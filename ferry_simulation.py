"""
Ferry network discrete-event simulation (SimPy).

Time handling:
- The simulation clock (`env.now`) is in **seconds since 06:00**.
- Many input parameters are expressed in minutes and converted to seconds internally.

Model sketch:
- Passenger arrivals follow a piecewise-constant non-homogeneous Poisson process with rates that
  change by period (AM peak / midday / PM peak / low).
- Passengers pass through turnstiles, queue at terminals (finite capacity → balking), board ferries
  subject to fixed dwell time and vessel capacity, and may transfer if a shuttle is enabled.
"""

import simpy
import numpy as np
import pandas as pd
import math
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Constants
SIM_START = 6 * 3600  # 06:00 in seconds
SIM_END = 22 * 3600   # 22:00 in seconds
WARMUP = 1 * 3600     # 3600 seconds
TOTAL_TIME = SIM_END - SIM_START

# Periods
# AM Peak: 07:00-09:00 (3600 - 10800), Midday: 09:00-17:00 (10800 - 39600), PM Peak: 17:00-19:00 (39600 - 46800), Low: 19:00-22:00 (46800 - 57600)
PERIODS = [
    ('am_peak', 3600, 10800),
    ('midday', 10800, 39600),
    ('pm_peak', 39600, 46800),
    ('low', 46800, 57600)
]

def get_period(t):
    """Return the demand period name for simulation time `t` (seconds since 06:00)."""
    for name, start, end in PERIODS:
        if start <= t < end:
            return name
    return 'low' # Default

# Direction multipliers (Table 6): Asia→Europe vs Europe→Asia
DIR_MULT = {
    'Asia->Europe': {'am_peak': 1.0, 'midday': 0.4, 'pm_peak': 0.3, 'low': 0.15},
    'Europe->Asia': {'am_peak': 0.3, 'midday': 0.4, 'pm_peak': 1.0, 'low': 0.15},
}

# Given parameters (Table 5) for A2 & E2
A2_BASE_RATE = 1000.0 / 60.0 # pax/min
E2_BASE_RATE = 500.0 / 60.0  # pax/min
A2_SPLITS = {'E1': 0.55, 'E2': 0.45}
E2_SPLITS = {'A1': 0.60, 'A2': 0.40}

# Period change boundaries (seconds since 06:00)
PERIOD_CHANGE_POINTS = [3600, 10800, 39600, 46800]
TIME_EPS = 1e-9

def analyze_arrivals(csv_path: str) -> Dict[str, Dict[str, float]]:
    """Estimate per-minute destination rates by period from a raw arrival CSV.

    Expected columns:
    - `date` and `time` (combined into a timestamp)
    - `destination` as a human-readable stop name (mapped to terminal codes)

    If multiple dates are present, counts are converted to average **per-day** rates.
    Returns a dict: period -> {terminal_code -> rate_per_min}.
    """
    rates = {}
    try:
        df = pd.read_csv(csv_path)

        # CSVs often contain multiple days; convert total counts into *average per-day* rates.
        n_days = int(df['date'].nunique()) if 'date' in df.columns else 1
        n_days = max(n_days, 1)

        df['dt'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['sec'] = (df['dt'].dt.hour - 6) * 3600 + df['dt'].dt.minute * 60 + df['dt'].dt.second

        # Map destinations: 'Eminonu'->'E1', 'Besiktas'->'E2', 'Kadikoy'->'A1', 'Uskudar'->'A2'
        dest_map = {'Eminonu': 'E1', 'Besiktas': 'E2', 'Kadikoy': 'A1', 'Uskudar': 'A2'}

        for name, start, end in PERIODS:
            # Low period includes 06:00–07:00 AND 19:00–22:00
            if name == 'low':
                mask = ((df['sec'] >= 0) & (df['sec'] < 3600)) | ((df['sec'] >= start) & (df['sec'] < end))
                # Effective low-period denominator uses two windows: [06:00,07:00) and [19:00,22:00).
                duration_mins = ((3600 - 0) + (end - start)) / 60.0
            else:
                mask = (df['sec'] >= start) & (df['sec'] < end)
                duration_mins = (end - start) / 60.0

            sub = df[mask]
            rates[name] = {}
            if not sub.empty:
                counts = sub['destination'].value_counts()
                for dest, count in counts.items():
                    mapped = dest_map.get(dest, dest)
                    rates[name][mapped] = count / (duration_mins * n_days)
    except Exception as e:
        print(f"Warning: could not analyze {csv_path}. Using fallback rates.")
        for name, _, _ in PERIODS:
            rates[name] = {'E1': 10.0, 'E2': 5.0, 'A1': 10.0, 'A2': 5.0} # Fallbacks
    return rates

@dataclass
class Passenger:
    """Entity tracked through the system for KPIs (times are in seconds since 06:00)."""
    id: int
    origin: str
    destination: str
    arrival_time: float
    turnstile_end_time: Optional[float] = None
    board_time: Optional[float] = None
    disembark_time: Optional[float] = None
    route: List[str] = field(default_factory=list)
    balked: bool = False
    left_behind: bool = False
    left_behind_events: int = 0
    missed_connection: bool = False
    transferred: bool = False
    indirect_route: bool = False
    current_terminal: Optional[str] = None
    current_terminal_arrival_time: Optional[float] = None
    wait_times: List[Tuple[str, float]] = field(default_factory=list)
    last_left_behind_time: Optional[float] = None

class Terminal:
    def __init__(self, env: simpy.Environment, name: str, berths: int, capacity: int, turnstiles: int, dwell: float):
        """Terminal with limited turnstiles, finite waiting area, and berth resource.

        Parameters:
        - `dwell`: fixed dock time in minutes (converted to seconds).
        - `capacity`: max number of waiting passengers before new arrivals balk.
        """
        self.env = env
        self.name = name
        self.capacity = capacity
        self.dwell = dwell * 60
        self.berths = simpy.Resource(env, capacity=berths)
        self.turnstiles = simpy.Resource(env, capacity=turnstiles)
        self.waiting_passengers: List[Passenger] = []
        self.balked_count = 0
        self.berth_busy_time = 0.0
        self.berth_busy_time_post_warmup = 0.0

class FerryLine:
    def __init__(self, env: simpy.Environment, sim, name: str, origin: Terminal, dest: Terminal, 
                 capacity: int, peak_hw: int, offpeak_hw: int, travel_time: int, rng: np.random.Generator, weather_p: float):
        """A directed service with a repeating schedule and fixed-capacity trips.

        Headways and travel time are provided in minutes and converted to seconds.
        Weather cancellations are modeled as an independent Bernoulli trial per scheduled sailing.
        """
        self.env = env
        self.sim = sim
        self.name = name
        self.origin = origin
        self.dest = dest
        self.capacity = capacity
        self.peak_hw = peak_hw * 60
        self.offpeak_hw = offpeak_hw * 60
        self.travel_time = travel_time * 60
        self.rng = rng
        self.weather_p = weather_p
        
        self.total_capacity_offered = 0
        self.total_pax_carried = 0
        
        self.env.process(self.run_schedule())

    def get_headway(self, t):
        """Return headway (seconds) at simulation time `t` (seconds since 06:00)."""
        if (3600 <= t < 10800) or (39600 <= t < 46800):
            return self.peak_hw
        return self.offpeak_hw

    def run_schedule(self):
        """Generate departures across the full simulation horizon (06:00–22:00)."""
        last_departure = 0.0
        # First departure exactly at 06:00
        while last_departure < TOTAL_TIME:
            # Weather cancellation (independent per scheduled sailing)
            run_trip = (self.rng.random() >= self.weather_p)
            if run_trip:
                self.env.process(self.operate_trip())

            hw = self.get_headway(last_departure)
            next_dep = last_departure + hw

            # If the current headway would carry us into the next period (strictly beyond a boundary),
            # use the new period headway measured from the previous departure.
            current_period = get_period(last_departure + TIME_EPS)
            period_before_next = get_period(max(0.0, next_dep - TIME_EPS))
            if period_before_next != current_period:
                next_dep = last_departure + self.get_headway(next_dep)

            wait_time = max(0, next_dep - self.env.now)
            yield self.env.timeout(wait_time)
            last_departure = next_dep

    def operate_trip(self):
        """One sailing: berth dwell/boarding at origin → travel → disembark at destination."""
        # 1. Request origin berth
        req = self.origin.berths.request()
        yield req

        berth_start = self.env.now

        # 2–3. Dock → fixed dwell time; boarding consumes dwell time and uses FCFS selection.
        boarded: List[Passenger] = []
        start_dwell = self.env.now

        while (self.env.now - start_dwell) < self.origin.dwell and len(boarded) < self.capacity:
            time_left = self.origin.dwell - (self.env.now - start_dwell)
            if time_left <= 0:
                break

            # Only board passengers whose next hop matches this line's destination.
            eligible = [p for p in self.origin.waiting_passengers if p.route and p.route[0] == self.dest.name]
            if not eligible:
                # Wait for next passenger (or dwell expiry)
                yield self.env.timeout(min(time_left, 1.0))
                continue

            p = eligible[0]
            b_time = float(self.sim.board_rng.exponential(1.0))
            if b_time > time_left:
                # Not enough time left to board the next passenger (FCFS)
                break

            self.origin.waiting_passengers.remove(p)
            yield self.env.timeout(b_time)

            p.board_time = self.env.now
            if p.current_terminal_arrival_time is not None:
                p.wait_times.append((self.origin.name, p.board_time - p.current_terminal_arrival_time))
            boarded.append(p)

        # Ensure the full dwell time is spent at the berth (fixed dwell)
        remaining = self.origin.dwell - (self.env.now - start_dwell)
        if remaining > 0:
            yield self.env.timeout(remaining)

        # Left-behind = passengers who wanted this destination but could not board this departure
        # (because of capacity and/or remaining dwell time).
        left_behind = [p for p in self.origin.waiting_passengers if p.route and p.route[0] == self.dest.name]
        for p in left_behind:
            # Avoid duplicate left-behind counting when simultaneous departures trigger route switching.
            if p.last_left_behind_time is None or abs(p.last_left_behind_time - self.env.now) > TIME_EPS:
                p.left_behind_events += 1
                p.left_behind = True
                p.last_left_behind_time = self.env.now
            if p.transferred:
                # If they are mid-journey, being left behind implies a missed connection.
                p.missed_connection = True

            # Route switching (only when shuttle is active) with q=0.6, if an indirect path exists
            if self.sim.config.get('shuttle', False) and not p.indirect_route:
                if self.sim.route_rng.random() < 0.6:
                    via = self.sim.transfers.get(self.origin.name, {}).get(p.destination)
                    if via is not None and via != self.dest.name:
                        p.route = [via, p.destination]
                        p.indirect_route = True

        # Berth occupancy tracking (origin dwell)
        berth_end = self.env.now
        occupied = berth_end - berth_start
        self.origin.berth_busy_time += occupied
        post_start = max(berth_start, WARMUP)
        post_end = min(berth_end, TOTAL_TIME)
        if post_end > post_start:
            self.origin.berth_busy_time_post_warmup += (post_end - post_start)

        self.origin.berths.release(req)

        self.total_capacity_offered += self.capacity
        self.total_pax_carried += len(boarded)

        # 4. Travel
        yield self.env.timeout(self.travel_time)

        # 5. Request dest berth
        req_d = self.dest.berths.request()
        yield req_d

        # Disembark (instant); transfer pax re-enter via turnstiles
        for p in boarded:
            if len(p.route) > 1:
                p.route.pop(0)
                p.transferred = True
                self.env.process(self.sim.passenger_process(p, self.dest, arrival_time=self.env.now))
            else:
                p.disembark_time = self.env.now

        self.dest.berths.release(req_d)

class Simulation:
    def __init__(self, scenario_name: str, rep: int, config: Dict, kadikoy_rates, eminonu_rates):
        """Scenario replication wrapper: initializes RNG streams, assets, demand, and service lines."""
        self.env = simpy.Environment()
        self.scenario_name = scenario_name
        self.rep = rep
        self.config = config
        self.kadikoy_rates = kadikoy_rates
        self.eminonu_rates = eminonu_rates
        
        seed_seq = np.random.SeedSequence(config['base_seed'] + rep)
        streams = seed_seq.spawn(10)
        self.arr_rng = np.random.default_rng(streams[0])
        self.turn_rng = np.random.default_rng(streams[1])
        self.board_rng = np.random.default_rng(streams[2])
        self.weath_rng = np.random.default_rng(streams[3])
        self.route_rng = np.random.default_rng(streams[4])
        self.dest_rng = np.random.default_rng(streams[5])
        
        self.terminals = {
            'A1': Terminal(self.env, 'A1', 3, 1500, 8, 6),
            'A2': Terminal(self.env, 'A2', 2, 600, 4, 5),
            'E1': Terminal(self.env, 'E1', 3, 1200, 6, 6),
            'E2': Terminal(self.env, 'E2', 2, 800, 4, 5)
        }
        
        hw_mult = config.get('hw_multiplier', 1.0)
        weather_p = 0.2 if config.get('lodos', False) else 0.0
        
        # Helper to round and multiply headways
        def mhw(hw): return max(1, round(hw * hw_mult))
        
        self.lines = [
            FerryLine(self.env, self, 'L1', self.terminals['A1'], self.terminals['E1'], 400, mhw(15), mhw(30), 20, self.weath_rng, weather_p),
            FerryLine(self.env, self, 'L1_rev', self.terminals['E1'], self.terminals['A1'], 400, mhw(15), mhw(30), 20, self.weath_rng, weather_p),
            FerryLine(self.env, self, 'L2', self.terminals['A1'], self.terminals['E2'], 200, mhw(20), mhw(40), 25, self.weath_rng, weather_p),
            FerryLine(self.env, self, 'L2_rev', self.terminals['E2'], self.terminals['A1'], 200, mhw(20), mhw(40), 25, self.weath_rng, weather_p),
            FerryLine(self.env, self, 'L3', self.terminals['A2'], self.terminals['E1'], 200, mhw(20), mhw(40), 15, self.weath_rng, weather_p),
            FerryLine(self.env, self, 'L3_rev', self.terminals['E1'], self.terminals['A2'], 200, mhw(20), mhw(40), 15, self.weath_rng, weather_p),
            FerryLine(self.env, self, 'L4', self.terminals['A2'], self.terminals['E2'], 200, mhw(30), mhw(60), 20, self.weath_rng, weather_p),
            FerryLine(self.env, self, 'L4_rev', self.terminals['E2'], self.terminals['A2'], 200, mhw(30), mhw(60), 20, self.weath_rng, weather_p)
        ]
        
        if config.get('shuttle', False):
            self.lines.append(FerryLine(self.env, self, 'L5', self.terminals['E1'], self.terminals['E2'], 200, mhw(15), mhw(15), 10, self.weath_rng, weather_p))
            self.lines.append(FerryLine(self.env, self, 'L5_rev', self.terminals['E2'], self.terminals['E1'], 200, mhw(15), mhw(15), 10, self.weath_rng, weather_p))

        # Transfer mappings A1->(E1)->E2 etc. Table 8 indirect
        self.transfers = {
            'A1': {'E2': 'E1'},
            'A2': {'E2': 'E1'},
            'E2': {'A1': 'E1', 'A2': 'E1'}
        }

        # Fallback profiles used when a historical period has zero observed arrivals.
        self.hist_fallback = {
            'A1': self._build_historical_fallback(self.kadikoy_rates, ['E1', 'E2']),
            'E1': self._build_historical_fallback(self.eminonu_rates, ['A1', 'A2'])
        }
        
        self.all_passengers = []
        self.p_count = 0

    @staticmethod
    def _build_historical_fallback(period_rates: Dict[str, Dict[str, float]], destinations: List[str]) -> Dict[str, object]:
        """Build fallback total rate and destination splits from available historical periods."""
        totals = []
        dest_sum = {d: 0.0 for d in destinations}

        for period_name, _, _ in PERIODS:
            rates = period_rates.get(period_name, {})
            period_total = 0.0
            for d in destinations:
                rate = float(rates.get(d, 0.0))
                dest_sum[d] += rate
                period_total += rate
            if period_total > 0:
                totals.append(period_total)

        fallback_total = float(np.mean(totals)) if totals else 0.0
        split_total = sum(dest_sum.values())
        if split_total > 0:
            fallback_splits = {d: dest_sum[d] / split_total for d in destinations}
        else:
            fallback_splits = {d: 1.0 / len(destinations) for d in destinations}

        return {'total_rate': fallback_total, 'splits': fallback_splits}

    def _historical_profile(self, origin: str, period: str, destinations: List[str]) -> Tuple[float, Dict[str, float]]:
        """Return period total rate and OD splits, with fallback when period data is sparse."""
        source_rates = self.kadikoy_rates if origin == 'A1' else self.eminonu_rates
        rates = source_rates.get(period, {})
        dest_rates = {d: float(rates.get(d, 0.0)) for d in destinations}
        total_rate = sum(dest_rates.values())

        if total_rate > 0:
            return total_rate, {d: (dest_rates[d] / total_rate) for d in destinations}

        # If one period is sparse/zero, borrow the nearest period with observed demand first.
        period_names = [name for name, _, _ in PERIODS]
        if period in period_names:
            target_idx = period_names.index(period)
            best_distance = None
            nearest_total = 0.0
            nearest_splits: Optional[Dict[str, float]] = None

            for idx, p_name in enumerate(period_names):
                p_rates = source_rates.get(p_name, {})
                p_dest_rates = {d: float(p_rates.get(d, 0.0)) for d in destinations}
                p_total = sum(p_dest_rates.values())
                if p_total <= 0:
                    continue
                distance = abs(idx - target_idx)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    nearest_total = p_total
                    nearest_splits = {d: (p_dest_rates[d] / p_total) for d in destinations}

            if nearest_splits is not None and nearest_total > 0:
                return nearest_total, nearest_splits

        fallback = self.hist_fallback[origin]
        fallback_total = float(fallback['total_rate'])
        fallback_splits = dict(fallback['splits'])
        if fallback_total > 0:
            return fallback_total, fallback_splits

        # If the historical file has no demand at all, keep zero rate but stable split defaults.
        uniform_splits = {d: 1.0 / len(destinations) for d in destinations}
        return 0.0, uniform_splits

    def generate_dyn_arrivals(self, origin, destinations, is_historical=False):
        """Generate arrivals for one origin using a piecewise Poisson process.

        Implementation detail:
        - Within each time segment where the total arrival rate is constant, inter-arrival times
          are exponential. If a sampled inter-arrival would cross a segment boundary, the process
          advances to the boundary and re-samples under the next segment's rate.
        """
        while self.env.now < TOTAL_TIME:
            t = self.env.now
            period = get_period(t + TIME_EPS)

            # Next time the period can change
            segment_end = TOTAL_TIME
            for b in PERIOD_CHANGE_POINTS:
                if b > (t + TIME_EPS):
                    segment_end = b
                    break
            time_left = segment_end - t
            if time_left <= TIME_EPS:
                yield self.env.timeout(TIME_EPS)
                continue

            if origin == 'A1':
                # A1 and E1 use CSV-derived historical rates by period with fallback for sparse periods.
                total_rate, splits = self._historical_profile('A1', period, destinations)
            elif origin == 'E1':
                total_rate, splits = self._historical_profile('E1', period, destinations)
            elif origin == 'A2':
                # A2/E2 use base rates multiplied by direction/period factors (Table 6) with fixed OD splits.
                splits = {d: float(A2_SPLITS.get(d, 0.0)) for d in destinations}
                mult = DIR_MULT['Asia->Europe'].get(period, 1.0)
                total_rate = A2_BASE_RATE * mult
            else:  # E2
                splits = {d: float(E2_SPLITS.get(d, 0.0)) for d in destinations}
                mult = DIR_MULT['Europe->Asia'].get(period, 1.0)
                total_rate = E2_BASE_RATE * mult

            # Normalise splits if needed
            sp_total = sum(splits.values())
            if sp_total > 0:
                splits = {d: (splits[d] / sp_total) for d in destinations}
            else:
                splits = {d: 1.0 / len(destinations) for d in destinations}

            if total_rate <= 0:
                # If no arrivals are expected in this segment, advance to the next period boundary.
                yield self.env.timeout(time_left)
                continue

            # Keep generating arrivals within this constant-rate segment.
            while self.env.now < segment_end - TIME_EPS and self.env.now < TOTAL_TIME:
                time_to_boundary = segment_end - self.env.now
                if time_to_boundary <= TIME_EPS:
                    break

                # `total_rate` is in pax/min; convert to seconds-based inter-arrival time.
                delta = float(self.arr_rng.exponential(1.0 / total_rate) * 60.0)
                if delta >= time_to_boundary:
                    # Crosses period boundary: move to boundary and sample again under new period rate.
                    yield self.env.timeout(time_to_boundary)
                    break

                yield self.env.timeout(max(delta, TIME_EPS))

                # Assign destination
                r = self.dest_rng.random()
                cumulative = 0.0
                dest = destinations[0]
                for d in destinations:
                    cumulative += splits.get(d, 0.0)
                    if r <= cumulative:
                        dest = d
                        break

                p = Passenger(id=self.p_count, origin=origin, destination=dest, arrival_time=self.env.now, route=[dest])
                self.p_count += 1
                self.all_passengers.append(p)
                self.env.process(self.passenger_process(p, self.terminals[origin], arrival_time=p.arrival_time))
            
    def passenger_process(self, p: Passenger, term: Terminal, arrival_time: Optional[float] = None):
        """Turnstiles → (optional) balking check → join waiting area for boarding."""
        if arrival_time is None:
            arrival_time = self.env.now

        p.current_terminal = term.name
        p.current_terminal_arrival_time = arrival_time

        with term.turnstiles.request() as req:
            yield req
            yield self.env.timeout(self.turn_rng.exponential(3.0))

        p.turnstile_end_time = self.env.now

        # Balking occurs after turnstiles if the waiting area is already full.
        if len(term.waiting_passengers) >= term.capacity:
            p.balked = True
            term.balked_count += 1
            return

        term.waiting_passengers.append(p)

    def run(self):
        """Start all demand processes and run the environment through the full horizon."""
        self.env.process(self.generate_dyn_arrivals('A1', ['E1', 'E2'], True))
        self.env.process(self.generate_dyn_arrivals('E1', ['A1', 'A2'], True))
        self.env.process(self.generate_dyn_arrivals('A2', ['E1', 'E2'], False))
        self.env.process(self.generate_dyn_arrivals('E2', ['A1', 'A2'], False))
        self.env.run(until=TOTAL_TIME)
        return self.collect_kpis()

    def collect_kpis(self):
        """Compute replication KPIs using only observations after the warm-up period."""
        # Filter warm-up observations to reduce initialization bias.
        total_pax = [p for p in self.all_passengers if p.arrival_time >= WARMUP]
        served_pax = [p for p in total_pax if p.disembark_time is not None]

        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        # Journey times (minutes)
        jt = [(p.disembark_time - p.arrival_time) / 60.0 for p in served_pax]
        jt_by_dest = {
            'E1': [(p.disembark_time - p.arrival_time) / 60.0 for p in served_pax if p.destination == 'E1'],
            'E2': [(p.disembark_time - p.arrival_time) / 60.0 for p in served_pax if p.destination == 'E2'],
            'A1': [(p.disembark_time - p.arrival_time) / 60.0 for p in served_pax if p.destination == 'A1'],
            'A2': [(p.disembark_time - p.arrival_time) / 60.0 for p in served_pax if p.destination == 'A2'],
        }

        # Wait times (minutes) per terminal, including transfers
        waits_by_term = {k: [] for k in ['A1', 'A2', 'E1', 'E2']}
        for p in served_pax:
            for term_code, wait_sec in p.wait_times:
                if term_code in waits_by_term:
                    waits_by_term[term_code].append(wait_sec / 60.0)

        total_arr = len(total_pax)
        total_pax_served = len(served_pax)
        obs_hours = (TOTAL_TIME - WARMUP) / 3600.0
        throughput = (total_pax_served / obs_hours) if obs_hours > 0 else 0.0

        loss_rate = sum(1 for p in total_pax if p.balked) / max(total_arr, 1)
        lb_rate = sum(1 for p in total_pax if p.left_behind_events > 0) / max(total_arr, 1)

        # Missed connections are only meaningful for transfer passengers.
        # Project definition: fraction of transfer passengers with second-leg wait > one L5 headway.
        transfer_pax = [p for p in served_pax if p.transferred]
        l5_lines = [l for l in self.lines if l.name in ('L5', 'L5_rev')]
        l5_headway_sec = float(min(l.peak_hw for l in l5_lines)) if l5_lines else (15.0 * 60.0)

        transfer_with_second_wait = [p for p in transfer_pax if len(p.wait_times) >= 2]
        for p in transfer_with_second_wait:
            # Robust to potential future extensions with additional transfers.
            p.missed_connection = p.wait_times[-1][1] > l5_headway_sec

        missed_conn_rate = (
            sum(1 for p in transfer_with_second_wait if p.missed_connection) / len(transfer_with_second_wait)
        ) if transfer_with_second_wait else 0.0
        if transfer_pax:
            if len(transfer_with_second_wait) < len(transfer_pax):
                print(
                    f"Warning: {len(transfer_pax) - len(transfer_with_second_wait)} transfer passengers "
                    "missing second-leg wait-time records."
                )

        # Load factors
        lf_L1 = sum(l.total_pax_carried for l in self.lines if 'L1' in l.name) / max(sum(l.total_capacity_offered for l in self.lines if 'L1' in l.name), 1)
        lf_L2 = sum(l.total_pax_carried for l in self.lines if 'L2' in l.name) / max(sum(l.total_capacity_offered for l in self.lines if 'L2' in l.name), 1)
        lf_L3 = sum(l.total_pax_carried for l in self.lines if 'L3' in l.name) / max(sum(l.total_capacity_offered for l in self.lines if 'L3' in l.name), 1)
        lf_L4 = sum(l.total_pax_carried for l in self.lines if 'L4' in l.name) / max(sum(l.total_capacity_offered for l in self.lines if 'L4' in l.name), 1)

        # Berth utilization (post-warm-up)
        obs_time = TOTAL_TIME - WARMUP
        denom_k = self.terminals['A1'].berths.capacity * obs_time
        denom_e = self.terminals['E1'].berths.capacity * obs_time
        berth_util_kadikoy = (self.terminals['A1'].berth_busy_time_post_warmup / denom_k) if denom_k > 0 else 0.0
        berth_util_eminonu = (self.terminals['E1'].berth_busy_time_post_warmup / denom_e) if denom_e > 0 else 0.0

        kpis = {
            'avg_journey_time': safe_mean(jt),
            'avg_jt_eminonu': safe_mean(jt_by_dest['E1']),
            'avg_jt_besiktas': safe_mean(jt_by_dest['E2']),
            'avg_jt_kadikoy': safe_mean(jt_by_dest['A1']),
            'avg_jt_uskudar': safe_mean(jt_by_dest['A2']),
            'loss_rate': loss_rate,
            'left_behind_rate': lb_rate,
            'throughput': throughput,
            'load_factor_L1': lf_L1,
            'load_factor_L2': lf_L2,
            'load_factor_L3': lf_L3,
            'load_factor_L4': lf_L4,
            'avg_wait_kadikoy': safe_mean(waits_by_term['A1']),
            'avg_wait_uskudar': safe_mean(waits_by_term['A2']),
            'avg_wait_eminonu': safe_mean(waits_by_term['E1']),
            'avg_wait_besiktas': safe_mean(waits_by_term['E2']),
            'berth_util_kadikoy': berth_util_kadikoy,
            'berth_util_eminonu': berth_util_eminonu,
            'missed_conn_rate': missed_conn_rate,
            'total_pax_served': total_pax_served
        }
        return kpis

def _sorted_scenarios(scenarios: List[str]) -> List[str]:
    """Return scenario names sorted by numeric suffix (S1, S2, ...)."""
    def _key(x: str):
        try:
            return int(x[1:]) if x.startswith('S') else 10**9
        except ValueError:
            return 10**9
    return sorted(scenarios, key=_key)

def _plot_kpi_with_ci(summary_df: pd.DataFrame, kpi: str, title: str, ylabel: str, out_path: str):
    """Create a per-scenario mean chart with 95% CI error bars for one KPI."""
    sub = summary_df[summary_df['kpi'] == kpi].copy()
    if sub.empty:
        return

    order = _sorted_scenarios(sub['scenario'].unique().tolist())
    sub['scenario'] = pd.Categorical(sub['scenario'], categories=order, ordered=True)
    sub = sub.sort_values('scenario')

    y = sub['mean'].to_numpy(dtype=float)
    yerr_low = np.maximum(0.0, y - sub['ci_lower'].to_numpy(dtype=float))
    yerr_high = np.maximum(0.0, sub['ci_upper'].to_numpy(dtype=float) - y)

    plt.figure(figsize=(9, 5))
    plt.bar(sub['scenario'].astype(str), y, color='#2F6690')
    plt.errorbar(
        sub['scenario'].astype(str),
        y,
        yerr=[yerr_low, yerr_high],
        fmt='none',
        ecolor='black',
        elinewidth=1,
        capsize=4,
        capthick=1
    )
    plt.title(title)
    plt.xlabel('Scenario')
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def generate_output_plots(results_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: str = 'plots'):
    """Generate scenario-comparison plots to track simulation outputs."""
    os.makedirs(out_dir, exist_ok=True)

    _plot_kpi_with_ci(
        summary_df,
        'avg_journey_time',
        'Average Journey Time by Scenario (95% CI)',
        'Minutes',
        os.path.join(out_dir, 'avg_journey_time_ci.png')
    )
    _plot_kpi_with_ci(
        summary_df,
        'throughput',
        'Throughput by Scenario (95% CI)',
        'Passengers per Hour',
        os.path.join(out_dir, 'throughput_ci.png')
    )

    # Multi-KPI service-quality comparison (mean values).
    quality_kpis = ['loss_rate', 'left_behind_rate', 'missed_conn_rate']
    quality = summary_df[summary_df['kpi'].isin(quality_kpis)].copy()
    if not quality.empty:
        order = _sorted_scenarios(quality['scenario'].unique().tolist())
        pivot_q = quality.pivot(index='scenario', columns='kpi', values='mean').reindex(order)
        ax = pivot_q.plot(kind='bar', figsize=(10, 5), color=['#D1495B', '#EDA35A', '#00798C'])
        ax.set_title('Service Quality Rates by Scenario')
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Rate')
        ax.grid(axis='y', alpha=0.25)
        ax.legend(title='KPI')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'service_quality_rates.png'), dpi=200)
        plt.close()

    # Replication spread for avg journey time (boxplot).
    rep_jt = results_df[results_df['kpi'] == 'avg_journey_time'].copy()
    if not rep_jt.empty:
        order = _sorted_scenarios(rep_jt['scenario'].unique().tolist())
        data = [rep_jt[rep_jt['scenario'] == s]['value'].to_numpy(dtype=float) for s in order]
        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=order, showmeans=True)
        plt.title('Replication Distribution: Average Journey Time')
        plt.xlabel('Scenario')
        plt.ylabel('Minutes')
        plt.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'avg_journey_time_boxplot.png'), dpi=200)
        plt.close()

def main():
    """Run all scenarios/replications and write `results.csv` and `summary.csv`."""
    print("Performing input analysis on CSV files...")
    k_rates = analyze_arrivals('arrivals_kadikoy.csv')
    e_rates = analyze_arrivals('arrivals_eminonu.csv')
    
    scenarios = {
        'S1': {'shuttle': False, 'lodos': False, 'hw_multiplier': 1.0, 'base_seed': 42},
        'S2': {'shuttle': False, 'lodos': False, 'hw_multiplier': 2.0 / 3.0, 'base_seed': 42},
        'S3': {'shuttle': True, 'lodos': False, 'hw_multiplier': 1.0, 'base_seed': 42},
        'S4': {'shuttle': False, 'lodos': True, 'hw_multiplier': 1.0, 'base_seed': 42},
        'S5': {'shuttle': True, 'lodos': True, 'hw_multiplier': 2.0 / 3.0, 'base_seed': 42},
        'S6': {'shuttle': False, 'lodos': False, 'hw_multiplier': 0.8, 'base_seed': 42}
    }

    expected_50pct = {15: 10, 20: 13, 30: 20, 40: 27, 60: 40}
    for s_name, config in scenarios.items():
        if np.isclose(config.get('hw_multiplier', 1.0), 2.0 / 3.0):
            rounded = {base: max(1, round(base * config['hw_multiplier'])) for base in expected_50pct}
            if rounded != expected_50pct:
                raise ValueError(f"Scenario {s_name} headway rounding mismatch: expected {expected_50pct}, got {rounded}")
    
    results = []
    
    for s_name, config in scenarios.items():
        print(f"Running {s_name}...")
        for rep in range(1, 21):
            sim = Simulation(s_name, rep, config, k_rates, e_rates)
            kpis = sim.run()
            for k, v in kpis.items():
                results.append({'scenario': s_name, 'replication': rep, 'kpi': k, 'value': v})
                
    # Write results.csv
    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)
    
    # Write summary.csv (95% t-CI, df = 19 for 20 replications)
    t_crit_95_df19 = 2.093024054
    summary = df.groupby(['scenario', 'kpi'])['value'].agg(
        mean='mean',
        std=lambda x: float(np.std(x, ddof=1)),
        n='count'
    ).reset_index()
    summary['ci_half_width'] = t_crit_95_df19 * summary['std'] / np.sqrt(summary['n'])
    summary['ci_lower'] = summary['mean'] - summary['ci_half_width']
    summary['ci_upper'] = summary['mean'] + summary['ci_half_width']
    summary_out = summary[['scenario', 'kpi', 'mean', 'ci_lower', 'ci_upper', 'std']]
    summary_out.to_csv('summary.csv', index=False)

    generate_output_plots(df, summary_out, out_dir='plots')
    print("Done! Outputs saved to results.csv, summary.csv, and plots/")

if __name__ == '__main__':
    main()
