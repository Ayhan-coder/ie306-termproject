**IE 306 — Systems Simulation**  
**Term Project: Istanbul Ferry Terminal Operations**  
**(Group Project — Teams of 3)**  

**Spring 2026** | **Due Date: Last day of finals**

### Overview
Istanbul’s Bosphorus ferry network carries hundreds of thousands of commuters daily between the Asian and European sides of the city. In this project your team will build a **discrete-event simulation** of a four-terminal ferry network using **SimPy**. The network connects two Asian-side terminals (Kadıköy and Üsküdar) to two European-side terminals (Eminönü and Beşiktaş) via four regular ferry lines and one optional shuttle line.

Your model must faithfully represent:
- time-varying passenger demand,
- schedule-based ferry operations with berth contention,
- batch boarding with dwell-time limits,
- route switching for left-behind passengers,
- weather disruptions.

You will design and run experiments comparing operational interventions (increased frequency, a European-side shuttle line) under both normal and storm conditions.

Unlike the assignments, this project requires you to **make and defend your own modeling decisions**. The system description below defines *what* must be modeled; *how* you implement each aspect in SimPy is up to you. Your report should justify your design choices.

### 1 System Description

**Bosphorus ferry network** (see Figure 1)

- **Asian Side**  
  - Kadıköy (A1) – 3 berths  
  - Üsküdar (A2) – 2 berths  

- **European Side**  
  - Eminönü (E1) – 3 berths  
  - Beşiktaş (E2) – 2 berths  

**Lines**  
- **L1 (20 min)**: A1 ↔ E1  
- **L2 (25 min)**: A1 ↔ E2  
- **L3 (15 min)**: A2 ↔ E1  
- **L4 (20 min)**: A2 ↔ E2  
- **L5 shuttle (10 min)**: E1 ↔ E2 (optional, dashed line)

*Only cross-Bosphorus OD pairs are served by the ferry system.*

#### 1.1 Terminal Network

| Terminal   | Code | Berths | Waiting cap. | Turnstiles | Dwell time |
|------------|------|--------|--------------|------------|------------|
| Kadıköy    | A1   | 3      | 1 500 pax    | 8          | 6 min      |
| Üsküdar    | A2   | 2      | 600 pax      | 4          | 5 min      |
| Eminönü    | E1   | 3      | 1 200 pax    | 6          | 6 min      |
| Beşiktaş   | E2   | 2      | 800 pax      | 4          | 5 min      |

**Key rules**  
- **Berths**: A ferry arriving when all berths are occupied waits offshore.  
- **Waiting area**: Passengers enter via turnstiles *first*. If the waiting area is full after passing the turnstile, the passenger **balks** (leaves and is counted as lost).  
- **Dwell time**: Fixed per terminal. Boarding stops at capacity **or** dwell-time expiry (whichever first). Left-behind passengers stay for the next sailing.

#### 1.2 Ferry Lines

| Line | Route      | Type          | Capacity | Peak hdwy | Off-peak hdwy | Travel time |
|------|------------|---------------|----------|-----------|---------------|-------------|
| L1   | A1 ↔ E1    | Large vapur   | 400 pax  | 15 min    | 30 min        | 20 min      |
| L2   | A1 ↔ E2    | Small motor   | 200 pax  | 20 min    | 40 min        | 25 min      |
| L3   | A2 ↔ E1    | Small motor   | 200 pax  | 20 min    | 40 min        | 15 min      |
| L4   | A2 ↔ E2    | Small motor   | 200 pax  | 30 min    | 60 min        | 20 min      |
| L5   | E1 ↔ E2    | Small motor   | 200 pax  | 15 min    | 15 min        | 10 min      |

- Lines operate **bidirectionally** with independent schedules from each end.  
- First departure in each direction: **06:00**.  
- **Headway transitions**: Next departure uses the new period’s headway measured from the *previous* departure (not from the boundary).

#### 1.3 Passenger Arrivals
Passengers arrive as a **non-homogeneous Poisson process**. Only cross-Bosphorus OD pairs are modeled.

**Time-of-day periods**

| Period    | Hours              |
|-----------|--------------------|
| AM peak   | 07:00–09:00        |
| Midday    | 09:00–17:00        |
| PM peak   | 17:00–19:00        |
| Low       | 06:00–07:00, 19:00–22:00 |

**1.3.1 Terminals with Historical Data (A1 & E1)**  
Six months of passenger arrival records are provided in `arrivals_kadikoy.csv` and `arrivals_eminonu.csv`.

**Your input-analysis task** (must be shown in the report):
1. Compute inter-arrival times per period.  
2. Fit exponential distribution → estimate λ for each period (histograms + Q–Q plots).  
3. Kolmogorov–Smirnov goodness-of-fit test (statistic + p-value). Discuss validity.  
4. Estimate destination split (fraction to each possible destination).  
5. Use fitted λ and splits in the simulation.

**1.3.2 Terminals with Given Parameters (A2 & E2)**

| Origin          | Base rate (pax/hr) | Destination 1          | Destination 2          |
|-----------------|--------------------|------------------------|------------------------|
| Üsküdar (A2)    | 1 000              | 55 % → Eminönü         | 45 % → Beşiktaş        |
| Beşiktaş (E2)   | 500                | 60 % → Kadıköy         | 40 % → Üsküdar         |

**1.3.3 Effective Arrival Rate**  
\[
\lambda_{o\to d}(t) = \lambda_o^{\text{base}} \times p_{o\to d} \times m_{\text{dir}}(\text{period}(t),\ \text{direction}(o,d))
\]

**Direction multipliers** ($m_{\text{dir}}$)

| Period   | Asia → Europe | Europe → Asia |
|----------|---------------|---------------|
| AM peak  | 1.0           | 0.3           |
| Midday   | 0.4           | 0.4           |
| PM peak  | 0.3           | 1.0           |
| Low      | 0.15          | 0.15          |

#### 1.4 Terminal Service
- **Turnstiles**: Exp(3 s) per passenger, multiple parallel.  
- **Boarding**: Exp(1 s) per passenger, FCFS among passengers whose *next leg* matches the ferry’s line (includes transfers).

#### 1.5 Ferry Operations
Each scheduled departure is an independent process (no vessel tracking).  
1. Request berth (wait offshore if busy).  
2. Dock → dwell time.  
3. Board passengers.  
4. Depart → travel time.  
5. Disembark at destination (instant), release berth.

#### 1.6 Route Choice & Left-Behind Passengers
- Primary route = direct line (see Table 7).  
- **Route switching** (only when L5 shuttle is active): left-behind passengers switch to indirect route with probability **q = 0.6** (if an indirect path exists). Transfer passengers re-enter via turnstiles.

#### 1.7 Weather Disruption
- **Calm**: no cancellations.  
- **Lodos**: each scheduled sailing independently cancelled with probability **p_cancel = 0.20**.

### 2 Scenarios (mandatory)

| #  | Description       | Frequency | Shuttle (L5) | Weather | Purpose                  |
|----|-------------------|-----------|--------------|---------|--------------------------|
| S1 | Baseline          | Standard  | No           | Calm    | Reference                |
| S2 | High frequency    | +50 %     | No           | Calm    | Capacity investment      |
| S3 | Shuttle network   | Standard  | Yes          | Calm    | Network design           |
| S4 | Storm stress test | Standard  | No           | Lodos   | Disruption impact        |
| S5 | Full intervention | +50 %     | Yes          | Lodos   | Combined mitigation      |

**+50 % frequency** = multiply every headway by 2/3 and round to nearest minute.

**Additional scenario (S6+)**: You **must** design and run at least one extra scenario of your own choosing. Explain purpose/motivation in the report.

### 3 Simulation Requirements
- **Random streams**: Separate `numpy.random.default_rng()` streams for every source of randomness. Use **Common Random Numbers** (same base seed per replication across scenarios).  
- **Simulation window**: 06:00–22:00 (57 600 s).  
- **Warm-up**: Discard first 1 hour (06:00–07:00).  
- **Replications**: 20.  
- **End-of-day**: Passengers still in system are excluded from all KPIs.

### 4 Key Performance Indicators (KPIs)
(Exact KPI names required for output CSV)

| KPI name                | Definition |
|-------------------------|----------|
| `avg_journey_time`      | Avg. journey time, all passengers |
| `avg_jt_eminonu`        | Avg. journey time → Eminönü |
| `avg_jt_besiktas`       | Avg. journey time → Beşiktaş |
| `avg_jt_kadikoy`        | Avg. journey time → Kadıköy |
| `avg_jt_uskudar`        | Avg. journey time → Üsküdar |
| `loss_rate`             | Passenger loss rate (balks) |
| `left_behind_rate`      | Left-behind rate |
| `throughput`            | Throughput (pax/h, post-warm-up) |
| `load_factor_L1` … `L4` | Load factor per line |
| `avg_wait_kadikoy` … `besiktas` | Avg. wait time at each terminal |
| `berth_util_kadikoy`, `berth_util_eminonu` | Berth utilization |
| `missed_conn_rate`      | Missed-connection rate (transfer pax) |
| `total_pax_served`      | Total passengers served (post-warm-up) |

### 5 Output Format
Your notebook must produce **two CSV files** (`results.csv` and `summary.csv`). Exact column names and KPI strings are mandatory.

---

**Next steps (if you want to proceed with the project)**  
1. **Input analysis** on the two historical CSV files (I can run it with Python).  
2. Build the SimPy model.  
3. Run the 5+ scenarios and generate the required CSVs.

Would you like me to **(a)** finish the clean Markdown version with any missing pages, **(b)** start the full input analysis on `arrivals_kadikoy.csv` + `arrivals_eminonu.csv` right now, or **(c)** something else?