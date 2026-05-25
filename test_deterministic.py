import unittest
import simpy
from ferry_simulation import Simulation

class TestDeterministicVerify(unittest.TestCase):
    def test_single_ferry_trip(self):
        \"\"\"Deterministic test for single ferry routing and arrival times.\"\"\"
        scenarios = {
            'S1_test': {'shuttle': False, 'lodos': False, 'hw_multiplier': 1.0, 'base_seed': 42},
        }
        
        # Override arrival rates to 0 to only inject one specific passenger manually.
        k_rates = {}
        e_rates = {}
        
        sim = Simulation('S1_test', 1, scenarios['S1_test'], k_rates, e_rates)
        sim.weath_rng.random = lambda: 1.0 # no weather cancellations
        
        # Manually create one passenger at time = 8000 (after 3600 warmup).
        from ferry_simulation import Passenger
        p = Passenger(id=999, origin='A1', destination='E1', arrival_time=8000, route=['E1'])
        
        def mock_generate_dyn_arrivals(origin, destinations, is_historical):
            if origin == 'A1':
                yield sim.env.timeout(8000)
                sim.all_passengers.append(p)
                sim.env.process(sim.passenger_process(p, sim.terminals['A1'], arrival_time=p.arrival_time))
            # No other arrivals
            yield sim.env.timeout(999999)
            
        sim.generate_dyn_arrivals = mock_generate_dyn_arrivals
        
        sim.run()
        
        # Verify
        self.assertIsNotNone(p.board_time, \"Passenger should have boarded.\")
        self.assertIsNotNone(p.disembark_time, \"Passenger should have arrived at E1.\")
        self.assertEqual(p.destination, 'E1', \"Route should lead to E1.\")
        self.assertFalse(p.balked)
        self.assertTrue(p.board_time >= 8000)

if __name__ == '__main__':
    unittest.main()
