import unittest

import numpy as np

import CounterfactualBranch as CB
import nmcc_testbed
from HazardDatabase import HazardRegimeConfig, HazardRegimeProcess


def regime_core(seed=5301):
    core = nmcc_testbed.build(
        grid=7, cell_x=4, cell_y=4, population=80, stop_time=35,
        hazard_count=2, candidate_count=32, scenario_seed=seed,
    )
    config = HazardRegimeConfig(
        enabled=True,
        wind_shift_rate_per_minute=0.08,
        intensity_switch_rate_per_minute=0.06,
        closure_rate_per_minute=0.04,
        secondary_ignition_rate_per_minute=0.04,
        episode_horizon=35,
    )
    regime = HazardRegimeProcess(
        config, np.random.default_rng(seed + 1), core.cellTracker, core.mapDS
    )
    regime.base_wind_speed_m_per_minute = 20.0
    regime.reset(0)
    core.hazard_regime = regime
    core.hazardDS.set_regime(regime)
    core.hazardRegimeEnabled = True
    return core


class V53RRegimeCRNTests(unittest.TestCase):
    def test_physical_spread_updates_the_public_intensity_filter(self):
        core = regime_core(5300)
        regime = core.hazard_regime
        self.assertEqual(regime.intensity_belief_updates, 0)
        regime.advance(1, core.hazardDS, core.mapDS)
        core.hazardDS.spreadUpdate()
        self.assertEqual(regime.intensity_belief_updates, 1)
        self.assertAlmostEqual(float(np.sum(regime.intensity_belief)), 1.0)
        self.assertTrue(np.isfinite(regime.intensity_belief).all())
        self.assertIn("intensity_belief_mean", regime.public)

    def test_same_seed_event_tape_and_advance_trace_are_identical(self):
        first, second = regime_core(), regime_core()
        self.assertEqual(first.hazard_regime.scheduled, second.hazard_regime.scheduled)
        trace_first, trace_second = [], []
        for minute in range(1, 16):
            for core, trace in ((first, trace_first), (second, trace_second)):
                core.hazard_regime.advance(minute, core.hazardDS, core.mapDS)
                trace.append(core.hazard_regime.snapshot())
        for left, right in zip(trace_first, trace_second):
            left.pop("rng_state")
            right.pop("rng_state")
            self.assertEqual(left, right)

    def test_snapshot_restore_round_trip_and_action_independence(self):
        core = regime_core(5307)
        snapshot = CB.capture(core, label="v53r-regime")
        digest = snapshot.state_digest()
        for _ in range(5):
            CB.advance_one_timestep(core)
        CB.restore(core, snapshot)
        self.assertEqual(CB.capture(core).state_digest(), digest)
        CB.assert_hazard_is_action_independent(
            core, steps=5, cell_index=core.feasible_cells()[0]
        )


if __name__ == "__main__":
    unittest.main()
