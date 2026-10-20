import unittest

import numpy as np

import CounterfactualBranch as CB
import nmcc_testbed
from HazardDatabase import HazardRegimeConfig, HazardRegimeProcess


class V53RIdentityTests(unittest.TestCase):
    def test_disabled_regime_preserves_legacy_trajectory_bit_for_bit(self):
        baseline = nmcc_testbed.build(
            grid=6, cell_x=4, cell_y=4, population=250, stop_time=20,
            hazard_count=2, scenario_seed=5312,
        )
        redesigned = nmcc_testbed.build(
            grid=6, cell_x=4, cell_y=4, population=250, stop_time=20,
            hazard_count=2, scenario_seed=5312,
        )
        regime = HazardRegimeProcess(
            HazardRegimeConfig(enabled=False, episode_horizon=20),
            np.random.default_rng(991),
            redesigned.cellTracker,
            redesigned.mapDS,
        )
        redesigned.hazard_regime = regime
        redesigned.hazardDS.set_regime(regime)
        for minute in range(8):
            CB.advance_one_timestep(baseline)
            regime.advance(minute + 1, redesigned.hazardDS, redesigned.mapDS)
            CB.advance_one_timestep(redesigned)
        self.assertEqual(
            CB.live_state_digest(baseline), CB.live_state_digest(redesigned)
        )


if __name__ == "__main__":
    unittest.main()
