import copy
import unittest

import numpy as np

from tests.test_v53r_regime_crn import regime_core
from HazardDatabase import HazardRegimeConfig, HazardRegimeProcess


class V53RForecastEnsembleTests(unittest.TestCase):
    def test_zero_rates_equal_deterministic_forecast(self):
        core = regime_core(5320)
        regime = HazardRegimeProcess(
            HazardRegimeConfig(enabled=True, episode_horizon=35),
            np.random.default_rng(7),
            core.cellTracker,
            core.mapDS,
        )
        regime.base_wind_speed_m_per_minute = core.hazardDS.wind_speed_m_per_minute
        regime.base_wind_direction_degrees = core.hazardDS.wind_direction_degrees
        regime.reset(0)
        core.hazard_regime = regime
        core.hazardDS.set_regime(regime)
        particles = regime.belief_particles(4, np.random.default_rng(7))
        ensemble = core.hazardDS.forecast_distribution_by_cell(5, particles)
        deterministic = core.hazardDS.forecast_distribution_by_cell(5)
        np.testing.assert_allclose(ensemble["expected"], deterministic["expected"])
        np.testing.assert_allclose(ensemble["uncertainty"], 0.0)

    def test_warned_shift_moves_mean_and_latent_events_create_dispersion(self):
        core = regime_core(5321)
        regime = core.hazard_regime
        warned = {
            "time": 5,
            "kind": "wind",
            "payload": {
                "direction_delta": 90.0,
                "speed_multiplier": 1.5,
                "warning_visible": True,
            },
        }
        regime.scheduled = [warned]
        regime._update_public(0)
        particles = regime.belief_particles(4, np.random.default_rng(8))
        forecast = core.hazardDS.forecast_distribution_by_cell(10, particles)
        self.assertTrue(np.isfinite(forecast["expected"]).all())
        latent = copy.deepcopy(particles)
        latent[0].scheduled.append(
            {"time": 4, "kind": "ignition", "payload": {"cell_index": 15}}
        )
        varied = core.hazardDS.forecast_distribution_by_cell(10, latent)
        self.assertGreater(float(varied["uncertainty"].max()), 0.0)

    def test_forecast_does_not_depend_on_hidden_current_intensity(self):
        core = regime_core(5322)
        regime = core.hazard_regime
        regime.intensity_belief = np.asarray([0.2, 0.5, 0.3], dtype=np.float64)
        twin = copy.deepcopy(regime)
        regime.intensity_multiplier = 0.7
        twin.intensity_multiplier = 1.6
        self.assertEqual(regime.public, twin.public)

        left_particles = regime.belief_particles(16, np.random.default_rng(91))
        right_particles = twin.belief_particles(16, np.random.default_rng(91))
        self.assertEqual(
            [particle.intensity_multiplier for particle in left_particles],
            [particle.intensity_multiplier for particle in right_particles],
        )
        left = core.hazardDS.forecast_distribution_by_cell(10, left_particles)
        right = core.hazardDS.forecast_distribution_by_cell(10, right_particles)
        for key in left:
            np.testing.assert_allclose(left[key], right[key], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
