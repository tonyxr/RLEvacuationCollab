import unittest

from CityProfiles import load_city_suite
from population_candidate_backtest import (
    _condition_seed,
    _source_training_action_intervals,
    _source_training_horizons,
    _validate_matched_rows,
    build_execution_plan,
)


class PopulationCandidatePlanTests(unittest.TestCase):
    def test_full_plan_encodes_true_60_minute_horizon(self):
        cities = load_city_suite().cities
        populations = (10000, 20000, 30000, 40000, 50000)
        candidates = (25, 50, 75, 100, 125)
        plan = build_execution_plan(
            cities=cities,
            populations=populations,
            candidates=candidates,
            replications=5,
            policy_count=8,
            horizon_timesteps=60,
            shelter_action_interval=2,
        )
        self.assertEqual(25, plan["factor_cells_per_city"])
        self.assertEqual(625, plan["matched_scenario_cells"])
        self.assertEqual(5625, plan["total_episodes"])
        self.assertEqual(60, plan["horizon_timesteps"])
        self.assertEqual(61, plan["simulator_stop_time"])
        self.assertEqual(2, plan["shelter_action_interval"])

    def test_condition_seeds_are_stable_and_factor_specific(self):
        first = _condition_seed(7, 1, 1, 1, 1, 710)
        self.assertEqual(first, _condition_seed(7, 1, 1, 1, 1, 710))
        self.assertNotEqual(first, _condition_seed(7, 1, 2, 1, 1, 710))
        self.assertNotEqual(first, _condition_seed(7, 1, 1, 1, 1, 711))

    def test_source_training_horizon_uses_realized_transition_count(self):
        manifest = {
            "effective_overrides_by_city": {
                "state_college_pa": {"stopTime": 121},
                "reading_pa": {"stopTime": 81},
            }
        }
        self.assertEqual(
            {"state_college_pa": 120, "reading_pa": 80},
            _source_training_horizons(
                manifest,
                ["state_college_pa", "reading_pa"],
            ),
        )

    def test_source_training_action_interval_uses_legacy_default(self):
        manifest = {
            "effective_overrides_by_city": {
                "new": {"shelterActionInterval": 2},
                "legacy": {"stopTime": 61},
            }
        }
        self.assertEqual(
            {"new": 2, "legacy": 5},
            _source_training_action_intervals(manifest, ["new", "legacy"]),
        )

    def test_parity_requires_every_policy_and_shared_interface_digest(self):
        common = {
            "city_id": "state_college_pa",
            "population_level": 2500,
            "shelter_candidate_level": 25,
            "scale_replication": 1,
            "scenario_seed": 99,
            "initial_population": 2500,
            "initial_observation_digest": "observation",
            "hazard_trajectory_digest": "hazard",
            "maximum_dynamic_deployments": 24,
            "actual_candidate_count": 25,
            "horizon_timesteps": 120,
        }
        rows = [
            {
                **common,
                "deployment_strategy": "heuristic",
                "policy_replication": 0,
            },
            {
                **common,
                "deployment_strategy": "rl",
                "policy_replication": 1,
            },
            {
                **common,
                "deployment_strategy": "rl",
                "policy_replication": 2,
            },
        ]
        self.assertTrue(_validate_matched_rows(rows, 2)["verified"])
        rows[-1]["initial_observation_digest"] = "changed"
        self.assertFalse(_validate_matched_rows(rows, 2)["verified"])


if __name__ == "__main__":
    unittest.main()
