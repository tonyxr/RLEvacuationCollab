import unittest
from pathlib import Path

from CityProfiles import load_city_suite
from map_factorial_backtest import (
    _completion_audit,
    _source_training_congestion_contracts,
    _source_training_action_intervals,
    _source_training_map_contracts,
    _source_training_installation_budgets,
    build_map_execution_plan,
    load_map_design,
)


class MapFactorialDesignTests(unittest.TestCase):
    def test_design_encodes_requested_population_candidate_hazard_matrix(self):
        design = load_map_design()
        self.assertEqual(design.population_levels, (10000, 20000, 30000, 40000, 50000))
        self.assertEqual(design.shelter_candidate_levels, (5, 10, 15, 20))
        self.assertEqual(design.hazard_count_levels, (1, 2, 3, 4, 5))
        self.assertEqual(design.initial_shelters, 2)
        self.assertEqual(design.maximum_additional_shelters, 5)
        self.assertEqual(design.shelter_action_interval, 2)
        self.assertEqual(design.horizon_timesteps, 60)
        self.assertEqual(design.progress_milestones, (0, 10, 20, 30, 40, 50, 60))

    def test_full_plan_counts_every_city_condition_and_policy(self):
        design = load_map_design()
        cities = load_city_suite().cities
        plan = build_map_execution_plan(cities=cities, design=design)
        self.assertEqual(plan["factor_cells"], 500)
        self.assertEqual(plan["episodes"], 1500)
        self.assertEqual(plan["decision_sequence_maps"], 500)
        self.assertEqual(plan["decision_epoch_comparison_maps"], 500)
        self.assertEqual(plan["progress_comparison_maps"], 500)
        self.assertEqual(plan["congestion_diagnostic_graphs"], 500)
        self.assertEqual(plan["total_comparison_and_diagnostic_graphs"], 2000)
        self.assertEqual(plan["requested_pedestrian_trajectories"], 45_000_000)
        self.assertEqual(plan["maximum_person_transitions"], 2_700_000_000)
        self.assertEqual(plan["simulator_stop_time"], 61)
        self.assertEqual(plan["shelter_action_interval"], 2)

    def test_source_training_budget_is_explicitly_audited(self):
        manifest = {
            "effective_overrides_by_city": {
                "new": {"maxAdditionalShelters": 5},
                "legacy": {"stopTime": 15},
            }
        }
        self.assertEqual(
            _source_training_installation_budgets(manifest, ("new", "legacy")),
            {"new": 5, "legacy": 0},
        )

    def test_source_training_congestion_contract_is_explicitly_audited(self):
        manifest = {
            "effective_overrides_by_city": {
                "new": {
                    "timeStepMinutes": 1.0,
                    "congestionEnabled": True,
                    "congestionEffectiveWidthM": 3.0,
                    "congestionJamDensityPedPerM2": 5.4,
                    "congestionShape": 1.913,
                    "congestionMinimumSpeedRatio": 0.05,
                    "congestionSubstepSeconds": 10.0,
                },
                "legacy": {"stopTime": 15},
            }
        }
        contracts = _source_training_congestion_contracts(
            manifest, ("new", "legacy")
        )
        self.assertTrue(contracts["new"]["enabled"])
        self.assertEqual(contracts["new"]["model"], "weidmann_physical_link_v1")
        self.assertFalse(contracts["legacy"]["enabled"])
        self.assertEqual(contracts["legacy"]["model"], "none")

    def test_source_action_interval_and_map_footprint_are_audited(self):
        manifest = {
            "effective_overrides_by_city": {
                "new": {
                    "shelterActionInterval": 2,
                    "mapQueryMode": "point",
                    "mapCenterLat": 40.0,
                    "mapCenterLon": -75.0,
                    "mapRadiusM": 3000,
                    "cellX": 8,
                    "cellY": 8,
                },
                "legacy": {"stopTime": 15},
            }
        }
        self.assertEqual(
            _source_training_action_intervals(manifest, ("new", "legacy")),
            {"new": 2, "legacy": 5},
        )
        maps = _source_training_map_contracts(manifest, ("new", "legacy"))
        self.assertEqual(maps["new"]["radius_m"], 3000.0)
        self.assertEqual(maps["legacy"]["query_mode"], "place")

    def test_completion_audit_checks_policy_parity_budget_and_artifact_counts(self):
        design = load_map_design()
        plan = {
            "episodes": 3,
            "factor_cells": 1,
            "total_comparison_and_diagnostic_graphs": 4,
        }
        common = {
            "condition_id": "one",
            "initial_population": 2500,
            "safe_completed": 2460,
            "casualty": 40,
            "unfinished": 0,
            "actual_candidate_count": 10,
            "initial_observation_digest": "same-observation",
            "hazard_trajectory_digest": "same-hazard",
            "rl_heuristic_sequence_identical": False,
        }
        rows = [
            {**common, "deployment_strategy": "rl", "deployments_made": 5},
            {**common, "deployment_strategy": "heuristic", "deployments_made": 5},
            {
                **common,
                "deployment_strategy": "initial_only",
                "deployments_made": 0,
                "initial_observation_digest": "static-observation",
            },
        ]
        audit = _completion_audit(
            rows=rows,
            design=design,
            plan=plan,
            figure_paths=tuple(Path(f"figure-{index}.png") for index in range(4)),
        )
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["dynamic_initial_observation_parity"])
        self.assertTrue(audit["common_hazard_trajectory_within_condition"])
        self.assertEqual(audit["rl_heuristic_identical_sequence_conditions"], 0)


if __name__ == "__main__":
    unittest.main()
