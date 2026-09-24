import json
import tempfile
import unittest
from pathlib import Path

from CityProfiles import load_city_suite
from NMCCPIConfig import NMCC_PI_CORE_FIELDS
from TrainingCurriculum import (
    DEFAULT_TRAINING_CURRICULUM_PATH,
    build_curriculum_schedule,
    load_training_curriculum,
)
from multicity_backtest import balanced_rollout_episodes


ROOT = Path(__file__).resolve().parents[1]


class TrainingCurriculumTests(unittest.TestCase):
    def test_registered_curriculum_matches_confirmatory_budget(self):
        curriculum = load_training_curriculum(DEFAULT_TRAINING_CURRICULUM_PATH)
        self.assertEqual(curriculum.episodes_per_city, 120)
        self.assertEqual(curriculum.stages[-1].episodes_per_city, 84)

    def test_operational_curriculum_reaches_target_scale_with_declared_cohorts(self):
        curriculum = load_training_curriculum(
            ROOT / "config" / "staged_training_curriculum_operational.json"
        )
        self.assertEqual(curriculum.episodes_per_city, 20)
        self.assertEqual(curriculum.episodes_per_city * 5, 100)
        self.assertEqual(curriculum.stages[-1].episodes_per_city, 6)
        expected_sizes = [10, 20, 20, 20, 20, 20]
        self.assertEqual(
            [stage.variants[0].overrides["pedestrianGroupSize"] for stage in curriculum.stages],
            expected_sizes,
        )
        self.assertEqual(
            curriculum.stages[-1].variants[0].overrides["pedVol"],
            50000,
        )

    def test_v28_curriculum_uses_persistent_heldout_value_control(self):
        curriculum = load_training_curriculum(
            ROOT
            / "config"
            / "state_college_training_curriculum_2500_risk_value_v28.json"
        )
        self.assertEqual(curriculum.episodes_per_city, 104)
        overrides = curriculum.stages[0].variants[0].overrides
        self.assertEqual(overrides["nmccPiActorObjective"], "value_lcb")
        self.assertEqual(overrides["nmccPiFullHorizonDecisions"], 2)
        self.assertEqual(overrides["nmccPiExhaustiveDecisions"], 2)
        self.assertGreater(overrides["nmccPiReplayMaxEpisodes"], 1)
        self.assertGreater(overrides["nmccPiReplayEpochs"], 1)
        self.assertGreater(overrides["nmccPiValidationFraction"], 0.0)
        self.assertTrue(overrides["nmccPiReplayRefit"])
        self.assertEqual(overrides["nmccPiBasePolicy"], "risk_reduction")
        self.assertEqual(overrides["actorPrior"], "risk_time_reduction")
        self.assertEqual(overrides["nmccPiValidationGainZ"], 2.0)
        self.assertEqual(curriculum.learner_overrides["actorPrior"], "risk_time_reduction")
        self.assertNotIn("pedVol", curriculum.learner_overrides)
        city = load_city_suite().select(("state_college_pa",))
        rollout = balanced_rollout_episodes(len(city))
        schedule = build_curriculum_schedule(
            city,
            curriculum,
            launch_seed=27,
            rollout_episodes=rollout,
        )
        self.assertEqual(len(schedule), 104)
        self.assertEqual(len(schedule) % rollout, 0)
        for attribute, _, kind, _ in NMCC_PI_CORE_FIELDS:
            self.assertIn(attribute, overrides)
            expected = (int, float) if kind is float else kind
            self.assertIsInstance(overrides[attribute], expected)

    def test_v28_scenario_curriculum_applies_one_learner_to_all_factor_combinations(self):
        curriculum = load_training_curriculum(
            ROOT / "config" / "scenario_general_training_curriculum_risk_v28.json"
        )
        self.assertEqual(curriculum.schema_version, 2)
        self.assertEqual(curriculum.episodes_per_city, 120)
        contracts = []
        scenarios = set()
        for stage in curriculum.stages:
            for variant in stage.variants:
                overrides = variant.overrides
                scenarios.add((
                    overrides["pedVol"], overrides["hazardVol"], overrides["panicRate"]
                ))
                contracts.append(tuple(
                    (attribute, overrides[attribute])
                    for attribute, _, _, _ in NMCC_PI_CORE_FIELDS
                ))
        expected_scenarios = {
            (population, hazard_count, panic)
            for population in (1000, 2500, 5000)
            for hazard_count in (1, 3, 5)
            for panic in (0.1, 0.5, 0.9)
        }
        self.assertEqual(scenarios, expected_scenarios)
        self.assertTrue(all(contract == contracts[0] for contract in contracts[1:]))
        cities = load_city_suite().select(None)
        rollout_episodes = balanced_rollout_episodes(len(cities))
        schedule = build_curriculum_schedule(
            cities,
            curriculum,
            launch_seed=26,
            rollout_episodes=rollout_episodes,
        )
        self.assertEqual(len(schedule), 120 * len(cities))
        self.assertEqual(len(schedule) % rollout_episodes, 0)

    def test_schedule_is_reproducible_city_balanced_and_rollout_complete(self):
        cities = load_city_suite().cities
        curriculum = load_training_curriculum(DEFAULT_TRAINING_CURRICULUM_PATH)
        rollout = balanced_rollout_episodes(len(cities))
        first = build_curriculum_schedule(
            cities,
            curriculum,
            launch_seed=91,
            rollout_episodes=rollout,
        )
        second = build_curriculum_schedule(
            cities,
            curriculum,
            launch_seed=91,
            rollout_episodes=rollout,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 600)
        for start in range(0, len(first), rollout):
            block = first[start : start + rollout]
            for city in cities:
                self.assertEqual(
                    sum(item.city.city_id == city.city_id for item in block),
                    2,
                )
            self.assertEqual(len({item.stage_id for item in block}), 1)

    def test_factor_randomization_stage_realizes_exact_variant_weights_per_city(self):
        cities = load_city_suite().cities
        curriculum = load_training_curriculum(DEFAULT_TRAINING_CURRICULUM_PATH)
        schedule = build_curriculum_schedule(
            cities,
            curriculum,
            launch_seed=19,
            rollout_episodes=10,
        )
        robust = [item for item in schedule if item.stage_id == "S4"]
        for city in cities:
            variants = [
                item.variant_id
                for item in robust
                if item.city.city_id == city.city_id
            ]
            self.assertEqual(variants.count("small_high_disruption"), 4)
            self.assertEqual(variants.count("low_medium_high"), 4)
            self.assertEqual(variants.count("center"), 4)
            self.assertEqual(variants.count("high_medium_low"), 4)
            self.assertEqual(variants.count("large_low_disruption"), 4)

    def test_protected_stage_override_is_rejected(self):
        payload = {
            "schema_version": 1,
            "curriculum_id": "invalid",
            "description": "invalid protected change",
            "stages": [
                {
                    "stage_id": "bad",
                    "label": "bad",
                    "episodes_per_city": 2,
                    "variants": [
                        {
                            "variant_id": "bad",
                            "weight": 1,
                            "overrides": {"pedVol": 10, "cellX": 4},
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected"):
                load_training_curriculum(path)

    def test_learner_overrides_cannot_change_between_variants(self):
        payload = {
            "schema_version": 1,
            "curriculum_id": "invalid-learner-drift",
            "description": "learner settings must be fixed",
            "stages": [
                {
                    "stage_id": "bad",
                    "label": "bad",
                    "episodes_per_city": 2,
                    "variants": [
                        {
                            "variant_id": "a",
                            "weight": 1,
                            "overrides": {
                                "pedVol": 10,
                                "nmccEnabled": True,
                            },
                        },
                        {
                            "variant_id": "b",
                            "weight": 1,
                            "overrides": {
                                "pedVol": 10,
                                "nmccEnabled": False,
                            },
                        },
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical"):
                load_training_curriculum(path)


if __name__ == "__main__":
    unittest.main()
