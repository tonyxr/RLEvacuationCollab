import json
import tempfile
import unittest
from pathlib import Path

from CityProfiles import load_city_suite
from TrainingCurriculum import (
    DEFAULT_TRAINING_CURRICULUM_PATH,
    build_curriculum_schedule,
    load_training_curriculum,
)
from multicity_backtest import balanced_rollout_episodes


ROOT = Path(__file__).resolve().parents[1]


class TrainingCurriculumTests(unittest.TestCase):
    def test_v51_rollout_curricula_use_one_policy_iteration_stage(self):
        for filename, episodes, rollout, branches, tapes in (
            ("state_college_training_smoke_250_rollout_pi_v51.json", 4, 2, 4, 1),
            (
                "state_college_training_curriculum_2500_rollout_pi_v51.json",
                96,
                4,
                6,
                2,
            ),
        ):
            curriculum = load_training_curriculum(ROOT / "config" / filename)
            self.assertEqual([stage.stage_id for stage in curriculum.stages], ["P"])
            self.assertEqual(curriculum.episodes_per_city, episodes)
            overrides = curriculum.learner_overrides
            self.assertEqual(overrides["ppoRolloutEpisodes"], rollout)
            self.assertEqual(overrides["nmccPiMaxBranches"], branches)
            self.assertEqual(overrides["nmccPiTapes"], tapes)
            self.assertEqual(overrides["actorPrior"], "objective_greedy")
            self.assertEqual(overrides["actionTemperatureStart"], 1.0)
            self.assertEqual(overrides["actionTemperatureEnd"], 1.0)
            self.assertEqual(overrides["entropyCoefficientStart"], 0.0)
            self.assertEqual(overrides["entropyCoefficientEnd"], 0.0)

    def test_target_curricula_restore_decay_and_actor_update_scale(self):
        for filename, policy_stage_episodes in (
            ("state_college_training_smoke_250_target_v49.json", 6),
            ("state_college_training_curriculum_2500_target_v49.json", 80),
        ):
            curriculum = load_training_curriculum(ROOT / "config" / filename)
            overrides = curriculum.learner_overrides
            self.assertEqual(
                [stage.stage_id for stage in curriculum.stages], ["E", "P"]
            )
            self.assertEqual(
                curriculum.stages[-1].episodes_per_city,
                policy_stage_episodes,
            )
            self.assertEqual(
                overrides["explorationSchedule"], "exponential_half_life"
            )
            self.assertEqual(overrides["explorationRateStart"], 0.35)
            self.assertEqual(overrides["explorationRateEnd"], 0.02)
            self.assertEqual(overrides["actionTemperatureStart"], 1.25)
            self.assertEqual(overrides["actionTemperatureEnd"], 0.8)
            self.assertEqual(overrides["actorLearningRate"], 0.001)
            self.assertEqual(overrides["nmccPiReplayEpochs"], 24)

    def test_seattle_curriculum_has_three_exact_learning_stages(self):
        curriculum = load_training_curriculum(
            ROOT
            / "config"
            / "seattle_training_curriculum_population_foresight_v37.json"
        )
        self.assertEqual(curriculum.episodes_per_city, 160)
        self.assertEqual(
            [stage.stage_id for stage in curriculum.stages], ["E", "A", "F"]
        )
        self.assertEqual(
            [stage.episodes_per_city for stage in curriculum.stages], [24, 48, 88]
        )
        overrides = curriculum.learner_overrides
        self.assertEqual(
            overrides["cellPartitionMode"], "population_density_adaptive"
        )
        self.assertTrue(overrides["nmccPiContinuationEnabled"])
        self.assertEqual(overrides["nmccPiActorEpochs"], 256)
        self.assertEqual(overrides["nmccPiTapes"], 2)
        self.assertEqual(overrides["nmccPiWorkers"], 4)
        self.assertEqual(overrides["shelterCapacitySurplus"], 200)
        self.assertEqual(overrides["explorationSchedule"], "exponential_half_life")
        for stage in curriculum.stages:
            for variant in stage.variants:
                self.assertEqual(
                    10 * variant.overrides["shelterCapacityToken"],
                    variant.overrides["pedVol"] + 200,
                )

    def test_v37_curriculum_registers_decision_interval_continuation(self):
        curriculum = load_training_curriculum(
            ROOT
            / "config"
            / "state_college_training_curriculum_multiscale_env_action_v37.json"
        )
        overrides = curriculum.learner_overrides
        self.assertEqual(curriculum.episodes_per_city, 120)
        self.assertEqual(overrides["shelterActionInterval"], 15)
        self.assertEqual(overrides["nmccPiBranchHorizon"], 15)
        self.assertTrue(overrides["nmccPiContinuationEnabled"])
        self.assertEqual(overrides["nmccPiContinuationLossCoefficient"], 1.0)
        self.assertEqual(overrides["nmccPiActorEpochs"], 64)
        self.assertEqual(overrides["nmccPiFitTolerance"], 0.01)
        for stage in curriculum.stages:
            for variant in stage.variants:
                self.assertEqual(variant.overrides["initShelterVol"], 2)
                self.assertEqual(variant.overrides["maxAdditionalShelters"], 8)
                self.assertEqual(
                    10 * variant.overrides["shelterCapacityToken"],
                    variant.overrides["pedVol"],
                )

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
