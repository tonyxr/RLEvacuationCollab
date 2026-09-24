import json
import tempfile
import unittest
from pathlib import Path

from TrainingCurriculum import load_training_curriculum
from full_experiment_campaign import (
    DEFAULT_CURRICULUM,
    _behavior_gate_passes,
    _parse_args,
)


class ConvergenceFirstCampaignTests(unittest.TestCase):
    def test_default_curriculum_is_strictly_5000_pedestrians(self):
        curriculum = load_training_curriculum(DEFAULT_CURRICULUM)
        populations = {
            int(variant.overrides["pedVol"])
            for stage in curriculum.stages
            for variant in stage.variants
        }
        self.assertEqual(populations, {5000})
        hazards = {
            int(variant.overrides["hazardVol"])
            for stage in curriculum.stages
            for variant in stage.variants
        }
        panic_rates = {
            float(variant.overrides["panicRate"])
            for stage in curriculum.stages
            for variant in stage.variants
        }
        self.assertEqual(hazards, {3})
        self.assertEqual(panic_rates, {0.5})
        self.assertEqual(curriculum.episodes_per_city, 120)
        self.assertEqual(len(curriculum.stages), 1)

    def test_campaign_trains_exactly_one_policy(self):
        self.assertEqual(_parse_args([]).policy_replicates, 1)
        with self.assertRaises(SystemExit):
            _parse_args(["--policy-replicates", "2"])

    def test_behavior_gate_requires_multiple_held_out_scenarios(self):
        self.assertEqual(_parse_args([]).behavior_gate_replications_per_city, 5)
        with self.assertRaises(SystemExit):
            _parse_args(["--behavior-gate-replications-per-city", "1"])

    def test_behavior_gate_reads_only_explicit_learning_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning_assessment.json"
            self.assertFalse(_behavior_gate_passes(path))
            path.write_text(
                json.dumps({"qualifies_as_learning_well_cross_city": False}),
                encoding="utf-8",
            )
            self.assertFalse(_behavior_gate_passes(path))
            path.write_text(
                json.dumps({"qualifies_as_learning_well_cross_city": True}),
                encoding="utf-8",
            )
            self.assertTrue(_behavior_gate_passes(path))


if __name__ == "__main__":
    unittest.main()
