"""v65 curricula equal the v64 ones except for the compute-only keys."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from TrainingCurriculum import load_training_curriculum

ROOT = Path(__file__).resolve().parents[1]
V65_KEYS = {
    "hazardForecastKernel": "vectorized",
    "forecastMemoEnabled": True,
    "branchTaskChunking": "tape_affine",
    "observationEveryMinute": False,
}


class V65ConfigTests(unittest.TestCase):
    def test_v65_curricula_differ_from_v64_only_in_compute_keys(self):
        for size, verify in (("2500_paper", 0.01), ("250_smoke", 0.1)):
            v64 = load_training_curriculum(
                ROOT / "config" / f"state_college_spokane_regime_v64_{size}.json"
            )
            v65 = load_training_curriculum(
                ROOT / "config" / f"state_college_spokane_regime_v65_{size}.json"
            )
            with self.subTest(size=size):
                self.assertIn("v65", v65.curriculum_id)
                overrides = dict(v65.learner_overrides)
                for name, value in V65_KEYS.items():
                    self.assertEqual(overrides.pop(name), value)
                self.assertEqual(overrides.pop("forecastMemoVerifyFraction"), verify)
                # v65.1: the launch-time wind override is declared in the file.
                self.assertEqual(overrides.pop("hazardWindSpeedMPerMinute"), 16.67)
                self.assertEqual(overrides, dict(v64.learner_overrides))
                self.assertEqual(
                    [
                        (stage.stage_id, stage.episodes_per_city, stage.label)
                        for stage in v65.stages
                    ],
                    [
                        (stage.stage_id, stage.episodes_per_city, stage.label)
                        for stage in v64.stages
                    ],
                )
                raw64 = json.loads(
                    (ROOT / "config" / f"state_college_spokane_regime_v64_{size}.json").read_text()
                )
                raw65 = json.loads(
                    (ROOT / "config" / f"state_college_spokane_regime_v65_{size}.json").read_text()
                )
                self.assertEqual(raw64["stages"], raw65["stages"])
                # Every stage variant carries the declared wind speed, so the
                # training and evaluation environments both receive it.
                for stage in v65.stages:
                    for variant in stage.variants:
                        self.assertEqual(
                            variant.overrides["hazardWindSpeedMPerMinute"], 16.67
                        )


if __name__ == "__main__":
    unittest.main()
