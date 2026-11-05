"""Regression tests for the v65.1 preflight orchestrator."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from scripts.preflight_pipeline import (
    deterministic_torch_env,
    map_preflight_issues,
    micro_assembly_issues,
)


class V651PreflightPipelineTests(unittest.TestCase):
    def test_deterministic_training_pins_only_torch_openmp(self):
        base = {
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "7",
            "VECLIB_MAXIMUM_THREADS": "6",
        }
        observed = deterministic_torch_env(base)
        self.assertEqual(observed["OMP_NUM_THREADS"], "1")
        self.assertEqual(observed["OPENBLAS_NUM_THREADS"], "7")
        self.assertEqual(observed["VECLIB_MAXIMUM_THREADS"], "6")

    def test_complete_micro_assembly_need_not_converge(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "training_episode_summary.csv",
                "policies/policy_001/regional_policy.pt",
                "policies/policy_002/regional_policy.pt",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("evidence")
            (root / "experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "training_not_converged",
                        "training_policy_indices": [1, 2],
                        "training_shard_provenance": [{}, {}],
                    }
                )
            )
            status, issues = micro_assembly_issues(root)
            self.assertEqual(status, "training_not_converged")
            self.assertEqual(issues, [])

    def test_optional_zero_shot_timeout_is_warn_only(self):
        results = [
            {"city_id": "state_college_pa", "status": "ready"},
            {"city_id": "spokane_wa", "status": "ready"},
            {"city_id": "seattle_wa", "status": "ready"},
            {"city_id": "chicago_il", "status": "ready"},
            {"city_id": "malibu_ca", "status": "timeout", "error": ">1800s"},
        ]
        training, zero_shot = map_preflight_issues(results)
        self.assertEqual(training, [])
        self.assertEqual(zero_shot, ["malibu_ca:timeout (>1800s)"])

    def test_missing_training_city_is_a_hard_issue(self):
        results = [
            {"city_id": "state_college_pa", "status": "ready"},
            {"city_id": "seattle_wa", "status": "ready"},
            {"city_id": "chicago_il", "status": "ready"},
            {"city_id": "malibu_ca", "status": "ready"},
        ]
        training, zero_shot = map_preflight_issues(results)
        self.assertEqual(training, ["spokane_wa:missing"])
        self.assertEqual(zero_shot, [])


if __name__ == "__main__":
    unittest.main()
