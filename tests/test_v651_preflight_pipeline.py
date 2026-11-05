"""Regression tests for the v65.1 preflight orchestrator."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from types import SimpleNamespace
from unittest import mock
import sys

import scripts.preflight_pipeline as preflight

from scripts.preflight_pipeline import (
    deterministic_torch_env,
    map_preflight_issues,
    micro_assembly_issues,
)


class V651PreflightPipelineTests(unittest.TestCase):
    def test_repository_root_is_available_to_script_entrypoint_imports(self):
        self.assertIn(str(preflight.ROOT), sys.path)

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

    def test_completed_eval_is_revalidated_and_replaces_stale_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            logs = report / "logs"
            summary = (
                root
                / preflight.ASSEMBLED_ID
                / "evaluation_episode_summary.csv"
            )
            summary.parent.mkdir(parents=True)
            summary.write_text("preserved evidence\n")
            report.mkdir()
            (report / "state.json").write_text(
                json.dumps(
                    {
                        "multicity_eval": {"status": "done"},
                        "_checks": [
                            {
                                "check": "multicity --eval-only on the assembled launch",
                                "status": "FAIL",
                                "detail": "stale pre-repair audit",
                                "utc": "earlier",
                            }
                        ],
                    }
                )
            )
            args = SimpleNamespace(redo=None, keep_going=False, eval_timeout_h=1)
            with (
                mock.patch.object(preflight, "RUNS", root),
                mock.patch.object(preflight, "REPORT", report),
                mock.patch.object(preflight, "LOGS", logs),
                mock.patch.object(
                    preflight, "multicity_eval_summary_issues", return_value=[]
                ) as validate,
            ):
                pipeline = preflight.Pipeline(args)
                pipeline.step5_multicity_eval()

            validate.assert_called_once_with(summary)
            matching = [
                row
                for row in pipeline.checks
                if row["check"]
                == "multicity --eval-only on the assembled launch"
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["status"], "PASS")
            self.assertIn("issues=none", matching[0]["detail"])


if __name__ == "__main__":
    unittest.main()
