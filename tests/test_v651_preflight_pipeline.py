"""Regression tests for the v65.1 preflight orchestrator."""

from __future__ import annotations

import unittest

from scripts.preflight_pipeline import deterministic_torch_env, map_preflight_issues


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
