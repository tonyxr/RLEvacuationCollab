import tempfile
import unittest
from pathlib import Path

import torch

from DecisionInterface import (
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    INFRA_FEATURE_NAMES,
    PED_FEATURE_NAMES,
)
from GNN import EvacPolicy
from policy_sensitivity_audit import audit_policy


class PolicySensitivityAuditTests(unittest.TestCase):
    def _checkpoint(self, directory: str, *, trained_actor: bool) -> Path:
        policy = EvacPolicy(
            d_ped=len(PED_FEATURE_NAMES),
            d_hazard=len(HAZARD_FEATURE_NAMES),
            d_infra=len(INFRA_FEATURE_NAMES),
            d_global=len(GLOBAL_FEATURE_NAMES),
            d_candidate=len(CANDIDATE_FEATURE_NAMES),
        )
        if trained_actor:
            with torch.no_grad():
                policy.actor_cell.weight.fill_(0.1)
        path = Path(directory) / "policy.pt"
        torch.save(
            {
                "model_signature": {
                    "version": 10,
                    "grid_shape": (2, 2),
                    "cell_features": CELL_FEATURE_NAMES,
                    "global_features": GLOBAL_FEATURE_NAMES,
                    "candidate_features": CANDIDATE_FEATURE_NAMES,
                    "architecture": "test",
                },
                "policy_state_dict": policy.state_dict(),
            },
            path,
        )
        return path

    def test_zero_residual_actor_fails_required_dynamic_state_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_policy(self._checkpoint(directory, trained_actor=False))
        self.assertTrue(result["state_family_checks"]["dynamic_population"])
        self.assertFalse(result["state_family_checks"]["hazard_and_forecast"])
        self.assertFalse(
            result["state_family_checks"]["physical_environment_and_shelters"]
        )
        self.assertFalse(result["all_required_state_families_responsive"])

    def test_nonzero_actor_produces_finite_counterfactual_report(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_policy(self._checkpoint(directory, trained_actor=True))
        self.assertEqual(result["grid_shape"], [2, 2])
        self.assertEqual(
            set(result["local_feature_counterfactuals"]),
            set(CELL_FEATURE_NAMES),
        )
        for item in result["local_feature_counterfactuals"].values():
            self.assertTrue(torch.isfinite(torch.tensor(item["total_logit_contrast_delta"])))


if __name__ == "__main__":
    unittest.main()
