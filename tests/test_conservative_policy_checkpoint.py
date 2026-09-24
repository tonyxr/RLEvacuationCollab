import unittest

import torch

from conservative_policy_checkpoint import derive_payload


class ConservativeCheckpointTests(unittest.TestCase):
    def test_derivation_scales_only_final_actor_and_blocks_resume(self):
        payload = {
            "policy_state_dict": {
                "actor_cell.weight": torch.tensor([[2.0, -4.0]]),
                "actor_cell.bias": torch.tensor([1.0]),
                "critic.0.weight": torch.tensor([[3.0]]),
            },
            "training_resume_allowed": True,
        }
        result = derive_payload(payload, 0.25, "abc")
        torch.testing.assert_close(
            result["policy_state_dict"]["actor_cell.weight"],
            torch.tensor([[0.5, -1.0]]),
        )
        torch.testing.assert_close(
            result["policy_state_dict"]["actor_cell.bias"],
            torch.tensor([0.25]),
        )
        torch.testing.assert_close(
            result["policy_state_dict"]["critic.0.weight"],
            torch.tensor([[3.0]]),
        )
        self.assertFalse(result["training_resume_allowed"])
        self.assertEqual(
            result["checkpoint_derivation"]["parent_checkpoint_sha256"], "abc"
        )
        self.assertEqual(payload["policy_state_dict"]["actor_cell.bias"].item(), 1.0)

    def test_derivation_rejects_invalid_or_chained_scaling(self):
        payload = {
            "policy_state_dict": {
                "actor_cell.weight": torch.ones(1, 1),
                "actor_cell.bias": torch.ones(1),
            }
        }
        with self.assertRaises(ValueError):
            derive_payload(payload, 1.1, "abc")
        payload["checkpoint_derivation"] = {"method": "previous"}
        with self.assertRaises(ValueError):
            derive_payload(payload, 0.5, "abc")


if __name__ == "__main__":
    unittest.main()
