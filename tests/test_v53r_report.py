import math
import unittest

from scripts.value_decomposition import _summarize


class ValueDecompositionReportTests(unittest.TestCase):
    def test_decomposition_and_rl_seed_inference_use_matched_tapes(self):
        offsets = {
            "objective_greedy": 0.10,
            "regime_oracle_greedy": 0.06,
            "belief_rollout": 0.07,
            "hindsight_ceiling": 0.00,
            "deadline_greedy": 0.09,
            "rl": 0.05,
            "rl_forecast_ablated": 0.08,
        }
        rows = []
        for tape in range(4):
            for policy, offset in offsets.items():
                seeds = range(5) if policy.startswith("rl") else range(1)
                for policy_seed in seeds:
                    rows.append({
                        "scenario": "synthetic",
                        "tape": tape,
                        "policy_seed": policy_seed,
                        "policy": policy,
                        "normalized_cost": 0.5 + offset + 0.001 * tape,
                    })

        summary = _summarize(rows)

        self.assertEqual(len(summary), 1)
        result = summary[0]
        self.assertEqual(result["best_tuned_rule"], "deadline_greedy")
        self.assertAlmostEqual(result["adaptation_value"], 0.04)
        self.assertAlmostEqual(result["foresight_value"], 0.03)
        self.assertAlmostEqual(result["ceiling_value"], 0.10)
        self.assertAlmostEqual(result["rl_vs_best_improvement"], 0.04)
        self.assertAlmostEqual(result["rl_share"], 4.0 / 9.0)
        self.assertTrue(result["gate_1_pass"])
        self.assertTrue(math.isfinite(result["rl_vs_best_holm_p"]))


if __name__ == "__main__":
    unittest.main()
