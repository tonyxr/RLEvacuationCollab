#!/usr/bin/env python3
import unittest

import numpy as np

import RolloutPolicyIteration as RPI


class RolloutPolicyIterationTests(unittest.TestCase):
    def test_rollout_panel_gives_every_leftover_cell_a_sampling_chance(self):
        feasible = np.arange(12, dtype=np.int64)
        scores = np.linspace(12.0, 1.0, 12)
        rng = np.random.default_rng(41)
        seen = set()
        for _ in range(100):
            panel = RPI.select_rollout_panel(
                feasible,
                scores,
                max_branches=6,
                uniform_extras=2,
                rng=rng,
                must_include=(0,),
            )
            self.assertEqual(len(panel), 6)
            self.assertEqual(len(set(panel.tolist())), 6)
            self.assertIn(0, panel.tolist())
            seen.update(panel.tolist())
        self.assertEqual(seen, set(feasible.tolist()))

    def test_rollout_panel_includes_executed_and_base_actions(self):
        panel = RPI.select_rollout_panel(
            np.arange(10),
            np.arange(10, dtype=float),
            max_branches=5,
            uniform_extras=2,
            rng=np.random.default_rng(8),
            must_include=(0, 9),
        )
        self.assertIn(0, panel.tolist())
        self.assertIn(9, panel.tolist())

    def test_zero_branch_budget_is_exhaustive(self):
        feasible = np.asarray((1, 3, 4, 8), dtype=np.int64)
        panel = RPI.select_rollout_panel(
            feasible,
            np.arange(4, dtype=float),
            max_branches=0,
            uniform_extras=2,
            rng=np.random.default_rng(3),
        )
        np.testing.assert_array_equal(panel, feasible)

    def test_cost_estimate_prefers_lower_complete_cost(self):
        costs = np.asarray(
            (
                (0.42, 0.31, 0.37),
                (0.44, 0.30, 0.36),
                (0.43, 0.32, 0.38),
            ),
            dtype=np.float64,
        )
        target = RPI.estimate_rollout_costs(costs)
        self.assertEqual(int(np.argmax(target.score)), 1)
        self.assertGreaterEqual(target.paired_contrast_standard_error, 0.0)
        self.assertAlmostEqual(float(target.score.mean()), 0.0, places=6)

    def test_paired_noise_uses_common_future_contrasts(self):
        costs = np.asarray(
            (
                (0.20, 0.30, 0.40),
                (0.21, 0.31, 0.80),
                (0.19, 0.29, 0.10),
                (0.20, 0.30, 0.70),
            ),
            dtype=np.float64,
        )
        target = RPI.estimate_rollout_costs(costs)
        self.assertGreater(target.paired_contrast_standard_error, 0.0)

    def test_cost_estimate_preserves_between_state_effect_size(self):
        near_tie = np.asarray(
            ((0.500, 0.501), (0.510, 0.511), (0.490, 0.491)),
            dtype=np.float64,
        )
        clear_win = near_tie.copy()
        clear_win[:, 1] = clear_win[:, 0] + 0.100
        near = RPI.estimate_rollout_costs(near_tie)
        clear = RPI.estimate_rollout_costs(clear_win)
        self.assertGreater(
            float(np.ptp(clear.score)),
            50.0 * float(np.ptp(near.score)),
        )

    def test_cost_estimate_requires_replicated_shared_futures(self):
        with self.assertRaisesRegex(ValueError, "at least two shared futures"):
            RPI.estimate_rollout_costs(np.asarray(((0.2, 0.3),)))


if __name__ == "__main__":
    unittest.main()
