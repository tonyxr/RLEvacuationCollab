import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

import CounterfactualBranch as CB
import NMCCPolicyImprovement as NPI
import RolloutPolicyIteration as RPI
from RLBridge import ForwardRewardBreakdown, RLBridge
from DecisionInterface import RemainingTimeEstimate, estimate_clearance_tail


class RandomReferenceRewardTests(unittest.TestCase):
    def test_uniform_reference_sampling_keeps_chosen_and_uses_branch_budget(self):
        feasible = np.arange(10, dtype=np.int64)
        selected = NPI.select_uniform_reference_actions(
            feasible,
            max_branches=4,
            must_include=(7,),
            rng=np.random.default_rng(19),
        )
        self.assertEqual(selected.size, 4)
        self.assertIn(7, selected.tolist())
        self.assertEqual(np.unique(selected).size, selected.size)
        self.assertTrue(np.isin(selected, feasible).all())

    def test_random_reference_advantage_expands_sample_to_uniform_feasible_mean(self):
        actions = np.asarray([1, 4, 7], dtype=np.int64)
        values = np.asarray([2.0, -1.0, 0.0], dtype=np.float64)
        advantages, baseline = NPI.uniform_reference_advantages(
            actions,
            values,
            feasible_count=10,
            chosen_action=7,
        )
        self.assertAlmostEqual(baseline, 0.45)
        np.testing.assert_allclose(advantages, values - 0.45)
        self.assertAlmostEqual(advantages[2], -0.45)

    def test_complete_table_uses_exact_uniform_mean(self):
        actions = np.arange(4, dtype=np.int64)
        values = np.asarray([0.1, 0.4, -0.2, 0.3], dtype=np.float64)
        advantages, baseline = NPI.uniform_reference_advantages(
            actions,
            values,
            feasible_count=4,
            chosen_action=1,
        )
        self.assertAlmostEqual(baseline, float(values.mean()))
        self.assertAlmostEqual(float(advantages.mean()), 0.0)

    def test_better_and_worse_than_random_actions_keep_their_sign(self):
        actions = np.arange(5, dtype=np.int64)
        # Values are savings, so the cell below the uniform mean must remain a
        # genuinely negative training example rather than being clipped away.
        values = np.asarray([3.0, 1.0, 0.0, -1.0, -3.0])
        advantages, baseline = NPI.uniform_reference_advantages(
            actions,
            values,
            feasible_count=5,
            chosen_action=0,
        )
        self.assertAlmostEqual(baseline, 0.0)
        self.assertGreater(advantages[0], 0.0)
        self.assertLess(advantages[-1], 0.0)
        self.assertAlmostEqual(float(advantages.mean()), 0.0)

    def test_reference_potential_reward_telescopes_to_savings(self):
        intervals = [
            ForwardRewardBreakdown(
                reward=0.3,
                random_reference_value_before=1.0,
                random_reference_value_after=0.5,
                plain_interval_cost=0.2,
                exposure_interval_cost=0.0,
                expected_casualty_interval_cost=0.0,
                terminal_plain_residual_cost=0.0,
                terminal_exposure_residual_cost=0.0,
                decision_discount=1.0,
            ),
            ForwardRewardBreakdown(
                reward=0.2,
                random_reference_value_before=0.5,
                random_reference_value_after=0.0,
                plain_interval_cost=0.3,
                exposure_interval_cost=0.0,
                expected_casualty_interval_cost=0.0,
                terminal_plain_residual_cost=0.0,
                terminal_exposure_residual_cost=0.0,
                decision_discount=1.0,
            ),
        ]
        self.assertAlmostEqual(sum(item.reward for item in intervals), 0.5)
        self.assertAlmostEqual(
            sum(item.reference_potential_reduction for item in intervals),
            1.0,
        )
        self.assertAlmostEqual(
            sum(item.reward for item in intervals),
            intervals[0].random_reference_value_before
            - sum(item.interval_cost for item in intervals),
        )

    def test_random_reference_cost_to_go_is_complete_backward_sum(self):
        bridge = object.__new__(RLBridge)
        bridge.gamma = 1.0
        targets = bridge._cost_to_go_targets(
            torch.tensor([0.2, 0.3, 0.4]),
            torch.tensor([0.0, 0.0, 1.0]),
            torch.tensor([15.0, 15.0, 15.0]),
        )
        torch.testing.assert_close(targets, torch.tensor([0.9, 0.7, 0.4]))

    def test_rollout_target_uses_complete_simulated_costs_and_uncertainty(self):
        costs = np.asarray(
            [
                [100.0, 104.0, 102.0],
                [90.0, 92.0, 91.0],
                [110.0, 116.0, 113.0],
            ],
            dtype=np.float64,
        ).T
        target = RPI.estimate_rollout_costs(costs)
        self.assertEqual(int(np.argmax(target.score)), 1)
        self.assertEqual(int(np.argmin(target.score)), 2)
        self.assertAlmostEqual(float(np.mean(target.score)), 0.0, places=7)
        self.assertGreaterEqual(target.paired_contrast_standard_error, 0.0)

    def test_foresight_interval_cost_excludes_endpoint_remaining_debt(self):
        result = CB.BranchResult(
            label="candidate",
            active_person_time=30.0,
            hazard_exposure_person_time=12.0,
            remaining_plain_person_time=400.0,
            remaining_hazard_exposure_person_time=80.0,
        )
        self.assertEqual(result.interval_hazard_weighted_person_time, 42.0)
        self.assertEqual(result.hazard_weighted_time_to_safety_cost, 522.0)

    def test_clearance_tail_uses_slow_cohort_and_never_rewards_a_casualty(self):
        histogram = defaultdict(int, {10.0: 1, 20.0: 1, 30.0: 1, 40.0: 1})
        core = SimpleNamespace(
            timeStepMinutes=1.0,
            clearanceTailFraction=0.20,
            pedDS=SimpleNamespace(
                currTime=50,
                result={"casualty": 1},
                safeCompletionTimeHistogram=histogram,
            ),
        )
        remaining = RemainingTimeEstimate(
            plain_person_minutes=150.0,
            hazard_exposure_person_minutes=0.0,
            plain_minutes_by_person=np.asarray([10, 20, 30, 40, 50]),
        )
        tail = estimate_clearance_tail(
            core,
            episode_horizon=180,
            initial_population=10,
            remaining=remaining,
        )
        self.assertEqual(tail.tail_population, 2)
        self.assertEqual(tail.clearance_time_minutes, 360.0)
        self.assertEqual(tail.t90_minutes, 100.0)
        self.assertEqual(tail.tail_person_minutes, 460.0)
        self.assertEqual(tail.tail_mean_minutes, 230.0)

    def test_branch_cost_includes_enabled_clearance_tail_once(self):
        result = CB.BranchResult(
            label="candidate",
            active_person_time=30.0,
            hazard_exposure_person_time=12.0,
            expected_casualty_person_time=5.0,
            clearance_tail_person_time=20.0,
        )
        self.assertEqual(result.hazard_weighted_time_to_safety_cost, 67.0)


if __name__ == "__main__":
    unittest.main()
