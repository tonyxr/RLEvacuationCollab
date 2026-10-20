import unittest

import numpy as np

import CounterfactualBranch as CB
import NMCCPolicyImprovement as NPI
import RolloutPolicyIteration as RPI
import nmcc_testbed
from RewardProcessor import RewardProcessor


class V53ROCBATests(unittest.TestCase):
    def test_budget_is_exact_and_best_matches_exhaustive_on_audit_state(self):
        core = nmcc_testbed.build(
            grid=6, cell_x=4, cell_y=4, population=60, stop_time=24,
            hazard_count=2, candidate_count=32, scenario_seed=5331,
        )
        runner = CB.CounterfactualBrancher(
            core,
            horizon=4,
            reward_model=RewardProcessor(),
            initial_population=60,
            episode_horizon=24,
            workers=1,
        )
        cells = core.feasible_cells()[:3]
        seeds = (101, 211, 307, 401)
        snapshot = CB.capture(core, label="v53r-ocba")
        exhaustive = runner.candidate_effects_many_futures(
            cells,
            future_seeds=seeds,
            snapshot=snapshot,
            allocation="uniform",
        )
        exhaustive_mean = np.mean(
            [
                [
                    effect.waited.hazard_weighted_time_to_safety_cost
                    - effect.acted.hazard_weighted_time_to_safety_cost
                    for effect in table
                ]
                for table in exhaustive
            ],
            axis=0,
        )
        adaptive = runner.candidate_effects_many_futures(
            cells,
            future_seeds=seeds,
            snapshot=snapshot,
            allocation="ocba",
            futures_min=2,
        )
        self.assertEqual(int(adaptive.futures_spent.sum()), len(cells) * len(seeds))
        np.testing.assert_array_equal(
            adaptive.futures_spent,
            np.full(len(cells), len(seeds), dtype=np.int64),
        )
        np.testing.assert_allclose(adaptive.mean_advantage, exhaustive_mean)
        self.assertTrue(all(len(effects) == len(seeds) for effects in adaptive.effects))
        self.assertEqual(
            adaptive.selected_cell,
            int(cells[int(np.argmax(exhaustive_mean))]),
        )

    def test_rollout_legacy_ocba_uses_one_complete_paired_rectangle(self):
        cells = (3, 7, 11)
        seeds = (101, 211, 307, 401)
        common_future = np.asarray((10.0, 100.0, 30.0, 250.0))[:, None]
        action_cost = np.asarray((0.30, 0.10, 0.20))[None, :]
        costs = common_future + action_cost

        class StubBrancher:
            def __init__(self):
                self.calls = []

            def rollout_policy_costs(self, requested_cells, **kwargs):
                self.calls.append(
                    (
                        tuple(requested_cells),
                        tuple(kwargs["future_seeds"]),
                        kwargs["mode"],
                    )
                )
                return costs.copy()

        brancher = StubBrancher()
        estimate = CB.CounterfactualBrancher.rollout_policy_costs_paired(
            brancher,
            cells,
            snapshot=object(),
            future_seeds=seeds,
            decision_index=2,
            remaining_deployments=4,
            maximum_deployments=8,
            frozen_policy=object(),
            mode="belief",
            futures_min=2,
        )
        authoritative = RPI.estimate_rollout_costs(costs)
        self.assertEqual(brancher.calls, [(cells, seeds, "belief")])
        np.testing.assert_allclose(estimate.mean_cost, authoritative.mean_cost)
        np.testing.assert_allclose(estimate.score, authoritative.score)
        np.testing.assert_array_equal(
            estimate.futures_spent,
            np.full(len(cells), len(seeds), dtype=np.int64),
        )
        self.assertAlmostEqual(estimate.paired_contrast_standard_error, 0.0)
        self.assertGreater(float(np.mean(estimate.standard_error)), 20.0)

    def test_legacy_name_is_reported_as_paired_uniform(self):
        self.assertEqual(NPI.effective_future_allocation("ocba"), "paired_uniform")
        self.assertEqual(NPI.effective_future_allocation("uniform"), "paired_uniform")


if __name__ == "__main__":
    unittest.main()
