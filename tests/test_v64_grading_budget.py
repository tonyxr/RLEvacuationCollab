"""v64 stage-dependent rollout grading budget tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import CounterfactualBranch as CB
import NMCCPolicyImprovement as NPI
from RLBridge import RLBridge
from nmcc_testbed import build


def _bridge(directory: str, core, **overrides) -> RLBridge:
    values = dict(
        train_mode=True,
        deployment_strategy="rl",
        target_active_shelters=len(core.shelterDS.shelterList) + 3,
        shelter_action_interval=3,
        first_decision_time=1,
        policy_seed=7,
        rollout_episodes=1,
        actor_lr=1e-3,
        actor_epochs=1,
        critic_epochs=1,
        nmcc_policy_improvement=True,
        nmcc_pi_tapes=4,
        nmcc_pi_late_tapes=2,
        nmcc_pi_early_decisions=1,
        nmcc_pi_workers=1,
        nmcc_pi_exhaustive_decisions=1,
        nmcc_pi_max_branches=4,
        nmcc_pi_actor_epochs=1,
        nmcc_pi_replay_epochs=1,
        checkpoint_path=os.path.join(directory, "policy.pt"),
        diagnostics_path=os.path.join(directory, "diagnostics.csv"),
    )
    values.update(overrides)
    return RLBridge(core, **values)


class V64GradingBudgetTests(unittest.TestCase):
    def test_early_and_late_decisions_use_configured_shared_tapes(self):
        core = build(
            stop_time=9,
            population=30,
            group_size=10,
            spacing_m=120,
            candidate_count=8,
            shelter_capacity_token=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(directory, core)
            core.rl = bridge
            observed_seeds = []
            original = NPI.region_defer_panel_costs

            def recording_panel(*args, **kwargs):
                observed_seeds.append(tuple(kwargs["future_seeds"]))
                return original(*args, **kwargs)

            with mock.patch.object(
                NPI,
                "region_defer_panel_costs",
                side_effect=recording_panel,
            ):
                for step in range(1, int(core.stopTime)):
                    CB.advance_one_timestep(core)
                    bridge.step(
                        simulation_time=step,
                        is_terminal=(step == core.stopTime - 1),
                    )

            self.assertGreaterEqual(len(bridge.improvement_records), 2)
            self.assertEqual(len(observed_seeds), len(bridge.improvement_records))
            for record, seeds in zip(bridge.improvement_records, observed_seeds):
                decision_index = int(record["decision_index"])
                expected_count = 4 if decision_index < 1 else 2
                self.assertEqual(int(record["tape_count"]), expected_count)
                self.assertEqual(len(seeds), expected_count)
                self.assertEqual(
                    seeds,
                    tuple(
                        NPI.tape_seed(
                            int(core.scenario_seed) ^ 0x524F4C4C,
                            decision_index,
                            tape_index,
                        )
                        for tape_index in range(expected_count)
                    ),
                )
                self.assertEqual(
                    int(record["future_samples_total"]),
                    int(record["branched"]) * expected_count,
                )

            first_record = bridge.improvement_records[0]
            self.assertEqual(
                int(first_record["branched"]), int(first_record["feasible"])
            )
            later_transition = bridge.traj[1]
            later_record = bridge.improvement_records[1]
            exact = later_transition.improvement_exact_mask.numpy() > 0.5
            self.assertEqual(int(np.sum(exact)), 4)
            self.assertTrue(exact[int(later_transition.action.item())])
            self.assertTrue(exact[int(later_record["base_action"])])

    def test_late_tape_budget_of_one_is_rejected(self):
        core = build(population=20, group_size=10, candidate_count=8)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "zero.*at least two"):
                _bridge(directory, core, nmcc_pi_late_tapes=1)


if __name__ == "__main__":
    unittest.main()
