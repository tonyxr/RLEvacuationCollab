#!/usr/bin/env python3
"""Executable contract for direct PPO with exact NMCC credit."""

import csv
import os
import tempfile
import unittest

import numpy as np
import torch

import CounterfactualBranch as CB
import nmcc_testbed
from RLBridge import RLBridge


class DirectRLNMCCTests(unittest.TestCase):
    def test_objective_greedy_uses_exact_immediate_nmcc_objective_without_mutation(self):
        core = nmcc_testbed.build(
            grid=7,
            cell_x=3,
            cell_y=3,
            population=40,
            stop_time=12,
            spread_rate=(5, 2),
            casualty_rate=(40, 9),
            panic_rate=0.2,
            hazard_count=1,
            scenario_seed=31,
            spacing_m=120,
            candidate_count=8,
            shelter_capacity_token=10,
        )
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="objective_greedy",
            target_active_shelters=len(core.shelterDS.shelterList) + 1,
            shelter_action_interval=3,
            policy_seed=31,
        )
        observation = bridge.observation_builder.build(
            decision_index=0,
            simulation_time=1,
            remaining_deployments=bridge.remaining_deployments,
        )
        digest_before = CB.live_state_digest(core)
        values = bridge._objective_greedy_values(observation)
        decision = bridge._select_action(observation)
        digest_after = CB.live_state_digest(core)
        feasible = np.flatnonzero(observation.action_mask)
        effects = bridge._objective_greedy_brancher.candidate_effects(
            observation.candidate_cell_indices[feasible]
        )
        denominator = bridge.initial_population * bridge.horizon_minutes
        expected = np.asarray(
            [
                1.0
                / (
                    1.0
                    + effect.acted.hazard_weighted_time_to_safety_cost
                    / denominator
                )
                for effect in effects
            ]
        )
        self.assertEqual(digest_before, digest_after)
        self.assertTrue(np.isfinite(values[feasible]).all())
        np.testing.assert_allclose(values[feasible], expected)
        self.assertTrue(np.all((values[feasible] > 0.0) & (values[feasible] <= 1.0)))
        self.assertEqual(
            decision.action_index,
            int(feasible[int(np.argmax(values[feasible]))]),
        )

    def test_objective_greedy_complete_ledger_keeps_panicked_debt(self):
        core = nmcc_testbed.build(
            grid=7,
            cell_x=3,
            cell_y=3,
            population=40,
            stop_time=12,
            spread_rate=(5, 2),
            casualty_rate=(40, 9),
            panic_rate=0.2,
            hazard_count=1,
            scenario_seed=37,
            spacing_m=120,
            candidate_count=8,
            shelter_capacity_token=10,
        )
        pedestrians = list(core.pedDS.pedAgentList.values())
        pedestrians[0].terminated = True
        pedestrians[1].panicked = True
        core.pedDS.result["casualty"] = 1
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="objective_greedy",
            target_active_shelters=len(core.shelterDS.shelterList) + 1,
            shelter_action_interval=3,
            policy_seed=37,
        )
        brancher = CB.CounterfactualBrancher(
            core,
            horizon=0,
            reward_model=bridge.reward_model,
            initial_population=bridge.initial_population,
            episode_horizon=bridge.horizon,
        )
        evaluator = CB._ImmediateCandidateEvaluator(brancher)
        self.assertEqual(evaluator.rational_population, 38)
        self.assertEqual(evaluator.panicked_population, 1)
        self.assertEqual(evaluator.casualty_population, 1)
        self.assertAlmostEqual(
            evaluator.common_plain_person_time,
            bridge.horizon_minutes,
        )

    def test_rollout_policy_iteration_fits_simulator_graded_pairs(self):
        core = nmcc_testbed.build(
            grid=7,
            cell_x=3,
            cell_y=3,
            population=40,
            stop_time=12,
            spread_rate=(5, 2),
            casualty_rate=(40, 9),
            panic_rate=0.2,
            hazard_count=1,
            scenario_seed=7,
            spacing_m=120,
            candidate_count=8,
            shelter_capacity_token=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            bridge = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=len(core.shelterDS.shelterList) + 2,
                shelter_action_interval=3,
                policy_seed=7,
                rollout_episodes=1,
                actor_lr=1e-3,
                actor_epochs=1,
                critic_epochs=1,
                nmcc_policy_improvement=True,
                nmcc_pi_branch_horizon=3,
                nmcc_pi_workers=2,
                nmcc_pi_exhaustive_decisions=0,
                nmcc_pi_max_branches=2,
                nmcc_pi_actor_epochs=4,
                nmcc_pi_epsilon=0.1,
                exploration_rate_start=0.4,
                exploration_rate_end=0.4,
                action_temperature_start=1.5,
                action_temperature_end=1.5,
                counterfactual_credit=False,
                actor_prior="objective_greedy",
                actor_prior_scale=1.0,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "diagnostics.csv"),
            )
            with torch.no_grad():
                bridge.policy.actor_cell.weight.copy_(
                    torch.linspace(
                        -0.05,
                        0.05,
                        bridge.policy.actor_cell.weight.numel(),
                    ).reshape_as(bridge.policy.actor_cell.weight)
                )
            core.rl = bridge
            for step in range(1, int(core.stopTime)):
                CB.advance_one_timestep(core)
                bridge.step(
                    simulation_time=step,
                    is_terminal=(step == core.stopTime - 1),
                )

            self.assertTrue(bridge.traj)
            self.assertEqual(len(bridge.improvement_records), len(bridge.traj))
            self.assertTrue(
                all(
                    int(record["tape_count"]) == int(bridge.nmcc_pi_tapes)
                    for record in bridge.improvement_records
                )
            )
            for transition in bridge.traj:
                mask = transition.observation_history[-1].action_mask
                exact = transition.improvement_exact_mask.to(torch.bool)
                self.assertEqual(
                    int(exact.sum().item()),
                    min(2, int(mask.sum().item())),
                )
                self.assertTrue(bool(exact[int(transition.action.item())]))
                unbranched = mask & ~exact
                self.assertTrue(
                    torch.all(transition.improvement_advantage[unbranched] == 0.0)
                )
                target = transition.improvement_target
                self.assertAlmostEqual(float(target.sum().item()), 1.0, places=6)
                self.assertTrue(torch.all(target[~mask] == 0.0))
                torch.testing.assert_close(
                    target[unbranched],
                    transition.improvement_prior[unbranched],
                )

            episodes = bridge._group_episode_indices(bridge.traj)
            masks = torch.stack(
                [item.observation_history[-1].action_mask for item in bridge.traj]
            )
            selected, raw_logits, _ = bridge._evaluate_nmcc_actor_sequences(
                bridge.traj,
                episodes,
                action_temperature=1.0,
            )
            self.assertTrue(
                torch.equal(selected, torch.arange(len(bridge.traj)))
            )
            actor_prior = torch.softmax(
                bridge._safe_masked_logits(raw_logits, masks), dim=-1
            )
            stored_prior = torch.stack(
                [item.improvement_prior for item in bridge.traj]
            )
            torch.testing.assert_close(stored_prior, actor_prior)
            exploratory = bridge._exploratory_probabilities(
                raw_logits / 1.5,
                masks,
            )
            self.assertGreater(
                float(torch.max(torch.abs(exploratory - actor_prior)).item()),
                1e-5,
            )
            for index, transition in enumerate(bridge.traj):
                action = int(transition.action.item())
                self.assertAlmostEqual(
                    float(transition.log_probability.exp().item()),
                    float(exploratory[index, action].item()),
                    places=5,
                )

            fit_transitions = list(bridge.traj)
            fit_episodes = bridge._group_episode_indices(fit_transitions)
            fit_masks = torch.stack(
                [item.observation_history[-1].action_mask for item in fit_transitions]
            )
            behavior_entropy = float(np.mean(bridge.behavior_entropies))
            result = bridge.end_episode()
            post_actor = bridge._actor_distribution_diagnostics(
                fit_transitions,
                fit_episodes,
                fit_masks,
            )
            self.assertEqual(result["rollout_pi_target_coverage"], 1.0)
            self.assertEqual(result["rollout_pi_mean_graded_candidates"], 2.0)
            self.assertGreater(result["rollout_pi_mean_feasible_candidates"], 2.0)
            self.assertGreater(result["actor_gradient_steps"], 0.0)
            self.assertTrue(np.isfinite(result["policy_loss"]))
            self.assertGreater(result["actor_gradient_norm"], 0.0)
            self.assertGreater(result["actor_head_gradient_norm"], 0.0)
            self.assertGreaterEqual(result["rollout_pi_actor_kl"], 0.0)
            self.assertLessEqual(
                result["rollout_pi_target_kl_after"],
                result["rollout_pi_target_kl_before"],
            )
            self.assertLessEqual(
                result["rollout_pi_target_step_kl"],
                bridge.nmcc_pi_epsilon + 1e-6,
            )
            self.assertGreaterEqual(
                result["rollout_pi_weighted_pairwise_accuracy_after"],
                0.0,
            )
            self.assertAlmostEqual(
                result["behavior_policy_entropy"], behavior_entropy, places=7
            )
            self.assertAlmostEqual(
                result["actor_policy_entropy_after"],
                post_actor["entropy"],
                places=7,
            )
            self.assertAlmostEqual(
                result["actor_residual_rms_after_update"],
                post_actor["residual_rms"],
                places=7,
            )
            self.assertGreaterEqual(
                result["actor_residual_abs_max_after_update"],
                result["actor_residual_rms_after_update"],
            )
            payload = torch.load(
                bridge.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertNotIn("improvement_replay", payload)
            self.assertIn("actor_replay", payload)
            self.assertEqual(len(payload["actor_replay"]), 1)
            self.assertNotIn("continuation_replay", payload)
            self.assertEqual(len(bridge.actor_replay), 1)
            self.assertFalse(hasattr(bridge, "continuation_replay"))
            self.assertEqual(result["rollout_pi_actor_replay_episodes"], 1.0)
            self.assertEqual(result["rollout_pi_actor_replay_transitions"], 2.0)
            self.assertEqual(result["sequence_episodes"], 1.0)
            self.assertEqual(payload["model_signature"]["version"], 57)
            self.assertEqual(
                payload["model_signature"]["residual_correction"],
                "unbounded_linear_actor_readout",
            )
            self.assertFalse(
                payload["model_signature"]["counterfactual_rollout_contract"][
                    "correction_gate"
                ]
            )
            with open(bridge.diagnostics_path, newline="") as handle:
                header = next(csv.reader(handle))
            self.assertIn("rollout_pi_target_coverage", header)
            self.assertIn("rollout_pi_actor_kl", header)
            self.assertNotIn("approximate_kl", header)
            self.assertNotIn("value_loss", header)

    def test_rollout_policy_target_does_not_self_grade_with_critic(self):
        core = nmcc_testbed.build(
            grid=7,
            cell_x=3,
            cell_y=3,
            population=40,
            stop_time=12,
            spread_rate=(5, 2),
            casualty_rate=(40, 9),
            panic_rate=0.2,
            hazard_count=1,
            scenario_seed=71,
            spacing_m=120,
            candidate_count=8,
            shelter_capacity_token=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            bridge = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=len(core.shelterDS.shelterList) + 2,
                shelter_action_interval=3,
                policy_seed=71,
                rollout_episodes=1,
                actor_lr=1e-3,
                critic_lr=1e-3,
                actor_epochs=1,
                critic_epochs=2,
                nmcc_policy_improvement=True,
                nmcc_pi_branch_horizon=3,
                nmcc_pi_workers=1,
                nmcc_pi_continuation_enabled=False,
                nmcc_pi_continuation_loss_coef=1.0,
                nmcc_pi_actor_epochs=8,
                nmcc_pi_fit_tolerance=0.05,
                nmcc_pi_epsilon=0.1,
                counterfactual_credit=False,
                actor_prior="objective_greedy",
                actor_prior_scale=1.0,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "diagnostics.csv"),
            )
            core.rl = bridge
            critic_before = {
                name: parameter.detach().clone()
                for name, parameter in bridge.policy.named_parameters()
                if name.startswith("critic_heads.")
            }
            self.assertEqual(bridge._ensure_brancher().horizon, 3)
            self.assertEqual(bridge._ensure_reward_brancher().horizon, 0)
            for step in range(1, int(core.stopTime)):
                CB.advance_one_timestep(core)
                bridge.step(
                    simulation_time=step,
                    is_terminal=(step == core.stopTime - 1),
                )

            self.assertEqual(len(bridge.traj), 2)
            result = bridge.end_episode()
            self.assertGreater(
                result["rollout_pi_rollout_horizon_timesteps"], 3.0
            )
            self.assertNotIn("nmcc_pi_continuation_loss", result)
            self.assertTrue(
                all(
                    torch.equal(critic_before[name], parameter.detach())
                    for name, parameter in bridge.policy.named_parameters()
                    if name in critic_before
                )
            )
            self.assertEqual(result["critic_gradient_steps"], 0.0)

    def test_forward_nmcc_requires_exactly_one_decision_interval(self):
        core = nmcc_testbed.build(
            grid=7,
            cell_x=3,
            cell_y=3,
            population=20,
            stop_time=10,
            spread_rate=(5, 2),
            casualty_rate=(40, 9),
            panic_rate=0.0,
            hazard_count=1,
            scenario_seed=73,
            spacing_m=120,
            candidate_count=8,
            shelter_capacity_token=10,
        )
        with self.assertRaisesRegex(ValueError, "equal the shelter decision interval"):
            RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=len(core.shelterDS.shelterList) + 1,
                shelter_action_interval=3,
                nmcc_policy_improvement=True,
                nmcc_pi_branch_horizon=2,
                nmcc_pi_continuation_enabled=True,
            )

    def test_sampled_actions_train_the_actor_with_signed_exact_nmcc_credit(self):
        core = nmcc_testbed.build(
            grid=8,
            cell_x=3,
            cell_y=3,
            population=90,
            stop_time=22,
            spread_rate=(5, 2),
            casualty_rate=(40, 9),
            panic_rate=0.5,
            hazard_count=1,
            scenario_seed=19,
            spacing_m=120,
            candidate_count=12,
            shelter_capacity_token=10,
        )
        initial_shelters = len(core.shelterDS.shelterList)
        with tempfile.TemporaryDirectory() as directory:
            bridge = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=initial_shelters + 3,
                shelter_action_interval=4,
                policy_seed=19,
                rollout_episodes=1,
                actor_epochs=2,
                critic_epochs=2,
                minibatch_size=2,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "diagnostics.csv"),
                counterfactual_credit=True,
                counterfactual_horizon=4,
                counterfactual_weight=1.0,
                nmcc_joint_counterfactual_weight=1.0,
                nmcc_natural_loss_coef=0.0,
                nmcc_policy_improvement=False,
                actor_prior="none",
                actor_prior_scale=0.0,
                representation_mode="actor_owned",
            )
            core.rl = bridge
            actor_before = {
                name: parameter.detach().clone()
                for name, parameter in bridge.actor_named_parameters
            }

            for step in range(1, int(core.stopTime)):
                CB.advance_one_timestep(core)
                bridge.step(
                    simulation_time=step,
                    is_terminal=(step == core.stopTime - 1),
                )
            result = bridge.end_episode()

            self.assertTrue(np.isfinite(result["episode_return"]))
            self.assertNotAlmostEqual(result["episode_return"], 0.0)
            self.assertAlmostEqual(
                result["remaining_risk_time_reduction"],
                result["raw_remaining_risk_time_reduction"],
            )
            self.assertEqual(result["optimizer_updated"], 1.0)
            self.assertEqual(result["counterfactual_fraction"], 1.0)
            self.assertGreater(
                result["counterfactual_advantage_sd"],
                0.0,
            )
            self.assertGreater(
                result["counterfactual_positive_fraction"]
                + result["counterfactual_negative_fraction"],
                0.0,
            )
            self.assertGreater(result["actor_gradient_norm"], 0.0)
            self.assertEqual(result["actor_update_rejected"], 0.0)
            self.assertFalse(bridge.policy.auxiliary_models_enabled)
            self.assertNotIn("nmcc_pi_gate_passed", result)
            self.assertNotIn("nmcc_pi_validation_gain", result)
            self.assertTrue(
                any(
                    not torch.equal(actor_before[name], parameter.detach())
                    for name, parameter in bridge.actor_named_parameters
                )
            )
            payload = torch.load(
                bridge.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertFalse(
                any(key.startswith("improvement_") for key in payload)
            )
            signature = payload["model_signature"]
            self.assertNotIn("nmcc_policy_improvement", signature)
            self.assertEqual(
                signature["learner_hyperparameters"]["kl_control"],
                "diagnostic_only_for_direct_ppo",
            )
            self.assertEqual(
                signature["counterfactual_rollout_contract"]["reference_action"],
                "independent_categorical_draw_from_same_behavior_policy",
            )
            self.assertTrue(
                all(
                    "reference_action" in record
                    and "remaining_risk_time_reduction_difference" in record
                    for record in bridge.counterfactual_records
                )
            )


if __name__ == "__main__":
    unittest.main()
