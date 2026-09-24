#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RLBridge/GNN-side tests for NMCC policy improvement (NMCC-PI).

These require torch and therefore the project's pinned ``rlevacuation``
environment.  The torch-free properties (exact KL of the E-step target,
monotone movement toward higher exact value, CRN pairing, no disturbance of
the live episode, tapes independent of the live future) are covered in
``tests/test_nmcc_policy_improvement.py``.

What is pinned here is the learner contract:

* NMCC-PI is inert unless switched on, and switching it on changes only the
  actor target, not the executed-action semantics;
* the intervention-value loss is blind to any cell-independent level, so the
  natural momentum of the scenario cannot be learned as intervention value;
* one training episode on the synthetic testbed produces exact targets at
  every decision and an M-step that moves the policy *toward* the target and
  toward the branch-best cell while staying inside the trust region.
"""

import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import GNN
import NMCCPIConfig
import nmcc_testbed
from RLBridge import RLBridge


def make_core():
    return nmcc_testbed.build(
        grid=4,
        cell_x=2,
        cell_y=2,
        population=12,
        stop_time=11,
        spread_rate=(4, 2),
        casualty_rate=(40, 9),
        panic_rate=0.5,
        hazard_count=1,
        scenario_seed=11,
        spacing_m=100,
        candidate_count=8,
        shelter_capacity_token=6,
    )


def bridge(**kwargs):
    scratch = tempfile.mkdtemp(prefix="rlevac-nmcc-test-")
    defaults = dict(
        train_mode=True,
        deployment_strategy="rl",
        target_active_shelters=4,
        shelter_action_interval=5,
        policy_seed=11,
        rollout_episodes=1,
        checkpoint_path=os.path.join(scratch, "policy.pt"),
        diagnostics_path=os.path.join(scratch, "diagnostics.csv"),
    )
    defaults.update(kwargs)
    return RLBridge(make_core(), **defaults)


class ContractTests(unittest.TestCase):
    def test_config_default_matches_the_network_constant(self):
        self.assertEqual(NMCCPIConfig.DEFAULT_ACTOR_PRIOR_SCALE, GNN.HEURISTIC_PRIOR_SCALE)

    def test_policy_improvement_is_off_by_default(self):
        rl = bridge()
        self.assertFalse(rl.nmcc_policy_improvement)
        self.assertEqual(rl.actor_prior, "active_population")
        self.assertFalse(rl._model_signature()["nmcc_policy_improvement"]["enabled"])

    def test_invalid_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            bridge(nmcc_pi_base_policy="oracle")
        with self.assertRaises(ValueError):
            bridge(nmcc_pi_max_branches=1)
        with self.assertRaises(ValueError):
            bridge(actor_prior="unknown")
        with self.assertRaises(ValueError):
            # A trust region tighter than the E-step step would forbid the
            # policy from ever reaching its own target.
            bridge(nmcc_policy_improvement=True, nmcc_pi_epsilon=0.5, nmcc_pi_kl_cap=0.4)

    def test_actor_prior_enters_the_signature(self):
        rl = bridge(actor_prior="route_time_saving")
        signature = rl._model_signature()
        self.assertEqual(signature["actor_prior"]["kind"], "route_time_saving")
        self.assertIn("actor_prior", rl._inference_signature(signature))

    def test_value_controller_rejects_a_mismatched_base_prior(self):
        with self.assertRaisesRegex(ValueError, "same base rule"):
            bridge(
                nmcc_policy_improvement=True,
                nmcc_pi_actor_objective="value_lcb",
                nmcc_pi_base_policy="risk_reduction",
                actor_prior="active_population",
            )

    def test_replay_gate_is_disjoint_from_fit_and_early_stopping(self):
        rl = bridge(nmcc_policy_improvement=True)
        rl.improvement_replay = [
            {"episode_id": episode_id, "transitions": [object()]}
            for episode_id in range(20)
        ]
        training, validation, gate = rl._split_improvement_replay()
        partitions = [
            {item["episode_id"] for item in records}
            for records in (training, validation, gate)
        ]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        self.assertEqual(set.union(*partitions), set(range(20)))

    def test_staged_score_controller_and_schedules_are_compatible(self):
        rl = bridge(
            nmcc_policy_improvement=True,
            nmcc_natural_pretrain_rollouts=2,
            nmcc_causal_pretrain_rollouts=3,
            nmcc_controller_warmup_rollouts=2,
            nmcc_pi_actor_objective="score_ranking",
            exploration_rate_start=0.4,
            exploration_rate_end=0.05,
            learning_rate_schedule="cosine",
            lr_warmup_updates=1,
            lr_decay_updates=10,
        )
        self.assertEqual(rl._current_nmcc_training_phase(), "natural_pretrain")
        self.assertFalse(rl._actor_training_enabled())
        self.assertAlmostEqual(rl._current_exploration_rate(), 0.4)
        rl.rollout_updates_completed = 2
        self.assertEqual(rl._current_nmcc_training_phase(), "causal_pretrain")
        rl.rollout_updates_completed = 5
        self.assertEqual(rl._current_nmcc_training_phase(), "controller_warmup")
        self.assertTrue(rl._actor_training_enabled())
        initial_actor, initial_critic = rl._apply_learning_rate_schedule()
        rl.rollout_updates_completed = 15
        later_actor, later_critic = rl._apply_learning_rate_schedule()
        self.assertLess(later_actor, initial_actor)
        self.assertLess(later_critic, initial_critic)
        signature = rl._model_signature()
        self.assertEqual(
            signature["nmcc_policy_improvement"]["actor_objective"],
            "score_ranking",
        )

    def test_value_controller_stays_on_prior_until_heldout_gain_gate_opens(self):
        rl = bridge(
            nmcc_policy_improvement=True,
            nmcc_pi_actor_objective="value_lcb",
            actor_prior="risk_time_reduction",
        )
        rl.observation_builder.maximum_shelter_forecast_danger = 1.0
        observation = rl.observation_builder.build(
            decision_index=0,
            simulation_time=1,
            remaining_deployments=rl.remaining_deployments,
        )
        step = rl._advance_recurrent_observation(
            observation, cache_for_training=False
        )
        action_mask = torch.ones_like(step.frame.action_mask, dtype=torch.bool)
        closed = rl._controller_logits(step, action_mask)
        expected = rl._safe_masked_logits(
            step.prior_logits, action_mask.reshape(1, -1)
        )
        torch.testing.assert_close(closed, expected)

        feasible = torch.nonzero(action_mask).flatten()
        self.assertGreaterEqual(feasible.numel(), 2)
        base = int(expected.argmax(dim=-1).item())
        selected = next(int(item) for item in feasible.tolist() if int(item) != base)
        samples = torch.zeros_like(step.improvement_value_samples)
        samples[:, :, selected] = 100.0
        step.improvement_value_samples = samples
        rl.improvement_gate_history = [0.01] * rl.nmcc_pi_gate_updates
        opened = rl._controller_logits(step, action_mask)
        self.assertEqual(int(opened.argmax(dim=-1).item()), selected)
        initial_exploration = rl._current_exploration_rate()
        rl.rollout_updates_completed = rl.exploration_decay_updates
        self.assertLess(rl._current_exploration_rate(), initial_exploration)

    def test_intervention_lcb_is_invariant_to_unidentified_member_offsets(self):
        rl = bridge(
            nmcc_policy_improvement=True,
            nmcc_pi_actor_objective="value_lcb",
            actor_prior="risk_time_reduction",
        )
        samples = torch.tensor(
            [[[1.0, 2.0, 0.0], [10.0, 13.0, 8.0], [-4.0, -3.5, -6.0]]]
        )
        base = torch.tensor([0])
        original = rl._base_relative_improvement_lcb(samples, base)
        offsets = torch.tensor([[[100.0], [-17.0], [4.5]]])
        shifted = rl._base_relative_improvement_lcb(samples + offsets, base)
        torch.testing.assert_close(original, shifted)
        self.assertEqual(float(original[0, 0]), 0.0)

    def test_intervention_value_projection_is_critic_owned(self):
        rl = bridge(nmcc_policy_improvement=True)
        critic = {name for name, _ in rl.critic_head_named_parameters}
        self.assertTrue(any(name.startswith("improvement_node.") for name in critic))
        self.assertTrue(any(name.startswith("improvement_candidate.") for name in critic))
        self.assertTrue(any(name.startswith("improvement_value_heads.") for name in critic))
        self.assertTrue(any(name.startswith("improvement_direct_heads.") for name in critic))

    def test_operational_benefit_is_masked_but_deadline_never_relaxes_safety(self):
        rl = bridge()
        builder = rl.observation_builder
        count = builder.number_of_actions
        records = tuple(
            (str(index), index, 1.0, 0.5, 0.5) for index in range(count)
        )
        builder.require_candidate_operational_benefit = True
        builder.minimum_candidate_hazard_safety_margin = 0.5
        builder._mask_candidate_forecast_danger = torch.zeros(count).numpy()
        builder._mask_candidate_hazard_safety = torch.ones(count).numpy()
        builder._mask_candidate_hazard_safety[0] = 0.1
        builder._mask_candidate_reroutable_population = torch.zeros(count).numpy()
        builder._mask_candidate_risk_time_reduction = torch.zeros(count).numpy()
        builder._mask_force_capacity_token = False
        ordinary = builder._candidate_action_mask(1, torch.zeros(count).numpy(), records)
        self.assertFalse(ordinary.any())
        builder._mask_force_capacity_token = True
        deadline = builder._candidate_action_mask(1, torch.zeros(count).numpy(), records)
        self.assertFalse(bool(deadline[0]))
        self.assertTrue(bool(deadline[1:].all()))


class ImprovementValueLossTests(unittest.TestCase):
    def setUp(self):
        self.rl = bridge(nmcc_policy_improvement=True)
        generator = torch.Generator().manual_seed(3)
        self.rows, self.members, self.actions = 4, 3, 5
        self.behavior = torch.softmax(torch.randn(self.rows, self.actions, generator=generator), -1)
        self.exact = torch.zeros(self.rows, self.actions)
        self.exact[:, :4] = 1.0
        values = torch.randn(self.rows, self.actions, generator=generator) * 0.05
        weights = self.behavior * self.exact
        weights = weights / weights.sum(-1, keepdim=True)
        self.advantage = (values - (weights * values).sum(-1, keepdim=True)) * self.exact
        self.available = torch.ones(self.rows, dtype=torch.bool)
        self.bootstrap = torch.ones(self.rows, self.members)

    def loss(self, samples):
        return self.rl._improvement_value_loss(
            samples, self.available, self.behavior, self.exact, self.advantage, self.bootstrap
        )

    def test_exact_prediction_has_zero_loss(self):
        scale = self.rl.nmcc_pi_value_scale
        samples = (self.advantage / scale).unsqueeze(1).expand(-1, self.members, -1).clone()
        self.assertLess(float(self.loss(samples)), 1e-10)
        self.assertLess(float(self.rl._improvement_pairwise_loss(
            samples, self.available, self.exact, self.advantage, self.bootstrap
        )), 1e-10)

    def test_open_value_controller_exactly_recovers_physical_return_order(self):
        rl = bridge(
            nmcc_policy_improvement=True,
            nmcc_pi_actor_objective="value_lcb",
            actor_prior="risk_time_reduction",
        )
        prior = torch.tensor(
            [[0.9, 0.2, 0.6, 0.1, 0.0]]
        ).expand(self.rows, -1).clone()
        samples = (
            self.advantage / rl.nmcc_pi_value_scale
        ).unsqueeze(1).expand(-1, self.members, -1).clone()
        self.assertLess(float(rl._improvement_value_loss(
            samples, self.available, self.behavior, self.exact,
            self.advantage, self.bootstrap,
        )), 1e-10)
        self.assertLess(float(rl._improvement_pairwise_loss(
            samples, self.available, self.exact, self.advantage,
            self.bootstrap,
        )), 1e-10)
        base = prior.argmax(dim=-1)
        correction = rl._base_relative_improvement_lcb(samples, base)
        controller = prior.gather(1, base[:, None]) + (
            rl.nmcc_pi_value_scale
            * correction
            / rl.nmcc_pi_ranking_temperature
        )
        exact_best = torch.where(
            self.exact > 0.5,
            self.advantage,
            torch.full_like(self.advantage, float("-inf")),
        ).argmax(dim=-1)
        controller = torch.where(
            self.exact > 0.5,
            controller,
            torch.full_like(controller, float("-inf")),
        )
        self.assertTrue(torch.equal(controller.argmax(dim=-1), exact_best))

    def test_cell_independent_level_is_invisible(self):
        """V_wait / natural momentum: a per-state constant must not change the loss."""
        samples = torch.randn(self.rows, self.members, self.actions)
        shifted = samples + 17.0 * torch.randn(self.rows, self.members, 1)
        torch.testing.assert_close(self.loss(samples), self.loss(shifted))

    def test_unbranched_cells_do_not_enter_the_loss(self):
        samples = torch.randn(self.rows, self.members, self.actions)
        changed = samples.clone()
        changed[..., 4] += 100.0
        torch.testing.assert_close(self.loss(samples), self.loss(changed))

    def test_world_model_uses_the_exact_environment_reward_decomposition(self):
        outcomes = torch.tensor([[0.2, 0.1, 0.3, 0.4, 0.5, 0.5]])
        components = self.rl.policy._outcomes_to_components(outcomes)
        torch.testing.assert_close(
            components,
            torch.tensor([[0.2, -0.3, -0.3, -0.4]]),
        )

    def test_validation_scores_the_same_prior_plus_correction_as_deployment(self):
        action_count = self.rl.num_candidate_actions
        exact = torch.zeros(action_count)
        exact[:3] = 1.0
        advantage = torch.zeros(action_count)
        advantage[0] = -0.5
        advantage[2] = 0.5
        behavior = exact / exact.sum()
        transition = SimpleNamespace(
            improvement_advantage=advantage,
            improvement_behavior=behavior,
            improvement_exact_mask=exact,
            improvement_bootstrap=torch.ones(self.rl.nmcc_ensemble_size),
            improvement_base_action=torch.tensor(1),
        )
        samples = torch.zeros(1, self.rl.nmcc_ensemble_size, action_count)
        samples[:, :, 2] = 1.0
        priors = torch.zeros(1, action_count)
        priors[:, 1] = 100.0  # a varying prior must not be counted twice
        outputs = (
            torch.tensor([0]),
            None, None, None, None, None, None, None, None, None,
            samples,
            priors,
        )
        with mock.patch.object(
            self.rl,
            "_evaluate_recurrent_sequences",
            return_value=outputs,
        ):
            metrics = self.rl._improvement_validation_metrics(
                [transition], [[0]]
            )
        self.assertAlmostEqual(metrics["gain"], 0.5)
        self.assertAlmostEqual(metrics["top1"], 1.0)


class TrustRegionLineSearchTests(unittest.TestCase):
    def test_line_search_lands_inside_and_near_the_boundary(self):
        rl = bridge(nmcc_policy_improvement=True)
        snapshot = {
            name: parameter.detach().clone() for name, parameter in rl.actor_named_parameters
        }
        with torch.no_grad():
            rl.policy.actor_cell.weight.add_(1.0)
        moved = {
            name: parameter.detach().clone() for name, parameter in rl.actor_named_parameters
        }

        def statistics():
            # KL proxy that is exactly quadratic in the step fraction.
            delta = (rl.policy.actor_cell.weight - snapshot["actor_cell.weight"]).square().sum()
            return {"kl": float(delta.item())}

        full = float((moved["actor_cell.weight"] - snapshot["actor_cell.weight"]).square().sum())
        cap = 0.25 * full
        fraction = rl._line_search_to_trust_region(snapshot, cap, statistics)
        self.assertLessEqual(statistics()["kl"], cap + 1e-9)
        self.assertAlmostEqual(fraction, 0.5, delta=2 ** -8)


class RepresentationOwnershipTests(unittest.TestCase):
    def test_roles_partition_every_parameter(self):
        rl = bridge()
        roles = rl.policy.parameter_roles()
        names = [name for group in roles.values() for name, _ in group]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {name for name, _ in rl.policy.named_parameters()})
        representation = [name for name, _ in roles["representation"]]
        self.assertTrue(any(name.startswith("temporal.") for name in representation))
        self.assertFalse(any(name.startswith("temporal_actor_context") for name in representation))

    def test_legacy_mode_keeps_the_critic_off_the_representation(self):
        rl = bridge()
        critic = {name for name, _ in rl.critic_named_parameters}
        representation = {name for name, _ in rl.representation_named_parameters}
        self.assertFalse(critic & representation)

    def test_shared_mode_gives_the_representation_to_both_passes(self):
        rl = bridge(representation_mode="shared_phasic")
        actor = {name for name, _ in rl.actor_named_parameters}
        critic = {name for name, _ in rl.critic_named_parameters}
        representation = {name for name, _ in rl.representation_named_parameters}
        self.assertEqual(actor & critic, representation)
        self.assertTrue(any(name.startswith("temporal.") for name in representation))
        rl._set_optimizer_partition_trainable(actor=False, critic=True)
        for name, parameter in rl.policy.named_parameters():
            expected = name in critic
            self.assertEqual(parameter.requires_grad, expected, msg=name)
        rl._set_optimizer_partition_trainable(actor=True, critic=False)
        for name, parameter in rl.policy.named_parameters():
            self.assertEqual(parameter.requires_grad, name in actor, msg=name)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            bridge(representation_mode="frozen")


class TrainingEpisodeTests(unittest.TestCase):
    def test_value_replay_refit_handles_a_one_candidate_minibatch(self):
        import CounterfactualBranch as CB
        import nmcc_testbed

        core = nmcc_testbed.build(
            grid=8,
            cell_x=3,
            cell_y=3,
            population=90,
            stop_time=24,
            spread_rate=(6, 3),
            casualty_rate=(40, 9),
            panic_rate=0.5,
            hazard_count=2,
            scenario_seed=123,
            spacing_m=150,
            candidate_count=8,
            shelter_capacity_token=30,
        )
        with tempfile.TemporaryDirectory() as directory:
            rl = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=3,
                shelter_action_interval=6,
                policy_seed=7,
                rollout_episodes=1,
                minibatch_size=1,
                epochs=1,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "diagnostics.csv"),
                nmcc_policy_improvement=True,
                nmcc_pi_actor_objective="value_lcb",
                nmcc_pi_exhaustive_decisions=1,
                nmcc_pi_max_branches=4,
                nmcc_pi_replay_epochs=2,
                nmcc_pi_early_stopping_patience=1,
                nmcc_pi_min_validation_states=1,
                actor_prior="risk_time_reduction",
                action_temperature_start=1.0,
                action_temperature_end=1.0,
                representation_mode="shared_phasic",
            )
            core.rl = rl
            for step in range(1, int(core.stopTime)):
                CB.advance_one_timestep(core)
                rl.step(simulation_time=step, is_terminal=(step == core.stopTime - 1))
            result = rl.end_episode()

            self.assertEqual(len(rl.improvement_replay), 1)
            template = rl.improvement_replay[0]
            rl.improvement_replay = []
            for episode_id in range(4):
                record = copy.deepcopy(template)
                record["episode_id"] = episode_id
                rl.improvement_replay.append(record)
            # Episode 2 is a training minibatch with no within-state contrast.
            # It must be skipped rather than calling backward() on a constant.
            for transition in rl.improvement_replay[2]["transitions"]:
                if transition.improvement_exact_mask is None:
                    continue
                mask = torch.zeros_like(transition.improvement_exact_mask)
                mask[0] = 1.0
                transition.improvement_exact_mask = mask
                transition.improvement_advantage = torch.zeros_like(
                    transition.improvement_advantage
                )
            rl.improvement_replay_seen = 4
            rl.improvement_replay_next_id = 4
            replay = rl._fit_improvement_replay()

        self.assertEqual(result["optimizer_updated"], 1.0)
        self.assertEqual(result["nmcc_pi_coverage"], 1.0)
        self.assertGreaterEqual(result["nmcc_pi_branched"], 2.0)
        self.assertGreater(replay["nmcc_pi_replay_training_states"], 0.0)
        self.assertGreater(replay["nmcc_pi_replay_epochs_completed"], 0.0)


if __name__ == "__main__":
    unittest.main()
