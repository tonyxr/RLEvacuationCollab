"""v64 deployable early-search strategy tests."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import CounterfactualBranch as CB
from DecisionInterface import BENCHMARK_POLICY_CONTRACTS
from RLBridge import RLBridge
from backtest import DYNAMIC_STRATEGIES as SINGLE_DYNAMIC
from backtest import LEARNED_STRATEGIES as SINGLE_LEARNED
from multicity_backtest import DYNAMIC_STRATEGIES as MULTI_DYNAMIC
from multicity_backtest import LEARNED_STRATEGIES as MULTI_LEARNED
from nmcc_testbed import build


def _bridge(directory: str, core, **overrides) -> RLBridge:
    values = dict(
        train_mode=True,
        deployment_strategy="rl",
        target_active_shelters=len(core.shelterDS.shelterList) + 3,
        shelter_action_interval=3,
        first_decision_time=1,
        policy_seed=13,
        rollout_episodes=1,
        actor_lr=1e-3,
        actor_epochs=1,
        critic_epochs=1,
        nmcc_pi_replay_epochs=1,
        nmcc_pi_workers=1,
        rl_early_search_decisions=3,
        rl_early_search_top_k=3,
        rl_early_search_futures=4,
        checkpoint_path=os.path.join(directory, "policy.pt"),
        diagnostics_path=os.path.join(directory, "diagnostics.csv"),
    )
    values.update(overrides)
    return RLBridge(core, **values)


def _evaluation_bridge(directory: str, core, **overrides) -> RLBridge:
    bridge = _bridge(directory, core, **overrides)
    bridge.train_mode = False
    bridge.deployment_strategy = "rl_early_search"
    bridge.policy.eval()
    return bridge


def _observation_and_step(bridge: RLBridge):
    observation = bridge.observation_builder.build(
        decision_index=bridge.decision_index,
        simulation_time=1,
        remaining_deployments=bridge.remaining_deployments,
    )
    observation = bridge._enrich_policy_observation(observation)
    recurrent_step = bridge._advance_recurrent_observation(observation)
    return observation, recurrent_step


class _DeterministicBrancher:
    def __init__(self, best_index: int):
        self.best_index = int(best_index)
        self.calls = []

    def rollout_policy_costs(self, cells, **kwargs):
        self.calls.append((tuple(cells), dict(kwargs)))
        costs = np.full(
            (len(kwargs["future_seeds"]), len(cells)),
            10.0,
            dtype=np.float64,
        )
        costs[:, self.best_index] = 1.0
        return costs


class V64EarlySearchTests(unittest.TestCase):
    @staticmethod
    def _run_episode(strategy: str, scenario_seed: int) -> dict:
        core = build(
            stop_time=10,
            population=30,
            group_size=10,
            scenario_seed=scenario_seed,
            candidate_count=6,
            shelter_capacity_token=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            bridge = _evaluation_bridge(
                directory,
                core,
                policy_seed=31,
                rl_early_search_decisions=3,
                rl_early_search_top_k=2,
                rl_early_search_futures=2,
            )
            bridge.deployment_strategy = strategy
            core.rl = bridge
            for step in range(1, int(core.stopTime)):
                CB.advance_one_timestep(core)
                bridge.step(
                    simulation_time=step,
                    is_terminal=(step == core.stopTime - 1),
                )
            diagnostics = bridge.end_episode()
            if bridge._counterfactual_brancher is not None:
                bridge._counterfactual_brancher.close()
            return diagnostics

    def test_six_scenario_rollout_improvement_is_within_one_se(self):
        differences = []
        losses = []
        for scenario_seed in range(710, 716):
            baseline = self._run_episode("rl", scenario_seed)
            searched = self._run_episode("rl_early_search", scenario_seed)
            difference = float(searched["objective_normalized_cost"]) - float(
                baseline["objective_normalized_cost"]
            )
            differences.append(difference)
            if difference > 0.0:
                losses.append(difference)
        mean_difference = float(np.mean(differences))
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(6.0))
        self.assertLessEqual(mean_difference, standard_error + 1e-12)
        self.assertLessEqual(len(losses), 1)
        if losses:
            self.assertLessEqual(losses[0], standard_error + 1e-12)

    def test_search_chooses_lowest_mean_panel_action_without_live_mutation(self):
        core = build(
            stop_time=10,
            population=30,
            group_size=10,
            candidate_count=8,
            shelter_capacity_token=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            bridge = _evaluation_bridge(directory, core)
            observation, recurrent_step = _observation_and_step(bridge)
            mask = np.asarray(observation.action_mask, dtype=bool)
            feasible = np.flatnonzero(mask)
            defer = int(observation.defer_action_index)
            logits = bridge._behavior_policy(recurrent_step)[0]
            scores = logits.detach().cpu().numpy().reshape(-1)
            install = feasible[feasible != defer]
            ranked = sorted(
                install.tolist(),
                key=lambda action: (-float(scores[action]), int(action)),
            )
            panel = ranked[: bridge.rl_early_search_top_k] + [defer]
            base = bridge._select_rl(
                observation,
                deterministic=True,
                recurrent_step=recurrent_step,
            )
            best_index = next(
                index
                for index, action in enumerate(panel)
                if action != int(base.action_index)
            )
            brancher = _DeterministicBrancher(best_index)
            bridge._counterfactual_brancher = brancher
            before = CB.live_state_digest(core)
            decision = bridge._select_rl_with_early_search(
                observation,
                recurrent_step,
            )
            self.assertEqual(CB.live_state_digest(core), before)
            self.assertEqual(decision.action_index, panel[best_index])
            self.assertEqual(decision.strategy, "rl_early_search")
            self.assertEqual(bridge.early_search_decisions, 1)
            self.assertEqual(bridge.early_search_changed, 1)
            call = brancher.calls[0]
            self.assertEqual(len(call[1]["future_seeds"]), 4)
            self.assertEqual(call[1]["mode"], "belief")

    def test_determinism_on_six_testbed_scenarios_and_panel_optimality(self):
        for scenario_seed in range(610, 616):
            decisions = []
            for repeat in range(2):
                core = build(
                    stop_time=8,
                    population=20,
                    group_size=10,
                    scenario_seed=scenario_seed,
                    candidate_count=6,
                    shelter_capacity_token=10,
                )
                with tempfile.TemporaryDirectory() as directory:
                    bridge = _evaluation_bridge(
                        directory,
                        core,
                        policy_seed=23,
                        rl_early_search_top_k=2,
                        rl_early_search_futures=2,
                    )
                    observation, recurrent_step = _observation_and_step(bridge)
                    brancher = bridge._ensure_brancher()
                    captured = []
                    original = brancher.rollout_policy_costs

                    def recording(cells, **kwargs):
                        matrix = original(cells, **kwargs)
                        captured.append((tuple(cells), np.asarray(matrix)))
                        return matrix

                    before = CB.live_state_digest(core)
                    with mock.patch.object(
                        brancher,
                        "rollout_policy_costs",
                        side_effect=recording,
                    ):
                        decision = bridge._select_rl_with_early_search(
                            observation,
                            recurrent_step,
                        )
                    self.assertEqual(CB.live_state_digest(core), before)
                    cells, matrix = captured[0]
                    best_cell = int(cells[int(np.argmin(np.mean(matrix, axis=0)))])
                    selected_cell = (
                        -1
                        if decision.action_index == observation.defer_action_index
                        else int(
                            observation.candidate_cell_indices[
                                decision.action_index
                            ]
                        )
                    )
                    self.assertEqual(selected_cell, best_cell)
                    decisions.append(int(decision.action_index))
                    brancher.close()
            self.assertEqual(decisions[0], decisions[1])

    def test_switch_off_and_training_mode_bypass_search(self):
        core = build(population=20, group_size=10, candidate_count=8)
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(
                directory,
                core,
                rl_early_search_decisions=0,
            )
            observation, recurrent_step = _observation_and_step(bridge)
            brancher = SimpleNamespace(
                rollout_policy_costs=mock.Mock(
                    side_effect=AssertionError("search should be bypassed")
                )
            )
            bridge._counterfactual_brancher = brancher
            expected = bridge._select_rl(
                observation,
                deterministic=True,
                recurrent_step=recurrent_step,
            )
            bridge.train_mode = False
            disabled = bridge._select_rl_with_early_search(
                observation,
                recurrent_step,
            )
            self.assertEqual(disabled.action_index, expected.action_index)
            bridge.rl_early_search_decisions = 3
            bridge.train_mode = True
            training = bridge._select_rl_with_early_search(
                observation,
                recurrent_step,
            )
            self.assertEqual(training.action_index, expected.action_index)
            brancher.rollout_policy_costs.assert_not_called()

    def test_registration_and_eval_checkpoint_loading(self):
        for strategies in (
            SINGLE_DYNAMIC,
            SINGLE_LEARNED,
            MULTI_DYNAMIC,
            MULTI_LEARNED,
        ):
            self.assertIn("rl_early_search", strategies)
        self.assertIn("rl_early_search", BENCHMARK_POLICY_CONTRACTS)
        self.assertEqual(
            BENCHMARK_POLICY_CONTRACTS["rl_early_search"]["role"],
            "deployable_hybrid",
        )
        with tempfile.TemporaryDirectory() as directory:
            training_core = build(
                population=20,
                group_size=10,
                candidate_count=8,
            )
            training = _bridge(directory, training_core)
            training._save_checkpoint()
            evaluation_core = build(
                population=20,
                group_size=10,
                candidate_count=8,
            )
            evaluation = RLBridge(
                evaluation_core,
                train_mode=False,
                deployment_strategy="rl_early_search",
                target_active_shelters=(
                    len(evaluation_core.shelterDS.shelterList) + 3
                ),
                shelter_action_interval=3,
                first_decision_time=1,
                policy_seed=13,
                rollout_episodes=1,
                actor_lr=1e-3,
                actor_epochs=1,
                critic_epochs=1,
                nmcc_pi_replay_epochs=1,
                nmcc_pi_workers=1,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "eval.csv"),
            )
            self.assertEqual(evaluation.deployment_strategy, "rl_early_search")
            self.assertIsNotNone(evaluation.policy)


if __name__ == "__main__":
    unittest.main()
