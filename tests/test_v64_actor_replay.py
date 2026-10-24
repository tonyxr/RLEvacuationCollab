import os
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from RLBridge import ActorFitEpisode, ActorReplayEpisode, RLBridge
from tests.test_rl_framework import make_core, make_observation


def _bridge(directory, **overrides):
    values = dict(
        core=make_core(),
        train_mode=True,
        deployment_strategy="rl",
        target_active_shelters=2,
        checkpoint_path=os.path.join(directory, "policy.pt"),
        diagnostics_path=os.path.join(directory, "diagnostics.csv"),
        nmcc_policy_improvement=True,
        nmcc_pi_tapes=2,
        nmcc_pi_replay_max_episodes=20,
        nmcc_pi_replay_max_iterations=40,
        nmcc_pi_replay_half_life_iterations=1.0,
        nmcc_pi_early_decisions=1,
        nmcc_pi_early_decision_weight=2.0,
    )
    values.update(overrides)
    core = values.pop("core")
    return RLBridge(core, **values)


def _frame(bridge):
    return bridge._advance_recurrent_observation(make_observation()).frame


def _transition(frame, offset=0.0):
    actions = int(frame.action_mask.numel())
    scores = torch.arange(actions, dtype=torch.float32) + float(offset)
    return SimpleNamespace(
        observation_history=(frame,),
        improvement_advantage=scores,
        improvement_exact_mask=frame.action_mask.detach().cpu().to(torch.float32),
        improvement_eta_floor=torch.as_tensor(0.01, dtype=torch.float32),
        scenario_seed=torch.as_tensor((20261024,), dtype=torch.long),
    )


class V64ActorReplayTests(unittest.TestCase):
    def test_zero_iteration_window_retains_only_latest_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(
                directory,
                nmcc_pi_replay_max_iterations=0,
                nmcc_pi_replay_max_episodes=10,
            )
            frame = _frame(bridge)
            transitions = [_transition(frame), _transition(frame, 1.0)]
            bridge.actor_rollout_updates = 0
            bridge._append_actor_replay(transitions, ((0,), (1,)))
            bridge.actor_rollout_updates = 1
            bridge._append_actor_replay(transitions, ((0,), (1,)))
            self.assertEqual(len(bridge.actor_replay), 2)
            self.assertEqual(
                {episode.iteration for episode in bridge.actor_replay}, {1}
            )

    def test_replay_aggregates_iterations_then_caps_episode_count(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(directory, nmcc_pi_replay_max_episodes=10)
            frame = _frame(bridge)
            transitions = [_transition(frame), _transition(frame, 1.0)]
            for iteration in range(3):
                bridge.actor_rollout_updates = iteration
                bridge._append_actor_replay(transitions, ((0,), (1,)))
            self.assertEqual(len(bridge.actor_replay), 6)
            self.assertEqual(
                {episode.iteration for episode in bridge.actor_replay},
                {0, 1, 2},
            )

            capped = _bridge(
                os.path.join(directory, "capped"),
                nmcc_pi_replay_max_episodes=4,
            )
            capped_frame = _frame(capped)
            capped_transitions = [
                _transition(capped_frame),
                _transition(capped_frame, 1.0),
            ]
            for iteration in range(3):
                capped.actor_rollout_updates = iteration
                capped._append_actor_replay(
                    capped_transitions, ((0,), (1,))
                )
            self.assertEqual(len(capped.actor_replay), 4)
            self.assertEqual(
                [episode.iteration for episode in capped.actor_replay],
                [1, 1, 2, 2],
            )

    def test_recency_stage_weights_and_weighted_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(directory)
            frame = _frame(bridge)
            actions = int(frame.action_mask.numel())
            scores = torch.zeros((2, actions), dtype=torch.float32)
            masks = frame.action_mask.detach().cpu().to(torch.float32).repeat(2, 1)
            floors = torch.full((2,), 0.01, dtype=torch.float32)
            bridge.actor_rollout_updates = 3
            bridge.actor_replay = [
                ActorReplayEpisode(
                    frames=(frame, frame),
                    target_scores=scores,
                    exact_masks=masks,
                    eta_floors=floors,
                    iteration=2,
                ),
                ActorReplayEpisode(
                    frames=(frame, frame),
                    target_scores=scores,
                    exact_masks=masks,
                    eta_floors=floors,
                    iteration=1,
                ),
            ]
            fitted = bridge._actor_fit_episodes()
            torch.testing.assert_close(
                fitted[0].frame_weights,
                torch.as_tensor((1.0, 0.5)),
            )
            torch.testing.assert_close(
                fitted[1].frame_weights,
                torch.as_tensor((0.5, 0.25)),
            )

            feasible = frame.action_mask.detach().cpu().to(torch.bool)
            feasible_actions = torch.nonzero(feasible).reshape(-1)
            first_target = torch.zeros(actions)
            second_target = torch.zeros(actions)
            first_target[feasible_actions[0]] = 1.0
            second_target[feasible_actions[1]] = 1.0
            exact = feasible.to(torch.float32).reshape(1, -1)
            fit_one = ActorFitEpisode(
                frames=(frame,),
                target_probabilities=first_target.reshape(1, -1),
                target_scores=torch.arange(actions, dtype=torch.float32).reshape(1, -1),
                exact_masks=exact,
                frame_weights=torch.ones(1),
            )
            fit_two = ActorFitEpisode(
                frames=(frame,),
                target_probabilities=second_target.reshape(1, -1),
                target_scores=torch.arange(actions - 1, -1, -1, dtype=torch.float32).reshape(1, -1),
                exact_masks=exact,
                frame_weights=torch.ones(1),
            )
            loss_one = bridge._actor_fit_objective((fit_one,))[0].detach()
            loss_two = bridge._actor_fit_objective((fit_two,))[0].detach()
            fit_one.frame_weights = torch.as_tensor((0.5,))
            fit_two.frame_weights = torch.as_tensor((0.25,))
            combined = bridge._actor_fit_objective((fit_one, fit_two))[0].detach()
            expected = (0.5 * loss_one + 0.25 * loss_two) / 0.75
            torch.testing.assert_close(combined, expected)

    def test_checkpoint_round_trip_preserves_replay_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(directory)
            source_edges = torch.as_tensor(
                ((0, 1), (1, 0)), dtype=torch.long
            )
            frame = replace(
                _frame(bridge),
                spatial_edge_index=source_edges,
            )
            transition = _transition(frame)
            bridge.actor_rollout_updates = 7
            bridge._append_actor_replay((transition,), ((0,),))
            bridge._save_checkpoint()
            restored = _bridge(directory)
            self.assertEqual(len(restored.actor_replay), 1)
            self.assertEqual(restored.actor_replay[0].iteration, 7)
            self.assertEqual(
                restored.actor_replay_scenario_seeds, [20261024]
            )
            self.assertEqual(
                len(restored.actor_replay[0].frames),
                len(bridge.actor_replay[0].frames),
            )
            restored_frame = restored.actor_replay[0].frames[0]
            torch.testing.assert_close(
                restored_frame.spatial_edge_index,
                source_edges,
            )
            torch.testing.assert_close(
                restored._graph_from_frame(restored_frame).edge_index,
                source_edges,
            )
            payload = torch.load(
                bridge.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertNotIn("spatial_edge_index", payload)

    def test_checkpoint_registry_accumulates_multiple_city_networks(self):
        with tempfile.TemporaryDirectory() as directory:
            first_core = make_core()
            first_core.cityID = "city_a"
            first = _bridge(directory, core=first_core)
            first._save_checkpoint()

            second_core = make_core()
            second_core.cityID = "city_b"
            second = _bridge(directory, core=second_core)
            second._save_checkpoint()

            payload = torch.load(
                second.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                set(payload["training_topology_registry"]),
                {"city_a", "city_b"},
            )

    def test_checkpoint_registry_records_same_city_adaptive_topologies(self):
        with tempfile.TemporaryDirectory() as directory:
            first_core = make_core()
            first_core.cityID = "city_a"
            first = _bridge(directory, core=first_core)
            first._save_checkpoint()

            changed_edges = np.asarray(((0, 1), (1, 0)), dtype=np.int64)
            second_core = make_core()
            second_core.cityID = "city_a"
            with mock.patch(
                "DecisionInterface.RegionalObservationBuilder._regional_spatial_edges",
                return_value=changed_edges,
            ):
                second = _bridge(directory, core=second_core)
                second._save_checkpoint()

            payload = torch.load(
                second.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            registry = payload["training_topology_registry"]["city_a"]
            self.assertEqual(len(registry["spatial_topologies"]), 2)
            self.assertEqual(
                sum(
                    item["episode_count"]
                    for item in registry["spatial_topologies"].values()
                ),
                2,
            )

    def test_checkpoint_registry_rejects_same_city_source_network_drift(self):
        def provenance(graph_digest):
            return SimpleNamespace(
                locationDrive=object(),
                graph_provenance=lambda: {
                    "query_sha256": "1" * 64,
                    "consolidated_graph_cache_sha256": graph_digest,
                    "graph_nodes": 100,
                    "graph_edges": 200,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            first_core = make_core()
            first_core.cityID = "city_a"
            first_core.OSMProcessor = provenance("2" * 64)
            first = _bridge(directory, core=first_core)
            first._save_checkpoint()

            second_core = make_core()
            second_core.cityID = "city_a"
            second_core.OSMProcessor = provenance("3" * 64)
            with self.assertRaisesRegex(ValueError, "source network changed"):
                _bridge(directory, core=second_core)

    def test_packed_frames_keep_distinct_spatial_topologies(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(directory)
            frame = _frame(bridge)
            first_edges = torch.as_tensor(
                ((0, 1), (1, 0)), dtype=torch.long
            )
            second_edges = torch.as_tensor(
                ((0, 3), (3, 0)), dtype=torch.long
            )
            packed = bridge._packed_graph_from_frames(
                (
                    replace(frame, spatial_edge_index=first_edges),
                    replace(frame, spatial_edge_index=second_edges),
                )
            )
            expected = torch.cat(
                (first_edges, second_edges + frame.cell_features.size(0)),
                dim=1,
            )
            torch.testing.assert_close(packed.edge_index, expected)

    def test_iteration_checkpoint_is_evaluation_only_and_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = _bridge(
                directory,
                save_iteration_checkpoints=True,
            )
            bridge.episodes_completed = 12
            bridge.actor_rollout_updates = 7
            bridge._save_checkpoint()

            iteration_path = os.path.join(
                directory,
                "iteration_checkpoints",
                "regional_policy_iteration_007.pt",
            )
            self.assertTrue(os.path.exists(iteration_path))
            payload = torch.load(
                iteration_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertFalse(payload["training_resume_allowed"])
            self.assertEqual(
                payload["checkpoint_purpose"],
                "fixed_panel_evaluation",
            )
            self.assertEqual(payload["episodes_completed"], 12)
            self.assertEqual(payload["actor_rollout_updates"], 7)
            self.assertIn("model_signature", payload)
            self.assertIn("inference_signature", payload)
            self.assertEqual(
                set(payload["policy_state_dict"]),
                set(bridge.policy.state_dict()),
            )

            evaluated = _bridge(
                os.path.join(directory, "evaluation"),
                train_mode=False,
                checkpoint_path=iteration_path,
            )
            for name, tensor in bridge.policy.state_dict().items():
                torch.testing.assert_close(
                    evaluated.policy.state_dict()[name], tensor
                )

            resume_path = bridge.checkpoint_path
            bridge.checkpoint_path = iteration_path
            with self.assertRaisesRegex(RuntimeError, "evaluation-only"):
                bridge._load_checkpoint(require_training_state=True)
            bridge.checkpoint_path = resume_path


if __name__ == "__main__":
    unittest.main()
