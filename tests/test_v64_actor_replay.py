import os
import tempfile
import unittest
from types import SimpleNamespace

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
            frame = _frame(bridge)
            transition = _transition(frame)
            bridge.actor_rollout_updates = 7
            bridge._append_actor_replay((transition,), ((0,),))
            bridge._save_checkpoint()
            restored = _bridge(directory)
            self.assertEqual(len(restored.actor_replay), 1)
            self.assertEqual(restored.actor_replay[0].iteration, 7)
            self.assertEqual(
                len(restored.actor_replay[0].frames),
                len(bridge.actor_replay[0].frames),
            )


if __name__ == "__main__":
    unittest.main()
