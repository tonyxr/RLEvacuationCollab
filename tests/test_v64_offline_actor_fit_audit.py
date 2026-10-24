"""Tests for the v64 offline actor representability audit."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import os
import tempfile
import unittest

import torch

from DecisionInterface import CANDIDATE_FEATURE_NAMES, CELL_FEATURE_NAMES
from RLBridge import ActorReplayEpisode, RLBridge
from scripts.offline_actor_fit_audit import (
    FRONT_CANDIDATE_FEATURES,
    FRONT_CELL_FEATURES,
    _ablate_front,
    _scenario_split,
    run_audit,
)
from tests.test_rl_framework import make_core, make_observation


class V64OfflineActorAuditTests(unittest.TestCase):
    def test_ablation_is_exact_and_does_not_mutate_source(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "diagnostics.csv"),
            )
            frame = bridge._advance_recurrent_observation(make_observation()).frame
            episode = ActorReplayEpisode(
                frames=(frame,),
                target_scores=torch.zeros((1, frame.action_mask.numel())),
                exact_masks=frame.action_mask.reshape(1, -1).to(torch.float32),
                eta_floors=torch.ones(1),
                iteration=0,
            )
            source_cells = frame.cell_features.clone()
            source_candidates = frame.candidate_features.clone()
            ablated = _ablate_front(episode)
            torch.testing.assert_close(frame.cell_features, source_cells)
            torch.testing.assert_close(frame.candidate_features, source_candidates)
            for name in FRONT_CELL_FEATURES:
                self.assertTrue(
                    torch.all(
                        ablated.frames[0].cell_features[
                            :, CELL_FEATURE_NAMES.index(name)
                        ]
                        == 0.0
                    )
                )
            for name in FRONT_CANDIDATE_FEATURES:
                self.assertTrue(
                    torch.all(
                        ablated.frames[0].candidate_features[
                            :, CANDIDATE_FEATURE_NAMES.index(name)
                        ]
                        == 0.0
                    )
                )

    def test_checkpoint_audit_uses_disjoint_scenario_split(self):
        training, held_out = _scenario_split(
            [11, 11, 12, 13, 14, 15], train_fraction=0.8
        )
        training_seeds = {([11, 11, 12, 13, 14, 15][index]) for index in training}
        held_out_seeds = {([11, 11, 12, 13, 14, 15][index]) for index in held_out}
        self.assertTrue(training_seeds)
        self.assertTrue(held_out_seeds)
        self.assertFalse(training_seeds & held_out_seeds)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            bridge = RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                checkpoint_path=checkpoint,
                diagnostics_path=os.path.join(directory, "diagnostics.csv"),
                nmcc_policy_improvement=True,
                nmcc_pi_tapes=2,
                nmcc_pi_replay_epochs=1,
            )
            frame = bridge._advance_recurrent_observation(make_observation()).frame
            actions = int(frame.action_mask.numel())
            exact = frame.action_mask.reshape(1, -1).to(torch.float32)
            topology_a = torch.as_tensor(
                ((0, 1), (1, 0)), dtype=torch.long
            )
            topology_b = torch.as_tensor(
                ((0, 3), (3, 0)), dtype=torch.long
            )
            bridge.actor_replay = [
                ActorReplayEpisode(
                    frames=(
                        replace(
                            frame,
                            spatial_edge_index=(
                                topology_a if index % 2 == 0 else topology_b
                            ),
                        ),
                    ),
                    target_scores=(
                        torch.arange(actions, dtype=torch.float32).reshape(1, -1)
                        + index
                    ),
                    exact_masks=exact,
                    eta_floors=torch.full((1,), 0.01),
                    iteration=index,
                )
                for index in range(5)
            ]
            bridge.actor_replay_scenario_seeds = [11, 12, 13, 14, 15]
            bridge.actor_rollout_updates = 4
            bridge._save_checkpoint()
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            self.assertNotIn("spatial_edge_index", payload)
            report = run_audit(
                Namespace(
                    checkpoint=checkpoint,
                    fit_seeds=(1, 2, 3),
                    train_fraction=0.8,
                    epochs=1,
                    learning_rate=0.001,
                    epsilon=0.10,
                    half_life=5.0,
                    early_decisions=3,
                    early_weight=2.0,
                )
            )
            self.assertEqual(report["model_version"], 64)
            self.assertEqual(report["replay_episodes"], 5)
            self.assertEqual(len(report["rows"]), 6)
            self.assertFalse(
                set(report["training_scenario_seeds"])
                & set(report["held_out_scenario_seeds"])
            )
            self.assertEqual(
                {row["variant"] for row in report["rows"]},
                {"full", "no_front"},
            )


if __name__ == "__main__":
    unittest.main()
