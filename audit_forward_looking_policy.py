#!/usr/bin/env python3
"""Test whether a frozen RL policy uses foresight to improve siting decisions.

This is a read-only evaluation.  At selected held-out decision states it:

* records the deterministic learned and objective-greedy choices;
* recomputes the learned choice after removing current momentum, all temporal
  history, or the spatial hazard forecast; and
* grades every distinct choice to the physical episode horizon on identical
  random futures while the frozen learned policy handles later decisions.

Decision zero can grade every feasible cell, which permits a direct comparison
of the learned and myopic rankings against full-horizon simulator value.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import CounterfactualBranch as CB
import NMCCPolicyImprovement as NPI
import RLBridge as RB
import RolloutPolicyIteration as RPI
from DecisionInterface import RegionalObservation
from rollout_headroom_experiment import evaluation_overrides


_ORIGINAL_ADVANCE = RB.RLBridge._advance_recurrent_observation
_ORIGINAL_SELECT = RB.RLBridge._select_action
_AUDIT = None


def _seed(launch_seed: int, stream: int, index: int = 0) -> int:
    sequence = np.random.SeedSequence([int(launch_seed), int(stream), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _clone_state(state):
    if state is None:
        return None
    return tuple(value.detach().clone() for value in state)


def _masked_argmax(logits: torch.Tensor, mask: np.ndarray) -> int:
    feasible = torch.as_tensor(mask, dtype=torch.bool, device=logits.device).reshape(-1)
    flat = logits.detach().reshape(-1)
    masked = flat.masked_fill(~feasible, torch.finfo(flat.dtype).min)
    return int(torch.argmax(masked).item())


def _spearman(score: np.ndarray, value: np.ndarray) -> float:
    score = np.asarray(score, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    if score.size < 3 or np.ptp(score) == 0.0 or np.ptp(value) == 0.0:
        return float("nan")
    score_rank = np.argsort(np.argsort(score)).astype(np.float64)
    value_rank = np.argsort(np.argsort(value)).astype(np.float64)
    return float(np.corrcoef(score_rank, value_rank)[0, 1])


def _policy_logits(
    bridge,
    observation: RegionalObservation,
    *,
    recurrent_state,
    momentum: torch.Tensor,
) -> torch.Tensor:
    (
        cell_features,
        global_features,
        route_edge_index,
        route_edge_weight,
        candidate_cell_index,
        candidate_features,
        _,
    ) = bridge._policy_tensors(observation)
    graph = bridge._graph(
        cell_features,
        global_features,
        route_edge_index=route_edge_index,
        route_edge_weight=route_edge_weight,
        candidate_cell_index=candidate_cell_index,
        candidate_features=candidate_features,
    )
    with torch.no_grad():
        logits, _, _, _ = bridge.policy.forward_nmcc_actor_recurrent(
            graph,
            recurrent_state,
            momentum,
        )
    return logits.detach()


def _advance_with_audit(self, observation):
    self._foresight_prior_state = _clone_state(self.recurrent_state)
    self._foresight_prior_observation = self.previous_temporal_observation
    return _ORIGINAL_ADVANCE(self, observation)


def _select_with_audit(self, observation, recurrent_step=None):
    decision = _ORIGINAL_SELECT(self, observation, recurrent_step)
    config = _AUDIT
    if (
        config is None
        or self.deployment_strategy != "rl"
        or recurrent_step is None
        or int(self.deployments_made) not in config["decisions"]
    ):
        return decision

    decision_index = int(self.deployments_made)
    feasible = np.flatnonzero(observation.action_mask).astype(np.int64)
    actual_logits = recurrent_step.logits.detach().reshape(-1)
    actual_action = int(decision.action_index)
    prior_state = getattr(self, "_foresight_prior_state", None)
    zero_momentum = torch.zeros_like(recurrent_step.frame.momentum_features)

    no_current_logits = _policy_logits(
        self,
        observation,
        recurrent_state=prior_state,
        momentum=zero_momentum,
    )
    no_history_logits = _policy_logits(
        self,
        observation,
        recurrent_state=None,
        momentum=zero_momentum,
    )
    hazard_blind_observation = replace(
        observation,
        forecast_danger_by_cell=np.asarray(
            observation.danger_by_cell, dtype=np.float32
        ).copy(),
        danger_momentum_by_cell=np.zeros_like(
            observation.danger_by_cell, dtype=np.float32
        ),
    )
    no_forecast_logits = _policy_logits(
        self,
        hazard_blind_observation,
        recurrent_state=prior_state,
        momentum=recurrent_step.frame.momentum_features,
    )

    variants = {
        "rl": actual_action,
        "objective_greedy": int(
            self.objective_greedy_policy.select(
                observation,
                candidate_values=self._objective_greedy_values(observation),
                deterministic=True,
            ).action_index
        ),
        "no_current_momentum": _masked_argmax(
            no_current_logits, observation.action_mask
        ),
        "no_temporal_history": _masked_argmax(
            no_history_logits, observation.action_mask
        ),
        "no_hazard_forecast": _masked_argmax(
            no_forecast_logits, observation.action_mask
        ),
    }

    if decision_index in config["exhaustive_decisions"]:
        panel_actions = feasible.tolist()
    else:
        panel_actions = list(dict.fromkeys(variants.values()))
        ranked = feasible[
            np.argsort(
                -actual_logits.detach().cpu().numpy().reshape(-1)[feasible],
                kind="stable",
            )
        ]
        for action in ranked:
            if int(action) not in panel_actions:
                panel_actions.append(int(action))
            if len(panel_actions) >= 2:
                break
    panel_actions = np.asarray(panel_actions, dtype=np.int64)
    panel_cells = np.asarray(
        observation.candidate_cell_indices[panel_actions], dtype=np.int64
    )
    if len(set(panel_cells.tolist())) != len(panel_cells):
        raise RuntimeError("Foresight audit action panel contains duplicate cells")

    episode_seed = int(getattr(self.core, "scenario_seed", self.policy_seed))
    future_seeds = tuple(
        NPI.tape_seed(
            episode_seed ^ 0x464F5245,
            decision_index,
            tape_index,
        )
        for tape_index in range(int(config["tapes"]))
    )
    frozen_policy = RPI.FrozenResidualPolicy.capture(
        self.policy,
        self.edge_index,
        initial_population=self.initial_population,
        shelter_action_interval=self.shelter_action_interval,
        previous_observation=observation,
        recurrent_state=self.recurrent_state,
    )
    brancher = self._ensure_brancher()
    snapshot = CB.capture(self.core, label="forward-looking-policy-audit")
    before = CB.live_state_digest(self.core)
    costs = brancher.rollout_policy_costs(
        panel_cells,
        snapshot=snapshot,
        future_seeds=future_seeds,
        decision_index=decision_index,
        remaining_deployments=int(self.remaining_deployments),
        maximum_deployments=int(self.maximum_deployments),
        frozen_policy=frozen_policy,
    )
    if CB.live_state_digest(self.core) != before:
        raise RuntimeError("Foresight audit mutated the live simulator")

    action_to_position = {
        int(action): position for position, action in enumerate(panel_actions)
    }
    mean_cost = costs.mean(axis=0)
    greedy_values = self._objective_greedy_values(observation)[panel_actions]
    actor_scores = actual_logits.detach().cpu().numpy().reshape(-1)[panel_actions]
    record = {
        "scenario": int(config["scenario"]),
        "scenario_seed": episode_seed,
        "decision_index": decision_index,
        "simulation_time": int(observation.simulation_time),
        "momentum": recurrent_step.frame.momentum_features.detach().cpu().numpy().tolist(),
        "variants": {name: int(action) for name, action in variants.items()},
        "variant_cells": {
            name: int(observation.candidate_cell_indices[action])
            for name, action in variants.items()
        },
        "panel_actions": panel_actions.tolist(),
        "panel_cells": panel_cells.tolist(),
        "cost_by_tape": costs.tolist(),
        "mean_cost": mean_cost.tolist(),
        "actor_scores": actor_scores.tolist(),
        "objective_greedy_scores": np.asarray(greedy_values, float).tolist(),
        "actor_spearman_with_full_horizon_value": _spearman(
            actor_scores, -mean_cost
        ),
        "greedy_spearman_with_full_horizon_value": _spearman(
            np.asarray(greedy_values, float), -mean_cost
        ),
    }
    for name, action in variants.items():
        position = action_to_position[int(action)]
        record[f"{name}_mean_cost"] = float(mean_cost[position])
        record[f"{name}_cost_by_tape"] = costs[:, position].tolist()
        record[f"{name}_regret_within_panel"] = float(
            mean_cost[position] - mean_cost.min()
        )
    with config["decision_path"].open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(
        "[FORESIGHT AUDIT] "
        f"scenario={config['scenario']} k={decision_index} "
        f"RL-greedy={record['rl_mean_cost'] - record['objective_greedy_mean_cost']:+.5f} "
        f"actions={record['variant_cells']}",
        flush=True,
    )
    return decision


RB.RLBridge._advance_recurrent_observation = _advance_with_audit
RB.RLBridge._select_action = _select_with_audit


def _paired(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "standard_error": None}
    standard_error = (
        float(array.std(ddof=1) / math.sqrt(array.size))
        if array.size > 1
        else None
    )
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "standard_error": standard_error,
        "positive": int(np.sum(array > 0.0)),
        "negative": int(np.sum(array < 0.0)),
    }


def _summarize(decision_path: Path, episode_path: Path, output: Path) -> dict:
    decisions = [
        json.loads(line)
        for line in decision_path.read_text().splitlines()
        if line.strip()
    ]
    episodes = [
        json.loads(line)
        for line in episode_path.read_text().splitlines()
        if line.strip()
    ]
    deviations = [
        row for row in decisions if row["variants"]["rl"] != row["variants"]["objective_greedy"]
    ]
    rl_saving = [
        row["objective_greedy_mean_cost"] - row["rl_mean_cost"]
        for row in deviations
    ]
    ablations = {}
    for name in ("no_current_momentum", "no_temporal_history", "no_hazard_forecast"):
        changed = [row for row in decisions if row["variants"][name] != row["variants"]["rl"]]
        ablations[name] = {
            "action_changed": len(changed),
            "decision_states": len(decisions),
            "cost_increase_when_changed": _paired(
                [row[f"{name}_mean_cost"] - row["rl_mean_cost"] for row in changed]
            ),
        }
    exhaustive = [
        row
        for row in decisions
        if len(row["panel_actions"]) > len(set(row["variants"].values()))
    ]
    summary = {
        "decision_states": len(decisions),
        "episodes": len(episodes),
        "rl_deviations_from_objective_greedy": len(deviations),
        "rl_full_horizon_saving_when_deviating": _paired(rl_saving),
        "ablations": ablations,
        "exhaustive_rank_audit": {
            "states": len(exhaustive),
            "actor_mean_spearman": (
                float(np.nanmean([row["actor_spearman_with_full_horizon_value"] for row in exhaustive]))
                if exhaustive else None
            ),
            "greedy_mean_spearman": (
                float(np.nanmean([row["greedy_spearman_with_full_horizon_value"] for row in exhaustive]))
                if exhaustive else None
            ),
        },
        "episode_results": episodes,
    }
    (output / "foresight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--scenarios", type=int, default=6)
    parser.add_argument("--tapes", type=int, default=2)
    parser.add_argument("--decisions", default="0,2,4,6")
    parser.add_argument("--exhaustive-decisions", default="0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--machine", default="local")
    args = parser.parse_args()

    from backtest import _run_episode

    run = Path(args.run).resolve()
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "policies" / "policy_001" / "regional_policy.pt"
    diagnostics = run / "policies" / "policy_001" / "learner_diagnostics.csv"
    manifest = run / "experiment_manifest.json"
    for required in (checkpoint, diagnostics, manifest):
        if not required.exists():
            raise FileNotFoundError(required)
    overrides, launch_seed = evaluation_overrides(
        str(manifest), args.curriculum, None
    )
    decisions = {int(value) for value in args.decisions.split(",") if value.strip()}
    exhaustive = {
        int(value) for value in args.exhaustive_decisions.split(",") if value.strip()
    }
    if not exhaustive.issubset(decisions):
        raise ValueError("Exhaustive decisions must be included in --decisions")

    decision_path = output / "decision_audit.jsonl"
    episode_path = output / "episode_audit.jsonl"
    decision_path.unlink(missing_ok=True)
    episode_path.unlink(missing_ok=True)
    global _AUDIT
    for scenario in range(1, int(args.scenarios) + 1):
        scenario_seed = _seed(launch_seed, 500 + 2, scenario)
        policy_seed = _seed(launch_seed, 600 + 2, scenario)
        _AUDIT = {
            "scenario": scenario,
            "decisions": decisions,
            "exhaustive_decisions": exhaustive,
            "tapes": int(args.tapes),
            "decision_path": decision_path,
        }
        row = _run_episode(
            replication=scenario,
            machine=args.machine,
            phase=str(output / "work" / f"scenario_{scenario:03d}"),
            strategy="rl",
            train_mode=False,
            scenario_seed=scenario_seed,
            policy_seed=policy_seed,
            checkpoint_path=str(checkpoint),
            diagnostics_path=str(diagnostics),
            overrides=copy.deepcopy(overrides),
        )
        keep = {
            key: row.get(key)
            for key in (
                "scenario_seed",
                "objective_normalized_cost",
                "realized_hazard_weighted_time_to_safety",
                "casualty",
                "safe_completed",
                "unfinished",
                "episode_wall_time_s",
            )
        }
        keep["scenario"] = scenario
        with episode_path.open("a") as handle:
            handle.write(json.dumps(keep, default=str) + "\n")
    _AUDIT = None
    summary = _summarize(decision_path, episode_path, output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
