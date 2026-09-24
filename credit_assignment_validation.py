#!/usr/bin/env python3
"""Mechanistic validation for delayed reward and recurrent credit assignment.

This diagnostic does not train or mutate a checkpoint.  It checks three
properties independently of downstream policy performance:

1. interval reward components telescope exactly to the same post-action
   objective computed at the terminal boundary;
2. complete duration-aware Monte Carlo return-to-go propagates terminal
   outcomes to every earlier actor decision, is invariant to critic error, and
   preserves factorized branch isolation; and
3. the trained GNN-LSTM has a differentiable path from a final decision back to
   an observation 10--60 frames earlier, while a reset-memory ablation does not.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from DecisionInterface import (
    CANDIDATE_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    INFRA_FEATURE_NAMES,
    MOMENTUM_FEATURE_NAMES,
    OutcomeSnapshot,
    PED_FEATURE_NAMES,
)
from GNN import EvacPolicy, fit_gnn, grid_edge_index
from RLBridge import RLBridge
from RewardProcessor import REWARD_COMPONENT_NAMES, RewardProcessor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(*, safe: int, casualties: int, population: int) -> OutcomeSnapshot:
    active = max(0, int(population) - int(safe) - int(casualties))
    return OutcomeSnapshot(
        safe_completed=int(safe),
        casualties=int(casualties),
        shelter_evacuated=int(safe),
        ordinary_arrivals=0,
        active_population=active,
        risk_mass=float(active),
    )


def reward_telescoping_audit(*, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    processor = RewardProcessor()
    population = 3000
    horizon = 60
    maximum_scalar_gap = 0.0
    maximum_component_gap = 0.0
    terminal_casualty_trials = 0
    for _ in range(int(trials)):
        interval_count = int(rng.integers(1, 8))
        remaining = population
        safe_total = 0
        casualty_total = 0
        active_time_total = 0.0
        exposure_time_total = 0.0
        component_sum = np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float64)
        scalar_sum = 0.0
        before = _snapshot(
            safe=safe_total,
            casualties=casualty_total,
            population=population,
        )
        for interval in range(interval_count):
            new_safe = min(remaining, int(rng.integers(0, 251)))
            remaining -= new_safe
            new_casualties = min(remaining, int(rng.integers(0, 31)))
            remaining -= new_casualties
            if interval == interval_count - 1 and new_casualties > 0:
                terminal_casualty_trials += 1
            safe_total += new_safe
            casualty_total += new_casualties
            active_time = float(rng.uniform(0.0, population * horizon / interval_count))
            exposure_time = float(rng.uniform(0.0, active_time))
            active_time_total += active_time
            exposure_time_total += exposure_time
            after = _snapshot(
                safe=safe_total,
                casualties=casualty_total,
                population=population,
            )
            reward = processor.evaluate(
                before=before,
                after=after,
                active_person_time=active_time,
                hazard_exposure_person_time=exposure_time,
                initial_population=population,
                horizon=horizon,
            )
            component_sum += reward.component_vector().astype(np.float64)
            scalar_sum += float(reward.total)
            before = after

        whole = processor.evaluate(
            before=_snapshot(safe=0, casualties=0, population=population),
            after=before,
            active_person_time=active_time_total,
            hazard_exposure_person_time=exposure_time_total,
            initial_population=population,
            horizon=horizon,
        )
        expected_components = whole.component_vector().astype(np.float64)
        maximum_scalar_gap = max(maximum_scalar_gap, abs(scalar_sum - whole.total))
        maximum_component_gap = max(
            maximum_component_gap,
            float(np.max(np.abs(component_sum - expected_components))),
        )
    return {
        "trials": int(trials),
        "terminal_casualty_trials": int(terminal_casualty_trials),
        "maximum_scalar_gap": float(maximum_scalar_gap),
        "maximum_component_gap": float(maximum_component_gap),
        "passed": bool(
            maximum_scalar_gap <= 1e-6 and maximum_component_gap <= 1e-6
        ),
    }


def temporal_credit_target_audit(*, trials: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    bridge = object.__new__(RLBridge)
    bridge.device = torch.device("cpu")
    bridge.gamma = 1.0
    maximum_closed_form_gap = 0.0
    maximum_cross_branch_leakage = 0.0
    full_credited_actions = 0
    one_step_credited_actions = 0
    total_actions = 0
    earliest_advantages = []
    for _ in range(int(trials)):
        transition_count = int(rng.integers(2, 8))
        durations = torch.as_tensor(
            rng.integers(1, 31, size=transition_count),
            dtype=torch.float32,
        )
        branch = int(rng.integers(0, len(REWARD_COMPONENT_NAMES)))
        terminal_reward = float(rng.uniform(0.001, 1.0))
        rewards = torch.zeros(transition_count, len(REWARD_COMPONENT_NAMES))
        rewards[-1, branch] = terminal_reward
        values = torch.zeros_like(rewards)
        dones = torch.zeros(transition_count)
        dones[-1] = 1.0
        actor_returns, td_targets = bridge._component_credit_targets(
            rewards,
            values,
            dones,
            durations,
        )
        alternative_actor_returns, _ = bridge._component_credit_targets(
            rewards,
            torch.randn_like(values),
            dones,
            durations,
        )
        expected = torch.zeros(transition_count)
        expected[-1] = terminal_reward
        for index in reversed(range(transition_count - 1)):
            expected[index] = (
                (bridge.gamma ** durations[index])
                * expected[index + 1]
            )
        maximum_closed_form_gap = max(
            maximum_closed_form_gap,
            float(torch.max(torch.abs(actor_returns[:, branch] - expected)).item()),
            float(torch.max(torch.abs(alternative_actor_returns - actor_returns)).item()),
        )
        other = [
            index for index in range(len(REWARD_COMPONENT_NAMES)) if index != branch
        ]
        maximum_cross_branch_leakage = max(
            maximum_cross_branch_leakage,
            float(torch.max(torch.abs(actor_returns[:, other])).item()),
        )
        full_credited_actions += int(torch.count_nonzero(actor_returns[:, branch]).item())
        one_step_credited_actions += int(torch.count_nonzero(td_targets[:, branch]).item())
        total_actions += transition_count
        earliest_advantages.append(float(actor_returns[0, branch].item()))
    return {
        "trials": int(trials),
        "total_actions": int(total_actions),
        "full_credited_actions": int(full_credited_actions),
        "one_step_credited_actions": int(one_step_credited_actions),
        "credit_coverage_fraction": float(full_credited_actions / total_actions),
        "critic_td0_nonzero_fraction": float(one_step_credited_actions / total_actions),
        "minimum_earliest_advantage": float(min(earliest_advantages)),
        "maximum_closed_form_gap": float(maximum_closed_form_gap),
        "maximum_cross_branch_leakage": float(maximum_cross_branch_leakage),
        "passed": bool(
            full_credited_actions == total_actions
            and maximum_closed_form_gap <= 1e-6
            and maximum_cross_branch_leakage <= 1e-12
            and min(earliest_advantages) > 0.0
        ),
    }


def _load_policy(checkpoint_path: Path) -> EvacPolicy:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    policy = EvacPolicy(
        len(PED_FEATURE_NAMES),
        len(HAZARD_FEATURE_NAMES),
        len(INFRA_FEATURE_NAMES),
        d_global=len(GLOBAL_FEATURE_NAMES),
        d_candidate=len(CANDIDATE_FEATURE_NAMES),
        d_momentum=len(MOMENTUM_FEATURE_NAMES),
    )
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _random_graph(generator: torch.Generator):
    node_count = 64
    action_count = 18
    return fit_gnn(
        torch.rand(node_count, len(PED_FEATURE_NAMES), generator=generator),
        torch.rand(node_count, len(HAZARD_FEATURE_NAMES), generator=generator),
        torch.rand(node_count, len(INFRA_FEATURE_NAMES), generator=generator),
        torch.rand(len(GLOBAL_FEATURE_NAMES), generator=generator),
        edge_index=torch.as_tensor(grid_edge_index(8, 8), dtype=torch.long),
        candidate_cell_index=torch.arange(action_count, dtype=torch.long),
        candidate_features=torch.rand(
            action_count,
            len(CANDIDATE_FEATURE_NAMES),
            generator=generator,
        ),
    )


def _delayed_gradient(
    policy: EvacPolicy,
    *,
    horizon: int,
    seed: int,
    reset_memory: bool,
    critic: bool,
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    graph = _random_graph(generator)
    cue = torch.randn(
        len(MOMENTUM_FEATURE_NAMES),
        generator=generator,
        requires_grad=True,
    )
    state = None
    output = None
    for frame in range(int(horizon)):
        momentum = cue if frame == 0 else torch.zeros_like(cue)
        if reset_memory:
            state = None
        output = policy.forward_recurrent(graph, state, momentum)
        state = output[-1]
    if output is None:
        raise RuntimeError("gradient horizon must be positive")
    objective = output[2][0, 1] if critic else output[3][0, 0]
    if not objective.requires_grad:
        return 0.0
    gradient = torch.autograd.grad(
        objective,
        cue,
        allow_unused=True,
        retain_graph=False,
    )[0]
    return 0.0 if gradient is None else float(torch.linalg.vector_norm(gradient).item())


def recurrent_gradient_audit(
    checkpoint_path: Path,
    *,
    seeds: int,
    horizons: tuple[int, ...],
) -> dict:
    policy = _load_policy(checkpoint_path)
    rows = []
    for horizon in horizons:
        for offset in range(int(seeds)):
            seed = 7300 + 101 * int(horizon) + offset
            actor_full = _delayed_gradient(
                policy,
                horizon=horizon,
                seed=seed,
                reset_memory=False,
                critic=False,
            )
            actor_reset = _delayed_gradient(
                policy,
                horizon=horizon,
                seed=seed,
                reset_memory=True,
                critic=False,
            )
            critic_full = _delayed_gradient(
                policy,
                horizon=horizon,
                seed=seed,
                reset_memory=False,
                critic=True,
            )
            critic_reset = _delayed_gradient(
                policy,
                horizon=horizon,
                seed=seed,
                reset_memory=True,
                critic=True,
            )
            rows.append(
                {
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "actor_full_gradient_norm": actor_full,
                    "actor_reset_gradient_norm": actor_reset,
                    "critic_full_gradient_norm": critic_full,
                    "critic_reset_gradient_norm": critic_reset,
                }
            )
    actor_full = [row["actor_full_gradient_norm"] for row in rows]
    actor_reset = [row["actor_reset_gradient_norm"] for row in rows]
    critic_full = [row["critic_full_gradient_norm"] for row in rows]
    critic_reset = [row["critic_reset_gradient_norm"] for row in rows]
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "horizons": list(horizons),
        "seeds_per_horizon": int(seeds),
        "rows": rows,
        "actor_full_minimum_gradient_norm": float(min(actor_full)),
        "actor_reset_maximum_gradient_norm": float(max(actor_reset)),
        "critic_full_minimum_gradient_norm": float(min(critic_full)),
        "critic_reset_maximum_gradient_norm": float(max(critic_reset)),
        "passed": bool(
            min(actor_full) > 0.0
            and max(actor_reset) == 0.0
            and min(critic_full) > 0.0
            and max(critic_reset) == 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--gradient-seeds", type=int, default=8)
    parser.add_argument("--gradient-horizons", default="10,30,60")
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.trials <= 0 or args.gradient_seeds <= 0:
        parser.error("trial and gradient-seed counts must be positive")
    args.gradient_horizons = tuple(
        int(value.strip())
        for value in str(args.gradient_horizons).split(",")
        if value.strip()
    )
    if not args.gradient_horizons or min(args.gradient_horizons) <= 0:
        parser.error("gradient horizons must be positive")
    return args


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "reward_telescoping": reward_telescoping_audit(
            trials=args.trials,
            seed=args.seed,
        ),
        "complete_actor_mc_and_critic_td0": temporal_credit_target_audit(
            trials=args.trials,
            seed=args.seed + 1,
        ),
        "recurrent_gradient": recurrent_gradient_audit(
            args.checkpoint,
            seeds=args.gradient_seeds,
            horizons=args.gradient_horizons,
        ),
    }
    result["passed"] = bool(
        result["reward_telescoping"]["passed"]
        and result["complete_actor_mc_and_critic_td0"]["passed"]
        and result["recurrent_gradient"]["passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
