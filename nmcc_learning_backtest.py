#!/usr/bin/env python3
"""Matched engineering backtest for v25 score fitting and v26 fitted NMCC value control.

This runs the production GNN/LSTM/RLBridge, cellular-automata transition
engine, hazard process, routing, pedestrians, reward, and exact CRN branches
on the repository's deterministic synthetic map.  It is intentionally small;
the State College curriculum is the confirmatory experiment, not this check.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random

import numpy as np

import CounterfactualBranch as CB
from RLBridge import RLBridge
import nmcc_testbed


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "runs" / "nmcc_value_backtest_v26.json"


def _strict(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _learner_kwargs(arm: str, rollout_episodes: int, branch_horizon: int) -> dict:
    common = dict(
        nmcc_policy_improvement=True,
        nmcc_pi_tapes=1,
        nmcc_pi_exhaustive_decisions=0,
        nmcc_pi_max_branches=8,
        nmcc_pi_base_policy="route_saving",
        nmcc_pi_branch_horizon=int(branch_horizon),
        nmcc_pi_value_scale=0.05,
        nmcc_pi_value_loss_coef=1.0,
        nmcc_pi_model_fill=False,
        actor_prior="route_time_saving",
        actor_prior_scale=1.0,
        representation_mode="shared_phasic",
        representation_clone_coef=1.0,
        representation_kl_cap=0.05,
        rollout_episodes=int(rollout_episodes),
        actor_lr=8e-4,
        critic_lr=3e-4,
        actor_epochs=1,
        critic_epochs=2,
        nmcc_pi_actor_epochs=12,
        minibatch_size=4,
        nmcc_guidance_max=0.0,
        nmcc_teacher_coef=0.0,
        action_temperature_start=1.0,
        action_temperature_end=1.0,
    )
    if arm == "v25_staged_score":
        return {
            **common,
            "nmcc_pi_actor_objective": "score_ranking",
            "nmcc_pi_epsilon": 0.2,
            "nmcc_pi_kl_cap": 0.3,
            "nmcc_pi_ranking_temperature": 0.03,
            "nmcc_pi_rank_margin": 0.25,
            "nmcc_pi_rank_margin_coef": 0.5,
            "nmcc_natural_pretrain_rollouts": 1,
            "nmcc_causal_pretrain_rollouts": 1,
            "nmcc_controller_warmup_rollouts": 1,
            "exploration_rate_start": 0.40,
            "exploration_rate_end": 0.05,
            "exploration_decay_updates": 12,
            "learning_rate_schedule": "cosine",
            "lr_warmup_updates": 1,
            "lr_decay_updates": 12,
            "actor_lr_min_fraction": 0.15,
            "critic_lr_min_fraction": 0.25,
        }
    if arm != "v26_fitted_value":
        raise ValueError(f"Unknown arm {arm!r}")
    return {
        **common,
        "nmcc_pi_actor_objective": "value_lcb",
        "nmcc_pi_epsilon": 0.2,
        "nmcc_pi_kl_cap": 0.3,
        "nmcc_pi_ranking_temperature": 0.05,
        "nmcc_pi_exhaustive_decisions": 2,
        "nmcc_pi_full_horizon_decisions": 2,
        "nmcc_pi_replay_max_episodes": 64,
        "nmcc_pi_replay_epochs": 24,
        "nmcc_pi_validation_fraction": 0.2,
        "nmcc_pi_early_stopping_patience": 4,
        "nmcc_pi_min_validation_states": 3,
        "nmcc_pi_validation_gain_z": 1.0,
        "nmcc_pi_replay_refit": True,
        "nmcc_pi_gate_updates": 3,
        "nmcc_natural_pretrain_rollouts": 1,
        "nmcc_causal_pretrain_rollouts": 1,
        "nmcc_controller_warmup_rollouts": 0,
        "exploration_rate_start": 0.40,
        "exploration_rate_end": 0.05,
        "exploration_decay_updates": 12,
        "learning_rate_schedule": "cosine",
        "lr_warmup_updates": 1,
        "lr_decay_updates": 12,
        "actor_lr_min_fraction": 0.15,
        "critic_lr_min_fraction": 0.25,
    }


def _episode(
    *,
    arm: str,
    strategy: str,
    train: bool,
    scenario_seed: int,
    policy_seed: int,
    checkpoint: Path,
    diagnostics: Path,
    population: int,
    stop_time: int,
    rollout_episodes: int,
    branch_horizon: int,
    finalize: bool = False,
) -> dict:
    random.seed(int(scenario_seed))
    np.random.seed(int(scenario_seed) & 0xFFFF_FFFF)
    capacity_token = max(10, int(math.ceil(population / 3)))
    core = nmcc_testbed.build(
        grid=8,
        cell_x=3,
        cell_y=3,
        population=int(population),
        stop_time=int(stop_time),
        hazard_count=2,
        casualty_rate=(40, 9),
        spread_rate=(8, 4),
        panic_rate=0.5,
        scenario_seed=int(scenario_seed),
        candidate_count=8,
        shelter_capacity_token=capacity_token,
    )
    core.maximumShelterForecastDanger = 0.85
    # The matched capacity-controlled evaluation must place all three tokens.
    # Forecast safety and physical feasibility remain hard constraints; the
    # optional route-benefit filter is tested separately and remains enabled
    # in the State College v25 curriculum.
    core.requireCandidateOperationalBenefit = False
    core.minimumCandidateReroutableFraction = 0.0
    core.minimumCandidateRouteTimeSaving = 0.0
    core.minimumCandidateHazardSafetyMargin = 0.0
    learner = (
        _learner_kwargs(arm, rollout_episodes, branch_horizon)
        if strategy == "rl"
        else {"rollout_episodes": int(rollout_episodes)}
    )
    bridge = RLBridge(
        core,
        train_mode=bool(train),
        deployment_strategy=strategy,
        target_active_shelters=int(len(core.shelterDS.shelterList)) + 3,
        # The tiny synthetic map clears much faster than State College. Three
        # early epochs guarantee that every policy can spend the same three
        # capacity tokens before a terminal evacuation censors later actions.
        shelter_action_interval=3,
        policy_seed=int(policy_seed),
        checkpoint_path=str(checkpoint),
        diagnostics_path=str(diagnostics),
        **learner,
    )
    core.rl = bridge
    cells = []
    capacity = 0.0
    for step in range(1, int(core.stopTime)):
        CB.advance_one_timestep(core)
        result = bridge.step(
            simulation_time=step,
            is_terminal=(step == int(core.stopTime) - 1),
        )
        if int(result["decision_made"]):
            cells.append(int(result["selected_cell"]))
            capacity += float(result["capacity_added"])
    result = bridge.end_episode(finalize_rollout=bool(finalize))
    outcome = core.pedDS.result
    return {
        **{key: float(value) for key, value in result.items()},
        "seed": int(scenario_seed),
        "cells": cells,
        "installed_shelters": len(cells),
        "installed_capacity": float(capacity),
        "safe_completed": int(outcome.get("arrival", 0)) + int(outcome.get("evacuated", 0)),
        "casualties": int(outcome.get("casualty", 0)),
        "unfinished": int(outcome.get("unfinished", 0)),
    }


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def _finite_number(value, default=float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _diagnostic_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {key: _finite_number(value) for key, value in raw.items()}
            # The optimizer ledger contains the complete post-action training
            # return, not the action-count-invariant full-episode objective.
            # Keep that distinction explicit when recovering a continuation
            # after a reporting interruption.
            row["objective_episode_return"] = float("nan")
            rows.append(row)
        return rows


def _plot(path: Path, training: dict, evaluation: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for arm, rows in training.items():
        x = np.arange(1, len(rows) + 1)
        training_return = [
            (
                _finite_number(r.get("objective_episode_return"))
                if math.isfinite(_finite_number(r.get("objective_episode_return")))
                else _finite_number(r.get("episode_return"))
            )
            for r in rows
        ]
        axes[0, 0].plot(x, training_return, label=arm)
        metric = (
            "nmcc_pi_validation_top1"
            if arm == "v26_fitted_value"
            else "nmcc_pi_top1_after"
        )
        axes[0, 1].plot(x, [r[metric] for r in rows], label=arm)
        axes[1, 0].plot(x, [r["actor_learning_rate"] for r in rows], label=arm)
    axes[0, 0].set_title("Training return diagnostic")
    axes[0, 1].set_title("Exact-branch top-1 (training / held-out)")
    axes[1, 0].set_title("Actor learning-rate schedule")
    differences = [row["v26_minus_base_return"] for row in evaluation]
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].bar(np.arange(len(differences)), differences)
    axes[1, 1].set_title("Held-out paired v26 - route-saving base return")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    artifacts = output.parent / f"{output.stem}_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    arms = ("v25_staged_score", "v26_fitted_value")
    if args.resume_v26_only:
        if args.overwrite:
            raise ValueError("--resume-v26-only and --overwrite are mutually exclusive")
        if not output.exists():
            raise FileNotFoundError("A completed backtest JSON is required for continuation")
        previous = json.loads(output.read_text(encoding="utf-8"))
        previous_arguments = previous.get("arguments", {})
        for name in (
            "train_episodes", "rollout_episodes", "population", "stop_time",
            "branch_horizon", "seed", "policy_seed",
        ):
            if int(previous_arguments.get(name, -1)) != int(getattr(args, name)):
                raise ValueError(f"Continuation argument {name!r} differs from the saved run")
        training = previous["training"]
        if set(training) != set(arms):
            raise ValueError("Saved backtest does not contain both registered arms")
        checkpoint = artifacts / "v26_fitted_value.pt"
        diagnostics = artifacts / "v26_fitted_value.csv"
        if not checkpoint.exists() or not diagnostics.exists():
            raise FileNotFoundError("v26 continuation checkpoint or diagnostics are missing")
        rows = training["v26_fitted_value"]
        completed = _diagnostic_rows(diagnostics)
        if len(completed) < len(rows):
            raise ValueError("Optimizer diagnostics are shorter than the saved training rows")
        if len(completed) > len(rows):
            rows.extend(completed[len(rows):])
        start = len(rows)
        stop = start + int(args.continuation_episodes)
        for index in range(start, stop):
            rows.append(
                _episode(
                    arm="v26_fitted_value",
                    strategy="rl",
                    train=True,
                    scenario_seed=int(args.seed + index),
                    policy_seed=int(args.policy_seed),
                    checkpoint=checkpoint,
                    diagnostics=diagnostics,
                    population=int(args.population),
                    stop_time=int(args.stop_time),
                    rollout_episodes=int(args.rollout_episodes),
                    branch_horizon=int(args.branch_horizon),
                    finalize=(index == stop - 1),
                )
            )
            print(
                f"[v26_fitted_value] continuation={index + 1}/{stop} "
                f"phase={int(rows[-1]['nmcc_training_phase_index'])} "
                f"return={rows[-1]['objective_episode_return']:.4f} "
                f"heldout_gain_lcb={rows[-1].get('nmcc_pi_validation_gain_lower', 0.0):.4f}",
                flush=True,
            )
    else:
        training = {}
    for arm in (() if args.resume_v26_only else arms):
        checkpoint = artifacts / f"{arm}.pt"
        diagnostics = artifacts / f"{arm}.csv"
        if args.overwrite:
            for path in (checkpoint, diagnostics):
                if path.exists():
                    path.unlink()
        elif checkpoint.exists() or diagnostics.exists():
            raise FileExistsError(f"Backtest artifact exists for {arm}; pass --overwrite")
        rows = []
        for index in range(int(args.train_episodes)):
            rows.append(
                _episode(
                    arm=arm,
                    strategy="rl",
                    train=True,
                    scenario_seed=int(args.seed + index),
                    policy_seed=int(args.policy_seed),
                    checkpoint=checkpoint,
                    diagnostics=diagnostics,
                    population=int(args.population),
                    stop_time=int(args.stop_time),
                    rollout_episodes=int(args.rollout_episodes),
                    branch_horizon=int(args.branch_horizon),
                    finalize=(index == int(args.train_episodes) - 1),
                )
            )
            print(
                f"[{arm}] episode={index + 1}/{args.train_episodes} "
                f"phase={int(rows[-1]['nmcc_training_phase_index'])} "
                f"return={rows[-1]['objective_episode_return']:.4f} "
                f"heldout_gain_lcb={rows[-1].get('nmcc_pi_validation_gain_lower', 0.0):.4f}",
                flush=True,
            )
        training[arm] = rows

    evaluation = []
    for index in range(int(args.eval_episodes)):
        seed = int(args.seed + 10000 + index)
        policies = {}
        for arm in arms:
            policies[arm] = _episode(
                arm=arm,
                strategy="rl",
                train=False,
                scenario_seed=seed,
                policy_seed=int(args.policy_seed),
                checkpoint=artifacts / f"{arm}.pt",
                diagnostics=artifacts / f"{arm}.csv",
                population=int(args.population),
                stop_time=int(args.stop_time),
                rollout_episodes=int(args.rollout_episodes),
                branch_horizon=int(args.branch_horizon),
            )
        policies["heuristic"] = _episode(
            arm="heuristic",
            strategy="heuristic",
            train=False,
            scenario_seed=seed,
            policy_seed=int(args.policy_seed),
            checkpoint=artifacts / "unused.pt",
            diagnostics=artifacts / "unused.csv",
            population=int(args.population),
            stop_time=int(args.stop_time),
            rollout_episodes=int(args.rollout_episodes),
            branch_horizon=int(args.branch_horizon),
        )
        policies["route_saving"] = _episode(
            arm="route_saving",
            strategy="route_saving",
            train=False,
            scenario_seed=seed,
            policy_seed=int(args.policy_seed),
            checkpoint=artifacts / "unused.pt",
            diagnostics=artifacts / "unused.csv",
            population=int(args.population),
            stop_time=int(args.stop_time),
            rollout_episodes=int(args.rollout_episodes),
            branch_horizon=int(args.branch_horizon),
        )
        capacity = {
            name: (row["installed_shelters"], round(row["installed_capacity"], 9))
            for name, row in policies.items()
        }
        if len(set(capacity.values())) != 1:
            raise RuntimeError(f"Equal-capacity gate failed on seed {seed}: {capacity}")
        evaluation.append(
            {
                "seed": seed,
                "policies": policies,
                "capacity": capacity,
                "v26_minus_v25_return": (
                    policies["v26_fitted_value"]["objective_episode_return"]
                    - policies["v25_staged_score"]["objective_episode_return"]
                ),
                "v26_minus_base_return": (
                    policies["v26_fitted_value"]["objective_episode_return"]
                    - policies["route_saving"]["objective_episode_return"]
                ),
            }
        )

    v26 = training["v26_fitted_value"]
    validation_rows = [
        row for row in v26 if row.get("nmcc_pi_validation_states", 0.0) > 0.0
    ]
    gain_rows = [
        row for row in validation_rows
        if math.isfinite(_finite_number(
            row.get("nmcc_pi_validation_gain_lower", float("nan"))
        ))
    ]
    phases = sorted({int(row["nmcc_training_phase_index"]) for row in v26})
    checks = {
        "system_then_control_phases_observed": phases == [0, 1, 3],
        "persistent_replay_grew": v26[-1]["nmcc_pi_replay_episodes"] >= 2.0,
        "heldout_states_evaluated": bool(validation_rows),
        "heldout_metrics_finite": bool(validation_rows) and all(
            math.isfinite(row["nmcc_pi_validation_loss"])
            and math.isfinite(row["nmcc_pi_validation_rank"])
            for row in validation_rows
        ),
        "heldout_correction_gate_opened": any(
            row["nmcc_pi_gate_passed"] > 0.5 for row in gain_rows
        ),
        "positive_heldout_paired_gain_lower_bound": bool(gain_rows)
        and gain_rows[-1]["nmcc_pi_validation_gain_lower"] > 0.0,
        "exploration_decayed": v26[-1]["exploration_rate"] < v26[0]["exploration_rate"],
        "capacity_parity": True,
        "finite_world_model_losses": all(
            math.isfinite(row["nmcc_natural_loss"])
            and math.isfinite(row["nmcc_causal_loss"])
            for row in v26
        ),
    }
    held_out_difference = float(np.mean([
        row["v26_minus_v25_return"] for row in evaluation
    ]))
    held_out_base_difference = float(np.mean([
        row["v26_minus_base_return"] for row in evaluation
    ]))
    checks["held_out_return_improved_over_fixed_base"] = (
        held_out_base_difference > 0.0
    )
    summary = {
        "v25_eval_return": float(np.mean([
            row["policies"]["v25_staged_score"]["objective_episode_return"]
            for row in evaluation
        ])),
        "v26_eval_return": float(np.mean([
            row["policies"]["v26_fitted_value"]["objective_episode_return"]
            for row in evaluation
        ])),
        "v26_minus_v25_return": held_out_difference,
        "route_saving_eval_return": float(np.mean([
            row["policies"]["route_saving"]["objective_episode_return"]
            for row in evaluation
        ])),
        "v26_minus_route_saving_return": held_out_base_difference,
        "v26_validation_gain_lower_mean": _mean(
            gain_rows, "nmcc_pi_validation_gain_lower"
        ),
        "v26_validation_gain_lower_final": (
            float(gain_rows[-1]["nmcc_pi_validation_gain_lower"])
            if gain_rows else float("nan")
        ),
        "v26_validation_top1": _mean(
            validation_rows, "nmcc_pi_validation_top1"
        ),
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
    }
    plot = output.with_suffix(".png")
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engineering_backtest_only": True,
        "arguments": vars(args),
        "summary": summary,
        "checks": checks,
        "training": training,
        "evaluation": evaluation,
        "plot": str(plot),
    }
    _write_json(output, payload)
    _plot(plot, training, evaluation)
    print(json.dumps({"summary": summary, "checks": checks}, indent=2), flush=True)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-episodes", type=int, default=8)
    result.add_argument("--eval-episodes", type=int, default=4)
    result.add_argument("--rollout-episodes", type=int, default=1)
    result.add_argument("--population", type=int, default=120)
    result.add_argument("--stop-time", type=int, default=24)
    result.add_argument("--branch-horizon", type=int, default=12)
    result.add_argument("--seed", type=int, default=25100)
    result.add_argument("--policy-seed", type=int, default=57)
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--resume-v26-only", action="store_true")
    result.add_argument("--continuation-episodes", type=int, default=24)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
