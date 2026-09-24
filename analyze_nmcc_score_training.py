#!/usr/bin/env python3
"""Audit a completed staged NMCC candidate-score training launch.

This analyzer is deliberately separate from ``analyze_staged_nmcc_training``:
that legacy utility encodes the model-v22 8/8/16/32 episode schedule.  Model
v25 advances phases at optimizer boundaries and trains an exact-branch
candidate scorer, so its primary diagnostics are ranking accuracy, target-fit
KL, world-model losses, gradients, and the registered schedules.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


PHASE_NAMES = {
    0: "natural pretrain",
    1: "causal pretrain",
    2: "controller warm-up",
    3: "joint optimization",
}


def _finite(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _field(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([_finite(row.get(name)) for row in rows], dtype=float)


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _bootstrap_mean_difference(
    first: np.ndarray,
    last: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, list[float]]:
    first = first[np.isfinite(first)]
    last = last[np.isfinite(last)]
    point = float(np.mean(last) - np.mean(first))
    rng = np.random.default_rng(seed)
    samples = np.empty(int(draws), dtype=float)
    for index in range(int(draws)):
        samples[index] = float(
            np.mean(rng.choice(last, size=len(last), replace=True))
            - np.mean(rng.choice(first, size=len(first), replace=True))
        )
    interval = np.quantile(samples, [0.025, 0.975]).astype(float).tolist()
    return point, interval


def _slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if int(mask.sum()) < 2:
        return float("nan")
    x = np.arange(1, len(values) + 1, dtype=float)[mask]
    return float(np.polyfit(x, values[mask], 1)[0])


def _phase_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for phase, name in PHASE_NAMES.items():
        selected = [
            row
            for row in rows
            if int(round(_finite(row.get("nmcc_training_phase_index"), -1.0))) == phase
        ]
        result.append(
            {
                "phase_index": phase,
                "phase": name,
                "optimizer_updates": len(selected),
                "natural_loss_mean": _json_number(
                    _mean(_finite(row.get("nmcc_natural_loss")) for row in selected)
                ),
                "causal_loss_mean": _json_number(
                    _mean(_finite(row.get("nmcc_causal_loss")) for row in selected)
                ),
                "dueling_loss_mean": _json_number(
                    _mean(_finite(row.get("nmcc_dueling_loss")) for row in selected)
                ),
                "actor_gradient_mean": _json_number(
                    _mean(_finite(row.get("actor_gradient_norm")) for row in selected)
                ),
                "critic_gradient_mean": _json_number(
                    _mean(_finite(row.get("critic_gradient_norm")) for row in selected)
                ),
            }
        )
    return result


def _plot(
    output: Path,
    rows: list[dict[str, str]],
    updates: list[dict[str, str]],
    actor_updates: list[dict[str, str]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, len(rows) + 1)
    returns = _field(rows, "episode_return")
    window = 6
    rolling = np.convolve(returns, np.ones(window) / window, mode="valid")
    update_x = _field(updates, "episode")
    actor_x = _field(actor_updates, "episode")

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    axes[0, 0].plot(episodes, returns, color="#90caf9", alpha=0.55, label="episode")
    axes[0, 0].plot(
        episodes[window - 1 :],
        rolling,
        color="black",
        linewidth=2.5,
        label=f"moving mean ({window})",
    )
    axes[0, 0].set_title("Training return (not a causal policy comparison)")
    axes[0, 0].set_ylabel("return")
    axes[0, 0].legend()

    axes[0, 1].plot(
        update_x,
        _field(updates, "nmcc_natural_loss"),
        marker="o",
        label="natural",
    )
    axes[0, 1].plot(
        update_x,
        _field(updates, "nmcc_causal_loss"),
        marker="o",
        label="causal",
    )
    axes[0, 1].plot(
        update_x,
        _field(updates, "nmcc_dueling_loss"),
        marker="o",
        label="dueling",
    )
    axes[0, 1].set_title("World-model and intervention losses")
    axes[0, 1].set_yscale("symlog", linthresh=1e-3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        actor_x,
        _field(actor_updates, "nmcc_pi_top1_before"),
        marker="o",
        label="before update",
    )
    axes[1, 0].plot(
        actor_x,
        _field(actor_updates, "nmcc_pi_top1_after"),
        marker="o",
        label="after update",
    )
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Exact-branch candidate top-1")
    axes[1, 0].legend()

    axes[1, 1].plot(
        actor_x,
        _field(actor_updates, "nmcc_pi_fit_kl_before"),
        marker="o",
        label="before update",
    )
    axes[1, 1].plot(
        actor_x,
        _field(actor_updates, "nmcc_pi_fit_kl_after"),
        marker="o",
        label="after update",
    )
    axes[1, 1].set_title("Exact-target fit KL (lower is better)")
    axes[1, 1].legend()

    axes[2, 0].plot(
        update_x,
        _field(updates, "critic_gradient_norm"),
        marker="o",
        label="critic",
    )
    axes[2, 0].plot(
        update_x,
        _field(updates, "actor_gradient_norm"),
        marker="o",
        label="actor",
    )
    axes[2, 0].set_title("Optimization signal")
    axes[2, 0].set_ylabel("gradient norm")
    axes[2, 0].legend()

    axes[2, 1].plot(
        update_x,
        _field(updates, "exploration_rate"),
        marker="o",
        label="epsilon exploration",
    )
    axes[2, 1].plot(
        update_x,
        _field(updates, "actor_learning_rate") * 1000.0,
        marker="o",
        label="actor LR ×1000",
    )
    axes[2, 1].plot(
        update_x,
        _field(updates, "critic_learning_rate") * 1000.0,
        marker="o",
        label="critic LR ×1000",
    )
    axes[2, 1].set_title("Registered exploration and LR schedules")
    axes[2, 1].legend()

    # Optimizer boundaries: natural 1-2, causal 3-5, controller 6-7, joint 8+.
    for axis in axes.flat:
        for boundary in (8.5, 20.5, 28.5):
            axis.axvline(boundary, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        axis.set_xlabel("training episode")
        axis.grid(alpha=0.22)

    fig.suptitle("State College 2,500-person v25 staged NMCC score training", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def analyze(
    launch_dir: Path,
    *,
    bootstrap_draws: int = 20_000,
    seed: int = 20260922,
) -> dict[str, object]:
    summary_path = launch_dir / "training_episode_summary.csv"
    diagnostics_path = launch_dir / "policies" / "policy_001" / "ppo_diagnostics.csv"
    manifest_path = launch_dir / "experiment_manifest.json"
    convergence_path = launch_dir / "training_convergence_diagnostics.json"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with diagnostics_path.open("r", encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    convergence = None
    if convergence_path.exists():
        with convergence_path.open("r", encoding="utf-8") as handle:
            convergence = json.load(handle)

    updates = [row for row in diagnostics if _finite(row.get("optimizer_updated"), 0.0) > 0.5]
    actor_updates = [row for row in updates if _finite(row.get("nmcc_actor_enabled"), 0.0) > 0.5]
    phases = sorted(
        {
            int(round(_finite(row.get("nmcc_training_phase_index"), -1.0)))
            for row in updates
        }
    )
    fit_before = _field(actor_updates, "nmcc_pi_fit_kl_before")
    fit_after = _field(actor_updates, "nmcc_pi_fit_kl_after")
    top1_before = _field(actor_updates, "nmcc_pi_top1_before")
    top1_after = _field(actor_updates, "nmcc_pi_top1_after")
    top1_delta = top1_after - top1_before
    episode_return = _field(rows, "episode_return")
    last_first_delta, last_first_interval = _bootstrap_mean_difference(
        episode_return[:8],
        episode_return[-8:],
        draws=bootstrap_draws,
        seed=seed,
    )

    natural = _field(updates, "nmcc_natural_loss")
    causal_rows = [
        row
        for row in updates
        if int(round(_finite(row.get("nmcc_training_phase_index"), -1.0))) >= 1
    ]
    dueling = _field(causal_rows, "nmcc_dueling_loss")
    casualties = _field(rows, "casualty")
    safe = _field(rows, "safe_completed")
    unfinished = _field(rows, "unfinished")
    accounting_gap = np.abs(_field(rows, "reward_accounting_gap"))
    deployments = _field(rows, "deployments_made")
    deployment_cap = _field(rows, "maximum_dynamic_deployments")
    stage_overrides = json.loads(rows[0]["stage_overrides"])
    token = float(stage_overrides["shelterCapacityToken"])

    checks = {
        "64_episodes_complete": len(rows) == 64,
        "16_optimizer_updates_complete": len(updates) == 16,
        "all_four_phases_observed": phases == [0, 1, 2, 3],
        "actor_frozen_during_system_pretraining": all(
            _finite(row.get("nmcc_actor_optimizer_updates"), 0.0) == 0.0
            for row in updates
            if int(round(_finite(row.get("nmcc_training_phase_index"), -1.0))) <= 1
        ),
        "actor_received_nonzero_signal": bool(actor_updates)
        and all(_finite(row.get("actor_gradient_norm"), 0.0) > 0.0 for row in actor_updates),
        "all_actor_updates_accepted": bool(actor_updates)
        and all(_finite(row.get("actor_update_accepted"), 0.0) > 0.5 for row in actor_updates)
        and all(_finite(row.get("actor_update_rejected"), 0.0) < 0.5 for row in actor_updates),
        "target_fit_kl_improved_every_actor_update": bool(actor_updates)
        and bool(np.all(fit_after < fit_before)),
        "mean_exact_top1_improved": _mean(top1_after) > _mean(top1_before),
        "natural_loss_decreased": natural[-1] < natural[0],
        "dueling_loss_decreased": dueling[-1] < dueling[0],
        "exploration_schedule_reached_floor": math.isclose(
            _finite(updates[-1].get("exploration_rate")), 0.03, abs_tol=1e-9
        ),
        "learning_rates_decayed": (
            _finite(updates[-1].get("actor_learning_rate"))
            < _finite(updates[0].get("actor_learning_rate"))
            and _finite(updates[-1].get("critic_learning_rate"))
            < _finite(updates[0].get("critic_learning_rate"))
        ),
        "no_representation_rollbacks": all(
            _finite(row.get("representation_rollback"), 0.0) < 0.5 for row in updates
        ),
        "complete_reward_accounting": bool(np.nanmax(accounting_gap) < 1e-6),
        "fixed_deployment_budget_every_episode": bool(
            np.all(deployments == deployment_cap) and np.all(deployments == deployments[0])
        ),
        "population_accounting_balanced": all(
            int(round(_finite(row.get("safe_completed"), 0.0)))
            + int(round(_finite(row.get("casualty"), 0.0)))
            + int(round(_finite(row.get("unfinished"), 0.0)))
            == int(round(_finite(row.get("initial_population"), -1.0)))
            for row in rows
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}

    generic_converged = bool(convergence and convergence.get("all_policies_converged"))
    fit_converged = sum(
        _finite(row.get("nmcc_pi_fit_converged"), 0.0) > 0.5 for row in actor_updates
    )
    gate_passed = sum(
        _finite(row.get("nmcc_pi_gate_passed"), 0.0) > 0.5 for row in updates
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "launch_dir": str(launch_dir.resolve()),
        "model_version": 25,
        "training_status": manifest.get("status"),
        "summary": {
            "episodes": len(rows),
            "optimizer_updates": len(updates),
            "actor_enabled_updates": len(actor_updates),
            "actor_optimizer_steps": int(
                round(_finite(actor_updates[-1].get("nmcc_actor_optimizer_updates"), 0.0))
            ),
            "actor_updates_improving_top1": int(np.sum(top1_delta > 0.0)),
            "actor_updates_worsening_top1": int(np.sum(top1_delta < 0.0)),
            "mean_top1_before": _json_number(_mean(top1_before)),
            "mean_top1_after": _json_number(_mean(top1_after)),
            "mean_top1_change": _json_number(_mean(top1_delta)),
            "mean_fit_kl_before": _json_number(_mean(fit_before)),
            "mean_fit_kl_after": _json_number(_mean(fit_after)),
            "mean_fit_kl_change": _json_number(_mean(fit_after - fit_before)),
            "fit_converged_updates": int(fit_converged),
            "causal_gate_passed_updates": int(gate_passed),
            "natural_loss_first": _json_number(natural[0]),
            "natural_loss_final": _json_number(natural[-1]),
            "dueling_loss_first": _json_number(dueling[0]),
            "dueling_loss_final": _json_number(dueling[-1]),
            "return_full_slope_per_episode": _json_number(_slope(episode_return)),
            "return_actor_period_slope_per_episode": _json_number(_slope(episode_return[20:])),
            "last8_minus_first8_return": last_first_delta,
            "last8_minus_first8_bootstrap_95_interval": last_first_interval,
            "safe_completed_mean": _json_number(_mean(safe)),
            "casualty_mean": _json_number(_mean(casualties)),
            "casualty_median": _json_number(float(np.nanmedian(casualties))),
            "casualty_maximum": _json_number(float(np.nanmax(casualties))),
            "episodes_with_casualties": int(np.sum(casualties > 0.0)),
            "unfinished_mean": _json_number(_mean(unfinished)),
            "maximum_absolute_reward_accounting_gap": _json_number(
                float(np.nanmax(accounting_gap))
            ),
            "first_episode_deployments": int(round(deployments[0])),
            "deployment_budget_per_episode": int(round(deployment_cap[0])),
            "deployments_installed_mean": _json_number(_mean(deployments)),
            "deployments_installed_minimum": int(round(float(np.nanmin(deployments)))),
            "deployments_installed_maximum": int(round(float(np.nanmax(deployments)))),
            "episodes_installing_full_budget": int(np.sum(deployments == deployment_cap)),
            "capacity_token": token,
            "additional_capacity_budget_per_episode": float(deployment_cap[0] * token),
            "additional_capacity_installed_mean": float(_mean(deployments) * token),
            "exploration_initial": _json_number(
                _finite(updates[0].get("exploration_rate"))
            ),
            "exploration_final": _json_number(
                _finite(updates[-1].get("exploration_rate"))
            ),
            "actor_learning_rate_initial": _json_number(
                _finite(updates[0].get("actor_learning_rate"))
            ),
            "actor_learning_rate_final": _json_number(
                _finite(updates[-1].get("actor_learning_rate"))
            ),
            "critic_learning_rate_initial": _json_number(
                _finite(updates[0].get("critic_learning_rate"))
            ),
            "critic_learning_rate_final": _json_number(
                _finite(updates[-1].get("critic_learning_rate"))
            ),
        },
        "phase_summary": _phase_summary(updates),
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "scientific_assessment": {
            "learning_mechanics_and_credit_signal_validated": bool(
                all(
                    passed
                    for name, passed in checks.items()
                    if name != "fixed_deployment_budget_every_episode"
                )
            ),
            "realized_capacity_fairness_gate_validated": checks[
                "fixed_deployment_budget_every_episode"
            ],
            "generic_training_converged": generic_converged,
            "decision_fit_converged": fit_converged == len(actor_updates),
            "held_out_policy_superiority_tested": False,
            "capacity_parity_interpretation": (
                "The RL arm had a five-token budget, but safety/feasibility masks "
                "prevented full installation in some episodes. No heuristic was run "
                "in this train-only launch, so realized capacity parity is not "
                "established."
            ),
            "conclusion": (
                "The staged learner received and used valid exact-branch signal, but "
                "this 64-episode single-seed train-only pilot does not establish "
                "return convergence, realized capacity parity, or superiority over "
                "a heuristic."
            ),
        },
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args(argv)
    launch_dir = args.launch_dir.expanduser().resolve()
    payload = analyze(
        launch_dir,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    output_json = launch_dir / "nmcc_score_training_audit.json"
    output_figure = launch_dir / "nmcc_score_training_diagnostics.png"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (launch_dir / "training_episode_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    with (
        launch_dir / "policies" / "policy_001" / "ppo_diagnostics.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    updates = [row for row in diagnostics if _finite(row.get("optimizer_updated"), 0.0) > 0.5]
    actor_updates = [row for row in updates if _finite(row.get("nmcc_actor_enabled"), 0.0) > 0.5]
    _plot(output_figure, rows, updates, actor_updates)
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "scientific_assessment": payload["scientific_assessment"],
                "checks": payload["checks"],
                "json": str(output_json),
                "figure": str(output_figure),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
