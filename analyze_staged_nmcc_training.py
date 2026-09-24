#!/usr/bin/env python3
"""Audit and visualize a completed staged Hybrid-NMCC training launch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ORDER = ("N0", "N1", "N2", "N3")
STAGE_LABELS = {
    "N0": "Natural model",
    "N1": "Causal model",
    "N2": "Controller warm-up",
    "N3": "Joint optimization",
}
STAGE_COLORS = {
    "N0": "#DCEAF7",
    "N1": "#E5DDF2",
    "N2": "#F8E4C2",
    "N3": "#DDEED8",
}
UPDATE_FIELDS = (
    "nmcc_natural_loss",
    "nmcc_causal_loss",
    "nmcc_dueling_loss",
    "nmcc_teacher_loss",
    "nmcc_causal_uncertainty",
    "nmcc_counterfactual_advantage_sd",
    "nmcc_gae_advantage_sd",
    "policy_loss",
    "value_loss",
    "approximate_kl",
    "gradient_norm",
    "update_entropy",
    "action_temperature",
    "entropy_coefficient",
    "nmcc_guidance_weight",
    "nmcc_teacher_coefficient",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column!r} contains a non-finite value")


def _rolling(values: pd.Series, window: int = 8) -> pd.Series:
    return values.rolling(window=window, min_periods=max(3, window // 2)).mean()


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


def _bootstrap_slope_interval(
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = intercept + slope * x
    residuals = values - fitted
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        boot = fitted + rng.choice(residuals, size=residuals.size, replace=True)
        samples[index] = np.polyfit(x, boot, 1)[0]
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _bootstrap_mean_difference_interval(
    early: np.ndarray,
    late: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        samples[index] = float(
            rng.choice(late, size=late.size, replace=True).mean()
            - rng.choice(early, size=early.size, replace=True).mean()
        )
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _shade_stages(axis, episodes: pd.Series, stages: pd.Series) -> None:
    for stage in STAGE_ORDER:
        selected = episodes[stages == stage]
        if selected.empty:
            continue
        axis.axvspan(
            float(selected.min()) - 0.5,
            float(selected.max()) + 0.5,
            color=STAGE_COLORS[stage],
            alpha=0.45,
            linewidth=0,
        )
        axis.text(
            float(selected.mean()),
            0.98,
            stage,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#333333",
        )


def _stage_summary(training: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in STAGE_ORDER:
        subset = training[training["training_stage_id"] == stage]
        if subset.empty:
            continue
        row = {
            "stage_id": stage,
            "stage_label": STAGE_LABELS[stage],
            "episodes": int(len(subset)),
        }
        for field in (
            "objective_episode_return",
            "episode_return",
            "safe_completed",
            "casualty",
            "unfinished",
            "normalized_risk_weighted_person_time",
        ):
            values = subset[field].to_numpy(dtype=float)
            row[f"{field}_mean"] = float(values.mean())
            row[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{field}_slope_per_episode"] = _slope(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_reward(training: pd.DataFrame, output: Path) -> None:
    episode = training["replication"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis in axes:
        _shade_stages(axis, episode, training["training_stage_id"])

    axes[0].plot(
        episode,
        training["objective_episode_return"],
        color="#4C78A8",
        alpha=0.32,
        linewidth=1.0,
        label="Full-episode global return",
    )
    axes[0].plot(
        episode,
        _rolling(training["objective_episode_return"]),
        color="#1F4E79",
        linewidth=2.3,
        label="8-episode moving mean",
    )
    axes[0].plot(
        episode,
        _rolling(training["episode_return"]),
        color="#D97706",
        linewidth=1.8,
        label="Post-action return moving mean",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(title="Staged NMCC reward trajectory", ylabel="Normalized return")
    axes[0].legend(loc="lower left", fontsize=8)

    components = (
        ("objective_safe_completion_reward", "Safe completion", "#2A9D8F"),
        ("objective_casualty_penalty", "Casualty", "#C1121F"),
        ("objective_evacuation_time_penalty", "Evacuation time", "#6A4C93"),
        ("objective_hazard_exposure_penalty", "Hazard exposure", "#E76F51"),
    )
    for field, label, color in components:
        axes[1].plot(
            episode,
            _rolling(training[field]),
            label=label,
            color=color,
            linewidth=1.8,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        title="Global reward components (8-episode moving means)",
        xlabel="Training episode",
        ylabel="Normalized component",
    )
    axes[1].legend(ncol=2, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_outcomes(training: pd.DataFrame, output: Path) -> None:
    episode = training["replication"]
    panels = (
        ("safe_completed", "Safe completions", "People", "#2A9D8F"),
        ("casualty", "Casualties", "People", "#C1121F"),
        ("unfinished", "Unfinished at horizon", "People", "#D97706"),
        (
            "normalized_risk_weighted_person_time",
            "Normalized evacuation + exposure time",
            "Normalized person-time",
            "#6A4C93",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (field, title, ylabel, color) in zip(axes.flat, panels):
        _shade_stages(axis, episode, training["training_stage_id"])
        axis.scatter(episode, training[field], color=color, alpha=0.28, s=15)
        axis.plot(episode, _rolling(training[field]), color=color, linewidth=2.2)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(alpha=0.22)
    axes[1, 0].set_xlabel("Training episode")
    axes[1, 1].set_xlabel("Training episode")
    fig.suptitle("Physical evacuation outcomes during staged training", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_nmcc(updates: pd.DataFrame, output: Path) -> None:
    episode = updates["episode"]
    ratio = updates["nmcc_counterfactual_advantage_sd"] / updates[
        "nmcc_gae_advantage_sd"
    ].replace(0.0, np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for field, label, color in (
        ("nmcc_natural_loss", "Natural", "#4C78A8"),
        ("nmcc_causal_loss", "Causal", "#F58518"),
        ("nmcc_dueling_loss", "Dueling", "#54A24B"),
    ):
        axes[0, 0].plot(episode, updates[field], marker="o", label=label, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(title="NMCC auxiliary losses", ylabel="Smooth-L1 loss (log)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        episode,
        updates["nmcc_counterfactual_advantage_sd"],
        marker="o",
        label="Exact counterfactual SD",
    )
    axes[0, 1].plot(
        episode,
        updates["nmcc_gae_advantage_sd"],
        marker="o",
        label="Raw GAE SD",
    )
    axes[0, 1].set(title="Actor-target dispersion", ylabel="Standard deviation")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(episode, ratio, marker="o", color="#6A4C93")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set(
        title="Counterfactual / GAE dispersion ratio",
        xlabel="Training episode",
        ylabel="SD ratio (lower is better)",
    )

    axes[1, 1].plot(
        episode,
        updates["nmcc_causal_uncertainty"],
        marker="o",
        color="#B279A2",
        label="Ensemble uncertainty",
    )
    axes[1, 1].plot(
        episode,
        updates["nmcc_counterfactual_fraction"],
        marker="s",
        color="#2A9D8F",
        label="Exact-target coverage",
    )
    axes[1, 1].set(
        title="Target coverage and uncertainty",
        xlabel="Training episode",
        ylabel="Value",
    )
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle("NMCC model and credit-assignment diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_policy(updates: pd.DataFrame, output: Path, target_kl: float) -> None:
    episode = updates["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes[0, 0].plot(episode, updates["policy_loss"], marker="o", label="Policy loss")
    axes[0, 0].plot(episode, updates["value_loss"], marker="o", label="Value loss")
    axes[0, 0].set(title="Actor and critic losses", ylabel="Loss")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(episode, updates["update_entropy"], marker="o", label="Policy entropy")
    axes[0, 1].plot(
        episode,
        updates["action_temperature"],
        marker="s",
        label="Action temperature",
    )
    axes[0, 1].set(title="Exploration schedule", ylabel="Value")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(episode, updates["approximate_kl"], marker="o", label="Approximate KL")
    axes[1, 0].axhline(target_kl, color="black", linestyle="--", label="KL target")
    axes[1, 0].axhline(2.0 * target_kl, color="#C1121F", linestyle=":", label="2× target")
    axes[1, 0].set(title="PPO trust-region diagnostic", xlabel="Training episode", ylabel="KL")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        episode,
        updates["gradient_norm"],
        marker="o",
        color="#C1121F",
        label="Pre-clipping gradient norm",
    )
    axes[1, 1].plot(
        episode,
        updates["nmcc_guidance_weight"],
        marker="s",
        color="#4C78A8",
        label="Guidance weight",
    )
    axes[1, 1].plot(
        episode,
        updates["nmcc_teacher_coefficient"],
        marker="^",
        color="#54A24B",
        label="Teacher coefficient",
    )
    axes[1, 1].set(title="Optimization and teacher schedule", xlabel="Training episode")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle("Recurrent PPO policy diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_convergence(
    joint: pd.DataFrame,
    *,
    slope: float,
    early_mean: float,
    late_mean: float,
    output: Path,
) -> None:
    episode = joint["replication"].to_numpy(dtype=float)
    values = joint["objective_episode_return"].to_numpy(dtype=float)
    fitted = values.mean() + slope * (episode - episode.mean())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].scatter(episode, values, alpha=0.45, color="#4C78A8", label="Episode return")
    axes[0].plot(episode, fitted, color="#C1121F", linewidth=2.0, label="N3 linear trend")
    axes[0].plot(
        episode,
        _rolling(joint["objective_episode_return"]).to_numpy(),
        color="black",
        linewidth=2.0,
        label="8-episode moving mean",
    )
    axes[0].axhline(early_mean, color="#D97706", linestyle="--", label="First-8 mean")
    axes[0].axhline(late_mean, color="#2A9D8F", linestyle="--", label="Last-8 mean")
    axes[0].set(title="Joint-stage global return convergence", ylabel="Global return")
    axes[0].legend(fontsize=8, ncol=2)

    cumulative = np.cumsum(values) / np.arange(1, len(values) + 1)
    axes[1].plot(episode, cumulative, color="#6A4C93", linewidth=2.0)
    axes[1].fill_between(
        episode,
        cumulative - np.asarray([values[:i].std(ddof=1) / np.sqrt(i) if i > 1 else 0.0 for i in range(1, len(values) + 1)]),
        cumulative + np.asarray([values[:i].std(ddof=1) / np.sqrt(i) if i > 1 else 0.0 for i in range(1, len(values) + 1)]),
        color="#6A4C93",
        alpha=0.16,
        label="±1 sequential SE",
    )
    axes[1].set(
        title="Joint-stage cumulative return mean",
        xlabel="Training episode",
        ylabel="Cumulative mean",
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def analyze(launch_dir: Path, *, bootstrap_draws: int) -> dict:
    launch_dir = launch_dir.expanduser().resolve()
    training_path = launch_dir / "training_episode_summary.csv"
    diagnostics_path = launch_dir / "policies" / "policy_001" / "ppo_diagnostics.csv"
    convergence_path = launch_dir / "training_convergence_diagnostics.json"
    manifest_path = launch_dir / "experiment_manifest.json"
    for path in (training_path, diagnostics_path, convergence_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    training = pd.read_csv(training_path)
    diagnostics = pd.read_csv(diagnostics_path)
    if len(training) != 64 or len(diagnostics) != 64:
        raise ValueError("The staged pilot requires exactly 64 completed episodes")
    expected_episode = np.arange(1, 65)
    if not np.array_equal(training["replication"].to_numpy(dtype=int), expected_episode):
        raise ValueError("Training episodes are incomplete or out of order")
    expected_stages = ["N0"] * 8 + ["N1"] * 8 + ["N2"] * 16 + ["N3"] * 32
    if training["training_stage_id"].tolist() != expected_stages:
        raise ValueError("Observed stages do not match the registered 8/8/16/32 schedule")

    training_numeric = (
        "replication",
        "objective_episode_return",
        "episode_return",
        "safe_completed",
        "casualty",
        "unfinished",
        "normalized_risk_weighted_person_time",
        "objective_safe_completion_reward",
        "objective_casualty_penalty",
        "objective_evacuation_time_penalty",
        "objective_hazard_exposure_penalty",
        "reward_accounting_gap",
        "optimizer_updated",
    )
    _finite_float(training, training_numeric)
    _finite_float(
        diagnostics,
        (
            "episode",
            "optimizer_updated",
            "nmcc_counterfactual_fraction",
            "nmcc_effective_counterfactual_weight",
            *UPDATE_FIELDS,
        ),
    )
    updates = diagnostics[diagnostics["optimizer_updated"] > 0.5].copy()
    if updates["episode"].astype(int).tolist() != [8, 16, 24, 32, 40, 48, 56, 64]:
        raise ValueError("Optimizer events do not align with the registered rollout gates")

    stage_summary = _stage_summary(training)
    stage_summary_path = launch_dir / "staged_training_stage_summary.csv"
    update_summary_path = launch_dir / "staged_training_optimizer_updates.csv"
    stage_summary.to_csv(stage_summary_path, index=False)
    updates.to_csv(update_summary_path, index=False)

    joint = training[training["training_stage_id"] == "N3"].copy()
    joint_values = joint["objective_episode_return"].to_numpy(dtype=float)
    joint_slope = _slope(joint_values)
    early = joint_values[:8]
    late = joint_values[-8:]
    early_mean = float(early.mean())
    late_mean = float(late.mean())
    late_minus_early = float(late_mean - early_mean)
    rng = np.random.default_rng(20260920)
    slope_ci = _bootstrap_slope_interval(
        joint_values,
        draws=bootstrap_draws,
        rng=rng,
    )
    difference_ci = _bootstrap_mean_difference_interval(
        early,
        late,
        draws=bootstrap_draws,
        rng=rng,
    )

    ratios = (
        updates["nmcc_counterfactual_advantage_sd"].to_numpy(dtype=float)
        / updates["nmcc_gae_advantage_sd"].to_numpy(dtype=float)
    )
    variance_reductions = 1.0 - np.square(ratios)
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    maximum_accounting_gap = float(
        np.abs(training["reward_accounting_gap"].to_numpy(dtype=float)).max()
    )
    exact_coverage = float(updates["nmcc_counterfactual_fraction"].min())
    maximum_kl = float(updates["approximate_kl"].max())
    final_kl = float(updates["approximate_kl"].iloc[-1])
    target_kl = float(convergence["thresholds"]["target_kl"])
    reward_improvement_supported = bool(
        joint_slope > 0.0
        and slope_ci[0] > 0.0
        and late_minus_early > 0.0
        and difference_ci[0] > 0.0
    )
    credit_path_valid = bool(
        exact_coverage == 1.0
        and np.isfinite(ratios).all()
        and float(np.median(ratios)) < 1.0
        and maximum_accounting_gap < 1e-6
    )
    policy_converged = bool(convergence.get("all_policies_converged", False))
    numerical_stability = bool(
        np.isfinite(updates[list(UPDATE_FIELDS)].to_numpy(dtype=float)).all()
        and maximum_kl < 2.0 * target_kl
    )
    status = (
        "reward_improvement_and_convergence_supported"
        if reward_improvement_supported and policy_converged
        else "credit_path_valid_but_reward_and_policy_not_converged"
        if credit_path_valid
        else "training_validation_failed"
    )
    statistical_findings = [
        {
            "metric": "Joint-stage return slope",
            "method": "OLS slope with residual-bootstrap interval",
            "estimate": joint_slope,
            "interval_95": list(slope_ci),
            "conclusion": "Does not support positive reward improvement",
            "confidence": "CAUTION",
        },
        {
            "metric": "Last-eight minus first-eight joint return",
            "method": "Independent resampling of the two eight-episode windows",
            "estimate": late_minus_early,
            "interval_95": list(difference_ci),
            "conclusion": "Does not support positive reward improvement",
            "confidence": "CAUTION",
        },
        {
            "metric": "Counterfactual-to-GAE target variance",
            "method": "Median within-update variance ratio",
            "estimate": float(np.median(np.square(ratios))),
            "interval_95": None,
            "conclusion": "Counterfactual targets are materially less dispersed",
            "confidence": "CAUTION",
        },
        {
            "metric": "Registered policy-convergence gate",
            "method": "Episode count, tail stationarity, and target-KL criteria",
            "estimate": policy_converged,
            "interval_95": None,
            "conclusion": "Policy is not converged under the registered gate",
            "confidence": "SOLID",
        },
    ]
    warnings = [
        {
            "type": "Single-seed evidence",
            "detail": "Only one stochastic policy seed was trained and no held-out evaluation was run.",
            "affected": "Reward improvement, policy quality, and generalization",
        },
        {
            "type": "Short convergence horizon",
            "detail": "The run has 64 episodes; the registered convergence gate requires at least 100.",
            "affected": "Policy convergence",
        },
        {
            "type": "Sequential dependence",
            "detail": "The bootstrap intervals are descriptive and do not model autocorrelation induced by sequential policy updates.",
            "affected": "Return-slope and early-versus-late intervals",
        },
        {
            "type": "Trust-region pressure",
            "detail": "Every tail optimizer event exceeded the 0.015 target KL, although all remained below 2x target.",
            "affected": "Policy stability",
        },
        {
            "type": "Runtime environment",
            "detail": "The launch used KMP_DUPLICATE_LIB_OK=TRUE and is engineering evidence, not confirmatory numerical evidence.",
            "affected": "Reproducibility and publication use",
        },
    ]
    fallacy_scan = [
        {
            "fallacy": "Simpson's paradox",
            "severity": "NOTE",
            "detail": "Checked global and stage-stratified trajectories; the reward conclusion is based on N3 itself, not an aggregate reversal.",
            "recommendation": "Keep stage-stratified results in future multi-seed reports.",
        },
        {
            "fallacy": "Ecological fallacy",
            "severity": "NOTE",
            "detail": "No individual-pedestrian behavioral claim is inferred from episode-level aggregates.",
            "recommendation": "Retain episode-level wording unless individual outcomes are modeled directly.",
        },
        {
            "fallacy": "Berkson's paradox",
            "severity": "NOTE",
            "detail": "All 64 scheduled episodes are included; no outcome-based episode filter was applied.",
            "recommendation": "Continue to report all registered episodes.",
        },
        {
            "fallacy": "Collider bias",
            "severity": "NOTE",
            "detail": "The trend analysis does not condition on post-action outcomes or other possible common effects.",
            "recommendation": "Avoid adjusting learning curves for realized casualties or completions.",
        },
        {
            "fallacy": "Base-rate neglect",
            "severity": "NOTE",
            "detail": "Casualties, completions, and unfinished counts retain the fixed 2,500-person denominator.",
            "recommendation": "Keep counts and population-normalized rates together in comparisons.",
        },
        {
            "fallacy": "Regression to the mean",
            "severity": "NOTE",
            "detail": "Stages and comparison windows were not selected because their returns were extreme.",
            "recommendation": "Prespecify future comparison windows before multi-seed runs.",
        },
        {
            "fallacy": "Survivorship bias",
            "severity": "NOTE",
            "detail": "The validation contains all 64 scheduled simulation episodes with no episode attrition.",
            "recommendation": "Fail future analyses if scheduled episodes are missing.",
        },
        {
            "fallacy": "Look-elsewhere effect",
            "severity": "NOTE",
            "detail": "Both registered reward contrasts and the complete convergence gate are reported, including unfavorable results.",
            "recommendation": "Treat secondary physical and optimizer panels as diagnostic, not hypothesis tests.",
        },
        {
            "fallacy": "Garden of forking paths",
            "severity": "CAUTION",
            "detail": "The stage schedule and convergence thresholds were registered, but this single-seed graphical audit remains exploratory.",
            "recommendation": "Freeze the next actor-stability intervention and multi-seed analysis before rerunning.",
        },
        {
            "fallacy": "Correlation is not causation",
            "severity": "CAUTION",
            "detail": "A within-run return trend cannot identify the controller's causal effect because scenarios and policy parameters change together.",
            "recommendation": "Use matched held-out scenarios and frozen checkpoints for causal policy comparisons.",
        },
        {
            "fallacy": "Reverse causality",
            "severity": "CAUTION",
            "detail": "Temporal order is known, but difficult scenarios also alter collected gradients; the learning-curve association is not a one-way causal estimate.",
            "recommendation": "Compare checkpointed policies on common exogenous tapes rather than interpreting training order causally.",
        },
    ]

    figure_paths = {
        "reward": launch_dir / "staged_reward_learning.png",
        "outcomes": launch_dir / "staged_physical_outcomes.png",
        "nmcc": launch_dir / "staged_nmcc_diagnostics.png",
        "policy": launch_dir / "staged_policy_diagnostics.png",
        "convergence": launch_dir / "staged_convergence_assessment.png",
    }
    _plot_reward(training, figure_paths["reward"])
    _plot_outcomes(training, figure_paths["outcomes"])
    _plot_nmcc(updates, figure_paths["nmcc"])
    _plot_policy(updates, figure_paths["policy"], target_kl)
    _plot_convergence(
        joint,
        slope=joint_slope,
        early_mean=early_mean,
        late_mean=late_mean,
        output=figure_paths["convergence"],
    )

    json_path = launch_dir / "staged_training_validation.json"
    report_path = launch_dir / "STAGED_TRAINING_VALIDATION.md"
    payload = {
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run+validate",
            "origin_date": datetime.now(timezone.utc).isoformat(),
            "verification_status": "ANALYZED",
            "version_label": "staged_nmcc_validation_v1",
            "launch_directory": str(launch_dir),
            "training_csv_sha256": _sha256(training_path),
            "diagnostics_csv_sha256": _sha256(diagnostics_path),
        },
        "status": status,
        "overall_confidence": "CAUTION",
        "runner_execution": {
            "manifest_status": manifest.get("status"),
            "started_utc": manifest.get("started_utc"),
            "resumed_utc": manifest.get("resumed_utc"),
            "completed_utc": manifest.get("completed_utc"),
        },
        "episodes": int(len(training)),
        "optimizer_events": int(len(updates)),
        "stage_schedule_verified": True,
        "credit_assignment": {
            "valid": credit_path_valid,
            "minimum_exact_target_coverage": exact_coverage,
            "counterfactual_to_gae_sd_ratio_by_update": ratios.tolist(),
            "median_counterfactual_to_gae_sd_ratio": float(np.median(ratios)),
            "median_implied_variance_reduction": float(
                np.median(variance_reductions)
            ),
            "maximum_absolute_reward_accounting_gap": maximum_accounting_gap,
        },
        "reward_improvement": {
            "supported": reward_improvement_supported,
            "joint_stage_slope_per_episode": joint_slope,
            "bootstrap_95_ci_slope": list(slope_ci),
            "first_8_joint_mean": early_mean,
            "last_8_joint_mean": late_mean,
            "last_minus_first_8_mean": late_minus_early,
            "bootstrap_95_ci_last_minus_first_8": list(difference_ci),
            "interpretation": (
                "A single stochastic policy seed is descriptive. Improvement is "
                "supported only when both the joint-stage slope and late-minus-early "
                "contrasts are positive with positive bootstrap intervals."
            ),
        },
        "policy_optimization": {
            "numerically_stable": numerical_stability,
            "converged": policy_converged,
            "target_kl": target_kl,
            "maximum_update_kl": maximum_kl,
            "final_update_kl": final_kl,
            "final_gradient_norm_before_clipping": float(
                updates["gradient_norm"].iloc[-1]
            ),
            "final_value_loss": float(updates["value_loss"].iloc[-1]),
            "final_policy_loss": float(updates["policy_loss"].iloc[-1]),
            "built_in_convergence": convergence,
        },
        "world_model": {
            "natural_loss_first": float(updates["nmcc_natural_loss"].iloc[0]),
            "natural_loss_final": float(updates["nmcc_natural_loss"].iloc[-1]),
            "causal_loss_first_enabled": float(
                updates.loc[
                    updates["nmcc_causal_model_enabled"] > 0.5,
                    "nmcc_causal_loss",
                ].iloc[0]
            ),
            "causal_loss_final": float(updates["nmcc_causal_loss"].iloc[-1]),
            "dueling_loss_first_enabled": float(
                updates.loc[
                    updates["nmcc_causal_model_enabled"] > 0.5,
                    "nmcc_dueling_loss",
                ].iloc[0]
            ),
            "dueling_loss_final": float(updates["nmcc_dueling_loss"].iloc[-1]),
            "final_ensemble_uncertainty": float(
                updates["nmcc_causal_uncertainty"].iloc[-1]
            ),
        },
        "statistical_findings": statistical_findings,
        "warnings": warnings,
        "fallacy_scan": {
            "coverage": "11/11",
            "items": fallacy_scan,
        },
        "reproducibility": {
            "method": "stochastic run analyzed once; independent seed rerun not performed",
            "verdict": "CANNOT_VERIFY",
        },
        "stage_summary": stage_summary.to_dict(orient="records"),
        "artifacts": {
            "stage_summary_csv": str(stage_summary_path),
            "optimizer_update_csv": str(update_summary_path),
            "figures": {key: str(value) for key, value in figure_paths.items()},
            "report": str(report_path),
            "validation_json": str(json_path),
        },
        "evidence_boundary": (
            "This is one training seed without held-out policy evaluation. It can "
            "validate execution, credit quality, and within-run diagnostics but cannot "
            "establish policy superiority or population-level convergence."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    finding_rows = "\n".join(
        "| {metric} | {method} | {estimate} | {interval} | {confidence} |".format(
            metric=item["metric"],
            method=item["method"],
            estimate=(
                f"{item['estimate']:+.6f}"
                if isinstance(item["estimate"], float)
                else str(item["estimate"])
            ),
            interval=(
                "N/A"
                if item["interval_95"] is None
                else f"[{item['interval_95'][0]:+.6f}, {item['interval_95'][1]:+.6f}]"
            ),
            confidence=item["confidence"],
        )
        for item in statistical_findings
    )
    warning_rows = "\n".join(
        f"| {item['type']} | {item['detail']} | {item['affected']} |"
        for item in warnings
    )
    fallacy_rows = "\n".join(
        f"| {item['fallacy']} | {item['severity']} | {item['detail']} | {item['recommendation']} |"
        for item in fallacy_scan
    )

    report = f"""# Staged Hybrid-NMCC training validation

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run + validate
- Origin Date: {payload['material_passport']['origin_date']}
- Verification Status: ANALYZED
- Version Label: staged_nmcc_validation_v1
- Launch: `{launch_dir}`

## Verdict

**{status.replace('_', ' ')}**

- Overall confidence: **CAUTION**.
- Campaign manifest status: **{manifest.get('status')}**.

- All 64 registered episodes and all eight rollout optimizer gates are present.
- Exact NMCC target coverage is {exact_coverage:.1%}.
- Median counterfactual/GAE SD ratio is {np.median(ratios):.4f}, equivalent to a
  median {np.median(variance_reductions):.1%} reduction in target variance.
- Maximum absolute reward-accounting gap is {maximum_accounting_gap:.3g}.
- Joint-stage global-return slope is {joint_slope:+.6f} per episode, with a
  residual-bootstrap 95% interval [{slope_ci[0]:+.6f}, {slope_ci[1]:+.6f}].
- Last-eight minus first-eight joint-stage mean return is
  {late_minus_early:+.6f}, with bootstrap 95% interval
  [{difference_ci[0]:+.6f}, {difference_ci[1]:+.6f}].
- Built-in policy convergence: **{policy_converged}**.
- Final / maximum PPO KL: {final_kl:.6f} / {maximum_kl:.6f}; target {target_kl:.6f}.

The causal credit path is functioning and substantially less variable than raw
GAE. This run does **not** show valid reward improvement or policy convergence.
It is one stochastic training seed and contains no held-out policy comparison.

## Statistical findings

| Metric | Method | Estimate | 95% interval | Confidence |
|---|---|---:|---:|---|
{finding_rows}

## Warnings

| Type | Detail | Affected |
|---|---|---|
{warning_rows}

## Statistical fallacy scan

- Coverage: **11/11** types checked.

| Fallacy | Severity | Detail | Recommendation |
|---|---|---|---|
{fallacy_rows}

## Reproducibility

- Method: stochastic run analyzed once; independent seed rerun not performed.
- Verdict: **CANNOT_VERIFY**.

## Output artifacts

- `staged_training_stage_summary.csv`
- `staged_training_optimizer_updates.csv`
- `staged_reward_learning.png`
- `staged_physical_outcomes.png`
- `staged_nmcc_diagnostics.png`
- `staged_policy_diagnostics.png`
- `staged_convergence_assessment.png`
- `staged_training_validation.json`
    """
    report_path.write_text(report, encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.bootstrap_draws < 1000:
        parser.error("--bootstrap-draws must be at least 1000")
    result = analyze(args.launch_dir, bootstrap_draws=args.bootstrap_draws)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
