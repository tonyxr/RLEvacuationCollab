#!/usr/bin/env python3
"""Analyze staged v34 training and held-out shelter-placement evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_COLORS = ("#DCEAF7", "#F8E4C2", "#DDEED8", "#E8DDF0")
ROLLING_WINDOW = 8


def _numeric(frame: pd.DataFrame, names: tuple[str, ...]) -> None:
    for name in names:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")


def _rolling(values: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    return values.rolling(window=window, min_periods=max(2, window // 2)).mean()


def _stage_spans(training: pd.DataFrame) -> list[tuple[str, int, int]]:
    spans = []
    if "training_stage_id" not in training:
        return spans
    for stage, rows in training.groupby("training_stage_id", sort=False):
        episodes = rows["replication"].astype(int)
        spans.append((str(stage), int(episodes.min()), int(episodes.max())))
    return spans


def _shade(axis, spans: list[tuple[str, int, int]]) -> None:
    for index, (stage, first, last) in enumerate(spans):
        axis.axvspan(
            first - 0.5,
            last + 0.5,
            color=STAGE_COLORS[index % len(STAGE_COLORS)],
            alpha=0.35,
            linewidth=0,
        )
        axis.text(
            (first + last) / 2.0,
            0.98,
            stage,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )


def _decision_rows(root: Path, mode: str) -> pd.DataFrame:
    rows = []
    base = root / mode
    if not base.exists():
        return pd.DataFrame()
    for path in sorted(base.rglob("progress.csv")):
        match = re.search(r"rep_(\d+)_local", str(path))
        if match is None:
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        if "decision_made" not in frame:
            continue
        frame = frame[pd.to_numeric(frame["decision_made"], errors="coerce") > 0].copy()
        if frame.empty:
            continue
        parts = path.parts
        known = (
            "rl",
            "heuristic",
            "risk_reduction",
            "route_saving",
            "hazard_weighted",
            "accessibility_deficit",
            "random",
        )
        strategy = next((name for name in known if name in parts), "unknown")
        stage = "evaluation"
        if mode == "training":
            policy_positions = [i for i, part in enumerate(parts) if part.startswith("policy_")]
            if policy_positions and policy_positions[-1] + 1 < len(parts):
                stage = parts[policy_positions[-1] + 1]
        frame["episode"] = int(match.group(1))
        frame["strategy"] = strategy
        frame["stage"] = stage
        frame["evaluation_variant_id"] = next(
            (
                part
                for part in parts
                if part.startswith("state_college_n")
                and (part.endswith("_action") or part.endswith("_environment"))
            ),
            "",
        )
        frame = frame.sort_values("timestep")
        frame["decision_order"] = np.arange(1, len(frame) + 1)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    _numeric(
        result,
        (
            "reward",
            "selected_cell",
            "rerouted_population",
            "raw_remaining_risk_time_reduction",
            "raw_plain_time_reduction",
            "raw_hazard_exposure_reduction",
        ),
    )
    return result


def _normalized_components(training: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    population = pd.to_numeric(
        training.get("initial_population", training.get("stage_population", 1.0)),
        errors="coerce",
    ).replace(0.0, np.nan)
    horizon = pd.to_numeric(training.get("horizon_minutes", 120.0), errors="coerce")
    denominator = population * horizon
    plain = pd.to_numeric(training.get("raw_plain_time_reduction", 0.0), errors="coerce") / denominator
    exposure = pd.to_numeric(
        training.get("raw_hazard_exposure_reduction", 0.0), errors="coerce"
    ) / denominator
    return plain.fillna(0.0), exposure.fillna(0.0), denominator


def _plot_reward(training: pd.DataFrame, output: Path) -> None:
    spans = _stage_spans(training)
    episodes = training["replication"]
    plain, exposure, _ = _normalized_components(training)
    raw = plain + exposure
    reward = training["episode_return"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for axis in axes:
        _shade(axis, spans)
        axis.grid(alpha=0.22)
    axes[0].plot(episodes, reward, color="#4C78A8", alpha=0.28, linewidth=0.9)
    axes[0].plot(episodes, _rolling(reward), color="#174A7E", linewidth=2.4)
    axes[0].set(title="Positive matched-WAIT reward", ylabel="Normalized reward")
    axes[1].plot(episodes, raw, color="#777777", alpha=0.25, linewidth=0.9)
    axes[1].plot(episodes, _rolling(raw), color="#222222", linewidth=2.4)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(title="Signed causal exposure-removal gain", ylabel="WAIT − action")
    axes[2].plot(episodes, _rolling(plain), label="Plain time removed", color="#2A9D8F", linewidth=2.2)
    axes[2].plot(episodes, _rolling(exposure), label="Hazard exposure removed", color="#E76F51", linewidth=2.2)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set(title="Reward decomposition", xlabel="Training episode", ylabel="Normalized reduction")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_policy(training: pd.DataFrame, output: Path) -> None:
    spans = _stage_spans(training)
    episodes = training["replication"]
    panels = (
        ("entropy", "Behavior-policy entropy", "Entropy", "#6A4C93"),
        ("action_temperature", "Exploration temperature", "Temperature", "#8E44AD"),
        ("actor_learning_rate", "Actor learning-rate schedule", "Learning rate", "#1F77B4"),
        ("approximate_kl", "PPO policy movement at updates", "Approximate KL", "#F39C12"),
        ("actor_gradient_norm", "Actor gradient magnitude", "Gradient norm", "#D35400"),
        (
            "nmcc_counterfactual_advantage_sd",
            "Within-state causal action distinction",
            "Advantage SD",
            "#2A9D8F",
        ),
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for axis, (field, title, ylabel, color) in zip(axes.reshape(-1), panels):
        _shade(axis, spans)
        if field in training:
            values = pd.to_numeric(training[field], errors="coerce")
            if field in {"approximate_kl", "actor_gradient_norm", "nmcc_counterfactual_advantage_sd"}:
                mask = pd.to_numeric(training.get("optimizer_updated", 0.0), errors="coerce") > 0
                axis.plot(episodes[mask], values[mask], marker="o", markersize=4, color=color)
            else:
                axis.plot(episodes, values, color=color, linewidth=1.8)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(alpha=0.22)
    axes[-1, 0].set_xlabel("Training episode")
    axes[-1, 1].set_xlabel("Training episode")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_staged_learning(training: pd.DataFrame, output: Path) -> None:
    """Show whether environment pretraining and action learning both worked."""
    spans = _stage_spans(training)
    episodes = training["replication"]
    updates = pd.to_numeric(
        training.get("optimizer_updated", 0.0), errors="coerce"
    ).fillna(0.0) > 0.0
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for axis in axes.reshape(-1):
        _shade(axis, spans)
        axis.grid(alpha=0.22)

    panels = (
        ("nmcc_natural_loss", "Environment-dynamics prediction loss", "Loss", "#4C78A8"),
        (
            "critic_representation_gradient_norm",
            "Environment gradient reaching GNN/LSTM",
            "Gradient norm",
            "#2A9D8F",
        ),
        ("nmcc_actor_enabled", "Action-learning phase enabled", "Indicator", "#6A4C93"),
        (
            "actor_representation_gradient_norm",
            "Action gradient reaching GNN/LSTM",
            "Gradient norm",
            "#D35400",
        ),
    )
    for axis, (field, title, ylabel, color) in zip(axes.reshape(-1)[:4], panels):
        if field in training:
            values = pd.to_numeric(training[field], errors="coerce")
            mask = updates & values.notna()
            axis.plot(
                episodes[mask], values[mask], marker="o", markersize=4,
                color=color, linewidth=1.5,
            )
        axis.set(title=title, ylabel=ylabel)

    target_kl = pd.to_numeric(
        training.get("nmcc_pi_target_kl_to_prior", np.nan), errors="coerce"
    )
    fit_kl = pd.to_numeric(
        training.get("nmcc_pi_fit_kl_after", np.nan), errors="coerce"
    )
    mask = updates & target_kl.notna()
    axes[2, 0].plot(episodes[mask], target_kl[mask], marker="o", label="NMCC target", color="#F39C12")
    mask = updates & fit_kl.notna()
    axes[2, 0].plot(episodes[mask], fit_kl[mask], marker="s", label="Actor after fit", color="#C0392B")
    axes[2, 0].set(title="Requested versus achieved policy movement", ylabel="KL divergence")
    axes[2, 0].legend()

    for field, label, color in (
        ("nmcc_pi_top1_before", "Before NMCC", "#777777"),
        ("nmcc_pi_top1_after", "After NMCC", "#174A7E"),
    ):
        if field in training:
            values = pd.to_numeric(training[field], errors="coerce")
            mask = updates & values.notna()
            axes[2, 1].plot(
                episodes[mask], values[mask], marker="o", label=label,
                color=color, linewidth=1.5,
            )
    axes[2, 1].set(title="Policy agreement with NMCC-best action", ylabel="Top-1 fraction")
    axes[2, 1].set_ylim(-0.03, 1.03)
    axes[2, 1].legend()
    axes[2, 0].set_xlabel("Training episode")
    axes[2, 1].set_xlabel("Training episode")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_outcomes(training: pd.DataFrame, output: Path) -> None:
    spans = _stage_spans(training)
    episodes = training["replication"]
    panels = (
        ("safe_completed", "Safe completion share", "Population share", "#2A9D8F"),
        ("unfinished", "Unfinished share at episode end", "Population share", "#F39C12"),
        ("casualty", "Casualty share (reported, not rewarded)", "Population share", "#C0392B"),
        ("mean_safe_completion_time", "Mean time to genuine safety", "Minutes", "#4C78A8"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for axis, (field, title, ylabel, color) in zip(axes.reshape(-1), panels):
        _shade(axis, spans)
        if field in training:
            values = pd.to_numeric(training[field], errors="coerce")
            if field in {"safe_completed", "unfinished", "casualty"}:
                population = pd.to_numeric(
                    training.get("initial_population", 1.0), errors="coerce"
                ).replace(0.0, np.nan)
                values = values / population
            axis.plot(episodes, values, color=color, alpha=0.22, linewidth=0.8)
            axis.plot(episodes, _rolling(values), color=color, linewidth=2.2)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(alpha=0.22)
    axes[-1, 0].set_xlabel("Training episode")
    axes[-1, 1].set_xlabel("Training episode")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _attach_denominator(decisions: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return decisions
    _, _, denominator = _normalized_components(training)
    lookup = dict(zip(training["replication"].astype(int), denominator))
    result = decisions.copy()
    result["denominator"] = result["episode"].map(lookup).astype(float)
    for source, target in (
        ("raw_plain_time_reduction", "normalized_plain_reduction"),
        ("raw_hazard_exposure_reduction", "normalized_exposure_reduction"),
        ("raw_remaining_risk_time_reduction", "normalized_raw_reduction"),
    ):
        if source in result:
            result[target] = pd.to_numeric(result[source], errors="coerce") / result["denominator"]
    return result


def _plot_action_quality(decisions: pd.DataFrame, output: Path) -> None:
    if decisions.empty:
        return
    grouped = decisions.groupby("decision_order", sort=True)
    orders = np.asarray(sorted(grouped.groups), dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    series = (
        ("reward", "Positive causal reward", "Normalized reward", "#174A7E"),
        ("normalized_raw_reduction", "Signed causal gain", "Normalized WAIT − action", "#555555"),
        ("normalized_plain_reduction", "Plain remaining time removed", "Normalized reduction", "#2A9D8F"),
        ("normalized_exposure_reduction", "Danger exposure removed", "Normalized reduction", "#E76F51"),
    )
    for axis, (field, title, ylabel, color) in zip(axes.reshape(-1), series):
        if field in decisions:
            means = grouped[field].mean().reindex(orders)
            standard_error = grouped[field].sem().reindex(orders).fillna(0.0)
            axis.errorbar(orders, means, yerr=1.96 * standard_error, marker="o", color=color, capsize=3)
            axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title=title, ylabel=ylabel)
        axis.grid(alpha=0.22)
    axes[-1, 0].set_xlabel("Installation number")
    axes[-1, 1].set_xlabel("Installation number")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _cell_matrix(rows: pd.DataFrame, value: str | None = None, side: int = 8) -> np.ndarray:
    matrix = np.full((side, side), np.nan if value else 0.0, dtype=float)
    if "selected_cell" not in rows or (value is not None and value not in rows):
        return matrix
    valid = rows[pd.to_numeric(rows.get("selected_cell", -1), errors="coerce").between(0, side * side - 1)].copy()
    if valid.empty:
        return matrix
    valid["selected_cell"] = valid["selected_cell"].astype(int)
    if value is None:
        grouped = valid.groupby("selected_cell").size()
    else:
        grouped = valid.groupby("selected_cell")[value].mean()
    for cell, score in grouped.items():
        i, j = divmod(int(cell), side)
        matrix[i, j] = float(score)
    return matrix


def _heatmap(axis, matrix: np.ndarray, title: str, colorbar_label: str, cmap: str) -> None:
    image = axis.imshow(matrix.T, origin="lower", cmap=cmap, aspect="equal")
    axis.set(title=title, xlabel="Cell row i", ylabel="Cell column j")
    axis.set_xticks(range(matrix.shape[0]))
    axis.set_yticks(range(matrix.shape[1]))
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=colorbar_label)


def _plot_training_locations(decisions: pd.DataFrame, output: Path) -> None:
    if decisions.empty:
        return
    episodes = sorted(decisions["episode"].unique())
    splits = np.array_split(np.asarray(episodes), 3)
    labels = ("Early training", "Middle training", "Late training")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for column, (label, selected) in enumerate(zip(labels, splits)):
        subset = decisions[decisions["episode"].isin(selected)]
        _heatmap(axes[0, column], _cell_matrix(subset), f"{label}: selection frequency", "Selections", "Blues")
        _heatmap(
            axes[1, column],
            _cell_matrix(subset, "normalized_raw_reduction"),
            f"{label}: mean signed causal gain",
            "Normalized WAIT − action",
            "RdYlGn",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_decision_sequence(decisions: pd.DataFrame, output: Path) -> None:
    if decisions.empty:
        return
    orders = sorted(int(value) for value in decisions["decision_order"].unique())
    columns = 4
    rows = int(np.ceil(len(orders) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.7 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, order in zip(axes, orders):
        subset = decisions[decisions["decision_order"] == order]
        _heatmap(axis, _cell_matrix(subset), f"Installation {order}", "Selections", "viridis")
    for axis in axes[len(orders):]:
        axis.axis("off")
    fig.suptitle("Where the learned policy installs shelters by decision order", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_evaluation(evaluation: pd.DataFrame, output: Path) -> None:
    if evaluation.empty:
        return
    strategy_field = "deployment_strategy"
    if strategy_field not in evaluation:
        return
    metrics = (
        ("episode_return", "Matched-WAIT reward", False),
        ("safe_completed", "Safe completion share", True),
        ("unfinished", "Unfinished share", True),
        ("casualty", "Casualty share (reported only)", True),
    )
    strategies = list(dict.fromkeys(evaluation[strategy_field].astype(str)))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (field, title, normalize_population) in zip(axes.reshape(-1), metrics):
        if field not in evaluation:
            axis.axis("off")
            continue
        groups = []
        for strategy in strategies:
            frame = evaluation[evaluation[strategy_field] == strategy]
            values = pd.to_numeric(frame[field], errors="coerce")
            if normalize_population:
                values = values / pd.to_numeric(
                    frame.get("initial_population", 1.0), errors="coerce"
                ).replace(0.0, np.nan)
            groups.append(values.dropna())
        means = [values.mean() if len(values) else np.nan for values in groups]
        errors = [values.sem() * 1.96 if len(values) > 1 else 0.0 for values in groups]
        axis.bar(strategies, means, yerr=errors, capsize=4, color="#4C78A8", alpha=0.85)
        axis.set(title=title)
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Held-out fixed-policy evaluation (mean ± 95% normal CI)")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_evaluation_by_population(evaluation: pd.DataFrame, output: Path) -> None:
    required = {"evaluation_population", "deployment_strategy", "initial_population"}
    if evaluation.empty or not required.issubset(evaluation.columns):
        return
    populations = sorted(
        pd.to_numeric(evaluation["evaluation_population"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    strategies = list(dict.fromkeys(evaluation["deployment_strategy"].astype(str)))
    metrics = (
        ("episode_return", "Matched-WAIT reward", False),
        ("safe_completed", "Safe completion share", True),
        ("unfinished", "Unfinished share", True),
        ("casualty", "Casualty share (reported only)", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (field, title, normalize_population) in zip(axes.reshape(-1), metrics):
        if field not in evaluation:
            axis.axis("off")
            continue
        for strategy in strategies:
            frame = evaluation[evaluation["deployment_strategy"].astype(str) == strategy].copy()
            frame["value"] = pd.to_numeric(frame[field], errors="coerce")
            if normalize_population:
                frame["value"] = frame["value"] / pd.to_numeric(
                    frame["initial_population"], errors="coerce"
                ).replace(0.0, np.nan)
            grouped = frame.groupby("evaluation_population")["value"]
            means = grouped.mean().reindex(populations)
            errors = grouped.sem().fillna(0.0).reindex(populations) * 1.96
            axis.errorbar(
                populations,
                means,
                yerr=errors,
                marker="o",
                linewidth=1.7,
                capsize=3,
                label=strategy,
            )
        axis.set(title=title)
        axis.grid(alpha=0.22)
    axes[-1, 0].set_xlabel("Pedestrians")
    axes[-1, 1].set_xlabel("Pedestrians")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Held-out outcomes by population (mean ± 95% normal CI)")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(20_000, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return float(values.mean()), float(lower), float(upper)


def _paired_evaluation(evaluation: pd.DataFrame) -> dict[str, dict]:
    required = {"replication", "deployment_strategy", "episode_return", "unfinished"}
    if evaluation.empty or not required.issubset(evaluation.columns):
        return {}
    condition_field = (
        "evaluation_variant_id"
        if "evaluation_variant_id" in evaluation
        else None
    )
    index_fields = (
        [condition_field, "replication", "deployment_strategy"]
        if condition_field
        else ["replication", "deployment_strategy"]
    )
    indexed = evaluation.set_index(index_fields)
    strategies = [
        strategy
        for strategy in dict.fromkeys(evaluation["deployment_strategy"].astype(str))
        if strategy != "rl"
    ]
    result = {}
    conditions = (
        list(dict.fromkeys(evaluation[condition_field].astype(str)))
        if condition_field
        else [""]
    )
    for condition_index, condition in enumerate(conditions):
      for strategy_index, strategy in enumerate(strategies):
        rows = []
        subset = (
            evaluation[evaluation[condition_field].astype(str) == condition]
            if condition_field
            else evaluation
        )
        for replication in sorted(subset["replication"].unique()):
            try:
                prefix = (condition, replication) if condition_field else (replication,)
                rl = indexed.loc[prefix + ("rl",)]
                comparator = indexed.loc[prefix + (strategy,)]
            except KeyError:
                continue
            population = max(1.0, float(rl.get("initial_population", 1.0)))
            rows.append(
                (
                    float(rl["episode_return"] - comparator["episode_return"]),
                    float(rl["unfinished"] - comparator["unfinished"]) / population,
                )
            )
        if not rows:
            continue
        differences = np.asarray(rows, dtype=float)
        seed_offset = condition_index * 100 + strategy_index
        reward_mean, reward_lower, reward_upper = _bootstrap_mean_ci(
            differences[:, 0], seed=34_000 + seed_offset
        )
        unfinished_mean, unfinished_lower, unfinished_upper = _bootstrap_mean_ci(
            differences[:, 1], seed=35_000 + seed_offset
        )
        result[f"{condition}:{strategy}" if condition else strategy] = {
            "evaluation_variant_id": condition,
            "benchmark": strategy,
            "paired_scenarios": int(len(differences)),
            "reward_difference_rl_minus_benchmark": reward_mean,
            "reward_difference_95pct_bootstrap_ci": [reward_lower, reward_upper],
            "reward_wins": int(np.sum(differences[:, 0] > 0.0)),
            "reward_ties": int(np.sum(np.isclose(differences[:, 0], 0.0))),
            "unfinished_share_difference_rl_minus_benchmark": unfinished_mean,
            "unfinished_difference_95pct_bootstrap_ci": [unfinished_lower, unfinished_upper],
            "completion_wins": int(np.sum(differences[:, 1] < 0.0)),
            "completion_ties": int(np.sum(np.isclose(differences[:, 1], 0.0))),
            "reward_differences": differences[:, 0].tolist(),
            "unfinished_differences": differences[:, 1].tolist(),
        }
    return result


def _plot_paired_evaluation(evaluation: pd.DataFrame, output: Path) -> None:
    paired = _paired_evaluation(evaluation)
    if not paired:
        return
    strategies = list(paired)
    positions = np.arange(len(strategies), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    panels = (
        (
            "reward_differences",
            "reward_difference_rl_minus_benchmark",
            "reward_difference_95pct_bootstrap_ci",
            "RL reward − benchmark reward",
            "RL better →",
        ),
        (
            "unfinished_differences",
            "unfinished_share_difference_rl_minus_benchmark",
            "unfinished_difference_95pct_bootstrap_ci",
            "RL unfinished share − benchmark unfinished share",
            "← RL better",
        ),
    )
    for axis, (raw_field, mean_field, ci_field, title, direction) in zip(axes, panels):
        for index, strategy in enumerate(strategies):
            values = np.asarray(paired[strategy][raw_field], dtype=float)
            jitter = np.linspace(-0.11, 0.11, len(values)) if len(values) > 1 else np.zeros(1)
            axis.scatter(values, index + jitter, color="#7F8C8D", alpha=0.65, s=22, zorder=2)
            mean = paired[strategy][mean_field]
            lower, upper = paired[strategy][ci_field]
            axis.errorbar(
                mean,
                index,
                xerr=np.asarray([[mean - lower], [upper - mean]]),
                marker="D",
                color="#174A7E",
                markersize=6,
                capsize=4,
                linewidth=2,
                zorder=3,
            )
        axis.axvline(0.0, color="black", linewidth=1.0)
        axis.set(title=title, xlabel=direction)
        axis.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(positions, strategies)
    axes[0].invert_yaxis()
    fig.suptitle("Paired held-out differences (points = scenarios; diamond = mean ± 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _f3_policy_shift(decisions: pd.DataFrame) -> dict:
    if decisions.empty or "stage" not in decisions:
        return {}
    stage_order = (
        decisions.groupby("stage", sort=False)["episode"].max().sort_values()
    )
    if stage_order.empty:
        return {}
    final_stage = str(stage_order.index[-1])
    stage = decisions[decisions["stage"].astype(str) == final_stage].copy()
    episodes = np.asarray(sorted(stage["episode"].unique()), dtype=int)
    window = min(16, len(episodes) // 2)
    if window < 2:
        return {}
    early = stage[stage["episode"].isin(episodes[:window])]
    late = stage[stage["episode"].isin(episodes[-window:])]
    early_counts = np.nan_to_num(_cell_matrix(early), nan=0.0)
    late_counts = np.nan_to_num(_cell_matrix(late), nan=0.0)
    early_share = early_counts / max(float(early_counts.sum()), 1.0)
    late_share = late_counts / max(float(late_counts.sum()), 1.0)
    return {
        "window_episodes": int(window),
        "stage_id": final_stage,
        "early_episode_range": [int(episodes[0]), int(episodes[window - 1])],
        "late_episode_range": [int(episodes[-window]), int(episodes[-1])],
        "selection_total_variation_distance": float(0.5 * np.abs(late_share - early_share).sum()),
        "early": early,
        "late": late,
        "early_share": early_share,
        "late_share": late_share,
    }


def _plot_f3_policy_shift(decisions: pd.DataFrame, output: Path) -> None:
    shift = _f3_policy_shift(decisions)
    if not shift:
        return
    early = shift["early"]
    late = shift["late"]
    early_share = shift["early_share"]
    late_share = shift["late_share"]
    maximum = max(float(early_share.max()), float(late_share.max()))
    difference = late_share - early_share
    difference_limit = max(float(np.abs(difference).max()), 1.0e-6)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, matrix, title in (
        (axes[0, 0], early_share, f"Early {shift['stage_id']} episodes {shift['early_episode_range'][0]}–{shift['early_episode_range'][1]}"),
        (axes[0, 1], late_share, f"Late {shift['stage_id']} episodes {shift['late_episode_range'][0]}–{shift['late_episode_range'][1]}"),
    ):
        image = axis.imshow(matrix.T, origin="lower", cmap="Blues", aspect="equal", vmin=0.0, vmax=maximum)
        axis.set(title=title, xlabel="Cell row i", ylabel="Cell column j")
        axis.set_xticks(range(8))
        axis.set_yticks(range(8))
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Selection share")
    image = axes[1, 0].imshow(
        difference.T,
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[1, 0].set(
        title=f"Late − early selection share (TV distance {shift['selection_total_variation_distance']:.2f})",
        xlabel="Cell row i",
        ylabel="Cell column j",
    )
    axes[1, 0].set_xticks(range(8))
    axes[1, 0].set_yticks(range(8))
    plt.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Share change")
    for subset, label, color in ((early, "Early F3", "#7F8C8D"), (late, "Late F3", "#174A7E")):
        grouped = subset.groupby("decision_order")["normalized_raw_reduction"].mean()
        axes[1, 1].plot(grouped.index, grouped.values, marker="o", label=label, color=color)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        title="Causal action gain within the same stage",
        xlabel="Installation number",
        ylabel="Normalized WAIT − action",
    )
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend()
    fig.suptitle(
        f"Learned placement behavior shift within final curriculum stage {shift['stage_id']}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def _plot_evaluation_locations(decisions: pd.DataFrame, output: Path) -> None:
    if decisions.empty:
        return
    strategies = list(dict.fromkeys(decisions["strategy"].astype(str)))
    columns = min(3, len(strategies))
    rows = int(np.ceil(len(strategies) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.7 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis, strategy in zip(axes, strategies):
        subset = decisions[decisions["strategy"] == strategy]
        _heatmap(axis, _cell_matrix(subset), strategy, "Selections", "magma")
    for axis in axes[len(strategies):]:
        axis.axis("off")
    fig.suptitle("Held-out shelter installation locations", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.polyfit(np.arange(values.size, dtype=float), values, 1)[0])


def _summary(training: pd.DataFrame, decisions: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    stages = {}
    if "training_stage_id" in training:
        for stage, frame in training.groupby("training_stage_id", sort=False):
            values = frame["episode_return"].to_numpy(dtype=float)
            stages[str(stage)] = {
                "episodes": int(len(frame)),
                "mean_reward": float(np.mean(values)),
                "reward_slope_per_episode": _slope(values),
                "first_rollout_mean": float(np.mean(values[: min(ROLLING_WINDOW, len(values))])),
                "last_rollout_mean": float(np.mean(values[-min(ROLLING_WINDOW, len(values)) :])),
            }
    update_rows = training[pd.to_numeric(training.get("optimizer_updated", 0.0), errors="coerce") > 0]
    top_cells = []
    if not decisions.empty:
        for cell, frame in decisions.groupby("selected_cell"):
            if int(cell) < 0:
                continue
            top_cells.append(
                {
                    "cell": int(cell),
                    "selections": int(len(frame)),
                    "mean_raw_gain": float(frame.get("normalized_raw_reduction", pd.Series([0.0])).mean()),
                    "mean_plain_reduction": float(frame.get("normalized_plain_reduction", pd.Series([0.0])).mean()),
                    "mean_exposure_reduction": float(frame.get("normalized_exposure_reduction", pd.Series([0.0])).mean()),
                }
            )
        top_cells.sort(key=lambda row: (-row["selections"], -row["mean_raw_gain"]))
    evaluation_summary = {}
    if not evaluation.empty and "deployment_strategy" in evaluation:
        for strategy, frame in evaluation.groupby("deployment_strategy", sort=False):
            evaluation_summary[str(strategy)] = {
                field: float(pd.to_numeric(frame[field], errors="coerce").mean())
                for field in ("episode_return", "safe_completed", "unfinished", "casualty")
                if field in frame
            }
    evaluation_by_population = {}
    if not evaluation.empty and {
        "deployment_strategy",
        "evaluation_population",
    }.issubset(evaluation.columns):
        for (population, strategy), frame in evaluation.groupby(
            ["evaluation_population", "deployment_strategy"], sort=True
        ):
            denominator = pd.to_numeric(
                frame.get("initial_population", 1.0), errors="coerce"
            ).replace(0.0, np.nan)
            evaluation_by_population.setdefault(str(int(population)), {})[
                str(strategy)
            ] = {
                "mean_reward": float(
                    pd.to_numeric(frame["episode_return"], errors="coerce").mean()
                ),
                "mean_safe_completion_share": float(
                    (
                        pd.to_numeric(frame["safe_completed"], errors="coerce")
                        / denominator
                    ).mean()
                ),
                "mean_unfinished_share": float(
                    (
                        pd.to_numeric(frame["unfinished"], errors="coerce")
                        / denominator
                    ).mean()
                ),
                "mean_casualty_share": float(
                    (
                        pd.to_numeric(frame["casualty"], errors="coerce")
                        / denominator
                    ).mean()
                ),
            }
    shift = _f3_policy_shift(decisions)
    shift_summary = {
        key: value
        for key, value in shift.items()
        if key not in {"early", "late", "early_share", "late_share"}
    }
    return {
        "training_episodes": int(len(training)),
        "stages": stages,
        "optimizer_updates": int(len(update_rows)),
        "mean_update_kl": float(update_rows["approximate_kl"].mean()) if len(update_rows) and "approximate_kl" in update_rows else 0.0,
        "final_entropy": float(training["entropy"].iloc[-1]) if "entropy" in training else 0.0,
        "decision_rows": int(len(decisions)),
        "top_selected_cells": top_cells[:12],
        "evaluation": evaluation_summary,
        "evaluation_by_population": evaluation_by_population,
        "paired_evaluation": _paired_evaluation(evaluation),
        "f3_policy_shift": shift_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("launch_dir")
    args = parser.parse_args()
    root = Path(args.launch_dir).expanduser().resolve()
    training_path = root / "training_episode_summary.csv"
    if not training_path.exists():
        raise FileNotFoundError(training_path)
    output = root / "counterfactual_exposure_analysis"
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(training_path).sort_values("replication").reset_index(drop=True)
    _numeric(
        training,
        (
            "replication",
            "episode_return",
            "raw_plain_time_reduction",
            "raw_hazard_exposure_reduction",
            "initial_population",
            "stage_population",
            "horizon_minutes",
            "entropy",
            "action_temperature",
            "actor_learning_rate",
            "approximate_kl",
            "actor_gradient_norm",
            "nmcc_counterfactual_advantage_sd",
            "nmcc_natural_loss",
            "critic_representation_gradient_norm",
            "nmcc_actor_enabled",
            "actor_representation_gradient_norm",
            "nmcc_pi_target_kl_to_prior",
            "nmcc_pi_fit_kl_after",
            "nmcc_pi_top1_before",
            "nmcc_pi_top1_after",
            "optimizer_updated",
        ),
    )
    decisions = _attach_denominator(_decision_rows(root, "training"), training)
    evaluation_path = root / "evaluation_postfix_episode_summary.csv"
    if not evaluation_path.exists():
        evaluation_path = root / "evaluation_episode_summary.csv"
    evaluation = pd.read_csv(evaluation_path) if evaluation_path.exists() else pd.DataFrame()
    evaluation_mode = "evaluation_postfix" if (root / "evaluation_postfix").exists() else "evaluation"
    evaluation_decisions = _decision_rows(root, evaluation_mode)

    figures = []
    plotters = (
        ("01_reward_and_components.png", lambda path: _plot_reward(training, path)),
        ("02_policy_learning_and_convergence.png", lambda path: _plot_policy(training, path)),
        ("03_evacuation_outcomes.png", lambda path: _plot_outcomes(training, path)),
        ("04_action_quality_by_installation.png", lambda path: _plot_action_quality(decisions, path)),
        ("05_training_location_heatmaps.png", lambda path: _plot_training_locations(decisions, path)),
        ("06_location_by_installation_order.png", lambda path: _plot_decision_sequence(decisions, path)),
        ("12_staged_learning_diagnostics.png", lambda path: _plot_staged_learning(training, path)),
    )
    for name, plotter in plotters:
        path = output / name
        plotter(path)
        if path.exists():
            figures.append(str(path))
    if not evaluation.empty:
        path = output / "07_held_out_policy_comparison.png"
        _plot_evaluation(evaluation, path)
        if path.exists():
            figures.append(str(path))
    if not evaluation_decisions.empty:
        path = output / "08_held_out_location_heatmaps.png"
        _plot_evaluation_locations(evaluation_decisions, path)
        if path.exists():
            figures.append(str(path))
    if not evaluation.empty:
        path = output / "09_paired_held_out_differences.png"
        _plot_paired_evaluation(evaluation, path)
        if path.exists():
            figures.append(str(path))
        path = output / "11_held_out_outcomes_by_population.png"
        _plot_evaluation_by_population(evaluation, path)
        if path.exists():
            figures.append(str(path))
    if not decisions.empty:
        path = output / "10_f3_policy_shift.png"
        _plot_f3_policy_shift(decisions, path)
        if path.exists():
            figures.append(str(path))

    payload = _summary(training, decisions, evaluation)
    payload["figures"] = figures
    summary_path = output / "analysis_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(summary_path), "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
