#!/usr/bin/env python3
"""Generate the prespecified E0--E6 publication figures from experiment tables.

Unlike ``generate_paper_figures.py`` (the sealed one-seed smoke-study
reporter), this script is designed for the full experiment.  It never trains,
reruns, or mutates a simulation.  It validates result-table schemas, accounts
for independent policy seeds and matched scenario replications, renders every
figure whose source data are available, and writes an explicit readiness
report for figures that cannot yet be supported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

# Figure generation is a batch artifact pipeline.  Default to a non-GUI
# backend before pyplot is imported so direct function calls, unit tests,
# nohup campaigns, and headless workers never initialize macOS AppKit.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from ExperimentSuite import (
    DEFAULT_EXPERIMENT_SUITE_PATH,
    ExperimentSuite,
    OutputTableSpec,
    load_experiment_suite,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LAUNCH_DIR = (
    PROJECT_ROOT / "runs" / "multicity_five_city_lr1e3_learning_audit_arm_20260906"
)
STRATEGY_ORDER = (
    "rl",
    "risk_reduction",
    "heuristic",
    "hazard_weighted",
    "accessibility_deficit",
    "random",
    "static_greedy",
    "rl_precommit",
    "initial_only",
)
STRATEGY_LABEL = {
    "rl": "RL regional priority",
    "risk_reduction": "Future risk-time reduction",
    "heuristic": "Active-population heuristic",
    "hazard_weighted": "Hazard-weighted demand",
    "accessibility_deficit": "Accessibility deficit",
    "random": "Random feasible region",
    "static_greedy": "Static demand-greedy",
    "rl_precommit": "RL precommitment",
    "initial_only": "Static predeployment",
    "pooled_all_cities": "Pooled all-city RL",
    "leave_one_city_out": "Leave-one-city-out RL",
}
STRATEGY_COLOR = {
    "rl": "#0072B2",
    "risk_reduction": "#F0E442",
    "heuristic": "#E69F00",
    "hazard_weighted": "#D55E00",
    "accessibility_deficit": "#CC79A7",
    "random": "#999999",
    "static_greedy": "#009E73",
    "rl_precommit": "#56B4E9",
    "initial_only": "#009E73",
    "pooled_all_cities": "#56B4E9",
    "leave_one_city_out": "#0072B2",
}
CITY_LABEL = {
    "state_college_pa": "State College",
    "reading_pa": "Reading",
    "spokane_wa": "Spokane",
    "seattle_wa": "Seattle",
    "chicago_il": "Chicago",
}
METRIC_DIRECTION = {
    "episode_return": 1.0,
    "objective_episode_return": 1.0,
    "safe_fraction": 1.0,
    "casualty_fraction": -1.0,
    "unfinished_fraction": -1.0,
    "restricted_mean_time_to_safety": -1.0,
    "normalized_risk_weighted_person_time": -1.0,
    "simulation_runtime_s": -1.0,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f(row: Mapping, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key!r} value in result row")
    return value


def _i(row: Mapping, key: str, default: int | None = None) -> int:
    value = row.get(key, "")
    if (value is None or str(value).strip() == "") and default is not None:
        return int(default)
    return int(float(value))


def _city(row: Mapping) -> str:
    value = row.get("city_id") or row.get("held_out_city")
    if value is None or not str(value).strip():
        raise KeyError("Result row requires city_id or held_out_city")
    return str(value)


def _city_order(rows: Sequence[Mapping]) -> list[str]:
    ranks: dict[str, int] = {}
    for row in rows:
        city = _city(row)
        rank = _i(row, "city_scale_rank", default=len(ranks) + 1)
        ranks.setdefault(city, rank)
    return sorted(ranks, key=lambda city: (ranks[city], city))


def _policy_id(row: Mapping) -> int:
    return _i(row, "policy_replication", default=0)


def _strategy(row: Mapping, field: str = "deployment_strategy") -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise KeyError(f"Result row requires {field}")
    return value


def _metric(row: Mapping, metric: str) -> float:
    if metric == "episode_return":
        objective = str(row.get("objective_episode_return", "")).strip()
        return float(objective) if objective else _f(row, "episode_return")
    if metric in {"safe_fraction", "casualty_fraction", "unfinished_fraction"}:
        population = max(1.0, _f(row, "initial_population"))
        numerator = {
            "safe_fraction": "safe_completed",
            "casualty_fraction": "casualty",
            "unfinished_fraction": "unfinished",
        }[metric]
        return _f(row, numerator) / population
    return _f(row, metric)


def _scenario_key(row: Mapping) -> tuple[str, ...]:
    """Matched scenario identity, including any active experimental factors."""
    values = [_city(row)]
    for key in (
        "capacity_regime",
        "hazard_regime",
        "demand_pattern",
        "perturbation",
        "level",
        "city_scenario_replication",
        "scenario_seed",
    ):
        if key in row and str(row.get(key, "")).strip():
            values.append(f"{key}={row[key]}")
    if len(values) == 1:
        raise KeyError("A result row lacks a scenario identity")
    return tuple(values)


def validate_result_table(path: Path, spec: OutputTableSpec) -> dict:
    """Return a machine-readable schema audit for one result table."""
    status = {
        "table_id": spec.table_id,
        "experiment": spec.experiment,
        "path": str(path),
        "required_for_complete_suite": spec.required_for_complete_suite,
        "exists": path.exists(),
        "valid": False,
        "row_count": 0,
        "missing_columns": [],
        "sha256": None,
    }
    if not path.exists():
        return status
    rows = _read_csv(path)
    fields = set(rows[0]) if rows else set()
    missing = sorted(set(spec.required_columns).difference(fields))
    status.update(
        {
            "valid": bool(rows) and not missing,
            "row_count": len(rows),
            "missing_columns": missing,
            "sha256": _sha256(path),
        }
    )
    return status


def audit_suite_inputs(
    suite: ExperimentSuite, launch_dir: Path
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict]]:
    """Validate and load every available table declared by the suite."""
    tables: dict[str, list[dict[str, str]]] = {}
    statuses: dict[str, dict] = {}
    for table_id, spec in suite.output_tables.items():
        path = suite.table_path(launch_dir, table_id)
        status = validate_result_table(path, spec)
        statuses[table_id] = status
        if status["valid"]:
            tables[table_id] = _read_csv(path)
    return tables, statuses


def balanced_training_blocks(training: Sequence[Mapping]) -> list[dict]:
    """Aggregate each policy seed into complete equal-city training blocks."""
    if not training:
        raise ValueError("Training table is empty")
    cities = _city_order(training)
    expected = set(cities)
    output = []
    policy_ids = sorted({_policy_id(row) for row in training})
    if not policy_ids or policy_ids[0] <= 0:
        raise ValueError("Training rows require positive policy_replication values")
    metrics = (
        "episode_return",
        "safe_completion_reward",
        "casualty_penalty",
        "risk_time_penalty",
        "entropy",
        "heuristic_agreement_rate",
    )
    for policy_id in policy_ids:
        rows = sorted(
            (row for row in training if _policy_id(row) == policy_id),
            key=lambda row: _i(row, "replication"),
        )
        if len(rows) % len(cities):
            raise ValueError(
                f"Policy {policy_id} ends before a complete equal-city training block"
            )
        for offset in range(0, len(rows), len(cities)):
            block_rows = rows[offset : offset + len(cities)]
            observed = {_city(row) for row in block_rows}
            if observed != expected:
                raise ValueError(
                    f"Policy {policy_id} block {offset // len(cities) + 1} is "
                    f"city-imbalanced: {sorted(observed)}"
                )
            record = {
                "policy_replication": policy_id,
                "block": offset // len(cities) + 1,
            }
            for metric in metrics:
                record[metric] = float(np.mean([_metric(row, metric) for row in block_rows]))
            output.append(record)
    block_counts = {
        policy_id: sum(row["policy_replication"] == policy_id for row in output)
        for policy_id in policy_ids
    }
    if len(set(block_counts.values())) != 1:
        raise ValueError(f"Policy seeds have unequal training lengths: {block_counts}")
    return output


def _matrix_by_city(
    rows: Sequence[Mapping],
    strategy: str,
    metric: str,
    *,
    strategy_field: str = "deployment_strategy",
) -> dict[str, np.ndarray]:
    """Construct complete policy-seed by scenario matrices within each city."""
    selected = [row for row in rows if _strategy(row, strategy_field) == strategy]
    if not selected:
        raise ValueError(f"No rows for {strategy_field}={strategy!r}")
    output: dict[str, np.ndarray] = {}
    for city in _city_order(selected):
        city_rows = [row for row in selected if _city(row) == city]
        policy_ids = sorted({_policy_id(row) for row in city_rows})
        keys = sorted({_scenario_key(row) for row in city_rows})
        by_identity: dict[tuple[int, tuple[str, ...]], Mapping] = {}
        for row in city_rows:
            identity = (_policy_id(row), _scenario_key(row))
            if identity in by_identity:
                raise ValueError(f"Duplicate result row for {strategy!r}: {identity}")
            by_identity[identity] = row
        missing = [
            (policy_id, key)
            for policy_id in policy_ids
            for key in keys
            if (policy_id, key) not in by_identity
        ]
        if missing:
            raise ValueError(
                f"Incomplete policy-by-scenario matrix for {strategy!r}/{city}: {missing[:3]}"
            )
        output[city] = np.asarray(
            [
                [_metric(by_identity[(policy_id, key)], metric) for key in keys]
                for policy_id in policy_ids
            ],
            dtype=float,
        )
    return output


def paired_difference_matrices(
    rows: Sequence[Mapping],
    strategy: str,
    reference: str,
    metric: str,
    *,
    strategy_field: str = "deployment_strategy",
) -> dict[str, np.ndarray]:
    """Return positive-is-better paired differences by city/policy/scenario."""
    left_rows = [row for row in rows if _strategy(row, strategy_field) == strategy]
    right_rows = [row for row in rows if _strategy(row, strategy_field) == reference]
    if not left_rows or not right_rows:
        raise ValueError(f"Cannot pair {strategy!r} and {reference!r}")
    direction = METRIC_DIRECTION.get(metric, 1.0)
    result: dict[str, np.ndarray] = {}
    for city in _city_order(left_rows):
        left_city = [row for row in left_rows if _city(row) == city]
        right_city = [row for row in right_rows if _city(row) == city]
        right_by_key: dict[tuple[str, ...], float] = {}
        for row in right_city:
            key = _scenario_key(row)
            value = _metric(row, metric)
            if key in right_by_key and not math.isclose(
                right_by_key[key], value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Non-identical duplicate reference row for {reference!r}/{key}")
            right_by_key[key] = value
        policy_ids = sorted({_policy_id(row) for row in left_city})
        left_by_identity: dict[tuple[int, tuple[str, ...]], float] = {}
        for row in left_city:
            identity = (_policy_id(row), _scenario_key(row))
            if identity in left_by_identity:
                raise ValueError(f"Duplicate comparison row for {strategy!r}/{identity}")
            left_by_identity[identity] = _metric(row, metric)
        keys = sorted(right_by_key)
        if not keys:
            raise ValueError(f"No reference scenarios for {city}")
        for policy_id in policy_ids:
            observed = {key for candidate_policy, key in left_by_identity if candidate_policy == policy_id}
            if observed != set(keys):
                raise ValueError(
                    f"Unmatched scenarios for {strategy!r}/{reference!r}/{city}/policy {policy_id}"
                )
        result[city] = direction * np.asarray(
            [
                [
                    left_by_identity[(policy_id, key)] - right_by_key[key]
                    for key in keys
                ]
                for policy_id in policy_ids
            ],
            dtype=float,
        )
    if set(result) != {_city(row) for row in right_rows}:
        raise ValueError(f"{strategy!r} and {reference!r} do not cover the same cities")
    return result


def hierarchical_bootstrap(
    matrices: Mapping[str, np.ndarray],
    *,
    seed: int,
    draws: int,
    confidence_level: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap global policy seeds and scenarios nested within fixed cities.

    Policy indices are resampled once per draw and shared across cities because
    one trained seed is evaluated in every city.  Scenarios are independently
    resampled within each fixed city.  Cities are then macro-averaged and are
    never resampled.
    """
    if not matrices:
        raise ValueError("No matrices supplied to hierarchical bootstrap")
    arrays = {city: np.asarray(values, dtype=float) for city, values in matrices.items()}
    policy_counts = {values.shape[0] for values in arrays.values()}
    if len(policy_counts) != 1 or any(values.ndim != 2 or values.shape[1] == 0 for values in arrays.values()):
        raise ValueError("Every city requires the same non-empty policy-seed dimension")
    policy_count = next(iter(policy_counts))
    estimate = float(np.mean([values.mean() for values in arrays.values()]))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=float)
    for draw in range(int(draws)):
        policy_indices = rng.integers(0, policy_count, size=policy_count)
        city_means = []
        for values in arrays.values():
            scenario_count = values.shape[1]
            scenario_indices = rng.integers(0, scenario_count, size=scenario_count)
            city_means.append(values[np.ix_(policy_indices, scenario_indices)].mean())
        samples[draw] = float(np.mean(city_means))
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return estimate, float(lower), float(upper)


def _pointwise_policy_summary(
    blocks: Sequence[Mapping], metric: str, *, seed: int, draws: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    policy_ids = sorted({int(row["policy_replication"]) for row in blocks})
    block_ids = sorted({int(row["block"]) for row in blocks})
    lookup = {
        (int(row["policy_replication"]), int(row["block"])): float(row[metric])
        for row in blocks
    }
    matrix = np.asarray(
        [[lookup[(policy, block)] for block in block_ids] for policy in policy_ids],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    if len(policy_ids) == 1:
        return np.asarray(block_ids), mean, mean.copy(), mean.copy()
    rng = np.random.default_rng(int(seed))
    samples = np.empty((int(draws), len(block_ids)), dtype=float)
    for draw in range(int(draws)):
        indices = rng.integers(0, len(policy_ids), size=len(policy_ids))
        samples[draw] = matrix[indices].mean(axis=0)
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    return np.asarray(block_ids), mean, lower, upper


def _style() -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=320 if suffix == "png" else None, facecolor="white")
        paths.append(path)
    return paths


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray(
        [values[max(0, index - window + 1) : index + 1].mean() for index in range(len(values))]
    )


def _summary_row(
    figure_id: str,
    comparison: str,
    metric: str,
    estimate: float,
    lower: float,
    upper: float,
    **fields,
) -> dict:
    return {
        "figure_id": figure_id,
        "comparison": comparison,
        "metric": metric,
        "estimate": float(estimate),
        "bootstrap_95_ci_lower": float(lower),
        "bootstrap_95_ci_upper": float(upper),
        **fields,
    }


def _plot_training(
    training: Sequence[Mapping],
    checkpoint: Sequence[Mapping] | None,
    output_dir: Path,
    *,
    seed: int,
    draws: int,
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    blocks = balanced_training_blocks(training)
    policy_ids = sorted({int(row["policy_replication"]) for row in blocks})
    block_ids = sorted({int(row["block"]) for row in blocks})
    lookup = {
        (int(row["policy_replication"]), int(row["block"])): row for row in blocks
    }
    x, mean, lower, upper = _pointwise_policy_summary(
        blocks, "episode_return", seed=seed, draws=draws
    )
    window = min(8, len(x))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for policy_id in policy_ids:
        values = np.asarray(
            [float(lookup[(policy_id, block)]["episode_return"]) for block in block_ids]
        )
        axes[0].plot(
            x,
            values,
            color="#7F7F7F",
            alpha=0.20,
            linewidth=0.7,
            marker=".",
            markersize=2.5,
        )
    if len(policy_ids) > 1:
        axes[0].fill_between(
            x,
            lower,
            upper,
            color="#56B4E9",
            alpha=0.28,
            label="95% policy-seed band",
        )
    axes[0].plot(
        x,
        _moving_average(mean, window),
        color="#0072B2",
        linewidth=2.4,
        label=f"{window}-block moving average",
    )
    axes[0].set(
        title="Balanced training return",
        xlabel="Equal-city training block",
        ylabel="Undiscounted episode return",
    )
    axes[0].legend(frameon=False)

    summary: list[dict] = []
    if checkpoint:
        checkpoints = sorted({_i(row, "training_episode_per_city") for row in checkpoint})
        estimates, lows, highs = [], [], []
        for index, episode in enumerate(checkpoints):
            subset = [
                row for row in checkpoint if _i(row, "training_episode_per_city") == episode
            ]
            matrices = paired_difference_matrices(
                subset, "rl", "heuristic", "episode_return"
            )
            estimate, lo, hi = hierarchical_bootstrap(
                matrices, seed=seed + 50 + index, draws=draws
            )
            estimates.append(estimate)
            lows.append(lo)
            highs.append(hi)
            summary.append(
                _summary_row(
                    "F01",
                    "rl_minus_heuristic_at_checkpoint",
                    "episode_return",
                    estimate,
                    lo,
                    hi,
                    training_episode_per_city=episode,
                )
            )
        estimates_array = np.asarray(estimates)
        axes[1].fill_between(checkpoints, lows, highs, color="#009E73", alpha=0.25)
        axes[1].plot(checkpoints, estimates_array, color="#009E73", linewidth=2.3, marker="o")
        axes[1].axhline(0.0, color="#333333", linewidth=1.0)
        axes[1].set(
            title="Fixed development-set performance",
            xlabel="Training episodes per city",
            ylabel="RL − heuristic return",
        )
    else:
        components = (
            ("safe_completion_reward", "Safe completion", "#0072B2"),
            ("casualty_penalty", "Casualty", "#D55E00"),
            ("risk_time_penalty", "Risk-weighted person-time", "#009E73"),
            ("episode_return", "Total", "#111111"),
        )
        for index, (metric, label, color) in enumerate(components):
            component_x, component_mean, _, _ = _pointwise_policy_summary(
                blocks, metric, seed=seed + 10 + index, draws=draws
            )
            axes[1].plot(
                component_x,
                _moving_average(component_mean, window),
                color=color,
                linewidth=2.0,
                label=label,
            )
        axes[1].axhline(0.0, color="#777777", linewidth=0.8)
        axes[1].set(
            title="Reward decomposition",
            xlabel="Equal-city training block",
            ylabel="Normalized contribution",
        )
        axes[1].legend(frameon=False, ncol=2)
        axes[1].text(
            0.02,
            0.02,
            "Checkpoint evaluations not yet available",
            transform=axes[1].transAxes,
            fontsize=8,
            color="#666666",
        )
    fig.suptitle(
        f"Multi-city PPO learning ({len(policy_ids)} independent policy seed"
        f"{'s' if len(policy_ids) != 1 else ''})"
    )
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "01_training_reward_and_convergence")
    plt.close(fig)
    return paths, summary


def _load_ppo_diagnostics(launch_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((launch_dir / "policies").glob("policy_*/ppo_diagnostics.csv")):
        try:
            policy_id = int(path.parent.name.split("_")[-1])
        except ValueError as error:
            raise ValueError(f"Invalid policy diagnostics directory: {path.parent}") from error
        for row in _read_csv(path):
            copied = dict(row)
            copied["policy_replication"] = policy_id
            rows.append(copied)
    return rows


def _diagnostic_seed_series(
    diagnostics: Sequence[Mapping], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    policy_ids = sorted({_policy_id(row) for row in diagnostics})
    by_policy = {}
    for policy_id in policy_ids:
        updates = [
            row
            for row in diagnostics
            if _policy_id(row) == policy_id and _f(row, "optimizer_updated") > 0.5
        ]
        updates.sort(key=lambda row: _i(row, "optimizer_updates", default=_i(row, "episode")))
        if updates:
            by_policy[policy_id] = updates
    if not by_policy:
        raise ValueError("No PPO optimizer updates were recorded")
    common_length = min(len(rows) for rows in by_policy.values())
    matrix = np.asarray(
        [[_f(row, metric) for row in by_policy[policy_id][:common_length]] for policy_id in sorted(by_policy)]
    )
    mean = matrix.mean(axis=0)
    if matrix.shape[0] == 1:
        lower = upper = mean.copy()
    else:
        lower, upper = np.quantile(matrix, [0.025, 0.975], axis=0)
    return np.arange(1, common_length + 1), mean, lower, upper


def training_update_averages(
    training: Sequence[Mapping], diagnostics: Sequence[Mapping]
) -> list[dict]:
    """Align optimizer diagnostics with means over their contributing episodes.

    An update is triggered only after a complete multi-episode rollout.  The
    optimization fields on that update row are already means over PPO epochs
    and minibatches.  This function adds means over the episodes in the same
    rollout, preventing the final episode of a rollout from being presented as
    though it were the update's average training outcome.
    """
    if not training or not diagnostics:
        raise ValueError("Training rows and PPO diagnostics are both required")
    output: list[dict] = []
    policy_ids = sorted({_policy_id(row) for row in diagnostics})
    for policy_id in policy_ids:
        training_rows = sorted(
            (row for row in training if _policy_id(row) == policy_id),
            key=lambda row: _i(row, "replication"),
        )
        by_episode = {_i(row, "replication"): row for row in training_rows}
        diagnostic_rows = [
            row for row in diagnostics if _policy_id(row) == policy_id
        ]
        if not training_rows or not diagnostic_rows:
            raise ValueError(
                f"Policy {policy_id} lacks training rows or PPO diagnostics"
            )
        # A continuation launch deliberately keeps the checkpoint's global
        # episode counter while its launch-local training table restarts at
        # replication 1.  Infer that constant, auditable offset from the first
        # row of each source instead of assuming every checkpoint began at 0.
        episode_offset = max(_i(row, "episode") for row in diagnostic_rows) - max(
            by_episode
        )
        updates = sorted(
            (
                row
                for row in diagnostic_rows
                if _f(row, "optimizer_updated") > 0.5
            ),
            key=lambda row: _i(row, "episode"),
        )
        previous_end = min(by_episode) - 1
        for update_index, update in enumerate(updates, start=1):
            source_episode_end = _i(update, "episode")
            episode_end = source_episode_end - episode_offset
            contributing = [
                by_episode[episode]
                for episode in range(previous_end + 1, episode_end + 1)
                if episode in by_episode
            ]
            expected = episode_end - previous_end
            if len(contributing) != expected or not contributing:
                raise ValueError(
                    f"Policy {policy_id} PPO update {update_index} lacks its complete "
                    "contributing episode rollout"
                )
            decisions = sum(_f(row, "decisions") for row in contributing)
            weighted_agreement = sum(
                _f(row, "heuristic_agreement_rate") * _f(row, "decisions")
                for row in contributing
            )
            record = {
                "policy_replication": policy_id,
                "rollout_update": update_index,
                "episode_start": previous_end + 1,
                "episode_end": episode_end,
                "source_episode_end": source_episode_end,
                "episode_index_offset": episode_offset,
                "contributing_episodes": len(contributing),
                "optimizer_steps_cumulative": _i(update, "optimizer_updates"),
                "epochs_completed": _f(update, "epochs_completed"),
                "approximate_kl": _f(update, "approximate_kl"),
                "clip_fraction": _f(update, "clip_fraction"),
                "explained_variance": _f(update, "explained_variance"),
                "policy_loss": _f(update, "policy_loss"),
                "value_loss": _f(update, "value_loss"),
                "gradient_norm": _f(update, "gradient_norm"),
                "update_entropy": _f(update, "update_entropy"),
                "mean_episode_return": float(
                    np.mean([_metric(row, "episode_return") for row in contributing])
                ),
                "mean_safe_completion_reward": float(
                    np.mean([_f(row, "safe_completion_reward") for row in contributing])
                ),
                "mean_casualty_penalty": float(
                    np.mean([_f(row, "casualty_penalty") for row in contributing])
                ),
                "mean_risk_time_penalty": float(
                    np.mean([_f(row, "risk_time_penalty") for row in contributing])
                ),
                "mean_behavior_entropy": float(
                    np.mean([_f(row, "entropy") for row in contributing])
                ),
                "decision_weighted_heuristic_agreement": (
                    weighted_agreement / decisions if decisions > 0.0 else 0.0
                ),
            }
            output.append(record)
            previous_end = episode_end
    if not output:
        raise ValueError("No completed PPO rollout updates were found")
    return output


def _update_seed_series(
    rows: Sequence[Mapping], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pointwise policy-seed means and empirical 95% seed intervals."""
    policy_ids = sorted({_policy_id(row) for row in rows})
    by_policy = {
        policy_id: sorted(
            (row for row in rows if _policy_id(row) == policy_id),
            key=lambda row: _i(row, "rollout_update"),
        )
        for policy_id in policy_ids
    }
    common_length = min(len(values) for values in by_policy.values())
    if common_length <= 0:
        raise ValueError("Every policy seed requires at least one PPO update")
    matrix = np.asarray(
        [
            [_f(row, metric) for row in by_policy[policy_id][:common_length]]
            for policy_id in policy_ids
        ],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    if matrix.shape[0] == 1:
        lower = upper = mean.copy()
    else:
        lower, upper = np.quantile(matrix, [0.025, 0.975], axis=0)
    return np.arange(1, common_length + 1), mean, lower, upper


def _plot_ppo_diagnostics(
    training: Sequence[Mapping], diagnostics: Sequence[Mapping], output_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not diagnostics:
        raise ValueError("PPO diagnostics are unavailable")
    updates = training_update_averages(training, diagnostics)
    policy_count = len({_policy_id(row) for row in updates})
    update_count = min(
        sum(_policy_id(row) == policy for row in updates)
        for policy in {_policy_id(row) for row in updates}
    )
    fig, axes = plt.subplots(4, 1, figsize=(11.8, 13.8), sharex=True)

    def series(axis, metric, label, color, *, linestyle="-", marker="o"):
        x, mean, lower, upper = _update_seed_series(updates, metric)
        if policy_count > 1:
            axis.fill_between(x, lower, upper, color=color, alpha=0.15)
        axis.plot(
            x,
            mean,
            color=color,
            label=label,
            linewidth=1.8,
            marker=marker,
            markersize=3.4,
            linestyle=linestyle,
        )
        return x, mean

    trust = axes[0]
    series(trust, "approximate_kl", "Mean approximate KL", "#0072B2")
    trust.axhline(
        0.03,
        color="#56B4E9",
        linestyle="--",
        linewidth=1.2,
        label="Target KL = 0.03",
    )
    trust_right = trust.twinx()
    series(trust_right, "clip_fraction", "Mean clip fraction", "#E69F00")
    trust.set(title="Trust-region behavior", ylabel="KL divergence")
    trust_right.set_ylabel("Clip fraction")
    lines = trust.get_lines() + trust_right.get_lines()
    trust.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")

    exploration = axes[1]
    series(
        exploration,
        "mean_behavior_entropy",
        "Rollout-mean behavior entropy",
        "#CC79A7",
    )
    series(
        exploration,
        "update_entropy",
        "Epoch/minibatch-mean update entropy",
        "#9467BD",
    )
    exploration_right = exploration.twinx()
    series(
        exploration_right,
        "decision_weighted_heuristic_agreement",
        "Decision-weighted heuristic agreement",
        "#009E73",
    )
    exploration.set(title="Exploration and policy concentration", ylabel="Entropy (nats)")
    exploration_right.set_ylabel("Agreement fraction")
    exploration_right.set_ylim(-0.02, 1.02)
    lines = exploration.get_lines() + exploration_right.get_lines()
    exploration.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")

    critic = axes[2]
    series(critic, "explained_variance", "Explained variance", "#7F7F7F")
    critic.axhline(0.0, color="#777777", linewidth=0.8)
    critic.set_ylim(-1.05, 1.05)
    critic_right = critic.twinx()
    series(critic_right, "value_loss", "Mean value loss", "#BCBD22")
    series(critic_right, "policy_loss", "Mean policy loss", "#17BECF")
    critic_right.set_yscale("symlog", linthresh=1e-4)
    critic.set(title="Critic and optimization fit", ylabel="Explained variance")
    critic_right.set_ylabel("Epoch/minibatch-mean loss")
    lines = [
        line
        for line in critic.get_lines() + critic_right.get_lines()
        if not line.get_label().startswith("_")
    ]
    critic.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")

    outcome = axes[3]
    for metric, label, color in (
        ("mean_episode_return", "Total return", "#111111"),
        ("mean_safe_completion_reward", "Safe completion", "#0072B2"),
        ("mean_casualty_penalty", "Casualty", "#D55E00"),
        ("mean_risk_time_penalty", "Risk-weighted person-time", "#009E73"),
    ):
        series(outcome, metric, label, color)
    outcome.axhline(0.0, color="#777777", linewidth=0.8)
    outcome.set(
        title="Rollout-averaged training outcomes",
        xlabel="PPO rollout update (optimization statistics average its completed training epochs)",
        ylabel="Mean normalized reward contribution",
    )
    outcome.legend(frameon=False, ncol=2, loc="best")

    fig.suptitle(
        f"Training PPO update diagnostics ({update_count} rollout updates; "
        f"{policy_count} policy seed{'s' if policy_count != 1 else ''})"
    )
    fig.text(
        0.5,
        0.967,
        "Optimization statistics average all epoch/minibatch steps; outcome statistics average all episodes in the corresponding rollout.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    paths = _save_figure(fig, output_dir, "02_ppo_optimization_diagnostics")
    plt.close(fig)
    table_path = output_dir / "02_ppo_update_averages.csv"
    _write_csv(table_path, updates)
    return [*paths, table_path]


def _available_strategies(rows: Sequence[Mapping]) -> list[str]:
    observed = {_strategy(row) for row in rows}
    return [strategy for strategy in STRATEGY_ORDER if strategy in observed] + sorted(
        observed.difference(STRATEGY_ORDER)
    )


def _absolute_interval(
    rows: Sequence[Mapping],
    strategy: str,
    metric: str,
    *,
    seed: int,
    draws: int,
) -> tuple[float, float, float]:
    return hierarchical_bootstrap(
        _matrix_by_city(rows, strategy, metric), seed=seed, draws=draws
    )


def _plot_primary_performance(
    rows: Sequence[Mapping], output_dir: Path, *, seed: int, draws: int
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    strategies = _available_strategies(rows)
    metrics = (
        ("episode_return", "Policy-objective return", 1.0, "Return"),
        ("safe_fraction", "Safe completion", 100.0, "Population (%)"),
        ("casualty_fraction", "Casualty", 100.0, "Population (%)"),
        (
            "restricted_mean_time_to_safety",
            "Restricted mean time to safety",
            1.0,
            "Timesteps",
        ),
    )
    y = np.arange(len(strategies))
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.7))
    summary = []
    for metric_index, ((metric, title, scale, xlabel), axis) in enumerate(
        zip(metrics, axes.flat)
    ):
        for strategy_index, strategy in enumerate(strategies):
            estimate, lower, upper = _absolute_interval(
                rows,
                strategy,
                metric,
                seed=seed + metric_index * 100 + strategy_index,
                draws=draws,
            )
            value = scale * estimate
            axis.errorbar(
                value,
                strategy_index,
                xerr=np.asarray(
                    [[scale * (estimate - lower)], [scale * (upper - estimate)]]
                ),
                fmt="o",
                color=STRATEGY_COLOR.get(strategy, "#444444"),
                capsize=3,
                markersize=7,
                linewidth=1.5,
            )
            summary.append(
                _summary_row(
                    "F03",
                    "absolute_policy_performance",
                    metric,
                    estimate,
                    lower,
                    upper,
                    strategy=strategy,
                )
            )
        axis.set_yticks(y, [STRATEGY_LABEL.get(value, value) for value in strategies])
        axis.invert_yaxis()
        axis.set(title=title, xlabel=xlabel)
    fig.suptitle("Matched held-out evacuation performance across five cities")
    fig.text(
        0.5,
        0.005,
        "Equal-city macro means; intervals resample policy seeds and matched scenarios within fixed cities.",
        ha="center",
        fontsize=8.2,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    paths = _save_figure(fig, output_dir, "03_primary_policy_performance")
    plt.close(fig)
    return paths, summary


def _plot_city_forest(
    rows: Sequence[Mapping], output_dir: Path, *, seed: int, draws: int
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    metrics = (
        ("episode_return", "Return improvement", 1.0),
        ("safe_fraction", "Safe-completion improvement (pp)", 100.0),
        ("casualty_fraction", "Casualty reduction (pp)", 100.0),
    )
    cities = _city_order(rows)
    labels = [CITY_LABEL.get(city, city) for city in cities] + ["Equal-city macro"]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    summary = []
    for metric_index, ((metric, title, scale), axis) in enumerate(zip(metrics, axes)):
        matrices = paired_difference_matrices(rows, "rl", "heuristic", metric)
        estimates, lows, highs = [], [], []
        for city_index, city in enumerate(cities):
            result = hierarchical_bootstrap(
                {city: matrices[city]},
                seed=seed + 100 * metric_index + city_index,
                draws=draws,
            )
            estimates.append(result[0])
            lows.append(result[1])
            highs.append(result[2])
            summary.append(
                _summary_row(
                    "F04",
                    "rl_minus_heuristic",
                    metric,
                    *result,
                    city_id=city,
                    scope="city",
                )
            )
        macro = hierarchical_bootstrap(
            matrices, seed=seed + 100 * metric_index + 90, draws=draws
        )
        estimates.append(macro[0])
        lows.append(macro[1])
        highs.append(macro[2])
        summary.append(
            _summary_row(
                "F04",
                "rl_minus_heuristic",
                metric,
                *macro,
                city_id="all",
                scope="equal_city_macro",
            )
        )
        values = scale * np.asarray(estimates)
        errors = np.vstack(
            (
                values - scale * np.asarray(lows),
                scale * np.asarray(highs) - values,
            )
        )
        axis.axvline(0.0, color="#333333", linewidth=1.0)
        axis.errorbar(
            values[:-1],
            y[:-1],
            xerr=errors[:, :-1],
            fmt="o",
            color="#56B4E9",
            capsize=3,
            linewidth=1.4,
        )
        axis.errorbar(
            values[-1],
            y[-1],
            xerr=errors[:, -1:],
            fmt="D",
            color="#0072B2",
            capsize=3.5,
            linewidth=1.8,
            markersize=7,
        )
        axis.set(title=title, xlabel="Positive favors RL")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
    fig.suptitle("RL regional priority versus active-population heuristic")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save_figure(fig, output_dir, "04_city_specific_paired_effects")
    plt.close(fig)
    return paths, summary


def _plot_regime_heatmap(
    rows: Sequence[Mapping],
    suite: ExperimentSuite,
    output_dir: Path,
    *,
    seed: int,
    draws: int,
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    capacities = list(suite.factors["capacity_regime"])
    hazards = list(suite.factors["hazard_regime"])
    panels = (
        ("episode_return", "RL return improvement", 1.0),
        ("casualty_fraction", "RL casualty reduction (pp)", 100.0),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    summary = []
    for panel_index, ((metric, title, scale), axis) in enumerate(zip(panels, axes)):
        values = np.full((len(hazards), len(capacities)), np.nan, dtype=float)
        for hazard_index, hazard in enumerate(hazards):
            for capacity_index, capacity in enumerate(capacities):
                subset = [
                    row
                    for row in rows
                    if row["hazard_regime"] == hazard
                    and row["capacity_regime"] == capacity
                ]
                matrices = paired_difference_matrices(
                    subset, "rl", "heuristic", metric
                )
                result = hierarchical_bootstrap(
                    matrices,
                    seed=seed + panel_index * 100 + hazard_index * 10 + capacity_index,
                    draws=draws,
                )
                values[hazard_index, capacity_index] = scale * result[0]
                summary.append(
                    _summary_row(
                        "F05",
                        "rl_minus_heuristic",
                        metric,
                        *result,
                        hazard_regime=hazard,
                        capacity_regime=capacity,
                    )
                )
        bound = max(1e-9, float(np.nanmax(np.abs(values))))
        image = axis.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
        for hazard_index in range(len(hazards)):
            for capacity_index in range(len(capacities)):
                axis.text(
                    capacity_index,
                    hazard_index,
                    f"{values[hazard_index, capacity_index]:+.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(values[hazard_index, capacity_index]) > 0.55 * bound else "#111111",
                    fontweight="bold",
                )
        axis.set_xticks(np.arange(len(capacities)), [value.title() for value in capacities])
        axis.set_yticks(np.arange(len(hazards)), [value.title() for value in hazards])
        axis.set(title=title, xlabel="Initial shelter-capacity regime", ylabel="Hazard regime")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("Where learned regional prioritization improves on the benchmark")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save_figure(fig, output_dir, "05_capacity_hazard_performance_heatmap")
    plt.close(fig)
    return paths, summary


def _plot_safety_frontier(
    rows: Sequence[Mapping], output_dir: Path, *, seed: int, draws: int
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    strategies = _available_strategies(rows)
    fig, axis = plt.subplots(figsize=(7.8, 5.8))
    summary = []
    for index, strategy in enumerate(strategies):
        casualty = _absolute_interval(
            rows, strategy, "casualty_fraction", seed=seed + 10 * index, draws=draws
        )
        time = _absolute_interval(
            rows,
            strategy,
            "restricted_mean_time_to_safety",
            seed=seed + 10 * index + 1,
            draws=draws,
        )
        safe = _absolute_interval(
            rows, strategy, "safe_fraction", seed=seed + 10 * index + 2, draws=draws
        )
        axis.errorbar(
            100.0 * casualty[0],
            time[0],
            xerr=np.asarray(
                [[100.0 * (casualty[0] - casualty[1])], [100.0 * (casualty[2] - casualty[0])]]
            ),
            yerr=np.asarray([[time[0] - time[1]], [time[2] - time[0]]]),
            fmt="none",
            color=STRATEGY_COLOR.get(strategy, "#444444"),
            alpha=0.65,
            capsize=3,
        )
        axis.scatter(
            100.0 * casualty[0],
            time[0],
            s=45.0 + 2.2 * 100.0 * safe[0],
            color=STRATEGY_COLOR.get(strategy, "#444444"),
            edgecolor="white",
            linewidth=0.8,
            label=f"{STRATEGY_LABEL.get(strategy, strategy)} ({100.0 * safe[0]:.0f}% safe)",
            zorder=3,
        )
        for metric, result in (
            ("casualty_fraction", casualty),
            ("restricted_mean_time_to_safety", time),
            ("safe_fraction", safe),
        ):
            summary.append(
                _summary_row(
                    "F06",
                    "absolute_policy_performance",
                    metric,
                    *result,
                    strategy=strategy,
                )
            )
    axis.annotate(
        "Preferred direction",
        xy=(0.04, 0.05),
        xytext=(0.25, 0.22),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        color="#333333",
    )
    axis.set(
        title="Safety–casualty–timeliness frontier",
        xlabel="Casualties (% of initial population; lower is better)",
        ylabel="Restricted mean time to safety (lower is better)",
    )
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    paths = _save_figure(fig, output_dir, "06_safety_casualty_timeliness_frontier")
    plt.close(fig)
    return paths, summary


def _mean_interval(values: Sequence[float], seed: int, draws: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        raise ValueError("Cannot estimate an interval from no values")
    estimate = float(array.mean())
    if len(array) == 1:
        return estimate, estimate, estimate
    rng = np.random.default_rng(int(seed))
    samples = np.asarray(
        [rng.choice(array, size=len(array), replace=True).mean() for _ in range(int(draws))]
    )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def _plot_scalability(
    rows: Sequence[Mapping], output_dir: Path, *, seed: int, draws: int
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    methods = sorted({_strategy(row, "method") for row in rows})
    metrics = (
        ("action_count", "Action-space size", "Available actions", True),
        ("decision_latency_ms", "Decision latency", "Milliseconds", True),
        ("episode_return", "Decision quality", "Episode return", False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    summary = []
    method_colors = {
        "hierarchical_region": "#0072B2",
        "flat_candidate": "#D55E00",
    }
    for panel_index, ((metric, title, ylabel, log_y), axis) in enumerate(
        zip(metrics, axes)
    ):
        for method_index, method in enumerate(methods):
            method_rows = [row for row in rows if _strategy(row, "method") == method]
            candidate_counts = sorted({_i(row, "candidate_count") for row in method_rows})
            estimates, lows, highs = [], [], []
            for count_index, count in enumerate(candidate_counts):
                values = [
                    _f(row, metric)
                    for row in method_rows
                    if _i(row, "candidate_count") == count
                ]
                result = _mean_interval(
                    values,
                    seed + panel_index * 100 + method_index * 20 + count_index,
                    draws,
                )
                estimates.append(result[0])
                lows.append(result[1])
                highs.append(result[2])
                summary.append(
                    _summary_row(
                        "F07",
                        "scalability_absolute",
                        metric,
                        *result,
                        method=method,
                        candidate_count=count,
                    )
                )
            estimates_array = np.asarray(estimates)
            axis.plot(
                candidate_counts,
                estimates_array,
                marker="o",
                linewidth=2.0,
                color=method_colors.get(method, "#444444"),
                label=method.replace("_", " ").title(),
            )
            axis.fill_between(
                candidate_counts,
                lows,
                highs,
                color=method_colors.get(method, "#444444"),
                alpha=0.18,
            )
        axis.set_xscale("log")
        if log_y and all(_f(row, metric) > 0.0 for row in rows):
            axis.set_yscale("log")
        axis.set(title=title, xlabel="Shelter candidates", ylabel=ylabel)
    axes[0].legend(frameon=False)
    fig.suptitle("Why shared contextual exact-site scoring is computationally scalable")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save_figure(fig, output_dir, "07_hierarchical_action_scalability")
    plt.close(fig)
    return paths, summary


def paired_same_policy_matrices(
    rows: Sequence[Mapping],
    strategy: str,
    reference: str,
    metric: str,
    *,
    strategy_field: str,
) -> dict[str, np.ndarray]:
    """Pair two learned variants by both policy seed and scenario."""
    left = [row for row in rows if _strategy(row, strategy_field) == strategy]
    right = [row for row in rows if _strategy(row, strategy_field) == reference]
    direction = METRIC_DIRECTION.get(metric, 1.0)
    if not left or not right:
        raise ValueError(f"Cannot pair {strategy!r} and {reference!r}")
    result = {}
    for city in _city_order(left):
        left_city = [row for row in left if _city(row) == city]
        right_city = [row for row in right if _city(row) == city]
        left_lookup = {
            (_policy_id(row), _scenario_key(row)): _metric(row, metric)
            for row in left_city
        }
        right_lookup = {
            (_policy_id(row), _scenario_key(row)): _metric(row, metric)
            for row in right_city
        }
        if len(left_lookup) != len(left_city) or len(right_lookup) != len(right_city):
            raise ValueError(f"Duplicate variant rows for {city}")
        if set(left_lookup) != set(right_lookup):
            raise ValueError(f"Unmatched policy-seed/scenario variants for {city}")
        policies = sorted({identity[0] for identity in left_lookup})
        keys = sorted({identity[1] for identity in left_lookup})
        result[city] = direction * np.asarray(
            [
                [
                    left_lookup[(policy, key)] - right_lookup[(policy, key)]
                    for key in keys
                ]
                for policy in policies
            ]
        )
    return result


def _plot_ablation(
    rows: Sequence[Mapping],
    suite: ExperimentSuite,
    output_dir: Path,
    *,
    seed: int,
    draws: int,
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    configured = list(suite.experiments["E4"]["design"]["ablations"])
    observed = {_strategy(row, "ablation") for row in rows}
    variants = [value for value in configured if value in observed and value != "full_model"]
    if "full_model" not in observed or not variants:
        raise ValueError("Ablation table requires full_model and at least one ablation")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, max(4.3, 0.55 * len(variants) + 2.0)))
    summary = []
    for panel_index, (metric, title, scale) in enumerate(
        (
            ("episode_return", "Return retained by the full model", 1.0),
            ("casualty_fraction", "Casualty reduction from the full model (pp)", 100.0),
        )
    ):
        results = []
        for variant_index, variant in enumerate(variants):
            matrices = paired_same_policy_matrices(
                rows,
                "full_model",
                variant,
                metric,
                strategy_field="ablation",
            )
            result = hierarchical_bootstrap(
                matrices,
                seed=seed + panel_index * 100 + variant_index,
                draws=draws,
            )
            results.append(result)
            summary.append(
                _summary_row(
                    "F08",
                    "full_model_minus_ablation",
                    metric,
                    *result,
                    ablation=variant,
                )
            )
        estimates = scale * np.asarray([result[0] for result in results])
        lower = scale * np.asarray([result[1] for result in results])
        upper = scale * np.asarray([result[2] for result in results])
        y = np.arange(len(variants))
        axes[panel_index].axvline(0.0, color="#333333", linewidth=1.0)
        axes[panel_index].errorbar(
            estimates,
            y,
            xerr=np.vstack((estimates - lower, upper - estimates)),
            fmt="D",
            color="#0072B2",
            capsize=3.5,
            linewidth=1.7,
        )
        axes[panel_index].set_yticks(
            y, [variant.replace("_", " ").title() for variant in variants]
        )
        axes[panel_index].invert_yaxis()
        axes[panel_index].set(title=title, xlabel="Positive favors the full model")
    fig.suptitle("Reward and hierarchical policy ablations")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save_figure(fig, output_dir, "08_reward_and_architecture_ablations")
    plt.close(fig)
    return paths, summary


def _plot_transfer(
    rows: Sequence[Mapping], output_dir: Path, *, seed: int, draws: int
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    observed = {_strategy(row) for row in rows}
    comparators = [
        value
        for value in ("leave_one_city_out", "pooled_all_cities")
        if value in observed
    ]
    if "heuristic" not in observed or not comparators:
        raise ValueError("Transfer table requires heuristic and a learned transfer policy")
    cities = _city_order(rows)
    offsets = np.linspace(-0.12, 0.12, len(comparators))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
    summary = []
    for panel_index, (metric, title, scale) in enumerate(
        (
            ("episode_return", "Held-out-city return improvement", 1.0),
            ("casualty_fraction", "Held-out-city casualty reduction (pp)", 100.0),
        )
    ):
        axes[panel_index].axvline(0.0, color="#333333", linewidth=1.0)
        for comparator_index, comparator in enumerate(comparators):
            matrices = paired_difference_matrices(
                rows, comparator, "heuristic", metric
            )
            city_results = []
            for city_index, city in enumerate(cities):
                result = hierarchical_bootstrap(
                    {city: matrices[city]},
                    seed=seed + panel_index * 200 + comparator_index * 50 + city_index,
                    draws=draws,
                )
                city_results.append(result)
                summary.append(
                    _summary_row(
                        "F09",
                        f"{comparator}_minus_heuristic",
                        metric,
                        *result,
                        held_out_city=city,
                    )
                )
            estimates = scale * np.asarray([result[0] for result in city_results])
            lower = scale * np.asarray([result[1] for result in city_results])
            upper = scale * np.asarray([result[2] for result in city_results])
            axes[panel_index].errorbar(
                estimates,
                np.arange(len(cities)) + offsets[comparator_index],
                xerr=np.vstack((estimates - lower, upper - estimates)),
                fmt="o",
                color=STRATEGY_COLOR.get(comparator, "#444444"),
                capsize=3,
                label=STRATEGY_LABEL.get(comparator, comparator),
            )
        axes[panel_index].set_yticks(
            np.arange(len(cities)), [CITY_LABEL.get(city, city) for city in cities]
        )
        axes[panel_index].invert_yaxis()
        axes[panel_index].set(title=title, xlabel="Positive favors learned policy")
    axes[0].legend(frameon=False)
    fig.suptitle("Transfer to cities excluded from training")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save_figure(fig, output_dir, "09_leave_one_city_out_transfer")
    plt.close(fig)
    return paths, summary


def _plot_robustness(
    rows: Sequence[Mapping], output_dir: Path, *, seed: int, draws: int
) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    conditions = sorted(
        {(str(row["perturbation"]), str(row["level"])) for row in rows}
    )
    if not conditions:
        raise ValueError("Robustness table has no perturbation conditions")
    labels = [
        f"{perturbation.replace('_', ' ').title()} — {level}"
        for perturbation, level in conditions
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.3, max(5.0, 0.42 * len(labels) + 2.0)), sharey=True)
    summary = []
    for panel_index, (metric, title, scale) in enumerate(
        (
            ("episode_return", "Return improvement", 1.0),
            ("casualty_fraction", "Casualty reduction (pp)", 100.0),
        )
    ):
        results = []
        for condition_index, (perturbation, level) in enumerate(conditions):
            subset = [
                row
                for row in rows
                if str(row["perturbation"]) == perturbation and str(row["level"]) == level
            ]
            matrices = paired_difference_matrices(
                subset, "rl", "heuristic", metric
            )
            result = hierarchical_bootstrap(
                matrices,
                seed=seed + panel_index * 100 + condition_index,
                draws=draws,
            )
            results.append(result)
            summary.append(
                _summary_row(
                    "F10",
                    "rl_minus_heuristic",
                    metric,
                    *result,
                    perturbation=perturbation,
                    level=level,
                )
            )
        estimates = scale * np.asarray([result[0] for result in results])
        lower = scale * np.asarray([result[1] for result in results])
        upper = scale * np.asarray([result[2] for result in results])
        axes[panel_index].axvline(0.0, color="#333333", linewidth=1.0)
        axes[panel_index].errorbar(
            estimates,
            np.arange(len(conditions)),
            xerr=np.vstack((estimates - lower, upper - estimates)),
            fmt="D",
            color="#0072B2",
            capsize=3,
            linewidth=1.6,
        )
        axes[panel_index].set_yticks(np.arange(len(conditions)), labels)
        axes[panel_index].invert_yaxis()
        axes[panel_index].set(title=title, xlabel="Positive favors RL")
    fig.suptitle("Policy robustness under out-of-design operating perturbations")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = _save_figure(fig, output_dir, "10_operating_robustness")
    plt.close(fig)
    return paths, summary


def _plot_population_candidate_stress(
    rows: Sequence[Mapping],
    suite: ExperimentSuite,
    output_dir: Path,
    *,
    seed: int,
    draws: int,
) -> tuple[list[Path], list[dict]]:
    """Render the prespecified 5x5 population/candidate scaling experiment."""
    import matplotlib.pyplot as plt

    populations = list(suite.population_levels)
    candidates = list(suite.shelter_candidate_levels)
    horizons = {_i(row, "horizon_timesteps") for row in rows}
    if horizons != {suite.scale_horizon_timesteps}:
        raise ValueError(
            "Scale-stress rows must all use the prespecified "
            f"{suite.scale_horizon_timesteps}-transition horizon"
        )
    panels = (
        (
            suite.primary_metric,
            "RL policy-objective return improvement",
            1.0,
            True,
        ),
        ("safe_fraction", "RL safe-completion improvement (pp)", 100.0, True),
        ("casualty_fraction", "RL casualty reduction (pp)", 100.0, True),
        ("simulation_runtime_s", "Mean RL episode runtime (s)", 1.0, False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2))
    summary = []
    for panel_index, ((metric, title, scale, paired), axis) in enumerate(
        zip(panels, axes.flat)
    ):
        values = np.full((len(populations), len(candidates)), np.nan, dtype=float)
        for population_index, population in enumerate(populations):
            for candidate_index, candidate in enumerate(candidates):
                subset = [
                    row
                    for row in rows
                    if _i(row, "population_level") == population
                    and _i(row, "shelter_candidate_level") == candidate
                ]
                if not subset:
                    continue
                if paired:
                    matrices = paired_difference_matrices(
                        subset, "rl", "heuristic", metric
                    )
                    comparison = "rl_minus_heuristic"
                else:
                    matrices = _matrix_by_city(
                        subset, "rl", "simulation_runtime_s"
                    )
                    comparison = "rl_absolute"
                result = hierarchical_bootstrap(
                    matrices,
                    seed=seed + panel_index * 100 + population_index * 10 + candidate_index,
                    draws=draws,
                )
                values[population_index, candidate_index] = scale * result[0]
                summary.append(
                    _summary_row(
                        "F13",
                        comparison,
                        metric,
                        *result,
                        population_level=population,
                        shelter_candidate_level=candidate,
                        horizon_timesteps=suite.scale_horizon_timesteps,
                    )
                )
        if np.isnan(values).all():
            raise ValueError("Scale-stress table contains none of the configured factor cells")
        if paired:
            bound = max(1e-9, float(np.nanmax(np.abs(values))))
            image = axis.imshow(
                values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto"
            )
        else:
            image = axis.imshow(values, cmap="viridis", aspect="auto")
            bound = max(1e-9, float(np.nanmax(np.abs(values))))
        for population_index in range(len(populations)):
            for candidate_index in range(len(candidates)):
                value = values[population_index, candidate_index]
                if np.isnan(value):
                    label = "—"
                    color = "#666666"
                else:
                    label = f"{value:+.2f}" if paired else f"{value:.1f}"
                    color = (
                        "white"
                        if abs(value) > 0.55 * bound or not paired
                        else "#111111"
                    )
                axis.text(
                    candidate_index,
                    population_index,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8.5,
                    fontweight="bold" if not np.isnan(value) else "normal",
                )
        axis.set_xticks(np.arange(len(candidates)), [str(value) for value in candidates])
        axis.set_yticks(
            np.arange(len(populations)), [f"{value:,}" for value in populations]
        )
        axis.set(
            title=title,
            xlabel="Sampled shelter candidates",
            ylabel="Initial pedestrians",
        )
        fig.colorbar(image, ax=axis, shrink=0.82)
    observed_cells = {
        (_i(row, "population_level"), _i(row, "shelter_candidate_level"))
        for row in rows
    }
    expected_cell_count = len(populations) * len(candidates)
    evidence_label = (
        ""
        if len(observed_cells) == expected_cell_count
        else f"; pilot evidence, {len(observed_cells)}/{expected_cell_count} cells"
    )
    fig.suptitle(
        "Population and shelter-candidate scale stress "
        f"({suite.scale_horizon_timesteps} transitions{evidence_label})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = _save_figure(fig, output_dir, "13_population_candidate_scale_stress")
    plt.close(fig)
    return paths, summary


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _copy_or_compose_maps(
    rows: Sequence[Mapping], source_index: Path, output_dir: Path
) -> tuple[list[Path], dict[str, list[str]]]:
    """Create a self-contained set of final-placement and progress map panels."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    if not all(_truthy(row.get("non_interventional", False)) for row in rows):
        raise ValueError("Map figures require non-interventional visualization evidence")
    cities = _city_order(rows)
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    by_family: dict[str, list[str]] = {"F11": [], "F12": []}
    existing_dir = source_index.parent / "maps"
    for city in cities:
        city_rows = [row for row in rows if _city(row) == city]
        strategy_rows = {
            _strategy(row, "strategy"): row for row in city_rows
        }
        strategies = [value for value in STRATEGY_ORDER if value in strategy_rows]
        if not strategies:
            strategies = sorted(strategy_rows)
        products = (
            ("shelter_installation_map", "shelter_installations", "F11", "Final shelter configuration"),
            ("evacuation_progress_map", "evacuation_progress", "F12", "Matched evacuation progress"),
        )
        for field, stem, family, title in products:
            destination = map_dir / f"{family.lower()}_{stem}_{city}.png"
            existing = existing_dir / f"{stem}_{city}.png"
            if existing.exists():
                shutil.copy2(existing, destination)
            else:
                source_paths = [Path(str(strategy_rows[value][field])) for value in strategies]
                missing = [str(path) for path in source_paths if not path.exists()]
                if missing:
                    raise FileNotFoundError(
                        f"Missing map sources for {family}/{city}: {missing[:3]}"
                    )
                if field == "shelter_installation_map":
                    fig, axes = plt.subplots(1, len(strategies), figsize=(4.0 * len(strategies), 4.3))
                else:
                    fig, axes = plt.subplots(len(strategies), 1, figsize=(16.0, 3.0 * len(strategies)))
                axes_array = np.atleast_1d(axes).flat
                for axis, strategy, path in zip(axes_array, strategies, source_paths):
                    axis.imshow(mpimg.imread(path))
                    axis.set_title(STRATEGY_LABEL.get(strategy, strategy))
                    axis.axis("off")
                fig.suptitle(f"{title} — {CITY_LABEL.get(city, city)}")
                fig.text(
                    0.995,
                    0.005,
                    "Road network © OpenStreetMap contributors (ODbL)",
                    ha="right",
                    fontsize=7,
                    color="#555555",
                )
                fig.tight_layout(rect=(0, 0.02, 1, 0.95))
                fig.savefig(destination, dpi=260, facecolor="white")
                plt.close(fig)
            generated.append(destination)
            by_family[family].append(str(destination))
    return generated, by_family


def _design_checks(
    suite: ExperimentSuite, tables: Mapping[str, Sequence[Mapping]]
) -> list[dict]:
    """Check sample sizes and factorial completeness separately from CSV schemas."""
    checks = []

    training = tables.get("training_summary")
    if training:
        cities = _city_order(training)
        policy_ids = sorted({_policy_id(row) for row in training})
        counts = {
            f"policy_{policy:03d}/{city}": sum(
                _policy_id(row) == policy and _city(row) == city for row in training
            )
            for policy in policy_ids
            for city in cities
        }
        checks.append(
            {
                "check": "confirmatory_training_sample",
                "passed": len(policy_ids) == suite.policy_seeds
                and set(counts.values()) == {suite.train_episodes_per_city},
                "expected_policy_seeds": suite.policy_seeds,
                "observed_policy_seeds": len(policy_ids),
                "expected_episodes_per_policy_city": suite.train_episodes_per_city,
                "observed_counts": counts,
            }
        )

    evaluation = tables.get("evaluation_summary")
    if evaluation:
        rl_policies = {
            _policy_id(row)
            for row in evaluation
            if _strategy(row) == "rl"
        }
        parity_passed = False
        try:
            paired_difference_matrices(
                evaluation, "rl", "heuristic", "episode_return"
            )
            parity_passed = True
        except (KeyError, ValueError):
            parity_passed = False
        checks.append(
            {
                "check": "primary_evaluation_policy_matrix",
                "passed": len(rl_policies) == suite.policy_seeds and parity_passed,
                "expected_policy_seeds": suite.policy_seeds,
                "observed_policy_seeds": len(rl_policies),
                "scenario_parity": parity_passed,
            }
        )

    regime = tables.get("regime_evaluation")
    if regime:
        observed_cells = {
            (
                str(row["capacity_regime"]),
                str(row["hazard_regime"]),
                str(row["demand_pattern"]),
            )
            for row in regime
        }
        expected_cells = {
            (
                cell["capacity_regime"],
                cell["hazard_regime"],
                cell["demand_pattern"],
            )
            for cell in suite.factor_cells
        }
        replicate_counts = []
        for city in _city_order(regime):
            for capacity, hazard, demand in expected_cells:
                reference = [
                    row
                    for row in regime
                    if _city(row) == city
                    and _strategy(row) == "heuristic"
                    and row["capacity_regime"] == capacity
                    and row["hazard_regime"] == hazard
                    and row["demand_pattern"] == demand
                ]
                replicate_counts.append(len({_scenario_key(row) for row in reference}))
        checks.append(
            {
                "check": "evaluation_factorial",
                "passed": observed_cells == expected_cells
                and set(replicate_counts) == {suite.replications_per_factor_cell},
                "expected_factor_cells": len(expected_cells),
                "observed_factor_cells": len(observed_cells),
                "expected_replications_per_city_cell": suite.replications_per_factor_cell,
                "observed_replication_counts": sorted(set(replicate_counts)),
            }
        )

    maps = tables.get("map_figure_index")
    if maps:
        checks.append(
            {
                "check": "map_visualization_non_interventional",
                "passed": all(_truthy(row.get("non_interventional", False)) for row in maps),
                "rows": len(maps),
            }
        )
    scale = tables.get("scale_stress_evaluation")
    if scale:
        expected_cells = {
            (population, candidates)
            for population in suite.population_levels
            for candidates in suite.shelter_candidate_levels
        }
        observed_cells = {
            (_i(row, "population_level"), _i(row, "shelter_candidate_level"))
            for row in scale
        }
        horizons = {_i(row, "horizon_timesteps") for row in scale}
        replicate_counts = []
        for city in _city_order(scale):
            for population, candidates in expected_cells:
                reference = [
                    row
                    for row in scale
                    if _city(row) == city
                    and _strategy(row) == "heuristic"
                    and _i(row, "population_level") == population
                    and _i(row, "shelter_candidate_level") == candidates
                ]
                replicate_counts.append(
                    len({_i(row, "scale_replication") for row in reference})
                )
        rl_policies = {
            _policy_id(row) for row in scale if _strategy(row) == "rl"
        }
        checks.append(
            {
                "check": "population_candidate_scale_matrix",
                "passed": observed_cells == expected_cells
                and horizons == {suite.scale_horizon_timesteps}
                and set(replicate_counts) == {suite.scale_replications_per_cell}
                and len(rl_policies) == suite.policy_seeds,
                "expected_cells": len(expected_cells),
                "observed_cells": len(observed_cells),
                "expected_policy_seeds": suite.policy_seeds,
                "observed_policy_seeds": len(rl_policies),
                "expected_horizon_timesteps": suite.scale_horizon_timesteps,
                "observed_horizon_timesteps": sorted(horizons),
                "expected_replications_per_city_cell": suite.scale_replications_per_cell,
                "observed_replication_counts": sorted(set(replicate_counts)),
            }
        )
    return checks


def _artifact_manifest(
    *,
    suite: ExperimentSuite,
    launch_dir: Path,
    output_dir: Path,
    generated: Sequence[Path],
    table_statuses: Mapping[str, Mapping],
    figure_statuses: Sequence[Mapping],
) -> Path:
    artifact_paths = sorted({Path(path).resolve() for path in generated if Path(path).exists()})
    payload = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "suite_config": {
            "path": str(suite.source_path),
            "sha256": suite.source_sha256,
        },
        "source_launch": str(launch_dir),
        "report_is_read_only_with_respect_to_simulation": True,
        "statistical_method": (
            "Policy seeds are resampled globally across fixed cities; matched "
            "scenarios are resampled independently within city; cities are equally "
            "weighted and are not resampled."
        ),
        "source_tables": {
            table_id: {
                "path": status["path"],
                "sha256": status.get("sha256"),
                "row_count": status.get("row_count", 0),
                "valid": status.get("valid", False),
            }
            for table_id, status in table_statuses.items()
        },
        "figure_statuses": list(figure_statuses),
        "artifacts": {
            str(path.relative_to(output_dir.resolve())): {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        },
    }
    path = output_dir / "full_experiment_figure_manifest.json"
    _write_json(path, payload)
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, default=DEFAULT_LAUNCH_DIR)
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_EXPERIMENT_SUITE_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        help="Override the suite's plotting-bootstrap draws (useful for tests only).",
    )
    parser.add_argument(
        "--require-complete-suite",
        action="store_true",
        help="Fail unless every E0--E6 table and confirmatory sample check passes.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Write readiness diagnostics without rendering figures.",
    )
    parser.add_argument(
        "--skip-maps",
        action="store_true",
        help="Do not copy or compose the OSM map panels.",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_draws is not None and args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    suite = load_experiment_suite(args.suite_config)
    launch_dir = args.launch_dir.expanduser().resolve()
    if not launch_dir.is_dir():
        raise FileNotFoundError(launch_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else launch_dir / "full_experiment_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, table_statuses = audit_suite_inputs(suite, launch_dir)
    design_checks = _design_checks(suite, tables)
    required_table_failures = [
        table_id
        for table_id, status in table_statuses.items()
        if status["required_for_complete_suite"] and not status["valid"]
    ]
    failed_design_checks = [
        check["check"] for check in design_checks if not check["passed"]
    ]
    suite_complete = not required_table_failures and not failed_design_checks

    family_by_id = {family.figure_id: family for family in suite.figure_families}
    figure_statuses: list[dict] = []

    def record_figure(
        figure_id: str,
        status: str,
        paths: Sequence[Path] = (),
        detail: str = "",
    ) -> None:
        figure_statuses.append(
            {
                "figure_id": figure_id,
                "title": family_by_id[figure_id].title,
                "status": status,
                "detail": detail,
                "artifacts": [str(path) for path in paths],
            }
        )

    if args.require_complete_suite and not suite_complete:
        readiness = {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "complete": False,
            "required_table_failures": required_table_failures,
            "failed_design_checks": failed_design_checks,
            "scale_stress_population_levels": list(suite.population_levels),
            "scale_stress_shelter_candidate_levels": list(
                suite.shelter_candidate_levels
            ),
            "scale_stress_horizon_timesteps": suite.scale_horizon_timesteps,
            "table_statuses": table_statuses,
            "design_checks": design_checks,
            "figure_statuses": [],
        }
        readiness_path = output_dir / "full_suite_readiness.json"
        _write_json(readiness_path, readiness)
        raise RuntimeError(
            "Full-suite reporting contract is not satisfied; inspect "
            f"{readiness_path}"
        )

    if args.validate_only:
        readiness = {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "complete": suite_complete,
            "expected_policy_seeds": suite.policy_seeds,
            "expected_training_episodes_per_city": suite.train_episodes_per_city,
            "expected_evaluation_scenarios_per_city": suite.evaluation_scenarios_per_city,
            "scale_stress_population_levels": list(suite.population_levels),
            "scale_stress_shelter_candidate_levels": list(
                suite.shelter_candidate_levels
            ),
            "scale_stress_horizon_timesteps": suite.scale_horizon_timesteps,
            "required_table_failures": required_table_failures,
            "failed_design_checks": failed_design_checks,
            "table_statuses": table_statuses,
            "design_checks": design_checks,
            "figure_statuses": [],
        }
        readiness_path = output_dir / "full_suite_readiness.json"
        _write_json(readiness_path, readiness)
        print(f"[FULL SUITE VALIDATION] complete={suite_complete} artifact={readiness_path}")
        return 0 if suite_complete or not args.require_complete_suite else 2

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    _style()
    draws = int(args.bootstrap_draws or suite.bootstrap_draws)
    seed = 845773
    generated: list[Path] = []
    statistical_summary: list[dict] = []

    training = tables.get("training_summary")
    if training:
        paths, rows = _plot_training(
            training,
            tables.get("checkpoint_evaluation"),
            output_dir,
            seed=seed,
            draws=draws,
        )
        generated.extend(paths)
        statistical_summary.extend(rows)
        detail = (
            "Includes fixed checkpoint evaluation."
            if tables.get("checkpoint_evaluation")
            else "Training-only fallback; checkpoint evaluation table is not yet available."
        )
        record_figure("F01", "generated", paths, detail)
        diagnostics = _load_ppo_diagnostics(launch_dir)
        if diagnostics:
            paths = _plot_ppo_diagnostics(training, diagnostics, output_dir)
            generated.extend(paths)
            record_figure("F02", "generated", paths)
        else:
            record_figure("F02", "skipped", detail="No policy PPO diagnostic files were found.")
    else:
        record_figure("F01", "skipped", detail="Missing valid training_summary table.")
        record_figure("F02", "skipped", detail="Missing valid training_summary table.")

    evaluation = tables.get("evaluation_summary")
    regime = tables.get("regime_evaluation")
    legacy_objective_path = launch_dir / "paper_figures" / "policy_objective_evaluation.csv"
    legacy_objective = _read_csv(legacy_objective_path) if legacy_objective_path.exists() else None
    performance_rows = regime or legacy_objective or evaluation
    if performance_rows:
        paths, rows = _plot_primary_performance(
            performance_rows, output_dir, seed=seed + 1000, draws=draws
        )
        generated.extend(paths)
        statistical_summary.extend(rows)
        source = (
            "regime_evaluation"
            if regime
            else "legacy policy-objective table"
            if legacy_objective
            else "evaluation_summary"
        )
        record_figure("F03", "generated", paths, f"Source: {source}.")
        paths, rows = _plot_safety_frontier(
            performance_rows, output_dir, seed=seed + 2000, draws=draws
        )
        generated.extend(paths)
        statistical_summary.extend(rows)
        record_figure("F06", "generated", paths, f"Source: {source}.")
    else:
        record_figure("F03", "skipped", detail="No valid evaluation table is available.")
        record_figure("F06", "skipped", detail="No valid evaluation table is available.")

    forest_rows = evaluation or performance_rows
    if forest_rows:
        paths, rows = _plot_city_forest(
            forest_rows, output_dir, seed=seed + 3000, draws=draws
        )
        generated.extend(paths)
        statistical_summary.extend(rows)
        record_figure("F04", "generated", paths)
    else:
        record_figure("F04", "skipped", detail="No valid matched RL/heuristic evaluation is available.")

    if regime:
        paths, rows = _plot_regime_heatmap(
            regime, suite, output_dir, seed=seed + 4000, draws=draws
        )
        generated.extend(paths)
        statistical_summary.extend(rows)
        record_figure("F05", "generated", paths)
    else:
        record_figure("F05", "skipped", detail="Missing valid regime_evaluation table.")

    optional_plotters = (
        (
            "F07",
            "scalability_evaluation",
            lambda data: _plot_scalability(data, output_dir, seed=seed + 5000, draws=draws),
        ),
        (
            "F08",
            "ablation_evaluation",
            lambda data: _plot_ablation(
                data, suite, output_dir, seed=seed + 6000, draws=draws
            ),
        ),
        (
            "F09",
            "transfer_evaluation",
            lambda data: _plot_transfer(data, output_dir, seed=seed + 7000, draws=draws),
        ),
        (
            "F10",
            "robustness_evaluation",
            lambda data: _plot_robustness(data, output_dir, seed=seed + 8000, draws=draws),
        ),
    )
    for figure_id, table_id, plotter in optional_plotters:
        data = tables.get(table_id)
        if data:
            paths, rows = plotter(data)
            generated.extend(paths)
            statistical_summary.extend(rows)
            record_figure(figure_id, "generated", paths)
        else:
            record_figure(figure_id, "skipped", detail=f"Missing valid {table_id} table.")

    scale_data = tables.get("scale_stress_evaluation")
    if scale_data:
        paths, rows = _plot_population_candidate_stress(
            scale_data, suite, output_dir, seed=seed + 9000, draws=draws
        )
        generated.extend(paths)
        statistical_summary.extend(rows)
        record_figure("F13", "generated", paths)
    else:
        record_figure(
            "F13",
            "skipped",
            detail="Missing valid scale_stress_evaluation table.",
        )

    map_rows = tables.get("map_figure_index")
    if map_rows and not args.skip_maps:
        map_paths, by_family = _copy_or_compose_maps(
            map_rows,
            suite.table_path(launch_dir, "map_figure_index"),
            output_dir,
        )
        generated.extend(map_paths)
        record_figure("F11", "generated", [Path(path) for path in by_family["F11"]])
        record_figure("F12", "generated", [Path(path) for path in by_family["F12"]])
    elif args.skip_maps:
        record_figure("F11", "skipped", detail="Map generation disabled by --skip-maps.")
        record_figure("F12", "skipped", detail="Map generation disabled by --skip-maps.")
    else:
        record_figure("F11", "skipped", detail="Missing valid map_figure_index table.")
        record_figure("F12", "skipped", detail="Missing valid map_figure_index table.")

    summary_path = output_dir / "figure_statistical_summary.csv"
    _write_csv(summary_path, statistical_summary)
    if summary_path.exists():
        generated.append(summary_path)
    readiness = {
        "schema_version": 1,
        "suite_id": suite.suite_id,
        "complete": suite_complete,
        "report_mode": "complete" if suite_complete else "partial_available_data",
        "expected_policy_seeds": suite.policy_seeds,
        "expected_training_episodes_per_city": suite.train_episodes_per_city,
        "expected_evaluation_factor_cells": len(suite.factor_cells),
        "expected_evaluation_scenarios_per_city": suite.evaluation_scenarios_per_city,
        "scale_stress_population_levels": list(suite.population_levels),
        "scale_stress_shelter_candidate_levels": list(
            suite.shelter_candidate_levels
        ),
        "scale_stress_horizon_timesteps": suite.scale_horizon_timesteps,
        "required_table_failures": required_table_failures,
        "failed_design_checks": failed_design_checks,
        "table_statuses": table_statuses,
        "design_checks": design_checks,
        "figure_statuses": figure_statuses,
        "interpretation_warning": (
            None
            if suite_complete
            else "Generated figures describe only available pilot/smoke evidence and must not be presented as the complete E0--E6 experiment."
        ),
    }
    readiness_path = output_dir / "full_suite_readiness.json"
    _write_json(readiness_path, readiness)
    generated.append(readiness_path)
    manifest_path = _artifact_manifest(
        suite=suite,
        launch_dir=launch_dir,
        output_dir=output_dir,
        generated=generated,
        table_statuses=table_statuses,
        figure_statuses=figure_statuses,
    )
    print(
        f"[FULL FIGURES COMPLETE] mode={readiness['report_mode']} artifact={manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
