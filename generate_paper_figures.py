#!/usr/bin/env python3
"""Generate auditable paper figures from a sealed multi-city experiment.

The script never trains or overwrites the sealed experiment.  It reads the
completed RL/heuristic evaluation, evaluates two preregistered controls on the
same held-out scenarios when they are not already cached, and reruns one
matched scenario per city with read-only visualization enabled.  The controls
are:

* ``random``: the same online decision interface with a random feasible exact
  candidate-site action; and
* ``initial_only``: the full shelter budget placed before evacuation begins.
  This is a static, anticipative comparator, not an equal online competitor.

Every derived table, map, and figure is recorded in a checksum manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LAUNCH_DIR = (
    PROJECT_ROOT / "runs" / "multicity_five_city_lr1e3_learning_audit_arm_20260906"
)
POLICY_ORDER = ("rl", "heuristic", "random", "initial_only")
POLICY_LABEL = {
    "rl": "RL",
    "heuristic": "Active-population\nheuristic",
    "random": "Random feasible\nregion",
    "initial_only": "Static\npredeployment",
}
POLICY_COLOR = {
    "rl": "#0072B2",
    "heuristic": "#E69F00",
    "random": "#999999",
    "initial_only": "#009E73",
}
CITY_LABEL = {
    "state_college_pa": "State College",
    "reading_pa": "Reading",
    "spokane_wa": "Spokane",
    "seattle_wa": "Seattle",
    "chicago_il": "Chicago",
}
METRIC_DIRECTIONS = {
    "policy_objective_return": 1.0,
    "safe_fraction": 1.0,
    "casualty_fraction": -1.0,
    "restricted_mean_time_to_safety": -1.0,
    "normalized_risk_weighted_person_time": -1.0,
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = (
        "replication",
        "city_scenario_replication",
        "policy_replication",
        "city_id",
        "city_scale_rank",
        "deployment_strategy",
        "scenario_seed",
        "policy_seed",
        "episode_return",
        "safe_completed",
        "casualty",
        "unfinished",
        "restricted_mean_time_to_safety",
        "normalized_risk_weighted_person_time",
        "initial_population",
        "visualization_manifest",
    )
    fields = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in fields]
    fieldnames.extend(sorted(fields.difference(fieldnames)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _i(row: dict, key: str) -> int:
    return int(float(row[key]))


def _scenario_key(row: dict) -> tuple[str, int, int]:
    return (
        str(row["city_id"]),
        _i(row, "city_scenario_replication"),
        _i(row, "scenario_seed"),
    )


def _metric(row: dict, name: str) -> float:
    if name == "policy_objective_return":
        return _f(row, "objective_episode_return")
    if name == "safe_fraction":
        return _f(row, "safe_completed") / max(1.0, _f(row, "initial_population"))
    if name == "casualty_fraction":
        return _f(row, "casualty") / max(1.0, _f(row, "initial_population"))
    if name == "unfinished_fraction":
        return _f(row, "unfinished") / max(1.0, _f(row, "initial_population"))
    return _f(row, name)


def _city_order(rows: Sequence[dict]) -> list[str]:
    ranks = {}
    for row in rows:
        city = str(row["city_id"])
        ranks[city] = _i(row, "city_scale_rank")
    return sorted(ranks, key=ranks.get)


def _validate_sealed_inputs(launch_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest_path = launch_dir / "experiment_manifest.json"
    training_path = launch_dir / "training_episode_summary.csv"
    evaluation_path = launch_dir / "evaluation_episode_summary.csv"
    parity_path = launch_dir / "interface_parity.json"
    for path in (manifest_path, training_path, evaluation_path, parity_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError("Paper figures require a completed, sealed experiment")
    parity = _read_json(parity_path)
    if parity.get("verified") is not True:
        raise RuntimeError("RL/heuristic interface parity has not been verified")
    training = _read_csv(training_path)
    evaluation = _read_csv(evaluation_path)
    if not training or not evaluation:
        raise RuntimeError("Training and evaluation tables must be non-empty")
    strategies = {str(row["deployment_strategy"]) for row in evaluation}
    if not {"rl", "heuristic"}.issubset(strategies):
        raise RuntimeError("The sealed evaluation must contain RL and heuristic rows")
    return manifest, training, evaluation


def _validate_matched_matrix(rows: Sequence[dict], strategies: Sequence[str]) -> None:
    by_strategy = {
        strategy: {_scenario_key(row): row for row in rows if row["deployment_strategy"] == strategy}
        for strategy in strategies
    }
    reference = set(by_strategy[strategies[0]])
    if not reference:
        raise RuntimeError("Evaluation scenario matrix is empty")
    for strategy in strategies:
        if set(by_strategy[strategy]) != reference:
            missing = sorted(reference.difference(by_strategy[strategy]))
            extra = sorted(set(by_strategy[strategy]).difference(reference))
            raise RuntimeError(
                f"Incomplete matched matrix for {strategy}: missing={missing[:3]} extra={extra[:3]}"
            )
        if len(by_strategy[strategy]) != sum(
            row["deployment_strategy"] == strategy for row in rows
        ):
            raise RuntimeError(f"Duplicate evaluation rows found for {strategy}")


def _representative_rows(evaluation: Sequence[dict], city_replication: int) -> list[dict]:
    selected = [
        row
        for row in evaluation
        if row["deployment_strategy"] == "heuristic"
        and _i(row, "city_scenario_replication") == int(city_replication)
    ]
    cities = _city_order(evaluation)
    by_city = {str(row["city_id"]): row for row in selected}
    if set(by_city) != set(cities):
        raise RuntimeError(
            f"Representative scenario {city_replication} is unavailable for every city"
        )
    return [by_city[city] for city in cities]


def _run_supplemental_controls(
    *,
    launch_dir: Path,
    output_dir: Path,
    manifest: dict,
    evaluation: Sequence[dict],
    map_scenario_replication: int,
) -> list[dict]:
    """Evaluate random and static controls on the sealed held-out scenarios."""
    from backtest import _run_episode

    output_path = output_dir / "supplemental_evaluation_episode_summary.csv"
    if output_path.exists():
        rows = _read_csv(output_path)
        _validate_matched_matrix(rows, ("random", "initial_only"))
        return rows

    reference_rows = sorted(
        (row for row in evaluation if row["deployment_strategy"] == "heuristic"),
        key=lambda row: _i(row, "replication"),
    )
    checkpoint_path = launch_dir / "policies" / "policy_001" / "regional_policy.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    effective = manifest.get("effective_overrides_by_city", {})
    rows: list[dict] = []
    for strategy in ("random", "initial_only"):
        for reference in reference_rows:
            city = str(reference["city_id"])
            if city not in effective:
                raise RuntimeError(f"Missing effective overrides for {city}")
            visualize = (
                _i(reference, "city_scenario_replication")
                == int(map_scenario_replication)
            )
            result = _run_episode(
                replication=_i(reference, "replication"),
                machine="paper",
                phase=str(output_dir / "supplemental_runs" / city),
                strategy=strategy,
                train_mode=False,
                scenario_seed=_i(reference, "scenario_seed"),
                policy_seed=_i(reference, "policy_seed"),
                checkpoint_path=str(checkpoint_path),
                diagnostics_path=str(output_dir / "unused_eval_diagnostics.csv"),
                overrides=dict(effective[city]),
                visualization_enabled=visualize,
                visualization_milestones="quartiles",
            )
            result.update(
                {
                    "replication": _i(reference, "replication"),
                    "city_scenario_replication": _i(
                        reference, "city_scenario_replication"
                    ),
                    "city_scale_rank": _i(reference, "city_scale_rank"),
                    "policy_replication": 0,
                }
            )
            if _scenario_key(result) != _scenario_key(reference):
                raise AssertionError("Supplemental control changed the scenario identity")
            if result["hazard_trajectory_digest"] != reference["hazard_trajectory_digest"]:
                raise AssertionError("Supplemental control changed the exogenous hazard path")
            rows.append(result)
            print(
                f"[SUPPLEMENTAL] {strategy} {city} "
                f"scenario={result['city_scenario_replication']} "
                f"return={float(result['episode_return']):.4f}",
                flush=True,
            )
    _write_csv(output_path, rows)
    _validate_matched_matrix(rows, ("random", "initial_only"))
    return rows


def _run_policy_objective_evaluation(
    *,
    launch_dir: Path,
    output_dir: Path,
    manifest: dict,
    sealed_evaluation: Sequence[dict],
) -> list[dict]:
    """Evaluate every policy with an action-count-invariant episode objective.

    The sealed experiment predates the separate policy-objective accumulator.
    Its RL/heuristic action returns remain the primary learning comparison and
    are reproduced exactly here.  These derived reruns add the complete
    t=0..terminal objective needed for a meaningful static-policy comparison.
    """
    from backtest import _run_episode

    output_path = output_dir / "policy_objective_evaluation.csv"
    if output_path.exists():
        rows = _read_csv(output_path)
        _validate_matched_matrix(rows, POLICY_ORDER)
        required = (
            "objective_episode_return",
            "objective_safe_completion_reward",
            "objective_casualty_penalty",
            "objective_risk_time_penalty",
            "objective_risk_weighted_person_time",
        )
        if not all(all(str(row.get(key, "")).strip() for key in required) for row in rows):
            raise RuntimeError("Cached policy-objective evaluation uses an obsolete schema")
        return rows

    references = sorted(
        (row for row in sealed_evaluation if row["deployment_strategy"] == "heuristic"),
        key=lambda row: _i(row, "replication"),
    )
    sealed_by_key = {
        (str(row["deployment_strategy"]), _scenario_key(row)): row
        for row in sealed_evaluation
    }
    checkpoint_path = launch_dir / "policies" / "policy_001" / "regional_policy.pt"
    effective = manifest.get("effective_overrides_by_city", {})
    rows = []
    for reference in references:
        city = str(reference["city_id"])
        for strategy in POLICY_ORDER:
            result = _run_episode(
                replication=_i(reference, "replication"),
                machine="paper",
                phase=str(output_dir / "policy_objective_runs" / city),
                strategy=strategy,
                train_mode=False,
                scenario_seed=_i(reference, "scenario_seed"),
                policy_seed=_i(reference, "policy_seed"),
                checkpoint_path=str(checkpoint_path),
                diagnostics_path=str(output_dir / "unused_eval_diagnostics.csv"),
                overrides=dict(effective[city]),
                visualization_enabled=False,
            )
            result.update(
                {
                    "replication": _i(reference, "replication"),
                    "city_scenario_replication": _i(
                        reference, "city_scenario_replication"
                    ),
                    "city_scale_rank": _i(reference, "city_scale_rank"),
                    "policy_replication": 1 if strategy == "rl" else 0,
                }
            )
            if _scenario_key(result) != _scenario_key(reference):
                raise AssertionError("Policy-objective rerun changed scenario identity")
            if result["hazard_trajectory_digest"] != reference["hazard_trajectory_digest"]:
                raise AssertionError("Policy-objective rerun changed the exogenous hazard path")
            if strategy in {"rl", "heuristic"}:
                sealed = sealed_by_key[(strategy, _scenario_key(reference))]
                if not math.isclose(
                    float(result["episode_return"]),
                    float(sealed["episode_return"]),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise AssertionError(
                        f"Objective instrumentation changed sealed action return for {strategy}/{city}"
                    )
                result["sealed_action_return_reproduced"] = 1
            else:
                result["sealed_action_return_reproduced"] = ""
            rows.append(result)
            print(
                f"[OBJECTIVE EVAL] {city} scenario={result['city_scenario_replication']} "
                f"strategy={strategy} objective={float(result['objective_episode_return']):.4f}",
                flush=True,
            )
    _write_csv(output_path, rows)
    _validate_matched_matrix(rows, POLICY_ORDER)
    return rows


def _validate_visualized_rerun(result: dict, reference: dict) -> dict:
    exact_fields = (
        "scenario_seed",
        "policy_seed",
        "city_id",
        "deployment_strategy",
        "initial_observation_digest",
        "hazard_trajectory_digest",
        "safe_completed",
        "casualty",
        "unfinished",
        "deployments_made",
    )
    mismatches = {
        key: {"expected": reference.get(key), "observed": result.get(key)}
        for key in exact_fields
        if str(result.get(key)) != str(reference.get(key))
    }
    numeric_fields = (
        "episode_return",
        "restricted_mean_time_to_safety",
        "normalized_risk_weighted_person_time",
    )
    for key in numeric_fields:
        if not math.isclose(
            float(result[key]), float(reference[key]), rel_tol=1e-10, abs_tol=1e-10
        ):
            mismatches[key] = {
                "expected": float(reference[key]),
                "observed": float(result[key]),
            }
    manifest_path = Path(str(result.get("visualization_manifest", "")))
    if not manifest_path.exists():
        mismatches["visualization_manifest"] = {
            "expected": "existing manifest",
            "observed": str(manifest_path),
        }
    return {"verified": not mismatches, "mismatches": mismatches}


def _run_dynamic_map_reruns(
    *,
    launch_dir: Path,
    output_dir: Path,
    manifest: dict,
    evaluation: Sequence[dict],
    map_scenario_replication: int,
) -> list[dict]:
    """Reproduce one sealed RL/heuristic scenario per city with maps enabled."""
    from backtest import _run_episode

    cache_path = output_dir / "dynamic_map_episode_summary.csv"
    validation_path = output_dir / "dynamic_map_rerun_validation.json"
    if cache_path.exists() and validation_path.exists():
        validation = _read_json(validation_path)
        if validation.get("verified") is not True:
            raise RuntimeError("Cached visualization reruns did not reproduce sealed outcomes")
        rows = _read_csv(cache_path)
        if len(rows) != 2 * len(_city_order(evaluation)):
            raise RuntimeError("Cached visualization rerun matrix is incomplete")
        if not all(Path(str(row["visualization_manifest"])).exists() for row in rows):
            raise RuntimeError("A cached visualization manifest is missing")
        return rows

    checkpoint_path = launch_dir / "policies" / "policy_001" / "regional_policy.pt"
    effective = manifest.get("effective_overrides_by_city", {})
    representative = _representative_rows(evaluation, map_scenario_replication)
    sealed_by_key = {
        (str(row["deployment_strategy"]), _scenario_key(row)): row
        for row in evaluation
    }
    rows = []
    checks = []
    for strategy in ("rl", "heuristic"):
        for scenario in representative:
            city = str(scenario["city_id"])
            result = _run_episode(
                replication=_i(scenario, "replication"),
                machine="paper",
                phase=str(output_dir / "map_runs" / city),
                strategy=strategy,
                train_mode=False,
                scenario_seed=_i(scenario, "scenario_seed"),
                policy_seed=_i(scenario, "policy_seed"),
                checkpoint_path=str(checkpoint_path),
                diagnostics_path=str(output_dir / "unused_eval_diagnostics.csv"),
                overrides=dict(effective[city]),
                visualization_enabled=True,
                visualization_milestones="quartiles",
            )
            result.update(
                {
                    "replication": _i(scenario, "replication"),
                    "city_scenario_replication": _i(
                        scenario, "city_scenario_replication"
                    ),
                    "city_scale_rank": _i(scenario, "city_scale_rank"),
                    "policy_replication": 1 if strategy == "rl" else 0,
                }
            )
            reference = sealed_by_key[(strategy, _scenario_key(scenario))]
            check = _validate_visualized_rerun(result, reference)
            check.update({"strategy": strategy, "city_id": city})
            checks.append(check)
            if not check["verified"]:
                raise AssertionError(
                    f"Visualization changed the sealed outcome for {strategy}/{city}: "
                    f"{check['mismatches']}"
                )
            rows.append(result)
            print(f"[MAP RERUN VERIFIED] {strategy} {city}", flush=True)
    _write_csv(cache_path, rows)
    _write_json(
        validation_path,
        {
            "verified": all(check["verified"] for check in checks),
            "criterion": (
                "Visualization is non-interventional when exact outcomes, random-stream "
                "digests, observations, and returns reproduce the sealed evaluation."
            ),
            "checks": checks,
        },
    )
    return rows


def _map_manifests(
    dynamic_rows: Sequence[dict], supplemental_rows: Sequence[dict]
) -> dict[tuple[str, str], Path]:
    manifests = {}
    for row in [*dynamic_rows, *supplemental_rows]:
        value = str(row.get("visualization_manifest", "")).strip()
        if not value:
            continue
        path = Path(value)
        if path.exists():
            manifests[(str(row["city_id"]), str(row["deployment_strategy"]))] = path
    return manifests


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(values.size):
        start = max(0, index - int(window) + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def _training_blocks(training: Sequence[dict]) -> tuple[list[str], list[dict]]:
    cities = _city_order(training)
    city_count = len(cities)
    ordered = sorted(training, key=lambda row: _i(row, "replication"))
    if len(ordered) % city_count:
        raise RuntimeError("Training episodes do not form complete equal-city blocks")
    blocks = []
    for offset in range(0, len(ordered), city_count):
        rows = ordered[offset : offset + city_count]
        found = {str(row["city_id"]) for row in rows}
        if found != set(cities):
            raise RuntimeError(f"Training block {offset // city_count + 1} is city-imbalanced")
        blocks.append(
            {
                "block": offset // city_count + 1,
                "rows": rows,
                **{
                    metric: float(np.mean([_metric(row, metric) for row in rows]))
                    for metric in (
                        "episode_return",
                        "safe_completion_reward",
                        "casualty_penalty",
                        "risk_time_penalty",
                        "heuristic_agreement_rate",
                        "entropy",
                    )
                },
            }
        )
    return cities, blocks


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


def _save_figure(fig, output_dir: Path, stem: str, *, vector: bool = True) -> list[Path]:
    paths = []
    for suffix in (("png", "svg") if vector else ("png",)):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=320 if suffix == "png" else None, facecolor="white")
        paths.append(path)
    return paths


def _plot_training(
    training: Sequence[dict], output_dir: Path, diagnostics_path: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    cities, blocks = _training_blocks(training)
    x = np.asarray([block["block"] for block in blocks], dtype=float)
    window = min(5, len(blocks))
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for city in cities:
        city_rows = sorted(
            (row for row in training if row["city_id"] == city),
            key=lambda row: _i(row, "city_training_replication"),
        )
        axes[0].plot(
            np.arange(1, len(city_rows) + 1),
            [_metric(row, "episode_return") for row in city_rows],
            color="#A6A6A6",
            linewidth=0.75,
            alpha=0.27,
        )
    returns = np.asarray([block["episode_return"] for block in blocks])
    axes[0].plot(x, returns, color="#56B4E9", linewidth=1.0, marker="o", markersize=2.5, label="Equal-city block mean")
    axes[0].plot(x, _moving_average(returns, window), color="#0072B2", linewidth=2.5, label=f"{window}-block moving mean")
    axes[0].set(title="Training return", xlabel="Equal-city training block (5 episodes)", ylabel="Undiscounted episode return")
    axes[0].legend(frameon=False)

    components = (
        ("safe_completion_reward", "Safe-completion term", "#0072B2"),
        ("casualty_penalty", "Casualty term", "#D55E00"),
        ("risk_time_penalty", "Risk-time term", "#009E73"),
        ("episode_return", "Total return", "#111111"),
    )
    for metric, label, color in components:
        values = np.asarray([block[metric] for block in blocks])
        axes[1].plot(x, _moving_average(values, window), color=color, linewidth=2.0, label=label)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set(title="Reward decomposition", xlabel="Equal-city training block (5 episodes)", ylabel="Normalized reward contribution")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Balanced multi-city PPO training")
    fig.tight_layout()
    paths.extend(_save_figure(fig, output_dir, "01_training_reward_convergence"))
    plt.close(fig)

    if not diagnostics_path.exists():
        raise FileNotFoundError(diagnostics_path)
    diagnostics = _read_csv(diagnostics_path)
    updates = [row for row in diagnostics if _f(row, "optimizer_updated") > 0.5]
    if not updates:
        raise RuntimeError("No PPO optimizer updates were recorded")
    update_x = np.arange(1, len(updates) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    axes[0, 0].plot(update_x, [_f(row, "approximate_kl") for row in updates], color="#D55E00", marker="o")
    axes[0, 0].axhline(0.03, color="#555555", linestyle="--", linewidth=1.0, label="Target KL = 0.03")
    axes[0, 0].set(title="PPO policy update size", xlabel="Optimizer update", ylabel="Approximate KL divergence")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(update_x, [_f(row, "explained_variance") for row in updates], color="#0072B2", marker="o")
    axes[0, 1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0, 1].set(title="Critic fit", xlabel="Optimizer update", ylabel="Explained variance")
    entropy = np.asarray([block["entropy"] for block in blocks])
    agreement = np.asarray([block["heuristic_agreement_rate"] for block in blocks])
    axes[1, 0].plot(x, _moving_average(entropy, window), color="#CC79A7", linewidth=2.2)
    axes[1, 0].set(title="Policy entropy", xlabel="Equal-city training block", ylabel="Entropy (nats)")
    axes[1, 1].plot(x, _moving_average(agreement, window), color="#009E73", linewidth=2.2)
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set(title="Behavioral overlap with benchmark", xlabel="Equal-city training block", ylabel="Action agreement rate")
    fig.suptitle("PPO learning diagnostics")
    fig.tight_layout()
    paths.extend(_save_figure(fig, output_dir, "02_training_ppo_diagnostics"))
    plt.close(fig)
    return paths


def _stratified_ci(
    rows: Sequence[dict], metric: str, *, seed: int, draws: int = 10000
) -> tuple[float, float, float]:
    cities = _city_order(rows)
    grouped = {
        city: np.asarray([_metric(row, metric) for row in rows if row["city_id"] == city])
        for city in cities
    }
    estimate = float(np.mean([values.mean() for values in grouped.values()]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = np.mean(
            [rng.choice(values, size=values.size, replace=True).mean() for values in grouped.values()]
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper)


def _paired_rows(rows: Sequence[dict], strategy: str, reference: str = "heuristic") -> list[tuple[dict, dict]]:
    left = {_scenario_key(row): row for row in rows if row["deployment_strategy"] == strategy}
    right = {_scenario_key(row): row for row in rows if row["deployment_strategy"] == reference}
    if set(left) != set(right):
        raise RuntimeError(f"{strategy} and {reference} do not share the same scenarios")
    return [(left[key], right[key]) for key in sorted(left)]


def _paired_improvement_ci(
    rows: Sequence[dict], strategy: str, metric: str, *, seed: int, draws: int = 10000
) -> tuple[float, float, float, dict[str, float]]:
    pairs = _paired_rows(rows, strategy)
    direction = METRIC_DIRECTIONS[metric]
    by_city = {}
    for strategy_row, heuristic_row in pairs:
        city = str(strategy_row["city_id"])
        by_city.setdefault(city, []).append(
            direction * (_metric(strategy_row, metric) - _metric(heuristic_row, metric))
        )
    arrays = {city: np.asarray(values, dtype=float) for city, values in by_city.items()}
    city_means = {city: float(values.mean()) for city, values in arrays.items()}
    estimate = float(np.mean(list(city_means.values())))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = np.mean(
            [rng.choice(values, size=values.size, replace=True).mean() for values in arrays.values()]
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return estimate, float(lower), float(upper), city_means


def _plot_evaluation(rows: Sequence[dict], output_dir: Path, launch_seed: int) -> tuple[list[Path], list[dict]]:
    import matplotlib.pyplot as plt

    _validate_matched_matrix(rows, POLICY_ORDER)
    paths = []
    summary = []
    metrics = (
        ("policy_objective_return", "Policy-objective return", "Higher is better", 1.0),
        ("safe_fraction", "Safe completion", "Percent of population", 100.0),
        ("casualty_fraction", "Casualty", "Percent of population", 100.0),
        ("restricted_mean_time_to_safety", "Restricted mean time to safety", "Timesteps", 1.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    y = np.arange(len(POLICY_ORDER))
    for metric_index, ((metric, title, xlabel, scale), axis) in enumerate(zip(metrics, axes.flat)):
        estimates, lower, upper = [], [], []
        for strategy_index, strategy in enumerate(POLICY_ORDER):
            policy_rows = [row for row in rows if row["deployment_strategy"] == strategy]
            estimate, lo, hi = _stratified_ci(
                policy_rows,
                metric,
                seed=int(launch_seed + 100 * metric_index + strategy_index),
            )
            estimates.append(scale * estimate)
            lower.append(scale * lo)
            upper.append(scale * hi)
            summary.append(
                {
                    "comparison": "absolute",
                    "strategy": strategy,
                    "reference_strategy": "",
                    "metric": metric,
                    "estimate": estimate,
                    "bootstrap_95_ci_lower": lo,
                    "bootstrap_95_ci_upper": hi,
                    "positive_favors_strategy": "",
                }
            )
        estimates_array = np.asarray(estimates)
        errors = np.vstack((estimates_array - np.asarray(lower), np.asarray(upper) - estimates_array))
        for index, strategy in enumerate(POLICY_ORDER):
            axis.errorbar(
                estimates_array[index],
                y[index],
                xerr=errors[:, index : index + 1],
                fmt="o",
                color=POLICY_COLOR[strategy],
                capsize=3,
                markersize=7,
                linewidth=1.5,
            )
        axis.set_yticks(y, [POLICY_LABEL[strategy] for strategy in POLICY_ORDER])
        axis.invert_yaxis()
        axis.set(title=title, xlabel=xlabel)
        if metric == "safe_fraction":
            axis.xaxis.set_major_formatter(lambda value, position: f"{value:.0f}%")
        elif metric == "casualty_fraction":
            axis.xaxis.set_major_formatter(lambda value, position: f"{value:.1f}%")
    fig.suptitle("Matched held-out policy performance across five cities")
    fig.text(
        0.5,
        0.005,
        "Points are equal-city macro means; intervals bootstrap scenarios within city. "
        "Static predeployment uses the full budget before t = 0.",
        ha="center",
        fontsize=8.3,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    paths.extend(_save_figure(fig, output_dir, "03_evaluation_policy_performance"))
    plt.close(fig)

    comparison_metrics = (
        ("policy_objective_return", "Policy-objective return improvement"),
        ("safe_fraction", "Safe-completion improvement (percentage points)"),
        ("casualty_fraction", "Casualty reduction (percentage points)"),
        ("restricted_mean_time_to_safety", "Time-to-safety reduction (timesteps)"),
    )
    comparators = ("rl", "random", "initial_only")
    cities = _city_order(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    for metric_index, ((metric, title), axis) in enumerate(zip(comparison_metrics, axes.flat)):
        scale = 100.0 if "fraction" in metric else 1.0
        axis.axvline(0.0, color="#333333", linewidth=1.0)
        for strategy_index, strategy in enumerate(comparators):
            estimate, lo, hi, city_means = _paired_improvement_ci(
                rows,
                strategy,
                metric,
                seed=int(launch_seed + 500 + 100 * metric_index + strategy_index),
            )
            y_value = strategy_index
            offsets = np.linspace(-0.12, 0.12, len(cities))
            axis.scatter(
                [scale * city_means[city] for city in cities],
                y_value + offsets,
                s=19,
                color=POLICY_COLOR[strategy],
                alpha=0.35,
                marker="o",
                label="City means" if strategy_index == 0 else None,
            )
            axis.errorbar(
                scale * estimate,
                y_value,
                xerr=np.asarray([[scale * (estimate - lo)], [scale * (hi - estimate)]]),
                fmt="D",
                color=POLICY_COLOR[strategy],
                capsize=3.5,
                markersize=7,
                linewidth=1.7,
            )
            summary.append(
                {
                    "comparison": "paired_improvement",
                    "strategy": strategy,
                    "reference_strategy": "heuristic",
                    "metric": metric,
                    "estimate": estimate,
                    "bootstrap_95_ci_lower": lo,
                    "bootstrap_95_ci_upper": hi,
                    "positive_favors_strategy": 1,
                }
            )
        axis.set_yticks(np.arange(len(comparators)), [POLICY_LABEL[strategy] for strategy in comparators])
        axis.invert_yaxis()
        axis.set(title=title, xlabel="Positive favors policy over heuristic")
    fig.suptitle("Matched performance relative to active-population heuristic")
    fig.text(
        0.5,
        0.005,
        "Diamonds: equal-city macro means and scenario-bootstrap intervals; translucent points: city means.",
        ha="center",
        fontsize=8.3,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    paths.extend(_save_figure(fig, output_dir, "04_evaluation_vs_heuristic"))
    plt.close(fig)

    cities = _city_order(rows)
    fig, axes = plt.subplots(len(cities), 1, figsize=(10.6, 10.8), sharex=True)
    for axis, city in zip(axes, cities):
        safe, casualty, unfinished = [], [], []
        for strategy in POLICY_ORDER:
            policy_rows = [
                row for row in rows if row["city_id"] == city and row["deployment_strategy"] == strategy
            ]
            safe.append(100.0 * np.mean([_metric(row, "safe_fraction") for row in policy_rows]))
            casualty.append(100.0 * np.mean([_metric(row, "casualty_fraction") for row in policy_rows]))
            unfinished.append(100.0 * np.mean([_metric(row, "unfinished_fraction") for row in policy_rows]))
        x = np.arange(len(POLICY_ORDER))
        axis.bar(x, safe, color="#009E73", label="Safe")
        axis.bar(x, casualty, bottom=safe, color="#D55E00", label="Casualty")
        axis.bar(x, unfinished, bottom=np.asarray(safe) + np.asarray(casualty), color="#A6A6A6", label="Unfinished")
        axis.set_ylim(0, 100)
        axis.set_ylabel(f"{CITY_LABEL.get(city, city)}\nPopulation (%)")
        axis.grid(axis="x", visible=False)
    axes[-1].set_xticks(np.arange(len(POLICY_ORDER)), [POLICY_LABEL[strategy].replace("\n", " ") for strategy in POLICY_ORDER])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.971),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Evacuation outcomes by city and policy", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    paths.extend(_save_figure(fig, output_dir, "05_evaluation_outcomes_by_city"))
    plt.close(fig)
    return paths, summary


def _artifact_path(manifest: dict, name: str) -> Path:
    artifact = manifest["artifacts"].get(name)
    if not artifact:
        raise KeyError(f"Visualization manifest lacks {name}")
    path = Path(artifact["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    if _sha256(path) != artifact["sha256"]:
        raise RuntimeError(f"Visualization artifact checksum mismatch: {path}")
    return path


def _compose_map_figures(
    *,
    manifests: dict[tuple[str, str], Path],
    cities: Sequence[str],
    output_dir: Path,
) -> tuple[list[Path], list[dict]]:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    paths = []
    index_rows = []
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    for city in cities:
        missing = [strategy for strategy in POLICY_ORDER if (city, strategy) not in manifests]
        if missing:
            raise RuntimeError(f"Missing representative maps for {city}: {missing}")
        loaded = {strategy: _read_json(manifests[(city, strategy)]) for strategy in POLICY_ORDER}
        final_images = {}
        progress_images = {}
        for strategy, manifest in loaded.items():
            final_time = max(int(value) for value in manifest["captured_milestones"])
            final_images[strategy] = _artifact_path(manifest, f"milestone_t{final_time:04d}.png")
            progress_images[strategy] = _artifact_path(manifest, "evacuation_milestones.png")
            index_rows.append(
                {
                    "city_id": city,
                    "strategy": strategy,
                    "scenario_seed": manifest["scenario_seed"],
                    "policy_seed": manifest["policy_seed"],
                    "visualization_manifest": str(manifests[(city, strategy)]),
                    "shelter_installation_map": str(final_images[strategy]),
                    "evacuation_progress_map": str(progress_images[strategy]),
                    "non_interventional": manifest["non_interventional"],
                }
            )

        fig, axes = plt.subplots(1, 4, figsize=(15.8, 4.25))
        for axis, strategy in zip(axes, POLICY_ORDER):
            axis.imshow(mpimg.imread(final_images[strategy]))
            axis.set_title(POLICY_LABEL[strategy].replace("\n", " "))
            axis.axis("off")
        fig.suptitle(f"Final shelter configuration — {CITY_LABEL.get(city, city)}")
        fig.text(
            0.995,
            0.005,
            "Road network © OpenStreetMap contributors (ODbL)",
            ha="right",
            fontsize=7,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        paths.extend(_save_figure(fig, map_dir, f"shelter_installations_{city}", vector=False))
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(16.2, 12.3))
        for axis, strategy in zip(axes, POLICY_ORDER):
            axis.imshow(mpimg.imread(progress_images[strategy]))
            axis.set_ylabel(
                POLICY_LABEL[strategy].replace("\n", " "),
                rotation=90,
                labelpad=8,
                fontweight="bold",
            )
            axis.axis("off")
        fig.suptitle(f"Matched evacuation progress — {CITY_LABEL.get(city, city)}")
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        paths.extend(_save_figure(fig, map_dir, f"evacuation_progress_{city}", vector=False))
        plt.close(fig)
    _write_csv(output_dir / "map_figure_index.csv", index_rows)
    return paths, index_rows


def _build_artifact_manifest(
    *,
    launch_dir: Path,
    output_dir: Path,
    figure_paths: Sequence[Path],
    table_paths: Sequence[Path],
    combined_rows: Sequence[dict],
    map_index: Sequence[dict],
) -> Path:
    all_paths = sorted({Path(path).resolve() for path in [*figure_paths, *table_paths]})
    payload = {
        "schema_version": 1,
        "source_experiment": str(launch_dir.resolve()),
        "sealed_source_unchanged": True,
        "matched_scenarios_per_city": len(
            {
                _i(row, "city_scenario_replication")
                for row in combined_rows
                if row["city_id"] == _city_order(combined_rows)[0]
                and row["deployment_strategy"] == "heuristic"
            }
        ),
        "strategies": list(POLICY_ORDER),
        "policy_interpretation": {
            "rl": "learned online regional priority policy",
            "heuristic": "online region with most active pedestrians; equal RL benchmark",
            "random": "random feasible region using the same online action interface",
            "initial_only": "full shelter budget predeployed before evacuation; static anticipative comparator",
        },
        "representative_map_count": len(map_index),
        "artifacts": {
            str(path.relative_to(output_dir.resolve())): {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in all_paths
        },
    }
    manifest_path = output_dir / "paper_figure_manifest.json"
    _write_json(manifest_path, payload)
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, default=DEFAULT_LAUNCH_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--map-scenario-replication", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    launch_dir = args.launch_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else launch_dir / "paper_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    _style()

    manifest, training, sealed_evaluation = _validate_sealed_inputs(launch_dir)
    supplemental = _run_supplemental_controls(
        launch_dir=launch_dir,
        output_dir=output_dir,
        manifest=manifest,
        evaluation=sealed_evaluation,
        map_scenario_replication=args.map_scenario_replication,
    )
    dynamic_maps = _run_dynamic_map_reruns(
        launch_dir=launch_dir,
        output_dir=output_dir,
        manifest=manifest,
        evaluation=sealed_evaluation,
        map_scenario_replication=args.map_scenario_replication,
    )
    combined = _run_policy_objective_evaluation(
        launch_dir=launch_dir,
        output_dir=output_dir,
        manifest=manifest,
        sealed_evaluation=sealed_evaluation,
    )
    _validate_matched_matrix(combined, POLICY_ORDER)
    combined_path = output_dir / "combined_matched_evaluation.csv"
    _write_csv(combined_path, combined)

    figure_paths = []
    figure_paths.extend(
        _plot_training(
            training,
            output_dir,
            launch_dir / "policies" / "policy_001" / "ppo_diagnostics.csv",
        )
    )
    evaluation_paths, statistical_summary = _plot_evaluation(
        combined, output_dir, int(manifest["launch_seed"])
    )
    figure_paths.extend(evaluation_paths)
    summary_path = output_dir / "figure_statistical_summary.csv"
    _write_csv(summary_path, statistical_summary)

    map_manifests = _map_manifests(dynamic_maps, supplemental)
    map_paths, map_index = _compose_map_figures(
        manifests=map_manifests,
        cities=_city_order(combined),
        output_dir=output_dir,
    )
    figure_paths.extend(map_paths)
    table_paths = (
        combined_path,
        output_dir / "supplemental_evaluation_episode_summary.csv",
        output_dir / "dynamic_map_episode_summary.csv",
        output_dir / "dynamic_map_rerun_validation.json",
        output_dir / "policy_objective_evaluation.csv",
        output_dir / "figure_statistical_summary.csv",
        output_dir / "map_figure_index.csv",
    )
    artifact_manifest = _build_artifact_manifest(
        launch_dir=launch_dir,
        output_dir=output_dir,
        figure_paths=figure_paths,
        table_paths=table_paths,
        combined_rows=combined,
        map_index=map_index,
    )
    print(f"[PAPER FIGURES COMPLETE] {artifact_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
