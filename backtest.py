#!/usr/bin/env python3
"""Reproducible matched-seed backtest for regional shelter policies.

RL and every dynamic benchmark use the same regional observation contract,
exact-candidate action mask, deployment schedule, and budget. The primary
confirmatory contrast remains RL versus the active-population heuristic; the
additional policies are predeclared diagnostic benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from DecisionInterface import (
    BENCHMARK_POLICY_CONTRACTS,
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
)
from GNN import HEURISTIC_PRIOR_SCALE, RESIDUAL_LOGIT_BOUND
from RLBridge import (
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ROLLOUT_EPISODES,
    DEFAULT_TARGET_KL,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
DYNAMIC_STRATEGIES = (
    "rl",
    "risk_reduction",
    "route_saving",
    "heuristic",
    "hazard_weighted",
    "accessibility_deficit",
    "random",
)
STATIC_STRATEGIES = ("initial_only", "static_greedy", "rl_precommit")
LEARNED_STRATEGIES = ("rl", "rl_precommit")

EVALUATION_METRICS = {
    "episode_return": "higher",
    "safe_completed": "higher",
    "casualty": "lower",
    "unfinished": "lower",
    "restricted_mean_time_to_safety": "lower",
    "normalized_risk_weighted_person_time": "lower",
}


def _seed(launch_seed: int, stream: int, index: int = 0) -> int:
    sequence = np.random.SeedSequence([int(launch_seed), int(stream), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _strict_json_value(value):
    """Convert NumPy values and non-finite diagnostics to strict JSON values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _json_dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(
            _strict_json_value(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    os.replace(temporary, path)


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    preferred = [
        "replication",
        "policy_replication",
        "deployment_strategy",
        "scenario_seed",
        "policy_seed",
        "episode_return",
        "safe_completed",
        "shelter_evacuated",
        "arrival",
        "casualty",
        "unfinished",
        "restricted_mean_time_to_safety",
        "mean_safe_completion_time",
        "mean_evacuation_time",
        "normalized_risk_weighted_person_time",
        "risk_weighted_person_time",
        "decisions",
        "deployments_made",
        "maximum_dynamic_deployments",
        "active_shelters",
        "initial_observation_digest",
    ]
    all_columns = {key for row in rows for key in row}
    columns = [key for key in preferred if key in all_columns]
    columns.extend(sorted(all_columns.difference(columns)))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_metadata() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args,
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "unavailable"

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain") not in {"", "unavailable"}),
    }


def _dependency_versions() -> dict:
    versions = {}
    for package in ("numpy", "pandas", "matplotlib", "torch", "osmnx", "geopandas", "shapely"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _parse_overrides(items: Iterable[str]) -> dict:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must use NAME=VALUE, got {item!r}")
        name, raw = item.split("=", 1)
        name = name.strip()
        raw = raw.strip()
        if not name:
            raise ValueError("Override name cannot be empty")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[name] = value
    return overrides


def _run_episode(
    *,
    replication: int,
    machine: str,
    phase: str,
    strategy: str,
    train_mode: bool,
    scenario_seed: int,
    policy_seed: int,
    checkpoint_path: str,
    diagnostics_path: str,
    overrides: dict,
    visualization_enabled: bool = False,
    visualization_milestones="quartiles",
    visualization_individual_snapshots: bool = True,
    visualization_vector_outputs: bool = True,
) -> dict:
    from Core import Core

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    core = Core(machine)
    core.initSimulator(
        replication=replication,
        machine=machine,
        config_overrides=overrides,
        phase=phase,
        train_mode=train_mode,
        deployment_strategy=strategy,
        run_tag=strategy,
        scenario_seed=scenario_seed,
        policy_seed=policy_seed,
        checkpoint_path=checkpoint_path,
        diagnostics_path=diagnostics_path,
        visualization_enabled=visualization_enabled,
        visualization_milestones=visualization_milestones,
        visualization_individual_snapshots=visualization_individual_snapshots,
        visualization_vector_outputs=visualization_vector_outputs,
    )
    if core.episode_summary is None:
        raise RuntimeError("Core completed without an episode summary")
    result = dict(core.episode_summary)
    result["episode_wall_time_s"] = float(time.perf_counter() - wall_start)
    result["episode_cpu_time_s"] = float(time.process_time() - cpu_start)
    horizon_minutes = max(1e-12, float(result.get("horizon_minutes", 0.0)))
    result["wall_time_per_simulated_minute_s"] = float(
        result["episode_wall_time_s"] / horizon_minutes
    )
    result["replication"] = int(replication)
    result["maximum_dynamic_deployments"] = int(core.rl.maximum_deployments)
    result["deployments_made"] = int(core.rl.deployments_made)
    return result


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    sample_indices = rng.integers(0, values.size, size=(int(draws), values.size))
    means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _paired_randomization_pvalue(values: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    observed = abs(float(values.mean()))
    if values.size == 0:
        return float("nan")
    # The policy-training seed is the independent unit.  With the default five
    # seeds there are only 2^5 sign assignments, so enumerate the exact null
    # distribution instead of reporting spuriously precise Monte Carlo values.
    exact_assignments = 1 << int(values.size)
    if values.size <= 20 and exact_assignments <= int(draws):
        exceedances = 0
        for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
            permuted = abs(float(np.mean(np.asarray(signs, dtype=float) * values)))
            exceedances += int(permuted >= observed - 1e-15)
        return float(exceedances / exact_assignments)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(draws), values.size))
    null_means = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(null_means >= observed)) / (int(draws) + 1))


def _evaluation_metric(row: dict, metric: str) -> float:
    """Return the action-count-invariant value used for policy evaluation."""
    if metric == "episode_return":
        objective = row.get("objective_episode_return")
        if objective is not None and str(objective).strip():
            return float(objective)
    if metric == "normalized_risk_weighted_person_time":
        objective = row.get("objective_risk_weighted_person_time")
        population = row.get("initial_population")
        horizon = row.get("horizon_transitions")
        if all(
            value is not None and str(value).strip()
            for value in (objective, population, horizon)
        ):
            denominator = float(population) * float(horizon)
            if denominator <= 0.0:
                raise ValueError(
                    "initial_population and horizon_transitions must be positive"
                )
            return float(objective) / denominator
    return float(row[metric])


def _paired_analysis(rows: list[dict], launch_seed: int, draws: int) -> list[dict]:
    heuristic_by_scenario = {
        int(row["replication"]): row
        for row in rows
        if str(row["deployment_strategy"]) == "heuristic"
    }
    rl_by_key = {
        (int(row.get("policy_replication", 1)), int(row["replication"])): row
        for row in rows
        if str(row["deployment_strategy"]) == "rl"
    }
    policy_replications = sorted({key[0] for key in rl_by_key})
    replications = sorted(heuristic_by_scenario)
    if not policy_replications or not replications:
        raise RuntimeError("No paired RL/heuristic replications were found")
    missing = [
        (policy_replication, replication)
        for policy_replication in policy_replications
        for replication in replications
        if (policy_replication, replication) not in rl_by_key
    ]
    if missing:
        raise RuntimeError(f"Incomplete policy-by-scenario evaluation matrix: {missing[:3]}")

    # Positive improvement always means RL is better.
    output = []
    inferentially_eligible = len(policy_replications) >= 2 and len(replications) >= 2
    for metric_index, (metric, direction) in enumerate(EVALUATION_METRICS.items()):
        rl_values = np.asarray(
            [
                [
                    _evaluation_metric(
                        rl_by_key[(policy_replication, replication)], metric
                    )
                    for replication in replications
                ]
                for policy_replication in policy_replications
            ],
            dtype=float,
        )
        heuristic_values = np.asarray(
            [
                _evaluation_metric(heuristic_by_scenario[replication], metric)
                for replication in replications
            ],
            dtype=float,
        )
        raw_difference = rl_values - heuristic_values[None, :]
        improvement = raw_difference if direction == "higher" else -raw_difference
        rng = np.random.default_rng(_seed(launch_seed, 91, metric_index))
        policy_indices = rng.integers(
            0,
            len(policy_replications),
            size=(int(draws), len(policy_replications)),
        )
        scenario_indices = rng.integers(
            0,
            len(replications),
            size=(int(draws), len(replications)),
        )
        bootstrap_means = np.empty(int(draws), dtype=float)
        for draw in range(int(draws)):
            bootstrap_means[draw] = improvement[
                np.ix_(policy_indices[draw], scenario_indices[draw])
            ].mean()
        ci_low, ci_high = (
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        )
        policy_mean_improvements = improvement.mean(axis=1)
        p_value = _paired_randomization_pvalue(policy_mean_improvements, rng, draws)
        standard_deviation = (
            float(policy_mean_improvements.std(ddof=1))
            if len(policy_replications) > 1
            else 0.0
        )
        standard_error = standard_deviation / math.sqrt(len(policy_replications))
        output.append(
            {
                "metric": metric,
                "preferred_direction": direction,
                "policy_replications": len(policy_replications),
                "scenario_replications": len(replications),
                "inferentially_eligible": inferentially_eligible,
                "paired_replications": int(improvement.size),
                "rl_mean": float(rl_values.mean()),
                "heuristic_mean": float(heuristic_values.mean()),
                "mean_rl_improvement": float(improvement.mean()),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "paired_standard_error": standard_error,
                "paired_effect_size_dz": (
                    float(improvement.mean()) / standard_deviation
                    if standard_deviation > 0.0
                    else 0.0
                ),
                "two_sided_randomization_p": p_value,
                "rl_win_rate": float(np.mean(improvement > 0.0)),
                "tie_rate": float(np.mean(improvement == 0.0)),
                "superiority_ci_excludes_zero": bool(
                    inferentially_eligible and ci_low > 0.0
                ),
            }
        )
    return output


def _benchmark_analysis(
    rows: list[dict],
    launch_seed: int,
    draws: int,
    benchmark_strategies: Iterable[str] | None = None,
) -> list[dict]:
    """Estimate paired RL improvements against every registered comparator.

    Non-learned benchmarks contribute one outcome per held-out scenario and
    are broadcast across independently trained RL policy seeds. Learned timing
    controls such as ``rl_precommit`` are instead paired by checkpoint seed.
    Policy seeds and scenarios are resampled separately, matching the primary
    analysis without promoting these diagnostic contrasts to confirmatory
    status.
    """
    rl_by_key = {
        (int(row.get("policy_replication", 1)), int(row["replication"])): row
        for row in rows
        if str(row["deployment_strategy"]) == "rl"
    }
    policy_replications = sorted({key[0] for key in rl_by_key})
    replications = sorted({key[1] for key in rl_by_key})
    if not policy_replications or not replications:
        raise RuntimeError("No RL policy-by-scenario evaluation matrix was found")
    missing_rl = [
        (policy, replication)
        for policy in policy_replications
        for replication in replications
        if (policy, replication) not in rl_by_key
    ]
    if missing_rl:
        raise RuntimeError(f"Incomplete RL policy-by-scenario matrix: {missing_rl[:3]}")

    observed = {
        str(row["deployment_strategy"])
        for row in rows
        if str(row["deployment_strategy"]) != "rl"
    }
    requested = (
        tuple(str(strategy) for strategy in benchmark_strategies)
        if benchmark_strategies is not None
        else tuple(sorted(observed))
    )
    missing_strategies = sorted(set(requested).difference(observed))
    if missing_strategies:
        raise RuntimeError(
            f"No evaluation rows were found for benchmarks: {missing_strategies}"
        )

    output = []
    inferentially_eligible = len(policy_replications) >= 2 and len(replications) >= 2
    for benchmark_index, benchmark_strategy in enumerate(requested):
        benchmark_rows = [
            row
            for row in rows
            if str(row["deployment_strategy"]) == benchmark_strategy
        ]
        counts = {}
        for row in benchmark_rows:
            counts[int(row["replication"])] = counts.get(int(row["replication"]), 0) + 1
        broadcast = bool(
            set(counts) == set(replications)
            and all(counts[replication] == 1 for replication in replications)
        )
        if broadcast:
            benchmark_by_scenario = {
                int(row["replication"]): row for row in benchmark_rows
            }
            benchmark_by_key = None
        else:
            benchmark_by_scenario = None
            benchmark_by_key = {
                (int(row.get("policy_replication", 1)), int(row["replication"])): row
                for row in benchmark_rows
            }
            missing_benchmark = [
                (policy, replication)
                for policy in policy_replications
                for replication in replications
                if (policy, replication) not in benchmark_by_key
            ]
            if missing_benchmark:
                raise RuntimeError(
                    f"Incomplete {benchmark_strategy} policy-by-scenario matrix: "
                    f"{missing_benchmark[:3]}"
                )

        for metric_index, (metric, direction) in enumerate(
            EVALUATION_METRICS.items()
        ):
            rl_values = np.asarray(
                [
                    [
                        _evaluation_metric(rl_by_key[(policy, replication)], metric)
                        for replication in replications
                    ]
                    for policy in policy_replications
                ],
                dtype=float,
            )
            if broadcast:
                benchmark_values = np.asarray(
                    [
                        _evaluation_metric(
                            benchmark_by_scenario[replication], metric
                        )
                        for replication in replications
                    ],
                    dtype=float,
                )[None, :]
            else:
                benchmark_values = np.asarray(
                    [
                        [
                            _evaluation_metric(
                                benchmark_by_key[(policy, replication)], metric
                            )
                            for replication in replications
                        ]
                        for policy in policy_replications
                    ],
                    dtype=float,
                )
            raw_difference = rl_values - benchmark_values
            improvement = (
                raw_difference if direction == "higher" else -raw_difference
            )
            rng = np.random.default_rng(
                _seed(launch_seed, 291 + benchmark_index, metric_index)
            )
            policy_indices = rng.integers(
                0,
                len(policy_replications),
                size=(int(draws), len(policy_replications)),
            )
            scenario_indices = rng.integers(
                0,
                len(replications),
                size=(int(draws), len(replications)),
            )
            bootstrap_means = np.empty(int(draws), dtype=float)
            for draw in range(int(draws)):
                bootstrap_means[draw] = improvement[
                    np.ix_(policy_indices[draw], scenario_indices[draw])
                ].mean()
            ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))
            policy_mean_improvements = improvement.mean(axis=1)
            standard_deviation = (
                float(policy_mean_improvements.std(ddof=1))
                if len(policy_replications) > 1
                else 0.0
            )
            output.append(
                {
                    "comparison_role": (
                        "primary"
                        if benchmark_strategy == "heuristic"
                        else "secondary_diagnostic"
                    ),
                    "benchmark_strategy": benchmark_strategy,
                    "metric": metric,
                    "preferred_direction": direction,
                    "policy_replications": len(policy_replications),
                    "scenario_replications": len(replications),
                    "inferentially_eligible": inferentially_eligible,
                    "paired_replications": int(improvement.size),
                    "rl_mean": float(rl_values.mean()),
                    "benchmark_mean": float(benchmark_values.mean()),
                    "mean_rl_improvement": float(improvement.mean()),
                    "bootstrap_95_ci_low": float(ci_low),
                    "bootstrap_95_ci_high": float(ci_high),
                    "paired_standard_error": (
                        standard_deviation / math.sqrt(len(policy_replications))
                    ),
                    "paired_effect_size_dz": (
                        float(improvement.mean()) / standard_deviation
                        if standard_deviation > 0.0
                        else 0.0
                    ),
                    "two_sided_randomization_p": _paired_randomization_pvalue(
                        policy_mean_improvements, rng, draws
                    ),
                    "rl_win_rate": float(np.mean(improvement > 0.0)),
                    "tie_rate": float(np.mean(improvement == 0.0)),
                    "superiority_ci_excludes_zero": bool(
                        inferentially_eligible and ci_low > 0.0
                    ),
                }
            )
    return output


def _training_convergence(
    rows: list[dict],
    *,
    minimum_episodes: int,
    window_fraction: float,
    trend_threshold: float,
    shift_threshold: float,
    target_kl: float,
) -> dict:
    """Audit numerical stability and tail stationarity for every policy seed."""
    by_policy = {}
    for row in rows:
        by_policy.setdefault(int(row.get("policy_replication", 1)), []).append(row)
    policies = []
    required_fields = (
        "episode_return",
        "entropy",
        "approximate_kl",
        "gradient_norm",
        "policy_loss",
        "value_loss",
    )
    for policy_replication, policy_rows in sorted(by_policy.items()):
        policy_rows.sort(key=lambda row: int(row["replication"]))
        numeric = {
            field: np.asarray([float(row[field]) for row in policy_rows], dtype=float)
            for field in required_fields
        }
        finite = bool(all(np.isfinite(values).all() for values in numeric.values()))
        episodes = len(policy_rows)
        window = max(2, int(math.ceil(float(window_fraction) * episodes)))
        window = min(window, episodes // 2) if episodes >= 4 else 0
        enough = episodes >= int(minimum_episodes) and window >= 2

        trend_span_sd = float("inf")
        window_shift_sd = float("inf")
        tail_return_mean = float("nan")
        tail_return_std = float("nan")
        kl_violation_rate = float("inf")
        if finite and window >= 2:
            returns = numeric["episode_return"]
            comparison = returns[-2 * window:]
            previous = comparison[:window]
            tail = comparison[window:]
            tail_return_mean = float(tail.mean())
            tail_return_std = float(tail.std(ddof=1))
            scale = float(comparison.std(ddof=1))
            x = np.arange(window, dtype=float)
            slope = float(np.polyfit(x, tail, deg=1)[0])
            if scale <= 1e-12:
                trend_span_sd = 0.0 if np.allclose(tail, tail[0]) else float("inf")
                window_shift_sd = 0.0 if np.allclose(previous.mean(), tail.mean()) else float("inf")
            else:
                trend_span_sd = abs(slope) * float(window - 1) / scale
                window_shift_sd = abs(float(tail.mean() - previous.mean())) / scale
            tail_rows = policy_rows[-window:]
            update_kl = np.asarray(
                [
                    float(row["approximate_kl"])
                    for row in tail_rows
                    if float(row.get("optimizer_updated", 1.0)) > 0.5
                ],
                dtype=float,
            )
            kl_violation_rate = (
                float(np.mean(update_kl > target_kl))
                if update_kl.size
                else float("inf")
            )

        rollout_complete = bool(
            not policy_rows
            or float(policy_rows[-1].get("rollout_episodes_pending", 0.0)) == 0.0
        )
        optimization_events = sum(
            float(row.get("optimizer_updated", 1.0)) > 0.5
            for row in policy_rows
        )
        fitted_pi = any("nmcc_pi_replay_episodes" in row for row in policy_rows)
        validation_rows = [
            row for row in policy_rows
            if np.isfinite(float(row.get("nmcc_pi_validation_gain_lower", float("nan"))))
        ]
        heldout_gain_lower = (
            float(validation_rows[-1]["nmcc_pi_validation_gain_lower"])
            if validation_rows else float("nan")
        )
        heldout_states = (
            int(float(validation_rows[-1].get("nmcc_pi_validation_states", 0)))
            if validation_rows else 0
        )
        heldout_gate = bool(
            validation_rows
            and float(validation_rows[-1].get("nmcc_pi_gate_passed", 0.0)) > 0.5
            and heldout_gain_lower > 0.0
        )

        if fitted_pi:
            # Fitted policy improvement is selected by exact held-out paired
            # gain, not by stationarity of noisy, unpaired training returns or
            # PPO's unrelated 0.015 KL cap.
            converged = bool(
                enough and finite and rollout_complete
                and optimization_events > 0 and heldout_states > 0 and heldout_gate
            )
        else:
            converged = bool(
                enough
                and finite
                and rollout_complete
                and optimization_events > 0
                and trend_span_sd <= float(trend_threshold)
                and window_shift_sd <= float(shift_threshold)
                and kl_violation_rate <= 0.10
            )
        policies.append(
            {
                "policy_replication": int(policy_replication),
                "episodes": int(episodes),
                "window_episodes": int(window),
                "all_diagnostics_finite": finite,
                "rollout_complete": rollout_complete,
                "optimization_events": int(optimization_events),
                "tail_return_mean": tail_return_mean,
                "tail_return_standard_deviation": tail_return_std,
                "tail_trend_span_standard_deviations": trend_span_sd,
                "adjacent_window_shift_standard_deviations": window_shift_sd,
                "tail_kl_violation_rate": kl_violation_rate,
                "convergence_mode": (
                    "episode_heldout_paired_gain" if fitted_pi else "ppo_tail_stationarity"
                ),
                "heldout_paired_gain_lower_bound": heldout_gain_lower,
                "heldout_validation_states": heldout_states,
                "heldout_controller_gate_passed": heldout_gate,
                "tail_entropy_mean": (
                    float(numeric["entropy"][-window:].mean())
                    if finite and window >= 2
                    else float("nan")
                ),
                "converged": converged,
            }
        )
    return {
        "definition": (
            "PPO policies require tail stationarity and KL compliance. Fitted NMCC-PI "
            "policies instead require finite complete training and a positive exact "
            "episode-heldout paired-gain lower bound that opens the controller gate."
        ),
        "thresholds": {
            "minimum_episodes": int(minimum_episodes),
            "window_fraction": float(window_fraction),
            "tail_trend_span_standard_deviations": float(trend_threshold),
            "adjacent_window_shift_standard_deviations": float(shift_threshold),
            "target_kl": float(target_kl),
            "maximum_tail_kl_violation_rate": 0.10,
        },
        "policies": policies,
        "all_policies_converged": bool(policies and all(row["converged"] for row in policies)),
    }


def _performance_assessment(
    analysis: list[dict],
    *,
    allow_fixed_policy: bool = False,
) -> dict:
    primary = next((row for row in analysis if row["metric"] == "episode_return"), None)
    if primary is None:
        return {"status": "unavailable", "primary_metric": "episode_return"}
    policy_replications = int(primary.get("policy_replications", 0))
    scenario_replications = int(primary.get("scenario_replications", 0))
    scenarios_per_city = int(
        primary.get("scenarios_per_city", scenario_replications)
    )
    fixed_policy_design = bool(allow_fixed_policy and policy_replications == 1)
    eligible = scenarios_per_city >= 2 and (
        policy_replications >= 2 or fixed_policy_design
    )
    lower = float(primary["bootstrap_95_ci_low"])
    upper = float(primary["bootstrap_95_ci_high"])
    if not eligible:
        status = "descriptive_only_insufficient_replication"
    elif lower > 0.0:
        status = "rl_superior"
    elif upper < 0.0:
        status = "rl_inferior"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "primary_metric": "episode_return",
        "mean_rl_improvement": float(primary["mean_rl_improvement"]),
        "bootstrap_95_ci": [lower, upper],
        "policy_replications": policy_replications,
        "scenario_replications": scenario_replications,
        "scenarios_per_city": scenarios_per_city,
        "inferentially_eligible": eligible,
        "inference_scope": (
            "conditional_on_one_fixed_trained_policy"
            if fixed_policy_design
            else "joint_over_policy_and_scenario_seeds"
        ),
        "criterion": (
            (
                "For the preregistered single-policy design, classification is "
                "conditional on the frozen trained checkpoint and requires at "
                "least two held-out scenarios per city; superiority requires the "
                "paired scenario-bootstrap interval to lie above zero."
            )
            if fixed_policy_design
            else (
                "Inferential classification requires at least two policy seeds and "
                "two held-out scenarios; RL superiority then requires the paired "
                "two-way-bootstrap interval to lie above zero."
            )
        ),
    }


def _verify_matched_interface(rows: list[dict], strategies: tuple[str, ...]) -> dict:
    dynamic = tuple(strategy for strategy in strategies if strategy in DYNAMIC_STRATEGIES)
    benchmark_by_replication = {
        int(row["replication"]): row
        for row in rows
        if row["deployment_strategy"] == "heuristic"
    }
    mismatches = []
    compared = 0
    for row in rows:
        strategy = row["deployment_strategy"]
        if strategy not in dynamic or strategy == "heuristic":
            continue
        replication = int(row["replication"])
        reference = benchmark_by_replication.get(replication)
        if reference is None:
            mismatches.append({"replication": replication, "reason": "missing heuristic"})
            continue
        compared += 1
        required_provenance_present = all(
            (
                isinstance(row.get("random_stream_seeds"), dict),
                isinstance(reference.get("random_stream_seeds"), dict),
                bool(row.get("hazard_trajectory_digest")),
                bool(reference.get("hazard_trajectory_digest")),
            )
        )
        if any(
            (
                not required_provenance_present,
                int(row["scenario_seed"]) != int(reference["scenario_seed"]),
                row["initial_observation_digest"] != reference["initial_observation_digest"],
                int(row["maximum_dynamic_deployments"])
                != int(reference["maximum_dynamic_deployments"]),
                (
                    "deployments_made" in row
                    and "deployments_made" in reference
                    and int(row["deployments_made"])
                    != int(reference["deployments_made"])
                ),
                (
                    "total_shelter_capacity" in row
                    and "total_shelter_capacity" in reference
                    and not np.isclose(
                        float(row["total_shelter_capacity"]),
                        float(reference["total_shelter_capacity"]),
                        rtol=0.0,
                        atol=1e-9,
                    )
                ),
                row.get("random_stream_seeds") != reference.get("random_stream_seeds"),
                row.get("hazard_trajectory_digest")
                != reference.get("hazard_trajectory_digest"),
                row.get("city_id") != reference.get("city_id"),
                row.get("map_spec") != reference.get("map_spec"),
                row.get("cell_partition") != reference.get("cell_partition"),
            )
        ):
            mismatches.append(
                {
                    "replication": replication,
                    "strategy": strategy,
                    "policy_replication": int(row.get("policy_replication", 0)),
                    "reason": (
                        "city/map, scenario/component seed, initial observation, realized "
                        "deployment count/capacity, or budget mismatch"
                    ),
                }
            )
    if mismatches:
        raise RuntimeError(f"Matched-interface verification failed: {mismatches[:3]}")
    return {
        "verified": True,
        "scenario_replications": len(benchmark_by_replication),
        "policy_scenario_comparisons": compared,
        "strategies": list(dynamic),
        "checks": [
            "same scenario seed",
            "byte-identical first regional observation and feasible-action mask",
            "same maximum dynamic deployment budget",
            "same realized dynamic deployment count and total installed capacity",
            "same independent hazard and pedestrian-outcome random streams",
            "byte-identical full exogenous hazard trajectory",
            "same RegionalObservation schema",
            "same exact-candidate action table and feasibility mask",
            "same direct candidate-site execution contract",
            "same city identifier and OSM query specification",
            "same cell-partition mode and exact partition-edge digest",
        ],
    }


def _write_paper_table(path: str, analysis: list[dict]) -> None:
    inferentially_eligible = bool(
        analysis and all(row.get("inferentially_eligible", True) for row in analysis)
    )
    fixed_policy = bool(
        analysis
        and all(
            row.get("inference_scope")
            == "conditional_on_one_fixed_trained_policy"
            for row in analysis
        )
    )
    interpretation = (
        "Positive improvement favors RL. Confidence intervals condition on the one "
        "frozen policy and resample held-out scenarios within each fixed city; a "
        "policy-seed randomization p-value is not defined."
        if fixed_policy
        else (
            "Positive improvement favors RL. Confidence intervals are percentile paired "
            "bootstrap intervals; p-values use paired sign randomization."
            if inferentially_eligible
            else "Positive improvement favors RL. This pilot lacks the minimum two policy "
            "seeds and two held-out scenarios per analysis stratum; its resampling ranges "
            "and p-values are descriptive and cannot establish superiority."
        )
    )
    lines = [
        "# Paired RL versus active-population heuristic",
        "",
        interpretation,
        "`episode_return` denotes the action-count-invariant full-episode policy objective.",
        "",
        "| Metric | RL mean | Heuristic mean | RL improvement | 95% CI | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis:
        p_value = row.get("two_sided_randomization_p")
        p_text = "NA" if p_value is None else f"{float(p_value):.4g}"
        lines.append(
            "| {metric} | {rl_mean:.5g} | {heuristic_mean:.5g} | {mean_rl_improvement:.5g} | "
            "[{bootstrap_95_ci_low:.5g}, {bootstrap_95_ci_high:.5g}] | {p_text} |".format(
                **row,
                p_text=p_text,
            )
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_benchmark_table(path: str, analysis: list[dict]) -> None:
    inferentially_eligible = bool(
        analysis and all(row.get("inferentially_eligible", False) for row in analysis)
    )
    interpretation = (
        "Intervals separately resample policy seeds and matched scenarios. The "
        "maximum-active-population contrast is primary; all others are secondary."
        if inferentially_eligible
        else "This engineering backtest is descriptive because it lacks at least two "
        "independent policy seeds and two held-out scenarios. The maximum-active-"
        "population contrast remains primary; all others are secondary."
    )
    lines = [
        "# Paired RL versus registered benchmark policies",
        "",
        interpretation,
        "Positive improvement favors RL. `episode_return` is the action-count-invariant full-episode objective.",
        "",
        "| Role | Benchmark | Metric | RL mean | Benchmark mean | RL improvement | 95% CI | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis:
        lines.append(
            "| {comparison_role} | {benchmark_strategy} | {metric} | {rl_mean:.5g} | "
            "{benchmark_mean:.5g} | {mean_rl_improvement:.5g} | "
            "[{bootstrap_95_ci_low:.5g}, {bootstrap_95_ci_high:.5g}] | "
            "{two_sided_randomization_p:.4g} |".format(**row)
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _plot_outputs(
    launch_dir: str,
    training: list[dict],
    analysis: list[dict],
    benchmark_analysis: list[dict] | None = None,
) -> list[str]:
    try:
        import matplotlib

        # The campaign runner imports PyTorch before finalizing figures.  On
        # headless macOS workers, deferring backend selection until pyplot is
        # imported can select the interactive MacOSX backend and deadlock the
        # completed run during figure construction.  These are file artifacts,
        # so force the non-interactive backend before pyplot is imported.
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        with open(os.path.join(launch_dir, "plotting_unavailable.txt"), "w", encoding="utf-8") as handle:
            handle.write(f"Matplotlib unavailable: {exc}\n")
        return []

    outputs = []
    if training:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        by_policy = {}
        for row in training:
            by_policy.setdefault(int(row.get("policy_replication", 1)), []).append(row)
        aligned_returns = []
        aligned_episodes = None
        for policy_replication, policy_rows in sorted(by_policy.items()):
            policy_rows.sort(key=lambda row: int(row["replication"]))
            episodes = np.asarray([row["replication"] for row in policy_rows], dtype=float)
            returns = np.asarray([row["episode_return"] for row in policy_rows], dtype=float)
            ax.plot(
                episodes,
                returns,
                alpha=0.22,
                linewidth=0.8,
                label=f"policy seed {policy_replication}",
            )
            if aligned_episodes is None:
                aligned_episodes = episodes
            if np.array_equal(episodes, aligned_episodes):
                aligned_returns.append(returns)
        if aligned_returns:
            mean_return = np.stack(aligned_returns).mean(axis=0)
            window = max(5, min(25, len(mean_return) // 10 if len(mean_return) >= 10 else 5))
            kernel = np.ones(window, dtype=float) / window
            moving = (
                np.convolve(mean_return, kernel, mode="valid")
                if len(mean_return) >= window
                else mean_return
            )
            moving_x = (
                aligned_episodes[window - 1:]
                if len(mean_return) >= window
                else aligned_episodes
            )
            ax.plot(moving_x, moving, color="black", linewidth=2.4, label=f"cross-seed moving mean ({window})")
        ax.set(xlabel="Training episode", ylabel="Undiscounted return", title="RL training convergence")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(launch_dir, "training_convergence.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

        # Decompose the simple reward and show the diagnostics that can reveal
        # apparent reward stationarity caused by policy collapse or oversized
        # PPO updates.  Lines are cross-policy means when episode indices align.
        fields = (
            ("safe_completion_reward", "Safe-completion term"),
            ("casualty_penalty", "Casualty term"),
            ("risk_time_penalty", "Risk-time term"),
        )
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
        series_by_field = {field: [] for field, _ in fields}
        entropy_series = []
        kl_series = []
        common_episodes = None
        for policy_rows in by_policy.values():
            policy_rows.sort(key=lambda row: int(row["replication"]))
            episodes = np.asarray([float(row["replication"]) for row in policy_rows])
            if common_episodes is None:
                common_episodes = episodes
            if not np.array_equal(episodes, common_episodes):
                continue
            for field, _ in fields:
                series_by_field[field].append(
                    np.asarray([float(row[field]) for row in policy_rows])
                )
            entropy_series.append(np.asarray([float(row["entropy"]) for row in policy_rows]))
            kl_series.append(
                np.asarray(
                    [
                        float(row["approximate_kl"])
                        if float(row.get("optimizer_updated", 1.0)) > 0.5
                        else np.nan
                        for row in policy_rows
                    ]
                )
            )

        def moving_mean(values):
            if values.size < 5:
                return common_episodes, values
            window = max(5, min(25, values.size // 10))
            kernel = np.ones(window, dtype=float) / float(window)
            return common_episodes[window - 1:], np.convolve(values, kernel, mode="valid")

        reward_ax = axes[0, 0]
        for field, label in fields:
            if series_by_field[field]:
                values = np.stack(series_by_field[field]).mean(axis=0)
                x_values, y_values = moving_mean(values)
                reward_ax.plot(x_values, y_values, label=label)
        reward_ax.set(title="Reward decomposition", ylabel="Normalized reward")
        reward_ax.legend(fontsize=8)

        entropy_ax = axes[0, 1]
        if entropy_series:
            x_values, y_values = moving_mean(np.stack(entropy_series).mean(axis=0))
            entropy_ax.plot(x_values, y_values, color="tab:purple")
        entropy_ax.set(title="Policy entropy", ylabel="Entropy")

        kl_ax = axes[1, 0]
        if kl_series:
            stacked_kl = np.stack(kl_series)
            valid = np.isfinite(stacked_kl).any(axis=0)
            values = np.full(stacked_kl.shape[1], np.nan, dtype=float)
            values[valid] = np.nanmean(stacked_kl[:, valid], axis=0)
            kl_ax.plot(
                common_episodes[valid],
                values[valid],
                color="tab:orange",
                marker="o",
                markersize=3,
            )
        kl_ax.axhline(
            DEFAULT_TARGET_KL,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="target KL",
        )
        kl_ax.set(title="PPO update size", xlabel="Training episode", ylabel="Approximate KL")
        kl_ax.legend(fontsize=8)

        agreement_ax = axes[1, 1]
        agreement_series = [
            np.asarray([float(row["heuristic_agreement_rate"]) for row in rows])
            for rows in by_policy.values()
            if common_episodes is not None and len(rows) == len(common_episodes)
        ]
        if agreement_series:
            x_values, y_values = moving_mean(np.stack(agreement_series).mean(axis=0))
            agreement_ax.plot(x_values, y_values, color="tab:green")
        agreement_ax.set(
            title="Behavioral overlap with benchmark",
            xlabel="Training episode",
            ylabel="Action agreement rate",
            ylim=(-0.02, 1.02),
        )
        for diagnostic_ax in axes.flat:
            diagnostic_ax.grid(alpha=0.25)
        fig.tight_layout()
        diagnostic_path = os.path.join(launch_dir, "training_diagnostics.png")
        fig.savefig(diagnostic_path, dpi=180)
        plt.close(fig)
        outputs.append(diagnostic_path)

    if not analysis:
        return outputs

    labels = [row["metric"] for row in analysis]
    estimates = np.asarray([row["mean_rl_improvement"] for row in analysis])
    lower = np.asarray([row["bootstrap_95_ci_low"] for row in analysis])
    upper = np.asarray([row["bootstrap_95_ci_high"] for row in analysis])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    ax.errorbar(estimates, y, xerr=np.vstack((estimates - lower, upper - estimates)), fmt="o")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(y, labels=labels)
    ax.set(xlabel="Mean paired improvement (positive favors RL)", title="RL versus heuristic: paired 95% intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(launch_dir, "paired_policy_comparison.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    if benchmark_analysis:
        metric_order = tuple(EVALUATION_METRICS)
        benchmark_order = [
            strategy
            for strategy in (
                "risk_reduction",
                "route_saving",
                "heuristic",
                "hazard_weighted",
                "accessibility_deficit",
                "random",
                "static_greedy",
                "rl_precommit",
                "initial_only",
            )
            if any(
                row["benchmark_strategy"] == strategy
                for row in benchmark_analysis
            )
        ]
        benchmark_labels = {
            "risk_reduction": "Future risk-time reduction",
            "route_saving": "Demand-weighted route saving",
            "heuristic": "Max active demand",
            "hazard_weighted": "Hazard-weighted demand",
            "accessibility_deficit": "Accessibility deficit",
            "random": "Random feasible candidate",
            "static_greedy": "Static demand greedy",
            "rl_precommit": "RL precommit",
            "initial_only": "Static initial-only",
        }
        metric_labels = {
            "episode_return": "Full-episode objective",
            "safe_completed": "Safe completions",
            "casualty": "Casualty reduction",
            "unfinished": "Unfinished reduction",
            "restricted_mean_time_to_safety": "Time-to-safety reduction",
            "normalized_risk_weighted_person_time": "Risk-time reduction",
        }
        lookup = {
            (row["benchmark_strategy"], row["metric"]): row
            for row in benchmark_analysis
        }
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
        y = np.arange(len(benchmark_order))
        for metric, axis in zip(metric_order, axes.flat):
            subset = [lookup[(strategy, metric)] for strategy in benchmark_order]
            estimates = np.asarray(
                [row["mean_rl_improvement"] for row in subset], dtype=float
            )
            lower = np.asarray(
                [row["bootstrap_95_ci_low"] for row in subset], dtype=float
            )
            upper = np.asarray(
                [row["bootstrap_95_ci_high"] for row in subset], dtype=float
            )
            axis.errorbar(
                estimates,
                y,
                xerr=np.vstack((estimates - lower, upper - estimates)),
                fmt="o",
                capsize=3,
            )
            axis.axvline(0.0, color="black", linewidth=1.0)
            axis.set_yticks(
                y,
                labels=[benchmark_labels.get(value, value) for value in benchmark_order],
            )
            axis.invert_yaxis()
            axis.set(title=metric_labels.get(metric, metric), xlabel="Paired RL improvement")
            axis.grid(axis="x", alpha=0.25)
        fig.suptitle("RL versus registered benchmark policies")
        fig.text(
            0.5,
            0.01,
            "Positive values favor RL; initial-only is an anticipative timing control.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        benchmark_path = os.path.join(
            launch_dir, "all_benchmark_comparisons.png"
        )
        fig.savefig(benchmark_path, dpi=180)
        plt.close(fig)
        outputs.append(benchmark_path)
    return outputs


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-seed", type=int, default=20260902)
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--machine", default="local")
    parser.add_argument("--policy-replicates", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=320)
    parser.add_argument("--eval-replications", type=int, default=50)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument(
        "--require-convergence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop before held-out evaluation if any trained policy fails the preregistered convergence audit.",
    )
    parser.add_argument("--convergence-min-episodes", type=int, default=100)
    parser.add_argument("--convergence-window-fraction", type=float, default=0.20)
    parser.add_argument("--convergence-trend-threshold", type=float, default=0.50)
    parser.add_argument("--convergence-shift-threshold", type=float, default=0.50)
    parser.add_argument(
        "--strategies",
        default="rl,risk_reduction,heuristic,hazard_weighted,accessibility_deficit,random",
    )
    parser.add_argument("--hazard-mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument(
        "--visualize-eval-pairs",
        type=int,
        default=1,
        help="Render the first N held-out scenario pairs (0 disables rendering).",
    )
    parser.add_argument(
        "--visualization-strategies",
        default="rl,heuristic",
        help="Comma-separated evaluated strategies to render.",
    )
    parser.add_argument(
        "--visualization-policy-replication",
        type=int,
        default=1,
        help="Trained RL policy replicate used for milestone maps.",
    )
    parser.add_argument(
        "--visualization-milestones",
        default="quartiles",
        help="'quartiles' or comma-separated simulator times; endpoints are always included.",
    )
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_only and args.train_only:
        parser.error("--eval-only and --train-only are mutually exclusive")
    if (
        args.policy_replicates <= 0
        or args.train_episodes <= 0
        or args.eval_replications <= 0
        or args.bootstrap_draws <= 0
        or args.convergence_min_episodes <= 0
    ):
        parser.error("episode, replication, and bootstrap counts must be positive")
    if not 0.0 < args.convergence_window_fraction <= 0.5:
        parser.error("--convergence-window-fraction must be in (0, 0.5]")
    if args.convergence_trend_threshold < 0.0 or args.convergence_shift_threshold < 0.0:
        parser.error("convergence thresholds must be non-negative")
    if args.visualize_eval_pairs < 0:
        parser.error("--visualize-eval-pairs must be non-negative")
    if args.train_episodes % DEFAULT_ROLLOUT_EPISODES != 0:
        parser.error(
            f"--train-episodes must be a multiple of the fixed "
            f"{DEFAULT_ROLLOUT_EPISODES}-episode on-policy rollout"
        )
    if not 1 <= args.visualization_policy_replication <= args.policy_replicates:
        parser.error("--visualization-policy-replication must identify a trained policy replicate")
    strategies = tuple(item.strip().lower() for item in args.strategies.split(",") if item.strip())
    allowed = set(DYNAMIC_STRATEGIES).union(STATIC_STRATEGIES)
    invalid = sorted(set(strategies).difference(allowed))
    if invalid:
        parser.error(f"unsupported strategies: {invalid}")
    if not args.train_only and not {"rl", "heuristic"}.issubset(strategies):
        parser.error("evaluation must include both rl and heuristic for the primary paired test")
    args.strategies = strategies
    visualization_strategies = tuple(
        item.strip().lower()
        for item in args.visualization_strategies.split(",")
        if item.strip()
    )
    invalid_visualization = sorted(set(visualization_strategies).difference(allowed))
    if invalid_visualization:
        parser.error(f"unsupported visualization strategies: {invalid_visualization}")
    missing_visualization = sorted(set(visualization_strategies).difference(strategies))
    if not args.train_only and args.visualize_eval_pairs > 0 and missing_visualization:
        parser.error(
            "visualization strategies must also be present in --strategies: "
            f"{missing_visualization}"
        )
    if args.train_only:
        args.visualize_eval_pairs = 0
    args.visualization_strategies = visualization_strategies
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    launch_id = args.launch_id or f"regional_backtest_seed_{args.launch_seed}"
    launch_dir = os.path.join(RUNS_ROOT, launch_id)
    training_summary_path = os.path.join(launch_dir, "training_episode_summary.csv")
    evaluation_summary_path = os.path.join(launch_dir, "evaluation_episode_summary.csv")
    checkpoint_paths = {
        policy_replication: os.path.join(
            launch_dir,
            "policies",
            f"policy_{policy_replication:03d}",
            "regional_policy.pt",
        )
        for policy_replication in range(1, args.policy_replicates + 1)
    }
    diagnostics_paths = {
        policy_replication: os.path.join(
            launch_dir,
            "policies",
            f"policy_{policy_replication:03d}",
            "ppo_diagnostics.csv",
        )
        for policy_replication in range(1, args.policy_replicates + 1)
    }
    os.makedirs(launch_dir, exist_ok=True)

    existing_checkpoints = [path for path in checkpoint_paths.values() if os.path.exists(path)]
    if not args.eval_only and existing_checkpoints and not args.resume:
        raise FileExistsError(
            f"Checkpoint already exists at {existing_checkpoints[0]}; choose a new --launch-id or pass --resume"
        )
    missing_checkpoints = [path for path in checkpoint_paths.values() if not os.path.exists(path)]
    if args.eval_only and missing_checkpoints:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {missing_checkpoints[0]}")
    if args.eval_only and os.path.exists(evaluation_summary_path):
        raise FileExistsError(
            f"Evaluation already exists at {evaluation_summary_path}; use a new --launch-id "
            "to preserve provenance"
        )
    if args.resume and not args.train_only and os.path.exists(evaluation_summary_path):
        raise FileExistsError(
            "This launch already has evaluation results. Continue training with --train-only, "
            "then use --eval-only after choosing whether to replace the prior evaluation."
        )

    overrides = _parse_overrides(args.override)
    overrides["hazardEvolutionMode"] = args.hazard_mode
    manifest_path = os.path.join(launch_dir, "experiment_manifest.json")
    manifest = {
        "schema_version": 3,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, os.path.abspath(__file__), *(argv or sys.argv[1:])],
        "launch_id": launch_id,
        "launch_seed": int(args.launch_seed),
        "policy_replicates": int(args.policy_replicates),
        "train_episodes": int(args.train_episodes),
        "eval_replications": int(args.eval_replications),
        "strategies": list(args.strategies),
        "benchmark_policy_contracts": {
            strategy: BENCHMARK_POLICY_CONTRACTS[strategy]
            for strategy in args.strategies
            if strategy in BENCHMARK_POLICY_CONTRACTS
        },
        "visualization": {
            "eval_pairs": int(args.visualize_eval_pairs),
            "strategies": list(args.visualization_strategies),
            "rl_policy_replication": int(args.visualization_policy_replication),
            "milestones": str(args.visualization_milestones),
            "basemap": "the episode's OpenStreetMap road graph",
            "non_interventional": True,
        },
        "overrides": overrides,
        "reward_equation": (
            "r_k=(Delta safe_k-3 Delta casualty_k)/P"
            "-active_person_time_k/(P H)"
            "-hazard_exposure_person_time_k/(P H)"
        ),
        "evaluation_objective_equation": (
            "R=(safe_completed-3 casualty)/P"
            "-full_episode_active_person_time/(P H)"
            "-full_episode_hazard_exposure_person_time/(P H)"
        ),
        "ppo_design": {
            "rollout_episodes": int(DEFAULT_ROLLOUT_EPISODES),
            "advantage_normalization": "across the complete multi-episode on-policy rollout",
            "policy_architecture": (
                "bounded active-population-prior residual exact-candidate scorer "
                "with pedestrian, hazard, infrastructure, spatial, and route context"
            ),
            "heuristic_prior_logit_scale": float(HEURISTIC_PRIOR_SCALE),
            "residual_logit_bound": float(RESIDUAL_LOGIT_BOUND),
            "entropy_coefficient": float(DEFAULT_ENTROPY_COEF),
            "entropy_normalization": "log_feasible_action_count",
            "kl_control": "full_rollout_epoch_stop_and_adaptive_learning_rate",
        },
        "randomization_design": {
            "matched_scenario_seed": True,
            "hazard_stream_isolated_from_policy": True,
            "pedestrian_hazard_draw": "counter-based by scenario, timestep, and pedestrian id",
            "policy_rng_isolated_from_scenario": True,
        },
        "convergence_design": {
            "required_before_evaluation": bool(args.require_convergence),
            "minimum_episodes": int(args.convergence_min_episodes),
            "window_fraction": float(args.convergence_window_fraction),
            "trend_threshold_standard_deviations": float(args.convergence_trend_threshold),
            "shift_threshold_standard_deviations": float(args.convergence_shift_threshold),
            "target_kl": DEFAULT_TARGET_KL,
        },
        "cell_observation_features": list(CELL_FEATURE_NAMES),
        "global_observation_features": list(GLOBAL_FEATURE_NAMES),
        "action": "one exact feasible forecast-safe shelter candidate",
        "hidden_lower_level_optimizer": False,
        "git": _git_metadata(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "dependencies": _dependency_versions(),
    }
    _json_dump(manifest_path, manifest)

    training_rows = _read_csv(training_summary_path) if (args.resume or args.eval_only) else []
    if not args.eval_only:
        for policy_replication in range(1, args.policy_replicates + 1):
            training_policy_seed = _seed(args.launch_seed, 10, policy_replication)
            completed = max(
                (
                    int(row["replication"])
                    for row in training_rows
                    if int(row.get("policy_replication", 1)) == policy_replication
                ),
                default=0,
            )
            for episode in range(completed + 1, args.train_episodes + 1):
                scenario_seed = _seed(
                    args.launch_seed,
                    100 + policy_replication,
                    episode,
                )
                print(
                    f"[TRAIN] policy={policy_replication}/{args.policy_replicates} "
                    f"episode={episode}/{args.train_episodes} scenario_seed={scenario_seed}",
                    flush=True,
                )
                episode_overrides = dict(overrides)
                episode_overrides["finalizePpoRollout"] = bool(
                    episode == args.train_episodes
                )
                row = _run_episode(
                    replication=episode,
                    machine=args.machine,
                    phase=os.path.join(
                        launch_id,
                        "training",
                        f"policy_{policy_replication:03d}",
                    ),
                    strategy="rl",
                    train_mode=True,
                    scenario_seed=scenario_seed,
                    policy_seed=training_policy_seed,
                    checkpoint_path=checkpoint_paths[policy_replication],
                    diagnostics_path=diagnostics_paths[policy_replication],
                    overrides=episode_overrides,
                )
                row["policy_replication"] = policy_replication
                training_rows.append(row)
                _write_csv(training_summary_path, training_rows)

    convergence = _training_convergence(
        training_rows,
        minimum_episodes=args.convergence_min_episodes,
        window_fraction=args.convergence_window_fraction,
        trend_threshold=args.convergence_trend_threshold,
        shift_threshold=args.convergence_shift_threshold,
        target_kl=DEFAULT_TARGET_KL,
    )
    convergence_path = os.path.join(launch_dir, "training_convergence_diagnostics.json")
    _json_dump(convergence_path, convergence)
    pre_evaluation_plots = _plot_outputs(launch_dir, training_rows, [])
    if (
        args.require_convergence
        and not convergence["all_policies_converged"]
    ):
        manifest.update(
            {
                "status": "training_not_converged",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "training_convergence": convergence,
                "artifacts": {
                    "training_summary": training_summary_path,
                    "training_convergence": convergence_path,
                    "plots": pre_evaluation_plots,
                },
            }
        )
        _json_dump(manifest_path, manifest)
        raise RuntimeError(
            "Training did not satisfy the preregistered convergence audit; "
            "held-out evaluation was not opened. Continue all policy seeds with "
            "--resume --train-only and a larger --train-episodes target."
        )

    evaluation_rows = []
    interface_verification = None
    paired_analysis = []
    benchmark_analysis = []
    performance_assessment = {"status": "not_evaluated"}
    if not args.train_only:
        for replication in range(1, args.eval_replications + 1):
            scenario_seed = _seed(args.launch_seed, 21, replication)
            evaluation_policy_seed = _seed(args.launch_seed, 22, replication)
            for strategy in args.strategies:
                policy_indices = (
                    range(1, args.policy_replicates + 1)
                    if strategy in LEARNED_STRATEGIES
                    else (0,)
                )
                for policy_replication in policy_indices:
                    print(
                        f"[EVAL] pair={replication}/{args.eval_replications} "
                        f"strategy={strategy} policy={policy_replication} "
                        f"scenario_seed={scenario_seed}",
                        flush=True,
                    )
                    checkpoint_index = (
                        policy_replication if strategy in LEARNED_STRATEGIES else 1
                    )
                    visualize_episode = bool(
                        replication <= args.visualize_eval_pairs
                        and strategy in args.visualization_strategies
                        and (
                            strategy not in LEARNED_STRATEGIES
                            or policy_replication == args.visualization_policy_replication
                        )
                    )
                    row = _run_episode(
                        replication=replication,
                        machine=args.machine,
                        phase=os.path.join(
                            launch_id,
                            "evaluation",
                            (
                                f"policy_{policy_replication:03d}"
                                if strategy in LEARNED_STRATEGIES
                                else "benchmarks"
                            ),
                        ),
                        strategy=strategy,
                        train_mode=False,
                        scenario_seed=scenario_seed,
                        policy_seed=evaluation_policy_seed,
                        checkpoint_path=checkpoint_paths[checkpoint_index],
                        diagnostics_path=diagnostics_paths[checkpoint_index],
                        overrides=overrides,
                        visualization_enabled=visualize_episode,
                        visualization_milestones=args.visualization_milestones,
                    )
                    row["policy_replication"] = policy_replication
                    evaluation_rows.append(row)
        _write_csv(evaluation_summary_path, evaluation_rows)
        interface_verification = _verify_matched_interface(evaluation_rows, args.strategies)
        _json_dump(os.path.join(launch_dir, "interface_parity.json"), interface_verification)
        paired_analysis = _paired_analysis(
            evaluation_rows,
            args.launch_seed,
            args.bootstrap_draws,
        )
        _write_csv(os.path.join(launch_dir, "paired_comparison.csv"), paired_analysis)
        _write_paper_table(os.path.join(launch_dir, "paired_comparison.md"), paired_analysis)
        benchmark_analysis = _benchmark_analysis(
            evaluation_rows,
            args.launch_seed,
            args.bootstrap_draws,
            benchmark_strategies=(
                strategy for strategy in args.strategies if strategy != "rl"
            ),
        )
        _write_csv(
            os.path.join(launch_dir, "benchmark_comparison.csv"),
            benchmark_analysis,
        )
        _write_benchmark_table(
            os.path.join(launch_dir, "benchmark_comparison.md"),
            benchmark_analysis,
        )
        performance_assessment = _performance_assessment(paired_analysis)
        _json_dump(
            os.path.join(launch_dir, "performance_assessment.json"),
            performance_assessment,
        )

    plot_paths = _plot_outputs(
        launch_dir,
        training_rows,
        paired_analysis,
        benchmark_analysis,
    )
    manifest.update(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_paths": checkpoint_paths,
            "interface_verification": interface_verification,
            "training_convergence": convergence,
            "performance_assessment": performance_assessment,
            "artifacts": {
                "training_summary": training_summary_path,
                "ppo_diagnostics": diagnostics_paths,
                "training_convergence": convergence_path,
                "evaluation_summary": evaluation_summary_path,
                "paired_comparison": os.path.join(launch_dir, "paired_comparison.csv"),
                "paper_table": os.path.join(launch_dir, "paired_comparison.md"),
                "benchmark_comparison": (
                    os.path.join(launch_dir, "benchmark_comparison.csv")
                    if benchmark_analysis
                    else None
                ),
                "benchmark_table": (
                    os.path.join(launch_dir, "benchmark_comparison.md")
                    if benchmark_analysis
                    else None
                ),
                "performance_assessment": os.path.join(
                    launch_dir,
                    "performance_assessment.json",
                ) if paired_analysis else None,
                "plots": plot_paths,
                "evacuation_visualizations": (
                    os.path.join(launch_dir, "evaluation")
                    if args.visualize_eval_pairs > 0
                    else None
                ),
            },
        }
    )
    _json_dump(manifest_path, manifest)
    print(f"[COMPLETE] artifacts={launch_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
