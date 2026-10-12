#!/usr/bin/env python3
"""Reproducible matched-seed backtest for regional shelter policies.

RL and every dynamic benchmark use the same regional observation contract,
exact-candidate action mask, deployment schedule, and budget. The primary
contrast is RL versus complete-ledger objective-greedy; simpler policies have
explicit floor, practice-rule, feedback, or appendix-diagnostic roles.
"""

from __future__ import annotations

import argparse
import ast
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
from GNN import HEURISTIC_PRIOR_SCALE
from RLBridge import (
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ROLLOUT_EPISODES,
    DEFAULT_TARGET_KL,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
DYNAMIC_STRATEGIES = (
    "rl",
    "objective_greedy",
    "risk_reduction",
    "route_saving",
    "heuristic",
    "hazard_weighted",
    "accessibility_deficit",
    "random",
    "initial_only",
)
STATIC_STRATEGIES = ("static_greedy", "rl_precommit")
LEARNED_STRATEGIES = ("rl", "rl_precommit")
PRIMARY_BENCHMARK_STRATEGY = "objective_greedy"

EVALUATION_METRICS = {
    "objective_normalized_cost": "lower",
    "safe_completed": "higher",
    "expected_casualties": "lower",
    "casualty": "lower",
    "unfinished": "lower",
    "restricted_mean_time_to_safety": "lower",
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
        "objective_normalized_cost",
        "objective_episode_score",
        "safe_completed",
        "shelter_evacuated",
        "arrival",
        "expected_casualties",
        "casualty",
        "unfinished",
        "restricted_mean_time_to_safety",
        "mean_safe_completion_time",
        "mean_evacuation_time",
        "remaining_risk_time_reduction",
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
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "torch",
        "osmnx",
        "networkx",
        "geopandas",
        "shapely",
        "scikit-learn",
        "tqdm",
    ):
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
    installed_shelters = int(len(core.shelterDS.shelterList))
    expected_shelters = int(core.targetActiveShelters)
    if installed_shelters != expected_shelters:
        raise RuntimeError(
            f"Benchmark shelter-budget mismatch for {strategy}: installed "
            f"{installed_shelters}, expected {expected_shelters}"
        )
    if bool(core.requirePopulationMatchedShelterCapacity):
        required_shelters = int(core.initShelterVol + core.maxAdditionalShelters)
        if expected_shelters != required_shelters:
            raise RuntimeError(
                f"Matched-capacity run targeted {expected_shelters} shelters, "
                f"expected the complete {required_shelters}-shelter budget"
            )
        installed_capacity = int(
            sum(
                int(getattr(shelter, "shelterCap", 0))
                for shelter in core.shelterDS.shelterList.values()
            )
        )
        required_capacity = int(core.pedVol + core.shelterCapacitySurplus)
        if installed_capacity != required_capacity:
            raise RuntimeError(
                f"Matched-capacity run installed {installed_capacity} places, "
                f"expected {required_capacity}"
            )
        unassigned = core.finalInstallationUnassignedRationalPopulation
        if unassigned is None or int(unassigned) != 0:
            raise RuntimeError(
                "Final shelter installation did not verify zero unassigned "
                f"rational population: {unassigned}"
            )
    if strategy == "initial_only":
        expected_additions = int(core.targetActiveShelters) - int(
            core.initShelterVol
        )
        if int(core.rl.maximum_deployments) != expected_additions:
            raise RuntimeError(
                "initial_only must start with the common initial shelters and "
                "retain the complete sequential installation budget"
            )
        if len(core.rl.static_precommit_schedule) != expected_additions:
            raise RuntimeError(
                "initial_only must commit every scheduled site at time zero"
            )
        expected_times = tuple(
            int(core.shelterActionInterval) * index
            for index in range(1, expected_additions + 1)
        )
        if tuple(core.rl.deployment_times) != expected_times:
            raise RuntimeError(
                "initial_only did not execute its precommitted sites on the "
                f"common schedule: expected {expected_times}, observed "
                f"{tuple(core.rl.deployment_times)}"
            )
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
    """Return one explicitly named physical evaluation metric."""
    return float(row[metric])


def _paired_analysis(rows: list[dict], launch_seed: int, draws: int) -> list[dict]:
    benchmark_by_scenario = {
        int(row["replication"]): row
        for row in rows
        if str(row["deployment_strategy"]) == PRIMARY_BENCHMARK_STRATEGY
    }
    rl_by_key = {
        (int(row.get("policy_replication", 1)), int(row["replication"])): row
        for row in rows
        if str(row["deployment_strategy"]) == "rl"
    }
    policy_replications = sorted({key[0] for key in rl_by_key})
    replications = sorted(benchmark_by_scenario)
    if not policy_replications or not replications:
        raise RuntimeError(
            "No paired RL/objective-greedy replications were found"
        )
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
    fixed_policy_inference = len(policy_replications) == 1
    inferentially_eligible = len(replications) >= 2
    inference_scope = (
        "conditional_on_one_fixed_trained_policy"
        if fixed_policy_inference
        else "joint_over_policy_and_scenario_seeds"
    )
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
        benchmark_values = np.asarray(
            [
                _evaluation_metric(benchmark_by_scenario[replication], metric)
                for replication in replications
            ],
            dtype=float,
        )
        raw_difference = rl_values - benchmark_values[None, :]
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
        p_value = (
            None
            if fixed_policy_inference
            else _paired_randomization_pvalue(
                policy_mean_improvements,
                rng,
                draws,
            )
        )
        standard_deviation = (
            float(policy_mean_improvements.std(ddof=1))
            if len(policy_replications) > 1
            else 0.0
        )
        standard_error = (
            float(bootstrap_means.std(ddof=1))
            if fixed_policy_inference and len(bootstrap_means) > 1
            else standard_deviation / math.sqrt(len(policy_replications))
        )
        output.append(
            {
                "metric": metric,
                "preferred_direction": direction,
                "policy_replications": len(policy_replications),
                "scenario_replications": len(replications),
                "inferentially_eligible": inferentially_eligible,
                "inference_scope": inference_scope,
                "paired_replications": int(improvement.size),
                "benchmark_strategy": PRIMARY_BENCHMARK_STRATEGY,
                "rl_mean": float(rl_values.mean()),
                "benchmark_mean": float(benchmark_values.mean()),
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
                    "comparison_role": BENCHMARK_POLICY_CONTRACTS.get(
                        benchmark_strategy,
                        {},
                    ).get("role", "timing_diagnostic"),
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
    for policy_replication, policy_rows in sorted(by_policy.items()):
        policy_rows.sort(key=lambda row: int(row["replication"]))
        rollout_pi = any(
            float(row.get("rollout_pi_actor_enabled", 0.0)) > 0.5
            for row in policy_rows
        )
        required_fields = (
            (
                "episode_return",
                "behavior_policy_entropy",
                "rollout_pi_actor_kl",
                "actor_gradient_norm",
                "policy_loss",
            )
            if rollout_pi
            else (
                "episode_return",
                "behavior_policy_entropy",
                "approximate_kl",
                "gradient_norm",
                "policy_loss",
                "value_loss",
            )
        )
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
                    float(
                        row[
                            "rollout_pi_actor_kl"
                            if rollout_pi
                            else "approximate_kl"
                        ]
                    )
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
        actor_fit_rows = [
            row
            for row in policy_rows
            if float(row.get("optimizer_updated", 0.0)) > 0.5
            and float(row.get("rollout_pi_actor_enabled", 0.0)) > 0.5
        ]

        actor_fit_finite = bool(
            actor_fit_rows
            and all(
                np.isfinite(float(row["rollout_pi_target_kl_before"]))
                and np.isfinite(float(row["rollout_pi_target_kl_after"]))
                and np.isfinite(
                    float(row["actor_policy_entropy_after"])
                )
                for row in actor_fit_rows
            )
        )
        actor_fit_improvement_fraction = (
            float(
                np.mean(
                    [
                        float(row["rollout_pi_target_kl_after"])
                        <= float(row["rollout_pi_target_kl_before"])
                        for row in actor_fit_rows
                    ]
                )
            )
            if actor_fit_rows
            else float("nan")
        )
        branch_target_coverage = (
            float(
                np.mean(
                    [
                        float(row["rollout_pi_target_coverage"])
                        for row in actor_fit_rows
                    ]
                )
            )
            if actor_fit_rows
            else float("nan")
        )

        if rollout_pi:
            # Rollout policy iteration has no KL/controller gate. Convergence is
            # the ordinary stationarity audit plus evidence that its complete
            # simulator-graded categorical actor fits were numerically valid.
            # Entropy and retained-policy KL remain descriptive diagnostics.
            converged = bool(
                enough and finite and rollout_complete
                and optimization_events > 0
                and actor_fit_finite
                and branch_target_coverage >= 1.0 - 1e-9
                and trend_span_sd <= float(trend_threshold)
                and window_shift_sd <= float(shift_threshold)
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
                    "rollout_policy_iteration_tail_stationarity"
                    if rollout_pi
                    else "ppo_tail_stationarity"
                ),
                "rollout_pi_fit_diagnostics_finite": actor_fit_finite,
                "rollout_pi_target_kl_improvement_fraction": (
                    actor_fit_improvement_fraction
                ),
                "mean_rollout_pi_target_coverage": branch_target_coverage,
                "tail_entropy_mean": (
                    float(numeric["behavior_policy_entropy"][-window:].mean())
                    if finite and window >= 2
                    else float("nan")
                ),
                "converged": converged,
            }
        )
    return {
        "definition": (
            "PPO policies require tail stationarity and KL compliance. Simulator-graded "
            "rollout-policy-iteration policies require tail stationarity, complete branch "
            "targets and finite categorical actor fits; retained-policy KL is descriptive and no controller "
            "gate accepts or rejects an actor update."
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
    primary_metric = "objective_normalized_cost"
    primary = next((row for row in analysis if row["metric"] == primary_metric), None)
    if primary is None:
        return {"status": "unavailable", "primary_metric": primary_metric}
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
        "primary_metric": primary_metric,
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
    def provenance_mapping(value):
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    dynamic = tuple(strategy for strategy in strategies if strategy in DYNAMIC_STRATEGIES)
    reference_by_replication = {
        int(row["replication"]): row
        for row in rows
        if row["deployment_strategy"] == PRIMARY_BENCHMARK_STRATEGY
    }
    mismatches = []
    compared = 0
    for row in rows:
        strategy = row["deployment_strategy"]
        if strategy not in dynamic or strategy == PRIMARY_BENCHMARK_STRATEGY:
            continue
        replication = int(row["replication"])
        reference = reference_by_replication.get(replication)
        if reference is None:
            mismatches.append(
                {
                    "replication": replication,
                    "reason": "missing objective_greedy reference",
                }
            )
            continue
        compared += 1
        row_random_streams = provenance_mapping(row.get("random_stream_seeds"))
        reference_random_streams = provenance_mapping(
            reference.get("random_stream_seeds")
        )
        required_provenance_present = all(
            (
                row_random_streams is not None,
                reference_random_streams is not None,
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
                row_random_streams != reference_random_streams,
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
        "scenario_replications": len(reference_by_replication),
        "reference_strategy": PRIMARY_BENCHMARK_STRATEGY,
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
        "# Paired RL versus complete-ledger objective-greedy",
        "",
        interpretation,
        "`episode_return` denotes the action-count-invariant full-episode policy objective.",
        "",
        "| Metric | RL mean | Objective-greedy mean | RL improvement | 95% CI | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis:
        p_value = row.get("two_sided_randomization_p")
        p_text = "NA" if p_value is None else f"{float(p_value):.4g}"
        lines.append(
            "| {metric} | {rl_mean:.5g} | {benchmark_mean:.5g} | {mean_rl_improvement:.5g} | "
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
        "complete-ledger objective-greedy contrast is primary; every other "
        "benchmark keeps its registered diagnostic role."
        if inferentially_eligible
        else "This engineering backtest is descriptive because it lacks at least two "
        "independent policy seeds and two held-out scenarios. Complete-ledger "
        "objective-greedy remains primary; all others keep diagnostic roles."
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
        ax.set(
            xlabel="Training episode",
            ylabel="Return (savings versus uniform random)",
            title="RL training convergence",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(launch_dir, "training_convergence.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

        # Plot the signed uniform-random-relative return and report casualties
        # separately. Capacity arrival, natural evacuation progress, and
        # casualties are outcomes rather than separate reward components.
        fields = (
            (
                "episode_return",
                "Savings versus uniform-random siting",
            ),
        )
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
        axes = axes.reshape(-1)
        series_by_field = {field: [] for field, _ in fields}
        casualty_series = []
        safe_series = []
        unfinished_series = []
        behavior_entropy_series = []
        actor_entropy_series = []
        kl_series = []
        rollout_pi_training = False
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
                    np.asarray(
                        [
                            float(
                                row.get(
                                    field,
                                    row.get("episode_return", 0.0),
                                )
                            )
                            for row in policy_rows
                        ]
                    )
                )
            casualty_series.append(
                np.asarray([float(row.get("casualty", 0.0)) for row in policy_rows])
            )
            safe_series.append(
                np.asarray(
                    [float(row.get("safe_completed", 0.0)) for row in policy_rows]
                )
            )
            unfinished_series.append(
                np.asarray(
                    [float(row.get("unfinished", 0.0)) for row in policy_rows]
                )
            )
            behavior_entropy_series.append(
                np.asarray(
                    [float(row["behavior_policy_entropy"]) for row in policy_rows]
                )
            )
            actor_entropy_series.append(
                np.asarray(
                    [
                        float(row.get("actor_policy_entropy_after", float("nan")))
                        if float(row.get("optimizer_updated", 0.0)) > 0.5
                        else float("nan")
                        for row in policy_rows
                    ]
                )
            )
            rollout_pi_training = rollout_pi_training or any(
                float(row.get("rollout_pi_actor_enabled", 0.0)) > 0.5
                for row in policy_rows
            )
            kl_series.append(
                np.asarray(
                    [
                        float(
                            row[
                                "rollout_pi_actor_kl"
                                if rollout_pi_training
                                else "approximate_kl"
                            ]
                        )
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

        reward_ax = axes[0]
        for field, label in fields:
            if series_by_field[field]:
                values = np.stack(series_by_field[field]).mean(axis=0)
                x_values, y_values = moving_mean(values)
                reward_ax.plot(x_values, y_values, label=label)
        reward_ax.set(
            title="Uniform-random-relative policy return",
            ylabel="Normalized signed return",
        )
        reward_ax.legend(fontsize=8)

        casualty_ax = axes[1]
        if casualty_series:
            x_values, y_values = moving_mean(
                np.stack(casualty_series).mean(axis=0)
            )
            casualty_ax.plot(x_values, y_values, color="tab:red")
        casualty_ax.set(
            title="Casualty trend (reported only)",
            ylabel="Moving-mean casualties per episode",
        )

        entropy_ax = axes[2]
        if behavior_entropy_series:
            x_values, y_values = moving_mean(
                np.stack(behavior_entropy_series).mean(axis=0)
            )
            entropy_ax.plot(
                x_values,
                y_values,
                color="tab:purple",
                label="sampled behavior policy",
            )
        if actor_entropy_series:
            stacked_actor_entropy = np.stack(actor_entropy_series)
            valid = np.isfinite(stacked_actor_entropy).any(axis=0)
            actor_values = np.full(stacked_actor_entropy.shape[1], np.nan, dtype=float)
            actor_values[valid] = np.nanmean(stacked_actor_entropy[:, valid], axis=0)
            entropy_ax.plot(
                common_episodes[valid],
                actor_values[valid],
                color="tab:blue",
                marker="o",
                markersize=3,
                label="raw actor after fit",
            )
        entropy_ax.set(
            title="Behavior and trainable-actor concentration",
            ylabel="Normalized feasible-action entropy [0, 1]",
        )
        entropy_ax.legend(fontsize=8)

        kl_ax = axes[3]
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
        if rollout_pi_training:
            kl_ax.set(
                title="Residual-actor movement (descriptive)",
                xlabel="Training episode",
                ylabel="KL(new actor || iteration prior)",
            )
        else:
            kl_ax.axhline(
                DEFAULT_TARGET_KL,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="target KL",
            )
            kl_ax.set(
                title="PPO update size",
                xlabel="Training episode",
                ylabel="Approximate KL",
            )
            kl_ax.legend(fontsize=8)

        agreement_ax = axes[4]
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

        outcome_ax = axes[5]
        if safe_series:
            x_values, y_values = moving_mean(
                np.stack(safe_series).mean(axis=0)
            )
            outcome_ax.plot(x_values, y_values, label="Safely completed")
        if unfinished_series:
            x_values, y_values = moving_mean(
                np.stack(unfinished_series).mean(axis=0)
            )
            outcome_ax.plot(x_values, y_values, label="Unfinished")
        outcome_ax.set(
            title="Terminal evacuation outcomes",
            xlabel="Training episode",
            ylabel="Pedestrians",
        )
        outcome_ax.legend(fontsize=8)
        for diagnostic_ax in axes:
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
    ax.set(
        xlabel="Mean paired improvement (positive favors RL)",
        title="RL versus complete-ledger objective-greedy: paired 95% intervals",
    )
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
                "objective_greedy",
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
            "objective_greedy": "Objective-matched greedy",
            "risk_reduction": "Future HWTTS reduction",
            "route_saving": "Demand-weighted route saving",
            "heuristic": "Max active demand",
            "hazard_weighted": "Hazard-weighted demand",
            "accessibility_deficit": "Accessibility deficit",
            "random": "Random feasible candidate",
            "static_greedy": "Static demand greedy",
            "rl_precommit": "RL precommit",
            "initial_only": "Static precommitted schedule",
        }
        metric_labels = {
            "objective_normalized_cost": "Normalized objective-cost reduction",
            "safe_completed": "Safe completions",
            "expected_casualties": "Expected-casualty reduction",
            "casualty": "Casualty reduction",
            "unfinished": "Unfinished reduction",
            "restricted_mean_time_to_safety": "Time-to-safety reduction",
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
        default="rl,objective_greedy,heuristic,random,initial_only",
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
        default="rl,objective_greedy",
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
    if not args.train_only and not {
        "rl",
        PRIMARY_BENCHMARK_STRATEGY,
    }.issubset(strategies):
        parser.error(
            "evaluation must include both rl and objective_greedy for the "
            "primary paired test"
        )
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
            "learner_diagnostics.csv",
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
            "r_k=(mean_{a in feasible} J_k(a)-J_k(chosen))/(P H_minutes)"
        ),
        "evaluation_objective_equation": (
            "cost=[plain_time+hazard_exposure_time+expected_casualty_"
            "person_time+clearance_tail_person_time+terminal_residual]/"
            "(P H_minutes); lower_is_better; display_score=1/(1+cost)"
        ),
        "unavailable_shelter_time": "full_episode_horizon_120_minutes",
        "terminal_residual_in_learning_reward": False,
        "casualty_role": (
            "expected_casualty_intensity_is_priced_in_the_objective;_"
            "realized_casualties_are_a_secondary_stochastic_outcome"
        ),
        "learner_design": {
            "learner_batch_episodes": int(
                overrides.get("ppoRolloutEpisodes", DEFAULT_ROLLOUT_EPISODES)
            ),
            "optimizer_family": (
                "simulator_graded_rollout_policy_iteration_when_enabled; "
                "otherwise_direct_ppo"
            ),
            "policy_architecture": (
                "unbounded linear prior-residual exact-candidate scorer "
                "with pedestrian, hazard, infrastructure, spatial, and route context"
            ),
            "heuristic_prior_logit_scale": float(HEURISTIC_PRIOR_SCALE),
            "residual_correction": "unbounded_linear_actor_readout",
            "entropy_coefficient_start": float(
                overrides.get("entropyCoefficientStart", DEFAULT_ENTROPY_COEF)
            ),
            "entropy_coefficient_end": float(
                overrides.get("entropyCoefficientEnd", DEFAULT_ENTROPY_COEF)
            ),
            "entropy_normalization": "log_feasible_action_count",
            "rollout_policy_iteration_kl_role": (
                "descriptive_exact_KL_of_raw_actor_to_iteration_prior; no gate or rollback"
            ),
            "direct_ppo_kl_role": "trust_region_diagnostic",
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
                rollout_size = int(
                    episode_overrides.get(
                        "ppoRolloutEpisodes", DEFAULT_ROLLOUT_EPISODES
                    )
                )
                episode_overrides.update(
                    {
                        "trainingStageId": "joint",
                        "trainingStageUpdateIndex": int(
                            (episode - 1) // rollout_size
                        ),
                        "trainingStageTotalUpdates": int(
                            math.ceil(args.train_episodes / rollout_size)
                        ),
                    }
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
                "learner_diagnostics": diagnostics_paths,
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
