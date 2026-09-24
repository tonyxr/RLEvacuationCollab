#!/usr/bin/env python3
"""Pooled five-city PPO training and matched, city-stratified backtesting.

One residual policy is trained on a balanced interleaving of city maps. Every
city uses the identical observation fields, 8x8 regional context graph,
20-slot exact-candidate action contract, reward equation, and deployment
budget rule. Held-out scenarios are paired within city and never used for
training or convergence decisions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import sys
import itertools

import numpy as np

from CellPartitioning import normalize_partition_mode
from CityProfiles import DEFAULT_CITY_PROFILE_PATH, CityProfile, CitySuite, load_city_suite
from TrainingCurriculum import (
    CurriculumEpisode,
    build_curriculum_schedule,
    load_training_curriculum,
)
from PolicyCache import PolicyCache, source_fingerprint


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
DEFAULT_POLICY_CACHE = os.path.join(PROJECT_ROOT, "cache", "trained_policies")
DEFAULT_FULL_CURRICULUM = os.path.join(
    PROJECT_ROOT,
    "config",
    "staged_training_curriculum_5000_convergence.json",
)
DEFAULT_POOLED_LEARNING_RATE = 3e-4
DEFAULT_TARGET_KL = 0.015
POLICY_DYNAMICS_SOURCES = (
    "CAProcessor.py",
    "Core.py",
    "DecisionInterface.py",
    "GNN.py",
    "HazardDatabase.py",
    "MapDatabase.py",
    "NetworkCongestion.py",
    "OSMProcessor.py",
    "PedestrianDatabase.py",
    "RLBridge.py",
    "RewardProcessor.py",
    "SocialForce.py",
)
DYNAMIC_STRATEGIES = (
    "rl",
    "risk_reduction",
    "heuristic",
    "hazard_weighted",
    "accessibility_deficit",
    "random",
)
STATIC_STRATEGIES = ("initial_only", "static_greedy", "rl_precommit")
LEARNED_STRATEGIES = ("rl", "rl_precommit")


def _curriculum_is_monotone_extension(previous: object, requested: object) -> bool:
    """Return whether ``requested`` preserves every episode in ``previous``.

    A failed stationarity audit may justify collecting more training episodes,
    but it must not authorize changing the population, hazard, or variant mix
    seen by an already-started policy.  Source paths and hashes necessarily
    differ for a registered extension, so compatibility is established from
    the immutable semantic fields.  Only the final existing stage may grow;
    additional stages may be appended after it.
    """
    if not isinstance(previous, dict) or not isinstance(requested, dict):
        return False
    if previous == requested:
        return True
    for field in ("schema_version", "curriculum_id", "description"):
        if previous.get(field) != requested.get(field):
            return False
    previous_stages = previous.get("stages")
    requested_stages = requested.get("stages")
    if not isinstance(previous_stages, list) or not isinstance(requested_stages, list):
        return False
    if not previous_stages or len(requested_stages) < len(previous_stages):
        return False
    for index, previous_stage in enumerate(previous_stages):
        requested_stage = requested_stages[index]
        if not isinstance(previous_stage, dict) or not isinstance(requested_stage, dict):
            return False
        for field in ("stage_id", "label", "variants"):
            if previous_stage.get(field) != requested_stage.get(field):
                return False
        previous_episodes = int(previous_stage.get("episodes_per_city", 0))
        requested_episodes = int(requested_stage.get("episodes_per_city", 0))
        if index < len(previous_stages) - 1:
            if requested_episodes != previous_episodes:
                return False
        elif requested_episodes < previous_episodes:
            return False
    return int(requested.get("episodes_per_city", 0)) >= int(
        previous.get("episodes_per_city", 0)
    )
MINIMUM_ROLLOUT_EPISODES = 8


def _manifest_time_fields(previous_manifest: dict | None) -> dict[str, str]:
    """Preserve the launch origin while recording each resume explicitly."""
    now = datetime.now(timezone.utc).isoformat()
    if isinstance(previous_manifest, dict):
        started = previous_manifest.get("started_utc")
        return {
            "started_utc": str(started) if started else now,
            "resumed_utc": now,
        }
    return {"started_utc": now}


def _manifest_command_fields(
    previous_manifest: dict | None,
    current_command: list[str],
) -> dict[str, list[str]]:
    """Keep the originating command and record a resume separately."""
    if isinstance(previous_manifest, dict):
        original = previous_manifest.get("command")
        return {
            "command": list(original) if isinstance(original, list) else current_command,
            "resume_command": current_command,
        }
    return {"command": current_command}


def _seed(launch_seed: int, stream: int, index: int = 0) -> int:
    sequence = np.random.SeedSequence([int(launch_seed), int(stream), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _policy_cache_contract(
    resume_contract: dict,
    *,
    train_episodes_per_city: int,
    total_train_episodes: int,
    policy_replication: int,
    launch_seed: int,
) -> dict:
    """Exact contract shared by policy-cache publication and restoration."""
    return {
        "runner": "multicity_backtest",
        "training_contract": resume_contract,
        "train_episodes_per_city": int(train_episodes_per_city),
        "total_train_episodes": int(total_train_episodes),
        "policy_replication": int(policy_replication),
        "policy_seed": int(_seed(launch_seed, 10, policy_replication)),
    }


def _strict_json_value(value):
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


def _parse_overrides(items) -> dict:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must use NAME=VALUE, got {item!r}")
        name, raw = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Override name cannot be empty")
        try:
            value = json.loads(raw.strip())
        except json.JSONDecodeError:
            value = raw.strip()
        overrides[name] = value
    return overrides


def _paired_randomization_pvalue(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> float:
    observed = abs(float(values.mean()))
    exact_assignments = 1 << int(values.size)
    if values.size <= 20 and exact_assignments <= int(draws):
        exceedances = sum(
            abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-15
            for signs in itertools.product((-1.0, 1.0), repeat=int(values.size))
        )
        return float(exceedances / exact_assignments)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(draws), values.size))
    null_means = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(null_means >= observed)) / (int(draws) + 1))


def balanced_city_schedule(
    cities: tuple[CityProfile, ...],
    episodes_per_city: int,
    launch_seed: int,
) -> tuple[CityProfile, ...]:
    """Return a deterministic block-randomized schedule with exact balance."""
    episodes_per_city = int(episodes_per_city)
    if not cities or episodes_per_city <= 0:
        raise ValueError("cities and episodes_per_city must be non-empty and positive")
    schedule = []
    for cycle in range(episodes_per_city):
        order = np.random.default_rng(_seed(launch_seed, 31, cycle)).permutation(len(cities))
        schedule.extend(cities[int(index)] for index in order)
    counts = {city.city_id: 0 for city in cities}
    for city in schedule:
        counts[city.city_id] += 1
    if any(count != episodes_per_city for count in counts.values()):
        raise AssertionError(f"Internal city-schedule imbalance: {counts}")
    return tuple(schedule)


def balanced_rollout_episodes(
    city_count: int,
    minimum_episodes: int = MINIMUM_ROLLOUT_EPISODES,
) -> int:
    """Smallest rollout at least ``minimum_episodes`` containing whole city blocks."""
    city_count = int(city_count)
    minimum_episodes = int(minimum_episodes)
    if city_count <= 0 or minimum_episodes <= 0:
        raise ValueError("city_count and minimum_episodes must be positive")
    return city_count * int(math.ceil(minimum_episodes / city_count))


def _city_overrides(
    suite: CitySuite,
    city: CityProfile,
    user_overrides: dict,
    hazard_mode: str,
) -> dict:
    # Map identity is owned by the preregistered profile and cannot be changed
    # through an accidental generic override. Common simulator parameters may
    # be intentionally overridden, and are recorded in the manifest.
    result = dict(suite.common_experiment)
    result.update(user_overrides)
    protected = set(city.core_overrides())
    conflict = sorted(protected.intersection(user_overrides))
    if conflict:
        raise ValueError(
            "City map fields must be changed in the versioned profile file, not "
            f"with --override: {conflict}"
        )
    result.update(city.core_overrides())
    result["hazardEvolutionMode"] = str(hazard_mode)
    return result


def _validate_shared_interface(overrides_by_city: dict[str, dict]) -> tuple[int, int]:
    grids = {
        (int(overrides["cellX"]), int(overrides["cellY"]))
        for overrides in overrides_by_city.values()
    }
    if len(grids) != 1:
        raise ValueError(
            "Pooled training requires one grid shape so every city has the same action space"
        )
    grid = next(iter(grids))
    if grid[0] <= 0 or grid[1] <= 0:
        raise ValueError("The shared city grid must be positive")
    partition_contracts = {
        (
            normalize_partition_mode(
                overrides.get("cellPartitionMode", "node_density_adaptive")
            ),
            float(overrides.get("cellPartitionMinWidthFraction", 1e-4)),
        )
        for overrides in overrides_by_city.values()
    }
    if len(partition_contracts) != 1:
        raise ValueError(
            "Pooled training requires one shared cell-partition mode and minimum width"
        )
    return grid


def _validate_resume_rows(
    rows: list[dict],
    schedule: tuple,
    policy_replicates: int,
) -> None:
    for policy_replication in range(1, int(policy_replicates) + 1):
        policy_rows = sorted(
            (
                row
                for row in rows
                if int(row.get("policy_replication", 1)) == policy_replication
            ),
            key=lambda row: int(row["replication"]),
        )
        for expected_episode, row in enumerate(policy_rows, start=1):
            if expected_episode > len(schedule):
                raise ValueError("Training summary contains more episodes than requested")
            scheduled = schedule[expected_episode - 1]
            expected_city = (
                scheduled.city.city_id
                if isinstance(scheduled, CurriculumEpisode)
                else scheduled.city_id
            )
            if int(row["replication"]) != expected_episode or row.get("city_id") != expected_city:
                raise ValueError(
                    "Training summary does not match the deterministic city schedule at "
                    f"policy={policy_replication}, episode={expected_episode}"
                )
            if isinstance(scheduled, CurriculumEpisode):
                curriculum_fields = {
                    "training_stage_id": scheduled.stage_id,
                    "training_variant_id": scheduled.variant_id,
                    "stage_index": scheduled.stage_index,
                    "stage_city_replication": scheduled.stage_city_replication,
                }
                mismatches = {
                    key: {"expected": value, "observed": row.get(key)}
                    for key, value in curriculum_fields.items()
                    if str(row.get(key)) != str(value)
                }
                if mismatches:
                    raise ValueError(
                        "Training summary does not match the deterministic curriculum "
                        f"at policy={policy_replication}, episode={expected_episode}: "
                        f"{mismatches}"
                    )


def _equal_city_training_blocks(
    rows: list[dict],
    cities: tuple[CityProfile, ...],
) -> list[dict]:
    """Aggregate training diagnostics into complete, equal-city cycle means."""
    city_ids = {city.city_id for city in cities}
    if not city_ids:
        raise ValueError("At least one city is required for convergence aggregation")
    output = []
    policy_ids = sorted({int(row.get("policy_replication", 1)) for row in rows})
    for policy_id in policy_ids:
        policy_rows = sorted(
            (
                row
                for row in rows
                if int(row.get("policy_replication", 1)) == policy_id
            ),
            key=lambda row: int(row["replication"]),
        )
        if len(policy_rows) % len(cities) != 0:
            raise ValueError(
                f"Policy {policy_id} training rows do not end at an equal-city block"
            )
        for start in range(0, len(policy_rows), len(cities)):
            block = policy_rows[start:start + len(cities)]
            observed = {str(row["city_id"]) for row in block}
            if observed != city_ids:
                raise ValueError(
                    f"Policy {policy_id} block {start // len(cities) + 1} is not "
                    f"city-balanced: {sorted(observed)}"
                )
            update_rows = [
                row for row in block if float(row.get("optimizer_updated", 0.0)) > 0.5
            ]
            update = update_rows[-1] if update_rows else None
            output.append(
                {
                    "policy_replication": policy_id,
                    "replication": start // len(cities) + 1,
                    "episode_return": float(
                        np.mean([float(row["episode_return"]) for row in block])
                    ),
                    "entropy": float(
                        np.mean([float(row["entropy"]) for row in block])
                    ),
                    "approximate_kl": (
                        float(update["approximate_kl"]) if update else 0.0
                    ),
                    "gradient_norm": (
                        float(update["gradient_norm"]) if update else 0.0
                    ),
                    "policy_loss": float(update["policy_loss"]) if update else 0.0,
                    "value_loss": float(update["value_loss"]) if update else 0.0,
                    "optimizer_updated": float(update is not None),
                    "rollout_episodes_pending": float(
                        block[-1].get("rollout_episodes_pending", 0.0)
                    ),
                }
            )
    return output


def _multicity_training_convergence(
    rows: list[dict],
    cities: tuple[CityProfile, ...],
    *,
    minimum_episodes: int,
    window_fraction: float,
    trend_threshold: float,
    shift_threshold: float,
    target_kl: float,
) -> dict:
    """Audit stationarity on equal-city means while retaining PPO diagnostics."""
    from backtest import _training_convergence

    blocks = _equal_city_training_blocks(rows, cities)
    minimum_blocks = int(math.ceil(int(minimum_episodes) / len(cities)))
    result = _training_convergence(
        blocks,
        minimum_episodes=minimum_blocks,
        window_fraction=window_fraction,
        trend_threshold=trend_threshold,
        shift_threshold=shift_threshold,
        target_kl=target_kl,
    )
    episode_counts = {
        policy: sum(
            int(row.get("policy_replication", 1)) == policy for row in rows
        )
        for policy in {int(row.get("policy_replication", 1)) for row in rows}
    }
    for policy in result["policies"]:
        policy_id = int(policy["policy_replication"])
        policy["equal_city_blocks"] = int(policy["episodes"])
        policy["training_episodes"] = int(episode_counts[policy_id])
        policy["window_equal_city_blocks"] = int(policy.pop("window_episodes"))
        policy["episodes"] = int(episode_counts[policy_id])
    result["definition"] = (
        "Every policy has the minimum episode count, finite PPO diagnostics, a "
        "complete on-policy rollout, stationary equal-city block-mean tail return, "
        "and no more than 10% tail optimizer events above target KL."
    )
    result["thresholds"]["minimum_equal_city_blocks"] = minimum_blocks
    result["thresholds"]["minimum_episodes"] = int(minimum_episodes)
    result["return_aggregation"] = (
        "One mean per complete block containing every selected city exactly once"
    )
    return result


def _stratified_paired_analysis(
    rows: list[dict],
    cities: tuple[CityProfile, ...],
    launch_seed: int,
    draws: int,
) -> list[dict]:
    """Macro-average cities and bootstrap policy seeds plus scenarios within city.

    The five named cities are fixed design sites, not a random sample of all US
    cities. Therefore cities are averaged but not resampled. Policy seeds are
    resampled jointly across cities and scenarios are resampled independently
    within each city.
    """
    from backtest import EVALUATION_METRICS, _evaluation_metric

    metrics = EVALUATION_METRICS
    city_matrices = {}
    policy_ids = None
    scenarios_per_city = None
    for city in cities:
        city_rows = [row for row in rows if row.get("city_id") == city.city_id]
        heuristic = {
            int(row["city_scenario_replication"]): row
            for row in city_rows
            if row["deployment_strategy"] == "heuristic"
        }
        rl = {
            (
                int(row["policy_replication"]),
                int(row["city_scenario_replication"]),
            ): row
            for row in city_rows
            if row["deployment_strategy"] == "rl"
        }
        city_policy_ids = sorted({key[0] for key in rl})
        city_scenarios = sorted(heuristic)
        if not city_policy_ids or not city_scenarios:
            raise RuntimeError(f"Missing paired evaluation rows for {city.city_id}")
        if policy_ids is None:
            policy_ids = city_policy_ids
            scenarios_per_city = len(city_scenarios)
        if city_policy_ids != policy_ids or len(city_scenarios) != scenarios_per_city:
            raise RuntimeError("Every city must have the same complete policy-by-scenario matrix")
        missing = [
            (policy, scenario)
            for policy in policy_ids
            for scenario in city_scenarios
            if (policy, scenario) not in rl
        ]
        if missing:
            raise RuntimeError(f"Incomplete evaluation matrix for {city.city_id}: {missing[:3]}")
        city_matrices[city.city_id] = (heuristic, rl, city_scenarios)

    output = []
    fixed_policy_inference = len(policy_ids) == 1
    inferentially_eligible = int(scenarios_per_city) >= 2
    inference_scope = (
        "conditional_on_one_fixed_trained_policy"
        if fixed_policy_inference
        else "joint_over_policy_and_scenario_seeds"
    )
    for metric_index, (metric, direction) in enumerate(metrics.items()):
        improvements = []
        rl_city_means = []
        heuristic_city_means = []
        for city in cities:
            heuristic, rl, scenarios = city_matrices[city.city_id]
            rl_values = np.asarray(
                [
                    [
                        _evaluation_metric(rl[(policy, scenario)], metric)
                        for scenario in scenarios
                    ]
                    for policy in policy_ids
                ],
                dtype=float,
            )
            heuristic_values = np.asarray(
                [
                    _evaluation_metric(heuristic[scenario], metric)
                    for scenario in scenarios
                ],
                dtype=float,
            )
            difference = rl_values - heuristic_values[None, :]
            improvements.append(difference if direction == "higher" else -difference)
            rl_city_means.append(float(rl_values.mean()))
            heuristic_city_means.append(float(heuristic_values.mean()))

        cube = np.stack(improvements, axis=0)  # city, policy, scenario
        rng = np.random.default_rng(_seed(launch_seed, 191, metric_index))
        policy_draws = rng.integers(0, len(policy_ids), size=(int(draws), len(policy_ids)))
        scenario_draws = rng.integers(
            0,
            scenarios_per_city,
            size=(int(draws), len(cities), scenarios_per_city),
        )
        bootstrap = np.empty(int(draws), dtype=float)
        for draw in range(int(draws)):
            city_means = []
            for city_index in range(len(cities)):
                city_means.append(
                    cube[city_index][
                        np.ix_(policy_draws[draw], scenario_draws[draw, city_index])
                    ].mean()
                )
            bootstrap[draw] = float(np.mean(city_means))
        lower, upper = np.quantile(bootstrap, (0.025, 0.975))
        policy_means = cube.mean(axis=(0, 2))
        standard_deviation = (
            float(policy_means.std(ddof=1)) if len(policy_ids) > 1 else 0.0
        )
        output.append(
            {
                "scope": "macro_all_cities",
                "city_id": "ALL",
                "metric": metric,
                "preferred_direction": direction,
                "cities": len(cities),
                "policy_replications": len(policy_ids),
                "scenario_replications": int(len(cities) * scenarios_per_city),
                "scenarios_per_city": int(scenarios_per_city),
                "inferentially_eligible": inferentially_eligible,
                "inference_scope": inference_scope,
                "paired_replications": int(cube.size),
                "rl_mean": float(np.mean(rl_city_means)),
                "heuristic_mean": float(np.mean(heuristic_city_means)),
                "mean_rl_improvement": float(cube.mean()),
                "bootstrap_95_ci_low": float(lower),
                "bootstrap_95_ci_high": float(upper),
                "paired_standard_error": (
                    float(np.std(bootstrap, ddof=1))
                    if fixed_policy_inference and len(bootstrap) > 1
                    else standard_deviation / math.sqrt(len(policy_ids))
                ),
                "paired_effect_size_dz": (
                    float(policy_means.mean()) / standard_deviation
                    if standard_deviation > 0.0
                    else None
                ),
                "two_sided_randomization_p": (
                    None
                    if fixed_policy_inference
                    else _paired_randomization_pvalue(policy_means, rng, draws)
                ),
                "rl_win_rate": float(np.mean(cube > 0.0)),
                "tie_rate": float(np.mean(cube == 0.0)),
                "superiority_ci_excludes_zero": bool(
                    inferentially_eligible and lower > 0.0
                ),
                "city_sampling_note": (
                    "Fixed-site macro estimand: cities are equally weighted and not resampled"
                ),
            }
        )
    return output


def _all_paired_analysis(
    rows: list[dict],
    cities: tuple[CityProfile, ...],
    launch_seed: int,
    draws: int,
) -> list[dict]:
    from backtest import _paired_analysis

    result = _stratified_paired_analysis(rows, cities, launch_seed, draws)
    for city in cities:
        city_rows = [row for row in rows if row.get("city_id") == city.city_id]
        analysis = _paired_analysis(
            city_rows,
            _seed(launch_seed, 192, city.scale_rank),
            draws,
        )
        for row in analysis:
            row["scope"] = "city"
            row["city_id"] = city.city_id
            row["city_scale_rank"] = city.scale_rank
        result.extend(analysis)
    return result


def _learning_assessment(
    convergence: dict,
    performance: dict,
    analysis: list[dict],
) -> dict:
    """Conservative separation of optimization, inference, and transfer claims."""
    macro = {
        row["metric"]: row
        for row in analysis
        if row.get("scope") == "macro_all_cities"
    }
    city_returns = [
        row
        for row in analysis
        if row.get("scope") == "city" and row.get("metric") == "episode_return"
    ]
    city_casualties = [
        row
        for row in analysis
        if row.get("scope") == "city" and row.get("metric") == "casualty"
    ]
    training_converged = bool(convergence.get("all_policies_converged", False))
    evaluated = "episode_return" in macro
    inferentially_eligible = bool(performance.get("inferentially_eligible", False))
    return_improvement = (
        float(macro["episode_return"]["mean_rl_improvement"])
        if evaluated
        else None
    )
    casualty_improvement = (
        float(macro["casualty"]["mean_rl_improvement"])
        if "casualty" in macro
        else None
    )
    all_city_returns_nonnegative = bool(
        city_returns
        and all(float(row["mean_rl_improvement"]) >= 0.0 for row in city_returns)
    )
    all_city_casualties_nonworsening = bool(
        city_casualties
        and all(float(row["mean_rl_improvement"]) >= 0.0 for row in city_casualties)
    )
    aggregate_casualty_nonworsening = bool(
        casualty_improvement is not None and casualty_improvement >= 0.0
    )

    if not training_converged:
        status = "training_not_converged"
    elif not evaluated:
        status = "training_converged_evaluation_not_opened"
    elif not inferentially_eligible:
        status = (
            "descriptively_promising_not_confirmed"
            if return_improvement is not None and return_improvement > 0.0
            else "learning_not_demonstrated"
        )
    else:
        status = str(performance.get("status", "inconclusive"))

    qualifies_as_learning_well = bool(
        status == "rl_superior"
        and all_city_returns_nonnegative
        and aggregate_casualty_nonworsening
        and all_city_casualties_nonworsening
    )
    return {
        "status": status,
        "training_converged": training_converged,
        "evaluation_available": evaluated,
        "inferentially_eligible": inferentially_eligible,
        "inference_scope": performance.get("inference_scope"),
        "mean_return_improvement": return_improvement,
        "mean_casualty_improvement": casualty_improvement,
        "all_city_return_point_estimates_nonnegative": all_city_returns_nonnegative,
        "aggregate_casualty_point_estimate_nonworsening": aggregate_casualty_nonworsening,
        "all_city_casualty_point_estimates_nonworsening": all_city_casualties_nonworsening,
        "qualifies_as_learning_well_cross_city": qualifies_as_learning_well,
        "criterion": (
            "A 'learning well cross-city' conclusion requires converged training, a "
            "superior return under the declared inference scope, nonnegative return "
            "point estimates in every city, and nonworsening aggregate and city "
            "casualty point estimates. With one policy, this conclusion is explicitly "
            "conditional on that frozen trained checkpoint."
        ),
    }


def _scale_trend_diagnostics(
    analysis: list[dict],
    cities: tuple[CityProfile, ...],
) -> list[dict]:
    """Describe whether city-level effects deteriorate with prespecified rank.

    Five deliberately selected sites do not support a powered population-level
    trend test, so this intentionally reports slopes and correlations without
    p-values or a pass/fail threshold.
    """
    city_rows = [row for row in analysis if row.get("scope") == "city"]
    metrics = sorted({str(row["metric"]) for row in city_rows})
    by_key = {(row["city_id"], row["metric"]): row for row in city_rows}
    ranks = np.asarray([city.scale_rank for city in cities], dtype=float)
    log_population = np.log10(
        np.asarray([city.census_2020_population for city in cities], dtype=float)
    )
    output = []
    for metric in metrics:
        effects = np.asarray(
            [float(by_key[(city.city_id, metric)]["mean_rl_improvement"]) for city in cities],
            dtype=float,
        )
        if len(cities) >= 2:
            rank_slope, rank_intercept = np.polyfit(ranks, effects, 1)
            population_slope, population_intercept = np.polyfit(
                log_population, effects, 1
            )
            rank_correlation = (
                float(np.corrcoef(ranks, effects)[0, 1])
                if float(effects.std()) > 0.0
                else 0.0
            )
        else:
            rank_slope = population_slope = rank_correlation = None
            rank_intercept = population_intercept = float(effects[0])
        output.append(
            {
                "metric": metric,
                "cities": len(cities),
                "effect_by_increasing_scale_rank": {
                    city.city_id: float(effect)
                    for city, effect in zip(cities, effects)
                },
                "slope_per_scale_rank": (
                    float(rank_slope) if rank_slope is not None else None
                ),
                "rank_intercept": float(rank_intercept),
                "pearson_correlation_with_scale_rank": rank_correlation,
                "slope_per_log10_census_population": (
                    float(population_slope) if population_slope is not None else None
                ),
                "log10_population_intercept": float(population_intercept),
                "interpretation": (
                    "descriptive only; five fixed cities, no inferential p-value"
                ),
            }
        )
    return output


def _preflight_maps(
    cities: tuple[CityProfile, ...],
    output_path: str,
    *,
    consolidate_intersections: bool = True,
    intersection_tolerance_m: float = 5.0,
    required_shelter_candidates: int | None = None,
) -> dict:
    import osmnx as ox
    from OSMProcessor import OSMProcessor

    ox.settings.cache_folder = os.path.join(PROJECT_ROOT, "cache")
    results = []
    for city in cities:
        status = "ready"
        error = None
        graph_nodes = graph_edges = stamped_nodes = 0
        cache_path = None
        try:
            processor = OSMProcessor(
                city.address,
                query_mode="point",
                center_point=(city.center_lat, city.center_lon),
                radius_m=city.radius_m,
                verbose=False,
            )
            processor.setLocationDrive()
            if consolidate_intersections and hasattr(
                processor, "consolidateIntersections"
            ):
                processor.consolidateIntersections(intersection_tolerance_m)
            processor.setNodeEdgeSets()
            processor.setIntersectionStreetCount()
            processor.setBuildingOnly()
            graph_nodes = int(processor.locationDrive.number_of_nodes())
            graph_edges = int(processor.locationDrive.number_of_edges())
            stamped_nodes = int(len(processor.buildingNodes))
            cache_path = processor._graph_cache_path()
            minimum_candidates = (
                1
                if required_shelter_candidates is None
                else int(required_shelter_candidates)
            )
            if (
                graph_nodes < 2
                or graph_edges < 1
                or stamped_nodes < minimum_candidates
            ):
                raise RuntimeError(
                    "map must contain a connected road graph and at least "
                    f"{minimum_candidates} stamped building/amenity nodes"
                )
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "city_id": city.city_id,
                "status": status,
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
                "stamped_building_or_amenity_nodes": stamped_nodes,
                "graph_cache_path": cache_path,
                "map_provenance": (
                    processor.graph_provenance() if status == "ready" else None
                ),
                "error": error,
            }
        )
        _json_dump(
            output_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "requested_cities": [item.city_id for item in cities],
                "intersection_consolidation": {
                    "enabled": bool(consolidate_intersections),
                    "tolerance_m": float(intersection_tolerance_m),
                },
                "required_shelter_candidates": required_shelter_candidates,
                "results": results,
                "complete": False,
                "all_ready": False,
            },
        )
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_cities": [item.city_id for item in cities],
        "intersection_consolidation": {
            "enabled": bool(consolidate_intersections),
            "tolerance_m": float(intersection_tolerance_m),
        },
        "required_shelter_candidates": required_shelter_candidates,
        "results": results,
        "complete": len(results) == len(cities),
        "all_ready": bool(
            len(results) == len(cities)
            and all(row["status"] == "ready" for row in results)
        ),
    }
    _json_dump(output_path, payload)
    return payload


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-profiles", default=DEFAULT_CITY_PROFILE_PATH)
    parser.add_argument(
        "--cities",
        default="all",
        help="'all' or comma-separated city_ids from the profile catalog",
    )
    parser.add_argument("--launch-seed", type=int, default=20260906)
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--machine", default="local")
    parser.add_argument(
        "--policy-replicates",
        type=int,
        default=1,
        help=(
            "Independent training seeds. The active campaign uses one frozen "
            "policy; specify more only for a separately labeled sensitivity run."
        ),
    )
    parser.add_argument(
        "--training-policy-indices",
        default=None,
        help=(
            "Optional comma-separated subset of the declared policy replications to "
            "train in this launch. This is a compute shard: it is valid only with "
            "--train-only and preserves the master launch seed, scenario corpus, and "
            "actual policy-replication seed indices."
        ),
    )
    parser.add_argument("--train-episodes-per-city", type=int, default=120)
    parser.add_argument(
        "--training-curriculum",
        default=DEFAULT_FULL_CURRICULUM,
        help=(
            "Optional validated JSON curriculum. Its stage episode counts must sum "
            "to --train-episodes-per-city."
        ),
    )
    parser.add_argument("--eval-replications-per-city", type=int, default=20)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_POOLED_LEARNING_RATE,
        help="Conservative pooled-PPO learning rate for the convergence-first policy.",
    )
    parser.add_argument(
        "--strategies",
        default="rl,risk_reduction,heuristic,hazard_weighted,accessibility_deficit,random",
    )
    parser.add_argument(
        "--hazard-mode",
        choices=("stochastic", "deterministic"),
        default="stochastic",
    )
    parser.add_argument(
        "--require-convergence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--convergence-min-episodes", type=int, default=100)
    parser.add_argument("--convergence-window-fraction", type=float, default=0.20)
    parser.add_argument("--convergence-trend-threshold", type=float, default=0.50)
    parser.add_argument("--convergence-shift-threshold", type=float, default=0.50)
    parser.add_argument("--visualize-eval-pairs-per-city", type=int, default=1)
    parser.add_argument("--visualization-strategies", default="rl,heuristic")
    parser.add_argument("--visualization-policy-replication", type=int, default=1)
    parser.add_argument("--visualization-milestones", default="quartiles")
    parser.add_argument("--override", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--validate-profiles-only", action="store_true")
    parser.add_argument("--preflight-maps-only", action="store_true")
    parser.add_argument(
        "--policy-cache-dir",
        default=DEFAULT_POLICY_CACHE,
        help="Content-addressed destination for completed training checkpoints.",
    )
    parser.add_argument(
        "--no-policy-cache",
        action="store_true",
        help="Do not publish completed policy checkpoints to the shared cache.",
    )
    args = parser.parse_args(argv)

    exclusive = sum(
        bool(value)
        for value in (
            args.eval_only,
            args.train_only,
            args.validate_profiles_only,
            args.preflight_maps_only,
        )
    )
    if exclusive > 1:
        parser.error("execution-mode flags are mutually exclusive")
    positive = (
        args.policy_replicates,
        args.train_episodes_per_city,
        args.eval_replications_per_city,
        args.bootstrap_draws,
        args.convergence_min_episodes,
    )
    if any(value <= 0 for value in positive):
        parser.error("replicate, episode, bootstrap, and convergence counts must be positive")
    if not math.isfinite(args.learning_rate) or not 0.0 < args.learning_rate <= 1e-2:
        parser.error("--learning-rate must be finite and in (0, 0.01]")
    if not 0.0 < args.convergence_window_fraction <= 0.5:
        parser.error("--convergence-window-fraction must be in (0, 0.5]")
    if args.convergence_trend_threshold < 0.0 or args.convergence_shift_threshold < 0.0:
        parser.error("convergence thresholds must be non-negative")
    if args.visualize_eval_pairs_per_city < 0:
        parser.error("--visualize-eval-pairs-per-city must be non-negative")
    if not 1 <= args.visualization_policy_replication <= args.policy_replicates:
        parser.error("invalid --visualization-policy-replication")
    strategies = tuple(item.strip().lower() for item in args.strategies.split(",") if item.strip())
    allowed = set(DYNAMIC_STRATEGIES).union(STATIC_STRATEGIES)
    invalid = sorted(set(strategies).difference(allowed))
    if invalid:
        parser.error(f"unsupported strategies: {invalid}")
    if not args.train_only and not args.validate_profiles_only and not args.preflight_maps_only:
        if not {"rl", "heuristic"}.issubset(strategies):
            parser.error("evaluation requires rl and heuristic")
    args.strategies = strategies
    args.visualization_strategies = tuple(
        item.strip().lower()
        for item in args.visualization_strategies.split(",")
        if item.strip()
    )
    invalid_visual = sorted(set(args.visualization_strategies).difference(allowed))
    if invalid_visual or set(args.visualization_strategies).difference(strategies):
        parser.error("visualization strategies must be valid evaluated strategies")
    if args.training_policy_indices is None:
        args.training_policy_indices = tuple(range(1, args.policy_replicates + 1))
    else:
        try:
            selected = tuple(
                int(item.strip())
                for item in str(args.training_policy_indices).split(",")
                if item.strip()
            )
        except ValueError:
            parser.error("--training-policy-indices must contain comma-separated integers")
        if not selected or len(selected) != len(set(selected)):
            parser.error("--training-policy-indices must be non-empty and unique")
        if any(index < 1 or index > args.policy_replicates for index in selected):
            parser.error("--training-policy-indices must lie within --policy-replicates")
        if not args.train_only:
            parser.error("--training-policy-indices is valid only with --train-only")
        args.training_policy_indices = tuple(sorted(selected))
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    suite = load_city_suite(args.city_profiles)
    requested = None if args.cities.strip().lower() == "all" else args.cities.split(",")
    cities = suite.select(requested)
    user_overrides = _parse_overrides(args.override)
    if "learningRate" in user_overrides:
        raise ValueError("Use --learning-rate instead of --override learningRate=...")
    if "ppoRolloutEpisodes" in user_overrides:
        raise ValueError(
            "ppoRolloutEpisodes is determined by the number of selected cities; "
            "do not set it with --override"
        )
    rollout_episodes = balanced_rollout_episodes(len(cities))
    overrides_by_city = {
        city.city_id: _city_overrides(suite, city, user_overrides, args.hazard_mode)
        for city in cities
    }
    for overrides in overrides_by_city.values():
        overrides["ppoRolloutEpisodes"] = rollout_episodes
        overrides["learningRate"] = float(args.learning_rate)
    training_curriculum = (
        load_training_curriculum(args.training_curriculum)
        if args.training_curriculum
        else None
    )
    evaluation_overrides_by_city = {
        city_id: {
            **overrides,
            **(
                training_curriculum.learner_overrides
                if training_curriculum is not None
                else {}
            ),
        }
        for city_id, overrides in overrides_by_city.items()
    }
    grid = _validate_shared_interface(evaluation_overrides_by_city)
    if (
        training_curriculum is not None
        and training_curriculum.episodes_per_city != int(args.train_episodes_per_city)
    ):
        raise ValueError(
            "Curriculum episodes per city "
            f"({training_curriculum.episodes_per_city}) do not match "
            f"--train-episodes-per-city ({args.train_episodes_per_city})"
        )
    map_or_profile_only = bool(
        args.validate_profiles_only or args.preflight_maps_only
    )
    schedule = (
        ()
        if map_or_profile_only
        else (
            build_curriculum_schedule(
                cities,
                training_curriculum,
                launch_seed=args.launch_seed,
                rollout_episodes=rollout_episodes,
            )
            if training_curriculum is not None
            else balanced_city_schedule(
                cities, args.train_episodes_per_city, args.launch_seed
            )
        )
    )
    total_train_episodes = len(cities) * int(args.train_episodes_per_city)
    if not map_or_profile_only and total_train_episodes % rollout_episodes != 0:
        raise ValueError(
            f"Total training episodes ({total_train_episodes}) must be a multiple of "
            f"the {rollout_episodes}-episode city-balanced PPO rollout"
        )

    launch_id = args.launch_id or f"multicity_backtest_seed_{args.launch_seed}"
    launch_dir = os.path.join(RUNS_ROOT, launch_id)
    os.makedirs(launch_dir, exist_ok=True)
    manifest_path = os.path.join(launch_dir, "experiment_manifest.json")
    dynamics_fingerprint = source_fingerprint(
        PROJECT_ROOT,
        POLICY_DYNAMICS_SOURCES,
    )
    resume_contract = {
        "city_profile_sha256": suite.source_sha256,
        "city_ids": [city.city_id for city in cities],
        "shared_grid": list(grid),
        "policy_replicates": int(args.policy_replicates),
        "training_policy_indices": list(args.training_policy_indices),
        "eval_replications_per_city": int(args.eval_replications_per_city),
        "ppo_rollout_episodes": int(rollout_episodes),
        "learning_rate": float(args.learning_rate),
        "effective_overrides_by_city": overrides_by_city,
        "training_curriculum": (
            None if training_curriculum is None else training_curriculum.as_dict()
        ),
        "policy_dynamics_source_sha256": dynamics_fingerprint,
    }
    previous_manifest = None
    continuation_provenance = None
    continuation_history = []
    if os.path.exists(manifest_path) and (args.resume or args.eval_only):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        continuation_history = list(
            previous_manifest.get("training_continuation_history", [])
        )
        previous_continuation = previous_manifest.get("training_continuation")
        if (
            isinstance(previous_continuation, dict)
            and (
                not continuation_history
                or continuation_history[-1] != previous_continuation
            )
        ):
            continuation_history.append(previous_continuation)
        previous_contract = {
            key: previous_manifest.get(key) for key in resume_contract
        }
        # Launches created before compute-shard support implicitly trained the
        # full declared policy range. Preserve exact backward-compatible resume
        # without weakening any scientific configuration field.
        if previous_contract.get("training_policy_indices") is None:
            previous_contract["training_policy_indices"] = list(
                range(1, int(previous_manifest.get("policy_replicates", 0)) + 1)
            )
        curriculum_extension = _curriculum_is_monotone_extension(
            previous_contract.get("training_curriculum"),
            resume_contract.get("training_curriculum"),
        )
        mismatches = {}
        for key, value in resume_contract.items():
            if previous_contract.get(key) == value:
                continue
            if key == "training_curriculum" and curriculum_extension:
                continue
            mismatches[key] = {
                "previous": previous_contract.get(key),
                "requested": value,
            }
        previous_train_per_city = int(
            previous_manifest.get("train_episodes_per_city", 0)
        )
        if int(args.train_episodes_per_city) < previous_train_per_city:
            mismatches["train_episodes_per_city"] = {
                "previous": previous_train_per_city,
                "requested": int(args.train_episodes_per_city),
                "reason": "training continuation cannot contract the episode target",
            }
        if mismatches:
            raise ValueError(
                "Resume/evaluation contract differs from the recorded experiment: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if int(args.train_episodes_per_city) > previous_train_per_city:
            if not curriculum_extension:
                raise ValueError(
                    "Training continuation requires a monotone curriculum extension "
                    "that preserves the complete recorded schedule prefix"
                )
            continuation_provenance = {
                "reason": "predeclared_stationarity_audit_not_passed",
                "previous_status": previous_manifest.get("status"),
                "previous_train_episodes_per_city": previous_train_per_city,
                "requested_train_episodes_per_city": int(
                    args.train_episodes_per_city
                ),
                "previous_completed_utc": previous_manifest.get("completed_utc"),
                "optimizer_and_rng_state_resumed": True,
                "convergence_thresholds_unchanged": True,
            }
    elif (
        os.path.exists(manifest_path)
        and not args.validate_profiles_only
        and not args.preflight_maps_only
    ):
        raise FileExistsError(
            "Experiment manifest already exists; use --resume/--eval-only or a new --launch-id"
        )
    profile_snapshot_path = os.path.join(launch_dir, "city_profile_snapshot.json")
    _json_dump(
        profile_snapshot_path,
        {
            "schema_version": suite.schema_version,
            "source_path": suite.source_path,
            "source_sha256": suite.source_sha256,
            "selection_basis": suite.selection_basis,
            "common_experiment": suite.common_experiment,
            "selected_cities": [city.as_dict() for city in cities],
        },
    )
    if args.validate_profiles_only:
        print(f"[CITY PROFILES VALID] snapshot={profile_snapshot_path}", flush=True)
        return 0
    if args.preflight_maps_only:
        preflight_path = os.path.join(launch_dir, "map_preflight.json")
        result = _preflight_maps(
            cities,
            preflight_path,
            consolidate_intersections=bool(
                suite.common_experiment.get("intersectionConsolidationEnabled", True)
            ),
            intersection_tolerance_m=float(
                suite.common_experiment.get("intersectionConsolidationToleranceM", 5.0)
            ),
            required_shelter_candidates=int(
                suite.common_experiment["shelterCanVol"]
            ),
        )
        print(f"[MAP PREFLIGHT] all_ready={result['all_ready']} artifact={preflight_path}")
        return 0 if result["all_ready"] else 2

    # Heavy learning imports are intentionally delayed until after the
    # profile/map-only exits.  This keeps map preflight independent of PyTorch
    # and avoids initializing two OpenMP runtimes in a non-learning command.
    from backtest import (
        _dependency_versions,
        _git_metadata,
        _performance_assessment,
        _plot_outputs,
        _read_csv,
        _run_episode,
        _verify_matched_interface,
        _write_csv,
        _write_paper_table,
    )
    from DecisionInterface import (
        BENCHMARK_POLICY_CONTRACTS,
        CANDIDATE_FEATURE_NAMES,
        CELL_FEATURE_NAMES,
        GLOBAL_FEATURE_NAMES,
        MOMENTUM_FEATURE_NAMES,
    )
    from GNN import HEURISTIC_PRIOR_SCALE, RESIDUAL_LOGIT_BOUND
    from RLBridge import DEFAULT_ENTROPY_COEF, DEFAULT_ROLLOUT_EPISODES

    if int(DEFAULT_ROLLOUT_EPISODES) != MINIMUM_ROLLOUT_EPISODES:
        raise RuntimeError(
            "multicity runner and RL bridge disagree on the minimum PPO rollout length"
        )

    training_summary_path = os.path.join(launch_dir, "training_episode_summary.csv")
    evaluation_summary_path = os.path.join(launch_dir, "evaluation_episode_summary.csv")
    checkpoint_paths = {
        policy: os.path.join(
            launch_dir, "policies", f"policy_{policy:03d}", "regional_policy.pt"
        )
        for policy in range(1, args.policy_replicates + 1)
    }
    diagnostics_paths = {
        policy: os.path.join(
            launch_dir, "policies", f"policy_{policy:03d}", "ppo_diagnostics.csv"
        )
        for policy in range(1, args.policy_replicates + 1)
    }
    preexisting = [path for path in checkpoint_paths.values() if os.path.exists(path)]
    if preexisting and not args.resume and not args.eval_only:
        raise FileExistsError("Checkpoints exist; use --resume or a new --launch-id")

    cache_events = {}
    restored_training_rows = []
    if not args.no_policy_cache and not args.resume and not args.eval_only:
        policy_cache = PolicyCache(args.policy_cache_dir)
        for policy_replication in args.training_policy_indices:
            contract = _policy_cache_contract(
                resume_contract,
                train_episodes_per_city=args.train_episodes_per_city,
                total_train_episodes=total_train_episodes,
                policy_replication=policy_replication,
                launch_seed=args.launch_seed,
            )
            event = policy_cache.restore(
                contract,
                checkpoint_paths[policy_replication],
                required_metadata={
                    "training_episode_count": int(total_train_episodes),
                    "converged": True,
                },
            )
            if event is None:
                continue
            cached_rows = event["metadata"].get("training_rows", [])
            if (
                len(cached_rows) != total_train_episodes
                or any(
                    int(row.get("policy_replication", -1)) != policy_replication
                    for row in cached_rows
                )
            ):
                raise RuntimeError("Policy cache contains an invalid training ledger")
            restored_training_rows.extend(cached_rows)
            cache_events[str(policy_replication)] = {
                key: value for key, value in event.items() if key != "metadata"
            }

    missing = [path for path in checkpoint_paths.values() if not os.path.exists(path)]
    if args.eval_only and missing:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {missing[0]}")
    if os.path.exists(evaluation_summary_path):
        raise FileExistsError(
            "Held-out evaluation has already been opened for this launch. Training "
            "may not resume and evaluation may not be overwritten; use a new launch id."
        )

    current_command = [
        sys.executable,
        os.path.abspath(__file__),
        *(argv or sys.argv[1:]),
    ]
    manifest = {
        "schema_version": 1,
        "status": "running",
        **_manifest_time_fields(previous_manifest),
        **_manifest_command_fields(previous_manifest, current_command),
        "launch_id": launch_id,
        "launch_seed": int(args.launch_seed),
        # Use the exact same fields for manifest persistence and resume
        # validation so a new contract field cannot be validated but omitted.
        **resume_contract,
        "city_scale_order": [city.as_dict() for city in cities],
        "shared_action_count": int(suite.common_experiment["shelterCanVol"]),
        "train_episodes_per_city": int(args.train_episodes_per_city),
        "total_train_episodes_per_policy": int(total_train_episodes),
        "strategies": list(args.strategies),
        "benchmark_policy_contracts": {
            strategy: BENCHMARK_POLICY_CONTRACTS[strategy]
            for strategy in args.strategies
            if strategy in BENCHMARK_POLICY_CONTRACTS
        },
        "common_overrides": user_overrides,
        "training_city_schedule": [
            (
                scheduled.manifest_dict()
                if isinstance(scheduled, CurriculumEpisode)
                else scheduled.city_id
            )
            for scheduled in schedule
        ],
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
        "policy_design": {
            "pooled_across_cities": bool(len(cities) > 1),
            "city_identifier_observed_by_policy": False,
            "actor": (
                "shared bounded active-population-prior residual exact-candidate "
                "scorer with relational graph context and an episode-level LSTM"
            ),
            "action_space": "exact feasible shelter candidate",
            "heuristic_prior_logit_scale": float(HEURISTIC_PRIOR_SCALE),
            "residual_logit_bound": float(RESIDUAL_LOGIT_BOUND),
            "decision_interval_timesteps": int(
                next(iter(overrides_by_city.values()))["shelterActionInterval"]
            ),
            "transition_boundary": "next_executed_action_or_true_environment_terminal",
            "final_action_accounting": "complete_to_environment_terminal",
            "recurrent_sequence_sampling": "whole_episode",
            "critic_factorization": "safe_casualty_time_exposure_heads",
            "rollout_episodes": int(rollout_episodes),
            "rollout_city_blocks": int(rollout_episodes // len(cities)),
            "every_optimizer_batch_is_city_balanced": True,
            "learning_rate": float(args.learning_rate),
            "entropy_coefficient": float(DEFAULT_ENTROPY_COEF),
            "entropy_normalization": "log_feasible_action_count",
            "kl_control": "full_rollout_epoch_stop_and_adaptive_learning_rate",
        },
        "estimand": {
            "primary": "equal-city macro mean RL improvement over heuristic",
            "cities_fixed_not_resampled": True,
            "policy_seeds_resampled": bool(args.policy_replicates > 1),
            "scenarios_resampled_within_city": True,
            "inference_scope": (
                "conditional_on_one_fixed_trained_policy"
                if args.policy_replicates == 1
                else "joint_over_policy_and_scenario_seeds"
            ),
        },
        "cell_observation_features": list(CELL_FEATURE_NAMES),
        "global_observation_features": list(GLOBAL_FEATURE_NAMES),
        "momentum_observation_features": list(MOMENTUM_FEATURE_NAMES),
        "candidate_observation_features": list(CANDIDATE_FEATURE_NAMES),
        "git": _git_metadata(),
        "python": {"version": sys.version, "executable": sys.executable},
        "dependencies": _dependency_versions(),
        "policy_cache": {
            "enabled": not bool(args.no_policy_cache),
            "root": os.path.abspath(args.policy_cache_dir),
            "events": cache_events,
        },
    }
    if continuation_provenance is not None:
        manifest["training_continuation"] = continuation_provenance
        continuation_history.append(continuation_provenance)
    if continuation_history:
        manifest["training_continuation_history"] = continuation_history
    _json_dump(manifest_path, manifest)

    training_rows = (
        _read_csv(training_summary_path)
        if (args.resume or args.eval_only)
        else list(restored_training_rows)
    )
    if restored_training_rows:
        _write_csv(training_summary_path, training_rows)
    _validate_resume_rows(training_rows, schedule, args.policy_replicates)
    if not args.eval_only:
        for policy_replication in args.training_policy_indices:
            completed = sum(
                int(row.get("policy_replication", 1)) == policy_replication
                for row in training_rows
            )
            policy_seed = _seed(args.launch_seed, 10, policy_replication)
            city_occurrence = {city.city_id: 0 for city in cities}
            for scheduled in schedule[:completed]:
                city = (
                    scheduled.city
                    if isinstance(scheduled, CurriculumEpisode)
                    else scheduled
                )
                city_occurrence[city.city_id] += 1
            for episode in range(completed + 1, total_train_episodes + 1):
                scheduled = schedule[episode - 1]
                city = (
                    scheduled.city
                    if isinstance(scheduled, CurriculumEpisode)
                    else scheduled
                )
                city_occurrence[city.city_id] += 1
                scenario_seed = _seed(
                    args.launch_seed,
                    100 + city.scale_rank,
                    city_occurrence[city.city_id],
                )
                print(
                    f"[TRAIN] policy={policy_replication}/{args.policy_replicates} "
                    f"episode={episode}/{total_train_episodes} city={city.city_id}"
                    + (
                        f" stage={scheduled.stage_id} variant={scheduled.variant_id} "
                        f"population={scheduled.stage_overrides['pedVol']}"
                        if isinstance(scheduled, CurriculumEpisode)
                        else ""
                    ),
                    flush=True,
                )
                episode_overrides = dict(overrides_by_city[city.city_id])
                episode_overrides["finalizePpoRollout"] = bool(
                    episode == total_train_episodes
                )
                if isinstance(scheduled, CurriculumEpisode):
                    episode_overrides.update(scheduled.stage_overrides)
                phase_parts = [
                    launch_id,
                    "training",
                    f"policy_{policy_replication:03d}",
                ]
                if isinstance(scheduled, CurriculumEpisode):
                    phase_parts.extend((scheduled.stage_id, scheduled.variant_id))
                phase_parts.append(city.city_id)
                row = _run_episode(
                    replication=episode,
                    machine=args.machine,
                    phase=os.path.join(*phase_parts),
                    strategy="rl",
                    train_mode=True,
                    scenario_seed=scenario_seed,
                    policy_seed=policy_seed,
                    checkpoint_path=checkpoint_paths[policy_replication],
                    diagnostics_path=diagnostics_paths[policy_replication],
                    overrides=episode_overrides,
                )
                training_metadata = {
                    "policy_replication": policy_replication,
                    "city_id": city.city_id,
                    "city_scale_rank": city.scale_rank,
                    "city_training_replication": city_occurrence[city.city_id],
                }
                if isinstance(scheduled, CurriculumEpisode):
                    training_metadata.update(
                        {
                            "training_stage_id": scheduled.stage_id,
                            "training_stage_label": scheduled.stage_label,
                            "training_variant_id": scheduled.variant_id,
                            "stage_index": scheduled.stage_index,
                            "stage_replication": scheduled.stage_replication,
                            "stage_city_replication": scheduled.stage_city_replication,
                            "stage_population": int(scheduled.stage_overrides["pedVol"]),
                            "stage_overrides": json.dumps(
                                dict(scheduled.stage_overrides),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
                row.update(training_metadata)
                training_rows.append(row)
                _write_csv(training_summary_path, training_rows)

    convergence = _multicity_training_convergence(
        training_rows,
        cities,
        minimum_episodes=args.convergence_min_episodes,
        window_fraction=args.convergence_window_fraction,
        trend_threshold=args.convergence_trend_threshold,
        shift_threshold=args.convergence_shift_threshold,
        target_kl=DEFAULT_TARGET_KL,
    )
    convergence_path = os.path.join(launch_dir, "training_convergence_diagnostics.json")
    _json_dump(convergence_path, convergence)
    plots = _plot_outputs(launch_dir, training_rows, [])
    if args.require_convergence and not convergence["all_policies_converged"]:
        manifest.update(
            {
                "status": "training_not_converged",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "training_convergence": convergence,
            }
        )
        _json_dump(manifest_path, manifest)
        raise RuntimeError("Pooled policies did not pass the training-only convergence audit")

    # Only scientifically usable policies enter the content-addressed cache.
    # A completed episode count alone is insufficient: restoring a stationary-
    # failure checkpoint would silently poison future campaigns.
    if (
        not args.no_policy_cache
        and not args.eval_only
        and convergence["all_policies_converged"]
    ):
        policy_cache = PolicyCache(args.policy_cache_dir)
        published_cache_events = dict(manifest["policy_cache"]["events"])
        for policy_replication in args.training_policy_indices:
            completed = sum(
                int(row.get("policy_replication", 1)) == policy_replication
                for row in training_rows
            )
            if completed != total_train_episodes:
                continue
            policy_contract = _policy_cache_contract(
                resume_contract,
                train_episodes_per_city=args.train_episodes_per_city,
                total_train_episodes=total_train_episodes,
                policy_replication=policy_replication,
                launch_seed=args.launch_seed,
            )
            policy_rows = [
                _strict_json_value(row)
                for row in training_rows
                if int(row.get("policy_replication", 1)) == policy_replication
            ]
            published_cache_events[str(policy_replication)] = policy_cache.store(
                policy_contract,
                checkpoint_paths[policy_replication],
                metadata={
                    "training_episode_count": int(total_train_episodes),
                    "training_rows": policy_rows,
                    "converged": True,
                    "convergence_policy": next(
                        row
                        for row in convergence["policies"]
                        if int(row["policy_replication"]) == policy_replication
                    ),
                },
            )
        manifest["policy_cache"]["events"] = published_cache_events
        _json_dump(manifest_path, manifest)

    evaluation_rows = []
    analysis = []
    scale_trends = []
    parity = None
    performance = {"status": "not_evaluated"}
    if not args.train_only:
        global_replication = 0
        for city in cities:
            for city_replication in range(1, args.eval_replications_per_city + 1):
                global_replication += 1
                scenario_seed = _seed(args.launch_seed, 500 + city.scale_rank, city_replication)
                evaluation_policy_seed = _seed(
                    args.launch_seed, 600 + city.scale_rank, city_replication
                )
                for strategy in args.strategies:
                    policy_indices = (
                        range(1, args.policy_replicates + 1)
                        if strategy in LEARNED_STRATEGIES
                        else (0,)
                    )
                    for policy_replication in policy_indices:
                        visualize = bool(
                            city_replication <= args.visualize_eval_pairs_per_city
                            and strategy in args.visualization_strategies
                            and (
                                strategy not in LEARNED_STRATEGIES
                                or policy_replication == args.visualization_policy_replication
                            )
                        )
                        checkpoint_index = (
                            policy_replication
                            if strategy in LEARNED_STRATEGIES
                            else 1
                        )
                        print(
                            f"[EVAL] city={city.city_id} pair={city_replication}/"
                            f"{args.eval_replications_per_city} strategy={strategy} "
                            f"policy={policy_replication}",
                            flush=True,
                        )
                        row = _run_episode(
                            replication=global_replication,
                            machine=args.machine,
                            phase=os.path.join(
                                launch_id,
                                "evaluation",
                                city.city_id,
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
                            overrides=evaluation_overrides_by_city[city.city_id],
                            visualization_enabled=visualize,
                            visualization_milestones=args.visualization_milestones,
                        )
                        row.update(
                            {
                                "policy_replication": policy_replication,
                                "city_id": city.city_id,
                                "city_scale_rank": city.scale_rank,
                                "city_scenario_replication": city_replication,
                            }
                        )
                        evaluation_rows.append(row)
        _write_csv(evaluation_summary_path, evaluation_rows)
        parity = _verify_matched_interface(evaluation_rows, args.strategies)
        parity["cities"] = [city.city_id for city in cities]
        _json_dump(os.path.join(launch_dir, "interface_parity.json"), parity)
        analysis = _all_paired_analysis(
            evaluation_rows, cities, args.launch_seed, args.bootstrap_draws
        )
        _write_csv(os.path.join(launch_dir, "paired_comparison_by_city.csv"), analysis)
        macro = [row for row in analysis if row["scope"] == "macro_all_cities"]
        _write_paper_table(os.path.join(launch_dir, "paired_comparison_macro.md"), macro)
        performance = _performance_assessment(
            macro,
            allow_fixed_policy=bool(args.policy_replicates == 1),
        )
        performance["estimand"] = "fixed-site equal-city macro average"
        _json_dump(os.path.join(launch_dir, "performance_assessment.json"), performance)
        scale_trends = _scale_trend_diagnostics(analysis, cities)
        _json_dump(os.path.join(launch_dir, "scale_trend_diagnostics.json"), {
            "schema_version": 1,
            "scale_definition": "prespecified rank by 2020 Census municipal population",
            "inferential": False,
            "diagnostics": scale_trends,
        })
        plots = _plot_outputs(launch_dir, training_rows, macro)

    learning_assessment = _learning_assessment(
        convergence,
        performance,
        analysis,
    )
    learning_assessment_path = os.path.join(
        launch_dir,
        "learning_assessment.json",
    )
    _json_dump(learning_assessment_path, learning_assessment)

    is_training_shard = bool(
        args.train_only
        and tuple(args.training_policy_indices)
        != tuple(range(1, args.policy_replicates + 1))
    )
    manifest.update(
        {
            "status": "training_shard_complete" if is_training_shard else "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "training_convergence": convergence,
            "interface_verification": parity,
            "performance_assessment": performance,
            "learning_assessment": learning_assessment,
            "artifacts": {
                "city_profile_snapshot": profile_snapshot_path,
                "training_summary": training_summary_path,
                "ppo_diagnostics": diagnostics_paths,
                "training_convergence": convergence_path,
                "learning_assessment": learning_assessment_path,
                "evaluation_summary": evaluation_summary_path if evaluation_rows else None,
                "paired_comparison": (
                    os.path.join(launch_dir, "paired_comparison_by_city.csv")
                    if analysis
                    else None
                ),
                "paper_table": (
                    os.path.join(launch_dir, "paired_comparison_macro.md")
                    if analysis
                    else None
                ),
                "scale_trend_diagnostics": (
                    os.path.join(launch_dir, "scale_trend_diagnostics.json")
                    if scale_trends
                    else None
                ),
                "plots": plots,
            },
        }
    )
    _json_dump(manifest_path, manifest)
    print(f"[COMPLETE] artifacts={launch_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
