#!/usr/bin/env python3
"""Matched 5x5 population/shelter-candidate scale-stress backtest.

The experiment evaluates trained regional policies and the active-population
heuristic on identical stochastic scenarios.  It varies only the initialized
pedestrian population and the number of sampled OSM shelter candidates.  A
60-transition horizon is implemented as ``stopTime=61`` because the
simulator advances over ``range(1, stopTime)``.

The full design is expensive (5 cities x 25 factor cells x 5 scenarios, with
eight RL policies plus one heuristic).  ``--pilot`` executes one matched pair
at the smallest factor cell to validate runtime and accounting without
mislabeling it as full evidence; ``--dry-run`` only writes the execution plan.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

from CityProfiles import DEFAULT_CITY_PROFILE_PATH, load_city_suite
from ExperimentSuite import DEFAULT_EXPERIMENT_SUITE_PATH, load_experiment_suite
from multicity_backtest import RUNS_ROOT, _city_overrides


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_LAUNCH = (
    PROJECT_ROOT / "runs" / "multicity_five_city_lr1e3_learning_audit_arm_20260906"
)
STRATEGIES = ("rl", "heuristic")


def _strict(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_strict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = (
        "city_id",
        "city_scale_rank",
        "population_level",
        "shelter_candidate_level",
        "horizon_timesteps",
        "shelter_action_interval_timesteps",
        "scale_replication",
        "policy_replication",
        "deployment_strategy",
        "scenario_seed",
        "policy_seed",
        "episode_return",
        "objective_episode_return",
        "safe_completed",
        "casualty",
        "unfinished",
        "restricted_mean_time_to_safety",
        "normalized_risk_weighted_person_time",
        "initial_population",
        "actual_candidate_count",
        "simulation_runtime_s",
    )
    fields = {str(key) for row in rows for key in row}
    fieldnames = [field for field in preferred if field in fields]
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


def _condition_seed(
    launch_seed: int,
    city_rank: int,
    population_index: int,
    candidate_index: int,
    replication: int,
    stream: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(launch_seed),
            int(stream),
            int(city_rank),
            int(population_index),
            int(candidate_index),
            int(replication),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _checkpoint_paths(source_launch: Path, policy_count: int) -> dict[int, Path]:
    return {
        policy: source_launch
        / "policies"
        / f"policy_{policy:03d}"
        / "regional_policy.pt"
        for policy in range(1, int(policy_count) + 1)
    }


def _source_training_horizons(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> dict[str, int]:
    """Recover realized transition horizons from a training manifest."""
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    horizons = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping) or "stopTime" not in overrides:
            raise ValueError(
                f"Source experiment manifest has no stopTime for city {city_id!r}"
            )
        stop_time = int(overrides["stopTime"])
        if stop_time <= 1:
            raise ValueError(
                f"Source experiment stopTime must exceed one for city {city_id!r}"
            )
        horizons[str(city_id)] = stop_time - 1
    return horizons


def _source_training_action_intervals(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> dict[str, int]:
    """Recover cadence, assigning the documented legacy default of five."""
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    intervals = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Source manifest lacks overrides for city {city_id!r}")
        intervals[str(city_id)] = int(overrides.get("shelterActionInterval", 5))
    return intervals


def _source_training_installation_budgets(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> dict[str, int]:
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    budgets = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Source manifest lacks overrides for city {city_id!r}")
        budgets[str(city_id)] = int(overrides.get("maxAdditionalShelters", 0))
    return budgets


def _map_contract(overrides: Mapping) -> dict:
    center = None
    if overrides.get("mapCenterLat") is not None and overrides.get("mapCenterLon") is not None:
        center = [float(overrides["mapCenterLat"]), float(overrides["mapCenterLon"])]
    return {
        "query_mode": str(overrides.get("mapQueryMode", "place")),
        "center": center,
        "radius_m": (
            None if overrides.get("mapRadiusM") is None else float(overrides["mapRadiusM"])
        ),
        "grid": [int(overrides.get("cellX", 0)), int(overrides.get("cellY", 0))],
        "cell_partition": {
            "mode": str(
                overrides.get("cellPartitionMode", "node_density_adaptive")
            ),
            "minimum_axis_width_fraction": float(
                overrides.get("cellPartitionMinWidthFraction", 1e-4)
            ),
        },
    }


def _congestion_contract(overrides: Mapping) -> dict:
    enabled = bool(overrides.get("congestionEnabled", False))
    return {
        "time_step_minutes": float(overrides.get("timeStepMinutes", 1.0)),
        "enabled": enabled,
        "model": "weidmann_physical_link_v1" if enabled else "none",
        "effective_width_m": (
            float(overrides.get("congestionEffectiveWidthM", 3.0)) if enabled else None
        ),
        "jam_density_ped_per_m2": (
            float(overrides.get("congestionJamDensityPedPerM2", 5.4)) if enabled else None
        ),
        "shape": float(overrides.get("congestionShape", 1.913)) if enabled else None,
        "minimum_speed_ratio": (
            float(overrides.get("congestionMinimumSpeedRatio", 0.05)) if enabled else None
        ),
        "integration_substep_seconds": (
            float(overrides.get("congestionSubstepSeconds", 10.0)) if enabled else None
        ),
    }


def _source_training_environment_contracts(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    maps = {}
    congestion = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Source manifest lacks overrides for city {city_id!r}")
        maps[str(city_id)] = _map_contract(overrides)
        congestion[str(city_id)] = _congestion_contract(overrides)
    return maps, congestion


def _row_identity(row: Mapping) -> tuple:
    return (
        str(row["city_id"]),
        int(float(row["population_level"])),
        int(float(row["shelter_candidate_level"])),
        int(float(row["scale_replication"])),
        str(row["deployment_strategy"]),
        int(float(row.get("policy_replication", 0))),
    )


def build_execution_plan(
    *,
    cities,
    populations: Sequence[int],
    candidates: Sequence[int],
    replications: int,
    policy_count: int,
    horizon_timesteps: int,
    shelter_action_interval: int = 2,
) -> dict:
    scenario_cells = len(cities) * len(populations) * len(candidates) * int(replications)
    episodes_per_cell = int(policy_count) + 1
    return {
        "city_ids": [city.city_id for city in cities],
        "population_levels": list(populations),
        "shelter_candidate_levels": list(candidates),
        "horizon_timesteps": int(horizon_timesteps),
        "shelter_action_interval": int(shelter_action_interval),
        "simulator_stop_time": int(horizon_timesteps) + 1,
        "replications_per_city_factor_cell": int(replications),
        "policy_replications": int(policy_count),
        "factor_cells_per_city": len(populations) * len(candidates),
        "matched_scenario_cells": scenario_cells,
        "episodes_per_scenario_cell": episodes_per_cell,
        "total_episodes": scenario_cells * episodes_per_cell,
        "total_requested_pedestrian_trajectories": int(
            len(cities)
            * len(candidates)
            * int(replications)
            * episodes_per_cell
            * sum(populations)
        ),
    }


def _validate_matched_rows(rows: Sequence[Mapping], policy_count: int) -> dict:
    """Verify interface and exogenous-path parity in every completed cell."""
    groups: dict[tuple, list[Mapping]] = {}
    for row in rows:
        key = (
            str(row["city_id"]),
            int(float(row["population_level"])),
            int(float(row["shelter_candidate_level"])),
            int(float(row["scale_replication"])),
        )
        groups.setdefault(key, []).append(row)
    complete = 0
    checks = []
    for key, group in sorted(groups.items()):
        heuristic = [row for row in group if row["deployment_strategy"] == "heuristic"]
        rl = [row for row in group if row["deployment_strategy"] == "rl"]
        expected_rl = int(policy_count)
        group_complete = len(heuristic) == 1 and len(rl) == expected_rl
        mismatches = []
        if group_complete:
            reference = heuristic[0]
            exact_fields = (
                "scenario_seed",
                "initial_population",
                "initial_observation_digest",
                "hazard_trajectory_digest",
                "maximum_dynamic_deployments",
                "actual_candidate_count",
                "horizon_timesteps",
                "shelter_action_interval_timesteps",
            )
            for candidate in rl:
                for field in exact_fields:
                    if str(candidate.get(field)) != str(reference.get(field)):
                        mismatches.append(
                            {
                                "policy_replication": candidate.get("policy_replication"),
                                "field": field,
                                "rl": candidate.get(field),
                                "heuristic": reference.get(field),
                            }
                        )
            group_complete = not mismatches
        if group_complete:
            complete += 1
        checks.append(
            {
                "cell": list(key),
                "complete": group_complete,
                "rl_rows": len(rl),
                "heuristic_rows": len(heuristic),
                "mismatches": mismatches,
            }
        )
    return {
        "verified": bool(groups) and complete == len(groups),
        "completed_matched_cells": complete,
        "observed_cells": len(groups),
        "checks": checks,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", type=Path, default=DEFAULT_EXPERIMENT_SUITE_PATH)
    parser.add_argument("--city-profiles", default=DEFAULT_CITY_PROFILE_PATH)
    parser.add_argument("--source-launch-dir", type=Path, default=DEFAULT_SOURCE_LAUNCH)
    parser.add_argument("--launch-id", default="population_candidate_scale_seed_20260908")
    parser.add_argument("--launch-seed", type=int, default=20260908)
    parser.add_argument("--cities", default="all")
    parser.add_argument("--machine", default="scale")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run one policy, one city, one replication, and the smallest 5x5 cell.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render the first matched RL/heuristic pair only.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    experiment = load_experiment_suite(args.suite_config)
    city_suite = load_city_suite(args.city_profiles)
    requested = None if args.cities.strip().lower() == "all" else args.cities.split(",")
    cities = city_suite.select(requested)
    populations = experiment.population_levels
    candidates = experiment.shelter_candidate_levels
    replications = experiment.scale_replications_per_cell
    policy_count = experiment.policy_seeds
    if args.pilot:
        cities = cities[:1]
        populations = populations[:1]
        candidates = candidates[:1]
        replications = 1
        policy_count = 1
    plan = build_execution_plan(
        cities=cities,
        populations=populations,
        candidates=candidates,
        replications=replications,
        policy_count=policy_count,
        horizon_timesteps=experiment.scale_horizon_timesteps,
        shelter_action_interval=experiment.scale_shelter_action_interval,
    )
    source_launch = args.source_launch_dir.expanduser().resolve()
    if not source_launch.is_dir():
        raise FileNotFoundError(source_launch)
    source_manifest_path = source_launch / "experiment_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if source_manifest.get("status") != "complete":
        raise ValueError(
            f"Source training launch is not complete: {source_manifest_path}"
        )
    source_horizons = _source_training_horizons(
        source_manifest,
        [city.city_id for city in cities],
    )
    source_action_intervals = _source_training_action_intervals(
        source_manifest,
        [city.city_id for city in cities],
    )
    source_installation_budgets = _source_training_installation_budgets(
        source_manifest,
        [city.city_id for city in cities],
    )
    source_map_contracts, source_congestion_contracts = (
        _source_training_environment_contracts(
            source_manifest,
            [city.city_id for city in cities],
        )
    )
    expected_overrides = {
        city.city_id: _city_overrides(city_suite, city, {}, "stochastic")
        for city in cities
    }
    expected_map_contracts = {
        city_id: _map_contract(overrides)
        for city_id, overrides in expected_overrides.items()
    }
    expected_congestion_contracts = {
        city_id: _congestion_contract(overrides)
        for city_id, overrides in expected_overrides.items()
    }
    source_policy_replicates = int(source_manifest.get("policy_replicates", 0))
    source_training_converged = bool(
        source_manifest.get("training_convergence", {}).get(
            "all_policies_converged", False
        )
    )
    horizon_mismatches = {
        city_id: source_horizon
        for city_id, source_horizon in source_horizons.items()
        if source_horizon != int(experiment.scale_horizon_timesteps)
    }
    action_interval_mismatches = {
        city_id: source_interval
        for city_id, source_interval in source_action_intervals.items()
        if source_interval != int(experiment.scale_shelter_action_interval)
    }
    installation_budget_mismatches = {
        city_id: source_budget
        for city_id, source_budget in source_installation_budgets.items()
        if source_budget != int(
            city_suite.common_experiment.get("maxAdditionalShelters", 0)
        )
    }
    map_contract_mismatches = {
        city_id: {
            "source": source_map_contracts[city_id],
            "evaluation": expected_map_contracts[city_id],
        }
        for city_id in source_map_contracts
        if source_map_contracts[city_id] != expected_map_contracts[city_id]
    }
    congestion_contract_mismatches = {
        city_id: {
            "source": source_congestion_contracts[city_id],
            "evaluation": expected_congestion_contracts[city_id],
        }
        for city_id in source_congestion_contracts
        if source_congestion_contracts[city_id]
        != expected_congestion_contracts[city_id]
    }
    checkpoints = _checkpoint_paths(source_launch, policy_count)
    missing_checkpoints = [str(path) for path in checkpoints.values() if not path.exists()]
    plan["source_launch"] = str(source_launch)
    plan["checkpoint_paths"] = {str(key): str(value) for key, value in checkpoints.items()}
    plan["missing_checkpoints"] = missing_checkpoints
    plan["mode"] = "pilot" if args.pilot else "full_confirmatory"
    plan["source_training_manifest"] = str(source_manifest_path)
    plan["source_training_manifest_sha256"] = _sha256(source_manifest_path)
    plan["source_training_horizon_by_city"] = source_horizons
    plan["source_training_horizon_mismatches"] = horizon_mismatches
    plan["confirmatory_source_horizon_compatible"] = not horizon_mismatches
    plan["source_training_action_interval_by_city"] = source_action_intervals
    plan["source_training_action_interval_mismatches"] = action_interval_mismatches
    plan["confirmatory_source_action_interval_compatible"] = not action_interval_mismatches
    plan["source_training_installation_budget_by_city"] = source_installation_budgets
    plan["source_training_installation_budget_mismatches"] = installation_budget_mismatches
    plan["source_training_map_contract_by_city"] = source_map_contracts
    plan["evaluation_map_contract_by_city"] = expected_map_contracts
    plan["source_training_map_contract_mismatches"] = map_contract_mismatches
    plan["source_training_congestion_contract_by_city"] = source_congestion_contracts
    plan["evaluation_congestion_contract_by_city"] = expected_congestion_contracts
    plan["source_training_congestion_contract_mismatches"] = congestion_contract_mismatches
    plan["source_policy_replicates"] = source_policy_replicates
    plan["source_training_converged"] = source_training_converged

    launch_id = f"{args.launch_id}_pilot" if args.pilot and not args.launch_id.endswith("_pilot") else args.launch_id
    launch_dir = Path(RUNS_ROOT) / launch_id
    launch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = launch_dir / "scale_stress_manifest.json"
    table_path = launch_dir / "full_suite_tables" / "scale_stress_evaluation.csv"
    parity_path = launch_dir / "full_suite_tables" / "scale_stress_interface_parity.json"
    manifest = {
        "schema_version": 1,
        "status": "planned" if args.dry_run else "running",
        "mode": plan["mode"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "launch_id": launch_id,
        "launch_seed": int(args.launch_seed),
        "experiment_suite": {
            "path": str(experiment.source_path),
            "sha256": experiment.source_sha256,
        },
        "city_profiles": {
            "path": city_suite.source_path,
            "sha256": city_suite.source_sha256,
        },
        "execution_plan": plan,
        "scientific_contract": {
            "policies": list(STRATEGIES),
            "same_regional_observation_action_interface": True,
            "exact_candidate_actions_without_lower_level_substitution": True,
            "common_random_numbers": True,
            "primary_metric": "objective_episode_return",
            "horizon_note": "60 one-minute transitions are encoded as stopTime=61",
            "decision_cadence": (
                "one regional shelter decision every "
                f"{experiment.scale_shelter_action_interval} transitions"
            ),
            "candidate_factor": "shelterCanVol sampled OSM candidate-site count",
            "checkpoint_contract": (
                "inference architecture must match; a confirmatory source policy must "
                "also have been trained at the 60-transition design horizon and "
                "two-transition decision cadence"
            ),
        },
        "limitations": (
            [
                "Pilot uses a policy trained under a different environment or "
                "decision contract and is for computational validation only, not "
                "policy-performance evidence."
            ]
            if args.pilot and (
                horizon_mismatches
                or action_interval_mismatches
                or installation_budget_mismatches
                or map_contract_mismatches
                or congestion_contract_mismatches
            )
            else []
        ),
        "artifacts": {
            "evaluation": str(table_path),
            "interface_parity": str(parity_path),
        },
    }
    _write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(f"[SCALE STRESS PLAN] artifact={manifest_path}")
        return 0
    if missing_checkpoints:
        raise FileNotFoundError(
            "Scale stress requires every declared policy checkpoint; first missing: "
            f"{missing_checkpoints[0]}"
        )
    if horizon_mismatches and not args.pilot:
        raise ValueError(
            "Confirmatory scale stress requires policies trained at the same "
            f"{experiment.scale_horizon_timesteps}-transition horizon; observed "
            f"training horizons: {horizon_mismatches}"
        )
    if action_interval_mismatches and not args.pilot:
        raise ValueError(
            "Confirmatory scale stress requires policies trained with the same "
            f"{experiment.scale_shelter_action_interval}-transition shelter-action "
            f"interval; observed: {action_interval_mismatches}"
        )
    if installation_budget_mismatches and not args.pilot:
        raise ValueError(
            "Confirmatory scale stress requires policies trained with the same "
            f"installation budget; observed: {installation_budget_mismatches}"
        )
    if map_contract_mismatches and not args.pilot:
        raise ValueError(
            "Confirmatory scale stress requires policies trained on the exact "
            f"enlarged OSM footprints; observed: {map_contract_mismatches}"
        )
    if congestion_contract_mismatches and not args.pilot:
        raise ValueError(
            "Confirmatory scale stress requires policies trained under the exact "
            f"pedestrian-congestion law; observed: {congestion_contract_mismatches}"
        )
    if not args.pilot and source_policy_replicates != int(policy_count):
        raise ValueError(
            "Confirmatory scale stress requires exactly the declared number of "
            f"independently trained policies; expected={policy_count}, "
            f"source_manifest={source_policy_replicates}"
        )
    if not args.pilot and not source_training_converged:
        raise ValueError(
            "Confirmatory scale stress requires a source launch that passed its "
            "recorded all-policy training convergence gate"
        )
    if table_path.exists() and not args.resume:
        raise FileExistsError(
            f"Scale-stress table already exists; use --resume or a new --launch-id: {table_path}"
        )

    from backtest import _run_episode

    rows = _read_csv(table_path) if args.resume else []
    completed = {_row_identity(row) for row in rows}
    city_overrides = expected_overrides
    source_learning_rate = source_manifest.get("learning_rate")
    source_rollout_episodes = source_manifest.get("ppo_rollout_episodes")
    for overrides in city_overrides.values():
        if source_learning_rate is not None:
            overrides["learningRate"] = float(source_learning_rate)
        if source_rollout_episodes is not None:
            overrides["ppoRolloutEpisodes"] = int(source_rollout_episodes)
    global_replication = 0
    for city in cities:
        for population_index, population in enumerate(populations, start=1):
            for candidate_index, candidate_count in enumerate(candidates, start=1):
                for scale_replication in range(1, int(replications) + 1):
                    global_replication += 1
                    scenario_seed = _condition_seed(
                        args.launch_seed,
                        city.scale_rank,
                        population_index,
                        candidate_index,
                        scale_replication,
                        710,
                    )
                    policy_seed = _condition_seed(
                        args.launch_seed,
                        city.scale_rank,
                        population_index,
                        candidate_index,
                        scale_replication,
                        711,
                    )
                    for strategy in STRATEGIES:
                        policy_ids = range(1, policy_count + 1) if strategy == "rl" else (0,)
                        for policy_replication in policy_ids:
                            identity = (
                                city.city_id,
                                int(population),
                                int(candidate_count),
                                int(scale_replication),
                                strategy,
                                int(policy_replication),
                            )
                            if identity in completed:
                                continue
                            overrides = dict(city_overrides[city.city_id])
                            overrides.update(
                                {
                                    "pedVol": int(population),
                                    "shelterCanVol": int(candidate_count),
                                    "initShelterVol": min(
                                        int(overrides.get("initShelterVol", 5)),
                                        int(candidate_count),
                                    ),
                                    "stopTime": int(experiment.scale_horizon_timesteps) + 1,
                                    "shelterActionInterval": int(
                                        experiment.scale_shelter_action_interval
                                    ),
                                    "maxAdditionalShelters": int(
                                        city_suite.common_experiment.get(
                                            "maxAdditionalShelters", 0
                                        )
                                    ),
                                }
                            )
                            checkpoint_index = policy_replication if strategy == "rl" else 1
                            visualize = bool(
                                args.visualize
                                and global_replication == 1
                                and policy_replication in {0, 1}
                            )
                            print(
                                f"[SCALE] city={city.city_id} population={population} "
                                f"candidates={candidate_count} rep={scale_replication}/"
                                f"{replications} strategy={strategy} policy={policy_replication}",
                                flush=True,
                            )
                            started = time.perf_counter()
                            result = _run_episode(
                                replication=global_replication,
                                machine=args.machine,
                                phase=str(
                                    launch_dir
                                    / "episodes"
                                    / city.city_id
                                    / f"population_{population}"
                                    / f"candidates_{candidate_count}"
                                    / (
                                        f"policy_{policy_replication:03d}"
                                        if strategy == "rl"
                                        else "benchmark"
                                    )
                                ),
                                strategy=strategy,
                                train_mode=False,
                                scenario_seed=scenario_seed,
                                policy_seed=policy_seed,
                                checkpoint_path=str(checkpoints[checkpoint_index]),
                                diagnostics_path=str(
                                    launch_dir / "unused_evaluation_diagnostics.csv"
                                ),
                                overrides=overrides,
                                visualization_enabled=visualize,
                                visualization_milestones="quartiles",
                            )
                            runtime = time.perf_counter() - started
                            actual_candidates = int(result["active_shelters"]) + int(
                                result["remaining_candidates"]
                            )
                            if int(result["initial_population"]) != int(population):
                                raise RuntimeError(
                                    f"Population factor was not realized for {identity}: "
                                    f"initialized={result['initial_population']}"
                                )
                            if actual_candidates != int(candidate_count):
                                raise RuntimeError(
                                    f"Candidate factor was not realized for {identity}: "
                                    f"requested={candidate_count}, actual={actual_candidates}"
                                )
                            result.update(
                                {
                                    "city_id": city.city_id,
                                    "city_scale_rank": city.scale_rank,
                                    "scale_replication": scale_replication,
                                    "policy_replication": policy_replication,
                                    "population_level": population,
                                    "shelter_candidate_level": candidate_count,
                                    "actual_candidate_count": actual_candidates,
                                    "horizon_timesteps": experiment.scale_horizon_timesteps,
                                    "shelter_action_interval_timesteps": (
                                        experiment.scale_shelter_action_interval
                                    ),
                                    "stop_time_config": experiment.scale_horizon_timesteps + 1,
                                    "simulation_runtime_s": float(runtime),
                                }
                            )
                            rows.append(result)
                            completed.add(identity)
                            _write_csv(table_path, rows)

    parity = _validate_matched_rows(rows, policy_count)
    _write_json(parity_path, parity)
    if not parity["verified"]:
        raise RuntimeError(f"Scale-stress interface parity failed; inspect {parity_path}")
    expected_rows = int(plan["total_episodes"])
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Scale-stress row count is incomplete: expected={expected_rows}, observed={len(rows)}"
        )
    manifest.update(
        {
            "status": "pilot_complete" if args.pilot else "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": len(rows),
            "interface_parity": parity,
            "artifact_hashes": {
                "evaluation": _sha256(table_path),
                "interface_parity": _sha256(parity_path),
                "checkpoints": {
                    str(policy): _sha256(path) for policy, path in checkpoints.items()
                },
            },
        }
    )
    _write_json(manifest_path, manifest)
    print(f"[SCALE STRESS COMPLETE] mode={plan['mode']} artifact={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
