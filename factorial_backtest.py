#!/usr/bin/env python3
"""Resumable full-factorial evaluation of cached multicity PPO policies.

The evaluation crosses five cities, five population levels, five hazard-source
counts, and five panic levels. RL is compared with the five declared benchmark
policies under common random numbers. Every completed episode is journaled and
fsynced before the next episode begins, so a long run can safely resume.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from CityProfiles import DEFAULT_CITY_PROFILE_PATH, load_city_suite
from multicity_backtest import LEARNED_STRATEGIES, RUNS_ROOT, _city_overrides, _seed


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DESIGN = PROJECT_ROOT / "config" / "evacuation_factorial_experiment.json"
EXPECTED_POPULATIONS = (5000, 10000, 15000, 20000, 25000)
EXPECTED_HAZARDS = (1, 2, 3, 4, 5)
EXPECTED_PANIC = (0.1, 0.3, 0.5, 0.7, 0.9)
EXPECTED_STRATEGIES = (
    "rl",
    "initial_only",
    "risk_reduction",
    "heuristic",
    "random",
    "hazard_weighted",
    "accessibility_deficit",
)


@dataclass(frozen=True)
class FactorialDesign:
    experiment_id: str
    populations: tuple[int, ...]
    hazards: tuple[int, ...]
    panic_levels: tuple[float, ...]
    strategies: tuple[str, ...]
    replications: int
    source_path: str
    source_sha256: str
    payload: dict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_journal(path: Path, row: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_strict(row), sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid evaluation journal line {line_number}: {path}"
                ) from exc
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fields = {str(key) for row in rows for key in row}
    preferred = (
        "city_id",
        "population_level",
        "hazard_count",
        "panic_level",
        "scenario_replication",
        "policy_replication",
        "deployment_strategy",
        "scenario_seed",
        "policy_seed",
        "objective_episode_return",
        "safe_completed",
        "casualty",
        "unfinished",
        "safe_completion_coverage_rate",
        "shelter_service_coverage_rate",
        "restricted_mean_time_to_safety",
        "mean_pedestrian_distance_m",
        "mean_end_to_end_deployment_latency_ms",
        "p95_end_to_end_deployment_latency_ms",
        "episode_wall_time_s",
    )
    fieldnames = [name for name in preferred if name in fields]
    fieldnames.extend(sorted(fields.difference(fieldnames)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_design(path: str | Path = DEFAULT_DESIGN) -> FactorialDesign:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "population_levels",
        "hazard_count_levels",
        "panic_levels",
        "strategies",
        "scenario_replications_per_factor_cell",
        "individual_pedestrian_required",
        "panic_definition",
        "coverage_metrics",
        "latency_metric",
    }
    if set(payload) != required or int(payload["schema_version"]) != 1:
        raise ValueError("Factorial experiment fields or schema version are invalid")
    populations = tuple(int(value) for value in payload["population_levels"])
    hazards = tuple(int(value) for value in payload["hazard_count_levels"])
    panic = tuple(float(value) for value in payload["panic_levels"])
    strategies = tuple(str(value).strip().lower() for value in payload["strategies"])
    if populations != EXPECTED_POPULATIONS:
        raise ValueError("Population levels must be 5,000 through 25,000 by 5,000")
    if hazards != EXPECTED_HAZARDS:
        raise ValueError("Hazard-source levels must be 1 through 5")
    if panic != EXPECTED_PANIC:
        raise ValueError("Panic levels must be 10%, 30%, 50%, 70%, and 90%")
    if strategies != EXPECTED_STRATEGIES:
        raise ValueError("Factorial strategies do not match the declared benchmark set")
    if payload["individual_pedestrian_required"] is not True:
        raise ValueError("The panic experiment must require individual pedestrians")
    replications = int(payload["scenario_replications_per_factor_cell"])
    if replications <= 0:
        raise ValueError("scenario replications must be positive")
    return FactorialDesign(
        experiment_id=str(payload["experiment_id"]),
        populations=populations,
        hazards=hazards,
        panic_levels=panic,
        strategies=strategies,
        replications=replications,
        source_path=str(resolved),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
    )


def execution_plan(city_count: int, policy_count: int, design: FactorialDesign) -> dict:
    factor_cells = (
        int(city_count)
        * len(design.populations)
        * len(design.hazards)
        * len(design.panic_levels)
    )
    episodes_per_scenario = len(design.strategies) - 1 + int(policy_count)
    return {
        "cities": int(city_count),
        "factor_cells": factor_cells,
        "scenario_replications_per_factor_cell": design.replications,
        "policy_replications": int(policy_count),
        "episodes_per_scenario_replication": episodes_per_scenario,
        "total_episodes": factor_cells * design.replications * episodes_per_scenario,
    }


def validate_source_training_manifest(
    manifest: Mapping,
    *,
    allow_nonconverged: bool = False,
) -> None:
    """Fail closed before a checkpoint is used for confirmatory evaluation."""
    if manifest.get("status") != "complete":
        raise ValueError("Source training launch is not complete")
    converged = bool(
        manifest.get("training_convergence", {}).get(
            "all_policies_converged", False
        )
    )
    if not allow_nonconverged and not converged:
        raise ValueError(
            "Source policies did not pass the recorded convergence audit; use "
            "--allow-nonconverged-source only for a clearly labeled diagnostic run"
        )


def _identity(row: Mapping) -> tuple:
    return (
        str(row["city_id"]),
        int(row["population_level"]),
        int(row["hazard_count"]),
        float(row["panic_level"]),
        int(row["scenario_replication"]),
        int(row["policy_replication"]),
        str(row["deployment_strategy"]),
    )


def _condition_seed(
    launch_seed: int,
    city_rank: int,
    population: int,
    hazards: int,
    panic: float,
    replication: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(launch_seed),
            int(city_rank),
            int(population),
            int(hazards),
            int(round(float(panic) * 100)),
            int(replication),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _aggregate(rows: Sequence[Mapping]) -> list[dict]:
    numeric = (
        "objective_episode_return",
        "safe_completed",
        "casualty",
        "unfinished",
        "safe_completion_coverage_rate",
        "shelter_service_coverage_rate",
        "restricted_mean_time_to_safety",
        "mean_pedestrian_distance_m",
        "mean_end_to_end_deployment_latency_ms",
        "episode_wall_time_s",
    )
    groups = {}
    for row in rows:
        key = (
            row["city_id"],
            int(row["population_level"]),
            int(row["hazard_count"]),
            float(row["panic_level"]),
            row["deployment_strategy"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        record = {
            "city_id": key[0],
            "population_level": key[1],
            "hazard_count": key[2],
            "panic_level": key[3],
            "deployment_strategy": key[4],
            "episode_count": len(values),
        }
        for name in numeric:
            samples = [float(row[name]) for row in values if row.get(name) is not None]
            record[f"mean_{name}"] = float(np.mean(samples)) if samples else 0.0
        output.append(record)
    return output


def _latency_pairs(rows: Sequence[Mapping]) -> list[dict]:
    dynamic_heuristics = {
        (
            row["city_id"],
            int(row["population_level"]),
            int(row["hazard_count"]),
            float(row["panic_level"]),
            int(row["scenario_replication"]),
            str(row["deployment_strategy"]),
        ): row
        for row in rows
        if row["deployment_strategy"]
        in {
            "risk_reduction",
            "heuristic",
            "random",
            "hazard_weighted",
            "accessibility_deficit",
        }
    }
    output = []
    for row in rows:
        if row["deployment_strategy"] != "rl":
            continue
        key = (
            row["city_id"],
            int(row["population_level"]),
            int(row["hazard_count"]),
            float(row["panic_level"]),
            int(row["scenario_replication"]),
        )
        rl_latency = float(row["mean_end_to_end_deployment_latency_ms"])
        for strategy in (
            "risk_reduction",
            "heuristic",
            "random",
            "hazard_weighted",
            "accessibility_deficit",
        ):
            reference = dynamic_heuristics.get((*key, strategy))
            if reference is None:
                continue
            heuristic_latency = float(
                reference["mean_end_to_end_deployment_latency_ms"]
            )
            output.append(
                {
                    "city_id": key[0],
                    "population_level": key[1],
                    "hazard_count": key[2],
                    "panic_level": key[3],
                    "scenario_replication": key[4],
                    "policy_replication": int(row["policy_replication"]),
                    "heuristic_strategy": strategy,
                    "rl_end_to_end_latency_ms": rl_latency,
                    "heuristic_end_to_end_latency_ms": heuristic_latency,
                    "rl_minus_heuristic_latency_ms": rl_latency - heuristic_latency,
                    "rl_to_heuristic_latency_ratio": (
                        rl_latency / heuristic_latency
                        if heuristic_latency > 0.0
                        else None
                    ),
                }
            )
    return output


def _training_times(source_launch: Path) -> list[dict]:
    path = source_launch / "training_episode_summary.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for policy in sorted({int(row["policy_replication"]) for row in rows}):
        subset = [row for row in rows if int(row["policy_replication"]) == policy]
        output.append(
            {
                "policy_replication": policy,
                "training_episodes": len(subset),
                "training_wall_time_s": float(
                    sum(float(row.get("episode_wall_time_s", 0.0)) for row in subset)
                ),
                "training_cpu_time_s": float(
                    sum(float(row.get("episode_cpu_time_s", 0.0)) for row in subset)
                ),
            }
        )
    return output


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-launch-dir", required=False)
    parser.add_argument("--city-profiles", default=DEFAULT_CITY_PROFILE_PATH)
    parser.add_argument("--design", default=str(DEFAULT_DESIGN))
    parser.add_argument("--launch-id", default=None)
    parser.add_argument("--launch-seed", type=int, default=20260914)
    parser.add_argument("--machine", default="local")
    parser.add_argument("--policy-indices", default="all")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--visualize-first-replication", action="store_true")
    parser.add_argument(
        "--allow-nonconverged-source",
        action="store_true",
        help="Explicitly permit diagnostic evaluation of a complete but nonconverged training launch.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    design = load_design(args.design)
    suite = load_city_suite(args.city_profiles)
    replications = design.replications if args.replications is None else int(args.replications)
    if replications <= 0:
        raise ValueError("--replications must be positive")
    if args.dry_run:
        policy_count = 1
        print(json.dumps(execution_plan(len(suite.cities), policy_count, design), indent=2))
        return 0
    if not args.source_launch_dir:
        raise ValueError("--source-launch-dir is required unless --dry-run is used")

    source_launch = Path(args.source_launch_dir).expanduser().resolve()
    source_manifest_path = source_launch / "experiment_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    validate_source_training_manifest(
        source_manifest,
        allow_nonconverged=bool(args.allow_nonconverged_source),
    )
    if source_manifest.get("city_profile_sha256") != suite.source_sha256:
        raise ValueError("Source policy city profile differs from the factorial profile")
    declared_policy_count = int(source_manifest["policy_replicates"])
    if str(args.policy_indices).strip().lower() == "all":
        policy_indices = tuple(range(1, declared_policy_count + 1))
    else:
        policy_indices = tuple(
            int(item.strip()) for item in str(args.policy_indices).split(",") if item.strip()
        )
    if not policy_indices or len(policy_indices) != len(set(policy_indices)):
        raise ValueError("--policy-indices must be non-empty and unique")
    if any(index < 1 or index > declared_policy_count for index in policy_indices):
        raise ValueError("--policy-indices contains an unavailable policy")
    checkpoints = {
        index: source_launch / "policies" / f"policy_{index:03d}" / "regional_policy.pt"
        for index in policy_indices
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source policy checkpoint is missing: {missing[0]}")

    launch_id = args.launch_id or f"factorial_seed_{args.launch_seed}"
    launch_dir = Path(RUNS_ROOT) / launch_id
    manifest_path = launch_dir / "experiment_manifest.json"
    journal_path = launch_dir / "evaluation_episode_journal.jsonl"
    contract = {
        "design_sha256": design.source_sha256,
        "city_profile_sha256": suite.source_sha256,
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_checkpoint_sha256": {
            str(index): _sha256(path) for index, path in checkpoints.items()
        },
        "policy_indices": list(policy_indices),
        "launch_seed": int(args.launch_seed),
        "replications": replications,
        "allow_nonconverged_source": bool(args.allow_nonconverged_source),
    }
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Factorial launch exists; pass --resume or use a new id")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != contract:
            raise ValueError("Factorial resume contract differs from the existing launch")
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a factorial launch without a manifest")
        manifest = {
            "schema_version": 1,
            "status": "running",
            "started_utc": _now(),
            "experiment_id": design.experiment_id,
            "contract": contract,
            "execution_plan": execution_plan(len(suite.cities), len(policy_indices), design),
            "design": design.payload,
            "inference_scope": (
                "conditional_on_one_fixed_trained_policy"
                if len(policy_indices) == 1
                else "joint_over_policy_and_scenario_seeds"
            ),
        }
        _write_json(manifest_path, manifest)

    rows = _read_journal(journal_path)
    identities = {_identity(row) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("Factorial journal contains duplicate episode identities")
    from backtest import _run_episode

    episode_index = len(rows)
    try:
        for city, population, hazards, panic, replication in itertools.product(
            suite.cities,
            design.populations,
            design.hazards,
            design.panic_levels,
            range(1, replications + 1),
        ):
            scenario_seed = _condition_seed(
                args.launch_seed,
                city.scale_rank,
                population,
                hazards,
                panic,
                replication,
            )
            base_overrides = _city_overrides(
                suite,
                city,
                {},
                "stochastic",
            )
            base_overrides.update(
                {
                    "pedVol": int(population),
                    "pedestrianGroupSize": 1,
                    "hazardVol": int(hazards),
                    "panicRate": float(panic),
                }
            )
            for strategy in design.strategies:
                indices = policy_indices if strategy in LEARNED_STRATEGIES else (0,)
                for policy_index in indices:
                    identity = (
                        city.city_id,
                        int(population),
                        int(hazards),
                        float(panic),
                        int(replication),
                        int(policy_index),
                        strategy,
                    )
                    if identity in identities:
                        continue
                    episode_index += 1
                    policy_seed = (
                        _seed(int(source_manifest["launch_seed"]), 10, policy_index)
                        if policy_index
                        else _condition_seed(
                            args.launch_seed,
                            city.scale_rank,
                            population,
                            hazards,
                            panic,
                            replication + 1_000_000,
                        )
                    )
                    print(
                        f"[FACTORIAL] episode={episode_index} city={city.city_id} "
                        f"population={population} hazards={hazards} panic={panic:.1f} "
                        f"rep={replication} strategy={strategy} policy={policy_index}",
                        flush=True,
                    )
                    checkpoint = checkpoints[policy_index or policy_indices[0]]
                    row = _run_episode(
                        replication=episode_index,
                        machine=args.machine,
                        phase=str(
                            launch_dir
                            / "episodes"
                            / city.city_id
                            / f"population_{population}"
                            / f"hazards_{hazards}"
                            / f"panic_{int(round(panic * 100)):02d}"
                        ),
                        strategy=strategy,
                        train_mode=False,
                        scenario_seed=scenario_seed,
                        policy_seed=policy_seed,
                        checkpoint_path=str(checkpoint),
                        diagnostics_path=str(checkpoint.parent / "ppo_diagnostics.csv"),
                        overrides=base_overrides,
                        visualization_enabled=bool(
                            args.visualize_first_replication
                            and replication == 1
                            and policy_index in {0, policy_indices[0]}
                            and strategy in {"rl", "heuristic", "initial_only"}
                        ),
                    )
                    row.update(
                        {
                            "city_id": city.city_id,
                            "city_scale_rank": city.scale_rank,
                            "population_level": int(population),
                            "hazard_count": int(hazards),
                            "panic_level": float(panic),
                            "scenario_replication": int(replication),
                            "policy_replication": int(policy_index),
                            "deployment_strategy": strategy,
                            "source_checkpoint_sha256": contract[
                                "source_checkpoint_sha256"
                            ][str(policy_index or policy_indices[0])],
                        }
                    )
                    _append_journal(journal_path, row)
                    rows.append(row)
                    identities.add(identity)
                    manifest["completed_episodes"] = len(rows)
                    manifest["updated_utc"] = _now()
                    _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": _now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_episodes": len(rows),
            }
        )
        _write_json(manifest_path, manifest)
        raise

    evaluation_path = launch_dir / "evaluation_episode_summary.csv"
    aggregate_path = launch_dir / "factorial_summary.csv"
    latency_path = launch_dir / "rl_vs_heuristic_deployment_latency.csv"
    training_time_path = launch_dir / "training_time_by_policy.csv"
    _write_csv(evaluation_path, rows)
    _write_csv(aggregate_path, _aggregate(rows))
    _write_csv(latency_path, _latency_pairs(rows))
    _write_csv(training_time_path, _training_times(source_launch))
    manifest.update(
        {
            "status": "complete",
            "completed_utc": _now(),
            "completed_episodes": len(rows),
            "artifacts": {
                "episode_journal": str(journal_path),
                "evaluation_summary": str(evaluation_path),
                "factorial_summary": str(aggregate_path),
                "rl_vs_heuristic_deployment_latency": str(latency_path),
                "training_time_by_policy": str(training_time_path),
            },
        }
    )
    _write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
