#!/usr/bin/env python3
"""Evaluate a frozen checkpoint after an explicitly recorded environment fix."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

from backtest import _read_csv, _run_episode, _write_csv
from CityProfiles import DEFAULT_CITY_PROFILE_PATH, load_city_suite
from multicity_backtest import (
    POLICY_DYNAMICS_SOURCES,
    PROJECT_ROOT,
    _city_overrides,
    _seed,
)
from PolicyCache import source_fingerprint
from TrainingCurriculum import load_training_curriculum


DEFAULT_STRATEGIES = (
    "rl",
    "objective_greedy",
    "heuristic",
    "random",
    "initial_only",
)
REGISTERED_STRATEGIES = DEFAULT_STRATEGIES + (
    "risk_reduction",
    "hazard_weighted",
    "accessibility_deficit",
    "route_saving",
    "static_greedy",
    "rl_precommit",
)


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _evaluate_job(job: dict) -> dict:
    row = _run_episode(
        replication=int(job["replication"]),
        machine="local",
        phase=str(job["phase"]),
        strategy=str(job["strategy"]),
        train_mode=False,
        scenario_seed=int(job["scenario_seed"]),
        policy_seed=int(job["policy_seed"]),
        checkpoint_path=str(job["checkpoint_path"]),
        diagnostics_path=str(job["diagnostics_path"]),
        overrides=dict(job["overrides"]),
        visualization_enabled=False,
    )
    row.update(dict(job["result_fields"]))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch_dir")
    parser.add_argument("--replications", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated registered strategies; defaults preserve the historical contract.",
    )
    parser.add_argument(
        "--output-stem",
        default="evaluation_postfix",
        help="Artifact stem used for the summary, manifest, and episode directory.",
    )
    parser.add_argument("--city", default="state_college_pa")
    parser.add_argument("--city-profiles", default=DEFAULT_CITY_PROFILE_PATH)
    parser.add_argument(
        "--training-curriculum",
        default="config/state_college_training_curriculum_2500_counterfactual_exposure_v34.json",
    )
    args = parser.parse_args()
    if args.replications <= 0:
        parser.error("--replications must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    strategies = tuple(
        item.strip().lower()
        for item in str(args.strategies).split(",")
        if item.strip()
    )
    invalid = sorted(set(strategies).difference(REGISTERED_STRATEGIES))
    if not strategies or invalid:
        parser.error(f"unsupported or empty strategies: {invalid}")
    output_stem = str(args.output_stem).strip()
    if not output_stem or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in output_stem):
        parser.error("--output-stem must contain only letters, digits, '_' or '-'")

    root = Path(args.launch_dir).expanduser().resolve()
    training_path = root / "training_episode_summary.csv"
    checkpoint_path = root / "policies" / "policy_001" / "regional_policy.pt"
    diagnostics_path = root / "policies" / "policy_001" / "learner_diagnostics.csv"
    parent_manifest_path = root / "experiment_manifest.json"
    for required in (training_path, checkpoint_path, diagnostics_path, parent_manifest_path):
        if not required.exists():
            raise FileNotFoundError(required)
    training_rows = _read_csv(str(training_path))
    parent_manifest = json.loads(parent_manifest_path.read_text())
    suite = load_city_suite(args.city_profiles)
    city = suite.select([args.city])[0]
    curriculum = load_training_curriculum(args.training_curriculum)
    expected_training_rows = int(curriculum.episodes_per_city)
    if len(training_rows) != expected_training_rows:
        raise RuntimeError(
            "Frozen-policy evaluation requires all "
            f"{expected_training_rows} training episodes"
        )
    final_stage = curriculum.stages[-1]
    evaluation_variants = tuple(final_stage.variants)
    if not evaluation_variants:
        raise RuntimeError("Target-stage evaluation requires at least one variant")
    base_overrides = _city_overrides(suite, city, {}, "stochastic")
    base_overrides.update(curriculum.learner_overrides)
    base_overrides["learningRate"] = float(parent_manifest["learning_rate"])

    current_fingerprint = source_fingerprint(PROJECT_ROOT, POLICY_DYNAMICS_SOURCES)
    training_fingerprint = parent_manifest["policy_dynamics_source_sha256"]
    changed_sources = {
        name: {
            "training_sha256": training_fingerprint.get(name),
            "evaluation_sha256": current_fingerprint.get(name),
        }
        for name in POLICY_DYNAMICS_SOURCES
        if training_fingerprint.get(name) != current_fingerprint.get(name)
    }
    output_path = root / f"{output_stem}_episode_summary.csv"
    manifest_path = root / f"{output_stem}_manifest.json"
    contract = {
        "schema_version": 2,
        "parent_launch_id": parent_manifest["launch_id"],
        "frozen_checkpoint": str(checkpoint_path),
        "city_id": city.city_id,
        "replications": int(args.replications),
        "workers": int(args.workers),
        "strategies": list(strategies),
        "output_stem": output_stem,
        "scenario_seed_contract": (
            "seed(parent_launch_seed, 500+city_scale_rank, "
            "evaluation_variant_ordinal*10000+replication)"
        ),
        "policy_seed_contract": (
            "seed(parent_launch_seed, 600+city_scale_rank, "
            "evaluation_variant_ordinal*10000+replication)"
        ),
        "target_stage_id": final_stage.stage_id,
        "target_variants": [
            {
                "variant_id": variant.variant_id,
                "pedVol": int(variant.overrides["pedVol"]),
                "shelterCapacityToken": int(
                    variant.overrides["shelterCapacityToken"]
                ),
            }
            for variant in evaluation_variants
        ],
        "common_effective_overrides": base_overrides,
        "training_policy_dynamics_source_sha256": training_fingerprint,
        "evaluation_policy_dynamics_source_sha256": current_fingerprint,
        "changed_sources": changed_sources,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous_contract = {key: previous.get(key) for key in contract}
        if previous_contract != contract:
            raise ValueError("Post-fix evaluation contract differs from its existing manifest")
        started_utc = previous["started_utc"]
    else:
        started_utc = datetime.now(timezone.utc).isoformat()

    rows = _read_csv(str(output_path)) if output_path.exists() else []
    completed = {
        (
            str(row["evaluation_variant_id"]),
            int(row["city_scenario_replication"]),
            str(row["deployment_strategy"]),
        )
        for row in rows
    }
    manifest = {
        **contract,
        "status": "running",
        "started_utc": started_utc,
        "completed_evaluations": len(rows),
    }
    _json_dump(manifest_path, manifest)

    jobs = []
    try:
        for variant_index, variant in enumerate(evaluation_variants, start=1):
            overrides = {**base_overrides, **dict(variant.overrides)}
            for replication in range(1, args.replications + 1):
                seed_index = variant_index * 10000 + replication
                scenario_seed = _seed(
                    int(parent_manifest["launch_seed"]),
                    500 + city.scale_rank,
                    seed_index,
                )
                policy_seed = _seed(
                    int(parent_manifest["launch_seed"]),
                    600 + city.scale_rank,
                    seed_index,
                )
                for strategy in strategies:
                    key = (variant.variant_id, replication, strategy)
                    if key in completed:
                        continue
                    print(
                        f"[POSTFIX EVAL] population={overrides['pedVol']} "
                        f"pair={replication}/{args.replications} strategy={strategy}",
                        flush=True,
                    )
                    learned_strategy = strategy in ("rl", "rl_precommit")
                    policy_replication = 1 if learned_strategy else 0
                    phase_group = "policy_001" if learned_strategy else "benchmarks"
                    jobs.append(
                        {
                            "replication": replication,
                            "phase": str(
                                Path(root.name)
                                / output_stem
                                / city.city_id
                                / variant.variant_id
                                / phase_group
                            ),
                            "strategy": strategy,
                            "scenario_seed": scenario_seed,
                            "policy_seed": policy_seed,
                            "checkpoint_path": str(checkpoint_path),
                            "diagnostics_path": str(diagnostics_path),
                            "overrides": dict(overrides),
                            "result_fields": {
                                "policy_replication": policy_replication,
                                "city_id": city.city_id,
                                "city_scale_rank": city.scale_rank,
                                "city_scenario_replication": replication,
                                "evaluation_variant_id": variant.variant_id,
                                "evaluation_population": int(overrides["pedVol"]),
                                "evaluation_shelter_capacity": int(
                                    overrides["shelterCapacityToken"]
                                ),
                                "evaluation_contract": "multiscale_target_stage_v2",
                            },
                        }
                    )

        def retain(row: dict) -> None:
            key = (
                str(row["evaluation_variant_id"]),
                int(row["city_scenario_replication"]),
                str(row["deployment_strategy"]),
            )
            rows.append(row)
            completed.add(key)
            rows.sort(
                key=lambda item: (
                    int(item["evaluation_population"]),
                    int(item["city_scenario_replication"]),
                    strategies.index(str(item["deployment_strategy"])),
                )
            )
            _write_csv(str(output_path), rows)
            manifest["completed_evaluations"] = len(rows)
            _json_dump(manifest_path, manifest)

        if int(args.workers) == 1:
            for job in jobs:
                retain(_evaluate_job(job))
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = {executor.submit(_evaluate_job, job): job for job in jobs}
                for future in as_completed(futures):
                    retain(future.result())
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "failure": f"{type(error).__name__}: {error}",
            }
        )
        _json_dump(manifest_path, manifest)
        raise

    manifest.update(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_evaluations": len(rows),
        }
    )
    _json_dump(manifest_path, manifest)
    print(f"[POSTFIX EVAL COMPLETE] rows={len(rows)} summary={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
