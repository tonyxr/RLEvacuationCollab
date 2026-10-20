#!/usr/bin/env python3
"""Run one auditable visualization episode per configured city.

This is a descriptive portability run, not a cross-city performance test.  It
uses one frozen policy, one simulator contract, and one scenario seed in every
city, and writes each completed city immediately so an interrupted long map
run can be resumed without repeating earlier cities.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from backtest import _run_episode
from CityProfiles import DEFAULT_CITY_PROFILE_PATH, load_city_suite
from TrainingCurriculum import load_training_curriculum


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / (
    "runs/state_college_2500_regime_v61_origin_grouped_E20_P80_"
    "eval20_seed6110_20261015_r2/policies/policy_001/regional_policy.pt"
)
DEFAULT_CURRICULUM = (
    PROJECT_ROOT / "config/state_college_regime_v61_origin_grouped_2500_full.json"
)
DEFAULT_EXECUTION_ORDER = (
    "state_college_pa",
    "spokane_wa",
    "seattle_wa",
    "chicago_il",
    "malibu_ca",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args), cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument(
        "--city-profiles", type=Path, default=Path(DEFAULT_CITY_PROFILE_PATH)
    )
    parser.add_argument("--scenario-seed", type=int, default=2646201033)
    parser.add_argument("--policy-seed", type=int, default=6110)
    parser.add_argument("--milestones", default="0,60,120,180")
    parser.add_argument("--machine", default="c1")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(DEFAULT_EXECUTION_ORDER),
        help="City identifiers in execution order; completed cities are skipped.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    curriculum_path = args.curriculum.expanduser().resolve()
    profiles_path = args.city_profiles.expanduser().resolve()
    for path in (checkpoint, curriculum_path, profiles_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    suite = load_city_suite(str(profiles_path))
    cities = suite.select(args.cities)
    curriculum = load_training_curriculum(str(curriculum_path))
    final_variant = curriculum.stages[-1].variants[0]
    shared = dict(suite.common_experiment)
    shared.update(dict(final_variant.overrides))
    shared["hazardEvolutionMode"] = "stochastic"
    if (int(shared["cellX"]), int(shared["cellY"])) != (16, 16):
        raise ValueError("Contribution-one run requires the trained 16x16 interface")
    if int(shared["pedVol"]) != 2500:
        raise ValueError("Contribution-one run requires 2,500 represented pedestrians")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "experiment_manifest.json"
    summary_path = output / "city_episode_summary.csv"
    rows_by_city: dict[str, dict] = {}
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows_by_city = {
                str(row["city_id"]): dict(row) for row in csv.DictReader(handle)
            }

    git_commit = _git_value("rev-parse", "HEAD")
    git_diff_sha256 = hashlib.sha256(
        _git_value("diff", "--binary", "HEAD").encode("utf-8")
    ).hexdigest()
    runner_sha256 = _sha256(Path(__file__).resolve())
    visualizer_sha256 = _sha256(PROJECT_ROOT / "EvacuationVisualizer.py")
    requested_contract = {
        "execution_order": [city.city_id for city in cities],
        "scenario_seed_shared_across_cities": int(args.scenario_seed),
        "policy_seed": int(args.policy_seed),
        "milestones": str(args.milestones),
        "checkpoint_sha256": _sha256(checkpoint),
        "curriculum_sha256": curriculum.source_sha256,
        "city_profiles_sha256": suite.source_sha256,
        "git_commit": git_commit,
        "git_diff_sha256": git_diff_sha256,
        "runner_sha256": runner_sha256,
        "visualizer_sha256": visualizer_sha256,
    }
    previous_manifest = None
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if previous_manifest.get("run_contract") != requested_contract:
            raise ValueError(
                "Existing output has a different run contract; choose a new output path"
            )

    manifest = {
        "schema_version": 1,
        "experiment": "contribution_1_five_city_dynamic_visualization",
        "status": "running",
        "started_utc": (
            previous_manifest.get("started_utc")
            if previous_manifest is not None
            else datetime.now(timezone.utc).isoformat()
        ),
        "claim_supported": (
            "One fixed OSM-to-simulation pipeline and frozen policy execute "
            "closed-loop episodes and expose spatial crowd, hazard, and shelter "
            "deployment trajectories across five heterogeneous city networks."
        ),
        "claims_not_supported": [
            "cross-city policy effectiveness",
            "statistical policy generalization",
            "superiority over city-specific benchmarks",
        ],
        "run_contract": requested_contract,
        "execution_order": requested_contract["execution_order"],
        "scenario_seed_shared_across_cities": requested_contract[
            "scenario_seed_shared_across_cities"
        ],
        "policy_seed": requested_contract["policy_seed"],
        "milestones": requested_contract["milestones"],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": requested_contract["checkpoint_sha256"],
        },
        "curriculum": {
            "path": str(curriculum_path),
            "sha256": curriculum.source_sha256,
            "id": curriculum.curriculum_id,
            "final_stage": curriculum.stages[-1].stage_id,
            "final_variant": final_variant.variant_id,
        },
        "city_profiles": {
            "path": str(profiles_path),
            "sha256": suite.source_sha256,
        },
        "effective_shared_contract": shared,
        "git": {
            "commit": git_commit,
            "dirty": bool(_git_value("status", "--porcelain") not in {"", "unavailable"}),
            "diff_sha256": git_diff_sha256,
        },
        "source_sha256": {
            "runner": runner_sha256,
            "visualizer": visualizer_sha256,
        },
        "python": sys.version,
        "completed_cities": sorted(rows_by_city),
    }
    _write_json(manifest_path, manifest)

    for replication, city in enumerate(cities, start=1):
        existing = rows_by_city.get(city.city_id)
        if existing and Path(str(existing.get("visualization_manifest", ""))).is_file():
            print(f"[C1 SKIP] {city.city_id} already complete", flush=True)
            continue

        print(f"[C1 RUN] {replication}/{len(cities)} {city.city_id}", flush=True)
        overrides = dict(shared)
        overrides.update(city.core_overrides())
        try:
            result = _run_episode(
                replication=replication,
                machine=args.machine,
                phase=str(output / "episodes" / city.city_id),
                strategy="rl",
                train_mode=False,
                scenario_seed=int(args.scenario_seed),
                policy_seed=int(args.policy_seed),
                checkpoint_path=str(checkpoint),
                diagnostics_path=str(output / "unused_evaluation_diagnostics.csv"),
                overrides=overrides,
                visualization_enabled=True,
                visualization_milestones=str(args.milestones),
                visualization_individual_snapshots=False,
                visualization_vector_outputs=False,
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failed_city"] = city.city_id
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(manifest_path, manifest)
            raise
        result.update(
            {
                "city_id": city.city_id,
                "city_display_name": city.display_name,
                "city_scale_rank": int(city.scale_rank),
                "scenario_seed": int(args.scenario_seed),
                "policy_seed": int(args.policy_seed),
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            }
        )
        rows_by_city[city.city_id] = result
        ordered_rows = [
            rows_by_city[item.city_id]
            for item in cities
            if item.city_id in rows_by_city
        ]
        _write_csv(summary_path, ordered_rows)
        manifest["completed_cities"] = [row["city_id"] for row in ordered_rows]
        _write_json(manifest_path, manifest)
        print(
            f"[C1 COMPLETE] {city.city_id} "
            f"cost={float(result['objective_normalized_cost']):.6f} "
            f"visualization={result['visualization_manifest']}",
            flush=True,
        )

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    print(f"[C1 ALL COMPLETE] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
