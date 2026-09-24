#!/usr/bin/env python3
"""Re-run a prespecified sample of a completed multicity evaluation.

The audit is deliberately narrower than a second full confirmatory run.  It
checks that one matched RL/heuristic scenario per city reproduces the recorded
scientific outputs under the same scenario streams and frozen checkpoint.  It
never compares wall-clock or CPU timings and never overwrites the source run.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping

from backtest import _run_episode


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LAUNCH = PROJECT_ROOT / "runs" / "full_e0_e6_confirmatory_staged_v2_20260914"

EXACT_FIELDS = (
    "initial_population",
    "safe_completed",
    "shelter_evacuated",
    "arrival",
    "casualty",
    "unfinished",
    "deployments_made",
    "active_shelters",
    "initial_observation_digest",
    "hazard_trajectory_digest",
)

FLOAT_FIELDS = (
    "episode_return",
    "objective_episode_return",
    "safe_completion_reward",
    "casualty_penalty",
    "risk_time_penalty",
    "normalized_risk_weighted_person_time",
    "restricted_mean_time_to_safety",
    "mean_safe_completion_time",
    "mean_evacuation_time",
    "risk_weighted_person_time",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty reproducibility table")
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


def _policy_seed(row: Mapping) -> int:
    streams = row.get("random_stream_seeds")
    if isinstance(streams, str):
        streams = ast.literal_eval(streams)
    if not isinstance(streams, Mapping) or "policy" not in streams:
        raise ValueError("Evaluation row lacks a parseable policy random-stream seed")
    return int(streams["policy"])


def _same_float(left, right, *, absolute_tolerance: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=float(absolute_tolerance),
    )


def compare_rows(
    original: Mapping,
    rerun: Mapping,
    *,
    absolute_tolerance: float = 1e-10,
) -> list[dict]:
    """Return field-level comparisons for scientific, non-timing outcomes."""
    comparisons = []
    for field in EXACT_FIELDS:
        original_value = original.get(field)
        rerun_value = rerun.get(field)
        match = str(original_value) == str(rerun_value)
        comparisons.append(
            {
                "field": field,
                "comparison": "exact",
                "original": original_value,
                "rerun": rerun_value,
                "absolute_difference": 0.0 if match else None,
                "status": "MATCH" if match else "MISMATCH",
            }
        )
    for field in FLOAT_FIELDS:
        original_value = original.get(field)
        rerun_value = rerun.get(field)
        if original_value in (None, "") or rerun_value in (None, ""):
            match = False
            difference = None
        else:
            difference = abs(float(original_value) - float(rerun_value))
            match = _same_float(
                original_value,
                rerun_value,
                absolute_tolerance=absolute_tolerance,
            )
        comparisons.append(
            {
                "field": field,
                "comparison": f"absolute_tolerance_{absolute_tolerance:g}",
                "original": original_value,
                "rerun": rerun_value,
                "absolute_difference": difference,
                "status": "MATCH" if match else "MISMATCH",
            }
        )
    return comparisons


def _selected_rows(rows: list[dict], scenario_replication: int) -> list[dict]:
    selected = []
    cities = sorted({str(row["city_id"]) for row in rows})
    for city in cities:
        for strategy, policy_replication in (("rl", 1), ("heuristic", 0)):
            matches = [
                row
                for row in rows
                if str(row["city_id"]) == city
                and int(row["city_scenario_replication"]) == int(scenario_replication)
                and str(row["deployment_strategy"]) == strategy
                and int(row.get("policy_replication", 0)) == policy_replication
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {city}/{strategy}/scenario {scenario_replication} row; "
                    f"observed {len(matches)}"
                )
            selected.append(matches[0])
    return selected


def run_audit(
    launch_dir: Path,
    *,
    scenario_replication: int = 1,
    absolute_tolerance: float = 1e-10,
) -> dict:
    manifest_path = launch_dir / "experiment_manifest.json"
    evaluation_path = launch_dir / "evaluation_episode_summary.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Reproducibility audit requires a completed source launch")
    rows = _read_csv(evaluation_path)
    selected = _selected_rows(rows, scenario_replication)
    overrides_by_city = manifest.get("effective_overrides_by_city")
    if not isinstance(overrides_by_city, Mapping):
        raise RuntimeError("Source manifest lacks effective_overrides_by_city")

    output_dir = launch_dir / "reproducibility_sample"
    comparison_rows: list[dict] = []
    rerun_rows: list[dict] = []
    for original in selected:
        city = str(original["city_id"])
        strategy = str(original["deployment_strategy"])
        policy_replication = int(original.get("policy_replication", 0))
        checkpoint_index = policy_replication if strategy == "rl" else 1
        checkpoint = launch_dir / "policies" / f"policy_{checkpoint_index:03d}" / "regional_policy.pt"
        diagnostics = launch_dir / "policies" / f"policy_{checkpoint_index:03d}" / "ppo_diagnostics.csv"
        rerun = _run_episode(
            replication=int(original["replication"]),
            machine="reproduction",
            phase=str(output_dir / city / strategy),
            strategy=strategy,
            train_mode=False,
            scenario_seed=int(original["scenario_seed"]),
            policy_seed=_policy_seed(original),
            checkpoint_path=str(checkpoint),
            diagnostics_path=str(diagnostics),
            overrides=dict(overrides_by_city[city]),
            visualization_enabled=False,
        )
        rerun.update(
            {
                "city_id": city,
                "deployment_strategy": strategy,
                "policy_replication": policy_replication,
                "city_scenario_replication": int(scenario_replication),
            }
        )
        rerun_rows.append(rerun)
        for comparison in compare_rows(
            original,
            rerun,
            absolute_tolerance=absolute_tolerance,
        ):
            comparison.update(
                {
                    "city_id": city,
                    "deployment_strategy": strategy,
                    "policy_replication": policy_replication,
                    "city_scenario_replication": int(scenario_replication),
                }
            )
            comparison_rows.append(comparison)

    passed = all(row["status"] == "MATCH" for row in comparison_rows)
    rerun_path = output_dir / "rerun_episode_summary.csv"
    comparison_path = output_dir / "reproducibility_comparison.csv"
    _write_csv(rerun_path, rerun_rows)
    _write_csv(comparison_path, comparison_rows)
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scope": "one_prespecified_matched_rl_heuristic_scenario_per_city",
        "source_launch": str(launch_dir.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_evaluation_sha256": _sha256(evaluation_path),
        "scenario_replication": int(scenario_replication),
        "cities": sorted({str(row["city_id"]) for row in selected}),
        "strategies": ["rl_policy_001", "heuristic"],
        "scientific_fields_checked": len(EXACT_FIELDS) + len(FLOAT_FIELDS),
        "field_comparisons": len(comparison_rows),
        "mismatches": sum(row["status"] != "MATCH" for row in comparison_rows),
        "timing_metrics_compared": False,
        "absolute_float_tolerance": float(absolute_tolerance),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PARTIALLY_REPRODUCIBLE" if passed else "NOT_REPRODUCIBLE",
        "limitation": (
            "This deterministic spot re-run verifies a prespecified sample, not the "
            "entire stochastic confirmatory matrix or a second independent training run."
        ),
        "artifacts": {
            "rerun_episode_summary": str(rerun_path.resolve()),
            "comparison": str(comparison_path.resolve()),
        },
    }
    report_path = output_dir / "reproducibility_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path.resolve())
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-dir", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--scenario-replication", type=int, default=1)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)
    if args.scenario_replication <= 0:
        parser.error("--scenario-replication must be positive")
    if args.absolute_tolerance < 0.0 or not math.isfinite(args.absolute_tolerance):
        parser.error("--absolute-tolerance must be finite and non-negative")
    report = run_audit(
        args.launch_dir.resolve(),
        scenario_replication=args.scenario_replication,
        absolute_tolerance=args.absolute_tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
