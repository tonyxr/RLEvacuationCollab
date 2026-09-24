#!/usr/bin/env python3
"""Paired calibration of weighted-cohort and individual evacuation simulation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from CityProfiles import DEFAULT_CITY_PROFILE_PATH, load_city_suite
from backtest import _run_episode
from multicity_backtest import _city_overrides, _seed


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS = {
    "objective_episode_return": 0.10,
    "safe_completion_fraction": 0.10,
    "casualty_fraction": 0.05,
    "normalized_risk_weighted_person_time": 0.10,
}


def _normalized_metrics(row: dict) -> dict[str, float]:
    population = float(row["initial_population"])
    horizon = float(row["horizon_transitions"])
    return {
        "objective_episode_return": float(row["objective_episode_return"]),
        "safe_completion_fraction": float(row["safe_completed"]) / population,
        "casualty_fraction": float(row["casualty"]) / population,
        "normalized_risk_weighted_person_time": (
            float(row["objective_risk_weighted_person_time"])
            / (population * horizon)
        ),
    }


def summarize_fidelity(rows: list[dict], cohort_size: int) -> dict:
    by_key = {
        (int(row["scenario_replication"]), int(row["maximum_persons_per_agent"])): row
        for row in rows
    }
    replications = sorted({key[0] for key in by_key})
    diagnostics = {}
    for metric, tolerance in METRICS.items():
        differences = []
        for replication in replications:
            individual = _normalized_metrics(by_key[(replication, 1)])[metric]
            cohort = _normalized_metrics(by_key[(replication, int(cohort_size))])[metric]
            differences.append(cohort - individual)
        values = np.asarray(differences, dtype=float)
        diagnostics[metric] = {
            "mean_paired_difference_cohort_minus_individual": float(values.mean()),
            "mean_absolute_paired_difference": float(np.abs(values).mean()),
            "maximum_absolute_paired_difference": float(np.abs(values).max()),
            "acceptance_tolerance": float(tolerance),
            "mean_bias_within_tolerance": bool(abs(float(values.mean())) <= tolerance),
        }
    return {
        "criterion": (
            "Exploratory training-surrogate gate: absolute paired mean bias must not "
            "exceed the prespecified metric tolerance. This does not validate cohort "
            "results as confirmatory individual-microsimulation evidence."
        ),
        "cohort_size": int(cohort_size),
        "scenario_replications": len(replications),
        "metrics": diagnostics,
        "passes_training_surrogate_gate": bool(
            all(item["mean_bias_within_tolerance"] for item in diagnostics.values())
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-profiles", default=DEFAULT_CITY_PROFILE_PATH)
    parser.add_argument("--city", default="state_college_pa")
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--cohort-size", type=int, default=20)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--stop-time", type=int, default=15)
    parser.add_argument("--launch-seed", type=int, default=20260913)
    parser.add_argument("--launch-id", default="cohort_fidelity_20260913")
    args = parser.parse_args()
    if min(args.population, args.cohort_size, args.replications) <= 0:
        parser.error("population, cohort size, and replications must be positive")
    if args.stop_time < 2:
        parser.error("stop time must be at least two")

    suite = load_city_suite(args.city_profiles)
    city = suite.select((args.city,))[0]
    launch_dir = PROJECT_ROOT / "runs" / args.launch_id
    if launch_dir.exists():
        raise FileExistsError(f"launch already exists: {launch_dir}")
    launch_dir.mkdir(parents=True)
    rows = []
    for replication in range(1, args.replications + 1):
        scenario_seed = _seed(args.launch_seed, 701, replication)
        for group_size in (1, args.cohort_size):
            overrides = _city_overrides(
                suite,
                city,
                {
                    "pedVol": args.population,
                    "pedestrianGroupSize": group_size,
                    "stopTime": args.stop_time,
                    "shelterCanVol": 20,
                    "initShelterVol": 2,
                    "maxAdditionalShelters": 5,
                },
                "stochastic",
            )
            row = _run_episode(
                replication=replication,
                machine="local",
                phase=str(
                    launch_dir
                    / "episodes"
                    / f"rep_{replication:03d}"
                    / f"group_{group_size:04d}"
                ),
                strategy="heuristic",
                train_mode=False,
                scenario_seed=scenario_seed,
                policy_seed=0,
                checkpoint_path=str(launch_dir / "unused.pt"),
                diagnostics_path=str(launch_dir / "unused.csv"),
                overrides=overrides,
            )
            row.update(
                {
                    "scenario_replication": replication,
                    "requested_population": args.population,
                    "calibration_role": (
                        "individual_reference" if group_size == 1 else "weighted_cohort"
                    ),
                }
            )
            rows.append(row)

    assessment = summarize_fidelity(rows, args.cohort_size)
    _write_csv(launch_dir / "cohort_fidelity_episode_summary.csv", rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "city_id": city.city_id,
        "population": args.population,
        "individual_group_size": 1,
        "candidate_cohort_size": args.cohort_size,
        "stop_time": args.stop_time,
        "launch_seed": args.launch_seed,
        "assessment": assessment,
    }
    (launch_dir / "cohort_fidelity_assessment.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment["passes_training_surrogate_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
