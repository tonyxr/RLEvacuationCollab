#!/usr/bin/env python3
"""Assemble independently executed policy-training shards into one launch.

The shard mechanism changes only wall-clock scheduling.  Every shard declares
the same master launch seed, city schedule, scenario corpus, curriculum, and
total policy count, while ``training_policy_indices`` assigns disjoint neural
initializations to workers.  This assembler fails closed unless the indices
cover the complete declared set and every per-policy episode table,
diagnostics file, and checkpoint is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"

CONTRACT_FIELDS = (
    "schema_version",
    "launch_seed",
    "city_profile_sha256",
    "city_ids",
    "shared_grid",
    "policy_replicates",
    "eval_replications_per_city",
    "ppo_rollout_episodes",
    "learning_rate",
    "effective_overrides_by_city",
    "training_curriculum",
    "train_episodes_per_city",
    "total_train_episodes_per_policy",
    "training_city_schedule",
    "reward_equation",
    "policy_design",
    "cell_observation_features",
    "global_observation_features",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def validate_training_shards(shard_dirs: Sequence[Path]) -> dict:
    """Validate shard contracts and return assembly-ready provenance."""
    if not shard_dirs:
        raise ValueError("At least one training shard is required")
    records = []
    reference = None
    selected_indices: set[int] = set()
    convergence_rows = []
    from backtest import _read_csv

    for shard_dir in shard_dirs:
        resolved = shard_dir.expanduser().resolve()
        manifest_path = resolved / "experiment_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "training_shard_complete":
            raise ValueError(
                f"Training shard is not complete: {manifest_path} "
                f"status={manifest.get('status')!r}"
            )
        contract = {field: manifest.get(field) for field in CONTRACT_FIELDS}
        if reference is None:
            reference = contract
        elif contract != reference:
            mismatches = {
                field: {"reference": reference[field], "observed": contract[field]}
                for field in CONTRACT_FIELDS
                if contract[field] != reference[field]
            }
            raise ValueError(f"Training shard contract mismatch: {mismatches}")
        indices = tuple(int(value) for value in manifest.get("training_policy_indices", []))
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"Invalid training_policy_indices in {manifest_path}")
        overlap = selected_indices.intersection(indices)
        if overlap:
            raise ValueError(f"Policy replications appear in multiple shards: {sorted(overlap)}")
        selected_indices.update(indices)

        table_path = resolved / "training_episode_summary.csv"
        rows = _read_csv(str(table_path))
        expected_episodes = int(manifest["total_train_episodes_per_policy"])
        row_counts = {
            index: sum(int(float(row.get("policy_replication", 0))) == index for row in rows)
            for index in indices
        }
        if any(count != expected_episodes for count in row_counts.values()):
            raise ValueError(
                f"Incomplete policy episode table in {table_path}: "
                f"expected={expected_episodes}, observed={row_counts}"
            )
        unexpected = sorted(
            {
                int(float(row.get("policy_replication", 0)))
                for row in rows
            }.difference(indices)
        )
        if unexpected:
            raise ValueError(f"Unexpected policy rows in {table_path}: {unexpected}")

        policy_artifacts = {}
        for index in indices:
            policy_dir = resolved / "policies" / f"policy_{index:03d}"
            checkpoint = policy_dir / "regional_policy.pt"
            diagnostics = policy_dir / "ppo_diagnostics.csv"
            if not checkpoint.is_file() or not diagnostics.is_file():
                raise FileNotFoundError(
                    checkpoint if not checkpoint.is_file() else diagnostics
                )
            policy_artifacts[index] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "diagnostics": str(diagnostics),
                "diagnostics_sha256": _sha256(diagnostics),
            }
        convergence = manifest.get("training_convergence", {})
        observed_convergence = {
            int(row["policy_replication"]): row
            for row in convergence.get("policies", [])
        }
        missing_convergence = sorted(set(indices).difference(observed_convergence))
        if missing_convergence:
            raise ValueError(
                f"Shard convergence audit lacks policies {missing_convergence}: {manifest_path}"
            )
        convergence_rows.extend(observed_convergence[index] for index in indices)
        records.append(
            {
                "path": str(resolved),
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "training_summary": str(table_path),
                "training_summary_sha256": _sha256(table_path),
                "training_policy_indices": list(indices),
                "policies": policy_artifacts,
                "manifest_payload": manifest,
                "training_rows": rows,
            }
        )

    declared = int(reference["policy_replicates"])
    expected_indices = set(range(1, declared + 1))
    if selected_indices != expected_indices:
        raise ValueError(
            "Training shards do not cover the complete declared policy set: "
            f"missing={sorted(expected_indices.difference(selected_indices))}, "
            f"extra={sorted(selected_indices.difference(expected_indices))}"
        )
    return {
        "contract": reference,
        "records": records,
        "convergence_rows": sorted(
            convergence_rows, key=lambda row: int(row["policy_replication"])
        ),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = RUNS_ROOT / args.launch_id
    if (output_dir / "experiment_manifest.json").exists():
        raise FileExistsError(
            f"Assembled launch already exists; use a new --launch-id: {output_dir}"
        )
    validated = validate_training_shards(args.shards)
    records = validated["records"]
    reference_manifest = records[0]["manifest_payload"]
    output_dir.mkdir(parents=True, exist_ok=True)

    from backtest import _plot_outputs, _write_csv
    from multicity_backtest import _learning_assessment

    combined_rows = []
    for record in records:
        combined_rows.extend(record["training_rows"])
        for policy_index, paths in record["policies"].items():
            destination = output_dir / "policies" / f"policy_{policy_index:03d}"
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(paths["checkpoint"], destination / "regional_policy.pt")
            shutil.copy2(paths["diagnostics"], destination / "ppo_diagnostics.csv")
    combined_rows.sort(
        key=lambda row: (
            int(float(row.get("policy_replication", 0))),
            int(float(row.get("replication", 0))),
        )
    )
    training_path = output_dir / "training_episode_summary.csv"
    _write_csv(str(training_path), combined_rows)
    snapshot_source = Path(records[0]["path"]) / "city_profile_snapshot.json"
    snapshot_destination = output_dir / "city_profile_snapshot.json"
    shutil.copy2(snapshot_source, snapshot_destination)

    convergence_template = reference_manifest["training_convergence"]
    convergence = {
        **convergence_template,
        "policies": validated["convergence_rows"],
        "all_policies_converged": all(
            bool(row.get("converged", False))
            for row in validated["convergence_rows"]
        ),
    }
    convergence_path = output_dir / "training_convergence_diagnostics.json"
    _write_json(convergence_path, convergence)
    plots = _plot_outputs(str(output_dir), combined_rows, [])
    performance = {"status": "not_evaluated"}
    learning_assessment = _learning_assessment(convergence, performance, [])
    learning_path = output_dir / "learning_assessment.json"
    _write_json(learning_path, learning_assessment)

    manifest = dict(reference_manifest)
    manifest.update(
        {
            "status": (
                "complete"
                if convergence["all_policies_converged"]
                else "training_not_converged"
            ),
            "launch_id": args.launch_id,
            "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
            "started_utc": min(record["manifest_payload"]["started_utc"] for record in records),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "training_policy_indices": list(
                range(1, int(reference_manifest["policy_replicates"]) + 1)
            ),
            "training_convergence": convergence,
            "training_shard_provenance": [
                {key: value for key, value in record.items() if key not in {"manifest_payload", "training_rows"}}
                for record in records
            ],
            "interface_verification": None,
            "performance_assessment": performance,
            "learning_assessment": learning_assessment,
            "artifacts": {
                "city_profile_snapshot": str(snapshot_destination),
                "training_summary": str(training_path),
                "ppo_diagnostics": {
                    str(index): str(
                        output_dir / "policies" / f"policy_{index:03d}" / "ppo_diagnostics.csv"
                    )
                    for index in range(1, int(reference_manifest["policy_replicates"]) + 1)
                },
                "training_convergence": str(convergence_path),
                "learning_assessment": str(learning_path),
                "evaluation_summary": None,
                "paired_comparison": None,
                "paper_table": None,
                "scale_trend_diagnostics": None,
                "plots": plots,
            },
        }
    )
    manifest_path = output_dir / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        f"[SHARDS ASSEMBLED] status={manifest['status']} "
        f"policies={manifest['policy_replicates']} artifact={manifest_path}",
        flush=True,
    )
    return 0 if convergence["all_policies_converged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
