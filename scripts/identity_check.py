#!/usr/bin/env python3
"""Bitwise identity checks between two runs of the same seeds (v65.1).

training:  compare two ``training_episode_summary.csv`` files row by row on
           every shared column except timing (e.g. v65 smoke vs v64 r3).
eval:      compare ``multicity_backtest --eval-only`` rows with the rows a
           panel collected for the same (strategy, policy, city, pair).

Exit status 0 means identical on every compared column; 1 lists the first
differences.  Timing, latency, path and process columns are excluded by
pattern and listed in the report so the exclusion is visible.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from copy import deepcopy
from pathlib import Path

EXCLUDE = re.compile(
    r"(wall|cpu|latency|_seconds|seconds_|_time_s$|_ms$|episode_artifacts_dir|run_dir|phase|"
    r"^replication$|visualization|^panel_|^job_key$|^arm$|^captured_|checkpoint_path|^label_|"
    r"^evaluation_population$|^evaluation_variant_id$|^panel_id$|^machine$)"
)

# These fields were added by the preregistered v65.1 compute/provenance patch.
# They describe how an otherwise identical episode was computed; they are not
# scientific state.  Keep this allow-list narrow and fail closed on every
# other finite value.
PREREGISTERED_STAGE_METADATA = frozenset(
    {
        "branchTaskChunking",
        "forecastMemoEnabled",
        "forecastMemoVerifyFraction",
        "hazardForecastKernel",
        "hazardWindSpeedMPerMinute",
        "observationEveryMinute",
    }
)
PROVENANCE_PATH_FIELDS = frozenset(
    {"graph_cache_path", "consolidated_graph_cache_path"}
)


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _structure(text: str):
    """Parse a CSV cell written either as JSON (panel collect) or as a Python
    repr (backtest._write_csv) so that equal values compare equal."""
    import ast
    import json

    class _Floats(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in {"nan", "inf"}:
                return ast.copy_location(ast.Constant(float(node.id)), node)
            return node

    def python_literal(value: str):
        tree = _Floats().visit(ast.parse(value, mode="eval"))
        return ast.literal_eval(ast.fix_missing_locations(tree))

    for parse in (json.loads, python_literal):
        try:
            return parse(text)
        except Exception:
            continue
    return text


def _equal(x, y) -> bool:
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_equal(a, b) for a, b in zip(x, y))
    if isinstance(x, dict) and isinstance(y, dict):
        return set(x) == set(y) and all(_equal(x[k], y[k]) for k in x)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
        return x == y or (math.isnan(x) and math.isnan(y))
    return x == y


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
        return x == y or (math.isnan(x) and math.isnan(y))
    except (TypeError, ValueError):
        pass
    return _equal(_structure(a), _structure(b))


def normalize_preregistered_training_metadata(row: dict) -> dict:
    """Remove only preregistered compute metadata and machine-local paths.

    Hashes, map dimensions, configuration values, and every scientific field
    remain in the comparison.  This lets a detached baseline worktree and the
    production worktree share exact evidence without treating their absolute
    cache paths as dynamics.
    """
    normalized = deepcopy(row)
    stage = _structure(normalized.get("stage_overrides", ""))
    if isinstance(stage, dict):
        for key in PREREGISTERED_STAGE_METADATA:
            stage.pop(key, None)
        normalized["stage_overrides"] = repr(stage)
    provenance = _structure(normalized.get("map_provenance", ""))
    if isinstance(provenance, dict):
        for key in PROVENANCE_PATH_FIELDS:
            provenance.pop(key, None)
        normalized["map_provenance"] = repr(provenance)
    return normalized


def compare_rows(left: list[dict], right: list[dict], keys: list[tuple]) -> tuple[list, list, list]:
    columns = sorted(set(left[0]) & set(right[0])) if left and right else []
    compared = [c for c in columns if not EXCLUDE.search(c)]
    excluded = [c for c in columns if EXCLUDE.search(c)]
    differences = []
    for key, a, b in zip(keys, left, right):
        for column in compared:
            if not _same(a.get(column, ""), b.get(column, "")):
                differences.append((key, column, a.get(column), b.get(column)))
    return compared, excluded, differences


def command_training(args) -> int:
    reference = _read(Path(args.reference))
    candidate = _read(Path(args.candidate))
    if args.normalize_preregistered_metadata:
        reference = [normalize_preregistered_training_metadata(row) for row in reference]
        candidate = [normalize_preregistered_training_metadata(row) for row in candidate]
    policy = str(args.policy)
    reference = [r for r in reference if str(r.get("policy_replication", "1")) == policy]
    candidate = [r for r in candidate if str(r.get("policy_replication", "1")) == policy]
    count = min(len(reference), len(candidate))
    if args.rows:
        count = min(count, args.rows)
    if count == 0:
        print("no common rows")
        return 1
    keys = [(r["replication"], r.get("city_id"), r.get("training_stage_id")) for r in reference[:count]]
    for a, b in zip(reference[:count], candidate[:count]):
        if (a["replication"], a.get("city_id")) != (b["replication"], b.get("city_id")):
            print(f"schedule differs at replication {a['replication']} vs {b['replication']}")
            return 1
    compared, excluded, differences = compare_rows(reference[:count], candidate[:count], keys)
    return _report(f"training rows 1..{count} (policy {policy})", compared, excluded, differences,
                   set(reference[0]) ^ set(candidate[0]))


def checkpoint_tensor_differences(reference: Path, candidate: Path) -> tuple[int, int, list]:
    """Return exact tensor coverage and differences for two Torch checkpoints."""
    import torch

    def tensors(value, prefix=""):
        found = {}
        if torch.is_tensor(value):
            found[prefix] = value.detach().cpu()
        elif isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                found.update(tensors(item, child))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                found.update(tensors(item, f"{prefix}[{index}]"))
        return found

    left = tensors(torch.load(reference, map_location="cpu", weights_only=False))
    right = tensors(torch.load(candidate, map_location="cpu", weights_only=False))
    differences = []
    for path in sorted(set(left) | set(right)):
        if path not in left or path not in right:
            differences.append((path, "missing tensor path"))
            continue
        a, b = left[path], right[path]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            differences.append((path, f"{a.dtype}/{tuple(a.shape)} != {b.dtype}/{tuple(b.shape)}"))
    return len(left), len(right), differences


def command_checkpoint(args) -> int:
    left, right, differences = checkpoint_tensor_differences(
        Path(args.reference), Path(args.candidate)
    )
    print(f"[CHECKPOINT IDENTITY] tensor paths: reference={left}, candidate={right}")
    if differences:
        print(f"[IDENTITY FAIL] {len(differences)} tensor difference(s); first 25:")
        for path, detail in differences[:25]:
            print(f"  {path}: {detail}")
        return 1
    if left == 0 or right == 0:
        print("[IDENTITY FAIL] checkpoint contains no tensors")
        return 1
    print("[IDENTITY PASS] every checkpoint tensor is bitwise equal")
    return 0


def command_eval(args) -> int:
    reference = _read(Path(args.multicity))
    panel = _read(Path(args.panel))
    index = {}
    for row in panel:
        key = (row["deployment_strategy"], str(int(float(row["policy_replication"]))), row["city_id"],
               str(int(float(row["city_scenario_replication"]))))
        index.setdefault(key, row)
    pairs, keys, missing = [], [], []
    for row in reference:
        key = (row["deployment_strategy"], str(int(float(row["policy_replication"]))), row["city_id"],
               str(int(float(row["city_scenario_replication"]))))
        if key in index:
            pairs.append((row, index[key]))
            keys.append(key)
        else:
            missing.append(key)
    if not pairs:
        print("no common episodes")
        return 1
    compared, excluded, differences = compare_rows([a for a, _ in pairs], [b for _, b in pairs], keys)
    if missing:
        print(f"{len(missing)} multicity episode(s) have no panel counterpart, e.g. {missing[:3]}")
    return _report(f"{len(pairs)} evaluation episodes", compared, excluded, differences,
                   set(reference[0]) ^ set(panel[0]), strict_missing=bool(missing) and args.require_all)


def _report(scope, compared, excluded, differences, asymmetric, strict_missing=False) -> int:
    print(f"[IDENTITY] {scope}: compared {len(compared)} columns, excluded {len(excluded)} timing/path columns")
    print(f"  excluded: {', '.join(excluded)}")
    if asymmetric:
        print(f"  columns present on one side only (not compared): {', '.join(sorted(asymmetric))[:800]}")
    if differences:
        print(f"[IDENTITY FAIL] {len(differences)} differing values; first 25:")
        for key, column, a, b in differences[:25]:
            print(f"  {key} {column}: {a!r} != {b!r}")
        return 1
    if strict_missing:
        return 1
    print("[IDENTITY PASS] bitwise equal on every compared column")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("training")
    training.add_argument("--reference", required=True)
    training.add_argument("--candidate", required=True)
    training.add_argument("--policy", type=int, default=1)
    training.add_argument("--rows", type=int, default=0)
    training.add_argument("--normalize-preregistered-metadata", action="store_true")
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("--multicity", required=True)
    evaluation.add_argument("--panel", required=True)
    evaluation.add_argument("--require-all", action="store_true")
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("--reference", required=True)
    checkpoint.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    if args.command == "training":
        return command_training(args)
    if args.command == "checkpoint":
        return command_checkpoint(args)
    return command_eval(args)


if __name__ == "__main__":
    sys.exit(main())
