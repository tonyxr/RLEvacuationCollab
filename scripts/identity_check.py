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
from pathlib import Path

EXCLUDE = re.compile(
    r"(wall|cpu|latency|_seconds|seconds_|_time_s$|_ms$|episode_artifacts_dir|run_dir|phase|"
    r"^replication$|visualization|^panel_|^job_key$|^arm$|^captured_|checkpoint_path|^label_|"
    r"^evaluation_population$|^evaluation_variant_id$|^panel_id$|^machine$)"
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
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("--multicity", required=True)
    evaluation.add_argument("--panel", required=True)
    evaluation.add_argument("--require-all", action="store_true")
    args = parser.parse_args(argv)
    return command_training(args) if args.command == "training" else command_eval(args)


if __name__ == "__main__":
    sys.exit(main())
