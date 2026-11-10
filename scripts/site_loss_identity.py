#!/usr/bin/env python3
"""Gate: the site_loss panel's unstressed (base) episodes equal the campaign's.

    python scripts/site_loss_identity.py \
        --site-loss runs/paper_v65_site_loss/panels/site_loss/evaluation_episode_summary.csv \
        --reference runs/paper_v65/panels/g2_seed1/evaluation_episode_summary.csv

Rows are matched on (deployment_strategy, policy_replication, city_id,
city_scenario_replication). Only site_loss rows with ``label_cell == base``
are used. Every shared column is compared with the rules of
``scripts/identity_check.py`` (JSON and Python-literal cells parse to equal
values; NaN equals NaN). The same timing, path and label columns are excluded,
plus:

* ``map_provenance``: it records the absolute cache path, which differs
  between the campaign checkout and the worktree that runs the site_loss
  panel; the graph digests inside it are checked separately;
* every ``scenario_*`` column.

Exit 0 when every common episode is bitwise equal on every compared column
and at least ``--min-common`` episodes are common. This proves that the
worktree, the hook machinery and the episode store reproduce the campaign's
evaluation exactly for unstressed episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import identity_check as IC  # noqa: E402

EXTRA = re.compile(r"(^map_provenance$|^scenario_)")
KEY = ("deployment_strategy", "policy_replication", "city_id", "city_scenario_replication")


def _key(row: dict) -> tuple:
    return (row["deployment_strategy"], str(int(float(row["policy_replication"]))), row["city_id"],
            str(int(float(row["city_scenario_replication"]))))


def _graph_digests(row: dict) -> tuple:
    try:
        provenance = json.loads(row.get("map_provenance") or "{}")
    except json.JSONDecodeError:
        provenance = IC._structure(row.get("map_provenance") or "{}")
    if not isinstance(provenance, dict):
        return (None, None)
    return (provenance.get("graph_cache_sha256"), provenance.get("consolidated_graph_cache_sha256"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site-loss", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--min-common", type=int, default=4)
    parser.add_argument("--where", action="append", default=[],
                        help="extra row filter COLUMN=VALUE, e.g. label_recourse=off (repeatable)")
    args = parser.parse_args(argv)
    filters = [tuple(item.split("=", 1)) for item in args.where]
    with args.site_loss.open(newline="", encoding="utf-8") as handle:
        base = [r for r in csv.DictReader(handle) if r.get("label_cell") == "base"
                and all(r.get(column) == value for column, value in filters)]
    with args.reference.open(newline="", encoding="utf-8") as handle:
        reference = {_key(r): r for r in csv.DictReader(handle)}
    pairs = [(reference[_key(r)], r) for r in base if _key(r) in reference]
    print(f"[SITE-LOSS IDENTITY] {len(base)} base rows, {len(pairs)} with a reference episode")
    if len(pairs) < args.min_common:
        print(f"[FAIL] fewer than {args.min_common} common episodes")
        return 1
    columns = sorted(set(pairs[0][0]) & set(pairs[0][1]))
    compared = [c for c in columns if not IC.EXCLUDE.search(c) and not EXTRA.search(c)]
    differences = []
    for ref, row in pairs:
        for column in compared:
            if not IC._same(ref.get(column, ""), row.get(column, "")):
                differences.append((_key(row), column, ref.get(column), row.get(column)))
        if _graph_digests(ref) != _graph_digests(row):
            differences.append((_key(row), "map_provenance graph digests", _graph_digests(ref), _graph_digests(row)))
    print(f"  compared {len(compared)} columns on {len(pairs)} episodes")
    if differences:
        print(f"[FAIL] {len(differences)} differing values; first 20:")
        for key, column, a, b in differences[:20]:
            print(f"  {key} {column}: {str(a)[:120]!r} != {str(b)[:120]!r}")
        return 1
    print("[PASS] base episodes are bitwise equal to the reference panel on every compared column")
    return 0


if __name__ == "__main__":
    sys.exit(main())
