#!/usr/bin/env python3
"""Table 1: per-city network size, candidates, build time and memory.

Each city is built in a fresh subprocess (so time and peak memory are not
shared), with the same OSM pipeline as ``multicity_backtest.py
--preflight-maps-only``.  ``cache_hit`` says whether the graph came from the
local cache; report the cached build time and state it.

    python scripts/city_build_profile.py --out runs/paper_v65/city_build_profile.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = r"""
import json, os, sys, time, resource, platform
sys.path.insert(0, {root!r}); os.chdir({root!r})
from CityProfiles import load_city_suite
from multicity_backtest import _preflight_maps
suite = load_city_suite({profiles!r})
city = suite.select([{city!r}])[0]
start = time.perf_counter()
result = _preflight_maps((city,), {tmp!r},
    consolidate_intersections=bool(suite.common_experiment.get("intersectionConsolidationEnabled", True)),
    intersection_tolerance_m=float(suite.common_experiment.get("intersectionConsolidationToleranceM", 5.0)),
    required_shelter_candidates=int(suite.common_experiment["shelterCanVol"]))
elapsed = time.perf_counter() - start
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_mb = peak / 1048576 if platform.system() == "Darwin" else peak / 1024
row = result["results"][0]
print("RESULT " + json.dumps({{"city_id": city.city_id, "status": row["status"], "graph_nodes": row["graph_nodes"],
    "graph_edges": row["graph_edges"], "candidate_nodes": row["stamped_building_or_amenity_nodes"],
    "radius_m": city.radius_m, "census_2020_population": getattr(city, "census_2020_population", None),
    "build_s": round(elapsed, 1), "peak_rss_mb": round(peak_mb, 0), "error": row["error"]}}))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cities", default="all")
    parser.add_argument("--city-profiles", default=str(ROOT / "config" / "city_profiles.json"))
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from CityProfiles import load_city_suite

    suite = load_city_suite(args.city_profiles)
    cities = suite.select(None if args.cities == "all" else args.cities.split(","))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for city in cities:
        cache_before = sorted(p.name for p in (ROOT / "cache").glob("graph_*.graphml"))
        code = CHILD.format(root=str(ROOT), profiles=args.city_profiles, city=city.city_id,
                            tmp=str(out.parent / f".preflight_{city.city_id}.json"))
        started = time.time()
        try:
            completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                       timeout=args.timeout, cwd=str(ROOT))
            line = next((l for l in completed.stdout.splitlines() if l.startswith("RESULT ")), None)
            row = json.loads(line[7:]) if line else {"city_id": city.city_id, "status": "failed",
                                                      "error": completed.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            row = {"city_id": city.city_id, "status": "timeout", "error": f">{args.timeout}s"}
        cache_after = sorted(p.name for p in (ROOT / "cache").glob("graph_*.graphml"))
        row["cache_hit"] = cache_before == cache_after
        row["wall_s_including_imports"] = round(time.time() - started, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps({"schema_version": 1, "cities": rows}, indent=2) + "\n")
    return 0 if all(r.get("status") == "ready" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
