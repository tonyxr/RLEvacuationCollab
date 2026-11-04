#!/bin/bash
# Usage: runbook/train_status.sh   -- progress, speed and ETA of the three shards
source "$(cd "$(dirname "$0")" && pwd)/env.sh"
"$PY" - <<'PY'
import csv, json, os, statistics, time
from pathlib import Path
for k in (1, 2, 3):
    d = Path(f"runs/paper_v65_seed{k}")
    pid_file = Path(f"runs/_logs/train_seed{k}.pid")
    alive = False
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0); alive = True
        except (OSError, ValueError):
            pass
    rows = list(csv.DictReader((d / "training_episode_summary.csv").open())) if (d / "training_episode_summary.csv").exists() else []
    status = json.loads((d / "experiment_manifest.json").read_text()).get("status") if (d / "experiment_manifest.json").exists() else "-"
    e = [float(r["episode_wall_time_s"]) for r in rows if r.get("training_stage_id") == "E"]
    p = [float(r["episode_wall_time_s"]) for r in rows if r.get("training_stage_id") == "P"]
    recent = p[-8:] or e[-8:]
    left = 200 - len(rows)
    eta = left * (statistics.median(recent) if recent else float("nan")) / 3600
    gap = max((abs(float(r.get("reward_accounting_gap") or 0)) for r in rows), default=0.0)
    print(f"seed {k}: {len(rows):3d}/200 episodes  status={status:24s} {'ALIVE' if alive else 'stopped'}  "
          f"E median {statistics.median(e)/60 if e else float('nan'):.1f} min  "
          f"P median {statistics.median(p)/60 if p else float('nan'):.1f} min (last 8: {statistics.median(recent)/60 if recent else float('nan'):.1f})  "
          f"ETA {eta:.1f} h  max|ledger gap| {gap:.1e}")
PY
