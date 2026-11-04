#!/usr/bin/env python3
"""Log CPU and memory of the model's processes once a minute (macOS/Linux).

    python scripts/resource_monitor.py --out runs/paper_v65/resources.csv --every 60 &

Columns: utc, processes, rss_gb (sum over python processes whose command
line mentions this repository), cpu_percent (sum, 100 = one core), the
largest single process, system free memory and swap used (macOS: vm_stat
and sysctl vm.swapusage).  Stop it with ``kill`` or Ctrl-C.  A swap figure
that keeps growing means too many concurrent jobs: reduce workers.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])


def sample() -> dict:
    output = subprocess.run(["ps", "-A", "-o", "pid=,rss=,%cpu=,command="], capture_output=True, text=True).stdout
    rss_kb, cpu, count, largest = 0, 0.0, 0, 0
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        command = parts[3]
        if "python" not in command.lower() or (ROOT not in command and "NMCCBranchWorker" not in command
                                                and "multicity_backtest" not in command and "panel_eval" not in command):
            continue
        count += 1
        rss_kb += int(parts[1])
        largest = max(largest, int(parts[1]))
        cpu += float(parts[2])
    row = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "processes": count,
           "rss_gb": round(rss_kb / 1048576, 2), "largest_process_gb": round(largest / 1048576, 2),
           "cpu_percent": round(cpu, 1), "free_gb": "", "swap_used_gb": ""}
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", vm).group(1))
        free = sum(int(re.search(rf"{name}:\s+(\d+)", vm).group(1)) for name in ("Pages free", "Pages inactive"))
        row["free_gb"] = round(free * page / 1e9, 2)
        swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        used = re.search(r"used = ([\d.]+)M", swap)
        row["swap_used_gb"] = round(float(used.group(1)) / 1024, 2) if used else ""
    except Exception:
        pass
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=float, default=60.0)
    parser.add_argument("--count", type=int, default=0, help="stop after N samples (0 = forever)")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    taken = 0
    while True:
        row = sample()
        new = not path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        taken += 1
        if args.count and taken >= args.count:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
