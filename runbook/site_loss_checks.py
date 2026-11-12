#!/usr/bin/env python3
"""Checks used by runbook/site_loss_run.sh (kept out of the shell script so
that macOS /bin/bash 3.2 never parses Python inside a command substitution).

    python runbook/site_loss_checks.py stack
    python runbook/site_loss_checks.py imports ROOT
    python runbook/site_loss_checks.py fingerprint FROZEN_JSON ROOT
    python runbook/site_loss_checks.py rows CSV
    python runbook/site_loss_checks.py status MANIFEST
    python runbook/site_loss_checks.py seed3 MAIN
    python runbook/site_loss_checks.py resources MAIN WORKERS [OTHER_PANEL_PID ...]
    python runbook/site_loss_checks.py gates STORE OUT EXPECTED PANEL IDENTITY_EXIT IDENTITY_LOG
    python runbook/site_loss_checks.py recourse_gates STORE OUT EXPECTED PANEL PANEL_ID ANALYSIS_EXIT ANALYSIS_LOG
                                                      IDENTITY_EXIT IDENTITY_LOG
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

CELLS = ("base", "t1_m05", "t1_m35", "t2_m05", "t2_m35")
DESIGNED_MINUTE = {"t1_m05": 34, "t2_m05": 34, "t1_m35": 64, "t2_m35": 64}
OUTPUTS = ("manipulation_check.csv", "target_delivery.csv", "tape_delivery.csv",
           "site_loss_penalty.csv", "penalty_contrasts.csv", "stressed_levels.csv",
           "site_loss_penalty.png", "site_loss_penalty.pdf", "site_loss_summary.md")


def stack() -> int:
    import numpy
    import torch

    print(sys.version.split()[0], torch.__version__, numpy.__version__)
    return 0


def imports(root: str) -> int:
    root = os.path.realpath(root)
    sys.path.insert(0, root)
    import multicity_backtest
    import scenario_hooks

    files = [os.path.realpath(m.__file__) for m in (multicity_backtest, scenario_hooks)]
    print("yes" if all(f.startswith(root + os.sep) for f in files) else "no: " + ",".join(files))
    return 0


def fingerprint(frozen_json: str, root: str) -> int:
    sys.path.insert(0, os.path.realpath(root))
    from multicity_backtest import POLICY_DYNAMICS_SOURCES, PROJECT_ROOT
    from PolicyCache import source_fingerprint

    frozen = json.loads(Path(frozen_json).read_text())
    current = dict(source_fingerprint(PROJECT_ROOT, POLICY_DYNAMICS_SOURCES))
    changed = sorted(k for k in set(frozen) | set(current) if frozen.get(k) != current.get(k))
    print("identical" if not changed and len(current) == 32 else "changed:" + ",".join(changed))
    return 0


def rows(path: str) -> int:
    with open(path, newline="", encoding="utf-8") as handle:
        print(sum(1 for _ in csv.DictReader(handle)))
    return 0


def status(manifest: str) -> int:
    print(json.loads(Path(manifest).read_text()).get("status"))
    return 0


def seed3(main: str) -> int:
    directory = Path(main) / "runs" / "paper_v65_seed3"
    state = json.loads((directory / "experiment_manifest.json").read_text()).get("status")
    with (directory / "training_episode_summary.csv").open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    print(f"{state}, {count}/200 rows")
    return 0


def _memory() -> tuple[float, float]:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9, psutil.swap_memory().used / 1e9
    except ImportError:  # macOS without psutil: free + inactive + speculative pages
        text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", text).group(1))
        pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", text)}
        swap = re.search(r"used = ([\d.]+)M", subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
        return sum(pages.values()) * page / 1e9, (float(swap.group(1)) / 1024 if swap else float("nan"))


def resources(main: str, workers: str, *others: str) -> int:
    available, swap = _memory()
    disk = shutil.disk_usage(main).free / 1e9
    print(f"[INFO] available memory {available:.1f} GB, swap used {swap:.1f} GB, free disk {disk:.0f} GB")
    print(f"[INFO] other panel_eval run processes: {' '.join(others) or 'none'}")
    reasons = []
    if available < 2.0:
        reasons.append("available memory < 2.0 GB")
    if disk < 20:
        reasons.append("free disk < 20 GB (macOS swap lives on the boot volume)")
    if others and int(workers) > 1:
        reasons.append(f"a campaign panel is running, so use 1 worker (asked {workers})")
    print("RESOURCES OK" if not reasons else "RESOURCES WAIT: " + "; ".join(reasons))
    return 1 if reasons else 0


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def gates(store: str, out: str, expected: str, panel: str, identity_exit: str, identity_log: str) -> int:
    store, out, expected = Path(store), Path(out), int(expected)
    failed = []

    def gate(ok: bool, text: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
        if not ok:
            failed.append(text)

    summary = store / "panels" / "site_loss" / "evaluation_episode_summary.csv"
    table = []
    if summary.is_file():
        with summary.open(newline="", encoding="utf-8") as handle:
            table = list(csv.DictReader(handle))
    gate(len(table) == expected, f"collected rows: {len(table)} (want {expected})")
    per_cell = {cell: sum(r.get("label_cell") == cell for r in table) for cell in CELLS}
    gate(all(n == expected // 5 for n in per_cell.values()),
         "rows per cell: " + ", ".join(f"{c} {n}" for c, n in per_cell.items()) + f" (want {expected // 5} each)")

    done = {r.get("job_key") for r in table}
    failures = store / "panels" / "site_loss" / "failures.jsonl"
    records = []
    if failures.is_file():
        records = [json.loads(line) for line in failures.read_text().splitlines() if line.strip()]
    unresolved = [r for r in records if r.get("key") not in done]
    gate(not unresolved, f"failures.jsonl: {len(records)} attempt record(s), "
                         f"{len(unresolved)} without a finished episode")
    for r in records:
        detail = (r.get("detail") or "").strip().splitlines()
        print(f"    {'UNRESOLVED' if r.get('key') not in done else 'retried ok'} {r.get('arm')} "
              f"{r.get('city_id')} r{r.get('replication')} p{r.get('policy')} {r.get('kind')}: "
              f"{detail[-1][:110] if detail else ''}")

    def read(name: str) -> list[dict]:
        path = out / name
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    check, targets, tapes = read("manipulation_check.csv"), read("target_delivery.csv"), read("tape_delivery.csv")
    stressed = [r for r in check if r["cell"] != "base"]
    missed = [r for r in targets if r["status"] == "missed"]
    inconsistent = [r for r in tapes if r["status"] == "inconsistent"]
    gate(len(stressed) == 20 and bool(targets) and not missed and not inconsistent,
         f"treatment delivery: {len(stressed)} cell x policy rows (want 20); {len(targets)} targeted shelters, "
         f"{sum(r['status'] == 'delivered' for r in targets)} delivered, "
         f"{sum(r['status'] == 'pre-empted' for r in targets)} pre-empted, {len(missed)} missed; "
         f"{len(inconsistent)} tape(s) inconsistent across policies")
    for r in stressed:
        designed = _float(r["expected_failure_minute"])
        note = ""
        if DESIGNED_MINUTE.get(r["cell"]) != round(designed):
            note = f"  WARN: designed minute differs from {DESIGNED_MINUTE.get(r['cell'])}"
        print(f"    {r['cell']:7s} {r['policy']:24s} delivered {_float(r['share_delivered']):.0%} "
              f"pre-empted {_float(r['share_preempted']):.0%} missed {_float(r['share_missed']):.0%}  "
              f"minute {_float(r['delivered_minute_min']):.0f}-{_float(r['delivered_minute_max']):.0f} "
              f"(designed {designed:.0f})  sites lost {_float(r['incremental_sites_lost_mean']):.2f}  "
              f"released {_float(r['released_people_mean']):.0f}{note}")
    for r in tapes:
        if r["status"] == "pre-empted":
            print(f"    PRE-EMPTED {r['cell']} {r['city_id']} tape {r['tape']}: initial shelter {r['target']} "
                  f"already overrun at minute {r['failure_minute']} in the base episode (all policies)")
    for r in inconsistent:
        print(f"    INCONSISTENT {r['cell']} {r['city_id']} tape {r['tape']} target {r['target']}: {r['statuses_seen']}")
    for r in missed[:10]:
        key = r.get("job_key") or ""
        print(f"    MISSED {r['cell']} {r['policy']} {r['city_id']} r{r['tape']} p{r['seed']} target {r['target']}: "
              f"failure {r['failure_minute'] or 'never'}, base {r['base_failure_minute'] or 'none'}, designed "
              f"{r['expected_failure_minute']}; {store / 'episode_store' / key[:2] / key / 'scenario.json'}")
    for r in check:
        if r["cell"] == "base":
            print(f"    base    {r['policy']:24s} initial shelters overrun per episode "
                  f"{_float(r.get('initial_shelters_overrun_mean')):.2f}, all overruns "
                  f"{_float(r['other_overruns_mean']):.2f}")

    log = Path(identity_log)
    gate(int(identity_exit) == 0, f"base cells bitwise equal to the G2 reference (full text: {log})")
    if log.is_file():
        for line in log.read_text().strip().splitlines()[-4:]:
            print(f"    {line}")
    for name in OUTPUTS:
        gate((out / name).is_file(), f"output {name}")

    walls: dict[str, list[float]] = {}
    for r in table:
        walls.setdefault(r.get("label_cell", "?"), []).append(_float(r.get("panel_wall_s")))
    for cell, values in sorted(walls.items()):
        print(f"[INFO] median episode wall time {cell}: {statistics.median(values):.0f} s "
              f"over {len(values)} episodes")
    if table and panel == "pilot":
        median = statistics.median(_float(r.get("panel_wall_s")) for r in table)
        print("[INFO] full-run projection (1,280 new episodes): "
              + ", ".join(f"{w} worker(s) {1280 * median / w / 3600:.1f} h" for w in (1, 2, 3)))
    print("GATES PASS" if not failed else f"GATES FAIL ({len(failed)} gate(s))")
    return 1 if failed else 0


RECOURSE_OUTPUTS = ("target_status.csv", "zero_order_identity.csv", "recourse_value.csv", "recourse_penalty.csv",
                    "recourse_contrasts.csv", "recourse_orders.csv", "recourse_value.png", "recourse_summary.md")
NEW_EPISODES = {"rpilot": ("rfull", 2560, ("fL30", "fL15")), "rrollpilot": ("rroll", 108, ("roll",))}


def recourse_gates(store, out, expected, panel, panel_id, analysis_exit, analysis_log, identity_exit,
                   identity_log) -> int:
    store, out, expected = Path(store), Path(out), int(expected)
    failed = []

    def gate(ok: bool, text: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
        if not ok:
            failed.append(text)

    def read(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    table = read(store / "panels" / panel_id / "evaluation_episode_summary.csv")
    gate(len(table) == expected, f"collected rows: {len(table)} (want {expected})")
    modes = sorted({r.get("label_recourse") for r in table})
    per_mode = {m: sum(r.get("label_recourse") == m for r in table) for m in modes}
    gate(bool(modes) and len(set(per_mode.values())) == 1 and "off" in modes,
         "rows per recourse mode: " + ", ".join(f"{m} {n}" for m, n in per_mode.items()))
    done = {r.get("job_key") for r in table}
    failures = store / "panels" / panel_id / "failures.jsonl"
    records = [json.loads(x) for x in failures.read_text().splitlines() if x.strip()] if failures.is_file() else []
    unresolved = [r for r in records if r.get("key") not in done]
    gate(not unresolved, f"failures.jsonl: {len(records)} attempt record(s), {len(unresolved)} without a finished episode")
    for r in records:
        detail = (r.get("detail") or "").strip().splitlines()
        print(f"    {'UNRESOLVED' if r.get('key') not in done else 'retried ok'} {r.get('arm')} {r.get('city_id')} "
              f"r{r.get('replication')} p{r.get('policy')} {r.get('kind')}: {detail[-1][:110] if detail else ''}")
    log = Path(analysis_log)
    gate(int(analysis_exit) == 0, f"analysis --check (full text: {log})")
    if log.is_file():
        for line in log.read_text().strip().splitlines()[-1:]:
            print(f"    {line[:300]}")
    gate(int(identity_exit) == 0, f"off arms bitwise equal to the G2 reference (full text: {identity_log})")
    if Path(identity_log).is_file():
        for line in Path(identity_log).read_text().strip().splitlines()[-3:]:
            print(f"    {line}")
    identity = read(out / "zero_order_identity.csv")
    bad = [r for r in identity if str(r.get("differing_columns")) != "0"]
    gate(not bad, f"zero-order episodes equal to their off twin: {len(identity) - len(bad)}/{len(identity)}")
    for r in bad[:5]:
        print(f"    DIFFERS {r['mode']} {r['cell']} {r['policy']} {r['city_id']} r{r['tape']} p{r['seed']}: "
              f"{r['differing_columns']} column(s), first {r['first_difference']}")
    status = read(out / "target_status.csv")
    counts: dict = {}
    for r in status:
        counts.setdefault(r["mode"], {}).setdefault(r["status"], 0)
        counts[r["mode"]][r["status"]] += 1
    missed = sum(c.get("missed", 0) for c in counts.values())
    gate(bool(status) and missed == 0, "targeted shelters: " + "; ".join(
        f"{m}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) for m, c in sorted(counts.items())))
    for r in read(out / "recourse_orders.csv"):
        if r["cell"] == "base" and _float(r["closed_by_order_per_episode"]) == 0.0:
            continue
        print(f"    {r['mode']:5s} {r['cell']:7s} {r['policy']:24s} closed/ep {_float(r['closed_by_order_per_episode']):.2f}"
              f"  +{_float(r['minutes_after_ignition_mean']):.0f} min after ignition  danger {_float(r['danger_at_order_mean']):.1f}"
              f"  occupants {_float(r['occupants_evacuated_mean']):.0f}  reservations {_float(r['reservations_released_mean']):.0f}"
              f"  false alarms/ep {_float(r['false_alarm_regions_per_episode']):.2f}  controller {_float(r['controller_wall_s_mean']):.1f}s")
    names = RECOURSE_OUTPUTS + (("rollout_comparison.csv",) if panel_id.endswith("rollout") else ())
    for name in names:
        gate((out / name).is_file(), f"output {name}")
    walls: dict[str, list[float]] = {}
    for r in table:
        walls.setdefault(r.get("label_recourse", "?"), []).append(_float(r.get("panel_wall_s")))
    for mode, values in sorted(walls.items()):
        print(f"[INFO] median episode wall time {mode}: {statistics.median(values):.0f} s over {len(values)} episodes")
    if panel in NEW_EPISODES:
        target, count, used = NEW_EPISODES[panel]
        values = [v for m in used for v in walls.get(m, [])]
        if values:
            median = statistics.median(values)
            print(f"[INFO] {target} projection ({count} new episodes): "
                  + ", ".join(f"{w} worker(s) {count * median / w / 3600:.1f} h" for w in (1, 2, 3)))
    print("GATES PASS" if not failed else f"GATES FAIL ({len(failed)} gate(s))")
    return 1 if failed else 0


COMMANDS = {"stack": stack, "imports": imports, "fingerprint": fingerprint, "rows": rows,
            "status": status, "seed3": seed3, "resources": resources, "gates": gates,
            "recourse_gates": recourse_gates}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(2)
    sys.exit(COMMANDS[sys.argv[1]](*sys.argv[2:]))
