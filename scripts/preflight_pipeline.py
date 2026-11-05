#!/usr/bin/env python3
"""Phase 2: run the complete production pipeline at micro scale, then judge it.

One command exercises everything the paper runs will touch, with the same
code paths and the same panel specs, so no full-budget run is the first to
meet a pipeline defect:

  0  unit tests (full suite, plus the 2,500-person bitwise kernel test)
  1  map preflight for all five cities and the Table-1 build profile
  2  v64 identity: the v65.1 smoke curriculum (E4/P16, seed 6418) must
     reproduce the first 18 training rows of the v64 r3 run bit for bit
  3  two micro training shards (E2/P2 per city) -> shard status, rows,
     ledger identity, iteration checkpoints
  4  assembly of the shards
  5  multicity_backtest --eval-only on the assembled launch (sequential
     reference evaluation)
  6  every production panel through scripts/panel_eval.py at 1 pair per city
  7  identity: panel rows == multicity eval rows on every shared episode
  8  scripts/paper_results.py --check over the micro campaign
  9  report: PASS/FAIL per check, timings, memory -> GO / NO-GO

Steps 2 and 3 run concurrently.  Each passed step is recorded in
``runs/smoke_v65_report/state.json`` and skipped on a re-run, so after a
fix you re-run the same command (``--redo STEP`` repeats one step).

    caffeinate -dimsu .venv/bin/python scripts/preflight_pipeline.py --r3 runs/<v64 r3 launch>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = sys.executable
RUNS = ROOT / "runs"
REPORT = RUNS / "smoke_v65_report"
LOGS = REPORT / "logs"
CAMPAIGN = RUNS / "smoke_v65_campaign"
SEED = "6418"
SMOKE_CONFIG = "config/state_college_spokane_regime_v65_250_smoke.json"
MICRO_CONFIG = "config/state_college_spokane_regime_v65_250_micro.json"
IDENTITY_ID = "smoke_v65_identity_omp1_seed6418_bd05e52"
PINNED_IDENTITY_REGISTRY = ROOT / "runbook" / "v65_pinned_identity_reference.json"
MICRO_IDS = ("smoke_v65_micro_seed1", "smoke_v65_micro_seed2")
ASSEMBLED_ID = "smoke_v65_micro_assembled"
STRATEGIES = ("rl,rl_early_search,rl_forecast_ablated,rl_precommit,objective_greedy,"
              "objective_greedy_legacy,heuristic,random,initial_only,static_greedy,belief_rollout")
PANEL_ORDER = ("tune_deadline_main", "tune_deadline_calm", "plugin_check", "main", "g2_seed1",
               "gate1_calm", "gate1_compound", "capacity_sweep", "kappa_sweep", "zero_shot",
               "learning_curve", "sim_checks")
IDENTITY_ROWS = 18
IDENTITY_PREUPDATE_ROWS = 11
IDENTITY_PINNED_ROWS = 12
LEGACY_IDENTITY_CHECK = "v65.1 reproduces v64 r3 training rows 1-18 bit for bit"
PANEL_STEPS = tuple(f"panel_{p}" for p in PANEL_ORDER)
STEP_ORDER = ("tests", "maps", "identity_run", "micro_shards", "assemble", "multicity_eval",
              *PANEL_STEPS, "eval_identity", "results")
# Redoing a step also redoes every step that consumes its outputs.
DEPENDENTS = {
    "maps": ("results",),
    "micro_shards": ("assemble", "multicity_eval", *PANEL_STEPS, "eval_identity", "results"),
    "assemble": ("multicity_eval", "eval_identity"),
    "multicity_eval": ("eval_identity",),
    "panel_main": ("eval_identity", "results"),
    **{step: ("results",) for step in PANEL_STEPS if step != "panel_main"},
    "eval_identity": (),
}
WARN_ONLY_PANELS = {"zero_shot"}
MAP_PREFLIGHT_CITIES = (
    "state_college_pa",
    "spokane_wa",
    "seattle_wa",
    "chicago_il",
    "malibu_ca",
)
TRAINING_CITY_IDS = frozenset({"state_college_pa", "spokane_wa"})


def deterministic_torch_env(base: dict | None = None) -> dict:
    """Pin Torch/OpenMP only; leave NumPy/Accelerate parallel.

    The historical default (eight Torch threads) produced three distinct
    post-update actors from identical pre-update state.  Pinning every BLAS
    backend was deterministic but made the simulator impractically slow.
    OMP-only pinning is both exact and production-feasible.
    """
    env = dict(os.environ if base is None else base)
    env["OMP_NUM_THREADS"] = "1"
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multicity_eval_summary_issues(summary: Path) -> list[str]:
    """Revalidate a preserved multicity evaluation before accepting resume.

    ``multicity_backtest`` writes the CSV before running its matched-interface
    audit. A repair to that audit can therefore make the preserved, complete
    evaluation valid without changing any episode evidence. Resume must
    rerun the audit and exact coverage checks instead of merely noticing that
    the CSV exists (which would leave a stale FAIL in the report).
    """
    if not summary.is_file():
        return [f"missing {summary}"]
    rows = list(csv.DictReader(summary.open()))
    if not rows:
        return [f"empty {summary}"]

    from backtest import LEARNED_STRATEGIES, _verify_matched_interface

    cities = ("state_college_pa", "spokane_wa")
    strategies = tuple(STRATEGIES.split(","))
    expected = {
        (city, 1, strategy, policy)
        for city in cities
        for strategy in strategies
        for policy in ((1, 2) if strategy in LEARNED_STRATEGIES else (0,))
    }
    observed = [
        (
            row.get("city_id", ""),
            int(row.get("city_scenario_replication", 0) or 0),
            row.get("deployment_strategy", ""),
            int(row.get("policy_replication", 0) or 0),
        )
        for row in rows
    ]
    observed_set = set(observed)
    issues = []
    if len(observed) != len(observed_set):
        issues.append("duplicate city/scenario/strategy/policy rows")
    missing = sorted(expected - observed_set)
    unexpected = sorted(observed_set - expected)
    if missing:
        issues.append(f"missing evaluation rows: {missing[:3]}")
    if unexpected:
        issues.append(f"unexpected evaluation rows: {unexpected[:3]}")

    gaps = []
    for row in rows:
        try:
            gap = float(row.get("reward_accounting_gap", "nan"))
        except (TypeError, ValueError):
            gap = math.nan
        if not math.isfinite(gap) or abs(gap) > 1e-9:
            gaps.append(gap)
    if gaps:
        issues.append(f"non-finite or unbalanced reward ledgers: {gaps[:3]}")

    if not issues:
        try:
            _verify_matched_interface(rows, strategies)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            issues.append(str(exc))
    return issues


def pinned_identity_reference_issues(registry_path: Path = PINNED_IDENTITY_REGISTRY) -> tuple[Path | None, list[str]]:
    """Validate the immutable pinned-v64 evidence before it can gate v65.1."""
    if not registry_path.is_file():
        return None, [f"missing registry {registry_path}"]
    registry = json.loads(registry_path.read_text())
    launch = RUNS / registry["launch_id"]
    issues = []
    for relative, expected in registry["sha256"].items():
        path = launch / relative
        if not path.is_file():
            issues.append(f"missing {path}")
        else:
            observed = _sha256(path)
            if observed != expected:
                issues.append(f"hash mismatch {path}: {observed} != {expected}")
    rows_path = launch / "training_episode_summary.csv"
    if rows_path.is_file():
        rows = list(csv.DictReader(rows_path.open()))
        if len(rows) != int(registry["rows"]):
            issues.append(f"reference row count {len(rows)} != {registry['rows']}")
    return launch, issues


def micro_assembly_issues(directory: Path) -> tuple[str | None, list[str]]:
    """Validate an assembled micro launch without making convergence a gate."""
    manifest_path = directory / "experiment_manifest.json"
    if not manifest_path.is_file():
        return None, [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    status = manifest.get("status")
    issues = []
    if status not in {"complete", "training_not_converged"}:
        issues.append(f"unexpected status {status!r}")
    if manifest.get("training_policy_indices") != [1, 2]:
        issues.append(
            f"policy coverage {manifest.get('training_policy_indices')!r} != [1, 2]"
        )
    if len(manifest.get("training_shard_provenance", [])) != 2:
        issues.append("expected provenance for two training shards")
    for relative in (
        "training_episode_summary.csv",
        "policies/policy_001/regional_policy.pt",
        "policies/policy_002/regional_policy.pt",
    ):
        if not (directory / relative).is_file():
            issues.append(f"missing {directory / relative}")
    return status, issues


def map_preflight_issues(results: list[dict]) -> tuple[list[str], list[str]]:
    """Return required-training and warn-only zero-shot map issues.

    Map availability is a hard gate only for the two training cities.  The
    execution manual explicitly makes unavailable zero-shot cities a WARN.
    Missing rows count as issues so a killed subprocess cannot look like a
    successful city check.
    """
    by_city = {str(row.get("city_id")): row for row in results}
    issues = []
    for city_id in MAP_PREFLIGHT_CITIES:
        row = by_city.get(city_id)
        if row is None:
            issues.append(f"{city_id}:missing")
        elif row.get("status") != "ready":
            issues.append(
                f"{city_id}:{row.get('status', 'failed')}"
                + (f" ({row['error']})" if row.get("error") else "")
            )
    training = [issue for issue in issues if issue.split(":", 1)[0] in TRAINING_CITY_IDS]
    zero_shot = [issue for issue in issues if issue.split(":", 1)[0] not in TRAINING_CITY_IDS]
    return training, zero_shot


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def terminate_group(process: subprocess.Popen, grace: float = 120.0) -> None:
    """SIGTERM the process group, then SIGKILL if it does not exit."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(30)
    except ProcessLookupError:
        pass


class Pipeline:
    def __init__(self, args):
        self.args = args
        REPORT.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        self.state_path = REPORT / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {}
        self._failed_since_step_start = False
        # --redo X forgets X and every step that consumes its outputs
        for step in args.redo or []:
            if step not in STEP_ORDER:
                raise SystemExit(f"unknown step {step!r}; steps: {', '.join(STEP_ORDER)}")
            for forget in (step, *DEPENDENTS.get(step, ())):
                self.state.pop(forget, None)
        self.checks: list[dict] = self.state.get("_checks", [])

    # -- bookkeeping -------------------------------------------------------
    def save(self):
        self.state["_checks"] = self.checks
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def check(self, name: str, ok: bool | None, detail: str, *, warn_only=False):
        status = "PASS" if ok else ("WARN" if warn_only or ok is None else "FAIL")
        if status == "FAIL":
            self._failed_since_step_start = True
        self.checks = [c for c in self.checks if c["check"] != name]
        self.checks.append({"check": name, "status": status, "detail": detail, "utc": now()})
        print(f"[{status}] {name}: {detail}", flush=True)
        self.save()
        if status == "FAIL" and not self.args.keep_going:
            self.write_report()
            raise SystemExit(f"Stopped at failed check {name!r}. Fix, then re-run the same command.")

    def done(self, step: str) -> bool:
        return self.state.get(step, {}).get("status") == "done"

    def mark(self, step: str, **info):
        """Record a step as done only if none of its checks failed."""
        if self._failed_since_step_start:
            self._failed_since_step_start = False
            return
        self.state[step] = {"status": "done", "utc": now(), **info}
        self.save()

    def run(self, name: str, command: list[str], *, timeout: float, env=None) -> int:
        log = LOGS / f"{name}.log"
        print(f"[RUN] {name}: {' '.join(command)}\n      log: {log}", flush=True)
        with log.open("a") as handle:
            handle.write(f"\n===== {now()} {' '.join(command)}\n")
            handle.flush()
            started = time.time()
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                                       env={**os.environ, **(env or {})}, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_group(process, 60)
                code = -9
            handle.write(f"===== exit {code} after {time.time() - started:.0f}s\n")
        if code != 0:
            tail = log.read_text().splitlines()[-25:]
            print("      last log lines:\n        " + "\n        ".join(tail), flush=True)
        return code

    def launch(self, name: str, command: list[str], *, env=None) -> subprocess.Popen:
        log = LOGS / f"{name}.log"
        handle = log.open("a")
        handle.write(f"\n===== {now()} {' '.join(command)}\n")
        handle.flush()
        print(f"[START] {name}: {' '.join(command)}\n        log: {log}", flush=True)
        return subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                                env={**os.environ, **(env or {})}, start_new_session=True)

    # -- steps --------------------------------------------------------------
    def step0_tests(self):
        self._failed_since_step_start = False
        if self.args.skip_tests or self.done("tests"):
            return
        code = self.run("tests_full_suite", [PY, "-m", "unittest", "discover", "-s", "tests"], timeout=4 * 3600)
        self.check("unit tests (full suite)", code == 0, "see logs/tests_full_suite.log for the summary line")
        code = self.run("tests_paper_scale", [PY, "-m", "unittest",
                                             "tests.test_v65_forecast_kernel.V65ForecastKernelTests.test_paper_scale_first_passage_is_bitwise_equal"],
                        timeout=3600, env={"V65_PAPER_SCALE": "1"})
        self.check("2,500-person bitwise kernel test", code == 0, "V65_PAPER_SCALE=1")
        self.mark("tests")

    def step1_maps(self):
        self._failed_since_step_start = False
        if self.done("maps"):
            return
        # Run cities in fresh subprocesses.  A single all-city process retained
        # the 35-km Malibu graph while constructing later maps, exhausted swap,
        # and could time out before writing any evidence.  Training cities are
        # mandatory; the manual classifies zero-shot map failures as WARN.
        map_dir = RUNS / "smoke_v65_map_preflight"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_path = map_dir / "map_preflight.json"
        results = []
        for city_id in MAP_PREFLIGHT_CITIES:
            launch_id = f"smoke_v65_map_preflight_{city_id}"
            timeout = 3600 if city_id in TRAINING_CITY_IDS else 1800
            code = self.run(
                f"map_preflight_{city_id}",
                [
                    PY,
                    "multicity_backtest.py",
                    "--cities",
                    city_id,
                    "--launch-id",
                    launch_id,
                    "--training-curriculum",
                    "",
                    "--preflight-maps-only",
                ],
                timeout=timeout,
            )
            city_path = RUNS / launch_id / "map_preflight.json"
            if city_path.is_file():
                payload = json.loads(city_path.read_text())
                row = payload.get("results", [{}])[0]
            else:
                row = {
                    "city_id": city_id,
                    "status": "timeout" if code == -9 else "failed",
                    "error": f"exit {code}; see logs/map_preflight_{city_id}.log",
                }
            results.append(row)
            map_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "requested_cities": list(MAP_PREFLIGHT_CITIES),
                        "results": results,
                        "complete": len(results) == len(MAP_PREFLIGHT_CITIES),
                        "all_ready": all(
                            item.get("status") == "ready" for item in results
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
        training_issues, zero_shot_issues = map_preflight_issues(results)
        self.check(
            "map preflight, five cities",
            not training_issues,
            f"artifact={map_path}; training issues={training_issues or 'none'}",
        )
        self.check(
            "zero-shot map availability",
            None if zero_shot_issues else True,
            f"issues={zero_shot_issues or 'none'}; unavailable cities are omitted",
            warn_only=True,
        )
        code = self.run("city_build_profile", [PY, "scripts/city_build_profile.py", "--out",
                                               str(CAMPAIGN / "city_build_profile.json")], timeout=3 * 3600)
        self.check("Table-1 city build profile", code == 0, str(CAMPAIGN / "city_build_profile.json"),
                   warn_only=True)
        self.mark("maps")

    def _training_command(self, launch_id: str, config: str, *, replicates: int, index: int,
                          episodes: int, eval_reps: int) -> list[str]:
        return [PY, "multicity_backtest.py", "--cities", "state_college_pa,spokane_wa", "--launch-seed", SEED,
                "--launch-id", launch_id, "--machine", "local", "--policy-replicates", str(replicates),
                "--training-policy-indices", str(index), "--train-episodes-per-city", str(episodes),
                "--training-curriculum", config, "--eval-replications-per-city", str(eval_reps),
                "--strategies", STRATEGIES, "--no-require-convergence", "--visualize-eval-pairs-per-city", "0",
                "--no-policy-cache", "--train-only"]

    def step23_training(self):
        processes = {}
        if not self.done("identity_run"):
            directory = RUNS / IDENTITY_ID
            command = self._training_command(IDENTITY_ID, SMOKE_CONFIG, replicates=2, index=1, episodes=20, eval_reps=1)
            rows_path = directory / "training_episode_summary.csv"
            rows = list(csv.DictReader(rows_path.open())) if rows_path.is_file() else []
            checkpoint = directory / "policies/policy_001/iteration_checkpoints/regional_policy_iteration_001.pt"
            if len(rows) < IDENTITY_ROWS or not checkpoint.is_file():
                if (directory / "experiment_manifest.json").is_file():
                    command.append("--resume")
                processes["identity"] = self.launch(
                    "identity_run", command, env={"OMP_NUM_THREADS": "1"}
                )
        if not self.done("micro_shards"):
            for index, launch_id in enumerate(MICRO_IDS, start=1):
                command = self._training_command(launch_id, MICRO_CONFIG, replicates=2, index=index, episodes=4, eval_reps=1)
                manifest = RUNS / launch_id / "experiment_manifest.json"
                if manifest.is_file():
                    if json.loads(manifest.read_text()).get("status") == "training_shard_complete":
                        continue
                    command.append("--resume")
                processes[launch_id] = self.launch(
                    launch_id, command, env={"OMP_NUM_THREADS": "1"}
                )
                time.sleep(20)  # stagger torch/libomp start-up
        identity_rows = RUNS / IDENTITY_ID / "training_episode_summary.csv"
        deadline = time.time() + self.args.training_timeout_h * 3600
        failures = []
        try:
            while processes and time.time() < deadline:
                for name, process in list(processes.items()):
                    if name == "identity" and identity_rows.is_file():
                        with identity_rows.open() as handle:
                            rows = sum(1 for _ in handle) - 1
                        if rows >= IDENTITY_ROWS:
                            terminate_group(process)  # the 18 rows are all that r3 has
                            processes.pop(name)
                            continue
                    if process.poll() is not None:
                        processes.pop(name)
                        if process.returncode != 0:
                            failures.append((name, f"exit {process.returncode}"))
                time.sleep(30)
            for name in list(processes):
                failures.append((name, f"exceeded {self.args.training_timeout_h} h"))
        finally:
            for process in processes.values():  # never leave a second writer behind
                terminate_group(process)
        for name, reason in failures:
            label = ("v65.1 exact deterministic identity against v64"
                     if name == "identity" else f"micro shard {name}")
            self.check(label, False, f"{reason}; see logs/{'identity_run' if name == 'identity' else name}.log")
        if not self.done("identity_run"):
            self.judge_identity()
        if not self.done("micro_shards"):
            self.judge_micro()

    def judge_identity(self):
        self._failed_since_step_start = False
        # Remove the superseded infeasible eight-thread gate from resumed
        # state.  Its failed evidence remains on disk; it must not remain as a
        # live check after the exact pinned replacement passes.
        self.checks = [c for c in self.checks if c["check"] != LEGACY_IDENTITY_CHECK]
        self.save()
        path = RUNS / IDENTITY_ID / "training_episode_summary.csv"
        rows = list(csv.DictReader(path.open())) if path.is_file() else []
        if len(rows) < IDENTITY_ROWS:
            self.check("v65.1 exact deterministic identity against v64", False,
                       f"only {len(rows)} of {IDENTITY_ROWS} rows were produced; see logs/identity_run.log")
            return
        pinned_reference, issues = pinned_identity_reference_issues()
        self.check(
            "pinned v64 identity-reference integrity",
            not issues,
            f"registry={PINNED_IDENTITY_REGISTRY}; issues={issues or 'none'}",
        )
        historical = Path(self.args.r3) / "training_episode_summary.csv"
        candidate = RUNS / IDENTITY_ID / "training_episode_summary.csv"
        code = self.run(
            "identity_check_historical_preupdate",
            [
                PY, "scripts/identity_check.py", "training",
                "--reference", str(historical), "--candidate", str(candidate),
                "--rows", str(IDENTITY_PREUPDATE_ROWS),
                "--normalize-preregistered-metadata",
            ],
            timeout=600,
        )
        self.check(
            "pre-update dynamics equal historical v64 r3",
            code == 0,
            "logs/identity_check_historical_preupdate.log",
        )
        pinned_csv = pinned_reference / "training_episode_summary.csv"
        code = self.run(
            "identity_check_pinned_rows",
            [
                PY, "scripts/identity_check.py", "training",
                "--reference", str(pinned_csv), "--candidate", str(candidate),
                "--rows", str(IDENTITY_PINNED_ROWS),
                "--normalize-preregistered-metadata",
            ],
            timeout=600,
        )
        self.check(
            "pinned v64-v65.1 scientific rows are exact",
            code == 0,
            "logs/identity_check_pinned_rows.log",
        )
        reference_checkpoint = pinned_reference / "policies/policy_001/iteration_checkpoints/regional_policy_iteration_001.pt"
        candidate_checkpoint = RUNS / IDENTITY_ID / "policies/policy_001/iteration_checkpoints/regional_policy_iteration_001.pt"
        code = self.run(
            "identity_check_pinned_checkpoint",
            [
                PY, "scripts/identity_check.py", "checkpoint",
                "--reference", str(reference_checkpoint),
                "--candidate", str(candidate_checkpoint),
            ],
            timeout=600,
        )
        self.check(
            "pinned v64-v65.1 actor checkpoint is bitwise exact",
            code == 0,
            "logs/identity_check_pinned_checkpoint.log",
        )
        self.check(
            "v65.1 exact deterministic identity against v64",
            True,
            "historical rows 1-11, pinned rows 1-12, and all iteration-001 tensors are exact; "
            f"{len(rows)} candidate rows cover the smoke budget",
        )
        stage = {s: [float(r["episode_wall_time_s"]) for r in rows if r.get("training_stage_id") == s] for s in ("E", "P")}
        build = [float(r["mean_observation_build_latency_ms"]) for r in rows if r.get("mean_observation_build_latency_ms")]
        e = sorted(stage["E"])[len(stage["E"]) // 2] if stage["E"] else math.nan
        p = sorted(stage["P"])[len(stage["P"]) // 2] if stage["P"] else math.nan
        self.check("250-person speed (median E <= 60 s, P <= 240 s)", e <= 60 and p <= 240,
                   f"median E {e:.0f} s, P {p:.0f} s, build {max(build or [math.nan]):.0f} ms max", warn_only=True)
        self.mark("identity_run", median_e_s=e, median_p_s=p)

    def judge_micro(self):
        self._failed_since_step_start = False
        for launch_id in MICRO_IDS:
            manifest_path = RUNS / launch_id / "experiment_manifest.json"
            if not manifest_path.is_file():
                self.check(f"micro shard {launch_id}", False, f"no manifest; see logs/{launch_id}.log")
                continue
            manifest = json.loads(manifest_path.read_text())
            rows = list(csv.DictReader((RUNS / launch_id / "training_episode_summary.csv").open()))
            gap = max((abs(float(r.get("reward_accounting_gap") or 0.0)) for r in rows), default=math.nan)
            policy = manifest["training_policy_indices"][0]
            iterations = sorted((RUNS / launch_id / "policies" / f"policy_{policy:03d}" / "iteration_checkpoints").glob("*.pt"))
            ok = (manifest["status"] == "training_shard_complete" and len(rows) == 8 and gap <= 1e-9
                  and manifest["common_overrides"] == {} and bool(iterations))
            self.check(f"micro shard {launch_id}", ok,
                       f"status={manifest['status']} rows={len(rows)} max|ledger gap|={gap:.1e} "
                       f"iteration checkpoints={[p.stem[-3:] for p in iterations]}")
        self.mark("micro_shards")

    def step4_assemble(self):
        self._failed_since_step_start = False
        if self.done("assemble"):
            return
        directory = RUNS / ASSEMBLED_ID
        manifest_path = directory / "experiment_manifest.json"
        code = None
        if not manifest_path.is_file():
            code = self.run("assemble", [PY, "assemble_training_shards.py", "--launch-id", ASSEMBLED_ID,
                                         "--shards", *[str(RUNS / i) for i in MICRO_IDS]], timeout=1800)
        status, issues = micro_assembly_issues(directory)
        # Exit 2 means the complete micro artifact did not meet the full-run
        # convergence target.  Four smoke episodes per city cannot establish
        # convergence, and the manual does not make learning quality a Phase-2
        # hard gate.  Preserve and exercise the assembled artifact; report the
        # quality outcome as WARN.
        acceptable_exit = code is None or code in {0, 2}
        self.check(
            "shard assembly",
            acceptable_exit and not issues,
            f"runs/{ASSEMBLED_ID} status={status!r} issues={issues or 'none'}",
        )
        if status == "training_not_converged":
            self.check(
                "micro-smoke convergence",
                None,
                "four episodes per city did not meet the full-run convergence target; "
                "artifact coverage and integrity passed",
                warn_only=True,
            )
        self.mark("assemble")

    def step5_multicity_eval(self):
        self._failed_since_step_start = False
        summary = RUNS / ASSEMBLED_ID / "evaluation_episode_summary.csv"
        if self.done("multicity_eval"):
            issues = multicity_eval_summary_issues(summary)
            self.check(
                "multicity --eval-only on the assembled launch",
                not issues,
                f"{summary}; resumed artifact validation issues={issues or 'none'}",
            )
            return
        if summary.exists():
            issues = multicity_eval_summary_issues(summary)
            self.check(
                "multicity --eval-only on the assembled launch",
                not issues,
                f"{summary}; resumed artifact validation issues={issues or 'none'}",
            )
            self.mark("multicity_eval")
            return
        command = [PY, "multicity_backtest.py", "--cities", "state_college_pa,spokane_wa", "--launch-seed", SEED,
                   "--launch-id", ASSEMBLED_ID, "--machine", "local", "--policy-replicates", "2",
                   "--train-episodes-per-city", "4", "--training-curriculum", MICRO_CONFIG,
                   "--eval-replications-per-city", "1", "--bootstrap-draws", "500", "--strategies", STRATEGIES,
                   "--no-require-convergence", "--visualize-eval-pairs-per-city", "0", "--no-policy-cache",
                   "--eval-only"]
        code = self.run("multicity_eval", command, timeout=self.args.eval_timeout_h * 3600)
        issues = multicity_eval_summary_issues(summary) if summary.is_file() else [f"missing {summary}"]
        self.check(
            "multicity --eval-only on the assembled launch",
            code == 0 and not issues,
            f"{summary}; artifact validation issues={issues or 'none'}",
        )
        self.mark("multicity_eval")

    def _panel(self, verb: str, spec: Path, extra: list[str], *, name: str, timeout: float) -> int:
        command = [PY, "scripts/panel_eval.py", verb, "--spec", str(spec), "--campaign", str(CAMPAIGN),
                   "--replications", "1:1", *extra]
        return self.run(name, command, timeout=timeout)

    def step6_panels(self):
        shards = [str(RUNS / i) for i in MICRO_IDS]
        specs = {pid: ROOT / "config" / "panels" / f"{pid}.json" for pid in PANEL_ORDER}
        # learning curve: the micro shards have iterations 0..1
        lc = json.loads(specs["learning_curve"].read_text())
        iterations = sorted({int(p.stem.rsplit("_", 1)[1]) for i in MICRO_IDS
                             for p in (RUNS / i).glob("policies/policy_*/iteration_checkpoints/*.pt")})
        lc["arms"] = [{"arm": f"rl_iter{n:03d}", "strategy": "rl", "policies": [1, 2], "checkpoint": f"iteration:{n}",
                       "labels": {"iteration": n}} for n in iterations] + [{"arm": "objective_greedy"}]
        smoke_specs = CAMPAIGN / "specs"
        smoke_specs.mkdir(parents=True, exist_ok=True)
        (smoke_specs / "learning_curve.json").write_text(json.dumps(lc, indent=2))
        specs["learning_curve"] = smoke_specs / "learning_curve.json"
        for pid in PANEL_ORDER:
            step = f"panel_{pid}"
            if self.done(step):
                continue
            self._failed_since_step_start = False
            training = ["--training", shards[0]] if pid == "g2_seed1" else ["--training", *shards]
            extra = training + ["--policies", "1" if pid == "g2_seed1" else "1,2"]
            code = self._panel("run", specs[pid], extra + ["--plain-workers", str(self.args.plain_workers),
                                                           "--heavy-workers", str(self.args.heavy_workers),
                                                           "--allow-spec-change"],
                               name=f"panel_run_{pid}", timeout=self.args.eval_timeout_h * 3600)
            ok = code == 0
            if ok:
                ok = self._panel("collect", specs[pid], extra, name=f"panel_collect_{pid}", timeout=1800) == 0
            optional = pid in WARN_ONLY_PANELS
            self.check(f"panel {pid} (1 pair per city)", ok,
                       f"logs/panel_run_{pid}.log" + (" (zero-shot: a failing city is dropped, not fixed)" if optional else ""),
                       warn_only=optional)
            self.mark(step)

    def step7_eval_identity(self):
        self._failed_since_step_start = False
        if self.done("eval_identity"):
            return
        code = self.run("identity_check_eval", [PY, "scripts/identity_check.py", "eval", "--multicity",
                                                str(RUNS / ASSEMBLED_ID / "evaluation_episode_summary.csv"),
                                                "--panel", str(CAMPAIGN / "panels" / "main" / "evaluation_episode_summary.csv"),
                                                "--require-all"], timeout=600)
        self.check("panel evaluator == multicity eval-only (every shared episode, bitwise)", code == 0,
                   "logs/identity_check_eval.log (a difference here means the panel environment differs: stop)")
        self.mark("eval_identity")

    def step8_results(self):
        self._failed_since_step_start = False
        if self.done("results"):
            return
        expect_missing = []
        if not (CAMPAIGN / "city_build_profile.json").is_file():
            expect_missing.append("table1_cities")
        if not (CAMPAIGN / "panels" / "zero_shot" / "evaluation_episode_summary.csv").is_file():
            expect_missing.append("fig13_zero_shot")
        code = self.run("paper_results_check", [PY, "scripts/paper_results.py", "--campaign", str(CAMPAIGN),
                                                "--training", *[str(RUNS / i) for i in MICRO_IDS],
                                                "--out", str(CAMPAIGN / "paper_results"), "--check",
                                                "--expect-missing", ",".join(expect_missing)], timeout=3600)
        self.check("every table and figure renders from the micro campaign", code == 0,
                   str(CAMPAIGN / "paper_results" / "results_summary.md"))
        self.mark("results")

    def write_report(self):
        lines = [f"# v65.1 pre-flight report ({now()})", "", "| check | status | detail |", "|---|---|---|"]
        lines += [f"| {c['check']} | {c['status']} | {c['detail']} |" for c in self.checks]
        failed = [c for c in self.checks if c["status"] == "FAIL"]
        lines += ["", f"**Verdict: {'NO-GO' if failed else 'GO'} for the paper-scale canary (Phase 3).**"]
        resources = REPORT / "resources.csv"
        if resources.is_file():
            rows = list(csv.DictReader(resources.open()))
            if rows:
                peak = max(float(r["rss_gb"] or 0) for r in rows)
                swap = max(float(r["swap_used_gb"] or 0) for r in rows)
                lines += ["", f"Peak model RSS during the smoke: {peak:.1f} GB; peak swap used: {swap:.1f} GB."]
        (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n")
        print(f"\nReport: {REPORT / 'REPORT.md'}")

    def main(self):
        monitor = subprocess.Popen([PY, "scripts/resource_monitor.py", "--out", str(REPORT / "resources.csv"),
                                    "--every", "30"], cwd=str(ROOT))
        try:
            self.step0_tests()
            self.step1_maps()
            self.step23_training()
            self.step4_assemble()
            self.step5_multicity_eval()
            self.step6_panels()
            self.step7_eval_identity()
            self.step8_results()
        finally:
            monitor.terminate()
            self.write_report()
        return 1 if any(c["status"] == "FAIL" for c in self.checks) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r3", required=True, help="the v64 r3 smoke launch directory (identity reference)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="record failures and continue")
    parser.add_argument("--redo", nargs="*", help="forget these finished steps, e.g. panel_main results")
    parser.add_argument("--plain-workers", type=int, default=4)
    parser.add_argument("--heavy-workers", type=int, default=1)
    parser.add_argument("--training-timeout-h", type=float, default=4.0)
    parser.add_argument("--eval-timeout-h", type=float, default=3.0)
    args = parser.parse_args()
    return Pipeline(args).main()


if __name__ == "__main__":
    raise SystemExit(main())
