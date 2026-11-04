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
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS = ROOT / "runs"
REPORT = RUNS / "smoke_v65_report"
LOGS = REPORT / "logs"
CAMPAIGN = RUNS / "smoke_v65_campaign"
SEED = "6418"
SMOKE_CONFIG = "config/state_college_spokane_regime_v65_250_smoke.json"
MICRO_CONFIG = "config/state_college_spokane_regime_v65_250_micro.json"
IDENTITY_ID = "smoke_v65_identity_seed6418"
MICRO_IDS = ("smoke_v65_micro_seed1", "smoke_v65_micro_seed2")
ASSEMBLED_ID = "smoke_v65_micro_assembled"
STRATEGIES = ("rl,rl_early_search,rl_forecast_ablated,rl_precommit,objective_greedy,"
              "objective_greedy_legacy,heuristic,random,initial_only,static_greedy,belief_rollout")
PANEL_ORDER = ("tune_deadline_main", "tune_deadline_calm", "plugin_check", "main", "g2_seed1",
               "gate1_calm", "gate1_compound", "capacity_sweep", "kappa_sweep", "zero_shot",
               "learning_curve", "sim_checks")
IDENTITY_ROWS = 18
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

    def launch(self, name: str, command: list[str]) -> subprocess.Popen:
        log = LOGS / f"{name}.log"
        handle = log.open("a")
        handle.write(f"\n===== {now()} {' '.join(command)}\n")
        handle.flush()
        print(f"[START] {name}: {' '.join(command)}\n        log: {log}", flush=True)
        return subprocess.Popen(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)

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
        # No curriculum: the two-city curriculum's 4-episode learner batch is not
        # divisible by five cities, and map preflight needs no learner settings.
        code = self.run("map_preflight", [PY, "multicity_backtest.py", "--cities", "all", "--launch-id",
                                          "smoke_v65_map_preflight", "--training-curriculum", "",
                                          "--preflight-maps-only"], timeout=3 * 3600)
        detail = "runs/smoke_v65_map_preflight/map_preflight.json"
        if code != 0:
            path = RUNS / "smoke_v65_map_preflight" / "map_preflight.json"
            if not path.is_file():
                self.check("map preflight, five cities", False, "no map_preflight.json; see logs/map_preflight.log")
                return
            payload = json.loads(path.read_text())
            failed = [r["city_id"] for r in payload["results"] if r["status"] != "ready"]
            training_ok = not {"state_college_pa", "spokane_wa"} & set(failed)
            self.check("map preflight, five cities", training_ok,
                       f"failed: {failed} (zero-shot cities may be dropped; training cities may not)",
                       warn_only=training_ok)
        else:
            self.check("map preflight, five cities", True, detail)
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
            if (directory / "experiment_manifest.json").is_file():
                command.append("--resume")
            processes["identity"] = self.launch("identity_run", command)
        if not self.done("micro_shards"):
            for index, launch_id in enumerate(MICRO_IDS, start=1):
                command = self._training_command(launch_id, MICRO_CONFIG, replicates=2, index=index, episodes=4, eval_reps=1)
                manifest = RUNS / launch_id / "experiment_manifest.json"
                if manifest.is_file():
                    if json.loads(manifest.read_text()).get("status") == "training_shard_complete":
                        continue
                    command.append("--resume")
                processes[launch_id] = self.launch(launch_id, command)
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
            label = ("v65.1 reproduces v64 r3 training rows 1-18 bit for bit"
                     if name == "identity" else f"micro shard {name}")
            self.check(label, False, f"{reason}; see logs/{'identity_run' if name == 'identity' else name}.log")
        if not self.done("identity_run"):
            self.judge_identity()
        if not self.done("micro_shards"):
            self.judge_micro()

    def judge_identity(self):
        self._failed_since_step_start = False
        path = RUNS / IDENTITY_ID / "training_episode_summary.csv"
        rows = list(csv.DictReader(path.open())) if path.is_file() else []
        if len(rows) < IDENTITY_ROWS:
            self.check("v65.1 reproduces v64 r3 training rows 1-18 bit for bit", False,
                       f"only {len(rows)} of {IDENTITY_ROWS} rows were produced; see logs/identity_run.log")
            return
        code = self.run("identity_check_training", [PY, "scripts/identity_check.py", "training", "--reference",
                                                    str(Path(self.args.r3) / "training_episode_summary.csv"),
                                                    "--candidate", str(RUNS / IDENTITY_ID / "training_episode_summary.csv"),
                                                    "--rows", str(IDENTITY_ROWS)], timeout=600)
        self.check("v65.1 reproduces v64 r3 training rows 1-18 bit for bit", code == 0,
                   "logs/identity_check_training.log")
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
        if (RUNS / ASSEMBLED_ID).exists():
            shutil.rmtree(RUNS / ASSEMBLED_ID)
        code = self.run("assemble", [PY, "assemble_training_shards.py", "--launch-id", ASSEMBLED_ID,
                                     "--shards", *[str(RUNS / i) for i in MICRO_IDS]], timeout=1800)
        self.check("shard assembly", code == 0, f"runs/{ASSEMBLED_ID}")
        self.mark("assemble")

    def step5_multicity_eval(self):
        self._failed_since_step_start = False
        if self.done("multicity_eval"):
            return
        summary = RUNS / ASSEMBLED_ID / "evaluation_episode_summary.csv"
        if summary.exists():
            self.mark("multicity_eval")
            return
        command = [PY, "multicity_backtest.py", "--cities", "state_college_pa,spokane_wa", "--launch-seed", SEED,
                   "--launch-id", ASSEMBLED_ID, "--machine", "local", "--policy-replicates", "2",
                   "--train-episodes-per-city", "4", "--training-curriculum", MICRO_CONFIG,
                   "--eval-replications-per-city", "1", "--bootstrap-draws", "500", "--strategies", STRATEGIES,
                   "--no-require-convergence", "--visualize-eval-pairs-per-city", "0", "--no-policy-cache",
                   "--eval-only"]
        code = self.run("multicity_eval", command, timeout=self.args.eval_timeout_h * 3600)
        self.check("multicity --eval-only on the assembled launch", code == 0 and summary.is_file(), str(summary))
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
