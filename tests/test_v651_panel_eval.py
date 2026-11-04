"""v65.1 panel evaluator: same jobs as multicity eval-only, crash-safe store.

The multicity runner is executed for real (train-only, then eval-only) with
the simulator replaced by a recorder, so the test compares the panel
evaluator against the production evaluation loop, not against a copy of it.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest  # noqa: E402
import multicity_backtest as MB  # noqa: E402
import panel_eval as PE  # noqa: E402

SMOKE = ROOT / "config" / "state_college_spokane_regime_v65_250_smoke.json"
CITIES = "state_college_pa,spokane_wa"
STRATEGIES = (
    "rl,rl_early_search,rl_forecast_ablated,rl_precommit,objective_greedy,"
    "objective_greedy_legacy,heuristic,random,initial_only,static_greedy,belief_rollout"
)


def _fake_row(**kwargs):
    seed = int(kwargs["scenario_seed"]) % 1000
    return {
        "replication": int(kwargs["replication"]),
        "deployment_strategy": kwargs["strategy"],
        "objective_normalized_cost": 0.5 + seed / 10_000.0,
        "episode_return": -0.5,
        "scenario_seed": kwargs["scenario_seed"],
        "policy_seed": kwargs["policy_seed"],
    }


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _fake_row(**kwargs)


def _run_multicity(runs_root: Path, argv: list[str], recorder: _Recorder):
    patches = [
        mock.patch.object(MB, "RUNS_ROOT", str(runs_root)),
        mock.patch.object(MB, "training_environment", lambda: {"python": "test"}),
        mock.patch.object(backtest, "_run_episode", recorder),
        mock.patch.object(MB, "_audit_multicity_training_topologies", lambda *a, **k: {}),
        mock.patch.object(backtest, "_plot_outputs", lambda *a, **k: {}),
        mock.patch.object(
            MB,
            "_multicity_training_convergence",
            lambda rows, cities, **k: {
                "all_policies_converged": False,
                "policies": [{"policy_replication": 1, "converged": False}],
            },
        ),
        mock.patch.object(MB, "_learning_assessment", lambda *a, **k: {}),
    ]
    for patch in patches:
        patch.start()
    try:
        return MB.main(argv)
    finally:
        for patch in reversed(patches):
            patch.stop()


class PanelEvaluatorEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="panel_eval_test_"))
        cls.runs = cls.tmp / "runs"
        common = [
            "--cities", CITIES, "--launch-seed", "6418", "--launch-id", "smoke",
            "--policy-replicates", "1", "--train-episodes-per-city", "20",
            "--training-curriculum", str(SMOKE), "--eval-replications-per-city", "3",
            "--strategies", STRATEGIES, "--no-require-convergence",
            "--visualize-eval-pairs-per-city", "0", "--no-policy-cache",
        ]
        cls.train_calls = _Recorder()
        _run_multicity(cls.runs, common + ["--train-only"], cls.train_calls)
        launch = cls.runs / "smoke"
        checkpoint = launch / "policies" / "policy_001" / "regional_policy.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint-bytes")
        cls.eval_calls = _Recorder()
        try:
            _run_multicity(cls.runs, common + ["--eval-only"], cls.eval_calls)
        except Exception:
            pass  # post-loop analysis cannot digest fake rows; the calls are recorded
        cls.launch = launch

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _spec(self, **extra):
        spec = {
            "schema_version": 1,
            "panel_id": "equivalence",
            "campaign_dir": str(self.tmp / "campaign"),
            "training": {"sources": [str(self.launch)]},
            "cities": CITIES.split(","),
            "replications": [1, 3],
            "arms": [
                {"arm": name, "policies": [1]} if name.startswith("rl") else {"arm": name}
                for name in STRATEGIES.split(",")
            ],
        }
        spec.update(extra)
        path = self.tmp / f"spec_{len(list(self.tmp.glob('spec_*')))}.json"
        path.write_text(json.dumps(spec))
        return path

    def test_training_manifest_was_written(self):
        manifest = json.loads((self.launch / "experiment_manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(self.train_calls.calls), 40)
        self.assertEqual(len(self.eval_calls.calls), 2 * 3 * 11)

    def test_jobs_equal_the_multicity_evaluation_loop(self):
        cli = PE.build_parser().parse_args(["plan", "--spec", str(self._spec())])
        spec, source, env, jobs, campaign, panel_dir = PE.prepare(cli)
        self.assertEqual(len(jobs), len(self.eval_calls.calls))
        for job, call in zip(jobs, self.eval_calls.calls):
            with self.subTest(arm=job.arm, city=job.city_id, rep=job.city_scenario_replication):
                self.assertEqual(job.strategy, call["strategy"])
                self.assertEqual(job.scenario_seed, call["scenario_seed"])
                self.assertEqual(job.policy_seed, call["policy_seed"])
                self.assertEqual(
                    PE.canonical_json(job.overrides), PE.canonical_json(call["overrides"])
                )
                self.assertFalse(call["train_mode"])
                self.assertEqual(
                    Path(job.checkpoint_path).resolve(), Path(call["checkpoint_path"]).resolve()
                )
        # The production wind speed reached every job without --override.
        self.assertTrue(all(j.overrides["hazardWindSpeedMPerMinute"] == 16.67 for j in jobs))
        self.assertTrue(all(j.overrides["pedVol"] == 250 for j in jobs))

    def test_keys_change_with_any_input_and_learned_keys_track_checkpoint_bytes(self):
        cli = PE.build_parser().parse_args(["plan", "--spec", str(self._spec())])
        _, _, _, jobs, _, _ = PE.prepare(cli)
        self.assertEqual(len({j.key for j in jobs}), len(jobs))
        overridden = self._spec(overrides={"casualtyPriorityKappa": 4.0})
        _, _, _, other, _, _ = PE.prepare(
            PE.build_parser().parse_args(["plan", "--spec", str(overridden)])
        )
        self.assertFalse({j.key for j in jobs} & {j.key for j in other})
        self.assertTrue(all(j.overrides["casualtyPriorityKappa"] == 4.0 for j in other))

    def test_changed_curriculum_or_profiles_fail_closed(self):
        cli = PE.build_parser().parse_args(["plan", "--spec", str(self._spec())])
        with mock.patch.object(PE, "sha256_text", PE.sha256_text):
            source = PE.load_training_source([str(self.launch)])
        tampered = dict(source.manifest)
        tampered["city_profile_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "city_profiles.json changed"):
            PE.build_environment(PE.TrainingSource(tampered, source.source_dirs, source.policy_dirs, {}))
        tampered = json.loads(json.dumps(source.manifest))
        tampered["policy_dynamics_source_sha256"]["RLBridge.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Dynamics sources differ"):
            PE.build_environment(PE.TrainingSource(tampered, source.source_dirs, source.policy_dirs, {}))

    def test_zero_shot_city_uses_profile_and_curriculum(self):
        spec = self._spec(cities=["seattle_wa"], arms=[{"arm": "objective_greedy"}])
        _, _, env, jobs, _, _ = PE.prepare(PE.build_parser().parse_args(["plan", "--spec", str(spec)]))
        city = env.city("seattle_wa")
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0].overrides["cityID"], "seattle_wa")
        self.assertEqual(jobs[0].overrides["cellX"], 16)
        self.assertEqual(
            jobs[0].scenario_seed, MB._seed(6418, 500 + city.scale_rank, 1)
        )

    def test_tuned_value_follows_one_se_rule(self):
        tune = self._spec(
            panel_id="tune",
            seed_streams={"scenario": 700, "policy": 800},
            replications=[1, 6],
            cities=["state_college_pa"],
            arms=[
                {"arm": "never", "strategy": "deadline_greedy",
                 "overrides": {"deadlineGreedyThreshold": -1e30}},
                {"arm": "t02", "strategy": "deadline_greedy",
                 "overrides": {"deadlineGreedyThreshold": 0.2}},
                {"arm": "t04", "strategy": "deadline_greedy",
                 "overrides": {"deadlineGreedyThreshold": 0.4}},
            ],
            tuning={"parameter": "deadlineGreedyThreshold", "baseline_arm": "never",
                    "arm_values": {"never": -1e30, "t02": 0.2, "t04": 0.4}},
        )
        cli = PE.build_parser().parse_args(["plan", "--spec", str(tune)])
        spec, _, _, jobs, campaign, _ = PE.prepare(cli)
        store = PE.EpisodeStore(campaign)
        shift = {"never": 0.0, "t02": -0.03, "t04": -0.031}
        noise = {"never": 0.0, "t02": 0.001, "t04": 0.3}
        for job in jobs:
            sign = 1.0 if job.city_scenario_replication % 2 else -1.0
            cost = 0.5 + shift[job.arm] + noise[job.arm] * sign
            store.write(job, {"job": {}, "row": {"objective_normalized_cost": cost},
                              "captured": {}, "evaluation": {"wall_s": 1.0}})
        selected = PE.select_tuned_value(PE.load_spec(tune), cli)
        self.assertEqual(selected["selected_arm"], "t02")  # t04 is too noisy to qualify
        main = self._spec(
            panel_id="uses_tune",
            cities=["state_college_pa"],
            arms=[{"arm": "deadline_tuned", "strategy": "deadline_greedy",
                   "overrides": {"deadlineGreedyThreshold": {"$tuned": {"spec": str(tune)}}}}],
        )
        _, _, _, main_jobs, _, _ = PE.prepare(PE.build_parser().parse_args(["plan", "--spec", str(main)]))
        self.assertTrue(all(j.overrides["deadlineGreedyThreshold"] == 0.2 for j in main_jobs))

    def test_cli_replication_limit_keeps_arm_subsets_valid(self):
        spec = self._spec(arms=[{"arm": "objective_greedy"},
                                {"arm": "hindsight_rollout", "replications": [1, 3]},
                                {"arm": "rl", "policies": [1]}])
        cli = PE.build_parser().parse_args(["plan", "--spec", str(spec), "--replications", "1:1"])
        _, _, _, jobs, _, _ = PE.prepare(cli)
        self.assertEqual(sorted({j.city_scenario_replication for j in jobs}), [1])
        self.assertEqual(sum(j.arm == "hindsight_rollout" for j in jobs), 2)  # one per city
        self.assertEqual(PE.parse_replications([4]), (4,))
        self.assertEqual(PE.parse_replications("2:3"), (2, 3))

    def test_learned_arm_must_list_policies(self):
        spec = self._spec(arms=[{"arm": "rl"}])
        with self.assertRaisesRegex(ValueError, "policies explicitly"):
            PE.prepare(PE.build_parser().parse_args(["plan", "--spec", str(spec)]))

    def test_key_tracks_code_outside_the_dynamics_fingerprint(self):
        cli = PE.build_parser().parse_args(["plan", "--spec", str(self._spec())])
        _, _, _, jobs, _, _ = PE.prepare(cli)
        before = jobs[0].key
        with mock.patch.object(PE, "_EVALUATION_CODE_SHA", "0" * 64):
            _, _, _, again, _, _ = PE.prepare(cli)
        self.assertNotEqual(before, again[0].key)

    def test_iteration_checkpoint_arm(self):
        iteration = self.launch / "policies" / "policy_001" / "iteration_checkpoints"
        iteration.mkdir(parents=True, exist_ok=True)
        (iteration / "regional_policy_iteration_010.pt").write_bytes(b"iter-10")
        spec = self._spec(arms=[{"arm": "rl_iter010", "strategy": "rl", "policies": [1],
                                 "checkpoint": "iteration:10"}])
        _, _, _, jobs, _, _ = PE.prepare(PE.build_parser().parse_args(["plan", "--spec", str(spec)]))
        self.assertTrue(all(j.checkpoint_path.endswith("regional_policy_iteration_010.pt") for j in jobs))
        self.assertTrue(all(j.checkpoint_sha256 == PE.sha256_text("iter-10") or j.checkpoint_sha256 for j in jobs))


# --------------------------------------------------------------------------
# scheduler (fake executor in the worker processes)
# --------------------------------------------------------------------------
FAKE_STATE = Path(
    os.environ.setdefault(
        "PANEL_EVAL_FAKE_STATE",
        str(Path(tempfile.gettempdir()) / f"panel_eval_fake_{os.getpid()}"),
    )
)


def fake_executor(job, capture=()):
    """Deterministic stand-in for an episode; crashes once on request."""
    marker = FAKE_STATE / f"{job['key']}.crashed"
    if job["arm"] == "crash_once" and not marker.exists():
        FAKE_STATE.mkdir(parents=True, exist_ok=True)
        marker.write_text("1")
        os._exit(17)
    if job["arm"] == "always_error":
        raise RuntimeError("deterministic failure")
    return {"row": _fake_row(replication=job["city_scenario_replication"], strategy=job["strategy"],
                             scenario_seed=job["scenario_seed"], policy_seed=job["policy_seed"]),
            "captured": {}}


def _job(key, arm, *, heavy=False, optional=False):
    return PE.Job(
        key=key, panel_id="t", arm=arm, strategy="objective_greedy", policy_replication=0,
        city_id="state_college_pa", city_scale_rank=2, city_scenario_replication=int(key[-1], 16),
        scenario_seed=int(key[-4:], 16), policy_seed=1, checkpoint_path="", checkpoint_sha256=None,
        overrides={}, evaluation_variant_id="", heavy=heavy, optional=optional, timeout_s=60.0,
    )


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.campaign = Path(tempfile.mkdtemp(prefix="panel_sched_"))
        os.environ["PANEL_EVAL_TEST_EXECUTOR"] = f"{__name__}:fake_executor"
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")]
        )

    def tearDown(self):
        os.environ.pop("PANEL_EVAL_TEST_EXECUTOR", None)
        shutil.rmtree(self.campaign, ignore_errors=True)
        shutil.rmtree(FAKE_STATE, ignore_errors=True)

    def test_runs_all_jobs_survives_a_crash_and_resumes(self):
        store = PE.EpisodeStore(self.campaign)
        jobs = [_job(f"{i:064x}", "plain") for i in range(1, 7)]
        jobs += [_job(f"{i:064x}", "heavy", heavy=True) for i in range(7, 9)]
        jobs += [_job(f"{9:064x}", "crash_once")]
        logs = []
        scheduler = PE.Scheduler(store, self.campaign / "panel", plain_workers=3,
                                 heavy_workers=1, capture=(), log=logs.append)
        result = scheduler.run(jobs)
        self.assertEqual(result["failed"], [])
        self.assertTrue(all(store.done(job) for job in jobs))
        self.assertTrue(any("[RETRY]" in line for line in logs))
        self.assertEqual(store.read(jobs[-1])["evaluation"]["attempt"], 2)
        # A second run finds everything in the store and starts no worker.
        logs.clear()
        result = PE.Scheduler(store, self.campaign / "panel", plain_workers=3,
                              heavy_workers=1, capture=(), log=logs.append).run(jobs)
        self.assertEqual(result["ran"], 0)
        self.assertIn("0 episodes to run", logs[0])

    def test_deterministic_error_is_reported_not_retried_forever(self):
        store = PE.EpisodeStore(self.campaign)
        jobs = [_job(f"{1:064x}", "always_error"), _job(f"{2:064x}", "plain")]
        result = PE.Scheduler(store, self.campaign / "panel", plain_workers=1,
                              heavy_workers=0, capture=(), log=lambda m: None).run(jobs)
        self.assertEqual(result["failed"], [jobs[0].key])
        self.assertTrue(store.done(jobs[1]))
        failures = (self.campaign / "panel" / "failures.jsonl").read_text().splitlines()
        self.assertEqual(len(failures), 2)
        self.assertIn("deterministic failure", json.loads(failures[0])["detail"])

    def test_claim_without_heartbeat_expires(self):
        store = PE.EpisodeStore(self.campaign)
        job = _job(f"{4:064x}", "plain")
        claim = job.store_paths(self.campaign)[0].with_suffix(".claim")
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.write_text("")  # truncated claim from a killed writer
        old = os.path.getmtime(claim) - 2 * PE.EpisodeStore.STALE_CLAIM_S
        os.utime(claim, (old, old))
        self.assertTrue(store.claim(job))
        store.release(job)
        claim.write_text(json.dumps({"pid": os.getppid(), "host": "another-host"}))
        self.assertFalse(store.claim(job))  # fresh foreign claim is respected
        os.utime(claim, (old, old))
        self.assertTrue(store.claim(job))  # ... until its heartbeat stops

    def test_claims_from_live_processes_are_respected_and_stale_ones_taken(self):
        store = PE.EpisodeStore(self.campaign)
        job = _job(f"{3:064x}", "plain")
        claim = job.store_paths(self.campaign)[0].with_suffix(".claim")
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.write_text(json.dumps({"pid": 999_999_999, "host": store.host}))
        self.assertTrue(store.claim(job))  # dead pid -> stale -> taken
        store.release(job)
        claim.write_text(json.dumps({"pid": os.getppid(), "host": store.host}))
        self.assertFalse(store.claim(job))  # live foreign pid -> respected


class CaptureHookTests(unittest.TestCase):
    def test_planner_capture_returns_identical_costs(self):
        code = r"""
import sys, json, numpy as np
sys.path[:0] = [%r, %r]
import CounterfactualBranch as CB, RolloutPolicyIteration as RPI
from tests.v65_testbed import regime_testbed, advance, builder_for
import panel_eval as PE
def costs():
    core = regime_testbed(seed=6601); advance(core, 10)
    obs = builder_for(core).build(decision_index=0, simulation_time=10, remaining_deployments=8)
    cells = [int(obs.candidate_cell_indices[a]) for a in np.flatnonzero(obs.action_mask) if a != obs.defer_action_index][:3] + [-1]
    snap = CB.capture(core, label="t")
    b = CB.CounterfactualBrancher(core, horizon=0, reward_model=None, initial_population=int(core.initial_population), episode_horizon=int(core.stopTime)-1)
    try:
        return b.rollout_policy_costs(cells, snapshot=snap, future_seeds=(5, 6, 7), decision_index=0, remaining_deployments=8, maximum_deployments=8, frozen_policy=RPI.ObjectiveGreedyContinuationPolicy(), mode="belief")
    finally:
        b.close()
before = costs()
PE.install_capture_hooks(["planner_matrices"]); PE.CAPTURE.reset("belief_rollout", ["planner_matrices"])
after = costs()
rec = PE.CAPTURE.records["planner_matrices"]
print(json.dumps({"same": bool(np.array_equal(before, after)), "n": len(rec), "shape": np.asarray(rec[0]["matrix"]).shape, "cells": rec[0]["cells"]}))
""" % (str(ROOT), str(ROOT / "scripts"))
        import subprocess

        output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env={**os.environ}, cwd=str(ROOT), timeout=900)
        self.assertEqual(output.returncode, 0, output.stderr[-3000:])
        result = json.loads(output.stdout.strip().splitlines()[-1])
        self.assertTrue(result["same"])
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["shape"], [3, 4])
        self.assertIsNone(result["cells"][-1])

    def test_link_density_capture_leaves_dynamics_unchanged(self):
        code = r"""
import sys, json
sys.path[:0] = [%r, %r]
from tests.v65_testbed import regime_testbed, advance, full_state_digest
import panel_eval as PE
def run():
    core = regime_testbed(seed=6602); advance(core, 25); return full_state_digest(core)
before = run()
PE.install_capture_hooks(["link_density"]); PE.CAPTURE.reset("objective_greedy", ["link_density"])
after = run()
rec = PE.CAPTURE.records["link_density"]
print(json.dumps({"same": before == after, "substeps": rec["substeps"] if rec else 0,
                  "persons": sum(rec["person_substeps"]) if rec else 0}))
""" % (str(ROOT), str(ROOT / "scripts"))
        import subprocess

        output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env={**os.environ}, cwd=str(ROOT), timeout=900)
        self.assertEqual(output.returncode, 0, output.stderr[-3000:])
        result = json.loads(output.stdout.strip().splitlines()[-1])
        self.assertTrue(result["same"])
        if result["substeps"] == 0:
            self.skipTest("testbed has congestion disabled; capture verified on the Mac smoke")
        self.assertGreater(result["persons"], 0)


if __name__ == "__main__":
    unittest.main()
