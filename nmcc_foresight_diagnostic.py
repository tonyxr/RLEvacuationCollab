#!/usr/bin/env python3
"""Does the zero-horizon NMCC label rank shelter sites like their real future value?

Diagnosis for the proposed NMCC foresight redesign (2026-09-29).

For decision states reached by the frozen v36 policy on unseen State College
scenarios, every feasible candidate cell is installed in turn and the episode
is continued to the physical horizon by the same frozen v36 policy.  Branches
share the complete pre-decision trajectory (deterministic replay, verified by
the observation digest) and, per tape, the identical post-decision exogenous
random streams.  Three candidate scores are compared with that ground truth:

* ``C0``   - the production zero-horizon label (``_ImmediateCandidateEvaluator``);
             this is exactly the value table the v36 actor was trained on;
* ``C15``  - realized cost over one decision interval plus the endpoint
             time-to-safety estimate (the TODO P0 congestion-aware label);
* ``Tfull``- realized hazard-weighted person-minutes of the decision-time
             rational cohort from the decision to the horizon, under the
             frozen policy's own continuation.  This is Q^pi(s, a) up to a
             state constant: the quantity policy improvement needs.

It also records post-decision-state descriptors of every candidate, so the
analysis can test whether the foresight gap is learnable from the
post-decision state (the basis of the proposed design).

Only production code paths are used.  Two class-level wrappers are installed
in each worker process: ``RLBridge._select_action`` (forces the candidate at
decision k and reseeds the exogenous tape) and ``RLBridge.step`` (accumulates
cohort mass after the decision).  Nothing in the model code is modified.

Usage (repository root, the project's own virtual environment):

    .venv/bin/python nmcc_foresight_diagnostic.py collect \
        --out runs/nmcc_foresight_diagnostic_20260929 \
        --populations 2500 10000 --scenarios 3 --decisions 1 3 5 --tapes 2 \
        --workers 6
    .venv/bin/python nmcc_foresight_diagnostic.py analyze \
        --out runs/nmcc_foresight_diagnostic_20260929

``collect`` is resumable: every finished branch is one JSON file, and a
re-run skips it.  ``analyze`` needs only numpy, pandas and matplotlib.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = "runs/state_college_v36_complete_multiscale_nmcc_20260927"
SEED_STREAM = 777  # not used by training (city rank streams) or evaluation (500/600)


# =============================================================================
# Worker side: production episode with two class-level wrappers
# =============================================================================

class _ProbeDone(Exception):
    """Raised after the decision-k label table is recorded (probe jobs)."""


class _Plan:
    def __init__(self):
        self.reset(None)

    def reset(self, job):
        self.job = job
        self.active = job is not None
        self.k = int(job["k"]) if job else None
        self.mode = job["mode"] if job else None
        self.cell = job.get("cell") if job else None
        self.tape_seed = job.get("tape_seed") if job else None
        self.interval = int(job.get("interval", 15)) if job else 15
        self.tk = None
        self.rational_ids = None
        self.all_ids = None
        self.bridge = None
        self.out = {}
        self.mass = np.zeros(4)  # rational plain, rational exposure, all plain, all exposure
        self.mass15 = None
        self.endpoint15 = None
        self.trajectory = []


_PLAN = _Plan()
_WRAPPED = False


def _cohorts(core):
    rational, everyone = set(), set()
    for pedestrian in core.pedDS.pedAgentList.values():
        if bool(getattr(pedestrian, "terminated", False)):
            continue
        pid = int(pedestrian.agentID)
        everyone.add(pid)
        if not bool(getattr(pedestrian, "panicked", False)):
            rational.add(pid)
    return rational, everyone


def _label_table(bridge, cells):
    """Zero-horizon label for every candidate plus post-decision descriptors.

    Replicates ``_ImmediateCandidateEvaluator.effects`` so that the resulting
    assignment of every candidate is available; the reward of each candidate
    is cross-checked against the production ``effects`` call.
    """
    import CounterfactualBranch as CB

    core = bridge.core
    brancher = CB.CounterfactualBrancher(
        core,
        horizon=0,
        reward_model=bridge.reward_model,
        initial_population=int(bridge.initial_population),
        episode_horizon=int(bridge.horizon),
    )
    ev = CB._ImmediateCandidateEvaluator(brancher)
    population = ev.population.astype(np.float64)
    n0 = float(bridge.initial_population)
    nh = n0 * float(bridge.horizon_minutes)
    free_flow = float(getattr(core, "maxSpeed", 64.0))
    wait_assign = ev.wait_assignment
    cell_y = int(core.cellTracker.cellYNum)

    nodes, capacities = {}, {}
    for cell in cells:
        node = core.shelterDS.previewShelterCandidate(divmod(int(cell), cell_y), core.cellTracker)
        if node is None:
            raise RuntimeError(f"no deployable candidate in cell {cell}")
        nodes[int(cell)] = node
        capacities[int(cell)] = int(core.shelterDS.configuredShelterCapacity(node))
    columns = {cell: ev._distance_column(node) for cell, node in nodes.items()}

    production = {
        int(effect.cell_index): effect.reward_breakdown(
            bridge.reward_model,
            initial_population=int(bridge.initial_population),
            episode_horizon_minutes=float(bridge.horizon_minutes),
        ).raw_total
        for effect in ev.effects([int(c) for c in cells])
    }

    table = []
    for cell in cells:
        cell = int(cell)
        candidate = CB._ImmediateShelter(
            shelter_id=int(core.shelterDS.nextID),
            node=nodes[cell],
            capacity=capacities[cell],
            flow=0,
        )
        shelters = ev.base_shelters + (candidate,)
        distances = dict(ev.base_distances)
        distances[candidate.shelter_id] = columns[cell]
        assignment = ev._allocate(shelters, distances)
        acted = ev._branch_result(
            label="acted", assignment=assignment, shelters=shelters,
            installed_cell=cell, installed_capacity=candidate.capacity,
        )
        effect = CB.PairedEffect(cell_index=cell, horizon=0, acted=acted,
                                 waited=ev.wait_result, common_noise=True)
        breakdown = effect.reward_breakdown(
            bridge.reward_model,
            initial_population=int(bridge.initial_population),
            episode_horizon_minutes=float(bridge.horizon_minutes),
        )
        if abs(breakdown.raw_total - production[cell]) > 1e-9:
            raise AssertionError(f"label replication mismatch in cell {cell}")

        assigned = assignment >= 0
        by_id = {s.shelter_id: s for s in shelters}
        route = np.zeros(len(assignment))
        for i in np.flatnonzero(assigned):
            route[i] = ev._route_cost(i, by_id[int(assignment[i])].node)[0]
        w_assigned = population[assigned]
        new = assignment == candidate.shelter_id
        moved = (wait_assign >= 0) & (assignment != wait_assign)
        newly = (wait_assign < 0) & assigned
        shares = np.array([population[assignment == s.shelter_id].sum() for s in shelters])
        shares = shares / max(1.0, shares.sum())
        unassigned = ~assigned
        others = [c for c in columns if c != cell]
        if others and unassigned.any():
            best = np.min(np.stack([columns[c] for c in others]), axis=0)[unassigned]
            best = np.where(np.isfinite(best), best, np.nan)
            wu = population[unassigned]
            ok = np.isfinite(best)
            future_distance = float(np.nansum(best * wu) / max(1.0, wu[ok].sum()))
            future_within_15 = float(wu[ok & (best <= free_flow * 15.0)].sum() / n0)
        else:
            future_distance, future_within_15 = 0.0, 0.0

        def wmean(x, w):
            return float((x * w).sum() / w.sum()) if w.sum() > 0 else 0.0

        table.append({
            "cell": cell,
            "label_raw_total": float(breakdown.raw_total),
            "label_total": float(breakdown.total),
            "cost_plain_action": float(acted.plain_time_to_safety_cost),
            "cost_exposure_action": float(acted.hazard_exposure_to_safety_cost),
            "cost_plain_wait": float(ev.wait_result.plain_time_to_safety_cost),
            "cost_exposure_wait": float(ev.wait_result.hazard_exposure_to_safety_cost),
            "f_cost_plain": float(acted.plain_time_to_safety_cost) / nh,
            "f_cost_exposure": float(acted.hazard_exposure_to_safety_cost) / nh,
            "f_unassigned_share": float(population[unassigned].sum() / n0),
            "f_new_served_share": float(population[new].sum() / n0),
            "f_newly_reserved_share": float(population[newly].sum() / n0),
            "f_rerouted_share": float(population[moved].sum() / n0),
            "f_mean_route_min": wmean(route[assigned], w_assigned),
            "f_p90_route_min": float(np.percentile(route[assigned], 90)) if assigned.any() else 0.0,
            "f_new_route_min": wmean(route[new], population[new]),
            "f_assignment_hhi": float((shares ** 2).sum()),
            "f_future_distance_m": future_distance,
            "f_future_within15_share": future_within_15,
            "f_candidate_danger": float(ev._cell_danger(divmod(cell, cell_y))),
            "f_capacity_share": float(candidate.capacity / n0),
        })
    return table


def _install_wrappers():
    global _WRAPPED
    if _WRAPPED:
        return
    import CounterfactualBranch as CB
    import RLBridge as RB
    from DecisionInterface import estimate_remaining_time_to_safety

    original_select = RB.RLBridge._select_action
    original_step = RB.RLBridge.step

    def select(self, observation, recurrent_step=None):
        plan = _PLAN
        if not plan.active or plan.tk is not None or int(self.deployments_made) != plan.k - 1:
            return original_select(self, observation, recurrent_step)
        plan.tk = int(observation.simulation_time)
        plan.bridge = self
        plan.rational_ids, plan.all_ids = _cohorts(self.core)
        feasible = np.flatnonzero(np.asarray(observation.action_mask, dtype=bool))
        cells = [int(c) for c in np.asarray(observation.candidate_cell_indices)[feasible]]
        decision = original_select(self, observation, recurrent_step)
        plan.out.update({
            "tk": plan.tk,
            "digest": str(self._observation_digest(observation)),
            "feasible_cells": cells,
            "policy_cell": int(observation.candidate_cell_indices[int(decision.action_index)]),
            "rational_cohort": len(plan.rational_ids),
            "all_cohort": len(plan.all_ids),
        })
        if plan.mode == "probe":
            plan.out["labels"] = _label_table(self, cells)
            raise _ProbeDone()
        target = int(plan.cell)
        index = int(np.flatnonzero(
            np.asarray(observation.candidate_cell_indices) == target)[0])
        if not bool(observation.action_mask[index]):
            raise RuntimeError(f"forced cell {target} is infeasible at decision {plan.k}")
        CB.CounterfactualBrancher._reseed_exogenous_streams(
            type("_Holder", (), {"core": self.core})(), int(plan.tape_seed)
        )
        return dataclasses.replace(decision, action_index=index)

    def step(self, *, simulation_time, is_terminal=False):
        plan = _PLAN
        t = int(simulation_time)
        if plan.active and plan.tk is not None and t > plan.tk:
            rp, re_, _ = CB.controllable_time_mass(
                self.core, plan.rational_ids
            )
            ap, ae, _ = CB.controllable_time_mass(self.core, plan.all_ids)
            minutes = float(getattr(self.core, "timeStepMinutes", 1.0))
            plan.mass += np.array([rp, re_, ap, ae]) * minutes
            if t == plan.tk + plan.interval:
                estimate = estimate_remaining_time_to_safety(
                    self.core, episode_horizon=int(self.horizon),
                    pedestrian_ids=plan.rational_ids,
                )
                plan.mass15 = plan.mass.copy()
                plan.endpoint15 = (float(estimate.plain_person_minutes),
                                   float(estimate.hazard_exposure_person_minutes))
            if t % 5 == 0:
                plan.trajectory.append((t, rp, re_))
        return original_step(self, simulation_time=simulation_time, is_terminal=is_terminal)

    RB.RLBridge._select_action = select
    RB.RLBridge.step = step
    _WRAPPED = True


def run_job(job: dict) -> dict:
    """Execute one probe or branch job in a worker process."""
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _install_wrappers()
    from backtest import _run_episode

    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _PLAN.reset(job)
    started = time.perf_counter()
    summary, status = None, "ok"
    with open(log_path, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            summary = _run_episode(
                replication=int(job["replication"]), machine="local",
                phase=str(job["phase"]), strategy="rl", train_mode=False,
                scenario_seed=int(job["scenario_seed"]), policy_seed=int(job["policy_seed"]),
                checkpoint_path=str(job["checkpoint"]), diagnostics_path=str(job["diagnostics"]),
                overrides=dict(job["overrides"]), visualization_enabled=False,
            )
        except _ProbeDone:
            status = "probe_done"
        except Exception:  # recorded, never silently dropped
            status = "error"
            traceback.print_exc()
    result = {key: job[key] for key in ("mode", "population", "scenario", "scenario_seed", "k")}
    result.update({"cell": job.get("cell"), "tape": job.get("tape"), "status": status,
                   "wall_s": time.perf_counter() - started})
    result.update(_PLAN.out)
    if job["mode"] == "branch" and status == "ok":
        if _PLAN.tk is None:
            result["status"] = "error_decision_not_reached"
        else:
            result["realized"] = _PLAN.mass.tolist()
            result["realized15"] = None if _PLAN.mass15 is None else _PLAN.mass15.tolist()
            result["endpoint15"] = _PLAN.endpoint15
            result["trajectory"] = _PLAN.trajectory
            result["summary"] = {k: summary.get(k) for k in (
                "episode_return", "safe_completed", "unfinished", "casualty",
                "restricted_mean_time_to_safety", "initial_population",
                "congestion_person_minutes", "congested_person_minute_share")}
    _PLAN.reset(None)
    return result


# =============================================================================
# Driver: plan, probe, branch
# =============================================================================

def _seed(*parts) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1, dtype=np.uint32)[0])


def _base_jobs(args):
    run = PROJECT_ROOT / args.run
    manifest = json.loads((run / "evaluation_postfix_manifest.json").read_text())
    parent = json.loads((run / "experiment_manifest.json").read_text())
    common = dict(manifest["common_effective_overrides"])
    variants = {int(v["pedVol"]): v for v in manifest["target_variants"]}
    out = (PROJECT_ROOT / args.out).resolve()
    jobs = []
    for population in args.populations:
        variant = variants[int(population)]
        overrides = {**common, "pedVol": int(variant["pedVol"]),
                     "shelterCapacityToken": int(variant["shelterCapacityToken"])}
        for scenario in range(1, args.scenarios + 1):
            scenario_seed = _seed(parent["launch_seed"], SEED_STREAM, population, scenario)
            policy_seed = _seed(parent["launch_seed"], SEED_STREAM + 1, population, scenario)
            for k in args.decisions:
                jobs.append({
                    "population": int(population), "scenario": scenario, "k": int(k),
                    "scenario_seed": scenario_seed, "policy_seed": policy_seed,
                    "overrides": overrides, "interval": int(overrides.get("shelterActionInterval", 15)),
                    "checkpoint": str(run / "policies/policy_001/regional_policy.pt"),
                    "diagnostics": str(
                        run / "policies/policy_001/learner_diagnostics.csv"
                    ),
                    "replication": scenario,
                    "tag": f"n{population}_s{scenario}_k{k}",
                    "out": str(out),
                })
    return jobs


def _execute(jobs, workers, done_path, label):
    total = len(jobs)
    if not jobs:
        return []
    results = []
    print(f"[{label}] {total} jobs on {workers} workers", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for n, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            result = future.result()
            target = Path(job["result"])
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(result))
            tmp.replace(target)
            results.append(result)
            rate = (time.perf_counter() - started) / n
            print(f"[{label}] {n}/{total} {job['tag']} status={result['status']} "
                  f"wall={result['wall_s']:.0f}s eta={(total - n) * rate / 60:.0f} min", flush=True)
            with open(done_path, "a") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} {label} {n}/{total} {job['tag']} {result['status']}\n")
    return results


def collect(args):
    out = (PROJECT_ROOT / args.out).resolve()
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    progress = out / "progress.log"
    base = _base_jobs(args)

    probes = []
    for job in base:
        target = out / "jobs" / f"probe_{job['tag']}.json"
        if target.exists() and json.loads(target.read_text())["status"] == "probe_done":
            continue
        probes.append({**job, "mode": "probe", "result": str(target),
                       "log": str(out / "logs" / f"probe_{job['tag']}.log"),
                       "phase": str(out / "episodes" / f"probe_{job['tag']}")})
    _execute(probes, args.workers, progress, "probe")

    branches = []
    for job in base:
        probe = json.loads((out / "jobs" / f"probe_{job['tag']}.json").read_text())
        if probe["status"] != "probe_done":
            raise RuntimeError(f"probe {job['tag']} failed; see its log")
        for tape in range(args.tapes):
            tape_seed = _seed(job["scenario_seed"], job["k"], 9000 + tape)
            for cell in probe["feasible_cells"]:
                tag = f"{job['tag']}_c{cell}_t{tape}"
                target = out / "jobs" / f"branch_{tag}.json"
                if target.exists() and json.loads(target.read_text())["status"] == "ok":
                    continue
                branches.append({**job, "mode": "branch", "cell": int(cell), "tape": tape,
                                 "tape_seed": tape_seed, "tag": tag, "result": str(target),
                                 "log": str(out / "logs" / f"branch_{tag}.log"),
                                 "phase": str(out / "episodes" / f"branch_{tag}")})
    # Longest (largest population, earliest decision) first for better packing.
    branches.sort(key=lambda j: (-j["population"], j["k"]))
    _execute(branches, args.workers, progress, "branch")
    print(f"[collect] complete: {out}", flush=True)


# =============================================================================
# Analysis (numpy / pandas / matplotlib only)
# =============================================================================

def _spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def load_table(out: Path):
    import pandas as pd

    probes, rows = {}, []
    for path in sorted((out / "jobs").glob("probe_*.json")):
        p = json.loads(path.read_text())
        probes[(p["population"], p["scenario"], p["k"])] = p
    for path in sorted((out / "jobs").glob("branch_*.json")):
        b = json.loads(path.read_text())
        if b["status"] != "ok":
            continue
        key = (b["population"], b["scenario"], b["k"])
        probe = probes[key]
        if b["digest"] != probe["digest"]:
            raise AssertionError(f"pre-decision state differs from probe in {path.name}")
        label = next(item for item in probe["labels"] if item["cell"] == b["cell"])
        n = float(b["population"])
        realized = np.asarray(b["realized"])
        row = {"population": b["population"], "scenario": b["scenario"], "k": b["k"],
               "tk": b["tk"], "cell": b["cell"], "tape": b["tape"],
               "policy_cell": probe["policy_cell"],
               # per-capita hazard-weighted minutes, lower is better
               "T": (realized[0] + realized[1]) / n,
               "T_all": (realized[2] + realized[3]) / n,
               "safe_share": b["summary"]["safe_completed"] / n,
               "unfinished_share": b["summary"]["unfinished"] / n,
               "rmtts": b["summary"]["restricted_mean_time_to_safety"]}
        if b["realized15"] is not None and b["endpoint15"] is not None:
            r15 = np.asarray(b["realized15"])
            row["C15"] = (r15[0] + r15[1] + b["endpoint15"][0] + b["endpoint15"][1]) / n
        else:
            row["C15"] = row["T"]
        row["C0"] = (label["cost_plain_action"] + label["cost_exposure_action"]) / n
        row["label_raw_total"] = label["label_raw_total"]
        row.update({k: v for k, v in label.items() if k.startswith("f_")})
        rows.append(row)
    return pd.DataFrame(rows), probes


FEATURES = ["f_cost_plain", "f_cost_exposure", "f_unassigned_share", "f_new_served_share",
            "f_newly_reserved_share", "f_rerouted_share", "f_mean_route_min", "f_p90_route_min",
            "f_new_route_min", "f_assignment_hhi", "f_future_distance_m",
            "f_future_within15_share", "f_candidate_danger"]


def _ridge_within_state(train, test, cols, alpha):
    """Within-state ridge: demean X and y per state, fit on train states only."""
    def demean(df, c):
        return df[c] - df.groupby("state")[c].transform("mean")
    xtr = np.column_stack([demean(train, c) for c in cols])
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    xtr = (xtr - mu) / sd
    ytr = demean(train, "Tbar").to_numpy()
    beta = np.linalg.solve(xtr.T @ xtr + alpha * np.eye(xtr.shape[1]), xtr.T @ ytr)
    xte = np.column_stack([demean(test, c) for c in cols])
    return ((xte - mu) / sd) @ beta


def analyze(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = (PROJECT_ROOT / args.out).resolve()
    df, probes = load_table(out)
    if df.empty:
        raise SystemExit("no finished branches")
    df["state"] = df.population.astype(str) + "_s" + df.scenario.astype(str) + "_k" + df.k.astype(str)
    df["group"] = df.population.astype(str) + "_s" + df.scenario.astype(str)
    tapes = sorted(df.tape.unique())
    wide = df.pivot_table(index=["state", "group", "population", "scenario", "k", "cell"],
                          columns="tape", values=["T", "C15"]).reset_index()
    wide.columns = ["_".join(str(c) for c in col if str(c) != "").strip("_") for col in wide.columns]
    feat = df[df.tape == tapes[0]].set_index(["state", "cell"])[
        ["C0", "label_raw_total", "policy_cell", "tk"] + FEATURES]
    w = wide.set_index(["state", "cell"]).join(feat).reset_index()
    tcols = [f"T_{t}" for t in tapes]
    w = w.dropna(subset=tcols).reset_index(drop=True)
    w["Tbar"] = w[tcols].mean(axis=1)
    two = len(tapes) >= 2

    # ---------------- per-state statistics
    per_state = []
    for state, g in w.groupby("state"):
        if len(g) < 4:
            continue
        rec = {"state": state, "population": int(g.population.iloc[0]), "k": int(g.k.iloc[0]),
               "group": g.group.iloc[0], "candidates": len(g),
               "Tbar_sd": float(g.Tbar.std(ddof=1)),
               "rho_C0_T": _spearman(g.C0, g.Tbar)}
        best = g.Tbar.min()
        rec["regret_C0"] = float(g.Tbar.iloc[int(np.argmin(g.C0.values))] - best)
        rec["regret_random"] = float(g.Tbar.mean() - best)
        pol = g[g.cell == g.policy_cell.iloc[0]]
        rec["regret_policy"] = float(pol.Tbar.iloc[0] - best) if len(pol) else np.nan
        rec["top1_C0"] = float(np.argmin(g.C0.values) == np.argmin(g.Tbar.values))
        if two:
            a, b = g[f"T_{tapes[0]}"].values, g[f"T_{tapes[1]}"].values
            rec["rho_TA_TB"] = _spearman(a, b)
            rec["noise_sd"] = float(np.std(a - b, ddof=1) / np.sqrt(2))
            rec["rho_C0_TA"] = 0.5 * (_spearman(g.C0, a) + _spearman(g.C0, b))
            ca, cb = g[f"C15_{tapes[0]}"].values, g[f"C15_{tapes[1]}"].values
            rec["rho_C15_T_disjoint"] = 0.5 * (_spearman(ca, b) + _spearman(cb, a))
            # selection on one tape, evaluation on the other (no winner's curse)
            rec["regret_C15_disjoint"] = 0.5 * (
                (b[np.argmin(ca)] - b.min()) + (a[np.argmin(cb)] - a.min()))
            rec["regret_tapeoracle_disjoint"] = 0.5 * (
                (b[np.argmin(a)] - b.min()) + (a[np.argmin(b)] - a.min()))
            rec["regret_C0_pertape"] = 0.5 * ((a[np.argmin(g.C0.values)] - a.min())
                                             + (b[np.argmin(g.C0.values)] - b.min()))
        else:
            rec["rho_C15_T"] = _spearman(g[f"C15_{tapes[0]}"], g.Tbar)
        per_state.append(rec)
    ps = pd.DataFrame(per_state)

    # ---------------- learnability: leave-one-scenario-group-out within-state ridge
    preds = {"C0_only": np.full(len(w), np.nan), "post_decision": np.full(len(w), np.nan)}
    for group in w.group.unique():
        tr, te = w[w.group != group], w[w.group == group]
        if tr.state.nunique() < 3:
            continue
        idx = te.index
        preds["C0_only"][idx] = _ridge_within_state(tr, te, ["C0"], 1e-6)
        preds["post_decision"][idx] = _ridge_within_state(tr, te, ["C0"] + FEATURES, float(args.alpha))
    for name, p in preds.items():
        w[f"pred_{name}"] = p
    learn = []
    for state, g in w.groupby("state"):
        if len(g) < 4 or g[f"pred_post_decision"].isna().all():
            continue
        best = g.Tbar.min()
        rec = {"state": state}
        for name in preds:
            p = g[f"pred_{name}"].values
            rec[f"rho_{name}"] = _spearman(p, g.Tbar)
            rec[f"regret_{name}"] = float(g.Tbar.iloc[int(np.nanargmin(p))] - best)
        learn.append(rec)
    lr = pd.DataFrame(learn)

    # ---------------- summary
    def m(series):
        s = pd.Series(series).dropna()
        return {"mean": float(s.mean()), "se": float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
                "n": int(len(s))}

    summary = {"states": int(ps.shape[0]), "branches": int(len(df)), "tapes": len(tapes),
               "per_decision": {}, "overall": {}}
    cols = [c for c in ps.columns if c.startswith(("rho_", "regret_", "top1_", "Tbar_sd", "noise_sd"))]
    for c in cols:
        summary["overall"][c] = m(ps[c])
    for k, g in ps.groupby("k"):
        summary["per_decision"][int(k)] = {c: m(g[c]) for c in cols}
    if not lr.empty:
        summary["learnability_leave_scenario_out"] = {c: m(lr[c]) for c in lr.columns if c != "state"}
    if two and "rho_TA_TB" in ps:
        r_ab = ps.rho_TA_TB.clip(lower=1e-3)
        rel_mean = (2 * r_ab / (1 + r_ab))
        summary["overall"]["rho_C0_T_reliability_corrected"] = m(ps.rho_C0_T / np.sqrt(rel_mean))
    (out / "analysis").mkdir(exist_ok=True)
    (out / "analysis" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    ps.to_csv(out / "analysis" / "per_state.csv", index=False)
    w.to_csv(out / "analysis" / "candidates.csv", index=False)
    if not lr.empty:
        lr.to_csv(out / "analysis" / "learnability.csv", index=False)
    _figures(w, ps, lr, out / "analysis", two, tapes)
    print(json.dumps(summary, indent=1, default=float))


def _figures(w, ps, lr, out, two, tapes):
    import matplotlib.pyplot as plt

    blue, orange, aqua, grey, ink = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e1", "font.size": 9,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    ks = sorted(ps.k.unique())

    # D1: rank agreement with the full-horizon truth
    fig, ax = plt.subplots(figsize=(8.5, 4))
    series = [("rho_C0_T", "Zero-horizon label C0 (v36 target)", blue)]
    if two:
        series += [("rho_C15_T_disjoint", "One-interval label C15 (tape-disjoint)", orange),
                   ("rho_TA_TB", "Noise ceiling: tape A vs tape B", grey)]
    width = 0.8 / len(series)
    for i, (col, lab, c) in enumerate(series):
        means = [ps[ps.k == k][col].mean() for k in ks]
        ses = [ps[ps.k == k][col].std(ddof=1) / np.sqrt((ps.k == k).sum()) for k in ks]
        x = np.arange(len(ks)) + (i - (len(series) - 1) / 2) * width
        ax.bar(x, means, width * 0.9, yerr=1.96 * np.asarray(ses), color=c, label=lab, capsize=3)
        for k_i, k in enumerate(ks):
            v = ps[ps.k == k][col].values
            ax.scatter(np.full(v.size, x[k_i]), v, s=10, color=ink, alpha=0.5, zorder=3)
    ax.axhline(0, color=ink, lw=0.8)
    ax.set_xticks(range(len(ks)), [f"installation {k}" for k in ks])
    ax.set_ylabel("Within-state Spearman with full-horizon value")
    ax.set_title("D1  Does the NMCC label rank sites like their real future value?")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "D1_rank_agreement.png", dpi=170)
    plt.close(fig)

    # D2: regret
    fig, ax = plt.subplots(figsize=(8.5, 4))
    series = [("regret_random", "Random feasible site", grey),
              ("regret_policy", "Frozen v36 choice", aqua),
              ("regret_C0", "Greedy on C0 label", blue)]
    if two:
        series += [("regret_C15_disjoint", "Greedy on C15 (tape-disjoint)", orange),
                   ("regret_tapeoracle_disjoint", "Best on other tape (noise floor)", "#4a3aa7")]
    width = 0.8 / len(series)
    for i, (col, lab, c) in enumerate(series):
        means = [ps[ps.k == k][col].mean() for k in ks]
        ses = [ps[ps.k == k][col].std(ddof=1) / np.sqrt((ps.k == k).sum()) for k in ks]
        x = np.arange(len(ks)) + (i - (len(series) - 1) / 2) * width
        ax.bar(x, means, width * 0.9, yerr=1.96 * np.asarray(ses), color=c, label=lab, capsize=3)
    ax.set_xticks(range(len(ks)), [f"installation {k}" for k in ks])
    ax.set_ylabel("Regret vs best site (hazard-weighted min / capita)")
    ax.set_title("D2  Cost of acting greedily on each score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "D2_regret.png", dpi=170)
    plt.close(fig)

    # D3: within-state scatter C0 vs truth
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 3.8), squeeze=False)
    for ax, k in zip(axes[0], ks):
        g = w[w.k == k]
        x = g.C0 - g.groupby("state").C0.transform("mean")
        y = g.Tbar - g.groupby("state").Tbar.transform("mean")
        for pop, c in zip(sorted(g.population.unique()), [blue, orange, aqua, "#eda100"]):
            s = g.population == pop
            ax.scatter(x[s], y[s], s=14, color=c, alpha=0.8, label=f"N={pop:,}")
        ax.axhline(0, color=ink, lw=0.7)
        ax.axvline(0, color=ink, lw=0.7)
        ax.set_title(f"Installation {k}: Spearman {_spearman(x, y):+.2f} (pooled, centred)")
        ax.set_xlabel("C0 post-decision cost, centred (min/capita)")
        ax.set_ylabel("Realized future cost, centred (min/capita)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "D3_label_vs_truth.png", dpi=170)
    plt.close(fig)

    # D4: learnability
    if lr is not None and not lr.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, metric, lab in [(axes[0], "rho", "Within-state Spearman with truth"),
                                (axes[1], "regret", "Regret (min / capita)")]:
            vals = [lr[f"{metric}_C0_only"], lr[f"{metric}_post_decision"]]
            ax.bar([0, 1], [v.mean() for v in vals],
                   yerr=[1.96 * v.std(ddof=1) / np.sqrt(len(v)) for v in vals],
                   color=[blue, orange], capsize=4, width=0.6)
            for i, v in enumerate(vals):
                ax.scatter(np.full(len(v), i), v, s=10, color=ink, alpha=0.5, zorder=3)
            ax.set_xticks([0, 1], ["C0 only", "C0 + post-decision\nstate descriptors"])
            ax.set_ylabel(lab)
        fig.suptitle("D4  Is the foresight gap learnable from the post-decision state? (leave-scenario-out)")
        fig.tight_layout()
        fig.savefig(out / "D4_learnability.png", dpi=170)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--run", default=DEFAULT_RUN)
    c.add_argument("--out", default="runs/nmcc_foresight_diagnostic_20260929")
    c.add_argument("--populations", type=int, nargs="+", default=[2500, 10000])
    c.add_argument("--scenarios", type=int, default=3)
    c.add_argument("--decisions", type=int, nargs="+", default=[1, 3, 5])
    c.add_argument("--tapes", type=int, default=2)
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = sub.add_parser("analyze")
    a.add_argument("--out", default="runs/nmcc_foresight_diagnostic_20260929")
    a.add_argument("--alpha", type=float, default=10.0)
    args = ap.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
