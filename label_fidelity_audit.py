#!/usr/bin/env python3
"""Label-fidelity audit: does the actor's NMCC label rank shelter sites the way
the evaluation objective does?

For each decision state, every feasible cell is branched under common random
numbers to the END of the episode, and four labels are computed per cell:

  J_true      full-horizon realised hazard-weighted time-to-safety of the WHOLE
              population (active + panicked + casualties) + terminal residual.
              This is the evaluation objective.
  L_code      the Stage-A NMCC label as implemented: k-step realised cost of
              the decision-time controllable cohort (non-panicked, alive) plus
              the controllable remaining-time potential at step k.
  L_ctrl_full full-horizon cost of the controllable cohort only (isolates the
              cohort-exclusion effect from truncation).
  L_all_k     k-step realised cost of the whole population plus whole-population
              potential at step k (isolates truncation from exclusion).
  L_greedy    zero-horizon objective-greedy value (the benchmark heuristic).

Costs are lower-is-better; Spearman is reported against J_true.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

import CounterfactualBranch as CB
import nmcc_testbed as T
from DecisionInterface import estimate_remaining_time_to_safety
from RewardProcessor import RewardProcessor


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def population_cost_rate(core):
    snap = CB.outcome_snapshot(core)
    return float(snap.active_population + snap.casualties + snap.hazard_exposure_mass)


def run_branch(core, brancher, base, cell, timeline, seed, k, stop):
    CB.restore(core, base)
    brancher._reseed_exogenous_streams(seed)
    cohort = {
        int(p.agentID)
        for p in core.pedDS.pedAgentList.values()
        if not p.terminated and not getattr(p, "panicked", False)
    }
    cell_pair = core.cellTracker.unflatten_cell(cell)
    sid = core.shelterDS.newShelter({"cell": cell_pair}, core.cellTracker)
    shelter = core.shelterDS.shelterList[sid]
    if hasattr(core.cellTracker, "addShelter"):
        core.cellTracker.addShelter(cell_pair, shelter)
    core.pedDS.reroute_active_to_nearest_shelter(new_shelter=shelter)

    ctrl = allc = 0.0
    out = {}
    for step, frame in enumerate(timeline):
        CB.advance_one_timestep(core, hazard_frame=frame)
        a, e, casualty_cost = CB.controllable_time_mass(core, cohort)
        ctrl += a + e + casualty_cost
        allc += population_cost_rate(core)
        if step == k - 1:
            d_ctrl = estimate_remaining_time_to_safety(
                core, episode_horizon=stop, pedestrian_ids=cohort, include_panicked=False
            ).hazard_weighted_person_minutes
            d_all = estimate_remaining_time_to_safety(
                core, episode_horizon=stop, include_panicked=True
            ).hazard_weighted_person_minutes
            dead = CB.outcome_snapshot(core).casualties
            remaining = stop - int(core.pedDS.currTime)
            out["L_code"] = ctrl + d_ctrl
            out["L_all_k"] = allc + d_all + dead * remaining
        if CB.outcome_snapshot(core).active_population <= 0:
            break
    snap = CB.outcome_snapshot(core)
    terminal_all = estimate_remaining_time_to_safety(
        core, episode_horizon=stop, include_panicked=True
    ).hazard_weighted_person_minutes
    terminal_ctrl = estimate_remaining_time_to_safety(
        core, episode_horizon=stop, pedestrian_ids=cohort, include_panicked=False
    ).hazard_weighted_person_minutes
    out["J_true"] = allc + terminal_all
    out["L_ctrl_full"] = ctrl + terminal_ctrl
    out["panic"] = int(core.pedDS.result.get("panic_onset", 0))
    out["casualty"] = int(snap.casualties)
    return out


def audit_state(seed, *, pop, stop, decision_t, k, tapes, panic, hazards):
    core = T.build(
        population=pop,
        stop_time=stop,
        panic_rate=panic,
        hazard_count=hazards,
        scenario_seed=seed,
        casualty_rate=(40, 9),
        spread_rate=(30, 25),
    )
    for _ in range(decision_t):
        CB.advance_one_timestep(core)
    cells = core.feasible_cells()
    base = CB.capture(core, label="audit")
    full_h = stop - int(core.pedDS.currTime)
    brancher = CB.CounterfactualBrancher(
        core,
        horizon=full_h,
        reward_model=RewardProcessor(),
        initial_population=pop,
        episode_horizon=stop,
    )
    greedy = CB.CounterfactualBrancher(
        core, horizon=0, reward_model=RewardProcessor(),
        initial_population=pop, episode_horizon=stop,
    )
    CB.restore(core, base)
    g_effects = greedy.candidate_effects(cells, snapshot=base)
    # raw_total is WAIT-minus-action gain (higher better) -> negate to a cost
    L_greedy = [
        -e.reward_breakdown(RewardProcessor(), initial_population=pop,
                            episode_horizon_minutes=stop).raw_total
        for e in g_effects
    ]
    labels = {key: np.zeros((tapes, len(cells))) for key in
              ("J_true", "L_code", "L_ctrl_full", "L_all_k", "panic", "casualty")}
    for tape in range(tapes):
        tseed = seed * 1000 + tape
        timeline = brancher._hazard_timeline(base, tseed)
        for j, cell in enumerate(cells):
            res = run_branch(core, brancher, base, cell, timeline, tseed, k, stop)
            for key in labels:
                labels[key][tape, j] = res[key]
    CB.restore(core, base)
    mean = {key: value.mean(axis=0) for key, value in labels.items()}
    J = mean["J_true"]
    # Single-tape label (what one NMCC tape sees) vs multi-tape truth
    row = {
        "seed": seed,
        "cells": len(cells),
        "J_true_spread_sd": float(np.std(J)),
        "J_true_mean": float(np.mean(J)),
        "panic_mean": float(mean["panic"].mean()),
        "panic_spread_across_cells": float(np.ptp(mean["panic"])),
        "casualty_mean": float(mean["casualty"].mean()),
        "rho_L_code": spearman(mean["L_code"], J),
        "rho_L_ctrl_full": spearman(mean["L_ctrl_full"], J),
        "rho_L_all_k": spearman(mean["L_all_k"], J),
        "rho_L_greedy": spearman(L_greedy, J),
        "rho_panic_vs_L_code": spearman(mean["L_code"], -mean["panic"]),
        "regret_argmin_L_code": float(J[np.argmin(mean["L_code"])] - J.min()),
        "regret_argmin_L_ctrl_full": float(J[np.argmin(mean["L_ctrl_full"])] - J.min()),
        "regret_argmin_L_all_k": float(J[np.argmin(mean["L_all_k"])] - J.min()),
        "regret_argmin_greedy": float(J[int(np.argmin(L_greedy))] - J.min()),
        "regret_random": float(J.mean() - J.min()),
    }
    return row


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [11, 12]
    cfg = dict(pop=300, stop=60, decision_t=5, k=5, tapes=3, panic=0.5, hazards=3)
    if len(sys.argv) > 2:
        cfg.update(json.loads(sys.argv[2]))
    for seed in seeds:
        t0 = time.time()
        row = audit_state(seed, **cfg)
        row["seconds"] = round(time.time() - t0, 1)
        row["cfg"] = cfg
        print(json.dumps(row), flush=True)
