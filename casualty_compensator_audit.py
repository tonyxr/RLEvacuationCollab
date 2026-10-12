#!/usr/bin/env python3
"""Realized casualties vs their compensator (integrated hazard) as a siting signal.

At one decision state per scenario, every feasible cell is branched to the
horizon under T common-random-number tapes.  For each branch we record
  N  = realized casualties (Bernoulli/binomial draws), and
  L  = sum over steps and living exposed people of the casualty probability
       actually used by the simulator for that draw (the compensator).
E[N] = E[L]; L is the Rao-Blackwellized version.  We compare, per state,
the site-effect SNR: var_across_cells(mean over tapes) / mean within-cell
tape variance of the paired contrast, and agreement of the two rankings.
"""
import json, sys, time
import numpy as np
import CounterfactualBranch as CB
import nmcc_testbed as T
import PedestrianDatabase as PD
from RewardProcessor import RewardProcessor

LEDGER = {"L": 0.0}
_orig = PD.PedDS._hazard_casualty_count


def _tracked(self, pedestrian_id, represented_population, probability):
    LEDGER["L"] += max(1, int(represented_population)) * max(0.0, min(1.0, float(probability)))
    return _orig(self, pedestrian_id, represented_population, probability)


PD.PedDS._hazard_casualty_count = _tracked
POP, STOP, TOKEN = 300, 60, 33


def branch(core, br, base, cell, timeline, seed):
    CB.restore(core, base)
    br._reseed_exogenous_streams(seed)
    pair = core.cellTracker.unflatten_cell(cell)
    sid = core.shelterDS.newShelter({"cell": pair}, core.cellTracker)
    sh = core.shelterDS.shelterList[sid]
    if hasattr(core.cellTracker, "addShelter"):
        core.cellTracker.addShelter(pair, sh)
    core.pedDS.reroute_active_to_nearest_shelter(new_shelter=sh)
    n0 = CB.outcome_snapshot(core).casualties
    LEDGER["L"] = 0.0
    for frame in timeline:
        CB.advance_one_timestep(core, hazard_frame=frame)
    return CB.outcome_snapshot(core).casualties - n0, LEDGER["L"]


def state(seed, tapes, decision_t):
    core = T.build(population=POP, stop_time=STOP, panic_rate=0.5, hazard_count=3,
                   scenario_seed=seed, casualty_rate=(40, 9), spread_rate=(30, 25),
                   shelter_capacity_token=TOKEN)
    for _ in range(decision_t):
        CB.advance_one_timestep(core)
    cells = core.feasible_cells()
    base = CB.capture(core, label="cas")
    br = CB.CounterfactualBrancher(core, horizon=STOP - int(core.pedDS.currTime),
                                   reward_model=RewardProcessor(), initial_population=POP,
                                   episode_horizon=STOP)
    N = np.zeros((tapes, len(cells))); L = np.zeros_like(N)
    for t in range(tapes):
        ts = seed * 1000 + t
        tl = br._hazard_timeline(base, ts)
        for j, c in enumerate(cells):
            N[t, j], L[t, j] = branch(core, br, base, c, tl, ts)
    CB.restore(core, base)

    def snr(X):
        C = X - X.mean(axis=1, keepdims=True)          # paired within-tape contrast vs site mean
        between = np.var(C.mean(axis=0), ddof=1)       # site effect variance (incl. noise/T)
        within = np.mean(np.var(C, axis=0, ddof=1))    # tape noise of one contrast
        signal = max(between - within / X.shape[0], 0.0)
        return signal / within if within > 0 else float("nan")

    rank = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rank(N.mean(0)), rank(L.mean(0)))[0, 1]) if np.std(N.mean(0)) > 0 else float("nan")
    # does the realized ranking agree with itself on disjoint tapes? (reliability)
    h = tapes // 2
    rel = lambda X: float(np.corrcoef(X[:h].mean(0), X[h:].mean(0))[0, 1]) if np.std(X[:h].mean(0)) > 0 and np.std(X[h:].mean(0)) > 0 else float("nan")
    return {"seed": seed, "cells": len(cells), "mean_N": float(N.mean()), "mean_L": float(L.mean()),
            "snr_N": snr(N), "snr_L": snr(L), "rho_N_L": rho,
            "split_half_rel_N": rel(N), "split_half_rel_L": rel(L),
            "tape_sd_contrast_N": float(np.sqrt(np.mean(np.var(N - N.mean(1, keepdims=True), axis=0, ddof=1)))),
            "tape_sd_contrast_L": float(np.sqrt(np.mean(np.var(L - L.mean(1, keepdims=True), axis=0, ddof=1)))),
            "site_range_L": float(np.ptp(L.mean(0)))}


if __name__ == "__main__":
    for s in [int(x) for x in sys.argv[1].split(",")]:
        t0 = time.time(); r = state(s, int(sys.argv[2]), 5); r["sec"] = round(time.time() - t0, 1)
        print(json.dumps(r), flush=True)
