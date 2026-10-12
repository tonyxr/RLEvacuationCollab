#!/usr/bin/env python3
"""Does the reward rise when siting is random?  D-potential vs random-policy potential.

For each scenario an episode is run under a policy (uniform random or
objective-greedy).  At every decision state s_k two potentials are recorded:

  D(s_k)    the current heuristic forecast debt (estimate_remaining_time_to_safety,
            complete ledger), i.e. the v44 potential;
  V_mu(s_k) the cost-to-go of the uniform-random siting policy from s_k,
            estimated by M Monte Carlo rollouts (random decisions from k on).

Per-decision reward under potential Phi:  r_k = Phi(s_k) - C_k - Phi(s_{k+1}),
with Phi(terminal) = 0 and the terminal unresolved debt charged inside the last
C_k (complete ledger).  C_k is realized complete-ledger cost between decisions.

Prediction: under a random policy E[r_k] > 0 with D (installing any shelter
reduces forecast debt) but E[r_k] = 0 with V_mu (performance-difference lemma).
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

POP, STOP, TOKEN = 300, 60, 33
DECISIONS = [5, 10, 15, 20, 25, 30, 35, 40]


def build(seed):
    return T.build(population=POP, stop_time=STOP, panic_rate=0.5, hazard_count=3,
                   scenario_seed=seed, casualty_rate=(40, 9), spread_rate=(30, 25),
                   shelter_capacity_token=TOKEN)


def cost_rate(core):
    s = CB.outcome_snapshot(core)
    return float(s.active_population + s.casualties + s.hazard_exposure_mass)


def debt(core):
    return estimate_remaining_time_to_safety(
        core, episode_horizon=STOP, include_panicked=True, include_casualties=True
    ).hazard_weighted_person_minutes


def install(core, cell):
    pair = core.cellTracker.unflatten_cell(cell)
    sid = core.shelterDS.newShelter({"cell": pair}, core.cellTracker)
    if sid is None:
        raise RuntimeError("install failed")
    sh = core.shelterDS.shelterList[sid]
    if hasattr(core.cellTracker, "addShelter"):
        core.cellTracker.addShelter(pair, sh)
    core.pedDS.reroute_active_to_nearest_shelter(new_shelter=sh)


def greedy_cell(core, cells):
    br = CB.CounterfactualBrancher(core, horizon=0, reward_model=RewardProcessor(),
                                   initial_population=POP, episode_horizon=STOP)
    eff = br.candidate_effects(cells)
    vals = [e.reward_breakdown(RewardProcessor(), initial_population=POP,
                               episode_horizon_minutes=STOP).raw_total for e in eff]
    return int(cells[int(np.argmax(vals))])


def reseed(core, brancher, key):
    brancher._reseed_exogenous_streams(int(np.random.SeedSequence(key).generate_state(1)[0]))


def random_cost_to_go(core, brancher, base, k, rng, tape_key):
    """Cost from decision k to the end with uniform-random decisions k..K-1."""
    CB.restore(core, base)
    reseed(core, brancher, tape_key)
    total = 0.0
    t = int(core.pedDS.currTime)
    while t < STOP:
        if t in DECISIONS:
            cells = core.feasible_cells()
            if cells:
                install(core, cells[int(rng.integers(len(cells)))])
        CB.advance_one_timestep(core)
        total += cost_rate(core)
        t = int(core.pedDS.currTime)
    total += debt(core)  # terminal unresolved debt
    CB.restore(core, base)
    return total


def episode(seed, policy, M):
    core = build(seed)
    brancher = CB.CounterfactualBrancher(core, horizon=1, reward_model=RewardProcessor(),
                                         initial_population=POP, episode_horizon=STOP)
    act_rng = np.random.default_rng([seed, 99])
    pre_cost = 0.0
    while int(core.pedDS.currTime) < DECISIONS[0]:
        CB.advance_one_timestep(core)
        pre_cost += cost_rate(core)
    D, V, C = [], [], []
    interval = 0.0
    for k, td in enumerate(DECISIONS):
        assert int(core.pedDS.currTime) == td
        if k > 0:
            C.append(interval)
            interval = 0.0
        base = CB.capture(core, label=f"d{k}")
        D.append(debt(core))
        V.append(float(np.mean([
            random_cost_to_go(core, brancher, base, k, np.random.default_rng([seed, k, m, 7]),
                              [seed, k, m])
            for m in range(M)])))
        cells = core.feasible_cells()
        cell = (cells[int(act_rng.integers(len(cells)))] if policy == "random"
                else greedy_cell(core, cells))
        install(core, cell)
        reseed(core, brancher, [seed, k, 0])  # factual future shares tape 0 with rollout m=0
        nxt = DECISIONS[k + 1] if k + 1 < len(DECISIONS) else STOP
        while int(core.pedDS.currTime) < nxt:
            CB.advance_one_timestep(core)
            interval += cost_rate(core)
    # last interval runs to the horizon; charge terminal unresolved debt
    interval += debt(core)
    C.append(interval)
    D, V, C = map(np.asarray, (D, V, C))
    D_next = np.append(D[1:], 0.0)
    V_next = np.append(V[1:], 0.0)
    r_D = D - C - D_next
    r_V = V - C - V_next
    realized = pre_cost + C.sum()
    return {
        "seed": seed, "policy": policy, "M": M,
        "r_D": r_D.tolist(), "r_V": r_V.tolist(),
        "return_D": float(r_D.sum()), "return_V": float(r_V.sum()),
        "V0": float(V[0]), "D0": float(D[0]), "realized_cost": float(realized),
        "panic": int(core.pedDS.result.get("panic_onset", 0)),
        "casualty": int(CB.outcome_snapshot(core).casualties),
    }


if __name__ == "__main__":
    policy = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(",")]
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    for s in seeds:
        t0 = time.time()
        row = episode(s, policy, M)
        row["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(row), flush=True)
