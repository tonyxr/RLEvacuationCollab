#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference learner that verifies NMCC-PI on the real stochastic dynamics.

The GNN actor cannot be trained in the environment where this change was made
(torch is not installable there), so this script trains the simplest faithful
stand-in: a linear softmax policy over the actor's own per-cell inputs --
``RegionalObservation.candidate_features()`` for the cell's resolved site, the
host cell's row of ``policy_features()``, and three time interactions.

It starts *exactly* where the current actor starts at zero residual: logits
equal to ``relative_active`` (active population over its maximum), whose argmax
is the active-population heuristic.  Only the NMCC-PI update then changes it.
If the update works as designed, the frozen, deterministically evaluated policy
must move away from the heuristic and toward the rollout's choices on held-out
scenarios it never trained on.

Protocol
--------
* Training scenarios and held-out evaluation scenarios use disjoint seeds.
* Each iteration collects ``--episodes`` on-policy episodes (actions sampled
  from the current policy), branches cells at each decision with
  ``NMCCPolicyImprovement.BranchValuer`` under independent future tapes, forms
  the KL-constrained target, and fits it by cross-entropy with an M-step KL cap.
* After every iteration the frozen policy is evaluated by argmax on the
  held-out scenarios, paired against the heuristic and greedy route saving on
  the same scenarios.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np

import CounterfactualBranch as CB
import headroom_lib as H
import NMCCPolicyImprovement as PI
from DecisionInterface import RegionalObservationBuilder, RegionalShelterExecutor

HORIZON, INTERVAL, BUDGET = 60, 10, 5
MAP = dict(spacing_m=150, candidate_count=20)


# ----------------------------------------------------------------------------- features
def features(observation) -> np.ndarray:
    """Per-action features, built only from what the GNN actor also receives."""
    cand = observation.candidate_features().astype(np.float64)  # (A, 8)
    cell_matrix, global_vector = observation.policy_features()
    cells = np.asarray(observation.candidate_cell_indices, dtype=np.int64)
    host = np.asarray(cell_matrix, dtype=np.float64)[cells]  # (A, d_cell)
    active = np.asarray(observation.active_by_cell, dtype=np.float64)[cells]
    relative_active = active / active.max() if active.max() > 0 else np.zeros_like(active)
    time_remaining = float(np.asarray(global_vector, dtype=np.float64)[0])
    interactions = np.stack(
        (cand[:, 7] * time_remaining, cand[:, 6] * time_remaining, relative_active * time_remaining),
        axis=1,
    )
    return np.concatenate((relative_active[:, None], cand, host, interactions), axis=1)


class LinearPolicy:
    """logits = X @ w.  w[0] multiplies relative_active: w = e_0 is the current actor."""

    def __init__(self, dimension: int):
        self.w = np.zeros(dimension, dtype=np.float64)
        self.w[0] = 1.0  # HEURISTIC_PRIOR_SCALE * relative_active, residual 0
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        self.steps = 0

    def probs(self, X, mask, w=None):
        return PI.softmax(X @ (self.w if w is None else w), mask)

    def fit(self, records, *, lr, max_epochs, kl_cap, l2):
        """M-step: minimize sum CE(q, pi_w) + l2 ||w - w_old||^2, KL(pi_old||pi_w) <= kl_cap."""
        w_old = self.w.copy()
        accepted = w_old.copy()
        history = []
        for epoch in range(int(max_epochs)):
            grad = np.zeros_like(self.w)
            loss = 0.0
            for rec in records:
                X, mask, q = rec.features, rec.action_mask, rec.target.target
                p = self.probs(X, mask)
                grad += X[mask].T @ (p[mask] - q[mask])
                loss -= float(np.sum(q[mask] * np.log(np.clip(p[mask], 1e-300, None))))
            grad = grad / len(records) + 2 * l2 * (self.w - w_old)
            self.steps += 1
            self.m = 0.9 * self.m + 0.1 * grad
            self.v = 0.999 * self.v + 0.001 * grad * grad
            mhat = self.m / (1 - 0.9 ** self.steps)
            vhat = self.v / (1 - 0.999 ** self.steps)
            self.w = self.w - lr * mhat / (np.sqrt(vhat) + 1e-8)
            kl = float(np.mean([PI._kl(r.target.pi_old, self.probs(r.features, r.action_mask)) for r in records]))
            history.append((epoch, loss / len(records), kl))
            if kl > kl_cap:
                self.w = accepted  # transactional: keep the last step inside the trust region
                break
            accepted = self.w.copy()
        return history


# ----------------------------------------------------------------------------- episodes
def run_policy_episode(seed, choose):
    """One full episode; choose(observation) -> action.  Returns return and outcomes."""
    core = H.build_core(seed, **MAP)
    builder = RegionalObservationBuilder(core, initial_population=800, horizon=HORIZON, maximum_deployments=BUDGET)
    executor = RegionalShelterExecutor(core)
    reward = PI.RewardProcessor()
    clock = PI.EpisodeClock(horizon=HORIZON, interval=INTERVAL, budget=BUDGET, population=800)
    cells = []
    while clock.t < HORIZON:
        PI.step_and_score(core, clock, reward)
        if clock.is_decision(clock.t):
            obs = builder.build(decision_index=clock.deployed, simulation_time=clock.t,
                                remaining_deployments=BUDGET - clock.deployed)
            if obs.has_feasible_action:
                action = int(choose(obs))
                receipt = executor.execute(obs, _Decision(action))
                clock.deployed += 1
                cells.append((clock.t, int(receipt.executed_cell)))
    out = CB.outcome_snapshot(core)
    return {"return": clock.accumulated, "unfinished": int(out.active_population),
            "safe": int(out.safe_completed), "casualty": int(out.casualties), "cells": cells}


class _Decision:
    def __init__(self, action_index):
        self.action_index = int(action_index)


def collect_episode(args):
    """Worker: one on-policy training episode with NMCC-PI branch targets."""
    seed, w, cfg = args
    rng = np.random.default_rng(seed * 31 + 7)
    core = H.build_core(seed, **MAP)
    builder = RegionalObservationBuilder(core, initial_population=800, horizon=HORIZON, maximum_deployments=BUDGET)
    executor = RegionalShelterExecutor(core)
    reward = PI.RewardProcessor()
    valuer = PI.BranchValuer(core, builder=builder, executor=executor, reward_model=reward,
                             base_policy=cfg["base_policy"], tapes=cfg["tapes"])
    clock = PI.EpisodeClock(horizon=HORIZON, interval=INTERVAL, budget=BUDGET, population=800)
    records = []
    while clock.t < HORIZON:
        PI.step_and_score(core, clock, reward)
        if not clock.is_decision(clock.t):
            continue
        obs = builder.build(decision_index=clock.deployed, simulation_time=clock.t,
                            remaining_deployments=BUDGET - clock.deployed)
        if not obs.has_feasible_action:
            continue
        mask = np.asarray(obs.action_mask, dtype=bool)
        X = features(obs)
        pi = PI.softmax(X @ w, mask)
        action = int(rng.choice(pi.size, p=pi))
        feasible = np.flatnonzero(mask)
        branch = PI.select_branch_actions(feasible, pi, decision_index=clock.deployed,
                                          exhaustive_decisions=cfg["exhaustive_decisions"],
                                          max_branches=cfg["max_branches"], must_include=(action,), rng=rng)
        values = valuer.value(obs, clock, branch, episode_seed=seed)
        target = PI.improvement_target(pi, mask, values.actions, values.mean_values(),
                                       epsilon=cfg["epsilon"], eta_min=cfg["eta_min"])
        records.append(PI.DecisionRecord(features=X, action_mask=mask, target=target, values=values,
                                         executed_action=action, simulation_time=clock.t,
                                         decision_index=clock.deployed))
        executor.execute(obs, _Decision(action))
        clock.deployed += 1
    return records, clock.accumulated


def evaluate(w, seeds):
    def choose(obs):
        X = features(obs)
        mask = np.asarray(obs.action_mask, dtype=bool)
        return int(np.argmax(np.where(mask, X @ w, -np.inf)))
    return [run_policy_episode(s, choose) for s in seeds]


def _eval_worker(args):
    w, seeds = args
    return evaluate(w, seeds)


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else float("nan")
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()), "n": int(d.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=4, help="training episodes per iteration")
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=0.5, help="E-step KL per state")
    ap.add_argument("--kl-cap", type=float, default=0.5, help="M-step mean KL cap")
    ap.add_argument("--eta-min", type=float, default=0.005)
    ap.add_argument("--tapes", type=int, default=1)
    ap.add_argument("--exhaustive-decisions", type=int, default=2)
    ap.add_argument("--max-branches", type=int, default=6)
    ap.add_argument("--base-policy", default="route_saving")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--output", default="nmcc_pi_reference_report.json")
    args = ap.parse_args()
    cfg = dict(epsilon=args.epsilon, eta_min=args.eta_min, tapes=args.tapes,
               exhaustive_decisions=args.exhaustive_decisions, max_branches=args.max_branches,
               base_policy=args.base_policy)

    eval_seeds = list(range(201, 201 + args.eval_seeds))
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        halves = [eval_seeds[i::args.workers] for i in range(args.workers)]
        ref = {}
        for name, pol in (("heuristic", H.heuristic), ("route_saving", H.route_saving)):
            ref[name] = {s: H.run(H.build_core(s, **MAP), pol)["return"] for s in eval_seeds}
        probe_obs = None
        core = H.build_core(301, **MAP)
        b = RegionalObservationBuilder(core, initial_population=800, horizon=HORIZON, maximum_deployments=BUDGET)
        CB.advance_one_timestep(core)
        dim = features(b.build(decision_index=0, simulation_time=1, remaining_deployments=BUDGET)).shape[1]
        policy = LinearPolicy(dim)

        def run_eval():
            parts = pool.map(_eval_worker, [(policy.w.copy(), h) for h in halves])
            by_seed = {}
            for h, res in zip(halves, parts):
                for s, r in zip(h, res):
                    by_seed[s] = r
            returns = [by_seed[s]["return"] for s in eval_seeds]
            return returns, by_seed

        curve = []
        returns, by_seed = run_eval()
        curve.append({"iteration": 0, "episodes_trained": 0, "mean_return": float(np.mean(returns)),
                      "vs_heuristic": paired(returns, [ref["heuristic"][s] for s in eval_seeds]),
                      "vs_route_saving": paired(returns, [ref["route_saving"][s] for s in eval_seeds]),
                      "w": policy.w.tolist()})
        print(f"[{time.time()-t0:5.0f}s] iter 0 (current actor, residual 0): eval={np.mean(returns):+.4f} "
              f"heuristic={np.mean(list(ref['heuristic'].values())):+.4f} "
              f"route_saving={np.mean(list(ref['route_saving'].values())):+.4f}", flush=True)

        train_seed = 301
        diagnostics = []
        for iteration in range(1, args.iterations + 1):
            jobs = [(train_seed + k, policy.w.copy(), cfg) for k in range(args.episodes)]
            train_seed += args.episodes
            results = pool.map(collect_episode, jobs)
            records = [r for recs, _ in results for r in recs]
            behavior_returns = [ret for _, ret in results]
            entropy_before = float(np.mean([
                -np.sum(r.target.pi_old[r.action_mask] * np.log(np.clip(r.target.pi_old[r.action_mask], 1e-300, None)))
                / math.log(max(2, r.action_mask.sum())) for r in records]))
            top1_before = float(np.mean([int(np.argmax(np.where(r.action_mask, r.target.pi_old, -1))) == r.target.best_action
                                         for r in records if r.target.branched.sum() > 1]))
            history = policy.fit(records, lr=args.lr, max_epochs=args.epochs, kl_cap=args.kl_cap, l2=args.l2)
            post = [policy.probs(r.features, r.action_mask) for r in records]
            top1_after = float(np.mean([int(np.argmax(np.where(r.action_mask, p, -1))) == r.target.best_action
                                        for r, p in zip(records, post) if r.target.branched.sum() > 1]))
            entropy_after = float(np.mean([
                -np.sum(p[r.action_mask] * np.log(np.clip(p[r.action_mask], 1e-300, None))) / math.log(max(2, r.action_mask.sum()))
                for r, p in zip(records, post)]))
            realized_kl = float(np.mean([PI._kl(r.target.pi_old, p) for r, p in zip(records, post)]))
            within_sd = float(np.mean([np.std(r.values.mean_values()) for r in records if r.values.actions.size > 1]))
            returns, by_seed = run_eval()
            row = {"iteration": iteration, "episodes_trained": iteration * args.episodes,
                   "decisions": len(records), "behavior_return": float(np.mean(behavior_returns)),
                   "mean_return": float(np.mean(returns)),
                   "vs_heuristic": paired(returns, [ref["heuristic"][s] for s in eval_seeds]),
                   "vs_route_saving": paired(returns, [ref["route_saving"][s] for s in eval_seeds]),
                   "entropy_before": entropy_before, "entropy_after": entropy_after,
                   "top1_branch_best_before": top1_before, "top1_branch_best_after": top1_after,
                   "realized_kl": realized_kl, "m_step_epochs": len(history),
                   "mean_eta": float(np.mean([r.target.eta for r in records if np.isfinite(r.target.eta)])),
                   "within_state_value_sd": within_sd, "w": policy.w.tolist()}
            curve.append(row)
            print(f"[{time.time()-t0:5.0f}s] iter {iteration}: eval={row['mean_return']:+.4f} "
                  f"vsH={row['vs_heuristic']['mean']:+.4f}[{row['vs_heuristic']['ci95'][0]:+.3f},{row['vs_heuristic']['ci95'][1]:+.3f}] "
                  f"vsRS={row['vs_route_saving']['mean']:+.4f} | entropy {entropy_before:.2f}->{entropy_after:.2f} "
                  f"top1 {top1_before:.2f}->{top1_after:.2f} KL={realized_kl:.3f} epochs={len(history)}", flush=True)

    report = {"config": vars(args), "eval_seeds": eval_seeds,
              "reference_returns": {k: [v[s] for s in eval_seeds] for k, v in ref.items()},
              "curve": curve,
              "feature_count": dim}
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
