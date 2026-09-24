#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much can any cell-priority policy improve on the heuristic?

Before investing further in the learner, this measures the size of the prize
and the shape of the signal a learner would have to find.

Policies, all on matched scenario seeds (identical hazards and keyed shocks):

* ``heuristic``      maximum active population (the registered comparator)
* ``accessibility``  accessibility-deficit benchmark
* ``uniform``        uniform over feasible cells
* ``prior_T1``       softmax(relative_active / 1.0): the v22/v23 *training*
                     behavior policy when the learned residual is zero
* ``rollout``        one-step rollout over the heuristic (Bertsekas): at each
                     decision, every feasible cell is installed in a branch and
                     the episode is continued with the heuristic to the horizon;
                     the best cell is chosen.  Branches share the episode's own
                     noise, so this is a *perfect-information* one-step lookahead
                     and an upper bound on what one-step improvement can buy.

At every rollout decision the script also records, for every feasible cell,
the ``L``-step reward after installing it and the ``L``-step reward of a
matched ``WAIT`` branch.  That is exactly the NMCC Variant-A target, so the
report can say (i) how much of that target's variance is between-state
("deploying now is worth a lot") rather than between-cell ("this cell beats
that one"), and (ii) whether the short-horizon effect even ranks cells the way
the full-horizon value does.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

import CounterfactualBranch as CB
import headroom_lib as H


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def reseed_exogenous(core, seed):
    """Replace every exogenous stream with an independent tape.

    Hazard spread and the keyed person-level shocks are drawn from the new tape
    from this point forward; the current state (fire front, who has already
    panicked, who is where) is unchanged.  This is the information set of a
    real operator: the present is observed, the future is not.
    """
    mask = (1 << 64) - 1
    key = int(np.random.SeedSequence([int(seed)]).generate_state(1, dtype=np.uint64)[0])
    core.hazardDS.rng = np.random.default_rng(key & mask)
    core.pedDS.set_hazard_random_seed((key ^ 0x5DEECE66D) & mask)
    core.pedDS.panic_random_seed = (key ^ 0x9E3779B97F4A7C15) & mask


class RolloutPolicy:
    """One-step rollout over the heuristic.

    ``mc_tapes = 0``  perfect information: branches share the episode's own
                      future noise (an upper bound, not implementable).
    ``mc_tapes = K``  implementable: each cell is scored by the mean over K
                      independent future tapes.  The tapes are common across
                      cells within a decision, so the comparison between cells
                      is still CRN-paired.
    """

    def __init__(self, *, short_horizon=10, mc_tapes=0, seed=0):
        self.short = int(short_horizon)
        self.mc_tapes = int(mc_tapes)
        self.seed = int(seed)
        self.records = []

    def _continue_with_heuristic(self, ep):
        start = ep.ret
        short_reward = None
        steps = 0
        while ep.t < ep.horizon:
            ep.step_dynamics()
            steps += 1
            if steps == self.short:
                short_reward = ep.ret - start
            if ep.is_decision(ep.t):
                obs = ep.observe()
                if obs.has_feasible_action:
                    ep.install(obs, H.heuristic(ep, obs, None))
        if short_reward is None:
            short_reward = ep.ret - start
        return ep.ret - start, short_reward

    def __call__(self, ep, obs, rng):
        feasible = np.flatnonzero(obs.action_mask)
        heuristic_action = H.heuristic(ep, obs, None)
        base = CB.capture(ep.core, label="rollout")
        book = ep.bookkeeping()

        full, short = {}, {}
        tapes = [None] if self.mc_tapes <= 0 else [
            self.seed * 7919 + ep.t * 104729 + k for k in range(self.mc_tapes)]
        for action in feasible:
            values_full, values_short = [], []
            for tape in tapes:
                CB.restore(ep.core, base); ep.set_bookkeeping(book)
                if tape is not None:
                    reseed_exogenous(ep.core, tape)
                ep.install(obs, int(action))
                f, sh = self._continue_with_heuristic(ep)
                values_full.append(f); values_short.append(sh)
            full[int(action)] = float(np.mean(values_full))
            short[int(action)] = float(np.mean(values_short))

        # matched WAIT branch over the short horizon: the NMCC Variant-A baseline
        CB.restore(ep.core, base); ep.set_bookkeeping(book)
        if tapes[0] is not None:
            reseed_exogenous(ep.core, tapes[0])
        start = ep.ret
        for _ in range(self.short):
            if ep.t >= ep.horizon:
                break
            ep.step_dynamics()
        wait_short = ep.ret - start

        CB.restore(ep.core, base); ep.set_bookkeeping(book)
        best = max(full, key=full.get)
        cells = [int(obs.candidate_cell_indices[a]) for a in feasible]
        self.records.append({
            "t": int(ep.t),
            "feasible": int(feasible.size),
            "cells": cells,
            "active_by_cell": [float(obs.active_by_cell[c]) for c in cells],
            "full_value": [full[int(a)] for a in feasible],
            "short_reward": [short[int(a)] for a in feasible],
            "wait_short_reward": float(wait_short),
            "heuristic_action": int(heuristic_action),
            "rollout_action": int(best),
            "heuristic_is_best": bool(int(heuristic_action) == int(best)),
            "heuristic_regret": float(full[int(best)] - full[int(heuristic_action)]),
            "spearman_short_vs_full": spearman([short[int(a)] for a in feasible],
                                               [full[int(a)] for a in feasible]),
            "spearman_active_vs_full": spearman([float(obs.active_by_cell[c]) for c in cells],
                                                [full[int(a)] for a in feasible]),
        })
        return int(best)


def paired_summary(values_a, values_b):
    d = np.asarray(values_a) - np.asarray(values_b)
    n = d.size
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": float(d.mean()), "se": float(se), "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
            "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()), "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--rollout-seeds", type=int, default=6)
    ap.add_argument("--random-draws", type=int, default=3)
    ap.add_argument("--short-horizon", type=int, default=10)
    ap.add_argument("--output", default="headroom_report.json")
    ap.add_argument("--mc-tapes", type=int, default=0,
                    help="0 = perfect-information rollout; K = implementable MC rollout")
    ap.add_argument("--first-seed", type=int, default=101)
    ap.add_argument("--rollout-only", action="store_true")
    args = ap.parse_args()

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    results = {k: [] for k in ("heuristic", "accessibility", "uniform", "prior_T1", "rollout")}
    rollout_records = []
    t0 = time.time()
    for index, seed in enumerate(seeds):
        results["heuristic"].append(H.run(H.build_core(seed, spacing_m=150, candidate_count=20), H.heuristic))
        if args.rollout_only:
            if index < args.rollout_seeds:
                policy = RolloutPolicy(short_horizon=args.short_horizon, mc_tapes=args.mc_tapes, seed=seed)
                results["rollout"].append(H.run(H.build_core(seed, spacing_m=150, candidate_count=20), policy))
                for record in policy.records:
                    record["seed"] = int(seed)
                rollout_records.extend(policy.records)
            print(f"seed {seed} done ({time.time() - t0:.0f}s): heuristic={results['heuristic'][-1]['return']:+.4f} "
                  f"rollout={results['rollout'][-1]['return']:+.4f}", flush=True)
            continue
        results["accessibility"].append(H.run(H.build_core(seed, spacing_m=150, candidate_count=20), H.accessibility))
        for name, policy in (("uniform", H.uniform), ("prior_T1", H.prior_sampling(1.0))):
            draws = [H.run(H.build_core(seed, spacing_m=150, candidate_count=20), policy,
                           rng=np.random.default_rng(seed * 1000 + k)) for k in range(args.random_draws)]
            results[name].append({"return": float(np.mean([d["return"] for d in draws])),
                                  "safe": float(np.mean([d["safe"] for d in draws])),
                                  "unfinished": float(np.mean([d["unfinished"] for d in draws])),
                                  "casualty": float(np.mean([d["casualty"] for d in draws])),
                                  "deployed": float(np.mean([d["deployed"] for d in draws]))})
        if index < args.rollout_seeds:
            policy = RolloutPolicy(short_horizon=args.short_horizon, mc_tapes=args.mc_tapes, seed=seed)
            results["rollout"].append(H.run(H.build_core(seed, spacing_m=150, candidate_count=20), policy))
            for record in policy.records:
                record["seed"] = int(seed)
            rollout_records.extend(policy.records)
        print(f"seed {seed} done ({time.time() - t0:.0f}s): "
              + " ".join(f"{k}={results[k][-1]['return']:+.4f}" for k in results if len(results[k]) == index + 1),
              flush=True)

    ret = {k: [r["return"] for r in v] for k, v in results.items()}
    report_mode = "perfect_information" if args.mc_tapes <= 0 else f"monte_carlo_{args.mc_tapes}_tapes"
    n_roll = len(ret["rollout"])
    report = {
        "config": {"grid": 20, "spacing_m": 150, "cells": "8x8", "population": 800, "capacity_token": 160,
                   "hazards": 3, "spread_rate": [4, 2], "casualty_rate": [40, 9], "panic_rate": 0.5,
                   "candidates": 20, "budget": 5, "interval": 10, "horizon": 60,
                   "short_horizon": args.short_horizon, "seeds": seeds,
                   "rollout_mode": report_mode},
        "mean_return": {k: float(np.mean(v)) for k, v in ret.items() if v},
        "return_sd_across_scenarios": {k: float(np.std(v, ddof=1)) for k, v in ret.items() if len(v) > 1},
        "mean_unfinished": {k: float(np.mean([r["unfinished"] for r in v])) for k, v in results.items() if v},
        "vs_heuristic": {k: paired_summary(ret[k], ret["heuristic"][:len(ret[k])]) for k in ret if k != "heuristic" and ret[k]},
    }

    recs = rollout_records
    if recs:
        # Variance decomposition of the NMCC Variant-A target A = R_L(s,c) - R_L(s,WAIT).
        per_state = [np.asarray(r["short_reward"]) - r["wait_short_reward"] for r in recs if r["feasible"] >= 2]
        all_a = np.concatenate(per_state)
        between = float(np.var([a.mean() for a in per_state]))
        within = float(np.mean([a.var() for a in per_state]))
        full_within = float(np.mean([np.var(r["full_value"]) for r in recs if r["feasible"] >= 2]))
        report["decision_analysis"] = {
            "decisions": len(recs),
            "heuristic_already_best_fraction": float(np.mean([r["heuristic_is_best"] for r in recs])),
            "mean_heuristic_regret": float(np.mean([r["heuristic_regret"] for r in recs])),
            "regret_by_decision_time": {str(t): float(np.mean([r["heuristic_regret"] for r in recs if r["t"] == t]))
                                        for t in sorted({r["t"] for r in recs})},
            "nmcc_target_total_variance": float(all_a.var()),
            "nmcc_target_between_state_variance": between,
            "nmcc_target_within_state_variance": within,
            "nmcc_target_between_state_share": between / (between + within) if (between + within) > 0 else float("nan"),
            "full_value_within_state_sd": math.sqrt(full_within),
            "short_effect_within_state_sd": math.sqrt(within),
            "spearman_short_vs_full_mean": float(np.nanmean([r["spearman_short_vs_full"] for r in recs])),
            "spearman_active_vs_full_mean": float(np.nanmean([r["spearman_active_vs_full"] for r in recs])),
        }
    report["rollout_decisions"] = recs
    with open(args.output, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rollout_decisions"}, indent=2))


if __name__ == "__main__":
    main()
