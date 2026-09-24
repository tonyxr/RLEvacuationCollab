#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit: is the action-differential signal strong enough, and does the optimizer
turn it into a policy change of the right size?

The failure being checked is: "valid delayed rewards reach the actor, but the
action-differential signal is weak, and the optimizer converts it into an
extremely small policy change."  It is split into three measurable links.

1. Signal.  At real decision states of the calibrated testbed, every feasible
   cell is branched to the horizon under ``--tapes`` independent CRN tapes.
   Within one tape all cells share the same noise, so the cell-centered
   contrast A_k(s, c) = Q_k(s, c) - mean_c Q_k(s, c) is paired.  From the tapes
   we separate, per decision index:

   * the true within-state spread of cell values (the signal), and
   * the tape-to-tape noise of a one-tape contrast.

   The E-step target built from a subset of tapes is then scored against the
   remaining, independent tapes, which are an unbiased estimate of the true
   advantage.  Its *valid improvement* sum_c (q - pi_old)(c) A_true(c) is the
   expected return gain of following the target instead of the behavior
   policy, in the reward's own units, and can be compared with the largest
   gain available at that state, max_c A_true(c) - E_pi_old A_true.

2. Target size.  How far the E-step asks the policy to move
   (KL(q || pi_old)), and how often the temperature floor eta_min binds.

3. Conversion.  A policy network replica (``learner_flow_experiment``: encoder,
   residual head B * tanh(w . gelu(W2 h)), Adam, clipping at 0.5), fed the
   actor's own per-cell features, is fitted to the targets with the v23 budget
   and with the v24 M-step.  Realized KL(pi_old || pi_new) and the realized
   valid improvement are reported on the training states and on held-out
   episodes.

Behavior policy: the v24 zero-residual actor, softmax(route_time_saving / max)
at temperature 1, sampled on-policy.

    python nmcc_pi_signal_audit.py collect --episodes 10 --tapes 4 --workers 2
    python nmcc_pi_signal_audit.py analyze
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import time

import numpy as np

import headroom_lib as H
import learner_flow_experiment as L
import NMCCPolicyImprovement as PI
from DecisionInterface import RegionalObservationBuilder, RegionalShelterExecutor
from nmcc_pi_reference import BUDGET, HORIZON, INTERVAL, MAP, _Decision, features

SAVING_COLUMN = 1 + 7  # features(): [relative_active, candidate_features(8), ...]


def behavior_policy(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    saving = np.clip(X[:, SAVING_COLUMN], 0.0, None)
    top = saving[mask].max() if mask.any() else 0.0
    prior = saving / top if top > 0 else np.zeros_like(saving)
    return PI.softmax(prior, mask)


# ----------------------------------------------------------------------------- collection
def collect_episode(args):
    seed, tapes = args
    rng = np.random.default_rng(seed * 17 + 3)
    core = H.build_core(seed, **MAP)
    builder = RegionalObservationBuilder(core, initial_population=800, horizon=HORIZON, maximum_deployments=BUDGET)
    executor = RegionalShelterExecutor(core)
    reward = PI.RewardProcessor()
    valuer = PI.BranchValuer(core, builder=builder, executor=executor, reward_model=reward,
                             base_policy="route_saving", tapes=int(tapes))
    clock = PI.EpisodeClock(horizon=HORIZON, interval=INTERVAL, budget=BUDGET, population=800)
    states = []
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
        pi = behavior_policy(X, mask)
        feasible = np.flatnonzero(mask)
        values = valuer.value(obs, clock, feasible, episode_seed=seed)
        states.append(dict(seed=seed, decision_index=int(clock.deployed), t=int(clock.t),
                           X=X, mask=mask, pi=pi, actions=values.actions, values=values.values,
                           wait=values.wait_values))
        action = int(rng.choice(pi.size, p=pi))
        executor.execute(obs, _Decision(action))
        clock.deployed += 1
    return states


# ----------------------------------------------------------------------------- analysis helpers
def centered(values):
    """Per-tape cell-centered contrasts, (tapes, cells)."""
    return values - values.mean(axis=1, keepdims=True)


def full_vector(state, cell_values):
    out = np.zeros(state["mask"].size)
    out[state["actions"]] = cell_values
    return out


def target_from(state, tape_rows, epsilon, eta_min):
    q = PI.improvement_target(state["pi"], state["mask"], state["actions"],
                              state["values"][tape_rows].mean(axis=0),
                              epsilon=epsilon, eta_min=eta_min)
    return q


def valid_gain(prob, pi, a_true):
    return float(np.dot(prob - pi, a_true))


def signal_table(states):
    rows = {}
    for s in states:
        K = s["values"].shape[0]
        if s["actions"].size < 2 or K < 2:
            continue
        a = centered(s["values"])
        noise_var = float(a.var(axis=0, ddof=1).mean())  # one-tape contrast noise
        mean_a = a.mean(axis=0)
        signal_var = max(0.0, float(mean_a.var()) - noise_var / K)
        rows.setdefault(s["decision_index"], []).append(
            (s["actions"].size, math.sqrt(signal_var), math.sqrt(noise_var)))
    table = {}
    for d, items in sorted(rows.items()):
        arr = np.array(items)
        table[int(d)] = dict(states=len(items), cells=float(arr[:, 0].mean()),
                             signal_sd=float(arr[:, 1].mean()), noise_sd_one_tape=float(arr[:, 2].mean()),
                             snr_one_tape=float(np.mean(arr[:, 1] ** 2 / np.maximum(arr[:, 2] ** 2, 1e-12))))
    return table


def target_quality(states, *, signal_tapes, epsilon, eta_min):
    """Score targets built from ``signal_tapes`` tapes against the other tapes."""
    by_decision = {}
    for s in states:
        K = s["values"].shape[0]
        if s["actions"].size < 2 or K <= signal_tapes:
            continue
        rows = list(range(signal_tapes))
        held = list(range(signal_tapes, K))
        q = target_from(s, rows, epsilon, eta_min)
        a_true = full_vector(s, centered(s["values"][held]).mean(axis=0))
        pi = s["pi"] / s["pi"].sum()
        a_true = np.where(s["mask"], a_true - np.dot(pi, a_true), 0.0)
        best = float(a_true[s["actions"]].max())
        by_decision.setdefault(s["decision_index"], []).append(dict(
            gain=valid_gain(q.target, pi, a_true), available=best,
            kl=float(q.kl_to_old), eta_floor=float(q.eta <= eta_min * (1 + 1e-9)),
            top1=float(int(q.best_action) == int(s["actions"][int(np.argmax(a_true[s["actions"]]))])),
        ))
    out = {}
    for d, items in sorted(by_decision.items()):
        g = np.array([i["gain"] for i in items])
        av = np.array([i["available"] for i in items])
        out[int(d)] = dict(states=len(items), valid_gain=float(g.mean()), available_gain=float(av.mean()),
                           captured=float(g.sum() / max(av.sum(), 1e-12)),
                           gain_positive=float((g > 0).mean()),
                           target_kl=float(np.mean([i["kl"] for i in items])),
                           eta_at_floor=float(np.mean([i["eta_floor"] for i in items])),
                           top1_matches_heldout_best=float(np.mean([i["top1"] for i in items])))
    return out


# ----------------------------------------------------------------------------- conversion (replica M-step)
def standardize(train, test):
    X = np.concatenate([s["X"][s["mask"]] for s in train])
    # A feature that is (nearly) constant on the training states must not be
    # blown up on held-out states: floor the scale and clip standardized values.
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-2)
    return mu, sd


def pad_batch(states, mu, sd, prior_from_behavior=True):
    C = max(s["mask"].size for s in states)
    d = states[0]["X"].shape[1]
    x = np.zeros((len(states), C, d))
    mask = np.zeros((len(states), C), dtype=bool)
    prior = np.zeros((len(states), C))
    for i, s in enumerate(states):
        n = s["mask"].size
        x[i, :n] = np.clip((s["X"] - mu) / sd, -5.0, 5.0)
        mask[i, :n] = s["mask"]
        # logits whose softmax is the behavior policy (prior with zero residual)
        prior[i, :n] = np.where(s["mask"], np.log(np.clip(s["pi"], 1e-300, None)), 0.0)
    return dict(x=x, mask=mask, prior=prior)


def m_step(states, targets, *, lr, epochs, kl_cap, trust, seed, steps_per_epoch=2, width=64,
           fit_tolerance=0.2, test_states=None):
    """Fit the replica actor to the targets.

    ``trust``:
      * ``"sampled_stop"``  -- v23: stop at the first epoch whose KL(pi_old||pi) > cap.
      * ``"reverse_halving"`` -- v24 before this audit: KL(pi_old||pi) cap, roll the
        epoch back and halve lr, at most 3 consecutive times.
      * ``"forward_linesearch"`` -- v24 now: KL(pi||pi_old) cap (the E-step's
        direction); on overshoot, bisect the epoch's parameter step to the
        boundary and stop.
    """
    rng = np.random.default_rng(seed)
    mu, sd = standardize(states, test_states)
    batch = pad_batch(states, mu, sd)
    C = batch["mask"].shape[1]
    q = np.zeros((len(states), C))
    for i, t in enumerate(targets):
        q[i, :t.size] = t
    p = L.make_params(rng, batch["x"].shape[-1], width, "zero")
    opt = L.Adam(p, lr)
    keys = L.REP + L.ACTOR_HEAD
    behavior, _ = L.policy_logp(p, batch)
    logq = np.log(np.maximum(q, 1e-300))
    requested = float(L.kl(logq, behavior, batch["mask"]).mean())

    def trust_kl(now):
        if trust == "forward_linesearch":
            return float(L.kl(now, behavior, batch["mask"]).mean())
        return float(L.kl(behavior, now, batch["mask"]).mean())

    rollbacks, consecutive, epochs_run, reached = 0, 0, 0, 0
    S = len(states)
    for _ in range(epochs):
        snap = {k: v.copy() for k, v in p.items()}
        ostate = opt.state()
        order = rng.permutation(S)
        for part in np.array_split(order, steps_per_epoch):
            sub = {k: v[part] for k, v in batch.items()}
            g = L.actor_grads(p, sub, q[part])
            L.clip(g, keys)
            opt.step(g, keys)
        epochs_run += 1
        now, _ = L.policy_logp(p, batch)
        if trust_kl(now) > kl_cap:
            if trust == "forward_linesearch":
                current = {k: v.copy() for k, v in p.items()}
                lo, hi = 0.0, 1.0
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    for k in p:
                        p[k] = snap[k] + mid * (current[k] - snap[k])
                    if trust_kl(L.policy_logp(p, batch)[0]) <= kl_cap:
                        lo = mid
                    else:
                        hi = mid
                for k in p:
                    p[k] = snap[k] + lo * (current[k] - snap[k])
                reached = 1
                break
            p.update(snap)
            opt.load(ostate)
            opt.lr *= 0.5
            rollbacks += 1
            consecutive += 1
            if trust == "sampled_stop" or consecutive >= 3:
                break
            continue
        consecutive = 0
        if float(L.kl(logq, now, batch["mask"]).mean()) <= fit_tolerance * requested:
            break
    now, _ = L.policy_logp(p, batch)
    result = dict(requested_kl=requested,
                  realized_kl=float(L.kl(behavior, now, batch["mask"]).mean()),
                  realized_forward_kl=float(L.kl(now, behavior, batch["mask"]).mean()),
                  residual_fit_kl=float(L.kl(logq, now, batch["mask"]).mean()),
                  fit_fraction=1.0 - float(L.kl(logq, now, batch["mask"]).mean()) / max(requested, 1e-12),
                  epochs=epochs_run, rollbacks=rollbacks, trust_region_reached=reached,
                  final_lr=float(opt.lr))
    return p, mu, sd, result


def policy_probs(p, states, mu, sd):
    batch = pad_batch(states, mu, sd)
    logp, _ = L.policy_logp(p, batch)
    return [np.exp(np.where(batch["mask"][i], logp[i], -np.inf))[: s["mask"].size] for i, s in enumerate(states)]


def true_advantage(s, held):
    a = full_vector(s, centered(s["values"][held]).mean(axis=0))
    pi = s["pi"] / s["pi"].sum()
    return np.where(s["mask"], a - np.dot(pi, a), 0.0)


def conversion(states, *, signal_tapes, epsilon, eta_min, budgets, folds, seed):
    """Leave-episodes-out: fit on train episodes, score valid gain on both."""
    seeds = sorted({s["seed"] for s in states})
    chunks = [seeds[i::folds] for i in range(folds)]
    K = states[0]["values"].shape[0]
    held_rows = list(range(signal_tapes, K))
    results = {name: [] for name in budgets}
    for f, test_seeds in enumerate(chunks):
        train = [s for s in states if s["seed"] not in test_seeds and s["actions"].size >= 2]
        test = [s for s in states if s["seed"] in test_seeds and s["actions"].size >= 2]
        targets = [target_from(s, list(range(signal_tapes)), epsilon, eta_min).target for s in train]
        for name, cfg in budgets.items():
            p, mu, sd, fit = m_step(train, targets, seed=seed + f, test_states=test, **cfg)
            row = dict(fit)
            for label, group in (("train", train), ("heldout", test)):
                probs = policy_probs(p, group, mu, sd)
                gains, avail, kls, top1 = [], [], [], []
                for s, pr in zip(group, probs):
                    a = true_advantage(s, held_rows)
                    pi = s["pi"] / s["pi"].sum()
                    gains.append(float(np.dot(pr - pi, a)))
                    avail.append(float(a[s["actions"]].max()))
                    m = s["mask"]
                    kls.append(float(np.sum(pi[m] * (np.log(pi[m]) - np.log(np.clip(pr[m], 1e-300, None))))))
                    top1.append(float(int(np.argmax(np.where(m, pr, -1))) == int(s["actions"][int(np.argmax(a[s["actions"]]))])))
                row[f"{label}_valid_gain"] = float(np.mean(gains))
                row[f"{label}_available_gain"] = float(np.mean(avail))
                row[f"{label}_realized_kl"] = float(np.mean(kls))
                row[f"{label}_top1_best"] = float(np.mean(top1))
                row[f"{label}_behavior_top1_best"] = float(np.mean([
                    float(int(np.argmax(np.where(s["mask"], s["pi"], -1))) ==
                          int(s["actions"][int(np.argmax(true_advantage(s, held_rows)[s["actions"]]))]))
                    for s in group]))
            results[name].append(row)
    summary = {}
    for name, rows in results.items():
        summary[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return summary


BUDGETS = {
    # v23: one actor epoch at lr 1e-4 inside a 0.015 KL trust region.
    "v23_budget": dict(lr=1e-4, epochs=1, kl_cap=0.015, trust="sampled_stop"),
    # v24 before this audit: KL(pi_old || pi) cap 0.5, halve lr on violation.
    "v24_prior_rule": dict(lr=1e-3, epochs=32, kl_cap=0.5, trust="reverse_halving"),
    # v24 now: KL(pi || pi_old) cap 0.6, line search to the boundary.
    "v24_current": dict(lr=1e-3, epochs=32, kl_cap=0.6, trust="forward_linesearch"),
}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--episodes", type=int, default=10)
    c.add_argument("--tapes", type=int, default=4)
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--first-seed", type=int, default=401)
    c.add_argument("--output", default="runs/nmcc_pi_signal_audit_states_20260921.pkl")
    a = sub.add_parser("analyze")
    a.add_argument("--input", default="runs/nmcc_pi_signal_audit_states_20260921.pkl")
    a.add_argument("--epsilon", type=float, default=0.5)
    a.add_argument("--eta-min", type=float, default=0.03)
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--output", default="runs/nmcc_pi_signal_audit_20260921.json")
    args = ap.parse_args()

    if args.command == "collect":
        t0 = time.time()
        jobs = [(args.first_seed + k, args.tapes) for k in range(args.episodes)]
        with mp.Pool(args.workers) as pool:
            states = []
            for i, result in enumerate(pool.imap_unordered(collect_episode, jobs)):
                states.extend(result)
                print(f"[{time.time() - t0:5.0f}s] episode {i + 1}/{len(jobs)}: {len(result)} decisions", flush=True)
        with open(args.output, "wb") as handle:
            pickle.dump(states, handle)
        print("wrote", args.output, len(states), "states")
        return

    with open(args.input, "rb") as handle:
        states = pickle.load(handle)
    K = states[0]["values"].shape[0]
    report = dict(states=len(states), tapes=K, epsilon=args.epsilon, eta_min=args.eta_min)
    report["signal"] = signal_table(states)
    report["target_quality"] = {
        f"{n}_tape": target_quality(states, signal_tapes=n, epsilon=args.epsilon, eta_min=args.eta_min)
        for n in range(1, K // 2 + 1)
    }
    report["conversion"] = {
        f"{n}_tape": conversion(states, signal_tapes=n, epsilon=args.epsilon, eta_min=args.eta_min,
                                budgets=BUDGETS, folds=args.folds, seed=11)
        for n in (1, K // 2)
    }
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"{len(states)} states, {K} tapes")
    print("\nSIGNAL (per decision index)")
    for d, r in report["signal"].items():
        print(f"  d={d}: cells={r['cells']:.1f} signal_sd={r['signal_sd']:.4f} noise_sd(1 tape)={r['noise_sd_one_tape']:.4f} SNR={r['snr_one_tape']:.2f}")
    for name, table in report["target_quality"].items():
        print(f"\nE-STEP TARGET from {name} (scored on the independent tapes)")
        for d, r in table.items():
            print(f"  d={d}: valid gain {r['valid_gain']:+.4f} of {r['available_gain']:.4f} available "
                  f"({100 * r['captured']:.0f}%), positive in {100 * r['gain_positive']:.0f}% | "
                  f"KL(q||pi_old)={r['target_kl']:.3f} eta@floor={r['eta_at_floor']:.2f} top1={r['top1_matches_heldout_best']:.2f}")
    for name, table in report["conversion"].items():
        print(f"\nM-STEP CONVERSION, targets from {name}")
        for budget, r in table.items():
            print(f"  {budget:15s} requested KL(q||pi_old) {r['requested_kl']:.3f} fit {100 * r['fit_fraction']:.0f}% "
                  f"KL(pi_old||pi) {r['realized_kl']:.3f} KL(pi||pi_old) {r['realized_forward_kl']:.3f} "
                  f"edge={r['trust_region_reached']:.2f} rollbacks={r['rollbacks']:.1f} final_lr={r['final_lr']:.1e} | "
                  f"(held-out KL {r['heldout_realized_kl']:.3f}) | valid gain train {r['train_valid_gain']:+.4f}/{r['train_available_gain']:.4f} "
                  f"held-out {r['heldout_valid_gain']:+.4f}/{r['heldout_available_gain']:.4f} | "
                  f"top1 best {r['train_behavior_top1_best']:.2f}->{r['train_top1_best']:.2f} "
                  f"(held-out {r['heldout_behavior_top1_best']:.2f}->{r['heldout_top1_best']:.2f}) epochs={r['epochs']:.0f}")


if __name__ == "__main__":
    main()
