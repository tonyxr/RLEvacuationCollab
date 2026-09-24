#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit of the v25 learning signal: will the actor see which cell is better?

The v25 controller fits a listwise ``score_ranking`` target,

    target(c) ∝ exp(Q_h(s, c) / T)   over the branched subset S (|S| = 6),

where ``Q_h`` is the return accrued in an ``h = nmccPiBranchHorizon``-step
CRN branch (20 in the State College curriculum) under the route-saving base
policy, from one tape. Deployment is the argmax score. This audit measures,
on real decision states of the calibrated testbed, whether that label points
to cells that are actually better **over the whole episode**, which is what
the reward the policy is evaluated on measures.

Collection branches *every* feasible cell under ``--tapes`` independent CRN
tapes to the full horizon and records the return accrued at several offsets
along the same trajectory, so truncated and full-horizon values come from
identical simulations. Analysis then rebuilds any target variant (horizon,
tapes, temperature, subset rule, MPO vs listwise) from the stored values and
scores it against **full-horizon values on tapes it did not use**.

Behavior: the v25 initial controller (zero residual => scores are the
route-time-saving prior; epsilon-greedy with epsilon = 0.35 over the argmax).
The v25 State College mask settings (operational benefit required, hazard
safety margin >= 0.05, forecast danger <= 0.6) are applied.

    python nmcc_v25_signal_audit.py collect --episodes 12 --tapes 3 --workers 2
    python nmcc_v25_signal_audit.py analyze
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import time

import numpy as np

import CounterfactualBranch as CB
import headroom_lib as H
import learner_flow_experiment as L
import NMCCPolicyImprovement as PI
from DecisionInterface import PolicyDecision, RegionalObservationBuilder, RegionalShelterExecutor
from nmcc_pi_reference import BUDGET, HORIZON, INTERVAL, MAP, features

OFFSETS = (10, 20, 30)  # truncation points (timesteps after the decision)
SAVING_COLUMN = 1 + 7
V25_MASK = dict(
    requireCandidateOperationalBenefit=True,
    minimumCandidateReroutableFraction=0.0,
    minimumCandidateRouteTimeSaving=0.0,
    minimumCandidateHazardSafetyMargin=0.05,
    maximumShelterForecastDanger=0.6,
)


def prior_scores(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    saving = np.clip(X[:, SAVING_COLUMN], 0.0, None)
    top = saving[mask].max() if mask.any() else 0.0
    return saving / top if top > 0 else np.zeros_like(saving)


def build(seed):
    core = H.build_core(seed, **MAP)
    for key, value in V25_MASK.items():
        setattr(core, key, value)
    builder = RegionalObservationBuilder(core, initial_population=800, horizon=HORIZON, maximum_deployments=BUDGET)
    return core, builder, RegionalShelterExecutor(core)


def continue_with_checkpoints(core, builder, executor, clock, reward):
    """Base-policy continuation to the horizon, recording return at OFFSETS."""
    start_t, start = clock.t, clock.accumulated
    marks = {}
    while clock.t < clock.horizon:
        PI.step_and_score(core, clock, reward)
        elapsed = clock.t - start_t
        if elapsed in OFFSETS:
            marks[elapsed] = clock.accumulated - start
        if clock.is_decision(clock.t):
            obs = builder.build(decision_index=clock.deployed, simulation_time=clock.t,
                                remaining_deployments=clock.budget - clock.deployed)
            if obs.has_feasible_action:
                executor.execute(obs, PolicyDecision(action_index=int(PI.route_saving_policy(obs)),
                                                     strategy="audit_base"))
                clock.deployed += 1
    full = clock.accumulated - start
    return [marks.get(o, full) for o in OFFSETS] + [full]


def collect_episode(args):
    seed, tapes, epsilon = args
    rng = np.random.default_rng(seed * 13 + 5)
    core, builder, executor = build(seed)
    reward = PI.RewardProcessor()
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
        scores = prior_scores(X, mask)
        feasible = np.flatnonzero(mask)
        values = np.zeros((tapes, feasible.size, len(OFFSETS) + 1))
        base = CB.capture(core, label="v25-audit")
        try:
            for tape in range(tapes):
                tseed = PI.tape_seed(seed, clock.deployed, tape)
                for column, action in enumerate(feasible):
                    CB.restore(core, base)
                    PI.reseed_exogenous(core, tseed)
                    branch = clock.copy()
                    executor.execute(obs, PolicyDecision(action_index=int(action), strategy="audit_branch"))
                    branch.deployed += 1
                    values[tape, column] = continue_with_checkpoints(core, builder, executor, branch, reward)
        finally:
            CB.restore(core, base)
        greedy = int(feasible[int(np.argmax(scores[feasible]))])
        action = greedy if rng.random() >= epsilon else int(rng.choice(feasible))
        states.append(dict(seed=seed, decision_index=int(clock.deployed), t=int(clock.t), X=X, mask=mask,
                           scores=scores, actions=feasible, values=values, executed=action))
        executor.execute(obs, PolicyDecision(action_index=action, strategy="audit_behavior"))
        clock.deployed += 1
    return states


# ----------------------------------------------------------------------------- analysis
def softmax_masked(z, mask):
    return PI.softmax(z, mask)


def spearman(a, b):
    return PI.spearman(np.asarray(a), np.asarray(b))


def truth(state, held_tapes):
    """Full-horizon within-state advantage (centered on the cell mean) from held-out tapes."""
    v = state["values"][held_tapes, :, -1]
    v = v - v.mean(axis=1, keepdims=True)
    return v.mean(axis=0)


def subset_v25(state, rng, max_branches=6):
    """v25 branch subset: top half by score probability (incl. executed) plus random support."""
    mask = state["mask"]
    probs = softmax_masked(state["scores"], mask)
    chosen = PI.select_branch_actions(state["actions"], probs, decision_index=state["decision_index"],
                                      exhaustive_decisions=0, max_branches=max_branches,
                                      must_include=(state["executed"],), rng=rng)
    return np.array([int(np.flatnonzero(state["actions"] == a)[0]) for a in chosen])


def label_values(state, columns, *, horizon_index, tapes):
    return state["values"][tapes][:, columns, horizon_index].mean(axis=0)


def evaluate_variant(states, *, horizon_index, signal_tapes, held_tapes, subset, target, T, epsilon, eta_min, seed=3):
    rng = np.random.default_rng(seed)
    by_d = {}
    for s in states:
        n = s["actions"].size
        if n < 2:
            continue
        cols = subset_v25(s, rng) if subset == "v25" else np.arange(n)
        a_true = truth(s, held_tapes)
        greedy_col = int(np.argmax(s["scores"][s["actions"]]))
        q = label_values(s, cols, horizon_index=horizon_index, tapes=signal_tapes)
        if target == "score":
            w = np.exp((q - q.max()) / T)
            w /= w.sum()
        else:  # MPO around the behavior score softmax
            pi = softmax_masked(s["scores"], s["mask"])[s["actions"]][cols]
            pi = pi / pi.sum()
            tgt = PI.improvement_target(pi, np.ones(cols.size, bool), np.arange(cols.size), q,
                                        epsilon=epsilon, eta_min=eta_min)
            w = tgt.target
        label = int(cols[int(np.argmax(q))])
        best = int(np.argmax(a_true))
        row = dict(
            label_gain=float(a_true[label] - a_true[greedy_col]),       # argmax label vs current greedy
            target_gain=float(np.dot(w, a_true[cols]) - a_true[greedy_col]),
            available=float(a_true[best] - a_true[greedy_col]),
            best_in_subset=float(best in set(cols.tolist())),
            label_is_best=float(label == best),
            label_worse=float(a_true[label] < a_true[greedy_col] - 1e-9),
            rank_corr=float(spearman(q, a_true[cols])) if cols.size >= 3 else float("nan"),
            target_max=float(w.max()),
        )
        by_d.setdefault(s["decision_index"], []).append(row)
    out = {}
    for d, rows in sorted(by_d.items()):
        out[int(d)] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
        out[int(d)]["states"] = len(rows)
    rows = [r for rs in by_d.values() for r in rs]
    out["all"] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out["all"]["states"] = len(rows)
    return out


def horizon_table(states, tapes):
    """Does an h-step contrast rank cells like the full-horizon contrast?"""
    K = tapes
    res = {}
    for hi, name in enumerate([f"h{o}" for o in OFFSETS] + ["full"]):
        per_d = {}
        for s in states:
            if s["actions"].size < 3:
                continue
            short = s["values"][[0], :, hi].mean(axis=0)
            full = truth(s, list(range(1, K)))
            short_c = short - short.mean()
            per_d.setdefault(s["decision_index"], []).append(
                (spearman(short, full), float(np.std(short_c)), float(np.std(full))))
        res[name] = {int(d): dict(rank_corr_with_full=float(np.nanmean([r[0] for r in v])),
                                  contrast_sd=float(np.mean([r[1] for r in v])),
                                  full_sd=float(np.mean([r[2] for r in v])), states=len(v))
                     for d, v in sorted(per_d.items())}
    return res


# --- replica M-step for the v25 objective -------------------------------------
def backprop_logits(p, batch, dz):
    out = L.forward(p, batch["x"])
    dpre = dz * L.BOUND * (1 - np.tanh(out["pre"]) ** 2)
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["w"] = np.einsum("sc,scm->m", dpre, out["g2"])
    dg2 = dpre[..., None] * p["w"]
    da2 = dg2 * L.gelu_grad(out["a2"])
    g["W2"] = np.einsum("scm,scn->mn", da2, out["h"])
    dh = da2 @ p["W2"]
    da1 = dh * L.gelu_grad(out["a1"])
    g["W1"] = np.einsum("scm,scd->md", da1, batch["x"])
    g["b1"] = da1.sum((0, 1))
    return g


def padded(states, mu, sd):
    C = max(s["mask"].size for s in states)
    d = states[0]["X"].shape[1]
    x = np.zeros((len(states), C, d))
    mask = np.zeros((len(states), C), bool)
    prior = np.zeros((len(states), C))
    for i, s in enumerate(states):
        n = s["mask"].size
        x[i, :n] = np.clip((s["X"] - mu) / sd, -5, 5)
        mask[i, :n] = s["mask"]
        prior[i, :n] = s["scores"]
    return dict(x=x, mask=mask, prior=prior)


def fit_v25(train, labels, *, lr, epochs, margin, margin_coef, seed, objective="score", T=0.03,
            epsilon=0.2, eta_min=0.03, kl_cap=0.3):
    """labels: per state (columns in actions order, values). Returns params & stats."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([s["X"][s["mask"]] for s in train])
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-2)
    batch = padded(train, mu, sd)
    S, C = batch["mask"].shape
    exact = np.zeros((S, C), bool)
    tgt = np.zeros((S, C))
    best = np.zeros(S, int)
    for i, (s, (cols, q)) in enumerate(zip(train, labels)):
        cells = s["actions"][cols]
        exact[i, cells] = True
        if objective == "score":
            w = np.exp((q - q.max()) / T)
            w /= w.sum()
            tgt[i, cells] = w
        else:
            pi = PI.softmax(s["scores"], s["mask"])
            t = PI.improvement_target(pi, s["mask"], cells, q, epsilon=epsilon, eta_min=eta_min).target
            tgt[i, : t.size] = t
        best[i] = int(cells[int(np.argmax(q))])
    p = L.make_params(rng, batch["x"].shape[-1], 64, "zero")
    opt = L.Adam(p, lr)
    keys = L.REP + L.ACTOR_HEAD
    behavior, _ = L.policy_logp(p, batch)
    for _ in range(epochs):
        out = L.forward(p, batch["x"])
        logits = batch["prior"] + out["r"]
        if objective == "score":
            lp = L.masked_log_softmax(logits, exact)
            probs = np.where(exact, np.exp(np.where(exact, lp, 0.0)), 0.0)
            dz = (probs - tgt) / S
            bscore = logits[np.arange(S), best][:, None]
            rivals = exact.copy()
            rivals[np.arange(S), best] = False
            viol = (margin - (bscore - logits) > 0) & rivals
            n = max(1, rivals.sum())
            dz = dz + margin_coef * (viol.astype(float) - np.eye(C)[best] * viol.sum(1, keepdims=True)) / n
        else:
            lp = L.masked_log_softmax(logits, batch["mask"])
            dz = np.where(batch["mask"], (np.exp(np.where(batch["mask"], lp, 0.0)) - tgt) / S, 0.0)
        snap = {k: v.copy() for k, v in p.items()}
        g = backprop_logits(p, batch, dz)
        L.clip(g, keys)
        opt.step(g, keys)
        now, _ = L.policy_logp(p, batch)
        if objective == "mpo" and float(L.kl(now, behavior, batch["mask"]).mean()) > kl_cap:
            p.update(snap)
            break
    return p, mu, sd


def argmax_policy_gain(p, mu, sd, states, held_tapes):
    batch = padded(states, mu, sd)
    out = L.forward(p, batch["x"])
    logits = batch["prior"] + out["r"]
    gains, avail, best_hits = [], [], []
    for i, s in enumerate(states):
        a_true = truth(s, held_tapes)
        feas = s["actions"]
        choice = int(np.argmax(logits[i, feas]))
        greedy = int(np.argmax(s["scores"][feas]))
        gains.append(float(a_true[choice] - a_true[greedy]))
        avail.append(float(a_true.max() - a_true[greedy]))
        best_hits.append(float(choice == int(np.argmax(a_true))))
    return float(np.mean(gains)), float(np.mean(avail)), float(np.mean(best_hits))


def conversion(states, variants, *, folds, held_tapes, seed=7):
    seeds = sorted({s["seed"] for s in states})
    chunks = [seeds[i::folds] for i in range(folds)]
    results = {}
    for name, v in variants.items():
        rows = []
        for f, test_seeds in enumerate(chunks):
            rng = np.random.default_rng(seed + f)
            train = [s for s in states if s["seed"] not in test_seeds and s["actions"].size >= 2]
            test = [s for s in states if s["seed"] in test_seeds and s["actions"].size >= 2]
            labels = []
            for s in train:
                cols = subset_v25(s, rng) if v["subset"] == "v25" else np.arange(s["actions"].size)
                q = label_values(s, cols, horizon_index=v["horizon_index"], tapes=v["signal_tapes"])
                labels.append((cols, q))
            p, mu, sd = fit_v25(train, labels, lr=v["lr"], epochs=v["epochs"], margin=v.get("margin", 0.25),
                                margin_coef=v.get("margin_coef", 0.5), seed=seed + f,
                                objective=v["objective"], T=v.get("T", 0.03))
            tr = argmax_policy_gain(p, mu, sd, train, held_tapes)
            te = argmax_policy_gain(p, mu, sd, test, held_tapes)
            rows.append(dict(train_gain=tr[0], train_available=tr[1], train_best=tr[2],
                             heldout_gain=te[0], heldout_available=te[1], heldout_best=te[2]))
        results[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results[name]["heldout_gain_per_fold"] = [r["heldout_gain"] for r in rows]
    return results


# --- sequential updates: does the policy keep moving in the right direction? ----
def cosine_lr(base, update, warmup=1, decay=12, floor=0.15):
    if update < warmup:
        return base * (update + 1) / warmup
    progress = min(1.0, (update - warmup) / max(1, decay))
    return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * progress)))


def _labels(state, variant, rng, K):
    tape = [int(rng.integers(K))]
    cols = subset_v25(state, rng) if variant["subset"] == "v25" else np.arange(state["actions"].size)
    q = label_values(state, cols, horizon_index=variant["horizon_index"], tapes=tape)
    return cols, q


def sequential_run(states, variant, *, heldout_seeds, updates, episodes_per_update, seed):
    """Replay the registered update schedule on the stored decision states.

    Each update draws ``episodes_per_update`` training episodes; each of their
    states is labelled once, from one random tape and (for |S| = 6) a fresh
    v25 subset -- exactly the information one real rollout would produce.
    ``replay=False`` (v25) fits only the newest labels; ``replay=True`` keeps
    every label collected so far, which is valid because a fixed-base-policy
    branch value does not depend on the actor.  ``objective``:

    * ``score``: v25 listwise CE (T = 0.03) + best-vs-rival margin on the
      branched subset, forward-KL trust region, cosine LR, 12 epochs;
    * ``mpo``: v24 KL-constrained target around the current policy;
    * ``value``: within-state advantage regression head (the replica of
      ``improvement_value_heads``), acting greedily on prior + prediction.

    Held-out episodes are scored after every update with the deterministic
    argmax policy against the full-horizon value averaged over all tapes.
    """
    rng = np.random.default_rng(seed)
    K = states[0]["values"].shape[0]
    train = [s for s in states if s["seed"] not in heldout_seeds and s["actions"].size >= 2]
    test = [s for s in states if s["seed"] in heldout_seeds and s["actions"].size >= 2]
    train_seeds = sorted({s["seed"] for s in train})
    X = np.concatenate([s["X"][s["mask"]] for s in train])
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-2)
    p = L.make_params(rng, X.shape[1], 64, "zero")
    opt = L.Adam(p, variant["lr"])
    objective = variant["objective"]
    keys = L.REP + (L.CRITIC_HEAD if objective == "value" else L.ACTOR_HEAD)
    all_tapes = list(range(K))
    value_weight = float(variant.get("value_weight", 1.0))
    curve = []
    memory = []  # (state, cols, q)

    def decision_logits(batch):
        out = L.forward(p, batch["x"])
        if objective == "value":
            return batch["prior"] + value_weight * out["y"], out
        return batch["prior"] + out["r"], out

    def score(group):
        batch = padded(group, mu, sd)
        logits, out = decision_logits(batch)
        gains, hits = [], []
        for i, s in enumerate(group):
            a = truth(s, all_tapes)
            feas = s["actions"]
            c = int(np.argmax(logits[i, feas]))
            g = int(np.argmax(s["scores"][feas]))
            gains.append(float(a[c] - a[g]))
            hits.append(float(c == int(np.argmax(a))))
        resid = (logits - batch["prior"])[batch["mask"]]
        return float(np.mean(gains)), float(np.mean(hits)), float(np.sqrt(np.mean(resid ** 2))), \
            float(np.mean(np.abs(resid) > 0.9 * L.BOUND))

    g0 = score(test)
    curve.append(dict(update=0, heldout_gain=g0[0], heldout_best=g0[1], residual_rms=g0[2], saturated=g0[3]))
    for u in range(updates):
        chosen = set(rng.choice(train_seeds, size=min(episodes_per_update, len(train_seeds)), replace=False).tolist())
        fresh = [(st,) + _labels(st, variant, rng, K) for st in train if st["seed"] in chosen]
        memory.extend(fresh)
        data = memory if variant.get("replay") else fresh
        batch_states = [d[0] for d in data]
        batch = padded(batch_states, mu, sd)
        S, C = batch["mask"].shape
        tgt = np.zeros((S, C))
        exact = np.zeros((S, C), bool)
        adv = np.zeros((S, C))
        best = np.zeros(S, int)
        behavior, _ = L.policy_logp(p, batch)
        for i, (st, cols, q) in enumerate(data):
            cells = st["actions"][cols]
            exact[i, cells] = True
            best[i] = int(cells[int(np.argmax(q))])
            adv[i, cells] = (q - q.mean()) / 0.05
            if objective == "score":
                w = np.exp((q - q.max()) / variant.get("T", 0.03))
                tgt[i, cells] = w / w.sum()
            elif objective == "mpo":
                pi = np.exp(np.where(batch["mask"][i], behavior[i], -np.inf))[: st["mask"].size]
                t = PI.improvement_target(pi, st["mask"], cells, q, epsilon=variant.get("epsilon", 0.5),
                                          eta_min=0.03).target
                tgt[i, : t.size] = t
        lr = cosine_lr(variant["lr"], u) if variant.get("schedule") == "cosine" else variant["lr"]
        opt.lr = lr
        for _ in range(variant["epochs"]):
            if objective == "value":
                g, _ = L.value_grads(p, batch, adv, exact.astype(float))
                L.clip(g, keys)
                opt.step(g, keys)
                continue
            out = L.forward(p, batch["x"])
            logits = batch["prior"] + out["r"]
            if objective == "score":
                lp = L.masked_log_softmax(logits, exact)
                probs = np.where(exact, np.exp(np.where(exact, lp, 0.0)), 0.0)
                dz = (probs - tgt) / S
                if variant.get("margin_coef", 0.5) > 0:
                    bscore = logits[np.arange(S), best][:, None]
                    rivals = exact.copy()
                    rivals[np.arange(S), best] = False
                    viol = ((variant.get("margin", 0.25) - (bscore - logits)) > 0) & rivals
                    n = max(1, rivals.sum())
                    onehot = np.zeros((S, C))
                    onehot[np.arange(S), best] = 1.0
                    dz = dz + variant.get("margin_coef", 0.5) * (viol - onehot * viol.sum(1, keepdims=True)) / n
            else:
                lp = L.masked_log_softmax(logits, batch["mask"])
                dz = np.where(batch["mask"], (np.exp(np.where(batch["mask"], lp, 0.0)) - tgt) / S, 0.0)
            snap = {k: v.copy() for k, v in p.items()}
            g = backprop_logits(p, batch, dz)
            L.clip(g, keys)
            opt.step(g, keys)
            now, _ = L.policy_logp(p, batch)
            cap = variant.get("kl_cap")
            if cap is not None and float(L.kl(now, behavior, batch["mask"]).mean()) > cap:
                current = {k: v.copy() for k, v in p.items()}
                lo, hi = 0.0, 1.0
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    for k in p:
                        p[k] = snap[k] + mid * (current[k] - snap[k])
                    if float(L.kl(L.policy_logp(p, batch)[0], behavior, batch["mask"]).mean()) <= cap:
                        lo = mid
                    else:
                        hi = mid
                for k in p:
                    p[k] = snap[k] + lo * (current[k] - snap[k])
                break
        gu = score(test)
        curve.append(dict(update=u + 1, lr=lr, labelled_states=len(data), heldout_gain=gu[0], heldout_best=gu[1],
                          residual_rms=gu[2], saturated=gu[3]))
    return curve


H20, FULL = 1, 3
SEQUENTIAL = {
    "A v25 registered: h=20, |S|=6, listwise+margin, no replay":
        dict(horizon_index=H20, subset="v25", objective="score", lr=5e-4, schedule="cosine", epochs=12, kl_cap=0.3),
    "B v25 + replay of all exact labels":
        dict(horizon_index=H20, subset="v25", objective="score", lr=5e-4, schedule="cosine", epochs=12, kl_cap=0.3,
             replay=True),
    "C v25 + full-horizon labels":
        dict(horizon_index=FULL, subset="v25", objective="score", lr=5e-4, schedule="cosine", epochs=12, kl_cap=0.3),
    "D value regression + replay, h=20":
        dict(horizon_index=H20, subset="v25", objective="value", lr=1e-3, epochs=24, replay=True),
    "E value regression + replay, full horizon":
        dict(horizon_index=FULL, subset="v25", objective="value", lr=1e-3, epochs=24, replay=True),
    "F listwise + replay, full horizon":
        dict(horizon_index=FULL, subset="v25", objective="score", lr=5e-4, schedule="cosine", epochs=12, kl_cap=0.3,
             replay=True),
}


def sequential(states, *, folds=4, updates=11, episodes_per_update=4, repeats=3):
    seeds = sorted({s["seed"] for s in states})
    chunks = [seeds[i::folds] for i in range(folds)]
    out = {}
    for name, v in SEQUENTIAL.items():
        curves = []
        for f, held in enumerate(chunks):
            for r in range(repeats):
                curves.append(sequential_run(states, v, heldout_seeds=set(held), updates=updates,
                                             episodes_per_update=episodes_per_update, seed=100 * f + r))
        keys = ("heldout_gain", "heldout_best", "residual_rms", "saturated")
        mean = [{k: float(np.mean([c[i][k] for c in curves])) for k in keys} for i in range(updates + 1)]
        se = [float(np.std([c[i]["heldout_gain"] for c in curves], ddof=1) / np.sqrt(len(curves)))
              for i in range(updates + 1)]
        out[name] = dict(mean=mean, heldout_gain_se=se, runs=len(curves))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--episodes", type=int, default=12)
    c.add_argument("--tapes", type=int, default=3)
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--epsilon", type=float, default=0.35)
    c.add_argument("--first-seed", type=int, default=501)
    c.add_argument("--output", default="runs/nmcc_v25_signal_audit_states_20260921.pkl")
    a = sub.add_parser("analyze")
    a.add_argument("--input", default="runs/nmcc_v25_signal_audit_states_20260921.pkl")
    a.add_argument("--folds", type=int, default=4)
    a.add_argument("--output", default="runs/nmcc_v25_signal_audit_20260921.json")
    q = sub.add_parser("sequential")
    q.add_argument("--input", default="runs/nmcc_v25_signal_audit_states_20260921.pkl")
    q.add_argument("--folds", type=int, default=4)
    q.add_argument("--repeats", type=int, default=3)
    q.add_argument("--updates", type=int, default=11)
    q.add_argument("--output", default="runs/nmcc_v25_sequential_audit_20260921.json")
    args = ap.parse_args()

    if args.command == "collect":
        t0 = time.time()
        jobs = [(args.first_seed + k, args.tapes, args.epsilon) for k in range(args.episodes)]
        states = []
        with mp.Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(collect_episode, jobs)):
                states.extend(result)
                print(f"[{time.time() - t0:5.0f}s] episode {i + 1}/{len(jobs)}: {len(result)} decisions", flush=True)
        with open(args.output, "wb") as handle:
            pickle.dump(dict(offsets=OFFSETS, states=states), handle)
        print("wrote", args.output, len(states), "states")
        return

    with open(args.input, "rb") as handle:
        payload = pickle.load(handle)
    states = payload["states"]
    if args.command == "sequential":
        result = sequential(states, folds=args.folds, updates=args.updates, repeats=args.repeats)
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
        for name, r in result.items():
            print(name)
            for i, m in enumerate(r["mean"]):
                if i in (0, 1, 2, 4, 6, 8, len(r["mean"]) - 1):
                    print(f"   update {i:2d}: held-out gain {m['heldout_gain']:+.4f} (se {r['heldout_gain_se'][i]:.4f}) "
                          f"best {m['heldout_best']:.2f} residual rms {m['residual_rms']:.2f} saturated {m['saturated']:.2f}")
        return
    K = states[0]["values"].shape[0]
    held = list(range(1, K))
    H20 = OFFSETS.index(20)
    FULL = len(OFFSETS)
    report = dict(states=len(states), tapes=K, offsets=list(OFFSETS))
    report["horizon"] = horizon_table(states, K)
    variants = {
        "v25_registered (h=20, 1 tape, |S|=6, listwise T=0.03)":
            dict(horizon_index=H20, signal_tapes=[0], subset="v25", target="score", T=0.03),
        "full horizon, |S|=6, listwise T=0.03":
            dict(horizon_index=FULL, signal_tapes=[0], subset="v25", target="score", T=0.03),
        "full horizon, all cells, listwise T=0.03":
            dict(horizon_index=FULL, signal_tapes=[0], subset="all", target="score", T=0.03),
        "h=20, all cells, listwise T=0.03":
            dict(horizon_index=H20, signal_tapes=[0], subset="all", target="score", T=0.03),
        "full horizon, all cells, MPO eps=0.5":
            dict(horizon_index=FULL, signal_tapes=[0], subset="all", target="mpo", T=0.03),
    }
    report["targets"] = {name: evaluate_variant(states, held_tapes=held, epsilon=0.5, eta_min=0.03, **v)
                         for name, v in variants.items()}
    fits = {
        "v25_registered: h=20, |S|=6, listwise+margin, lr 5e-4, 12 epochs":
            dict(horizon_index=H20, signal_tapes=[0], subset="v25", objective="score", lr=5e-4, epochs=12),
        "v25 late-schedule lr (7.5e-5), 12 epochs":
            dict(horizon_index=H20, signal_tapes=[0], subset="v25", objective="score", lr=7.5e-5, epochs=12),
        "full horizon, |S|=6, listwise+margin, lr 5e-4, 12 epochs":
            dict(horizon_index=FULL, signal_tapes=[0], subset="v25", objective="score", lr=5e-4, epochs=12),
        "full horizon, all cells, listwise+margin, lr 5e-4, 12 epochs":
            dict(horizon_index=FULL, signal_tapes=[0], subset="all", objective="score", lr=5e-4, epochs=12),
        "full horizon, all cells, listwise+margin, lr 1e-3, 32 epochs":
            dict(horizon_index=FULL, signal_tapes=[0], subset="all", objective="score", lr=1e-3, epochs=32),
        "full horizon, all cells, MPO, lr 1e-3, 32 epochs":
            dict(horizon_index=FULL, signal_tapes=[0], subset="all", objective="mpo", lr=1e-3, epochs=32),
    }
    report["conversion"] = conversion(states, fits, folds=args.folds, held_tapes=held)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"{len(states)} states, {K} tapes")
    print("\nHORIZON: rank correlation of an h-step one-tape contrast with the full-horizon truth (other tapes)")
    for name, table in report["horizon"].items():
        print(f"  {name:5s} " + "  ".join(f"d{d}: rho={r['rank_corr_with_full']:+.2f} sd={r['contrast_sd']:.3f}/{r['full_sd']:.3f}"
                                         for d, r in table.items()))
    print("\nTARGET VALIDITY (gain in full-horizon return vs the current greedy choice, per state)")
    for name, table in report["targets"].items():
        r = table["all"]
        print(f"  {name}")
        for d, row in table.items():
            print(f"     {str(d):>3}: label gain {row['label_gain']:+.4f} target gain {row['target_gain']:+.4f} of {row['available']:.4f} "
                  f"| best in S {row['best_in_subset']:.2f} label=best {row['label_is_best']:.2f} "
                  f"label worse than greedy {row['label_worse']:.2f} rho {row['rank_corr']:+.2f} max w {row['target_max']:.2f} (n={row['states']})")
    print("\nCONVERSION: argmax-policy gain vs current greedy, full-horizon truth (leave-episodes-out)")
    for name, r in report["conversion"].items():
        print(f"  {name}: train {r['train_gain']:+.4f}/{r['train_available']:.4f} best {r['train_best']:.2f} | "
              f"held-out {r['heldout_gain']:+.4f}/{r['heldout_available']:.4f} best {r['heldout_best']:.2f} "
              f"folds {['%+.3f' % g for g in r['heldout_gain_per_fold']]}")


if __name__ == "__main__":
    main()
