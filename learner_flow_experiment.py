#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch-free experiment: does the actor/critic parameterization let signal flow?

A numpy replica of the parts of ``EvacPolicy`` that decide gradient flow:

    representation   h_c = gelu(W1 x_c + b1)                  (encoder, shared)
    actor head       r_c = B * tanh(w . gelu(W2 h_c))          (residual logits)
                     logits_c = prior_c + r_c
    value head       y_c = u . gelu(W3 h_c)                     (intervention value)

It is trained with Adam (lr, betas, eps as in the learner) and global-norm
clipping at 0.5, on synthetic decision states whose cell values depend
non-linearly on the cell features, so a good policy and a good value head both
need the representation to learn.

Three questions are measured, one per failure mode raised in review:

1. Initialization delay: with the legacy zero readout, what fraction of the
   first actor step's gradient reaches the representation, and how many
   steps does it take before the residual can move the policy?
2. Step validity: does each accepted M-step reduce KL(q || pi) without
   exceeding the trust region, under the v23 one-epoch budget versus the
   v24 32-epoch budget with backtracking?
3. Critic isolation: how well does the value head rank cells when it may
   only fit a readout on an actor-owned representation, versus when it also
   trains the representation under a policy-preservation constraint?

Backpropagation is hand-written and checked against finite differences at
start-up.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

BOUND = 4.0
FEATURES = 8  # input width; only 4 directions carry value (see synthetic_states)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def gelu_grad(x):
    k = math.sqrt(2.0 / math.pi)
    inner = k * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * k * (1.0 + 3 * 0.044715 * x * x)


class Adam:
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def state(self):
        return ({k: v.copy() for k, v in self.m.items()},
                {k: v.copy() for k, v in self.v.items()}, self.t)

    def load(self, state):
        self.m, self.v, self.t = ({k: v.copy() for k, v in state[0].items()},
                                  {k: v.copy() for k, v in state[1].items()}, state[2])

    def step(self, grads, keys):
        self.t += 1
        for k in keys:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1**self.t)
            vh = self.v[k] / (1 - self.b2**self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def make_params(rng, d, m, readout):
    p = {
        "W1": rng.normal(0, 1 / math.sqrt(d), (m, d)),
        "b1": np.zeros(m),
        "W2": rng.normal(0, 1 / math.sqrt(m), (m, m)),
        "w": np.zeros(m),
        "W3": rng.normal(0, 1 / math.sqrt(m), (m, m)),
        "u": np.zeros(m),
    }
    if readout == "scaled":
        rms = 0.05
        p["w"] = rng.normal(0, rms / (BOUND * 0.5 * math.sqrt(m)), m)
        p["u"] = rng.uniform(-0.1 / math.sqrt(m), 0.1 / math.sqrt(m), m)
    return p


def forward(p, x):
    a1 = x @ p["W1"].T + p["b1"]
    h = gelu(a1)
    a2 = h @ p["W2"].T
    g2 = gelu(a2)
    pre = g2 @ p["w"]
    r = BOUND * np.tanh(pre)
    a3 = h @ p["W3"].T
    g3 = gelu(a3)
    y = g3 @ p["u"]
    return dict(a1=a1, h=h, a2=a2, g2=g2, pre=pre, r=r, a3=a3, g3=g3, y=y)


def masked_log_softmax(z, mask):
    z = np.where(mask, z, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return np.where(mask, z - lse, -np.inf)


def policy_logp(p, batch):
    out = forward(p, batch["x"])
    return masked_log_softmax(batch["prior"] + out["r"], batch["mask"]), out


def kl(p_log, q_log, mask):
    pl = np.where(mask, p_log, 0.0)
    ql = np.where(mask, q_log, 0.0)
    pp = np.where(mask, np.exp(pl), 0.0)
    return (pp * (pl - ql)).sum(-1)


def actor_grads(p, batch, target):
    """Gradient of mean_s KL(target || pi_theta) w.r.t. all parameters."""
    logp, out = policy_logp(p, batch)
    S = logp.shape[0]
    dz = (np.exp(logp) - target) / S  # d/dlogits of CE, zero on masked cells
    dz = np.where(batch["mask"], dz, 0.0)
    dpre = dz * BOUND * (1 - np.tanh(out["pre"]) ** 2)
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["w"] = np.einsum("sc,scm->m", dpre, out["g2"])
    dg2 = dpre[..., None] * p["w"]
    da2 = dg2 * gelu_grad(out["a2"])
    g["W2"] = np.einsum("scm,scn->mn", da2, out["h"])
    dh = da2 @ p["W2"]
    da1 = dh * gelu_grad(out["a1"])
    g["W1"] = np.einsum("scm,scd->md", da1, batch["x"])
    g["b1"] = da1.sum((0, 1))
    return g


def value_grads(p, batch, advantage, exact, clone_ref=None, clone_coef=1.0):
    """Within-state centered value regression (+ optional clone KL)."""
    out = forward(p, batch["x"])
    y = out["y"]
    w = exact / exact.sum(-1, keepdims=True)
    yc = y - (w * y).sum(-1, keepdims=True)
    err = (yc - advantage) * exact
    n = exact.sum()
    dyc = 2 * err / n
    dy = dyc - w * dyc.sum(-1, keepdims=True)
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["u"] = np.einsum("sc,scm->m", dy, out["g3"])
    dg3 = dy[..., None] * p["u"]
    da3 = dg3 * gelu_grad(out["a3"])
    g["W3"] = np.einsum("scm,scn->mn", da3, out["h"])
    dh = da3 @ p["W3"]
    if clone_ref is not None:
        logp = masked_log_softmax(batch["prior"] + out["r"], batch["mask"])
        S = logp.shape[0]
        dz = np.where(batch["mask"], (np.exp(logp) - np.exp(clone_ref)) / S, 0.0) * clone_coef
        dpre = dz * BOUND * (1 - np.tanh(out["pre"]) ** 2)
        dg2 = dpre[..., None] * p["w"]
        da2 = dg2 * gelu_grad(out["a2"])
        dh = dh + da2 @ p["W2"]
    da1 = dh * gelu_grad(out["a1"])
    g["W1"] = np.einsum("scm,scd->md", da1, batch["x"])
    g["b1"] = da1.sum((0, 1))
    loss = float((err**2).sum() / n)
    return g, loss


def clip(g, keys, max_norm=0.5):
    norm = math.sqrt(sum(float((g[k] ** 2).sum()) for k in keys))
    if norm > max_norm:
        for k in keys:
            g[k] = g[k] * (max_norm / (norm + 1e-6))
    return norm


def grad_norm(g, keys):
    return math.sqrt(sum(float((g[k] ** 2).sum()) for k in keys))


def synthetic_states(rng, S, C, d, projection):
    """Cells whose value is a non-linear function of their features."""
    x = rng.normal(size=(S, C, d))
    feasible = rng.random((S, C)) < 0.8
    feasible[:, 0] = True
    z = x @ projection
    value = 0.08 * np.tanh(z[..., 0] * z[..., 1]) + 0.05 * np.sin(z[..., 2]) - 0.03 * z[..., 3] ** 2
    prior_feature = np.clip(x[..., 0], 0, None)
    prior = prior_feature / np.maximum(prior_feature.max(-1, keepdims=True), 1e-8)
    return dict(x=x, mask=feasible, prior=prior), value


def mpo_target(behavior_log, value, mask, epsilon=0.5, eta_min=0.03):
    b = np.exp(behavior_log)
    adv = value - np.where(mask, b * value, 0).sum(-1, keepdims=True)
    targets = np.zeros_like(b)
    for s in range(b.shape[0]):
        m = mask[s]

        def q_of(eta):
            z = np.log(b[s, m]) + adv[s, m] / eta
            z -= z.max()
            q = np.exp(z)
            return q / q.sum()

        def kl_of(eta):
            q = q_of(eta)
            return float((q * (np.log(q) - np.log(b[s, m]))).sum())

        lo, hi = math.log(eta_min), math.log(1e3)
        if kl_of(eta_min) <= epsilon:
            eta = eta_min
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if kl_of(math.exp(mid)) > epsilon:
                    lo = mid
                else:
                    hi = mid
            eta = math.exp(hi)
        targets[s, m] = q_of(eta)
    return targets, adv


def check_gradients(rng):
    d, m = 5, 6
    p = make_params(rng, d, m, "scaled")
    p["w"] = rng.normal(0, 0.3, m)
    p["u"] = rng.normal(0, 0.3, m)
    proj = rng.normal(size=(d, 4))
    batch, value = synthetic_states(rng, 3, 4, d, proj)
    target = np.where(batch["mask"], rng.dirichlet(np.ones(4), 3), 0.0)
    target /= target.sum(-1, keepdims=True)
    exact = batch["mask"].astype(float)
    adv = value - value.mean(-1, keepdims=True)
    shifted = {k: v.copy() for k, v in p.items()}
    shifted["w"] = shifted["w"] + 0.2
    ref = policy_logp(shifted, batch)[0]

    def actor_loss(q):
        logp, _ = policy_logp(q, batch)
        safe = np.where(batch["mask"], logp, 0.0)
        return float((target * (np.log(np.maximum(target, 1e-300)) - safe)).sum() / 3)

    def value_loss(q):
        out = forward(q, batch["x"])
        y = out["y"]
        w = exact / exact.sum(-1, keepdims=True)
        yc = y - (w * y).sum(-1, keepdims=True)
        base = float((((yc - adv) * exact) ** 2).sum() / exact.sum())
        logp = masked_log_softmax(batch["prior"] + out["r"], batch["mask"])
        return base + float(kl(ref, logp, batch["mask"]).mean())

    worst = 0.0
    for loss_fn, grad_fn in (
        (actor_loss, lambda q: actor_grads(q, batch, target)),
        (value_loss, lambda q: value_grads(q, batch, adv, exact, clone_ref=ref)[0]),
    ):
        analytic = grad_fn(p)
        for key in ("W1", "b1", "W2", "w", "W3", "u"):
            if not np.any(analytic[key]):
                continue
            idx = tuple(rng.integers(0, s) for s in p[key].shape)
            q1 = {k: v.copy() for k, v in p.items()}
            q2 = {k: v.copy() for k, v in p.items()}
            q1[key][idx] += 1e-6
            q2[key][idx] -= 1e-6
            numeric = (loss_fn(q1) - loss_fn(q2)) / 2e-6
            worst = max(worst, abs(numeric - analytic[key][idx]) / max(1e-8, abs(numeric)))
    return worst


REP = ("W1", "b1")
ACTOR_HEAD = ("W2", "w")
CRITIC_HEAD = ("W3", "u")


def run_actor(readout, epochs, rollouts, *, seed, lr=3e-4, kl_cap=0.5, backtrack=True, steps_per_epoch=2):
    rng = np.random.default_rng(seed)
    d, m, S, C = FEATURES, 64, 40, 15
    proj = rng.normal(size=(d, 4))
    p = make_params(rng, d, m, readout)
    opt = Adam(p, lr)
    keys = REP + ACTOR_HEAD
    held, held_value = synthetic_states(np.random.default_rng(999), 400, C, d, proj)
    history = []
    first_share = None
    for rollout in range(rollouts):
        batch, value = synthetic_states(rng, S, C, d, proj)
        behavior, _ = policy_logp(p, batch)
        target, _ = mpo_target(behavior, value, batch["mask"])
        fit_before = float(kl(np.log(np.maximum(target, 1e-300)), behavior, batch["mask"]).mean())
        consecutive = 0
        for _ in range(epochs):
            snap = {k: v.copy() for k, v in p.items()}
            ostate = opt.state()
            for part in np.array_split(np.arange(S), steps_per_epoch):
                sub = {k: v[part] for k, v in batch.items()}
                g = actor_grads(p, sub, target[part])
                rep, head = grad_norm(g, REP), grad_norm(g, ACTOR_HEAD)
                if first_share is None:
                    first_share = rep / (rep + head) if rep + head > 0 else 0.0
                clip(g, keys)
                opt.step(g, keys)
            now, _ = policy_logp(p, batch)
            trust = float(kl(behavior, now, batch["mask"]).mean())
            if trust > kl_cap:
                p.update(snap)
                opt.params = p
                opt.load(ostate)
                opt.lr *= 0.5
                consecutive += 1
                if not backtrack or consecutive >= 3:
                    break
                continue
            consecutive = 0
            fit = float(kl(np.log(np.maximum(target, 1e-300)), now, batch["mask"]).mean())
            if fit <= 0.2 * fit_before:
                break
        now, _ = policy_logp(p, batch)
        fit_after = float(kl(np.log(np.maximum(target, 1e-300)), now, batch["mask"]).mean())
        held_logp, held_out = policy_logp(p, held)
        greedy = np.argmax(np.where(held["mask"], held_logp, -np.inf), -1)
        best_value = np.where(held["mask"], held_value, -np.inf).max(-1)
        chosen = held_value[np.arange(held_value.shape[0]), greedy]
        prior_choice = np.argmax(np.where(held["mask"], held["prior"], -np.inf), -1)
        history.append(
            dict(
                rollout=rollout + 1,
                fit_before=fit_before,
                fit_after=fit_after,
                residual_rms=float(np.sqrt(np.mean(held_out["r"][held["mask"]] ** 2))),
                held_regret=float(np.mean(best_value - chosen)),
                prior_regret=float(np.mean(best_value - held_value[np.arange(len(prior_choice)), prior_choice])),
            )
        )
    return dict(first_step_representation_share=first_share, history=history)


def run_critic(mode, rollouts, *, seed, lr=3e-4, epochs=4, readout="scaled", clone_coef=1.0):
    """Fit the within-state value head; representation frozen or shared.

    In ``shared_phasic`` mode the representation is also trained, with the
    policy-preservation term clone_coef * KL(pi_ref || pi); the policy drift
    each critic pass causes is recorded.
    """
    rng = np.random.default_rng(seed)
    d, m, S, C = FEATURES, 64, 40, 15
    proj = rng.normal(size=(d, 4))
    p = make_params(rng, d, m, readout)
    # A non-trivial actor so that representation changes can move the policy.
    p["w"] = np.random.default_rng(seed + 7).normal(0, 0.15, m)
    opt = Adam(p, lr)
    keys = CRITIC_HEAD + (REP if mode == "shared_phasic" else ())
    held, held_value = synthetic_states(np.random.default_rng(999), 400, C, d, proj)
    ranks, drifts = [], []
    for _ in range(rollouts):
        batch, value = synthetic_states(rng, S, C, d, proj)
        exact = batch["mask"].astype(float)
        adv = (value - (exact * value).sum(-1, keepdims=True) / exact.sum(-1, keepdims=True)) * exact
        ref = policy_logp(p, batch)[0]
        for _ in range(epochs):
            for part in np.array_split(np.arange(S), 2):
                sub = {k: v[part] for k, v in batch.items()}
                g, _ = value_grads(
                    p, sub, adv[part] / 0.05, exact[part],
                    clone_ref=ref[part] if (mode == "shared_phasic" and clone_coef > 0) else None,
                    clone_coef=clone_coef,
                )
                clip(g, keys)
                opt.step(g, keys)
        drifts.append(float(kl(ref, policy_logp(p, batch)[0], batch["mask"]).mean()))
        pred = forward(p, held["x"])["y"]
        scores = []
        for s in range(held_value.shape[0]):
            msk = held["mask"][s]
            a, b = pred[s, msk], held_value[s, msk]
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            if ra.std() > 0 and rb.std() > 0:
                scores.append(float(np.corrcoef(ra, rb)[0, 1]))
        ranks.append(float(np.mean(scores)))
    return dict(within_state_spearman=ranks, final=ranks[-1], policy_drift_kl=drifts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--features", type=int, default=8,
                        help="input width; value depends on a 4-dimensional projection")
    parser.add_argument("--output", default="runs/learner_flow_experiment_20260921.json")
    args = parser.parse_args()
    global FEATURES
    FEATURES = int(args.features)

    worst = check_gradients(np.random.default_rng(0))
    print(f"gradient check: worst relative error {worst:.2e}")
    if worst > 1e-4:
        raise SystemExit("hand-written gradients disagree with finite differences")

    configurations = {
        "v23_zero_readout_1_epoch": dict(readout="zero", epochs=1, backtrack=False, kl_cap=0.015),
        "v24_zero_readout_32_epochs": dict(readout="zero", epochs=32, backtrack=True),
        "scaled_readout_32_epochs": dict(readout="scaled", epochs=32, backtrack=True),
    }
    report = {"gradient_check_relative_error": worst, "features": FEATURES, "rollouts": args.rollouts, "seeds": args.seeds, "actor": {}, "critic": {}}
    for name, cfg in configurations.items():
        runs = [run_actor(rollouts=args.rollouts, seed=s, **cfg) for s in range(args.seeds)]
        report["actor"][name] = runs
        share = np.mean([r["first_step_representation_share"] for r in runs])
        last = [r["history"][-1] for r in runs]
        first = [r["history"][0] for r in runs]
        print(
            f"{name:30s} first-step representation share={share:.3f}  "
            f"residual RMS after 1 rollout={np.mean([h['residual_rms'] for h in first]):.3f}  "
            f"after {args.rollouts}={np.mean([h['residual_rms'] for h in last]):.3f}  "
            f"fit KL {np.mean([h['fit_before'] for h in first]):.3f}->{np.mean([h['fit_after'] for h in first]):.3f} (rollout 1)  "
            f"held-out regret {np.mean([h['prior_regret'] for h in last]):.4f}->{np.mean([h['held_regret'] for h in last]):.4f}"
        )
    critic_configurations = {
        "actor_owned_zero_readout": dict(mode="actor_owned", readout="zero"),
        "actor_owned_scaled_readout": dict(mode="actor_owned", readout="scaled"),
        "shared_phasic_zero_readout_no_clone": dict(mode="shared_phasic", readout="zero", clone_coef=0.0),
        "shared_phasic_zero_readout_clone_1": dict(mode="shared_phasic", readout="zero", clone_coef=1.0),
    }
    for name, cfg in critic_configurations.items():
        runs = [run_critic(rollouts=args.rollouts, seed=s, **cfg) for s in range(args.seeds)]
        report["critic"][name] = runs
        curve = np.mean([r["within_state_spearman"] for r in runs], axis=0)
        drift = np.mean([r["policy_drift_kl"] for r in runs], axis=0)
        print(
            f"critic {name:28s} held-out within-state Spearman: "
            + " ".join(f"{v:.2f}" for v in curve)
            + f" | max policy drift per pass {drift.max():.4f}"
        )
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
