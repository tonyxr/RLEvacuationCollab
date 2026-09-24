#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operating characteristics of the live v26 (`value_lcb`) deployment gate.

Replays the gate rule implemented in ``RLBridge._improvement_validation_metrics``
and ``RLBridge.improvement_gate_passed`` against the real v25 audit labels
(16 calibrated-testbed episodes, every feasible cell branched to the full
horizon on 3 independent CRN tapes).

The gate rule as implemented:

1. Episodes are split by ``episode_id % round(1/validation_fraction)``; the
   validation episodes are never fitted.
2. For every validation state with at least two branched cells and the base
   cell among them, the controller's choice is the argmax **over the branched
   cells** of ``base_level + scale * LCB / temperature``.
3. ``gain(state) = A(chosen) - A(base)`` from that episode's own stored label
   tape; per-episode mean; then mean and standard error over episodes.
4. ``gain_lower = gain - z * se`` with ``z = 1.0`` is appended to the gate
   history, and the gate opens when the last ``nmccPiGateUpdates`` entries are
   all strictly positive.

Measured here, against truth held out from the gate (tapes 1 and 2):

* the per-refit open rate for models of known skill, including a no-skill one;
* the same rate when the epoch is chosen by validation loss on the *same*
  validation episodes, which is what early stopping does (selection optimism);
* the deployed gain when the controller may choose among **all** feasible
  cells, as it does at deployment, versus only the branched cells the gate
  scored.

    python live26_gate_check.py --trials 400
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np

import NMCCPolicyImprovement as NPI

STATES = "runs/nmcc_v25_signal_audit_states_20260921.pkl"
MEMBERS = 5
UNCERTAINTY_PENALTY = 1.0  # nmccPiModelUncertaintyPenalty
VALIDATION_GAIN_Z = 1.0  # nmccPiValidationGainZ
GATE_UPDATES = 3  # nmccPiGateUpdates
MIN_VALIDATION_STATES = 8  # nmccPiMinimumValidationStates
EXHAUSTIVE_DECISIONS = 2  # nmccPiExhaustiveDecisions
MAX_BRANCHES = 6  # nmccPiMaxBranches
REPLAY_EPOCHS = 24  # nmccPiReplayEpochs (epochs the early stopper selects over)
# The audit pool holds 16 episodes, so a 20% split gives 3 validation episodes.
# The registered 96-episode State College run reaches about 19, which shrinks
# the standard error by sqrt(3/19).  SE_SCALE emulates that larger pool.
SE_SCALE = 1.0


def load(path: str = STATES) -> list[dict]:
    with open(path, "rb") as handle:
        return list(pickle.load(handle)["states"])


def base_action(state: dict) -> int:
    feasible = state["actions"]
    return int(feasible[int(np.argmax(state["scores"][feasible]))])


def values(state: dict, tapes) -> np.ndarray:
    """Full-horizon value per feasible cell, in the order of state['actions']."""
    return state["values"][list(tapes), :, -1].mean(axis=0)


def branched_columns(state: dict, rng: np.random.Generator) -> np.ndarray:
    """Columns the v26 allocation would branch, as indices into state['actions']."""
    feasible = state["actions"]
    if int(state["decision_index"]) < EXHAUSTIVE_DECISIONS:
        return np.arange(feasible.size)
    probabilities = NPI.softmax(state["scores"], state["mask"])
    chosen = NPI.select_branch_actions(
        feasible,
        probabilities,
        decision_index=int(state["decision_index"]),
        exhaustive_decisions=EXHAUSTIVE_DECISIONS,
        max_branches=MAX_BRANCHES,
        must_include=(int(state["executed"]), base_action(state)),
        rng=rng,
    )
    lookup = {int(a): i for i, a in enumerate(feasible)}
    return np.asarray(sorted(lookup[int(a)] for a in chosen), dtype=np.int64)


def members_for(state: dict, noise: float, rng: np.random.Generator) -> np.ndarray:
    """(members, cells) ensemble predictions of the within-state advantage.

    ``noise = inf`` is the no-skill model: pure disagreement about a signal
    that is not there.
    """
    truth = values(state, (1, 2))
    truth = (truth - truth.mean()) / max(1e-9, float(np.std(values(state, (0, 1, 2)))))
    if not np.isfinite(noise):
        return rng.normal(size=(MEMBERS, truth.size))
    return truth[None, :] + noise * rng.normal(size=(MEMBERS, truth.size))


def controller_choice(members: np.ndarray, columns: np.ndarray, base_column: int) -> int:
    """Column chosen by prior + LCB, restricted to ``columns`` (the gate's view).

    The base level is constant within a state, so it does not affect the
    argmax; what remains is exactly the implemented base-relative LCB.
    """
    paired = members - members[:, [base_column]]
    lcb = paired.mean(axis=0) - UNCERTAINTY_PENALTY * paired.std(axis=0)
    masked = np.full(lcb.shape, -np.inf)
    masked[columns] = lcb[columns]
    return int(np.argmax(masked))


def trial(states, episodes, rng, noise, *, select_epochs: bool):
    """One refit: returns (gate statistic, true deployed gain, true gate-view gain)."""
    order = rng.permutation(episodes)
    validation = set(order[: max(1, int(round(0.2 * len(episodes))))].tolist())
    rows = []
    for state in states:
        if state["seed"] not in validation:
            continue
        columns = branched_columns(state, rng)
        feasible = state["actions"]
        base_column = int(np.flatnonzero(feasible == base_action(state))[0])
        if columns.size < 2 or base_column not in set(columns.tolist()):
            continue
        label = values(state, (0,))
        truth = values(state, (1, 2))
        candidates = []
        for _ in range(REPLAY_EPOCHS if select_epochs else 1):
            members = members_for(state, noise, rng)
            candidates.append(members)
        rows.append((state, columns, base_column, label, truth, candidates))
    if len(rows) < MIN_VALIDATION_STATES:
        return None
    # Early stopping picks one epoch by validation loss on these same episodes.
    if select_epochs:
        losses = []
        for epoch in range(REPLAY_EPOCHS):
            error = 0.0
            for _, columns, base_column, label, _, candidates in rows:
                members = candidates[epoch]
                paired = (members - members[:, [base_column]]).mean(axis=0)
                target = label - label[base_column]
                error += float(np.mean((paired[columns] - target[columns]) ** 2))
            losses.append(error)
        epoch = int(np.argmin(losses))
    else:
        epoch = 0
    gate_gain_by_episode: dict[int, list[float]] = {}
    true_gate_view: list[float] = []
    true_deployed: list[float] = []
    for state, columns, base_column, label, truth, candidates in rows:
        members = candidates[epoch]
        chosen = controller_choice(members, columns, base_column)
        deployed = controller_choice(members, np.arange(members.shape[1]), base_column)
        gate_gain_by_episode.setdefault(state["seed"], []).append(
            float(label[chosen] - label[base_column])
        )
        true_gate_view.append(float(truth[chosen] - truth[base_column]))
        true_deployed.append(float(truth[deployed] - truth[base_column]))
    episode_gains = [float(np.mean(v)) for v in gate_gain_by_episode.values()]
    count = len(episode_gains)
    gain = float(np.mean(episode_gains))
    se = float(np.std(episode_gains, ddof=1) / np.sqrt(count)) if count > 1 else np.inf
    return (
        gain - VALIDATION_GAIN_Z * se * SE_SCALE,
        float(np.mean(true_deployed)),
        float(np.mean(true_gate_view)),
    )


def run(trials: int, seed: int) -> dict:
    states = load()
    episodes = sorted({s["seed"] for s in states})
    rng = np.random.default_rng(seed)
    summary = {}
    for label, noise in (
        ("no_skill", float("inf")),
        ("noise_2.0", 2.0),
        ("noise_1.0", 1.0),
        ("noise_0.5", 0.5),
        ("noise_0.25", 0.25),
    ):
        for mode, select in (("single_epoch", False), ("early_stopped", True)):
            opened, gains, deployed, gate_view, consecutive = [], [], [], [], []
            history: list[float] = []
            for _ in range(trials):
                result = trial(states, episodes, rng, noise, select_epochs=select)
                if result is None:
                    continue
                statistic, true_deployed, true_gate = result
                opened.append(float(statistic > 0.0))
                gains.append(statistic)
                deployed.append(true_deployed)
                gate_view.append(true_gate)
                history.append(statistic)
                if len(history) >= GATE_UPDATES:
                    consecutive.append(
                        float(all(v > 0.0 for v in history[-GATE_UPDATES:]))
                    )
            summary[f"{label}|{mode}"] = {
                "refits": len(opened),
                "per_refit_open_rate": float(np.mean(opened)) if opened else float("nan"),
                "three_consecutive_open_rate": (
                    float(np.mean(consecutive)) if consecutive else float("nan")
                ),
                "mean_gate_statistic": float(np.mean(gains)) if gains else float("nan"),
                "true_gain_gate_view_branched_cells": (
                    float(np.mean(gate_view)) if gate_view else float("nan")
                ),
                "true_gain_deployed_all_feasible_cells": (
                    float(np.mean(deployed)) if deployed else float("nan")
                ),
            }
    return {
        "settings": {
            "trials": trials,
            "members": MEMBERS,
            "uncertainty_penalty": UNCERTAINTY_PENALTY,
            "validation_gain_z": VALIDATION_GAIN_Z,
            "gate_updates": GATE_UPDATES,
            "minimum_validation_states": MIN_VALIDATION_STATES,
            "replay_epochs_selected_over": REPLAY_EPOCHS,
            "labels": "tape 0, full horizon",
            "truth": "tapes 1 and 2, full horizon",
            "emulated_validation_episodes": float(3.0 / SE_SCALE**2),
            "episodes": len(episodes),
            "states": len(states),
        },
        "summary": summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--validation-episodes", type=int, default=3,
                        help="Emulated validation-episode count (scales the standard error)")
    parser.add_argument("--output", default="runs/live26_value_lcb_gate_check.json")
    args = parser.parse_args(argv)
    global SE_SCALE
    SE_SCALE = float(np.sqrt(3.0 / max(1.0, float(args.validation_episodes))))
    started = time.time()
    payload = run(args.trials, args.seed)
    payload["wall_seconds"] = time.time() - started
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
