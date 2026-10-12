#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which one-tape label ranks candidate cells best, at equal simulation cost?

``exposure_signal_decomposition.py`` shows where the difficulty is: within a
decision state the siting signal accumulates early (56% of its full-horizon
magnitude by +10 timesteps, 84% by +20) while the noise that survives
common-random-number pairing accumulates late (3% by +10, 47% by +20).  The
terminal return difference therefore measures the right target with the worst
signal-to-noise ratio of any window on the same trajectory.

Because the audit recorded the cumulative return at +10, +20, +30 and the full
horizon *along the same branch trajectories*, the early windows are free: they
cost no extra simulation.  That makes them usable as control variates for the
full-horizon contrast.

Estimators compared, all from **one** tape unless stated:

* ``full``            -- the terminal within-state contrast (v26 uses this at
                         the first two decisions).
* ``+10 / +20 / +30`` -- a truncated contrast (v26 uses about +20 later).
* ``blend``           -- the least-squares combination of the four cumulative
                         contrasts that best predicts the full-horizon cell
                         effect, with weights fitted on **other** states.
* ``full, 2 tapes``   -- twice the simulation cost, as an upper reference.

Truth is one held-out tape; every estimator sees only the other two, so nothing
is scored against its own noise.  A one-tape truth depresses every rank
agreement and inflates every regret by the same amount, so compare estimators
with each other rather than reading the absolute levels; the leakage-free
absolute levels against the *true* cell effect are in
``exposure_signal_decomposition.py``.

    python exposure_label_estimators.py
"""

from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np

STATES = "runs/nmcc_v25_signal_audit_states_20260921.pkl"
OFFSETS = ("+10", "+20", "+30", "full")


def load(path: str = STATES) -> list[dict]:
    with open(path, "rb") as handle:
        return list(pickle.load(handle)["states"])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    return float("nan") if denominator <= 0 else float(np.dot(ra, rb) / denominator)


def centered(values: np.ndarray) -> np.ndarray:
    """Within-state contrast: subtract the cell mean, which is what pairing
    identifies and what any level-invariant learner sees."""
    return values - values.mean()


def blend_weights(states, target_tapes, estimator_tape, exclude) -> np.ndarray:
    """Least-squares weights predicting the held-out cell effect from the four
    cumulative contrasts, pooled over every state except ``exclude``."""
    rows, targets = [], []
    for index, state in enumerate(states):
        if index == exclude:
            continue
        values = state["values"]
        if values.shape[1] < 3:
            continue
        design = np.stack(
            [centered(values[estimator_tape, :, o]) for o in range(len(OFFSETS))],
            axis=1,
        )
        truth = centered(values[list(target_tapes), :, len(OFFSETS) - 1].mean(axis=0))
        rows.append(design)
        targets.append(truth)
    if not rows:
        return np.zeros(len(OFFSETS))
    design = np.concatenate(rows, axis=0)
    truth = np.concatenate(targets, axis=0)
    solution, *_ = np.linalg.lstsq(design, truth, rcond=None)
    return solution


def evaluate(states, *, seed: int) -> dict:
    tapes = states[0]["values"].shape[0]
    records: dict[str, list[dict]] = {}
    weight_log = []
    for index, state in enumerate(states):
        values = state["values"]
        cells = values.shape[1]
        if cells < 3:
            continue
        for truth_tape in range(tapes):
            # Truth is one held-out tape; every estimator sees only the other
            # two, so no estimator is scored against its own noise.  A
            # one-tape truth inflates every regret equally and depresses every
            # rank agreement equally, so the comparison between estimators
            # stays valid while the absolute levels do not.
            target_tapes = [k for k in range(tapes) if k != truth_tape]
            estimator_tape = target_tapes[0]
            truth = centered(values[truth_tape, :, len(OFFSETS) - 1])
            best = int(np.argmax(truth))
            candidates = {
                label: centered(values[estimator_tape, :, offset])
                for offset, label in enumerate(OFFSETS)
            }
            weights = blend_weights(states, (truth_tape,), estimator_tape, index)
            weight_log.append(weights.tolist())
            design = np.stack([candidates[label] for label in OFFSETS], axis=1)
            candidates["blend"] = design @ weights
            # Two tapes disjoint from the truth tape: the only fair
            # extra-tape reference available with three collected tapes.
            candidates["full, 2 tapes"] = centered(
                values[target_tapes, :, len(OFFSETS) - 1].mean(axis=0)
            )
            for label, estimate in candidates.items():
                chosen = int(np.argmax(estimate))
                records.setdefault(label, []).append(
                    {
                        "spearman": spearman(estimate, truth),
                        "top1": float(chosen == best),
                        "regret": float(truth[best] - truth[chosen]),
                        "decision_index": int(state["decision_index"]),
                        "cells": cells,
                    }
                )
    summary = {}
    for label, rows in records.items():
        spearmans = [r["spearman"] for r in rows if np.isfinite(r["spearman"])]
        first = [r for r in rows if r["decision_index"] == 0]
        summary[label] = {
            "evaluations": len(rows),
            "mean_spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
            "top1_rate": float(np.mean([r["top1"] for r in rows])),
            "mean_regret": float(np.mean([r["regret"] for r in rows])),
            "first_decision_top1_rate": (
                float(np.mean([r["top1"] for r in first])) if first else float("nan")
            ),
            "first_decision_mean_regret": (
                float(np.mean([r["regret"] for r in first])) if first else float("nan")
            ),
            "simulation_cost_relative": 2.0 if label == "full, 2 tapes" else 1.0,
        }
    summary["_blend_weights_mean"] = np.mean(np.asarray(weight_log), axis=0).tolist()
    summary["_blend_weight_order"] = list(OFFSETS)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default=STATES)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", default="runs/exposure_label_estimators_20260927.json")
    args = parser.parse_args(argv)
    states = load(args.states)
    summary = evaluate(states, seed=args.seed)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"states": len(states), "summary": summary}, handle, indent=2)
    order = ["+10", "+20", "+30", "full", "blend", "full, 2 tapes"]
    print(f"{'label':>14s} {'cost':>5s} {'spearman':>9s} {'top-1':>7s} {'regret':>8s} "
          f"{'top-1 @t=1':>11s} {'regret @t=1':>12s}")
    for label in order:
        row = summary[label]
        print(f"{label:>14s} {row['simulation_cost_relative']:5.1f} "
              f"{row['mean_spearman']:9.3f} {row['top1_rate']:7.3f} {row['mean_regret']:8.4f} "
              f"{row['first_decision_top1_rate']:11.3f} {row['first_decision_mean_regret']:12.4f}")
    print()
    print("mean blend weights on the cumulative contrasts "
          f"{summary['_blend_weight_order']}: "
          + ", ".join(f"{w:+.2f}" for w in summary["_blend_weights_mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
