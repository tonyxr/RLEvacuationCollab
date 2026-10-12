#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much of the return difference between candidate cells is identifiable?

The concern this answers: every installation adds the same equal-sized capacity
token, so every action raises absorbed flow and therefore improves evacuation
progress.  If that common component dominates, a learner cannot tell an optimal
siting from a sub-optimal one.

The v25 signal audit collected, at real decision states of the calibrated
testbed, the full-horizon return of **every feasible cell** under **three
independent common-random-number tapes**, with the return also recorded at
+10, +20 and +30 timesteps along the same trajectories.  That is exactly the
design needed for a variance decomposition, so no new simulation is required.

Model, per decision state ``s`` and truncation offset ``o``:

    V(s, c, k) = mu(s) + a(s, c) + b(s, k) + e(s, c, k)

* ``mu(s)``   -- the state level: scenario severity plus "a shelter now exists
                 at all".  This is the nuisance the concern is about.
* ``a(s, c)`` -- the cell effect: the siting signal a controller must learn.
* ``b(s, k)`` -- the tape effect: exogenous future noise shared by every cell
                 at that state.  Common random numbers remove this **exactly**,
                 which is why a paired branch set is worth its cost.
* ``e``       -- the pairing residual: the part of the future noise that does
                 not cancel between cells.  This, not ``b``, is the noise floor
                 of a one-tape within-state contrast.

A two-way ANOVA without interaction gives unbiased estimates of the three
variance components per state; they are pooled over states with equal weight.
The reported signal-to-noise ratio is ``sd(a) / sd(e)``: how far apart two
cells are, in units of the noise that survives pairing.

    python exposure_signal_decomposition.py
"""

from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np

STATES = "runs/nmcc_v25_signal_audit_states_20260921.pkl"
OFFSET_LABELS = ("+10", "+20", "+30", "full")


def load(path: str = STATES) -> tuple[list[dict], tuple[int, ...]]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return list(payload["states"]), tuple(payload["offsets"])


def components(matrix: np.ndarray) -> dict[str, float]:
    """Variance components of one state's (tapes, cells) return matrix.

    Two-way ANOVA without interaction.  Expected mean squares:
        MS_cells    = sigma_e^2 + K * sigma_a^2
        MS_tapes    = sigma_e^2 + C * sigma_b^2
        MS_residual = sigma_e^2
    """
    tapes, cells = matrix.shape
    if tapes < 2 or cells < 2:
        return {}
    grand = matrix.mean()
    cell_mean = matrix.mean(axis=0)
    tape_mean = matrix.mean(axis=1)
    ss_cells = tapes * float(np.sum((cell_mean - grand) ** 2))
    ss_tapes = cells * float(np.sum((tape_mean - grand) ** 2))
    ss_total = float(np.sum((matrix - grand) ** 2))
    ss_residual = ss_total - ss_cells - ss_tapes
    df_residual = (tapes - 1) * (cells - 1)
    ms_cells = ss_cells / (cells - 1)
    ms_tapes = ss_tapes / (tapes - 1)
    ms_residual = ss_residual / df_residual if df_residual > 0 else float("nan")
    return {
        "level": grand,
        "cells": cells,
        "tapes": tapes,
        # Unbiased, clipped at zero: a negative estimate means the component is
        # indistinguishable from zero at this sample size.
        "var_cell": max(0.0, (ms_cells - ms_residual) / tapes),
        "var_tape": max(0.0, (ms_tapes - ms_residual) / cells),
        "var_residual": max(0.0, ms_residual),
        "observed_cell_spread": float(np.var(cell_mean)),
    }


def expected_rank_agreement(var_cell: float, var_residual: float, cells: float,
                            tapes: int, draws: int, rng) -> float:
    """Spearman between a ``tapes``-tape contrast and the true cell effect.

    A Gaussian surrogate at the measured variance components: it isolates how
    much of the ranking a finite tape budget can recover, without the confound
    of a particular model's inductive bias.
    """
    if var_cell <= 0.0 or not np.isfinite(var_residual):
        return float("nan")
    size = max(2, int(round(cells)))
    scores = []
    for _ in range(draws):
        truth = rng.normal(0.0, math.sqrt(var_cell), size=size)
        noise = rng.normal(0.0, math.sqrt(max(var_residual, 0.0) / tapes), size=size)
        a = np.argsort(np.argsort(truth)).astype(float)
        b = np.argsort(np.argsort(truth + noise)).astype(float)
        a -= a.mean()
        b -= b.mean()
        denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
        if denominator > 0:
            scores.append(float(np.dot(a, b) / denominator))
    return float(np.mean(scores)) if scores else float("nan")


def analyse(states, offsets, *, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    report = {}
    per_offset = {}
    for index, label in enumerate(OFFSET_LABELS):
        rows = []
        for state in states:
            matrix = state["values"][:, :, index]
            estimate = components(matrix)
            if estimate:
                estimate["decision_index"] = int(state["decision_index"])
                rows.append(estimate)
        if not rows:
            continue
        levels = np.asarray([r["level"] for r in rows])
        var_cell = float(np.mean([r["var_cell"] for r in rows]))
        var_tape = float(np.mean([r["var_tape"] for r in rows]))
        var_residual = float(np.mean([r["var_residual"] for r in rows]))
        cells = float(np.mean([r["cells"] for r in rows]))
        entry = {
            "states": len(rows),
            "mean_feasible_cells": cells,
            "between_state_level_sd": float(np.std(levels, ddof=1)),
            "cell_effect_sd": math.sqrt(var_cell),
            "tape_effect_sd": math.sqrt(var_tape),
            "pairing_residual_sd": math.sqrt(var_residual),
            "snr_cell_over_residual": (
                math.sqrt(var_cell / var_residual) if var_residual > 0 else float("inf")
            ),
            "cell_effect_over_state_level": (
                math.sqrt(var_cell) / float(np.std(levels, ddof=1))
                if np.std(levels, ddof=1) > 0 else float("nan")
            ),
            "unpaired_snr_cell_over_tape_plus_residual": (
                math.sqrt(var_cell / (var_tape + var_residual))
                if (var_tape + var_residual) > 0 else float("inf")
            ),
            "expected_rank_agreement": {
                f"{tapes}_tape": expected_rank_agreement(
                    var_cell, var_residual, cells, tapes, draws, rng
                )
                for tapes in (1, 2, 3, 5, 10)
            },
        }
        # Same decomposition restricted to the first decision, where the v25
        # review located nearly all of the available regret.
        first = [r for r in rows if r["decision_index"] == 0]
        if len(first) >= 2:
            v_cell = float(np.mean([r["var_cell"] for r in first]))
            v_res = float(np.mean([r["var_residual"] for r in first]))
            entry["first_decision"] = {
                "states": len(first),
                "cell_effect_sd": math.sqrt(v_cell),
                "pairing_residual_sd": math.sqrt(v_res),
                "snr_cell_over_residual": (
                    math.sqrt(v_cell / v_res) if v_res > 0 else float("inf")
                ),
            }
        per_offset[label] = entry
    report["by_offset"] = per_offset
    full = per_offset.get("full", {})
    if full:
        report["accumulation"] = {
            label: {
                "share_of_full_cell_effect_sd": (
                    per_offset[label]["cell_effect_sd"] / full["cell_effect_sd"]
                    if full["cell_effect_sd"] > 0 else float("nan")
                ),
                "share_of_full_residual_sd": (
                    per_offset[label]["pairing_residual_sd"] / full["pairing_residual_sd"]
                    if full["pairing_residual_sd"] > 0 else float("nan")
                ),
                "snr_relative_to_full": (
                    per_offset[label]["snr_cell_over_residual"]
                    / full["snr_cell_over_residual"]
                    if full["snr_cell_over_residual"] not in (0.0, float("inf"))
                    else float("nan")
                ),
            }
            for label in OFFSET_LABELS
            if label in per_offset
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default=STATES)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", default="runs/exposure_signal_decomposition_20260927.json")
    args = parser.parse_args(argv)
    states, offsets = load(args.states)
    report = analyse(states, offsets, draws=args.draws, seed=args.seed)
    report["source"] = {"states": len(states), "offsets": list(offsets), "path": args.states}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"{'offset':>7s} {'cells':>6s} {'state level':>12s} {'cell effect':>12s} "
          f"{'tape effect':>12s} {'pair resid':>11s} {'SNR':>6s} {'unpaired SNR':>13s}")
    for label, entry in report["by_offset"].items():
        print(f"{label:>7s} {entry['mean_feasible_cells']:6.1f} "
              f"{entry['between_state_level_sd']:12.4f} {entry['cell_effect_sd']:12.4f} "
              f"{entry['tape_effect_sd']:12.4f} {entry['pairing_residual_sd']:11.4f} "
              f"{entry['snr_cell_over_residual']:6.2f} "
              f"{entry['unpaired_snr_cell_over_tape_plus_residual']:13.2f}")
    print()
    print("expected within-state Spearman against the true cell effect")
    for label, entry in report["by_offset"].items():
        row = entry["expected_rank_agreement"]
        print(f"{label:>7s} " + "  ".join(
            f"{tapes}: {row[tapes]:.2f}" for tapes in row
        ))
    if "accumulation" in report:
        print()
        print("share of the full-horizon magnitude accumulated by each offset")
        for label, row in report["accumulation"].items():
            print(f"{label:>7s} cell effect {row['share_of_full_cell_effect_sd']:.2f}  "
                  f"pairing residual {row['share_of_full_residual_sd']:.2f}  "
                  f"SNR {row['snr_relative_to_full']:.2f}")
    first = {k: v.get("first_decision") for k, v in report["by_offset"].items()}
    print()
    print("first decision only")
    for label, row in first.items():
        if row:
            print(f"{label:>7s} states {row['states']:3d}  cell effect {row['cell_effect_sd']:.4f}  "
                  f"pairing residual {row['pairing_residual_sd']:.4f}  SNR {row['snr_cell_over_residual']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
