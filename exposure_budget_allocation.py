#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""At a fixed simulation budget, how should a branch spend it: horizon or tapes?

``nmccPiBranchHorizon`` truncates every branch (the configuration default is 20
of the 60 timesteps; the v26 curricula set 0, the full horizon) and
``nmccPiTapes`` is 1 in every curriculum.  Those two knobs trade off against
each other at constant cost: one branch costs about ``tapes * window``
simulator timesteps, so 1 tape to the full horizon and 3 tapes to +20 cost the
same.

The v25 signal audit recorded, at every decision state, the cumulative return
of every feasible cell at +10, +20, +30 and the horizon on 3 independent
common-random-number tapes.  That is enough to answer the question in closed
form, with no new simulation, because the quantity a controller must recover is
the **full-horizon** cell effect ``a_full(s, c)`` whatever window the label
uses:

* a ``k``-tape window-``o`` contrast estimates ``a_o + e_o / sqrt(k)``;
* truncation is a bias, measured by ``rho_signal = corr(a_o, a_full)``;
* tapes are variance, measured by ``var(e_o) / k``.

So the correlation of the label with the target a controller is graded on is

    corr = rho_signal / sqrt(1 + (var(e_o) / k) / var(a_o))

and both ingredients are identified from disjoint tape sets: the estimator uses
one tape, the ``a_full`` proxy uses the other two, so their residuals are
independent and the cross-covariance is unbiased for ``cov(a_o, a_full)``.  The
variances are the two-way ANOVA components, corrected for the residual the
proxy itself carries.

Reported per (window, tapes): the implied correlation with ``a_full``, the
expected Spearman and top-1 rate a controller ranking on that label would
achieve, and the cost in simulator timesteps per branch.

    python exposure_budget_allocation.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle

import numpy as np

STATES = "runs/nmcc_v25_signal_audit_states_20260921.pkl"
OFFSET_LABELS = ("+10", "+20", "+30", "full")
HORIZON = 60
# Timesteps a branch actually simulates at each window, from the first decision
# at t = 1.  The full horizon is 59 remaining timesteps.
WINDOW_STEPS = {"+10": 10, "+20": 20, "+30": 30, "full": HORIZON - 1}


def load(path: str = STATES) -> list[dict]:
    with open(path, "rb") as handle:
        return list(pickle.load(handle)["states"])


def anova(matrix: np.ndarray) -> tuple[float, float]:
    """(cell-effect variance, pairing-residual variance) for (tapes, cells)."""
    tapes, cells = matrix.shape
    if tapes < 2 or cells < 2:
        return float("nan"), float("nan")
    grand = matrix.mean()
    ss_cells = tapes * float(np.sum((matrix.mean(axis=0) - grand) ** 2))
    ss_tapes = cells * float(np.sum((matrix.mean(axis=1) - grand) ** 2))
    ss_total = float(np.sum((matrix - grand) ** 2))
    ms_cells = ss_cells / (cells - 1)
    ms_residual = (ss_total - ss_cells - ss_tapes) / ((tapes - 1) * (cells - 1))
    return max(0.0, (ms_cells - ms_residual) / tapes), max(0.0, ms_residual)


def centered(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


def signal_covariance(states, window: int, full: int) -> dict[str, float]:
    """Pooled cov(a_window, a_full) using tape sets with disjoint residuals.

    For every ordered split of the three tapes into one estimator tape and two
    proxy tapes, the estimator's residual is independent of the proxy's, so the
    empirical cross-product estimates the covariance of the signals alone.
    Pooled over splits and over states with equal weight.
    """
    products, window_var, full_var = [], [], []
    for state in states:
        values = np.asarray(state["values"], dtype=np.float64)
        tapes, cells, _ = values.shape
        if tapes < 3 or cells < 3:
            continue
        cross = []
        for estimator_tape in range(tapes):
            others = [k for k in range(tapes) if k != estimator_tape]
            a = centered(values[estimator_tape, :, window])
            b = centered(values[others, :, full].mean(axis=0))
            cross.append(float(np.dot(a, b) / (cells - 1)))
        products.append(float(np.mean(cross)))
        var_window, _ = anova(values[:, :, window])
        var_full, _ = anova(values[:, :, full])
        window_var.append(var_window)
        full_var.append(var_full)
    if not products:
        return {}
    covariance = float(np.mean(products))
    var_window = float(np.mean(window_var))
    var_full = float(np.mean(full_var))
    denominator = math.sqrt(var_window * var_full)
    return {
        "covariance": covariance,
        "var_window": var_window,
        "var_full": var_full,
        "signal_correlation": covariance / denominator if denominator > 0 else float("nan"),
    }


def rank_statistics(correlation: float, cells: int, draws: int, rng) -> dict[str, float]:
    """Expected Spearman and top-1 rate of a label with this correlation.

    A Gaussian surrogate at the measured correlation: it converts an estimator's
    correlation with the target into the two quantities a siting controller is
    actually judged on, without assuming a particular model class.
    """
    if not np.isfinite(correlation):
        return {"spearman": float("nan"), "top1": float("nan"), "regret_fraction": float("nan")}
    correlation = float(np.clip(correlation, -1.0, 1.0))
    truth = rng.normal(size=(draws, cells))
    independent = rng.normal(size=(draws, cells))
    label = correlation * truth + math.sqrt(max(0.0, 1.0 - correlation**2)) * independent
    ranks_truth = np.argsort(np.argsort(truth, axis=1), axis=1).astype(float)
    ranks_label = np.argsort(np.argsort(label, axis=1), axis=1).astype(float)
    ranks_truth -= ranks_truth.mean(axis=1, keepdims=True)
    ranks_label -= ranks_label.mean(axis=1, keepdims=True)
    numerator = np.sum(ranks_truth * ranks_label, axis=1)
    denominator = np.sqrt(
        np.sum(ranks_truth**2, axis=1) * np.sum(ranks_label**2, axis=1)
    )
    spearman = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), np.nan)
    best = np.argmax(truth, axis=1)
    chosen = np.argmax(label, axis=1)
    rows = np.arange(draws)
    # Regret in units of the cell-effect standard deviation, so it is
    # comparable across windows whose signal magnitudes differ.
    regret = truth[rows, best] - truth[rows, chosen]
    return {
        "spearman": float(np.nanmean(spearman)),
        "top1": float(np.mean(best == chosen)),
        "regret_in_cell_effect_sd": float(np.mean(regret)),
    }


def prior_correlation(states) -> dict[str, float]:
    """corr(prior score, full-horizon cell effect), and the prior's recall.

    A branch allocation that scores only the prior's top ``n`` cells cannot pick
    a better cell than the best one inside that subset, whatever the label's
    quality.  The recall curve is therefore a ceiling in its own right, and it is
    measured, not assumed.
    """
    full = len(OFFSET_LABELS) - 1
    products, prior_var, effect_var, ranks, sizes = [], [], [], [], []
    for state in states:
        values = np.asarray(state["values"], dtype=np.float64)
        feasible = np.asarray(state["actions"])
        cells = feasible.size
        if cells < 3:
            continue
        prior = centered(np.asarray(state["scores"], dtype=np.float64)[feasible])
        effect = centered(values[:, :, full].mean(axis=0))
        products.append(float(np.dot(prior, effect) / (cells - 1)))
        prior_var.append(float(np.dot(prior, prior) / (cells - 1)))
        var_cell, _ = anova(values[:, :, full])
        effect_var.append(var_cell)
        order = np.argsort(-np.asarray(state["scores"], dtype=np.float64)[feasible])
        best = int(np.argmax(values[:, :, full].mean(axis=0)))
        ranks.append(int(np.flatnonzero(order == best)[0]) + 1)
        sizes.append(cells)
    denominator = math.sqrt(float(np.mean(prior_var)) * float(np.mean(effect_var)))
    return {
        "correlation": float(np.mean(products)) / denominator if denominator > 0 else float("nan"),
        "mean_rank_of_best_cell": float(np.mean(ranks)),
        "mean_feasible_cells": float(np.mean(sizes)),
        "recall_at": {
            str(n): float(np.mean([rank <= n for rank in ranks]))
            for n in range(2, int(max(sizes)) + 1)
        },
    }


def subset_statistics(label_correlation: float, prior_corr: float, subset: int,
                      cells: int, draws: int, rng) -> dict[str, float]:
    """Top-1 rate and regret when only the prior's top ``subset`` cells are branched.

    Three jointly Gaussian variables per cell: the full-horizon cell effect
    ``a``, the prior score ``p`` with ``corr(p, a) = prior_corr``, and the label
    ``l`` with ``corr(l, a) = label_correlation``.  The label sees the
    simulator, not the prior, so its only dependence on ``p`` runs through
    ``a``; that gives ``corr(p, l) = prior_corr * label_correlation`` and makes
    the triple consistent without a further assumption.
    """
    if not np.isfinite(label_correlation):
        return {"top1": float("nan"), "regret_in_cell_effect_sd": float("nan")}
    label_correlation = float(np.clip(label_correlation, -1.0, 1.0))
    prior_corr = float(np.clip(prior_corr, -1.0, 1.0))
    effect = rng.normal(size=(draws, cells))
    prior = prior_corr * effect + math.sqrt(max(0.0, 1.0 - prior_corr**2)) * rng.normal(
        size=(draws, cells)
    )
    labels = label_correlation * effect + math.sqrt(
        max(0.0, 1.0 - label_correlation**2)
    ) * rng.normal(size=(draws, cells))
    subset = int(min(max(1, subset), cells))
    keep = np.argsort(-prior, axis=1)[:, :subset]
    rows = np.arange(draws)[:, None]
    masked = np.full(labels.shape, -np.inf)
    masked[rows, keep] = labels[rows, keep]
    chosen = np.argmax(masked, axis=1)
    best = np.argmax(effect, axis=1)
    flat = np.arange(draws)
    return {
        "top1": float(np.mean(best == chosen)),
        "regret_in_cell_effect_sd": float(
            np.mean(effect[flat, best] - effect[flat, chosen])
        ),
    }


def analyse(states, *, tape_counts, draws: int, seed: int, first_only: bool) -> dict:
    if first_only:
        states = [s for s in states if int(s["decision_index"]) == 0]
    rng = np.random.default_rng(seed)
    full = len(OFFSET_LABELS) - 1
    cells = int(round(float(np.mean([s["values"].shape[1] for s in states]))))
    report = {
        "states": len(states),
        "mean_feasible_cells": cells,
        "first_decision_only": bool(first_only),
        "windows": {},
    }
    for index, label in enumerate(OFFSET_LABELS):
        signal = signal_covariance(states, index, full)
        if not signal:
            continue
        _, var_residual = (float("nan"), float("nan"))
        residuals = []
        for state in states:
            values = np.asarray(state["values"], dtype=np.float64)
            if values.shape[0] < 2 or values.shape[1] < 2:
                continue
            _, var_res = anova(values[:, :, index])
            residuals.append(var_res)
        var_residual = float(np.mean(residuals)) if residuals else float("nan")
        entry = {
            "steps_per_branch": WINDOW_STEPS[label],
            "cell_effect_sd": math.sqrt(signal["var_window"]),
            "pairing_residual_sd": math.sqrt(var_residual),
            # How much of the full-horizon ordering this window can express at
            # all: the ceiling no number of tapes can pass.
            "truncation_signal_correlation": signal["signal_correlation"],
            "by_tapes": {},
        }
        for tapes in tape_counts:
            attenuation = (
                signal["signal_correlation"]
                / math.sqrt(1.0 + (var_residual / tapes) / signal["var_window"])
                if signal["var_window"] > 0 else float("nan")
            )
            statistics = rank_statistics(attenuation, cells, draws, rng)
            entry["by_tapes"][str(tapes)] = {
                "correlation_with_full_horizon_cell_effect": attenuation,
                "cost_steps": WINDOW_STEPS[label] * tapes,
                **statistics,
            }
        report["windows"][label] = entry
    # The frontier: at each achievable cost, which (window, tapes) pair
    # maximizes agreement with the full-horizon cell effect?
    options = []
    for label, entry in report["windows"].items():
        for tapes, row in entry["by_tapes"].items():
            options.append(
                {
                    "window": label,
                    "tapes": int(tapes),
                    "cost_steps": row["cost_steps"],
                    "spearman": row["spearman"],
                    "top1": row["top1"],
                    "correlation": row["correlation_with_full_horizon_cell_effect"],
                }
            )
    options.sort(key=lambda row: (row["cost_steps"], -row["spearman"]))
    frontier, best_so_far = [], -np.inf
    for row in options:
        if row["spearman"] > best_so_far:
            frontier.append(row)
            best_so_far = row["spearman"]
    report["cost_frontier"] = frontier
    report["equal_cost_pairs"] = [
        [a, b]
        for a, b in itertools.combinations(options, 2)
        if a["cost_steps"] == b["cost_steps"] and a["window"] != b["window"]
    ]

    # Third axis: how many cells to branch.  The total cost of one decision is
    # cells * tapes * window steps, so branching fewer cells buys tapes or
    # horizon.  The prior's recall bounds what a subset can ever reach.
    prior = prior_correlation(states)
    report["prior"] = prior
    allocations = []
    for label, entry in report["windows"].items():
        for tapes_text, row in entry["by_tapes"].items():
            tapes = int(tapes_text)
            for subset in range(2, cells + 1):
                statistics = subset_statistics(
                    row["correlation_with_full_horizon_cell_effect"],
                    prior["correlation"],
                    subset,
                    cells,
                    draws,
                    rng,
                )
                allocations.append(
                    {
                        "window": label,
                        "tapes": tapes,
                        "cells_branched": subset,
                        "cost_steps": entry["steps_per_branch"] * tapes * subset,
                        **statistics,
                    }
                )
    allocations.sort(key=lambda row: (row["cost_steps"], row["regret_in_cell_effect_sd"]))
    best, frontier = np.inf, []
    for row in allocations:
        if row["regret_in_cell_effect_sd"] < best:
            frontier.append(row)
            best = row["regret_in_cell_effect_sd"]
    report["allocation_frontier"] = frontier
    report["allocations"] = allocations
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default=STATES)
    parser.add_argument("--tapes", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--first-decision-only", action="store_true")
    parser.add_argument("--output", default="runs/exposure_budget_allocation_20260927.json")
    args = parser.parse_args(argv)
    states = load(args.states)
    report = analyse(
        states,
        tape_counts=args.tapes,
        draws=args.draws,
        seed=args.seed,
        first_only=args.first_decision_only,
    )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"{report['states']} decision states, {report['mean_feasible_cells']} feasible "
          f"cells on average"
          + (" (first decision only)" if report["first_decision_only"] else ""))
    print()
    print("truncation ceiling: correlation of a noiseless window contrast with the "
          "full-horizon cell effect")
    for label, entry in report["windows"].items():
        print(f"{label:>7s} {entry['steps_per_branch']:3d} steps/branch  "
              f"signal correlation {entry['truncation_signal_correlation']:.3f}  "
              f"cell effect {entry['cell_effect_sd']:.4f}  "
              f"pairing residual {entry['pairing_residual_sd']:.4f}")
    print()
    print("expected Spearman against the full-horizon cell effect "
          "(cost in simulator timesteps per branch)")
    tape_counts = sorted({int(t) for e in report["windows"].values() for t in e["by_tapes"]})
    print(f"{'window':>7s} " + " ".join(f"{t:>14d}" for t in tape_counts))
    for label, entry in report["windows"].items():
        cells = []
        for tapes in tape_counts:
            row = entry["by_tapes"].get(str(tapes))
            cells.append(
                f"{row['spearman']:.3f} @{row['cost_steps']:4d}" if row else " " * 14
            )
        print(f"{label:>7s} " + " ".join(f"{c:>14s}" for c in cells))
    print()
    print("cost frontier (each entry is the best label at its cost or below)")
    for row in report["cost_frontier"]:
        print(f"  {row['cost_steps']:4d} steps  {row['window']:>5s} x {row['tapes']} tapes  "
              f"spearman {row['spearman']:.3f}  top-1 {row['top1']:.3f}")
    print()
    print("equal-cost comparisons")
    seen = set()
    for a, b in report["equal_cost_pairs"]:
        key = (a["cost_steps"], a["window"], b["window"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  {a['cost_steps']:4d} steps: {a['window']:>5s} x {a['tapes']} "
              f"-> {a['spearman']:.3f}   vs   {b['window']:>5s} x {b['tapes']} "
              f"-> {b['spearman']:.3f}")
    print()
    prior = report["prior"]
    print(f"prior score: correlation with the full-horizon cell effect "
          f"{prior['correlation']:.3f}; the true best cell sits at prior rank "
          f"{prior['mean_rank_of_best_cell']:.1f} of "
          f"{prior['mean_feasible_cells']:.1f} on average")
    recall = prior["recall_at"]
    print("  recall of the best cell in the prior's top n: " + "  ".join(
        f"{n}:{recall[n]:.2f}" for n in ("2", "4", "6", "8", "12") if n in recall
    ))
    print()
    print("allocation frontier: lowest regret at each total cost per decision "
          "(cells x tapes x steps)")
    for row in report["allocation_frontier"]:
        print(f"  {row['cost_steps']:6d} steps  {row['cells_branched']:2d} cells x "
              f"{row['tapes']} tapes x {row['window']:>5s}  "
              f"top-1 {row['top1']:.3f}  regret {row['regret_in_cell_effect_sd']:.3f} sd")
    print()
    print("what the v26 settings buy, per decision")
    for cells_branched, tapes, window in (
        (int(round(prior["mean_feasible_cells"])), 1, "full"),
        (int(round(prior["mean_feasible_cells"])), 1, "+20"),
        (8, 1, "full"),
        (6, 1, "+20"),
        (6, 3, "full"),
        (4, 5, "full"),
    ):
        match = [
            row for row in report["allocations"]
            if row["window"] == window and row["tapes"] == tapes
            and row["cells_branched"] == min(cells_branched, report["mean_feasible_cells"])
        ]
        if match:
            row = match[0]
            print(f"  {row['cells_branched']:2d} cells x {row['tapes']} tapes x "
                  f"{row['window']:>5s} = {row['cost_steps']:5d} steps  "
                  f"top-1 {row['top1']:.3f}  regret {row['regret_in_cell_effect_sd']:.3f} sd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
