#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which reward component carries the siting signal at the first decision?

Every dynamic installation adds the same equal-capacity token, so every action
raises absorbed flow and therefore improves evacuation progress.  The question
this answers is where the *difference* between a good and a bad siting shows
up: in the safe-completion count, in the casualty count, in active person-time,
or in hazard-weighted exposure person-time.

If the differential lives in one component, then scoring candidates on that
component -- with the reward's own known weights -- is a variance reduction at
zero extra simulation cost, because the noisy components stop contributing
noise to the ranking.  If instead the differential is spread across components
in the same proportions as the noise, nothing is gained and the scalar return
is already the right target.

Collection branches **every feasible cell at the first decision** of each
episode, to the **full horizon**, on **three independent common-random-number
tapes**, recording the cumulative four-component reward vector at +10, +20,
+30 and the horizon along the same trajectories.  The scenario, masks and
behavior match ``nmcc_v25_signal_audit.py`` so the results compose with the
existing audit.

    python exposure_component_signal.py collect --episodes 10 --workers 2
    python exposure_component_signal.py analyze
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import time

import numpy as np

import CounterfactualBranch as CB
import headroom_lib as H
import NMCCPolicyImprovement as PI
from DecisionInterface import (
    PolicyDecision,
    RegionalObservationBuilder,
    RegionalShelterExecutor,
)
from nmcc_pi_reference import BUDGET, HORIZON, INTERVAL, MAP, features
from RewardProcessor import REWARD_COMPONENT_NAMES

OFFSETS = (10, 20, 30)
OFFSET_LABELS = ("+10", "+20", "+30", "full")
POPULATION = 800
SAVING_COLUMN = 1 + 7
# Seeds 501.. reproduce the first-decision states of
# runs/nmcc_v25_signal_audit_states_20260921.pkl exactly, so the summed
# component vector must equal that audit's stored scalar value cell by cell.
FIRST_SEED = 501
V25_AUDIT_STATES = "runs/nmcc_v25_signal_audit_states_20260921.pkl"
V25_MASK = dict(
    requireCandidateOperationalBenefit=True,
    minimumCandidateReroutableFraction=0.0,
    minimumCandidateRouteTimeSaving=0.0,
    minimumCandidateHazardSafetyMargin=0.05,
    maximumShelterForecastDanger=0.6,
)


def build(seed: int):
    core = H.build_core(seed, **MAP)
    for key, value in V25_MASK.items():
        setattr(core, key, value)
    builder = RegionalObservationBuilder(
        core, initial_population=POPULATION, horizon=HORIZON, maximum_deployments=BUDGET
    )
    return core, builder, RegionalShelterExecutor(core)


def prior_scores(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The v25 initial controller's scores, identical to the signal audit's."""
    saving = np.clip(grid[:, SAVING_COLUMN], 0.0, None)
    top = saving[mask].max() if mask.any() else 0.0
    return saving / top if top > 0 else np.zeros_like(saving)


def step_components(core, clock, reward_model) -> np.ndarray:
    """Advance one timestep; return that step's signed four-component reward.

    Identical dynamics and bookkeeping to ``NMCCPolicyImprovement.step_and_score``
    -- which returns only the scalar -- but keeping the four signed branches in
    float64 rather than the float32 of ``RewardBreakdown.component_vector``, so
    the sum over the horizon reproduces the scalar exactly.
    """
    clock.t += 1
    CB.advance_one_timestep(core)
    current = CB.outcome_snapshot(core)
    if clock.previous is None:
        clock.previous = current
    breakdown = reward_model.evaluate(
        before=clock.previous,
        after=current,
        active_person_time=float(current.active_population),
        hazard_exposure_person_time=float(current.hazard_exposure_mass),
        initial_population=int(clock.population),
        horizon=int(clock.horizon),
    )
    clock.accumulated += float(breakdown.total)
    clock.active_person_time += float(current.active_population)
    clock.hazard_exposure_person_time += float(current.hazard_exposure_mass)
    clock.previous = current
    return np.asarray(
        (
            breakdown.safe_completion_reward,
            breakdown.casualty_penalty,
            breakdown.evacuation_time_penalty,
            breakdown.hazard_exposure_penalty,
        ),
        dtype=np.float64,
    )


def continue_components(core, builder, executor, clock, reward_model) -> np.ndarray:
    """Base-policy continuation to the horizon.

    Returns the cumulative component vector at each offset and at the horizon,
    shape ``(len(OFFSETS) + 1, 4)``, all measured on one trajectory.
    """
    start_t = clock.t
    running = np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float64)
    marks: dict[int, np.ndarray] = {}
    while clock.t < clock.horizon:
        running = running + step_components(core, clock, reward_model)
        elapsed = clock.t - start_t
        if elapsed in OFFSETS:
            marks[elapsed] = running.copy()
        if clock.is_decision(clock.t):
            observation = builder.build(
                decision_index=clock.deployed,
                simulation_time=clock.t,
                remaining_deployments=clock.budget - clock.deployed,
            )
            if observation.has_feasible_action:
                executor.execute(
                    observation,
                    PolicyDecision(
                        action_index=int(PI.route_saving_policy(observation)),
                        strategy="component_audit_base",
                    ),
                )
                clock.deployed += 1
    return np.stack([marks.get(o, running) for o in OFFSETS] + [running])


def collect_episode(job) -> dict | None:
    seed, tapes, max_cells = job
    core, builder, executor = build(seed)
    reward_model = PI.RewardProcessor()
    clock = PI.EpisodeClock(
        horizon=HORIZON, interval=INTERVAL, budget=BUDGET, population=POPULATION
    )
    step_components(core, clock, reward_model)  # advance to the first decision, t = 1
    observation = builder.build(
        decision_index=0, simulation_time=clock.t, remaining_deployments=BUDGET
    )
    if not observation.has_feasible_action:
        return None
    mask = np.asarray(observation.action_mask, dtype=bool)
    feasible = np.flatnonzero(mask)
    if max_cells and feasible.size > max_cells:
        feasible = feasible[:max_cells]
    grid = features(observation)
    values = np.zeros((tapes, feasible.size, len(OFFSETS) + 1, len(REWARD_COMPONENT_NAMES)))
    wait = np.zeros((tapes, len(OFFSETS) + 1, len(REWARD_COMPONENT_NAMES)))
    snapshot = CB.capture(core, label="component-audit")
    try:
        for tape in range(tapes):
            seed_for_tape = PI.tape_seed(seed, 0, tape)
            for column, action in enumerate(feasible):
                CB.restore(core, snapshot)
                PI.reseed_exogenous(core, seed_for_tape)
                branch = clock.copy()
                executor.execute(
                    observation,
                    PolicyDecision(action_index=int(action), strategy="component_audit_branch"),
                )
                branch.deployed += 1
                values[tape, column] = continue_components(
                    core, builder, executor, branch, reward_model
                )
            # WAIT keeps the token for the next scheduled epoch, which is the
            # natural-momentum reference the world model is supposed to predict.
            CB.restore(core, snapshot)
            PI.reseed_exogenous(core, seed_for_tape)
            deferred = clock.copy()
            deferred.deployed = 0
            wait[tape] = continue_components(
                core, builder, executor, deferred, reward_model
            )
    finally:
        CB.restore(core, snapshot)
    return {
        "seed": int(seed),
        "simulation_time": int(clock.t),
        "actions": feasible,
        "mask": mask,
        "features": grid,
        "scores": prior_scores(np.asarray(grid), mask),
        "components": values,
        "wait_components": wait,
    }


def cross_check(records, path: str = V25_AUDIT_STATES) -> dict:
    """Do the summed components reproduce the v25 audit's scalar values?

    Same seeds, same masks, same tape keys, same base continuation, so for every
    first-decision state the component sum must equal the stored scalar return
    to floating-point tolerance.  A mismatch means the component collection is
    not measuring the same trajectories the existing audit measured, and no
    conclusion drawn from it composes with the earlier results.
    """
    try:
        with open(path, "rb") as handle:
            reference = pickle.load(handle)["states"]
    except OSError:
        return {"checked": 0, "note": f"reference {path} unavailable"}
    lookup = {
        int(state["seed"]): state
        for state in reference
        if int(state["decision_index"]) == 0
    }
    checked, worst = 0, 0.0
    for record in records:
        state = lookup.get(int(record["seed"]))
        if state is None or state["actions"].size != record["actions"].size:
            continue
        if not np.array_equal(np.asarray(state["actions"]), np.asarray(record["actions"])):
            continue
        mine = record["components"].sum(axis=-1)  # (tapes, cells, offsets + 1)
        theirs = np.asarray(state["values"], dtype=np.float64)
        if mine.shape != theirs.shape:
            continue
        worst = max(worst, float(np.max(np.abs(mine - theirs))))
        checked += 1
    return {"checked": checked, "max_absolute_difference": worst}


def collect(args) -> None:
    jobs = [
        (FIRST_SEED + index, args.tapes, args.max_cells) for index in range(args.episodes)
    ]
    started = time.time()
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            records = pool.map(collect_episode, jobs)
    else:
        records = [collect_episode(job) for job in jobs]
    records = [record for record in records if record is not None]
    payload = {
        "offsets": OFFSETS,
        "components": list(REWARD_COMPONENT_NAMES),
        "states": records,
        "wall_seconds": time.time() - started,
        "scenario": {"population": POPULATION, "horizon": HORIZON, "interval": INTERVAL,
                     "budget": BUDGET, "map": MAP, "masks": V25_MASK},
        "cross_check": cross_check(records),
    }
    with open(args.states, "wb") as handle:
        pickle.dump(payload, handle)
    print(f"collected {len(records)} first-decision states in "
          f"{payload['wall_seconds'] / 60:.1f} min -> {args.states}")
    check = payload["cross_check"]
    if check.get("checked"):
        print(f"cross-check against the v25 audit: {check['checked']} states, "
              f"largest scalar difference {check['max_absolute_difference']:.3e}")
    else:
        print(f"cross-check skipped: {check.get('note', 'no comparable states')}")


# ----------------------------------------------------------------- analysis


def variance_components(matrix: np.ndarray) -> tuple[float, float]:
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    return float("nan") if denominator <= 0 else float(np.dot(ra, rb) / denominator)


def analyze(args) -> None:
    with open(args.states, "rb") as handle:
        payload = pickle.load(handle)
    states = payload["states"]
    names = list(payload["components"])
    offset = len(OFFSET_LABELS) - 1  # full horizon
    report: dict = {"states": len(states), "components": names}

    def matrix_for(state, name: str, position: int) -> np.ndarray:
        block = state["components"]
        if name == "total":
            return block[:, :, position, :].sum(axis=-1)
        return block[:, :, position, names.index(name)]

    def pooled(name: str, position: int) -> tuple[float, float]:
        """Variance components pooled over states with equal weight.

        Pooling the variances and taking one ratio at the end is stable at 16
        states; averaging per-state ratios is not, because a state whose
        estimated total cell variance is near zero dominates the average.
        """
        cell, residual = [], []
        for state in states:
            var_cell, var_residual = variance_components(matrix_for(state, name, position))
            if np.isfinite(var_cell) and np.isfinite(var_residual):
                cell.append(var_cell)
                residual.append(var_residual)
        if not cell:
            return float("nan"), float("nan")
        return float(np.mean(cell)), float(np.mean(residual))

    # Common (every installation) versus differential (which cell) per
    # component.  The WAIT reference defers the whole token schedule by one
    # interval, so the common effect is the value of installing ten minutes
    # earlier -- exactly the capacity-driven progress the concern is about.
    total_cell_variance, _ = pooled("total", offset)
    per_component = {}
    for name in names + ["total"]:
        var_cell, var_residual = pooled(name, offset)
        common = [
            float(np.mean(matrix_for(state, name, offset).mean(axis=1)
                          - (state["wait_components"][:, offset, :].sum(axis=-1)
                             if name == "total"
                             else state["wait_components"][:, offset, names.index(name)])))
            for state in states
        ]
        common_effect = float(np.mean(common)) if common else float("nan")
        per_component[name] = {
            "cell_effect_sd": math.sqrt(var_cell) if np.isfinite(var_cell) else float("nan"),
            "pairing_residual_sd": (
                math.sqrt(var_residual) if np.isfinite(var_residual) else float("nan")
            ),
            "snr": (
                math.sqrt(var_cell / var_residual)
                if np.isfinite(var_residual) and var_residual > 0 else float("inf")
            ),
            "variance_share_of_total_cell_effect": (
                var_cell / total_cell_variance
                if np.isfinite(total_cell_variance) and total_cell_variance > 0
                else float("nan")
            ),
            "common_effect_over_wait": common_effect,
            # How discriminating is this component relative to how much of it
            # every installation buys?  Large means the component separates
            # sitings; small means it only registers that a shelter now exists.
            "differential_over_common": (
                math.sqrt(var_cell) / abs(common_effect)
                if np.isfinite(var_cell) and abs(common_effect) > 0 else float("nan")
            ),
        }
    report["full_horizon_by_component"] = per_component

    # Does scoring on one component rank cells better than the scalar return?
    # One tape estimates, one held-out tape as truth, averaged over both
    # assignments and over states with equal weight.
    # The truth is always the terminal return on a held-out tape: every
    # estimator, however truncated or reweighted, is judged as a surrogate for
    # the objective the controller is actually graded on.
    def reweighting(exclude: int, truth_tape: int, estimator_tape: int,
                    position: int) -> np.ndarray:
        """Least-squares component weights predicting the held-out terminal effect.

        Fitted on every state but ``exclude``, so the weights are never fitted on
        the state they are scored at.  If the reward's own weights already put
        the signal where it belongs, this cannot beat ``total`` by much; if one
        component carries the differential, the fit will find it.
        """
        design, target = [], []
        for index, state in enumerate(states):
            if index == exclude:
                continue
            block = state["components"]
            rows_ = block[estimator_tape, :, position, :]
            design.append(rows_ - rows_.mean(axis=0, keepdims=True))
            terminal = block[truth_tape, :, offset, :].sum(axis=-1)
            target.append(terminal - terminal.mean())
        if not design:
            return np.zeros(len(names))
        solution, *_ = np.linalg.lstsq(
            np.concatenate(design, axis=0), np.concatenate(target), rcond=None
        )
        return solution

    def estimate_for(block, label: str, tape: int, position: int,
                     state_index: int, truth_tape: int) -> np.ndarray:
        window = block[tape, :, position, :]
        if label == "total":
            return window.sum(axis=-1)
        if label == "time_terms_only":
            return window[:, 2:].sum(axis=-1)
        if label == "exposure_plus_safe":
            return window[:, 0] + window[:, 3]
        if label == "refitted_weights":
            return window @ reweighting(state_index, truth_tape, tape, position)
        return window[:, names.index(label)]

    labels = names + ["total", "time_terms_only", "exposure_plus_safe", "refitted_weights"]

    def score_table(position: int) -> dict:
        table, weight_log = {}, []
        for label in labels:
            rows = []
            for state_index, state in enumerate(states):
                block = state["components"]
                for truth_tape in range(block.shape[0]):
                    estimator_tape = [
                        k for k in range(block.shape[0]) if k != truth_tape
                    ][0]
                    truth = block[truth_tape, :, offset, :].sum(axis=-1)
                    truth = truth - truth.mean()
                    if label == "refitted_weights":
                        weight_log.append(
                            reweighting(
                                state_index, truth_tape, estimator_tape, position
                            ).tolist()
                        )
                    estimate = estimate_for(
                        block, label, estimator_tape, position, state_index, truth_tape
                    )
                    estimate = estimate - estimate.mean()
                    best = int(np.argmax(truth))
                    chosen = int(np.argmax(estimate))
                    rows.append(
                        {
                            "spearman": spearman(estimate, truth),
                            "top1": float(chosen == best),
                            "regret": float(truth[best] - truth[chosen]),
                        }
                    )
            finite = [r["spearman"] for r in rows if np.isfinite(r["spearman"])]
            table[label] = {
                "mean_spearman": float(np.mean(finite)) if finite else float("nan"),
                "top1_rate": float(np.mean([r["top1"] for r in rows])),
                "mean_regret": float(np.mean([r["regret"] for r in rows])),
            }
        if weight_log:
            table["_refitted_weights_mean"] = np.mean(
                np.asarray(weight_log), axis=0
            ).tolist()
            table["_refitted_weight_order"] = list(names)
        return table

    # Each estimator at each truncation: one component read early is the direct
    # operationalization of "the reduction speed of hazard-weighted exposure".
    by_window = {
        label: score_table(position)
        for position, label in enumerate(OFFSET_LABELS)
    }
    scoring = by_window[OFFSET_LABELS[-1]]
    report["scoring_against_total_return"] = scoring
    report["scoring_by_window"] = by_window

    # How much of each component's differential has accumulated by each offset,
    # and how much of its pairing noise: the "reduction speed" view.
    accumulation = {}
    for name in names + ["total"]:
        rows = {}
        for position, label in enumerate(OFFSET_LABELS):
            var_cell, var_residual = pooled(name, position)
            rows[label] = {
                "cell_effect_sd": math.sqrt(var_cell) if np.isfinite(var_cell) else float("nan"),
                "pairing_residual_sd": (
                    math.sqrt(var_residual) if np.isfinite(var_residual) else float("nan")
                ),
            }
        full = rows[OFFSET_LABELS[-1]]
        for label in OFFSET_LABELS:
            rows[label]["share_of_full_cell_effect"] = (
                rows[label]["cell_effect_sd"] / full["cell_effect_sd"]
                if full["cell_effect_sd"] > 0 else float("nan")
            )
            rows[label]["share_of_full_residual"] = (
                rows[label]["pairing_residual_sd"] / full["pairing_residual_sd"]
                if full["pairing_residual_sd"] > 0 else float("nan")
            )
        accumulation[name] = rows
    report["accumulation_by_component"] = accumulation

    # The WAIT reference: how large is the deployed-versus-deferred effect
    # relative to the cell-to-cell differential the controller must resolve?
    common, differential = [], []
    for state in states:
        block = state["components"]
        cells = block[:, :, offset, :].sum(axis=-1)
        waited = state["wait_components"][:, offset, :].sum(axis=-1)
        common.append(float(np.mean(cells.mean(axis=1) - waited)))
        differential.append(float(np.std(cells.mean(axis=0))))
    report["common_versus_differential"] = {
        "mean_installation_effect_over_wait": float(np.mean(common)),
        "mean_within_state_cell_sd": float(np.mean(differential)),
        "ratio": float(np.mean(common) / np.mean(differential))
        if np.mean(differential) > 0 else float("nan"),
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"{len(states)} first-decision states, full horizon\n")
    print(f"{'component':>22s} {'cell effect':>12s} {'pair resid':>11s} {'SNR':>6s} "
          f"{'var share':>10s} {'common':>9s} {'diff/common':>12s}")
    for name, row in per_component.items():
        print(f"{name:>22s} {row['cell_effect_sd']:12.4f} {row['pairing_residual_sd']:11.4f} "
              f"{row['snr']:6.2f} {row['variance_share_of_total_cell_effect']:10.2f} "
              f"{row['common_effect_over_wait']:+9.4f} {row['differential_over_common']:12.3f}")
    print()
    print("ranking cells by one component, scored against the total return")
    print(f"{'score':>22s} {'spearman':>9s} {'top-1':>7s} {'regret':>8s}")
    for name, row in scoring.items():
        if name.startswith("_"):
            continue
        print(f"{name:>22s} {row['mean_spearman']:9.3f} {row['top1_rate']:7.3f} "
              f"{row['mean_regret']:8.4f}")
    if "_refitted_weights_mean" in scoring:
        print("mean refitted component weights "
              f"{scoring['_refitted_weight_order']}: "
              + ", ".join(f"{w:+.2f}" for w in scoring["_refitted_weights_mean"]))
    print()
    print("Spearman against the terminal return, by estimator window "
          "(same simulation cost)")
    print(f"{'score':>22s} " + " ".join(f"{label:>7s}" for label in OFFSET_LABELS))
    for label in labels:
        print(f"{label:>22s} " + " ".join(
            f"{by_window[window][label]['mean_spearman']:7.3f}"
            for window in OFFSET_LABELS
        ))
    print()
    print("top-1 rate against the terminal return, by estimator window")
    print(f"{'score':>22s} " + " ".join(f"{label:>7s}" for label in OFFSET_LABELS))
    for label in labels:
        print(f"{label:>22s} " + " ".join(
            f"{by_window[window][label]['top1_rate']:7.3f}"
            for window in OFFSET_LABELS
        ))
    print()
    print("share of the full-horizon magnitude accumulated by each offset")
    for name, rows in accumulation.items():
        parts = "  ".join(
            f"{label} {rows[label]['share_of_full_cell_effect']:.2f}/"
            f"{rows[label]['share_of_full_residual']:.2f}"
            for label in OFFSET_LABELS[:-1]
        )
        print(f"{name:>22s} signal/noise  {parts}")
    print()
    cv = report["common_versus_differential"]
    print(f"installing anything rather than deferring: {cv['mean_installation_effect_over_wait']:+.4f} return")
    print(f"cell-to-cell differential within a state : {cv['mean_within_state_cell_sd']:.4f} return SD")
    print(f"ratio                                    : {cv['ratio']:.1f}x")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--episodes", type=int, default=10)
    c.add_argument("--tapes", type=int, default=3)
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--max-cells", type=int, default=0, help="0 uses every feasible cell")
    c.add_argument("--states", default="runs/exposure_component_states_20260927.pkl")
    c.set_defaults(function=collect)
    a = sub.add_parser("analyze")
    a.add_argument("--states", default="runs/exposure_component_states_20260927.pkl")
    a.add_argument("--output", default="runs/exposure_component_signal_20260927.json")
    a.set_defaults(function=analyze)
    args = parser.parse_args(argv)
    args.function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
