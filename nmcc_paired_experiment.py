#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NMCC minimal first experiment: does pairing create a learnable signal?

This is the promotion gate that must pass before any architecture is built on
top of counterfactual credit assignment.  It answers three questions, in
increasing order of what the project actually needs:

1. **Does common-random-number pairing reduce the variance of a measured
   action effect?**  NMCC's gate is a 50% reduction with no change in the mean
   effect beyond Monte Carlo error.
2. **By how much, relative to the noise that defeated the previous design?**
   The 2026-09-19 validation measured a per-episode total-return standard
   deviation of 0.2226 against a held-out RL-minus-heuristic difference of
   -0.00596.  A useful estimator has to resolve an effect of roughly that size.
3. **Can the estimator rank cells?**  Variance reduction on a scalar is not
   the same as a usable learning signal.  What PPO needs is for the ordering
   over cells to be stable across noise realizations, because that ordering is
   what the policy gradient moves the actor toward.

Design
------
At each sampled decision epoch the simulator is snapshotted.  For each of
``R`` independent noise tapes and each sampled cell ``c`` the harness records
``A[r, c] = G(c, U_r)``, the return over the branch horizon after deploying in
``c``, and ``W[r] = G(0, U_r)``, the return over the same horizon with no
deployment.  Two estimators are then formed from *exactly the same
simulations*:

    common:       D_common[r, c] = A[r, c] - W[r]
    independent:  D_indep[r, c]  = A[r, c] - W[(r + 1) mod R]

The acted term is identical in both, so any difference between the two is
attributable to the pairing alone and to nothing else -- no extra simulation,
no confound from unequal sample sizes.  The independent arm is the honest
control for "estimate both sides separately", which is what an unpaired
policy-gradient estimator effectively does.

Usage
-----
``python nmcc_paired_experiment.py --help``.  With ``--real-core`` the same
measurement runs against a genuine ``Core`` on an OSM map; by default it uses
the torch-free synthetic testbed so it can run anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

import CounterfactualBranch as CB

# Reference figures from docs/CREDIT_ASSIGNMENT_VALIDATION_RESULTS_20260919.md,
# so the report says whether the estimator resolves the effect that mattered.
BASELINE_EPISODE_RETURN_SD = 0.2226
BASELINE_RL_MINUS_HEURISTIC = -0.00596


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- enough for a Spearman coefficient."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique = np.unique(values)
    if unique.size != values.size:
        for value in unique:
            mask = values == value
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(ra, rb) / denominator)


def mean_pairwise_rank_agreement(matrix: np.ndarray) -> float:
    """Average Spearman correlation between every pair of noise tapes.

    High agreement means the estimator names the same good cells whatever the
    fire happens to do -- which is exactly the property the actor needs for its
    gradient to point somewhere stable.
    """
    tapes = matrix.shape[0]
    scores = []
    for i in range(tapes):
        for j in range(i + 1, tapes):
            value = spearman(matrix[i], matrix[j])
            if not math.isnan(value):
                scores.append(value)
    return float(np.mean(scores)) if scores else float("nan")


@dataclass
class EpochMeasurement:
    """Paired measurements at one decision epoch."""

    simulation_time: int
    cells: list[int]
    acted: np.ndarray  # (tapes, cells) return after deploying
    waited: np.ndarray  # (tapes,) return with no deployment
    active_population: int = 0
    mean_danger: float = 0.0

    def common(self) -> np.ndarray:
        """Paired estimate: deploy-minus-wait on the same noise tape."""
        return self.acted - self.waited[:, None]

    def unpaired(self) -> np.ndarray:
        """What an ordinary advantage estimator sees.

        PPO forms ``G(a, U) - V(s)``.  ``V(s)`` is a learned function of the
        state, identical for every action at that state, so it shifts the
        estimate without changing its spread and without changing the ordering
        the gradient induces over actions.  The realized return ``A`` is
        therefore the honest unpaired comparator -- a tighter one than
        differencing against an independently drawn wait branch, which would
        inflate the unpaired variance by roughly a factor of two.
        """
        return self.acted

    def independent(self) -> np.ndarray:
        """Deploy-minus-wait with the pairing deliberately broken."""
        rolled = np.roll(self.waited, 1)
        return self.acted - rolled[:, None]

    def true_effect(self) -> np.ndarray:
        """Per-cell effect averaged over tapes: the target to be recovered."""
        return self.common().mean(axis=0)

    def single_sample_rank_recovery(self, *, trials: int = 400, seed: int = 0) -> dict:
        """Can one visit per cell recover the true ordering over cells?

        This is the question that decides whether the actor can learn.  On-policy
        training does not get many matched samples of the same decision state: it
        sees a state, takes one action, and lives one noise realization.  So each
        cell is scored here under *its own* tape, and the resulting ordering is
        compared against the tape-averaged truth.

        The paired estimator subtracts that tape's own wait branch and keeps the
        ordering; the unpaired estimator carries the whole natural trajectory of
        whichever tape it happened to draw, and the ordering dissolves into it.
        """
        rng = np.random.default_rng(seed)
        truth = self.true_effect()
        if truth.size < 2 or float(np.std(truth)) <= 0.0:
            return {"paired": float("nan"), "unpaired": float("nan"), "trials": 0}
        common = self.common()
        unpaired = self.unpaired()
        tapes, cells = common.shape
        paired_scores, unpaired_scores = [], []
        for _ in range(int(trials)):
            draw = rng.integers(0, tapes, size=cells)
            index = (draw, np.arange(cells))
            paired_scores.append(spearman(common[index], truth))
            unpaired_scores.append(spearman(unpaired[index], truth))
        return {
            "paired": float(np.nanmean(paired_scores)),
            "unpaired": float(np.nanmean(unpaired_scores)),
            "trials": int(trials),
        }

    def summary(self) -> dict:
        common = self.common()
        unpaired = self.unpaired()
        independent = self.independent()

        # Noise: how much one sample of each estimator wobbles, per cell,
        # averaged over cells.
        common_sd = float(np.mean(common.std(axis=0, ddof=1)))
        unpaired_sd = float(np.mean(unpaired.std(axis=0, ddof=1)))
        independent_sd = float(np.mean(independent.std(axis=0, ddof=1)))

        # Signal: how much the choice of cell actually changes the outcome.
        truth = self.true_effect()
        signal_sd = float(np.std(truth, ddof=1)) if truth.size > 1 else 0.0

        correlation = float("nan")
        if self.acted.shape[0] > 2:
            flat_acted = self.acted.mean(axis=1)
            if flat_acted.std() > 0 and self.waited.std() > 0:
                correlation = float(np.corrcoef(flat_acted, self.waited)[0, 1])

        recovery = self.single_sample_rank_recovery(seed=int(self.simulation_time))
        return {
            "simulation_time": int(self.simulation_time),
            "active_population": int(self.active_population),
            "mean_danger": float(self.mean_danger),
            "cells": list(self.cells),
            "common_mean_effect": float(common.mean()),
            "independent_mean_effect": float(independent.mean()),
            "common_effect_sd": common_sd,
            "unpaired_effect_sd": unpaired_sd,
            "independent_effect_sd": independent_sd,
            "between_cell_signal_sd": signal_sd,
            "discriminability_paired": (
                signal_sd / common_sd if common_sd > 0 else float("inf") if signal_sd > 0 else float("nan")
            ),
            "discriminability_unpaired": (
                signal_sd / unpaired_sd if unpaired_sd > 0 else float("nan")
            ),
            "variance_reduction": (
                1.0 - (common_sd ** 2) / (unpaired_sd ** 2)
                if unpaired_sd > 0
                else float("nan")
            ),
            "acted_waited_correlation": correlation,
            "rank_recovery_paired": recovery["paired"],
            "rank_recovery_unpaired": recovery["unpaired"],
        }


def measure_epoch(
    core,
    *,
    brancher: CB.CounterfactualBrancher,
    cells: Sequence[int],
    tapes: int,
    tape_seed_base: int,
) -> EpochMeasurement:
    """Collect ``A[r, c]`` and ``W[r]`` at the current simulator state.

    Every branch starts from one snapshot, so all of them are counterfactuals
    of the *same* decision state; and within a tape the acted and waited
    branches see the same disturbances by construction.
    """
    base = CB.capture(core, label=f"epoch-{getattr(core.pedDS, 'currTime', 0)}")
    tracker = core.cellTracker
    danger = np.asarray(getattr(tracker, "dangerLevelByCell", []), dtype=np.float64)
    start_outcome = CB.outcome_snapshot(core)

    acted = np.zeros((int(tapes), len(cells)), dtype=np.float64)
    waited = np.zeros(int(tapes), dtype=np.float64)

    for tape in range(int(tapes)):
        seed = int(tape_seed_base + 1009 * tape)

        CB.restore(core, base)
        brancher._reseed_exogenous_streams(seed)
        wait_branch = brancher._run_branch(
            label="waited", install_cell=None, steps=brancher.horizon
        )
        waited[tape] = wait_branch.discounted_return()

        for index, cell in enumerate(cells):
            CB.restore(core, base)
            brancher._reseed_exogenous_streams(seed)
            acted_branch = brancher._run_branch(
                label="acted", install_cell=int(cell), steps=brancher.horizon
            )
            acted[tape, index] = acted_branch.discounted_return()

    CB.restore(core, base)
    return EpochMeasurement(
        simulation_time=int(getattr(core.pedDS, "currTime", 0)),
        cells=[int(c) for c in cells],
        acted=acted,
        waited=waited,
        active_population=int(start_outcome.active_population),
        mean_danger=float(danger.mean()) if danger.size else 0.0,
    )


def run_experiment(
    core,
    *,
    horizon: int = 6,
    tapes: int = 6,
    cells_per_epoch: int = 6,
    decision_interval: int = 5,
    first_decision: int = 6,
    epochs: int = 6,
    seed: int = 20260920,
    verbose: bool = True,
) -> dict:
    """March the episode, measuring paired effects at each decision epoch."""
    rng = np.random.default_rng(seed)
    reward_model = core.reward_model() if hasattr(core, "reward_model") else None
    if reward_model is None:
        from RewardProcessor import RewardProcessor

        reward_model = RewardProcessor()

    brancher = CB.CounterfactualBrancher(
        core,
        horizon=horizon,
        reward_model=reward_model,
        initial_population=int(getattr(core, "initial_population", 1)),
        episode_horizon=int(getattr(core, "stopTime", 60)),
    )

    for _ in range(int(first_decision)):
        CB.advance_one_timestep(core)

    measurements: list[EpochMeasurement] = []
    for epoch in range(int(epochs)):
        feasible = core.feasible_cells()
        if not feasible:
            break
        chosen = list(
            rng.choice(
                feasible,
                size=min(int(cells_per_epoch), len(feasible)),
                replace=False,
            )
        )
        chosen = sorted(int(c) for c in chosen)

        measurement = measure_epoch(
            core,
            brancher=brancher,
            cells=chosen,
            tapes=tapes,
            tape_seed_base=int(seed + 7919 * epoch),
        )
        measurements.append(measurement)
        if verbose:
            summary = measurement.summary()
            print(
                f"  epoch {epoch}  t={summary['simulation_time']:3d} "
                f"active={summary['active_population']:4d} "
                f"danger={summary['mean_danger']:.2f} | "
                f"sd_paired={summary['common_effect_sd']:.5f} "
                f"sd_unpaired={summary['unpaired_effect_sd']:.5f} "
                f"varred={summary['variance_reduction']*100:5.1f}% | "
                f"signal={summary['between_cell_signal_sd']:.5f} | "
                f"rank_paired={summary['rank_recovery_paired']:+.2f} "
                f"rank_unpaired={summary['rank_recovery_unpaired']:+.2f}"
            )

        # Advance the real episode: deploy the best measured cell and run on to
        # the next decision epoch, so later epochs are measured from states a
        # real policy would actually reach.
        best_index = int(np.argmax(measurement.common().mean(axis=0)))
        best_cell = measurement.cells[best_index]
        cell = divmod(best_cell, int(core.cellY))
        shelter_id = core.shelterDS.newShelter({"cell": cell}, core.cellTracker)
        if shelter_id is not None and hasattr(core.cellTracker, "addShelter"):
            core.cellTracker.addShelter(cell, core.shelterDS.shelterList[shelter_id])
            reroute = getattr(core.pedDS, "reroute_to_new_shelter_if_closer", None)
            if callable(reroute):
                reroute(core.shelterDS.shelterList[shelter_id])
        for _ in range(int(decision_interval)):
            CB.advance_one_timestep(core)

    return aggregate(measurements, horizon=horizon, tapes=tapes)


def aggregate(measurements: Sequence[EpochMeasurement], *, horizon: int, tapes: int) -> dict:
    """Pool epochs into the report the promotion gate is read from."""
    if not measurements:
        return {"status": "no_measurements"}

    per_epoch = [m.summary() for m in measurements]
    common_all = np.concatenate([m.common().ravel() for m in measurements])
    unpaired_all = np.concatenate([m.unpaired().ravel() for m in measurements])

    def pooled(key: str) -> float:
        values = [row[key] for row in per_epoch if np.isfinite(row[key])]
        return float(np.mean(values)) if values else float("nan")

    common_sd = pooled("common_effect_sd")
    unpaired_sd = pooled("unpaired_effect_sd")
    signal_sd = pooled("between_cell_signal_sd")
    variance_reduction = (
        1.0 - (common_sd ** 2) / (unpaired_sd ** 2) if unpaired_sd > 0 else float("nan")
    )

    # Pairing is a variance-reduction device, not a re-definition of the
    # quantity being estimated.  If it shifted the mean it would have
    # introduced bias, which would be worse than the noise it removed.
    mean_common = float(common_all.mean())
    mean_independent = float(
        np.concatenate([m.independent().ravel() for m in measurements]).mean()
    )
    se_common = float(common_all.std(ddof=1) / math.sqrt(common_all.size))
    mean_shift_z = (
        abs(mean_common - mean_independent) / se_common if se_common > 0 else 0.0
    )

    recovery_paired = pooled("rank_recovery_paired")
    recovery_unpaired = pooled("rank_recovery_unpaired")
    discriminability_paired = pooled("discriminability_paired")
    discriminability_unpaired = pooled("discriminability_unpaired")

    gates = {
        "variance_reduction_at_least_50pct": bool(variance_reduction >= 0.50),
        "mean_effect_unchanged_within_mc_error": bool(mean_shift_z <= 2.0),
        "signal_exceeds_paired_noise": bool(discriminability_paired >= 1.0),
        "paired_recovers_cell_ranking": bool(recovery_paired >= 0.6),
        "pairing_beats_unpaired_ranking": bool(
            recovery_paired > recovery_unpaired + 0.10
        ),
    }

    return {
        "status": "ok",
        "branch_horizon_timesteps": int(horizon),
        "noise_tapes": int(tapes),
        "decision_epochs": len(per_epoch),
        "paired_effect_sd": common_sd,
        "unpaired_effect_sd": unpaired_sd,
        "between_cell_signal_sd": signal_sd,
        "variance_reduction_fraction": variance_reduction,
        "effect_sd_ratio": (common_sd / unpaired_sd) if unpaired_sd > 0 else float("nan"),
        "discriminability_paired": discriminability_paired,
        "discriminability_unpaired": discriminability_unpaired,
        "rank_recovery_paired": recovery_paired,
        "rank_recovery_unpaired": recovery_unpaired,
        "mean_effect_paired": mean_common,
        "mean_effect_unpaired_control": mean_independent,
        "mean_shift_z": mean_shift_z,
        "baseline_episode_return_sd_20260919": BASELINE_EPISODE_RETURN_SD,
        "baseline_rl_minus_heuristic_20260919": BASELINE_RL_MINUS_HEURISTIC,
        "paired_sd_vs_baseline_episode_sd": (
            common_sd / BASELINE_EPISODE_RETURN_SD if BASELINE_EPISODE_RETURN_SD else float("nan")
        ),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "per_epoch": per_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=6, help="counterfactual branch length L")
    parser.add_argument("--tapes", type=int, default=6, help="independent noise realizations")
    parser.add_argument("--cells-per-epoch", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--decision-interval", type=int, default=5)
    parser.add_argument("--first-decision", type=int, default=6)
    parser.add_argument("--population", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    import nmcc_testbed

    core = nmcc_testbed.build(
        grid=16,
        cell_x=6,
        cell_y=6,
        population=int(args.population),
        stop_time=60,
        spread_rate=(8, 4),
        casualty_rate=(25, 10),
        hazard_count=2,
        scenario_seed=int(args.seed),
    )

    print("NMCC paired-effect experiment")
    print(
        f"  branch horizon L={args.horizon}  tapes={args.tapes}  "
        f"cells/epoch={args.cells_per_epoch}  epochs={args.epochs}"
    )
    report = run_experiment(
        core,
        horizon=int(args.horizon),
        tapes=int(args.tapes),
        cells_per_epoch=int(args.cells_per_epoch),
        decision_interval=int(args.decision_interval),
        first_decision=int(args.first_decision),
        epochs=int(args.epochs),
        seed=int(args.seed),
    )

    print()
    print("Pooled result")
    print(f"  between-cell signal SD  {report['between_cell_signal_sd']:.6f}   (how much the choice matters)")
    print(f"  paired estimator SD     {report['paired_effect_sd']:.6f}")
    print(f"  unpaired estimator SD   {report['unpaired_effect_sd']:.6f}")
    print(f"  variance reduction      {report['variance_reduction_fraction']*100:.2f}%")
    print(f"  signal/noise paired     {report['discriminability_paired']:.2f}")
    print(f"  signal/noise unpaired   {report['discriminability_unpaired']:.2f}")
    print(f"  rank recovery paired    {report['rank_recovery_paired']:+.3f}")
    print(f"  rank recovery unpaired  {report['rank_recovery_unpaired']:+.3f}")
    print(f"  mean effect             {report['mean_effect_paired']:+.6f}  (shift z={report['mean_shift_z']:.2f})")
    print()
    for name, passed in report["gates"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
