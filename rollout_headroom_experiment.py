#!/usr/bin/env python3
"""How much siting improvement exists beyond the base heuristics?

This is a measurement, not a learner.  It answers the question that decides
whether the NMCC actor *can* beat objective-greedy under the fixed
15-minute schedule: is there a better siting policy at all, and how large is
its margin?

At every scheduled decision the one-step rollout policy (Bertsekas) installs
each feasible cell in a branch, continues the episode with a fixed base policy
to the horizon on ``--tapes`` common-random-number futures, and executes the
cell with the lowest mean complete cost-to-go.  Rollout is guaranteed (in
expectation, up to Monte Carlo error) to be no worse than its base policy, so

    J(base) - J(rollout(base))

is a lower bound on the foresight headroom that any learned siting policy
could exploit beyond that base.  If it is small, no reward or optimizer change
can make RL beat the heuristics by much; if it is large, the learner is
failing to extract a signal that exists, and rollout targets show where.

The script also records, per decision, the full-horizon rollout cost of every
feasible cell, so the zero-horizon objective-greedy score and the simple
heuristic scores can be checked against full-horizon value on real
State College states (Spearman, top-1, regret).

Nothing in the model code is changed: ``RLBridge._select_action`` is wrapped
in this process only, and the live simulator is verified bitwise-unchanged
after every rollout decision.

Usage (one scenario per process; launch several in parallel):

    .venv/bin/python rollout_headroom_experiment.py --scenario 1 \\
        --strategies objective_greedy,heuristic,hazard_weighted,random,rollout:hazard_weighted \\
        --tapes 2 --out runs/rollout_headroom_v48_20261006

    # quick smoke on a 250-person population:
    .venv/bin/python rollout_headroom_experiment.py --scenario 1 --population 250 \\
        --strategies hazard_weighted,rollout:hazard_weighted --tapes 1 --out runs/rollout_smoke

Scenario seeds are the v48 evaluation seeds (launch seed 4810), so results pair
with the v48 RL evaluation once it exists.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

import CounterfactualBranch as CB
import NMCCPolicyImprovement as NPI
import RLBridge as RB
from CAProcessor import authoritative_grid_shape
from DecisionInterface import (
    PolicyDecision,
    RegionalObservationBuilder,
    estimate_remaining_time_to_safety,
)

DEFAULT_MANIFEST = (
    "runs/state_college_2500_target_v48_full96_seed4810_20261005/experiment_manifest.json"
)
DEFAULT_CURRICULUM = "config/state_college_training_curriculum_2500_target_v48.json"
CITY_ID = "state_college_pa"
CITY_SCALE_RANK = 2  # multicity_backtest stream offset for State College

ROLLOUT_ENV = "ROLLOUT_BASE_POLICY"
TAPES_ENV = "ROLLOUT_TAPES"
LOG_ENV = "ROLLOUT_DECISION_LOG"


# ---------------------------------------------------------------------------
# Scenario reconstruction (mirrors multicity_backtest evaluation exactly)
# ---------------------------------------------------------------------------

def _seed(launch_seed: int, stream: int, index: int = 0) -> int:
    sequence = np.random.SeedSequence([int(launch_seed), int(stream), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def evaluation_overrides(manifest_path: str, curriculum_path: str, population: int | None) -> tuple[dict, int]:
    from TrainingCurriculum import load_training_curriculum

    manifest = json.loads(Path(manifest_path).read_text())
    base = dict(manifest["effective_overrides_by_city"][CITY_ID])
    curriculum = load_training_curriculum(curriculum_path)
    final_variant = curriculum.stages[-1].variants[0]
    overrides = {**base, **curriculum.learner_overrides, **dict(final_variant.overrides)}
    if population is not None:
        overrides["pedVol"] = int(population)
        shelters = int(overrides["initShelterVol"]) + int(overrides["maxAdditionalShelters"])
        surplus = int(overrides.get("shelterCapacitySurplus", 200))
        total = int(population) + surplus
        if total % shelters:
            raise ValueError(
                f"population + surplus ({total}) must divide evenly over {shelters} shelters"
            )
        overrides["shelterCapacityToken"] = total // shelters
    return overrides, int(manifest["launch_seed"])


# ---------------------------------------------------------------------------
# Base policies and the complete continuation
# ---------------------------------------------------------------------------

def base_action(bridge, observation, base: str) -> int:
    if base == "heuristic":
        return int(bridge.heuristic_policy.select(observation, deterministic=True).action_index)
    if base == "hazard_weighted":
        return int(bridge.hazard_weighted_policy.select(observation, deterministic=True).action_index)
    if base == "objective_greedy":
        return int(
            bridge.objective_greedy_policy.select(
                observation,
                candidate_values=bridge._objective_greedy_values(observation),
                deterministic=True,
            ).action_index
        )
    raise ValueError(f"Unsupported rollout base policy {base!r}")


def _install(core, cell_index: int) -> None:
    _, cell_y = authoritative_grid_shape(core)
    cell = divmod(int(cell_index), cell_y)
    shelter_id = core.shelterDS.newShelter({"cell": cell}, core.cellTracker)
    if shelter_id is None:
        raise RuntimeError(f"Rollout failed to install feasible cell {cell_index}")
    shelter = core.shelterDS.shelterList[shelter_id]
    if hasattr(core.cellTracker, "addShelter"):
        core.cellTracker.addShelter(cell, shelter)
    core.pedDS.reroute_active_to_nearest_shelter(new_shelter=shelter)


def continuation_cost(
    bridge,
    builder: RegionalObservationBuilder,
    *,
    first_cell: int,
    base: str,
    episode_horizon: int,
) -> float:
    """Install ``first_cell`` now, then follow ``base`` on the common schedule.

    Same ledger, cohort, ordering and normalization as
    ``CounterfactualBrancher._run_uniform_random_continuation`` (the V_mu
    target), with the uniform chooser replaced by the base policy.
    """
    core = bridge.core
    interval = max(1, int(getattr(core, "shelterActionInterval", 1)))
    now = int(core.pedDS.currTime)
    decision_index = int(bridge.deployments_made)
    remaining = int(bridge.remaining_deployments)
    cohort = {
        int(p.agentID)
        for p in core.pedDS.pedAgentList.values()
        if not bool(getattr(p, "terminated", False))
    }
    step_minutes = max(1e-9, float(getattr(core, "timeStepMinutes", 1.0)))

    _install(core, first_cell)
    remaining -= 1
    decision_index += 1
    next_time = now + interval
    cost = 0.0
    while int(core.pedDS.currTime) < episode_horizon:
        current = int(core.pedDS.currTime)
        if remaining > 0 and current >= next_time:
            observation = builder.build(
                decision_index=decision_index,
                simulation_time=current,
                remaining_deployments=remaining,
            )
            if not np.any(observation.action_mask):
                raise RuntimeError("Base continuation exhausted feasible cells")
            action = base_action(bridge, observation, base)
            _install(core, int(observation.candidate_cell_indices[action]))
            remaining -= 1
            decision_index += 1
            next_time += interval
        CB.advance_one_timestep(core)
        plain, exposure, casualty = CB.controllable_time_mass(core, cohort)
        cost += step_minutes * (plain + exposure + casualty)
    residual = estimate_remaining_time_to_safety(
        core,
        episode_horizon=episode_horizon,
        pedestrian_ids=cohort,
        include_panicked=True,
        include_casualties=False,
    )
    cost += residual.hazard_weighted_person_minutes
    return float(cost / (float(bridge.initial_population) * float(episode_horizon) * step_minutes))


def _spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# In-process wrapper of the live decision
# ---------------------------------------------------------------------------

_ORIGINAL_SELECT = RB.RLBridge._select_action


def _rollout_select(self, observation, recurrent_step=None):
    base = os.environ.get(ROLLOUT_ENV, "")
    if not base or self.deployment_strategy != base:
        return _ORIGINAL_SELECT(self, observation, recurrent_step)

    tapes = int(os.environ.get(TAPES_ENV, "2"))
    started = time.perf_counter()
    feasible = np.flatnonzero(observation.action_mask)
    cells = observation.candidate_cell_indices[feasible].astype(int)
    base_choice = base_action(self, observation, base)
    greedy_values = self._objective_greedy_values(observation)[feasible]
    active = observation.active_by_cell[cells]
    hazard_score = active * (1.0 + observation.danger_by_cell[cells])

    episode_horizon = int(self.horizon)
    reseeder = CB.CounterfactualBrancher(
        self.core,
        horizon=0,
        reward_model=self.reward_model,
        initial_population=int(self.initial_population),
        episode_horizon=episode_horizon,
    )
    builder = RegionalObservationBuilder(
        self.core,
        initial_population=int(self.initial_population),
        horizon=episode_horizon,
        maximum_deployments=int(self.maximum_deployments),
    )
    episode_seed = int(getattr(self.core, "scenario_seed", self.policy_seed))
    before = CB.live_state_digest(self.core)
    snapshot = CB.capture(self.core, label="rollout-headroom")
    costs = np.zeros((tapes, feasible.size), dtype=np.float64)
    try:
        for tape in range(tapes):
            seed = NPI.tape_seed(episode_seed ^ 0x524F4C4C, int(self.deployments_made), tape)
            for j, cell in enumerate(cells):
                CB.restore(self.core, snapshot)
                reseeder._reseed_exogenous_streams(seed)
                costs[tape, j] = continuation_cost(
                    self,
                    builder,
                    first_cell=int(cell),
                    base=base,
                    episode_horizon=episode_horizon,
                )
    finally:
        CB.restore(self.core, snapshot)
    if CB.live_state_digest(self.core) != before:
        raise RuntimeError("Rollout evaluation changed the live simulator state")

    mean_cost = costs.mean(axis=0)
    best = int(np.argmin(mean_cost))
    base_position = int(np.flatnonzero(feasible == base_choice)[0])
    greedy_position = int(np.argmax(greedy_values))
    record = {
        "decision_index": int(self.deployments_made),
        "simulation_time": int(self.core.pedDS.currTime),
        "feasible": int(feasible.size),
        "tapes": tapes,
        "cells": cells.tolist(),
        "rollout_cost_by_tape": costs.tolist(),
        "rollout_best_cell": int(cells[best]),
        "base_cell": int(cells[base_position]),
        "greedy_cell": int(cells[greedy_position]),
        "cost_spread_sd": float(np.std(mean_cost)),
        "regret_base": float(mean_cost[base_position] - mean_cost[best]),
        "regret_greedy": float(mean_cost[greedy_position] - mean_cost[best]),
        "regret_uniform": float(mean_cost.mean() - mean_cost[best]),
        "tape_pairing_residual_sd": (
            float(np.std(costs - costs.mean(axis=0, keepdims=True)
                          - costs.mean(axis=1, keepdims=True) + costs.mean()))
            if tapes > 1 else float("nan")
        ),
        "spearman_greedy_vs_rollout": _spearman(-greedy_values, mean_cost),
        "spearman_active_vs_rollout": _spearman(-active, mean_cost),
        "spearman_hazard_weighted_vs_rollout": _spearman(-hazard_score, mean_cost),
        "seconds": float(time.perf_counter() - started),
    }
    log_path = os.environ.get(LOG_ENV)
    if log_path:
        with open(log_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    print(
        f"[rollout] t={record['simulation_time']} k={record['decision_index']} "
        f"regret base={record['regret_base']:.4f} greedy={record['regret_greedy']:.4f} "
        f"uniform={record['regret_uniform']:.4f} ({record['seconds']:.0f}s)",
        flush=True,
    )
    return PolicyDecision(action_index=int(feasible[best]), strategy=self.deployment_strategy)


RB.RLBridge._select_action = _rollout_select


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", type=int, required=True, help="evaluation replication, 1-based")
    parser.add_argument(
        "--strategies",
        default="objective_greedy,heuristic,hazard_weighted,random,rollout:hazard_weighted",
        help="comma list; rollout:<base> with base in heuristic|hazard_weighted|objective_greedy",
    )
    parser.add_argument("--tapes", type=int, default=2)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM)
    parser.add_argument("--out", required=True)
    parser.add_argument("--machine", default="local")
    args = parser.parse_args()

    from backtest import _run_episode

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    overrides, launch_seed = evaluation_overrides(args.manifest, args.curriculum, args.population)
    scenario_seed = _seed(launch_seed, 500 + CITY_SCALE_RANK, args.scenario)
    policy_seed = _seed(launch_seed, 600 + CITY_SCALE_RANK, args.scenario)

    for spec in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        rollout = spec.startswith("rollout:")
        strategy = spec.split(":", 1)[1] if rollout else spec
        tag = f"rollout_{strategy}" if rollout else strategy
        log_path = out / f"decisions_s{args.scenario:03d}_{tag}.jsonl"
        if rollout:
            os.environ[ROLLOUT_ENV] = strategy
            os.environ[TAPES_ENV] = str(int(args.tapes))
            os.environ[LOG_ENV] = str(log_path)
            if log_path.exists():
                log_path.unlink()
        else:
            os.environ.pop(ROLLOUT_ENV, None)
        print(f"[run] scenario={args.scenario} seed={scenario_seed} strategy={tag}", flush=True)
        work = out / "work" / f"s{args.scenario:03d}_{tag}"
        row = _run_episode(
            replication=int(args.scenario),
            machine=args.machine,
            phase=str(work),
            strategy=strategy,
            train_mode=False,
            scenario_seed=scenario_seed,
            policy_seed=policy_seed,
            checkpoint_path=str(work / "unused_checkpoint.pt"),
            diagnostics_path=str(work / "unused_diagnostics.csv"),
            overrides=overrides,
        )
        keep = {
            key: row.get(key)
            for key in (
                "scenario_seed",
                "objective_normalized_cost",
                "realized_hazard_weighted_time_to_safety",
                "objective_hazard_weighted_person_time_cost",
                "casualty",
                "safe_completed",
                "unfinished",
                "deployment_times",
                "episode_wall_time_s",
            )
        }
        keep.update(
            {
                "scenario": int(args.scenario),
                "strategy": tag,
                "population": int(overrides["pedVol"]),
                "tapes": int(args.tapes) if rollout else 0,
            }
        )
        with open(out / "episodes.jsonl", "a") as handle:
            handle.write(json.dumps(keep, default=str) + "\n")
        print(f"[done] {tag}: normalized cost {keep['objective_normalized_cost']}", flush=True)
    os.environ.pop(ROLLOUT_ENV, None)


def summarize(out_dir: str) -> None:
    """Paired summary across scenarios: python -c 'import rollout_headroom_experiment as r; r.summarize(\"runs/...\")'"""
    import pandas as pd

    out = Path(out_dir)
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines() if line]
    frame = pd.DataFrame(rows)
    table = frame.pivot_table(index="scenario", columns="strategy", values="objective_normalized_cost")
    print(table.round(4).to_string())
    print("\nmean normalized cost:\n" + table.mean().round(4).to_string())
    for column in table.columns:
        if column.startswith("rollout_"):
            base = column[len("rollout_"):]
            for reference in (base, "objective_greedy", "random"):
                if reference in table.columns:
                    diff = (table[reference] - table[column]).dropna()
                    if len(diff):
                        se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
                        print(
                            f"{reference} - {column}: {diff.mean():+.4f} (SE {se:.4f}), "
                            f"wins {int((diff > 0).sum())}/{len(diff)}"
                        )
    decisions = []
    for path in out.glob("decisions_*.jsonl"):
        decisions += [json.loads(line) for line in path.read_text().splitlines() if line]
    if decisions:
        d = pd.DataFrame(decisions)
        cols = [
            "regret_base", "regret_greedy", "regret_uniform", "cost_spread_sd",
            "tape_pairing_residual_sd", "spearman_greedy_vs_rollout",
            "spearman_active_vs_rollout", "spearman_hazard_weighted_vs_rollout",
        ]
        print("\nper-decision (full-horizon rollout costs):")
        print(d.groupby("decision_index")[cols].mean().round(4).to_string())
        print(d[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
