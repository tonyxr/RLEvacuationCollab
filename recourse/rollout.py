"""Simulator-graded optimal stopping for evacuation orders, on public-belief futures.

At a check time ``t`` with threatened regions ``C`` the decision maker can order
the evacuation of ``C`` now, or at one of the later offsets ``d`` (minutes),
or never within the look-ahead. Each alternative is simulated from the same
snapshot on ``M`` hazard futures sampled from the public belief (the same
belief particles the forecasts and the belief planner use), with common random
numbers across alternatives, for ``H`` minutes. Its cost is the objective's
hazard-weighted person-time accrued over the look-ahead plus the remaining
time-to-safety estimate at its end, i.e. exactly the ledger a placement
branch is graded with. Ordering now is chosen only if it has the lowest mean
cost; otherwise the decision is re-taken at the next check (a rollout policy
for the stopping problem whose base policy is "never order").

No shelter is placed inside the look-ahead: every alternative faces the same
capacity, so the comparison isolates the timing of the order. The live
simulator is restored exactly afterwards (checked by state digest).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

import CounterfactualBranch as CB
from DecisionInterface import estimate_clearance_tail, estimate_remaining_time_to_safety

from .transaction import execute_evacuation_orders

NEVER = -1


def reseed_exogenous_streams(core, seed: int) -> None:
    """Common random numbers for one future (mirrors the branch engine)."""
    key = int(np.random.SeedSequence(int(seed)).generate_state(1, dtype=np.uint64)[0])
    mask = (1 << 64) - 1
    hazard_store = getattr(core, "hazardDS", None)
    if hazard_store is not None:
        hazard_store.rng = np.random.default_rng(key & mask)
    ped_store = getattr(core, "pedDS", None)
    if ped_store is not None:
        if hasattr(ped_store, "set_hazard_random_seed"):
            ped_store.set_hazard_random_seed((key ^ 0x5DEECE66D) & mask)
        if hasattr(ped_store, "panic_random_seed"):
            ped_store.panic_random_seed = (key ^ 0x9E3779B97F4A7C15) & mask


def belief_timeline(core, snapshot, seed: int, steps: int) -> tuple:
    """One action-independent hazard future from the public belief (mirrors the branch engine)."""
    CB.restore(core, snapshot)
    reseed_exogenous_streams(core, seed)
    regime = getattr(core, "hazard_regime", None)
    if regime is not None and bool(getattr(regime, "enabled", False)):
        regime = regime.belief_particles(1, np.random.default_rng(int(seed)))[0]
        core.hazard_regime = regime
        core.hazardDS.set_regime(regime)
    frames = []
    start = int(snapshot.simulation_time)
    for offset in range(1, int(steps) + 1):
        if regime is not None and bool(getattr(regime, "enabled", False)):
            regime.advance(start + offset, core.hazardDS, core.mapDS)
        core.hazardDS.spreadUpdate()
        core.hazardDS.heatUpdate()
        core.hazardDS.smokeUpdate()
        core.hazardDS.terminateHazard()
        frames.append(CB._capture_hazard_frame(core))
    CB.restore(core, snapshot)
    return tuple(frames)


def branch_step(core, *, hazard_frame=None, order_cells: Sequence = (), resolve: Optional[Callable] = None) -> list:
    """``CounterfactualBranch.advance_one_timestep`` with orders executed in the closure phase.

    With no orders this performs exactly the calls of ``advance_one_timestep``
    in the same order (tested by digest equality).
    """
    core.pedDS.startDocument()
    if hazard_frame is None:
        regime = getattr(core, "hazard_regime", None)
        if regime is not None and regime.enabled:
            regime.advance(int(getattr(core.pedDS, "currTime", 0)) + 1, core.hazardDS, core.mapDS)
        core.hazardDS.spreadUpdate()
        core.hazardDS.heatUpdate()
        core.hazardDS.smokeUpdate()
        core.hazardDS.terminateHazard()
    else:
        CB._apply_hazard_frame(core, hazard_frame)
    now = int(getattr(core.pedDS, "currTime", 0)) + 1
    records = []
    if order_cells:
        records = execute_evacuation_orders(core, order_cells, simulation_time=now, resolve=resolve)
    core.pedDS.resolveHazardShelterClosures(simulation_time=now)
    core.pedDS.loadShelterLookup(core.shelterDS.shelterByOSMID)
    core.pedDS.pedestrianHazardInteraction()
    core.pedDS.interPedestrianInteraction()
    core.pedDS.pedestrianNetworkInteraction()
    core.cellTracker.cellUpdate(pedDS=core.pedDS, forceTracker=core.forceTracker)
    core.pedDS.docuStatus()
    return records


def branch_cost(core, *, timeline: Sequence, schedule: dict, steps: int) -> float:
    """Objective ledger of one alternative: person-time over the look-ahead plus the remaining estimate."""
    result = CB.BranchResult(
        label="recourse",
        active_person_time_weight=float(getattr(core, "activePersonTimeWeight", 1.0)),
        hazard_exposure_time_weight=float(getattr(core, "hazardExposureTimeWeight", 1.0)),
    )
    horizon = int(getattr(core, "stopTime", 0)) - 1
    step_minutes = max(1e-9, float(getattr(core, "timeStepMinutes", 1.0)))
    for index in range(int(steps)):
        branch_step(core, hazard_frame=timeline[index], order_cells=schedule.get(index + 1, ()))
        active = {int(getattr(p, "agentID")) for p in core.pedDS.pedAgentList.values()
                  if not bool(getattr(p, "terminated", False))}
        plain, exposure, casualty = CB.controllable_time_mass(core, active)
        result.active_person_time += plain * step_minutes
        result.hazard_exposure_person_time += exposure * step_minutes
        result.expected_casualty_person_time += casualty * step_minutes
        result.steps += 1
        if int(getattr(core.pedDS, "currTime", 0)) >= horizon:
            break
    active = {int(getattr(p, "agentID")) for p in core.pedDS.pedAgentList.values()
              if not bool(getattr(p, "terminated", False))}
    remaining = estimate_remaining_time_to_safety(
        core, episode_horizon=horizon, pedestrian_ids=active, include_panicked=True, include_casualties=False,
    )
    result.remaining_plain_person_time = float(remaining.plain_person_minutes)
    result.remaining_hazard_exposure_person_time = float(remaining.hazard_exposure_person_minutes)
    tail_weight = float(getattr(core, "clearanceTailExtraWeight", 0.0))
    if tail_weight > 0.0:
        tail = estimate_clearance_tail(core, episode_horizon=horizon,
                                       initial_population=int(getattr(core, "initial_population", 1)),
                                       remaining=remaining)
        result.clearance_tail_person_time = float(tail_weight * tail.tail_person_minutes)
    return float(result.hazard_weighted_time_to_safety_cost)


@dataclass
class StoppingEvaluation:
    offsets: tuple
    mean_cost: dict
    costs: dict
    best: int
    futures: int
    steps: int

    def as_record(self) -> dict:
        return {
            "offsets": list(self.offsets),
            "mean_cost": {str(k): float(v) for k, v in self.mean_cost.items()},
            "best_offset": int(self.best),
            "futures": int(self.futures),
            "steps": int(self.steps),
        }


def evaluate_stopping(
    core,
    cells: Sequence,
    *,
    offsets: Sequence[int],
    futures: int,
    horizon_minutes: int,
    seed: int,
) -> StoppingEvaluation:
    """Mean look-ahead cost of ordering ``cells`` at each offset, or never."""
    step_minutes = max(1e-9, float(getattr(core, "timeStepMinutes", 1.0)))
    now = int(getattr(core.pedDS, "currTime", 0))
    remaining = max(0, int(getattr(core, "stopTime", 0)) - 1 - now)
    steps = max(1, min(int(round(horizon_minutes / step_minutes)), remaining))
    alternatives = tuple(sorted({int(o) for o in offsets if 0 <= int(round(o / step_minutes)) < steps})) + (NEVER,)
    snapshot = CB.capture(core, label="recourse-stopping")
    before = CB.live_state_digest(core)
    costs = {alt: [] for alt in alternatives}
    try:
        for member in range(int(futures)):
            future_seed = int(np.random.SeedSequence([int(seed), int(member)]).generate_state(1)[0])
            timeline = belief_timeline(core, snapshot, future_seed, steps)
            for alt in alternatives:
                CB.restore(core, snapshot)
                reseed_exogenous_streams(core, future_seed)
                schedule = {} if alt == NEVER else {int(round(alt / step_minutes)) + 1: tuple(cells)}
                costs[alt].append(branch_cost(core, timeline=timeline, schedule=schedule, steps=steps))
    finally:
        CB.restore(core, snapshot)
    if CB.live_state_digest(core) != before:
        raise RuntimeError("recourse stopping evaluation mutated the live simulator")
    mean = {alt: float(np.mean(values)) for alt, values in costs.items()}
    best = min(alternatives, key=lambda alt: (mean[alt], alt == 0))  # a tie never orders now
    return StoppingEvaluation(offsets=alternatives, mean_cost=mean, costs=costs, best=int(best),
                              futures=int(futures), steps=int(steps))
