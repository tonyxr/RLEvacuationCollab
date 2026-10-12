#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Natural-momentum counterfactual branching under common random numbers.

This module implements the NMCC Stage-0 prerequisites for the cell-priority
action space, plus the paired-branch effect estimator built on top of them.

The problem
-----------
A realized trajectory reports ``G(a, U)``: the return after action ``a`` under
one realization ``U`` of every uncontrolled disturbance -- hazard spread,
casualty shocks, panic onset, panicked route choice.  A policy gradient
computed from that single number cannot tell whether a low return came from
the chosen cell or from ``U``.  In this simulator the uncontrolled part
dominates: the 2026-09-19 validation measured a per-episode total-return
standard deviation of 0.2226 against a held-out RL-minus-heuristic difference
of -0.00596.  The quantity being estimated is roughly forty times smaller than
the noise it is embedded in, which is why the policy converged to agreement
with its own prior rather than to an improvement over it.

The remedy is to isolate the controllable intervention. Every feasible shelter
and ``WAIT`` start from one exact captured state and receive the same hazard,
panic, and casualty random future. Candidate ranking compares their signed
effects within that state. Under common random numbers,

    Var[G(a, U) - G(b, U)] = Var[G(a)] + Var[G(b)] - 2 Cov[G(a), G(b)],

and because both branches ride the same hazard front and movement dynamics,
that covariance sits close to the variances and the difference variance
collapses. This is the classical
common-random-numbers variance reduction of Kleinman, Spall and Naiman (1999)
used as a within-state control variate. Centering the complete action table on
its uniform feasible-site mean produces the random-installation baseline
without mixing natural system evolution into the action contrast.

Why this is affordable
----------------------
Simulator-graded policy iteration does run selected branches to the episode
end. Three properties keep the comparison tractable:

1. **Coverage panels.** Each state grades the frozen actor's strongest cells
   plus the least-evaluated feasible cells. Persistent counts rotate neglected
   cells into later panels; a zero panel budget requests exhaustive grading.
2. **Shared exogenous evolution.** Each hazard future is advanced once and
   reproduced for every candidate through keyed common random numbers, while
   independent candidate branches run concurrently in simulator-only workers.
3. **Matched route maintenance.** ``WAIT`` and every action both repair stale
   routes before evolving, so ordinary routing maintenance cannot be credited
   to a shelter installation.

Validity
--------
Paired branches are valid only if both branches see the same disturbances.
This module provides exact snapshot/restore, and the accompanying tests prove
the two properties NMCC names as prerequisites: factual replay from a restored
snapshot is bitwise identical, and branch order changes neither branch.

The simulator was already most of the way there.  ``PedestrianDatabase``
addresses casualty, panic-susceptibility and panicked-route-choice shocks by
immutable keys (episode seed, mechanism, pedestrian identifier, timestep), so
those draws are already invariant to how many people another branch has
already evacuated.  Hazard evolution consumes a sequential generator, but
``HazardDatabase._spreadUpdateStochastic`` reads only hazard state and cell
states -- never shelters or pedestrians -- so hazard is exogenous and its draw
sequence is reproduced exactly by restoring the generator state.
:func:`assert_hazard_is_action_independent` checks that claim on the live
object graph rather than trusting this paragraph.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import math
import os
import pickle
import random
import select
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import numpy as np

from CAProcessor import authoritative_grid_shape
from DecisionInterface import (
    OutcomeSnapshot,
    RegionalObservationBuilder,
    RemainingTimeEstimate,
    estimate_clearance_tail,
    estimate_remaining_time_to_safety,
)
from RolloutPolicyIteration import FrozenResidualPolicy


# Attributes of ``Core`` that carry branch-varying simulation state.  Anything
# listed here is deep-copied into a snapshot; anything not listed is either
# static infrastructure (the road graph) or belongs to the learner rather than
# the simulated world (``rl``, loggers, the visualizer).
STATE_ATTRIBUTES = (
    "pedDS",
    "hazardDS",
    "shelterDS",
    "cellTracker",
    "forceTracker",
    "guidanceDS",
)

# Mutable per-timestep fields that live on the *shared* map objects.  These are
# captured as flat arrays instead of by deep copy, which is what keeps a branch
# in the millisecond range on a full OSM graph.
NODE_FLOW_FIELDS = ("nodeFlow",)
EDGE_FLOW_FIELDS = (
    "edgeFlow",
    "congestionOccupancy",
    "congestionDensityPedPerM2",
    "congestionSpeedRatio",
)


def _iter_map_nodes(core) -> list:
    """Return every ``Node`` object reachable from the map store, in a stable order."""
    map_store = getattr(core, "mapDS", None)
    if map_store is None:
        return []
    by_local = getattr(map_store, "nodeListByLocalID", None)
    if isinstance(by_local, dict) and by_local:
        return [by_local[key] for key in sorted(by_local)]
    raw = getattr(map_store, "rawNodeList", None) or []
    return list(raw)


def _iter_map_edges(core) -> list:
    """Return every ``Edge`` object reachable from the map store, in a stable order."""
    map_store = getattr(core, "mapDS", None)
    if map_store is None:
        return []
    by_local = getattr(map_store, "edgeListByLocalID", None)
    if isinstance(by_local, dict) and by_local:
        return [by_local[key] for key in sorted(by_local)]
    raw = getattr(map_store, "rawEdgeList", None) or []
    return list(raw)


def _shared_memo(core) -> dict[int, Any]:
    """Seed a ``deepcopy`` memo with the objects branches deliberately share.

    The road graph is static infrastructure.  Its topology, geometry and the
    derived routing caches (``_routing_tree_to_target_cache`` and friends) are
    pure functions of the map and a target node, so sharing them across
    branches is not merely an optimization -- a cache entry computed inside one
    branch is exactly the entry the other branch would have computed.  Sharing
    them is what keeps branch cost proportional to the population rather than
    to the size of the city.

    Pedestrians hold direct references to ``Node`` and ``Edge`` objects, so
    every one of them must be seeded individually; seeding only ``mapDS`` would
    let ``deepcopy`` clone the graph by following a pedestrian's route.
    """
    memo: dict[int, Any] = {}
    for obj in (
        getattr(core, "mapDS", None),
        getattr(core, "OSMProcessor", None),
        getattr(core, "congestionModel", None),
        getattr(getattr(core, "mapDS", None), "locationDrive", None),
    ):
        if obj is not None:
            memo[id(obj)] = obj
    for node in _iter_map_nodes(core):
        memo[id(node)] = node
    for edge in _iter_map_edges(core):
        memo[id(edge)] = edge
    return memo


def _capture_map_flow(nodes: Sequence, edges: Sequence) -> tuple[np.ndarray, np.ndarray]:
    node_state = np.asarray(
        [[float(getattr(node, name, 0.0) or 0.0) for name in NODE_FLOW_FIELDS] for node in nodes],
        dtype=np.float64,
    ).reshape(len(nodes), len(NODE_FLOW_FIELDS))
    edge_state = np.asarray(
        [[float(getattr(edge, name, 0.0) or 0.0) for name in EDGE_FLOW_FIELDS] for edge in edges],
        dtype=np.float64,
    ).reshape(len(edges), len(EDGE_FLOW_FIELDS))
    return node_state, edge_state


def _restore_map_flow(
    nodes: Sequence,
    edges: Sequence,
    node_state: np.ndarray,
    edge_state: np.ndarray,
) -> None:
    for index, node in enumerate(nodes):
        for offset, name in enumerate(NODE_FLOW_FIELDS):
            setattr(node, name, type(getattr(node, name, 0.0))(node_state[index, offset]))
    for index, edge in enumerate(edges):
        for offset, name in enumerate(EDGE_FLOW_FIELDS):
            current = getattr(edge, name, 0.0)
            setattr(edge, name, type(current)(edge_state[index, offset]))


@dataclass
class SimulatorSnapshot:
    """An exact, restorable capture of every branch-varying quantity.

    A snapshot deliberately excludes the learner (``core.rl``), the training
    logger and the visualizer.  A counterfactual branch must not append to the
    PPO rollout, advance the recurrent state, or write a decision row: it is a
    question asked of the world, not an experience the agent lived through.
    """

    state: dict[str, Any]
    node_flow: np.ndarray
    edge_flow: np.ndarray
    hazard_bit_generator_state: Optional[dict]
    python_random_state: tuple
    numpy_legacy_state: tuple
    simulation_time: int
    grid_shape: tuple[int, int]
    label: str = ""

    def state_digest(self) -> str:
        """Content hash of the branch-varying state, for replay audits."""
        hasher = hashlib.sha256()
        hasher.update(self.node_flow.tobytes())
        hasher.update(self.edge_flow.tobytes())
        hasher.update(repr(self.hazard_bit_generator_state).encode("utf-8"))
        hasher.update(str(self.simulation_time).encode("utf-8"))
        hasher.update(np.asarray(self.grid_shape, dtype=np.int64).tobytes())
        return hasher.hexdigest()


@dataclass
class HazardFrame:
    """Exact exogenous hazard state after one future timestep.

    Shelter placement cannot alter hazard evolution.  NMCC therefore evolves
    each common-random-number future once, stores these compact frames, and
    replays them in every candidate branch.  Pedestrian, shelter, congestion,
    and routing state remain candidate-specific and continue to evolve through
    the full simulator.
    """

    hazard_list: dict[Any, Any]
    hazard_bit_generator_state: Optional[dict]
    grid_shape: tuple[int, int]
    cell_state: np.ndarray
    cell_heat: np.ndarray
    cell_smoke: np.ndarray


def _capture_hazard_frame(core) -> HazardFrame:
    grid_shape = authoritative_grid_shape(core)
    cells = [
        core.cellTracker.cellList[key]
        for key in core.cellTracker.cell_ids()
    ]
    hazard_rng = getattr(getattr(core, "hazardDS", None), "rng", None)
    rng_state = None
    if hazard_rng is not None and hasattr(hazard_rng, "bit_generator"):
        rng_state = copy.deepcopy(hazard_rng.bit_generator.state)
    return HazardFrame(
        hazard_list=copy.deepcopy(core.hazardDS.hazardList, _shared_memo(core)),
        hazard_bit_generator_state=rng_state,
        grid_shape=grid_shape,
        cell_state=np.asarray(
            [int(getattr(cell, "impactedLevel", 0)) for cell in cells],
            dtype=np.int16,
        ),
        cell_heat=np.asarray(
            [float(getattr(cell, "heat", 0.0)) for cell in cells],
            dtype=np.float64,
        ),
        cell_smoke=np.asarray(
            [float(getattr(cell, "smoke", 0.0)) for cell in cells],
            dtype=np.float64,
        ),
    )


def _apply_hazard_frame(core, frame: HazardFrame) -> None:
    """Install one precomputed hazard frame without touching candidate state."""
    grid_shape = authoritative_grid_shape(core)
    if tuple(frame.grid_shape) != grid_shape:
        raise RuntimeError(
            "NMCC hazard replay grid changed inside a branch: "
            f"frame={tuple(frame.grid_shape)}, live={grid_shape}"
        )
    core.hazardDS.hazardList = copy.deepcopy(
        frame.hazard_list,
        _shared_memo(core),
    )
    hazard_rng = getattr(core.hazardDS, "rng", None)
    if hazard_rng is not None and frame.hazard_bit_generator_state is not None:
        hazard_rng.bit_generator.state = copy.deepcopy(
            frame.hazard_bit_generator_state
        )
    cells = [
        core.cellTracker.cellList[key]
        for key in core.cellTracker.cell_ids()
    ]
    expected = int(core.cellTracker.number_of_cells)
    frame_shapes = {
        "state": np.asarray(frame.cell_state).shape,
        "heat": np.asarray(frame.cell_heat).shape,
        "smoke": np.asarray(frame.cell_smoke).shape,
    }
    if any(shape != (expected,) for shape in frame_shapes.values()):
        raise RuntimeError(
            "NMCC hazard replay arrays do not match the exact branch grid: "
            f"grid={grid_shape}, arrays={frame_shapes}"
        )
    for index, cell in enumerate(cells):
        cell.impactedLevel = int(frame.cell_state[index])
        cell.heat = float(frame.cell_heat[index])
        cell.smoke = float(frame.cell_smoke[index])


def capture(core, *, label: str = "") -> SimulatorSnapshot:
    """Capture the simulator's branch-varying state without disturbing it."""
    grid_shape = authoritative_grid_shape(core)
    memo = _shared_memo(core)
    state: dict[str, Any] = {}
    for name in STATE_ATTRIBUTES:
        value = getattr(core, name, None)
        if value is None:
            continue
        state[name] = copy.deepcopy(value, memo)

    nodes = _iter_map_nodes(core)
    edges = _iter_map_edges(core)
    node_flow, edge_flow = _capture_map_flow(nodes, edges)

    hazard_store = getattr(core, "hazardDS", None)
    hazard_rng = getattr(hazard_store, "rng", None)
    hazard_state = None
    if hazard_rng is not None and hasattr(hazard_rng, "bit_generator"):
        hazard_state = copy.deepcopy(hazard_rng.bit_generator.state)

    return SimulatorSnapshot(
        state=state,
        node_flow=node_flow,
        edge_flow=edge_flow,
        hazard_bit_generator_state=hazard_state,
        python_random_state=random.getstate(),
        numpy_legacy_state=np.random.get_state(),
        simulation_time=int(getattr(getattr(core, "pedDS", None), "currTime", 0)),
        grid_shape=grid_shape,
        label=str(label),
    )


def restore(core, snapshot: SimulatorSnapshot) -> None:
    """Restore a captured state, leaving the snapshot reusable.

    The stored objects are copied on the way out as well as on the way in, so
    one snapshot can seed any number of branches.  Handing out the stored
    objects directly would let the first branch mutate the snapshot and would
    silently invalidate every later comparison drawn from it.
    """
    memo = _shared_memo(core)
    for name, value in snapshot.state.items():
        setattr(core, name, copy.deepcopy(value, memo))

    _restore_map_flow(
        _iter_map_nodes(core),
        _iter_map_edges(core),
        snapshot.node_flow,
        snapshot.edge_flow,
    )

    hazard_store = getattr(core, "hazardDS", None)
    hazard_rng = getattr(hazard_store, "rng", None)
    if hazard_rng is not None and snapshot.hazard_bit_generator_state is not None:
        hazard_rng.bit_generator.state = copy.deepcopy(snapshot.hazard_bit_generator_state)

    # The hazard store keeps its own pointer to the cell tracker; after a
    # restore both objects are new, and a stale pointer would let the hazard
    # spread into a grid nobody else can see.
    if hazard_store is not None and hasattr(hazard_store, "cellTracker"):
        restored_tracker = getattr(core, "cellTracker", None)
        if restored_tracker is not None:
            setter = getattr(hazard_store, "setCellTracker", None)
            if callable(setter):
                setter(restored_tracker)
            else:
                hazard_store.cellTracker = restored_tracker

    restored_shape = authoritative_grid_shape(core)
    if restored_shape != tuple(snapshot.grid_shape):
        raise RuntimeError(
            "Restored NMCC snapshot changed cell-grid dimensionality: "
            f"snapshot={tuple(snapshot.grid_shape)}, restored={restored_shape}"
        )

    random.setstate(snapshot.python_random_state)
    np.random.set_state(snapshot.numpy_legacy_state)


def live_state_digest(core) -> str:
    """Hash the live simulator state; two runs agree only if they truly match.

    This covers the quantities a branch can change: per-person position,
    progress, wellness and terminal status; shelter occupancy; cell hazard
    level; and the cumulative outcome ledger.  It is the instrument behind the
    bitwise-replay and branch-order-invariance audits.
    """
    hasher = hashlib.sha256()
    ped_store = getattr(core, "pedDS", None)
    if ped_store is not None:
        hasher.update(str(int(getattr(ped_store, "currTime", 0))).encode("utf-8"))
        for key in sorted(getattr(ped_store, "result", {})):
            hasher.update(f"{key}={ped_store.result[key]}".encode("utf-8"))
        agents = getattr(ped_store, "pedAgentList", {})
        for key in sorted(agents):
            ped = agents[key]
            for name in (
                "lastX",
                "lastY",
                "currSpeed",
                "terminated",
                "panicked",
                "affected",
                "group_size",
            ):
                hasher.update(f"{name}={getattr(ped, name, None)!r}".encode("utf-8"))
    shelter_store = getattr(core, "shelterDS", None)
    if shelter_store is not None:
        for key in sorted(getattr(shelter_store, "shelterList", {})):
            shelter = shelter_store.shelterList[key]
            hasher.update(
                f"{key}:{getattr(shelter, 'shelterCap', 0)}:"
                f"{getattr(shelter, 'shelterFlow', 0)}:"
                f"{getattr(shelter, 'status', 0)}".encode("utf-8")
            )
        for pedestrian_id, reservation in sorted(
            getattr(shelter_store, "reservationByPedestrian", {}).items()
        ):
            hasher.update(
                f"reservation:{pedestrian_id}:{reservation}".encode("utf-8")
            )
    tracker = getattr(core, "cellTracker", None)
    if tracker is not None:
        danger = np.asarray(getattr(tracker, "dangerLevelByCell", []), dtype=np.float64)
        counts = np.asarray(getattr(tracker, "countByCell", []), dtype=np.float64)
        hasher.update(danger.tobytes())
        hasher.update(counts.tobytes())
    return hasher.hexdigest()


def advance_one_timestep(core, *, hazard_frame: Optional[HazardFrame] = None) -> None:
    """Advance the simulated world one timestep, with no learner involvement.

    This mirrors ``Core.simulationEnumerator`` exactly, minus three things that
    must not happen inside a counterfactual: ``rl.step`` (which would append to
    the PPO rollout and advance the recurrent state), ``_record_hazard_state``
    (which would poison the episode's hazard-trajectory digest that the matched
    backtest relies on), and logging.  The ordering below is load-bearing --
    hazard resolves before pedestrians react to it, and outcomes are committed
    before the decision boundary is observed.
    """
    core.pedDS.startDocument()
    if hazard_frame is None:
        core.hazardDS.spreadUpdate()
        core.hazardDS.heatUpdate()
        core.hazardDS.smokeUpdate()
        core.hazardDS.terminateHazard()
    else:
        _apply_hazard_frame(core, hazard_frame)
    core.pedDS.loadShelterLookup(core.shelterDS.shelterByOSMID)
    core.pedDS.pedestrianHazardInteraction()
    core.pedDS.interPedestrianInteraction()
    core.pedDS.pedestrianNetworkInteraction()
    core.cellTracker.cellUpdate(pedDS=core.pedDS, forceTracker=core.forceTracker)
    core.pedDS.docuStatus()


def outcome_snapshot(core) -> OutcomeSnapshot:
    """Read the reward-relevant outcome vector straight off the simulator.

    Deliberately avoids building a full ``RegionalObservation``: the candidate
    feature block costs ``O(cells * pedestrians)`` and none of it is needed to
    score a branch.  ``test_branch_outcomes_match_the_observation_builder``
    pins these values against the builder so the shortcut cannot drift.
    """
    results = getattr(core.pedDS, "result", {})
    arrivals = max(0, int(results.get("arrival", 0)))
    shelter_evacuated = max(0, int(results.get("evacuated", 0)))
    casualties = max(0, int(results.get("casualty", 0)))

    tracker = core.cellTracker
    counts = np.asarray(getattr(tracker, "countByCell", []), dtype=np.float64)
    danger = np.asarray(getattr(tracker, "dangerLevelByCell", []), dtype=np.float64)
    danger = np.clip(np.nan_to_num(danger, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    active_population = int(round(float(counts.sum())))
    risk_mass = float(np.sum(counts * (1.0 + danger)))

    return OutcomeSnapshot(
        safe_completed=arrivals + shelter_evacuated,
        casualties=casualties,
        shelter_evacuated=shelter_evacuated,
        ordinary_arrivals=arrivals,
        active_population=active_population,
        risk_mass=risk_mass,
    )


def casualty_loss_weight(core, simulation_time: Optional[int] = None) -> float:
    """Person-minute price of one expected casualty at this timestep."""
    horizon = max(1, int(getattr(getattr(core, "rl", None), "horizon", 0) or getattr(core, "stopTime", 1) - 1))
    time_index = int(
        getattr(getattr(core, "pedDS", None), "currTime", 0)
        if simulation_time is None
        else simulation_time
    )
    remaining = max(0, horizon - time_index)
    kappa = float(getattr(core, "casualtyPriorityKappa", 2.0))
    if not math.isfinite(kappa) or kappa < 1.0:
        raise ValueError("casualtyPriorityKappa must be finite and at least 1")
    return float(kappa * 2.0 * remaining)


def expected_casualty_step_cost(
    core,
    *,
    simulation_time: Optional[int] = None,
) -> float:
    """Return L(t) * sum_i g_i p_i for the most recent physical step."""
    intensity = float(
        getattr(
            getattr(core, "pedDS", None),
            "expectedCasualtyIntensityStep",
            0.0,
        )
    )
    if not math.isfinite(intensity) or intensity < 0.0:
        raise ValueError("Expected casualty intensity must be finite and non-negative")
    return float(intensity * casualty_loss_weight(core, simulation_time))


def controllable_time_mass(
    core,
    pedestrian_ids: set[int],
) -> tuple[float, float, float]:
    """Complete plain, danger and expected-casualty mass for one cohort.

    Every person remains in the ledger until safe. Panicked pedestrians keep
    accruing active and exposure time. Casualty draws do not enter the scalar
    objective; their exact conditional intensity is priced before the draw.
    Common-random-number branches difference shared exogenous evolution.
    """
    danger = np.asarray(
        getattr(core.cellTracker, "dangerLevelByCell", ()), dtype=np.float64
    ).reshape(-1)
    danger = np.clip(
        np.nan_to_num(danger, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    _, cell_y = authoritative_grid_shape(core)
    plain = 0.0
    exposure = 0.0
    for pedestrian in getattr(core.pedDS, "pedAgentList", {}).values():
        pedestrian_id = int(getattr(pedestrian, "agentID", -1))
        if pedestrian_id not in pedestrian_ids or bool(
            getattr(pedestrian, "terminated", False)
        ):
            continue
        population = max(1, int(getattr(pedestrian, "group_size", 1)))
        plain += population
        try:
            i, j = getattr(pedestrian, "currCell")
            index = int(i) * cell_y + int(j)
            local_danger = float(danger[index]) if 0 <= index < danger.size else 0.0
        except (TypeError, ValueError, IndexError):
            local_danger = 0.0
        exposure += population * local_danger
    casualty_cost = expected_casualty_step_cost(core)
    return float(plain), float(exposure), float(casualty_cost)


@dataclass
class BranchResult:
    """One branch's realized trajectory over the counterfactual horizon."""

    label: str
    rewards: list[float] = field(default_factory=list)
    reward_components: list[np.ndarray] = field(default_factory=list)
    active_person_time: float = 0.0
    hazard_exposure_person_time: float = 0.0
    expected_casualty_person_time: float = 0.0
    remaining_plain_person_time: float = 0.0
    remaining_hazard_exposure_person_time: float = 0.0
    terminal_residual_person_time: float = 0.0
    clearance_tail_person_time: float = 0.0
    clearance_tail_mean_minutes: float = 0.0
    clearance_t90_minutes: float = 0.0
    clearance_time_minutes: float = 0.0
    start_outcome: Optional[OutcomeSnapshot] = None
    end_outcome: Optional[OutcomeSnapshot] = None
    installed_cell: int = -1
    installed_capacity: float = 0.0
    steps: int = 0
    terminal: bool = False
    end_digest: str = ""
    # Learned return from the endpoint to the remaining deployment decisions.
    # CounterfactualBranch does not define this value; the NMCC learner supplies
    # an endpoint callback so physical branching and learned continuation remain
    # separate, auditable parts of the Bellman target.
    continuation_value: float = 0.0
    # Fixed uniform-reference-policy cost-to-go at the same endpoint.  Unlike
    # ``continuation_value`` this is learned only in Stage E and frozen.
    random_reference_value: float = 0.0
    # Optional compact observation captured at the branch endpoint.  It is
    # deliberately not simulator state: isolated physical branches return this
    # payload to the parent, which evaluates the shared continuation network in
    # one batch without running PyTorch in child processes.
    endpoint_payload: Any = None

    @property
    def plain_time_to_safety_cost(self) -> float:
        return float(self.active_person_time + self.remaining_plain_person_time)

    @property
    def hazard_exposure_to_safety_cost(self) -> float:
        return float(
            self.hazard_exposure_person_time
            + self.remaining_hazard_exposure_person_time
        )

    @property
    def hazard_weighted_time_to_safety_cost(self) -> float:
        return float(
            self.plain_time_to_safety_cost
            + self.hazard_exposure_to_safety_cost
            + self.expected_casualty_person_time
            + self.clearance_tail_person_time
        )

    @property
    def interval_hazard_weighted_person_time(self) -> float:
        """Realized plain plus danger-weighted person-time in this branch."""
        return float(
            self.active_person_time
            + self.hazard_exposure_person_time
            + self.expected_casualty_person_time
        )

    def discounted_return(self, gamma: float = 1.0) -> float:
        total = 0.0
        for index, value in enumerate(self.rewards):
            total += (gamma ** index) * float(value)
        return float(total)

    def discounted_component_return(self, gamma: float = 1.0) -> np.ndarray:
        """Discounted signed reward components in the registered order."""
        if not self.reward_components:
            return np.zeros(1, dtype=np.float64)
        total = np.zeros_like(
            np.asarray(self.reward_components[0], dtype=np.float64)
        )
        for index, value in enumerate(self.reward_components):
            total += (float(gamma) ** index) * np.asarray(value, dtype=np.float64)
        return total

    def outcome_vector(self) -> dict[str, float]:
        """Return physical branch outcomes separately from scalar reward.

        Casualties remain an explicit audit coordinate. There is no standalone
        casualty coefficient; failure to reach safety is represented in the
        time-to-safety cost instead.
        """
        start = self.start_outcome
        end = self.end_outcome
        if start is None or end is None:
            return {}
        return {
            "safe_completions": float(end.safe_completed - start.safe_completed),
            "casualties": float(end.casualties - start.casualties),
            "active_person_time": float(self.active_person_time),
            "hazard_exposure_person_time": float(self.hazard_exposure_person_time),
            "expected_casualty_person_time": float(
                self.expected_casualty_person_time
            ),
            "remaining_plain_person_time": float(
                self.remaining_plain_person_time
            ),
            "remaining_hazard_exposure_person_time": float(
                self.remaining_hazard_exposure_person_time
            ),
            "terminal_residual_person_time": float(
                self.terminal_residual_person_time
            ),
            "clearance_tail_person_time": float(
                self.clearance_tail_person_time
            ),
            "clearance_tail_mean_minutes": float(
                self.clearance_tail_mean_minutes
            ),
            "clearance_t90_minutes": float(self.clearance_t90_minutes),
            "clearance_time_minutes": float(self.clearance_time_minutes),
            "final_active_population": float(end.active_population),
            "final_risk_mass": float(end.risk_mass),
        }

    def normalized_outcome_vector(
        self,
        *,
        initial_population: int,
        episode_horizon: int,
    ) -> np.ndarray:
        """Return the auditable six-coordinate NMCC physical target.

        Every coordinate has a fixed population/horizon scale, so the natural
        model and intervention ensemble can be shared across populations and
        region resolutions. Final risk mass is bounded by twice the active
        population under the observation contract and is normalized likewise.
        """
        population = max(1, int(initial_population))
        horizon = max(1, int(episode_horizon))
        values = self.outcome_vector()
        if not values:
            return np.zeros(6, dtype=np.float32)
        return np.asarray(
            (
                values["safe_completions"] / population,
                values["casualties"] / population,
                values["active_person_time"] / (population * horizon),
                values["hazard_exposure_person_time"] / (population * horizon),
                values["final_active_population"] / population,
                values["final_risk_mass"] / (2.0 * population),
            ),
            dtype=np.float32,
        )


@dataclass
class PairedEffect:
    """The causal contribution of one deployment, measured against ``WAIT``."""

    cell_index: int
    horizon: int
    acted: BranchResult
    waited: BranchResult
    common_noise: bool = True

    @property
    def reward_difference(self) -> float:
        return self.acted.discounted_return() - self.waited.discounted_return()

    def discounted_difference(self, gamma: float = 1.0) -> float:
        return self.acted.discounted_return(gamma) - self.waited.discounted_return(gamma)

    def outcome_difference(self) -> dict[str, float]:
        acted = self.acted.outcome_vector()
        waited = self.waited.outcome_vector()
        return {key: acted[key] - waited.get(key, 0.0) for key in acted}

    def reward_breakdown(
        self,
        reward_model,
        *,
        initial_population: int,
        episode_horizon_minutes: float,
    ):
        """Return positive WAIT-minus-action exposure-removal reward."""
        return reward_model.evaluate_counterfactual(
            start=self.acted.start_outcome,
            action_after=self.acted.end_outcome,
            plain_time_wait=self.waited.plain_time_to_safety_cost,
            plain_time_action=self.acted.plain_time_to_safety_cost,
            hazard_exposure_wait=(
                self.waited.hazard_exposure_to_safety_cost
            ),
            hazard_exposure_action=(
                self.acted.hazard_exposure_to_safety_cost
            ),
            initial_population=initial_population,
            horizon_minutes=episode_horizon_minutes,
        )


@dataclass(frozen=True)
class _ImmediateShelter:
    shelter_id: int
    node: Any
    capacity: int
    flow: int


class _ImmediateCandidateEvaluator:
    """Exact zero-horizon assignment and route-cost evaluator.

    The dynamic brancher is required when time advances.  At horizon zero,
    however, copying thousands of pedestrian objects once per candidate adds
    no information: the only state transition is the deterministic capacity
    assignment.  This evaluator executes that same finite transfer algorithm
    on local arrays and evaluates the same shortest-route exposure equation,
    without mutating the simulator.
    """

    def __init__(self, brancher: "CounterfactualBrancher"):
        self.brancher = brancher
        self.core = brancher.core
        self.map_store = self.core.mapDS
        self.shelter_store = self.core.shelterDS
        all_active = tuple(
            pedestrian
            for pedestrian in self.core.pedDS.pedAgentList.values()
            if not bool(getattr(pedestrian, "terminated", False))
        )
        self.pedestrians = sorted(
            (
                pedestrian
                for pedestrian in all_active
                if not bool(getattr(pedestrian, "panicked", False))
            ),
            key=lambda pedestrian: int(pedestrian.agentID),
        )
        panicked_ids = {
            int(getattr(pedestrian, "agentID", -1))
            for pedestrian in all_active
            if bool(getattr(pedestrian, "panicked", False))
        }
        self.panicked_population = sum(
            max(1, int(getattr(pedestrian, "group_size", 1)))
            for pedestrian in all_active
            if bool(getattr(pedestrian, "panicked", False))
        )
        self.casualty_population = max(
            0,
            int(getattr(self.core.pedDS, "result", {}).get("casualty", 0)),
        )
        common_debt = estimate_remaining_time_to_safety(
            self.core,
            episode_horizon=int(self.brancher.episode_horizon),
            pedestrian_ids=panicked_ids,
            include_panicked=True,
            include_casualties=False,
        )
        self.common_plain_person_time = float(common_debt.plain_person_minutes)
        self.common_hazard_exposure_person_time = float(
            common_debt.hazard_exposure_person_minutes
        )
        self.common_plain_minutes_by_person = np.asarray(
            common_debt.plain_minutes_by_person, dtype=np.float64
        )
        self.population = np.asarray(
            [
                max(1, int(getattr(pedestrian, "group_size", 1)))
                for pedestrian in self.pedestrians
            ],
            dtype=np.int64,
        )
        self.rational_population = int(self.population.sum())
        results = getattr(self.core.pedDS, "result", {})
        safe_population = max(0, int(results.get("arrival", 0))) + max(
            0, int(results.get("evacuated", 0))
        )
        ledger_population = (
            self.rational_population
            + int(self.panicked_population)
            + int(self.casualty_population)
            + int(safe_population)
        )
        if ledger_population != int(self.brancher.initial_population):
            raise RuntimeError(
                "Immediate candidate evaluation requires a complete population "
                "ledger: rational active + panicked active + casualties + safe "
                f"was {ledger_population}, expected {self.brancher.initial_population}"
            )
        self.pedestrian_ids = np.asarray(
            [int(pedestrian.agentID) for pedestrian in self.pedestrians],
            dtype=np.int64,
        )
        self.anchors = [
            self.core.pedDS._route_anchor_node(pedestrian)
            for pedestrian in self.pedestrians
        ]
        if any(anchor is None for anchor in self.anchors):
            raise RuntimeError("Every rational active pedestrian needs a routing anchor")

        self.base_shelters = tuple(
            _ImmediateShelter(
                shelter_id=int(shelter.shelterID),
                node=shelter.nodeMapped,
                capacity=int(shelter.shelterCap),
                flow=int(shelter.shelterFlow),
            )
            for shelter in sorted(
                self.shelter_store.shelterList.values(),
                key=lambda value: int(value.shelterID),
            )
            if int(getattr(shelter, "status", 0)) == 0
            and getattr(shelter, "nodeMapped", None) is not None
        )
        self.base_by_id = {
            shelter.shelter_id: shelter for shelter in self.base_shelters
        }
        self.distance_by_node: dict[int, np.ndarray] = {}
        self.route_cost_by_pedestrian_and_node: dict[
            tuple[int, int], tuple[float, float]
        ] = {}
        self.path_metric_by_anchor_and_node: dict[
            tuple[int, int], tuple[float, float]
        ] = {}
        self.start_outcome = outcome_snapshot(self.core)
        self.state_digest = live_state_digest(self.core)
        self._prepare_danger()
        self.base_distances = {
            shelter.shelter_id: self._distance_column(shelter.node)
            for shelter in self.base_shelters
        }
        if self.base_shelters:
            for index in range(len(self.pedestrians)):
                if not any(
                    np.isfinite(distances[index])
                    for distances in self.base_distances.values()
                ):
                    raise RuntimeError(
                        f"Active pedestrian {self.pedestrian_ids[index]} cannot reach "
                        "an operational shelter"
                    )
        self.base_assignment = self._valid_current_assignments()
        self.wait_assignment = self._allocate(
            self.base_shelters,
            self.base_distances,
        )
        self.wait_result = self._branch_result(
            label="waited",
            assignment=self.wait_assignment,
        )

    def _prepare_danger(self) -> None:
        cell_x, cell_y = authoritative_grid_shape(self.core)
        self.cell_x = cell_x
        self.cell_y = cell_y
        cell_count = cell_x * cell_y
        current = np.asarray(
            getattr(
                self.core.cellTracker,
                "dangerLevelByCell",
                np.zeros(cell_count),
            ),
            dtype=np.float64,
        ).reshape(cell_count)
        current = np.clip(
            np.nan_to_num(current, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        forecast = current
        forecast_fn = getattr(self.core.hazardDS, "forecast_danger_by_cell", None)
        if callable(forecast_fn):
            forecast = np.asarray(
                forecast_fn(
                    max(1, int(getattr(self.core, "shelterActionInterval", 1)))
                ),
                dtype=np.float64,
            ).reshape(cell_count)
            forecast = np.clip(
                np.nan_to_num(forecast, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
        self.danger = np.maximum(current, forecast)
        self.time_step_minutes = max(
            1e-9, float(getattr(self.core, "timeStepMinutes", 1.0))
        )
        current_time = max(
            0,
            int(getattr(getattr(self.core, "pedDS", None), "currTime", 0)),
        )
        self.horizon_minutes = (
            float(max(0, self.brancher.episode_horizon - current_time))
            * self.time_step_minutes
        )
        maximum_speed = max(1e-9, float(getattr(self.core, "maxSpeed", 1.0)))
        self.speed_floor = max(0.5, 0.05 * maximum_speed)

    def _cell_danger(self, cell) -> float:
        if cell is None:
            return 0.0
        try:
            i, j = int(cell[0]), int(cell[1])
        except (TypeError, ValueError, IndexError):
            return 0.0
        if not (0 <= i < self.cell_x and 0 <= j < self.cell_y):
            return 0.0
        return float(self.danger[i * self.cell_y + j])

    def _distance_column(self, node) -> np.ndarray:
        node_id = int(node.OSMID)
        cached = self.distance_by_node.get(node_id)
        if cached is not None:
            return cached
        routing_tree = getattr(self.map_store, "_routing_tree", None)
        if callable(routing_tree):
            distances, _ = routing_tree(node_id)
            values = np.asarray(
                [
                    float(distances.get(int(anchor.OSMID), float("inf")))
                    if int(anchor.OSMID) != node_id
                    else 0.0
                    for anchor in self.anchors
                ],
                dtype=np.float64,
            )
        else:
            values = np.asarray(
                [
                    float(self.map_store.networkDistanceToTarget(anchor, node))
                    for anchor in self.anchors
                ],
                dtype=np.float64,
            )
        self.distance_by_node[node_id] = values
        return values

    def _valid_current_assignments(self) -> np.ndarray:
        assignment = np.full(len(self.pedestrians), -1, dtype=np.int64)
        for index, pedestrian in enumerate(self.pedestrians):
            shelter, reserved = self.shelter_store.reservationFor(pedestrian)
            if shelter is None or int(reserved) < int(self.population[index]):
                continue
            shelter_id = int(shelter.shelterID)
            spec = self.base_by_id.get(shelter_id)
            route = getattr(pedestrian, "routeFollowing", None)
            route_target = getattr(route, "endNode", None) if route is not None else None
            if (
                spec is not None
                and route_target is not None
                and getattr(route_target, "OSMID", None)
                == getattr(spec.node, "OSMID", None)
            ):
                assignment[index] = shelter_id
        return assignment

    def _allocate(
        self,
        shelters: Sequence[_ImmediateShelter],
        distances: dict[int, np.ndarray],
    ) -> np.ndarray:
        assignment = self.base_assignment.copy()
        remaining = {
            shelter.shelter_id: int(shelter.capacity - shelter.flow)
            for shelter in shelters
        }
        available_ids = set(remaining)
        for index, shelter_id in enumerate(assignment):
            if int(shelter_id) not in available_ids:
                assignment[index] = -1
                continue
            remaining[int(shelter_id)] -= int(self.population[index])
        if any(value < 0 for value in remaining.values()):
            raise RuntimeError("Existing shelter reservations exceed capacity")

        while True:
            proposals = []
            for index in range(len(self.pedestrians)):
                current = int(assignment[index])
                current_valid = current >= 0
                current_distance = (
                    float(distances[current][index])
                    if current_valid
                    else float("inf")
                )
                size = int(self.population[index])
                best = None
                for shelter in shelters:
                    shelter_id = shelter.shelter_id
                    if current_valid and shelter_id == current:
                        continue
                    distance = float(distances[shelter_id][index])
                    if not math.isfinite(distance):
                        continue
                    if not distance + 1e-9 < current_distance:
                        continue
                    if remaining[shelter_id] < size:
                        continue
                    improvement = (
                        current_distance - distance if current_valid else 0.0
                    )
                    key = (
                        int(current_valid),
                        float(improvement),
                        float(improvement) * size,
                        -float(distance),
                        -int(shelter_id),
                    )
                    if best is None or key > best[0]:
                        best = (key, shelter_id)
                if best is not None:
                    proposals.append(
                        (
                            best[0],
                            -int(self.pedestrian_ids[index]),
                            index,
                            current,
                            int(best[1]),
                        )
                    )
            if not proposals:
                break
            moved = False
            for _, _, index, expected_current, target in sorted(
                proposals, reverse=True
            ):
                if int(assignment[index]) != int(expected_current):
                    continue
                size = int(self.population[index])
                if remaining[target] < size:
                    continue
                if expected_current >= 0:
                    remaining[int(expected_current)] += size
                remaining[target] -= size
                assignment[index] = target
                moved = True
            if not moved:
                break
        return assignment

    def _path_metrics(self, anchor, node) -> tuple[float, float]:
        key = (int(anchor.OSMID), int(node.OSMID))
        cached = self.path_metric_by_anchor_and_node.get(key)
        if cached is not None:
            return cached
        route = self.map_store.shortestPath(anchor, node)
        if route is None:
            raise RuntimeError(
                f"Verified route {anchor.OSMID}->{node.OSMID} disappeared"
            )
        ordinary_distance = 0.0
        exposure_distance = 0.0
        for edge in (getattr(route, "edgeRemained", None) or ()):
            distance = max(0.0, float(getattr(edge, "edgeLen", 0.0)))
            ordinary_distance += distance
            exposure_distance += distance * self._cell_danger(
                getattr(edge, "cellID", None)
            )
        cached = (ordinary_distance, exposure_distance)
        self.path_metric_by_anchor_and_node[key] = cached
        return cached

    def _route_cost(self, pedestrian_index: int, node) -> tuple[float, float]:
        node_id = int(node.OSMID)
        key = (int(pedestrian_index), node_id)
        cached = self.route_cost_by_pedestrian_and_node.get(key)
        if cached is not None:
            return cached
        pedestrian = self.pedestrians[pedestrian_index]
        speed = max(
            self.speed_floor,
            float(getattr(pedestrian, "currSpeed", 0.0)),
        )
        current_distance = max(
            0.0, float(getattr(pedestrian, "edge_remain", 0.0))
        )
        current_edge = getattr(pedestrian, "currEdge", None)
        current_exposure_distance = current_distance * self._cell_danger(
            getattr(current_edge, "cellID", None)
            or getattr(pedestrian, "currCell", None)
        )
        route_distance, route_exposure_distance = self._path_metrics(
            self.anchors[pedestrian_index], node
        )
        ordinary = (current_distance + route_distance) / speed
        exposure = (
            current_exposure_distance + route_exposure_distance
        ) / speed
        if ordinary > self.horizon_minutes and ordinary > 0.0:
            scale = self.horizon_minutes / ordinary
            ordinary = self.horizon_minutes
            exposure *= scale
        cached = (float(ordinary), float(exposure))
        self.route_cost_by_pedestrian_and_node[key] = cached
        return cached

    def _branch_result(
        self,
        *,
        label: str,
        assignment: np.ndarray,
        shelters: Optional[Sequence[_ImmediateShelter]] = None,
        installed_cell: int = -1,
        installed_capacity: float = 0.0,
    ) -> BranchResult:
        shelter_by_id = {
            shelter.shelter_id: shelter
            for shelter in (shelters or self.base_shelters)
        }
        plain = float(self.common_plain_person_time)
        exposure = float(self.common_hazard_exposure_person_time)
        remaining_by_person = list(self.common_plain_minutes_by_person)
        for index, shelter_id in enumerate(assignment):
            size = int(self.population[index])
            shelter = shelter_by_id.get(int(shelter_id))
            if shelter is None:
                pedestrian = self.pedestrians[index]
                plain += size * self.horizon_minutes
                exposure += (
                    size
                    * self.horizon_minutes
                    * self._cell_danger(getattr(pedestrian, "currCell", None))
                )
                remaining_by_person.extend([self.horizon_minutes] * size)
                continue
            ordinary, hazard = self._route_cost(index, shelter.node)
            plain += size * ordinary
            exposure += size * hazard
            remaining_by_person.extend([ordinary] * size)

        result = BranchResult(label=label)
        result.start_outcome = self.start_outcome
        result.end_outcome = self.start_outcome
        result.remaining_plain_person_time = float(plain)
        result.remaining_hazard_exposure_person_time = float(exposure)
        tail_weight = float(
            getattr(self.core, "clearanceTailExtraWeight", 0.0)
        )
        if tail_weight > 0.0:
            tail = estimate_clearance_tail(
                self.core,
                episode_horizon=self.brancher.episode_horizon,
                initial_population=self.brancher.initial_population,
                remaining=RemainingTimeEstimate(
                    plain_person_minutes=float(plain),
                    hazard_exposure_person_minutes=float(exposure),
                    plain_minutes_by_person=np.asarray(
                        remaining_by_person, dtype=np.float64
                    ),
                ),
            )
            result.clearance_tail_person_time = float(
                tail_weight * tail.tail_person_minutes
            )
            result.clearance_tail_mean_minutes = float(tail.tail_mean_minutes)
            result.clearance_t90_minutes = float(tail.t90_minutes)
            result.clearance_time_minutes = float(tail.clearance_time_minutes)
        result.installed_cell = int(installed_cell)
        result.installed_capacity = float(installed_capacity)
        denominator = (
            float(self.brancher.initial_population)
            * float(self.brancher.episode_horizon)
            * self.time_step_minutes
        )
        normalized_cost = result.hazard_weighted_time_to_safety_cost / denominator
        result.rewards.append(float(-normalized_cost))
        result.reward_components.append(
            np.asarray((-normalized_cost,), dtype=np.float32)
        )
        result.end_digest = self.state_digest
        return result

    def effects(self, cell_indices: Sequence[int]) -> list[PairedEffect]:
        effects = []
        _, cell_y = authoritative_grid_shape(self.core)
        for cell_index in cell_indices:
            cell_index = int(cell_index)
            cell = divmod(cell_index, cell_y)
            node = self.shelter_store.previewShelterCandidate(
                cell, self.core.cellTracker
            )
            if node is None:
                raise RuntimeError(
                    f"Counterfactual branch has no deployable candidate in cell {cell_index}"
                )
            capacity = int(self.shelter_store.configuredShelterCapacity(node))
            if capacity <= 0:
                raise RuntimeError("Counterfactual candidate capacity must be positive")
            candidate = _ImmediateShelter(
                shelter_id=int(self.shelter_store.nextID),
                node=node,
                capacity=capacity,
                flow=0,
            )
            shelters = self.base_shelters + (candidate,)
            distances = dict(self.base_distances)
            distances[candidate.shelter_id] = self._distance_column(node)
            for index in range(len(self.pedestrians)):
                if not any(
                    np.isfinite(values[index]) for values in distances.values()
                ):
                    raise RuntimeError(
                        f"Active pedestrian {self.pedestrian_ids[index]} cannot reach "
                        f"the action branch's operational shelter set"
                    )
            assignment = self._allocate(shelters, distances)
            acted = self._branch_result(
                label="acted",
                assignment=assignment,
                shelters=shelters,
                installed_cell=cell_index,
                installed_capacity=capacity,
            )
            effects.append(
                PairedEffect(
                    cell_index=cell_index,
                    horizon=0,
                    acted=acted,
                    waited=self.wait_result,
                    common_noise=True,
                )
            )
        return effects


@dataclass(frozen=True)
class _PhysicalBranchTask:
    tape_index: int
    action_position: int
    install_cell: Optional[int]


@dataclass(frozen=True)
class _RandomContinuationTask:
    tape_index: int
    decision_index: int
    remaining_deployments: int


@dataclass(frozen=True)
class _RolloutContinuationTask:
    tape_index: int
    action_position: int
    install_cell: int
    decision_index: int
    remaining_deployments: int


@dataclass
class EndpointObservationSpec:
    """Information needed to build a compact neural endpoint observation."""

    initial_population: int
    episode_horizon: int
    maximum_deployments: int
    decision_index: int
    remaining_deployments: int


@dataclass
class _LocalBranchBatch:
    brancher: "CounterfactualBrancher"
    snapshot: SimulatorSnapshot
    future_seeds: tuple[Optional[int], ...]
    hazard_timelines: tuple[tuple[HazardFrame, ...], ...]
    endpoint_spec: Optional[EndpointObservationSpec]
    endpoint_builder: Optional[RegionalObservationBuilder]
    rollout_policy: Optional[FrozenResidualPolicy] = None


@dataclass
class _SpawnBranchBatch:
    core: Any
    horizon: int
    reward_model: Any
    initial_population: int
    episode_horizon: int
    future_seeds: tuple[Optional[int], ...]
    hazard_timelines: tuple[tuple[HazardFrame, ...], ...]
    endpoint_spec: Optional[EndpointObservationSpec]
    node_flow: np.ndarray
    edge_flow: np.ndarray
    python_random_state: tuple
    numpy_legacy_state: tuple
    rollout_policy: Optional[FrozenResidualPolicy] = None


_SPAWN_BRANCH_BATCH: Optional[_LocalBranchBatch] = None


_PORTABLE_CORE_ATTRIBUTES = (
    "mapDS",
    "OSMProcessor",
    "congestionModel",
    "cellX",
    "cellY",
    "timeStepMinutes",
    "maxSpeed",
    "shelterActionInterval",
    "shelterCapacityToken",
    "hazardVol",
    "panicRate",
    "stopTime",
    "casualtyPriorityKappa",
)


def _portable_branch_core(core, snapshot: SimulatorSnapshot) -> SimpleNamespace:
    """Build a simulator-only object graph suitable for isolated workers.

    The learner, optimizers, file handles, visualizer, and loggers are excluded.
    Mutable stores come from the exact decision snapshot; static network objects
    are serialized in the same object graph so pedestrian route-node identity is
    retained after unpickling.
    """
    attributes = {
        name: getattr(core, name)
        for name in _PORTABLE_CORE_ATTRIBUTES
        if hasattr(core, name)
    }
    attributes.update(snapshot.state)
    return SimpleNamespace(**attributes)


def _capture_endpoint_observation(
    brancher: "CounterfactualBrancher",
    result: BranchResult,
    spec: Optional[EndpointObservationSpec],
    builder: Optional[RegionalObservationBuilder],
) -> Any:
    if spec is None or builder is None or result.terminal:
        return None
    remaining = int(spec.remaining_deployments)
    decision_index = int(spec.decision_index)
    if result.installed_cell >= 0:
        remaining -= 1
        decision_index += 1
    observation = builder.build(
        decision_index=decision_index,
        simulation_time=int(getattr(brancher.core.pedDS, "currTime", 0)),
        remaining_deployments=max(0, remaining),
    )
    return brancher.enrich_observation(observation)


def _execute_physical_branch(
    batch: _LocalBranchBatch,
    task: _PhysicalBranchTask,
) -> tuple[int, int, BranchResult]:
    brancher = batch.brancher
    restore(brancher.core, batch.snapshot)
    seed = batch.future_seeds[task.tape_index]
    if seed is not None:
        brancher._reseed_exogenous_streams(seed)
    result = brancher._run_branch(
        label="waited" if task.install_cell is None else "acted",
        install_cell=task.install_cell,
        steps=brancher.horizon,
        hazard_timeline=batch.hazard_timelines[task.tape_index],
    )
    result.endpoint_payload = _capture_endpoint_observation(
        brancher,
        result,
        batch.endpoint_spec,
        batch.endpoint_builder,
    )
    return task.tape_index, task.action_position, result


def _execute_random_continuation(
    batch: _LocalBranchBatch,
    task: _RandomContinuationTask,
) -> tuple[int, float]:
    brancher = batch.brancher
    restore(brancher.core, batch.snapshot)
    seed = batch.future_seeds[task.tape_index]
    brancher._reseed_exogenous_streams(seed)
    if batch.endpoint_builder is None:
        raise RuntimeError("Random continuation worker has no observation builder")
    placement_seed = int(
        np.random.SeedSequence((0 if seed is None else int(seed), 0xA17C9E3D))
        .generate_state(1, dtype=np.uint64)[0]
    )
    cost = brancher._run_uniform_random_continuation(
        builder=batch.endpoint_builder,
        decision_index=task.decision_index,
        remaining_deployments=task.remaining_deployments,
        placement_rng=np.random.default_rng(placement_seed),
    )
    return task.tape_index, float(cost)


def _execute_rollout_continuation(
    batch: _LocalBranchBatch,
    task: _RolloutContinuationTask,
) -> tuple[int, int, float]:
    brancher = batch.brancher
    restore(brancher.core, batch.snapshot)
    seed = batch.future_seeds[task.tape_index]
    brancher._reseed_exogenous_streams(seed)
    if batch.endpoint_builder is None or batch.rollout_policy is None:
        raise RuntimeError("Rollout worker has no frozen policy or observation builder")
    cost = brancher._run_frozen_policy_continuation(
        builder=batch.endpoint_builder,
        first_cell=int(task.install_cell),
        decision_index=int(task.decision_index),
        remaining_deployments=int(task.remaining_deployments),
        frozen_policy=batch.rollout_policy,
    )
    return task.tape_index, task.action_position, float(cost)


def _execute_branch_task(batch: _LocalBranchBatch, task):
    if isinstance(task, _PhysicalBranchTask):
        return _execute_physical_branch(batch, task)
    if isinstance(task, _RandomContinuationTask):
        return _execute_random_continuation(batch, task)
    if isinstance(task, _RolloutContinuationTask):
        return _execute_rollout_continuation(batch, task)
    raise TypeError(f"Unsupported branch worker task: {type(task).__name__}")


def _initialize_spawn_branch_worker(batch: _SpawnBranchBatch) -> None:
    global _SPAWN_BRANCH_BATCH
    random.setstate(batch.python_random_state)
    np.random.set_state(batch.numpy_legacy_state)
    _restore_map_flow(
        _iter_map_nodes(batch.core),
        _iter_map_edges(batch.core),
        batch.node_flow,
        batch.edge_flow,
    )
    brancher = CounterfactualBrancher(
        batch.core,
        horizon=batch.horizon,
        reward_model=batch.reward_model,
        initial_population=batch.initial_population,
        episode_horizon=batch.episode_horizon,
        workers=1,
    )
    builder = None
    if batch.endpoint_spec is not None:
        builder = RegionalObservationBuilder(
            batch.core,
            initial_population=batch.endpoint_spec.initial_population,
            horizon=batch.endpoint_spec.episode_horizon,
            maximum_deployments=batch.endpoint_spec.maximum_deployments,
        )
    _SPAWN_BRANCH_BATCH = _LocalBranchBatch(
        brancher=brancher,
        snapshot=capture(batch.core, label="nmcc-spawn-worker-base"),
        future_seeds=batch.future_seeds,
        hazard_timelines=batch.hazard_timelines,
        endpoint_spec=batch.endpoint_spec,
        endpoint_builder=builder,
        rollout_policy=batch.rollout_policy,
    )


def _spawn_physical_branch_worker(
    task,
):
    if _SPAWN_BRANCH_BATCH is None:
        raise RuntimeError("NMCC spawned worker started without a branch batch")
    return _execute_branch_task(_SPAWN_BRANCH_BATCH, task)


def _read_exact(stream, byte_count: int) -> bytes:
    payload = bytearray()
    while len(payload) < int(byte_count):
        chunk = stream.read(int(byte_count) - len(payload))
        if not chunk:
            raise EOFError("NMCC worker closed its protocol stream")
        payload.extend(chunk)
    return bytes(payload)


def _write_message(stream, value: Any) -> None:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    framed = len(payload).to_bytes(8, byteorder="big", signed=False) + payload
    view = memoryview(framed)
    while view:
        written = stream.write(view)
        if written is None:
            # Buffered streams accept the complete view and report ``None``.
            break
        if written <= 0:
            raise BrokenPipeError("NMCC worker protocol write made no progress")
        view = view[written:]
    stream.flush()


def _read_message(stream) -> Any:
    size = int.from_bytes(_read_exact(stream, 8), byteorder="big", signed=False)
    return pickle.loads(_read_exact(stream, size))


def _read_worker_message(
    process: subprocess.Popen,
    *,
    deadline: float,
    context: str,
) -> Any:
    """Read one small framed worker response without an unbounded pipe wait."""
    stream = process.stdout
    if stream is None:
        raise RuntimeError(f"NMCC worker has no protocol stream during {context}")

    def read_exact(byte_count: int) -> bytes:
        payload = bytearray()
        descriptor = stream.fileno()
        while len(payload) < int(byte_count):
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"NMCC worker timed out during {context}")
            readable, _, _ = select.select(
                (descriptor,), (), (), min(remaining, 0.25)
            )
            if not readable:
                status = process.poll()
                if status is not None:
                    raise EOFError(
                        f"NMCC worker exited with status {status} during {context}"
                    )
                continue
            chunk = os.read(descriptor, int(byte_count) - len(payload))
            if not chunk:
                raise EOFError(
                    f"NMCC worker closed its protocol stream during {context}"
                )
            payload.extend(chunk)
        return bytes(payload)

    size = int.from_bytes(read_exact(8), byteorder="big", signed=False)
    return pickle.loads(read_exact(size))


def run_persistent_branch_worker(*, protocol_output=None) -> None:
    """Serve exact branch batches until the parent closes the worker.

    The interpreter imports the simulator once.  Each request still loads one
    exact decision snapshot and its common-noise tapes, so persistence changes
    orchestration cost only—not the physical model or branch independence.
    """
    if protocol_output is None:
        protocol_output = sys.stdout.buffer
    _write_message(protocol_output, "ready")
    while True:
        try:
            request = _read_message(sys.stdin.buffer)
        except EOFError:
            return
        if request is None:
            return
        batch_path, tasks_path, output_path = request
        with open(batch_path, "rb") as handle:
            batch = pickle.load(handle)
        with open(tasks_path, "rb") as handle:
            tasks = pickle.load(handle)
        _initialize_spawn_branch_worker(batch)
        try:
            completed = [_spawn_physical_branch_worker(task) for task in tasks]
        finally:
            # A process is shared by many decisions and episodes.  Do not retain
            # the previous simulator snapshot, frozen actor, or road-state
            # arrays while the worker is idle.
            global _SPAWN_BRANCH_BATCH
            _SPAWN_BRANCH_BATCH = None
        with open(output_path, "wb") as handle:
            pickle.dump(completed, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _write_message(protocol_output, True)


class _PersistentIsolatedBranchPool:
    """Long-lived simulator-only processes for exact NMCC branch batches."""

    STARTUP_TIMEOUT_SECONDS = 60.0
    BATCH_TIMEOUT_SECONDS = 7200.0

    def __init__(self, workers: int):
        self.worker_count = max(1, int(workers))
        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        executable = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "NMCCBranchWorker.py",
        )
        self.processes: list[subprocess.Popen] = []
        self.unreaped_process_ids: tuple[int, ...] = ()
        try:
            # Start and handshake each OpenMP/Torch interpreter before starting
            # the next one.  Concurrent imports race through libomp's macOS
            # shared-memory registration and caused intermittent error #179
            # after repeated episode-level pool churn.
            for worker_index in range(self.worker_count):
                process = subprocess.Popen(
                    (sys.executable, executable, "--persistent"),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    # Preserve worker tracebacks in the parent training log.  An
                    # exact branch failure is a model error and must be diagnosed,
                    # never hidden behind an EOF or silently retried.
                    stderr=None,
                    env=environment,
                    bufsize=0,
                    # On macOS CPython this selects posix_spawn.  Forking after
                    # NumPy/OpenMP initialization can deadlock before exec.
                    close_fds=False,
                )
                self.processes.append(process)
                try:
                    ready = _read_worker_message(
                        process,
                        deadline=time.monotonic() + self.STARTUP_TIMEOUT_SECONDS,
                        context=f"startup of worker {worker_index}",
                    )
                    if ready != "ready":
                        raise RuntimeError(
                            "worker returned an invalid ready marker"
                        )
                except (
                    EOFError,
                    OSError,
                    TimeoutError,
                    pickle.PickleError,
                ) as error:
                    raise self._failure(worker_index, error) from error
        except BaseException:
            self.close()
            raise
        self.batches_completed = 0

    def _failure(self, index: int, error: BaseException) -> RuntimeError:
        process = self.processes[index]
        status = process.poll()
        if status is None:
            try:
                status = process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                status = None
        return RuntimeError(
            f"Persistent exact NMCC worker {index} failed "
            f"(exit_status={status}): {error}"
        )

    def run(
        self,
        batch: _SpawnBranchBatch,
        tasks: Sequence[Any],
    ) -> list[Any]:
        try:
            process_count = min(self.worker_count, len(tasks))
            chunks = [
                tuple(tasks[index::process_count])
                for index in range(process_count)
            ]
            with tempfile.TemporaryDirectory(prefix="nmcc-branches-") as directory:
                batch_path = os.path.join(directory, "batch.pkl")
                with open(batch_path, "wb") as handle:
                    pickle.dump(batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
                output_paths = []
                for worker_index, chunk in enumerate(chunks):
                    process = self.processes[worker_index]
                    if process.poll() is not None:
                        raise self._failure(
                            worker_index,
                            RuntimeError(
                                f"exited with status {process.returncode}"
                            ),
                        )
                    tasks_path = os.path.join(
                        directory, f"tasks-{worker_index}.pkl"
                    )
                    output_path = os.path.join(
                        directory, f"result-{worker_index}.pkl"
                    )
                    with open(tasks_path, "wb") as handle:
                        pickle.dump(chunk, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    try:
                        _write_message(
                            process.stdin,
                            (batch_path, tasks_path, output_path),
                        )
                    except (BrokenPipeError, OSError) as error:
                        raise self._failure(worker_index, error) from error
                    output_paths.append(output_path)

                deadline = time.monotonic() + self.BATCH_TIMEOUT_SECONDS
                for worker_index in range(process_count):
                    process = self.processes[worker_index]
                    try:
                        complete = _read_worker_message(
                            process,
                            deadline=deadline,
                            context=f"branch batch on worker {worker_index}",
                        )
                        if complete is not True:
                            raise RuntimeError(
                                "worker returned an invalid completion marker"
                            )
                    except (
                        EOFError,
                        OSError,
                        TimeoutError,
                        pickle.PickleError,
                    ) as error:
                        raise self._failure(worker_index, error) from error
                completed = []
                for output_path in output_paths:
                    with open(output_path, "rb") as handle:
                        completed.extend(pickle.load(handle))
        except BaseException:
            # A failed/interrupted batch invalidates every process: some may
            # still be computing against temporary files that are about to be
            # removed.  Close the complete pool instead of leaving orphans.
            self.close()
            raise
        self.batches_completed += 1
        return completed

    def close(self) -> None:
        processes, self.processes = self.processes, []
        for process in processes:
            if process.poll() is None and process.stdin is not None:
                try:
                    _write_message(process.stdin, None)
                except (BrokenPipeError, OSError):
                    pass
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        def wait_for(process: subprocess.Popen, timeout: float) -> bool:
            try:
                process.wait(timeout=timeout)
                return True
            except (subprocess.TimeoutExpired, ChildProcessError):
                return process.poll() is not None

        for process in processes:
            if not wait_for(process, 5.0):
                process.terminate()
                if not wait_for(process, 2.0):
                    process.kill()
                    wait_for(process, 2.0)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self.unreaped_process_ids = tuple(
            int(process.pid) for process in processes if process.poll() is None
        )


_SHARED_BRANCH_POOL_LOCK = threading.RLock()
_SHARED_BRANCH_POOLS: dict[int, _PersistentIsolatedBranchPool] = {}


def _shared_persistent_branch_pool(workers: int) -> _PersistentIsolatedBranchPool:
    """Return the process-wide exact-branch pool for one worker count.

    A campaign constructs a fresh ``Core`` for every episode, but branch
    workers consume complete portable batches and therefore do not belong to a
    particular ``Core``.  Keeping them for the lifetime of the campaign avoids
    repeated Torch/OpenMP interpreter startup and teardown without sharing any
    simulated state between branches.
    """
    worker_count = max(1, int(workers))
    with _SHARED_BRANCH_POOL_LOCK:
        pool = _SHARED_BRANCH_POOLS.get(worker_count)
        if pool is not None and pool.processes and all(
            process.poll() is None for process in pool.processes
        ):
            return pool
        if pool is not None:
            pool.close()
        pool = _PersistentIsolatedBranchPool(worker_count)
        _SHARED_BRANCH_POOLS[worker_count] = pool
        return pool


def shutdown_persistent_branch_pools() -> None:
    """Close every campaign-level branch pool exactly once at process exit."""
    with _SHARED_BRANCH_POOL_LOCK:
        pools = tuple(_SHARED_BRANCH_POOLS.values())
        _SHARED_BRANCH_POOLS.clear()
    for pool in pools:
        pool.close()


atexit.register(shutdown_persistent_branch_pools)


class CounterfactualBrancher:
    """Run exact common-noise branches from a decision state.

    A horizon of zero measures the exact immediate reservation/route change;
    positive horizons additionally simulate physical evolution before the
    endpoint remaining-time-to-safety estimate.
    """

    def __init__(
        self,
        core,
        *,
        horizon: int,
        reward_model,
        initial_population: int,
        episode_horizon: int,
        workers: int = 1,
    ):
        if int(horizon) < 0:
            raise ValueError("Counterfactual branch horizon must be non-negative")
        if int(initial_population) <= 0:
            raise ValueError("initial_population must be positive")
        if int(episode_horizon) <= 0:
            raise ValueError("episode_horizon must be positive")
        self.core = core
        self.horizon = int(horizon)
        self.reward_model = reward_model
        self.initial_population = int(initial_population)
        self.episode_horizon = int(episode_horizon)
        self.workers = int(workers)
        if self.workers <= 0:
            raise ValueError("Counterfactual branch workers must be positive")
        self._worker_pool: Optional[_PersistentIsolatedBranchPool] = None

    def _pool(self) -> _PersistentIsolatedBranchPool:
        if self.workers <= 1:
            raise RuntimeError("Serial NMCC execution does not use a worker pool")
        if (
            self._worker_pool is None
            or not self._worker_pool.processes
            or any(process.poll() is not None for process in self._worker_pool.processes)
        ):
            self._worker_pool = _shared_persistent_branch_pool(self.workers)
        return self._worker_pool

    def close(self) -> None:
        # The pool owns no episode state; each request carries a complete
        # portable simulator batch.  Detach this brancher while leaving the
        # campaign-level workers alive for the next episode.
        self._worker_pool = None

    def enrich_observation(
        self,
        observation: RegionalObservation,
    ) -> RegionalObservation:
        """Attach the exact D(s) ledger and zero-horizon action gain.

        Both quantities use the same capacity-valid route estimator as the
        complete objective. They are deterministic functions of the current
        snapshot and are computed identically for live decisions and branch
        endpoints.
        """
        estimate = estimate_remaining_time_to_safety(
            self.core,
            episode_horizon=self.episode_horizon,
            include_panicked=True,
            include_casualties=False,
        )
        denominator = (
            float(self.initial_population)
            * float(self.episode_horizon)
            * max(1e-9, float(getattr(self.core, "timeStepMinutes", 1.0)))
        )
        if denominator <= 0.0:
            raise RuntimeError("Exact decision-feature denominator must be positive")
        debt_by_cell = np.asarray(
            estimate.hazard_weighted_by_cell,
            dtype=np.float64,
        ) / denominator
        gains = np.zeros(observation.number_of_actions, dtype=np.float64)
        feasible = np.flatnonzero(observation.action_mask)
        if feasible.size:
            cells = observation.candidate_cell_indices[feasible]
            effects = _ImmediateCandidateEvaluator(self).effects(cells)
            if len(effects) != int(feasible.size):
                raise RuntimeError("Exact assignment gain changed candidate count")
            gains[feasible] = np.asarray(
                [
                    effect.waited.hazard_weighted_time_to_safety_cost
                    - effect.acted.hazard_weighted_time_to_safety_cost
                    for effect in effects
                ],
                dtype=np.float64,
            ) / denominator
        if not np.isfinite(debt_by_cell).all() or not np.isfinite(gains).all():
            raise FloatingPointError("Exact decision features contain non-finite values")
        if np.any(debt_by_cell < -1e-9):
            raise RuntimeError("Capacity-valid evacuation debt became negative")
        if np.any(np.abs(gains) > 2.0 + 1e-6):
            raise RuntimeError("Exact assignment gain exceeded its ledger bound")
        return replace(
            observation,
            remaining_evacuation_debt_by_cell=np.maximum(
                debt_by_cell, 0.0
            ).astype(np.float32),
            candidate_assignment_gain=gains.astype(np.float32),
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can dismantle modules before finalizers run;
            # explicit episode cleanup remains the normal lifecycle.
            pass

    def _hazard_timeline(
        self,
        snapshot: SimulatorSnapshot,
        future_seed: Optional[int],
    ) -> tuple[HazardFrame, ...]:
        """Evolve one action-independent hazard future exactly once."""
        restore(self.core, snapshot)
        if future_seed is not None:
            self._reseed_exogenous_streams(future_seed)
        frames = []
        for _ in range(self.horizon):
            self.core.hazardDS.spreadUpdate()
            self.core.hazardDS.heatUpdate()
            self.core.hazardDS.smokeUpdate()
            self.core.hazardDS.terminateHazard()
            frames.append(_capture_hazard_frame(self.core))
        restore(self.core, snapshot)
        return tuple(frames)

    # -- single branch ----------------------------------------------------

    def _run_branch(
        self,
        *,
        label: str,
        install_cell: Optional[int],
        steps: int,
        hazard_timeline: Optional[Sequence[HazardFrame]] = None,
    ) -> BranchResult:
        core = self.core
        result = BranchResult(label=label)
        result.start_outcome = outcome_snapshot(core)
        reward_pedestrian_ids = {
            int(getattr(pedestrian, "agentID"))
            for pedestrian in getattr(core.pedDS, "pedAgentList", {}).values()
            if not bool(getattr(pedestrian, "terminated", False))
        }

        installed_shelter = None
        if install_cell is not None:
            cell_index = int(install_cell)
            _, cell_y = authoritative_grid_shape(core)
            cell = divmod(cell_index, cell_y)
            shelter_id = core.shelterDS.newShelter({"cell": cell}, core.cellTracker)
            if shelter_id is None:
                raise RuntimeError(
                    f"Counterfactual branch could not install a shelter in cell {cell_index}"
                )
            shelter = core.shelterDS.shelterList[shelter_id]
            if hasattr(core.cellTracker, "addShelter"):
                core.cellTracker.addShelter(cell, shelter)
            installed_shelter = shelter
            result.installed_cell = cell_index
            result.installed_capacity = max(0.0, float(getattr(shelter, "shelterCap", 0.0)))

        # WAIT receives the identical route-maintenance operation. Therefore a
        # stale route repaired in both branches cancels; only capacity access
        # or a strictly better destination enabled by the new shelter remains.
        core.pedDS.reroute_active_to_nearest_shelter(
            new_shelter=installed_shelter
        )

        previous = outcome_snapshot(core)
        time_step_minutes = max(
            1e-9, float(getattr(core, "timeStepMinutes", 1.0))
        )
        if hazard_timeline is not None and len(hazard_timeline) < int(steps):
            raise ValueError("NMCC hazard timeline is shorter than the branch")
        for step_index in range(int(steps)):
            advance_one_timestep(
                core,
                hazard_frame=(
                    None
                    if hazard_timeline is None
                    else hazard_timeline[step_index]
                ),
            )
            current = outcome_snapshot(core)

            # Keep the complete decision-time cohort in the cost ledger.
            # Common-random-number differencing removes shared panic, casualty,
            # traffic and fire evolution; an action-caused difference remains
            # valid shelter-placement credit rather than making a person
            # disappear from the objective.
            active_mass, exposure_mass, casualty_cost = controllable_time_mass(
                core, reward_pedestrian_ids
            )
            result.active_person_time += active_mass * time_step_minutes
            result.hazard_exposure_person_time += (
                exposure_mass * time_step_minutes
            )
            result.expected_casualty_person_time += (
                casualty_cost * time_step_minutes
            )

            environment_terminal = bool(
                int(getattr(core.pedDS, "currTime", 0)) >= self.episode_horizon
            )
            result.steps += 1
            previous = current
            if environment_terminal:
                result.terminal = True
                break

        result.end_outcome = previous
        remaining = estimate_remaining_time_to_safety(
            core,
            episode_horizon=self.episode_horizon,
            pedestrian_ids=reward_pedestrian_ids,
            include_panicked=True,
            include_casualties=False,
        )
        result.remaining_plain_person_time = float(
            remaining.plain_person_minutes
        )
        result.remaining_hazard_exposure_person_time = float(
            remaining.hazard_exposure_person_minutes
        )
        tail_weight = float(getattr(core, "clearanceTailExtraWeight", 0.0))
        if tail_weight > 0.0:
            tail = estimate_clearance_tail(
                core,
                episode_horizon=self.episode_horizon,
                initial_population=self.initial_population,
                remaining=remaining,
            )
            result.clearance_tail_person_time = float(
                tail_weight * tail.tail_person_minutes
            )
            result.clearance_tail_mean_minutes = float(tail.tail_mean_minutes)
            result.clearance_t90_minutes = float(tail.t90_minutes)
            result.clearance_time_minutes = float(tail.clearance_time_minutes)
        denominator = (
            float(self.initial_population)
            * float(self.episode_horizon)
            * time_step_minutes
        )
        normalized_cost = result.hazard_weighted_time_to_safety_cost / denominator
        result.rewards.append(float(-normalized_cost))
        result.reward_components.append(
            np.asarray((-normalized_cost,), dtype=np.float32)
        )
        result.end_digest = live_state_digest(core)
        return result

    def _run_uniform_random_continuation(
        self,
        *,
        builder: RegionalObservationBuilder,
        decision_index: int,
        remaining_deployments: int,
        placement_rng: np.random.Generator,
    ) -> float:
        """Realize the fixed uniform-siting policy from one decision state.

        The first shelter is selected immediately and later shelters follow the
        same fixed decision schedule as every learned or benchmark policy.  The
        returned quantity is the complete normalized cost-to-go, so averaging
        these independent futures gives an unbiased target for ``V_mu``.
        """
        core = self.core
        decision_index = int(decision_index)
        remaining_deployments = int(remaining_deployments)
        if decision_index < 0 or remaining_deployments <= 0:
            raise ValueError("Random continuation requires a deployment decision")
        interval = max(1, int(getattr(core, "shelterActionInterval", 1)))
        next_deployment_time = int(getattr(core.pedDS, "currTime", 0))
        cohort = {
            int(getattr(pedestrian, "agentID"))
            for pedestrian in core.pedDS.pedAgentList.values()
            if not bool(getattr(pedestrian, "terminated", False))
        }
        time_step_minutes = max(
            1e-9, float(getattr(core, "timeStepMinutes", 1.0))
        )
        complete_cost = 0.0

        while int(getattr(core.pedDS, "currTime", 0)) < self.episode_horizon:
            current_time = int(getattr(core.pedDS, "currTime", 0))
            if (
                remaining_deployments > 0
                and current_time >= next_deployment_time
            ):
                observation = builder.build(
                    decision_index=decision_index,
                    simulation_time=current_time,
                    remaining_deployments=remaining_deployments,
                )
                feasible = np.flatnonzero(observation.action_mask)
                if feasible.size == 0:
                    raise RuntimeError(
                        "Uniform reference policy exhausted feasible shelter cells "
                        "before its deployment budget"
                    )
                action = int(placement_rng.choice(feasible))
                cell_index = int(observation.candidate_cell_indices[action])
                _, cell_y = authoritative_grid_shape(core)
                cell = divmod(cell_index, cell_y)
                shelter_id = core.shelterDS.newShelter(
                    {"cell": cell}, core.cellTracker
                )
                if shelter_id is None:
                    raise RuntimeError(
                        f"Uniform reference failed to install feasible cell {cell_index}"
                    )
                shelter = core.shelterDS.shelterList[shelter_id]
                if hasattr(core.cellTracker, "addShelter"):
                    core.cellTracker.addShelter(cell, shelter)
                core.pedDS.reroute_active_to_nearest_shelter(
                    new_shelter=shelter
                )
                remaining_deployments -= 1
                decision_index += 1
                next_deployment_time += interval

            advance_one_timestep(core)
            plain, exposure, casualty = controllable_time_mass(core, cohort)
            complete_cost += time_step_minutes * (
                plain + exposure + casualty
            )
            if plain == 0.0 and exposure == 0.0 and casualty == 0.0:
                break

        residual = estimate_remaining_time_to_safety(
            core,
            episode_horizon=self.episode_horizon,
            pedestrian_ids=cohort,
            include_panicked=True,
            include_casualties=False,
        )
        complete_cost += residual.hazard_weighted_person_minutes
        tail_weight = float(getattr(core, "clearanceTailExtraWeight", 0.0))
        if tail_weight > 0.0:
            complete_cost += tail_weight * estimate_clearance_tail(
                core,
                episode_horizon=self.episode_horizon,
                initial_population=self.initial_population,
                remaining=residual,
            ).tail_person_minutes
        denominator = (
            float(self.initial_population)
            * float(self.episode_horizon)
            * time_step_minutes
        )
        return float(complete_cost / denominator)

    def _run_frozen_policy_continuation(
        self,
        *,
        builder: RegionalObservationBuilder,
        first_cell: int,
        decision_index: int,
        remaining_deployments: int,
        frozen_policy: FrozenResidualPolicy,
    ) -> float:
        """Install one candidate, then follow one frozen actor to the horizon.

        This is the simulator grader used by rollout policy iteration.  It uses
        the same complete ledger, physical dynamics, installation schedule and
        terminal residual as the fixed random-reference continuation.  Only
        the deterministic siting policy differs.
        """
        core = self.core
        decision_index = int(decision_index)
        remaining_deployments = int(remaining_deployments)
        if decision_index < 0 or remaining_deployments <= 0:
            raise ValueError("Frozen rollout requires a deployment decision")
        interval = max(1, int(getattr(core, "shelterActionInterval", 1)))
        next_deployment_time = int(getattr(core.pedDS, "currTime", 0)) + interval
        cohort = {
            int(getattr(pedestrian, "agentID"))
            for pedestrian in core.pedDS.pedAgentList.values()
            if not bool(getattr(pedestrian, "terminated", False))
        }
        time_step_minutes = max(
            1e-9, float(getattr(core, "timeStepMinutes", 1.0))
        )
        complete_cost = 0.0

        _, cell_y = authoritative_grid_shape(core)
        cell = divmod(int(first_cell), cell_y)
        shelter_id = core.shelterDS.newShelter({"cell": cell}, core.cellTracker)
        if shelter_id is None:
            raise RuntimeError(
                f"Frozen rollout failed to install feasible cell {first_cell}"
            )
        shelter = core.shelterDS.shelterList[shelter_id]
        if hasattr(core.cellTracker, "addShelter"):
            core.cellTracker.addShelter(cell, shelter)
        core.pedDS.reroute_active_to_nearest_shelter(new_shelter=shelter)
        remaining_deployments -= 1
        decision_index += 1
        previous_observation, recurrent_state = frozen_policy.initial_context()

        while int(getattr(core.pedDS, "currTime", 0)) < self.episode_horizon:
            current_time = int(getattr(core.pedDS, "currTime", 0))
            if (
                remaining_deployments > 0
                and current_time >= next_deployment_time
            ):
                observation = builder.build(
                    decision_index=decision_index,
                    simulation_time=current_time,
                    remaining_deployments=remaining_deployments,
                )
                if not np.any(observation.action_mask):
                    raise RuntimeError(
                        "Frozen rollout exhausted feasible shelter cells before "
                        "its deployment budget"
                    )
                observation = self.enrich_observation(observation)
                action, recurrent_state = frozen_policy.select(
                    observation,
                    previous_observation=previous_observation,
                    recurrent_state=recurrent_state,
                )
                candidate_cell = int(observation.candidate_cell_indices[action])
                _, cell_y = authoritative_grid_shape(core)
                cell = divmod(candidate_cell, cell_y)
                shelter_id = core.shelterDS.newShelter(
                    {"cell": cell}, core.cellTracker
                )
                if shelter_id is None:
                    raise RuntimeError(
                        "Frozen rollout selected an infeasible shelter cell"
                    )
                shelter = core.shelterDS.shelterList[shelter_id]
                if hasattr(core.cellTracker, "addShelter"):
                    core.cellTracker.addShelter(cell, shelter)
                core.pedDS.reroute_active_to_nearest_shelter(
                    new_shelter=shelter
                )
                previous_observation = observation
                remaining_deployments -= 1
                decision_index += 1
                next_deployment_time += interval

            advance_one_timestep(core)
            plain, exposure, casualty = controllable_time_mass(core, cohort)
            complete_cost += time_step_minutes * (
                plain + exposure + casualty
            )
            if plain == 0.0 and exposure == 0.0 and casualty == 0.0:
                break

        residual = estimate_remaining_time_to_safety(
            core,
            episode_horizon=self.episode_horizon,
            pedestrian_ids=cohort,
            include_panicked=True,
            include_casualties=False,
        )
        complete_cost += residual.hazard_weighted_person_minutes
        tail_weight = float(getattr(core, "clearanceTailExtraWeight", 0.0))
        if tail_weight > 0.0:
            complete_cost += tail_weight * estimate_clearance_tail(
                core,
                episode_horizon=self.episode_horizon,
                initial_population=self.initial_population,
                remaining=residual,
            ).tail_person_minutes
        denominator = (
            float(self.initial_population)
            * float(self.episode_horizon)
            * time_step_minutes
        )
        return float(complete_cost / denominator)

    def random_policy_costs(
        self,
        *,
        snapshot: SimulatorSnapshot,
        future_seeds: Sequence[int],
        decision_index: int,
        remaining_deployments: int,
        maximum_deployments: int,
    ) -> list[float]:
        """Estimate ``V_mu`` with independent complete uniform continuations."""
        seeds = tuple(int(seed) for seed in future_seeds)
        if not seeds:
            raise ValueError("Random-policy value requires at least one future")
        endpoint_spec = EndpointObservationSpec(
            initial_population=self.initial_population,
            episode_horizon=self.episode_horizon,
            maximum_deployments=int(maximum_deployments),
            decision_index=int(decision_index),
            remaining_deployments=int(remaining_deployments),
        )
        tasks = tuple(
            _RandomContinuationTask(
                tape_index=index,
                decision_index=int(decision_index),
                remaining_deployments=int(remaining_deployments),
            )
            for index in range(len(seeds))
        )
        local_builder = RegionalObservationBuilder(
            self.core,
            initial_population=self.initial_population,
            horizon=self.episode_horizon,
            maximum_deployments=int(maximum_deployments),
        )
        local_batch = _LocalBranchBatch(
            brancher=self,
            snapshot=snapshot,
            future_seeds=seeds,
            hazard_timelines=tuple(() for _ in seeds),
            endpoint_spec=endpoint_spec,
            endpoint_builder=local_builder,
        )
        try:
            if self.workers == 1:
                completed = [
                    _execute_random_continuation(local_batch, task)
                    for task in tasks
                ]
            else:
                spawn_batch = _SpawnBranchBatch(
                    core=_portable_branch_core(self.core, snapshot),
                    horizon=self.horizon,
                    reward_model=self.reward_model,
                    initial_population=self.initial_population,
                    episode_horizon=self.episode_horizon,
                    future_seeds=seeds,
                    hazard_timelines=tuple(() for _ in seeds),
                    endpoint_spec=endpoint_spec,
                    node_flow=snapshot.node_flow,
                    edge_flow=snapshot.edge_flow,
                    python_random_state=snapshot.python_random_state,
                    numpy_legacy_state=snapshot.numpy_legacy_state,
                )
                completed = self._pool().run(spawn_batch, tasks)
        finally:
            restore(self.core, snapshot)
        by_tape = {int(index): float(cost) for index, cost in completed}
        if set(by_tape) != set(range(len(seeds))):
            raise RuntimeError("Random-policy continuation batch is incomplete")
        return [by_tape[index] for index in range(len(seeds))]

    def rollout_policy_costs(
        self,
        cell_indices: Sequence[int],
        *,
        snapshot: SimulatorSnapshot,
        future_seeds: Sequence[int],
        decision_index: int,
        remaining_deployments: int,
        maximum_deployments: int,
        frozen_policy: FrozenResidualPolicy,
    ) -> np.ndarray:
        """Return complete common-noise costs for cells under a frozen policy.

        Rows are future tapes and columns preserve ``cell_indices`` order.
        The live simulator is restored after the batch and each isolated task
        receives the same pre-action snapshot and keyed disturbance seed.
        """
        cells = tuple(int(cell) for cell in cell_indices)
        seeds = tuple(int(seed) for seed in future_seeds)
        if len(cells) < 2 or len(cells) != len(set(cells)):
            raise ValueError("Rollout policy grading requires distinct candidate cells")
        if not seeds:
            raise ValueError("Rollout policy grading requires at least one future")
        endpoint_spec = EndpointObservationSpec(
            initial_population=self.initial_population,
            episode_horizon=self.episode_horizon,
            maximum_deployments=int(maximum_deployments),
            decision_index=int(decision_index),
            remaining_deployments=int(remaining_deployments),
        )
        tasks = tuple(
            _RolloutContinuationTask(
                tape_index=tape_index,
                action_position=action_position,
                install_cell=cell,
                decision_index=int(decision_index),
                remaining_deployments=int(remaining_deployments),
            )
            for tape_index in range(len(seeds))
            for action_position, cell in enumerate(cells)
        )
        builder = RegionalObservationBuilder(
            self.core,
            initial_population=self.initial_population,
            horizon=self.episode_horizon,
            maximum_deployments=int(maximum_deployments),
        )
        local_batch = _LocalBranchBatch(
            brancher=self,
            snapshot=snapshot,
            future_seeds=seeds,
            hazard_timelines=tuple(() for _ in seeds),
            endpoint_spec=endpoint_spec,
            endpoint_builder=builder,
            rollout_policy=frozen_policy,
        )
        try:
            if self.workers == 1:
                completed = [
                    _execute_rollout_continuation(local_batch, task)
                    for task in tasks
                ]
            else:
                spawn_batch = _SpawnBranchBatch(
                    core=_portable_branch_core(self.core, snapshot),
                    horizon=self.horizon,
                    reward_model=self.reward_model,
                    initial_population=self.initial_population,
                    episode_horizon=self.episode_horizon,
                    future_seeds=seeds,
                    hazard_timelines=tuple(() for _ in seeds),
                    endpoint_spec=endpoint_spec,
                    node_flow=snapshot.node_flow,
                    edge_flow=snapshot.edge_flow,
                    python_random_state=snapshot.python_random_state,
                    numpy_legacy_state=snapshot.numpy_legacy_state,
                    rollout_policy=frozen_policy,
                )
                completed = self._pool().run(spawn_batch, tasks)
        finally:
            restore(self.core, snapshot)

        costs = np.full((len(seeds), len(cells)), np.nan, dtype=np.float64)
        for tape_index, action_position, cost in completed:
            costs[int(tape_index), int(action_position)] = float(cost)
        if not np.isfinite(costs).all():
            raise RuntimeError("Frozen rollout batch did not return every cost")
        return costs

    # -- paired branches --------------------------------------------------

    def candidate_effects(
        self,
        cell_indices,
        *,
        snapshot: Optional[SimulatorSnapshot] = None,
        future_seed: Optional[int] = None,
        endpoint_value: Optional[Callable[[Any, BranchResult], float]] = None,
    ) -> list[PairedEffect]:
        """Evaluate every supplied shelter cell against one shared WAIT branch.

        All actions and WAIT start from the identical pre-intervention snapshot
        and consume the identical captured random streams.  WAIT is simulated
        once because it is the same control variate for the whole action table.
        The live simulator is restored even if any branch fails.
        """
        cells = [int(cell) for cell in cell_indices]
        if not cells:
            raise ValueError("candidate_effects requires at least one cell")
        if len(cells) != len(set(cells)):
            raise ValueError("candidate_effects requires unique candidate cells")
        if self.horizon == 0:
            if snapshot is not None:
                restore(self.core, snapshot)
            return _ImmediateCandidateEvaluator(self).effects(cells)
        base = snapshot if snapshot is not None else capture(
            self.core, label="nmcc-full-candidate-table"
        )
        effects: list[PairedEffect] = []
        try:
            hazard_timeline = self._hazard_timeline(base, future_seed)
            restore(self.core, base)
            if future_seed is not None:
                self._reseed_exogenous_streams(future_seed)
            waited = self._run_branch(
                label="waited",
                install_cell=None,
                steps=self.horizon,
                hazard_timeline=hazard_timeline,
            )
            if endpoint_value is not None and not waited.terminal:
                waited.continuation_value = float(endpoint_value(self.core, waited))
            for cell in cells:
                restore(self.core, base)
                if future_seed is not None:
                    self._reseed_exogenous_streams(future_seed)
                acted = self._run_branch(
                    label="acted",
                    install_cell=cell,
                    steps=self.horizon,
                    hazard_timeline=hazard_timeline,
                )
                if endpoint_value is not None and not acted.terminal:
                    acted.continuation_value = float(endpoint_value(self.core, acted))
                effects.append(
                    PairedEffect(
                        cell_index=cell,
                        horizon=self.horizon,
                        acted=acted,
                        waited=waited,
                        common_noise=True,
                    )
                )
        finally:
            restore(self.core, base)
        return effects

    def candidate_effects_many_futures(
        self,
        cell_indices,
        *,
        future_seeds: Sequence[Optional[int]],
        snapshot: Optional[SimulatorSnapshot] = None,
        endpoint_spec: Optional[EndpointObservationSpec] = None,
        reference_cell: Optional[int] = None,
    ) -> list[list[PairedEffect]]:
        """Evaluate candidate contrasts for all shared futures in one batch.

        Physical branches are independent conditional on the decision snapshot
        and future tape. Isolated workers receive a simulator-only object graph,
        excluding the learner and its numerical thread pools, and return only
        compact branch results and endpoint observations. Every worker still
        runs the exact pedestrian, routing, congestion, reservation, and
        exposure dynamics.

        By default the common comparator is ``WAIT``.  Passing
        ``reference_cell`` instead installs that equal-capacity shelter in the
        comparator branch.  This is the efficient sampled-policy-improvement
        path: subtracting one installed site directly from another removes the
        shared natural evolution and capacity contribution without spending a
        third physical branch on WAIT.
        """
        cells = tuple(int(cell) for cell in cell_indices)
        comparator_cell = (
            None if reference_cell is None else int(reference_cell)
        )
        seeds = tuple(None if seed is None else int(seed) for seed in future_seeds)
        if not cells:
            raise ValueError(
                "candidate_effects_many_futures requires at least one cell"
            )
        if len(cells) != len(set(cells)):
            raise ValueError(
                "candidate_effects_many_futures requires unique candidate cells"
            )
        if comparator_cell is not None and comparator_cell in set(cells):
            raise ValueError(
                "reference_cell must be distinct from every candidate cell"
            )
        if not seeds:
            raise ValueError(
                "candidate_effects_many_futures requires at least one future"
            )
        if self.horizon == 0:
            raise ValueError(
                "candidate_effects_many_futures is for physical positive-horizon NMCC"
            )

        base = snapshot if snapshot is not None else capture(
            self.core,
            label="nmcc-parallel-full-candidate-tables",
        )
        timelines = tuple(self._hazard_timeline(base, seed) for seed in seeds)
        tasks = tuple(
            _PhysicalBranchTask(
                tape_index=tape_index,
                action_position=action_position,
                install_cell=(
                    comparator_cell
                    if action_position < 0
                    else cells[action_position]
                ),
            )
            for tape_index in range(len(seeds))
            for action_position in (-1, *range(len(cells)))
        )
        local_builder = None
        if endpoint_spec is not None:
            local_builder = RegionalObservationBuilder(
                self.core,
                initial_population=endpoint_spec.initial_population,
                horizon=endpoint_spec.episode_horizon,
                maximum_deployments=endpoint_spec.maximum_deployments,
            )
        local_batch = _LocalBranchBatch(
            brancher=self,
            snapshot=base,
            future_seeds=seeds,
            hazard_timelines=timelines,
            endpoint_spec=endpoint_spec,
            endpoint_builder=local_builder,
        )

        try:
            if self.workers == 1:
                completed = [
                    _execute_physical_branch(local_batch, task) for task in tasks
                ]
            else:
                spawn_batch = _SpawnBranchBatch(
                    core=_portable_branch_core(self.core, base),
                    horizon=self.horizon,
                    reward_model=self.reward_model,
                    initial_population=self.initial_population,
                    episode_horizon=self.episode_horizon,
                    future_seeds=seeds,
                    hazard_timelines=timelines,
                    endpoint_spec=endpoint_spec,
                    node_flow=base.node_flow,
                    edge_flow=base.edge_flow,
                    python_random_state=base.python_random_state,
                    numpy_legacy_state=base.numpy_legacy_state,
                )
                completed = self._pool().run(spawn_batch, tasks)
        finally:
            restore(self.core, base)

        waited_by_tape: dict[int, BranchResult] = {}
        acted_by_tape: dict[int, dict[int, BranchResult]] = {
            tape_index: {} for tape_index in range(len(seeds))
        }
        for tape_index, action_position, result in completed:
            if action_position < 0:
                waited_by_tape[tape_index] = result
            else:
                acted_by_tape[tape_index][action_position] = result

        tables = []
        for tape_index in range(len(seeds)):
            waited = waited_by_tape[tape_index]
            tables.append(
                [
                    PairedEffect(
                        cell_index=cell,
                        horizon=self.horizon,
                        acted=acted_by_tape[tape_index][action_position],
                        waited=waited,
                        common_noise=True,
                    )
                    for action_position, cell in enumerate(cells)
                ]
            )
        return tables

    def paired_effect(
        self,
        cell_index: int,
        *,
        snapshot: Optional[SimulatorSnapshot] = None,
        common_noise: bool = True,
        independent_noise_seed: Optional[int] = None,
        wait_first: bool = False,
        endpoint_value: Optional[Callable[[Any, BranchResult], float]] = None,
    ) -> PairedEffect:
        """Measure one deployment's causal contribution over the horizon.

        ``common_noise=False`` reseeds the exogenous streams on the second
        branch.  That is the control condition for the variance-reduction
        audit, never a training configuration: it deliberately destroys the
        pairing so the two regimes can be compared.

        ``wait_first`` exists for the branch-order-invariance test.  If the
        order of the two branches changes either branch's outcome, the pairing
        is not valid and every effect measured through it would be an artifact.
        """
        if self.horizon == 0:
            if snapshot is not None:
                restore(self.core, snapshot)
            effect = _ImmediateCandidateEvaluator(self).effects(
                (int(cell_index),)
            )[0]
            if bool(common_noise):
                return effect
            return PairedEffect(
                cell_index=effect.cell_index,
                horizon=effect.horizon,
                acted=effect.acted,
                waited=effect.waited,
                common_noise=False,
            )
        base = snapshot if snapshot is not None else capture(self.core, label="decision")

        def run_acted() -> BranchResult:
            restore(self.core, base)
            result = self._run_branch(
                label="acted",
                install_cell=int(cell_index),
                steps=self.horizon,
            )
            if endpoint_value is not None and not result.terminal:
                result.continuation_value = float(endpoint_value(self.core, result))
            return result

        def run_waited() -> BranchResult:
            restore(self.core, base)
            if not common_noise:
                self._reseed_exogenous_streams(independent_noise_seed)
            result = self._run_branch(label="waited", install_cell=None, steps=self.horizon)
            if endpoint_value is not None and not result.terminal:
                result.continuation_value = float(endpoint_value(self.core, result))
            return result

        if wait_first:
            waited = run_waited()
            acted = run_acted()
        else:
            acted = run_acted()
            waited = run_waited()

        restore(self.core, base)
        return PairedEffect(
            cell_index=int(cell_index),
            horizon=self.horizon,
            acted=acted,
            waited=waited,
            common_noise=bool(common_noise),
        )

    def _reseed_exogenous_streams(self, seed: Optional[int]) -> None:
        """Break the pairing on purpose, for the independent-noise control."""
        key = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint64)[0]) if seed is not None else int(
            np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
        )
        mask = (1 << 64) - 1
        hazard_store = getattr(self.core, "hazardDS", None)
        if hazard_store is not None:
            hazard_store.rng = np.random.default_rng(key & mask)
        ped_store = getattr(self.core, "pedDS", None)
        if ped_store is not None:
            if hasattr(ped_store, "set_hazard_random_seed"):
                ped_store.set_hazard_random_seed((key ^ 0x5DEECE66D) & mask)
            if hasattr(ped_store, "panic_random_seed"):
                ped_store.panic_random_seed = (key ^ 0x9E3779B97F4A7C15) & mask


def counterfactual_advantage(
    effect: PairedEffect,
    *,
    gamma: float = 1.0,
    acted_bootstrap_value: float = 0.0,
    waited_bootstrap_value: float = 0.0,
    intervention_cost: float = 0.0,
) -> float:
    """NMCC's ``A_CF`` for one decision.

    ``sum_k gamma^k (r_k^a - r_k^0) + gamma^L (V(s_L^a) - V(s_L^0)) - c(a)``

    The bootstrap term is what keeps the branch horizon short: the critic, not
    the simulator, carries the comparison past ``L``.  The ``WAIT`` return is a
    control variate that does not depend on which cell was chosen, so the
    policy-gradient direction is preserved while the shared natural trajectory
    cancels.
    """
    horizon = max(effect.acted.steps, effect.waited.steps)
    difference = effect.discounted_difference(gamma)
    bootstrap = (gamma ** horizon) * (
        float(acted_bootstrap_value) - float(waited_bootstrap_value)
    )
    return float(difference + bootstrap - float(intervention_cost))


# -- Stage-0 audits ---------------------------------------------------------


def assert_replay_is_bitwise_identical(core, *, steps: int) -> str:
    """Prove a restored snapshot replays a factual trajectory exactly.

    NMCC calls this a scientific prerequisite rather than a nicety: training on
    invalid twins would manufacture high-confidence false causal labels, which
    is a worse failure than the noisy gradients it set out to fix.
    """
    base = capture(core, label="replay-audit")
    restore(core, base)
    for _ in range(int(steps)):
        advance_one_timestep(core)
    first = live_state_digest(core)

    restore(core, base)
    for _ in range(int(steps)):
        advance_one_timestep(core)
    second = live_state_digest(core)

    if first != second:
        raise AssertionError(
            "Factual replay from a restored snapshot is not reproducible: "
            f"{first} != {second}. Some mutable simulator state is escaping the "
            "snapshot, so paired counterfactuals would be invalid."
        )
    restore(core, base)
    return first


def assert_hazard_is_action_independent(core, *, steps: int, cell_index: int) -> None:
    """Check the exogeneity claim the pairing rests on.

    Hazard must evolve identically whether or not a shelter was installed.  If
    a shelter ever perturbed the fire front, the ``WAIT`` branch would no
    longer share the factual branch's hazard realization and the difference
    would confound the intervention with a different fire.
    """
    base = capture(core, label="hazard-exogeneity-audit")

    restore(core, base)
    for _ in range(int(steps)):
        advance_one_timestep(core)
    without = np.asarray(
        getattr(core.cellTracker, "dangerLevelByCell", []), dtype=np.float64
    ).copy()

    restore(core, base)
    _, cell_y = authoritative_grid_shape(core)
    cell = divmod(int(cell_index), cell_y)
    shelter_id = core.shelterDS.newShelter({"cell": cell}, core.cellTracker)
    if shelter_id is not None and hasattr(core.cellTracker, "addShelter"):
        core.cellTracker.addShelter(cell, core.shelterDS.shelterList[shelter_id])
    for _ in range(int(steps)):
        advance_one_timestep(core)
    with_action = np.asarray(
        getattr(core.cellTracker, "dangerLevelByCell", []), dtype=np.float64
    ).copy()

    restore(core, base)
    if not np.array_equal(without, with_action):
        raise AssertionError(
            "Installing a shelter changed the hazard trajectory. Hazard is not "
            "action-independent, so a WAIT branch does not share the factual "
            "branch's hazard realization and paired effects are confounded."
        )
