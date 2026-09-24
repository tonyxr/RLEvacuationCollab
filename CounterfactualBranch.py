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

The remedy is not to remove the stochasticity but to *share* it.  Comparing a
deployment against no deployment on the same noise realization gives

    Var[G(a, U) - G(0, U)] = Var[G(a)] + Var[G(0)] - 2 Cov[G(a), G(0)],

and because both branches ride the same hazard front, the same person-level
casualty shocks and the same panic draws, that covariance sits close to the
variances and the difference variance collapses.  This is the classical
common-random-numbers variance reduction of Kleinman, Spall and Naiman (1999)
used as a control variate: ``G(0, U)`` does not depend on which cell was
chosen, so subtracting it leaves the policy-gradient direction unchanged while
removing the shared natural trajectory.

Why this is affordable
----------------------
Running the whole episode twice per decision costs ``O(H)`` extra timesteps
per decision and is indeed prohibitive.  Three properties make the estimator
cheap instead:

1. **Bounded branch horizon.**  The counterfactual runs for ``L`` timesteps
   only; the critic closes the remaining gap through the bootstrap term
   ``gamma**L * (V(s_L^a) - V(s_L^0))``.  ``L`` is a bias/variance knob, not
   the episode length.
2. **One baseline serves every action.**  ``WAIT`` is action-independent, so a
   single no-deployment branch is a valid control variate for all
   ``number_of_cells`` actions simultaneously.  The cost is ``2x``, not
   ``|C|x`` -- which is what makes this tractable at full grid resolution.
3. **The factual branch is free during training.**  The real episode already
   simulates "deploy, then run on to the next decision epoch".  When ``L``
   equals the decision interval, only the ``WAIT`` branch is extra work.

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

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from DecisionInterface import OutcomeSnapshot


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
    label: str = ""

    def state_digest(self) -> str:
        """Content hash of the branch-varying state, for replay audits."""
        hasher = hashlib.sha256()
        hasher.update(self.node_flow.tobytes())
        hasher.update(self.edge_flow.tobytes())
        hasher.update(repr(self.hazard_bit_generator_state).encode("utf-8"))
        hasher.update(str(self.simulation_time).encode("utf-8"))
        return hasher.hexdigest()


def capture(core, *, label: str = "") -> SimulatorSnapshot:
    """Capture the simulator's branch-varying state without disturbing it."""
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
            hazard_store.cellTracker = restored_tracker

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


def advance_one_timestep(core) -> None:
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
    core.hazardDS.spreadUpdate()
    core.hazardDS.heatUpdate()
    core.hazardDS.smokeUpdate()
    core.hazardDS.terminateHazard()
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


@dataclass
class BranchResult:
    """One branch's realized trajectory over the counterfactual horizon."""

    label: str
    rewards: list[float] = field(default_factory=list)
    reward_components: list[np.ndarray] = field(default_factory=list)
    active_person_time: float = 0.0
    hazard_exposure_person_time: float = 0.0
    start_outcome: Optional[OutcomeSnapshot] = None
    end_outcome: Optional[OutcomeSnapshot] = None
    installed_cell: int = -1
    installed_capacity: float = 0.0
    steps: int = 0
    terminal: bool = False
    end_digest: str = ""

    def discounted_return(self, gamma: float = 1.0) -> float:
        total = 0.0
        for index, value in enumerate(self.rewards):
            total += (gamma ** index) * float(value)
        return float(total)

    def discounted_component_return(self, gamma: float = 1.0) -> np.ndarray:
        """Discounted signed reward components in the registered order."""
        if not self.reward_components:
            return np.zeros(4, dtype=np.float64)
        total = np.zeros_like(
            np.asarray(self.reward_components[0], dtype=np.float64)
        )
        for index, value in enumerate(self.reward_components):
            total += (float(gamma) ** index) * np.asarray(value, dtype=np.float64)
        return total

    def outcome_vector(self) -> dict[str, float]:
        """Vector-valued outcome, per NMCC's instruction to keep ``Y`` factored.

        Collapsing to a scalar reward before differencing would let a large
        safe-completion gain hide a casualty or exposure regression, which is
        precisely the failure mode an evacuation planner must be able to audit.
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


class CounterfactualBrancher:
    """Run paired ``act`` versus ``WAIT`` branches from a decision state.

    The branch horizon ``L`` should normally equal the deployment interval, so
    that the acted branch coincides with the trajectory the episode is about to
    live anyway and only the ``WAIT`` branch is additional work.
    """

    def __init__(
        self,
        core,
        *,
        horizon: int,
        reward_model,
        initial_population: int,
        episode_horizon: int,
    ):
        if int(horizon) <= 0:
            raise ValueError("Counterfactual branch horizon must be positive")
        if int(initial_population) <= 0:
            raise ValueError("initial_population must be positive")
        if int(episode_horizon) <= 0:
            raise ValueError("episode_horizon must be positive")
        self.core = core
        self.horizon = int(horizon)
        self.reward_model = reward_model
        self.initial_population = int(initial_population)
        self.episode_horizon = int(episode_horizon)

    # -- single branch ----------------------------------------------------

    def _run_branch(
        self,
        *,
        label: str,
        install_cell: Optional[int],
        steps: int,
    ) -> BranchResult:
        core = self.core
        result = BranchResult(label=label)
        result.start_outcome = outcome_snapshot(core)

        if install_cell is not None:
            cell_index = int(install_cell)
            cell = divmod(cell_index, int(core.cellY))
            shelter_id = core.shelterDS.newShelter({"cell": cell}, core.cellTracker)
            if shelter_id is None:
                raise RuntimeError(
                    f"Counterfactual branch could not install a shelter in cell {cell_index}"
                )
            shelter = core.shelterDS.shelterList[shelter_id]
            if hasattr(core.cellTracker, "addShelter"):
                core.cellTracker.addShelter(cell, shelter)
            reroute = getattr(core.pedDS, "reroute_to_new_shelter_if_closer", None)
            if callable(reroute):
                reroute(shelter)
            result.installed_cell = cell_index
            result.installed_capacity = max(0.0, float(getattr(shelter, "shelterCap", 0.0)))

        previous = result.start_outcome
        for _ in range(int(steps)):
            advance_one_timestep(core)
            current = outcome_snapshot(core)

            # Person-time accrues over the timestep just simulated, using the
            # same masses RLBridge accumulates between decision epochs.
            active_mass = float(current.active_population)
            exposure_mass = float(current.hazard_exposure_mass)
            result.active_person_time += active_mass
            result.hazard_exposure_person_time += exposure_mass

            breakdown = self.reward_model.evaluate(
                before=previous,
                after=current,
                active_person_time=active_mass,
                hazard_exposure_person_time=exposure_mass,
                initial_population=self.initial_population,
                horizon=self.episode_horizon,
            )
            result.rewards.append(float(breakdown.total))
            result.reward_components.append(breakdown.component_vector())
            result.steps += 1
            previous = current
            if current.active_population <= 0:
                result.terminal = True
                break

        result.end_outcome = previous
        result.end_digest = live_state_digest(core)
        return result

    # -- paired branches --------------------------------------------------

    def paired_effect(
        self,
        cell_index: int,
        *,
        snapshot: Optional[SimulatorSnapshot] = None,
        common_noise: bool = True,
        independent_noise_seed: Optional[int] = None,
        wait_first: bool = False,
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
        base = snapshot if snapshot is not None else capture(self.core, label="decision")

        def run_acted() -> BranchResult:
            restore(self.core, base)
            return self._run_branch(
                label="acted",
                install_cell=int(cell_index),
                steps=self.horizon,
            )

        def run_waited() -> BranchResult:
            restore(self.core, base)
            if not common_noise:
                self._reseed_exogenous_streams(independent_noise_seed)
            return self._run_branch(label="waited", install_cell=None, steps=self.horizon)

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
    cell = divmod(int(cell_index), int(core.cellY))
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
