#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact paired policy improvement for sequential shelter deployment.

Candidate branches share a snapshot and future-noise tape, then continue under
one fixed base rule. Their full-horizon, within-state value differences train
the conservative controller. A deferred ``WAIT`` branch separately supervises
natural system evolution. This module is torch-free and uses the same simulator
and shelter executor as live deployment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

import CounterfactualBranch as CB
from DecisionInterface import (
    ActivePopulationHeuristic,
    OutcomeSnapshot,
    PolicyDecision,
    RegionalObservation,
    RegionalObservationBuilder,
    RegionalShelterExecutor,
    RiskTimeReductionHeuristic,
    RouteTimeSavingHeuristic,
)
from RewardProcessor import RewardProcessor

# -----------------------------------------------------------------------------
# Base policies used for branch continuation.  They read only the shared
# observation, so they are cheap and identical in every environment.
# -----------------------------------------------------------------------------

_ACTIVE = ActivePopulationHeuristic()
_RISK_REDUCTION = RiskTimeReductionHeuristic()
_ROUTE_SAVING = RouteTimeSavingHeuristic()


def risk_reduction_policy(observation: RegionalObservation) -> int:
    """Greedy capacity-capped reduction in future active plus exposure time."""
    return int(_RISK_REDUCTION.select(observation).action_index)


def route_saving_policy(observation: RegionalObservation) -> int:
    """Compatibility name for the pre-v27 fixed continuation policy."""
    return int(_ROUTE_SAVING.select(observation).action_index)


def active_population_policy(observation: RegionalObservation) -> int:
    return int(_ACTIVE.select(observation).action_index)


BASE_POLICIES: dict[str, Callable[[RegionalObservation], int]] = {
    "risk_reduction": risk_reduction_policy,
    "route_saving": route_saving_policy,
    "active_population": active_population_policy,
}


# -----------------------------------------------------------------------------
# Noise tapes
# -----------------------------------------------------------------------------


def tape_seed(episode_seed: int, decision_index: int, tape_index: int) -> int:
    """A future-noise tape that is independent of the live episode's future.

    Derived through ``SeedSequence`` with a distinct domain tag, so it cannot
    coincide with the seeds ``Core`` uses for the factual trajectory.
    """
    sequence = np.random.SeedSequence(
        [int(episode_seed) & 0xFFFFFFFF, 0x4E4D4343, int(decision_index), int(tape_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def reseed_exogenous(core, seed: int) -> None:
    """Replace every exogenous stream, from now on, with one tape.

    The present -- fire front, who is where, who has already panicked -- is
    unchanged; only draws that have not happened yet come from the new tape.
    That is the information set of a real operator.
    """
    mask = (1 << 64) - 1
    key = int(seed) & mask
    core.hazardDS.rng = np.random.default_rng(key)
    core.pedDS.set_hazard_random_seed((key ^ 0x5DEECE66D) & mask)
    if hasattr(core.pedDS, "panic_random_seed"):
        core.pedDS.panic_random_seed = (key ^ 0x9E3779B97F4A7C15) & mask


# -----------------------------------------------------------------------------
# Episode clock: the decision schedule and reward accounting of RLBridge
# -----------------------------------------------------------------------------


@dataclass
class EpisodeClock:
    """Everything a branch needs to continue an episode exactly as RLBridge would.

    Decisions occur at ``t = first_decision + k * interval`` while deployments
    remain, after the dynamics of timestep ``t`` and before ``t + 1``.  Rewards
    are scored per timestep with ``RewardProcessor``; the reward is linear in
    completions, casualties and person-time, so per-step scoring sums exactly to
    RLBridge's per-interval scoring.
    """

    horizon: int
    interval: int
    budget: int
    population: int
    first_decision: int = 1
    t: int = 0
    deployed: int = 0
    accumulated: float = 0.0
    previous: Optional[OutcomeSnapshot] = None
    start_outcome: Optional[OutcomeSnapshot] = None
    active_person_time: float = 0.0
    hazard_exposure_person_time: float = 0.0
    # Earliest physical time at which an unused deployment token may be
    # reconsidered.  This is state, rather than an arithmetic property of the
    # episode clock: if a scheduled decision has no safe candidate, operators
    # retry on the next simulator tick instead of losing that token forever.
    next_decision_time: Optional[int] = None

    def __post_init__(self) -> None:
        if self.start_outcome is None:
            self.start_outcome = self.previous
        if self.next_decision_time is None:
            self.next_decision_time = (
                int(self.first_decision)
                if int(self.deployed) == 0
                else int(self.first_decision) + int(self.deployed) * int(self.interval)
            )

    def is_decision(self, t: int) -> bool:
        return (
            self.deployed < self.budget
            and int(t) >= int(self.next_decision_time)
        )

    def mark_deployed(self, t: int) -> None:
        """Consume one token and schedule the next interval from this action."""
        self.deployed += 1
        self.next_decision_time = int(t) + int(self.interval)

    def defer_deployment(self, t: int) -> None:
        """Keep the token but postpone reconsideration to the next epoch."""
        self.next_decision_time = int(t) + int(self.interval)

    def copy(self) -> "EpisodeClock":
        return EpisodeClock(
            horizon=self.horizon,
            interval=self.interval,
            budget=self.budget,
            population=self.population,
            first_decision=self.first_decision,
            t=self.t,
            deployed=self.deployed,
            accumulated=self.accumulated,
            previous=self.previous,
            start_outcome=self.start_outcome,
            active_person_time=self.active_person_time,
            hazard_exposure_person_time=self.hazard_exposure_person_time,
            next_decision_time=self.next_decision_time,
        )

    def normalized_outcome_vector(self) -> np.ndarray:
        """Six interpretable physical outcomes accumulated by this branch.

        The target uses fixed population/horizon denominators, so it remains
        comparable across branch lengths, pedestrian volumes, and cell grids.
        """
        start = self.start_outcome
        end = self.previous
        if start is None or end is None:
            return np.zeros(6, dtype=np.float32)
        population = float(max(1, int(self.population)))
        horizon_mass = population * float(max(1, int(self.horizon)))
        return np.asarray(
            (
                (end.safe_completed - start.safe_completed) / population,
                (end.casualties - start.casualties) / population,
                self.active_person_time / horizon_mass,
                self.hazard_exposure_person_time / horizon_mass,
                end.active_population / population,
                end.risk_mass / (2.0 * population),
            ),
            dtype=np.float32,
        )


def step_and_score(core, clock: EpisodeClock, reward_model: RewardProcessor) -> float:
    """Advance one timestep and add its reward to the clock."""
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
    return float(breakdown.total)


# -----------------------------------------------------------------------------
# Branch valuation
# -----------------------------------------------------------------------------


@dataclass
class StateValues:
    """Exact paired values at one decision state."""

    decision_index: int
    simulation_time: int
    actions: np.ndarray  # branched action indices
    values: np.ndarray  # (tapes, len(actions)) return-to-go after installing
    wait_values: np.ndarray  # (tapes,) return-to-go with no deployment now
    outcomes: np.ndarray  # (tapes, len(actions), six physical coordinates)
    wait_outcomes: np.ndarray  # (tapes, six physical coordinates)
    base_action: int

    def mean_values(self) -> np.ndarray:
        return self.values.mean(axis=0)

    def paired_effects(self) -> np.ndarray:
        """``Q(s, c) - Q(s, WAIT)`` per tape: the natural-momentum residual."""
        return self.values - self.wait_values[:, None]

    def paired_outcome_effects(self) -> np.ndarray:
        """Candidate-caused physical change relative to natural momentum."""
        return self.outcomes - self.wait_outcomes[:, None, :]

    def contrast_standard_error(self) -> np.ndarray:
        """Standard error of each cell's within-state contrast across tapes.

        With one tape there is no estimate; callers fall back to ``eta_min``.
        """
        tapes = self.values.shape[0]
        if tapes < 2:
            return np.full(self.values.shape[1], np.nan)
        centered = self.values - self.values.mean(axis=1, keepdims=True)
        return centered.std(axis=0, ddof=1) / math.sqrt(tapes)


class BranchValuer:
    """Exact CRN-paired valuation of cells at a decision state.

    Every branch starts from one snapshot; within a tape, every cell and the
    ``WAIT`` branch share the same future disturbances, so their differences
    are paired.  Tapes are independent of the live episode's own future.  After
    valuation the simulator is restored exactly, so asking the question never
    changes the episode that asked it.
    """

    def __init__(
        self,
        core,
        *,
        builder: RegionalObservationBuilder,
        executor: RegionalShelterExecutor,
        reward_model: Optional[RewardProcessor] = None,
        base_policy: str = "risk_reduction",
        tapes: int = 1,
        branch_horizon: Optional[int] = None,
        full_horizon_decisions: int = 0,
    ):
        if base_policy not in BASE_POLICIES:
            raise ValueError(f"Unknown base policy {base_policy!r}; choose from {sorted(BASE_POLICIES)}")
        if int(tapes) <= 0:
            raise ValueError("tapes must be positive")
        self.core = core
        self.builder = builder
        self.executor = executor
        self.reward_model = reward_model if reward_model is not None else RewardProcessor()
        self.base_policy_name = str(base_policy)
        self.base_policy = BASE_POLICIES[self.base_policy_name]
        self.tapes = int(tapes)
        self.branch_horizon = None if branch_horizon is None else int(branch_horizon)
        self.full_horizon_decisions = max(0, int(full_horizon_decisions))

    def _install(self, observation: RegionalObservation, action: int) -> None:
        self.executor.execute(
            observation, PolicyDecision(action_index=int(action), strategy="nmcc_pi_branch")
        )

    def _continue(self, clock: EpisodeClock, *, full_horizon: bool = False) -> float:
        """Continue with the base policy to the horizon; return reward accrued."""
        start = clock.accumulated
        stop = clock.horizon
        if self.branch_horizon is not None and not bool(full_horizon):
            stop = min(stop, clock.t + self.branch_horizon)
        while clock.t < stop:
            step_and_score(self.core, clock, self.reward_model)
            if clock.is_decision(clock.t):
                observation = self.builder.build(
                    decision_index=clock.deployed,
                    simulation_time=clock.t,
                    remaining_deployments=clock.budget - clock.deployed,
                )
                if observation.has_feasible_action:
                    self._install(observation, self.base_policy(observation))
                    clock.mark_deployed(clock.t)
        return float(clock.accumulated - start)

    def value(
        self,
        observation: RegionalObservation,
        clock: EpisodeClock,
        requested_actions: "Sequence[int] | np.ndarray",
        *,
        episode_seed: int,
    ) -> StateValues:
        actions = np.asarray(sorted({int(a) for a in requested_actions}), dtype=np.int64)
        if actions.size == 0:
            raise ValueError("At least one action must be branched")
        if not bool(np.all(np.asarray(observation.action_mask)[actions])):
            raise ValueError("Every branched action must be feasible")

        base = CB.capture(self.core, label=f"nmcc-pi-{clock.deployed}")
        values = np.zeros((self.tapes, actions.size), dtype=np.float64)
        wait_values = np.zeros(self.tapes, dtype=np.float64)
        outcomes = np.zeros((self.tapes, actions.size, 6), dtype=np.float32)
        wait_outcomes = np.zeros((self.tapes, 6), dtype=np.float32)
        full_horizon = int(clock.deployed) < int(self.full_horizon_decisions)
        try:
            for tape in range(self.tapes):
                seed = tape_seed(episode_seed, clock.deployed, tape)
                for column, action in enumerate(actions):
                    CB.restore(self.core, base)
                    reseed_exogenous(self.core, seed)
                    branch_clock = clock.copy()
                    self._install(observation, int(action))
                    branch_clock.mark_deployed(branch_clock.t)
                    values[tape, column] = self._continue(
                        branch_clock,
                        full_horizon=full_horizon,
                    )
                    outcomes[tape, column] = branch_clock.normalized_outcome_vector()
                # WAIT: keep the token and let the base policy spend it at the
                # next epoch -- the natural momentum of the system with this
                # decision deferred, not a permanently wasted token.
                CB.restore(self.core, base)
                reseed_exogenous(self.core, seed)
                wait_clock = clock.copy()
                wait_clock.defer_deployment(wait_clock.t)
                wait_values[tape] = self._continue(
                    wait_clock,
                    full_horizon=full_horizon,
                )
                wait_outcomes[tape] = wait_clock.normalized_outcome_vector()
        finally:
            CB.restore(self.core, base)

        return StateValues(
            decision_index=int(clock.deployed),
            simulation_time=int(clock.t),
            actions=actions,
            values=values,
            wait_values=wait_values,
            outcomes=outcomes,
            wait_outcomes=wait_outcomes,
            base_action=int(self.base_policy(observation)),
        )


# -----------------------------------------------------------------------------
# Which cells to branch
# -----------------------------------------------------------------------------


def select_branch_actions(
    feasible: np.ndarray,
    probabilities: np.ndarray,
    *,
    decision_index: int,
    exhaustive_decisions: int,
    max_branches: int,
    must_include: Sequence[int] = (),
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose the cells to branch at this decision.

    The early decisions carry nearly all of the value (on the testbed the
    heuristic's regret is ~0.21 at t = 1 and <= 0.03 afterwards), so they are
    branched exhaustively.  Later decisions branch the policy's most probable
    cells -- where a ranking error would actually change behavior -- plus
    uniformly sampled others to keep support; NMCC's own Stage-2 protocol warns
    that branching only preferred actions preserves selection bias.
    """
    feasible = np.asarray(feasible, dtype=np.int64)
    if feasible.size == 0:
        return feasible
    if int(decision_index) < int(exhaustive_decisions) or feasible.size <= int(max_branches):
        return feasible.copy()
    chosen = {int(a) for a in must_include if int(a) in set(feasible.tolist())}
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = feasible[np.argsort(-probabilities[feasible], kind="mergesort")]
    top = max(1, int(max_branches) // 2)
    for action in order:
        if len(chosen) >= top:
            break
        chosen.add(int(action))
    remaining = [int(a) for a in feasible if int(a) not in chosen]
    extra = max(0, int(max_branches) - len(chosen))
    if extra and remaining:
        picks = rng.choice(len(remaining), size=min(extra, len(remaining)), replace=False)
        chosen.update(remaining[int(i)] for i in picks)
    return np.asarray(sorted(chosen), dtype=np.int64)


# -----------------------------------------------------------------------------
# KL-constrained improvement target (MPO E-step)
# -----------------------------------------------------------------------------


def _target_given_eta(pi_old: np.ndarray, advantage: np.ndarray, eta: float) -> np.ndarray:
    logits = np.log(np.clip(pi_old, 1e-300, None)) + advantage / float(eta)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.clip(q[mask], 1e-300, None)))))


@dataclass
class ImprovementTarget:
    """Target distribution over the full action table for one decision state."""

    target: np.ndarray  # (num_actions,) probability, zero where infeasible
    pi_old: np.ndarray  # (num_actions,)
    eta: float
    kl_to_old: float
    advantage: np.ndarray  # (num_actions,) within-state advantage, 0 where unbranched
    branched: np.ndarray  # (num_actions,) bool
    best_action: int  # branch-best action


def score_ranking_target(
    action_mask: np.ndarray,
    branched_actions: np.ndarray,
    branched_values: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Listwise target for an actor whose output is a candidate score.

    Only exactly branched candidates receive supervision.  Unobserved
    candidates receive no implicit negative label; the actor learns their
    ordering when they enter a later branch set.  At execution, the feasible
    candidate with the greatest learned score is installed.
    """
    mask = np.asarray(action_mask, dtype=bool)
    actions = np.asarray(branched_actions, dtype=np.int64)
    values = np.asarray(branched_values, dtype=np.float64)
    if actions.ndim != 1 or values.shape != actions.shape or actions.size < 1:
        raise ValueError("score ranking requires at least one paired candidate value")
    if float(temperature) <= 0.0 or not np.isfinite(float(temperature)):
        raise ValueError("ranking temperature must be finite and positive")
    if not bool(np.all(mask[actions])):
        raise ValueError("score target contains an infeasible candidate")
    standardized = (values - float(np.max(values))) / float(temperature)
    weights = np.exp(np.clip(standardized, -700.0, 0.0))
    weights /= float(weights.sum())
    target = np.zeros(mask.size, dtype=np.float64)
    target[actions] = weights
    return target


def improvement_target(
    pi_old: np.ndarray,
    action_mask: np.ndarray,
    branched_actions: np.ndarray,
    branched_values: np.ndarray,
    *,
    epsilon: float,
    eta_min: float,
    eta_max: float = 1e3,
    iterations: int = 60,
) -> ImprovementTarget:
    """Solve ``q ∝ pi_old exp(A / eta)`` on the branched cells with ``KL(q||pi_old)=epsilon``.

    Only branched cells are reweighted.  Their total probability mass is kept
    equal to ``pi_old``'s, and unbranched cells keep ``pi_old`` exactly: with no
    exact value for them there is no evidence to move them, and inventing one
    from an unvalidated model would reintroduce the false-label risk NMCC was
    designed to avoid.

    ``KL(q || pi_old)`` decreases monotonically in ``eta``, so ``eta`` is found
    by bisection in log space.  If even ``eta_min`` cannot reach ``epsilon`` --
    the branched values are nearly equal -- the target stops at ``eta_min``:
    a noise-level difference must not be turned into a confident label.
    """
    pi_old = np.asarray(pi_old, dtype=np.float64)
    mask = np.asarray(action_mask, dtype=bool)
    branched_actions = np.asarray(branched_actions, dtype=np.int64)
    branched_values = np.asarray(branched_values, dtype=np.float64)
    if branched_actions.size != branched_values.size:
        raise ValueError("branched_actions and branched_values must align")
    if not np.all(mask[branched_actions]):
        raise ValueError("Branched actions must be feasible")
    if epsilon <= 0 or eta_min <= 0:
        raise ValueError("epsilon and eta_min must be positive")

    pi = np.where(mask, pi_old, 0.0)
    total = pi.sum()
    if total <= 0:
        raise ValueError("pi_old assigns no mass to feasible actions")
    pi = pi / total

    sub_pi = pi[branched_actions]
    sub_mass = float(sub_pi.sum())
    branched = np.zeros(pi.size, dtype=bool)
    branched[branched_actions] = True
    advantage_full = np.zeros(pi.size, dtype=np.float64)
    best_action = int(branched_actions[int(np.argmax(branched_values))])

    if branched_actions.size < 2 or sub_mass <= 0.0:
        return ImprovementTarget(pi, pi, float("inf"), 0.0, advantage_full, branched, best_action)

    conditional = sub_pi / sub_mass
    advantage = branched_values - float(np.dot(conditional, branched_values))
    advantage_full[branched_actions] = advantage

    def assemble(eta: float) -> np.ndarray:
        q = pi.copy()
        q[branched_actions] = sub_mass * _target_given_eta(conditional, advantage, eta)
        return q

    q_floor = assemble(eta_min)
    if _kl(q_floor, pi) <= epsilon:
        eta = float(eta_min)
        q = q_floor
    else:
        lo, hi = math.log(eta_min), math.log(eta_max)
        for _ in range(int(iterations)):
            mid = 0.5 * (lo + hi)
            if _kl(assemble(math.exp(mid)), pi) > epsilon:
                lo = mid
            else:
                hi = mid
        eta = math.exp(hi)
        q = assemble(eta)
    return ImprovementTarget(q, pi, float(eta), _kl(q, pi), advantage_full, branched, best_action)


# -----------------------------------------------------------------------------
# Training record for one decision
# -----------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """Everything the learner needs from one decision, independent of framework."""

    features: Optional[np.ndarray]  # (num_actions, d) for the reference learner
    action_mask: np.ndarray
    target: ImprovementTarget
    values: StateValues
    executed_action: int
    simulation_time: int
    decision_index: int
    extras: dict = field(default_factory=dict)


def softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.where(mask, np.asarray(logits, dtype=np.float64) / float(temperature), -np.inf)
    z = z - z[mask].max()
    w = np.where(mask, np.exp(z), 0.0)
    return w / w.sum()


def spearman(first: "Sequence[float] | np.ndarray", second: "Sequence[float] | np.ndarray") -> float:
    """Spearman rank correlation with average ranks for ties; NaN if undefined."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")

    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(x.size, dtype=np.float64)
        r[order] = np.arange(x.size, dtype=np.float64)
        for value in np.unique(x):
            tied = x == value
            if tied.sum() > 1:
                r[tied] = r[tied].mean()
        return r

    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    return float("nan") if denominator <= 0.0 else float(np.dot(ra, rb) / denominator)


def within_state_rank_agreement(
    predictions: np.ndarray,
    advantages: np.ndarray,
    exact_masks: np.ndarray,
    *,
    minimum_cells: int = 3,
) -> float:
    """Mean within-state Spearman between predicted and exact advantages.

    Only states with at least ``minimum_cells`` exactly branched cells count;
    ranking two cells is too coarse to be evidence.  This is the ranking gate
    NMCC's Stage-2 protocol specifies but no earlier run measured.
    """
    scores = []
    for prediction, advantage, mask in zip(predictions, advantages, exact_masks):
        mask = np.asarray(mask) > 0.5
        if mask.sum() < int(minimum_cells):
            continue
        value = spearman(np.asarray(prediction)[mask], np.asarray(advantage)[mask])
        if np.isfinite(value):
            scores.append(value)
    return float(np.mean(scores)) if scores else float("nan")
