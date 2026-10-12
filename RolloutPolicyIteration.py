#!/usr/bin/env python3
"""Simulator-graded targets for residual rollout policy iteration.

This module deliberately contains no reward or simulator logic.  It provides
the two learner-side operations needed by rollout policy iteration:

* a frozen, deterministic residual policy that can be serialized to isolated
  simulator workers and used for every later decision in a branch; and
* paired cost estimates and conservative categorical targets made from
  complete common-random-number rollouts.

The base score is the exact same-state objective-greedy assignment gain already
present in ``RegionalObservation.candidate_assignment_gain``.  The neural
actor learns only the correction justified by the simulator-graded rollout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
import torch

from DecisionInterface import (
    HAZARD_FEATURE_SLICE,
    INFRA_FEATURE_SLICE,
    PED_FEATURE_SLICE,
    RegionalObservation,
)
from GNN import fit_gnn


def momentum_array(
    previous: Optional[RegionalObservation],
    current: RegionalObservation,
    *,
    initial_population: int,
    shelter_action_interval: int,
) -> np.ndarray:
    """Return the four existing causal momentum rates without changing them."""
    if previous is None:
        return np.zeros(4, dtype=np.float32)
    elapsed = int(current.simulation_time) - int(previous.simulation_time)
    if elapsed <= 0:
        return np.zeros(4, dtype=np.float32)

    population = float(max(1, int(initial_population)))
    interval_scale = float(max(1, int(shelter_action_interval))) / float(elapsed)

    def burdens(item: RegionalObservation) -> tuple[float, float, float]:
        active = np.asarray(item.active_by_cell, dtype=np.float64)
        congestion_delay = np.clip(
            1.0
            - np.asarray(item.congestion_speed_ratio_by_cell, dtype=np.float64),
            0.0,
            1.0,
        )
        route_scale = max(
            float(item.time_step_minutes),
            float(item.horizon) * float(item.time_step_minutes),
        )
        route_time = np.clip(
            np.asarray(item.mean_route_time_by_cell, dtype=np.float64) / route_scale,
            0.0,
            1.0,
        )
        danger = np.maximum(
            np.asarray(item.danger_by_cell, dtype=np.float64),
            np.asarray(item.forecast_danger_by_cell, dtype=np.float64),
        )
        return (
            float(np.sum(active * congestion_delay) / population),
            float(np.sum(active * route_time) / population),
            float(np.sum(active * danger) / population),
        )

    previous_congestion, previous_route, previous_exposure = burdens(previous)
    current_congestion, current_route, current_exposure = burdens(current)
    values = np.asarray(
        (
            (
                previous.outcome.active_population
                - current.outcome.active_population
            )
            / population,
            previous_congestion - current_congestion,
            previous_route - current_route,
            previous_exposure - current_exposure,
        ),
        dtype=np.float32,
    )
    return np.clip(values * interval_scale, -1.0, 1.0).astype(
        np.float32, copy=False
    )


@dataclass
class FrozenResidualPolicy:
    """CPU snapshot of one policy-iteration actor.

    A branch owns its recurrent state locally.  The object is immutable during
    selection, so every candidate and every common-noise tape starts with the
    exact same actor, previous observation, and recurrent history.
    """

    policy: object
    spatial_edge_index: torch.Tensor
    initial_population: int
    shelter_action_interval: int
    previous_observation: RegionalObservation
    recurrent_state: Optional[tuple[torch.Tensor, torch.Tensor]]

    @classmethod
    def capture(
        cls,
        policy,
        spatial_edge_index: torch.Tensor,
        *,
        initial_population: int,
        shelter_action_interval: int,
        previous_observation: RegionalObservation,
        recurrent_state: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> "FrozenResidualPolicy":
        frozen = copy.deepcopy(policy).to(torch.device("cpu"))
        frozen.eval()
        state = None
        if recurrent_state is not None:
            state = tuple(
                value.detach().to(torch.device("cpu")).clone()
                for value in recurrent_state
            )
        return cls(
            policy=frozen,
            spatial_edge_index=spatial_edge_index.detach().cpu().clone(),
            initial_population=int(initial_population),
            shelter_action_interval=int(shelter_action_interval),
            previous_observation=previous_observation,
            recurrent_state=state,
        )

    def initial_context(
        self,
    ) -> tuple[RegionalObservation, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        state = None
        if self.recurrent_state is not None:
            state = tuple(value.clone() for value in self.recurrent_state)
        return self.previous_observation, state

    @staticmethod
    def _masked_argmax(logits: torch.Tensor, mask: np.ndarray) -> int:
        feasible = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
        flat = logits.detach().reshape(-1).cpu()
        if flat.shape != feasible.shape or not bool(feasible.any()):
            raise RuntimeError("Frozen rollout policy received an invalid action table")
        masked = flat.masked_fill(~feasible, torch.finfo(flat.dtype).min)
        return int(torch.argmax(masked).item())

    def select(
        self,
        observation: RegionalObservation,
        *,
        previous_observation: RegionalObservation,
        recurrent_state: Optional[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        cell_features_np, global_features_np = observation.policy_features()
        candidate_features_np = observation.candidate_features()
        cell_features = torch.as_tensor(cell_features_np, dtype=torch.float32)
        graph = fit_gnn(
            x_ped=cell_features[:, PED_FEATURE_SLICE],
            x_hazard=cell_features[:, HAZARD_FEATURE_SLICE],
            x_infra=cell_features[:, INFRA_FEATURE_SLICE],
            x_global=torch.as_tensor(global_features_np, dtype=torch.float32),
            edge_index=self.spatial_edge_index,
            route_edge_index=torch.as_tensor(
                observation.route_edge_index, dtype=torch.long
            ),
            route_edge_weight=torch.as_tensor(
                observation.route_edge_weight, dtype=torch.float32
            ),
            candidate_cell_index=torch.as_tensor(
                observation.candidate_cell_indices, dtype=torch.long
            ),
            candidate_features=torch.as_tensor(
                candidate_features_np, dtype=torch.float32
            ),
        )
        momentum = torch.as_tensor(
            momentum_array(
                previous_observation,
                observation,
                initial_population=self.initial_population,
                shelter_action_interval=self.shelter_action_interval,
            ),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits, _, _, next_state = self.policy.forward_nmcc_actor_recurrent(
                graph,
                recurrent_state,
                momentum,
            )
        action = self._masked_argmax(logits, observation.action_mask)
        return action, tuple(value.detach() for value in next_state)


def select_rollout_panel(
    feasible_actions: Sequence[int],
    base_scores: np.ndarray,
    *,
    max_branches: int,
    must_include: Sequence[int] = (),
    uniform_extras: int = 2,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a strong panel while giving every feasible cell a real chance.

    Required actions enter first. The remaining budget is split between the
    strongest objective-greedy scores and a uniform sample without replacement
    from all other feasible actions. This does not use actor logits or stale
    cross-iteration coverage counts, and every leftover candidate has nonzero
    probability of receiving a complete simulator grade.
    """
    feasible = np.asarray(feasible_actions, dtype=np.int64).reshape(-1)
    scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
    if feasible.size == 0:
        raise ValueError("Rollout panel requires at least one feasible action")
    if scores.shape != feasible.shape or not np.isfinite(scores).all():
        raise ValueError(
            "Rollout panel requires one finite base score per feasible action"
        )
    if int(max_branches) <= 0 or feasible.size <= int(max_branches):
        return feasible.copy()

    budget = int(max_branches)
    uniform_extras = max(0, int(uniform_extras))
    required = list(dict.fromkeys(int(action) for action in must_include))
    feasible_set = set(feasible.tolist())
    if any(action not in feasible_set for action in required):
        raise ValueError("Required rollout contender is not feasible")
    if len(required) > budget:
        raise ValueError("Required rollout contenders exceed the branch budget")

    selected = list(required)
    remaining = [action for action in feasible.tolist() if action not in selected]
    random_slots = min(uniform_extras, budget - len(selected), len(remaining))
    top_slots = budget - len(selected) - random_slots
    position = {int(action): index for index, action in enumerate(feasible)}
    ranked = sorted(
        remaining,
        key=lambda action: (-float(scores[position[action]]), int(action)),
    )
    selected.extend(ranked[:top_slots])
    remaining = [action for action in remaining if action not in set(selected)]
    if random_slots:
        chosen = rng.choice(
            np.asarray(remaining, dtype=np.int64),
            size=random_slots,
            replace=False,
        )
        selected.extend(int(action) for action in np.asarray(chosen).reshape(-1))
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError("Rollout panel construction did not fill its budget")
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class RolloutCostEstimate:
    score: np.ndarray
    mean_cost: np.ndarray
    standard_error: np.ndarray
    paired_contrast_standard_error: float


def estimate_rollout_costs(costs: np.ndarray) -> RolloutCostEstimate:
    """Summarize complete shared-future costs for categorical improvement.

    ``costs`` has shape ``(common futures, evaluated actions)``. Utilities stay
    in physical person-minute units; the KL-constrained E-step, rather than a
    normalization or regression scale, controls the size of a policy change.
    """
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(
            "Rollout grading requires at least two shared futures and "
            "two actions"
        )
    if not np.isfinite(values).all():
        raise ValueError("Rollout costs contain non-finite values")
    mean_cost = values.mean(axis=0)
    utility = -mean_cost
    score = utility - utility.mean()

    standard_error = values.std(axis=0, ddof=1) / math.sqrt(
        float(values.shape[0])
    )
    paired_difference = values[:, :, None] - values[:, None, :]
    paired_se = paired_difference.std(axis=0, ddof=1) / math.sqrt(
        float(values.shape[0])
    )
    off_diagonal = paired_se[np.triu_indices(values.shape[1], k=1)]
    paired_noise = float(np.median(off_diagonal))
    return RolloutCostEstimate(
        score=score.astype(np.float32),
        mean_cost=mean_cost.astype(np.float64),
        standard_error=standard_error.astype(np.float64),
        paired_contrast_standard_error=paired_noise,
    )
