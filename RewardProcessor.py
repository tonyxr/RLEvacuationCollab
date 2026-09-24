#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stateless outcome reward for exact shelter-candidate decisions.

For decision interval ``k`` the training reward is

    r_k = (Delta safe_k - 3 Delta casualty_k) / P
          - active_person_time_k / (P H)
          - hazard_exposure_person_time_k / (P H)

The two person-time terms are intentionally separate: the first rewards faster
completion for everyone still evacuating; the second additionally penalizes
time spent in hazardous regions.  Since normalized danger is in ``[0, 1]``, a
person can avoid at most two normalized units of future time cost by leaving
the active population.  The casualty coefficient of three therefore makes a
death strictly worse than that artificial shortcut. No shelter-service shaping
or site-selection bonus is included; policy and evaluation optimize the same
population outcome objective.
"""

from dataclasses import asdict, dataclass

import numpy as np

from DecisionInterface import OutcomeSnapshot


REWARD_COMPONENT_NAMES = (
    "safe_completion",
    "casualty",
    "evacuation_time",
    "hazard_exposure",
)

# One source of truth for the physical objective.  The recurrent critic,
# natural/causal outcome model, exact branch scorer, training reward and held-
# out evaluator all import these constants; changing a reward weight can no
# longer leave the GNN optimizing a stale hard-coded decomposition.
DEFAULT_SAFE_COMPLETION_WEIGHT = 1.0
DEFAULT_CASUALTY_WEIGHT = 3.0
DEFAULT_EVACUATION_TIME_WEIGHT = 1.0
DEFAULT_HAZARD_EXPOSURE_WEIGHT = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    safe_completion_reward: float
    casualty_penalty: float
    evacuation_time_penalty: float
    hazard_exposure_penalty: float
    total: float
    new_safe_completions: int
    new_casualties: int
    active_person_time: float
    hazard_exposure_person_time: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def component_vector(self) -> np.ndarray:
        """Return the signed objective branches in the registered order."""
        return np.asarray(
            (
                self.safe_completion_reward,
                self.casualty_penalty,
                self.evacuation_time_penalty,
                self.hazard_exposure_penalty,
            ),
            dtype=np.float32,
        )

    @property
    def risk_time_penalty(self) -> float:
        """Backward-compatible aggregate of the two explicit time terms."""
        return self.evacuation_time_penalty + self.hazard_exposure_penalty

    @property
    def risk_weighted_person_time(self) -> float:
        return self.active_person_time + self.hazard_exposure_person_time

    @property
    def shelter_service_reward(self) -> float:
        """Deprecated diagnostic: service shaping is excluded from the reward."""
        return 0.0

    @property
    def attributed_shelter_service(self) -> int:
        return 0


class RewardProcessor:
    """Pure interval reward with fixed, interpretable normalization."""

    def __init__(
        self,
        *,
        casualty_weight: float = DEFAULT_CASUALTY_WEIGHT,
        evacuation_time_weight: float = DEFAULT_EVACUATION_TIME_WEIGHT,
        hazard_exposure_weight: float = DEFAULT_HAZARD_EXPOSURE_WEIGHT,
    ):
        evacuation_time_weight = float(evacuation_time_weight)
        hazard_exposure_weight = float(hazard_exposure_weight)
        if not np.isfinite(evacuation_time_weight) or evacuation_time_weight <= 0.0:
            raise ValueError("evacuation_time_weight must be finite and positive")
        if not np.isfinite(hazard_exposure_weight) or hazard_exposure_weight <= 0.0:
            raise ValueError("hazard_exposure_weight must be finite and positive")
        casualty_weight = float(casualty_weight)
        maximum_avoided_time_cost = evacuation_time_weight + hazard_exposure_weight
        if not np.isfinite(casualty_weight) or casualty_weight <= maximum_avoided_time_cost:
            raise ValueError(
                "casualty_weight must exceed the sum of the evacuation-time and "
                "hazard-exposure weights so casualties cannot reduce total cost"
            )
        self.casualty_weight = casualty_weight
        self.evacuation_time_weight = evacuation_time_weight
        self.hazard_exposure_weight = hazard_exposure_weight

    def evaluate(
        self,
        *,
        before: OutcomeSnapshot,
        after: OutcomeSnapshot,
        active_person_time: float,
        hazard_exposure_person_time: float,
        initial_population: int,
        horizon: int,
    ) -> RewardBreakdown:
        population = int(initial_population)
        horizon = int(horizon)
        if population <= 0:
            raise ValueError("initial_population must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        active_time = float(active_person_time)
        exposure_time = float(hazard_exposure_person_time)
        if not np.isfinite(active_time) or active_time < 0.0:
            raise ValueError("active_person_time must be finite and non-negative")
        if not np.isfinite(exposure_time) or exposure_time < 0.0:
            raise ValueError(
                "hazard_exposure_person_time must be finite and non-negative"
            )

        new_safe = int(after.safe_completed - before.safe_completed)
        new_casualties = int(after.casualties - before.casualties)
        if new_safe < 0 or new_casualties < 0:
            raise ValueError("Cumulative population outcomes must be monotone")
        safe_reward = (
            DEFAULT_SAFE_COMPLETION_WEIGHT * float(new_safe) / float(population)
        )
        casualty_penalty = -self.casualty_weight * float(new_casualties) / float(population)
        evacuation_time_penalty = -(
            self.evacuation_time_weight * active_time / float(population * horizon)
        )
        hazard_exposure_penalty = -(
            self.hazard_exposure_weight * exposure_time / float(population * horizon)
        )
        total = (
            safe_reward
            + casualty_penalty
            + evacuation_time_penalty
            + hazard_exposure_penalty
        )
        if not np.isfinite(total):
            raise FloatingPointError("Reward total is non-finite")

        return RewardBreakdown(
            safe_completion_reward=safe_reward,
            casualty_penalty=casualty_penalty,
            evacuation_time_penalty=evacuation_time_penalty,
            hazard_exposure_penalty=hazard_exposure_penalty,
            total=float(total),
            new_safe_completions=new_safe,
            new_casualties=new_casualties,
            active_person_time=active_time,
            hazard_exposure_person_time=exposure_time,
        )
