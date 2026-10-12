#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Counterfactual shelter-site reward in hazard-weighted time-to-safety units.

The active model compares a selected site with the uniform average over all
feasible sites.  Every branch installs the same capacity token, so this
centering removes the capacity value shared by every action and preserves only
location, route, congestion, exposure, and continuation effects.  Archived
utilities can still request the former WAIT comparison through
``evaluate_counterfactual``.

    r_k = (mean_a J(s_k, a) - J(s_k, chosen)) / (P H_minutes)

``J`` is measured in hazard-weighted pedestrian-minutes.  The site advantage is
signed: a worse-than-random site is negative and a better-than-random site is
positive.  Realised full-episode time-to-safety is accounted separately so the
evaluation objective is independent of this variance-reducing baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from DecisionInterface import OutcomeSnapshot


REWARD_COMPONENT_NAMES = ("forward_hazard_weighted_progress",)

# Retained as neutral compatibility constants for modules that import them as
# architecture metadata.  The new reward has no separately weighted terms.
DEFAULT_EVACUATION_TIME_WEIGHT = 0.0
DEFAULT_HAZARD_EXPOSURE_WEIGHT = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    """One installation's auditable signed site advantage."""

    total: float
    raw_total: float
    remaining_risk_time_before: float
    remaining_risk_time_after: float
    raw_remaining_risk_time_reduction: float
    remaining_risk_time_reduction: float
    plain_time_wait: float
    plain_time_action: float
    raw_plain_time_reduction: float
    hazard_exposure_wait: float
    hazard_exposure_action: float
    raw_hazard_exposure_reduction: float
    new_safe_completions: int
    new_casualties: int
    initial_population: int
    horizon_minutes: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def component_vector(self) -> np.ndarray:
        return np.asarray((self.total,), dtype=np.float32)

    def raw_component_vector(self) -> np.ndarray:
        """Signed causal gain used to rank actions before display clipping."""
        return np.asarray((self.raw_total,), dtype=np.float32)

    # Deprecated report aliases remain read-only so existing CSV consumers can
    # read new runs without influencing the optimized objective.
    @property
    def safe_completion_reward(self) -> float:
        return 0.0

    @property
    def casualty_penalty(self) -> float:
        return 0.0

    @property
    def evacuation_time_penalty(self) -> float:
        return 0.0

    @property
    def hazard_exposure_penalty(self) -> float:
        return float(self.total)

    @property
    def terminal_residual_penalty(self) -> float:
        return 0.0

    @property
    def terminal_residual_person_time(self) -> float:
        return 0.0

    @property
    def risk_time_penalty(self) -> float:
        return float(self.total)

    @property
    def risk_weighted_person_time(self) -> float:
        return float(self.remaining_risk_time_reduction)

    @property
    def active_person_time(self) -> float:
        return 0.0

    @property
    def hazard_exposure_person_time(self) -> float:
        return float(self.remaining_risk_time_reduction)

    @property
    def evacuation_time_weight(self) -> float:
        return 0.0

    @property
    def hazard_exposure_weight(self) -> float:
        return 1.0

    @property
    def shelter_service_reward(self) -> float:
        return 0.0

    @property
    def attributed_shelter_service(self) -> int:
        return 0


class RewardProcessor:
    """Convert time-to-safety cost contrasts into normalized reward."""

    evacuation_time_weight = DEFAULT_EVACUATION_TIME_WEIGHT
    hazard_exposure_weight = DEFAULT_HAZARD_EXPOSURE_WEIGHT

    @staticmethod
    def _cost(name: str, value: float) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def evaluate_counterfactual(
        self,
        *,
        start: OutcomeSnapshot,
        action_after: OutcomeSnapshot,
        plain_time_wait: float,
        plain_time_action: float,
        hazard_exposure_wait: float,
        hazard_exposure_action: float,
        initial_population: int,
        horizon_minutes: int | float,
        signed: bool = False,
    ) -> RewardBreakdown:
        """Score the matched WAIT-minus-action exposure-to-safety cost."""
        population = int(initial_population)
        horizon = float(horizon_minutes)
        if population <= 0:
            raise ValueError("initial_population must be positive")
        if not np.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("horizon_minutes must be finite and positive")

        wait_plain = self._cost("plain_time_wait", plain_time_wait)
        action_plain = self._cost("plain_time_action", plain_time_action)
        wait_exposure = self._cost(
            "hazard_exposure_wait", hazard_exposure_wait
        )
        action_exposure = self._cost(
            "hazard_exposure_action", hazard_exposure_action
        )
        plain_reduction = wait_plain - action_plain
        exposure_reduction = wait_exposure - action_exposure
        wait_total = wait_plain + wait_exposure
        action_total = action_plain + action_exposure
        raw_reduction = plain_reduction + exposure_reduction
        denominator = float(population) * horizon
        raw_total = raw_reduction / denominator
        reduction = raw_reduction if bool(signed) else max(0.0, raw_reduction)
        total = reduction / denominator
        if not np.isfinite(raw_total) or not np.isfinite(total):
            raise FloatingPointError("Reward total is non-finite")

        new_safe = int(action_after.safe_completed - start.safe_completed)
        new_casualties = int(action_after.casualties - start.casualties)
        if new_safe < 0 or new_casualties < 0:
            raise ValueError("Cumulative population outcomes must be monotone")

        return RewardBreakdown(
            total=float(total),
            raw_total=float(raw_total),
            remaining_risk_time_before=float(wait_total),
            remaining_risk_time_after=float(action_total),
            raw_remaining_risk_time_reduction=float(raw_reduction),
            remaining_risk_time_reduction=float(reduction),
            plain_time_wait=float(wait_plain),
            plain_time_action=float(action_plain),
            raw_plain_time_reduction=float(plain_reduction),
            hazard_exposure_wait=float(wait_exposure),
            hazard_exposure_action=float(action_exposure),
            raw_hazard_exposure_reduction=float(exposure_reduction),
            new_safe_completions=new_safe,
            new_casualties=new_casualties,
            initial_population=population,
            horizon_minutes=horizon,
        )

    def evaluate_random_site_advantage(
        self,
        *,
        start: OutcomeSnapshot,
        action_after: OutcomeSnapshot,
        random_plain_time: float,
        action_plain_time: float,
        random_hazard_exposure: float,
        action_hazard_exposure: float,
        initial_population: int,
        horizon_minutes: int | float,
    ) -> RewardBreakdown:
        """Return chosen-site value relative to a uniform feasible-site draw.

        ``random_*`` is the arithmetic mean across the complete feasible set,
        not a sampled control action.  Equal capacity therefore cancels exactly
        and the result has zero mean over feasible sites in every state.
        """
        return self.evaluate_counterfactual(
            start=start,
            action_after=action_after,
            plain_time_wait=random_plain_time,
            plain_time_action=action_plain_time,
            hazard_exposure_wait=random_hazard_exposure,
            hazard_exposure_action=action_hazard_exposure,
            initial_population=initial_population,
            horizon_minutes=horizon_minutes,
            signed=True,
        )

    def evaluate(
        self,
        *,
        before: OutcomeSnapshot,
        after: OutcomeSnapshot,
        remaining_risk_time_before: float,
        remaining_risk_time_after: float,
        initial_population: int,
        horizon: int | float,
    ) -> RewardBreakdown:
        # Compatibility path for offline utilities that already provide one
        # combined risk-time cost rather than its plain/exposure split.
        return self.evaluate_counterfactual(
            start=before,
            action_after=after,
            plain_time_wait=remaining_risk_time_before,
            plain_time_action=remaining_risk_time_after,
            hazard_exposure_wait=0.0,
            hazard_exposure_action=0.0,
            initial_population=initial_population,
            horizon_minutes=horizon,
        )
