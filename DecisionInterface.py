#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Administrator-facing decision contract for dynamic shelter deployment.

At each decision epoch the policy sees the same quantities available on an
emergency-operations dashboard: regional evacuee counts, congestion delay,
route time, current and forecast hazard, available shelter capacity, remaining
time and deployment budget, plus a map-linked table of one action slot per
regional cell. An action names one regional cell to prioritize; the specific
building installed within that cell is resolved by a shared, deterministic
lower layer (``ShelterDatabase._candidate_index``:
maximum remaining capacity, tie-broken by OSM identifier) that is identical
for the RL policy and every heuristic benchmark.  This keeps the upper-level
decision -- which cell to prioritize -- as the only quantity that differs
across policies, so a policy comparison isolates cell-prioritization
behavior rather than building-level tie-breaking.
"""

from dataclasses import dataclass
import hashlib
from typing import Any, Optional, Protocol, Sequence

import numpy as np

from CAProcessor import CellTracker, authoritative_grid_shape


PED_FEATURE_NAMES = (
    "active_population_fraction",
    "unassigned_population_fraction",
    "congestion_delay_fraction",
    "mean_route_time_fraction",
    "slow_tail_population_fraction",
    "slow_tail_time_fraction",
    "long_route_population_fraction",
    "exposed_population_fraction",
    "panicked_population_fraction",
    "remaining_evacuation_debt_fraction",
    "susceptible_population_fraction",
    "threatened_population_fraction",
)

HAZARD_FEATURE_NAMES = (
    "danger",
    "danger_spread_momentum",
    "forecast_expected_danger",
    "forecast_escalation_probability",
    "forecast_uncertainty",
    "forecast_overrun_probability",
    "forecast_next_shelter_overrun_probability",
    "forecast_overrun_lead_time_fraction",
    "hazard_source_proximity",
    "closure_warning_incidence_fraction",
    "front_arrival_probability",
    "front_lead_time_fraction",
)

INFRA_FEATURE_NAMES = (
    "remaining_shelter_capacity_fraction",
    "operational_shelter_capacity_fraction",
    "shelter_utilization_fraction",
    "sheltered_population_fraction",
    "enroute_reserved_population_fraction",
    "forecast_reevacuation_population_fraction",
    "forecast_enroute_reroute_population_fraction",
    "deployable_capacity_fraction",
    "remaining_candidate_share",
    "road_node_share",
    "closed_road_incidence_fraction",
    "region_east_position_fraction",
    "region_north_position_fraction",
    "region_area_fraction",
)

CELL_FEATURE_NAMES = PED_FEATURE_NAMES + HAZARD_FEATURE_NAMES + INFRA_FEATURE_NAMES

PED_FEATURE_SLICE = slice(0, len(PED_FEATURE_NAMES))
HAZARD_FEATURE_SLICE = slice(
    PED_FEATURE_SLICE.stop,
    PED_FEATURE_SLICE.stop + len(HAZARD_FEATURE_NAMES),
)
INFRA_FEATURE_SLICE = slice(HAZARD_FEATURE_SLICE.stop, len(CELL_FEATURE_NAMES))

GLOBAL_FEATURE_NAMES = (
    "time_remaining_fraction",
    "active_population_fraction",
    "unassigned_population_fraction",
    "remaining_deployments_fraction",
    "network_load_share",
    "wind_east_velocity_fraction",
    "wind_north_velocity_fraction",
    "hazard_footprint_fraction",
    "forecast_hazard_growth_fraction",
    "population_network_density_fraction",
    "deployment_capacity_coverage_fraction",
    "forecast_reevacuation_population_fraction",
    "forecast_enroute_reroute_population_fraction",
    "forecast_safe_capacity_coverage_fraction",
    "next_shelter_overrun_probability",
    "next_shelter_overrun_lead_time_fraction",
    "slow_tail_population_fraction",
    "slow_tail_time_fraction",
    "threatened_population_fraction",
)

REGIME_GLOBAL_FEATURE_NAMES = (
    "wind_forecast_cos",
    "wind_forecast_sin",
    "wind_forecast_speed_fraction",
    "wind_warning",
    "closure_warning_share",
    "remaining_tokens_fraction",
    "remaining_epochs_fraction",
    "deadline_slack_fraction",
)


def _feasible_install_actions(observation) -> np.ndarray:
    feasible = np.flatnonzero(observation.action_mask)
    defer_index = int(getattr(observation, "defer_action_index", -1))
    return feasible[feasible != defer_index]

# Causal ordering is explicit: every momentum feature is computed only from
# the current dashboard observation and an earlier cached observation. Positive
# values indicate operational improvement; worsening forecast exposure is
# therefore negative.
MOMENTUM_FEATURE_NAMES = (
    "active_population_clearance_velocity",
    "congestion_delay_reduction_velocity",
    "route_time_reduction_velocity",
    "forecast_exposure_reduction_velocity",
    "forecast_reevacuation_reduction_velocity",
    "forecast_safe_capacity_growth_velocity",
    "threatened_population_reduction_velocity",
)

CANDIDATE_FEATURE_NAMES = (
    "is_defer_action",
    "candidate_capacity_fraction",
    "candidate_east_position_fraction",
    "candidate_north_position_fraction",
    "nearest_open_shelter_distance_fraction",
    "current_danger",
    "forecast_expected_danger",
    "forecast_escalation_probability",
    "forecast_uncertainty",
    "forecast_overrun_probability",
    "forecast_next_overrun_probability",
    "forecast_overrun_lead_time_fraction",
    "forecast_reevacuation_response_fraction",
    "hazard_safety_margin",
    "closure_warning_incidence_fraction",
    "closed_road_incidence_fraction",
    "public_regional_objective_score",
    "threat_capture_fraction",
    "front_approach_fraction",
    "front_arrival_probability",
    "front_lead_time_fraction",
)


def public_regional_objective_values(
    observation,
    *,
    clearance_tail_weight: float,
    casualty_priority_kappa: float,
    active_person_time_weight: float = 1.0,
    hazard_exposure_time_weight: float = 1.0,
    panic_danger_level: float = 3.0,
    variant: str = "legacy",
) -> np.ndarray:
    """Score candidates from only the public regional observation.

    This is the single information contract used by both the actor's residual
    base policy and the deployable objective-greedy comparator.  Scores are
    normalized within the current feasible action set; this preserves the
    ordering while giving the neural prior a stable, resolution-independent
    scale.  No pedestrian record, route, reservation, simulator branch, or
    latent regime value is read here.
    """
    objective_variant = str(variant).strip().lower()
    if objective_variant not in {"legacy", "front_aware"}:
        raise ValueError(
            "Public objective variant must be legacy or front_aware"
        )
    values = np.full(observation.number_of_actions, -np.inf, dtype=np.float64)
    feasible = _feasible_install_actions(observation)
    if feasible.size == 0:
        return values
    cells = np.asarray(observation.candidate_cell_indices[feasible], dtype=np.int64)
    population = float(max(1, int(observation.initial_population)))
    active = np.asarray(observation.active_by_cell[cells], dtype=np.float64)
    capacities = np.asarray(observation.candidate_capacities[feasible], dtype=np.float64)
    coverage = np.minimum(active, capacities) / np.maximum(active, 1.0)
    debt = np.asarray(
        observation.remaining_evacuation_debt_by_cell[cells], dtype=np.float64
    )
    horizon_minutes = max(
        float(observation.time_step_minutes),
        float(observation.horizon) * float(observation.time_step_minutes),
    )
    tail = (
        max(0.0, float(clearance_tail_weight))
        * np.asarray(observation.slow_tail_time_by_cell[cells], dtype=np.float64)
        / (population * horizon_minutes)
    )
    danger = np.maximum(
        np.asarray(observation.danger_by_cell[cells], dtype=np.float64),
        np.asarray(observation.candidate_forecast_danger[feasible], dtype=np.float64),
    )
    time_remaining = max(
        0.0,
        float(observation.horizon - observation.simulation_time)
        / float(max(1, observation.horizon)),
    )
    expected_harm = (
        max(0.0, float(casualty_priority_kappa))
        * (active / population)
        * danger
        * time_remaining
    )
    displaced_access = np.asarray(
        observation.candidate_forecast_reevacuation_response[feasible],
        dtype=np.float64,
    )
    if objective_variant == "legacy":
        scores = coverage * (debt + tail + expected_harm) + displaced_access
    else:
        survival = 1.0 - np.asarray(
            observation.candidate_forecast_overrun_probability[feasible],
            dtype=np.float64,
        )
        effective_capacity = capacities * survival
        coverage = np.minimum(active, effective_capacity) / np.maximum(
            active, 1.0
        )
        capture = np.asarray(
            observation.candidate_threat_capture_fraction[feasible],
            dtype=np.float64,
        )
        d_panic = float(np.clip(panic_danger_level / 5.0, 0.0, 1.0))
        horizon = float(observation.horizon)
        simulation_time = float(observation.simulation_time)
        rescue_value = (
            float(observation.configured_panic_fraction)
            * (
                float(active_person_time_weight)
                + float(hazard_exposure_time_weight) * d_panic
            )
            * (2.0 * horizon - simulation_time)
            / max(1.0, horizon)
        )
        scores = (
            survival * coverage * (debt + tail)
            + rescue_value * capture
            + displaced_access
        )
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise FloatingPointError("Public regional objective scores must be finite and non-negative")
    scale = max(float(np.max(scores)), 1e-12)
    values[feasible] = scores / scale
    return values


def public_objective_kwargs(core) -> dict:
    """Return the shared public-objective configuration for every caller."""
    return dict(
        clearance_tail_weight=float(
            getattr(core, "clearanceTailExtraWeight", 0.0)
        ),
        casualty_priority_kappa=float(
            getattr(core, "casualtyPriorityKappa", 2.0)
        ),
        active_person_time_weight=float(
            getattr(core, "activePersonTimeWeight", 1.0)
        ),
        hazard_exposure_time_weight=float(
            getattr(core, "hazardExposureTimeWeight", 1.0)
        ),
        panic_danger_level=float(getattr(core, "panicDangerThreshold", 3)),
        variant=str(getattr(core, "publicObjectiveVariant", "legacy")),
    )

BENCHMARK_POLICY_CONTRACTS = {
    "rl": {
        "role": "learned_primary_policy",
        "label": "front-aware residual reinforcement-learning policy",
        "information": "the complete public regional observation contract",
        "timing": "dynamic",
    },
    "rl_forecast_ablated": {
        "role": "learned_forecast_ablation",
        "label": "forecast-ablated residual reinforcement-learning policy",
        "information": "current public regional state with predictive inputs ablated",
        "timing": "dynamic",
    },
    "rl_precommit": {
        "role": "learned_timing_ablation",
        "label": "time-zero precommitted reinforcement-learning policy",
        "information": "the initial public regional observation only",
        "timing": "static_precommit",
    },
    "objective_greedy": {
        "role": "primary_myopic_comparator",
        "label": "public-regional objective-matched myopic policy",
        "score": (
            "survival-weighted coverage of actor-visible regional remaining "
            "debt and clearance-tail burden, plus rescue-valued threat "
            "capture and displaced-demand access"
        ),
        "information": (
            "the same public configured-n-by-n regional population, debt, tail, "
            "panic-front forecast, region-to-site road distances, and "
            "candidate-capacity summaries exposed to the actor; "
            "no per-person routes, candidate rerouting branch, recurrent "
            "history, or continuation value"
        ),
        "actor_relation": "identical base score; RL adds only its learned residual",
        "timing": "dynamic",
    },
    "objective_greedy_legacy": {
        "role": "overrun_blind_ablation",
        "label": "legacy public-regional objective-matched myopic policy",
        "score": (
            "capacity-covered share of actor-visible regional remaining debt, "
            "clearance-tail burden, and current/forecast danger burden"
        ),
        "information": (
            "public regional population, debt, tail, danger, forecast and "
            "candidate-capacity summaries; no site-survival or front-race term"
        ),
        "actor_relation": "v63 actor base-score ablation",
        "timing": "dynamic",
    },
    "forecast_greedy": {
        "role": "forecast_myopic_comparator",
        "label": "forecast-ensemble weighted demand greedy",
        "information": (
            "actor-visible current regional demand, candidate capacity, and "
            "public forecast-ensemble mean danger"
        ),
        "timing": "installs at every epoch",
    },
    "regime_oracle_greedy": {
        "role": "information_value_reference",
        "label": "realized-regime one-interval greedy",
        "information": "realized regime tape through the next decision interval",
        "timing": "installs at every epoch; not deployable",
    },
    "deadline_greedy": {
        "role": "strongest_tuned_rule",
        "label": "deadline-feasible threshold regional-objective greedy",
        "information": (
            "public regional objective summaries and budget/deadline slack; "
            "no exact candidate rerouting"
        ),
        "timing": (
            "may defer; threshold is selected on paired validation tapes and "
            "retained only after at least a one-standard-error win over never-defer"
        ),
    },
    "belief_rollout": {
        "role": "public_information_planner",
        "label": "belief-future rollout with greedy continuation",
        "information": (
            "the same public warnings and regional observation, plus futures "
            "sampled from the documented belief model; no realized latent tape"
        ),
        "timing": "top six public-score regions plus defer in Gate 1",
    },
    "hindsight_rollout": {
        "role": "clairvoyant_ceiling",
        "label": "realized-tape rollout ceiling",
        "information": "complete realized exogenous tape",
        "timing": "region plus defer; never reported as a deployable policy",
    },
    "risk_reduction": {
        "role": "appendix_diagnostic",
        "label": "maximum observable local hazard-weighted demand served",
        "score": (
            "min(active_population_at_candidate_i, candidate_capacity_i) * "
            "(1 + max(current_danger_i, forecast_danger_i))"
        ),
        "timing": "dynamic",
    },
    "route_saving": {
        "role": "appendix_diagnostic",
        "label": "legacy alias of observable local hazard-weighted demand served",
        "score": (
            "min(active_population_at_candidate_i, candidate_capacity_i) * "
            "(1 + max(current_danger_i, forecast_danger_i))"
        ),
        "timing": "dynamic",
    },
    "heuristic": {
        "role": "practice_rule",
        "label": "maximum active population",
        "score": "active_population_at_candidate_i",
        "timing": "dynamic",
    },
    "hazard_weighted": {
        "role": "appendix_diagnostic",
        "label": "hazard-weighted active demand",
        "score": "active_population_at_candidate_i * (1 + danger_at_candidate_i)",
        "timing": "dynamic",
    },
    "accessibility_deficit": {
        "role": "appendix_diagnostic",
        "label": "maximum accessibility deficit",
        "score": (
            "max(active_population_i - remaining_capacity_i, 0) * "
            "(1 + nearest_usable_shelter_distance_i / study_area_diagonal)"
        ),
        "distance": "Euclidean distance between projected-metre cell centroids",
        "timing": "dynamic",
    },
    "random": {
        "role": "floor",
        "label": "uniform random feasible regional action",
        "score": "not applicable",
        "timing": "dynamic; includes defer when defer is feasible, exactly as V_mu",
    },
    "initial_only": {
        "role": "value_of_feedback",
        "label": "initial-state objective-greedy precommitment",
        "score": "shared public-regional objective score on a frozen time-zero state",
        "timing": "fixed schedule at the common online decision epochs",
    },
    "static_greedy": {
        "role": "static_demand_reference",
        "label": "time-zero demand precommitment",
        "information": "time-zero public regional active demand and candidate table only",
        "score": "active population in the candidate region",
        "timing": "all sites chosen before the episode; common installation schedule",
    },
}


def _finite_vector(name: str, values, length: int, *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    if array.shape != (int(length),):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


@dataclass(frozen=True)
class OutcomeSnapshot:
    """Population outcomes at one well-defined simulator boundary."""

    safe_completed: int
    casualties: int
    shelter_evacuated: int
    ordinary_arrivals: int
    active_population: int
    risk_mass: float

    def __post_init__(self):
        for name in (
            "safe_completed",
            "casualties",
            "shelter_evacuated",
            "ordinary_arrivals",
            "active_population",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        risk_mass = float(self.risk_mass)
        if not np.isfinite(risk_mass) or risk_mass < 0.0:
            raise ValueError("risk_mass must be finite and non-negative")
        object.__setattr__(self, "risk_mass", risk_mass)

    @property
    def hazard_exposure_mass(self) -> float:
        """Current active-population times normalized-danger mass."""
        return max(0.0, self.risk_mass - float(self.active_population))


@dataclass(frozen=True)
class RemainingTimeEstimate:
    """Capacity-valid time-to-safety split into plain and hazard parts."""

    plain_person_minutes: float
    hazard_exposure_person_minutes: float
    plain_by_cell: Optional[np.ndarray] = None
    hazard_exposure_by_cell: Optional[np.ndarray] = None
    plain_minutes_by_person: Optional[np.ndarray] = None

    def __post_init__(self):
        for name in (
            "plain_person_minutes",
            "hazard_exposure_person_minutes",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        plain_by_cell = (
            np.empty(0, dtype=np.float64)
            if self.plain_by_cell is None
            else np.asarray(self.plain_by_cell, dtype=np.float64).reshape(-1).copy()
        )
        hazard_by_cell = (
            np.empty(0, dtype=np.float64)
            if self.hazard_exposure_by_cell is None
            else np.asarray(
                self.hazard_exposure_by_cell, dtype=np.float64
            ).reshape(-1).copy()
        )
        plain_by_person = (
            np.empty(0, dtype=np.float64)
            if self.plain_minutes_by_person is None
            else np.asarray(
                self.plain_minutes_by_person, dtype=np.float64
            ).reshape(-1).copy()
        )
        if plain_by_cell.shape != hazard_by_cell.shape:
            raise ValueError("Remaining-time cell ledgers must have equal shape")
        if (
            not np.isfinite(plain_by_cell).all()
            or not np.isfinite(hazard_by_cell).all()
            or np.any(plain_by_cell < 0.0)
            or np.any(hazard_by_cell < 0.0)
            or not np.isfinite(plain_by_person).all()
            or np.any(plain_by_person < 0.0)
        ):
            raise ValueError("Remaining-time ledgers must be finite and non-negative")
        if plain_by_cell.size:
            # Realized-casualty debt is retained only for compatibility with
            # archived callers and has no current cell. The target model calls
            # this estimator with ``include_casualties=False``, in which case
            # these inequalities are equalities.
            if (
                float(plain_by_cell.sum()) > self.plain_person_minutes + 1e-6
                or float(hazard_by_cell.sum())
                > self.hazard_exposure_person_minutes + 1e-6
            ):
                raise ValueError("Remaining-time cell ledgers exceed their totals")
        if (
            plain_by_person.size
            and not np.isclose(
                float(plain_by_person.sum()),
                self.plain_person_minutes,
                rtol=1e-9,
                atol=1e-6,
            )
        ):
            raise ValueError("Per-person remaining times do not sum to their total")
        object.__setattr__(self, "plain_by_cell", plain_by_cell)
        object.__setattr__(self, "hazard_exposure_by_cell", hazard_by_cell)
        object.__setattr__(self, "plain_minutes_by_person", plain_by_person)

    @property
    def hazard_weighted_person_minutes(self) -> float:
        return float(
            self.plain_person_minutes + self.hazard_exposure_person_minutes
        )

    @property
    def hazard_weighted_by_cell(self) -> np.ndarray:
        return self.plain_by_cell + self.hazard_exposure_by_cell


def estimate_remaining_time_to_safety(
    core,
    *,
    episode_horizon: int,
    pedestrian_ids: Optional[set] = None,
    include_panicked: bool = True,
    include_casualties: bool = False,
) -> RemainingTimeEstimate:
    """Return capacity-valid plain and hazard-weighted remaining time.

    Time is evaluated from the route and capacity reservation that are true at
    the instant this function is called.  Each route segment is weighted by
    ``1 + max(current danger, public forecast danger)`` in its host cell.  A
    pedestrian without a capacity-reserved, operational, route-matched shelter
    contributes the time remaining in the episode.  Consequently provisional
    routes that merely keep overflow pedestrians moving cannot create reward,
    while the old fixed-horizon reservation cliff does not recur at every
    decision. ``include_casualties`` is retained only for call compatibility;
    realized deaths never enter this forecast because their conditional loss
    is charged through the pre-draw expected-casualty intensity ledger.

    Both values are expressed in pedestrian-minutes. Ordinary route time is
    capped at one episode horizon per pedestrian. In the v41 State College
    curriculum that cap is 180 minutes.
    """
    horizon = max(1, int(episode_horizon))
    current_time = max(
        0,
        int(getattr(getattr(core, "pedDS", None), "currTime", 0)),
    )
    pedestrians = getattr(getattr(core, "pedDS", None), "pedAgentList", {})
    active = [
        pedestrian
        for pedestrian in pedestrians.values()
        if not bool(getattr(pedestrian, "terminated", False))
        and (
            bool(include_panicked)
            or not bool(getattr(pedestrian, "panicked", False))
        )
        and (
            pedestrian_ids is None
            or int(getattr(pedestrian, "agentID", -1)) in pedestrian_ids
        )
    ]
    tracker = getattr(core, "cellTracker", None)
    cell_x, cell_y = authoritative_grid_shape(core)
    cell_count = cell_x * cell_y
    current_danger = np.asarray(
        getattr(tracker, "dangerLevelByCell", np.zeros(cell_count)),
        dtype=np.float64,
    ).reshape(cell_count)
    current_danger = np.clip(
        np.nan_to_num(current_danger, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )
    forecast_danger = current_danger
    forecast = getattr(getattr(core, "hazardDS", None), "forecast_danger_by_cell", None)
    if callable(forecast):
        forecast_danger = np.asarray(
            forecast(max(1, int(getattr(core, "shelterActionInterval", 1)))),
            dtype=np.float64,
        ).reshape(cell_count)
        forecast_danger = np.clip(
            np.nan_to_num(forecast_danger, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )

    time_step_minutes = max(
        1e-9,
        float(getattr(core, "timeStepMinutes", 1.0)),
    )
    horizon_minutes = float(horizon) * time_step_minutes
    remaining_horizon_minutes = (
        float(max(0, horizon - current_time)) * time_step_minutes
    )
    unresolved_horizon_minutes = (
        remaining_horizon_minutes
        if current_time < horizon
        else horizon_minutes
    )
    maximum_speed = max(1e-9, float(getattr(core, "maxSpeed", 1.0)))
    speed_floor = max(0.5, 0.05 * maximum_speed)

    def cell_danger(cell) -> float:
        if cell is None:
            return 0.0
        try:
            i, j = int(cell[0]), int(cell[1])
        except (TypeError, ValueError, IndexError):
            return 0.0
        if not (0 <= i < cell_x and 0 <= j < cell_y):
            return 0.0
        index = i * cell_y + j
        return max(float(current_danger[index]), float(forecast_danger[index]))

    shelter_store = getattr(core, "shelterDS", None)
    reservation_for = getattr(shelter_store, "reservationFor", None)
    del include_casualties
    plain_total = 0.0
    exposure_total = 0.0
    plain_by_cell = np.zeros(cell_count, dtype=np.float64)
    exposure_by_cell = np.zeros(cell_count, dtype=np.float64)
    plain_by_person = []

    def current_cell_index(pedestrian) -> Optional[int]:
        try:
            i, j = getattr(pedestrian, "currCell")
            i, j = int(i), int(j)
        except (TypeError, ValueError, IndexError):
            return None
        if not (0 <= i < cell_x and 0 <= j < cell_y):
            return None
        return i * cell_y + j

    for pedestrian in active:
        group_size = max(1, int(getattr(pedestrian, "group_size", 1)))
        origin_cell = current_cell_index(pedestrian)
        route = getattr(pedestrian, "routeFollowing", None)
        assigned_shelter, assigned_population = (
            reservation_for(pedestrian)
            if callable(reservation_for)
            else (None, 0)
        )
        route_target = getattr(route, "endNode", None) if route is not None else None
        shelter_target = getattr(assigned_shelter, "nodeMapped", None)
        has_available_shelter = bool(
            not bool(getattr(pedestrian, "panicked", False))
            and route is not None
            and assigned_shelter is not None
            and int(assigned_population) >= group_size
            and int(getattr(assigned_shelter, "status", 0)) == 0
            and route_target is not None
            and shelter_target is not None
            and getattr(route_target, "OSMID", None)
            == getattr(shelter_target, "OSMID", None)
        )
        if not has_available_shelter:
            plain_contribution = float(group_size) * unresolved_horizon_minutes
            exposure_contribution = (
                float(group_size)
                * unresolved_horizon_minutes
                * cell_danger(getattr(pedestrian, "currCell", None))
            )
            plain_total += plain_contribution
            exposure_total += exposure_contribution
            plain_by_person.extend([unresolved_horizon_minutes] * group_size)
            if origin_cell is not None:
                plain_by_cell[origin_cell] += plain_contribution
                exposure_by_cell[origin_cell] += exposure_contribution
            continue

        speed = max(
            speed_floor,
            float(getattr(pedestrian, "currSpeed", 0.0)),
        )
        ordinary_minutes = 0.0
        exposure_minutes = 0.0

        current_distance = max(
            0.0,
            float(getattr(pedestrian, "edge_remain", 0.0)),
        )
        if current_distance > 0.0:
            minutes = current_distance / speed
            ordinary_minutes += minutes
            current_edge = getattr(pedestrian, "currEdge", None)
            exposure_minutes += minutes * cell_danger(
                getattr(current_edge, "cellID", None)
                or getattr(pedestrian, "currCell", None)
            )

        for edge in (getattr(route, "edgeRemained", None) or ()):
            distance = max(0.0, float(getattr(edge, "edgeLen", 0.0)))
            minutes = distance / speed
            ordinary_minutes += minutes
            exposure_minutes += minutes * cell_danger(
                getattr(edge, "cellID", None)
            )

        if ordinary_minutes > horizon_minutes and ordinary_minutes > 0.0:
            scale = horizon_minutes / ordinary_minutes
            ordinary_minutes = horizon_minutes
            exposure_minutes *= scale
        plain_contribution = float(group_size) * ordinary_minutes
        exposure_contribution = float(group_size) * exposure_minutes
        plain_total += plain_contribution
        exposure_total += exposure_contribution
        plain_by_person.extend([ordinary_minutes] * group_size)
        if origin_cell is not None:
            plain_by_cell[origin_cell] += plain_contribution
            exposure_by_cell[origin_cell] += exposure_contribution
    return RemainingTimeEstimate(
        plain_total,
        exposure_total,
        plain_by_cell=plain_by_cell,
        hazard_exposure_by_cell=exposure_by_cell,
        plain_minutes_by_person=np.asarray(plain_by_person, dtype=np.float64),
    )


@dataclass(frozen=True)
class ClearanceTailEstimate:
    """Restricted completion-time tail used by every policy objective.

    ``tail_person_minutes`` is the sum, not the mean, of the slowest declared
    population fraction, so it remains in the same person-minute units as the
    complete evacuation ledger.  ``clearance_time_minutes`` is T100 and is
    reported for interpretation; the smoother tail sum is the optimized term.
    """

    tail_person_minutes: float
    tail_mean_minutes: float
    t90_minutes: float
    clearance_time_minutes: float
    tail_population: int


def estimate_clearance_tail(
    core,
    *,
    episode_horizon: int,
    initial_population: int,
    remaining: Optional[RemainingTimeEstimate] = None,
    tail_fraction: Optional[float] = None,
) -> ClearanceTailEstimate:
    """Estimate the complete-episode slow-tail time from the current state.

    Safely completed people retain their realized completion times. Active
    people use current elapsed time plus their capacity-valid remaining route
    time. A casualty is assigned ``2H`` because it never reaches safety; this
    is also the maximum completion time of someone still unresolved at H, so
    death can never make the clearance term look better.
    """

    population = int(initial_population)
    if population <= 0:
        raise ValueError("initial_population must be positive")
    horizon = max(1, int(episode_horizon))
    step_minutes = max(1e-9, float(getattr(core, "timeStepMinutes", 1.0)))
    horizon_minutes = float(horizon) * step_minutes
    fraction = float(
        getattr(core, "clearanceTailFraction", 0.10)
        if tail_fraction is None
        else tail_fraction
    )
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("clearance tail fraction must be in (0, 1]")
    if remaining is None:
        remaining = estimate_remaining_time_to_safety(
            core,
            episode_horizon=horizon,
            include_panicked=True,
            include_casualties=False,
        )

    completion_times = []
    histogram = getattr(
        getattr(core, "pedDS", None), "safeCompletionTimeHistogram", {}
    )
    for completion_time, count in histogram.items():
        completion_times.extend([float(completion_time)] * max(0, int(count)))

    elapsed_minutes = min(
        horizon_minutes,
        max(0.0, float(getattr(core.pedDS, "currTime", 0)) * step_minutes),
    )
    completion_times.extend(
        elapsed_minutes + np.asarray(
            remaining.plain_minutes_by_person, dtype=np.float64
        )
    )
    casualties = max(
        0, int(getattr(core.pedDS, "result", {}).get("casualty", 0))
    )
    completion_times.extend([2.0 * horizon_minutes] * casualties)
    if len(completion_times) != population:
        raise RuntimeError(
            "Clearance ledger population mismatch: "
            f"safe + active + casualty = {len(completion_times)}, "
            f"initialized = {population}"
        )

    ordered = np.sort(np.asarray(completion_times, dtype=np.float64))
    tail_population = max(1, int(np.ceil(fraction * population)))
    tail = ordered[-tail_population:]
    t90_index = min(population - 1, max(0, int(np.ceil(0.90 * population)) - 1))
    return ClearanceTailEstimate(
        tail_person_minutes=float(tail.sum()),
        tail_mean_minutes=float(tail.mean()),
        t90_minutes=float(ordered[t90_index]),
        clearance_time_minutes=float(ordered[-1]),
        tail_population=tail_population,
    )


def estimate_remaining_hazard_weighted_travel_time(
    core,
    *,
    episode_horizon: int,
) -> float:
    """Backward-compatible scalar total of the split time-to-safety estimate."""
    return estimate_remaining_time_to_safety(
        core,
        episode_horizon=episode_horizon,
    ).hazard_weighted_person_minutes


@dataclass(frozen=True)
class RegionalObservation:
    """Operational observation supplied unchanged to every dynamic policy.

    Cell arrays are dashboard summaries.  Candidate arrays define a
    fixed-size, cell-indexed action table: slot ``a`` always refers to
    regional cell ``candidate_cell_indices[a]`` (identically ``a`` under the
    cell-priority action space), and ``candidate_osm_node_ids[a]`` is
    whichever building the shared deterministic lower layer would install
    there right now.  Because that resolution is re-run every decision
    epoch, the specific building behind a slot can change across epochs as
    earlier candidates in that cell are installed; the slot-to-cell mapping
    itself does not.  A cell with no remaining candidate keeps its slot in
    the table as a permanently infeasible placeholder, disabled by
    ``action_mask``.
    """

    decision_index: int
    simulation_time: int
    horizon: int
    initial_population: int
    remaining_deployments: int
    maximum_deployments: int
    maximum_speed: float
    active_by_cell: np.ndarray
    mean_speed_by_cell: np.ndarray
    danger_by_cell: np.ndarray
    remaining_capacity_by_cell: np.ndarray
    deployable_capacity_by_cell: np.ndarray
    candidate_count_by_cell: np.ndarray
    action_mask: np.ndarray
    outcome: OutcomeSnapshot
    operational_shelter_capacity_by_cell: Optional[np.ndarray] = None
    congestion_speed_ratio_by_cell: Optional[np.ndarray] = None
    candidate_osm_node_ids: Optional[tuple[str, ...]] = None
    candidate_cell_indices: Optional[np.ndarray] = None
    candidate_capacities: Optional[np.ndarray] = None
    candidate_east_positions: Optional[np.ndarray] = None
    candidate_north_positions: Optional[np.ndarray] = None
    shelter_utilization_by_cell: Optional[np.ndarray] = None
    sheltered_population_by_cell: Optional[np.ndarray] = None
    enroute_reserved_population_by_cell: Optional[np.ndarray] = None
    forecast_overrun_probability_by_cell: Optional[np.ndarray] = None
    forecast_next_shelter_overrun_probability_by_cell: Optional[np.ndarray] = None
    forecast_overrun_lead_time_fraction_by_cell: Optional[np.ndarray] = None
    forecast_reevacuation_population_by_cell: Optional[np.ndarray] = None
    forecast_enroute_reroute_population_by_cell: Optional[np.ndarray] = None
    network_node_count_by_cell: Optional[np.ndarray] = None
    mean_route_time_by_cell: Optional[np.ndarray] = None
    slow_tail_population_by_cell: Optional[np.ndarray] = None
    slow_tail_time_by_cell: Optional[np.ndarray] = None
    assigned_population_by_cell: Optional[np.ndarray] = None
    unassigned_population_by_cell: Optional[np.ndarray] = None
    outgoing_assignment_flow_by_cell: Optional[np.ndarray] = None
    incoming_assignment_flow_by_cell: Optional[np.ndarray] = None
    long_route_share_by_cell: Optional[np.ndarray] = None
    stable_wellness_by_cell: Optional[np.ndarray] = None
    exposed_wellness_by_cell: Optional[np.ndarray] = None
    panicked_wellness_by_cell: Optional[np.ndarray] = None
    forecast_danger_by_cell: Optional[np.ndarray] = None
    danger_momentum_by_cell: Optional[np.ndarray] = None
    forecast_escalation_probability_by_cell: Optional[np.ndarray] = None
    forecast_uncertainty_by_cell: Optional[np.ndarray] = None
    hazard_source_proximity_by_cell: Optional[np.ndarray] = None
    closure_warning_by_cell: Optional[np.ndarray] = None
    closed_road_by_cell: Optional[np.ndarray] = None
    region_east_positions: Optional[np.ndarray] = None
    region_north_positions: Optional[np.ndarray] = None
    region_area_fractions: Optional[np.ndarray] = None
    spatial_edge_index: Optional[np.ndarray] = None
    route_edge_index: Optional[np.ndarray] = None
    route_edge_weight: Optional[np.ndarray] = None
    candidate_nearest_shelter_distances: Optional[np.ndarray] = None
    candidate_forecast_danger: Optional[np.ndarray] = None
    candidate_hazard_safety_margin: Optional[np.ndarray] = None
    candidate_current_danger: Optional[np.ndarray] = None
    candidate_forecast_escalation: Optional[np.ndarray] = None
    candidate_forecast_uncertainty: Optional[np.ndarray] = None
    candidate_forecast_overrun_probability: Optional[np.ndarray] = None
    candidate_forecast_next_overrun_probability: Optional[np.ndarray] = None
    candidate_forecast_overrun_lead_time_fraction: Optional[np.ndarray] = None
    candidate_forecast_reevacuation_response: Optional[np.ndarray] = None
    candidate_closure_warning: Optional[np.ndarray] = None
    candidate_closed_road: Optional[np.ndarray] = None
    candidate_reroutable_population: Optional[np.ndarray] = None
    candidate_risk_time_reduction: Optional[np.ndarray] = None
    remaining_evacuation_debt_by_cell: Optional[np.ndarray] = None
    candidate_public_objective_score: Optional[np.ndarray] = None
    susceptible_population_by_cell: Optional[np.ndarray] = None
    threatened_population_by_cell: Optional[np.ndarray] = None
    front_arrival_probability_by_cell: Optional[np.ndarray] = None
    front_lead_time_fraction_by_cell: Optional[np.ndarray] = None
    candidate_threat_capture_fraction: Optional[np.ndarray] = None
    candidate_front_approach_fraction: Optional[np.ndarray] = None
    candidate_front_arrival_probability: Optional[np.ndarray] = None
    candidate_front_lead_time_fraction: Optional[np.ndarray] = None
    network_load_share: float = 0.0
    time_step_minutes: float = 1.0
    wind_speed_fraction: float = 0.0
    wind_east_direction_fraction: float = 0.0
    wind_north_direction_fraction: float = 0.0
    hazard_spread_fraction: float = 0.0
    population_network_density_fraction: float = 0.0
    hazard_instance_fraction: float = 0.0
    configured_panic_fraction: float = 0.0
    deployment_capacity_coverage_fraction: float = 0.0
    forecast_reevacuation_population_fraction: float = 0.0
    forecast_enroute_reroute_population_fraction: float = 0.0
    forecast_safe_capacity_coverage_fraction: float = 0.0
    next_shelter_overrun_probability: float = 0.0
    next_shelter_overrun_lead_time_fraction: float = 1.0
    threatened_population_fraction: float = 0.0
    forecast_ensemble_mean: Optional[np.ndarray] = None
    forecast_ensemble_dispersion: Optional[np.ndarray] = None
    forecast_escalation: Optional[np.ndarray] = None
    wind_forecast_cos: float = 0.0
    wind_forecast_sin: float = 0.0
    wind_forecast_speed: float = 0.0
    wind_warning: float = 0.0
    closure_warning_share: float = 0.0
    remaining_tokens: int = 0
    remaining_epochs: int = 0
    deadline_slack: int = 0
    regime_features_enabled: bool = False
    defer_action_index: int = -1

    def __post_init__(self):
        population = int(self.initial_population)
        horizon = int(self.horizon)
        if population <= 0:
            raise ValueError("initial_population must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        object.__setattr__(self, "decision_index", int(self.decision_index))
        object.__setattr__(self, "simulation_time", int(self.simulation_time))
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "initial_population", population)
        object.__setattr__(self, "remaining_deployments", max(0, int(self.remaining_deployments)))
        object.__setattr__(self, "maximum_deployments", max(0, int(self.maximum_deployments)))
        object.__setattr__(self, "maximum_speed", max(float(self.maximum_speed), 1e-6))

        raw_active = np.asarray(self.active_by_cell).reshape(-1)
        cell_count = int(raw_active.size)
        if cell_count <= 0:
            raise ValueError("RegionalObservation requires at least one cell")
        for name in (
            "active_by_cell",
            "mean_speed_by_cell",
            "danger_by_cell",
            "remaining_capacity_by_cell",
            "deployable_capacity_by_cell",
            "candidate_count_by_cell",
        ):
            object.__setattr__(
                self,
                name,
                _finite_vector(name, getattr(self, name), cell_count, nonnegative=True),
            )
        if self.network_node_count_by_cell is None:
            object.__setattr__(
                self,
                "network_node_count_by_cell",
                np.zeros(cell_count, dtype=np.float32),
            )
        else:
            object.__setattr__(
                self,
                "network_node_count_by_cell",
                _finite_vector(
                    "network_node_count_by_cell",
                    self.network_node_count_by_cell,
                    cell_count,
                    nonnegative=True,
                ),
            )
        if self.congestion_speed_ratio_by_cell is None:
            congestion_ratio = np.ones(cell_count, dtype=np.float32)
        else:
            congestion_ratio = _finite_vector(
                "congestion_speed_ratio_by_cell",
                self.congestion_speed_ratio_by_cell,
                cell_count,
                nonnegative=True,
            )
        object.__setattr__(
            self,
            "congestion_speed_ratio_by_cell",
            np.clip(congestion_ratio, 0.0, 1.0).astype(np.float32, copy=False),
        )
        danger = np.clip(self.danger_by_cell, 0.0, 1.0)
        object.__setattr__(self, "danger_by_cell", danger.astype(np.float32, copy=False))
        if self.shelter_utilization_by_cell is None:
            shelter_utilization = np.zeros(cell_count, dtype=np.float32)
        else:
            shelter_utilization = _finite_vector(
                "shelter_utilization_by_cell",
                self.shelter_utilization_by_cell,
                cell_count,
                nonnegative=True,
            )
        object.__setattr__(
            self,
            "shelter_utilization_by_cell",
            np.clip(shelter_utilization, 0.0, 1.0).astype(np.float32, copy=False),
        )

        for name in (
            "operational_shelter_capacity_by_cell",
            "sheltered_population_by_cell",
            "enroute_reserved_population_by_cell",
            "forecast_reevacuation_population_by_cell",
            "forecast_enroute_reroute_population_by_cell",
            "susceptible_population_by_cell",
            "threatened_population_by_cell",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                (
                    np.zeros(cell_count, dtype=np.float32)
                    if value is None
                    else _finite_vector(name, value, cell_count, nonnegative=True)
                ),
            )

        def bounded_cell_vector(name: str, default: float = 0.0) -> np.ndarray:
            value = getattr(self, name)
            if value is None:
                result = np.full(cell_count, default, dtype=np.float32)
            else:
                result = _finite_vector(name, value, cell_count, nonnegative=True)
            return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

        object.__setattr__(
            self,
            "forecast_ensemble_mean",
            bounded_cell_vector("forecast_ensemble_mean"),
        )
        object.__setattr__(
            self,
            "forecast_ensemble_dispersion",
            bounded_cell_vector("forecast_ensemble_dispersion"),
        )
        object.__setattr__(
            self,
            "forecast_escalation",
            bounded_cell_vector("forecast_escalation"),
        )
        object.__setattr__(
            self,
            "front_arrival_probability_by_cell",
            bounded_cell_vector("front_arrival_probability_by_cell"),
        )
        object.__setattr__(
            self,
            "front_lead_time_fraction_by_cell",
            bounded_cell_vector(
                "front_lead_time_fraction_by_cell", default=1.0
            ),
        )

        mean_route_time = (
            np.zeros(cell_count, dtype=np.float32)
            if self.mean_route_time_by_cell is None
            else _finite_vector(
                "mean_route_time_by_cell",
                self.mean_route_time_by_cell,
                cell_count,
                nonnegative=True,
            )
        )
        object.__setattr__(self, "mean_route_time_by_cell", mean_route_time)
        for name in (
            "slow_tail_population_by_cell",
            "slow_tail_time_by_cell",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                (
                    np.zeros(cell_count, dtype=np.float32)
                    if value is None
                    else _finite_vector(name, value, cell_count, nonnegative=True)
                ),
            )
        for name in (
            "assigned_population_by_cell",
            "unassigned_population_by_cell",
            "outgoing_assignment_flow_by_cell",
            "incoming_assignment_flow_by_cell",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                (
                    np.zeros(cell_count, dtype=np.float32)
                    if value is None
                    else _finite_vector(
                        name, value, cell_count, nonnegative=True
                    )
                ),
            )
        for name in (
            "long_route_share_by_cell",
            "stable_wellness_by_cell",
            "exposed_wellness_by_cell",
            "panicked_wellness_by_cell",
            "forecast_danger_by_cell",
            "forecast_escalation_probability_by_cell",
            "forecast_uncertainty_by_cell",
            "forecast_overrun_probability_by_cell",
            "forecast_next_shelter_overrun_probability_by_cell",
            "hazard_source_proximity_by_cell",
            "closure_warning_by_cell",
            "closed_road_by_cell",
        ):
            object.__setattr__(self, name, bounded_cell_vector(name))
        # A lead-time fraction of one means "no threshold crossing inside the
        # forecast window".  Zero means the cell is already overrun.  Using
        # the generic zero default would therefore turn a missing optional
        # forecast into the strongest possible warning.
        object.__setattr__(
            self,
            "forecast_overrun_lead_time_fraction_by_cell",
            bounded_cell_vector(
                "forecast_overrun_lead_time_fraction_by_cell", default=1.0
            ),
        )
        danger_momentum = (
            np.zeros(cell_count, dtype=np.float32)
            if self.danger_momentum_by_cell is None
            else _finite_vector(
                "danger_momentum_by_cell",
                self.danger_momentum_by_cell,
                cell_count,
            )
        )
        object.__setattr__(
            self,
            "danger_momentum_by_cell",
            np.clip(danger_momentum, -1.0, 1.0).astype(np.float32, copy=False),
        )
        remaining_debt = (
            np.zeros(cell_count, dtype=np.float32)
            if self.remaining_evacuation_debt_by_cell is None
            else _finite_vector(
                "remaining_evacuation_debt_by_cell",
                self.remaining_evacuation_debt_by_cell,
                cell_count,
                nonnegative=True,
            )
        )
        object.__setattr__(
            self,
            "remaining_evacuation_debt_by_cell",
            remaining_debt.astype(np.float32, copy=False),
        )

        if self.region_east_positions is None:
            east = np.linspace(0.0, 1.0, cell_count, dtype=np.float32)
        else:
            east = bounded_cell_vector("region_east_positions")
        north = bounded_cell_vector("region_north_positions", default=0.5)
        if self.region_area_fractions is None:
            area = np.full(cell_count, 1.0 / cell_count, dtype=np.float32)
        else:
            area = _finite_vector(
                "region_area_fractions",
                self.region_area_fractions,
                cell_count,
                nonnegative=True,
            )
            area_total = float(area.sum())
            if area_total <= 0.0:
                raise ValueError("region_area_fractions must have positive total area")
            area = area / area_total
        object.__setattr__(self, "region_east_positions", east)
        object.__setattr__(self, "region_north_positions", north)
        object.__setattr__(self, "region_area_fractions", area.astype(np.float32))

        def validated_edges(name: str, values) -> np.ndarray:
            if values is None:
                return np.empty((2, 0), dtype=np.int64)
            edges = np.asarray(values, dtype=np.int64)
            if edges.size == 0:
                return np.empty((2, 0), dtype=np.int64)
            if edges.ndim != 2 or edges.shape[0] != 2:
                raise ValueError(f"{name} must have shape (2, edges)")
            if np.any(edges < 0) or np.any(edges >= cell_count):
                raise ValueError(f"{name} contains an invalid regional node")
            return edges.copy()

        spatial_edges = validated_edges("spatial_edge_index", self.spatial_edge_index)
        route_edges = validated_edges("route_edge_index", self.route_edge_index)
        if self.route_edge_weight is None:
            route_weights = np.ones(route_edges.shape[1], dtype=np.float32)
        else:
            route_weights = _finite_vector(
                "route_edge_weight",
                self.route_edge_weight,
                route_edges.shape[1],
                nonnegative=True,
            )
        object.__setattr__(self, "spatial_edge_index", spatial_edges)
        object.__setattr__(self, "route_edge_index", route_edges)
        object.__setattr__(self, "route_edge_weight", route_weights)

        network_load_share = float(self.network_load_share)
        if not np.isfinite(network_load_share) or not 0.0 <= network_load_share <= 1.0:
            raise ValueError("network_load_share must be finite and in [0, 1]")
        object.__setattr__(self, "network_load_share", network_load_share)
        time_step_minutes = float(self.time_step_minutes)
        if not np.isfinite(time_step_minutes) or time_step_minutes <= 0.0:
            raise ValueError("time_step_minutes must be finite and positive")
        object.__setattr__(self, "time_step_minutes", time_step_minutes)
        for name in (
            "wind_speed_fraction",
            "hazard_spread_fraction",
            "population_network_density_fraction",
            "hazard_instance_fraction",
            "configured_panic_fraction",
            "deployment_capacity_coverage_fraction",
            "forecast_reevacuation_population_fraction",
            "forecast_enroute_reroute_population_fraction",
            "forecast_safe_capacity_coverage_fraction",
            "next_shelter_overrun_probability",
            "next_shelter_overrun_lead_time_fraction",
            "threatened_population_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "wind_forecast_cos",
            "wind_forecast_sin",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "wind_forecast_speed",
            "wind_warning",
            "closure_warning_share",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        remaining_tokens = max(0, int(self.remaining_tokens))
        remaining_epochs = max(0, int(self.remaining_epochs))
        slack = int(self.deadline_slack)
        if slack != remaining_epochs - remaining_tokens:
            raise ValueError("deadline_slack must equal remaining_epochs - remaining_tokens")
        object.__setattr__(self, "remaining_tokens", remaining_tokens)
        object.__setattr__(self, "remaining_epochs", remaining_epochs)
        object.__setattr__(self, "deadline_slack", slack)
        object.__setattr__(self, "regime_features_enabled", bool(self.regime_features_enabled))
        object.__setattr__(self, "defer_action_index", int(self.defer_action_index))
        for name in (
            "wind_east_direction_fraction",
            "wind_north_direction_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
            object.__setattr__(self, name, value)

        mask = np.asarray(self.action_mask, dtype=np.bool_).reshape(-1).copy()
        if self.candidate_cell_indices is None:
            # Compatibility for synthetic/legacy callers: one candidate slot
            # per cell. Authoritative simulator observations always supply the
            # exact candidate table built at episode initialization.
            if mask.shape != (cell_count,):
                raise ValueError(
                    "candidate_cell_indices are required when action_mask is not cell-sized"
                )
            candidate_cells = np.arange(cell_count, dtype=np.int64)
            candidate_capacities = np.where(
                mask & (self.deployable_capacity_by_cell <= 0.0),
                1.0,
                self.deployable_capacity_by_cell,
            ).astype(np.float32)
            candidate_ids = tuple(f"legacy-cell-{index}" for index in range(cell_count))
        else:
            candidate_cells = np.asarray(
                self.candidate_cell_indices,
                dtype=np.int64,
            ).reshape(-1).copy()
            action_count = int(candidate_cells.size)
            if np.any(candidate_cells < 0) or np.any(candidate_cells >= cell_count):
                raise ValueError("candidate_cell_indices contains an invalid regional cell")
            candidate_capacities = _finite_vector(
                "candidate_capacities",
                self.candidate_capacities,
                action_count,
                nonnegative=True,
            )
            if self.candidate_osm_node_ids is None:
                raise ValueError("candidate_osm_node_ids are required with candidate_cell_indices")
            candidate_ids = tuple(str(value) for value in self.candidate_osm_node_ids)
            if len(candidate_ids) != action_count:
                raise ValueError(
                    "candidate_osm_node_ids and candidate_cell_indices must have equal length"
                )
            if len(set(candidate_ids)) != action_count:
                raise ValueError("candidate_osm_node_ids must be unique within an episode")
            if mask.shape != (action_count,):
                raise ValueError(
                    f"action_mask must have shape ({action_count},), got {mask.shape}"
                )
        action_count = int(candidate_cells.size)
        if self.candidate_east_positions is None:
            # Compatibility only. Authoritative observations below use exact
            # projected site coordinates normalized to the study-area bounds.
            denominator = float(max(1, cell_count - 1))
            candidate_east = candidate_cells.astype(np.float32) / denominator
        else:
            candidate_east = _finite_vector(
                "candidate_east_positions",
                self.candidate_east_positions,
                action_count,
            )
        if self.candidate_north_positions is None:
            candidate_north = np.full(action_count, 0.5, dtype=np.float32)
        else:
            candidate_north = _finite_vector(
                "candidate_north_positions",
                self.candidate_north_positions,
                action_count,
            )
        if np.any((candidate_east < 0.0) | (candidate_east > 1.0)):
            raise ValueError("candidate_east_positions must be in [0, 1]")
        if np.any((candidate_north < 0.0) | (candidate_north > 1.0)):
            raise ValueError("candidate_north_positions must be in [0, 1]")

        def bounded_candidate_vector(name: str, default: float = 0.0) -> np.ndarray:
            value = getattr(self, name)
            if value is None:
                result = np.full(action_count, default, dtype=np.float32)
            else:
                result = _finite_vector(name, value, action_count, nonnegative=True)
            return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

        candidate_nearest = bounded_candidate_vector(
            "candidate_nearest_shelter_distances",
            default=1.0,
        )
        candidate_forecast = bounded_candidate_vector("candidate_forecast_danger")
        candidate_safety = bounded_candidate_vector(
            "candidate_hazard_safety_margin",
            default=1.0,
        )
        candidate_current_danger = bounded_candidate_vector(
            "candidate_current_danger"
        )
        candidate_escalation = bounded_candidate_vector(
            "candidate_forecast_escalation"
        )
        candidate_uncertainty = bounded_candidate_vector(
            "candidate_forecast_uncertainty"
        )
        candidate_overrun_probability = bounded_candidate_vector(
            "candidate_forecast_overrun_probability"
        )
        candidate_next_overrun_probability = bounded_candidate_vector(
            "candidate_forecast_next_overrun_probability"
        )
        candidate_overrun_lead_time = bounded_candidate_vector(
            "candidate_forecast_overrun_lead_time_fraction", default=1.0
        )
        candidate_reevacuation_response = bounded_candidate_vector(
            "candidate_forecast_reevacuation_response"
        )
        candidate_closure_warning = bounded_candidate_vector(
            "candidate_closure_warning"
        )
        candidate_closed_road = bounded_candidate_vector(
            "candidate_closed_road"
        )
        candidate_reroutable = bounded_candidate_vector(
            "candidate_reroutable_population"
        )
        candidate_risk_reduction = bounded_candidate_vector(
            "candidate_risk_time_reduction"
        )
        candidate_public_score = _finite_vector(
            "candidate_public_objective_score",
            (
                np.zeros(action_count, dtype=np.float32)
                if self.candidate_public_objective_score is None
                else self.candidate_public_objective_score
            ),
            action_count,
            nonnegative=True,
        )
        if np.any(candidate_public_score > 1.0 + 1e-6):
            raise ValueError(
                "candidate_public_objective_score must lie in [0, 1]"
            )
        candidate_public_score = np.clip(candidate_public_score, 0.0, 1.0).astype(
            np.float32, copy=False
        )
        candidate_threat_capture = bounded_candidate_vector(
            "candidate_threat_capture_fraction"
        )
        candidate_front_approach = bounded_candidate_vector(
            "candidate_front_approach_fraction", default=0.5
        )
        candidate_front_arrival = bounded_candidate_vector(
            "candidate_front_arrival_probability"
        )
        candidate_front_lead = bounded_candidate_vector(
            "candidate_front_lead_time_fraction", default=1.0
        )
        if self.remaining_deployments <= 0 and mask.any():
            raise ValueError("action_mask cannot allow deployment after the budget is exhausted")
        install_feasible = mask.copy()
        if 0 <= int(self.defer_action_index) < action_count:
            install_feasible[int(self.defer_action_index)] = False
        if np.any(install_feasible & (candidate_capacities <= 0.0)):
            raise ValueError("feasible candidates must have positive capacity")
        object.__setattr__(self, "candidate_osm_node_ids", candidate_ids)
        object.__setattr__(self, "candidate_cell_indices", candidate_cells)
        object.__setattr__(self, "candidate_capacities", candidate_capacities)
        object.__setattr__(self, "candidate_east_positions", candidate_east)
        object.__setattr__(self, "candidate_north_positions", candidate_north)
        object.__setattr__(
            self,
            "candidate_nearest_shelter_distances",
            candidate_nearest,
        )
        object.__setattr__(self, "candidate_forecast_danger", candidate_forecast)
        object.__setattr__(self, "candidate_hazard_safety_margin", candidate_safety)
        object.__setattr__(self, "candidate_current_danger", candidate_current_danger)
        object.__setattr__(self, "candidate_forecast_escalation", candidate_escalation)
        object.__setattr__(self, "candidate_forecast_uncertainty", candidate_uncertainty)
        object.__setattr__(
            self,
            "candidate_forecast_overrun_probability",
            candidate_overrun_probability,
        )
        object.__setattr__(
            self,
            "candidate_forecast_next_overrun_probability",
            candidate_next_overrun_probability,
        )
        object.__setattr__(
            self,
            "candidate_forecast_overrun_lead_time_fraction",
            candidate_overrun_lead_time,
        )
        object.__setattr__(
            self,
            "candidate_forecast_reevacuation_response",
            candidate_reevacuation_response,
        )
        object.__setattr__(self, "candidate_closure_warning", candidate_closure_warning)
        object.__setattr__(self, "candidate_closed_road", candidate_closed_road)
        object.__setattr__(
            self,
            "candidate_reroutable_population",
            candidate_reroutable,
        )
        object.__setattr__(
            self,
            "candidate_risk_time_reduction",
            candidate_risk_reduction,
        )
        object.__setattr__(
            self,
            "candidate_public_objective_score",
            candidate_public_score,
        )
        object.__setattr__(
            self,
            "candidate_threat_capture_fraction",
            candidate_threat_capture,
        )
        object.__setattr__(
            self,
            "candidate_front_approach_fraction",
            candidate_front_approach,
        )
        object.__setattr__(
            self,
            "candidate_front_arrival_probability",
            candidate_front_arrival,
        )
        object.__setattr__(
            self,
            "candidate_front_lead_time_fraction",
            candidate_front_lead,
        )
        object.__setattr__(self, "action_mask", mask)

        if int(round(float(self.active_by_cell.sum()))) != self.outcome.active_population:
            raise ValueError("active_by_cell does not sum to outcome.active_population")
        classified = self.outcome.safe_completed + self.outcome.casualties
        if classified + self.outcome.active_population > population:
            raise ValueError("population outcomes exceed initial_population")

    @property
    def number_of_cells(self) -> int:
        return int(self.active_by_cell.size)

    @property
    def number_of_actions(self) -> int:
        return int(self.action_mask.size)

    @property
    def has_feasible_action(self) -> bool:
        return bool(self.action_mask.any())

    def policy_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed-width features for any runtime number of regions.

        The feature width and units are invariant to the selected cell
        resolution. Counts are scenario shares, route times are fractions of
        the fixed episode duration, and geometry is normalized to the study
        extent. This is the contract consumed by the shared GNN encoders.
        """
        population_scale = float(max(1, self.initial_population))
        route_time_scale = max(
            self.time_step_minutes,
            float(self.horizon) * self.time_step_minutes,
        )
        road_node_total = float(max(1.0, self.network_node_count_by_cell.sum()))
        pedestrian_features = np.stack(
            (
                np.clip(self.active_by_cell / population_scale, 0.0, 1.0),
                np.clip(
                    self.unassigned_population_by_cell / population_scale,
                    0.0,
                    1.0,
                ),
                np.where(
                    self.active_by_cell > 0.0,
                    1.0 - self.congestion_speed_ratio_by_cell,
                    0.0,
                ),
                np.clip(self.mean_route_time_by_cell / route_time_scale, 0.0, 1.0),
                np.clip(
                    self.slow_tail_population_by_cell / population_scale,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.slow_tail_time_by_cell
                    / (population_scale * route_time_scale),
                    0.0,
                    1.0,
                ),
                self.long_route_share_by_cell,
                self.exposed_wellness_by_cell,
                self.panicked_wellness_by_cell,
                np.clip(
                    self.remaining_evacuation_debt_by_cell,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.susceptible_population_by_cell / population_scale,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.threatened_population_by_cell / population_scale,
                    0.0,
                    1.0,
                ),
            ),
            axis=-1,
        )
        hazard_features = np.stack(
            (
                self.danger_by_cell,
                self.danger_momentum_by_cell,
                self.forecast_danger_by_cell,
                self.forecast_escalation_probability_by_cell,
                self.forecast_uncertainty_by_cell,
                self.forecast_overrun_probability_by_cell,
                self.forecast_next_shelter_overrun_probability_by_cell,
                self.forecast_overrun_lead_time_fraction_by_cell,
                self.hazard_source_proximity_by_cell,
                self.closure_warning_by_cell,
                self.front_arrival_probability_by_cell,
                self.front_lead_time_fraction_by_cell,
            ),
            axis=-1,
        )
        infrastructure_features = np.stack(
            (
                np.clip(self.remaining_capacity_by_cell / population_scale, 0.0, 1.0),
                np.clip(
                    self.operational_shelter_capacity_by_cell / population_scale,
                    0.0,
                    1.0,
                ),
                self.shelter_utilization_by_cell,
                np.clip(
                    self.sheltered_population_by_cell / population_scale,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.enroute_reserved_population_by_cell / population_scale,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.forecast_reevacuation_population_by_cell
                    / population_scale,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.forecast_enroute_reroute_population_by_cell
                    / population_scale,
                    0.0,
                    1.0,
                ),
                np.clip(self.deployable_capacity_by_cell / population_scale, 0.0, 1.0),
                np.clip(
                    self.candidate_count_by_cell
                    / max(1.0, float(self.candidate_count_by_cell.sum())),
                    0.0,
                    1.0,
                ),
                np.clip(self.network_node_count_by_cell / road_node_total, 0.0, 1.0),
                self.closed_road_by_cell,
                self.region_east_positions,
                self.region_north_positions,
                self.region_area_fractions,
            ),
            axis=-1,
        )
        cell_features = np.concatenate(
            (pedestrian_features, hazard_features, infrastructure_features),
            axis=-1,
        ).astype(np.float32)
        time_remaining = max(0.0, float(self.horizon - self.simulation_time)) / float(self.horizon)
        deployment_scale = float(max(1, self.maximum_deployments))
        area = np.asarray(self.region_area_fractions, dtype=np.float64)
        hazard_footprint = float(
            np.clip(np.sum(area * self.danger_by_cell), 0.0, 1.0)
        )
        forecast_growth = float(
            np.clip(
                np.sum(
                    area
                    * (
                        self.forecast_danger_by_cell
                        - self.danger_by_cell
                    )
                ),
                -1.0,
                1.0,
            )
        )
        global_features = np.asarray(
            (
                time_remaining,
                self.outcome.active_population / population_scale,
                np.clip(
                    self.unassigned_population_by_cell.sum() / population_scale,
                    0.0,
                    1.0,
                ),
                self.remaining_deployments / deployment_scale,
                self.network_load_share,
                self.wind_east_direction_fraction,
                self.wind_north_direction_fraction,
                hazard_footprint,
                forecast_growth,
                self.population_network_density_fraction,
                self.deployment_capacity_coverage_fraction,
                self.forecast_reevacuation_population_fraction,
                self.forecast_enroute_reroute_population_fraction,
                self.forecast_safe_capacity_coverage_fraction,
                self.next_shelter_overrun_probability,
                self.next_shelter_overrun_lead_time_fraction,
                np.clip(
                    self.slow_tail_population_by_cell.sum() / population_scale,
                    0.0,
                    1.0,
                ),
                np.clip(
                    self.slow_tail_time_by_cell.sum()
                    / (population_scale * route_time_scale),
                    0.0,
                    1.0,
                ),
                self.threatened_population_fraction,
            ),
            dtype=np.float32,
        )
        if self.regime_features_enabled:
            epoch_scale = float(max(1, self.maximum_deployments, self.remaining_epochs))
            global_features = np.concatenate(
                (
                    global_features,
                    np.asarray(
                        (
                            self.wind_forecast_cos,
                            self.wind_forecast_sin,
                            self.wind_forecast_speed,
                            self.wind_warning,
                            self.closure_warning_share,
                            self.remaining_tokens / epoch_scale,
                            self.remaining_epochs / epoch_scale,
                            np.clip(self.deadline_slack / epoch_scale, -1.0, 1.0),
                        ),
                        dtype=np.float32,
                    ),
                )
            )
        if not np.isfinite(cell_features).all() or not np.isfinite(global_features).all():
            raise ValueError("Policy features contain non-finite values")
        return cell_features, global_features

    def candidate_features(self) -> np.ndarray:
        """Return one plainly interpretable feature vector per candidate site."""
        population_scale = float(max(1, self.initial_population))
        defer = np.zeros(self.number_of_actions, dtype=np.float32)
        if 0 <= int(self.defer_action_index) < self.number_of_actions:
            defer[int(self.defer_action_index)] = 1.0
        features = np.stack(
            (
                defer,
                np.clip(self.candidate_capacities / population_scale, 0.0, 1.0),
                self.candidate_east_positions,
                self.candidate_north_positions,
                self.candidate_nearest_shelter_distances,
                self.candidate_current_danger,
                self.candidate_forecast_danger,
                self.candidate_forecast_escalation,
                self.candidate_forecast_uncertainty,
                self.candidate_forecast_overrun_probability,
                self.candidate_forecast_next_overrun_probability,
                self.candidate_forecast_overrun_lead_time_fraction,
                self.candidate_forecast_reevacuation_response,
                self.candidate_hazard_safety_margin,
                self.candidate_closure_warning,
                self.candidate_closed_road,
                self.candidate_public_objective_score,
                self.candidate_threat_capture_fraction,
                self.candidate_front_approach_fraction,
                self.candidate_front_arrival_probability,
                self.candidate_front_lead_time_fraction,
            ),
            axis=-1,
        ).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Candidate features contain non-finite values")
        return features


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    strategy: str
    log_probability: Optional[Any] = None
    value: Optional[Any] = None
    value_components: Optional[Any] = None


@dataclass(frozen=True)
class RegionalActionReceipt:
    observation_id: tuple[int, int]
    requested_candidate: int
    executed_candidate: int
    requested_cell: int
    executed_cell: int
    shelter_id: int
    candidate_osm_node_id: str
    candidate_x_m: float
    candidate_y_m: float
    candidate_cell_i: int
    candidate_cell_j: int
    capacity_added: float
    rerouted_population: int
    installed: bool = True


class RegionalPolicy(Protocol):
    def select(self, observation: RegionalObservation, *, deterministic: bool) -> PolicyDecision:
        ...


def _defer_if_nothing_installable(
    observation: RegionalObservation,
    strategy_name: str,
):
    defer = int(getattr(observation, "defer_action_index", -1))
    if (
        _feasible_install_actions(observation).size == 0
        and defer >= 0
        and bool(observation.action_mask[defer])
    ):
        return PolicyDecision(action_index=defer, strategy=strategy_name)
    return None


class ActivePopulationHeuristic:
    """Benchmark: choose a site in the region with most active evacuees."""

    name = "heuristic"

    def select(self, observation: RegionalObservation, *, deterministic: bool = True) -> PolicyDecision:
        del deterministic
        fallback = _defer_if_nothing_installable(observation, self.name)
        if fallback is not None:
            return fallback
        feasible = _feasible_install_actions(observation)
        if feasible.size == 0:
            raise RuntimeError("The heuristic received an observation with no feasible action")
        cells = observation.candidate_cell_indices[feasible]
        active = observation.active_by_cell[cells]
        # The stable candidate-table order supplies the deterministic tie break.
        action = int(feasible[int(np.argmax(active))])
        return PolicyDecision(action_index=action, strategy=self.name)


class RiskTimeReductionHeuristic:
    """Coarse benchmark using only the current regional dashboard.

    This deliberately is not an oracle ranking of candidate interventions. It
    caps the active population in the candidate's own cell by the site's
    capacity and weights that local demand by observed/forecast local danger.
    It never scans pedestrian routes or simulates candidate consequences.
    """

    name = "risk_reduction"

    def select(
        self,
        observation: RegionalObservation,
        *,
        deterministic: bool = True,
    ) -> PolicyDecision:
        del deterministic
        fallback = _defer_if_nothing_installable(observation, self.name)
        if fallback is not None:
            return fallback
        feasible = _feasible_install_actions(observation)
        if feasible.size == 0:
            raise RuntimeError(
                "The risk-reduction heuristic received an observation with no feasible action"
            )
        cells = observation.candidate_cell_indices[feasible]
        local_demand = np.minimum(
            observation.active_by_cell[cells],
            observation.candidate_capacities[feasible],
        )
        local_danger = np.maximum(
            observation.danger_by_cell[cells],
            observation.candidate_forecast_danger[feasible],
        )
        score = local_demand * (1.0 + local_danger)
        action = int(feasible[int(np.argmax(score))])
        return PolicyDecision(action_index=action, strategy=self.name)


class ObjectiveMatchedGreedyHeuristic:
    """Choose the largest public-regional immediate objective proxy.

    The bridge supplies one value per feasible action computed only from the
    actor-visible regional dashboard.  The selector deliberately has no
    access to per-person routes, exact candidate rerouting, temporal memory or
    a continuation value.  This makes it an observation-matched heuristic,
    while the learned actor may use history and simulator-graded training.
    """

    name = "objective_greedy"

    def select(
        self,
        observation: RegionalObservation,
        *,
        candidate_values,
        deterministic: bool = True,
    ) -> PolicyDecision:
        del deterministic
        fallback = _defer_if_nothing_installable(observation, self.name)
        if fallback is not None:
            return fallback
        feasible = _feasible_install_actions(observation)
        if feasible.size == 0:
            raise RuntimeError(
                "The objective-greedy heuristic received an observation with no feasible action"
            )
        values = np.asarray(candidate_values, dtype=np.float64)
        expected_shape = (observation.number_of_actions,)
        if values.shape != expected_shape:
            raise ValueError(
                "candidate_values must have shape "
                f"{expected_shape}, got {values.shape}"
            )
        if not np.isfinite(values[feasible]).all():
            raise ValueError("Feasible objective-greedy candidate values must be finite")
        # np.argmax preserves stable action-table order for exact ties.
        action = int(feasible[int(np.argmax(values[feasible]))])
        return PolicyDecision(action_index=action, strategy=self.name)


class ForecastGreedyHeuristic:
    """Greedy public-forecast rule; timing is deliberately fixed."""

    name = "forecast_greedy"

    def scores(self, observation: RegionalObservation) -> np.ndarray:
        scores = np.full(observation.number_of_actions, -np.inf, dtype=np.float64)
        feasible = _feasible_install_actions(observation)
        if feasible.size:
            cells = observation.candidate_cell_indices[feasible]
            demand = np.minimum(
                observation.active_by_cell[cells],
                observation.candidate_capacities[feasible],
            )
            scores[feasible] = demand * (
                1.0 + observation.forecast_ensemble_mean[cells]
            )
        return scores

    def select(
        self,
        observation,
        *,
        candidate_values=None,
        deterministic=True,
    ):
        del deterministic
        fallback = _defer_if_nothing_installable(observation, self.name)
        if fallback is not None:
            return fallback
        scores = (
            self.scores(observation)
            if candidate_values is None
            else np.asarray(candidate_values, dtype=np.float64)
        )
        if scores.shape != (observation.number_of_actions,):
            raise ValueError("forecast candidate_values must match the action table")
        feasible = _feasible_install_actions(observation)
        if feasible.size == 0:
            raise RuntimeError("Forecast-greedy has no feasible installation")
        action = int(feasible[int(np.argmax(scores[feasible]))])
        return PolicyDecision(action_index=action, strategy=self.name)


class DeadlineGreedyHeuristic(ForecastGreedyHeuristic):
    """Forecast greedy with one tuned, deadline-feasible defer threshold."""

    name = "deadline_greedy"

    def __init__(self, threshold: float = -np.inf):
        self.threshold = float(threshold)

    def select(
        self,
        observation,
        *,
        candidate_values=None,
        deterministic=True,
    ):
        del deterministic
        scores = (
            self.scores(observation)
            if candidate_values is None
            else np.asarray(candidate_values, dtype=np.float64)
        )
        if scores.shape != (observation.number_of_actions,):
            raise ValueError("deadline candidate_values must match the action table")
        feasible = _feasible_install_actions(observation)
        defer = int(observation.defer_action_index)
        if (
            defer >= 0
            and bool(observation.action_mask[defer])
            and (
                feasible.size == 0
                or float(np.max(scores[feasible])) < self.threshold
            )
        ):
            return PolicyDecision(action_index=defer, strategy=self.name)
        if feasible.size == 0:
            raise RuntimeError("Deadline-greedy has no feasible action")
        action = int(feasible[int(np.argmax(scores[feasible]))])
        return PolicyDecision(action_index=action, strategy=self.name)


class BeliefRolloutPolicy:
    """Select the minimum-cost action from externally computed belief rollouts."""

    name = "belief_rollout"

    def select(self, observation, *, candidate_costs, deterministic=True):
        del deterministic
        costs = np.asarray(candidate_costs, dtype=np.float64)
        if costs.shape != (observation.number_of_actions,):
            raise ValueError("candidate_costs must match the action table")
        feasible = np.flatnonzero(observation.action_mask)
        feasible = feasible[np.isfinite(costs[feasible])]
        if feasible.size == 0:
            raise RuntimeError("Belief rollout requires at least one graded feasible action")
        action = int(feasible[int(np.argmin(costs[feasible]))])
        return PolicyDecision(action_index=action, strategy=self.name)


class HindsightRolloutPolicy(BeliefRolloutPolicy):
    """Clairvoyant ceiling; evaluation must never report it as a policy."""

    name = "hindsight_rollout"


class RouteTimeSavingHeuristic(RiskTimeReductionHeuristic):
    """Compatibility alias for pre-v27 experiment definitions."""

    name = "route_saving"


class HazardWeightedDemandHeuristic:
    """Choose a candidate in the region with greatest current HWTTS mass.

    The score ``N_i * (1 + D_i)`` is the instantaneous cell contribution to
    hazard-weighted time-to-safety. Stable candidate-table order resolves ties.
    """

    name = "hazard_weighted"

    def select(
        self,
        observation: RegionalObservation,
        *,
        deterministic: bool = True,
    ) -> PolicyDecision:
        del deterministic
        fallback = _defer_if_nothing_installable(observation, self.name)
        if fallback is not None:
            return fallback
        feasible = _feasible_install_actions(observation)
        if feasible.size == 0:
            raise RuntimeError(
                "The hazard-weighted heuristic received an observation with no feasible action"
            )
        cells = observation.candidate_cell_indices[feasible]
        score = observation.active_by_cell[cells] * (
            1.0 + observation.danger_by_cell[cells]
        )
        action = int(feasible[int(np.argmax(score))])
        return PolicyDecision(action_index=action, strategy=self.name)


class AccessibilityDeficitHeuristic:
    """Prioritize capacity-short cells that are remote from usable shelters.

    For cell ``i``, the deterministic score is

    ``max(N_i - Q_i, 0) * (1 + d(i, S+) / D)``,

    where ``N_i`` is active demand, ``Q_i`` is remaining capacity already in
    the cell, ``S+`` is the set of cells with positive remaining capacity, and
    ``D`` is the study-area diagonal.  Cell centroids are expressed in the
    simulator's projected metre coordinate system.  If no usable shelter
    remains, normalized distance is one for every cell.  The score is a
    transparent capacitated p-median-style accessibility deficit, not a claim
    to solve the full dynamic location-allocation problem.
    """

    name = "accessibility_deficit"

    def __init__(self, cell_centers):
        centers = np.asarray(cell_centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] <= 0:
            raise ValueError("cell_centers must have shape (number_of_cells, 2)")
        if not np.isfinite(centers).all():
            raise ValueError("cell_centers must be finite")
        self.cell_centers = centers.copy()
        spans = np.ptp(self.cell_centers, axis=0)
        self.study_area_diagonal = max(float(np.hypot(spans[0], spans[1])), 1e-12)

    def select(
        self,
        observation: RegionalObservation,
        *,
        deterministic: bool = True,
    ) -> PolicyDecision:
        del deterministic
        fallback = _defer_if_nothing_installable(observation, self.name)
        if fallback is not None:
            return fallback
        if observation.number_of_cells != int(self.cell_centers.shape[0]):
            raise ValueError(
                "cell_centers and RegionalObservation must contain the same number of cells"
            )
        feasible = _feasible_install_actions(observation)
        if feasible.size == 0:
            raise RuntimeError(
                "The accessibility-deficit heuristic received an observation with no feasible action"
            )

        usable_shelter_cells = np.flatnonzero(
            observation.remaining_capacity_by_cell > 0.0
        )
        if usable_shelter_cells.size:
            displacement = (
                self.cell_centers[:, None, :]
                - self.cell_centers[usable_shelter_cells][None, :, :]
            )
            nearest_distance = np.sqrt(np.sum(displacement * displacement, axis=2)).min(
                axis=1
            )
            normalized_distance = np.clip(
                nearest_distance / self.study_area_diagonal,
                0.0,
                1.0,
            )
        else:
            normalized_distance = np.ones(
                observation.number_of_cells,
                dtype=np.float64,
            )

        unmet_demand = np.maximum(
            observation.active_by_cell - observation.remaining_capacity_by_cell,
            0.0,
        )
        cells = observation.candidate_cell_indices[feasible]
        score = unmet_demand[cells] * (1.0 + normalized_distance[cells])
        action = int(feasible[int(np.argmax(score))])
        return PolicyDecision(action_index=action, strategy=self.name)


class UniformRegionalPolicy:
    """Random benchmark with a policy-specific RNG, independent of scenario RNG."""

    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    def select(self, observation: RegionalObservation, *, deterministic: bool = False) -> PolicyDecision:
        del deterministic
        action = uniform_feasible_action(observation, self.rng)
        return PolicyDecision(action_index=action, strategy=self.name)


def uniform_feasible_action(observation: RegionalObservation, rng) -> int:
    """Shared uniform-reference action rule used online and in V_mu rollouts."""
    feasible = np.flatnonzero(observation.action_mask)
    if feasible.size == 0:
        raise RuntimeError("The uniform policy received an observation with no feasible action")
    return int(rng.choice(feasible))


class RegionalObservationBuilder:
    """Create one authoritative operational observation from domain stores."""

    def __init__(self, core, *, initial_population: int, horizon: int, maximum_deployments: int):
        self.core = core
        self.initial_population = int(initial_population)
        self.horizon = int(horizon)
        self.maximum_deployments = max(0, int(maximum_deployments))
        self.nx, self.ny = authoritative_grid_shape(core)
        self.number_of_cells = self.nx * self.ny
        self.action_space = str(getattr(core, "actionSpace", "cell")).strip().lower()
        self.defer_enabled = self.action_space == "region_defer"
        (
            self.region_east_positions,
            self.region_north_positions,
            self.region_area_fractions,
            self.study_area_diagonal,
        ) = self._region_geometry()
        self.network_node_count_by_cell = self._network_node_counts()
        self.spatial_edge_index = self._regional_spatial_edges()
        self.network_load_share = self._network_load_share()
        road_nodes = float(max(1.0, self.network_node_count_by_cell.sum()))
        # Bounded, dimensionless scenario descriptors.  They condition one
        # shared policy across cities and experiment scales without creating
        # a trainable city id or a fixed maximum pedestrian/hazard count.
        self.population_network_density_fraction = float(
            self.initial_population / (self.initial_population + road_nodes)
        )
        hazard_instances = float(max(0, int(getattr(core, "hazardVol", 0))))
        self.hazard_instance_fraction = float(
            hazard_instances / (hazard_instances + 1.0)
        )
        self.configured_panic_fraction = float(
            np.clip(float(getattr(core, "panicRate", 0.0)), 0.0, 1.0)
        )
        capacity_token = max(0.0, float(getattr(core, "shelterCapacityToken", 0)))
        self.deployment_capacity_coverage_fraction = float(np.clip(
            self.maximum_deployments * capacity_token
            / float(max(1, self.initial_population)),
            0.0,
            1.0,
        ))
        self._candidate_nodes = ()
        self._candidate_records = ()
        self._raw_candidate_total = 0
        self._position_bounds = self._initial_position_bounds()
        self._site_region_distance_cache: dict[
            tuple[str, int], np.ndarray
        ] = {}
        self._region_road_nodes = self._road_nodes_by_region()
        if self._raw_candidate_total == 0 and self.maximum_deployments > 0:
            raise ValueError("A positive deployment budget requires at least one shelter candidate")

    def _region_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute normalized physical centroids and areas for runtime cells."""
        tracker = getattr(self.core, "cellTracker", None)
        x_edges = np.asarray(getattr(tracker, "xEdges", ()), dtype=np.float64)
        y_edges = np.asarray(getattr(tracker, "yEdges", ()), dtype=np.float64)
        if (
            x_edges.shape != (self.nx + 1,)
            or y_edges.shape != (self.ny + 1,)
            or not np.isfinite(x_edges).all()
            or not np.isfinite(y_edges).all()
            or np.any(np.diff(x_edges) <= 0.0)
            or np.any(np.diff(y_edges) <= 0.0)
        ):
            raise RuntimeError(
                "CellTracker boundaries do not match the authoritative grid "
                f"shape {(self.nx, self.ny)}"
            )
        x_span = max(float(x_edges[-1] - x_edges[0]), 1e-12)
        y_span = max(float(y_edges[-1] - y_edges[0]), 1e-12)
        east = []
        north = []
        areas = []
        for i in range(self.nx):
            for j in range(self.ny):
                east.append(
                    (0.5 * (x_edges[i] + x_edges[i + 1]) - x_edges[0]) / x_span
                )
                # Local-map y increases southward. Expose a conventional
                # physical north coordinate to the policy by inverting y.
                north.append(
                    (y_edges[-1] - 0.5 * (y_edges[j] + y_edges[j + 1]))
                    / y_span
                )
                areas.append(
                    (x_edges[i + 1] - x_edges[i])
                    * (y_edges[j + 1] - y_edges[j])
                )
        areas = np.asarray(areas, dtype=np.float64)
        areas /= max(float(areas.sum()), 1e-12)
        return (
            np.asarray(east, dtype=np.float32),
            np.asarray(north, dtype=np.float32),
            areas.astype(np.float32),
            max(float(np.hypot(x_span, y_span)), 1e-12),
        )

    def _regional_spatial_edges(self) -> np.ndarray:
        """Build one stable regional graph from adjacency and road crossings.

        The same Moore-neighbor edges as the hazard CA carry spatial context
        even where a cell is road-empty. Road links are then added, including
        any
        nonlocal connections introduced by an irregular network topology.
        """
        pairs = set()
        for i in range(self.nx):
            for j in range(self.ny):
                source = i * self.ny + j
                for di, dj in CellTracker.MOORE_NEIGHBOR_OFFSETS:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        pairs.add((source, ni * self.ny + nj))

        edges = getattr(getattr(self.core, "mapDS", None), "edgeListByLocalID", {})
        for edge in edges.values():
            start_cell = getattr(getattr(edge, "startNode", None), "cellID", None)
            end_cell = getattr(getattr(edge, "endNode", None), "cellID", None)
            if start_cell is None or end_cell is None:
                continue
            si, sj = int(start_cell[0]), int(start_cell[1])
            ti, tj = int(end_cell[0]), int(end_cell[1])
            if not (
                0 <= si < self.nx
                and 0 <= sj < self.ny
                and 0 <= ti < self.nx
                and 0 <= tj < self.ny
            ):
                continue
            source = si * self.ny + sj
            target = ti * self.ny + tj
            if source != target:
                pairs.add((source, target))
                pairs.add((target, source))
        if not pairs:
            return np.empty((2, 0), dtype=np.int64)
        return np.asarray(sorted(pairs), dtype=np.int64).T

    def _initial_position_bounds(self) -> tuple[float, float, float, float]:
        """Compute stable position-normalization bounds, once, at construction.

        Bounds are derived from the full raw candidate universe so that
        ``candidate_east_positions``/``candidate_north_positions`` stay on a
        consistent scale across decision epochs even though the specific
        building resolved for a cell can change once an earlier candidate in
        that cell is installed. This method also validates OSM-identifier
        uniqueness across the whole map up front (the per-cell resolution in
        ``_cell_action_records`` relies on that invariant) and records the
        total candidate count so ``__init__`` can fail closed on an empty map.
        """
        candidate_grid = getattr(self.core.shelterDS, "shelterCanByCell", None)
        if candidate_grid is None:
            self._raw_candidate_total = 0
            return 0.0, 1.0, 0.0, 1.0
        raw_x = []
        raw_y = []
        seen = set()
        for i in range(self.nx):
            for j in range(self.ny):
                for node in candidate_grid[i][j]:
                    osm_id = str(getattr(node, "OSMID", ""))
                    if not osm_id or osm_id in seen:
                        raise ValueError("Shelter candidates require unique non-empty OSM identifiers")
                    seen.add(osm_id)
                    x_coord = float(getattr(node, "nodeX", 0.0))
                    y_coord = float(getattr(node, "nodeY", 0.0))
                    if not np.isfinite(x_coord) or not np.isfinite(y_coord):
                        raise ValueError("Shelter candidate coordinates must be finite")
                    raw_x.append(x_coord)
                    raw_y.append(y_coord)
        self._raw_candidate_total = len(seen)
        tracker = getattr(self.core, "cellTracker", None)
        x_edges = np.asarray(getattr(tracker, "xEdges", ()), dtype=float)
        y_edges = np.asarray(getattr(tracker, "yEdges", ()), dtype=float)
        raw_x = np.asarray(raw_x, dtype=float)
        raw_y = np.asarray(raw_y, dtype=float)

        def bounds(edges: np.ndarray, values: np.ndarray) -> tuple[float, float]:
            if edges.size >= 2 and np.isfinite(edges).all() and edges[-1] > edges[0]:
                return float(edges[0]), float(edges[-1])
            if values.size and np.isfinite(values).all() and values.max() > values.min():
                return float(values.min()), float(values.max())
            return 0.0, 1.0

        x_min, x_max = bounds(x_edges, raw_x)
        y_min, y_max = bounds(y_edges, raw_y)
        return x_min, x_max, y_min, y_max

    def _cell_action_records(
        self,
        preview_nodes: Sequence[Optional[Any]],
    ) -> tuple[tuple[tuple[str, int, float, float, float], ...], tuple[Optional[Any], ...]]:
        """Resolve exactly one action slot per regional cell.

        Each slot is the site the shared deterministic lower layer
        (``ShelterDatabase._candidate_index``: maximum remaining capacity,
        tied-broken by OSM identifier) would install in that cell right now.
        This is the identical rule used to install whichever cell any
        heuristic benchmark or the RL policy chooses, so the action space is
        cell-indexed and every policy differs only in cell prioritization,
        never in building-level tie-breaking.  A cell with no remaining
        candidate contributes an infeasible placeholder slot (unique id,
        zero capacity) so the action space stays a fixed-size, cell-indexed
        table for the whole episode.
        """
        x_min, x_max, y_min, y_max = self._position_bounds
        records = []
        nodes = []
        for i in range(self.nx):
            for j in range(self.ny):
                cell_index = i * self.ny + j
                node = preview_nodes[cell_index] if preview_nodes is not None else None
                if node is None:
                    records.append((f"empty-cell-{cell_index}", cell_index, 0.0, 0.5, 0.5))
                    nodes.append(None)
                    continue
                capacity_function = getattr(
                    self.core.shelterDS,
                    "configuredShelterCapacity",
                    None,
                )
                capacity = (
                    float(capacity_function(node))
                    if callable(capacity_function)
                    else max(0.0, float(getattr(node, "nodeCap", 0.0)))
                )
                x_coord = float(getattr(node, "nodeX", 0.0))
                y_coord = float(getattr(node, "nodeY", 0.0))
                east = (
                    float(np.clip((x_coord - x_min) / (x_max - x_min), 0.0, 1.0))
                    if x_max > x_min
                    else 0.5
                )
                north = (
                    float(np.clip((y_max - y_coord) / (y_max - y_min), 0.0, 1.0))
                    if y_max > y_min
                    else 0.5
                )
                records.append(
                    (str(getattr(node, "OSMID", "")), cell_index, capacity, east, north)
                )
                nodes.append(node)
        return tuple(records), tuple(nodes)

    @property
    def number_of_actions(self) -> int:
        return self.number_of_cells + int(self.defer_enabled)

    def _candidate_action_mask(
        self,
        remaining_deployments: int,
        forecast_danger: np.ndarray,
        records: Sequence[tuple[str, int, float, float, float]],
    ) -> np.ndarray:
        """Return the structurally executable shelter-cell actions.

        Forecast danger, distance from the hazard, reroutable population, and
        analytical risk reduction are observations for the learned policy.
        They must not censor experience: a physically installable but poor
        location remains selectable and its realized negative return is part
        of training.  Only an exhausted deployment budget or the absence of a
        positive-capacity candidate makes an action impossible.
        """
        del forecast_danger
        return np.asarray(
            [
                remaining_deployments > 0 and float(capacity) > 0.0
                for _, _, capacity, _, _ in records
            ],
            dtype=np.bool_,
        )

    def _network_node_counts(self) -> np.ndarray:
        counts = np.zeros(self.number_of_cells, dtype=np.float32)
        nodes = getattr(getattr(self.core, "mapDS", None), "nodeListByLocalID", {})
        for node in nodes.values():
            cell = getattr(node, "cellID", None)
            if cell is None:
                continue
            i, j = int(cell[0]), int(cell[1])
            if 0 <= i < self.nx and 0 <= j < self.ny:
                counts[i * self.ny + j] += 1.0
        return counts

    def _network_load_share(self) -> float:
        """Return initial demand relative to modeled physical link storage.

        The observation otherwise normalizes population by itself, which makes
        differently loaded copies of the same map difficult to distinguish at
        the first decision.  Deduplicating counter-flow edges uses the same
        physical-link identity as the congestion model.
        """
        map_database = getattr(self.core, "mapDS", None)
        edges = getattr(map_database, "edgeListByLocalID", {})
        congestion = getattr(self.core, "congestionModel", None)
        if not edges or congestion is None:
            return 0.0
        effective_width = max(1e-12, float(congestion.effective_width_m))
        jam_density = max(1e-12, float(congestion.jam_density_ped_per_m2))
        physical_lengths = {}
        for edge in edges.values():
            key = congestion.physical_link_key(edge)
            physical_lengths[key] = max(
                float(physical_lengths.get(key, 0.0)),
                max(0.0, float(getattr(edge, "edgeLen", 0.0))),
            )
        storage = sum(physical_lengths.values()) * effective_width * jam_density
        if storage <= 0.0:
            return 0.0
        population = float(max(1, self.initial_population))
        return float(np.clip(population / (population + storage), 0.0, 1.0))

    @staticmethod
    def _remaining_route_distance(pedestrian) -> float:
        """Return the currently assigned route's remaining physical distance."""
        route = getattr(pedestrian, "routeFollowing", None)
        if route is None:
            return float("inf")
        distance = max(0.0, float(getattr(pedestrian, "edge_remain", 0.0)))
        for edge in (getattr(route, "edgeRemained", None) or ()):
            distance += max(0.0, float(getattr(edge, "edgeLen", 0.0)))
        return float(distance)

    def _node_cell_index(self, node) -> Optional[int]:
        if node is None:
            return None
        cell = getattr(node, "cellID", None)
        if cell is None:
            tracker = getattr(self.core, "cellTracker", None)
            locate = getattr(tracker, "locateCell", None)
            if callable(locate):
                try:
                    cell = locate(float(node.nodeX), float(node.nodeY))
                except Exception:
                    cell = None
        if cell is None:
            return None
        i, j = int(cell[0]), int(cell[1])
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return None
        return i * self.ny + j

    def _road_nodes_by_region(self) -> tuple[tuple[object, ...], ...]:
        nodes_by_region: list[list[object]] = [
            [] for _ in range(self.number_of_cells)
        ]
        map_database = getattr(self.core, "mapDS", None)
        for node in getattr(map_database, "nodeListByLocalID", {}).values():
            cell_index = self._node_cell_index(node)
            if cell_index is not None:
                nodes_by_region[cell_index].append(node)
        return tuple(tuple(nodes) for nodes in nodes_by_region)

    def _site_region_distances_m(self, site_node) -> np.ndarray:
        """Road distance from each region to one candidate site (metres)."""
        map_database = getattr(self.core, "mapDS", None)
        key = (
            str(getattr(site_node, "OSMID", "")),
            int(getattr(map_database, "routing_version", 0)),
        )
        cached = self._site_region_distance_cache.get(key)
        if cached is not None:
            return cached
        result = np.full(self.number_of_cells, np.inf, dtype=np.float64)
        tree = getattr(map_database, "_routing_tree", None)
        if callable(tree) and site_node is not None:
            try:
                distances, _ = tree(int(site_node.OSMID))
            except Exception:
                distances = {}
            for index, nodes in enumerate(self._region_road_nodes):
                result[index] = min(
                    (
                        float(distances.get(int(node.OSMID), np.inf))
                        for node in nodes
                    ),
                    default=np.inf,
                )
        centers = self._region_centers_m()
        euclidean = np.hypot(
            centers[:, 0] - float(getattr(site_node, "nodeX", 0.0)),
            centers[:, 1] - float(getattr(site_node, "nodeY", 0.0)),
        )
        result = np.where(np.isfinite(result), result, euclidean)
        if len(self._site_region_distance_cache) >= 64:
            self._site_region_distance_cache.pop(
                next(iter(self._site_region_distance_cache))
            )
        self._site_region_distance_cache[key] = result
        return result

    def _pedestrian_layer(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Aggregate individual route and wellness telemetry by region."""
        active = np.zeros(self.number_of_cells, dtype=np.float32)
        speed_sum = np.zeros(self.number_of_cells, dtype=np.float32)
        congestion_ratio_sum = np.zeros(self.number_of_cells, dtype=np.float32)
        route_time_sum = np.zeros(self.number_of_cells, dtype=np.float32)
        slow_tail_population = np.zeros(self.number_of_cells, dtype=np.float32)
        slow_tail_time = np.zeros(self.number_of_cells, dtype=np.float32)
        routed_population = np.zeros(self.number_of_cells, dtype=np.float32)
        long_route_population = np.zeros(self.number_of_cells, dtype=np.float32)
        stable_population = np.zeros(self.number_of_cells, dtype=np.float32)
        exposed_population = np.zeros(self.number_of_cells, dtype=np.float32)
        panicked_population = np.zeros(self.number_of_cells, dtype=np.float32)
        assigned_population = np.zeros(self.number_of_cells, dtype=np.float32)
        unassigned_population = np.zeros(self.number_of_cells, dtype=np.float32)
        outgoing_assignment_flow = np.zeros(
            self.number_of_cells, dtype=np.float32
        )
        incoming_assignment_flow = np.zeros(
            self.number_of_cells, dtype=np.float32
        )
        route_flow = {}
        active_route_records: list[tuple[float, int, int]] = []
        reservation_for = getattr(self.core.shelterDS, "reservationFor", None)
        time_step = max(1e-6, float(getattr(self.core, "timeStepMinutes", 1.0)))
        route_threshold = max(
            float(getattr(self.core, "shelterActionInterval", 1)) * time_step,
            0.25 * float(self.horizon) * time_step,
        )
        maximum_route_time = float(self.horizon) * time_step
        speed_floor = max(1e-6, 0.1 * float(self.core.maxSpeed))
        for pedestrian in self.core.pedDS.pedAgentList.values():
            if bool(getattr(pedestrian, "terminated", False)):
                continue
            cell = getattr(pedestrian, "currCell", None)
            if cell is None:
                continue
            i, j = int(cell[0]), int(cell[1])
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                continue
            index = i * self.ny + j
            group_size = max(1, int(getattr(pedestrian, "group_size", 1)))
            active[index] += group_size
            speed_sum[index] += group_size * max(0.0, float(getattr(pedestrian, "currSpeed", 0.0)))
            congestion_ratio_sum[index] += group_size * float(
                np.clip(
                    getattr(pedestrian, "congestion_speed_ratio", 1.0),
                    0.0,
                    1.0,
                )
            )
            is_panicked = bool(getattr(pedestrian, "panicked", False))
            is_exposed = bool(getattr(pedestrian, "affected", False))
            if is_panicked:
                panicked_population[index] += group_size
            elif is_exposed:
                exposed_population[index] += group_size
            else:
                stable_population[index] += group_size

            route = getattr(pedestrian, "routeFollowing", None)
            route_distance = self._remaining_route_distance(pedestrian)
            assigned_shelter, reserved_population = (
                reservation_for(pedestrian)
                if callable(reservation_for)
                else (None, 0)
            )
            route_target = getattr(route, "endNode", None) if route is not None else None
            shelter_target = getattr(assigned_shelter, "nodeMapped", None)
            has_valid_assignment = bool(
                not is_panicked
                and route is not None
                and assigned_shelter is not None
                and int(reserved_population) >= group_size
                and int(getattr(assigned_shelter, "status", 0)) == 0
                and route_target is not None
                and shelter_target is not None
                and getattr(route_target, "OSMID", None)
                == getattr(shelter_target, "OSMID", None)
                and np.isfinite(route_distance)
            )
            if has_valid_assignment:
                assigned_population[index] += group_size
            else:
                unassigned_population[index] += group_size
            tail_route_time = maximum_route_time
            if not is_panicked and np.isfinite(route_distance):
                speed = max(
                    speed_floor,
                    float(getattr(pedestrian, "currSpeed", 0.0)),
                )
                route_time = min(maximum_route_time, route_distance / speed)
                if has_valid_assignment:
                    tail_route_time = route_time
                route_time_sum[index] += group_size * route_time
                routed_population[index] += group_size
                if route_time > route_threshold:
                    long_route_population[index] += group_size
                target_index = self._node_cell_index(
                    route_target
                )
                if target_index is not None and target_index != index:
                    if has_valid_assignment:
                        outgoing_assignment_flow[index] += group_size
                        incoming_assignment_flow[target_index] += group_size
                    # Shelter-region information must flow back to the origin
                    # region where the long-route demand is observed.
                    # Use both directions. The origin receives information
                    # about its assigned shelter, while the shelter region
                    # receives the magnitude and condition of assigned demand.
                    for key in ((target_index, index), (index, target_index)):
                        route_flow[key] = float(route_flow.get(key, 0.0)) + group_size
            # The optimized clearance term concerns the slowest population,
            # not the mean route in a region.  Preserve the current location of
            # that mass explicitly.  A pedestrian without a capacity-valid
            # route (including persistent panic) receives the same unresolved
            # horizon used by the complete cost ledger.
            active_route_records.append((tail_route_time, index, group_size))

        tail_population = min(
            int(round(float(active.sum()))),
            max(
                1,
                int(
                    np.ceil(
                        float(getattr(self.core, "clearanceTailFraction", 0.10))
                        * float(self.initial_population)
                    )
                ),
            ),
        )
        remaining_tail = tail_population
        for route_time, index, group_size in sorted(
            active_route_records,
            key=lambda item: (-float(item[0]), int(item[1])),
        ):
            if remaining_tail <= 0:
                break
            included = min(int(group_size), int(remaining_tail))
            slow_tail_population[index] += included
            slow_tail_time[index] += included * float(route_time)
            remaining_tail -= included
        mean_speed = np.divide(
            speed_sum,
            active,
            out=np.zeros_like(speed_sum),
            where=active > 0.0,
        )
        mean_route_time = np.divide(
            route_time_sum,
            routed_population,
            out=np.zeros_like(route_time_sum),
            where=routed_population > 0.0,
        )
        mean_congestion_ratio = np.divide(
            congestion_ratio_sum,
            active,
            out=np.ones_like(congestion_ratio_sum),
            where=active > 0.0,
        )
        long_route_share = np.divide(
            long_route_population,
            active,
            out=np.zeros_like(active),
            where=active > 0.0,
        )
        stable_share = np.divide(
            stable_population,
            active,
            out=np.zeros_like(active),
            where=active > 0.0,
        )
        exposed_share = np.divide(
            exposed_population,
            active,
            out=np.zeros_like(active),
            where=active > 0.0,
        )
        panicked_share = np.divide(
            panicked_population,
            active,
            out=np.zeros_like(active),
            where=active > 0.0,
        )
        if route_flow:
            items = sorted(route_flow.items())
            route_edges = np.asarray([item[0] for item in items], dtype=np.int64).T
            route_weights = np.asarray(
                [item[1] / max(1, self.initial_population) for item in items],
                dtype=np.float32,
            )
        else:
            route_edges = np.empty((2, 0), dtype=np.int64)
            route_weights = np.empty(0, dtype=np.float32)
        return (
            active,
            mean_speed,
            mean_congestion_ratio,
            assigned_population,
            unassigned_population,
            outgoing_assignment_flow,
            incoming_assignment_flow,
            mean_route_time,
            slow_tail_population,
            slow_tail_time,
            long_route_share,
            stable_share,
            exposed_share,
            panicked_share,
            route_edges,
            route_weights,
        )

    def _regional_capacity(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list,
    ]:
        remaining = np.zeros(self.number_of_cells, dtype=np.float32)
        installed = np.zeros(self.number_of_cells, dtype=np.float32)
        sheltered = np.zeros(self.number_of_cells, dtype=np.float32)
        reserved_enroute = np.zeros(self.number_of_cells, dtype=np.float32)
        deployable = np.zeros(self.number_of_cells, dtype=np.float32)
        candidate_count = np.zeros(self.number_of_cells, dtype=np.float32)
        preview_nodes: list = [None] * self.number_of_cells
        shelter_grid = getattr(self.core.shelterDS, "shelterByCell", None)
        candidate_grid = getattr(self.core.shelterDS, "shelterCanByCell", None)
        if shelter_grid is None or candidate_grid is None:
            return (
                remaining,
                installed,
                sheltered,
                reserved_enroute,
                np.zeros(self.number_of_cells, dtype=np.float32),
                deployable,
                candidate_count,
                preview_nodes,
            )
        reserved_population = getattr(
            self.core.shelterDS, "reservedPopulation", None
        )
        available_capacity = getattr(
            self.core.shelterDS, "availableCapacity", None
        )

        for i in range(self.nx):
            for j in range(self.ny):
                index = i * self.ny + j
                for shelter in shelter_grid[i][j]:
                    status = int(getattr(shelter, "status", 0))
                    if status == 2:
                        # Installed-but-overrun capacity remains visible in
                        # audit outputs, but it is not usable capacity and must
                        # not dilute operational utilization in the actor's
                        # observation.
                        continue
                    capacity = max(0.0, float(getattr(shelter, "shelterCap", 0.0)))
                    flow = max(0.0, float(getattr(shelter, "shelterFlow", 0.0)))
                    reserved = max(
                        0.0,
                        float(reserved_population(shelter))
                        if callable(reserved_population)
                        else 0.0,
                    )
                    installed[index] += capacity
                    sheltered[index] += min(flow, capacity)
                    reserved_enroute[index] += min(
                        reserved,
                        max(0.0, capacity - flow),
                    )
                    if status == 0:
                        remaining[index] += (
                            float(available_capacity(shelter))
                            if callable(available_capacity)
                            else max(0.0, capacity - flow)
                        )
                candidates = candidate_grid[i][j]
                candidate_count[index] = float(len(candidates))
                if candidates:
                    preview = self.core.shelterDS.previewShelterCandidate(
                        (i, j),
                        self.core.cellTracker,
                    )
                    if preview is not None:
                        capacity_function = getattr(
                            self.core.shelterDS,
                            "configuredShelterCapacity",
                            None,
                        )
                        deployable[index] = (
                            float(capacity_function(preview))
                            if callable(capacity_function)
                            else max(
                                0.0,
                                float(getattr(preview, "nodeCap", 0.0)),
                            )
                        )
                        preview_nodes[index] = preview
        committed = sheltered + reserved_enroute
        utilization = np.divide(
            committed,
            installed,
            out=np.zeros_like(committed),
            where=installed > 0.0,
        )
        return (
            remaining,
            installed,
            sheltered,
            reserved_enroute,
            utilization,
            deployable,
            candidate_count,
            preview_nodes,
        )

    def _candidate_reevacuation_response(
        self,
        records: Sequence[tuple[str, int, float, float, float]],
        forecast_reevacuation_population: np.ndarray,
        overrun_lead_time_fraction: np.ndarray,
        candidate_overrun_probability_by_cell: np.ndarray,
    ) -> np.ndarray:
        """Capacity-, timing-, and safety-aware access to re-evacuees."""
        displaced = np.asarray(
            forecast_reevacuation_population, dtype=np.float64
        )
        lead = np.asarray(overrun_lead_time_fraction, dtype=np.float64)
        east = np.asarray(self.region_east_positions, dtype=np.float64)
        north = np.asarray(self.region_north_positions, dtype=np.float64)
        population = float(max(1, self.initial_population))
        result = np.zeros(len(records), dtype=np.float32)
        if not np.any(displaced > 0.0):
            return result
        normalizer = float(np.sqrt(2.0))
        for action, (candidate_id, _cell, capacity, x_pos, y_pos) in enumerate(records):
            if candidate_id == "__defer__":
                continue
            distances = np.hypot(east - float(x_pos), north - float(y_pos))
            proximity = 1.0 - np.clip(distances / normalizer, 0.0, 1.0)
            # Imminent failures receive more weight because the shelter placed
            # now is the last capacity token guaranteed to exist beforehand.
            timing = 1.0 - 0.5 * np.clip(lead, 0.0, 1.0)
            reachable_displaced = min(
                max(0.0, float(capacity)),
                float(np.sum(displaced * proximity * timing)),
            )
            survival = 1.0 - float(
                np.clip(
                    candidate_overrun_probability_by_cell[int(_cell)],
                    0.0,
                    1.0,
                )
            )
            result[action] = float(
                np.clip(
                    survival * reachable_displaced / population,
                    0.0,
                    1.0,
                )
            )
        return result

    def _reservation_arrivals_by_shelter_cell(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return grouped reservation ETA records keyed by shelter cell.

        Records are ``(arrival_steps, represented_population)``.  They use
        the same grouped pedestrian objects as the simulator, so the work is
        proportional to the number of cohorts rather than the represented
        population.
        """
        records: list[list[tuple[int, int]]] = [
            [] for _ in range(self.number_of_cells)
        ]
        shelter_store = getattr(self.core, "shelterDS", None)
        pedestrian_store = getattr(self.core, "pedDS", None)
        reservations = getattr(shelter_store, "reservationByPedestrian", {})
        shelters = getattr(shelter_store, "shelterList", {})
        pedestrians = getattr(pedestrian_store, "pedAgentList", {})
        time_step = max(
            1e-9, float(getattr(self.core, "timeStepMinutes", 1.0))
        )
        speed_floor = max(1e-6, 0.1 * float(self.core.maxSpeed))
        for pedestrian_id, reservation in reservations.items():
            shelter_id, population = (int(value) for value in reservation)
            if population <= 0:
                raise RuntimeError("Shelter reservation population must be positive")
            shelter = shelters.get(shelter_id)
            pedestrian = pedestrians.get(int(pedestrian_id))
            if (
                shelter is None
                or pedestrian is None
                or bool(getattr(pedestrian, "terminated", False))
                or int(getattr(shelter, "status", 0)) != 0
            ):
                raise RuntimeError(
                    "Operational reservation index contains an invalid cohort"
                )
            cell = getattr(shelter, "cellLocated", None)
            if cell is None:
                raise RuntimeError("Operational shelter has no regional cell")
            i, j = (int(value) for value in cell)
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                raise RuntimeError("Operational shelter cell is outside the grid")
            distance = self._remaining_route_distance(pedestrian)
            if np.isfinite(distance):
                speed = max(
                    speed_floor,
                    float(getattr(pedestrian, "currSpeed", 0.0)),
                )
                arrival_steps = max(
                    1, int(np.ceil((distance / speed) / time_step))
                )
            else:
                arrival_steps = int(self.horizon) + 1
            records[i * self.ny + j].append((arrival_steps, population))
        return tuple(tuple(cell_records) for cell_records in records)

    def _susceptible_arrivals_by_cell(
        self,
        window_steps: int,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Return steps-to-safety records for cohorts without a panic trial."""
        window = max(1, int(window_steps))
        records: list[list[tuple[int, int]]] = [
            [] for _ in range(self.number_of_cells)
        ]
        reservation_for = getattr(self.core.shelterDS, "reservationFor", None)
        time_step = max(
            1e-9, float(getattr(self.core, "timeStepMinutes", 1.0))
        )
        speed_floor = max(1e-6, 0.1 * float(self.core.maxSpeed))
        for pedestrian in self.core.pedDS.pedAgentList.values():
            if (
                bool(getattr(pedestrian, "terminated", False))
                or bool(getattr(pedestrian, "panicked", False))
                or bool(getattr(pedestrian, "affected", False))
            ):
                continue
            cell = getattr(pedestrian, "currCell", None)
            if cell is None:
                continue
            i, j = int(cell[0]), int(cell[1])
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                continue
            group_size = max(1, int(getattr(pedestrian, "group_size", 1)))
            route = getattr(pedestrian, "routeFollowing", None)
            route_distance = self._remaining_route_distance(pedestrian)
            assigned_shelter, reserved_population = (
                reservation_for(pedestrian)
                if callable(reservation_for)
                else (None, 0)
            )
            route_target = getattr(route, "endNode", None) if route is not None else None
            shelter_target = getattr(assigned_shelter, "nodeMapped", None)
            has_valid_assignment = bool(
                route is not None
                and assigned_shelter is not None
                and int(reserved_population) >= group_size
                and int(getattr(assigned_shelter, "status", 0)) == 0
                and route_target is not None
                and shelter_target is not None
                and getattr(route_target, "OSMID", None)
                == getattr(shelter_target, "OSMID", None)
                and np.isfinite(route_distance)
            )
            if has_valid_assignment:
                speed = max(
                    speed_floor,
                    float(getattr(pedestrian, "currSpeed", 0.0)),
                )
                tau = int(np.ceil((route_distance / speed) / time_step))
            else:
                tau = window
            records[i * self.ny + j].append(
                (min(max(0, int(tau)), window), group_size)
            )
        return tuple(tuple(cell_records) for cell_records in records)

    def _front_layer(
        self,
        danger: np.ndarray,
        simulation_time: int,
        susceptible_arrivals: tuple[tuple[tuple[int, int], ...], ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Panic-level first passage and its threatened population."""
        interval = max(1, int(getattr(self.core, "shelterActionInterval", 1)))
        remaining = max(1, int(self.horizon) - int(simulation_time))
        window = min(
            remaining,
            max(1, int(getattr(self.core, "nmccFrontForecastIntervals", 4)))
            * interval,
        )
        level = float(
            getattr(
                self.core,
                "panicDangerThreshold",
                getattr(self.core.pedDS, "panic_danger_threshold", 3),
            )
        )
        hazard_store = getattr(self.core, "hazardDS", None)
        first_passage = getattr(
            hazard_store, "belief_overrun_first_passage_by_cell", None
        )
        if callable(first_passage):
            seed = self._belief_forecast_seed(danger, simulation_time)
            distribution = first_passage(
                window,
                level,
                ensemble_size=int(getattr(self.core, "forecastEnsembleSize", 8)),
                rng=np.random.default_rng(np.random.SeedSequence([seed, 64])),
                next_event_mask=None,
            )
            front_cdf = distribution.get("crossing_cdf_by_step")
            if front_cdf is None:
                probability = np.asarray(
                    distribution["crossing_probability"], dtype=np.float64
                ).reshape(self.number_of_cells)
                crossing_steps = np.asarray(
                    distribution["conditional_expected_first_crossing_steps"],
                    dtype=np.float64,
                ).reshape(self.number_of_cells)
                front_cdf = np.zeros(
                    (window + 1, self.number_of_cells), dtype=np.float64
                )
                for cell_index, cell_probability in enumerate(probability):
                    step = min(
                        window,
                        max(0, int(np.ceil(crossing_steps[cell_index]))),
                    )
                    front_cdf[step:, cell_index] = cell_probability
            front_cdf = np.asarray(front_cdf, dtype=np.float64).reshape(
                window + 1, self.number_of_cells
            )
            lead_steps = np.asarray(
                distribution["conditional_expected_first_crossing_steps"],
                dtype=np.float64,
            ).reshape(self.number_of_cells)
        else:
            threshold = float(np.clip(level / 5.0, 0.0, 1.0))
            forecast_method = getattr(hazard_store, "forecast_danger_by_cell", None)
            forecast = (
                np.asarray(forecast_method(window), dtype=np.float64).reshape(
                    self.number_of_cells
                )
                if callable(forecast_method)
                else np.asarray(danger, dtype=np.float64)
            )
            current_crossing = np.asarray(danger, dtype=np.float64) >= threshold - 1e-12
            eventual_crossing = forecast >= threshold - 1e-12
            front_cdf = np.zeros(
                (window + 1, self.number_of_cells), dtype=np.float64
            )
            front_cdf[0] = current_crossing.astype(np.float64)
            if window > 0:
                front_cdf[1:] = eventual_crossing.astype(np.float64)
            lead_steps = np.where(
                current_crossing,
                0.0,
                np.where(eventual_crossing, 1.0, float(window + 1)),
            )
        front_cdf = np.clip(
            np.nan_to_num(front_cdf, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        front_cdf = np.maximum.accumulate(front_cdf, axis=0)
        arrival = front_cdf[-1]
        lead = np.clip(
            np.nan_to_num(
                lead_steps / float(window + 1),
                nan=1.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        susceptible = np.zeros(self.number_of_cells, dtype=np.float64)
        threatened = np.zeros(self.number_of_cells, dtype=np.float64)
        for cell_index, cohorts in enumerate(susceptible_arrivals):
            for tau, population in cohorts:
                susceptible[cell_index] += int(population)
                threatened[cell_index] += int(population) * float(
                    front_cdf[min(max(0, int(tau)), window), cell_index]
                )
        return (
            front_cdf.astype(np.float32),
            arrival.astype(np.float32),
            lead.astype(np.float32),
            susceptible.astype(np.float32),
            threatened.astype(np.float32),
            window,
        )

    def _candidate_front_features(
        self,
        records: Sequence[tuple[str, int, float, float, float]],
        nodes: Sequence[object],
        front_cdf: np.ndarray,
        susceptible_arrivals: tuple[tuple[tuple[int, int], ...], ...],
        lead: np.ndarray,
        window: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return action-level rescue gain and fire-front geometry."""
        action_count = len(records)
        capture = np.zeros(action_count, dtype=np.float32)
        approach = np.full(action_count, 0.5, dtype=np.float32)
        arrival = np.zeros(action_count, dtype=np.float32)
        candidate_lead = np.ones(action_count, dtype=np.float32)
        population_scale = float(max(1, self.initial_population))
        time_step = max(
            1e-9, float(getattr(self.core, "timeStepMinutes", 1.0))
        )
        travel_denominator = max(1e-9, float(self.core.maxSpeed) * time_step)
        centers = self._region_centers_m()
        hazard_store = getattr(self.core, "hazardDS", None)
        source_points = []
        for hazard in getattr(hazard_store, "hazardList", {}).values():
            if not bool(getattr(hazard, "active", True)):
                continue
            source = getattr(hazard, "sourceNode", None)
            if source is not None:
                source_points.append(
                    np.asarray(
                        (
                            float(getattr(source, "nodeX", 0.0)),
                            float(getattr(source, "nodeY", 0.0)),
                        ),
                        dtype=np.float64,
                    )
                )
        cdf = np.asarray(front_cdf, dtype=np.float64)
        lead_values = np.asarray(lead, dtype=np.float64)
        for action, (record, node) in enumerate(zip(records, nodes)):
            candidate_id, host_cell, capacity, _x_pos, _y_pos = record
            if candidate_id == "__defer__" or node is None:
                continue
            host_cell = int(host_cell)
            arrival[action] = float(cdf[-1, host_cell])
            candidate_lead[action] = float(lead_values[host_cell])
            distances = self._site_region_distances_m(node)
            travel_steps = np.minimum(
                int(window),
                np.maximum(
                    0,
                    np.ceil(distances / travel_denominator).astype(np.int64),
                ),
            )
            site_point = np.asarray(
                (
                    float(getattr(node, "nodeX", 0.0)),
                    float(getattr(node, "nodeY", 0.0)),
                ),
                dtype=np.float64,
            )
            gain_total = 0.0
            approach_total = 0.0
            for cell_index, cohorts in enumerate(susceptible_arrivals):
                if not cohorts:
                    continue
                site_vector = site_point - centers[cell_index]
                directional_factor = 0.5
                if source_points:
                    source = min(
                        source_points,
                        key=lambda point: float(
                            np.sum((point - centers[cell_index]) ** 2)
                        ),
                    )
                    source_vector = source - centers[cell_index]
                    norm = float(
                        np.linalg.norm(site_vector)
                        * np.linalg.norm(source_vector)
                    )
                    if norm > 1e-12:
                        cosine = float(
                            np.clip(
                                np.dot(site_vector, source_vector) / norm,
                                -1.0,
                                1.0,
                            )
                        )
                        directional_factor = 0.5 * (1.0 + cosine)
                site_steps = int(travel_steps[cell_index])
                for tau, represented_population in cohorts:
                    tau = min(max(0, int(tau)), int(window))
                    if not (tau == int(window) or site_steps < tau):
                        continue
                    gain = int(represented_population) * max(
                        0.0,
                        float(cdf[tau, cell_index])
                        - float(cdf[min(site_steps, int(window)), cell_index]),
                    )
                    gain_total += gain
                    approach_total += gain * directional_factor
            capture[action] = float(
                np.clip(
                    min(max(0.0, float(capacity)), gain_total)
                    / population_scale,
                    0.0,
                    1.0,
                )
            )
            if gain_total > 0.0:
                approach[action] = float(
                    np.clip(approach_total / gain_total, 0.0, 1.0)
                )
        return capture, approach, arrival, candidate_lead

    def _forecast_shelter_transition_burdens(
        self,
        sheltered_population: np.ndarray,
        reserved_population: np.ndarray,
        reservation_arrivals: tuple[tuple[tuple[int, int], ...], ...],
        crossing_cdf_by_step: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split threatened reservations by whether arrival precedes failure.

        A cohort admitted before the crossing reverses safe stock and is
        counted as re-evacuation.  A crossing at or before its arrival only
        releases/reroutes the reservation.  The split integrates over the
        public-belief first-passage distribution rather than comparing ETA to
        one point forecast.
        """
        cdf = np.asarray(crossing_cdf_by_step, dtype=np.float64)
        if cdf.ndim != 2 or cdf.shape[1] != self.number_of_cells:
            raise ValueError(
                "crossing_cdf_by_step must have shape (steps, cells)"
            )
        if cdf.shape[0] < 1 or not np.isfinite(cdf).all():
            raise ValueError("crossing_cdf_by_step must be finite and non-empty")
        if np.any(cdf < -1e-9) or np.any(cdf > 1.0 + 1e-9):
            raise ValueError("crossing_cdf_by_step must contain probabilities")
        if np.any(np.diff(cdf, axis=0) < -1e-7):
            raise ValueError("First-passage crossing CDF must be monotone")
        final_probability = np.clip(cdf[-1], 0.0, 1.0)
        reevacuation = (
            np.asarray(sheltered_population, dtype=np.float64)
            * final_probability
        )
        reroute = np.zeros(self.number_of_cells, dtype=np.float64)
        reservation_totals = np.zeros(self.number_of_cells, dtype=np.float64)
        final_step = int(cdf.shape[0] - 1)
        for cell_index, records in enumerate(reservation_arrivals):
            for arrival_steps, population in records:
                population = int(population)
                reservation_totals[cell_index] += population
                by_arrival = float(
                    cdf[min(max(0, int(arrival_steps)), final_step), cell_index]
                )
                reroute[cell_index] += population * by_arrival
                reevacuation[cell_index] += population * max(
                    0.0, final_probability[cell_index] - by_arrival
                )
        if not np.allclose(
            reservation_totals,
            np.asarray(reserved_population, dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Reservation ETA records disagree with regional reserved population"
            )
        return (
            reevacuation.astype(np.float32),
            reroute.astype(np.float32),
        )

    def _region_centers_m(self) -> np.ndarray:
        tracker = getattr(self.core, "cellTracker", None)
        get_center = getattr(tracker, "getCellCenter", None)
        centers = []
        for i in range(self.nx):
            for j in range(self.ny):
                if callable(get_center):
                    try:
                        x_coord, y_coord = get_center((i, j))
                        centers.append((float(x_coord), float(y_coord)))
                        continue
                    except Exception:
                        pass
                centers.append(
                    (
                        float(self.region_east_positions[i * self.ny + j]),
                        float(self.region_north_positions[i * self.ny + j]),
                    )
                )
        return np.asarray(centers, dtype=np.float64)

    def _belief_forecast_seed(
        self,
        danger: np.ndarray,
        simulation_time: int,
    ) -> int:
        """Key finite belief sampling only to information the actor observes.

        The realized regime stream generated the hidden episode tape and must
        never seed a policy input.  A stable public-state key makes identical
        observable states use identical forecast quadrature, while changes in
        the observed fire front, public warnings, or decision time legitimately
        produce a new ensemble.
        """
        regime = getattr(self.core, "hazard_regime", None)
        public = getattr(regime, "public", {}) if regime is not None else {}
        digest = hashlib.sha256()
        digest.update(np.asarray(self.nx, dtype="<i8").tobytes())
        digest.update(np.asarray(self.ny, dtype="<i8").tobytes())
        digest.update(np.asarray(int(simulation_time), dtype="<i8").tobytes())
        digest.update(
            np.ascontiguousarray(danger, dtype="<f4").reshape(-1).tobytes()
        )
        for key in sorted(public):
            digest.update(str(key).encode("utf-8"))
            digest.update(repr(public[key]).encode("utf-8"))
        return int.from_bytes(digest.digest()[:8], "little", signed=False)

    def _hazard_layer(
        self,
        danger: np.ndarray,
        simulation_time: int,
        operational_shelter_cells: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
    ]:
        """Return public belief forecasts, including overrun first passage."""
        hazard_store = getattr(self.core, "hazardDS", None)
        interval = max(1, int(getattr(self.core, "shelterActionInterval", 1)))
        remaining = max(1, int(self.horizon) - int(simulation_time))
        # Two decision intervals give the actor enough lead to pre-position a
        # replacement before the next threatened shelter fails, without
        # pretending that a far-horizon cell order is precise.
        forecast_horizon = min(remaining, 2 * interval)
        first_passage_method = getattr(
            hazard_store, "belief_overrun_first_passage_by_cell", None
        )
        if callable(first_passage_method):
            seed = self._belief_forecast_seed(danger, simulation_time)
            distribution = first_passage_method(
                forecast_horizon,
                float(getattr(self.core, "shelterOverrunDangerLevel", 4)),
                ensemble_size=int(
                    getattr(self.core, "forecastEnsembleSize", 8)
                ),
                rng=np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, 63]
                    )
                ),
                next_event_mask=np.asarray(
                    operational_shelter_cells, dtype=np.bool_
                ),
            )
            forecast = np.asarray(
                distribution["expected"], dtype=np.float32
            ).reshape(self.number_of_cells)
            momentum = np.asarray(
                distribution["momentum"], dtype=np.float32
            ).reshape(self.number_of_cells)
            escalation = np.asarray(
                distribution["escalation_probability"], dtype=np.float32
            ).reshape(self.number_of_cells)
            uncertainty = np.asarray(
                distribution["uncertainty"], dtype=np.float32
            ).reshape(self.number_of_cells)
            overrun_probability = np.asarray(
                distribution["crossing_probability"], dtype=np.float32
            ).reshape(self.number_of_cells)
            crossing_cdf = distribution.get("crossing_cdf_by_step")
            if crossing_cdf is None:
                # Compatibility for a deterministic external forecaster: put
                # its crossing mass at the reported conditional lead.
                crossing_cdf = np.zeros(
                    (forecast_horizon + 1, self.number_of_cells),
                    dtype=np.float32,
                )
                reported_leads = np.asarray(
                    distribution["conditional_expected_first_crossing_steps"],
                    dtype=np.float64,
                ).reshape(self.number_of_cells)
                for cell_index, probability in enumerate(overrun_probability):
                    lead = min(
                        forecast_horizon,
                        max(0, int(np.ceil(reported_leads[cell_index]))),
                    )
                    crossing_cdf[lead:, cell_index] = float(probability)
            crossing_cdf = np.asarray(crossing_cdf, dtype=np.float32).reshape(
                forecast_horizon + 1, self.number_of_cells
            )
            next_overrun_probability = np.asarray(
                distribution["next_overrun_probability"], dtype=np.float32
            ).reshape(self.number_of_cells)
            next_shelter_overrun_probability = np.asarray(
                distribution["next_priority_overrun_probability"],
                dtype=np.float32,
            ).reshape(self.number_of_cells)
            lead_steps = np.asarray(
                distribution["conditional_expected_first_crossing_steps"],
                dtype=np.float32,
            ).reshape(self.number_of_cells)
            overrun_lead_time = np.clip(
                lead_steps / float(forecast_horizon + 1), 0.0, 1.0
            )
        else:
            forecast_method = getattr(
                hazard_store, "forecast_danger_by_cell", None
            )
            forecast = (
                np.asarray(
                    forecast_method(forecast_horizon), dtype=np.float32
                ).reshape(
                    self.number_of_cells
                )
                if callable(forecast_method)
                else danger.copy()
            )
            momentum = np.zeros_like(danger, dtype=np.float32)
            unaffected = np.maximum(1.0 - danger, 1e-12)
            escalation = np.where(
                danger < 1.0 - 1e-12,
                (forecast - danger) / unaffected,
                0.0,
            )
            escalation = np.clip(escalation, 0.0, 1.0)
            uncertainty = 4.0 * escalation * (1.0 - escalation)
            threshold = float(
                np.clip(
                    float(
                        getattr(self.core, "shelterOverrunDangerLevel", 4)
                    )
                    / 5.0,
                    0.0,
                    1.0,
                )
            )
            overrun_probability = (
                forecast >= threshold - 1e-12
            ).astype(np.float32)
            crossing_cdf = np.zeros(
                (forecast_horizon + 1, self.number_of_cells),
                dtype=np.float32,
            )
            crossing_cdf[0] = (danger >= threshold - 1e-12).astype(
                np.float32
            )
            if forecast_horizon > 0:
                crossing_cdf[1:] = overrun_probability
            crossing_cells = np.flatnonzero(
                (danger < threshold - 1e-12)
                & (overrun_probability > 0.0)
            )
            next_overrun_probability = np.zeros_like(danger, dtype=np.float32)
            if crossing_cells.size:
                next_overrun_probability[crossing_cells] = (
                    1.0 / float(crossing_cells.size)
                )
            shelter_crossing_cells = crossing_cells[
                np.asarray(
                    operational_shelter_cells, dtype=np.bool_
                )[crossing_cells]
            ]
            next_shelter_overrun_probability = np.zeros_like(
                danger, dtype=np.float32
            )
            if shelter_crossing_cells.size:
                next_shelter_overrun_probability[shelter_crossing_cells] = (
                    1.0 / float(shelter_crossing_cells.size)
                )
            overrun_lead_time = np.where(
                danger >= threshold - 1e-12,
                0.0,
                np.where(overrun_probability > 0.0, 0.5, 1.0),
            ).astype(np.float32)
        forecast = np.clip(
            np.nan_to_num(forecast, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        momentum = np.clip(
            np.nan_to_num(momentum, nan=0.0, posinf=1.0, neginf=-1.0),
            -1.0,
            1.0,
        )
        escalation = np.clip(
            np.nan_to_num(escalation, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        uncertainty = np.clip(
            np.nan_to_num(uncertainty, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        overrun_probability = np.clip(
            np.nan_to_num(
                overrun_probability, nan=0.0, posinf=1.0, neginf=0.0
            ),
            0.0,
            1.0,
        )
        crossing_cdf = np.clip(
            np.nan_to_num(crossing_cdf, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        next_overrun_probability = np.clip(
            np.nan_to_num(
                next_overrun_probability,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        next_shelter_overrun_probability = np.clip(
            np.nan_to_num(
                next_shelter_overrun_probability,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )
        overrun_lead_time = np.clip(
            np.nan_to_num(
                overrun_lead_time, nan=1.0, posinf=1.0, neginf=0.0
            ),
            0.0,
            1.0,
        )

        active_hazards = [
            hazard
            for hazard in getattr(hazard_store, "hazardList", {}).values()
            if bool(getattr(hazard, "active", True))
        ]
        source_points = []
        for hazard in active_hazards:
            source = getattr(hazard, "sourceNode", None)
            if source is not None:
                source_points.append(
                    (float(getattr(source, "nodeX", 0.0)), float(getattr(source, "nodeY", 0.0)))
                )
        centers = self._region_centers_m()
        if source_points:
            sources = np.asarray(source_points, dtype=np.float64)
            distances = np.sqrt(
                np.sum((centers[:, None, :] - sources[None, :, :]) ** 2, axis=-1)
            )
            nearest = np.min(distances, axis=1)
            proximity = 1.0 - np.clip(nearest / self.study_area_diagonal, 0.0, 1.0)
        else:
            proximity = np.zeros(self.number_of_cells, dtype=np.float64)

        wind_speed = max(
            0.0,
            float(getattr(hazard_store, "wind_speed_m_per_minute", 0.0)),
        )
        speed_reference = max(float(self.core.maxSpeed), 1e-6)
        wind_speed_fraction = float(
            np.clip(wind_speed / (wind_speed + speed_reference), 0.0, 1.0)
        )
        if wind_speed <= 0.0:
            wind_east = 0.0
            wind_north = 0.0
        else:
            angle = np.deg2rad(
                float(getattr(hazard_store, "wind_direction_degrees", 0.0))
            )
            wind_east = float(wind_speed_fraction * np.cos(angle))
            wind_north = float(wind_speed_fraction * np.sin(angle))
        spread_values = [
            np.clip(float(getattr(hazard, "spreadRate", 0.0)), 0.0, 1.0)
            for hazard in active_hazards
        ]
        spread_fraction = float(np.mean(spread_values)) if spread_values else 0.0
        return (
            forecast.astype(np.float32),
            momentum.astype(np.float32),
            escalation.astype(np.float32),
            uncertainty.astype(np.float32),
            overrun_probability.astype(np.float32),
            next_overrun_probability.astype(np.float32),
            next_shelter_overrun_probability.astype(np.float32),
            overrun_lead_time.astype(np.float32),
            crossing_cdf.astype(np.float32),
            proximity.astype(np.float32),
            wind_speed_fraction,
            wind_east,
            wind_north,
            spread_fraction,
        )

    def _network_or_euclidean_distance(self, start_node, target_node) -> float:
        if start_node is None or target_node is None:
            return float("inf")
        network_distance = getattr(
            getattr(self.core, "mapDS", None),
            "networkDistanceToTarget",
            None,
        )
        if callable(network_distance):
            try:
                distance = float(network_distance(start_node, target_node))
                if np.isfinite(distance):
                    return max(0.0, distance)
            except Exception:
                pass
        return float(
            np.hypot(
                float(getattr(start_node, "nodeX", 0.0))
                - float(getattr(target_node, "nodeX", 0.0)),
                float(getattr(start_node, "nodeY", 0.0))
                - float(getattr(target_node, "nodeY", 0.0)),
            )
        )

    def _road_disruption_layer(
        self,
        public: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map public closure warnings and observable closures to cells.

        A single global warning share cannot tell a spatial policy which side
        of the network is at risk.  Incidence is normalized by the number of
        road-edge endpoints in each cell, preserving the complete public map
        without revealing any unannounced future closure.
        """
        incident = np.zeros(self.number_of_cells, dtype=np.float32)
        warned = np.zeros(self.number_of_cells, dtype=np.float32)
        closed = np.zeros(self.number_of_cells, dtype=np.float32)
        warned_ids = {
            str(value) for value in public.get("closure_warning_edges", ())
        }
        closed_ids = {
            str(value)
            for value in getattr(self.core.mapDS, "closed_edges", {}).keys()
        }
        for key, edge in getattr(
            self.core.mapDS, "edgeListByLocalID", {}
        ).items():
            edge_id = str(getattr(edge, "edgeID", key))
            endpoint_cells = set()
            for node in (
                getattr(edge, "startNode", None),
                getattr(edge, "endNode", None),
            ):
                index = self._node_cell_index(node)
                if index is not None:
                    endpoint_cells.add(int(index))
            for index in endpoint_cells:
                incident[index] += 1.0
                if edge_id in warned_ids or str(key) in warned_ids:
                    warned[index] += 1.0
                if edge_id in closed_ids or str(key) in closed_ids:
                    closed[index] += 1.0
        warned = np.divide(
            warned,
            incident,
            out=np.zeros_like(warned),
            where=incident > 0.0,
        )
        closed = np.divide(
            closed,
            incident,
            out=np.zeros_like(closed),
            where=incident > 0.0,
        )
        return np.clip(warned, 0.0, 1.0), np.clip(closed, 0.0, 1.0)

    def _candidate_operational_features(
        self,
        danger: np.ndarray,
        forecast_danger: np.ndarray,
        forecast_escalation: np.ndarray,
        forecast_uncertainty: np.ndarray,
        forecast_overrun_probability: np.ndarray,
        forecast_next_overrun_probability: np.ndarray,
        forecast_overrun_lead_time_fraction: np.ndarray,
        reevacuation_response: np.ndarray,
        closure_warning: np.ndarray,
        closed_road: np.ndarray,
        records: Sequence[tuple[str, int, float, float, float]],
        nodes: Sequence[Optional[Any]],
    ) -> tuple[np.ndarray, ...]:
        """Compute only directly observable site geometry and hazard context.

        Candidate-specific pedestrian route scans used to expose an engineered
        ranking of every action here. Those privileged labels now belong only
        to NMCC's training-time simulator branches, never to observations or
        benchmark inputs.
        """
        action_count = len(records)
        nearest_shelter = np.ones(action_count, dtype=np.float32)
        candidate_danger = np.zeros(action_count, dtype=np.float32)
        candidate_forecast = np.zeros(action_count, dtype=np.float32)
        candidate_escalation = np.zeros(action_count, dtype=np.float32)
        candidate_uncertainty = np.zeros(action_count, dtype=np.float32)
        candidate_overrun = np.zeros(action_count, dtype=np.float32)
        candidate_next_overrun = np.zeros(action_count, dtype=np.float32)
        candidate_overrun_lead = np.ones(action_count, dtype=np.float32)
        safety_margin = np.ones(action_count, dtype=np.float32)
        candidate_warning = np.zeros(action_count, dtype=np.float32)
        candidate_closed = np.zeros(action_count, dtype=np.float32)
        if action_count == 0:
            return (
                nearest_shelter,
                candidate_danger,
                candidate_forecast,
                candidate_escalation,
                candidate_uncertainty,
                candidate_overrun,
                candidate_next_overrun,
                candidate_overrun_lead,
                reevacuation_response,
                safety_margin,
                candidate_warning,
                candidate_closed,
            )

        available_capacity = getattr(
            self.core.shelterDS, "availableCapacity", None
        )
        shelters = []
        for shelter in getattr(self.core.shelterDS, "shelterList", {}).values():
            if int(getattr(shelter, "status", 0)) != 0:
                continue
            available = (
                float(available_capacity(shelter))
                if callable(available_capacity)
                else float(getattr(shelter, "shelterCap", 0.0))
                - float(getattr(shelter, "shelterFlow", 0.0))
            )
            if available > 0.0:
                shelters.append(shelter)
        active_sources = [
            getattr(hazard, "sourceNode", None)
            for hazard in getattr(getattr(self.core, "hazardDS", None), "hazardList", {}).values()
            if bool(getattr(hazard, "active", True))
            and getattr(hazard, "sourceNode", None) is not None
        ]
        for action, (record, candidate_node) in enumerate(zip(records, nodes)):
            candidate_id, host_cell, _, _, _ = record
            if candidate_id == "__defer__":
                nearest_shelter[action] = 0.0
                safety_margin[action] = 0.0
                continue
            candidate_danger[action] = float(danger[host_cell])
            candidate_forecast[action] = float(forecast_danger[host_cell])
            candidate_escalation[action] = float(
                forecast_escalation[host_cell]
            )
            candidate_uncertainty[action] = float(
                forecast_uncertainty[host_cell]
            )
            candidate_overrun[action] = float(
                forecast_overrun_probability[host_cell]
            )
            candidate_next_overrun[action] = float(
                forecast_next_overrun_probability[host_cell]
            )
            candidate_overrun_lead[action] = float(
                forecast_overrun_lead_time_fraction[host_cell]
            )
            candidate_warning[action] = float(closure_warning[host_cell])
            candidate_closed[action] = float(closed_road[host_cell])
            if candidate_node is None:
                continue
            if shelters:
                distance = min(
                    self._network_or_euclidean_distance(
                        candidate_node,
                        getattr(shelter, "nodeMapped", None),
                    )
                    for shelter in shelters
                )
                nearest_shelter[action] = float(
                    np.clip(distance / self.study_area_diagonal, 0.0, 1.0)
                )
            if active_sources:
                source_distance = min(
                    float(
                        np.hypot(
                            float(candidate_node.nodeX) - float(source.nodeX),
                            float(candidate_node.nodeY) - float(source.nodeY),
                        )
                    )
                    for source in active_sources
                )
                safety_margin[action] = float(
                    np.clip(source_distance / self.study_area_diagonal, 0.0, 1.0)
                )

        return (
            nearest_shelter,
            candidate_danger,
            candidate_forecast,
            candidate_escalation,
            candidate_uncertainty,
            candidate_overrun,
            candidate_next_overrun,
            candidate_overrun_lead,
            reevacuation_response,
            safety_margin,
            candidate_warning,
            candidate_closed,
        )

    def build(
        self,
        *,
        decision_index: int,
        simulation_time: int,
        remaining_deployments: int,
    ) -> RegionalObservation:
        runtime_shape = authoritative_grid_shape(self.core)
        if runtime_shape != (self.nx, self.ny):
            raise RuntimeError(
                "RegionalObservationBuilder grid changed after construction: "
                f"built={(self.nx, self.ny)}, runtime={runtime_shape}"
            )
        (
            active,
            mean_speed,
            congestion_speed_ratio,
            assigned_population,
            unassigned_population,
            outgoing_assignment_flow,
            incoming_assignment_flow,
            mean_route_time,
            slow_tail_population,
            slow_tail_time,
            long_route_share,
            stable_wellness,
            exposed_wellness,
            panicked_wellness,
            route_edge_index,
            route_edge_weight,
        ) = self._pedestrian_layer()
        danger = np.asarray(
            getattr(self.core.cellTracker, "dangerLevelByCell", np.zeros(self.number_of_cells)),
            dtype=np.float32,
        ).reshape(self.number_of_cells)
        danger = np.nan_to_num(danger, nan=0.0, posinf=1.0, neginf=0.0)
        danger = np.clip(danger, 0.0, 1.0)
        (
            remaining_capacity,
            operational_shelter_capacity,
            sheltered_population,
            enroute_reserved_population,
            shelter_utilization,
            deployable_capacity,
            candidate_count,
            preview_nodes,
        ) = self._regional_capacity()
        reservation_arrivals = self._reservation_arrivals_by_shelter_cell()
        (
            forecast_danger,
            danger_momentum,
            forecast_escalation_probability,
            forecast_uncertainty,
            forecast_overrun_probability,
            forecast_next_overrun_probability,
            forecast_next_shelter_overrun_probability,
            forecast_overrun_lead_time_fraction,
            forecast_crossing_cdf_by_step,
            hazard_source_proximity,
            wind_speed_fraction,
            wind_east_fraction,
            wind_north_fraction,
            hazard_spread_fraction,
        ) = self._hazard_layer(
            danger,
            simulation_time,
            operational_shelter_capacity > 0.0,
        )
        front_interval = max(
            1, int(getattr(self.core, "shelterActionInterval", 1))
        )
        front_window = min(
            max(1, int(self.horizon) - int(simulation_time)),
            max(1, int(getattr(self.core, "nmccFrontForecastIntervals", 4)))
            * front_interval,
        )
        susceptible_arrivals = self._susceptible_arrivals_by_cell(
            front_window
        )
        (
            front_crossing_cdf_by_step,
            front_arrival_probability,
            front_lead_time_fraction,
            susceptible_population,
            threatened_population,
            front_window,
        ) = self._front_layer(
            danger,
            simulation_time,
            susceptible_arrivals,
        )
        regime = getattr(self.core, "hazard_regime", None)
        public = getattr(regime, "public", {}) if regime is not None else {}
        closure_warning_by_cell, closed_road_by_cell = (
            self._road_disruption_layer(public)
        )
        self._candidate_records, self._candidate_nodes = self._cell_action_records(
            preview_nodes
        )
        if self.defer_enabled:
            self._candidate_records = (
                ("__defer__", 0, 0.0, 0.5, 0.5),
                *self._candidate_records,
            )
            self._candidate_nodes = (None, *self._candidate_nodes)
        # An en-route cohort can reach the shelter before it fails.  Such a
        # cohort becomes a resident and later reverses the safe-completion
        # stock; only a crossing at or before its ETA is a pure reroute.  Split
        # those outcomes against the full first-passage CDF rather than one
        # mean lead time.
        (
            forecast_reevacuation_population,
            forecast_enroute_reroute_population,
        ) = self._forecast_shelter_transition_burdens(
            sheltered_population,
            enroute_reserved_population,
            reservation_arrivals,
            forecast_crossing_cdf_by_step,
        )
        forecast_safe_capacity = operational_shelter_capacity * (
            1.0 - forecast_overrun_probability
        )
        forecast_reevacuation_fraction = float(
            np.clip(
                forecast_reevacuation_population.sum()
                / float(max(1, self.initial_population)),
                0.0,
                1.0,
            )
        )
        forecast_enroute_reroute_fraction = float(
            np.clip(
                forecast_enroute_reroute_population.sum()
                / float(max(1, self.initial_population)),
                0.0,
                1.0,
            )
        )
        forecast_safe_capacity_fraction = float(
            np.clip(
                forecast_safe_capacity.sum()
                / float(max(1, self.initial_population)),
                0.0,
                1.0,
            )
        )
        candidate_reevacuation_response = self._candidate_reevacuation_response(
            self._candidate_records,
            forecast_reevacuation_population,
            forecast_overrun_lead_time_fraction,
            forecast_overrun_probability,
        )
        shelter_next_mass = forecast_next_shelter_overrun_probability
        next_shelter_overrun_probability = float(
            np.clip(shelter_next_mass.sum(), 0.0, 1.0)
        )
        next_shelter_overrun_lead_time = (
            float(
                np.sum(
                    shelter_next_mass * forecast_overrun_lead_time_fraction
                )
                / max(float(shelter_next_mass.sum()), 1e-12)
            )
            if np.any(shelter_next_mass > 0.0)
            else 1.0
        )
        capacity_coverage = self.deployment_capacity_coverage_fraction
        if float(getattr(self.core, "shelterCapacityToken", 0)) <= 0.0:
            site_capacities = sorted(
                (max(0.0, float(record[2])) for record in self._candidate_records),
                reverse=True,
            )
            capacity_coverage = float(np.clip(
                sum(site_capacities[: self.maximum_deployments])
                / float(max(1, self.initial_population)),
                0.0,
                1.0,
            ))
        (
            candidate_nearest_shelter,
            candidate_current_danger,
            candidate_forecast_danger,
            candidate_forecast_escalation,
            candidate_forecast_uncertainty,
            candidate_forecast_overrun_probability,
            candidate_forecast_next_overrun_probability,
            candidate_forecast_overrun_lead_time_fraction,
            candidate_forecast_reevacuation_response,
            candidate_hazard_safety,
            candidate_closure_warning,
            candidate_closed_road,
        ) = self._candidate_operational_features(
            danger,
            forecast_danger,
            forecast_escalation_probability,
            forecast_uncertainty,
            forecast_overrun_probability,
            forecast_next_overrun_probability,
            forecast_overrun_lead_time_fraction,
            candidate_reevacuation_response,
            closure_warning_by_cell,
            closed_road_by_cell,
            self._candidate_records,
            self._candidate_nodes,
        )
        (
            candidate_threat_capture_fraction,
            candidate_front_approach_fraction,
            candidate_front_arrival_probability,
            candidate_front_lead_time_fraction,
        ) = self._candidate_front_features(
            self._candidate_records,
            self._candidate_nodes,
            front_crossing_cdf_by_step,
            susceptible_arrivals,
            front_lead_time_fraction,
            front_window,
        )

        results = getattr(self.core.pedDS, "result", {})
        arrivals = max(0, int(results.get("arrival", 0)))
        shelter_evacuated = max(0, int(results.get("evacuated", 0)))
        casualties = max(0, int(results.get("casualty", 0)))
        active_population = int(round(float(active.sum())))
        safe_completed = arrivals + shelter_evacuated
        risk_mass = float(np.sum(active * (1.0 + danger)))
        outcome = OutcomeSnapshot(
            safe_completed=safe_completed,
            casualties=casualties,
            shelter_evacuated=shelter_evacuated,
            ordinary_arrivals=arrivals,
            active_population=active_population,
            risk_mass=risk_mass,
        )

        remaining_deployments = max(0, int(remaining_deployments))
        action_mask = self._candidate_action_mask(
            remaining_deployments,
            forecast_danger,
            self._candidate_records,
        )
        interval = max(1, int(getattr(self.core, "shelterActionInterval", 1)))
        # Decisions are taken after a physical transition and never on the
        # terminal transition itself.  Count only epochs that can actually be
        # executed; otherwise the mask can permit one defer too many and leave
        # a capacity token unspent at the horizon.
        terminal_time = int(self.horizon) - 1
        remaining_epochs = (
            0
            if int(simulation_time) > terminal_time
            else ((terminal_time - int(simulation_time)) // interval) + 1
        )
        deadline_slack = int(remaining_epochs - remaining_deployments)
        if self.defer_enabled:
            action_mask = action_mask.copy()
            action_mask[0] = bool(
                remaining_deployments > 0
                and (
                    remaining_deployments < remaining_epochs
                    or not bool(np.any(action_mask[1:]))
                )
            )
        forecast_direction = float(public.get("wind_forecast_direction", 0.0))
        forecast_speed_raw = max(0.0, float(public.get("wind_forecast_speed", 0.0)))
        speed_reference = max(float(self.core.maxSpeed), 1e-6)
        forecast_speed = float(
            np.clip(
                forecast_speed_raw / (forecast_speed_raw + speed_reference),
                0.0,
                1.0,
            )
        )
        forecast_angle = np.deg2rad(forecast_direction)
        edge_total = max(1, len(getattr(self.core.mapDS, "edgeListByLocalID", {})))
        closure_share = float(
            np.clip(
                len(public.get("closure_warning_edges", ())) / edge_total,
                0.0,
                1.0,
            )
        )
        return RegionalObservation(
            decision_index=decision_index,
            simulation_time=simulation_time,
            horizon=self.horizon,
            initial_population=self.initial_population,
            remaining_deployments=remaining_deployments,
            maximum_deployments=self.maximum_deployments,
            maximum_speed=float(self.core.maxSpeed),
            active_by_cell=active,
            mean_speed_by_cell=mean_speed,
            congestion_speed_ratio_by_cell=congestion_speed_ratio,
            danger_by_cell=danger,
            remaining_capacity_by_cell=remaining_capacity,
            deployable_capacity_by_cell=deployable_capacity,
            candidate_count_by_cell=candidate_count,
            action_mask=action_mask,
            outcome=outcome,
            candidate_osm_node_ids=tuple(record[0] for record in self._candidate_records),
            candidate_cell_indices=np.asarray(
                [record[1] for record in self._candidate_records],
                dtype=np.int64,
            ),
            candidate_capacities=np.asarray(
                [record[2] for record in self._candidate_records],
                dtype=np.float32,
            ),
            candidate_east_positions=np.asarray(
                [record[3] for record in self._candidate_records],
                dtype=np.float32,
            ),
            candidate_north_positions=np.asarray(
                [record[4] for record in self._candidate_records],
                dtype=np.float32,
            ),
            operational_shelter_capacity_by_cell=(
                operational_shelter_capacity
            ),
            shelter_utilization_by_cell=shelter_utilization,
            sheltered_population_by_cell=sheltered_population,
            enroute_reserved_population_by_cell=enroute_reserved_population,
            forecast_overrun_probability_by_cell=(
                forecast_overrun_probability
            ),
            forecast_next_shelter_overrun_probability_by_cell=(
                forecast_next_shelter_overrun_probability
            ),
            forecast_overrun_lead_time_fraction_by_cell=(
                forecast_overrun_lead_time_fraction
            ),
            forecast_reevacuation_population_by_cell=(
                forecast_reevacuation_population
            ),
            forecast_enroute_reroute_population_by_cell=(
                forecast_enroute_reroute_population
            ),
            network_node_count_by_cell=self.network_node_count_by_cell,
            mean_route_time_by_cell=mean_route_time,
            slow_tail_population_by_cell=slow_tail_population,
            slow_tail_time_by_cell=slow_tail_time,
            assigned_population_by_cell=assigned_population,
            unassigned_population_by_cell=unassigned_population,
            outgoing_assignment_flow_by_cell=outgoing_assignment_flow,
            incoming_assignment_flow_by_cell=incoming_assignment_flow,
            long_route_share_by_cell=long_route_share,
            stable_wellness_by_cell=stable_wellness,
            exposed_wellness_by_cell=exposed_wellness,
            panicked_wellness_by_cell=panicked_wellness,
            forecast_danger_by_cell=forecast_danger,
            danger_momentum_by_cell=danger_momentum,
            forecast_escalation_probability_by_cell=(
                forecast_escalation_probability
            ),
            forecast_uncertainty_by_cell=forecast_uncertainty,
            hazard_source_proximity_by_cell=hazard_source_proximity,
            closure_warning_by_cell=closure_warning_by_cell,
            closed_road_by_cell=closed_road_by_cell,
            region_east_positions=self.region_east_positions,
            region_north_positions=self.region_north_positions,
            region_area_fractions=self.region_area_fractions,
            spatial_edge_index=self.spatial_edge_index,
            route_edge_index=route_edge_index,
            route_edge_weight=route_edge_weight,
            candidate_nearest_shelter_distances=candidate_nearest_shelter,
            candidate_current_danger=candidate_current_danger,
            candidate_forecast_danger=candidate_forecast_danger,
            candidate_forecast_escalation=candidate_forecast_escalation,
            candidate_forecast_uncertainty=candidate_forecast_uncertainty,
            candidate_forecast_overrun_probability=(
                candidate_forecast_overrun_probability
            ),
            candidate_forecast_next_overrun_probability=(
                candidate_forecast_next_overrun_probability
            ),
            candidate_forecast_overrun_lead_time_fraction=(
                candidate_forecast_overrun_lead_time_fraction
            ),
            candidate_forecast_reevacuation_response=(
                candidate_forecast_reevacuation_response
            ),
            candidate_hazard_safety_margin=candidate_hazard_safety,
            candidate_closure_warning=candidate_closure_warning,
            candidate_closed_road=candidate_closed_road,
            susceptible_population_by_cell=susceptible_population,
            threatened_population_by_cell=threatened_population,
            front_arrival_probability_by_cell=front_arrival_probability,
            front_lead_time_fraction_by_cell=front_lead_time_fraction,
            candidate_threat_capture_fraction=(
                candidate_threat_capture_fraction
            ),
            candidate_front_approach_fraction=(
                candidate_front_approach_fraction
            ),
            candidate_front_arrival_probability=(
                candidate_front_arrival_probability
            ),
            candidate_front_lead_time_fraction=(
                candidate_front_lead_time_fraction
            ),
            network_load_share=self.network_load_share,
            time_step_minutes=float(getattr(self.core, "timeStepMinutes", 1.0)),
            wind_speed_fraction=wind_speed_fraction,
            wind_east_direction_fraction=wind_east_fraction,
            wind_north_direction_fraction=wind_north_fraction,
            hazard_spread_fraction=hazard_spread_fraction,
            population_network_density_fraction=(
                self.population_network_density_fraction
            ),
            hazard_instance_fraction=self.hazard_instance_fraction,
            configured_panic_fraction=self.configured_panic_fraction,
            deployment_capacity_coverage_fraction=(
                capacity_coverage
            ),
            forecast_reevacuation_population_fraction=(
                forecast_reevacuation_fraction
            ),
            forecast_enroute_reroute_population_fraction=(
                forecast_enroute_reroute_fraction
            ),
            forecast_safe_capacity_coverage_fraction=(
                forecast_safe_capacity_fraction
            ),
            next_shelter_overrun_probability=(
                next_shelter_overrun_probability
            ),
            next_shelter_overrun_lead_time_fraction=(
                next_shelter_overrun_lead_time
            ),
            threatened_population_fraction=float(
                np.clip(
                    threatened_population.sum()
                    / float(max(1, self.initial_population)),
                    0.0,
                    1.0,
                )
            ),
            forecast_ensemble_mean=forecast_danger,
            forecast_ensemble_dispersion=forecast_uncertainty,
            forecast_escalation=forecast_escalation_probability,
            wind_forecast_cos=float(np.cos(forecast_angle)),
            wind_forecast_sin=float(np.sin(forecast_angle)),
            wind_forecast_speed=forecast_speed,
            wind_warning=float(bool(public.get("wind_warning", False))),
            closure_warning_share=closure_share,
            remaining_tokens=remaining_deployments,
            remaining_epochs=remaining_epochs,
            deadline_slack=deadline_slack,
            regime_features_enabled=bool(regime is not None and regime.enabled),
            defer_action_index=0 if self.defer_enabled else -1,
        )


class RegionalShelterExecutor:
    """Install into the regional cell selected by a dynamic strategy.

    The action names a cell; the exact building installed within it is
    resolved by the shared deterministic lower layer
    (``ShelterDatabase.newShelter`` / ``_candidate_index``: maximum remaining
    capacity, tie-broken by OSM identifier) -- the identical rule used for
    every heuristic benchmark.  This executor asserts, fail-closed, that the
    building actually installed matches the one the observation predicted
    for this cell, so a silent drift between the observation's preview and
    the lower layer's live resolution can never pass unnoticed.
    """

    def __init__(self, core):
        self.core = core
        self.nx, self.ny = authoritative_grid_shape(core)

    def execute(
        self,
        observation: RegionalObservation,
        decision: PolicyDecision,
    ) -> RegionalActionReceipt:
        action = int(decision.action_index)
        if action < 0 or action >= observation.number_of_actions:
            raise ValueError(f"Candidate action {action} is outside the shared action space")
        if not bool(observation.action_mask[action]):
            raise ValueError(f"Candidate action {action} is infeasible under the shared action mask")
        expected_actions = self.nx * self.ny + int(
            observation.defer_action_index >= 0
        )
        if observation.number_of_actions != expected_actions:
            raise RuntimeError(
                "Observation action dimensionality does not match CellTracker: "
                f"actions={observation.number_of_actions}, grid={(self.nx, self.ny)}"
            )
        if action == int(observation.defer_action_index):
            return RegionalActionReceipt(
                observation_id=(
                    observation.decision_index,
                    observation.simulation_time,
                ),
                requested_candidate=action,
                executed_candidate=action,
                requested_cell=-1,
                executed_cell=-1,
                shelter_id=-1,
                candidate_osm_node_id="__defer__",
                candidate_x_m=0.0,
                candidate_y_m=0.0,
                candidate_cell_i=-1,
                candidate_cell_j=-1,
                capacity_added=0.0,
                rerouted_population=0,
                installed=False,
            )

        cell_index = int(observation.candidate_cell_indices[action])
        cell = divmod(cell_index, self.ny)
        predicted_osm_id = str(observation.candidate_osm_node_ids[action])
        shelter_id = self.core.shelterDS.newShelter({"cell": cell}, self.core.cellTracker)
        if shelter_id is None:
            raise RuntimeError(
                f"Feasible cell action {action} failed during shared-site-rule execution"
            )
        shelter = self.core.shelterDS.shelterList[shelter_id]
        installed_osm_id = str(getattr(shelter.nodeMapped, "OSMID", ""))
        if not predicted_osm_id.startswith("empty-cell-") and installed_osm_id != predicted_osm_id:
            raise RuntimeError(
                "Shared deterministic site rule diverged from the observation's "
                f"preview for cell {cell_index}: predicted {predicted_osm_id!r}, "
                f"installed {installed_osm_id!r}"
            )
        if hasattr(self.core.cellTracker, "addShelter"):
            self.core.cellTracker.addShelter(cell, shelter)

        rerouted = int(
            self.core.pedDS.reroute_active_to_nearest_shelter(
                new_shelter=shelter
            )
        )
        if (
            int(observation.remaining_deployments) == 1
            and bool(
                getattr(
                    self.core,
                    "requirePopulationMatchedShelterCapacity",
                    False,
                )
            )
        ):
            unassigned = int(
                self.core.pedDS.unassigned_rational_population()
            )
            self.core.finalInstallationUnassignedRationalPopulation = int(
                unassigned
            )
            self.core.shelterDS.assertPopulationMatchedAssignment(
                unassigned,
                self.core.pedVol,
                network_disruption_active=bool(
                    getattr(self.core.mapDS, "closed_edges", {})
                ),
                context="Final scheduled shelter",
            )

        return RegionalActionReceipt(
            observation_id=(observation.decision_index, observation.simulation_time),
            requested_candidate=action,
            executed_candidate=action,
            requested_cell=cell_index,
            executed_cell=cell_index,
            shelter_id=int(shelter_id),
            candidate_osm_node_id=str(getattr(shelter.nodeMapped, "OSMID", "")),
            candidate_x_m=float(shelter.nodeMapped.nodeX),
            candidate_y_m=float(shelter.nodeMapped.nodeY),
            candidate_cell_i=int(cell[0]),
            candidate_cell_j=int(cell[1]),
            capacity_added=max(0.0, float(getattr(shelter, "shelterCap", 0.0))),
            rerouted_population=max(0, rerouted),
        )
