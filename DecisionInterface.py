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
    "panicked_population_fraction",
    "remaining_evacuation_debt_fraction",
)

HAZARD_FEATURE_NAMES = (
    "danger",
    "danger_spread_momentum",
    "forecast_expected_danger",
    "forecast_escalation_probability",
    "forecast_uncertainty",
)

INFRA_FEATURE_NAMES = (
    "remaining_shelter_capacity_fraction",
    "shelter_utilization_fraction",
    "deployable_capacity_fraction",
    "road_node_share",
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
    "slow_tail_population_fraction",
    "slow_tail_time_fraction",
)

# Causal ordering is explicit: every momentum feature is computed only from
# the current dashboard observation and an earlier cached observation. Positive
# values indicate operational improvement; worsening forecast exposure is
# therefore negative.
MOMENTUM_FEATURE_NAMES = (
    "active_population_clearance_velocity",
    "congestion_delay_reduction_velocity",
    "route_time_reduction_velocity",
    "forecast_exposure_reduction_velocity",
)

CANDIDATE_FEATURE_NAMES = (
    "candidate_capacity_fraction",
    "candidate_east_position_fraction",
    "candidate_north_position_fraction",
    "nearest_open_shelter_distance_fraction",
    "exact_assignment_gain_fraction",
)

BENCHMARK_POLICY_CONTRACTS = {
    "objective_greedy": {
        "role": "primary_myopic_comparator",
        "label": "complete-ledger objective-matched myopic policy",
        "score": (
            "positive evaluation score of capacity-capped same-state summed "
            "plain remaining time plus forecast-danger-weighted remaining time "
            "for every person not yet safe"
        ),
        "information": (
            "exact current pedestrian routes, speeds, shelter reservations, "
            "candidate capacity, and the public one-interval hazard forecast; "
            "panicked people and casualties retain unresolved debt; no physical "
            "rollout, recurrent history, or continuation value"
        ),
        "timing": "dynamic",
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
        "label": "uniform random feasible candidate site",
        "score": "not applicable",
        "timing": "dynamic",
    },
    "initial_only": {
        "role": "value_of_feedback",
        "label": "initial-state objective-greedy precommitment",
        "score": (
            "sequential immediate evacuation-debt reduction on a frozen "
            "time-zero planning state"
        ),
        "timing": "fixed schedule at the common online decision epochs",
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
    congestion_speed_ratio_by_cell: Optional[np.ndarray] = None
    candidate_osm_node_ids: Optional[tuple[str, ...]] = None
    candidate_cell_indices: Optional[np.ndarray] = None
    candidate_capacities: Optional[np.ndarray] = None
    candidate_east_positions: Optional[np.ndarray] = None
    candidate_north_positions: Optional[np.ndarray] = None
    shelter_utilization_by_cell: Optional[np.ndarray] = None
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
    region_east_positions: Optional[np.ndarray] = None
    region_north_positions: Optional[np.ndarray] = None
    region_area_fractions: Optional[np.ndarray] = None
    spatial_edge_index: Optional[np.ndarray] = None
    route_edge_index: Optional[np.ndarray] = None
    route_edge_weight: Optional[np.ndarray] = None
    candidate_nearest_shelter_distances: Optional[np.ndarray] = None
    candidate_forecast_danger: Optional[np.ndarray] = None
    candidate_hazard_safety_margin: Optional[np.ndarray] = None
    candidate_reroutable_population: Optional[np.ndarray] = None
    candidate_risk_time_reduction: Optional[np.ndarray] = None
    remaining_evacuation_debt_by_cell: Optional[np.ndarray] = None
    candidate_assignment_gain: Optional[np.ndarray] = None
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

        def bounded_cell_vector(name: str, default: float = 0.0) -> np.ndarray:
            value = getattr(self, name)
            if value is None:
                result = np.full(cell_count, default, dtype=np.float32)
            else:
                result = _finite_vector(name, value, cell_count, nonnegative=True)
            return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

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
            "hazard_source_proximity_by_cell",
        ):
            object.__setattr__(self, name, bounded_cell_vector(name))
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
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
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
        candidate_reroutable = bounded_candidate_vector(
            "candidate_reroutable_population"
        )
        candidate_risk_reduction = bounded_candidate_vector(
            "candidate_risk_time_reduction"
        )
        candidate_assignment_gain = _finite_vector(
            "candidate_assignment_gain",
            (
                np.zeros(action_count, dtype=np.float32)
                if self.candidate_assignment_gain is None
                else self.candidate_assignment_gain
            ),
            action_count,
        )
        if np.any(np.abs(candidate_assignment_gain) > 2.0 + 1e-6):
            raise ValueError(
                "candidate_assignment_gain must lie in the complete-ledger "
                "normalized range [-2, 2]"
            )
        candidate_assignment_gain = np.clip(
            candidate_assignment_gain, -2.0, 2.0
        ).astype(np.float32, copy=False)
        if self.remaining_deployments <= 0 and mask.any():
            raise ValueError("action_mask cannot allow deployment after the budget is exhausted")
        if np.any(mask & (candidate_capacities <= 0.0)):
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
            "candidate_assignment_gain",
            candidate_assignment_gain,
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
                self.panicked_wellness_by_cell,
                np.clip(
                    self.remaining_evacuation_debt_by_cell,
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
            ),
            axis=-1,
        )
        infrastructure_features = np.stack(
            (
                np.clip(self.remaining_capacity_by_cell / population_scale, 0.0, 1.0),
                self.shelter_utilization_by_cell,
                np.clip(self.deployable_capacity_by_cell / population_scale, 0.0, 1.0),
                np.clip(self.network_node_count_by_cell / road_node_total, 0.0, 1.0),
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
            ),
            dtype=np.float32,
        )
        if not np.isfinite(cell_features).all() or not np.isfinite(global_features).all():
            raise ValueError("Policy features contain non-finite values")
        return cell_features, global_features

    def candidate_features(self) -> np.ndarray:
        """Return one plainly interpretable feature vector per candidate site."""
        population_scale = float(max(1, self.initial_population))
        features = np.stack(
            (
                np.clip(self.candidate_capacities / population_scale, 0.0, 1.0),
                self.candidate_east_positions,
                self.candidate_north_positions,
                self.candidate_nearest_shelter_distances,
                self.candidate_assignment_gain,
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


class RegionalPolicy(Protocol):
    def select(self, observation: RegionalObservation, *, deterministic: bool) -> PolicyDecision:
        ...


class ActivePopulationHeuristic:
    """Benchmark: choose a site in the region with most active evacuees."""

    name = "heuristic"

    def select(self, observation: RegionalObservation, *, deterministic: bool = True) -> PolicyDecision:
        del deterministic
        feasible = np.flatnonzero(observation.action_mask)
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
        feasible = np.flatnonzero(observation.action_mask)
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
    """Choose the largest exact *immediate* gain under the RL objective.

    The bridge computes one value for every feasible action with the shared
    zero-horizon assignment/route evaluator.  Keeping selection in this small
    class makes the benchmark rule explicit while leaving physical route and
    capacity accounting in the same authoritative implementation used by the
    reward and NMCC.  This policy has no temporal memory or continuation value.
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
        feasible = np.flatnonzero(observation.action_mask)
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
        feasible = np.flatnonzero(observation.action_mask)
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
        if observation.number_of_cells != int(self.cell_centers.shape[0]):
            raise ValueError(
                "cell_centers and RegionalObservation must contain the same number of cells"
            )
        feasible = np.flatnonzero(observation.action_mask)
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
        feasible = np.flatnonzero(observation.action_mask)
        if feasible.size == 0:
            raise RuntimeError("The random policy received an observation with no feasible action")
        action = int(self.rng.choice(feasible))
        return PolicyDecision(action_index=action, strategy=self.name)


class RegionalObservationBuilder:
    """Create one authoritative operational observation from domain stores."""

    def __init__(self, core, *, initial_population: int, horizon: int, maximum_deployments: int):
        self.core = core
        self.initial_population = int(initial_population)
        self.horizon = int(horizon)
        self.maximum_deployments = max(0, int(maximum_deployments))
        self.nx, self.ny = authoritative_grid_shape(core)
        self.number_of_cells = self.nx * self.ny
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
        return self.number_of_cells

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        remaining = np.zeros(self.number_of_cells, dtype=np.float32)
        installed = np.zeros(self.number_of_cells, dtype=np.float32)
        fulfilled = np.zeros(self.number_of_cells, dtype=np.float32)
        deployable = np.zeros(self.number_of_cells, dtype=np.float32)
        candidate_count = np.zeros(self.number_of_cells, dtype=np.float32)
        preview_nodes: list = [None] * self.number_of_cells
        shelter_grid = getattr(self.core.shelterDS, "shelterByCell", None)
        candidate_grid = getattr(self.core.shelterDS, "shelterCanByCell", None)
        if shelter_grid is None or candidate_grid is None:
            return remaining, fulfilled, deployable, candidate_count, preview_nodes
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
                    capacity = max(0.0, float(getattr(shelter, "shelterCap", 0.0)))
                    flow = max(0.0, float(getattr(shelter, "shelterFlow", 0.0)))
                    reserved = max(
                        0.0,
                        float(reserved_population(shelter))
                        if callable(reserved_population)
                        else 0.0,
                    )
                    installed[index] += capacity
                    fulfilled[index] += min(flow + reserved, capacity)
                    if int(getattr(shelter, "status", 0)) == 0:
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
        utilization = np.divide(
            fulfilled,
            installed,
            out=np.zeros_like(fulfilled),
            where=installed > 0.0,
        )
        return remaining, utilization, deployable, candidate_count, preview_nodes

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

    def _hazard_layer(
        self,
        danger: np.ndarray,
    ) -> tuple[
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
        """Return a momentum-based forecast distribution and wind vector."""
        hazard_store = getattr(self.core, "hazardDS", None)
        horizon = max(1, int(getattr(self.core, "shelterActionInterval", 1)))
        distribution_method = getattr(
            hazard_store, "forecast_distribution_by_cell", None
        )
        if callable(distribution_method):
            distribution = distribution_method(horizon)
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
        else:
            forecast_method = getattr(
                hazard_store, "forecast_danger_by_cell", None
            )
            forecast = (
                np.asarray(forecast_method(horizon), dtype=np.float32).reshape(
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

    def _candidate_operational_features(
        self,
        danger: np.ndarray,
        forecast_danger: np.ndarray,
        records: Sequence[tuple[str, int, float, float, float]],
        nodes: Sequence[Optional[Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute only directly observable site geometry and hazard context.

        Candidate-specific pedestrian route scans used to expose an engineered
        ranking of every action here. Those privileged labels now belong only
        to NMCC's training-time simulator branches, never to observations or
        benchmark inputs.
        """
        action_count = len(records)
        nearest_shelter = np.ones(action_count, dtype=np.float32)
        candidate_forecast = np.zeros(action_count, dtype=np.float32)
        safety_margin = np.ones(action_count, dtype=np.float32)
        if action_count == 0:
            return (
                nearest_shelter,
                candidate_forecast,
                safety_margin,
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
            _, host_cell, _, _, _ = record
            candidate_forecast[action] = float(forecast_danger[host_cell])
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
            candidate_forecast,
            safety_margin,
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
            forecast_danger,
            danger_momentum,
            forecast_escalation_probability,
            forecast_uncertainty,
            hazard_source_proximity,
            wind_speed_fraction,
            wind_east_fraction,
            wind_north_fraction,
            hazard_spread_fraction,
        ) = self._hazard_layer(danger)
        (
            remaining_capacity,
            shelter_utilization,
            deployable_capacity,
            candidate_count,
            preview_nodes,
        ) = self._regional_capacity()
        self._candidate_records, self._candidate_nodes = self._cell_action_records(
            preview_nodes
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
            candidate_forecast_danger,
            candidate_hazard_safety,
        ) = self._candidate_operational_features(
            danger,
            forecast_danger,
            self._candidate_records,
            self._candidate_nodes,
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
            shelter_utilization_by_cell=shelter_utilization,
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
            region_east_positions=self.region_east_positions,
            region_north_positions=self.region_north_positions,
            region_area_fractions=self.region_area_fractions,
            spatial_edge_index=self.spatial_edge_index,
            route_edge_index=route_edge_index,
            route_edge_weight=route_edge_weight,
            candidate_nearest_shelter_distances=candidate_nearest_shelter,
            candidate_forecast_danger=candidate_forecast_danger,
            candidate_hazard_safety_margin=candidate_hazard_safety,
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
        if observation.number_of_actions != self.nx * self.ny:
            raise RuntimeError(
                "Observation action dimensionality does not match CellTracker: "
                f"actions={observation.number_of_actions}, grid={(self.nx, self.ny)}"
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
            if unassigned != 0:
                raise RuntimeError(
                    "Final scheduled shelter left capacity-eligible rational "
                    f"pedestrians unassigned: {unassigned}"
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
