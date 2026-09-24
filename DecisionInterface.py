#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Administrator-facing decision contract for dynamic shelter deployment.

At each decision epoch the policy sees the same quantities available on an
emergency-operations dashboard: regional evacuee counts, mobility delay,
hazard severity, available shelter capacity, remaining time and deployment
budget, plus a map-linked table of one action slot per regional cell, its
capacity, and its relative location.  An action names one regional cell to
prioritize; the specific building installed within that cell is resolved by
a shared, deterministic lower layer (``ShelterDatabase._candidate_index``:
maximum remaining capacity, tie-broken by OSM identifier) that is identical
for the RL policy and every heuristic benchmark.  This keeps the upper-level
decision -- which cell to prioritize -- as the only quantity that differs
across policies, so a policy comparison isolates cell-prioritization
behavior rather than building-level tie-breaking.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import numpy as np


PED_FEATURE_NAMES = (
    "active_population_fraction",
    "mobility_delay_fraction",
    "mean_route_time_fraction",
    "long_route_population_fraction",
    "stable_wellness_fraction",
    "exposed_wellness_fraction",
    "panicked_wellness_fraction",
)

HAZARD_FEATURE_NAMES = (
    "danger",
    "forecast_danger",
    "hazard_source_proximity",
)

INFRA_FEATURE_NAMES = (
    "remaining_shelter_capacity_fraction",
    "shelter_utilization_fraction",
    "deployable_capacity_fraction",
    "candidate_availability_fraction",
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
    "remaining_deployments_fraction",
    "network_load_share",
    "wind_speed_fraction",
    "wind_east_direction_fraction",
    "wind_north_direction_fraction",
    "hazard_spread_fraction",
    "population_network_density_fraction",
    "hazard_instance_fraction",
    "configured_panic_fraction",
    "deployment_capacity_coverage_fraction",
)

# Causal ordering is explicit: every momentum feature is computed only from
# the current dashboard observation and an earlier cached observation. Positive
# values indicate operational improvement except casualty incidence, whose
# adverse direction is intentionally kept explicit for interpretability.
MOMENTUM_FEATURE_NAMES = (
    "safe_completion_velocity",
    "casualty_incidence_velocity",
    "active_population_clearance_velocity",
    "hazard_exposure_reduction_velocity",
    "route_time_reduction_velocity",
    "long_route_population_reduction_velocity",
    "forecast_risk_reduction_velocity",
    "network_load_reduction_velocity",
    "time_since_last_deployment_fraction",
)

CANDIDATE_FEATURE_NAMES = (
    "candidate_capacity_fraction",
    "candidate_east_position_fraction",
    "candidate_north_position_fraction",
    "nearest_open_shelter_distance_fraction",
    "candidate_forecast_danger",
    "candidate_hazard_safety_margin",
    "reroutable_population_fraction",
    "risk_time_reduction_fraction",
)

BENCHMARK_POLICY_CONTRACTS = {
    "risk_reduction": {
        "label": "maximum capacity-capped reduction in future risk time",
        "score": "risk_time_reduction_fraction_at_candidate_i",
        "timing": "dynamic",
    },
    "route_saving": {
        "label": "legacy alias of maximum future-risk-time reduction",
        "score": "risk_time_reduction_fraction_at_candidate_i",
        "timing": "dynamic",
    },
    "heuristic": {
        "label": "maximum active population",
        "score": "active_population_at_candidate_i",
        "timing": "dynamic",
    },
    "hazard_weighted": {
        "label": "hazard-exposure-weighted demand",
        "score": "active_population_at_candidate_i * (1 + danger_at_candidate_i)",
        "timing": "dynamic",
    },
    "accessibility_deficit": {
        "label": "maximum accessibility deficit",
        "score": (
            "max(active_population_i - remaining_capacity_i, 0) * "
            "(1 + nearest_usable_shelter_distance_i / study_area_diagonal)"
        ),
        "distance": "Euclidean distance between projected-metre cell centroids",
        "timing": "dynamic",
    },
    "random": {
        "label": "uniform random feasible candidate site",
        "score": "not applicable",
        "timing": "dynamic",
    },
    "initial_only": {
        "label": "static initial shelters only",
        "score": "no post-initial deployment",
        "timing": "time zero",
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
    candidate_osm_node_ids: Optional[tuple[str, ...]] = None
    candidate_cell_indices: Optional[np.ndarray] = None
    candidate_capacities: Optional[np.ndarray] = None
    candidate_east_positions: Optional[np.ndarray] = None
    candidate_north_positions: Optional[np.ndarray] = None
    shelter_utilization_by_cell: Optional[np.ndarray] = None
    network_node_count_by_cell: Optional[np.ndarray] = None
    mean_route_time_by_cell: Optional[np.ndarray] = None
    long_route_share_by_cell: Optional[np.ndarray] = None
    stable_wellness_by_cell: Optional[np.ndarray] = None
    exposed_wellness_by_cell: Optional[np.ndarray] = None
    panicked_wellness_by_cell: Optional[np.ndarray] = None
    forecast_danger_by_cell: Optional[np.ndarray] = None
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
    network_load_share: float = 0.0
    time_step_minutes: float = 1.0
    wind_speed_fraction: float = 0.0
    wind_east_direction_fraction: float = 0.5
    wind_north_direction_fraction: float = 0.5
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
            "long_route_share_by_cell",
            "stable_wellness_by_cell",
            "exposed_wellness_by_cell",
            "panicked_wellness_by_cell",
            "forecast_danger_by_cell",
            "hazard_source_proximity_by_cell",
        ):
            object.__setattr__(self, name, bounded_cell_vector(name))

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
            "wind_east_direction_fraction",
            "wind_north_direction_fraction",
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
        speed_fraction = np.clip(
            self.mean_speed_by_cell / self.maximum_speed,
            0.0,
            1.0,
        )
        mobility_delay = np.where(
            self.active_by_cell > 0.0,
            1.0 - speed_fraction,
            0.0,
        )
        route_time_scale = max(
            self.time_step_minutes,
            float(self.horizon) * self.time_step_minutes,
        )
        candidate_total = float(max(1.0, self.candidate_count_by_cell.sum()))
        road_node_total = float(max(1.0, self.network_node_count_by_cell.sum()))
        pedestrian_features = np.stack(
            (
                np.clip(self.active_by_cell / population_scale, 0.0, 1.0),
                mobility_delay,
                np.clip(self.mean_route_time_by_cell / route_time_scale, 0.0, 1.0),
                self.long_route_share_by_cell,
                self.stable_wellness_by_cell,
                self.exposed_wellness_by_cell,
                self.panicked_wellness_by_cell,
            ),
            axis=-1,
        )
        hazard_features = np.stack(
            (
                self.danger_by_cell,
                self.forecast_danger_by_cell,
                self.hazard_source_proximity_by_cell,
            ),
            axis=-1,
        )
        infrastructure_features = np.stack(
            (
                np.clip(self.remaining_capacity_by_cell / population_scale, 0.0, 1.0),
                self.shelter_utilization_by_cell,
                np.clip(self.deployable_capacity_by_cell / population_scale, 0.0, 1.0),
                np.clip(self.candidate_count_by_cell / candidate_total, 0.0, 1.0),
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
        global_features = np.asarray(
            (
                time_remaining,
                self.outcome.active_population / population_scale,
                self.remaining_deployments / deployment_scale,
                self.network_load_share,
                self.wind_speed_fraction,
                self.wind_east_direction_fraction,
                self.wind_north_direction_fraction,
                self.hazard_spread_fraction,
                self.population_network_density_fraction,
                self.hazard_instance_fraction,
                self.configured_panic_fraction,
                self.deployment_capacity_coverage_fraction,
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
                self.candidate_forecast_danger,
                self.candidate_hazard_safety_margin,
                self.candidate_reroutable_population,
                self.candidate_risk_time_reduction,
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
    """Choose the safe site with the largest estimated reduction in ``T + E``.

    The candidate score is capacity capped. Each beneficiary's travel-time
    saving is weighted by ``1 + max(current danger, forecast danger)``. Thus
    the same rule values both earlier evacuation and earlier removal from an
    urgent area, without assigning different roles to successive shelters.
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
        score = np.asarray(
            observation.candidate_risk_time_reduction,
            dtype=np.float64,
        )[feasible]
        action = int(feasible[int(np.argmax(score))])
        return PolicyDecision(action_index=action, strategy=self.name)


class RouteTimeSavingHeuristic(RiskTimeReductionHeuristic):
    """Compatibility alias for pre-v27 experiment definitions."""

    name = "route_saving"


class HazardWeightedDemandHeuristic:
    """Choose a candidate in the region with greatest current exposure mass.

    The score ``N_i * (1 + D_i)`` is the cell contribution used by the
    simulator's hazard-weighted person-time objective.  It has no fitted
    coefficient: danger is already bounded in ``[0, 1]``, so exposed demand
    receives between one and two times the priority of otherwise identical
    demand. Stable candidate-table order resolves exact ties.
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
        self.nx = int(core.cellX)
        self.ny = int(core.cellY)
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
        self.maximum_shelter_forecast_danger = float(
            getattr(core, "maximumShelterForecastDanger", 0.6)
        )
        if not 0.0 <= self.maximum_shelter_forecast_danger <= 1.0:
            raise ValueError("maximumShelterForecastDanger must lie in [0, 1]")
        self.require_candidate_operational_benefit = bool(
            getattr(core, "requireCandidateOperationalBenefit", False)
        )
        self.minimum_candidate_reroutable_fraction = float(
            getattr(core, "minimumCandidateReroutableFraction", 0.0)
        )
        # The Core field keeps its pre-v27 name for configuration compatibility.
        self.minimum_candidate_risk_time_reduction = float(
            getattr(core, "minimumCandidateRouteTimeSaving", 0.0)
        )
        self.minimum_candidate_hazard_safety_margin = float(
            getattr(core, "minimumCandidateHazardSafetyMargin", 0.0)
        )
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
            x_edges = np.arange(self.nx + 1, dtype=np.float64)
            y_edges = np.arange(self.ny + 1, dtype=np.float64)
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
                north.append(
                    (0.5 * (y_edges[j] + y_edges[j + 1]) - y_edges[0]) / y_span
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

        Four-neighbor edges carry hazard/spatial proximity even where a cell is
        road-empty. Road links are then added explicitly, including any
        nonlocal connections introduced by an irregular network topology.
        """
        pairs = set()
        for i in range(self.nx):
            for j in range(self.ny):
                source = i * self.ny + j
                for di, dj in ((1, 0), (0, 1)):
                    ni, nj = i + di, j + dj
                    if ni < self.nx and nj < self.ny:
                        target = ni * self.ny + nj
                        pairs.add((source, target))
                        pairs.add((target, source))

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
                    float(np.clip((y_coord - y_min) / (y_max - y_min), 0.0, 1.0))
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
        """Return cells that are available, useful, and forecast-safe.

        Safety is enforced through the same administrator-visible forecast
        supplied to the policy.  If every remaining cell is forecast unsafe,
        deployment pauses instead of forcing an avoidably hazardous shelter.
        A cell resolves to a feasible slot only when the shared deterministic
        lower layer actually found a positive-capacity candidate there.
        """
        candidate_forecast_danger = getattr(
            self, "_mask_candidate_forecast_danger", None
        )
        candidate_hazard_safety = getattr(
            self, "_mask_candidate_hazard_safety", None
        )
        candidate_reroutable_population = getattr(
            self, "_mask_candidate_reroutable_population", None
        )
        candidate_risk_time_reduction = getattr(
            self, "_mask_candidate_risk_time_reduction", None
        )
        force_capacity_token = bool(getattr(self, "_mask_force_capacity_token", False))
        result = []
        for action, (_, _cell_index, capacity, _, _) in enumerate(records):
            cell_index = int(records[action][1])
            forecast = (
                float(candidate_forecast_danger[action])
                if candidate_forecast_danger is not None
                else float(forecast_danger[cell_index])
            )
            safety = (
                float(candidate_hazard_safety[action])
                if candidate_hazard_safety is not None
                else 1.0
            )
            reroutable = (
                float(candidate_reroutable_population[action])
                if candidate_reroutable_population is not None
                else 1.0
            )
            risk_reduction = (
                float(candidate_risk_time_reduction[action])
                if candidate_risk_time_reduction is not None
                else 1.0
            )
            operational = True
            if self.require_candidate_operational_benefit and not force_capacity_token:
                operational = (
                    reroutable > self.minimum_candidate_reroutable_fraction
                    and risk_reduction > self.minimum_candidate_risk_time_reduction
                )
            result.append(
                remaining_deployments > 0
                and capacity > 0.0
                and forecast <= self.maximum_shelter_forecast_danger
                and safety >= self.minimum_candidate_hazard_safety_margin
                and operational
            )
        return np.asarray(result, dtype=np.bool_)

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
    ]:
        """Aggregate individual route and wellness telemetry by region."""
        active = np.zeros(self.number_of_cells, dtype=np.float32)
        speed_sum = np.zeros(self.number_of_cells, dtype=np.float32)
        route_time_sum = np.zeros(self.number_of_cells, dtype=np.float32)
        routed_population = np.zeros(self.number_of_cells, dtype=np.float32)
        long_route_population = np.zeros(self.number_of_cells, dtype=np.float32)
        stable_population = np.zeros(self.number_of_cells, dtype=np.float32)
        exposed_population = np.zeros(self.number_of_cells, dtype=np.float32)
        panicked_population = np.zeros(self.number_of_cells, dtype=np.float32)
        route_flow = {}
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
            if not is_panicked and np.isfinite(route_distance):
                speed = max(
                    speed_floor,
                    float(getattr(pedestrian, "currSpeed", 0.0)),
                )
                route_time = min(maximum_route_time, route_distance / speed)
                route_time_sum[index] += group_size * route_time
                routed_population[index] += group_size
                if route_time > route_threshold:
                    long_route_population[index] += group_size
                target_index = self._node_cell_index(
                    getattr(route, "endNode", None) if route is not None else None
                )
                if target_index is not None and target_index != index:
                    # Shelter-region information must flow back to the origin
                    # region where the long-route demand is observed.
                    # Use both directions. The origin receives information
                    # about its assigned shelter, while the shelter region
                    # receives the magnitude and condition of assigned demand.
                    for key in ((target_index, index), (index, target_index)):
                        route_flow[key] = float(route_flow.get(key, 0.0)) + group_size
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
            mean_route_time,
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
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """Return current-front forecast, source proximity, and wind summary."""
        hazard_store = getattr(self.core, "hazardDS", None)
        forecast_method = getattr(hazard_store, "forecast_danger_by_cell", None)
        if callable(forecast_method):
            forecast = np.asarray(
                forecast_method(
                    max(1, int(getattr(self.core, "shelterActionInterval", 1)))
                ),
                dtype=np.float32,
            ).reshape(self.number_of_cells)
            forecast = np.clip(
                np.nan_to_num(forecast, nan=0.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
        else:
            forecast = danger.copy()

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
            wind_east = 0.5
            wind_north = 0.5
        else:
            angle = np.deg2rad(
                float(getattr(hazard_store, "wind_direction_degrees", 0.0))
            )
            wind_east = float(0.5 * (1.0 + np.cos(angle)))
            wind_north = float(0.5 * (1.0 + np.sin(angle)))
        spread_values = [
            np.clip(float(getattr(hazard, "spreadRate", 0.0)), 0.0, 1.0)
            for hazard in active_hazards
        ]
        spread_fraction = float(np.mean(spread_values)) if spread_values else 0.0
        return (
            forecast.astype(np.float32),
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute safety and a concise marginal ``T + E`` benefit per site."""
        action_count = len(records)
        nearest_shelter = np.ones(action_count, dtype=np.float32)
        candidate_forecast = np.zeros(action_count, dtype=np.float32)
        safety_margin = np.ones(action_count, dtype=np.float32)
        reroutable = np.zeros(action_count, dtype=np.float32)
        risk_time_reduction = np.zeros(action_count, dtype=np.float32)
        if action_count == 0:
            return (
                nearest_shelter,
                candidate_forecast,
                safety_margin,
                reroutable,
                risk_time_reduction,
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
        pedestrians = [
            pedestrian
            for pedestrian in self.core.pedDS.pedAgentList.values()
            if not bool(getattr(pedestrian, "terminated", False))
            and not bool(getattr(pedestrian, "panicked", False))
        ]
        anchor_method = getattr(self.core.pedDS, "_route_anchor_node", None)
        population_scale = float(max(1, self.initial_population))
        horizon_minutes = max(
            float(getattr(self.core, "timeStepMinutes", 1.0)),
            float(
                max(
                    1,
                    int(self.horizon)
                    - int(getattr(self.core.pedDS, "currTime", 0)),
                )
            )
            * float(getattr(self.core, "timeStepMinutes", 1.0)),
        )
        speed_floor = max(1e-6, 0.1 * float(self.core.maxSpeed))

        for action, (record, candidate_node) in enumerate(zip(records, nodes)):
            _, host_cell, capacity, _, _ = record
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

            potential_population = 0.0
            beneficiary_values = []
            for pedestrian in pedestrians:
                current_distance = self._remaining_route_distance(pedestrian)
                if callable(anchor_method):
                    anchor = anchor_method(pedestrian)
                else:
                    anchor = getattr(pedestrian, "currNode", None) or getattr(
                        pedestrian,
                        "edge_dest_node",
                        None,
                    )
                candidate_distance = self._network_or_euclidean_distance(
                    anchor,
                    candidate_node,
                )
                if not bool(getattr(pedestrian, "atNode", False)):
                    candidate_distance += max(
                        0.0,
                        float(getattr(pedestrian, "edge_remain", 0.0)),
                    )
                if not np.isfinite(candidate_distance) or candidate_distance >= current_distance:
                    continue
                group_size = max(1, int(getattr(pedestrian, "group_size", 1)))
                speed = max(
                    speed_floor,
                    float(getattr(pedestrian, "currSpeed", 0.0)),
                )
                if not np.isfinite(current_distance):
                    # With no reserved destination, the counterfactual is
                    # remaining active for the rest of the episode.
                    current_distance = speed * horizon_minutes
                if candidate_distance >= current_distance:
                    continue
                current_cell = getattr(pedestrian, "currCell", None)
                urgency = 0.0
                if current_cell is not None:
                    i, j = int(current_cell[0]), int(current_cell[1])
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        cell_index = i * self.ny + j
                        urgency = max(
                            float(danger[cell_index]),
                            float(forecast_danger[cell_index]),
                        )
                potential_population += group_size
                # This is the immediate, auditable proxy for reduction in the
                # scientific objective's two time terms: ordinary active time
                # plus danger-weighted exposure time.
                beneficiary_values.append(
                    (
                        (current_distance - candidate_distance)
                        / speed
                        * (1.0 + urgency),
                        group_size,
                    )
                )
            served = min(max(0.0, float(capacity)), potential_population)
            reroutable[action] = float(
                np.clip(served / population_scale, 0.0, 1.0)
            )
            if served > 0.0:
                unallocated = max(0.0, float(capacity))
                potential_risk_minutes = 0.0
                served = 0.0
                for per_person_value, group_size in sorted(
                    beneficiary_values, reverse=True
                ):
                    if float(group_size) > unallocated:
                        continue
                    potential_risk_minutes += group_size * per_person_value
                    served += group_size
                    unallocated -= group_size
                    if unallocated <= 0.0:
                        break
                reroutable[action] = float(
                    np.clip(served / population_scale, 0.0, 1.0)
                )
                risk_time_reduction[action] = float(
                    np.clip(
                        potential_risk_minutes
                        / (2.0 * population_scale * horizon_minutes),
                        0.0,
                        1.0,
                    )
                )
        return (
            nearest_shelter,
            candidate_forecast,
            safety_margin,
            reroutable,
            risk_time_reduction,
        )

    def build(
        self,
        *,
        decision_index: int,
        simulation_time: int,
        remaining_deployments: int,
    ) -> RegionalObservation:
        (
            active,
            mean_speed,
            mean_route_time,
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
            candidate_reroutable_population,
            candidate_risk_time_reduction,
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
        action_interval = max(
            1, int(getattr(self.core, "shelterActionInterval", 1))
        )
        remaining_decision_epochs = 1 + max(
            0, (int(self.horizon) - int(simulation_time)) // action_interval
        )
        force_capacity_token = (
            remaining_deployments > 0
            and remaining_deployments >= remaining_decision_epochs
        )
        self._mask_candidate_forecast_danger = candidate_forecast_danger
        self._mask_candidate_hazard_safety = candidate_hazard_safety
        self._mask_candidate_reroutable_population = candidate_reroutable_population
        self._mask_candidate_risk_time_reduction = candidate_risk_time_reduction
        self._mask_force_capacity_token = force_capacity_token
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
            long_route_share_by_cell=long_route_share,
            stable_wellness_by_cell=stable_wellness,
            exposed_wellness_by_cell=exposed_wellness,
            panicked_wellness_by_cell=panicked_wellness,
            forecast_danger_by_cell=forecast_danger,
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
            candidate_reroutable_population=candidate_reroutable_population,
            candidate_risk_time_reduction=candidate_risk_time_reduction,
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

        cell_index = int(observation.candidate_cell_indices[action])
        cell = divmod(cell_index, int(self.core.cellY))
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

        rerouted = 0
        reroute = getattr(self.core.pedDS, "reroute_to_new_shelter_if_closer", None)
        if callable(reroute):
            rerouted = int(reroute(shelter))

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
