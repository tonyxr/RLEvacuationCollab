#!/usr/bin/env python3
"""Deterministic pedestrian link-congestion model.

The simulator is microscopic, but OpenStreetMap does not reliably provide
walkable widths.  We therefore use one explicit effective-width assumption and
the Weidmann/Kladek pedestrian speed-density curve on each physical road link.
Opposing directed graph edges that represent the same OSM way segment share a
single density so counter-flow cannot disappear through graph duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class LinkCongestionState:
    """Synchronized state of one physical road link for one integration substep."""

    physical_key: tuple
    occupancy: int
    length_m: float
    effective_width_m: float
    density_ped_per_m2: float
    speed_ratio: float


class PedestrianCongestionModel:
    """Apply a bounded Weidmann pedestrian fundamental diagram to OSM links.

    ``shape`` and ``jam_density_ped_per_m2`` are the conventional Weidmann
    values 1.913 and 5.4.  ``minimum_speed_ratio`` is an explicit numerical
    regularization: it prevents a link whose entry queue exceeds its modeled
    storage area from becoming an irreversible absorbing gridlock state.
    """

    MODEL_NAME = "weidmann_physical_link_v1"

    def __init__(
        self,
        *,
        enabled: bool = True,
        effective_width_m: float = 3.0,
        jam_density_ped_per_m2: float = 5.4,
        shape: float = 1.913,
        minimum_speed_ratio: float = 0.05,
        integration_substep_seconds: float = 10.0,
    ):
        self.enabled = bool(enabled)
        self.effective_width_m = self._positive(
            "effective_width_m",
            effective_width_m,
        )
        self.jam_density_ped_per_m2 = self._positive(
            "jam_density_ped_per_m2",
            jam_density_ped_per_m2,
        )
        self.shape = self._positive("shape", shape)
        self.integration_substep_seconds = self._positive(
            "integration_substep_seconds",
            integration_substep_seconds,
        )
        self.minimum_speed_ratio = float(minimum_speed_ratio)
        if (
            not math.isfinite(self.minimum_speed_ratio)
            or not 0.0 <= self.minimum_speed_ratio < 1.0
        ):
            raise ValueError("minimum_speed_ratio must be finite and in [0, 1)")
        self._physical_edges = None
        self._physical_lengths = None
        self._last_occupied_keys = set()

    @staticmethod
    def _positive(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    def contract(self) -> dict:
        return {
            "model": self.MODEL_NAME,
            "enabled": self.enabled,
            "effective_width_m": self.effective_width_m,
            "jam_density_ped_per_m2": self.jam_density_ped_per_m2,
            "shape": self.shape,
            "minimum_speed_ratio": self.minimum_speed_ratio,
            "integration_substep_seconds": self.integration_substep_seconds,
            "counterflow": "combined_on_physical_osm_link",
            "synchronization": "occupancy_frozen_at_each_internal_substep",
        }

    @staticmethod
    def physical_link_key(edge) -> tuple:
        """Identify one physical OSM segment across graph directions."""
        start = getattr(getattr(edge, "startNode", None), "OSMID", None)
        end = getattr(getattr(edge, "endNode", None), "OSMID", None)
        osmid = getattr(edge, "OSMID", None)
        if start is None or end is None:
            return ("local_edge", int(getattr(edge, "edgeID", id(edge))))
        low, high = sorted((int(start), int(end)))
        return ("osm_segment", low, high, str(osmid))

    @staticmethod
    def intended_edge(pedestrian):
        """Return the edge occupied now, or the next edge queued at a node."""
        if not bool(getattr(pedestrian, "atNode", False)):
            current = getattr(pedestrian, "currEdge", None)
            if current is not None:
                return current
        panic_edge = getattr(pedestrian, "panic_next_edge", None)
        if panic_edge is not None:
            return panic_edge
        route = getattr(pedestrian, "routeFollowing", None)
        remaining = getattr(route, "edgeRemained", None)
        if remaining:
            return remaining[0]
        return None

    def speed_ratio(self, density_ped_per_m2: float) -> float:
        """Return ``v / v_free`` at a non-negative pedestrian density."""
        density = float(density_ped_per_m2)
        if not math.isfinite(density) or density < 0.0:
            raise ValueError("density_ped_per_m2 must be finite and non-negative")
        if not self.enabled or density <= 0.0:
            return 1.0
        if density >= self.jam_density_ped_per_m2:
            return self.minimum_speed_ratio
        exponent = -self.shape * (
            (1.0 / density) - (1.0 / self.jam_density_ped_per_m2)
        )
        ratio = 1.0 - math.exp(exponent)
        return float(min(1.0, max(self.minimum_speed_ratio, ratio)))

    def bind_edges(self, edges: Iterable) -> None:
        """Index immutable map topology once for efficient timestep updates."""
        if self._physical_edges is not None:
            raise RuntimeError("Congestion model is already bound to a road network")
        physical_edges = {}
        physical_lengths = {}
        for edge in tuple(edges):
            edge.edgeFlow = 0
            edge.congestionOccupancy = 0
            edge.congestionDensityPedPerM2 = 0.0
            edge.congestionSpeedRatio = 1.0
            key = self.physical_link_key(edge)
            physical_edges.setdefault(key, {})[id(edge)] = edge
            physical_lengths[key] = max(
                max(1e-6, float(getattr(edge, "edgeLen", 0.0))),
                float(physical_lengths.get(key, 0.0)),
            )
        self._physical_edges = physical_edges
        self._physical_lengths = physical_lengths
        self._last_occupied_keys = set()

    def snapshot(self, pedestrians: Iterable, edges: Iterable) -> tuple[dict, tuple, dict]:
        """Freeze link loads for one substep and return per-agent speed ratios.

        The snapshot is computed before any pedestrian moves.  Consequently,
        dictionary iteration order cannot give early agents more capacity than
        later agents.  Pedestrians waiting at nodes are assigned to their next
        route edge so a large simultaneous entry wave is represented in the
        current substep rather than one simulator transition late.
        """
        if self._physical_edges is None:
            self.bind_edges(edges)
        pedestrian_list = tuple(
            pedestrian
            for pedestrian in pedestrians
            if not bool(getattr(pedestrian, "terminated", False))
        )
        physical_edges = self._physical_edges
        physical_length = self._physical_lengths
        for key in self._last_occupied_keys:
            for edge in physical_edges.get(key, {}).values():
                edge.edgeFlow = 0
                edge.congestionOccupancy = 0
                edge.congestionDensityPedPerM2 = 0.0
                edge.congestionSpeedRatio = 1.0

        assignments = []
        physical_occupancy = {}
        directional_occupancy = {}
        active_population = 0
        for pedestrian in pedestrian_list:
            size = max(1, int(getattr(pedestrian, "group_size", 1)))
            active_population += size
            edge = self.intended_edge(pedestrian)
            if edge is None:
                assignments.append((pedestrian, None, size))
                continue
            key = self.physical_link_key(edge)
            length = max(1e-6, float(getattr(edge, "edgeLen", 0.0)))
            assignments.append((pedestrian, key, size))
            physical_occupancy[key] = int(physical_occupancy.get(key, 0)) + size
            physical_length[key] = max(length, float(physical_length.get(key, 0.0)))
            edge_identity = id(edge)
            directional_occupancy[edge_identity] = int(
                directional_occupancy.get(edge_identity, 0)
            ) + size

        states = {}
        for key, occupancy in physical_occupancy.items():
            length = physical_length[key]
            density = float(occupancy) / (length * self.effective_width_m)
            ratio = self.speed_ratio(density)
            states[key] = LinkCongestionState(
                physical_key=key,
                occupancy=int(occupancy),
                length_m=float(length),
                effective_width_m=self.effective_width_m,
                density_ped_per_m2=float(density),
                speed_ratio=float(ratio),
            )
            for edge in physical_edges.get(key, {}).values():
                edge.edgeFlow = int(directional_occupancy.get(id(edge), 0))
                edge.congestionOccupancy = int(occupancy)
                edge.congestionDensityPedPerM2 = float(density)
                edge.congestionSpeedRatio = float(ratio)
        self._last_occupied_keys = set(physical_occupancy)

        ratios = {}
        congested_population = 0
        assigned_population = 0
        weighted_ratio_terms = []
        minimum_ratio = 1.0
        for pedestrian, key, size in assignments:
            ratio = 1.0 if key is None else float(states[key].speed_ratio)
            ratios[id(pedestrian)] = ratio
            weighted_ratio_terms.append((
                int(getattr(pedestrian, "agentID", 0)),
                float(ratio) * size,
            ))
            minimum_ratio = min(minimum_ratio, ratio)
            if key is not None:
                assigned_population += size
            if ratio < 1.0 - 1e-12:
                congested_population += size

        weighted_ratio = math.fsum(
            term for _, term in sorted(weighted_ratio_terms)
        )
        metrics = {
            "active_population": int(active_population),
            "assigned_link_population": int(assigned_population),
            "occupied_physical_links": int(len(states)),
            "congested_population": int(congested_population),
            "mean_congestion_speed_ratio": (
                float(weighted_ratio) / float(active_population)
                if active_population > 0
                else 1.0
            ),
            "minimum_congestion_speed_ratio": (
                float(minimum_ratio) if active_population > 0 else 1.0
            ),
            "maximum_link_density_ped_per_m2": max(
                (state.density_ped_per_m2 for state in states.values()),
                default=0.0,
            ),
        }
        return ratios, tuple(states[key] for key in sorted(states, key=str)), metrics
