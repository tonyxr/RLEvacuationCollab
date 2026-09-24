#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A torch-free, OSM-free harness that drives the real evacuation dynamics.

Every stochastic mechanism the paired estimator has to survive is the real
one: ``HazardDatabase``'s cellular-automaton spread, ``PedestrianDatabase``'s
keyed casualty and panic draws, the social-force movement model, the
congestion model, and ``ShelterDatabase``'s deterministic site rule.  Only the
map is synthetic -- a grid city fed to ``MapDS`` in the same raw OpenStreetMap
tuple format ``OSMProcessor`` would produce -- and only the learner is absent.

This exists because the repository's pinned environment is unreachable from
the tool making these changes (no torch, and the package index is outside the
egress allowlist), while the entire dynamics layer is pure numpy.  The
credit-assignment machinery can therefore be exercised against real stochastic
dynamics here, and the same code paths run unchanged on a real OSM map.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

from CAProcessor import CellTracker
from HazardDatabase import HazardDS
from MapDatabase import MapDS
from NetworkCongestion import PedestrianCongestionModel
from PedestrianDatabase import PedDS
from RewardProcessor import RewardProcessor
from ShelterDatabase import ShelterDS
from SocialForce import ForceProcessor

CELL_VECTORS = (
    "heatByCell",
    "smokeByCell",
    "countByCell",
    "avgVelocityByCell",
    "shelterPressureByCell",
    "guidanceInterByCell",
    "dangerLevelByCell",
    "shelterFulfillByCell",
    "wellnessPenaltyByCell",
)

SHELTER_TYPES = ("school", "library", "community_centre", "hospital", "church")


def synthetic_raw_map(
    *,
    grid: int = 12,
    spacing_m: float = 90.0,
    center_lon: float = -77.86,
    center_lat: float = 40.79,
    shelter_every: int = 5,
    seed: int = 7,
):
    """Build raw OSM-style node and edge lists for a grid city.

    ``MapDS`` consumes ``(osmid, data)`` node tuples and ``(u, v, data)`` edge
    tuples, exactly as ``osmnx`` yields them, so a synthetic grid exercises the
    real map construction path rather than a stand-in for it.
    """
    rng = np.random.default_rng(seed)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))

    nodes = []
    index_of = {}
    osmid = 1000
    for i in range(grid):
        for j in range(grid):
            x_m = (i - grid / 2.0) * spacing_m
            y_m = (j - grid / 2.0) * spacing_m
            lon = center_lon + x_m / meters_per_deg_lon
            lat = center_lat + y_m / meters_per_deg_lat
            data = {"x": float(lon), "y": float(lat), "street_count": 4}
            if (i * grid + j) % shelter_every == 0:
                data["building_type"] = SHELTER_TYPES[
                    int(rng.integers(0, len(SHELTER_TYPES)))
                ]
            nodes.append((osmid, data))
            index_of[(i, j)] = osmid
            osmid += 1

    edges = []
    edge_osmid = 500_000
    for i in range(grid):
        for j in range(grid):
            here = index_of[(i, j)]
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni >= grid or nj >= grid:
                    continue
                there = index_of[(ni, nj)]
                length = float(spacing_m)
                for u, v in ((here, there), (there, here)):
                    edges.append((u, v, {"osmid": edge_osmid, "length": length}))
                edge_osmid += 1

    # ``MapDS`` routes over a live NetworkX graph (``locationDrive``); the
    # routing-tree cache is keyed on it, so the grid has to be a real graph
    # rather than an edge list.
    graph = nx.MultiDiGraph()
    for osm_id, data in nodes:
        graph.add_node(int(osm_id), **data)
    for u, v, data in edges:
        graph.add_edge(int(u), int(v), **data)
    return nodes, edges, graph


class TestbedCore:
    """A ``Core`` stand-in exposing exactly the surface the branch code uses.

    The attribute names match ``Core`` deliberately: ``CounterfactualBranch``
    is written against the production object, and anything that works here
    works there without translation.
    """

    def __init__(
        self,
        *,
        grid: int = 12,
        cell_x: int = 4,
        cell_y: int = 4,
        population: int = 240,
        group_size: int = 1,
        hazard_count: int = 2,
        stop_time: int = 40,
        max_speed: float = 80.0,
        scenario_seed: int = 20260920,
        panic_rate: float = 0.25,
        casualty_rate=(14, 9),
        spread_rate=(30, 25),
        speed_reduct=(25, 16),
        hazard_mode: str = "stochastic",
        time_step_minutes: float = 1.0,
        shelter_capacity_token=None,
        spacing_m: float = 90.0,
        candidate_count: int = 64,
    ):
        self.cellX = int(cell_x)
        self.cellY = int(cell_y)
        self.stopTime = int(stop_time)
        self.maxSpeed = float(max_speed)
        self.timeStepMinutes = float(time_step_minutes)
        self.maximumShelterForecastDanger = 0.6
        self.address = "synthetic-grid"
        self.scenario_seed = int(scenario_seed)
        self.shelterActionInterval = 4
        self.initial_population = int(population)

        seeds = {
            "hazard_evolution": int(
                np.random.SeedSequence([scenario_seed, 1]).generate_state(1, dtype=np.uint64)[0]
            ),
            "pedestrian_hazard_outcomes": int(
                np.random.SeedSequence([scenario_seed, 2]).generate_state(1, dtype=np.uint64)[0]
            ),
            "pedestrian_panic_behavior": int(
                np.random.SeedSequence([scenario_seed, 3]).generate_state(1, dtype=np.uint64)[0]
            ),
        }
        self.random_stream_seeds = seeds

        raw_nodes, raw_edges, graph = synthetic_raw_map(
            grid=grid, spacing_m=float(spacing_m), seed=scenario_seed
        )
        self.mapDS = MapDS(raw_nodes, raw_edges, self.address, graph)
        self.congestionModel = PedestrianCongestionModel(enabled=True)
        self.pedDS = PedDS(population, maximum_group_size=group_size)
        self.pedDS.set_hazard_random_seed(seeds["pedestrian_hazard_outcomes"])
        self.pedDS.configure_panic(
            rate=panic_rate,
            herd_probability=0.5,
            danger_threshold=3,
            random_seed=seeds["pedestrian_panic_behavior"],
        )
        self.hazardDS = HazardDS(
            hazard_count,
            casualty_rate,
            spread_rate,
            speed_reduct,
            rng=np.random.default_rng(seeds["hazard_evolution"]),
            wind_speed_m_per_minute=12.0,
            wind_direction_degrees=45.0,
            wind_influence=1.0,
            time_step_minutes=self.timeStepMinutes,
        )
        self.hazardDS.cell_state_evolution_mode = str(hazard_mode)
        self.cellTracker = CellTracker(self.cellX, self.cellY)
        self.forceTracker = ForceProcessor()
        # Production samples ``shelterCanVol`` candidates (20 for State
        # College); the default keeps the historical testbed behaviour.
        self.shelterDS = ShelterDS(candidateVol=int(candidate_count), initVol=2)
        self.shelterDS.deploymentCapacityToken = (
            None
            if shelter_capacity_token is None
            else int(shelter_capacity_token)
        )

        # MapDS/PedDS still use NumPy's legacy global generator for initial
        # origin/destination sampling. Make that initialization a pure
        # function of the declared scenario seed without contaminating the
        # caller or allowing earlier tests to change this testbed.
        global_numpy_state = np.random.get_state()
        np.random.seed(int(self.scenario_seed) & 0xFFFFFFFF)
        try:
            self._build()
        finally:
            np.random.set_state(global_numpy_state)

    def _build(self) -> None:
        self.mapDS.computeConvertUnit()
        self.mapDS.boundarySetter()

        bounds = self.mapDS.boundMeters
        x_length = float(abs(bounds[1] - bounds[0])) if len(bounds) >= 2 else 1000.0
        y_length = float(abs(bounds[3] - bounds[2])) if len(bounds) >= 4 else 1000.0
        self.cellTracker.initialCut(x_length, y_length)

        total_cells = int(self.cellX * self.cellY)
        for name in CELL_VECTORS:
            if getattr(self.cellTracker, name, None) is None:
                setattr(self.cellTracker, name, np.zeros(total_cells, dtype=float))

        self.mapDS.nodeInit(self.cellTracker)
        self.mapDS.edgeInit(self.cellTracker)
        self.mapDS.buildEdgeIndices()
        self.mapDS.computeNodeCapSum()
        self.congestionModel.bind_edges(self.mapDS.edgeListByLocalID.values())
        self.forceTracker.setupCellTracker(self.cellTracker)

        self.shelterDS.shelterCanList = self.mapDS.shelterCanList
        if not self.shelterDS.shelterCanList:
            raise RuntimeError("Synthetic map produced no shelter candidates")
        self.shelterDS.shelterPerCell(self.cellTracker, self.cellX, self.cellY)
        self.shelterDS.initShelter()

        self.hazardDS.setCellTracker(self.cellTracker)
        self.hazardDS.initHazard(self.mapDS, self.cellTracker)

        self.pedDS.initPedestrianAgent(self.mapDS, self.cellTracker, self.maxSpeed)
        self.shelterDS.shelterByOSMID = {
            shelter.nodeMapped.OSMID: shelter
            for shelter in self.shelterDS.shelterList.values()
        }
        self.pedDS.checkReady(
            mapDS=self.mapDS,
            cellTracker=self.cellTracker,
            maxSpeed=self.maxSpeed,
            hazardDS=self.hazardDS,
            shelterDS=self.shelterDS,
            forceTracker=self.forceTracker,
            congestionModel=self.congestionModel,
            timeStepMinutes=self.timeStepMinutes,
            casualtyReferenceExposureMinutes=60.0,
            evacuationHorizonTimesteps=self.stopTime,
        )
        self.pedDS.route_active_to_nearest_shelter()

    # -- convenience --------------------------------------------------------

    def reward_model(self) -> RewardProcessor:
        return RewardProcessor()

    def feasible_cells(self) -> list[int]:
        """Cells that still hold a deployable candidate."""
        cells = []
        grid = self.shelterDS.shelterCanByCell
        for i in range(self.cellX):
            for j in range(self.cellY):
                if grid[i][j]:
                    cells.append(i * self.cellY + j)
        return cells


def build(**kwargs) -> TestbedCore:
    return TestbedCore(**kwargs)
