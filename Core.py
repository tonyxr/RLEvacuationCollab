#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

7/28: Structure created, writen all import dependencies, variables
8/6: Worked on initSimulator function
8/7: Finished readInputCSV function
"""

import os
import osmnx as OSM
import csv
import hashlib
import json
import random
import warnings
import numpy as np

#import torch
#import torch.nn as nn
#import torch.optim as optim

"""Import other main Simulator modules"""
from OSMProcessor import OSMProcessor
from MapDatabase import MapDS
from HazardDatabase import HazardDS
from ShelterDatabase import ShelterDS
from PedestrianDatabase import PedDS
from CAProcessor import CellTracker
from CellPartitioning import (
    NODE_DENSITY_ADAPTIVE,
    build_cell_partition,
    normalize_partition_mode,
)
from SocialForce import ForceProcessor

"""Import RL components"""
from RLBridge import (
    DEFAULT_ACTOR_BASELINE_DECAY,
    DEFAULT_ACTOR_EPOCHS,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_ADVANTAGE_SCALE_FLOOR,
    DEFAULT_CLIP_EPS,
    DEFAULT_CRITIC_EPOCHS,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ROLLOUT_EPISODES,
    DEFAULT_TARGET_KL,
    DEFAULT_ENTROPY_COEF_END,
    DEFAULT_EXPLORATION_DECAY_UPDATES,
    DEFAULT_NMCC_CAUSAL_LOSS_COEF,
    DEFAULT_NMCC_DUELING_LOSS_COEF,
    DEFAULT_NMCC_GUIDANCE_MAX,
    DEFAULT_NMCC_GUIDANCE_RAMP_UPDATES,
    DEFAULT_NMCC_GUIDANCE_WARMUP_UPDATES,
    DEFAULT_NMCC_NATURAL_LOSS_COEF,
    DEFAULT_NMCC_TEACHER_COEF,
    DEFAULT_NMCC_TEACHER_DECAY_UPDATES,
    DEFAULT_NMCC_UNCERTAINTY_PENALTY,
    DEFAULT_TEMPERATURE_END,
    DEFAULT_TEMPERATURE_START,
    RLBridge,
)
from NMCCPIConfig import NMCC_PI_CORE_FIELDS
from GNN import DEFAULT_NMCC_ENSEMBLE_SIZE
from TrainingLogger import trainingLog
from EvacuationVisualizer import EvacuationVisualizer
from NetworkCongestion import PedestrianCongestionModel

"""Helpers of RL, help track training progress and timestep progress"""
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

#from reporter import log

import time
import math

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OSM.settings.cache_folder = os.path.join(PROJECT_ROOT, "cache")

"""Timer: efficiency debugging (how much step each step take)"""
class Timer:
    def __init__(self): self.t0 = time.perf_counter()
    def lap(self, label):
        t = time.perf_counter()
        dt = t - self.t0
        self.t0 = t
        print(f"[TIMER] {label}: {dt:.3f}s")

class Core:
    
    # Cache static map infrastructure per address so repeated replications
    # reuse the same prepared road/building/intersection datasets.
    _prepared_infra_cache = {}
    # Cache derived cell boundaries for fixed city + grid dimensions.
    _cell_partition_cache = {}

    @staticmethod
    def _shelter_target(
        *,
        available_candidates: int,
        initial_shelters: int,
        decision_windows: int,
        maximum_additions: int,
    ) -> int:
        """Return the common active-shelter budget for every policy.

        ``maximum_additions=0`` means time-window-limited legacy behavior.
        A positive value decouples the resource budget from the sampled
        candidate-pool size, which is necessary for candidate-scale studies.
        """
        available = max(0, int(available_candidates))
        initial = max(0, int(initial_shelters))
        windows = max(0, int(decision_windows))
        cap = int(maximum_additions)
        if cap < 0:
            raise ValueError("maximum_additions must be non-negative")
        additions = windows if cap == 0 else min(windows, cap)
        return int(min(available, initial + additions))
    
    def __init__(self, machine):
        
        
        """Input Parameters"""
        # Indicate the total duration of timestep
        self.stopTime = 0
        
        # Indicate the input-defined, initial (and max) volume of pedestrians in the network
        self.pedVol = 0
        # Maximum persons represented by one moving agent. One is the exact
        # individual microsimulation; values above one enable declared scalable
        # weighted-cohort training/evaluation.
        self.pedestrianGroupSize = 1
        
        # Indicate the input-defined, total volume of hazards we will simulate
        self.hazardVol = 0
        
        # Indicate the input-defined, ideal traveling speed of evacuees
        self.maxSpeed = 0

        # Temporal and pedestrian-congestion contract. ``maxSpeed`` is stored
        # in metres per minute; one simulator transition is one minute unless
        # an experiment explicitly declares another duration.
        self.timeStepMinutes = 1.0
        self.congestionEnabled = True
        self.congestionEffectiveWidthM = 3.0
        self.congestionJamDensityPedPerM2 = 5.4
        self.congestionShape = 1.913
        self.congestionMinimumSpeedRatio = 0.05
        self.congestionSubstepSeconds = 10.0

        # Paper Equations 19--22: network-constrained social-force dynamics.
        self.socialForceEnabled = True
        self.socialForceSelfCoefficient = 0.05
        self.socialForceImpactCoefficient = 0.50

        # OSM often encodes one physical junction as a cluster of nearby
        # nodes. The consolidated topology is the topology used for routing.
        self.intersectionConsolidationEnabled = True
        # OSMnx buffers each node by this radius, so 5 m merges nodes whose
        # buffers overlap within roughly 10 m. Real-map calibration showed the
        # formerly proposed 15 m radius over-collapsed pedestrian topology.
        self.intersectionConsolidationToleranceM = 5.0

        # Persistent panic follows the supplied experiment definition.
        self.panicRate = 0.0
        self.panicHerdProbability = 0.5
        self.panicDangerThreshold = 3

        # Administrators may reconsider deployment every ten one-minute
        # transitions. PPO uses variable-duration SMDP credit: an action owns
        # outcomes until the next action actually occurs, and the last action
        # remains accountable through the physical episode terminal boundary.
        self.shelterActionInterval = 10
        # Candidate sites forecast above this normalized action-window danger
        # are unavailable to every policy, including benchmarks. This encodes
        # the operational requirement that a new shelter must remain in a
        # plausibly safe region rather than asking PPO to discover a hard
        # safety rule through casualties.
        self.maximumShelterForecastDanger = 0.6
        self.requireCandidateOperationalBenefit = False
        self.minimumCandidateReroutableFraction = 0.0
        self.minimumCandidateRouteTimeSaving = 0.0
        self.minimumCandidateHazardSafetyMargin = 0.0
        
        # Input defined, indicate how many shelter candidate building nodes we will sample
        self.shelterCanVol = 0
        
        # Input defined, indicate how many guidance point candidates intersection nodes we will sample
        self.guidanceCanVol = 0
        
        # Input defined, indicate the mean and variance values [int, int] of speed reduction value of hazards
        self.hazardSpeedReduct = []
        
        # Indicate the spreading probability of hazards
        self.hazardSpreadRate = []
        
        # Input defined, indicate the mean and variance values [int, int] of casualty rate of hazards
        self.hazardCasualtyRate = []
        # Hazard casualty rates are cumulative Level-5 exposure probabilities
        # over this reference duration, not probabilities reapplied every minute.
        self.hazardCasualtyReferenceMinutes = 60.0
        # Observable exogenous wind used by both hazard evolution and the
        # administrator-facing action-window hazard forecast. Direction is the
        # direction toward which the hazard is transported, counter-clockwise
        # from projected east.
        self.hazardWindSpeedMPerMinute = 0.0
        self.hazardWindDirectionDegrees = 0.0
        self.hazardWindInfluence = 1.0
        
        # Input defined, indicate the initial volume of shelters we deploy in the network at t = 0
        self.initShelterVol = 0
        # Every installed site receives the same capacity token so policies
        # compete only on location/timing, never on an accidental difference
        # in total shelter resources. Zero explicitly requests legacy
        # site-specific OSM capacities.
        self.shelterCapacityToken = 500
        
        # Input defined, indicate the initial volume of guidance points we deploy in the network at t = 0
        self.initGuidanceVol = 0
        
        # Input defined, indicate the address name we used to extract needed map data, ex. Seattle, Washington, USA
        self.address = ""

        # Reproducible OSM query specification.  Legacy single-city runs use
        # the administrative place boundary.  Cross-city experiments use a
        # fixed point and radius so a geocoder update cannot silently move the
        # study area and large municipal boundaries cannot exhaust Overpass.
        self.cityID = "single_city"
        self.mapQueryMode = "place"
        self.mapCenterLat = None
        self.mapCenterLon = None
        self.mapRadiusM = None
        
        # indicates the number of X-axis cuts for cells
        self.cellX = 0
        
        # indicates the number of Y-axis cuts for cells
        self.cellY = 0

        # Spatial discretization is explicit because it changes the regional
        # observations and the meaning of every cell-indexed action.  The
        # adaptive default preserves the model's established quantile grid.
        self.cellPartitionMode = NODE_DENSITY_ADAPTIVE
        self.cellPartitionMinWidthFraction = 1e-4
        self.cellPartitionDiagnostics = None
        
        """RL parameters, will tune later"""
        # contrainer for learning rate used by the RL model
        self.learningRate = 0
        # v23 separates policy and critic/world-model optimization. The
        # legacy learningRate remains the explicit critic default for old
        # input files; actorLearningRate controls only accepted PPO steps.
        self.actorLearningRate = DEFAULT_ACTOR_LEARNING_RATE
        # Zero means inherit the experiment's legacy learningRate. Registered
        # v23 curricula set this field explicitly.
        self.criticLearningRate = 0.0
        self.actorPpoEpochs = DEFAULT_ACTOR_EPOCHS
        self.criticPpoEpochs = DEFAULT_CRITIC_EPOCHS
        self.actorBaselineDecay = DEFAULT_ACTOR_BASELINE_DECAY
        self.advantageScaleFloor = DEFAULT_ADVANTAGE_SCALE_FLOOR
        # Exploration vs. Exploitation prob
        self.explorationRate = 0
        # name of the optimizer used
        self.optimizer = "Adam"
        # The single-city default may be overridden by an experiment runner.
        # Multi-city training uses complete city blocks per PPO update so no
        # city is overrepresented in a gradient batch.
        self.ppoRolloutEpisodes = DEFAULT_ROLLOUT_EPISODES
        self.finalizePpoRollout = False
        # Hybrid Natural-Momentum Counterfactual Control. The master switch is
        # explicit and recorded in every run; scientific runs must never infer
        # NMCC from a checkpoint filename or silently change the actor target.
        self.nmccEnabled = False
        self.nmccCounterfactualHorizon = 0
        self.nmccCounterfactualWeight = 1.0
        self.nmccJointCounterfactualWeight = 1.0
        self.nmccInterventionCost = 0.0
        self.nmccEnsembleSize = DEFAULT_NMCC_ENSEMBLE_SIZE
        self.nmccNaturalLossCoefficient = DEFAULT_NMCC_NATURAL_LOSS_COEF
        self.nmccCausalLossCoefficient = DEFAULT_NMCC_CAUSAL_LOSS_COEF
        self.nmccDuelingLossCoefficient = DEFAULT_NMCC_DUELING_LOSS_COEF
        self.nmccTeacherCoefficient = DEFAULT_NMCC_TEACHER_COEF
        self.nmccTeacherDecayUpdates = DEFAULT_NMCC_TEACHER_DECAY_UPDATES
        self.nmccGuidanceMaximum = DEFAULT_NMCC_GUIDANCE_MAX
        self.nmccGuidanceWarmupUpdates = DEFAULT_NMCC_GUIDANCE_WARMUP_UPDATES
        self.nmccGuidanceRampUpdates = DEFAULT_NMCC_GUIDANCE_RAMP_UPDATES
        self.nmccUncertaintyPenalty = DEFAULT_NMCC_UNCERTAINTY_PENALTY
        self.entropyCoefficientEnd = DEFAULT_ENTROPY_COEF_END
        self.explorationDecayUpdates = DEFAULT_EXPLORATION_DECAY_UPDATES
        self.actionTemperatureStart = DEFAULT_TEMPERATURE_START
        self.actionTemperatureEnd = DEFAULT_TEMPERATURE_END
        # Optional staged-learning gates. Zero preserves the historical joint
        # optimizer from the first rollout. Positive counts refer to complete
        # PPO rollout batches, not individual gradient minibatches.
        self.nmccNaturalPretrainRollouts = 0
        self.nmccCausalPretrainRollouts = 0
        self.nmccControllerWarmupRollouts = 0
        # NMCC policy improvement (exact within-state branch targets) and the
        # actor's prior logit.  Defaults, types and constructor keywords come
        # from RLBridge.NMCC_PI_CORE_FIELDS.
        for attribute, _, _, default in NMCC_PI_CORE_FIELDS:
            setattr(self, attribute, default)

        """Pointers to main processing modules"""
        # Pointer for the NetworkDatabase
        self.mapDS = None
        
        # Pointer for the PedestrianDatabase
        self.pedDS = None
        
        # Pointer for the HazardDatabase
        self.hazardDS = None
        
        # Pointer for the CellularAutomataTracker
        self.cellTracker = None
        
        # Pointer for the ShelterDatabase
        self.shelterDS = None
        
        # Pointer for the SocialForceProcessor
        self.forceTracker = None

        self.congestionModel = None
        
        # Pointer to handler of all SOSM extracter functions
        self.OSMProcessor = None
        
        self.rl = None
        
        self.logger = None
        self.run_dir = None
        
        """Other Parameters, for auto-execution purposes"""
        self.machine = machine
        
        self.logName = ""
        
        self.currReplication = 0
        self.verbose = False
        self.profile_timing = False
        self.targetActiveShelters = 0
        # Zero preserves the general runner's time-window-limited behavior.
        # Experiments that vary the candidate pool should set an explicit
        # positive resource budget so "more candidates" does not silently mean
        # "install every candidate".
        self.maxAdditionalShelters = 0
        self.baseline_initial_shelter_ids = frozenset()
        self.static_predeployment_shelter_ids = frozenset()
        self.hazardEvolutionMode = "stochastic"
        self.scenario_seed = None
        self.policy_seed = None
        self.episode_summary = None
        self.visualizer = None
        self.visualization_manifest = None
        self.visualization_enabled = False
        self.visualization_milestones = "quartiles"
        self.visualization_individual_snapshots = True
        self.visualization_vector_outputs = True
        self.random_stream_seeds = {}
        self._hazard_trajectory_hasher = None

    """Getter Functions"""
    
    def getStopTime(self):
        return self.stopTime

    def getPedVol(self):
        return self.pedVol
    
    def getHazardVol(self):
        return self.hazardVol 
    
    def getMaxSpeed(self):
        return self.maxSpeed
    
    def getShelterCanVol(self):
        return self.shelterCanVol
    
    def getGuidanceCanVol(self):
        return self.guidanceCanVol
    
    def getSpeedReduct(self):
        return self.hazardSpeedReduct
    
    def getCasualtyRate(self):
        return self.hazardCasualtyRate
    
    def getInitShelterVol(self):
        return self.initShelterVol
    
    def getInitGuidanceVol(self):
        return self.initGuidanceVol
    
    """Handles all functionality at t = 0"""
    """Still need to add input parameters to function calls"""
    def initSimulator(
        self,
        replication,
        machine,
        config_overrides=None,
        phase: str = "train",
        train_mode: bool = True,
        deployment_strategy: str = "rl",
        run_tag: str = None,
        scenario_seed: int = None,
        policy_seed: int = None,
        checkpoint_path: str = None,
        diagnostics_path: str = None,
        visualization_enabled: bool = False,
        visualization_milestones="quartiles",
        visualization_individual_snapshots: bool = True,
        visualization_vector_outputs: bool = True,
    ):
        # read all input data
        self.readInputCSV()
        
        # runtime override for experiment scripts
        if config_overrides:
            self._apply_configuration_overrides(config_overrides)

        self._validate_effective_configuration()
        print(
            "[EFFECTIVE CONFIG] "
            + json.dumps(self._effective_configuration(), sort_keys=True),
            flush=True,
        )

        self.visualization_enabled = bool(visualization_enabled)
        self.visualization_milestones = visualization_milestones
        self.visualization_individual_snapshots = bool(
            visualization_individual_snapshots
        )
        self.visualization_vector_outputs = bool(visualization_vector_outputs)

        # Scenario randomness is deliberately isolated from policy randomness.
        # Reusing a scenario seed across policies supplies matched initial
        # conditions and common exogenous random-number streams for evaluation.
        self.scenario_seed = int(replication if scenario_seed is None else scenario_seed)
        self.policy_seed = int(
            self.scenario_seed + 1_000_003 if policy_seed is None else policy_seed
        )
        random.seed(self.scenario_seed)
        np.random.seed(self.scenario_seed)
        self.random_stream_seeds = {
            "initialization": int(self.scenario_seed),
            "hazard_evolution": int(
                np.random.SeedSequence([self.scenario_seed, 1]).generate_state(
                    1, dtype=np.uint64
                )[0]
            ),
            "pedestrian_hazard_outcomes": int(
                np.random.SeedSequence([self.scenario_seed, 2]).generate_state(
                    1, dtype=np.uint64
                )[0]
            ),
            "pedestrian_panic_behavior": int(
                np.random.SeedSequence([self.scenario_seed, 3]).generate_state(
                    1, dtype=np.uint64
                )[0]
            ),
            "policy": int(self.policy_seed),
        }

        # Guidance is a deprecated compatibility surface, never an active
        # experiment factor or candidate population.
        if self.guidanceCanVol or self.initGuidanceVol:
            warnings.warn(
                "Guidance inputs are deprecated and forced to zero",
                DeprecationWarning,
                stacklevel=2,
            )
        self.guidanceCanVol = 0
        self.initGuidanceVol = 0
        
        # For automated model excution once uploaded to a cloud-based computing platform
        phase_text = str(phase)
        phase_dir = (
            phase_text
            if os.path.isabs(phase_text)
            else os.path.join(PROJECT_ROOT, "runs", phase_text)
        )
        if run_tag:
            phase_dir = os.path.join(phase_dir, str(run_tag))
        self.run_dir = os.path.join(phase_dir, f"rep_{int(replication):03d}_{machine}")
        os.makedirs(self.run_dir, exist_ok = True)
        
        # Reuse static map infrastructure across replications that share
        # the same city/address. Only dynamic agents (pedestrians/hazards)
        # are regenerated each replication.
        map_spec = {
            "address": str(self.address).strip(),
            "query_mode": str(self.mapQueryMode).strip().lower(),
            "center": (
                None
                if self.mapCenterLat is None or self.mapCenterLon is None
                else [float(self.mapCenterLat), float(self.mapCenterLon)]
            ),
            "radius_m": None if self.mapRadiusM is None else float(self.mapRadiusM),
            "network_type": "walk",
            "intersection_consolidation": {
                "enabled": bool(self.intersectionConsolidationEnabled),
                "tolerance_m": float(self.intersectionConsolidationToleranceM),
            },
        }
        infra_cache_key = json.dumps(map_spec, sort_keys=True, separators=(",", ":"))
        cached_infra = Core._prepared_infra_cache.get(infra_cache_key)
        if cached_infra is None:
            self.OSMProcessor = OSMProcessor(
                self.address,
                query_mode=self.mapQueryMode,
                center_point=(
                    None
                    if self.mapCenterLat is None or self.mapCenterLon is None
                    else (float(self.mapCenterLat), float(self.mapCenterLon))
                ),
                radius_m=self.mapRadiusM,
                verbose=self.verbose,
            )

            # Call relevant OSMProcessor functions in order
            # Extract all relevant map data and setup the node, edges, intersection, buildings dataset
            # locationDrive is the overall container of all map data
            self.OSMProcessor.setLocationDrive()
            print("Network geometry extracted")
            if self.intersectionConsolidationEnabled:
                self.OSMProcessor.consolidateIntersections(
                    self.intersectionConsolidationToleranceM
                )
            # NOTE: OSMProcessor.setNetworkFeature() is intentionally skipped here:
            # it is not consumed downstream in the simulation path and is very expensive.
            # formally establish the node and edge set
            self.OSMProcessor.setNodeEdgeSets()
            print("Nodes and edge sets extracted")
            # get intersection and building sets ready
            self.OSMProcessor.setIntersectionStreetCount()
            self.OSMProcessor.setBuildingOnly()
            self.OSMProcessor.setIntersectionOnly()

            # Keep one prepared static infrastructure object and reuse it.
            # This avoids repeated download/sorting and also avoids expensive
            # deep-copy of large NetworkX/GeoPandas objects per replication.
            Core._prepared_infra_cache[infra_cache_key] = self.OSMProcessor
            print(f"[INFRA CACHE] Prepared static map infrastructure for {map_spec}")
        else:
            self.OSMProcessor = cached_infra
            self.OSMProcessor.verbose = bool(self.verbose)
            print(f"[INFRA CACHE] Reusing static map infrastructure for {map_spec}")
        
        # Initialize instances of processing modules
        self.mapDS = MapDS(
            self.OSMProcessor.nodeList,
            self.OSMProcessor.edgeList,
            self.address,
            self.OSMProcessor.locationDrive,
            effective_walkway_width_m=self.congestionEffectiveWidthM,
            jam_density_ped_per_m2=self.congestionJamDensityPedPerM2,
        )
        self.congestionModel = PedestrianCongestionModel(
            enabled=self.congestionEnabled,
            effective_width_m=self.congestionEffectiveWidthM,
            jam_density_ped_per_m2=self.congestionJamDensityPedPerM2,
            shape=self.congestionShape,
            minimum_speed_ratio=self.congestionMinimumSpeedRatio,
            integration_substep_seconds=self.congestionSubstepSeconds,
        )
        self.pedDS = PedDS(
            self.pedVol,
            maximum_group_size=self.pedestrianGroupSize,
        )
        self.pedDS.set_hazard_random_seed(
            self.random_stream_seeds["pedestrian_hazard_outcomes"]
        )
        self.pedDS.configure_panic(
            rate=self.panicRate,
            herd_probability=self.panicHerdProbability,
            danger_threshold=self.panicDangerThreshold,
            random_seed=self.random_stream_seeds["pedestrian_panic_behavior"],
        )
        self.hazardDS = HazardDS(
            self.hazardVol,
            self.hazardCasualtyRate,
            self.hazardSpreadRate,
            self.hazardSpeedReduct,
            rng=np.random.default_rng(self.random_stream_seeds["hazard_evolution"]),
            wind_speed_m_per_minute=self.hazardWindSpeedMPerMinute,
            wind_direction_degrees=self.hazardWindDirectionDegrees,
            wind_influence=self.hazardWindInfluence,
            time_step_minutes=self.timeStepMinutes,
        )
        self.hazardDS.cell_state_evolution_mode = str(self.hazardEvolutionMode).strip().lower()
        if self.hazardDS.cell_state_evolution_mode not in {"deterministic", "stochastic"}:
            raise ValueError(
                "hazardEvolutionMode must be either 'deterministic' or 'stochastic'"
            )
        self.cellTracker = CellTracker(self.cellX, self.cellY)
        self.forceTracker = ForceProcessor(
            self_coefficient=self.socialForceSelfCoefficient,
            impact_coefficient=self.socialForceImpactCoefficient,
        )
        self.shelterDS = ShelterDS(self.shelterCanVol, self.initShelterVol)
        
        # Call initializing functions here, functions needed at t = 0 (follow old model, with additions)
        self.mapDS.computeConvertUnit()
        self.mapDS.boundarySetter()
        
        self.hazardDS.setCellTracker(self.cellTracker)

        
        partition_cache_key = (
            infra_cache_key,
            int(self.cellX),
            int(self.cellY),
            str(self.cellPartitionMode),
            float(self.cellPartitionMinWidthFraction),
        )
        cached_partition = Core._cell_partition_cache.get(partition_cache_key)

        if cached_partition is None:
            x_vals = []
            y_vals = []
            for _, data in self.OSMProcessor.nodeList:
                lon = float(data["x"])
                lat = float(data["y"])
                x_m, y_m = self.mapDS.coordToMeters(lon, lat)
                x_vals.append(float(x_m))
                y_vals.append(float(y_m))

            partition = build_cell_partition(
                x_vals,
                y_vals,
                int(self.cellX),
                int(self.cellY),
                str(self.cellPartitionMode),
                min_width_fraction=float(self.cellPartitionMinWidthFraction),
            )
            x_edges = list(partition["x_edges"])
            y_edges = list(partition["y_edges"])
            xMin, xMax = float(x_edges[0]), float(x_edges[-1])
            yMin, yMax = float(y_edges[0]), float(y_edges[-1])
            xLength = float(xMax - xMin)
            yLength = float(yMax - yMin)

            cached_partition = {
                "xLength": xLength,
                "yLength": yLength,
                "xMin": xMin,
                "xMax": xMax,
                "yMin": yMin,
                "yMax": yMax,
                "x_edges": x_edges,
                "y_edges": y_edges,
                "diagnostics": dict(partition["diagnostics"]),
            }
            Core._cell_partition_cache[partition_cache_key] = cached_partition
            print(f"[INFRA CACHE] Prepared cell partitions for {partition_cache_key}")
        else:
            xLength = float(cached_partition["xLength"])
            yLength = float(cached_partition["yLength"])
            xMin = float(cached_partition["xMin"])
            xMax = float(cached_partition["xMax"])
            yMin = float(cached_partition["yMin"])
            yMax = float(cached_partition["yMax"])
            x_edges = list(cached_partition["x_edges"])
            y_edges = list(cached_partition["y_edges"])
            print(f"[INFRA CACHE] Reusing cell partitions for {partition_cache_key}")

        self.cellPartitionDiagnostics = dict(cached_partition["diagnostics"])

        print("network X span (occupied): ", xLength)
        print("network Y span (occupied): ", yLength)
        print("network X range (occupied): ", (xMin, xMax))
        print("network Y range (occupied): ", (yMin, yMax))
        
        self.cellTracker.initialCut(xLength, yLength, xEdges = x_edges, yEdges = y_edges)
        
        N = int(self.cellX * self.cellY)
        def _init_vec(name):
            if getattr(self.cellTracker, name, None) is None:
                setattr(self.cellTracker, name, np.zeros(N, dtype=float))
        
        for wire in [
            "countByCell",
            "avgVelocityByCell",
            "heatByCell",
            "smokeByCell",
            "dangerLevelByCell",
            "shelterFulfillByCell",
            "wellnessPenaltyByCell",
        ]:
            _init_vec(wire)

        self.mapDS.nodeInit(self.cellTracker)
        self.mapDS.edgeInit(self.cellTracker)
        self.mapDS.buildEdgeIndices()
        self.mapDS.computeNodeCapSum()
        self.congestionModel.bind_edges(self.mapDS.edgeListByLocalID.values())
        
        self.forceTracker.setupCellTracker(self.cellTracker)
        
        self.shelterDS.shelterCanList = self.mapDS.shelterCanList
        print(f"Shelter candidates detected: {len(self.shelterDS.shelterCanList)}")
        if not self.shelterDS.shelterCanList:
            raise RuntimeError(
                "The selected OSM study area contains no recognized shelter candidates. "
                "Verify that the matching building/amenity query was downloaded and that "
                f"the city profile is correct: {map_spec}"
            )
                
        self.shelterDS.deploymentCapacityToken = (
            None
            if int(self.shelterCapacityToken) == 0
            else int(self.shelterCapacityToken)
        )
        self.shelterDS.shelterPerCell(self.cellTracker, self.cellX, self.cellY)
        available_shelters = int(self.shelterDS.remainingCandidateCount())
        requested_initial_shelters = int(self.initShelterVol)
        shelter_action_interval = int(self.shelterActionInterval)
        deploy_windows = 0
        if int(self.stopTime) > 1:
            deploy_windows = ((int(self.stopTime) - 2) // shelter_action_interval) + 1
        self.targetActiveShelters = self._shelter_target(
            available_candidates=available_shelters,
            initial_shelters=self.initShelterVol,
            decision_windows=deploy_windows,
            maximum_additions=self.maxAdditionalShelters,
        )
        strategy_key = str(deployment_strategy).strip().lower()
        if strategy_key == "none":
            self.shelterDS.initVol = int(self.targetActiveShelters)
            print(
                f"[SHELTER FAIRNESS] strategy={deployment_strategy} pre-deploying "
                f"{self.shelterDS.initVol} initial shelters to match end-of-episode shelter budget."
            )
        self.shelterDS.initShelter()
        initialized_shelter_ids = sorted(int(value) for value in self.shelterDS.shelterList)
        self.baseline_initial_shelter_ids = frozenset(
            initialized_shelter_ids[:requested_initial_shelters]
        )
        predeployed_ids = list(initialized_shelter_ids[requested_initial_shelters:])
        if strategy_key == "static_greedy":
            node_mass = np.zeros(int(self.cellX * self.cellY), dtype=float)
            for node in self.mapDS.nodeListByLocalID.values():
                cell = getattr(node, "cellID", None)
                if cell is None:
                    continue
                i, j = int(cell[0]), int(cell[1])
                if 0 <= i < int(self.cellX) and 0 <= j < int(self.cellY):
                    node_mass[i * int(self.cellY) + j] += 1.0
            if float(node_mass.sum()) <= 0.0:
                node_mass[:] = 1.0
            expected_initial_demand = (
                float(self.pedVol) * node_mass / float(node_mass.sum())
            )
            additions = max(
                0,
                int(self.targetActiveShelters) - len(self.shelterDS.shelterList),
            )
            predeployed_ids.extend(
                self.shelterDS.predeployStaticDemandGreedy(
                    expected_initial_demand,
                    additions,
                    self.cellTracker,
                )
            )
            print(
                f"[SHELTER FAIRNESS] strategy=static_greedy pre-deployed "
                f"{len(predeployed_ids)} expected-demand sites at t=0.",
                flush=True,
            )
        self.static_predeployment_shelter_ids = frozenset(predeployed_ids)
        for shelter in self.shelterDS.shelterList.values():
            self.cellTracker.addShelter(shelter.cellLocated, shelter)
        if self.verbose:
            print("Shelters list: ", self.shelterDS.shelterList)
        
        self.hazardDS.initHazard(self.mapDS, self.cellTracker)
        self._hazard_trajectory_hasher = hashlib.sha256()
        self._record_hazard_state(0)
        
        self.pedDS.initPedestrianAgent(self.mapDS, self.cellTracker, self.maxSpeed)
        
        self.shelterDS.shelterByOSMID   = {sh.nodeMapped.OSMID: sh for sh in self.shelterDS.shelterList.values()}    
        
        self.pedDS.checkReady(mapDS = self.mapDS,
                              cellTracker = self.cellTracker,
                              maxSpeed = self.maxSpeed,
                              hazardDS = self.hazardDS,
                              shelterDS = self.shelterDS,
                              forceTracker = self.forceTracker,
                              congestionModel = self.congestionModel,
                              timeStepMinutes = self.timeStepMinutes,
                              casualtyReferenceExposureMinutes = (
                                  self.hazardCasualtyReferenceMinutes
                              ),
                              evacuationHorizonTimesteps = self.stopTime)
        self.pedDS.route_active_to_nearest_shelter()

        self.rl = RLBridge(
            self,
            train_mode = train_mode,
            lr = float(self.learningRate),
            actor_lr = float(self.actorLearningRate),
            critic_lr = float(self.criticLearningRate),
            gamma = 1.0,
            clip_eps = DEFAULT_CLIP_EPS,
            epochs = int(self.criticPpoEpochs),
            actor_epochs = int(self.actorPpoEpochs),
            critic_epochs = int(self.criticPpoEpochs),
            actor_baseline_decay = float(self.actorBaselineDecay),
            advantage_scale_floor = float(self.advantageScaleFloor),
            minibatch_size = 32,
            entropy_coef = DEFAULT_ENTROPY_COEF,
            target_kl = DEFAULT_TARGET_KL,
            rollout_episodes = int(self.ppoRolloutEpisodes),
            shelter_action_interval = int(self.shelterActionInterval),
            optimizer_name = self.optimizer,
            deployment_strategy = deployment_strategy,
            target_active_shelters = int(self.targetActiveShelters),
            policy_seed = self.policy_seed,
            checkpoint_path = checkpoint_path,
            diagnostics_path = diagnostics_path,
            counterfactual_credit=bool(self.nmccEnabled),
            counterfactual_horizon=(
                None
                if int(self.nmccCounterfactualHorizon) == 0
                else int(self.nmccCounterfactualHorizon)
            ),
            counterfactual_weight=float(self.nmccCounterfactualWeight),
            nmcc_joint_counterfactual_weight=float(
                self.nmccJointCounterfactualWeight
            ),
            counterfactual_intervention_cost=float(self.nmccInterventionCost),
            nmcc_ensemble_size=int(self.nmccEnsembleSize),
            nmcc_natural_loss_coef=float(self.nmccNaturalLossCoefficient),
            nmcc_causal_loss_coef=float(self.nmccCausalLossCoefficient),
            nmcc_dueling_loss_coef=float(self.nmccDuelingLossCoefficient),
            nmcc_teacher_coef=float(self.nmccTeacherCoefficient),
            nmcc_teacher_decay_updates=int(self.nmccTeacherDecayUpdates),
            nmcc_guidance_max=float(self.nmccGuidanceMaximum),
            nmcc_guidance_warmup_updates=int(self.nmccGuidanceWarmupUpdates),
            nmcc_guidance_ramp_updates=int(self.nmccGuidanceRampUpdates),
            nmcc_uncertainty_penalty=float(self.nmccUncertaintyPenalty),
            entropy_coef_end=float(self.entropyCoefficientEnd),
            exploration_decay_updates=int(self.explorationDecayUpdates),
            action_temperature_start=float(self.actionTemperatureStart),
            action_temperature_end=float(self.actionTemperatureEnd),
            nmcc_natural_pretrain_rollouts=int(
                self.nmccNaturalPretrainRollouts
            ),
            nmcc_causal_pretrain_rollouts=int(
                self.nmccCausalPretrainRollouts
            ),
            nmcc_controller_warmup_rollouts=int(
                self.nmccControllerWarmupRollouts
            ),
            **{
                keyword: getattr(self, attribute)
                for attribute, keyword, _, _ in NMCC_PI_CORE_FIELDS
            },
        )
        if strategy_key == "rl_precommit":
            receipts = self.rl.precommit_all()
            self.static_predeployment_shelter_ids = frozenset(
                int(receipt.shelter_id) for receipt in receipts
            )
            print(
                f"[SHELTER FAIRNESS] strategy=rl_precommit pre-deployed "
                f"{len(receipts)} learned-policy sites at t=0.",
                flush=True,
            )
        self.logger = trainingLog(run_dir = self.run_dir, window = 100, use_tensorboard = False)

        if self.visualization_enabled:
            self.visualizer = EvacuationVisualizer(
                self,
                milestones=self.visualization_milestones,
                render_individual_snapshots=self.visualization_individual_snapshots,
                render_vector_outputs=self.visualization_vector_outputs,
            )
            # This is the initialized state before any hazard, movement, or
            # shelter-deployment transition has occurred.
            self.visualizer.observe(0)

        run_metadata = {
            "schema_version": 5,
            "city_id": str(self.cityID),
            "map_spec": map_spec,
            "map_provenance": self.OSMProcessor.graph_provenance(),
            "replication": int(replication),
            "machine": str(machine),
            "phase": str(phase),
            "deployment_strategy": str(deployment_strategy),
            "train_mode": bool(train_mode),
            "scenario_seed": self.scenario_seed,
            "policy_seed": self.policy_seed,
            "random_stream_seeds": dict(self.random_stream_seeds),
            "hazard_evolution_mode": self.hazardDS.cell_state_evolution_mode,
            "initial_population": self.rl.initial_population,
            "grid_shape": [int(self.cellX), int(self.cellY)],
            "cell_partition": {
                **dict(self.cellPartitionDiagnostics),
                "minimum_axis_width_fraction": float(
                    self.cellPartitionMinWidthFraction
                ),
            },
            "horizon": int(self.stopTime) - 1,
            "time_step_minutes": float(self.timeStepMinutes),
            "horizon_minutes": (int(self.stopTime) - 1) * float(self.timeStepMinutes),
            "shelter_action_interval_timesteps": int(self.shelterActionInterval),
            "shelter_action_interval_minutes": (
                int(self.shelterActionInterval) * float(self.timeStepMinutes)
            ),
            "maximum_shelter_forecast_danger": float(
                self.maximumShelterForecastDanger
            ),
            "candidate_action_constraints": {
                "require_operational_benefit": bool(
                    self.requireCandidateOperationalBenefit
                ),
                "minimum_reroutable_fraction": float(
                    self.minimumCandidateReroutableFraction
                ),
                "minimum_route_time_saving": float(
                    self.minimumCandidateRouteTimeSaving
                ),
                "minimum_hazard_safety_margin": float(
                    self.minimumCandidateHazardSafetyMargin
                ),
            },
            "free_flow_speed_m_per_minute": float(self.maxSpeed),
            "free_flow_speed_m_per_second": float(self.maxSpeed) / 60.0,
            "congestion": self.congestionModel.contract(),
            "social_force": self.forceTracker.contract(),
            "panic_behavior": self.pedDS.panic_contract(),
            "initial_shelters": int(len(self.shelterDS.shelterList)),
            "maximum_dynamic_deployments": int(self.rl.maximum_deployments),
            "checkpoint_path": (
                self.rl.checkpoint_path
                if strategy_key in {"rl", "rl_precommit"}
                else None
            ),
            "ppo_rollout_episodes": int(self.rl.rollout_episodes),
            "reward": "(safe_completions - 3*casualties)/population - risk_weighted_person_time/(population*horizon)",
            "visualization_enabled": bool(self.visualization_enabled),
            "visualization_milestones": (
                list(self.visualizer.milestones) if self.visualizer is not None else []
            ),
        }
        with open(os.path.join(self.run_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(run_metadata, handle, indent=2, sort_keys=True)
        
        self.simulationEnumerator()
        
    """Main Functions"""
    def _apply_configuration_overrides(self, overrides) -> None:
        unknown = sorted(key for key in overrides if not hasattr(self, key))
        if unknown:
            raise KeyError(f"Unknown configuration override(s): {unknown}")
        for key, value in overrides.items():
            setattr(self, key, value)

    def _effective_configuration(self) -> dict:
        return {
            "city_id": str(self.cityID),
            "address": str(self.address),
            "map_query_mode": str(self.mapQueryMode),
            "map_center": (
                None
                if self.mapCenterLat is None or self.mapCenterLon is None
                else [float(self.mapCenterLat), float(self.mapCenterLon)]
            ),
            "map_radius_m": None if self.mapRadiusM is None else float(self.mapRadiusM),
            "stop_time": int(self.stopTime),
            "pedestrians": int(self.pedVol),
            "population_representation": {
                "mode": (
                    "individual"
                    if int(self.pedestrianGroupSize) == 1
                    else "weighted_cohort"
                ),
                "maximum_persons_per_agent": int(self.pedestrianGroupSize),
                "initialized_agent_count": int(
                    math.ceil(float(self.pedVol) / float(self.pedestrianGroupSize))
                ),
            },
            "hazards": int(self.hazardVol),
            "maximum_speed": float(self.maxSpeed),
            "maximum_speed_units": "metres_per_minute",
            "time_step_minutes": float(self.timeStepMinutes),
            "congestion": {
                "enabled": bool(self.congestionEnabled),
                "model": PedestrianCongestionModel.MODEL_NAME,
                "effective_width_m": float(self.congestionEffectiveWidthM),
                "jam_density_ped_per_m2": float(self.congestionJamDensityPedPerM2),
                "shape": float(self.congestionShape),
                "minimum_speed_ratio": float(self.congestionMinimumSpeedRatio),
                "integration_substep_seconds": float(self.congestionSubstepSeconds),
            },
            "social_force": {
                "enabled": bool(self.socialForceEnabled),
                "self_coefficient": float(self.socialForceSelfCoefficient),
                "impact_coefficient": float(self.socialForceImpactCoefficient),
            },
            "intersection_consolidation": {
                "enabled": bool(self.intersectionConsolidationEnabled),
                "tolerance_m": float(self.intersectionConsolidationToleranceM),
            },
            "panic": {
                "rate": float(self.panicRate),
                "danger_threshold": int(self.panicDangerThreshold),
                "herd_probability": float(self.panicHerdProbability),
                "random_probability": float(1.0 - self.panicHerdProbability),
                "persistent": True,
            },
            "grid": [int(self.cellX), int(self.cellY)],
            "cell_partition": {
                "mode": str(self.cellPartitionMode),
                "minimum_axis_width_fraction": float(
                    self.cellPartitionMinWidthFraction
                ),
            },
            "shelter_candidates": int(self.shelterCanVol),
            "maximum_additional_shelters": int(self.maxAdditionalShelters),
            "shelter_action_interval_timesteps": int(self.shelterActionInterval),
            "shelter_action_interval_minutes": (
                int(self.shelterActionInterval) * float(self.timeStepMinutes)
            ),
            "initial_shelters": int(self.initShelterVol),
            "shelter_capacity_token": int(self.shelterCapacityToken),
            "hazard_casualty_distribution": list(self.hazardCasualtyRate),
            "hazard_casualty_reference_exposure_minutes": float(
                self.hazardCasualtyReferenceMinutes
            ),
            "hazard_spread_distribution": list(self.hazardSpreadRate),
            "hazard_speed_distribution": list(self.hazardSpeedReduct),
            "hazard_evolution_mode": str(self.hazardEvolutionMode),
            "hazard_wind": {
                "speed_m_per_minute": float(self.hazardWindSpeedMPerMinute),
                "direction_degrees_from_east": float(
                    self.hazardWindDirectionDegrees
                ),
                "spread_influence": float(self.hazardWindInfluence),
            },
            "ppo_learning_rate": float(self.learningRate),
            "actor_learning_rate": float(self.actorLearningRate),
            "critic_learning_rate": float(self.criticLearningRate),
            "actor_ppo_epochs": int(self.actorPpoEpochs),
            "critic_ppo_epochs": int(self.criticPpoEpochs),
            "actor_baseline_decay": float(self.actorBaselineDecay),
            "advantage_scale_floor": float(self.advantageScaleFloor),
            "actor_credit_target": (
                "nmcc_policy_improvement_exact_branch_target"
                if bool(getattr(self, "nmccPolicyImprovement"))
                else "complete_episode_smdp_monte_carlo_return_to_go"
            ),
            "policy_improvement_learner": {
                keyword: getattr(self, attribute)
                for attribute, keyword, _, _ in NMCC_PI_CORE_FIELDS
            },
            "critic_target": "one_step_smdp_td0",
            "ppo_optimizer": str(self.optimizer),
            "ppo_rollout_episodes": int(self.ppoRolloutEpisodes),
            "ppo_finalize_rollout": bool(self.finalizePpoRollout),
            "epsilon_exploration": {
                "legacy": float(self.explorationRate),
                "start": float(self.explorationRateStart),
                "end": float(self.explorationRateEnd),
                "decay_updates": int(self.explorationDecayUpdates),
            },
            "nmcc": {
                "enabled": bool(self.nmccEnabled),
                "counterfactual_horizon_timesteps": int(
                    self.nmccCounterfactualHorizon
                ),
                "counterfactual_weight": float(self.nmccCounterfactualWeight),
                "joint_counterfactual_weight": float(
                    self.nmccJointCounterfactualWeight
                ),
                "intervention_cost": float(self.nmccInterventionCost),
                "ensemble_size": int(self.nmccEnsembleSize),
                "natural_loss_coefficient": float(
                    self.nmccNaturalLossCoefficient
                ),
                "causal_loss_coefficient": float(
                    self.nmccCausalLossCoefficient
                ),
                "dueling_loss_coefficient": float(
                    self.nmccDuelingLossCoefficient
                ),
                "teacher_coefficient": float(self.nmccTeacherCoefficient),
                "teacher_decay_updates": int(self.nmccTeacherDecayUpdates),
                "guidance_maximum": float(self.nmccGuidanceMaximum),
                "guidance_warmup_updates": int(
                    self.nmccGuidanceWarmupUpdates
                ),
                "guidance_ramp_updates": int(self.nmccGuidanceRampUpdates),
                "uncertainty_penalty": float(self.nmccUncertaintyPenalty),
                "staged_rollouts": {
                    "natural_pretrain": int(
                        self.nmccNaturalPretrainRollouts
                    ),
                    "causal_pretrain": int(
                        self.nmccCausalPretrainRollouts
                    ),
                    "controller_warmup": int(
                        self.nmccControllerWarmupRollouts
                    ),
                },
            },
            "exploration_schedule": {
                "entropy_start": float(DEFAULT_ENTROPY_COEF),
                "entropy_end": float(self.entropyCoefficientEnd),
                "temperature_start": float(self.actionTemperatureStart),
                "temperature_end": float(self.actionTemperatureEnd),
                "decay_updates": int(self.explorationDecayUpdates),
            },
        }

    def _validate_effective_configuration(self) -> None:
        self.stopTime = int(self.stopTime)
        self.pedVol = int(self.pedVol)
        self.pedestrianGroupSize = int(self.pedestrianGroupSize)
        self.hazardVol = int(self.hazardVol)
        self.maxSpeed = float(self.maxSpeed)
        self.timeStepMinutes = float(self.timeStepMinutes)
        if not isinstance(self.congestionEnabled, (bool, np.bool_)):
            raise ValueError("congestionEnabled must be boolean")
        self.congestionEnabled = bool(self.congestionEnabled)
        self.congestionEffectiveWidthM = float(self.congestionEffectiveWidthM)
        self.congestionJamDensityPedPerM2 = float(self.congestionJamDensityPedPerM2)
        self.congestionShape = float(self.congestionShape)
        self.congestionMinimumSpeedRatio = float(self.congestionMinimumSpeedRatio)
        self.congestionSubstepSeconds = float(self.congestionSubstepSeconds)
        if not isinstance(self.socialForceEnabled, (bool, np.bool_)):
            raise ValueError("socialForceEnabled must be boolean")
        self.socialForceEnabled = bool(self.socialForceEnabled)
        self.socialForceSelfCoefficient = float(self.socialForceSelfCoefficient)
        self.socialForceImpactCoefficient = float(self.socialForceImpactCoefficient)
        if not isinstance(self.intersectionConsolidationEnabled, (bool, np.bool_)):
            raise ValueError("intersectionConsolidationEnabled must be boolean")
        self.intersectionConsolidationEnabled = bool(self.intersectionConsolidationEnabled)
        self.intersectionConsolidationToleranceM = float(
            self.intersectionConsolidationToleranceM
        )
        self.panicRate = float(self.panicRate)
        self.panicHerdProbability = float(self.panicHerdProbability)
        self.panicDangerThreshold = int(self.panicDangerThreshold)
        self.cellX = int(self.cellX)
        self.cellY = int(self.cellY)
        self.cellPartitionMode = normalize_partition_mode(self.cellPartitionMode)
        self.cellPartitionMinWidthFraction = float(
            self.cellPartitionMinWidthFraction
        )
        self.shelterCanVol = int(self.shelterCanVol)
        self.initShelterVol = int(self.initShelterVol)
        self.shelterCapacityToken = int(self.shelterCapacityToken)
        self.maxAdditionalShelters = int(self.maxAdditionalShelters)
        self.shelterActionInterval = int(self.shelterActionInterval)
        self.maximumShelterForecastDanger = float(
            self.maximumShelterForecastDanger
        )
        if not isinstance(self.requireCandidateOperationalBenefit, (bool, np.bool_)):
            raise ValueError("requireCandidateOperationalBenefit must be boolean")
        self.requireCandidateOperationalBenefit = bool(
            self.requireCandidateOperationalBenefit
        )
        self.minimumCandidateReroutableFraction = float(
            self.minimumCandidateReroutableFraction
        )
        self.minimumCandidateRouteTimeSaving = float(
            self.minimumCandidateRouteTimeSaving
        )
        self.minimumCandidateHazardSafetyMargin = float(
            self.minimumCandidateHazardSafetyMargin
        )
        self.learningRate = float(self.learningRate)
        self.actorLearningRate = float(self.actorLearningRate)
        self.criticLearningRate = float(self.criticLearningRate)
        if self.criticLearningRate == 0.0:
            self.criticLearningRate = self.learningRate
        self.actorPpoEpochs = int(self.actorPpoEpochs)
        self.criticPpoEpochs = int(self.criticPpoEpochs)
        self.actorBaselineDecay = float(self.actorBaselineDecay)
        self.advantageScaleFloor = float(self.advantageScaleFloor)
        self.ppoRolloutEpisodes = int(self.ppoRolloutEpisodes)
        if not isinstance(self.finalizePpoRollout, (bool, np.bool_)):
            raise ValueError("finalizePpoRollout must be boolean")
        self.finalizePpoRollout = bool(self.finalizePpoRollout)
        self.explorationRate = float(self.explorationRate)
        if not isinstance(self.nmccEnabled, (bool, np.bool_)):
            raise ValueError("nmccEnabled must be boolean")
        self.nmccEnabled = bool(self.nmccEnabled)
        self.nmccCounterfactualHorizon = int(self.nmccCounterfactualHorizon)
        self.nmccCounterfactualWeight = float(self.nmccCounterfactualWeight)
        self.nmccJointCounterfactualWeight = float(
            self.nmccJointCounterfactualWeight
        )
        self.nmccInterventionCost = float(self.nmccInterventionCost)
        self.nmccEnsembleSize = int(self.nmccEnsembleSize)
        self.nmccNaturalLossCoefficient = float(self.nmccNaturalLossCoefficient)
        self.nmccCausalLossCoefficient = float(self.nmccCausalLossCoefficient)
        self.nmccDuelingLossCoefficient = float(self.nmccDuelingLossCoefficient)
        self.nmccTeacherCoefficient = float(self.nmccTeacherCoefficient)
        self.nmccTeacherDecayUpdates = int(self.nmccTeacherDecayUpdates)
        self.nmccGuidanceMaximum = float(self.nmccGuidanceMaximum)
        self.nmccGuidanceWarmupUpdates = int(self.nmccGuidanceWarmupUpdates)
        self.nmccGuidanceRampUpdates = int(self.nmccGuidanceRampUpdates)
        self.nmccUncertaintyPenalty = float(self.nmccUncertaintyPenalty)
        self.entropyCoefficientEnd = float(self.entropyCoefficientEnd)
        self.explorationDecayUpdates = int(self.explorationDecayUpdates)
        self.actionTemperatureStart = float(self.actionTemperatureStart)
        self.actionTemperatureEnd = float(self.actionTemperatureEnd)
        self.nmccNaturalPretrainRollouts = int(
            self.nmccNaturalPretrainRollouts
        )
        self.nmccCausalPretrainRollouts = int(self.nmccCausalPretrainRollouts)
        self.nmccControllerWarmupRollouts = int(
            self.nmccControllerWarmupRollouts
        )
        for attribute, _, kind, _ in NMCC_PI_CORE_FIELDS:
            value = getattr(self, attribute)
            if kind is bool:
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"{attribute} must be boolean")
                setattr(self, attribute, bool(value))
            elif kind is int:
                if isinstance(value, (bool, np.bool_)) or float(value) != int(value):
                    raise ValueError(f"{attribute} must be an integer")
                setattr(self, attribute, int(value))
            else:
                setattr(self, attribute, kind(value))
        self.optimizer = str(self.optimizer).strip()
        self.hazardEvolutionMode = str(self.hazardEvolutionMode).strip().lower()
        self.hazardCasualtyReferenceMinutes = float(
            self.hazardCasualtyReferenceMinutes
        )
        self.hazardWindSpeedMPerMinute = float(self.hazardWindSpeedMPerMinute)
        self.hazardWindDirectionDegrees = float(self.hazardWindDirectionDegrees)
        self.hazardWindInfluence = float(self.hazardWindInfluence)
        self.cityID = str(self.cityID).strip()
        self.mapQueryMode = str(self.mapQueryMode).strip().lower()
        if not str(self.address).strip():
            raise ValueError("address must not be empty")
        if not self.cityID:
            raise ValueError("cityID must not be empty")
        if self.mapQueryMode not in {"place", "point"}:
            raise ValueError("mapQueryMode must be either 'place' or 'point'")
        if self.mapQueryMode == "point":
            if self.mapCenterLat is None or self.mapCenterLon is None or self.mapRadiusM is None:
                raise ValueError(
                    "point map queries require mapCenterLat, mapCenterLon, and mapRadiusM"
                )
            self.mapCenterLat = float(self.mapCenterLat)
            self.mapCenterLon = float(self.mapCenterLon)
            self.mapRadiusM = float(self.mapRadiusM)
            if not math.isfinite(self.mapCenterLat) or not -90.0 <= self.mapCenterLat <= 90.0:
                raise ValueError("mapCenterLat must lie in [-90, 90]")
            if not math.isfinite(self.mapCenterLon) or not -180.0 <= self.mapCenterLon <= 180.0:
                raise ValueError("mapCenterLon must lie in [-180, 180]")
            if not math.isfinite(self.mapRadiusM) or self.mapRadiusM <= 0.0:
                raise ValueError("mapRadiusM must be finite and positive")
        else:
            self.mapCenterLat = None
            self.mapCenterLon = None
            self.mapRadiusM = None
        if self.stopTime < 2:
            raise ValueError("stopTime must be at least 2")
        if self.pedVol <= 0:
            raise ValueError("pedVol must be positive")
        if self.pedestrianGroupSize <= 0:
            raise ValueError("pedestrianGroupSize must be positive")
        if self.panicRate > 0.0 and self.pedestrianGroupSize != 1:
            raise ValueError(
                "panicRate > 0 requires pedestrianGroupSize=1 so panic and herding "
                "are computed for every active pedestrian"
            )
        if self.hazardVol < 0:
            raise ValueError("hazardVol must be non-negative")
        if (
            not math.isfinite(self.hazardCasualtyReferenceMinutes)
            or self.hazardCasualtyReferenceMinutes <= 0.0
        ):
            raise ValueError(
                "hazardCasualtyReferenceMinutes must be finite and positive"
            )
        if (
            not math.isfinite(self.hazardWindSpeedMPerMinute)
            or self.hazardWindSpeedMPerMinute < 0.0
        ):
            raise ValueError("hazardWindSpeedMPerMinute must be finite and non-negative")
        if not math.isfinite(self.hazardWindDirectionDegrees):
            raise ValueError("hazardWindDirectionDegrees must be finite")
        self.hazardWindDirectionDegrees %= 360.0
        if not math.isfinite(self.hazardWindInfluence) or self.hazardWindInfluence < 0.0:
            raise ValueError("hazardWindInfluence must be finite and non-negative")
        if not math.isfinite(self.maxSpeed) or self.maxSpeed <= 0.0:
            raise ValueError("maxSpeed must be finite and positive")
        if not math.isfinite(self.timeStepMinutes) or self.timeStepMinutes <= 0.0:
            raise ValueError("timeStepMinutes must be finite and positive")
        for name, value in (
            ("congestionEffectiveWidthM", self.congestionEffectiveWidthM),
            ("congestionJamDensityPedPerM2", self.congestionJamDensityPedPerM2),
            ("congestionShape", self.congestionShape),
            ("congestionSubstepSeconds", self.congestionSubstepSeconds),
            ("intersectionConsolidationToleranceM", self.intersectionConsolidationToleranceM),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.congestionMinimumSpeedRatio)
            or not 0.0 <= self.congestionMinimumSpeedRatio < 1.0
        ):
            raise ValueError(
                "congestionMinimumSpeedRatio must be finite and in [0, 1)"
            )
        for name, value in (
            ("socialForceSelfCoefficient", self.socialForceSelfCoefficient),
            ("socialForceImpactCoefficient", self.socialForceImpactCoefficient),
            ("panicRate", self.panicRate),
            ("panicHerdProbability", self.panicHerdProbability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not self.socialForceEnabled:
            raise ValueError("socialForceEnabled must remain true for the active model")
        if not 0 <= self.panicDangerThreshold <= 5:
            raise ValueError("panicDangerThreshold must lie in [0, 5]")
        if self.cellX <= 0 or self.cellY <= 0:
            raise ValueError("cellX and cellY must be positive")
        if (
            not math.isfinite(self.cellPartitionMinWidthFraction)
            or self.cellPartitionMinWidthFraction <= 0.0
            or self.cellPartitionMinWidthFraction * max(self.cellX, self.cellY) >= 1.0
        ):
            raise ValueError(
                "cellPartitionMinWidthFraction must be finite, positive, and "
                "smaller than 1/max(cellX, cellY)"
            )
        if self.shelterCanVol <= 0:
            raise ValueError("shelterCanVol must be positive")
        if not 0 <= self.initShelterVol <= self.shelterCanVol:
            raise ValueError("initShelterVol must lie in [0, shelterCanVol]")
        if self.shelterCapacityToken < 0:
            raise ValueError("shelterCapacityToken must be non-negative")
        if self.maxAdditionalShelters < 0:
            raise ValueError("maxAdditionalShelters must be non-negative")
        if self.shelterActionInterval <= 0:
            raise ValueError("shelterActionInterval must be positive")
        if (
            not math.isfinite(self.maximumShelterForecastDanger)
            or not 0.0 <= self.maximumShelterForecastDanger <= 1.0
        ):
            raise ValueError(
                "maximumShelterForecastDanger must be finite and in [0, 1]"
            )
        for name, value in (
            ("minimumCandidateReroutableFraction", self.minimumCandidateReroutableFraction),
            ("minimumCandidateRouteTimeSaving", self.minimumCandidateRouteTimeSaving),
            ("minimumCandidateHazardSafetyMargin", self.minimumCandidateHazardSafetyMargin),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not math.isfinite(self.learningRate) or not 0.0 < self.learningRate <= 1e-2:
            raise ValueError("learningRate must be in (0, 0.01] for PPO")
        for name, value in (
            ("actorLearningRate", self.actorLearningRate),
            ("criticLearningRate", self.criticLearningRate),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1e-2:
                raise ValueError(f"{name} must be in (0, 0.01]")
        if self.actorPpoEpochs <= 0 or self.criticPpoEpochs <= 0:
            raise ValueError("actorPpoEpochs and criticPpoEpochs must be positive")
        if not math.isfinite(self.actorBaselineDecay) or not 0.0 <= self.actorBaselineDecay < 1.0:
            raise ValueError("actorBaselineDecay must be in [0, 1)")
        if not math.isfinite(self.advantageScaleFloor) or self.advantageScaleFloor <= 0.0:
            raise ValueError("advantageScaleFloor must be finite and positive")
        if self.ppoRolloutEpisodes <= 0:
            raise ValueError("ppoRolloutEpisodes must be positive")
        if self.explorationRate != 0.0:
            raise ValueError(
                "explorationRate must be 0 for on-policy PPO; exploration is supplied by "
                "the categorical policy entropy"
            )
        if self.nmccCounterfactualHorizon < 0:
            raise ValueError("nmccCounterfactualHorizon must be non-negative")
        if not 0.0 <= self.nmccCounterfactualWeight <= 1.0:
            raise ValueError("nmccCounterfactualWeight must lie in [0, 1]")
        if not 0.0 <= self.nmccJointCounterfactualWeight <= 1.0:
            raise ValueError("nmccJointCounterfactualWeight must lie in [0, 1]")
        staged_rollouts = (
            self.nmccNaturalPretrainRollouts,
            self.nmccCausalPretrainRollouts,
            self.nmccControllerWarmupRollouts,
        )
        if any(value < 0 for value in staged_rollouts):
            raise ValueError("NMCC staged rollout counts must be non-negative")
        if any(staged_rollouts) and not (
            self.nmccEnabled or getattr(self, "nmccPolicyImprovement")
        ):
            raise ValueError(
                "NMCC staged rollouts require nmccEnabled or nmccPolicyImprovement"
            )
        if getattr(self, "nmccPolicyImprovement"):
            if self.nmccGuidanceMaximum != 0.0:
                raise ValueError("nmccPolicyImprovement requires nmccGuidanceMaximum=0")
        nonnegative_nmcc = {
            "nmccInterventionCost": self.nmccInterventionCost,
            "nmccNaturalLossCoefficient": self.nmccNaturalLossCoefficient,
            "nmccCausalLossCoefficient": self.nmccCausalLossCoefficient,
            "nmccDuelingLossCoefficient": self.nmccDuelingLossCoefficient,
            "nmccTeacherCoefficient": self.nmccTeacherCoefficient,
            "nmccGuidanceMaximum": self.nmccGuidanceMaximum,
            "nmccUncertaintyPenalty": self.nmccUncertaintyPenalty,
            "entropyCoefficientEnd": self.entropyCoefficientEnd,
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in nonnegative_nmcc.values()
        ):
            raise ValueError("NMCC coefficients must be finite and non-negative")
        if self.nmccEnsembleSize < 2:
            raise ValueError("nmccEnsembleSize must be at least two")
        if (
            self.nmccTeacherDecayUpdates <= 0
            or self.nmccGuidanceWarmupUpdates < 0
            or self.nmccGuidanceRampUpdates <= 0
            or self.explorationDecayUpdates <= 0
        ):
            raise ValueError("NMCC and exploration schedules are invalid")
        if not (
            0.0 < self.actionTemperatureEnd <= self.actionTemperatureStart
        ):
            raise ValueError("action temperatures must satisfy 0 < end <= start")
        if self.entropyCoefficientEnd > DEFAULT_ENTROPY_COEF:
            raise ValueError(
                "entropyCoefficientEnd cannot exceed the starting coefficient"
            )
        if self.optimizer.lower() not in {"adam", "adamw", "rmsprop"}:
            raise ValueError("optimizer must be Adam, AdamW, or RMSprop")
        if self.hazardEvolutionMode not in {"deterministic", "stochastic"}:
            raise ValueError("hazardEvolutionMode must be deterministic or stochastic")

    def _record_hazard_state(self, simulation_time: int) -> None:
        """Hash the exogenous hazard path for matched-policy verification."""
        if self._hazard_trajectory_hasher is None:
            raise RuntimeError("Hazard trajectory hasher is not initialized")
        self._hazard_trajectory_hasher.update(
            np.asarray([int(simulation_time)], dtype=np.int64).tobytes()
        )
        ordered_cells = sorted(self.cellTracker.cellList)
        states = np.asarray(
            [int(self.cellTracker.getCellState(cell)) for cell in ordered_cells],
            dtype=np.int8,
        )
        self._hazard_trajectory_hasher.update(states.tobytes())
        for hazard_id, hazard in sorted(self.hazardDS.hazardList.items()):
            values = np.asarray(
                [
                    int(hazard_id),
                    int(bool(getattr(hazard, "active", False))),
                    int(getattr(hazard, "age", 0)),
                ],
                dtype=np.int64,
            )
            self._hazard_trajectory_hasher.update(values.tobytes())

    """Read the input-defined parameter values accordingly from a CSV file"""
    def readInputCSV(self):
        line = 1
        with open(os.path.join(PROJECT_ROOT, "RLEvacuationParameter.csv")) as csvfile:
            parameterReader = csv.reader(csvfile)
            parameterList = list(parameterReader)
            for row in parameterList:
                # primary model parameters
                if line == 2:
                    self.stopTime = int(row[0])
                    self.address = str(row[1])
                    self.maxSpeed = int(row[2])
                    self.pedVol = int(row[3])
                    self.hazardVol = int(row[4])
                    self.cellX = int(row[5])
                    self.cellY = int(row[6])
                # guidance and shelter parameters
                elif line == 4:
                    self.guidanceCanVol = int(row[0])
                    self.shelterCanVol = int(row[1])
                    self.initShelterVol = int(row[2])
                    self.initGuidanceVol = int(row[3])
                # hazard parameters
                elif line == 6:
                    self.hazardCasualtyRate = [int(row[0]), int(row[1])]
                    self.hazardSpreadRate = [int(row[2]), int(row[3])]
                    self.hazardSpeedReduct = [int(row[4]), int(row[5])]
                # RL parameters
                elif line == 8:
                    self.learningRate = float(row[0])
                    self.explorationRate = float(row[1])
                    self.optimizer = str(row[2])
            
                line += 1
                
    """Controls the main execution of functions in the Simulator, all functionalities for t >= 1"""
    """Still need to add input parameters to function calls"""
    def simulationEnumerator(self):
        iterator = range(1, self.stopTime)
        last_simulation_time = 0
        
        # for output, visualized progress bar
        if _HAS_TQDM:
            iterator = tqdm(iterator, desc = "Sim timesteps", ncols = 120)
        
        for sim_time in iterator:
            last_simulation_time = int(sim_time)
            if not _HAS_TQDM:
                print("Current timestep is: ", sim_time)
                
            tmr = Timer() if self.profile_timing else None
                
            self.pedDS.startDocument()
            if tmr is not None: tmr.lap("startDocument")

            self.hazardDS.spreadUpdate()
            if tmr is not None: tmr.lap("hazard.spreadUpdate")

            self.hazardDS.heatUpdate()
            if tmr is not None: tmr.lap("hazard.heatUpdate")

            self.hazardDS.smokeUpdate()
            if tmr is not None: tmr.lap("hazard.smokeUpdate")

            self.hazardDS.terminateHazard()
            self._record_hazard_state(sim_time)
            if tmr is not None: tmr.lap("hazard.terminateHazard")

            # === PED/GU/SH LOOKUPS ===
            self.pedDS.loadShelterLookup(self.shelterDS.shelterByOSMID)
            if tmr is not None: tmr.lap("ped.loadShelterLookup")

            # === PEDESTRIAN INTERACTIONS ===
            self.pedDS.pedestrianHazardInteraction()
            if tmr is not None: tmr.lap("ped.hazardInteraction")

            self.pedDS.interPedestrianInteraction()
            if tmr is not None: tmr.lap("ped.interPedInteraction")

            self.pedDS.pedestrianNetworkInteraction()
            if tmr is not None: tmr.lap("ped.networkInteraction")

            # === CELL UPDATE ===
            self.cellTracker.cellUpdate(pedDS=self.pedDS, forceTracker=self.forceTracker)
            if tmr is not None: tmr.lap("cellTracker.cellUpdate")

            # Commit this transition's outcomes before the decision boundary is
            # observed.  This is required for the interval reward to include
            # final-step completions and casualties exactly once.
            self.pedDS.docuStatus()
            if tmr is not None: tmr.lap("ped.docuStatus")

            # === RL ===
            rl_out = self.rl.step(
                simulation_time=sim_time,
                is_terminal=(sim_time == self.stopTime - 1),
            )
            if tmr is not None: tmr.lap("rl.step")

            # === LOG/VISUALIZATION ===
            cumuResult = self.pedDS.result
            metrics = {
                "arrival": cumuResult.get("arrival", 0),
                "casualty": cumuResult.get("casualty", 0),
                "evacuated": cumuResult.get("evacuated", 0),
                "unfinished": cumuResult.get("unfinished", 0),
                "affected": cumuResult.get("affected", 0),
                "panic_onsets": cumuResult.get("panic_onset", 0),
                "panic_eligible_first_exposures": cumuResult.get(
                    "panic_eligible_first_exposure", 0
                ),
                "active_panicked": self.pedDS.active_panicked_count(),
                "panic_herd_choices": cumuResult.get("panic_herd_choices", 0),
                "panic_random_choices": cumuResult.get("panic_random_choices", 0),
                "added_shelters": rl_out.get("added_shelters", 0),
                "reward_raw": float(rl_out.get("reward", 0.0)),
                "reward_safe": rl_out.get("reward_safe", 0.0),
                "reward_casualty": rl_out.get("reward_casualty", 0.0),
                "reward_evacuation_time": rl_out.get(
                    "reward_evacuation_time", 0.0
                ),
                "reward_hazard_exposure": rl_out.get(
                    "reward_hazard_exposure", 0.0
                ),
                "reward_risk_time": rl_out.get("reward_risk_time", 0.0),
                "reward_shelter_service": rl_out.get(
                    "reward_shelter_service", 0.0
                ),
                "new_safe_completions": rl_out.get("new_safe_completions", 0),
                "new_casualties": rl_out.get("new_casualties", 0),
                "attributed_shelter_service": rl_out.get(
                    "attributed_shelter_service", 0
                ),
                "risk_weighted_person_time": rl_out.get("risk_weighted_person_time", 0.0),
                "active_person_time": rl_out.get("active_person_time", 0.0),
                "hazard_exposure_person_time": rl_out.get(
                    "hazard_exposure_person_time", 0.0
                ),
                "decision_made": rl_out.get("decision_made", 0),
                "selected_candidate": rl_out.get("selected_candidate", -1),
                "heuristic_candidate": rl_out.get("heuristic_candidate", -1),
                "selected_cell": rl_out.get("selected_cell", -1),
                "heuristic_cell": rl_out.get("heuristic_cell", -1),
                "completed_action": rl_out.get("completed_action", -1),
                "feasible_cells": rl_out.get("feasible_cells", 0),
                "feasible_candidates": rl_out.get("feasible_candidates", 0),
                "remaining_deployments": rl_out.get("remaining_deployments", 0),
                "capacity_added": rl_out.get("capacity_added", 0.0),
                "rerouted_population": rl_out.get("rerouted_population", 0),
                "episode_return": rl_out.get("episode_return", 0.0),
                "active_remaining": self.pedDS.remaining_active_count(),
                "mean_evacuation_time": float(getattr(self.pedDS, "mean_evacuation_time", lambda: 0.0)()),
                "mean_safe_completion_time": float(self.pedDS.mean_safe_completion_time()),
                "total_shelter_capacity": float(sum(getattr(sh, "shelterCap", 0.0) for sh in self.shelterDS.shelterList.values())),
                "shelter_utilization": 0.0,
                **dict(self.pedDS.lastCongestionMetrics),
            }
            total_capacity = float(metrics["total_shelter_capacity"])
            total_flow = float(sum(getattr(sh, "shelterFlow", 0.0) for sh in self.shelterDS.shelterList.values()))
            metrics["shelter_utilization"] = (total_flow / total_capacity) if total_capacity > 0.0 else 0.0
            population = float(max(1, self.rl.initial_population))
            metrics["safe_completion_rate"] = float(
                metrics["arrival"] + metrics["evacuated"]
            ) / population
            metrics["shelter_evacuation_rate"] = float(metrics["evacuated"]) / population
            eligible_for_panic = int(metrics["panic_eligible_first_exposures"])
            metrics["realized_panic_onset_rate"] = (
                float(metrics["panic_onsets"]) / float(eligible_for_panic)
                if eligible_for_panic > 0
                else 0.0
            )
            
            self.logger.log_step(t=sim_time, reward=float(rl_out.get("reward_norm", rl_out.get("reward", 0.0))), metrics=metrics)

            if self.visualizer is not None:
                self.visualizer.observe(sim_time, rl_out)
            
            if getattr(self.rl, "debug", False) and (sim_time % int(getattr(self.rl, "print_every", 1)) == 0):
                print(f"[CORE] t={sim_time} | reward={rl_out['reward']:.3f} | "
                      f"arr={metrics['arrival']} cas={metrics['casualty']} evac={metrics['evacuated']} "
                      f"affected={metrics['affected']} | "
                      f"added_sh={metrics['added_shelters']}")
            if tmr is not None: tmr.lap("logger.log_step")
       
            for name in ["countByCell","avgVelocityByCell","heatByCell","smokeByCell",
                "dangerLevelByCell","shelterFulfillByCell","wellnessPenaltyByCell"]:
                arr = getattr(self.cellTracker, name, None)
                if arr is None or len(np.asarray(arr).reshape(-1)) != (self.cellX * self.cellY):
                    print(f"[WIRE CHECK] {name} missing or wrong length")
            if self.rl.episode_done:
                break

        # Capture an early terminal boundary even when it was not one of the
        # requested milestones.  This happens before survivor finalization so
        # the map truthfully shows pedestrians still active at horizon expiry.
        if self.visualizer is not None:
            self.visualization_manifest = self.visualizer.finalize(
                terminal_time=last_simulation_time,
            )

        pending_before_finalize = self.pedDS.remaining_active_count()
        panicked_before_finalize = self.pedDS.active_panicked_count()
        finalized_remaining = 0
        if pending_before_finalize > 0 and hasattr(self.pedDS, "finalize_remaining_pedestrians"):
            finalized_remaining = int(self.pedDS.finalize_remaining_pedestrians(event="Unfinished"))
            self.pedDS.docuStatus()
            print(f"[EPISODE FINALIZE] unfinished={finalized_remaining} previously_active={pending_before_finalize}")
            
        rl_diagnostics = {}
        if self.rl is not None and hasattr(self.rl, "end_episode"):
            rl_diagnostics = self.rl.end_episode(
                finalize_rollout=bool(self.finalizePpoRollout)
            )
            if hasattr(self.shelterDS, "remainingCandidateCount"):
                print(f"[SHELTER POOL] remaining_candidates={self.shelterDS.remainingCandidateCount()} active_shelters={len(self.shelterDS.shelterList)}")
                
        total_arrival = int(self.pedDS.result.get("arrival", 0))
        total_casualty = int(self.pedDS.result.get("casualty", 0))
        total_evacuated = int(self.pedDS.result.get("evacuated", 0))
        total_unfinished = int(self.pedDS.result.get("unfinished", 0))
        total_classified = total_arrival + total_evacuated + total_casualty + total_unfinished
        actual_population = int(self.rl.initial_population)
        if total_classified != actual_population:
            print(
                f"[RESULT CHECK] arrival({total_arrival}) + casualty({total_casualty}) + evacuated({total_evacuated}) "
                f"+ unfinished({total_unfinished}) = {total_classified}; initialized={actual_population} "
                f"(difference={actual_population-total_classified})"
            )
        else:
            print(
                f"[RESULT CHECK] arrival({total_arrival}) + casualty({total_casualty}) + evacuated({total_evacuated}) "
                f"+ unfinished({total_unfinished}) = {total_classified}; initialized={actual_population} (balanced)"
            )

        horizon = max(1, int(self.stopTime) - 1)
        horizon_minutes = float(horizon) * float(self.timeStepMinutes)
        restricted_mean_time_to_safety = float(
            float(self.pedDS.safeCompletionTimeSum)
            + float(total_casualty + total_unfinished) * horizon_minutes
        ) / float(max(1, actual_population))

        self.episode_summary = {
            "schema_version": 6,
            "city_id": str(self.cityID),
            "address": str(self.address),
            "map_spec": {
                "query_mode": str(self.mapQueryMode),
                "center": (
                    None
                    if self.mapCenterLat is None or self.mapCenterLon is None
                    else [float(self.mapCenterLat), float(self.mapCenterLon)]
                ),
                "radius_m": None if self.mapRadiusM is None else float(self.mapRadiusM),
                "network_type": "walk",
            },
            "map_provenance": self.OSMProcessor.graph_provenance(),
            "deployment_strategy": self.rl.deployment_strategy,
            "scenario_seed": self.scenario_seed,
            "policy_seed": self.policy_seed,
            "random_stream_seeds": dict(self.random_stream_seeds),
            "hazard_trajectory_digest": self._hazard_trajectory_hasher.hexdigest(),
            "initial_population": actual_population,
            "population_representation": (
                "individual"
                if int(self.pedestrianGroupSize) == 1
                else "weighted_cohort"
            ),
            "maximum_persons_per_agent": int(self.pedestrianGroupSize),
            "initialized_agent_count": int(
                math.ceil(float(actual_population) / float(self.pedestrianGroupSize))
            ),
            "grid_shape": [int(self.cellX), int(self.cellY)],
            "cell_partition": {
                **dict(self.cellPartitionDiagnostics),
                "minimum_axis_width_fraction": float(
                    self.cellPartitionMinWidthFraction
                ),
            },
            "horizon_transitions": horizon,
            "time_step_minutes": float(self.timeStepMinutes),
            "horizon_minutes": horizon_minutes,
            "shelter_action_interval_timesteps": int(self.shelterActionInterval),
            "shelter_action_interval_minutes": (
                int(self.shelterActionInterval) * float(self.timeStepMinutes)
            ),
            "free_flow_speed_m_per_minute": float(self.maxSpeed),
            "free_flow_speed_m_per_second": float(self.maxSpeed) / 60.0,
            "arrival": total_arrival,
            "shelter_evacuated": total_evacuated,
            "safe_completed": total_arrival + total_evacuated,
            "safe_completion_coverage_rate": float(
                total_arrival + total_evacuated
            ) / float(max(1, actual_population)),
            "shelter_service_coverage_rate": float(total_evacuated) / float(
                max(1, actual_population)
            ),
            "casualty": total_casualty,
            "casualty_rate": float(total_casualty) / float(
                max(1, actual_population)
            ),
            "unfinished": total_unfinished,
            "mean_evacuation_time": float(self.pedDS.mean_evacuation_time()),
            "mean_safe_completion_time": float(self.pedDS.mean_safe_completion_time()),
            "restricted_mean_time_to_safety": restricted_mean_time_to_safety,
            "active_shelters": int(len(self.shelterDS.shelterList)),
            "remaining_candidates": int(self.shelterDS.remainingCandidateCount()),
            "initial_observation_digest": self.rl.initial_observation_digest,
            "precommit_initial_observation_digest": (
                self.rl.precommit_initial_observation_digest
            ),
            "ppo_rollout_episodes": int(self.rl.rollout_episodes),
            "visualization_manifest": (
                None
                if self.visualization_manifest is None
                else self.visualization_manifest.get("manifest_path")
            ),
            **self.pedDS.congestion_summary(),
            **self.pedDS.social_force_summary(),
            **self.pedDS.distance_summary(),
            **self.pedDS.panic_summary(),
            "active_panicked_at_horizon": int(panicked_before_finalize),
            **{key: float(value) for key, value in rl_diagnostics.items()},
        }
        with open(os.path.join(self.run_dir, "episode_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(self.episode_summary, handle, indent=2, sort_keys=True)

        if self.logger is not None:
            self.logger.close()
