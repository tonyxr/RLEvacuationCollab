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
import networkx as nx

#import torch
#import torch.nn as nn
#import torch.optim as optim

"""Import other main Simulator modules"""
from OSMProcessor import OSMProcessor
from MapDatabase import MapDS
from HazardDatabase import HazardDS, HazardRegimeConfig, HazardRegimeProcess
from StressScenarios import (
    StressScenarioSpec,
    bearing_degrees,
    clustered_origin_plan,
    downwind_front_speed,
    population_centroid,
    stress_rng,
    threat_source_index,
)
from ShelterDatabase import ShelterDS
from PedestrianDatabase import PedDS
from CAProcessor import CellTracker, authoritative_grid_shape
from CellPartitioning import (
    NODE_DENSITY_ADAPTIVE,
    POPULATION_DENSITY_ADAPTIVE,
    build_cell_partition,
    normalize_partition_mode,
    sample_population_origins,
)
from SocialForce import ForceProcessor

"""Import RL components"""
from RLBridge import (
    DEFAULT_ACTOR_EPOCHS,
    DEFAULT_ACTOR_LEARNING_RATE,
    DEFAULT_ADVANTAGE_SCALE_FLOOR,
    DEFAULT_CLIP_EPS,
    DEFAULT_CRITIC_EPOCHS,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_ROLLOUT_EPISODES,
    DEFAULT_ENTROPY_COEF_END,
    DEFAULT_EXPLORATION_DECAY_UPDATES,
    DEFAULT_NMCC_NATURAL_LOSS_COEF,
    DEFAULT_TEMPERATURE_END,
    DEFAULT_TEMPERATURE_START,
    RLBridge,
)
from NMCCPIConfig import NMCC_PI_ALL_CORE_FIELDS
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

    @staticmethod
    def _initial_shelter_volume(
        deployment_strategy: str,
        requested_initial: int,
        final_target: int,
    ) -> int:
        """Return the t=0 count while preserving one common final budget."""
        strategy = str(deployment_strategy).strip().lower()
        if strategy == "none":
            return int(final_target)
        return int(requested_initial)
    
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
        # Experiment invariant: when enabled, the initial and sequential
        # equal-capacity shelter budget must sum to the population plus the
        # explicitly declared integer surplus.  The surplus prevents
        # reservation fragmentation from leaving rational pedestrians without
        # capacity after the final scheduled installation.
        self.requirePopulationMatchedShelterCapacity = False
        self.shelterCapacitySurplus = 0
        # N-minus-k capacity resilience: after this many equal-capacity sites
        # are lost, the surviving planned network must still house the full
        # initial population. Zero preserves legacy experiment contracts.
        self.shelterFailureReserveCount = 0
        # Retained only so archived curricula still parse. These values do not
        # filter the policy's action space; risk and benefit are observations.
        self.maximumShelterForecastDanger = 1.0
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
        # Expected casualties are priced in the same person-minute ledger as
        # evacuation time.  kappa=2 makes one expected casualty cost twice the
        # maximum remaining living burden; sensitivity runs vary this declared
        # value rather than hiding it in reward weights.
        self.casualtyPriorityKappa = 2.0
        self.casualtyPriceRule = "legacy"
        self.publicObjectiveVariant = "legacy"
        # Explicit objective weights.  Defaults preserve historical runs;
        # experiment curricula may change them, but the same values are used
        # by the live reward, branch grading, heuristics and evaluation.
        self.activePersonTimeWeight = 1.0
        self.hazardExposureTimeWeight = 1.0
        # Optional coherent tail-clearance objective.  Older experiment files
        # leave the weight at zero; curricula that enable it state both the
        # CVaR population fraction and its extra weight explicitly.
        self.clearanceTailFraction = 0.10
        self.clearanceTailExtraWeight = 0.0
        # Observable exogenous wind used by both hazard evolution and the
        # administrator-facing action-window hazard forecast. Direction is the
        # direction toward which the hazard is transported, counter-clockwise
        # from projected east.
        self.hazardWindSpeedMPerMinute = 0.0
        self.hazardWindDirectionDegrees = 0.0
        self.hazardWindInfluence = 1.0
        # Physical front speed used by deterministic spread and to scale the
        # stochastic transition probability.  It is independent of the
        # decision-grid resolution and population-adaptive cell widths.
        self.hazardSpreadSpeedMPerMinute = 100.0 / 3.0
        # Danger severity is a source-local, elapsed-time process.  Only the
        # source cell can reach level 5, cells whose footprint is within the
        # declared radius can reach level 4, and all other affected cells cap
        # at level 3.  Values are cumulative minutes since first impact.
        self.hazardSeverityTransitionMinutes = [6.0, 15.0, 30.0, 50.0]
        self.hazardSevereRadiusM = 100.0
        # An installed shelter permanently fails at or above this integer CA
        # danger level; its current residents resume evacuation immediately.
        self.shelterOverrunDangerLevel = 4
        # v53r regime/timing controls are opt-in. Their disabled defaults leave
        # the historical simulator, action table and random streams untouched.
        self.hazardRegimeEnabled = False
        self.hazardWindShiftRatePerMinute = 0.0
        self.hazardWindShiftDegrees = [-90.0, -45.0, 45.0, 90.0]
        self.hazardWindSpeedMultipliers = [0.6, 1.0, 1.5]
        self.hazardWindMinDwellMinutes = 20
        self.hazardForecastLeadMinutes = 15
        self.hazardForecastMissProbability = 0.2
        self.hazardForecastFalseAlarmRatePerMinute = 0.0
        self.hazardIntensitySwitchRatePerMinute = 0.0
        self.hazardIntensityMultipliers = [0.7, 1.0, 1.6]
        self.hazardClosureRatePerMinute = 0.0
        self.hazardClosureDurationMinutes = [20, 60]
        self.hazardClosureWarningLeadMinutes = 10
        self.hazardSecondaryIgnitionRatePerMinute = 0.0
        self.hazardSecondaryIgnitionWindow = [10, 90]
        self.forecastEnsembleSize = 8
        self.nmccFrontForecastIntervals = 4
        self.actionSpace = "cell"
        self.nmccPiAllocation = "uniform"
        # Information set of the futures that grade the actor's rollout
        # targets. "belief" resamples every latent regime event from the
        # public signals at the decision (the actor's information set);
        # "realized" replays the episode's hidden regime tape and is retained
        # only as an explicitly labelled hindsight ablation.
        self.nmccPiTargetFutureMode = "belief"
        self.deadlineGreedyThreshold = -1.0e30
        self.constructedDecoyDemand = False
        # Evaluation stress scenarios (exogenous, pre-registered; off by
        # default and never set by the training curricula). See
        # ``StressScenarios`` and ``config/stress_family_evaluation.json``.
        self.stressScenarioFamily = "none"
        self.stressClusterShares = []
        self.stressClusterRadiusM = 300.0
        self.stressClusterMinSeparationM = 1500.0
        self.stressThreatenedCluster = -1
        self.stressThreatArrivalMinutes = 0.0
        self.stressWindAim = "none"
        self.hazard_regime = None
        
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
        # Parsed for archived curricula only; direct PPO uses its state-value
        self.advantageScaleFloor = DEFAULT_ADVANTAGE_SCALE_FLOOR
        # Positive conditioning multiplier applied only to PPO/NMCC learning
        # targets.  Reported physical rewards retain their normalized units.
        self.learningRewardScale = 1.0
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
        self.nmccNaturalLossCoefficient = DEFAULT_NMCC_NATURAL_LOSS_COEF
        self.entropyCoefficientStart = DEFAULT_ENTROPY_COEF
        self.entropyCoefficientEnd = DEFAULT_ENTROPY_COEF_END
        self.explorationDecayUpdates = DEFAULT_EXPLORATION_DECAY_UPDATES
        self.explorationSchedule = "half_cosine"
        self.actionTemperatureStart = DEFAULT_TEMPERATURE_START
        self.actionTemperatureEnd = DEFAULT_TEMPERATURE_END
        # NMCC policy improvement (exact within-state branch targets) and the
        # actor's prior logit.  Defaults, types and constructor keywords come
        # from the shared NMCC configuration contract.
        for attribute, _, _, default in NMCC_PI_ALL_CORE_FIELDS:
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
        # Curriculum-owned schedule coordinates.  They are supplied by the
        # runner for every episode so exploration and optimizer learning rates
        # can decay independently inside each learning stage.
        self.trainingStageId = "standalone"
        self.trainingStageUpdateIndex = 0
        self.trainingStageTotalUpdates = 1
        self.trainMode = False
        self.finalInstallationUnassignedRationalPopulation = None
        self.decoy_scenario_metadata = None
        self.stress_scenario_metadata = None
        self._stress_raw_node_xy = None
        self._stress_population_weights = None
        self._stress_anchor_indices = ()

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

    def _construct_decoy_origin_plan(self):
        """Create the preregistered decoy-demand start state.

        Three quarters of represented pedestrians start at an installed
        shelter and therefore clear quickly.  The remainder start beyond a
        true bridge link with every initial shelter on the opposite side.  A
        warned closure of both directed copies of that bridge is added to the
        exogenous tape.  The construction is deterministic conditional on the
        scenario/map and fails if the requested experiment cannot be formed.
        """
        if not self.constructedDecoyDemand:
            return None
        if self.hazard_regime is None or not self.hazard_regime.enabled:
            raise RuntimeError("The constructed decoy requires the hazard regime")
        shelters = tuple(
            sorted(
                self.shelterDS.shelterList.values(),
                key=lambda shelter: int(shelter.shelterID),
            )
        )
        if not shelters:
            raise RuntimeError("The constructed decoy requires an initial shelter")
        shelter_nodes = {
            int(shelter.nodeMapped.OSMID) for shelter in shelters
        }
        graph = nx.Graph(self.mapDS.locationDrive)
        if graph.number_of_nodes() < 3:
            raise RuntimeError("The constructed decoy requires a connected road graph")
        chosen = None
        for u, v in sorted(nx.bridges(graph), key=lambda pair: (int(pair[0]), int(pair[1]))):
            graph.remove_edge(u, v)
            components = (
                nx.node_connected_component(graph, u),
                nx.node_connected_component(graph, v),
            )
            graph.add_edge(u, v)
            for component, boundary in ((components[0], u), (components[1], v)):
                component = set(component)
                if len(component) < 3 or shelter_nodes & component:
                    continue
                if len(component) > max(3, int(0.45 * graph.number_of_nodes())):
                    continue
                chosen = (int(u), int(v), component, int(boundary))
                break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(
                "No bridge-separated road region without an initial shelter "
                "is available for the constructed decoy"
            )
        u, v, isolated, boundary = chosen
        subgraph = graph.subgraph(isolated)
        distances = nx.single_source_shortest_path_length(subgraph, boundary)
        small_origin = int(max(distances, key=lambda node: (distances[node], int(node))))
        large_origin = int(shelters[0].nodeMapped.OSMID)
        represented_agents = int(
            math.ceil(float(self.pedVol) / float(self.pedestrianGroupSize))
        )
        large_count = int(round(0.75 * represented_agents))
        large_count = min(max(1, large_count), represented_agents - 1)
        origins = [large_origin] * large_count + [small_origin] * (
            represented_agents - large_count
        )
        closure_edges = sorted({
            int(edge.edgeID)
            for pair in ((u, v), (v, u))
            for edge in self.mapDS.idx_uv_to_edges.get(pair, ())
        })
        if not closure_edges:
            raise RuntimeError("The decoy bridge has no initialized road edge")
        closure_time = max(1, int(self.shelterActionInterval))
        duration = max(1, int(self.stopTime))
        for edge_id in closure_edges:
            self.hazard_regime.scheduled.append({
                "time": closure_time,
                "kind": "closure",
                "payload": {"edge_id": edge_id, "duration": duration},
            })
        self.hazard_regime.scheduled.sort(
            key=lambda event: (int(event["time"]), str(event["kind"]))
        )
        self.hazard_regime._update_public(0)
        self.decoy_scenario_metadata = {
            "large_origin_osmid": large_origin,
            "small_origin_osmid": small_origin,
            "large_agent_count": large_count,
            "small_agent_count": represented_agents - large_count,
            "bridge_nodes": [u, v],
            "closure_edge_ids": closure_edges,
            "closure_time": closure_time,
        }
        return origins
    
    # -- evaluation stress scenarios -------------------------------------

    def _raw_node_xy(self, raw_nodes):
        """Local metric coordinates (x east, y south) of every raw road node."""
        xy = np.full((len(raw_nodes), 2), np.nan, dtype=np.float64)
        for index, (osmid, _) in enumerate(raw_nodes):
            node = self.mapDS.nodeListByOSMID.get(int(osmid))
            if node is not None:
                xy[index] = (float(node.nodeX), float(node.nodeY))
        return xy

    def _stress_hazard_sources(self, spec, raw_nodes):
        """Place hazard 0 at the pre-registered nominal arrival distance."""
        if spec.threatened_cluster < 0:
            return None
        if not self._stress_anchor_indices or self._stress_raw_node_xy is None:
            raise RuntimeError("Threatened-cluster scenario has no cluster anchors")
        anchors = list(self._stress_anchor_indices)
        threatened = anchors[spec.threatened_cluster]
        front_speed = downwind_front_speed(
            self.hazardSpreadSpeedMPerMinute,
            self.hazardWindSpeedMPerMinute,
            self.hazardWindInfluence,
        )
        distance = float(spec.threat_arrival_minutes) * front_speed
        source_index = threat_source_index(
            self._stress_raw_node_xy,
            threatened,
            distance,
            self.random_stream_seeds["hazard_evolution"],
            other_anchor_indices=[
                anchor for position, anchor in enumerate(anchors)
                if position != spec.threatened_cluster
            ],
        )
        node = self.mapDS.nodeListByOSMID.get(int(raw_nodes[source_index][0]))
        if node is None:
            raise RuntimeError("Threat source node is not in the road network")
        return [node]

    def _apply_stress_wind_aim(self, spec):
        """Point the public wind before t = 0 (hazard, forecast and regime)."""
        if spec.wind_aim == "random":
            direction = float(
                stress_rng(self.random_stream_seeds["hazard_regime"]).uniform(0.0, 360.0)
            )
        else:
            sources = [
                (float(hazard.sourceNode.nodeX), float(hazard.sourceNode.nodeY))
                for hazard in self.hazardDS.hazardList.values()
                if getattr(hazard, "sourceNode", None) is not None
            ]
            if not sources:
                raise RuntimeError("Aimed wind requires at least one hazard source")
            source_xy = np.mean(np.asarray(sources, dtype=np.float64), axis=0)
            if spec.wind_aim == "population_centroid":
                target_xy = population_centroid(
                    self._stress_raw_node_xy, self._stress_population_weights
                )
            else:
                anchor = self._stress_anchor_indices[spec.threatened_cluster]
                target_xy = self._stress_raw_node_xy[int(anchor)]
            direction = bearing_degrees(source_xy, target_xy)
        self.hazardWindDirectionDegrees = float(direction) % 360.0
        self.hazardDS.configure_wind(
            speed_m_per_minute=self.hazardWindSpeedMPerMinute,
            direction_degrees=self.hazardWindDirectionDegrees,
            influence=self.hazardWindInfluence,
            time_step_minutes=self.timeStepMinutes,
        )
        regime = getattr(self, "hazard_regime", None)
        if regime is not None:
            regime.base_wind_direction_degrees = self.hazardWindDirectionDegrees
            regime.wind_direction_degrees = self.hazardWindDirectionDegrees
            regime._update_public(0)

    def _stress_metadata(self, spec, raw_nodes, source_nodes):
        anchors = [int(raw_nodes[index][0]) for index in self._stress_anchor_indices]
        metadata = {
            **spec.as_dict(),
            "anchor_osmids": anchors,
            "wind_speed_m_per_minute": float(self.hazardWindSpeedMPerMinute),
            "wind_direction_degrees": float(self.hazardWindDirectionDegrees),
            "nominal_downwind_front_speed_m_per_minute": float(
                downwind_front_speed(
                    self.hazardSpreadSpeedMPerMinute,
                    self.hazardWindSpeedMPerMinute,
                    self.hazardWindInfluence,
                )
            ),
        }
        if source_nodes:
            source = source_nodes[0]
            anchor_xy = self._stress_raw_node_xy[
                int(self._stress_anchor_indices[spec.threatened_cluster])
            ]
            metadata.update({
                "threat_source_osmid": int(source.OSMID),
                "threat_distance_m": float(np.hypot(
                    float(source.nodeX) - anchor_xy[0],
                    float(source.nodeY) - anchor_xy[1],
                )),
            })
        return metadata

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

        self.trainMode = bool(train_mode)
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
            "pedestrian_origins": int(
                np.random.SeedSequence([self.scenario_seed, 4]).generate_state(
                    1, dtype=np.uint64
                )[0]
            ),
            "hazard_regime": int(
                np.random.SeedSequence([self.scenario_seed, 5]).generate_state(
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
            spread_speed_m_per_minute=self.hazardSpreadSpeedMPerMinute,
            severity_transition_minutes=self.hazardSeverityTransitionMinutes,
            severe_radius_m=self.hazardSevereRadiusM,
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
        self.shelterDS.hazardOverrunDangerLevel = int(
            self.shelterOverrunDangerLevel
        )
        
        # Call initializing functions here, functions needed at t = 0 (follow old model, with additions)
        self.mapDS.computeConvertUnit()
        self.mapDS.boundarySetter()
        
        self.hazardDS.setCellTracker(self.cellTracker)

        raw_nodes = list(self.OSMProcessor.nodeList)
        planned_birth_osmids = None
        planned_group_sizes = None
        population_weights = None
        population_partition_key = None
        if self.cellPartitionMode == POPULATION_DENSITY_ADAPTIVE:
            origin_plan = sample_population_origins(
                len(raw_nodes),
                int(self.pedVol),
                int(self.pedestrianGroupSize),
                self.random_stream_seeds["pedestrian_origins"],
            )
            stress_spec = StressScenarioSpec.from_core(self)
            if stress_spec.requires_origin_plan:
                self._stress_raw_node_xy = self._raw_node_xy(raw_nodes)
            if stress_spec.cluster_shares:
                origin_plan = clustered_origin_plan(
                    self._stress_raw_node_xy,
                    int(self.pedVol),
                    int(self.pedestrianGroupSize),
                    stress_spec,
                    self.random_stream_seeds["pedestrian_origins"],
                )
                self._stress_anchor_indices = tuple(origin_plan["anchor_indices"])
            planned_birth_osmids = [
                int(raw_nodes[index][0])
                for index in origin_plan["origin_indices"]
            ]
            planned_group_sizes = [
                int(value) for value in origin_plan["group_sizes"]
            ]
            population_weights = np.asarray(
                origin_plan["node_population"], dtype=float
            )
            self._stress_population_weights = population_weights
            population_partition_key = hashlib.sha256(
                population_weights.tobytes()
            ).hexdigest()

        partition_cache_key = (
            infra_cache_key,
            int(self.cellX),
            int(self.cellY),
            str(self.cellPartitionMode),
            float(self.cellPartitionMinWidthFraction),
            population_partition_key,
        )
        cached_partition = Core._cell_partition_cache.get(partition_cache_key)

        if cached_partition is None:
            x_vals = []
            y_vals = []
            for _, data in raw_nodes:
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
                population_weights=population_weights,
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
        grid_x, grid_y = authoritative_grid_shape(self)
        partition_shape = tuple(
            int(value)
            for value in self.cellPartitionDiagnostics.get("grid_shape", ())
        )
        if partition_shape != (grid_x, grid_y):
            raise RuntimeError(
                "Cell partition and CellTracker disagree: "
                f"partition={partition_shape}, tracker={(grid_x, grid_y)}"
            )

        N = int(self.cellTracker.number_of_cells)
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
        regime_config = HazardRegimeConfig(
            enabled=bool(self.hazardRegimeEnabled),
            wind_shift_rate_per_minute=float(
                self.hazardWindShiftRatePerMinute
            ),
            wind_shift_degrees=tuple(self.hazardWindShiftDegrees),
            wind_speed_multipliers=tuple(self.hazardWindSpeedMultipliers),
            wind_min_dwell_minutes=int(self.hazardWindMinDwellMinutes),
            forecast_lead_minutes=int(self.hazardForecastLeadMinutes),
            forecast_miss_probability=float(
                self.hazardForecastMissProbability
            ),
            forecast_false_alarm_rate_per_minute=float(
                self.hazardForecastFalseAlarmRatePerMinute
            ),
            intensity_switch_rate_per_minute=float(
                self.hazardIntensitySwitchRatePerMinute
            ),
            intensity_multipliers=tuple(self.hazardIntensityMultipliers),
            closure_rate_per_minute=float(self.hazardClosureRatePerMinute),
            closure_duration_minutes=tuple(
                self.hazardClosureDurationMinutes
            ),
            closure_warning_lead_minutes=int(
                self.hazardClosureWarningLeadMinutes
            ),
            secondary_ignition_rate_per_minute=float(
                self.hazardSecondaryIgnitionRatePerMinute
            ),
            secondary_ignition_window=tuple(
                self.hazardSecondaryIgnitionWindow
            ),
            episode_horizon=max(1, int(self.stopTime) - 1),
        )
        self.hazard_regime = HazardRegimeProcess(
            regime_config,
            np.random.default_rng(self.random_stream_seeds["hazard_regime"]),
            self.cellTracker,
            self.mapDS,
        )
        self.hazard_regime.base_wind_direction_degrees = float(
            self.hazardWindDirectionDegrees
        )
        self.hazard_regime.base_wind_speed_m_per_minute = float(
            self.hazardWindSpeedMPerMinute
        )
        self.hazard_regime.reset(0)
        self.hazardDS.set_regime(self.hazard_regime)
        
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
        self.shelterDS.shelterPerCell(self.cellTracker)
        authoritative_grid_shape(self)
        available_shelters = int(self.shelterDS.remainingCandidateCount())
        requested_initial_shelters = int(self.initShelterVol)
        planned_shelters = int(self.initShelterVol) + int(
            self.maxAdditionalShelters
        )
        if (
            self.requirePopulationMatchedShelterCapacity
            and available_shelters < planned_shelters
        ):
            raise RuntimeError(
                "Population-matched deployment requires at least "
                f"{planned_shelters} detected shelter candidates, but only "
                f"{available_shelters} are available"
            )
        shelter_action_interval = int(self.shelterActionInterval)
        deploy_windows = 0
        first_action_time = shelter_action_interval
        terminal_time = int(self.stopTime) - 1
        if terminal_time >= first_action_time:
            deploy_windows = (
                (terminal_time - first_action_time) // shelter_action_interval
            ) + 1
        self.targetActiveShelters = self._shelter_target(
            available_candidates=available_shelters,
            initial_shelters=self.initShelterVol,
            decision_windows=deploy_windows,
            maximum_additions=self.maxAdditionalShelters,
        )
        if (
            self.requirePopulationMatchedShelterCapacity
            and int(self.targetActiveShelters) != planned_shelters
        ):
            raise RuntimeError(
                "Population-matched deployment must schedule the complete "
                f"{planned_shelters}-shelter budget; computed target was "
                f"{self.targetActiveShelters}"
            )
        strategy_key = str(deployment_strategy).strip().lower()
        self.shelterDS.initVol = self._initial_shelter_volume(
            strategy_key,
            requested_initial_shelters,
            self.targetActiveShelters,
        )
        if strategy_key == "none":
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
        self.static_predeployment_shelter_ids = frozenset(predeployed_ids)
        for shelter in self.shelterDS.shelterList.values():
            self.cellTracker.addShelter(shelter.cellLocated, shelter)
        if self.verbose:
            print("Shelters list: ", self.shelterDS.shelterList)
        decoy_origins = self._construct_decoy_origin_plan()
        if decoy_origins is not None:
            planned_birth_osmids = decoy_origins
            planned_group_sizes = [1] * len(decoy_origins)
        
        stress_spec = StressScenarioSpec.from_core(self)
        stress_source_nodes = self._stress_hazard_sources(stress_spec, raw_nodes)
        self.hazardDS.initHazard(
            self.mapDS, self.cellTracker, source_nodes=stress_source_nodes
        )
        if stress_spec.wind_aim != "none":
            self._apply_stress_wind_aim(stress_spec)
        if stress_spec.active:
            self.stress_scenario_metadata = self._stress_metadata(
                stress_spec, raw_nodes, stress_source_nodes
            )
        self._hazard_trajectory_hasher = hashlib.sha256()
        self._record_hazard_state(0)
        
        self.pedDS.initPedestrianAgent(
            self.mapDS,
            self.cellTracker,
            self.maxSpeed,
            birth_node_osmids=planned_birth_osmids,
            group_sizes=planned_group_sizes,
        )
        
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
        self.pedDS.shelter_damage_danger_threshold = int(
            self.shelterOverrunDangerLevel
        )
        # Initial sites obey the same physics as sequentially installed ones.
        # This runs after the occupant registry is connected but before any
        # pedestrian is routed, so a source-overrun site contributes no
        # phantom operational capacity at t=0.
        self.pedDS.resolveHazardShelterClosures(simulation_time=0)
        self.pedDS.route_active_to_nearest_shelter()
        # Initialize every cell-indexed observation from the same authoritative
        # CA state before a t=0 precommitment or the first RL observation. In
        # particular, hazards installed by ``initHazard`` must not remain as
        # zeroes in the decision layer until the first physical timestep.
        self.cellTracker.cellUpdate(
            pedDS=self.pedDS,
            forceTracker=self.forceTracker,
        )
        authoritative_grid_shape(self)
        if (
            self.requirePopulationMatchedShelterCapacity
            and len(self.shelterDS.shelterList) == int(self.targetActiveShelters)
        ):
            unassigned = int(self.pedDS.unassigned_rational_population())
            self.finalInstallationUnassignedRationalPopulation = int(
                unassigned
            )
            self.shelterDS.assertPopulationMatchedAssignment(
                unassigned,
                self.pedVol,
                network_disruption_active=bool(
                    getattr(self.mapDS, "closed_edges", {})
                ),
                context="Complete shelter budget",
            )

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
            advantage_scale_floor = float(self.advantageScaleFloor),
            learning_reward_scale = float(self.learningRewardScale),
            minibatch_size = 32,
            entropy_coef = float(self.entropyCoefficientStart),
            rollout_episodes = int(self.ppoRolloutEpisodes),
            shelter_action_interval = int(self.shelterActionInterval),
            first_decision_time = int(self.shelterActionInterval),
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
            nmcc_natural_loss_coef=float(self.nmccNaturalLossCoefficient),
            entropy_coef_end=float(self.entropyCoefficientEnd),
            exploration_decay_updates=int(self.explorationDecayUpdates),
            exploration_schedule=str(self.explorationSchedule),
            action_temperature_start=float(self.actionTemperatureStart),
            action_temperature_end=float(self.actionTemperatureEnd),
            training_stage_id=str(self.trainingStageId),
            training_stage_update_index=int(self.trainingStageUpdateIndex),
            training_stage_total_updates=int(self.trainingStageTotalUpdates),
            **{
                keyword: getattr(self, attribute)
                for attribute, keyword, _, _ in NMCC_PI_ALL_CORE_FIELDS
            },
        )
        if strategy_key in {"initial_only", "static_greedy"}:
            schedule = self.rl.precommit_static_schedule()
            print(
                f"[SHELTER FAIRNESS] strategy={strategy_key} committed "
                f"{len(schedule)} initial-state sites for scheduled deployment "
                f"at t={self.shelterActionInterval},...,"
                f"{len(schedule) * self.shelterActionInterval}.",
                flush=True,
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
            "schema_version": 6,
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
            "grid_shape": list(authoritative_grid_shape(self)),
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
            "first_sequential_shelter_timestep": int(self.shelterActionInterval),
            "final_sequential_shelter_timestep": int(
                self.maxAdditionalShelters * self.shelterActionInterval
            ),
            "post_final_installation_timesteps": int(
                (self.stopTime - 1)
                - self.maxAdditionalShelters * self.shelterActionInterval
            ),
            "shelter_action_interval_minutes": (
                int(self.shelterActionInterval) * float(self.timeStepMinutes)
            ),
            "candidate_action_constraints": {
                "mask": "remaining_budget_and_positive_physical_capacity_only",
                "risk_and_benefit_features_are_observations_only": True,
            },
            "free_flow_speed_m_per_minute": float(self.maxSpeed),
            "free_flow_speed_m_per_second": float(self.maxSpeed) / 60.0,
            "congestion": self.congestionModel.contract(),
            "social_force": self.forceTracker.contract(),
            "panic_behavior": self.pedDS.panic_contract(),
            "initial_shelters": int(len(self.shelterDS.shelterList)),
            "maximum_dynamic_deployments": int(self.rl.maximum_deployments),
            "static_precommit_schedule": [
                {
                    "order": int(index + 1),
                    "candidate_osm_node_id": str(candidate_osm_node_id),
                    "cell_index": int(cell_index),
                    "scheduled_timestep": int(
                        self.shelterActionInterval * (index + 1)
                    ),
                }
                for index, (candidate_osm_node_id, cell_index) in enumerate(
                    self.rl.static_precommit_schedule
                )
            ],
            "checkpoint_path": (
                self.rl.checkpoint_path
                if strategy_key in {"rl", "rl_precommit"}
                else None
            ),
            "learner_batch_episodes": int(self.rl.rollout_episodes),
            "reward": (
                "V_uniform_random(state) - realized_complete_interval_cost - "
                "V_uniform_random(next_state), normalized by population*horizon"
            ),
            "reward_plain_time_term": (
                "realized_active_person_time_plus_terminal_remaining_time_to_safety"
            ),
            "reward_hazard_exposure_term": (
                "realized_danger_weighted_active_person_time_plus_terminal_exposure"
            ),
            "route_change_rule": (
                "none_to_reserved_or_strictly_shorter_capacity_valid_destination"
            ),
            "unavailable_shelter_time_minutes": float(
                (int(self.stopTime) - 1) * float(self.timeStepMinutes)
            ),
            "terminal_residual_in_learning_reward": (
                "charged_inside_the_final_complete_interval_cost"
            ),
            "evaluation_objective": (
                "minimize_normalized_complete_ledger_cost_composed_of_active_"
                "time_hazard_exposure_expected_casualty_and_clearance_tail"
            ),
            "casualty_role": (
                "rao_blackwellized_expected_casualty_intensity_weighted_by_"
                "declared_priority_kappa; realized_casualties_reported_secondarily"
            ),
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
        stage_id = str(self.trainingStageId).upper()
        comparison_horizon = {
            "E": "environment_dynamics_only_actor_frozen",
            "P": "complete_simulator_rollout_to_episode_horizon",
        }.get(stage_id, "complete_simulator_rollout_to_episode_horizon")
        if not self.trainMode:
            critic_target = "frozen_evaluation_value"
        elif stage_id == "E":
            critic_target = "random_reference_cost_to_go"
        elif bool(getattr(self, "nmccPolicyImprovement")):
            critic_target = (
                "complete_episode_smdp_monte_carlo_return_to_go_with_"
                "realized_regime_baseline"
            )
        else:
            critic_target = "current_policy_return_to_go"
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
                    else "bounded_initial_node_cohort"
                ),
                "maximum_persons_per_agent": int(self.pedestrianGroupSize),
                "initialized_agent_count": int(
                    math.ceil(float(self.pedVol) / float(self.pedestrianGroupSize))
                ),
                "initial_node_group_count": "recorded_after_initialization",
                "review_unit": "origin_node_cohort_per_simulator_transition",
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
            "planned_total_shelter_capacity": int(
                (self.initShelterVol + self.maxAdditionalShelters)
                * self.shelterCapacityToken
            ),
            "population_matched_shelter_capacity": bool(
                self.requirePopulationMatchedShelterCapacity
            ),
            "shelter_capacity_surplus": int(self.shelterCapacitySurplus),
            "shelter_failure_reserve_count": int(
                self.shelterFailureReserveCount
            ),
            "planned_capacity_after_reserved_failures": int(
                max(
                    0,
                    self.initShelterVol
                    + self.maxAdditionalShelters
                    - self.shelterFailureReserveCount,
                )
                * self.shelterCapacityToken
            ),
            "training_stage": {
                "id": str(self.trainingStageId),
                "update_index": int(self.trainingStageUpdateIndex),
                "total_updates": int(self.trainingStageTotalUpdates),
                "train_environment_dynamics": bool(
                    self.trainMode and stage_id == "E"
                ),
                "train_actor": bool(self.trainMode and stage_id != "E"),
                "train_shared_critic": bool(
                    self.trainMode
                    and stage_id not in {"E"}
                ),
                "critic_target": critic_target,
                "use_continuation_for_actions": bool(
                    not bool(getattr(self, "nmccPolicyImprovement"))
                    and ((not self.trainMode) or stage_id == "P")
                ),
            },
            "hazard_casualty_distribution": list(self.hazardCasualtyRate),
            "hazard_casualty_reference_exposure_minutes": float(
                self.hazardCasualtyReferenceMinutes
            ),
            "expected_casualty_priority_kappa": float(
                self.casualtyPriorityKappa
            ),
            "casualty_price_rule": str(self.casualtyPriceRule),
            "public_objective_variant": str(self.publicObjectiveVariant),
            "active_person_time_weight": float(self.activePersonTimeWeight),
            "hazard_exposure_time_weight": float(
                self.hazardExposureTimeWeight
            ),
            "clearance_tail_fraction": float(self.clearanceTailFraction),
            "clearance_tail_extra_weight": float(
                self.clearanceTailExtraWeight
            ),
            "hazard_spread_distribution": list(self.hazardSpreadRate),
            "hazard_speed_distribution": list(self.hazardSpeedReduct),
            "hazard_evolution_mode": str(self.hazardEvolutionMode),
            "hazard_spread_speed_m_per_minute": float(
                self.hazardSpreadSpeedMPerMinute
            ),
            "hazard_severity_transition_minutes": list(
                self.hazardSeverityTransitionMinutes
            ),
            "hazard_severe_radius_m": float(self.hazardSevereRadiusM),
            "shelter_overrun_danger_level": int(
                self.shelterOverrunDangerLevel
            ),
            "hazard_wind": {
                "speed_m_per_minute": float(self.hazardWindSpeedMPerMinute),
                "direction_degrees_from_east": float(
                    self.hazardWindDirectionDegrees
                ),
                "spread_influence": float(self.hazardWindInfluence),
            },
            "hazard_regime": {
                "enabled": bool(self.hazardRegimeEnabled),
                "wind_shift_rate_per_minute": float(
                    self.hazardWindShiftRatePerMinute
                ),
                "wind_shift_degrees": list(self.hazardWindShiftDegrees),
                "wind_speed_multipliers": list(
                    self.hazardWindSpeedMultipliers
                ),
                "wind_min_dwell_minutes": int(
                    self.hazardWindMinDwellMinutes
                ),
                "forecast_lead_minutes": int(
                    self.hazardForecastLeadMinutes
                ),
                "forecast_miss_probability": float(
                    self.hazardForecastMissProbability
                ),
                "forecast_false_alarm_rate_per_minute": float(
                    self.hazardForecastFalseAlarmRatePerMinute
                ),
                "intensity_switch_rate_per_minute": float(
                    self.hazardIntensitySwitchRatePerMinute
                ),
                "intensity_multipliers": list(
                    self.hazardIntensityMultipliers
                ),
                "closure_rate_per_minute": float(
                    self.hazardClosureRatePerMinute
                ),
                "closure_duration_minutes": list(
                    self.hazardClosureDurationMinutes
                ),
                "closure_warning_lead_minutes": int(
                    self.hazardClosureWarningLeadMinutes
                ),
                "secondary_ignition_rate_per_minute": float(
                    self.hazardSecondaryIgnitionRatePerMinute
                ),
                "secondary_ignition_window": list(
                    self.hazardSecondaryIgnitionWindow
                ),
                "intensity_information_contract": (
                    "latent_bayesian_posterior_from_observed_ca_front"
                ),
                "intensity_transition_model": "marked_poisson",
            },
            "forecast_ensemble_size": int(self.forecastEnsembleSize),
            "action_space": str(self.actionSpace),
            "nmcc_pi_allocation": str(self.nmccPiAllocation),
            "nmcc_pi_target_future_mode": str(self.nmccPiTargetFutureMode),
            "deadline_greedy_threshold": float(self.deadlineGreedyThreshold),
            "constructed_decoy_demand": bool(self.constructedDecoyDemand),
            "stress_scenario": {
                "family": str(self.stressScenarioFamily),
                "cluster_shares": [float(v) for v in (self.stressClusterShares or [])],
                "cluster_radius_m": float(self.stressClusterRadiusM),
                "cluster_min_separation_m": float(self.stressClusterMinSeparationM),
                "threatened_cluster": int(self.stressThreatenedCluster),
                "threat_arrival_minutes": float(self.stressThreatArrivalMinutes),
                "wind_aim": str(self.stressWindAim),
            },
            "ppo_learning_rate": float(self.learningRate),
            "actor_learning_rate": float(self.actorLearningRate),
            "critic_learning_rate": float(self.criticLearningRate),
            "actor_ppo_epochs": int(self.actorPpoEpochs),
            "critic_ppo_epochs": int(self.criticPpoEpochs),
            "advantage_normalization": (
                "none_categorical_target_retains_physical_cost_scale"
                if bool(getattr(self, "nmccPolicyImprovement"))
                else "zero_mean_unit_variance_with_machine_epsilon_only"
            ),
            "learning_reward_scale": float(self.learningRewardScale),
            "actor_credit_target": (
                "kl_constrained_categorical_rollout_policy_improvement"
                if bool(getattr(self, "nmccPolicyImprovement"))
                else (
                    "exact_common_noise_chosen_candidate_minus_policy_reference_candidate"
                    if bool(self.nmccEnabled)
                    else "complete_episode_return_minus_value_baseline"
                )
            ),
            **(
                {
                    "policy_improvement_learner": {
                        "candidate_coverage": (
                            "objective_greedy_contenders_plus_uniform_feasible_extras"
                        ),
                        "sampled_actions_per_state": int(
                            self.nmccPiMaxBranches
                        ),
                        "branch_comparator": (
                            "complete_costs_from_same_state_and_shared_futures"
                        ),
                        "branch_horizon": "to_episode_end",
                        "advantage": (
                            "complete_rollout_cost_categorical_advantages"
                        ),
                        "interval_cost_contract": (
                            "unchanged_complete_ledger_cost_to_episode_horizon"
                        ),
                        "actor_optimizer": (
                            "categorical_kl_fit_to_constrained_policy_target"
                        ),
                        "residual_correction": (
                            "unbounded_linear_actor_readout"
                        ),
                        "actor_epochs": int(self.nmccPiReplayEpochs),
                        "actor_replay_episodes": int(
                            self.nmccPiReplayMaxEpisodes
                        ),
                        "actor_replay_evidence": "physical_branch_scores",
                        "historical_probability_targets": False,
                        "random_reference_rollouts_per_state": int(
                            self.randomReferenceRollouts
                        ),
                        "continuation_policy": "frozen_current_residual_actor",
                        "shared_actor_critic_continuation_value": False,
                        "continuation_target": (
                            "belief_complete_simulator_cost"
                            if str(self.nmccPiTargetFutureMode) == "belief"
                            else "hindsight_realized_complete_simulator_cost"
                        ),
                        "target_future_mode": str(self.nmccPiTargetFutureMode),
                        "environment_reward_branch_horizon": 0,
                        "correction_gate": False,
                        "fallback_controller": False,
                        "model_filled_actions": False,
                        "entropy_diagnostics": {
                            "behavior": (
                                "sampled_policy_after_one_temperature_and_"
                                "uniform_mixture_application"
                            ),
                            "actor_before_after": (
                                "raw_temperature_one_trainable_policy"
                            ),
                        },
                    }
                }
                if bool(getattr(self, "nmccPolicyImprovement"))
                else {}
            ),
            "critic_target": critic_target,
            "optimizer": str(self.optimizer),
            "learner_batch_episodes": int(self.ppoRolloutEpisodes),
            "finalize_incomplete_learner_batch": bool(self.finalizePpoRollout),
            "counterfactual_branching": {
                "enabled": bool(
                    self.nmccEnabled or getattr(self, "nmccPolicyImprovement")
                ),
                "role": (
                    "simulator_graded_residual_rollout_policy_iteration"
                    if bool(getattr(self, "nmccPolicyImprovement"))
                    else "exact_credit_assignment_only"
                ),
                "comparison_horizon": (
                    "complete_simulation_to_episode_horizon"
                    if bool(getattr(self, "nmccPolicyImprovement"))
                    else comparison_horizon
                    if int(self.nmccPiBranchHorizon) > 0
                    else "immediate_pre_post_remaining_time_to_safety"
                ),
                "counterfactual_weight": float(self.nmccCounterfactualWeight),
                "reference_action": (
                    "objective_greedy_contenders_plus_uniform_feasible_extras"
                    if bool(getattr(self, "nmccPolicyImprovement"))
                    else "independent_draw_from_same_categorical_behavior_policy"
                ),
                "both_branches_install": bool(
                    self.nmccEnabled or getattr(self, "nmccPolicyImprovement")
                ),
                "missing_branch_behavior": "fail_training",
            },
            "exploration_schedule": {
                "kind": (
                    "exponential_half_life_categorical_temperature_entropy_and_uniform_mixture"
                    if self.explorationSchedule == "exponential_half_life"
                    else "half_cosine_categorical_temperature_entropy_and_uniform_mixture"
                ),
                "entropy_start": float(self.entropyCoefficientStart),
                "entropy_end": float(self.entropyCoefficientEnd),
                "temperature_start": float(self.actionTemperatureStart),
                "temperature_end": float(self.actionTemperatureEnd),
                "uniform_mixture_start": float(self.explorationRateStart),
                "uniform_mixture_end": float(self.explorationRateEnd),
                "schedule_updates": int(self.explorationDecayUpdates),
                "decay_updates": int(self.explorationDecayUpdates),
                "scope": "reset_independently_for_each_training_stage",
                "stage_id": str(self.trainingStageId),
                "stage_update_index": int(self.trainingStageUpdateIndex),
                "stage_total_updates": int(self.trainingStageTotalUpdates),
            },
            "learning_rate_schedule": {
                "kind": str(self.learningRateSchedule),
                "scope": "reset_independently_for_each_training_stage",
                "warmup_updates": int(self.learningRateWarmupUpdates),
                "decay_updates": int(self.learningRateDecayUpdates),
                "actor_start": float(self.actorLearningRate),
                "actor_end": float(
                    self.actorLearningRate
                    * self.actorLearningRateMinimumFraction
                ),
                "critic_start": float(self.criticLearningRate),
                "critic_end": float(
                    self.criticLearningRate
                    * self.criticLearningRateMinimumFraction
                ),
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
        self.casualtyPriorityKappa = float(self.casualtyPriorityKappa)
        self.casualtyPriceRule = str(self.casualtyPriceRule).strip().lower()
        self.publicObjectiveVariant = str(
            self.publicObjectiveVariant
        ).strip().lower()
        self.activePersonTimeWeight = float(self.activePersonTimeWeight)
        self.hazardExposureTimeWeight = float(self.hazardExposureTimeWeight)
        self.clearanceTailFraction = float(self.clearanceTailFraction)
        self.clearanceTailExtraWeight = float(self.clearanceTailExtraWeight)
        self.hazardSpreadSpeedMPerMinute = float(
            self.hazardSpreadSpeedMPerMinute
        )
        self.hazardSeverityTransitionMinutes = [
            float(value) for value in self.hazardSeverityTransitionMinutes
        ]
        self.hazardSevereRadiusM = float(self.hazardSevereRadiusM)
        self.shelterOverrunDangerLevel = int(self.shelterOverrunDangerLevel)
        self.cellX = int(self.cellX)
        self.cellY = int(self.cellY)
        self.cellPartitionMode = normalize_partition_mode(self.cellPartitionMode)
        self.cellPartitionMinWidthFraction = float(
            self.cellPartitionMinWidthFraction
        )
        if (
            not math.isfinite(self.casualtyPriorityKappa)
            or self.casualtyPriorityKappa < 1.0
        ):
            raise ValueError("casualtyPriorityKappa must be finite and at least 1")
        if self.casualtyPriceRule not in {"legacy", "harm_monotone"}:
            raise ValueError("casualtyPriceRule must be legacy or harm_monotone")
        if self.publicObjectiveVariant not in {"legacy", "front_aware"}:
            raise ValueError(
                "publicObjectiveVariant must be legacy or front_aware"
            )
        if (
            not math.isfinite(self.activePersonTimeWeight)
            or self.activePersonTimeWeight < 0.0
        ):
            raise ValueError("activePersonTimeWeight must be finite and non-negative")
        if (
            not math.isfinite(self.hazardExposureTimeWeight)
            or self.hazardExposureTimeWeight < 0.0
        ):
            raise ValueError(
                "hazardExposureTimeWeight must be finite and non-negative"
            )
        if self.activePersonTimeWeight + self.hazardExposureTimeWeight <= 0.0:
            raise ValueError(
                "At least one evacuation-time objective weight must be positive"
            )
        if (
            not math.isfinite(self.clearanceTailFraction)
            or not 0.0 < self.clearanceTailFraction <= 1.0
        ):
            raise ValueError("clearanceTailFraction must be in (0, 1]")
        if (
            not math.isfinite(self.clearanceTailExtraWeight)
            or self.clearanceTailExtraWeight < 0.0
        ):
            raise ValueError("clearanceTailExtraWeight must be finite and non-negative")
        if (
            not math.isfinite(self.hazardSpreadSpeedMPerMinute)
            or self.hazardSpreadSpeedMPerMinute <= 0.0
        ):
            raise ValueError(
                "hazardSpreadSpeedMPerMinute must be finite and positive"
            )
        if (
            len(self.hazardSeverityTransitionMinutes) != 4
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.hazardSeverityTransitionMinutes
            )
            or any(
                later <= earlier
                for earlier, later in zip(
                    self.hazardSeverityTransitionMinutes,
                    self.hazardSeverityTransitionMinutes[1:],
                )
            )
        ):
            raise ValueError(
                "hazardSeverityTransitionMinutes must contain four finite, "
                "positive, strictly increasing values"
            )
        if not math.isfinite(self.hazardSevereRadiusM) or self.hazardSevereRadiusM < 0.0:
            raise ValueError("hazardSevereRadiusM must be finite and non-negative")
        if not 1 <= self.shelterOverrunDangerLevel <= 5:
            raise ValueError("shelterOverrunDangerLevel must be in [1, 5]")
        self.shelterCanVol = int(self.shelterCanVol)
        self.initShelterVol = int(self.initShelterVol)
        self.shelterCapacityToken = int(self.shelterCapacityToken)
        self.shelterCapacitySurplus = int(self.shelterCapacitySurplus)
        self.shelterFailureReserveCount = int(self.shelterFailureReserveCount)
        self.maxAdditionalShelters = int(self.maxAdditionalShelters)
        self.shelterActionInterval = int(self.shelterActionInterval)
        self.trainingStageId = str(self.trainingStageId).strip()
        self.trainingStageUpdateIndex = int(self.trainingStageUpdateIndex)
        self.trainingStageTotalUpdates = int(self.trainingStageTotalUpdates)
        if not isinstance(
            self.requirePopulationMatchedShelterCapacity, (bool, np.bool_)
        ):
            raise ValueError(
                "requirePopulationMatchedShelterCapacity must be boolean"
            )
        self.requirePopulationMatchedShelterCapacity = bool(
            self.requirePopulationMatchedShelterCapacity
        )
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
        self.advantageScaleFloor = float(self.advantageScaleFloor)
        self.learningRewardScale = float(self.learningRewardScale)
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
        self.nmccNaturalLossCoefficient = float(self.nmccNaturalLossCoefficient)
        self.entropyCoefficientStart = float(self.entropyCoefficientStart)
        self.entropyCoefficientEnd = float(self.entropyCoefficientEnd)
        self.explorationDecayUpdates = int(self.explorationDecayUpdates)
        self.explorationSchedule = str(self.explorationSchedule).strip().lower()
        self.actionTemperatureStart = float(self.actionTemperatureStart)
        self.actionTemperatureEnd = float(self.actionTemperatureEnd)
        for attribute, _, kind, _ in NMCC_PI_ALL_CORE_FIELDS:
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
        self.hazardRegimeEnabled = bool(self.hazardRegimeEnabled)
        self.hazardWindShiftRatePerMinute = float(
            self.hazardWindShiftRatePerMinute
        )
        self.hazardWindShiftDegrees = [
            float(value) for value in self.hazardWindShiftDegrees
        ]
        self.hazardWindSpeedMultipliers = [
            float(value) for value in self.hazardWindSpeedMultipliers
        ]
        self.hazardWindMinDwellMinutes = int(
            self.hazardWindMinDwellMinutes
        )
        self.hazardForecastLeadMinutes = int(self.hazardForecastLeadMinutes)
        self.hazardForecastMissProbability = float(
            self.hazardForecastMissProbability
        )
        self.hazardForecastFalseAlarmRatePerMinute = float(
            self.hazardForecastFalseAlarmRatePerMinute
        )
        self.hazardIntensitySwitchRatePerMinute = float(
            self.hazardIntensitySwitchRatePerMinute
        )
        self.hazardIntensityMultipliers = [
            float(value) for value in self.hazardIntensityMultipliers
        ]
        self.hazardClosureRatePerMinute = float(
            self.hazardClosureRatePerMinute
        )
        self.hazardClosureDurationMinutes = [
            int(value) for value in self.hazardClosureDurationMinutes
        ]
        self.hazardClosureWarningLeadMinutes = int(
            self.hazardClosureWarningLeadMinutes
        )
        self.hazardSecondaryIgnitionRatePerMinute = float(
            self.hazardSecondaryIgnitionRatePerMinute
        )
        self.hazardSecondaryIgnitionWindow = [
            int(value) for value in self.hazardSecondaryIgnitionWindow
        ]
        self.forecastEnsembleSize = int(self.forecastEnsembleSize)
        self.nmccFrontForecastIntervals = int(
            self.nmccFrontForecastIntervals
        )
        self.actionSpace = str(self.actionSpace).strip().lower()
        self.nmccPiAllocation = str(self.nmccPiAllocation).strip().lower()
        self.nmccPiTargetFutureMode = (
            str(self.nmccPiTargetFutureMode).strip().lower()
        )
        self.deadlineGreedyThreshold = float(self.deadlineGreedyThreshold)
        if not isinstance(self.constructedDecoyDemand, (bool, np.bool_)):
            raise ValueError("constructedDecoyDemand must be boolean")
        self.constructedDecoyDemand = bool(self.constructedDecoyDemand)
        self.stressScenarioFamily = str(self.stressScenarioFamily)
        self.stressClusterShares = [
            float(value) for value in (self.stressClusterShares or [])
        ]
        self.stressClusterRadiusM = float(self.stressClusterRadiusM)
        self.stressClusterMinSeparationM = float(self.stressClusterMinSeparationM)
        self.stressThreatenedCluster = int(self.stressThreatenedCluster)
        self.stressThreatArrivalMinutes = float(self.stressThreatArrivalMinutes)
        self.stressWindAim = str(self.stressWindAim).strip().lower()
        stress_spec = StressScenarioSpec.from_core(self)
        if stress_spec.active and self.constructedDecoyDemand:
            raise ValueError(
                "Stress scenarios and constructedDecoyDemand are mutually exclusive"
            )
        if (
            stress_spec.requires_origin_plan
            and self.cellPartitionMode != POPULATION_DENSITY_ADAPTIVE
        ):
            raise ValueError(
                "Stress population and aimed-wind scenarios require "
                "cellPartitionMode=population_density_adaptive"
            )
        if stress_spec.threatened_cluster >= 0 and int(self.hazardVol) < 1:
            raise ValueError("A threatened-cluster scenario requires a hazard")
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
        HazardRegimeConfig(
            enabled=self.hazardRegimeEnabled,
            wind_shift_rate_per_minute=self.hazardWindShiftRatePerMinute,
            wind_shift_degrees=tuple(self.hazardWindShiftDegrees),
            wind_speed_multipliers=tuple(self.hazardWindSpeedMultipliers),
            wind_min_dwell_minutes=self.hazardWindMinDwellMinutes,
            forecast_lead_minutes=self.hazardForecastLeadMinutes,
            forecast_miss_probability=self.hazardForecastMissProbability,
            forecast_false_alarm_rate_per_minute=(
                self.hazardForecastFalseAlarmRatePerMinute
            ),
            intensity_switch_rate_per_minute=(
                self.hazardIntensitySwitchRatePerMinute
            ),
            intensity_multipliers=tuple(self.hazardIntensityMultipliers),
            closure_rate_per_minute=self.hazardClosureRatePerMinute,
            closure_duration_minutes=tuple(
                self.hazardClosureDurationMinutes
            ),
            closure_warning_lead_minutes=(
                self.hazardClosureWarningLeadMinutes
            ),
            secondary_ignition_rate_per_minute=(
                self.hazardSecondaryIgnitionRatePerMinute
            ),
            secondary_ignition_window=tuple(
                self.hazardSecondaryIgnitionWindow
            ),
            episode_horizon=max(1, int(self.stopTime) - 1),
        )
        if self.forecastEnsembleSize <= 0:
            raise ValueError("forecastEnsembleSize must be positive")
        if self.nmccFrontForecastIntervals < 1:
            raise ValueError("nmccFrontForecastIntervals must be at least 1")
        if self.actionSpace not in {"cell", "region_defer"}:
            raise ValueError("actionSpace must be cell or region_defer")
        if self.nmccPiAllocation not in {"uniform", "ocba"}:
            raise ValueError("nmccPiAllocation must be uniform or ocba")
        if self.nmccPiTargetFutureMode not in {"belief", "realized"}:
            raise ValueError("nmccPiTargetFutureMode must be belief or realized")
        if not math.isfinite(self.deadlineGreedyThreshold):
            raise ValueError("deadlineGreedyThreshold must be finite")
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
        if self.shelterCapacitySurplus < 0:
            raise ValueError("shelterCapacitySurplus must be non-negative")
        if self.shelterFailureReserveCount < 0:
            raise ValueError("shelterFailureReserveCount must be non-negative")
        if self.maxAdditionalShelters < 0:
            raise ValueError("maxAdditionalShelters must be non-negative")
        if self.shelterActionInterval <= 0:
            raise ValueError("shelterActionInterval must be positive")
        if not self.trainingStageId:
            raise ValueError("trainingStageId must not be empty")
        if self.trainingStageUpdateIndex < 0:
            raise ValueError("trainingStageUpdateIndex must be non-negative")
        if self.trainingStageTotalUpdates <= 0:
            raise ValueError("trainingStageTotalUpdates must be positive")
        if self.trainingStageUpdateIndex >= self.trainingStageTotalUpdates:
            raise ValueError(
                "trainingStageUpdateIndex must be smaller than "
                "trainingStageTotalUpdates"
            )
        if self.requirePopulationMatchedShelterCapacity:
            planned_shelters = int(self.initShelterVol) + int(
                self.maxAdditionalShelters
            )
            if planned_shelters <= 0:
                raise ValueError("Population-matched capacity requires shelters")
            if self.shelterFailureReserveCount >= planned_shelters:
                raise ValueError(
                    "shelterFailureReserveCount must be smaller than the "
                    "planned shelter count"
                )
            if int(self.shelterCanVol) < planned_shelters:
                raise ValueError(
                    "Population-matched capacity requires at least one "
                    "candidate for every planned shelter"
                )
            planned_capacity = planned_shelters * int(self.shelterCapacityToken)
            required_capacity = int(self.pedVol) + int(
                self.shelterCapacitySurplus
            )
            if planned_capacity != required_capacity:
                raise ValueError(
                    "Equal shelter capacity must sum to pedVol plus the "
                    "declared surplus: "
                    f"{planned_shelters} * {self.shelterCapacityToken} "
                    f"!= {self.pedVol} + {self.shelterCapacitySurplus}"
                )
            surviving_capacity = (
                planned_shelters - self.shelterFailureReserveCount
            ) * int(self.shelterCapacityToken)
            if surviving_capacity < int(self.pedVol):
                raise ValueError(
                    "Shelter failure reserve is infeasible: "
                    f"({planned_shelters} - {self.shelterFailureReserveCount}) "
                    f"* {self.shelterCapacityToken} = {surviving_capacity} "
                    f"< pedVol {self.pedVol}"
                )
            if int(self.initShelterVol) != 2 or int(self.maxAdditionalShelters) != 8:
                raise ValueError(
                    "Population-matched experiment requires 2 initial and "
                    "8 sequential shelters"
                )
            final_installation_time = (
                int(self.maxAdditionalShelters) * int(self.shelterActionInterval)
            )
            if int(self.stopTime) - 1 <= final_installation_time:
                raise ValueError(
                    "The episode must continue after the eighth installation"
                )
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
        if not math.isfinite(self.advantageScaleFloor) or self.advantageScaleFloor <= 0.0:
            raise ValueError("advantageScaleFloor must be finite and positive")
        if not math.isfinite(self.learningRewardScale) or self.learningRewardScale <= 0.0:
            raise ValueError("learningRewardScale must be finite and positive")
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
        nonnegative_nmcc = {
            "nmccInterventionCost": self.nmccInterventionCost,
            "nmccNaturalLossCoefficient": self.nmccNaturalLossCoefficient,
            "entropyCoefficientStart": self.entropyCoefficientStart,
            "entropyCoefficientEnd": self.entropyCoefficientEnd,
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in nonnegative_nmcc.values()
        ):
            raise ValueError("NMCC coefficients must be finite and non-negative")
        if self.explorationDecayUpdates <= 0:
            raise ValueError("explorationDecayUpdates must be positive")
        if self.explorationSchedule not in {
            "half_cosine",
            "exponential_half_life",
        }:
            raise ValueError(
                "explorationSchedule must be half_cosine or "
                "exponential_half_life"
            )
        if not (
            0.0 < self.actionTemperatureEnd <= self.actionTemperatureStart
        ):
            raise ValueError("action temperatures must satisfy 0 < end <= start")
        if self.entropyCoefficientEnd > self.entropyCoefficientStart:
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
        ordered_cells = tuple(self.cellTracker.cell_ids())
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
        regime = getattr(self, "hazard_regime", None)
        if regime is not None and regime.enabled:
            self._hazard_trajectory_hasher.update(
                json.dumps(
                    {
                        "wind_direction": regime.wind_direction_degrees,
                        "wind_speed_multiplier": regime.wind_speed_multiplier,
                        "intensity_multiplier": regime.intensity_multiplier,
                        "closures": sorted(regime.active_closures.items()),
                        "public": regime.public,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

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
    def _advance_exogenous_hazard(self, sim_time, timer=None):
        """Advance and record one complete exogenous hazard transition.

        The physical episode and the post-termination audit must consume the
        identical regime/spread/heat/smoke transition.  Keeping that sequence
        here prevents a policy's pedestrian stopping time from changing the
        matched-tape hazard digest used by evaluation.
        """
        if self.hazard_regime is not None and self.hazard_regime.enabled:
            self.hazard_regime.advance(sim_time, self.hazardDS, self.mapDS)

        self.hazardDS.spreadUpdate()
        if timer is not None:
            timer.lap("hazard.spreadUpdate")

        self.hazardDS.heatUpdate()
        if timer is not None:
            timer.lap("hazard.heatUpdate")

        self.hazardDS.smokeUpdate()
        if timer is not None:
            timer.lap("hazard.smokeUpdate")

        self.hazardDS.terminateHazard()
        self._record_hazard_state(sim_time)
        if timer is not None:
            timer.lap("hazard.terminateHazard")

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

            self._advance_exogenous_hazard(sim_time, tmr)

            closure = self.pedDS.resolveHazardShelterClosures(
                simulation_time=sim_time
            )
            if tmr is not None: tmr.lap("shelter.resolveHazardClosures")

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
                "shelter_closures": int(
                    getattr(self.shelterDS, "hazardOverrunCount", 0)
                ),
                "current_sheltered_population": int(
                    self.pedDS.currentShelteredPopulation()
                ),
                "cumulative_shelter_admissions": int(
                    self.pedDS.cumulativeShelterAdmissions
                ),
                "forced_reevacuated": int(
                    self.pedDS.cumulativeForcedReevacuated
                ),
                "closed_this_step": int(closure["closed_shelters"]),
                "reactivated_this_step": int(
                    closure["reactivated_population"]
                ),
                "retired_candidates_this_step": int(
                    closure.get("retired_candidates", 0)
                ),
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
                "reward_reference_potential": rl_out.get(
                    "reward_reference_potential", 0.0
                ),
                "reward_terminal_residual": rl_out.get(
                    "reward_terminal_residual", 0.0
                ),
                "reward_risk_time": rl_out.get("reward_risk_time", 0.0),
                "reward_remaining_risk_time_reduction": rl_out.get(
                    "reward_remaining_risk_time_reduction", 0.0
                ),
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
                "terminal_residual_person_time": rl_out.get(
                    "terminal_residual_person_time", 0.0
                ),
                "remaining_risk_time_before": rl_out.get(
                    "remaining_risk_time_before", 0.0
                ),
                "remaining_risk_time_after": rl_out.get(
                    "remaining_risk_time_after", 0.0
                ),
                "raw_remaining_risk_time_reduction": rl_out.get(
                    "raw_remaining_risk_time_reduction", 0.0
                ),
                "remaining_risk_time_reduction": rl_out.get(
                    "remaining_risk_time_reduction", 0.0
                ),
                "plain_time_wait": rl_out.get("plain_time_wait", 0.0),
                "plain_time_action": rl_out.get("plain_time_action", 0.0),
                "raw_plain_time_reduction": rl_out.get(
                    "raw_plain_time_reduction", 0.0
                ),
                "hazard_exposure_wait": rl_out.get(
                    "hazard_exposure_wait", 0.0
                ),
                "hazard_exposure_action": rl_out.get(
                    "hazard_exposure_action", 0.0
                ),
                "raw_hazard_exposure_reduction": rl_out.get(
                    "raw_hazard_exposure_reduction", 0.0
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
                "operational_shelter_capacity": float(sum(
                    getattr(sh, "shelterCap", 0.0)
                    for sh in self.shelterDS.shelterList.values()
                    if int(getattr(sh, "status", 0)) != 2
                )),
                "shelter_utilization": 0.0,
                **dict(self.pedDS.lastCongestionMetrics),
            }
            total_capacity = float(metrics["operational_shelter_capacity"])
            total_flow = float(sum(
                getattr(sh, "shelterFlow", 0.0)
                for sh in self.shelterDS.shelterList.values()
                if int(getattr(sh, "status", 0)) != 2
            ))
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
       
            authoritative_grid_shape(self)
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
        # Evaluate the physical terminal state before classifying survivors as
        # ``Unfinished`` and removing them from the live pedestrian database.
        # The realised HWTTS objective needs their capacity-valid remaining
        # route time (or full-horizon debt when none exists).
        rl_diagnostics = {}
        if self.rl is not None and hasattr(self.rl, "end_episode"):
            rl_diagnostics = self.rl.end_episode(
                finalize_rollout=bool(self.finalizePpoRollout)
            )
            if hasattr(self.shelterDS, "remainingCandidateCount"):
                print(f"[SHELTER POOL] remaining_candidates={self.shelterDS.remainingCandidateCount()} active_shelters={len(self.shelterDS.shelterList)}")

        finalized_remaining = 0
        if pending_before_finalize > 0 and hasattr(self.pedDS, "finalize_remaining_pedestrians"):
            finalized_remaining = int(self.pedDS.finalize_remaining_pedestrians(event="Unfinished"))
            self.pedDS.docuStatus()
            print(f"[EPISODE FINALIZE] unfinished={finalized_remaining} previously_active={pending_before_finalize}")

        # Policies can finish a minute or two apart after the final scheduled
        # installation. Complete only the exogenous hazard audit stream to the
        # configured horizon so matched-policy provenance compares the same
        # future length without extending pedestrian motion, rewards, or the
        # decision process.
        for audit_time in range(last_simulation_time + 1, int(self.stopTime)):
            self._advance_exogenous_hazard(audit_time)
                
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
            "schema_version": 7,
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
            "hazard_regime_events": (
                []
                if self.hazard_regime is None
                else [
                    {
                        "time": int(event["time"]),
                        "kind": str(event["kind"]),
                        "payload": dict(event["payload"]),
                    }
                    for event in self.hazard_regime.scheduled
                ]
            ),
            "decoy_scenario": self.decoy_scenario_metadata,
            "stress_scenario": self.stress_scenario_metadata,
            "initial_population": actual_population,
            "population_representation": (
                "individual"
                if int(self.pedestrianGroupSize) == 1
                else "bounded_initial_node_cohort"
            ),
            "maximum_persons_per_agent": int(self.pedestrianGroupSize),
            "initialized_agent_count": int(self.pedDS.initial_agent_count),
            "initial_node_group_count": int(len(self.pedDS.origin_groups)),
            "pedestrian_review_unit": "origin_node_cohort_per_simulator_transition",
            "grid_shape": list(authoritative_grid_shape(self)),
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
            "operational_shelters": int(sum(
                int(getattr(shelter, "status", 0)) != 2
                for shelter in self.shelterDS.shelterList.values()
            )),
            "overrun_shelters": int(sum(
                int(getattr(shelter, "status", 0)) == 2
                for shelter in self.shelterDS.shelterList.values()
            )),
            "shelter_overrun_events": [
                {
                    "shelter_id": int(shelter.shelterID),
                    "time": int(shelter.overrunTime),
                    "danger_level": int(shelter.overrunDangerLevel),
                    "forced_reevacuated_population": int(
                        shelter.forcedReevacuatedPopulation
                    ),
                }
                for shelter in sorted(
                    self.shelterDS.shelterList.values(),
                    key=lambda item: int(item.shelterID),
                )
                if int(getattr(shelter, "status", 0)) == 2
            ],
            "current_sheltered_population": int(
                self.pedDS.currentShelteredPopulation()
            ),
            "cumulative_shelter_admissions": int(
                self.pedDS.cumulativeShelterAdmissions
            ),
            "forced_reevacuated": int(
                self.pedDS.cumulativeForcedReevacuated
            ),
            "settled_stay_plain_person_minutes": float(
                self.pedDS.cumulativeSettledPlainPersonMinutes
            ),
            "settled_stay_exposure_person_minutes": float(
                self.pedDS.cumulativeSettledExposurePersonMinutes
            ),
            "total_shelter_capacity": int(
                sum(
                    int(getattr(shelter, "shelterCap", 0))
                    for shelter in self.shelterDS.shelterList.values()
                )
            ),
            "operational_shelter_capacity": int(
                sum(
                    int(getattr(shelter, "shelterCap", 0))
                    for shelter in self.shelterDS.shelterList.values()
                    if int(getattr(shelter, "status", 0)) != 2
                )
            ),
            "planned_capacity_surplus": int(self.shelterCapacitySurplus),
            "shelter_failure_reserve_count": int(
                self.shelterFailureReserveCount
            ),
            "planned_capacity_after_reserved_failures": int(
                max(
                    0,
                    self.initShelterVol
                    + self.maxAdditionalShelters
                    - self.shelterFailureReserveCount,
                )
                * self.shelterCapacityToken
            ),
            "unassigned_rational_after_final_installation": (
                -1
                if self.finalInstallationUnassignedRationalPopulation is None
                else int(
                    self.finalInstallationUnassignedRationalPopulation
                )
            ),
            "remaining_candidates": int(self.shelterDS.remainingCandidateCount()),
            "retired_candidates": int(self.shelterDS.retiredCandidateCount),
            "tokens_forfeited": int(
                max(0, self.rl.maximum_deployments - self.rl.deployments_made)
            ),
            "initial_observation_digest": self.rl.initial_observation_digest,
            "precommit_initial_observation_digest": (
                self.rl.precommit_initial_observation_digest
            ),
            "deployment_times": [
                int(value) for value in self.rl.deployment_times
            ],
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
