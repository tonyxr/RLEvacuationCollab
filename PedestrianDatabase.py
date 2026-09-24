#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/4: Constructed the skeletons
"""

import numpy as np
import math
from Pedestrian import Pedestrian
from SocialForce import ForceProcessor
from collections import defaultdict

class PedDS: 
    def __init__(self, pedNum, maximum_group_size=1):
        # total number of pedestrians in the environment
        self.pedNum = int(pedNum)
        # A represented agent may stand for a fixed cohort during scalable
        # training. Person-count accounting remains exact throughout the
        # simulator. The default of one preserves the individual microsimulation.
        self.maximum_group_size = int(maximum_group_size)
        if self.maximum_group_size <= 0:
            raise ValueError("maximum_group_size must be positive")
        # List of all pedestrian agent objects
        self.pedAgentList = {}
        """For stat documentation and analysis"""
        """Outcome definitions follow the original CASE 2024 model."""
        # the total number of arrivals at each timestep
        self.numArrival = {}
        # the total number of casualties at each timestep
        self.numCasualty = {}
        # the total number of successful evacuations at each timestep
        self.numEvacuated = {}
        # the total number of pedestrian agents impacted by any hazards at each timestep
        self.numAffected = {}
        
        """Pointer to other main model processors"""
        self.mapDS = None
        self.cellTracker = None
        self.hazardDS = None
        self.shelterDS = None
        # Social-force dynamics are intrinsic to the pedestrian process. Core
        # replaces this default with its configured processor.
        self.forceTracker = ForceProcessor()
        
        """Other key parameters"""
        self.currTime = 0
        self.maxSpeed = None
        self.timeStepMinutes = 1.0
        self.evacuation_horizon_timesteps = 1
        self.minOperationalSpeed = 1.0

        # allow to wait until X timesteps after simulation starts to document results 
        # (after the system is more stablized)
        self.docuStart = False
        # temporal container of event statistics at the current timestep
        self._step = dict(
            arrival=0,
            casualty=0,
            evacuated=0,
            unfinished=0,
            affected=0,
            panic_eligible_first_exposure=0,
            panic_onset=0,
            panic_herd_choices=0,
            panic_random_choices=0,
        )
        self.result = dict(self._step)
        self.evacuationTimeSum = 0.0
        self.evacuationCount = 0
        self.safeCompletionTimeSum = 0.0
        self.safeCompletionCount = 0
        self.travelDistanceByOutcome = defaultdict(float)
        self.travelPopulationByOutcome = defaultdict(int)
        
        self.shelter_osmid_map = None
        
        self.groups = defaultdict(set)
        self._route_distance_cache = {}
        self.hazard_random_seed = 0
        self.panic_random_seed = 0
        self.panic_rate = 0.0
        self.panic_herd_probability = 0.5
        self.panic_danger_threshold = 3
        self.casualty_reference_exposure_minutes = 60.0

        self.socialForcePersonSteps = 0.0
        self.socialSelfForcePersonSteps = 0.0
        self.socialImpactForcePersonSteps = 0.0
        self.socialSpeedRatioPersonSteps = 0.0

        # Link loads are synchronized at fixed internal substeps. These
        # accumulators are person-time weighted so summary statistics remain
        # valid if grouped pedestrians are reintroduced.
        self.congestionModel = None
        self.lastCongestionLinkStates = tuple()
        self.lastCongestionMetrics = {
            "active_population": 0,
            "assigned_link_population": 0,
            "occupied_physical_links": 0,
            "congested_population": 0,
            "mean_congestion_speed_ratio": 1.0,
            "minimum_congestion_speed_ratio": 1.0,
            "maximum_link_density_ped_per_m2": 0.0,
            "mean_effective_speed_m_per_minute": 0.0,
            "congestion_substeps": 1,
        }
        self.congestionPersonMinutes = 0.0
        self.congestionWeightedRatioSum = 0.0
        self.congestedPersonMinutes = 0.0
        self.maximumObservedLinkDensity = 0.0
        self.effectiveSpeedPersonMinuteSum = 0.0
        
        
    def _sample_group_sizes(self, total_population: int, min_size: int = 1, max_size: int = 50):
        remaining = int(max(0, total_population))
        if remaining <= 0:
            return []
        sizes = []
        while remaining > 0:
            sampled = int(np.random.randint(min_size, max_size + 1))
            gsize = int(min(remaining, sampled))
            sizes.append(gsize)
            remaining -= gsize
        return sizes
        
    
    def _speed_floor_value(self):
        base = float(self.maxSpeed) if self.maxSpeed is not None else 0.0
        dynamic_floor = 0.05 * base
        return float(max(self.minOperationalSpeed, dynamic_floor))

    def set_hazard_random_seed(self, seed: int) -> None:
        """Set the common-random-number key for pedestrian hazard outcomes."""
        self.hazard_random_seed = int(seed) & ((1 << 64) - 1)

    def configure_panic(
        self,
        *,
        rate: float,
        herd_probability: float = 0.5,
        danger_threshold: int = 3,
        random_seed: int = 0,
    ) -> None:
        rate = float(rate)
        herd_probability = float(herd_probability)
        danger_threshold = int(danger_threshold)
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("panic rate must be finite and in [0, 1]")
        if not math.isfinite(herd_probability) or not 0.0 <= herd_probability <= 1.0:
            raise ValueError("panic herd probability must be finite and in [0, 1]")
        if not 0 <= danger_threshold <= 5:
            raise ValueError("panic danger threshold must lie in [0, 5]")
        if rate > 0.0 and self.maximum_group_size != 1:
            raise ValueError(
                "panic behavior requires individual pedestrians (maximum_group_size=1)"
            )
        self.panic_rate = rate
        self.panic_herd_probability = herd_probability
        self.panic_danger_threshold = danger_threshold
        self.panic_random_seed = int(random_seed) & ((1 << 64) - 1)

    def panic_contract(self) -> dict:
        return {
            "model": "persistent_first_exposure_susceptibility_v2",
            "rate_among_first_exposed_pedestrians": self.panic_rate,
            "danger_level_threshold": self.panic_danger_threshold,
            "one_onset_trial_per_pedestrian": True,
            "persistent_after_onset": True,
            "herd_choice_probability": self.panic_herd_probability,
            "random_choice_probability": 1.0 - self.panic_herd_probability,
            "herd_rule": "incident physical edge with greatest frozen active occupancy",
            "population_representation": "individual_required_when_rate_positive",
        }

    def _panic_uniform(self, pedestrian_id: int, counter: int, channel: int) -> float:
        """Counter-based U(0,1) draw invariant to policy iteration order."""
        mask = (1 << 64) - 1
        value = int(self.panic_random_seed) & mask
        for term, multiplier in (
            (int(self.currTime) + 1, 0x9E3779B97F4A7C15),
            (int(pedestrian_id) + 1, 0xBF58476D1CE4E5B9),
            (int(counter) + 1, 0x94D049BB133111EB),
            (int(channel) + 1, 0xD6E8FEB86659FD93),
        ):
            value = (value + multiplier * term) & mask
            value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return float((value >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def _panic_susceptibility_uniform(self, pedestrian_id: int) -> float:
        """Return one policy-invariant latent panic-susceptibility draw.

        Susceptibility belongs to the individual, so it must not depend on the
        timestep at which a particular policy first exposes that person to a
        dangerous cell.  Route-choice draws remain time-varying and continue
        to use :meth:`_panic_uniform`.
        """
        mask = (1 << 64) - 1
        value = int(self.panic_random_seed) & mask
        value = (value + 0xBF58476D1CE4E5B9 * (int(pedestrian_id) + 1)) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return float((value >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def _hazard_uniform(self, pedestrian_id: int) -> float:
        """Return a deterministic U(0,1) shock for this person and timestep.

        Counter-based draws are invariant to iteration order and to how many
        pedestrians another policy has already evacuated. Matched policies
        therefore retain common casualty shocks after their states diverge.
        """
        mask = (1 << 64) - 1
        value = self.hazard_random_seed
        value = (value + 0x9E3779B97F4A7C15 * (int(self.currTime) + 1)) & mask
        value = (value + 0xBF58476D1CE4E5B9 * (int(pedestrian_id) + 1)) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return float((value >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    def _hazard_casualty_count(
        self,
        pedestrian_id: int,
        represented_population: int,
        probability: float,
    ) -> int:
        """Draw exact binomial casualties for a represented cohort.

        Individual mode retains the original counter-based Bernoulli draw.
        Cohort mode uses a deterministic, policy-paired seed keyed by episode
        hazard stream, timestep, and represented-agent identifier.
        """
        population = max(1, int(represented_population))
        probability = max(0.0, min(1.0, float(probability)))
        if probability <= 0.0:
            return 0
        if probability >= 1.0:
            return population
        if population == 1:
            return int(self._hazard_uniform(pedestrian_id) < probability)
        mask = (1 << 64) - 1
        seed = int(self.hazard_random_seed) & mask
        seed ^= (0x9E3779B97F4A7C15 * (int(self.currTime) + 1)) & mask
        seed ^= (0xBF58476D1CE4E5B9 * (int(pedestrian_id) + 1)) & mask
        rng = np.random.default_rng(seed)
        return int(rng.binomial(population, probability))
        
        
    def _distance_sq_to_node(self, ped, node):
        if ped is None or node is None:
            return float("inf")
        dx = float(ped.lastX) - float(node.nodeX)
        dy = float(ped.lastY) - float(node.nodeY)
        return dx * dx + dy * dy

    def _route_anchor_node(self, ped):
        """
        Pick a stable node for replanning:
        - at node: current node
        - on edge: preferred edge destination node (nearest upcoming node)
        - fallback: current node
        """
        if ped is None:
            return None
        if getattr(ped, "atNode", False) and getattr(ped, "currNode", None) is not None:
            return ped.currNode
        edge_dest = getattr(ped, "edge_dest_node", None)
        if edge_dest is not None:
            return edge_dest
        return getattr(ped, "currNode", None)
    
    
    def _route_distance(self, start_node, end_node):
        """
        Compute shortest-path distance between two nodes from route edge lengths.
        Returns +inf when no path can be found.
        """
        if start_node is None or end_node is None or self.mapDS is None:
            return float("inf")
        src = int(getattr(start_node, "OSMID", -1))
        dst = int(getattr(end_node, "OSMID", -2))
        if src == dst:
            return 0.0
        key = (src, dst)
        if key in self._route_distance_cache:
            return float(self._route_distance_cache[key])
        distance_lookup = getattr(self.mapDS, "networkDistanceToTarget", None)
        if callable(distance_lookup):
            distance = float(distance_lookup(start_node, end_node))
            self._route_distance_cache[key] = distance
            return distance
        try:
            route = self.mapDS.shortestPath(start_node, end_node)
        except Exception:
            route = None
        if route is None:
            self._route_distance_cache[key] = float("inf")
            return float("inf")
        dist = 0.0
        for edge in (getattr(route, "edgeRemained", None) or []):
            dist += float(max(0.0, getattr(edge, "edgeLen", 0.0)))
        self._route_distance_cache[key] = float(dist)
        return float(dist)
    
    @staticmethod
    def _cell_in_moore_ring(cell, center_cell):
        """
        True iff `cell` is one of the 8 neighboring cells around `center_cell`
        (Moore neighborhood ring, excluding center itself).
        """
        if cell is None or center_cell is None:
            return False
        try:
            ci, cj = int(cell[0]), int(cell[1])
            ai, aj = int(center_cell[0]), int(center_cell[1])
        except Exception:
            return False

        di = abs(ci - ai)
        dj = abs(cj - aj)
        return (di <= 1 and dj <= 1) and not (di == 0 and dj == 0)

    def _shelter_has_remaining_capacity(self, shelter, pedestrian=None):
        if shelter is None:
            return False
        return bool(
            self.shelterDS.availableCapacity(
                shelter, for_pedestrian=pedestrian
            ) > 0
        )

    def _reroute_pedestrian_to_shelter(self, ped, target_shelter):
        if ped is None or target_shelter is None or self.mapDS is None:
            return 0
        if bool(getattr(ped, "panicked", False)):
            return 0
        target_node = getattr(target_shelter, "nodeMapped", None)
        if target_node is None:
            return 0

        anchor = self._route_anchor_node(ped)
        if anchor is None:
            return 0

        try:
            newRoute = self.mapDS.shortestPath(anchor, target_node)
        except Exception:
            newRoute = None

        if newRoute is None:
            return 0

        reserved = int(self.shelterDS.reserveShelter(ped, target_shelter))
        if reserved <= 0:
            return 0
        ped.routeFollowing = newRoute
        return int(reserved)

    def _closest_open_shelter_from_ped(self, ped, exclude_osmid = None):
        if ped is None or self.shelterDS is None:
            return None

        best = None
        best_key = None
        anchor = self._route_anchor_node(ped)
        px = float(getattr(ped, "lastX", 0.0))
        py = float(getattr(ped, "lastY", 0.0))

        for shelter in self.shelterDS.shelterList.values():
            sh_node = getattr(shelter, "nodeMapped", None)
            if sh_node is None:
                continue
            sh_osmid = getattr(sh_node, "OSMID", None)
            if exclude_osmid is not None and sh_osmid == exclude_osmid:
                continue
            if not self._shelter_has_remaining_capacity(shelter, pedestrian=ped):
                continue

            distance = self._route_distance(anchor, sh_node)
            if not np.isfinite(distance):
                distance = math.hypot(
                    float(sh_node.nodeX) - px,
                    float(sh_node.nodeY) - py,
                )
            key = (distance, int(getattr(shelter, "shelterID", 0)))
            if best_key is None or key < best_key:
                best_key = key
                best = shelter

        return best

    def route_active_to_nearest_shelter(self) -> int:
        """Reserve initial capacity by risk-weighted time-to-safety benefit."""
        if self.mapDS is None or self.shelterDS is None:
            raise RuntimeError("Pedestrian, map, and shelter databases must be connected first")
        shelters = list(self.shelterDS.shelterList.values())
        if not shelters:
            return 0

        self._route_distance_cache = {}
        remaining_minutes = max(
            float(self.timeStepMinutes),
            float(self.evacuation_horizon_timesteps) * self.timeStepMinutes,
        )
        pairs = []
        for pedestrian in self.pedAgentList.values():
            if bool(getattr(pedestrian, "terminated", False)):
                continue
            if bool(getattr(pedestrian, "panicked", False)):
                continue
            anchor = self._route_anchor_node(pedestrian)
            if anchor is None:
                continue
            speed = max(
                self._speed_floor_value(),
                float(getattr(pedestrian, "currSpeed", 0.0)),
            )
            danger = 0.0
            if self.cellTracker is not None:
                danger = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            self.cellTracker.getCellState(pedestrian.currCell)
                        )
                        / 5.0,
                    ),
                )
            for shelter in shelters:
                distance = self._route_distance(anchor, shelter.nodeMapped)
                if not np.isfinite(distance):
                    continue
                benefit = (remaining_minutes - distance / speed) * (1.0 + danger)
                pairs.append((
                    -benefit,
                    distance,
                    int(shelter.shelterID),
                    int(pedestrian.agentID),
                    pedestrian,
                    shelter,
                ))

        assigned = set()
        reserved_population = 0
        for _, _, _, pedestrian_id, pedestrian, shelter in sorted(pairs):
            if pedestrian_id in assigned:
                continue
            if not self._shelter_has_remaining_capacity(
                shelter, pedestrian=pedestrian
            ):
                continue
            reserved = self._reroute_pedestrian_to_shelter(
                pedestrian, shelter
            )
            if reserved > 0:
                assigned.add(pedestrian_id)
                reserved_population += reserved
        return int(reserved_population)

    def reroute_to_new_shelter_if_closer(self, newShelter):
        """
        After a new shelter is installed, replan active pedestrians whose current
        intended shelter is farther from their current location than this new shelter.
        """
        if newShelter is None or self.mapDS is None or self.shelterDS is None:
            return 0
        
        if not self._shelter_has_remaining_capacity(newShelter):
            return 0

        rerouted_population = 0
        self._route_distance_cache = {}
        shelter_by_osmid = getattr(self.shelterDS, "shelterByOSMID", {}) or {}
        new_node = getattr(newShelter, "nodeMapped", None)
        if new_node is None:
            return 0
        beneficiaries = []
        for ped in list(self.pedAgentList.values()):
            if getattr(ped, "terminated", False):
                continue
            if bool(getattr(ped, "panicked", False)):
                continue
            anchor = self._route_anchor_node(ped)
            if anchor is None:
                continue
            route = getattr(ped, "routeFollowing", None)
            old_target = getattr(route, "endNode", None) if route is not None else None
            old_shelter = shelter_by_osmid.get(getattr(old_target, "OSMID", None)) if old_target is not None else None

            compare_shelter = old_shelter
            if compare_shelter is None:
                compare_shelter = self._closest_open_shelter_from_ped(
                    ped,
                    exclude_osmid=getattr(new_node, "OSMID", None),
                )
            
            old_node = getattr(compare_shelter, "nodeMapped", None) if compare_shelter is not None else None
            old_dist = self._route_distance(anchor, old_node) if old_node is not None else float("inf")
            new_dist = self._route_distance(anchor, new_node)
            if not np.isfinite(new_dist):
                new_dist = math.sqrt(max(0.0, self._distance_sq_to_node(ped, new_node)))
                old_dist = math.sqrt(max(0.0, self._distance_sq_to_node(ped, old_node))) if old_node is not None else float("inf")
            speed = max(
                self._speed_floor_value(),
                float(getattr(ped, "currSpeed", 0.0)),
            )
            if not np.isfinite(old_dist):
                remaining_minutes = max(
                    float(self.timeStepMinutes),
                    float(
                        max(
                            1,
                            int(self.evacuation_horizon_timesteps)
                            - int(self.currTime),
                        )
                    )
                    * float(self.timeStepMinutes),
                )
                old_dist = speed * remaining_minutes
            if new_dist + 1e-6 >= old_dist:
                continue
            danger = 0.0
            if self.cellTracker is not None and getattr(ped, "currCell", None) is not None:
                danger = max(
                    0.0,
                    min(1.0, float(self.cellTracker.getCellState(ped.currCell)) / 5.0),
                )
            marginal_risk_time = (old_dist - new_dist) / speed * (1.0 + danger)
            beneficiaries.append(
                (-marginal_risk_time, int(getattr(ped, "agentID", 0)), ped)
            )

        # Capacity goes first to the largest per-person reduction in the same
        # T+E proxy used by the policy. Stable identifiers remove iteration
        # order as an allocation mechanism.
        for _, _, ped in sorted(beneficiaries):
            reserved = self._reroute_pedestrian_to_shelter(ped, newShelter)
            rerouted_population += int(reserved)
            if not self._shelter_has_remaining_capacity(newShelter):
                break

        return int(rerouted_population)
    
    def remaining_active_count(self):
        return int(
            sum(
                0
                if getattr(p, "terminated", False)
                else max(1, int(getattr(p, "group_size", 1)))
                for p in self.pedAgentList.values()
            )
        )

    def finalize_remaining_pedestrians(self, event: str = "Arrival") -> int:
        active = [p for p in list(self.pedAgentList.values()) if not getattr(p, "terminated", False)]
        active_population = sum(max(1, int(getattr(p, "group_size", 1))) for p in active)
        for ped in active:
            self.terminatePedestrianAgent(ped, event)
        return int(active_population)

    def reroute_if_target_shelter_unavailable(self, ped):
        """
        If ped's current target shelter has no remaining capacity, reroute to the
        closest shelter (from current location) that still has positive remaining
        capacity.
        """
        if ped is None or self.shelterDS is None:
            return False
        if bool(getattr(ped, "panicked", False)):
            return False
        
        route = getattr(ped, "routeFollowing", None)
        target = getattr(route, "endNode", None) if route is not None else None
        if target is None:
            return False

        target_sh = self.shelterDS.shelterByOSMID.get(getattr(target, "OSMID", None))
        if target_sh is None:
            return False
        assigned_shelter, assigned_population = self.shelterDS.reservationFor(ped)
        if assigned_shelter is target_sh and int(assigned_population) > 0:
            return False

        # A route without a commitment must claim capacity before proceeding.
        if self._shelter_has_remaining_capacity(target_sh, pedestrian=ped):
            return bool(self._reroute_pedestrian_to_shelter(ped, target_sh))

        next_sh = self._closest_open_shelter_from_ped(
            ped,
            exclude_osmid = getattr(target_sh.nodeMapped, "OSMID", None)
        )
        if next_sh is None:
            return False

        return bool(self._reroute_pedestrian_to_shelter(ped, next_sh))
    
    """Competency check for other related processors"""
    def checkReady(self, mapDS = None, cellTracker = None, maxSpeed = None,
                   hazardDS = None, shelterDS = None, forceTracker = None,
                   congestionModel = None, timeStepMinutes = None,
                   casualtyReferenceExposureMinutes = None,
                   evacuationHorizonTimesteps = None):
        
        if mapDS is not None: self.mapDS = mapDS
        if cellTracker is not None: self.cellTracker = cellTracker
        if maxSpeed is not None: self.maxSpeed = float(maxSpeed)
        self.minOperationalSpeed = float(max(0.5, 0.05 * float(self.maxSpeed or 0.0)))
        if hazardDS is not None: self.hazardDS = hazardDS
        if shelterDS is not None:
            self.shelterDS = shelterDS
            self._shelter_osmid_map = {sh.nodeMapped.OSMID: sh for sh in self.shelterDS.shelterList.values()}
        if forceTracker is not None: self.forceTracker = forceTracker
        if congestionModel is not None: self.congestionModel = congestionModel
        if timeStepMinutes is not None:
            value = float(timeStepMinutes)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("timeStepMinutes must be finite and positive")
            self.timeStepMinutes = value
        if evacuationHorizonTimesteps is not None:
            self.evacuation_horizon_timesteps = max(
                1, int(evacuationHorizonTimesteps)
            )
        if casualtyReferenceExposureMinutes is not None:
            value = float(casualtyReferenceExposureMinutes)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "casualtyReferenceExposureMinutes must be finite and positive"
                )
            self.casualty_reference_exposure_minutes = value
    
    """This function is called at the start of the interaction process at each timestep to document all interaction events at this timestep"""
    def startDocument(self):
        if not self.docuStart:
            self._step = {
                key: 0
                for key in (
                    "arrival",
                    "casualty",
                    "evacuated",
                    "unfinished",
                    "affected",
                    "panic_eligible_first_exposure",
                    "panic_onset",
                    "panic_herd_choices",
                    "panic_random_choices",
                )
            }
            self.docuStart = True
            
    """Update the interaction events happened in the current timestep to the overall status set"""
    def bump(self, key, n=1):
        if not self.docuStart:
            self.startDocument()
        self._step[key] = int(self._step.get(key, 0)) + int(n)
    
    """At the end of each timestep/system iteration, update this timestep's overall statistics from the temporal container to official containers"""
    def docuStatus(self):
        for k, v in self._step.items():
            self.result[k] = int(self.result.get(k, 0)) + int(v)
            #self.result[k] = int(self.result.get(k, 0)) + int(self._step.get(k, 0))
        
        t = self.currTime
        self.numArrival[t] = self.result['arrival']
        self.numCasualty[t] = self.result['casualty']
        self.numEvacuated[t] = self.result["evacuated"]
        self.numAffected[t] = self.result["affected"]
        
        self.currTime += 1
        self.docuStart = False
        
    """Pedestrian Motion helper functions"""
    def unitVector(self, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        return ((0.0, 0.0), 0.0) if d == 0.0 else ((dx/d, dy/d), d)

    def updatePanicStates(self) -> int:
        """Evaluate fixed panic susceptibility on first qualifying exposure.

        ``panic_rate`` is the expected fraction of exposed pedestrians who
        panic, not a per-minute hazard that compounds toward one.  Once the
        first danger-threshold exposure is evaluated, a non-susceptible person
        remains rational; a susceptible person remains panicked.
        """
        if self.panic_rate <= 0.0 or self.cellTracker is None:
            return 0
        onsets = 0
        for ped in tuple(self.pedAgentList.values()):
            if bool(getattr(ped, "terminated", False)) or bool(
                getattr(ped, "panicked", False)
            ) or bool(getattr(ped, "panic_eligibility_evaluated", False)):
                continue
            if int(getattr(ped, "group_size", 1)) != 1:
                raise RuntimeError("panic onset encountered a non-individual pedestrian")
            danger = int(self.cellTracker.getCellState(ped.currCell))
            if danger < self.panic_danger_threshold:
                continue
            ped.panic_eligibility_evaluated = True
            self.bump("panic_eligible_first_exposure", 1)
            if self._panic_susceptibility_uniform(ped.agentID) < self.panic_rate:
                ped.panicked = True
                ped.panic_onset_time = int(self.currTime)
                ped.routeFollowing = None
                ped.panic_next_edge = None
                if self.shelterDS is not None:
                    self.shelterDS.releaseReservation(ped)
                self.bump("panic_onset", 1)
                onsets += 1
        return onsets

    def active_panicked_count(self) -> int:
        return int(
            sum(
                max(1, int(getattr(ped, "group_size", 1)))
                for ped in self.pedAgentList.values()
                if not bool(getattr(ped, "terminated", False))
                and bool(getattr(ped, "panicked", False))
            )
        )

    def _edge_load_key(self, edge):
        if self.congestionModel is not None:
            return self.congestionModel.physical_link_key(edge)
        return ("edge", int(getattr(edge, "edgeID", id(edge))))

    def _travelling_edge_loads(self, pedestrians) -> dict:
        """Freeze currently travelling physical-edge occupancy for herding."""
        loads = defaultdict(int)
        for ped in pedestrians:
            if bool(getattr(ped, "terminated", False)) or bool(
                getattr(ped, "atNode", False)
            ):
                continue
            edge = getattr(ped, "currEdge", None)
            if edge is not None:
                loads[self._edge_load_key(edge)] += max(
                    1, int(getattr(ped, "group_size", 1))
                )
        return dict(loads)

    def _choose_panic_edge(self, ped, frozen_loads):
        available = tuple(self.mapDS.incidentEdges(ped.currNode))
        if not available:
            return None
        counter = int(getattr(ped, "panic_decision_count", 0))
        herd = (
            self._panic_uniform(ped.agentID, counter, 1)
            < self.panic_herd_probability
        )
        if herd:
            largest = max(
                int(frozen_loads.get(self._edge_load_key(edge), 0))
                for edge in available
            )
            choices = tuple(
                edge
                for edge in available
                if int(frozen_loads.get(self._edge_load_key(edge), 0)) == largest
            )
            self.bump("panic_herd_choices", 1)
        else:
            choices = available
            self.bump("panic_random_choices", 1)
        draw = self._panic_uniform(ped.agentID, counter, 2)
        index = min(len(choices) - 1, int(draw * len(choices)))
        ped.panic_decision_count = counter + 1
        return choices[index]

    def _prepare_panic_choices(self, pedestrians, frozen_loads) -> None:
        for ped in pedestrians:
            if (
                bool(getattr(ped, "panicked", False))
                and bool(getattr(ped, "atNode", False))
                and getattr(ped, "panic_next_edge", None) is None
            ):
                # A panicked pedestrian enters any available shelter reached by
                # chance before making a further road choice.
                at_shelter = bool(
                    self.shelter_osmid_map
                    and getattr(ped, "currNode", None) is not None
                    and ped.currNode.OSMID in self.shelter_osmid_map
                )
                if not at_shelter:
                    ped.panic_next_edge = self._choose_panic_edge(ped, frozen_loads)
    
    """Load the active shelter lookup used for node-arrival admission."""
    def loadShelterLookup(self, shByOSMID):
        self.shelter_osmid_map = shByOSMID
    
    """Resolve pedestrian arrival at a network node or active shelter."""
    def arrive_node(self, ped, node):
        """Place ``ped`` at ``node`` and resolve a targeted shelter arrival.

        Returns ``True`` when movement may continue from the node.  Returns
        ``False`` after admission or when a full target has no feasible
        alternative.  A rejected pedestrian is never converted into a generic
        route-completion arrival.
        """
        ped.currNode = node
        ped.atNode = True
        ped.currEdge = None
        ped.edge_dest_mode = None
        ped.panic_next_edge = None
        
        ped.lastX, ped.lastY = node.nodeX, node.nodeY
        ped.currCell = self.cellTracker.locateCell(ped.lastX, ped.lastY)
        
        osmid = node.OSMID
        
        route = getattr(ped, "routeFollowing", None)
        target = getattr(route, "endNode", None) if route is not None else None
        is_target = target is None or getattr(target, "OSMID", None) == osmid

        # Only the route destination is an admission attempt. A route may pass
        # through another shelter node, which must remain an ordinary waypoint.
        if (
            is_target
            and self.shelter_osmid_map
            and osmid in self.shelter_osmid_map
            and self.shelterDS
        ):
            sh = self.shelter_osmid_map[osmid]
            group_size = max(1, int(getattr(ped, "group_size", 1)))
            if group_size > 1:
                admitted = int(
                    self.shelterDS.admitShelterPopulation(
                        group_size, sh, pedAgent=ped
                    )
                )
                if admitted <= 0:
                    if bool(getattr(ped, "panicked", False)):
                        return True
                    return bool(self.reroute_if_target_shelter_unavailable(ped))
                if admitted < group_size:
                    self.travelDistanceByOutcome["Evacuated"] += (
                        float(getattr(ped, "distance_travelled_m", 0.0)) * admitted
                    )
                    self.travelPopulationByOutcome["Evacuated"] += admitted
                    self.bump("evacuated", admitted)
                    event_time = (
                        float(self.currTime + 1) * float(self.timeStepMinutes)
                    )
                    self.evacuationTimeSum += event_time * admitted
                    self.evacuationCount += admitted
                    self.safeCompletionTimeSum += event_time * admitted
                    self.safeCompletionCount += admitted
                    ped.group_size = int(group_size - admitted)
                    return bool(self.reroute_if_target_shelter_unavailable(ped))
                self.terminatePedestrianAgent(ped, "Evacuated")
                return False
            status = self.shelterDS.updateShelterFlow(ped, sh)
            if status == 0:
                self.terminatePedestrianAgent(ped, "Evacuated")
                return False
            else:
                if bool(getattr(ped, "panicked", False)):
                    return True
                return bool(self.reroute_if_target_shelter_unavailable(ped))
        return True
    
    def advanceFromNode(self, ped, frozen_panic_loads=None):
        """
        When the pedestrian agent is at a node, choose the next edge the pedestrian should follow from its routeFollowing
        Start walking the agent in this timestep on that edge (emit from first element in routeFollowing's edgeRemain list)
        """
        
        if bool(getattr(ped, "panicked", False)):
            e = getattr(ped, "panic_next_edge", None)
            if e is None:
                e = self._choose_panic_edge(ped, frozen_panic_loads or {})
            ped.panic_next_edge = None
            if e is None:
                return False
        else:
            if ped.routeFollowing is None:
                return False
            try:
                e = ped.routeFollowing.getNextEdge()
            except (IndexError, AttributeError):
                return False
        
        # pick edge direction that is consistent with the current node, avoid possible error
        if ped.currNode and e.startNode.OSMID == ped.currNode.OSMID:
            start, end = e.startNode, e.endNode
        elif ped.currNode and e.endNode.OSMID == ped.currNode.OSMID:
            start, end = e.endNode, e.startNode
        else:
            start, end = e.startNode, e.endNode
        
        (ux, uy), seg_len = self.unitVector(start.nodeX, start.nodeY, end.nodeX, end.nodeY)
        ped.currEdge = e
        ped.atNode = False
        ped.edge_remain = float(max(0.0, e.edgeLen if hasattr(e, "edgeLen") else seg_len))
        ped.edge_vec = (ux, uy)
        ped.edge_dest_node = end
        return True
    
    """Primary functions"""
        
    def initPedestrianAgent(self, mapDS, cellTracker, maxSpeed):
        """
        Initialize each pedestrian at a sampled birth node. Core immediately
        assigns the nearest reachable active shelter after the databases are
        connected, so constructing a throwaway random origin--destination
        shortest path here would change no behavior and is prohibitively
        expensive for 50,000 pedestrians.
        """
        
        self.mapDS = mapDS
        self.cellTracker = cellTracker
        self.maxSpeed = float(maxSpeed)
        
        pedID = 0
        remaining_population = int(max(0, self.pedNum))
        while remaining_population > 0:
            represented_population = min(
                int(self.maximum_group_size),
                remaining_population,
            )
            group_id = pedID + 1
            #print("Current agent id, ", str(pedID))
            
            birthNode = mapDS.assignGenerationNode()
            assignedRoute = None
            currNode = birthNode
            
            currEdge = None
            lastX, lastY = currNode.nodeX, currNode.nodeY
            currCell = cellTracker.locateCell(lastX, lastY)
            currSpeed = maxSpeed
            
            affected = False
            atNode = True
            casualty = False
            evacuated = False
            arrival = False
            terminated = False
            
            agentID = pedID
            pedID += 1
            newAgent = Pedestrian(agentID, assignedRoute, currNode, currEdge, currCell, lastX, 
                                  lastY, currSpeed, affected, atNode, casualty, evacuated, arrival, terminated)
            
            newAgent.group_id = int(group_id)
            newAgent.group_size = int(represented_population)
            self.groups[newAgent.group_id].add(agentID)
            
            newAgent.edge_remain = 0.0 # at node, edge_remain = 0
            newAgent.edge_vec = None
            newAgent.edge_dest_node = None    
            newAgent.desired_speed = float(self.maxSpeed)
            
            self.pedAgentList[agentID] = newAgent
            remaining_population -= represented_population

        initialized_population = sum(
            max(1, int(getattr(pedestrian, "group_size", 1)))
            for pedestrian in self.pedAgentList.values()
        )
        if initialized_population != int(max(0, self.pedNum)):
            raise RuntimeError(
                f"Initialized population {initialized_population}, expected {self.pedNum}"
            )
            
    def terminatePedestrianAgent(self, ped, event: str):
        """
        - Check which exit conditions the given pedestrian matches,
        - Formally emit the pedestrian agent from the active environment
        - Document the exit instance accordingly
        """
        self.startDocument()
        
        # avoid double document
        if getattr(ped, "terminated", False):
            return

        if self.shelterDS is not None:
            self.shelterDS.releaseReservation(ped)

        group_size = max(1, int(getattr(ped, "group_size", 1)))
        self.travelDistanceByOutcome[str(event)] += (
            float(getattr(ped, "distance_travelled_m", 0.0)) * group_size
        )
        self.travelPopulationByOutcome[str(event)] += group_size
        
        # Step 1: (if statement) Check if casualty, if so, casualty count += 1. Set p_i's casualty and terminated status = True.
        if event == "Casualty":
            ped.casualty = True
            self.bump('casualty', int(getattr(ped, "group_size", 1)))
        # Step 2: (if statement) Check if evacuated, if so, evacuated count += 1. Set p_i's evacuated and terminated status = True. Call UpdateShelterFlow to update the flow. 
        elif event == "Evacuated":
            ped.evacuated = True
            gsize = int(getattr(ped, "group_size", 1))
            self.bump('evacuated', gsize)
            # Events occur during the simulator transition whose boundary is
            # currTime + 1; recording currTime alone makes every completion
            # one timestep early.
            event_time = float(self.currTime + 1) * float(self.timeStepMinutes)
            self.evacuationTimeSum += event_time * gsize
            self.evacuationCount += gsize
            self.safeCompletionTimeSum += event_time * gsize
            self.safeCompletionCount += gsize
        elif event == "Arrival":
            ped.arrival = True
            gsize = int(getattr(ped, "group_size", 1))
            self.bump("arrival", gsize)
            self.safeCompletionTimeSum += (
                float(self.currTime + 1) * float(self.timeStepMinutes) * gsize
            )
            self.safeCompletionCount += gsize
        elif event == "Unfinished":
            ped.unfinished = True
            self.bump("unfinished", int(getattr(ped, "group_size", 1)))
        else:
            raise ValueError(f"Unsupported pedestrian termination event: {event!r}")
        # Step 4: Remove $p_i$ from the active pedestrian agent list. 
        ped.terminated = True
        if ped.agentID in self.pedAgentList:
            del self.pedAgentList[ped.agentID]
    
    def mean_evacuation_time(self) -> float:
        if int(self.evacuationCount) <= 0:
            return 0.0
        return float(self.evacuationTimeSum) / float(self.evacuationCount)

    def mean_safe_completion_time(self) -> float:
        if int(self.safeCompletionCount) <= 0:
            return 0.0
        return float(self.safeCompletionTimeSum) / float(self.safeCompletionCount)
    
    def interPedestrianInteraction(self):
        """
        For group evacuating purpose,
        Unify the speed of all pedestrians in an evacuatinggroup
        (align the speed to the speed of the slowest traveling member in the group)
        """
        
        self.startDocument()
        
        # limit speed per cell to min desired speed seen in that cell
        group_min_speed = {}
        for ped in self.pedAgentList.values():
            if ped.terminated:
                continue
            gid = getattr(ped, "group_id", None)
            if gid is None:
                continue
            
            s = float(ped.currSpeed)
            if gid in group_min_speed: 
                if s < group_min_speed[gid]:
                    group_min_speed[gid] = s
            else:
                group_min_speed[gid] = s
                
        for ped in self.pedAgentList.values():
            if ped.terminated:
                continue
            gid = getattr(ped, "group_id", None)
            if gid is None:
                continue
            if gid in group_min_speed:
                ped.currSpeed = float(max(self._speed_floor_value(), group_min_speed[gid]))
        
    
    def pedestrianHazardInteraction(self):
        """
        Apply the paper's social-force free-speed update, persistent panic
        onset, and danger-level casualty process before network movement.
        """
        self.startDocument()
        if self.cellTracker is None:
            return 0
        if self.forceTracker is None:
            raise RuntimeError("The active pedestrian model requires a ForceProcessor")
        if getattr(self.forceTracker, "cellTracker", None) is not self.cellTracker:
            self.forceTracker.setupCellTracker(self.cellTracker)

        self.updatePanicStates()
        for ped in list(self.pedAgentList.values()):
            if ped.terminated:
                continue
            
            cellState = int(self.cellTracker.getCellState(ped.currCell))
            _, casualtyProb, exposed = self._hazard_effects(ped.currCell, cellState)

            previous_force_speed = float(
                getattr(ped, "social_force_speed", self.maxSpeed or ped.currSpeed or 0.0)
            )
            speed, self_force, impact_force = self.forceTracker.compute(
                baseSpeed=previous_force_speed,
                cell=ped.currCell,
                min_speed=0.0,
                max_speed=float(self.maxSpeed),
            )
            ped.social_force_speed = float(speed)
            ped.social_self_force = float(self_force)
            ped.social_impact_force = float(impact_force)
            ped.currSpeed = float(speed)
            ped.hazard_adjusted_speed = float(speed)

            group_size = max(1, int(getattr(ped, "group_size", 1)))
            self.socialForcePersonSteps += group_size
            self.socialSelfForcePersonSteps += float(self_force) * group_size
            self.socialImpactForcePersonSteps += float(impact_force) * group_size
            self.socialSpeedRatioPersonSteps += (
                float(speed) / float(max(1e-12, self.maxSpeed)) * group_size
            )

            casualties = self._hazard_casualty_count(
                getattr(ped, "agentID", 0),
                group_size,
                casualtyProb,
            )
            if casualties > 0:
                self.travelDistanceByOutcome["Casualty"] += (
                    float(getattr(ped, "distance_travelled_m", 0.0)) * casualties
                )
                self.travelPopulationByOutcome["Casualty"] += casualties
                self.bump("casualty", casualties)
                if casualties >= group_size:
                    if self.shelterDS is not None:
                        self.shelterDS.releaseReservation(ped)
                    ped.casualty = True
                    ped.terminated = True
                    if ped.agentID in self.pedAgentList:
                        del self.pedAgentList[ped.agentID]
                    continue
                ped.group_size = int(group_size - casualties)

            if exposed and not getattr(ped, "affected", False):
                ped.affected = True
                self.bump('affected', int(getattr(ped, "group_size", 1)))
        if self.shelterDS is not None:
            self.shelterDS.reconcileReservations(self.pedAgentList.values())
        return 0
    def _hazard_effects(self, cell, cell_state):
        """Combine configured active-hazard effects for one cell.

        Each hazard's sampled casualty and speed-reduction parameters are
        conditional maxima at state five. State severity scales them linearly.
        Independent overlapping effects combine through complementary
        probabilities, keeping both outputs in [0, 1] without ad hoc clipping.
        """
        if self.hazardDS is None or cell is None:
            return 0.0, 0.0, False
        cell_key = tuple(cell)
        # Per the documented wellness states, Level 2 starts exposure while
        # only Levels 4--5 are casualty-inducing. Configured casualty rates are
        # cumulative probabilities over ``casualty_reference_exposure_minutes``
        # at Level 5. Convert that probability to this simulator timestep and
        # severity before combining independent overlapping hazards.
        exposed_level = int(cell_state) >= 2
        casualty_severity = max(0.0, min(1.0, (float(cell_state) - 3.0) / 2.0))
        speed_survival = 1.0
        casualty_survival = 1.0
        exposed = False
        for hazard in self.hazardDS.hazardList.values():
            if not bool(getattr(hazard, "active", False)):
                continue
            impacted = {
                tuple(value) if isinstance(value, list) else value
                for value in getattr(hazard, "impactedCells", [])
            }
            if cell_key not in impacted:
                continue
            exposed = exposed or exposed_level
            speed_rate = max(0.0, min(1.0, float(getattr(hazard, "speedReduct", 0.0))))
            casualty_rate = max(0.0, min(1.0, float(getattr(hazard, "casualtyRate", 0.0))))
            speed_survival *= 1.0 - speed_rate * max(0.0, float(cell_state) / 5.0)
            exposure_fraction = (
                float(self.timeStepMinutes)
                * casualty_severity
                / float(self.casualty_reference_exposure_minutes)
            )
            if casualty_rate >= 1.0:
                step_casualty_probability = 1.0 if exposure_fraction > 0.0 else 0.0
            elif casualty_rate <= 0.0 or exposure_fraction <= 0.0:
                step_casualty_probability = 0.0
            else:
                step_casualty_probability = -math.expm1(
                    math.log1p(-casualty_rate) * exposure_fraction
                )
            casualty_survival *= 1.0 - step_casualty_probability
        return (
            1.0 - speed_survival,
            1.0 - casualty_survival,
            exposed,
        )
    
    def pedestrianNetworkInteraction(self):
        """
        Move each pedestrian along their route for one timestep.
        - If they reach their destination and are not 'affected', they 'Arrive'
        - If they reach destination AND are affected, they 'Evacuated'
        - If they hit a shelter, terminate as 'Evacuated'
        - Resolve shelter admission and continue unresolved movement.
        """
        self.startDocument()
        if self.mapDS is None or self.cellTracker is None:
            return 0

        active_at_start = [
            ped
            for ped in self.pedAgentList.values()
            if not getattr(ped, "terminated", False)
        ]

        # Preserve the hazard/group-adjusted free speed for all internal
        # congestion substeps. ``currSpeed`` becomes the realized whole-minute
        # movement rate only after integration is complete.
        distance_by_agent = {id(ped): 0.0 for ped in active_at_start}
        minimum_ratio_by_agent = {id(ped): 1.0 for ped in active_at_start}
        for ped in active_at_start:
            ped.hazard_adjusted_speed = max(0.0, float(ped.currSpeed))

        congestion_enabled = bool(
            self.congestionModel is not None
            and getattr(self.congestionModel, "enabled", False)
        )
        target_substep_seconds = (
            float(self.congestionModel.integration_substep_seconds)
            if congestion_enabled
            else float(self.timeStepMinutes) * 60.0
        )
        substeps = max(
            1,
            int(math.ceil(float(self.timeStepMinutes) * 60.0 / target_substep_seconds)),
        )
        substep_minutes = float(self.timeStepMinutes) / float(substeps)

        person_minutes = 0.0
        ratio_person_minutes = 0.0
        congested_person_minutes = 0.0
        assigned_person_minutes = 0.0
        effective_speed_person_minutes = 0.0
        minimum_ratio = 1.0
        maximum_density = 0.0
        maximum_occupied_links = 0

        for _substep in range(substeps):
            active_pedestrians = [
                ped
                for ped in self.pedAgentList.values()
                if not getattr(ped, "terminated", False)
            ]
            if not active_pedestrians:
                break

            if self.shelterDS is not None:
                self.shelterDS.reconcileReservations(active_pedestrians)

            # Every substep first updates intended routes, then freezes all
            # physical-link loads before anybody moves. Thus both speeds and
            # downstream link entry respond within the one-minute MDP step,
            # without iteration-order capacity advantages.
            for ped in active_pedestrians:
                self.reroute_if_target_shelter_unavailable(ped)

            frozen_panic_loads = self._travelling_edge_loads(active_pedestrians)
            self._prepare_panic_choices(active_pedestrians, frozen_panic_loads)

            if self.congestionModel is None:
                speed_ratios = {id(ped): 1.0 for ped in active_pedestrians}
                self.lastCongestionLinkStates = tuple()
                substep_population = sum(
                    max(1, int(getattr(ped, "group_size", 1)))
                    for ped in active_pedestrians
                )
                substep_metrics = {
                    "active_population": int(substep_population),
                    "assigned_link_population": 0,
                    "occupied_physical_links": 0,
                    "congested_population": 0,
                    "mean_congestion_speed_ratio": 1.0,
                    "minimum_congestion_speed_ratio": 1.0,
                    "maximum_link_density_ped_per_m2": 0.0,
                }
            else:
                (
                    speed_ratios,
                    self.lastCongestionLinkStates,
                    substep_metrics,
                ) = self.congestionModel.snapshot(
                    active_pedestrians,
                    self.mapDS.edgeListByLocalID.values(),
                )

            substep_population = int(substep_metrics["active_population"])
            duration_weight = float(substep_population) * substep_minutes
            person_minutes += duration_weight
            ratio_person_minutes += (
                float(substep_metrics["mean_congestion_speed_ratio"])
                * duration_weight
            )
            congested_person_minutes += (
                float(substep_metrics["congested_population"])
                * substep_minutes
            )
            assigned_person_minutes += (
                float(substep_metrics["assigned_link_population"])
                * substep_minutes
            )
            minimum_ratio = min(
                minimum_ratio,
                float(substep_metrics["minimum_congestion_speed_ratio"]),
            )
            maximum_density = max(
                maximum_density,
                float(substep_metrics["maximum_link_density_ped_per_m2"]),
            )
            maximum_occupied_links = max(
                maximum_occupied_links,
                int(substep_metrics["occupied_physical_links"]),
            )

            state_by_key = {
                state.physical_key: state
                for state in self.lastCongestionLinkStates
            }
            for ped in active_pedestrians:
                if getattr(ped, "terminated", False):
                    continue
                group_size = max(1, int(getattr(ped, "group_size", 1)))
                hazard_adjusted_speed = float(ped.hazard_adjusted_speed)
                intended_ratio = float(speed_ratios.get(id(ped), 1.0))
                effective_speed_person_minutes += (
                    hazard_adjusted_speed
                    * intended_ratio
                    * group_size
                    * substep_minutes
                )
                experienced_ratio = intended_ratio
                time_left = substep_minutes

                while time_left > 1e-9 and not ped.terminated:
                    if ped.atNode:
                        if (
                            self.shelter_osmid_map
                            and ped.currNode
                            and ped.currNode.OSMID in self.shelter_osmid_map
                        ):
                            may_continue = self.arrive_node(ped, ped.currNode)
                            if ped.terminated or not may_continue:
                                break

                        if (
                            ped.routeFollowing
                            and ped.currNode
                            and ped.currNode.OSMID == ped.routeFollowing.endNode.OSMID
                        ):
                            self.terminatePedestrianAgent(
                                ped,
                                "Evacuated" if ped.affected else "Arrival",
                            )
                            break

                        if not self.advanceFromNode(ped, frozen_panic_loads):
                            # Missing or exhausted routing data is not evidence
                            # of physical arrival. Keep the pedestrian active;
                            # the terminal ledger will classify unresolved
                            # agents as unfinished rather than falsely safe.
                            break

                    if (
                        not getattr(ped, "edge_vec", None)
                        or getattr(ped, "edge_remain", 0.0) <= 0.0
                    ):
                        ped.atNode = True
                        ped.currEdge = None
                        break

                    edge_ratio = 1.0
                    if self.congestionModel is not None and ped.currEdge is not None:
                        key = self.congestionModel.physical_link_key(ped.currEdge)
                        state = state_by_key.get(key)
                        edge_ratio = 1.0 if state is None else float(state.speed_ratio)
                    experienced_ratio = min(experienced_ratio, edge_ratio)
                    link_speed = hazard_adjusted_speed * edge_ratio
                    if link_speed <= 1e-12:
                        break

                    travel = min(link_speed * time_left, ped.edge_remain)
                    ped.lastX += ped.edge_vec[0] * travel
                    ped.lastY += ped.edge_vec[1] * travel
                    ped.currCell = self.cellTracker.locateCell(ped.lastX, ped.lastY)
                    ped.edge_remain -= travel
                    distance_by_agent[id(ped)] += travel
                    ped.distance_travelled_m = float(
                        getattr(ped, "distance_travelled_m", 0.0)
                    ) + float(travel)
                    time_left = max(0.0, time_left - (travel / link_speed))

                    if ped.edge_remain <= 1e-6:
                        dest_node = getattr(ped, "edge_dest_node", None)
                        if dest_node is None:
                            edge = ped.currEdge
                            if edge is not None:
                                if ped.currNode and edge.startNode.OSMID == ped.currNode.OSMID:
                                    dest_node = edge.endNode
                                else:
                                    dest_node = edge.startNode
                        if dest_node is not None:
                            may_continue = self.arrive_node(ped, dest_node)
                            if ped.terminated or not may_continue:
                                break

                minimum_ratio_by_agent[id(ped)] = min(
                    minimum_ratio_by_agent[id(ped)],
                    experienced_ratio,
                )

        for ped in active_at_start:
            ped.congestion_speed_ratio = minimum_ratio_by_agent[id(ped)]
            ped.currSpeed = (
                distance_by_agent[id(ped)] / float(self.timeStepMinutes)
            )

        initial_population = sum(
            max(1, int(getattr(ped, "group_size", 1))) for ped in active_at_start
        )
        mean_ratio = (
            ratio_person_minutes / person_minutes if person_minutes > 0.0 else 1.0
        )
        mean_effective_speed = (
            effective_speed_person_minutes / person_minutes
            if person_minutes > 0.0
            else 0.0
        )
        self.lastCongestionMetrics = {
            "active_population": int(initial_population),
            "assigned_link_population": int(round(
                assigned_person_minutes / float(self.timeStepMinutes)
            )),
            "occupied_physical_links": int(maximum_occupied_links),
            "congested_population": int(round(
                congested_person_minutes / float(self.timeStepMinutes)
            )),
            "mean_congestion_speed_ratio": float(mean_ratio),
            "minimum_congestion_speed_ratio": float(minimum_ratio),
            "maximum_link_density_ped_per_m2": float(maximum_density),
            "mean_effective_speed_m_per_minute": float(mean_effective_speed),
            "congestion_substeps": int(substeps),
        }
        self.congestionPersonMinutes += person_minutes
        self.congestionWeightedRatioSum += ratio_person_minutes
        self.congestedPersonMinutes += congested_person_minutes
        self.maximumObservedLinkDensity = max(
            self.maximumObservedLinkDensity,
            maximum_density,
        )
        self.effectiveSpeedPersonMinuteSum += effective_speed_person_minutes

        return 0

    def congestion_summary(self) -> dict:
        """Return episode-level, person-time-weighted congestion diagnostics."""
        person_minutes = float(self.congestionPersonMinutes)
        return {
            "congestion_model": (
                None
                if self.congestionModel is None
                else self.congestionModel.contract()
            ),
            "congestion_person_minutes": person_minutes,
            "mean_congestion_speed_ratio": (
                float(self.congestionWeightedRatioSum) / person_minutes
                if person_minutes > 0.0
                else 1.0
            ),
            "congested_person_minute_share": (
                float(self.congestedPersonMinutes) / person_minutes
                if person_minutes > 0.0
                else 0.0
            ),
            "maximum_link_density_ped_per_m2": float(
                self.maximumObservedLinkDensity
            ),
            "mean_effective_speed_m_per_minute": (
                float(self.effectiveSpeedPersonMinuteSum) / person_minutes
                if person_minutes > 0.0
                else 0.0
            ),
        }

    def social_force_summary(self) -> dict:
        person_steps = float(self.socialForcePersonSteps)
        denominator = person_steps if person_steps > 0.0 else 1.0
        return {
            "social_force_model": (
                None if self.forceTracker is None else self.forceTracker.contract()
            ),
            "social_force_person_timesteps": person_steps,
            "mean_social_self_force": float(
                self.socialSelfForcePersonSteps / denominator
            ),
            "mean_social_impact_force": float(
                self.socialImpactForcePersonSteps / denominator
            ),
            "mean_social_force_speed_ratio": float(
                self.socialSpeedRatioPersonSteps / denominator
            ),
        }

    def distance_summary(self) -> dict:
        total_distance = float(sum(self.travelDistanceByOutcome.values()))
        total_population = int(sum(self.travelPopulationByOutcome.values()))
        safe_events = ("Arrival", "Evacuated")
        safe_distance = float(
            sum(self.travelDistanceByOutcome.get(event, 0.0) for event in safe_events)
        )
        safe_population = int(
            sum(self.travelPopulationByOutcome.get(event, 0) for event in safe_events)
        )
        return {
            "total_pedestrian_distance_m": total_distance,
            "mean_pedestrian_distance_m": (
                total_distance / total_population if total_population else 0.0
            ),
            "mean_safe_completion_distance_m": (
                safe_distance / safe_population if safe_population else 0.0
            ),
        }

    def panic_summary(self) -> dict:
        eligible = int(self.result.get("panic_eligible_first_exposure", 0))
        onsets = int(self.result.get("panic_onset", 0))
        return {
            "panic_model": self.panic_contract(),
            "panic_eligible_first_exposures": eligible,
            "panic_onsets": onsets,
            "realized_panic_onset_rate": (
                float(onsets) / float(eligible) if eligible > 0 else 0.0
            ),
            "panic_herd_choices": int(self.result.get("panic_herd_choices", 0)),
            "panic_random_choices": int(self.result.get("panic_random_choices", 0)),
            "active_panicked_at_horizon": int(self.active_panicked_count()),
            "casualty_reference_exposure_minutes": float(
                self.casualty_reference_exposure_minutes
            ),
        }
    
