#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/4: Constructed the skeletons
"""

import numpy as np
import math
from Pedestrian import Pedestrian
from collections import defaultdict

class PedDS: 
    def __init__(self, pedNum):
        # total number of pedestrians in the environment
        self.pedNum = int(pedNum)
        # List of all pedestrian agent objects
        self.pedAgentList = {}
        """For stat documentation and analysis"""
        """For definition of arrivals, casualties, evacuation, guidance, affected --> see original CASE 2024 paper"""
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
        self.guidanceDS = None
        self.forceTracker = None
        
        """Other key parameters"""
        self.currTime = 0
        self.maxSpeed = None
        self.minOperationalSpeed = 1.0

        # allow to wait until X timesteps after simulation starts to document results 
        # (after the system is more stablized)
        self.docuStart = False
        # temporal container of event statistics at the current timestep
        self._step = dict(arrival=0, casualty=0, evacuated=0, affected=0)
        self.result = dict(arrival = 0, casualty = 0, evacuated = 0, affected = 0)
        self.evacuationTimeSum = 0.0
        self.evacuationCount = 0
        
        # default impact rate by cell state
        self._default_casualty_prob = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.005, 4: 0.01, 5: 0.02}
        self._default_speed_reduct  = {0: 0.0, 1: 0.025,  2: 0.05, 3: 0.10, 4: 0.12, 5: 0.25}
        
        self.shelter_osmid_map = None
        
        self.groups = defaultdict(set)
        
        
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
        if int(getattr(start_node, "OSMID", -1)) == int(getattr(end_node, "OSMID", -2)):
            return 0.0
        try:
            route = self.mapDS.shortestPath(start_node, end_node)
        except Exception:
            route = None
        if route is None:
            return float("inf")
        dist = 0.0
        for edge in (getattr(route, "edgeRemained", None) or []):
            dist += float(max(0.0, getattr(edge, "edgeLen", 0.0)))
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

    @staticmethod
    def _shelter_has_remaining_capacity(shelter):
        if shelter is None:
            return False
        if int(getattr(shelter, "status", 0)) != 0:
            return False
        cap = float(getattr(shelter, "shelterCap", 0.0))
        flow = float(getattr(shelter, "shelterFlow", 0.0))
        return (cap - flow) > 0.0

    def _reroute_pedestrian_to_shelter(self, ped, target_shelter):
        if ped is None or target_shelter is None or self.mapDS is None:
            return False
        target_node = getattr(target_shelter, "nodeMapped", None)
        if target_node is None:
            return False

        anchor = self._route_anchor_node(ped)
        if anchor is None:
            return False

        try:
            newRoute = self.mapDS.shortestPath(anchor, target_node)
        except Exception:
            newRoute = None

        if newRoute is None:
            return False

        ped.routeFollowing = newRoute
        return True

    def _closest_open_shelter_from_ped(self, ped, exclude_osmid = None):
        if ped is None or self.shelterDS is None:
            return None

        best = None
        best_d2 = float("inf")
        px, py = float(getattr(ped, "lastX", 0.0)), float(getattr(ped, "lastY", 0.0))

        for shelter in self.shelterDS.shelterList.values():
            sh_node = getattr(shelter, "nodeMapped", None)
            if sh_node is None:
                continue
            sh_osmid = getattr(sh_node, "OSMID", None)
            if exclude_osmid is not None and sh_osmid == exclude_osmid:
                continue
            if not self._shelter_has_remaining_capacity(shelter):
                continue

            dx = float(sh_node.nodeX) - px
            dy = float(sh_node.nodeY) - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = shelter

        return best

    def reroute_to_new_shelter_if_closer(self, newShelter):
        """
        After a new shelter is installed, replan active pedestrians whose current
        intended shelter is farther from their current location than this new shelter.
        """
        if newShelter is None or self.mapDS is None or self.shelterDS is None:
            return 0
        
        if not self._shelter_has_remaining_capacity(newShelter):
            return 0

        rerouted = 0
        shelter_by_osmid = getattr(self.shelterDS, "shelterByOSMID", {}) or {}
        new_node = getattr(newShelter, "nodeMapped", None)
        if new_node is None:
            return 0
        for ped in list(self.pedAgentList.values()):
            if getattr(ped, "terminated", False):
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

            if new_dist + 1e-6 < old_dist and self._reroute_pedestrian_to_shelter(ped, newShelter):
                rerouted += 1

        return rerouted
    
    def remaining_active_count(self):
        return int(sum(0 if getattr(p, "terminated", False) else 1 for p in self.pedAgentList.values()))

    def finalize_remaining_pedestrians(self, event: str = "Arrival") -> int:
        active = [p for p in list(self.pedAgentList.values()) if not getattr(p, "terminated", False)]
        for ped in active:
            self.terminatePedestrianAgent(ped, event)
        return int(len(active))

    def reroute_if_target_shelter_unavailable(self, ped):
        """
        If ped's current target shelter has no remaining capacity, reroute to the
        closest shelter (from current location) that still has positive remaining
        capacity.
        """
        if ped is None or self.shelterDS is None:
            return False
        
        route = getattr(ped, "routeFollowing", None)
        target = getattr(route, "endNode", None) if route is not None else None
        if target is None:
            return False

        target_sh = self.shelterDS.shelterByOSMID.get(getattr(target, "OSMID", None))
        if target_sh is None:
            return False
        if self._shelter_has_remaining_capacity(target_sh):
            return False

        next_sh = self._closest_open_shelter_from_ped(
            ped,
            exclude_osmid = getattr(target_sh.nodeMapped, "OSMID", None)
        )
        if next_sh is None:
            return False

        return self._reroute_pedestrian_to_shelter(ped, next_sh)
    
    """Competency check for other related processors"""
    def checkReady(self, mapDS = None, cellTracker = None, maxSpeed = None, 
                   hazardDS = None, shelterDS = None, guidanceDS = None, forceTracker = None):
        
        if mapDS is not None: self.mapDS = mapDS
        if cellTracker is not None: self.cellTracker = cellTracker
        if maxSpeed is not None: self.maxSpeed = float(maxSpeed)
        self.minOperationalSpeed = float(max(0.5, 0.05 * float(self.maxSpeed or 0.0)))
        if hazardDS is not None: self.hazardDS = hazardDS
        if shelterDS is not None:
            self.shelterDS = shelterDS
            self._shelter_osmid_map = {sh.nodeMapped.OSMID: sh for sh in self.shelterDS.shelterList.values()}
        if guidanceDS is not None:
            self.guidanceDS = guidanceDS
        if forceTracker is not None: self.forceTracker = forceTracker
    
    """This function is called at the start of the interaction process at each timestep to document all interaction events at this timestep"""
    def startDocument(self):
        if not self.docuStart:
            self._step = dict(arrival=0, casualty=0, evacuated=0, affected=0)
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
    
    """Load in the guidance and shelter lookup table from their respective databases"""
    def loadGuShLookup(self, guByOSMID, shByOSMID):
        self.shelter_osmid_map = shByOSMID
    
    """This function handles pedestrian agent's interaction with guidance points and shelters"""
    def arrive_node(self, ped, node):
        ped.currNode = node
        ped.atNode = True
        ped.currEdge = None
        ped.edge_dest_mode = None
        
        ped.lastX, ped.lastY = node.nodeX, node.nodeY
        ped.currCell = self.cellTracker.locateCell(ped.lastX, ped.lastY)
        
        osmid = node.OSMID
        
        # shelter arrival
        if self.shelter_osmid_map and osmid in self.shelter_osmid_map and self.shelterDS:
            sh = self.shelter_osmid_map[osmid]
            status = self.shelterDS.updateShelterFlow(ped, sh)
            if status == 0:
                self.terminatePedestrianAgent(ped, "Evacuated")
            else:
                self.reroute_if_target_shelter_unavailable(ped)
    
    def advanceFromNode(self, ped):
        """
        When the pedestrian agent is at a node, choose the next edge the pedestrian should follow from its routeFollowing
        Start walking the agent in this timestep on that edge (emit from first element in routeFollowing's edgeRemain list)
        """
        
        if ped.routeFollowing is None:
            return False
        
        try:
            e = ped.routeFollowing.getNextEdge()
        except Exception:
            # no more edge
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
        Initialize a pedestrian agent with:
            - Birth node
            - Destination node
            - Initial travel route
            - Base speed
            - Set of initial status variable values
        """
        
        self.mapDS = mapDS
        self.cellTracker = cellTracker
        self.maxSpeed = float(maxSpeed)
        
        pedID = 0
        group_sizes = self._sample_group_sizes(self.pedNum, min_size = 1, max_size = 50)
        
        for gsize in group_sizes:
            
            # Step 1: assign the new pedestrian agent ID, ID pointer + 1 for each iteration so agents have different IDs.
            agentID = pedID
            pedID += 1
            #print("Current agent id, ", str(pedID))
            
            # Step 2: Call MapDatabase's AssignGenerationNode and AssignTermination node to get $p_i$'s start and end node. 
            # Assign lastX and lastY with the assigned startNode's X and Y. 
            birthNode = mapDS.assignGenerationNode()
            destNode = mapDS.assignTerminationNode(birthNode)
        
            # Step 3: Call ShortestPathFinder to get the route, and assign to the parameter accordingly. 
            assignedRoute = mapDS.shortestPath(birthNode, destNode)
            # Step 4: In the extreme case, there exists no route between the assigned pair of start and destination node
            # re-assign destination node and plan a route again.
            if assignedRoute is None:
                birthNode = mapDS.assignGenerationNode()
                destNode = mapDS.assignTerminationNode(birthNode)
                assignedRoute = mapDS.shortestPath(birthNode, destNode)
            
            currNode = assignedRoute.startNode
            
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
            
            newAgent = Pedestrian(agentID, assignedRoute, currNode, currEdge, currCell, lastX, 
                                  lastY, currSpeed, affected, atNode, casualty, evacuated, arrival, terminated)
            
            newAgent.group_id = int(birthNode.OSMID)
            newAgent.group_size = int(max(1, gsize))
            self.groups[newAgent.group_id].add(agentID)
            
            newAgent.edge_remain = 0.0 # at node, edge_remain = 0
            newAgent.edge_vec = None
            newAgent.edge_dest_node = None    
            newAgent.desired_speed = float(self.maxSpeed)
            
            self.pedAgentList[agentID] = newAgent
            
            """
            if self.shelterDS is not None:
                self._shelter_osmid_map = {sh.nodeMapped.OSMID: sh for sh in self.shelterDS.shelterList.values()}
            if self.guidanceDS is not None:
                self._guidance_osmid_map = {gu.nodeMapped.OSMID: gu for gu in self.guidanceDS.guidanceList.values()}
            """
            
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
        
        # Step 1: (if statement) Check if casualty, if so, casualty count += 1. Set p_i's casualty and terminated status = True.
        if event == "Casualty":
            ped.casualty = True
            self.bump('casualty', int(getattr(ped, "group_size", 1)))
        # Step 2: (if statement) Check if evacuated, if so, evacuated count += 1. Set p_i's evacuated and terminated status = True. Call UpdateShelterFlow to update the flow. 
        elif event == "Evacuated":
            ped.evacuated = True
            gsize = int(getattr(ped, "group_size", 1))
            self.bump('evacuated', gsize)
            self.evacuationTimeSum += float(self.currTime)
            self.evacuationCount += gsize
        # Step 3: (else) arrival count += 1 and set $p_i$'s arrival and terminated status = True.
        else:
            ped.arrival = True
            self.bump("arrival", int(getattr(ped, "group_size", 1)))
        # Step 4: Remove $p_i$ from the active pedestrian agent list. 
        ped.terminated = True
        if ped.agentID in self.pedAgentList:
            del self.pedAgentList[ped.agentID]
    
    def mean_evacuation_time(self) -> float:
        if int(self.evacuationCount) <= 0:
            return 0.0
        return float(self.evacuationTimeSum) / float(self.evacuationCount)
    
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
        Note: all pedestrian agents in a given cell are subject to same scale of hazardous impact
        Review hazard effects per cell
        For each ped in a particular cell, assign effects of:
            - reduced speed (force due to smoke & heat)
            - possible casualty
            - mark as "affected" (if previously not)
        """
        self.startDocument()
        if self.cellTracker is None:
            return 0
        
        for ped in list(self.pedAgentList.values()):
            if ped.terminated:
                continue
            
            cellState = int(self.cellTracker.getCellState(ped.currCell))
            
            # speed reduction
            reduct = float(self._default_speed_reduct.get(cellState, 0.0))
            reduct = max(0.0, min(0.95, reduct))
            base_speed = float(getattr(ped, "desired_speed", self.maxSpeed or ped.currSpeed or 0.0))
            ped.currSpeed = max(self._speed_floor_value(), base_speed * (1.0 - reduct))
            
            casualtyProb = float(self._default_casualty_prob.get(cellState, 0.0))
            casualtyProb = max(0.0, min(1.0, casualtyProb))
            if cellState > 3 and casualtyProb > 0.0 and np.random.random() < casualtyProb: 
                self.terminatePedestrianAgent(ped, "Casualty")
                continue
                
            if cellState > 2 and not getattr(ped, "affected", False):
                ped.affected = True
                self.bump('affected', int(getattr(ped, "group_size", 1)))
        # Step 1: Based on each pedestrian's location, deem which emergency instance is impacting them (not needed if we do a unified casualty rate).
        
        # Step 2: Based on the casualty rate, sample by random number on if the pedestrian is to become a casualty. If so, call TerminatePedestrianAgent for pedestrian status change. 
        
        return 0
    
    def pedestrianNetworkInteraction(self):
        """
        Move each pedestrian along their route for one timestep.
        - If they reach their destination and are not 'affected', they 'Arrive'
        - If they reach destination AND are affected, they 'Evacuated'
        - If they hit a shelter, terminate as 'Evacuated'
        - If they hit guidance, reroute.
        """
        self.startDocument()
        if self.mapDS is None or self.cellTracker is None:
            return 0
        
        for ped in list(self.pedAgentList.values()):
            if getattr(ped, "terminated", False):
                continue
            
            # keep intended shelter feasible as capacities evolve over time
            self.reroute_if_target_shelter_unavailable(ped)
            
            step_left = max(self._speed_floor_value(), float(ped.currSpeed))
            
            while step_left > 1e-6 and not ped.terminated:
                if ped.atNode: 
                    if ped.routeFollowing and ped.currNode and ped.currNode.OSMID == ped.routeFollowing.endNode.OSMID:
                        if not ped.affected:
                            self.terminatePedestrianAgent(ped, "Arrival")
                            break
                        else:
                            self.terminatePedestrianAgent(ped, "Evacuated")
                            break
                    
                    if not self.advanceFromNode(ped):
                        if not ped.affected: 
                            self.terminatePedestrianAgent(ped, "Arrival")
                            break
                        else:
                            self.terminatePedestrianAgent(ped, "Evacuated")
                            break
                        
                if not getattr(ped, "edge_vec", None) or getattr(ped, "edge_remain", 0.0) <= 0.0:
                    ped.atNode = True
                    ped.currEdge = None
                    break
                    
                travel = min(step_left, ped.edge_remain)
                ped.lastX += ped.edge_vec[0] * travel
                ped.lastY += ped.edge_vec[1] * travel
                ped.currCell = self.cellTracker.locateCell(ped.lastX, ped.lastY)
                
                ped.edge_remain -= travel
                step_left -= travel
                
                if ped.edge_remain <= 1e-6:
                    dest_node = getattr(ped, "edge_dest_node", None)
                    if dest_node is None:
                        e = ped.currEdge
                        if e is not None:
                            if ped.currNode and e.startNode.OSMID == ped.currNode.OSMID:
                                dest_node = e.endNode
                            else:
                                dest_node = e.startNode
                                
                    if dest_node is not None:
                        self.arrive_node(ped, dest_node)
                            
        # Step 1: Extract the per-timestep speed of the pedestrian, which is equal to the distance the pedestrian (group) will travel.
        
        # Step 2.1: (If statement) If the pedestrian was traveling on an edge with remaining distance less or equal to current speed, 
        # the pedestrian is deemed to arrive at the next node (including shelter and guidance point case). 
        # Set currNode = nextNode, currEdge = None. Update the pedestrian's lastX and lastY to currNode's coordinates. 
        
        # Step 2.2: (Nested If statement): If the pedestrian (group) arrives at the intended destination node while not evacuating, 
        # call TerminatePedestrianAgent(p_i, "Arrival") to update the pedestrian agent's status as arrival.
        
        # Step 2.3: (Further Nested If statement): If the pedestrian arrives at the shelter, 
        # call TerminatePedestrianAgent(p_i, "Evacuated") for the pedestrian agent's status change, and call UpdateShelterFlow to update the shelter flow.
        
        # Step 2.4: (Nested If statement): If the pedestrian (group) arrives at a guidance point, 
        # call UpdateGuidePointFlow(p_i, gu_i) to update flow and receive the route to the desired shelter.
        
        # Step 3: (Else If Statement): Else if the pedestrian is currently positioned at a node and ready to travel to the next edge, set currNode = None, currEdge = nextEdge. 
        
        # Step 4: If the pedestrian agent $p_i$ will be traveling on an edge for this timestep, update the pedestrian agent's lastX and lastY with Euclidean distance.
        
        # Step 5: Update location
        
        return 0
    
