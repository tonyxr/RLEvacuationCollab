#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/5: Constructed the function skeletons
"""

from Shelter import Shelter

class ShelterDS:
    def __init__(self, candidateVol, initVol):
        self.shelterList = {}
        self.shelterByOSMID = {}
        
        self.shelterByCell = None
        self.shelterCanByCell = None
        
        self.shelterImplementedByTime = {}
        
        self.candidateVol = int(candidateVol)
        self.initVol = int(initVol)
        
        self.shelterCanList = {}
        self.nextID = 0

        # Capacity is committed when a pedestrian is routed, not when it
        # reaches the door.  These two indexes form one reservation ledger:
        # one destination per pedestrian and many pedestrians per shelter.
        self.reservationByPedestrian = {}
        self.reservationsByShelter = {}
        
        self.shelterPerCellList = None
        # When positive, every installed shelter consumes one identical
        # capacity token. This separates the location decision from resource
        # quantity and makes RL/heuristic comparisons capacity-fair.
        self.deploymentCapacityToken = None

    def configuredShelterCapacity(self, node) -> int:
        token = self.deploymentCapacityToken
        if token is not None:
            token = int(token)
            if token <= 0:
                raise ValueError("deploymentCapacityToken must be positive")
            return token if self.candidateSupportsConfiguredCapacity(node) else 0
        return int(max(0.0, float(getattr(node, "nodeCap", 100.0))))

    def candidateSupportsConfiguredCapacity(self, node) -> bool:
        """Whether a site's physical rating can supply one capacity token."""
        physical_capacity = int(
            max(0.0, float(getattr(node, "nodeCap", 0.0)))
        )
        token = self.deploymentCapacityToken
        if token is None:
            return physical_capacity > 0
        token = int(token)
        if token <= 0:
            raise ValueError("deploymentCapacityToken must be positive")
        return physical_capacity >= token

    @staticmethod
    def _pedestrian_id(pedAgent):
        if pedAgent is None or not hasattr(pedAgent, "agentID"):
            return None
        return int(pedAgent.agentID)

    @staticmethod
    def _shelter_id(shelter):
        if shelter is None or not hasattr(shelter, "shelterID"):
            return None
        return int(shelter.shelterID)

    def reservedPopulation(self, shelter, exclude_pedestrian=None) -> int:
        """Population committed to ``shelter`` but not yet admitted."""
        shelter_id = self._shelter_id(shelter)
        if shelter_id is None:
            return 0
        excluded = self._pedestrian_id(exclude_pedestrian)
        reservations = self.reservationsByShelter.get(shelter_id, {})
        return int(sum(
            int(population)
            for pedestrian_id, population in reservations.items()
            if pedestrian_id != excluded
        ))

    def availableCapacity(self, shelter, for_pedestrian=None) -> int:
        """Return capacity not occupied or promised to another pedestrian."""
        if shelter is None or int(getattr(shelter, "status", 0)) != 0:
            return 0
        return int(max(
            0,
            int(shelter.shelterCap)
            - int(shelter.shelterFlow)
            - self.reservedPopulation(shelter, exclude_pedestrian=for_pedestrian),
        ))

    def reservationFor(self, pedAgent):
        pedestrian_id = self._pedestrian_id(pedAgent)
        if pedestrian_id is None:
            return None, 0
        reservation = self.reservationByPedestrian.get(pedestrian_id)
        if reservation is None:
            return None, 0
        shelter_id, population = reservation
        return self.shelterList.get(int(shelter_id)), int(population)

    def _release_by_id(self, pedestrian_id) -> int:
        reservation = self.reservationByPedestrian.pop(int(pedestrian_id), None)
        if reservation is None:
            return 0
        shelter_id, population = reservation
        by_shelter = self.reservationsByShelter.get(int(shelter_id), {})
        by_shelter.pop(int(pedestrian_id), None)
        if not by_shelter:
            self.reservationsByShelter.pop(int(shelter_id), None)
        return int(population)

    def releaseReservation(self, pedAgent) -> int:
        pedestrian_id = self._pedestrian_id(pedAgent)
        return 0 if pedestrian_id is None else self._release_by_id(pedestrian_id)

    def reserveShelter(self, pedAgent, shelter, population=None) -> int:
        """Atomically move one pedestrian/cohort's capacity commitment.

        Represented cohorts are indivisible while travelling, so a route is
        committed only when the whole active cohort fits. Existing
        reservations are untouched when the requested destination cannot
        accept that commitment.
        """
        pedestrian_id = self._pedestrian_id(pedAgent)
        shelter_id = self._shelter_id(shelter)
        if pedestrian_id is None or shelter_id is None:
            return 0
        requested = (
            max(0, int(getattr(pedAgent, "group_size", 1)))
            if population is None
            else max(0, int(population))
        )
        requested = min(requested, max(0, int(getattr(pedAgent, "group_size", 1))))
        if requested <= 0 or int(getattr(shelter, "status", 0)) != 0:
            return 0

        assignable = self.availableCapacity(shelter, for_pedestrian=pedAgent)
        if requested > assignable:
            return 0
        assigned = requested

        self._release_by_id(pedestrian_id)
        self.reservationByPedestrian[pedestrian_id] = (shelter_id, assigned)
        self.reservationsByShelter.setdefault(shelter_id, {})[pedestrian_id] = assigned
        self.assertCapacityInvariant()
        return int(assigned)

    def _admit_population(self, requested_population, shelter, pedAgent=None) -> int:
        """Consume a reservation, while protecting every other reservation."""
        requested = max(0, int(requested_population))
        if requested <= 0 or shelter is None or int(getattr(shelter, "status", 0)) != 0:
            return 0
        admissible = self.availableCapacity(shelter, for_pedestrian=pedAgent)
        admitted = min(requested, admissible)
        assigned_shelter, _ = self.reservationFor(pedAgent)
        if assigned_shelter is shelter:
            self.releaseReservation(pedAgent)
        if admitted <= 0:
            return 0
        shelter.updateFlow(admitted)
        if shelter.shelterFlow >= shelter.shelterCap:
            shelter.status = 1
        self.assertCapacityInvariant()
        return int(admitted)

    def reconcileReservations(self, pedestrians) -> int:
        """Release stale promises and clamp cohort promises after casualties."""
        active = {
            int(ped.agentID): ped
            for ped in pedestrians
            if hasattr(ped, "agentID")
        }
        released = 0
        for pedestrian_id, (shelter_id, population) in tuple(
            self.reservationByPedestrian.items()
        ):
            pedestrian = active.get(int(pedestrian_id))
            shelter = self.shelterList.get(int(shelter_id))
            route = getattr(pedestrian, "routeFollowing", None)
            route_target = getattr(route, "endNode", None) if route is not None else None
            target_matches = bool(
                shelter is not None
                and route_target is not None
                and getattr(route_target, "OSMID", None)
                == getattr(getattr(shelter, "nodeMapped", None), "OSMID", None)
            )
            invalid = bool(
                pedestrian is None
                or getattr(pedestrian, "terminated", False)
                or getattr(pedestrian, "panicked", False)
                or shelter is None
                or int(getattr(shelter, "status", 0)) != 0
                or not target_matches
            )
            if invalid:
                released += self._release_by_id(pedestrian_id)
                continue
            retained = min(int(population), max(0, int(pedestrian.group_size)))
            if retained != int(population):
                released += int(population) - retained
                if retained <= 0:
                    self._release_by_id(pedestrian_id)
                else:
                    self.reservationByPedestrian[pedestrian_id] = (
                        int(shelter_id), retained
                    )
                    self.reservationsByShelter[int(shelter_id)][pedestrian_id] = retained
        self.assertCapacityInvariant()
        return int(released)

    def assertCapacityInvariant(self) -> None:
        """Fail immediately if occupied plus promised capacity is inconsistent."""
        for pedestrian_id, (shelter_id, population) in self.reservationByPedestrian.items():
            if int(population) <= 0:
                raise RuntimeError("Shelter reservations must be positive")
            reverse = self.reservationsByShelter.get(int(shelter_id), {})
            if int(reverse.get(int(pedestrian_id), 0)) != int(population):
                raise RuntimeError("Shelter reservation indexes disagree")
        for shelter_id, shelter in self.shelterList.items():
            reservations = self.reservationsByShelter.get(int(shelter_id), {})
            flow = int(shelter.shelterFlow)
            capacity = int(shelter.shelterCap)
            if flow < 0 or flow > capacity:
                raise RuntimeError(
                    f"Shelter {shelter_id} occupancy is outside [0, {capacity}]"
                )
            for pedestrian_id, population in reservations.items():
                if self.reservationByPedestrian.get(int(pedestrian_id)) != (
                    int(shelter_id), int(population)
                ):
                    raise RuntimeError("Shelter reservation indexes disagree")
            committed = flow + sum(
                int(value) for value in reservations.values()
            )
            if committed > capacity:
                raise RuntimeError(
                    f"Shelter {shelter_id} overcommitted: {committed} > "
                    f"{capacity}"
                )
            if int(getattr(shelter, "status", 0)) != 0 and reservations:
                raise RuntimeError(
                    f"Unavailable shelter {shelter_id} retains reservations"
                )
        missing = set(self.reservationsByShelter) - set(self.shelterList)
        if missing:
            raise RuntimeError("Reservation references a missing shelter")
        
    
    def remainingCandidateCount(self) -> int:
        if self.shelterCanByCell is None:
            return 0
        return int(sum(len(cell) for row in self.shelterCanByCell for cell in row))
    
    """Helper functions"""
    def allocID(self):
        sid = self.nextID
        self.nextID += 1
        return sid
    
    @staticmethod
    def ensureGrid(xn, yn):
        return [[[] for _ in range(yn)] for _ in range(xn)]
    
    """Primary functions"""
    """
    Build 2D arrays (candidates + active list). Sort candidates by descending capacity
    """
    def shelterPerCell(self, cellTracker, cellXNum, cellYNum, mapDS = None):
        cellXNum = int(cellXNum)
        cellYNum = int(cellYNum)
        
        self.shelterByCell = self.ensureGrid(cellXNum, cellYNum)
        self.shelterCanByCell = self.ensureGrid(cellXNum, cellYNum)
        
        for node in self.shelterCanList.values():
            if not self.candidateSupportsConfiguredCapacity(node):
                continue
            ci, cj = cellTracker.locateCell(node.nodeX, node.nodeY)
            if 0 <= ci < cellXNum and 0 <= cj < cellYNum:
                self.shelterCanByCell[ci][cj].append(node)
        
        for i in range(cellXNum):
            for j in range(cellYNum):
                self.shelterCanByCell[i][j].sort(key = lambda n: float(getattr(n, "nodeCap", 0.0)), reverse = True)
        # Step 3: Sort each sublist by descending shelter capacity
        
        # Step 4: Optionally rebalance candidates across cells (round-robin by cell)
        # so the RL can deploy shelters throughout the network instead of being
        # concentrated in one dense cell.
        target = int(self.candidateVol) if int(self.candidateVol) > 0 else None
        if target is not None:
            balanced = self.ensureGrid(cellXNum, cellYNum)
            filled = 0
            while filled < target:
                any_picked = False
                for i in range(cellXNum):
                    for j in range(cellYNum):
                        if filled >= target:
                            break
                        if self.shelterCanByCell[i][j]:
                            balanced[i][j].append(self.shelterCanByCell[i][j].pop(0))
                            filled += 1
                            any_picked = True
                if not any_picked:
                    break
            self.shelterCanByCell = balanced
            print(f"Shelter candidates sampled for deployment: {filled} (from detected pool)")
        
        self.shelterPerCellList = self.shelterCanByCell
        
    """
    Add one person to shelter if capacity remains
    Returns: 
        0 = admitted
        1 = full or unavailable
    """
    def updateShelterFlow(self, pedAgent, shelter):
        if shelter is None:
            return 1
        
        if getattr(shelter, "status", 0) != 0:
            return 1
        
        group_size = max(1, int(getattr(pedAgent, "group_size", 1)))
        remaining_capacity = self.availableCapacity(
            shelter, for_pedestrian=pedAgent
        )
        if remaining_capacity <= 0:
            return 1
        if group_size > remaining_capacity:
            # Admission is atomic for a represented pedestrian group. Do not
            # close the shelter while it can still accept a smaller group.
            return 1

        admitted = self._admit_population(group_size, shelter, pedAgent)
        if admitted != group_size:
            raise RuntimeError("Atomic shelter admission changed during commit")
        return 0

    def admitShelterPopulation(self, requested_population, shelter, pedAgent=None):
        """Admit as much of a represented cohort as capacity permits.

        This method is used only by the explicitly configured weighted-cohort
        simulator. ``updateShelterFlow`` retains atomic single-agent/group
        semantics for existing callers.
        """
        return self._admit_population(
            requested_population, shelter, pedAgent=pedAgent
        )
    
    """
    Euclidean nearest shelter among active ones
    Return closest shelter or none
    """
    def locateClosestShelter(self, x, y, openOnly = True):
        best = None
        best_d2 = float("inf")
        
        for shelter in self.shelterList.values():
            if openOnly and self.availableCapacity(shelter) <= 0:
                continue
            nx, ny = shelter.nodeMapped.nodeX, shelter.nodeMapped.nodeY
            d2 = (nx - x) ** 2 + (ny - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = shelter
        
        return best
        # Iteratively compare the Euclidean distance between the given (X, Y) and each open shelter's X and Y in the shelter's list (cross-cell evacuation is very possible here), 
        # return the ideal shelter entity. 
            
    """
    Simple status update:
        - If a shelter reaches capacity => status = 1 
        - (optiona, may develop later) extend with hazard impact to set status = 2
    """
    def shelterStatusUpdate(self):
        for shelter in self.shelterList.values():
            if shelter.shelterFlow >= shelter.shelterCap and getattr(shelter, "status", 0) == 0:
                shelter.status = 1
                
        # Step 1: (if statement): if scenario 1, change shelter status to full (status = 1).
        
        # Step 2: (if statement): if scenario 2, change shelter status to damaged (status = 2). 
        
        # Step 3: (Optional) If a shelter becomes unavailable to new pedestrian evacuees, 
        # it is removed from the available list and added to the closed list (so guidance knows which shelter to direct pedestrians away from).
    
    def newShelter(self, action, cellTracker):
        """
            Deploy a new shelter in a chosen cell.
            action should be {"cell": (ci, cj)}
    
            Returns:
                shelterID if deployed
                None if no valid candidate or out of range
        """
        if self.shelterCanByCell is None or self.shelterByCell is None:
            return None
        
        if not isinstance(action, dict) or "cell" not in action:
            return None
        
        # bound check 
        ci, cj = action["cell"]
        
        XNum = len(self.shelterCanByCell)
        YNum = len(self.shelterCanByCell[0]) if XNum > 0 else 0
        if not (0 <= ci < XNum and 0 <= cj < YNum):
            return None
        
        # check if this cell has any uninitialized candidate left
        cell_candidates = self.shelterCanByCell[ci][cj]
        if not cell_candidates:
            return None
        
        best_idx = self._candidate_index((ci, cj), cellTracker)
        if best_idx is None:
            return None

        node = cell_candidates.pop(best_idx)
        
        sid = self.allocID()
        cap = self.configuredShelterCapacity(node)
        flow = 0
        status = 0
        
        sh = Shelter(sid, node, (ci, cj), cap, flow, status)
        
        self.shelterList[sid] = sh
        self.shelterByOSMID[node.OSMID] = sh
        
        self.shelterByCell[ci][cj].append(sh)
            
        return sid
        # Step 1: Receive the action package from the RL actor.
            
        # Step 2: Locate the right shelter sub-list by cell's X and Y ID, extract and remove the first shelter candidate from the list.
        
        # Step 3: Declare a new shelter entity and add to the active shelters list.
        
        # Step 4: Communicate with guidance on potentially changing the shelter pointer to the newly established shelter.

    def newShelterCandidate(self, candidate_osm_id, cell, cellTracker):
        """Deploy the exact candidate selected by the administrator-facing action.

        Candidate identities are stable for the episode.  The cell is supplied
        as an integrity check, not as a second optimization stage.
        """
        del cellTracker
        if self.shelterCanByCell is None or self.shelterByCell is None:
            return None
        ci, cj = int(cell[0]), int(cell[1])
        if not (0 <= ci < len(self.shelterCanByCell)):
            return None
        if not (0 <= cj < len(self.shelterCanByCell[ci])):
            return None
        requested = str(candidate_osm_id)
        candidates = self.shelterCanByCell[ci][cj]
        matching = [
            index
            for index, node in enumerate(candidates)
            if str(getattr(node, "OSMID", "")) == requested
        ]
        if len(matching) != 1:
            return None
        node = candidates[matching[0]]
        if not self.candidateSupportsConfiguredCapacity(node):
            return None
        node = candidates.pop(matching[0])
        capacity = self.configuredShelterCapacity(node)
        if capacity <= 0:
            return None

        shelter_id = self.allocID()
        shelter = Shelter(shelter_id, node, (ci, cj), capacity, 0, 0)
        self.shelterList[shelter_id] = shelter
        self.shelterByOSMID[node.OSMID] = shelter
        self.shelterByCell[ci][cj].append(shelter)
        return shelter_id

    def _candidate_index(self, cell, cellTracker):
        """Solve the common lower-level problem without mutating candidates.

        Once an upper-level policy selects a region, the implementation layer
        installs the maximum-capacity feasible site in that region. This exact,
        deterministic rule is used for RL and every benchmark; OSM identifier
        and original order provide reproducible tie breaks.
        """
        del cellTracker
        if self.shelterCanByCell is None:
            return None
        ci, cj = int(cell[0]), int(cell[1])
        if not (0 <= ci < len(self.shelterCanByCell)):
            return None
        if not self.shelterCanByCell or not (0 <= cj < len(self.shelterCanByCell[ci])):
            return None
        candidates = self.shelterCanByCell[ci][cj]
        if not candidates:
            return None
        eligible = [
            index
            for index, candidate in enumerate(candidates)
            if self.candidateSupportsConfiguredCapacity(candidate)
        ]
        if not eligible:
            return None
        return int(
            min(
                eligible,
                key=lambda index: (
                    -max(0.0, float(getattr(candidates[index], "nodeCap", 0.0))),
                    str(getattr(candidates[index], "OSMID", "")),
                    index,
                ),
            )
        )

    def previewShelterCandidate(self, cell, cellTracker):
        """Expose the exact candidate the shared lower layer would install."""
        best_idx = self._candidate_index(cell, cellTracker)
        if best_idx is None:
            return None
        ci, cj = int(cell[0]), int(cell[1])
        return self.shelterCanByCell[ci][cj][best_idx]

    def initShelter(self):
        """
            Deploy up to self.initVol initial shelters,
            scanning the grid in a round-robin pattern (like initGuidance does).
        """
        
        if self.shelterCanByCell is None or self.shelterByCell is None:
            return None
            
        XNum = len(self.shelterCanByCell)
        YNum = len(self.shelterCanByCell[0]) if XNum > 0 else 0
        if XNum == 0 or YNum == 0:
            return None

        created = 0
        i = j = 0
        
        while created < self.initVol and XNum > 0 and YNum > 0:
            found = False
            for _ in range(XNum * YNum):
                best_idx = self._candidate_index((i, j), None)
                if best_idx is not None:
                    node = self.shelterCanByCell[i][j].pop(best_idx)
                    sid = self.allocID()
                    cap = self.configuredShelterCapacity(node)
                    sh = Shelter(sid, node, (i, j), cap, 0, 0)
                    self.shelterList[sid] = sh
                    self.shelterByCell[i][j].append(sh)
                    self.shelterByOSMID[node.OSMID] = sh 
                    
                    created += 1
                    found = True
                    break
                j += 1
                if j >= YNum:
                    j = 0
                    i += 1
                    if i >= XNum:
                        i = 0
            
            # if no more candidate in the cell
            if not found: 
                break
            
        return created

    def predeployStaticDemandGreedy(self, demand_by_cell, additions, cellTracker):
        """Predeploy a capacity-matched static set using only initial demand.

        The rule is deliberately non-anticipative: it receives an expected
        population mass for each regional cell, but no realized hazard path or
        future pedestrian state.  At each greedy step it installs the common
        lower-layer maximum-capacity candidate in the cell with the largest
        remaining demand that the candidate can cover.  Capacity and stable
        identifiers break ties.  Dynamic policies continue to use
        :meth:`newShelter`, so site feasibility and implementation semantics
        remain shared.

        Returns the installed shelter identifiers in decision order.
        """
        if self.shelterCanByCell is None or self.shelterByCell is None:
            return []
        x_count = len(self.shelterCanByCell)
        y_count = len(self.shelterCanByCell[0]) if x_count else 0
        expected = list(demand_by_cell)
        if len(expected) != x_count * y_count:
            raise ValueError(
                "demand_by_cell must contain one expected-demand value per cell"
            )
        remaining = [max(0.0, float(value)) for value in expected]
        installed = []
        for _ in range(max(0, int(additions))):
            choices = []
            for i in range(x_count):
                for j in range(y_count):
                    candidate = self.previewShelterCandidate((i, j), cellTracker)
                    if candidate is None:
                        continue
                    index = i * y_count + j
                    capacity = float(self.configuredShelterCapacity(candidate))
                    marginal_coverage = min(capacity, remaining[index])
                    choices.append(
                        (
                            -marginal_coverage,
                            -remaining[index],
                            -capacity,
                            index,
                            str(getattr(candidate, "OSMID", "")),
                            (i, j),
                        )
                    )
            if not choices:
                break
            selected = min(choices)[-1]
            shelter_id = self.newShelter({"cell": selected}, cellTracker)
            if shelter_id is None:
                raise RuntimeError(
                    "Static demand-greedy preview was feasible but installation failed"
                )
            shelter = self.shelterList[shelter_id]
            index = int(selected[0]) * y_count + int(selected[1])
            remaining[index] = max(
                0.0,
                remaining[index] - max(0.0, float(getattr(shelter, "shelterCap", 0.0))),
            )
            installed.append(int(shelter_id))
        return installed
