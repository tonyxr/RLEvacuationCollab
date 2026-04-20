#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/5: Constructed the function skeletons
"""

from Shelter import Shelter
import math

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
        
        self.shelterPerCellList = None
        
    
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
        
        if shelter.shelterFlow >= shelter.shelterCap:
            shelter.status = 1
            return 1
        
        shelter.updateFlow(1)
        # Step 1: Receive the pedestrian agent and shelter it arrives at
        
        # Step 2: (If statement) Conditionally check if the shelter has any capacity left. 
        # If there is no capacity or it is severely impacted, return status code = 1. 
        # So the PedestrianDatabase will call LocateClosestShelter and MapDatabase to find the nearest available shelter and plan a new route to that shelter. 
        # If the shelter still has status as open, change to unavailable. 
        
        # Step 3: Otherwise, add the new flow to the shelter's flow and return status code = 0. 
        # (If it returns 0, the pedestrian agent's status is changed to evacuated and added to the evacuated list.) 
        return 0
    
    """
    Euclidean nearest shelter among active ones
    Return closest shelter or none
    """
    def locateClosestShelter(self, x, y, openOnly = True):
        best = None
        best_d2 = float("inf")
        
        for shelter in self.shelterList.values():
            if openOnly and getattr(shelter, "status", 0) != 0:
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
        
        # Lower-layer heuristic scoring inside selected cell:
        # C^SH = W1 * APD - W2 * NDS
        # We use a local, observable proxy where:
        #   APD proxy: nearby pedestrian demand weighted by inverse distance.
        #   NDS: nearby danger weighted by inverse distance.
        # The candidate with highest score is installed first.
        best_idx = 0
        best_score = float("-inf")
        for idx, candidate in enumerate(cell_candidates):
            score = self._candidate_install_score(candidate, cellTracker)
            if score > best_score:
                best_score = score
                best_idx = idx

        node = cell_candidates.pop(best_idx)
        
        sid = self.allocID()
        cap = int(max(0.0, float(getattr(node, "nodeCap", 100.0))))
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
        
    def _candidate_install_score(self, node, cellTracker, w_apd = 1.0, w_nds = 1.0, radius = 2):
        """
        Compute lower-layer heuristic score for a candidate shelter location.

        C^SH = W1 * APD - W2 * NDS
        where APD and NDS are computed from cell-level observables around candidate.
        """
        cell_count = getattr(cellTracker, "countByCell", None)
        cell_danger = getattr(cellTracker, "dangerLevelByCell", None)
        if cell_count is None or cell_danger is None:
            return float(getattr(node, "nodeCap", 0.0))

        nx = int(getattr(cellTracker, "cellXNum", 0))
        ny = int(getattr(cellTracker, "cellYNum", 0))
        if nx <= 0 or ny <= 0:
            return float(getattr(node, "nodeCap", 0.0))

        ci, cj = cellTracker.locateCell(node.nodeX, node.nodeY)
        ci = int(ci)
        cj = int(cj)
        xmin = max(0, ci - int(radius))
        xmax = min(nx - 1, ci + int(radius))
        ymin = max(0, cj - int(radius))
        ymax = min(ny - 1, cj + int(radius))

        apd = 0.0
        nds = 0.0
        for i in range(xmin, xmax + 1):
            for j in range(ymin, ymax + 1):
                k = i * ny + j
                ped_load = float(cell_count[k])
                danger = float(cell_danger[k])
                center = cellTracker.getCellCenter((i, j))
                dx = float(node.nodeX) - float(center[0])
                dy = float(node.nodeY) - float(center[1])
                dist = math.sqrt(dx * dx + dy * dy)
                inv_dist = 1.0 / (1.0 + dist)
                apd += ped_load * inv_dist
                nds += danger * inv_dist

        # Small capacity prior to break ties in favor of higher-throughput sites.
        cap_bonus = 1e-3 * float(max(0.0, getattr(node, "nodeCap", 0.0)))
        return float(w_apd) * apd - float(w_nds) * nds + cap_bonus
        
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
                if self.shelterCanByCell[i][j]:
                    node = self.shelterCanByCell[i][j].pop(0)
                    sid = self.allocID()
                    cap = int(max(0.0, float(getattr(node, "nodeCap", 100.0))))
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
  