#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/6: Constructed the skeletons
9/1 - 9/7: implemented the rest of the primary and helper functions

"""

from .Cell import Cell
import math
import numpy as np

class CellTracker:
    def __init__(self, cellXNum, cellYNum):
        # (i, j) -> dict(state, center, heat, smoke, corners)
        self.cellList = {}
        # 3D list containing sub-lists, that consists of a list of guidance candidate nodes at each cell (x_i, y_i)
        self.cellWiseGuidanceCan = None
        # 3D list containing sub-lists, that consists of a list of shelter candidate nodes at each cell (x_i, y_i)
        self.cellWiseShelterCan = None
        # 
        self.cellWiseGuidances = None
    
        self.cellWiseShelters = None
        
        self.cellXNum = int(cellXNum)
        
        self.cellYNum = int(cellYNum)
        
        self.cellXLen = 0.0
        
        self.cellYLen = 0.0
        
        # [(northwest), (northeast), (southwest), (southeast)]
        self.cutBorders = None
        
        """Per-cell flattened features that RL needs"""
        self.countByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.avgVelocityByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.heatByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.smokeByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.dangerLevelByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.shelterFulfillByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.guidanceInterByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.wellnessPenaltyByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        
        self.last_snapshot = None
    
    """Primary functions"""
    def getCut(self):
        return self.cellXLen, self.cellYLen
    
    def initialCut(self, networkXLength, networkYLength):
        
        # check if necessary preparation is made for this function
        if self.cellXNum <= 0 or self.cellYNum <= 0:
            raise ValueError("cellXNum and cellYNum must be positive integers")
        
        # Step 1: Cut both X and Y ranges into equal-length segments of the defined dimensions (so each cell is either a square or rectangle)
        self.cellXLen = float(networkXLength) / self.cellXNum
        self.cellYLen = float(networkYLength) / self.cellYNum
        
        cutBorders = [[None for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        
        # Step 2: (Double for loop) Create a 2D list that contains the [northwest, northeast, southwest, southeast] coordinates of each cell, identifiable via cell\_dim[i][j] for $c_{ij}$
        
        for i in range(self.cellXNum):
            XLeft = i * self.cellXLen
            XRight = (i + 1) * self.cellXLen if i < self.cellXNum - 1 else float(networkXLength)
            
            for j in range(self.cellYNum):
                YDown = j * self.cellYLen
                YUp = (j + 1) * self.cellYLen if j < self.cellYNum - 1 else float(networkYLength)
                
                northWest = (XLeft, YDown)
                northEast = (XRight, YDown)
                southWest = (XLeft, YUp)
                southEast = (XRight, YUp)
                
                cutBorders[i][j] = [northWest, northEast, southWest, southEast]
                
                XCenter = 0.5 * (XLeft + XRight)
                YCenter = 0.5 * (YDown + YUp)
                
                newCell = Cell(
                        cellID = (i, j),
                        impactedLevel = 0,
                        shelterList = [],
                        guidanceList = [],
                        shCanList = [],
                        guCanList = [],
                        corners = (northWest, northEast, southWest, southEast),
                        center = (XCenter, YCenter)
                )
                
                self.cellList[(i, j)] = newCell
                
        self.cutBorders = cutBorders
        
        self.cellWiseGuidanceCan = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        self.cellWiseShelterCan  = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        self.cellWiseGuidances   = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        self.cellWiseShelters    = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        
        return cutBorders
    
    """
    def cellUpdate(self):
        # Step 1: Extract the cell's impact level
        
        # Step 2: For all pedestrian groups that are currently located in $ce_i$, aggregate their volume
        
        # Step 3: Iterate through active shelters in $ce_i$'s shelters list, aggregate their current flow and capacity, and compute the fulfillment ratio (total flow/total capacity)
        
        # Step 4: Iterate through active guidance points in $ce_i$'s guidance points list, aggregate their current flows
        
        # Step 5: Extract the current speed, then extract the combined forces by calling SelfForceUpdate and ImpactForceUpdate functions, and compute the new speed with the given equation
        
        # Step 6: Concat the computed parameter values as a set, add to the 2D array as an element
        
        return 0
    """
    def cellUpdate(self,
                   heat_range = (0.0, 80.0),
                   smoke_range = (0.0, 300.0),
                   heat_weight = 0.6,
                   neighbor_push = 0.4,
                   max_state = 5,
                   decay_if_safe = True, 
                   pedDS = None,
                   forceTracker = None, 
                   save_snapshot = True
            ):
        
        snapshot = [[None for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        
        """Helper functions"""
        def flatten(key, default = 0.0):
            out = np.zeros(total, dtype = float)
            
            k = 0
            for i in range(Nx):
                for j in range(Ny):
                    v = snapshot[i][j].get(key, default)
                    if v is None: v = default
                    out[k] = float(v)
                    k += 1
                    
            return out
        
        def _norm(arr, lo, hi):
            if hi <= lo:
                return np.zeros_like(arr)
            
            z = (arr - lo) / (hi - lo)
            return np.clip(z, 0.0, 1.0)
        
        def agents_in_cell(i, j):
            if pedDS is None:
                return []
            
            out = []
            
            for ped in pedDS.pedAgentList.values():
                if getattr(ped, "terminated", False):
                    continue
                c = getattr(ped, "currCell", None)
                if c is None:
                    continue
                
                ci = int(c[0])
                cj = int(c[1])
                if ci == i and cj == j:
                    out.append(ped)
            
            return out
        
        # function preparation, dimension and container for the state of the overall environment
        Nx, Ny = self.cellXNum, self.cellYNum
        total = Nx * Ny
        snapshot = [[None for _ in range(Ny)] for _ in range(Nx)]
        
        # containers for per-cell raw data
        heat_raw   = np.zeros(total, dtype=float)
        smoke_raw  = np.zeros(total, dtype=float)
        count_arr  = np.zeros(total, dtype=float)
        speed_arr  = np.zeros(total, dtype=float)
        shFulfill  = np.zeros(total, dtype=float)
        guFlow     = np.zeros(total, dtype=float)
        danger_arr = np.zeros(total, dtype=float)
        
        k_flat = 0
        
        for i in range(self.cellXNum):
            for j in range(self.cellYNum):
                cell = self.cellList[(i, j)]
                
                # cell state level (severity of hazarous impact)
                stateLevel = cell.getState() if hasattr(cell, "getState") else getattr(cell, "impactedLevel", 0)
                
                # group and volume of pedestrians in the currently reviewing cell
                agents = agents_in_cell(i, j)
                ped_volume = len(agents)
                
                # get shelter flow and total shelter capacity in the current cell
                sh_flow = 0.0
                sh_cap = 0.0
                
                for sh in cell.shelterList:
                    sh_flow += float(getattr(sh, "shelterFlow", 0.0))
                    sh_cap += float(getattr(sh, "shelterCap", 0.0))

                sh_fulfill = (sh_flow / sh_cap) if sh_cap > 0 else 0.0
                
                # get guidance flow in this cell
                gu_flow = 0.0
                for gu in cell.guidanceList:
                    gu_flow += float(getattr(gu, "guidedFlow", 0.0))
                
                # average speed before and after apply the computed, combined force to the cell
                avg_speed_before = None
                avg_speed_after = None
                
                if ped_volume > 0:
                    speeds_before = [float(getattr(ped, "currSpeed", 0.0)) for ped in agents]
                    avg_speed_before = sum(speeds_before) / ped_volume
                    
                    # apply forces
                    if forceTracker is not None and hasattr(forceTracker, "computeSpeed"):
                        speeds_after = []
                        for ped in agents:
                            base_v = float(getattr(ped, "currSpeed", 0.0))
                            new_v = float(forceTracker.computeSpeed(base_v, cell))
                            ped.currSpeed = new_v
                            speeds_after.append(new_v)
                        avg_speed_after = sum(speeds_after) / ped_volume
                    else:
                        avg_speed_after = avg_speed_before
                
                    # document per-cell snapshot/state for analysis
                snapshot[i][j] = {
                    "state": int(stateLevel),
                    "ped_volume": int(ped_volume),
                    "shelter_flow": float(sh_flow),
                    "shelter_cap": float(sh_cap),
                    "shelter_fulfillment": float(sh_fulfill),
                    "guidance_flow": float(gu_flow),
                    "avg_speed_before": (None if avg_speed_before is None else float(avg_speed_before)),
                    "avg_speed_after": (None if avg_speed_after is None else float(avg_speed_after))
                }
        
                heat_val = float(getattr(cell, "heat", 0.0))
                smoke_val = float(getattr(cell, "smoke", 0.0))
                
                heat_raw[k_flat] = heat_val
                smoke_raw[k_flat] = smoke_val
                count_arr[k_flat] = float(ped_volume)
                shFulfill[k_flat] = float(sh_fulfill)
                guFlow[k_flat] = float(gu_flow)
                
                # pick speed_after if valid else before, else 0
                if avg_speed_after is not None:
                    speed_arr[k_flat] = float(avg_speed_after)
                elif avg_speed_before is not None:
                    speed_arr[k_flat] = float(avg_speed_before)
                else:
                    speed_arr[k_flat] = 0.0
                    
                danger_arr[k_flat] = float(stateLevel)
                
                k_flat += 1
        
        heat_norm = _norm(heat_raw, *heat_range)
        smoke_norm = _norm(smoke_raw, *smoke_range)
        
        # danger level computed with weighted heat and smoke levels
        danger_combo = (float(heat_weight) * heat_norm + float(neighbor_push) * smoke_norm)
        
        wellness_penalty = danger_combo.copy()
        
        # write each parameter into its corresponding container
        self.heatByCell = heat_norm
        self.smokeByCell = smoke_norm
        self.dangerLevelByCell = danger_combo
        self.countByCell = count_arr
        self.avgVelocityByCell = speed_arr
        self.shelterFulfillByCell = shFulfill
        self.guidanceInterByCell = guFlow
        self.wellnessPenaltyByCell = wellness_penalty
        
        # save for logging purpose
        if save_snapshot:
            self.last_snapshot = snapshot
        
        return snapshot
    
    def update_all_cells(self, *args, **kwargs):
        return self.cellUpdate(*args, **kwargs)
                
    """This function receives a pair of (X, Y) coordinate and return the ID of the cell this point is in"""
    def locateCell(self, XPos, YPos):
        
        # check if functions are called in orders, check for prerequisite functions
        if self.cellXLen == 0 or self.cellYLen == 0.0:
            raise RuntimeError("Call initialCut() before locateCell().")
        
        XID = int(float(XPos) // self.cellXLen)
        YID = int(float(YPos) // self.cellYLen)
        
        # Cap values
        if XID < 0: XID = 0
        if YID < 0: YID = 0
        if XID >= self.cellXNum: XID = self.cellXNum - 1
        if YID >= self.cellYNum: YID = self.cellYNum - 1
        
        return [XID, YID]
    
    """return the 8-direction neighbors of given cell ID (i, j) pair"""
    def getNeighborCells(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        XID, YID = cellID
        neighbors = []
        for deltaI in (-1, 0, 1):
            for deltaJ in (-1, 0, 1):
                # ignore the given cell
                if deltaI == 0 and deltaJ == 0:
                    continue
                ni, nj = XID + deltaI, YID + deltaJ
                if 0 <= ni < self.cellXNum and 0 <= nj < self.cellYNum:
                    neighbors.append((ni, nj))
        
        return neighbors
    
    def getCellState(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        cell = self.cellList[cellID]
        return cell.getState() if hasattr(cell, "getState") else getattr(cell, "impactedLevel", 0)
    
    def setCellState(self, cellID, newState):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        cell = self.cellList[cellID]
        if hasattr(cell, "setState"):
            cell.setState(newState)
        else:
            cell.impactedLevel = newState
    
    def getCellCenter(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return self.cellList[cellID].getCenter()
    
    """the following two functions are called by HazardDatabase"""
    def setHeat(self, cellID, value):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        self.cellList[cellID].setHeat(value)
    
    def setSmoke(self, cellID, value):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        self.cellList[cellID].setSmoke(value)
        
    def getHeat(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return float(getattr(self.cellList[cellID], "heat", 0.0))

    def getSmoke(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return float(getattr(self.cellList[cellID], "smoke", 0.0))
    
    
    """Helper functions"""
    def getCell(self, i, j) -> Cell:
        return self.cellList[(i, j)]
    
    def addGuidanceCan(self, cellID, candidate):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].guCanList.append(candidate)
        if self.cellWiseGuidanceCan is not None:
            self.cellWiseGuidanceCan[i][j].append(candidate)
    
    def addShelterCan(self, cellID, candidate):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].shCanList.append(candidate)
        if self.cellWiseShelterCan is not None:
            self.cellWiseShelterCan[i][j].append(candidate)
    
    def addGuidance(self, cellID, guidanceObj):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].guidanceList.append(guidanceObj)
        if self.cellWiseGuidances is not None:
            self.cellWiseGuidances[i][j].append(guidanceObj)
    
    def addShelter(self, cellID, shelterObj):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].shelterList.append(shelterObj)
        if self.cellWiseShelters is not None:
            self.cellWiseShelters[i][j].append(shelterObj)
    
    def getCellCorners(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return self.cellList[cellID].corners
    