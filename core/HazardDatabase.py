#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/7: Constructed the skeletons
9/9: implemented the spreadUpdate function
"""

import numpy as np
import math
from .Hazard import hazardAgent

class HazardDS:
    def __init__(self, hazardNum, casualtyRate, spreadRate, speedReduct):
        # List of active hazard agents
        self.hazardList = {}
        # input parameter: number of total hazards presents in the environment
        self.totalHazard = int(hazardNum)
        
        # hazard casualty probability (probability causing a pedestrian casualty) range (mu, sigma)
        self.casualtyRate = casualtyRate
        # hazard spread probability range (mu, sigma)
        self.spreadRate = spreadRate
        # hazard speed reduction caused to pedestrians (mu, sigma)
        self.speedReduct = speedReduct
        # container for CAProcessor
        self.cellTracker = None
        
    """Helper functions"""
    """given a (mean, variance) pair, sample prob"""
    @staticmethod
    def sampleNormal(prob):
        """
        Given [mu, var], sample a value from the given Gaussian distribution
        """
        if isinstance(prob, (list, tuple)) and len(prob) == 2:
            mu, var = float(prob[0]), float(prob[1])
            std = math.sqrt(max(0.0, var))
            return float(np.random.normal(loc = mu, scale = std))
        return float(prob)
    
    @staticmethod
    def oneClip(x):
        """
        clamp to [0,1], a helper for probablistic sampling
        """
        return max(0.0, min(1.0, float(x)))
    
    @staticmethod
    def distance(a, b):
        """
        Returns the Euclidean distance between 2 points (x1, y1) and (x2, y2)
        """
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    
    def setCellTracker(self, cellTracker):
        self.cellTracker = cellTracker
        
    """Primary functions"""
    def initHazard(self, mapDS, cellTracker = None, lifespan = 60):
        self.cellTracker = cellTracker
        
        hid = 0
        
        for _ in range(self.totalHazard):
            localID = hid
            hid += 1
            
            # sample normal distribution based parameters
            localSpeedReduct = self.oneClip(self.sampleNormal(self.speedReduct))
            localCasualtyRate = self.oneClip(self.sampleNormal(self.casualtyRate))
            localSpreadRate = self.oneClip(self.sampleNormal(self.spreadRate)) / 10
            # define other parameters
            sourceNode = mapDS.assignGenerationNode()
            
            smokeLevel = 0.0
            heatLevel = 0.0
            impactedCells = []
            
            hazard = hazardAgent(localID, 
                                 localSpeedReduct, 
                                 localSpreadRate, 
                                 localCasualtyRate, 
                                 sourceNode, 
                                 smokeLevel, 
                                 heatLevel, 
                                 impactedCells)
        
            hazard.active = True
            hazard.age = 0
            hazard.lifespan = int(lifespan)
            hazard.sourceCell = None
            
            # sample hazard source location
            if self.cellTracker is not None and sourceNode is not None:
                sourceX, sourceY = float(sourceNode.nodeX), float(sourceNode.nodeY)
                sourceCell = self.cellTracker.locateCell(sourceX, sourceY)
                hazard.sourceCell = sourceCell
                hazard.impactedCells.append(hazard.sourceCell)
                
                if int(self.cellTracker.getCellState(sourceCell)) < 1:
                    self.cellTracker.setCellState(hazard.sourceCell, 1)
                    
            self.hazardList[localID] = hazard
    
    def spreadUpdate(self):
        
        if self.cellTracker is None:
            return 0
        
        totalChange = 0
        
        # evaluate the spread and new impact of each hazard
        for hazard in list(self.hazardList.values()):
            # ignore hazards that are no longer active (natually ceased or contained by effort)
            if not hazard.active:
                continue
            
            # for each impacted cell c_i, if any neighboring cell has state = 0, make them state = 1
            spread_rate = self.oneClip(float(hazard.spreadRate))
            fireFront = list(hazard.impactedCells)
            
            newlyImpacted = []
            
            for cell in fireFront:
                # get a list of neighboring cells of the given cell
                neighbors = self.cellTracker.getNeighborCells(cell)
                
                # probabilistically expose close neighbors of currently impacted cells(if neighbors are not impacted)
                for neighbor in neighbors:
                    if self.cellTracker.getCellState(neighbor) == 0 and np.random.random() < spread_rate:
                        self.cellTracker.setCellState(neighbor, 1)
                        if neighbor not in newlyImpacted:
                            newlyImpacted.append(neighbor)
                        totalChange += 1
                        
                # evaluate current cell, for cells at state 2 to 4 to next level
                cellState = int(self.cellTracker.getCellState(cell))
                if 1 <= cellState < 5:
                    if cellState == 1:
                        k_severe = sum(
                                1 for nb in neighbors
                                if int(self.cellTracker.getCellState(nb)) > 3
                        )
                        prob = self.oneClip(0.05 + 0.1*k_severe)
                    else:
                        prob = self.oneClip(0.1 * spread_rate)
                        
                    if np.random.random() < prob:
                        self.cellTracker.setCellState(cell, cellState + 1)
                        totalChange += 1
                        
            if newlyImpacted:
                for nb in newlyImpacted:
                    if nb not in hazard.impactedCells:
                        hazard.impactedCells.append(nb)
                
        return totalChange
            
        # Step 1: (If Statement, nested loop) If the cell is of State 0, iteratively check each neighboring cell to see if trigger Event 
        
        # Step 2: (Else If Statement) If the cell is of State 1, iteratively check and count the number of neighboring cells with State 3 or above
        
        # Step 2.1: Compute the overall SpreadRate = the SumSpreadRate = (\# of State 3 to 5 neighboring cells) * (SpreadRate of the impacting emergency)
        
        # Step 3: (Else Statement) If the cell is of State 2 or above, sample a random integer and compare it to the SpreadRate to determine if to evolve the cell state to the next level
        
    """Heat and Smoke update, following similar logic"""
    def heatUpdate(self):
        if self.cellTracker is None:
            return 0
        
        """May need more reasonable/theoratical proven values for heat"""
        updates = 0
        R = 60.0
        k = 5.0
        
        for hazard in list(self.hazardList.values()):
            # ignore hazards that are no longer active
            if not hazard.active:
                continue
            sourceID = hazard.sourceCell
            if sourceID is None:
                continue
            
            sourceX, sourceY = self.cellTracker.getCellCenter(sourceID)
            tempBase = float(getattr(hazard, "heatLevel", 0.0))
            
            for cell in getattr(hazard, "impactedCells", []):
                cellX, cellY = self.cellTracker.getCellCenter(cell)
                dis = self.distance((sourceX, sourceY), (cellX, cellY))
                cellState = self.cellTracker.getCellState(cell)
                newHeatLevel = tempBase + (k * cellState) * math.exp(-dis / R)
                self.cellTracker.setHeat(cell, newHeatLevel)
                updates += 1
                
        return updates
            
        # Step 1: Acquire the current timestep
        
        # Step 2: (for loop, if statement) For each cell $ce_i$, if $ce_i$ is impacted, then acquire its center location and the hazard instance impacting the cell. 
        # Otherwise, moved on to review the next cell
        
        # Step 3: Execute the equation according to the given equation and return the computed heatwave level
            
    def smokeUpdate(self):
        if self.cellTracker is None:
            return 0
        
        """May need more reasonable/theoratical proven values for smoke"""
        updates = 0
        R = 80.0
        k = 4.0
        
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue
            sourceCell = hazard.sourceCell
            if sourceCell is None:
                continue
            
            sourceX, sourceY = self.cellTracker.getCellCenter(sourceCell)
            smokeBase = float(hazard.smokeLevel)
            
            for cell in getattr(hazard, "impactedCells", []):
                cellX, cellY = self.cellTracker.getCellCenter(cell)
                dis = self.distance((sourceX, sourceY), (cellX, cellY))
                cellState = self.cellTracker.getCellState(cell)
                newSmokeLevel = smokeBase + (k * cellState) * math.exp(-dis / R)
                self.cellTracker.setSmoke(cell, newSmokeLevel)
                updates += 1
                
        return updates
        # Step 1: Acquire the current timestep
        
        # Step 2: (for loop, if statement) For each cell $ce_i$, if $ce_i$ is impacted, then acquire its center location and the hazard instance impacting the cell.
        # Otherwise, moved on to review the next cell.
        
        # Step 3: Execute the equation according to the given equation and return the computed smoke intensity level
            
    def terminateHazard(self):
        terminated = 0
        
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue
            
            hazard.age = int(getattr(hazard, "age", 0)) + 1
            lifespan = int(getattr(hazard, "lifespan", 0))
            
            # check on if the hazard will cease to continue impact
            if lifespan > 0 and hazard.age >= lifespan:
                hazard.active = False
                terminated += 1
                
                # once the hazard cease to exist, downgrade its impact to still require evacuation, but not lethal anymore
                for cell in hazard.impactedCells:
                    if self.cellTracker.getCellState(cell) > 2:
                        self.cellTracker.setCellState(cell, 2)
        # Step 1: (Loop) For each hazard, if the current timestep elapses to the defined timestep in the hazard's lifespan parameter, terminate the hazard
        
        # Step 2: (still brainstorming) For each cell impacted by this hazard, the cell state is assigned to 
        # State 2: impacted (for cells with State 2 and above, so impact no longer causes casualties and spread)
        
        return terminated
    
    
    