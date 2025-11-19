#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/7: Constructed the skeletons
"""

class Cell:
    def __init__(self, cellID, impactedLevel, shelterList, guidanceList, shCanList, guCanList, corners, center = None):
        # The model assigned ID for each distinctive cell in the network environment
        self.cellID = tuple(cellID)
        # The numerical impact level of the cell (see paper for clarification), range = 1 to 5
        self.impactedLevel = int(impactedLevel)
        # The list of active, functioning shelters in the cell
        self.shelterList = list(shelterList)
        # The list of active, functioning guidance points in the cell
        self.guidanceList = list(guidanceList)
        # The list of candidate locations for future shelter installations
        self.shCanList = list(shCanList)
        # The list of candidate locations for future guidance point installations
        self.guCanList = list(guCanList)
        # Base probability of the hazard severity to elevate at the cell at any timestep
        # Also governs the probability of the hazard to impact neighboring states
        self.spreadingProb = 0.0
        
        self.corners = corners
        
        self.center = (0.0, 0.0) if center is None else tuple(center)
        
        self.heat = 0.0
        
        self.smoke = 0.0 
        
    def setGuList(self, newGuList):
        self.guidanceList = list(newGuList)
    
    def setShList(self, newShelterList):
        self.shelterList = list(newShelterList)
    
    def setShCanList(self, newShCanList):
        self.shCanList = list(newShCanList)
    
    def setGuCanList(self, newGuCanList):
        self.guCanList = list(newGuCanList)
        
    def setState(self, s: int):
        self.impactedLevel = int(s)
    
    def getState(self) -> int:
        return int(self.impactedLevel)

    def setHeat(self, val: float):
        self.heat = float(val)
        
    def setSmoke(self, val: float):
        self.smoke = float(val)
        
    def getCenter(self):
        return self.center
    
    def setCenter(self, newCenter):
        self.center = newCenter
    