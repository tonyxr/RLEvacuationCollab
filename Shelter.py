#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/6: Constructed the skeletons
"""

class Shelter:
    OPEN = 0
    FULL = 1
    OVERRUN = 2

    def __init__(self, shelterID, nodeMapped, cellLocated, cap, flow, status):
        # Unique identifier of the shelter entity
        self.shelterID = shelterID
        # The node that this shelter is established upon
        self.nodeMapped = nodeMapped
        # The cell this shelter is in
        self.cellLocated = cellLocated
        # The capacity of this shelter
        self.shelterCap = cap
        # the current flow of this shelter
        self.shelterFlow = flow
        # the operational status of this shelter
        self.status = status
        # Hazard failure is permanent for one wildfire episode.  These fields
        # make the transition auditable in live runs, counterfactual snapshots
        # and exported shelter trajectories.
        self.overrunTime = None
        self.overrunDangerLevel = None
        self.forcedReevacuatedPopulation = 0
        
    def getShelterID(self):
        return self.shelterID
    
    def getNodeMapped(self):
        return self.nodeMapped
    
    def getCellLocated(self):
        return self.cellLocated
    
    def getShelterCap(self):
        return self.shelterCap
    
    def getShelterFlow(self):
        return self.shelterFlow
    
    def getStatus(self):
        return self.status

    def setStatus(self, status):
        status = int(status)
        if status not in {self.OPEN, self.FULL, self.OVERRUN}:
            raise ValueError("Shelter status must be OPEN, FULL, or OVERRUN")
        if int(getattr(self, "status", self.OPEN)) == self.OVERRUN and status != self.OVERRUN:
            raise RuntimeError("An overrun shelter cannot reopen during an episode")
        self.status = status
        return self.status

    def markOverrun(self, *, simulation_time=None, danger_level=None):
        """Permanently fail this shelter and record the causal transition."""
        if int(self.status) == self.OVERRUN:
            return False
        self.status = self.OVERRUN
        self.overrunTime = None if simulation_time is None else int(simulation_time)
        self.overrunDangerLevel = (
            None if danger_level is None else int(danger_level)
        )
        return True
    
    def updateFlow(self, flowAdded):
        self.shelterFlow += flowAdded
