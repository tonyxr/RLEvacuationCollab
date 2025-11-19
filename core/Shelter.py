#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/6: Constructed the skeletons
"""

class Shelter: 
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

    def setStatus(self):
        return 0
    
    def updateFlow(self, flowAdded):
        self.shelterFlow += flowAdded