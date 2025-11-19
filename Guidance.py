#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

7/30: constructed the skeletons
"""

class Guidance:
    def __init__(self, guidanceID, nodeMapped, cellLocated, shelterPointer, totalFlow, guidedFlow, guStatus):
        # Unique identifier of the guidance point entity.
        self.guidanceID = guidanceID
        # Corresponding to the intersection node upon which this guidance point is established.
        self.nodeMapped = nodeMapped
        # Indicator of which spatial cell the guidance point is located in (avoid having multiple guidance points in a single cell). 
        self.cellLocated = cellLocated
        # A pointer to the route to which shelter this guidance point is currently redirecting pedestrians. 
        self.shelterPointer = shelterPointer
        # (optional) Indicate how many total pedestrians have passed this guidance point. 
        self.totalFlow = totalFlow
        # Number of pedestrians guided by this guidance point
        self.guidedFlow = guidedFlow
        # Impacted status of the guidance point
        self.guStatus = guStatus
        
    """Getter Functions"""
    def getGuidanceID(self):
        return self.guidanceID
    
    def getNodeMapped(self):
        return self.nodeMapped
    
    def getCellLocated(self):
        return self.cellLocated
    
    def getShelterPointer(self):
        return self.shelterPointer
    
    def getTotalFlow(self):
        return self.totalFlow
    
    def getGuidedFlow(self):
        return self.guidedFlow
    
    def getStatus(self):
        return self.guStatus