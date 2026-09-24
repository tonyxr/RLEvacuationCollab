#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated guidance-point entity.

Guidance is outside the current planner-side shelter-deployment model.  The
class remains importable only so historical checkpoints and scripts fail
gracefully instead of encountering a missing module.
"""

import warnings

DEPRECATED = True
DEPRECATION_MESSAGE = (
    "Guidance is deprecated and excluded from the active simulation model"
)

class Guidance:
    """Deprecated compatibility type; no production code creates instances."""

    DEPRECATED = True

    def __init__(self, guidanceID, nodeMapped, cellLocated, shelterPointer, totalFlow, guidedFlow, guStatus):
        warnings.warn(
            DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
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
