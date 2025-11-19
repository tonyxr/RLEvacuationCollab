#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

Log:
7/30: structure created, first version done
"""

class Node:
    def __init__(self, localID, OSMID, nodeX, nodeY, cap, flow, cellID, buildingType):
        
        # unique model-assigned ID for enumeration purposes
        self.localID = localID
        
        # ID assigned in the OpenStreetMap system for the node
        self.OSMID = OSMID
        
        # The X component of the node's converted coordinate in meters
        self.nodeX = nodeX
        
        # The Y component of the node's converted coordinate in meters
        self.nodeY = nodeY
        
        # The pedestrian capacity of the node
        self.nodeCap = cap
        
        # The current, time-dependent pedestrian flow at the node
        self.nodeFlow = flow
        
        # Stores the building type of the node
        self.buildingType = buildingType
        
        # indicate the cell this node is in (for measuring impact)
        self.cellID = cellID
    
    """Getter functions"""
    def getLocalID(self):
        return self.localID
    
    def getOSMID(self):
        return self.OSMID
    
    def getNodeX(self):
        return self.nodeX
    
    def getNodeY(self):
        return self.nodeY
    
    def getNodeCap(self):
        return self.nodeCap
    
    def getNodeFlow(self):
        return self.nodeFlow
    
    def getBuildingType(self):
        return self.buildingType
    
    def getCellID(self):
        return self.cellID
    
    """Primary functions"""
    # Used to update the node flow
    # sign = 0 = positive, sign = 1 = negative
    def updateFlow(self, sign, numChange):
        if sign == 0:
            self.nodeFlow += numChange
        else:
            self.nodeFlow -= numChange
            
    # used to update building type during the mapping process
    def setBuildingType(self, buildingType):
        self.buildingType = buildingType
        