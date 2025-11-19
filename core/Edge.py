#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

Log:
7/30: structure created, first version done
"""

from Node import Node

class Edge: 
    def __init__(self, startNode, endNode, edgeID, OSMID, edgeLen, edgeCap, edgeFlow, cellID):
        
        # The model assigned, unique ID for the edge for enumeration purpose
        self.edgeID = edgeID
        
        # The ID for the edge assigned by the OpenStreetMap 
        self.OSMID = OSMID
        
        # The pointer to the node entity representing the start node v_i
        self.startNode = startNode
        
        # The pointer to the ndoe entity representing the end node v_j
        self.endNode = endNode
        
        # Represents the total traval distance of the represented street
        self.edgeLen = edgeLen
        
        # The pedestrian capacity of the edge
        self.edgeCap = edgeCap
        
        # The current, time-dependent pedestrian flow
        self.edgeFlow = edgeFlow
        
        # indicate the cell this edge is located (for impact purpose)
        self.cellID = cellID
        
    def getEdgeID(self):
        return self.edgeID
    
    def getOSMID(self):
        return self.OSMID
    
    def getStartNode(self):
        return self.startNode
    
    def getEndNode(self):
        return self.endNode
    
    def getEdgeLen(self):
        return self.edgeLen
    
    def getEdgeCap(self):
        return self.edgeCap
    
    def getEdgeFlow(self):
        return self.edgeFlow
    
    def getCellID(self):
        return self.cellID
    
    def updateFlow(self, sign, numChange):
        if sign == 0:
            self.edgeFlow += numChange
        else:
            self.edgeFlow -= numChange