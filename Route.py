#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/4: Constructed the skeletons
"""

from Edge import Edge
from Node import Node

class Route:
    def __init__(self, startNode, endNode, nodeRemained, edgeRemained):
        
        self.startNode = startNode
        
        self.endNode = endNode
        
        self.nodeRemained = nodeRemained
        
        self.edgeRemained = edgeRemained
        
    def getStartNode(self):
        return self.startNode
        
    def getEndNode(self):
        return self.endNode
        
    def getNextNode(self):
        nextNode = self.nodeRemained.pop(0)
        return nextNode
    
    def getNextEdge(self):
        nextEdge = self.edgeRemained.pop(0)
        return nextEdge