#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/3: Constructed the skeletons, parameters, getter functions and primary functions, may add more primary functions later.

"""

class Pedestrian:
    def __init__(self, agentID, routeFollowing, currNode, currEdge, currCell, lastX, lastY, currSpeed, affected, atNode, casualty, evacuated, arrival, terminated, guided):
        # The unique identifier of the pedestrian agent
        self.agentID = agentID
        # Pointer to the route entity the pedestrian agent is currently following
        self.routeFollowing = routeFollowing
        # Pointer to the current node the pedestrian is located at.
        # None if the agent is currently traveling on an edge
        self.currNode = currNode
        # Pointer to the current edge the pedestrian is traveling on,
        # None if the agent is currently on an node
        self.currEdge = currEdge
        # Pointer to the current cell the pedestrian agent is located in
        self.currCell = currCell
        # The X-axis coordinate of agent's current location
        self.lastX = lastX
        # The Y-axis coordinate of agent's current location
        self.lastY = lastY
        # The current traveling speed of the agent, subject to impact from combined Force
        self.currSpeed = currSpeed
        
        self.edge_remain = 0.0
        
        self.edge_vec = None
        
        self.desired_speed = 0.0
        
        self.group_id = 0
        
        """Binary statuses"""
        # Whether the agent is affected by any hazard
        self.affected = affected
        # Whether the agent is at a node
        self.atNode = atNode
        # Whether the agent has become a casualty
        self.casualty = casualty
        # Whether the agent has successfully evacuated
        self.evacuated = evacuated
        # Whether the agent has arrived to intended destination without being affected
        self.arrival = arrival 
        # Whether the agent has terminated from the network due to arrival, successful evacuation, or casualty
        self.terminated = terminated
        # Whether the agent has interacted with any guidance point
        self.guided = guided
        
        self.edge_dest_node = None
    
    """Getter Functions"""
    def getAgentID(self):
        return self.agentID
    
    def getRouteFollowing(self):
        return self.routeFollowing
    
    def getCurrNode(self):
        return self.currNode
    
    def getCurrEdge(self):
        return self.currEdge
    
    def getCurrCell(self):
        return self.currCell
    
    def getLastX(self):
        return self.lastX
    
    def getLastY(self):
        return self.lastY
    
    def getCurrSpeed(self):
        return self.currSpeed
    
    def getAffected(self):
        return self.affected
    
    def getAtNodeStatus(self):
        return self.atNode
    
    def getCasualtyStatus(self):
        return self.casualty
    
    def getEvacuatedStatus(self):
        return self.evacuated
    
    def getArrivalStatus(self):
        return self.arrival
    
    def getTerminatedStatus(self):
        return self.terminated
    
    def getGuidedStatus(self):
        return self.guided
    
    """Calculation functions"""
    def updateSpeed(self, forceTotal):
        self.currSpeed = self.currSpeed * forceTotal
        
    def setNewX(self, newX):
        self.lastX = newX
        
    def setNewY(self, newY):
        self.lastY = newY
    