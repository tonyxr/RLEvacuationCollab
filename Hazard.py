#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/6: Constructed the skeletons
"""

class hazardAgent: 
    def __init__(self, hazardID, speedReduct, spreadRate, casualtyRate, sourceNode, smokeLevel, tempLevel, impactedCells):
        self.hazardID = hazardID
        
        self.speedReduct = speedReduct
        
        self.casualtyRate = casualtyRate
        
        self.spreadRate = spreadRate
        
        self.sourceNode = sourceNode
        
        self.smokeLevel = smokeLevel
        
        self.tempLevel = tempLevel
        
        self.impactedCells = impactedCells
        
        self.active = True
        
        self.age = 0
        
        self.lifespan = 0
        
        self.sourceCell = None
                        
        """Smoke level helper parameters"""
        
        """Temp level helper parameters"""
    
    def updateSmokeLevel(self, newLevel):
        return 0
    
    def updateTempLevel(self, newlevel):
        return 0
    
    
    