#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:26:44 2025

@author: Xiaoru Shi
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

_HAS_PYG = False
if os.getenv("EVAC_ENABLE_PYG", "0") == "1":   
    try:
        from torch_geometric.nn import GATConv, global_mean_pool   
        from torch_geometric.data import Data                      
        _HAS_PYG = True
    except Exception:
        _HAS_PYG = False

class RewardProcessor:
    def __init__(self, 
        mode: str = "full",
        alpha: float = 1.0,
        beta: float = 0.01,
        cell_criticality_weight: float = 1.0,
        local_impact_weight: float = 1.5,
        step_penalty_weight: float = 0.01,
        evac_progress_weight: float = 0.1,
        casualty_delta_weight: float = 1.0,
        fulfillment_delta_weight: float = 0.05,
    ):
         # which reward (simple or full) mechanism to use
         self.mode = mode
         self.alpha = alpha
         self.beta = beta
         self.cell_criticality_weight = float(cell_criticality_weight)
         self.local_impact_weight = float(local_impact_weight)
         self.step_penalty_weight = float(step_penalty_weight)
         self.evac_progress_weight = float(evac_progress_weight)
         self.casualty_delta_weight = float(casualty_delta_weight)
         self.fulfillment_delta_weight = float(fulfillment_delta_weight)
         
         self.currFulfillment = 0
         self.lastFulfillment = 0
         
         self.currCasualty = 0
         self.lastCasualty = 0
         self.lastEvacuated = 0
         
         self.lastTotalSHInstalled = 0
         self.lastTotalGUInstalled = 0
         self.lastUsedShelterCapacity = 0.0
         
    def reset_episode(self):
         self.currFulfillment = 0
         self.lastFulfillment = 0
         self.currCasualty = 0
         self.lastCasualty = 0
         self.lastEvacuated = 0
         self.lastTotalSHInstalled = 0
         self.lastTotalGUInstalled = 0
         self.lastUsedShelterCapacity = 0.0
    
    """Simple reward mechanism, equation 9"""
    def simpleReward(self, 
                     numCasualties: int, 
                     t: int) -> float:
        
        return -self.alpha * float(numCasualties) - self.beta * float(t)
    
    def fullReward(self,
                   numCasualties: int, 
                   wellnessPenaltySum: float,
                   fulfillmentSum: float,
                   evacuatedTotal: int,
                   totalShelters: int,
                   cellCriticalityScore: float = 0.0,
                   localImpactScore: float = 0.0,
                   reroutedArrivalSpeedScore: float = 0.0,
                   timelyReroutedEvacScore: float = 0.0,
                   immediateReroutedCount: float = 0.0,
                   strandedCount: int = 0,
                   t: int = 0,
                   maxEpisodeSteps: int = 120,
                   ) -> float:
        
        # Simplified cell-selection reward:
        # 1) reward picking critical cells now
        # 2) reward short-horizon local improvement after picking that cell
        # 3) constant step cost to avoid dithering
        criticality_score = float(max(0.0, cellCriticalityScore))
        local_impact_raw = float(localImpactScore)
        local_impact_score = float(np.clip(local_impact_raw, -1.0, 1.0))
        
        # Outcome-linked shaping terms (dense deltas), normalized by active population scale
        live_population = float(max(1, evacuatedTotal + numCasualties + max(0, strandedCount)))
        delta_evacuated = float(evacuatedTotal - self.lastEvacuated) / live_population
        delta_casualty = float(numCasualties - self.lastCasualty) / live_population
        delta_fulfillment = float(fulfillmentSum - self.lastFulfillment) / live_population
        
        terminal_bonus = 0.0
        is_terminal = (t >= maxEpisodeSteps - 1) or (strandedCount <= 0)
        if is_terminal:
            evac_rate = float(evacuatedTotal) / live_population
            casualty_rate = float(numCasualties) / live_population
            terminal_bonus = evac_rate - casualty_rate
        
        totalReward = (
            self.cell_criticality_weight * criticality_score
            + self.local_impact_weight * local_impact_score
            + self.evac_progress_weight * delta_evacuated
            - self.casualty_delta_weight * delta_casualty
            + self.fulfillment_delta_weight * delta_fulfillment
            + terminal_bonus
            - self.step_penalty_weight
        )
        
        self.lastFulfillment = fulfillmentSum
        self.lastCasualty = numCasualties
        self.lastEvacuated = evacuatedTotal
        return float(totalReward)
    
    def rewardMode(self, **kwargs) -> float:
        
        if self.mode == "simple":
            return self.simpleReward(
                kwargs.get("numCasualties", 0),
                kwargs.get("t", 0),
            )
        else:
            return self.fullReward(
                kwargs.get("numCasualties", 0),
                kwargs.get("wellnessPenaltySum", 0.0),
                kwargs.get("fulfillmentSum", 0.0),
                kwargs.get("evacuatedTotal", 0),
                kwargs.get("totalShelters", 0),
                kwargs.get("cellCriticalityScore", kwargs.get("immediateReroutedCount", 0.0)),
                kwargs.get(
                    "localImpactScore",
                    kwargs.get("reroutedArrivalSpeedScore", 0.0) + kwargs.get("timelyReroutedEvacScore", 0.0),
                ),
                kwargs.get("reroutedArrivalSpeedScore", 0.0),
                kwargs.get("timelyReroutedEvacScore", 0.0),
                kwargs.get("immediateReroutedCount", 0.0),
                kwargs.get("strandedCount", 0),
                kwargs.get("t", 0),
                kwargs.get("maxEpisodeSteps", 120),
            )

def _safe_sum(x, default = 0.0) -> float:
    if x is None:
        return float(default)
    a = np.asarray(x, dtype = float).reshape(-1)
    if a.size == 0:
        return float(default)
    
    a = np.nan_to_num(a, nan = 0.0, posinf = 0.0, neginf = 0.0)
    return float(a.sum())

def extract_reward_terms(cellTracker) -> Dict[str, float]:
    wellness = getattr(cellTracker, "wellnessPenaltyByCell", None)
    fulfill = getattr(cellTracker, "shelterFulfillByCell", None)
    
    return dict(
        wellnessPenaltySum = _safe_sum(wellness, 0.0),
        fulfillmentSum = _safe_sum(fulfill, 0.0),
    )