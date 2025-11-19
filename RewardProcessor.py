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
        gamma_u_sh: float = 0.1,
        lambda_u_gu: float = 0.05,
        zeta_cost_sh: float = 0.001,
        eta_cost_gu: float = 0.001
    ):
         # which reward (simple or full) mechanism to use
         self.mode = mode
         self.alpha = alpha
         self.beta = beta
         self.gamma_u_sh = gamma_u_sh
         self.lambda_u_gu = lambda_u_gu
         self.zeta_cost_sh = zeta_cost_sh
         self.eta_cost_gu = eta_cost_gu
    
    """Simple reward mechanism, equation 9"""
    def simpleReward(self, 
                     numCasualties: int, 
                     t: int) -> float:
        
        return -self.alpha * float(numCasualties) - self.beta * float(t)
    
    def fullReward(self,
                   numCasualties: int, 
                   wellnessPenaltySum: float,
                   fulfillmentSum: float,
                   guidedSum: float,
                   totalShelters: int,
                   totalGuidances: int
                   ) -> float:
        
        actionReward = -self.zeta_cost_sh * float(totalShelters) - self.eta_cost_gu * float(totalGuidances)
        
        forceReward = self.gamma_u_sh * float(fulfillmentSum) + self.lambda_u_gu * float(guidedSum)
        
        totalReward = -self.alpha * float(numCasualties) - self.beta * float(wellnessPenaltySum) + actionReward + forceReward
        return float(totalReward)
    
    def rewardMode(self, **kwargs) -> float:
        if self.mode == "simple":
            return self.simpleReward(kwargs["numCasualties"], kwargs["t"])
        else:
            return self.fullReward(
                kwargs["numCasualties"],
                kwargs["wellnessPenaltySum"],
                kwargs["fulfillmentSum"],
                kwargs["guidedSum"],
                kwargs["totalShelters"],
                kwargs["totalGuidances"]
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
    guided  = getattr(cellTracker, "guidanceInterByCell", None)    
    
    return dict(
        wellnessPenaltySum = _safe_sum(wellness, 0.0),
        fulfillmentSum = _safe_sum(fulfill, 0.0),
        guidedSum = _safe_sum(guided, 0.0),
    )