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
        delta_evac: float = 0.15,
        gamma_u_sh: float = 0.1,
        zeta_cost_sh: float = 0.001,
        install_bonus_sh: float = 0.2,
        failed_install_penalty_sh: float = 0.05
    ):
         # which reward (simple or full) mechanism to use
         self.mode = mode
         self.alpha = alpha
         self.beta = beta
         self.delta_evac = delta_evac
         self.gamma_u_sh = gamma_u_sh
         self.zeta_cost_sh = zeta_cost_sh
         self.install_bonus_sh = install_bonus_sh
         self.failed_install_penalty_sh = failed_install_penalty_sh
         
         self.currFulfillment = 0
         self.lastFulfillment = 0
         
         self.currCasualty = 0
         self.lastCasualty = 0
         self.lastEvacuated = 0
         
         self.lastTotalSHInstalled = 0
         self.lastTotalGUInstalled = 0
    
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
                   shelterInstalledThisStep: int = 0,
                   shelterInstallAttemptsThisStep: int = 0
                   ) -> float:
        
        fulfillmentDiff = int(fulfillmentSum - self.lastFulfillment)
        casualtyDiff = int(numCasualties - self.lastCasualty)
        evacDiff = int(evacuatedTotal - self.lastEvacuated)
        
        actionReward = -self.zeta_cost_sh * float(totalShelters)        
        effectReward = self.gamma_u_sh * float(fulfillmentDiff)
        
        install_penalty = float(max(0, int(shelterInstallAttemptsThisStep) - int(shelterInstalledThisStep))) * self.failed_install_penalty_sh
        install_reward = float(max(0, int(shelterInstalledThisStep))) * self.install_bonus_sh
        
        totalReward = (
            -self.alpha * float(casualtyDiff)
            + self.delta_evac * float(evacDiff)
            + effectReward
            + install_reward
            - install_penalty
            + actionReward
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
                kwargs.get("shelterInstalledThisStep", 0),
                kwargs.get("shelterInstallAttemptsThisStep", 0),
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