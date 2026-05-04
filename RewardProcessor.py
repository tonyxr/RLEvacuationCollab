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
        delta_evac: float = 0.18,
        gamma_u_sh: float = 0.12,
        zeta_cost_sh: float = 0.001,
        install_bonus_sh: float = 0.12,
        failed_install_penalty_sh: float = 0.08,
        capacity_waste_penalty_sh: float = 0.10,
        timely_fill_bonus_sh: float = 0.16,
        open_shelter_penalty_sh: float = 0.01,
        wellness_penalty_coef: float = 0.005,
        shelter_install_cost_weight: float = 0.005,
        delayed_new_shelter_evac_weight: float = 0.2,
        rerouted_arrival_speed_weight: float = 0.3,
        immediate_reroute_reward_weight: float = 0.25,
        timely_rerouted_evac_weight: float = 0.35,
        casualty_penalty_weight: float = 2.5,
        stranded_penalty_weight: float = 0.02,
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
         self.capacity_waste_penalty_sh = capacity_waste_penalty_sh
         self.timely_fill_bonus_sh = timely_fill_bonus_sh
         self.open_shelter_penalty_sh = open_shelter_penalty_sh
         self.wellness_penalty_coef = wellness_penalty_coef
         self.shelter_install_cost_weight = float(shelter_install_cost_weight)
         self.delayed_new_shelter_evac_weight = float(delayed_new_shelter_evac_weight)
         self.rerouted_arrival_speed_weight = float(rerouted_arrival_speed_weight)
         self.immediate_reroute_reward_weight = float(immediate_reroute_reward_weight)
         self.timely_rerouted_evac_weight = float(timely_rerouted_evac_weight)
         self.casualty_penalty_weight = float(casualty_penalty_weight)
         self.stranded_penalty_weight = float(stranded_penalty_weight)
         
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
                   installedShelterCapacityThisStep: float = 0.0,
                   delayedNewShelterEvac: float = 0.0,
                   reroutedArrivalSpeedScore: float = 0.0,
                   timelyReroutedEvacScore: float = 0.0,
                   immediateReroutedCount: float = 0.0,
                   strandedCount: int = 0,
                   t: int = 0,
                   maxEpisodeSteps: int = 120,
                   ) -> float:
        
        install_cost = float(max(0.0, installedShelterCapacityThisStep))
        delayed_evac = float(max(0.0, delayedNewShelterEvac))
        rerouted_arrival_speed = float(max(0.0, reroutedArrivalSpeedScore))
        timely_rerouted_evac = float(max(0.0, timelyReroutedEvacScore))
        immediate_rerouted = float(max(0.0, immediateReroutedCount))
        stranded_count = float(max(0, int(strandedCount)))
        casualty_delta = float(max(0, int(numCasualties) - int(self.lastCasualty)))
        # Scale opening cost with shelter capacity volume and penalize low immediate utilization.
        waste_multiplier = 1.0 + (1.0 / (1.0 + immediate_rerouted))
        
        progress = float(max(0.0, min(1.0, float(t) / max(1.0, float(maxEpisodeSteps)))))
        early_focus = 1.0 - progress
        late_focus = progress

        immediate_weight = self.immediate_reroute_reward_weight * (1.0 + 1.25 * early_focus)
        timely_weight = self.timely_rerouted_evac_weight * (1.0 + 1.25 * late_focus)
        delayed_weight = self.delayed_new_shelter_evac_weight * (1.0 + 0.60 * late_focus)
        cost_weight = self.shelter_install_cost_weight
        casualty_weight = self.casualty_penalty_weight
        
        totalReward = (
            - cost_weight * install_cost * waste_multiplier
            + immediate_weight * immediate_rerouted
            + delayed_weight * delayed_evac
            + self.rerouted_arrival_speed_weight * rerouted_arrival_speed
            + timely_weight * timely_rerouted_evac
            - casualty_weight * casualty_delta
            - self.stranded_penalty_weight * stranded_count
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
                kwargs.get("installedShelterCapacityThisStep", 0.0),
                kwargs.get("delayedNewShelterEvac", 0.0),
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