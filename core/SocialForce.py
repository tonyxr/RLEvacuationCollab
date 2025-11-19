#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/7: Constructed the skeletons
"""

import math
from typing import Any, Optional, Tuple, Union

try:
    from .Cell import Cell as CellType
except Exception:
    CellType = Any
    
Number = Union[int, float]

class ForceProcessor:
    def __init__(self,
                 heat_range: Tuple[float, float] = (0.0, 80.0),
                 smoke_range: Tuple[float, float] = (0.0, 300.0),
                 heat_weight: float = 0.6,
                 smoke_weight: float = 0.4,
                 self_state_threshold: int = 1):
        
        self.currSelfForce: float = 0.0
        self.currImpactForce: float = 0.0
        
        self.heat_min, self.heat_max = float(heat_range[0]), float(heat_range[1])
        self.smoke_min, self.smoke_max = float(smoke_range[0]), float(smoke_range[1])
        
        self.heat_w = float(heat_weight)
        self.smoke_w = float(smoke_weight)
        wsum = self.heat_w + self.smoke_w
        if wsum > 0:
            self.heat_w /= wsum
            self.smoke_w /= wsum
        
        self.self_state_threshold = int(self_state_threshold)
        
        self.cellTracker = None
    
    """Helper functions"""
    @staticmethod
    def noneFound(x: float) -> bool:
        if x is None:
            return True
        if isinstance(x, (int, float)):
            return not math.isfinite(float(x))
        return False
    
    @staticmethod
    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    
    def setupCellTracker(self, cellTracker: Any) -> None:
        self.cellTracker = cellTracker
        
    def computeSpeed(self, 
                     baseSpeed: float,
                     cell: Any = None, 
                     k_self: float = 0.05, 
                     k_impact: float = 0.50, 
                     min_speed: float= 0.0,
                     max_speed: Optional[float] = None) -> float:
        
        self.selfForceUpdate(cell)
        self.impactForceUpdate(cell)
        
        v = 0.0 if self.noneFound(baseSpeed) else float(baseSpeed)
        v *= (1.0 + float(k_self) * float(self.currSelfForce))
        v *= max(0.0, 1.0 - float(k_impact) * float(self.currImpactForce))
        
        v = max(float(min_speed), v)
        if max_speed is not None and math.isfinite(float(max_speed)):
            v = min(float(max_speed), v)
        return v
        
    def normalize(self, value, vmin, vmax):
        """
        Normalize to [0, 1] with guards. Returns 0 if value is missing or invalid
        """
        if value is None or self.noneFound(value) or vmax <= vmin:
            return 0.0
        x = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
        return self._clamp01(x)
    
    def extractCellInfo(self, cell):
        state = None
        heat = None
        smoke = None
        
        # in case passing in tuple of cell ID (i, j)
        if isinstance(cell, (tuple, list)) and len(cell) == 2 and self.cellTracker is not None:
            try:
                i, j = int(cell[0]), int(cell[1])
                # state via CAProcessor
                try:
                    state = int(self.cellTracker.getCellState((i, j)))
                except Exception:
                    state = None
                # heat/smoke via Cell object
                try:
                    cobj = self.cellTracker.getCell(i, j)
                    heat = float(getattr(cobj, "heat", 0.0))
                    smoke = float(getattr(cobj, "smoke", 0.0))
                except Exception:
                    pass
                return state, heat, smoke
            except Exception:
                # fall through to other handlers
                pass
        
        """
        # in case passing in cell obj as dict
        if isinstance(cell, dict):
            state = cell.get('state', cell.get('impactedLevel', None))
            state = None if state is None else int(state)
            heat = cell.get('heat', None)
            smoke = cell.get('smoke', None)
            heat = None if heat is None else float(heat)
            smoke = None if smoke is None else float(smoke)
            return state, heat, smoke
        """
        
        # in case passig in cell object
        if cell is not None:
            if hasattr(cell, 'impactedLevel'):
                try: 
                    state = int(getattr(cell, 'impactedLevel'))
                except Exception:
                    state = None
            elif hasattr(cell, 'state'):
                try:
                    state = int(getattr(cell, 'state'))
                except Exception:
                    state = None
                
            if hasattr(cell, 'heat'):
                try:
                    heat = float(getattr(cell, 'heat'))
                except Exception:
                    heat = None
            if hasattr(cell, 'smoke'):
                try:
                    smoke = float(getattr(cell, 'smoke'))
                except Exception:
                    smoke = None
                
        return state, heat, smoke
        
    """Primary functions"""
    def selfForceUpdate(self, cell):
        # Step 1: Based on the given cell's impact level, conditionally assign and return self-force as 0 or 1
        state, _, _ = self.extractCellInfo(cell)
        self.currSelfForce = 1.0 if (state is not None and state >= self.self_state_threshold) else 0.0
        return self.currSelfForce
            
    def impactForceUpdate(self, cell):
        _, heat, smoke = self.extractCellInfo(cell)
        
        h_norm = self.normalize(heat, self.heat_min, self.heat_max)
        s_norm = self.normalize(smoke, self.smoke_min, self.smoke_max)
        
        impact = self.heat_w * (h_norm** 1.0) + self.smoke_w * (s_norm ** 1.0)
        self.currImpactForce = self._clamp01(float(impact))
        return self.currImpactForce
        # Step 1: Call HeatUpdate and SmokeUpdate to get the heatwave level and smoke intensity level for the given cell
        
        # Step 2: Compute the impact force according to the equation and return.
    
    def compute(self, 
                baseSpeed,
                cell,
                k_self,
                k_impact,
                min_speed,
                max_speed) -> Tuple[float, float, float]:
        
        v = self.computeSpeed(
            baseSpeed = baseSpeed,
            cell = cell,
            k_self = k_self,
            k_impact = k_impact,
            min_speed = min_speed,
            max_speed = max_speed
        )
        
        return v, self.currSelfForce, self.currImpactForce