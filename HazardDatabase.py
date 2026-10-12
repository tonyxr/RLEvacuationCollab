#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/7: Constructed the skeletons
9/9: implemented the spreadUpdate function
"""

import numpy as np
import math
from Hazard import hazardAgent

class HazardDS:
    def __init__(
        self,
        hazardNum,
        casualtyRate,
        spreadRate,
        speedReduct,
        *,
        rng=None,
        wind_speed_m_per_minute=0.0,
        wind_direction_degrees=0.0,
        wind_influence=1.0,
        time_step_minutes=1.0,
        spread_speed_m_per_minute=100.0 / 3.0,
    ):
        # List of active hazard agents
        self.hazardList = {}
        # input parameter: number of total hazards presents in the environment
        self.totalHazard = int(hazardNum)
        
        # hazard casualty probability (probability causing a pedestrian casualty) range (mu, sigma)
        self.casualtyRate = self._validate_percent_distribution(
            "hazard casualty rate", casualtyRate
        )
        # hazard spread probability range (mu, sigma)
        self.spreadRate = self._validate_percent_distribution(
            "hazard spread rate", spreadRate
        )
        # hazard speed reduction caused to pedestrians (mu, sigma)
        self.speedReduct = self._validate_percent_distribution(
            "hazard speed reduction", speedReduct
        )
        # container for CAProcessor
        self.cellTracker = None
        # Keep legacy stochastic evolution code and allow temporary deterministic mode.
        self.cell_state_evolution_mode = "deterministic"  # {"deterministic", "stochastic"}
        self.state_duration_by_level = {1: 4, 2: 5, 3: 6, 4: 7}
        self.spread_speed_m_per_minute = float(spread_speed_m_per_minute)
        if (
            not math.isfinite(self.spread_speed_m_per_minute)
            or self.spread_speed_m_per_minute <= 0.0
        ):
            raise ValueError("spread_speed_m_per_minute must be finite and positive")
        # Paper Equations 13--14 and Table 3 defaults.
        self.initial_heat_intensity = 200.0
        self.heat_spread_radius_m = 100.0
        self.heat_growth_rate = 0.03
        self.initial_smoke_intensity = 200.0
        self.smoke_spread_radius_m = 60.0
        self.smoke_growth_rate = 0.04
        self.configure_wind(
            speed_m_per_minute=wind_speed_m_per_minute,
            direction_degrees=wind_direction_degrees,
            influence=wind_influence,
            time_step_minutes=time_step_minutes,
        )
        # Hazard evolution owns its random stream. Policy-dependent pedestrian
        # events must never advance the exogenous hazard sequence in a matched
        # policy comparison.
        self.rng = rng if rng is not None else np.random.default_rng()

    def configure_wind(
        self,
        *,
        speed_m_per_minute,
        direction_degrees,
        influence=1.0,
        time_step_minutes=1.0,
    ):
        """Set the observable wind vector used by spread and forecasting.

        Direction is the direction toward which the hazard is transported,
        measured counter-clockwise from projected east. Zero speed recovers
        the previous isotropic spread process exactly.
        """
        speed = float(speed_m_per_minute)
        direction = float(direction_degrees)
        influence = float(influence)
        time_step = float(time_step_minutes)
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("wind speed must be finite and non-negative")
        if not math.isfinite(direction):
            raise ValueError("wind direction must be finite")
        if not math.isfinite(influence) or influence < 0.0:
            raise ValueError("wind influence must be finite and non-negative")
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step_minutes must be finite and positive")
        self.wind_speed_m_per_minute = speed
        self.wind_direction_degrees = direction % 360.0
        self.wind_influence = influence
        self.time_step_minutes = time_step
        
    """Helper functions"""
    @staticmethod
    def _validate_percent_distribution(name, values):
        """Validate a ``[mean, variance]`` pair expressed in percent units."""
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{name} must be a [mean_percent, variance_percent_squared] pair")
        mean, variance = float(values[0]), float(values[1])
        if not math.isfinite(mean) or not 0.0 <= mean <= 100.0:
            raise ValueError(f"{name} mean must be a finite percentage in [0, 100]")
        if not math.isfinite(variance) or variance < 0.0:
            raise ValueError(f"{name} variance must be finite and non-negative")
        return (mean, variance)

    @classmethod
    def sampleProbability(cls, percent_distribution, rng=None):
        """Sample a clipped probability from percent mean/variance inputs."""
        mean, variance = cls._validate_percent_distribution(
            "probability distribution", percent_distribution
        )
        normal = np.random.normal if rng is None else rng.normal
        sampled_percent = float(normal(loc=mean, scale=math.sqrt(variance)))
        return cls.oneClip(sampled_percent / 100.0)
    
    @staticmethod
    def oneClip(x):
        """
        clamp to [0,1], a helper for probablistic sampling
        """
        return max(0.0, min(1.0, float(x)))
    
    @staticmethod
    def distance(a, b):
        """
        Returns the Euclidean distance between 2 points (x1, y1) and (x2, y2)
        """
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def assert_grid_coherent(self, *, require_initialized=True):
        """Validate that every hazard cell belongs to the bound CA grid."""
        if self.cellTracker is None:
            raise RuntimeError("HazardDS has no CellTracker")
        validate = getattr(self.cellTracker, "assert_grid_coherent", None)
        if callable(validate):
            validate(require_initialized=require_initialized)
        if not require_initialized:
            return tuple(self.cellTracker.grid_shape)
        for hazard_id, hazard in self.hazardList.items():
            source = getattr(hazard, "sourceCell", None)
            if source is not None:
                try:
                    self.cellTracker.flatten_cell(source)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has source cell outside grid "
                        f"{self.cellTracker.grid_shape}: {source}"
                    ) from exc
            for cell in getattr(hazard, "impactedCells", ()):
                try:
                    self.cellTracker.flatten_cell(cell)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has impacted cell outside grid "
                        f"{self.cellTracker.grid_shape}: {cell}"
                    ) from exc
            for cell in getattr(hazard, "cellImpactStep", {}):
                try:
                    self.cellTracker.flatten_cell(cell)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has dwell-time cell outside grid "
                        f"{self.cellTracker.grid_shape}: {cell}"
                    ) from exc
            for cell in getattr(hazard, "cellFirstImpactStep", {}):
                try:
                    self.cellTracker.flatten_cell(cell)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has first-impact cell outside grid "
                        f"{self.cellTracker.grid_shape}: {cell}"
                    ) from exc
        return tuple(self.cellTracker.grid_shape)

    @staticmethod
    def _cell_tuple(cell):
        return tuple(cell) if isinstance(cell, list) else cell

    def _current_danger(self) -> np.ndarray:
        """Read danger directly from CA state in decision flattening order."""
        self.assert_grid_coherent(require_initialized=True)
        return self.cellTracker.state_vector(max_state=5, normalized=True)

    def _cell_center(self, cell):
        if self.cellTracker is None:
            return None
        try:
            center = self.cellTracker.getCellCenter(cell)
            return float(center[0]), float(center[1])
        except Exception:
            return None

    def directional_spread_multiplier(self, source_cell, target_cell) -> float:
        """Compute wind's dimensionless multiplier of physical front speed."""
        if self.wind_speed_m_per_minute <= 0.0 or self.wind_influence <= 0.0:
            return 1.0
        source = self._cell_center(source_cell)
        target = self._cell_center(target_cell)
        if source is None or target is None:
            return 1.0
        dx = target[0] - source[0]
        # MapDS stores local y in metres southward. Convert it to a physical
        # northward component before comparing with the public wind vector,
        # whose positive second component is north.
        north = source[1] - target[1]
        norm = math.hypot(dx, north)
        if norm <= 0.0:
            return 1.0
        angle = math.radians(self.wind_direction_degrees)
        alignment = (
            (dx / norm) * math.cos(angle)
            + (north / norm) * math.sin(angle)
        )
        strength = (
            self.wind_influence
            * self.wind_speed_m_per_minute
            / self.spread_speed_m_per_minute
        )
        return float(np.clip(math.exp(strength * alignment), 0.25, 4.0))

    def _neighbor_distance_m(self, source_cell, target_cell) -> float:
        source = self._cell_center(source_cell)
        target = self._cell_center(target_cell)
        if source is None or target is None:
            raise RuntimeError("Hazard spread requires physical cell centers")
        distance = self.distance(source, target)
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("Neighboring hazard cells require positive distance")
        return float(distance)

    def _spread_travel_steps(self, source_cell, target_cell) -> int:
        distance = self._neighbor_distance_m(source_cell, target_cell)
        speed = (
            self.spread_speed_m_per_minute
            * self.directional_spread_multiplier(source_cell, target_cell)
        )
        return max(
            1,
            int(math.ceil(distance / (speed * self.time_step_minutes))),
        )

    def directional_spread_probability(self, base_probability, source_cell, target_cell) -> float:
        """Scale stochastic spread by physical distance and wind transport."""
        base = self.oneClip(base_probability)
        if base <= 0.0 or base >= 1.0:
            return base
        multiplier = self.directional_spread_multiplier(source_cell, target_cell)
        distance = self._neighbor_distance_m(source_cell, target_cell)
        reference_distance = self.spread_speed_m_per_minute * self.time_step_minutes
        exposure = multiplier * reference_distance / distance
        return self.oneClip(1.0 - (1.0 - base) ** exposure)

    def _deterministic_forecast_states(self, horizon_steps: int) -> np.ndarray:
        """Advance a side-effect-free copy of the actual deterministic CA.

        This mirrors ``_spreadUpdateDeterministic`` followed by
        ``terminateHazard``.  It deliberately reuses CellTracker flattening
        and neighbor queries, so a rectangular grid cannot be reinterpreted by
        the forecast layer.
        """
        states = np.rint(self._current_danger() * 5.0).astype(np.int16)
        hazards = []
        for hazard in self.hazardList.values():
            impacted = [
                self._cell_tuple(cell)
                for cell in getattr(hazard, "impactedCells", ())
            ]
            entered = {
                self._cell_tuple(cell): int(step)
                for cell, step in getattr(hazard, "cellImpactStep", {}).items()
            }
            first_impacted = {
                self._cell_tuple(cell): int(step)
                for cell, step in getattr(
                    hazard, "cellFirstImpactStep", entered
                ).items()
            }
            source_cell = self._cell_tuple(
                getattr(hazard, "sourceCell", None)
            )
            if source_cell is not None and source_cell in impacted:
                first_impacted.setdefault(source_cell, 0)
            hazards.append(
                {
                    "active": bool(getattr(hazard, "active", True)),
                    "step": int(getattr(hazard, "step", 0)),
                    "age": int(getattr(hazard, "age", 0)),
                    "lifespan": int(getattr(hazard, "lifespan", 0)),
                    "impacted": impacted,
                    "entered": entered,
                    "first_impacted": first_impacted,
                }
            )

        for _ in range(max(0, int(horizon_steps))):
            for hazard in hazards:
                if not hazard["active"]:
                    continue
                hazard["step"] += 1
                fire_front = list(hazard["impacted"])
                for cell in fire_front:
                    hazard["entered"].setdefault(cell, hazard["step"])
                    hazard["first_impacted"].setdefault(
                        cell, hazard["step"]
                    )

                newly_impacted = []
                for cell in fire_front:
                    for neighbor in self.cellTracker.getNeighborCells(cell):
                        travel_steps = self._spread_travel_steps(cell, neighbor)
                        neighbor_index = self.cellTracker.flatten_cell(neighbor)
                        if (
                            hazard["step"]
                            - int(hazard["first_impacted"][cell])
                            >= travel_steps
                            and int(states[neighbor_index]) == 0
                        ):
                            states[neighbor_index] = 1
                            hazard["entered"][neighbor] = hazard["step"]
                            hazard["first_impacted"][neighbor] = hazard["step"]
                            if neighbor not in newly_impacted:
                                newly_impacted.append(neighbor)
                for cell in newly_impacted:
                    if cell not in hazard["impacted"]:
                        hazard["impacted"].append(cell)

                for cell in list(hazard["impacted"]):
                    index = self.cellTracker.flatten_cell(cell)
                    current_state = int(states[index])
                    if current_state < 1 or current_state >= 5:
                        continue
                    entered_at = int(
                        hazard["entered"].get(cell, hazard["step"])
                    )
                    dwell = int(
                        self.state_duration_by_level.get(current_state, 5)
                    )
                    if hazard["step"] - entered_at >= dwell:
                        states[index] = current_state + 1
                        hazard["entered"][cell] = hazard["step"]

            # Match the simulator's end-of-step hazard lifecycle exactly.
            for hazard in hazards:
                if not hazard["active"]:
                    continue
                hazard["age"] += 1
                if hazard["lifespan"] <= 0 or hazard["age"] < hazard["lifespan"]:
                    continue
                hazard["active"] = False
                for cell in hazard["impacted"]:
                    still_active = any(
                        other is not hazard
                        and other["active"]
                        and cell in other["impacted"]
                        for other in hazards
                    )
                    index = self.cellTracker.flatten_cell(cell)
                    if not still_active and int(states[index]) > 2:
                        states[index] = 2
        return states

    def forecast_danger_by_cell(self, horizon_steps: int) -> np.ndarray:
        """Forecast danger on the exact runtime CA grid without random draws."""
        if self.cellTracker is None:
            return np.empty(0, dtype=np.float32)
        horizon = max(0, int(horizon_steps))
        current = self._current_danger()
        if horizon == 0 or not any(
            bool(getattr(hazard, "active", True))
            for hazard in self.hazardList.values()
        ):
            return current.astype(np.float32)
        if str(self.cell_state_evolution_mode).lower() == "deterministic":
            states = self._deterministic_forecast_states(horizon)
            return np.clip(states.astype(np.float64) / 5.0, 0.0, 1.0).astype(
                np.float32
            )

        risk = current.copy()
        active_rates = [
            self.oneClip(float(getattr(hazard, "spreadRate", 0.0)))
            for hazard in self.hazardList.values()
            if bool(getattr(hazard, "active", True))
        ]
        base_probability = 1.0 - float(
            np.prod([1.0 - rate for rate in active_rates])
        )
        for _ in range(horizon):
            next_risk = risk.copy()
            for source_index in range(self.cellTracker.number_of_cells):
                source_risk = float(risk[source_index])
                if source_risk <= 0.0:
                    continue
                source_cell = self.cellTracker.unflatten_cell(source_index)
                for target_cell in self.cellTracker.getNeighborCells(source_cell):
                    target_index = self.cellTracker.flatten_cell(target_cell)
                    probability = self.directional_spread_probability(
                        base_probability,
                        source_cell,
                        target_cell,
                    )
                    transmitted = source_risk * probability
                    next_risk[target_index] = 1.0 - (
                        (1.0 - next_risk[target_index]) * (1.0 - transmitted)
                    )
            risk = np.clip(next_risk, 0.0, 1.0)
        return risk.astype(np.float32)

    def forecast_distribution_by_cell(self, horizon_steps: int) -> dict[str, np.ndarray]:
        """Summarize the next-front distribution from current spread momentum.

        The one-step forecast supplies the local front velocity.  The longer
        forecast supplies the expected danger at the next decision.  For a
        presently unaffected fraction ``1-current``, the implied escalation
        probability is the fraction expected to become affected by the
        horizon.  Its normalized Bernoulli variance is exposed as uncertainty.
        No simulator random stream is consumed and no realized future is used.
        """
        if self.cellTracker is None:
            empty = np.empty(0, dtype=np.float32)
            return {
                "expected": empty,
                "momentum": empty,
                "escalation_probability": empty,
                "uncertainty": empty,
            }
        node_count = int(self.cellTracker.number_of_cells)
        current = self._current_danger()
        one_step = np.asarray(
            self.forecast_danger_by_cell(1), dtype=np.float64
        ).reshape(node_count)
        expected = np.asarray(
            self.forecast_danger_by_cell(max(1, int(horizon_steps))),
            dtype=np.float64,
        ).reshape(node_count)
        momentum = np.clip(one_step - current, -1.0, 1.0)
        if str(self.cell_state_evolution_mode).lower() == "deterministic":
            escalation = (expected > current + 1e-12).astype(np.float64)
            uncertainty = np.zeros(node_count, dtype=np.float64)
        else:
            unaffected = np.maximum(1.0 - current, 1e-12)
            escalation = np.where(
                current < 1.0 - 1e-12,
                (expected - current) / unaffected,
                0.0,
            )
            escalation = np.clip(escalation, 0.0, 1.0)
            uncertainty = np.clip(
                4.0 * escalation * (1.0 - escalation), 0.0, 1.0
            )
        return {
            "expected": expected.astype(np.float32),
            "momentum": momentum.astype(np.float32),
            "escalation_probability": escalation.astype(np.float32),
            "uncertainty": uncertainty.astype(np.float32),
        }
    
    def setCellTracker(self, cellTracker):
        if cellTracker is None:
            raise ValueError("HazardDS requires a CellTracker")
        validate = getattr(cellTracker, "assert_grid_coherent", None)
        if callable(validate):
            validate(require_initialized=False)
        self.cellTracker = cellTracker
        if self.hazardList and getattr(cellTracker, "cellList", None):
            self.assert_grid_coherent(require_initialized=True)
        
    """Primary functions"""
    def initHazard(self, mapDS, cellTracker = None, lifespan = 60):
        # Episode initialization replaces the complete hazard realization.
        # Clear it before binding the new tracker so an earlier episode's
        # front cannot be validated against, or copied into, a new grid.
        self.hazardList.clear()
        self.setCellTracker(cellTracker)
        validate = getattr(self.cellTracker, "assert_grid_coherent", None)
        if callable(validate):
            validate(require_initialized=True)
        
        hid = 0
        
        for _ in range(self.totalHazard):
            localID = hid
            hid += 1
            
            # CSV inputs use percent means and percent-squared variances.
            localSpeedReduct = self.sampleProbability(self.speedReduct, self.rng)
            localCasualtyRate = self.sampleProbability(self.casualtyRate, self.rng)
            localSpreadRate = self.sampleProbability(self.spreadRate, self.rng)
            # define other parameters
            sourceNode = mapDS.assignGenerationNode()
            
            smokeLevel = 0.0
            heatLevel = 0.0
            impactedCells = []
            
            hazard = hazardAgent(localID, 
                                 localSpeedReduct, 
                                 localSpreadRate, 
                                 localCasualtyRate, 
                                 sourceNode, 
                                 smokeLevel, 
                                 heatLevel, 
                                 impactedCells)
        
            hazard.active = True
            hazard.age = 0
            hazard.lifespan = int(lifespan)
            hazard.sourceCell = None
            hazard.step = 0
            hazard.cellImpactStep = {}
            hazard.cellFirstImpactStep = {}
            
            # sample hazard source location
            if self.cellTracker is not None and sourceNode is not None:
                sourceX, sourceY = float(sourceNode.nodeX), float(sourceNode.nodeY)
                sourceCell = self.cellTracker.locateCell(sourceX, sourceY)
                hazard.sourceCell = sourceCell
                hazard.impactedCells.append(hazard.sourceCell)
                source_key = self._cell_tuple(hazard.sourceCell)
                hazard.cellImpactStep[source_key] = 0
                hazard.cellFirstImpactStep[source_key] = 0
                
                if int(self.cellTracker.getCellState(sourceCell)) < 1:
                    self.cellTracker.setCellState(hazard.sourceCell, 1)
                    
            self.hazardList[localID] = hazard
        self.assert_grid_coherent(require_initialized=True)
    
    def spreadUpdate(self):
        if self.cellTracker is not None:
            self.assert_grid_coherent(require_initialized=True)
        if str(self.cell_state_evolution_mode).lower() == "stochastic":
            return self._spreadUpdateStochastic()
        return self._spreadUpdateDeterministic()

    def _spreadUpdateStochastic(self):
        if self.cellTracker is None:
            return 0
        
        totalChange = 0
        
        # evaluate the spread and new impact of each hazard
        for hazard in list(self.hazardList.values()):
            # ignore hazards that are no longer active (natually ceased or contained by effort)
            if not hazard.active:
                continue
            
            # for each impacted cell c_i, if any neighboring cell has state = 0, make them state = 1
            spread_rate = self.oneClip(float(hazard.spreadRate))
            fireFront = list(hazard.impactedCells)
            
            newlyImpacted = []
            
            for cell in fireFront:
                # get a list of neighboring cells of the given cell
                neighbors = self.cellTracker.getNeighborCells(cell)
                
                # probabilistically expose close neighbors of currently impacted cells(if neighbors are not impacted)
                for neighbor in neighbors:
                    directional_rate = self.directional_spread_probability(
                        spread_rate,
                        cell,
                        neighbor,
                    )
                    if self.cellTracker.getCellState(neighbor) == 0 and self.rng.random() < directional_rate:
                        self.cellTracker.setCellState(neighbor, 1)
                        if neighbor not in newlyImpacted:
                            newlyImpacted.append(neighbor)
                        totalChange += 1
                        
                # evaluate current cell, for cells at state 2 to 4 to next level
                cellState = int(self.cellTracker.getCellState(cell))
                if 1 <= cellState < 5:
                    k_severe = sum(
                        1 for nb in neighbors
                        if int(self.cellTracker.getCellState(nb)) > 3
                    )
                    prob = self.oneClip(spread_rate * (1.0 + 0.5 * k_severe))
                        
                    if self.rng.random() < prob:
                        self.cellTracker.setCellState(cell, cellState + 1)
                        totalChange += 1
                        
            for nb in newlyImpacted:
                if nb not in hazard.impactedCells:
                    hazard.impactedCells.append(nb)
                
        return totalChange
    
    
    def _spreadUpdateDeterministic(self):
        if self.cellTracker is None:
            return 0

        totalChange = 0
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue

            # Initialize deterministic bookkeeping fields lazily.
            if not hasattr(hazard, "step"):
                hazard.step = 0
            if not hasattr(hazard, "cellImpactStep"):
                hazard.cellImpactStep = {}
            if not hasattr(hazard, "cellFirstImpactStep"):
                hazard.cellFirstImpactStep = dict(hazard.cellImpactStep)

            hazard.step += 1
            fireFront = list(getattr(hazard, "impactedCells", []))

            for c in fireFront:
                c_key = tuple(c) if isinstance(c, list) else c
                hazard.cellImpactStep.setdefault(c_key, hazard.step)
                hazard.cellFirstImpactStep.setdefault(c_key, hazard.step)

            # Deterministic spread: wind changes each neighbor's arrival delay.
            newlyImpacted = []
            for cell in fireFront:
                for nb in self.cellTracker.getNeighborCells(cell):
                    travel_steps = self._spread_travel_steps(cell, nb)
                    cell_key = tuple(cell) if isinstance(cell, list) else cell
                    if (
                        hazard.step
                        - int(hazard.cellFirstImpactStep[cell_key])
                        >= travel_steps
                    ):
                        if int(self.cellTracker.getCellState(nb)) == 0:
                            self.cellTracker.setCellState(nb, 1)
                            nb_key = tuple(nb) if isinstance(nb, list) else nb
                            hazard.cellImpactStep[nb_key] = hazard.step
                            hazard.cellFirstImpactStep[nb_key] = hazard.step
                            if nb not in newlyImpacted:
                                newlyImpacted.append(nb)
                            totalChange += 1
            for nb in newlyImpacted:
                if nb not in hazard.impactedCells:
                    hazard.impactedCells.append(nb)

            # Deterministic escalation by fixed dwell-time per state level.
            for cell in list(getattr(hazard, "impactedCells", [])):
                curr_state = int(self.cellTracker.getCellState(cell))
                if curr_state < 1 or curr_state >= 5:
                    continue
                cell_key = tuple(cell) if isinstance(cell, list) else cell
                entered_at = int(hazard.cellImpactStep.get(cell_key, hazard.step))
                dwell = int(self.state_duration_by_level.get(curr_state, 5))
                if (hazard.step - entered_at) >= dwell:
                    self.cellTracker.setCellState(cell, curr_state + 1)
                    hazard.cellImpactStep[cell_key] = hazard.step
                    totalChange += 1

        return totalChange
            
        # Step 1: (If Statement, nested loop) If the cell is of State 0, iteratively check each neighboring cell to see if trigger Event 
        
        # Step 2: (Else If Statement) If the cell is of State 1, iteratively check and count the number of neighboring cells with State 3 or above
        
        # Step 2.1: Compute the overall SpreadRate = the SumSpreadRate = (\# of State 3 to 5 neighboring cells) * (SpreadRate of the impacting emergency)
        
        # Step 3: (Else Statement) If the cell is of State 2 or above, sample a random integer and compare it to the SpreadRate to determine if to evolve the cell state to the next level
        
    """Heat and Smoke update, following similar logic"""
    def heatUpdate(self):
        """Update cell heat using the paper's Gaussian space-time field."""
        if self.cellTracker is None:
            return 0
        self.assert_grid_coherent(require_initialized=True)

        updates = 0
        contributions = {cell_id: 0.0 for cell_id in self.cellTracker.cellList}
        for hazard in list(self.hazardList.values()):
            # ignore hazards that are no longer active
            if not hazard.active:
                continue
            sourceID = hazard.sourceCell
            if sourceID is None:
                continue
            
            sourceX, sourceY = self.cellTracker.getCellCenter(sourceID)
            age = max(0.0, float(getattr(hazard, "age", 0)))
            for cell in self.cellTracker.cellList:
                cellX, cellY = self.cellTracker.getCellCenter(cell)
                dis = self.distance((sourceX, sourceY), (cellX, cellY))
                newHeatLevel = self.initial_heat_intensity * math.exp(
                    -(dis * dis) / (2.0 * self.heat_spread_radius_m ** 2)
                ) * (1.0 + self.heat_growth_rate * age)
                cell_key = tuple(cell) if isinstance(cell, list) else cell
                contributions[cell_key] += float(newHeatLevel)
                updates += 1

        for cell_id, value in contributions.items():
            self.cellTracker.setHeat(cell_id, value)
                
        return updates
            
        # Step 1: Acquire the current timestep
        
        # Step 2: (for loop, if statement) For each cell $ce_i$, if $ce_i$ is impacted, then acquire its center location and the hazard instance impacting the cell. 
        # Otherwise, moved on to review the next cell
        
        # Step 3: Execute the equation according to the given equation and return the computed heatwave level
            
    def smokeUpdate(self):
        """Update cell smoke using the paper's Gaussian space-time field."""
        if self.cellTracker is None:
            return 0
        self.assert_grid_coherent(require_initialized=True)

        updates = 0
        contributions = {cell_id: 0.0 for cell_id in self.cellTracker.cellList}
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue
            sourceCell = hazard.sourceCell
            if sourceCell is None:
                continue
            
            sourceX, sourceY = self.cellTracker.getCellCenter(sourceCell)
            age = max(0.0, float(getattr(hazard, "age", 0)))
            for cell in self.cellTracker.cellList:
                cellX, cellY = self.cellTracker.getCellCenter(cell)
                dis = self.distance((sourceX, sourceY), (cellX, cellY))
                newSmokeLevel = self.initial_smoke_intensity * math.exp(
                    -(dis * dis) / (2.0 * self.smoke_spread_radius_m ** 2)
                ) * (1.0 + self.smoke_growth_rate * age)
                cell_key = tuple(cell) if isinstance(cell, list) else cell
                contributions[cell_key] += float(newSmokeLevel)
                updates += 1

        for cell_id, value in contributions.items():
            self.cellTracker.setSmoke(cell_id, value)
                
        return updates
        # Step 1: Acquire the current timestep
        
        # Step 2: (for loop, if statement) For each cell $ce_i$, if $ce_i$ is impacted, then acquire its center location and the hazard instance impacting the cell.
        # Otherwise, moved on to review the next cell.
        
        # Step 3: Execute the equation according to the given equation and return the computed smoke intensity level
            
    def terminateHazard(self):
        if self.cellTracker is not None:
            self.assert_grid_coherent(require_initialized=True)
        terminated = 0
        
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue
            
            hazard.age = int(getattr(hazard, "age", 0)) + 1
            lifespan = int(getattr(hazard, "lifespan", 0))
            
            # check on if the hazard will cease to continue impact
            if lifespan > 0 and hazard.age >= lifespan:
                hazard.active = False
                terminated += 1
                
                # once the hazard cease to exist, downgrade its impact to still require evacuation, but not lethal anymore
                for cell in hazard.impactedCells:
                    cell_key = tuple(cell) if isinstance(cell, list) else cell
                    still_active = any(
                        other is not hazard
                        and getattr(other, "active", False)
                        and any(
                            (tuple(value) if isinstance(value, list) else value) == cell_key
                            for value in getattr(other, "impactedCells", [])
                        )
                        for other in self.hazardList.values()
                    )
                    if not still_active and self.cellTracker.getCellState(cell_key) > 2:
                        self.cellTracker.setCellState(cell_key, 2)
        return terminated
