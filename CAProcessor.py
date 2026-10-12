#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/6: Constructed the skeletons
9/1 - 9/7: implemented the rest of the primary and helper functions

"""

from Cell import Cell
import numpy as np


def authoritative_grid_shape(core, *, require_initialized=True):
    """Return the one cell-grid shape shared by simulation and decisions.

    ``cellX``/``cellY`` are configuration inputs.  Once a ``CellTracker`` is
    constructed, its shape is authoritative for every cell-indexed subsystem.
    Keeping this check at the boundary prevents a same-cell-count mismatch
    such as 2x8 versus 4x4 from silently reinterpreting hazard or action data.
    """
    tracker = getattr(core, "cellTracker", None)
    if tracker is None:
        raise RuntimeError("The cell grid is unavailable before CellTracker initialization")
    try:
        shape = (int(tracker.cellXNum), int(tracker.cellYNum))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("CellTracker must declare cellXNum and cellYNum") from exc
    configured = (int(getattr(core, "cellX", 0)), int(getattr(core, "cellY", 0)))
    if configured != shape:
        raise RuntimeError(
            "Configured cell grid and CellTracker disagree: "
            f"configured={configured}, tracker={shape}"
        )
    validate = getattr(tracker, "assert_grid_coherent", None)
    if callable(validate):
        validate(require_initialized=require_initialized)
    hazard_store = getattr(core, "hazardDS", None)
    hazard_tracker = getattr(hazard_store, "cellTracker", None)
    if hazard_tracker is not None and hazard_tracker is not tracker:
        raise RuntimeError("HazardDS and the simulator do not share the same CellTracker")
    hazard_validate = getattr(hazard_store, "assert_grid_coherent", None)
    if callable(hazard_validate) and hazard_tracker is tracker:
        hazard_validate(require_initialized=require_initialized)
    return shape


class CellTracker:
    # One authoritative cellular-neighborhood contract for hazard evolution,
    # forecasting, visualization, and the decision graph.  Keeping the
    # offsets here prevents those subsystems from silently using different
    # topologies on the same nominal grid.
    MOORE_NEIGHBOR_OFFSETS = tuple(
        (di, dj)
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if (di, dj) != (0, 0)
    )

    def __init__(self, cellXNum, cellYNum):
        # (i, j) -> dict(state, center, heat, smoke, corners)
        self.cellList = {}
        # 3D list containing sub-lists, that consists of a list of guidance candidate nodes at each cell (x_i, y_i)
        self.cellWiseGuidanceCan = None
        # 3D list containing sub-lists, that consists of a list of shelter candidate nodes at each cell (x_i, y_i)
        self.cellWiseShelterCan = None
        # 
        self.cellWiseGuidances = None

        self.cellWiseShelters = None
        
        self.cellXNum = int(cellXNum)
        self.cellYNum = int(cellYNum)
        if self.cellXNum <= 0 or self.cellYNum <= 0:
            raise ValueError("cellXNum and cellYNum must be positive integers")
        
        self.cellXLen = 0.0
        
        self.cellYLen = 0.0
        
        self.xEdges = None
        
        self.yEdges = None
        
        
        # [(northwest), (northeast), (southwest), (southeast)]
        self.cutBorders = None
        
        """Per-cell flattened features that RL needs"""
        self.countByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.avgVelocityByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.heatByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.smokeByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.dangerLevelByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        # active pedestrians / remaining shelter capacity in the cell
        self.shelterFulfillByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.shelterPressureByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.guidanceInterByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        self.wellnessPenaltyByCell = np.zeros(self.cellXNum * self.cellYNum, dtype=float)
        
        self.last_snapshot = None

    @property
    def grid_shape(self):
        return self.cellXNum, self.cellYNum

    @property
    def number_of_cells(self):
        return self.cellXNum * self.cellYNum

    def cell_ids(self):
        """Yield cells in the sole flattened-array order used by the model."""
        for index in range(self.number_of_cells):
            yield self.unflatten_cell(index)

    def _cell_id(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        if not isinstance(cellID, tuple) or len(cellID) != 2:
            raise ValueError("cellID must be an (x, y) pair")
        cell = (int(cellID[0]), int(cellID[1]))
        if not (
            0 <= cell[0] < self.cellXNum
            and 0 <= cell[1] < self.cellYNum
        ):
            raise ValueError(f"Cell {cell} is outside grid {self.grid_shape}")
        return cell

    def flatten_cell(self, cellID):
        i, j = self._cell_id(cellID)
        return i * self.cellYNum + j

    def unflatten_cell(self, cell_index):
        index = int(cell_index)
        if not 0 <= index < self.number_of_cells:
            raise ValueError(
                f"Cell index {index} is outside grid {self.grid_shape}"
            )
        return divmod(index, self.cellYNum)

    def assert_grid_coherent(self, *, require_initialized=True):
        """Validate every cell-indexed container against this exact shape."""
        expected_cells = {
            (i, j)
            for i in range(self.cellXNum)
            for j in range(self.cellYNum)
        }
        if require_initialized and set(self.cellList) != expected_cells:
            raise RuntimeError(
                "CellTracker cellList does not match its declared grid shape "
                f"{self.grid_shape}"
            )
        if require_initialized:
            for cell_id, cell in self.cellList.items():
                stored_id = tuple(getattr(cell, "cellID", cell_id))
                if stored_id != cell_id:
                    raise RuntimeError(
                        "CellTracker cell identity disagrees with its grid key: "
                        f"key={cell_id}, cellID={stored_id}"
                    )
        for axis, edges, size in (
            ("x", self.xEdges, self.cellXNum),
            ("y", self.yEdges, self.cellYNum),
        ):
            if require_initialized and (
                edges is None or np.asarray(edges).shape != (size + 1,)
            ):
                raise RuntimeError(
                    f"CellTracker {axis}Edges do not match grid shape {self.grid_shape}"
                )
        if require_initialized and (
            self.cutBorders is None
            or len(self.cutBorders) != self.cellXNum
            or any(len(column) != self.cellYNum for column in self.cutBorders)
        ):
            raise RuntimeError(
                "CellTracker cutBorders do not match grid shape "
                f"{self.grid_shape}"
            )
        for name in (
            "countByCell",
            "avgVelocityByCell",
            "heatByCell",
            "smokeByCell",
            "dangerLevelByCell",
            "shelterFulfillByCell",
            "shelterPressureByCell",
            "guidanceInterByCell",
            "wellnessPenaltyByCell",
        ):
            values = np.asarray(getattr(self, name, ())).reshape(-1)
            if values.size != self.number_of_cells:
                raise RuntimeError(
                    f"CellTracker {name} has {values.size} entries; "
                    f"grid {self.grid_shape} requires {self.number_of_cells}"
                )
        for name in (
            "cellWiseGuidanceCan",
            "cellWiseShelterCan",
            "cellWiseGuidances",
            "cellWiseShelters",
        ):
            grid = getattr(self, name, None)
            if grid is None:
                if require_initialized:
                    raise RuntimeError(f"CellTracker {name} is not initialized")
                continue
            if len(grid) != self.cellXNum or any(
                len(column) != self.cellYNum for column in grid
            ):
                raise RuntimeError(
                    f"CellTracker {name} does not match grid shape {self.grid_shape}"
                )
        if self.last_snapshot is not None and (
            len(self.last_snapshot) != self.cellXNum
            or any(len(column) != self.cellYNum for column in self.last_snapshot)
        ):
            raise RuntimeError(
                "CellTracker last_snapshot does not match grid shape "
                f"{self.grid_shape}"
            )
        return self.grid_shape
    
    """Primary functions"""
    def getCut(self):
        return self.cellXLen, self.cellYLen
    
    def initialCut(self, networkXLength, networkYLength, xEdges = None, yEdges = None):
        
        # check if necessary preparation is made for this function
        if self.cellXNum <= 0 or self.cellYNum <= 0:
            raise ValueError("cellXNum and cellYNum must be positive integers")
        
        def validated_edges(values, expected, axis_name):
            if len(values) != expected + 1:
                raise ValueError(
                    f"{axis_name}Edges must contain exactly {expected + 1} boundaries"
                )
            result = np.asarray(values, dtype=float)
            if not np.isfinite(result).all() or not np.all(np.diff(result) > 0.0):
                raise ValueError(
                    f"{axis_name}Edges must be finite and strictly increasing"
                )
            return [float(value) for value in result]

        # Step 1: Cut both projected-metre axes.  Explicit boundaries may be
        # equal-area or density-adaptive; CellTracker is deliberately agnostic.
        if xEdges is not None:
            self.xEdges = validated_edges(xEdges, self.cellXNum, "x")
            self.cellXLen = float(networkXLength) / self.cellXNum
        else:
            self.cellXLen = float(networkXLength) / self.cellXNum
            self.xEdges = [i * self.cellXLen for i in range(self.cellXNum)] + [float(networkXLength)]
            
        if yEdges is not None:
            self.yEdges = validated_edges(yEdges, self.cellYNum, "y")
            self.cellYLen = float(networkYLength) / self.cellYNum
        else:
            self.cellYLen = float(networkYLength) / self.cellYNum
            self.yEdges = [j * self.cellYLen for j in range(self.cellYNum)] + [float(networkYLength)]
        
        # ``initialCut`` is the episode boundary for every cell-indexed
        # container.  Clear any earlier realization before rebuilding the
        # configured shape so an accidental tracker reuse cannot retain stale
        # cellular state.
        self.cellList.clear()
        self.last_snapshot = None
        cutBorders = [[None for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        
        # Step 2: (Double for loop) Create a 2D list that contains the [northwest, northeast, southwest, southeast] coordinates of each cell, identifiable via cell\_dim[i][j] for $c_{ij}$
        
        for i in range(self.cellXNum):
            XLeft = self.xEdges[i]
            XRight = self.xEdges[i + 1]
            
            for j in range(self.cellYNum):
                YDown = self.yEdges[j]
                YUp = self.yEdges[j + 1]
                
                northWest = (XLeft, YDown)
                northEast = (XRight, YDown)
                southWest = (XLeft, YUp)
                southEast = (XRight, YUp)
                
                cutBorders[i][j] = [northWest, northEast, southWest, southEast]
                
                XCenter = 0.5 * (XLeft + XRight)
                YCenter = 0.5 * (YDown + YUp)
                
                newCell = Cell(
                        cellID = (i, j),
                        impactedLevel = 0,
                        shelterList = [],
                        guidanceList = [],
                        shCanList = [],
                        guCanList = [],
                        corners = (northWest, northEast, southWest, southEast),
                        center = (XCenter, YCenter)
                )
                
                self.cellList[(i, j)] = newCell
                
        self.cutBorders = cutBorders
        
        self.cellWiseGuidanceCan = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        self.cellWiseShelterCan  = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        self.cellWiseGuidances   = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        self.cellWiseShelters    = [[[] for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        for name in (
            "countByCell",
            "avgVelocityByCell",
            "heatByCell",
            "smokeByCell",
            "dangerLevelByCell",
            "shelterFulfillByCell",
            "shelterPressureByCell",
            "guidanceInterByCell",
            "wellnessPenaltyByCell",
        ):
            setattr(self, name, np.zeros(self.number_of_cells, dtype=float))
        self.assert_grid_coherent(require_initialized=True)
        return cutBorders
    
    """
    def cellUpdate(self):
        # Step 1: Extract the cell's impact level
        
        # Step 2: For all pedestrian groups that are currently located in $ce_i$, aggregate their volume
        
        # Step 3: Iterate through active shelters in $ce_i$'s shelters list, aggregate their current flow and capacity, and compute the fulfillment ratio (total flow/total capacity)
        
        # Step 4: Iterate through active guidance points in $ce_i$'s guidance points list, aggregate their current flows
        
        # Step 5: Extract the current speed, then extract the combined forces by calling SelfForceUpdate and ImpactForceUpdate functions, and compute the new speed with the given equation
        
        # Step 6: Concat the computed parameter values as a set, add to the 2D array as an element
        
        return 0
    """
    def cellUpdate(self,
                   heat_range = (0.0, 80.0),
                   smoke_range = (0.0, 300.0),
                   heat_weight = 0.6,
                   neighbor_push = 0.4,
                   max_state = 5,
                   decay_if_safe = True, 
                   pedDS = None,
                   forceTracker = None, 
                   save_snapshot = True
            ):
        
        snapshot = [[None for _ in range(self.cellYNum)] for _ in range(self.cellXNum)]
        
        """Helper functions"""
        def flatten(key, default = 0.0):
            out = np.zeros(total, dtype = float)
            
            k = 0
            for i in range(Nx):
                for j in range(Ny):
                    v = snapshot[i][j].get(key, default)
                    if v is None: v = default
                    out[k] = float(v)
                    k += 1
                    
            return out
        
        def _norm(arr, lo, hi):
            if hi <= lo:
                return np.zeros_like(arr)
            
            z = (arr - lo) / (hi - lo)
            return np.clip(z, 0.0, 1.0)
        
        # function preparation, dimension and container for the state of the overall environment
        Nx, Ny = self.cellXNum, self.cellYNum
        total = Nx * Ny
        snapshot = [[None for _ in range(Ny)] for _ in range(Nx)]

        # Index active agents once per timestep. The previous implementation
        # rescanned the entire population separately for every cell, making the
        # same calculation O(number_of_cells * population). This preserves
        # agent order and numerical behavior while reducing it to O(population).
        agents_by_cell = [[[] for _ in range(Ny)] for _ in range(Nx)]
        if pedDS is not None:
            for ped in pedDS.pedAgentList.values():
                if getattr(ped, "terminated", False):
                    continue
                cell = getattr(ped, "currCell", None)
                if cell is None:
                    continue
                i, j = int(cell[0]), int(cell[1])
                if 0 <= i < Nx and 0 <= j < Ny:
                    agents_by_cell[i][j].append(ped)
        
        # containers for per-cell raw data
        heat_raw   = np.zeros(total, dtype=float)
        smoke_raw  = np.zeros(total, dtype=float)
        count_arr  = np.zeros(total, dtype=float)
        speed_arr  = np.zeros(total, dtype=float)
        shFulfill  = np.zeros(total, dtype=float)
        shPressure = np.zeros(total, dtype=float)
        guFlow     = np.zeros(total, dtype=float)
        danger_arr = np.zeros(total, dtype=float)
        
        k_flat = 0
        
        for i in range(self.cellXNum):
            for j in range(self.cellYNum):
                cell = self.cellList[(i, j)]
                
                # cell state level (severity of hazarous impact)
                stateLevel = cell.getState() if hasattr(cell, "getState") else getattr(cell, "impactedLevel", 0)
                
                # group and volume of pedestrians in the currently reviewing cell
                agents = agents_by_cell[i][j]
                ped_volume = sum(
                    max(1, int(getattr(ped, "group_size", 1))) for ped in agents
                )
                
                # get shelter flow and total shelter capacity in the current cell
                sh_flow = 0.0
                sh_cap = 0.0
                
                for sh in cell.shelterList:
                    sh_flow += float(getattr(sh, "shelterFlow", 0.0))
                    sh_cap += float(getattr(sh, "shelterCap", 0.0))

                sh_fulfill = (sh_flow / sh_cap) if sh_cap > 0 else 0.0
                sh_remain = max(0.0, sh_cap - sh_flow)
                sh_pressure = (float(ped_volume) / max(1.0, sh_remain))
                
                # get guidance flow in this cell
                gu_flow = 0.0
                for gu in cell.guidanceList:
                    gu_flow += float(getattr(gu, "guidedFlow", 0.0))
                
                # average speed before and after apply the computed, combined force to the cell
                avg_speed_before = None
                avg_speed_after = None
                
                if ped_volume > 0:
                    speed_mass = sum(
                        float(getattr(ped, "currSpeed", 0.0))
                        * max(1, int(getattr(ped, "group_size", 1)))
                        for ped in agents
                    )
                    avg_speed_before = speed_mass / float(ped_volume)
                    
                    # Hazard effects have already been applied before movement.
                    # Cell aggregation is observational and must not mutate
                    # speed after the transition merely for logging/RL state.
                    avg_speed_after = avg_speed_before
                
                    # document per-cell snapshot/state for analysis
                snapshot[i][j] = {
                    "state": int(stateLevel),
                    "ped_volume": int(ped_volume),
                    "shelter_flow": float(sh_flow),
                    "shelter_cap": float(sh_cap),
                    "shelter_fulfillment": float(sh_fulfill),
                    "guidance_flow": float(gu_flow),
                    "avg_speed_before": (None if avg_speed_before is None else float(avg_speed_before)),
                    "avg_speed_after": (None if avg_speed_after is None else float(avg_speed_after))
                }
        
                heat_val = float(getattr(cell, "heat", 0.0))
                smoke_val = float(getattr(cell, "smoke", 0.0))
                
                heat_raw[k_flat] = heat_val
                smoke_raw[k_flat] = smoke_val
                count_arr[k_flat] = float(ped_volume)
                shFulfill[k_flat] = float(sh_fulfill)
                shPressure[k_flat] = float(sh_pressure)
                guFlow[k_flat] = float(gu_flow)
                
                # pick speed_after if valid else before, else 0
                if avg_speed_after is not None:
                    speed_arr[k_flat] = float(avg_speed_after)
                elif avg_speed_before is not None:
                    speed_arr[k_flat] = float(avg_speed_before)
                else:
                    speed_arr[k_flat] = 0.0
                    
                danger_arr[k_flat] = float(stateLevel)
                
                k_flat += 1
        
        heat_norm = _norm(heat_raw, *heat_range)
        smoke_norm = _norm(smoke_raw, *smoke_range)
        
        # The decision process uses the documented five-level hazard state as
        # its one normalized danger variable. This keeps D_i interpretable and
        # guarantees the [0, 1] bound used by the reward-dominance proof.
        danger_combo = np.clip(danger_arr / float(max(1, int(max_state))), 0.0, 1.0)
        
        wellness_penalty = danger_combo.copy()
        
        # write each parameter into its corresponding container
        self.heatByCell = heat_norm
        self.smokeByCell = smoke_norm
        self.dangerLevelByCell = danger_combo
        self.countByCell = count_arr
        self.avgVelocityByCell = speed_arr
        self.shelterFulfillByCell = shFulfill
        self.shelterPressureByCell = shPressure
        self.guidanceInterByCell = guFlow
        self.wellnessPenaltyByCell = wellness_penalty
        
        # save for logging purpose
        if save_snapshot:
            self.last_snapshot = snapshot
        
        self.assert_grid_coherent(require_initialized=True)
        return snapshot
    
    def update_all_cells(self, *args, **kwargs):
        return self.cellUpdate(*args, **kwargs)
                
    """This function receives a pair of (X, Y) coordinate and return the ID of the cell this point is in"""
    def locateCell(self, XPos, YPos):
        
        # check if functions are called in orders, check for prerequisite functions
        if self.cellXLen == 0 or self.cellYLen == 0.0:
            raise RuntimeError("Call initialCut() before locateCell().")
        
        x_val = float(XPos)
        y_val = float(YPos)
        
        if self.xEdges is not None and len(self.xEdges) == self.cellXNum + 1:
            XID = int(np.searchsorted(self.xEdges, x_val, side = "right") - 1)
        else:
            XID = int(x_val // self.cellXLen)
            
        if self.yEdges is not None and len(self.yEdges) == self.cellYNum + 1:
            YID = int(np.searchsorted(self.yEdges, y_val, side = "right") - 1)
        else:
            YID = int(y_val // self.cellYLen)
        
        # Cap values
        if XID < 0: XID = 0
        if YID < 0: YID = 0
        if XID >= self.cellXNum: XID = self.cellXNum - 1
        if YID >= self.cellYNum: YID = self.cellYNum - 1
        
        return [XID, YID]
    
    """return the 8-direction neighbors of given cell ID (i, j) pair"""
    def getNeighborCells(self, cellID):
        XID, YID = self._cell_id(cellID)
        neighbors = []
        for deltaI, deltaJ in self.MOORE_NEIGHBOR_OFFSETS:
            ni, nj = XID + deltaI, YID + deltaJ
            if 0 <= ni < self.cellXNum and 0 <= nj < self.cellYNum:
                neighbors.append((ni, nj))
        
        return neighbors

    def state_vector(self, *, max_state=5, normalized=False):
        """Return CA states in the same flattened order used by decisions."""
        values = np.asarray(
            [self.getCellState(cell_id) for cell_id in self.cell_ids()],
            dtype=np.float64,
        )
        if normalized:
            values = np.clip(values / float(max(1, int(max_state))), 0.0, 1.0)
        return values
    
    def getCellState(self, cellID):
        cellID = self._cell_id(cellID)
        cell = self.cellList[cellID]
        return cell.getState() if hasattr(cell, "getState") else getattr(cell, "impactedLevel", 0)
    
    def setCellState(self, cellID, newState):
        cellID = self._cell_id(cellID)
        cell = self.cellList[cellID]
        if hasattr(cell, "setState"):
            cell.setState(newState)
        else:
            cell.impactedLevel = newState
    
    def getCellCenter(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return self.cellList[cellID].getCenter()
    
    """the following two functions are called by HazardDatabase"""
    def setHeat(self, cellID, value):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        self.cellList[cellID].setHeat(value)
    
    def setSmoke(self, cellID, value):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        self.cellList[cellID].setSmoke(value)
        
    def getHeat(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return float(getattr(self.cellList[cellID], "heat", 0.0))

    def getSmoke(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return float(getattr(self.cellList[cellID], "smoke", 0.0))
    
    
    """Helper functions"""
    def getCell(self, i, j) -> Cell:
        return self.cellList[(i, j)]
    
    def addGuidanceCan(self, cellID, candidate):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].guCanList.append(candidate)
        if self.cellWiseGuidanceCan is not None:
            self.cellWiseGuidanceCan[i][j].append(candidate)
    
    def addShelterCan(self, cellID, candidate):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].shCanList.append(candidate)
        if self.cellWiseShelterCan is not None:
            self.cellWiseShelterCan[i][j].append(candidate)
    
    def addGuidance(self, cellID, guidanceObj):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].guidanceList.append(guidanceObj)
        if self.cellWiseGuidances is not None:
            self.cellWiseGuidances[i][j].append(guidanceObj)
    
    def addShelter(self, cellID, shelterObj):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        i, j = cellID
        self.cellList[cellID].shelterList.append(shelterObj)
        if self.cellWiseShelters is not None:
            self.cellWiseShelters[i][j].append(shelterObj)
    
    def getCellCorners(self, cellID):
        if isinstance(cellID, list):
            cellID = tuple(cellID)
        return self.cellList[cellID].corners
