#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

7/30: constructed the skeletons
"""

from Guidance import Guidance

class GuidanceDS:
    def __init__(self, candidateVol, initVol):
        self.guidanceList = {}
        self.guidanceByOSMID = {}
        
        self.guidanceByCell = None
        self.guidanceCanByCell = None
        
        self.guidanceImplementedByTime = {}
        
        self.candidateVol = int(candidateVol)
        
        self.initVol = int(initVol)
        
        self.guidanceCanList = {}
        
        self.nextID = 0
        
        self.pointPerCellList = None
        
    """Helper functions"""
    # sample the ID for the next implemented guidance point
    def allocID(self):
        gid = self.nextID
        self.nextID += 1
        return gid
    
    @staticmethod
    def ensureGrid(xn: int, yn: int):
        return [[[] for _ in range(int(yn))] for _ in range(int(xn))]
    
    """Primary functions"""    
    """
    When a pedestrian meets a guidance point:
        1) Increment guidance flow
        2) Redirect the pedestrian to the nearest open shelter
    """
    
    def updateGuidePointFlow(self, pedAgent, guPointAt, mapDS, shelterDS):
        if guPointAt is None:
            return 0
        
        if hasattr(guPointAt, "guideFlow"):
            guPointAt.guideFlow += 1
        elif hasattr(guPointAt, "flow"):
            guPointAt.flow += 1
            
        if (mapDS is not None) and (shelterDS is not None) and hasattr(pedAgent, "currNode"):
            targetShelter = shelterDS.locateClosestShelter(
                pedAgent.currNode.nodeX, 
                pedAgent.currNode.nodeY, 
                open_only = True,
            )
            
            if targetShelter is not None:
                try:
                    newRoute = mapDS.assignEvacuationRoute(guPointAt, targetShelter)
                    if newRoute is not None:
                        pedAgent.routeFollowing = newRoute
                        pedAgent.guided = True
                except Exception:
                    pass
        
        return 1
        
        # Step 1: Receive the interaction information and the guidance point it encounters. 
        
        # Step 2: Add the flow to the respective guidance point entity.
        
        # Step 3: Assign the pedestrian with the route to the currently desired shelter.
    
    """
    Deploy a new guidance point from the candidate list of the specified cell
    action: dict with either:
        - {"cell": (i, j)} OR
        - {"x": X, "y": Y} plus cellTracker to convert to (i, j)
    """
    def newPoint(self, action, cellTracker):
        if self.guidanceCanByCell is None or self.guidanceByCell is None:
            return None
        
        # when the action is passed back from RL model, extract the (x-axis cellID, y-axis cellID) the next guidance point will be implemented in
        
        # competency and bound check
        if isinstance(action, int):
            nx = len(self.guidanceCanByCell)
            ny = len(self.guidanceCanByCell[0]) if nx > 0 else 0
            if nx * ny == 0 or action < 0 or action >= nx * ny:
                return None
            cellX, cellY = divmod(int(action), ny)
            
        elif isinstance(action, dict) and "cell" in action:
            cellX, cellY = action["cell"]
        else:
            return None
        
        
        if not self.guidanceCanByCell[cellX][cellY]:
            return None
        
        # guidance candidates are sorted by decreasing # of path connections
        node = self.guidanceCanByCell[cellX][cellY].pop(0)
        
        gid = self.allocID()
        
        gu = Guidance(
            guidanceID = gid,
            nodeMapped = node,
            cellLocated = (cellX, cellY),
            shelterPointer = None,
            totalFlow = 0,
            guidedFlow = 0,
            guStatus = 0
        )
        
        self.guidanceList[gid] = gu
        self.guidanceByOSMID[node.OSMID] = gu
        self.guidanceByCell[cellX][cellY].append(gu)
        
        return gid
        # Step 1: Receive the action package from the GuidanceActor.
        
        # Step 2: Locate the correct cell and the guidance sub-list by cell's X and Y ID, 
        # extract and remove the first guidance point candidate from the list.
        
        # Step 3: Prepare and declare the new guidance point entity and add to the active guidance point list.
    
    """
    Set/compute pointer from a guidance point (or any node) to the nearest open shelter
    Returns a Route or None
    """
    def updateShelterPointer(self, mapDS, shelterDS, sourceNode):
        if sourceNode is None:
            return None
        
        targetShelter = shelterDS.locateClosestShelter(sourceNode.nodeX, sourceNode.nodeY, openOnly = True)
        if targetShelter is None:
            return None
        try:
            return mapDS.shortestPath(sourceNode, targetShelter.nodeMapped)
        except Exception:
            return None
        # Step 1: Compare Euclidean distance with all functional shelters, get the nearest shelter. 
        
        # Step 2: Call the PathFinder to compute the nearest route 
        
    """
    Build 2D arrays of candidates and active guidance points per cell
    Requires self.guidanceCanList to be filled with Node objects
    """
    def pointPerCell(self, cellTracker, cellX, cellY):
        
        self.guidanceByCell = self.ensureGrid(cellX, cellY)
        self.guidanceCanByCell = self.ensureGrid(cellX, cellY)
        
        for node in self.guidanceCanList.values():
            ci, cj = cellTracker.locateCell(node.nodeX, node.nodeY)
            if 0 <= ci < cellX and 0 <= cj < cellY:
                self.guidanceCanByCell[ci][cj].append(node)
        
        for i in range(cellX):
            for j in range(cellY):
                self.guidanceCanByCell[i][j].sort(key = lambda n: float(getattr(n, "nodeCap", 0.0)), reverse = True)
        # Step 1: Call CellularAutomata's getCut to receive the standard X and Y length of cells for the given community or city.
        
        # Step 2: Create an empty 3Darray where each element is a sub-list of guidance points, sorted by cell ID
        
        # Step 3: Review each candidate guidance point from the list received from MapDatabase, and assign each candidate intersection to a unique sub-list
        
        # Step 4: Among each subset, rank candidates in each subset based on decreasing number of connected streets. 
        
    # Deploy initial guidance points from candidates
    def initGuidance(self):
        if self.guidanceCanByCell is None or self.guidanceByCell is None:
            return 0
        
        XLen = len(self.guidanceCanByCell)
        YLen = len(self.guidanceCanByCell[0]) if XLen > 0 else 0
        
        if XLen == 0 or YLen == 0:
            return 0
        
        deployed = 0
        i = j = 0
        while deployed < self.initVol: 
            found = False
            
            # implement one initial guidance point in each cell
            for _ in range(XLen * YLen):
                if self.guidanceCanByCell[i][j]:
                    node = self.guidanceCanByCell[i][j].pop(0)
                    gid = self.allocID()
                    gu = Guidance(
                        guidanceID = gid, 
                        nodeMapped = node,
                        cellLocated = (i, j),
                        shelterPointer = None,
                        totalFlow = 0,
                        guidedFlow = 0,
                        guStatus = 0
                    )
                    
                    
                    self.guidanceList[gid] = gu
                    self.guidanceByCell[i][j].append(gu)
                    self.guidanceByOSMID[node.OSMID] = gu
                    
                    deployed += 1
                    found = True
                    break
                
                j += 1
                if j >= YLen:
                    j = 0
                    i += 1
                    if i >= XLen:
                        i = 0
        
            if not found:
                break
        return deployed
        