#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

7/30: constructed the skeletons

"""

from Edge import Edge
from Node import Node
from Route import Route

from ShelterTypes import canonical_shelter_site_type
import math
import numpy as np

from collections import defaultdict
from NetworkOptimization import routing_tree_to_target


class MapDS:
    def __init__(
        self,
        nodeList,
        edgeList,
        address,
        locationDrive,
        *,
        effective_walkway_width_m=3.0,
        jam_density_ped_per_m2=5.4,
    ):
        # House the node list directly extracted from OpenStreetMap
        self.rawNodeList = nodeList
        # House the edge list directly extracted from OpenStreetMap
        self.rawEdgeList = edgeList
        # 
        self.intersectionList = {}
        
        self.buildingList = {}
        
        self.shelterList = {}
        
        self.shelterCanList = {}
        
        self.guidanceList = {}
        
        self.guidanceCanList = {}
        
        self.address = address
        
        self.nodeListByOSMID = {}
        
        self.nodeListByLocalID = {}
        
        self.edgeListByOSMID = {}
        
        self.edgeListByLocalID = {}
        
        self.boundCoord = []
        
        self.boundMeters = []
        
        self.convertUnitX = None
        self.convertUnitY = None
        
        self.nodeCapSum = 0
        
        self.locationDrive = locationDrive

        self.effectiveWalkwayWidthM = float(effective_walkway_width_m)
        self.jamDensityPedPerM2 = float(jam_density_ped_per_m2)
        if not math.isfinite(self.effectiveWalkwayWidthM) or self.effectiveWalkwayWidthM <= 0.0:
            raise ValueError("effective_walkway_width_m must be finite and positive")
        if not math.isfinite(self.jamDensityPedPerM2) or self.jamDensityPedPerM2 <= 0.0:
            raise ValueError("jam_density_ped_per_m2 must be finite and positive")
        
        self.idx_uv_to_edges = defaultdict(list)
        self.idx_uv_best = {}
        self.incident_edges_by_osmid = defaultdict(list)
        self._node_pool = np.array([], dtype = object)
        self._shortest_path_osmid_cache = {}
        self._distance_to_target_cache = {}
        self._routing_tree_to_target_cache = {}
        
    """Getter functions"""
    
    def getNodeListByOSMID(self):
        return self.nodeListByOSMID
    
    def getNodeListByLocalID(self):
        return self.nodeListByLocalID
    
    def getEdgeListByOSMID(self):
        return self.edgeListByOSMID
    
    def getEdgeListByLocalID(self):
        return self.edgeListByLocalID
    
    def getIntersectionList(self):
        return self.intersectionList
    
    def getShelterList(self):
        return self.shelterList
    
    def getGuidanceList(self):
        return self.guidanceList
    
    """Main functions"""
    
    def boundarySetter(self):
        """
        This function iteratively reads in the X and Y coordinates separately, 
        then find the min and maxß X and Y, respectively, as 4-direction borders
        """

        xs, ys = [], []
        for n in self.rawNodeList:
            xs.append(float(n[1]['x']))
            ys.append(float(n[1]['y']))
            
        west = min(xs); east = max(xs)
        south = min(ys); north = max(ys)
        
        self.boundCoord = [east, west, south, north]
        
        if (self.convertUnitX is not None) and (self.convertUnitY is not None):
            self.boundMeters = [
                east  * self.convertUnitX,
                west  * self.convertUnitX,
                south * self.convertUnitY,
                north * self.convertUnitY,
            ]
    
    """Called at t = 0 once to set up convert units depends on the target environment's location in the Globe"""
    """Don't have to compute this at each node's initialization since target network typically don't span over many latitudes or longitude"""
    def computeConvertUnit(self):
        """Try find a more continuous conversion scheme for this function to resolve the "drift" issue when visualize"""
        if not self.rawNodeList:
            raise RuntimeError("rawNodeList is empty")

        lat = float(self.rawNodeList[0][1]['y'])
        # meters per degree latitude ~ constant
        self.convertUnitY = 111_132.0
        # meters per degree longitude scales by cos(lat)
        self.convertUnitX = 111_320.0 * math.cos(math.radians(lat))
    
    """This function is called at each node's initialization to convert the node's coord into meters"""
    def coordToMeters(self, nodeXCoord, nodeYCoord):
        if self.convertUnitX is None or self.convertUnitY is None:
            self.computeConvertUnit()
        if not self.boundCoord:
            self.boundarySetter()
            
        west = self.boundCoord[1]
        north = self.boundCoord[3]
        
        x_m = (nodeXCoord - west) * self.convertUnitX
        y_m = (north - nodeYCoord) * self.convertUnitY
        
        return x_m, y_m
    
    """Get a copy of list of buildings as a .txt file, for analysis purpose"""
    def writeBuildingList(self, buildings_gdf, path = "building.txt"):
        if buildings_gdf is None or len(buildings_gdf) == 0:
            return
        cols = [c for c in ['amenity', 'building', 'name'] if c in buildings_gdf.columns]
        out = buildings_gdf[cols].copy() if cols else buildings_gdf.copy()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(out.to_string())
    
    """This function receives the node list from OSMProcessor and create node instances in the network"""
    def nodeInit(self, cellTracker, min_street = 5):
        
        """
        Build local Nodes, compute capacity, and add to corresponding candidate list if meet conditions
        """
        
        """Need to print out the set of all building types to see if any other types should be added"""
        if self.convertUnitX is None or self.convertUnitY is None:
            self.computeConvertUnit()
        if not self.boundCoord:
            self.boundarySetter()
        
        localNodeID = 0
        cellID = []
        
        for node in self.rawNodeList:
            # Step 1: Extract basic info from OpenStreetMap's node dataset
            osmid = int(node[0])
            nodeXCoord = float(node[1]["x"])
            nodeYCoord = float(node[1]["y"])
            x_m, y_m = self.coordToMeters(nodeXCoord, nodeYCoord)
            
            buildingType = node[1].get('building_type', None)
            amenityType = node[1].get('amenity_type', None)
            
            siteType = canonical_shelter_site_type(buildingType, amenityType)
            
            # Step 2: assign unique local ID (+1 after declare each node)
            localID = localNodeID 
            localNodeID += 1
            
            nodeFlow = 0
            
            cellID = cellTracker.locateCell(x_m, y_m)
            
            # Step 3: Set Capacity by calling computeNodeCapacity
            nodeCap = self.computeNodeCapacity(siteType)
            
            # Step 4: Declare Node entity
            newNode = Node(localID, osmid, x_m, y_m, nodeCap, nodeFlow, cellID, siteType)

            self.nodeListByOSMID[osmid]  = newNode
            self.nodeListByLocalID[localID] = newNode
            
            # Guidance is deliberately deprecated.  Intersections remain part
            # of the road topology, but they are never exposed as candidates.

            # Step 5: conditionally put the node in ShelterList
            if siteType is not None:
                self.shelterCanList[localID] = newNode
         
        self._node_pool = np.array(list(self.nodeListByLocalID.values()), dtype = object)
            
            
    """This function receives the edge list from OSMProcessor and create node instances in the network"""
    def edgeInit(self, cellTracker):
        
        localEdgeID = 0
        startNode = None
        endNode = None
        OSMID = 0
        
        for edge in self.rawEdgeList: 
            # Step 1: Extract the start node ID, end node ID, edge length, and OSMID, start and end nodes
            startNodeID = int(edge[0])
            endNodeID = int(edge[1])
            
            startNode = self.nodeListByOSMID.get(startNodeID)
            endNode = self.nodeListByOSMID.get(endNodeID)
            
            # skip dangling edges
            if startNode is None or endNode is None:
                continue
            
            raw_osmid = edge[2].get("osmid")
            if isinstance(raw_osmid, list) and raw_osmid:
                raw_osmid = raw_osmid[0]
            try:
                OSMID = int(raw_osmid)
            except (TypeError, ValueError):
                # OSMnx may create short connector edges while rebuilding a
                # consolidated intersection. They have no source way id, so a
                # stable negative local id is used only as an internal key.
                OSMID = -(localEdgeID + 1)
            
            edgeLength = float(edge[2].get("length", 0.0))
            
            # Storage capacity is the modeled jam occupancy of the physical
            # walkway area. It replaces the former constant placeholder 100.
            edgeCap = self.computeEdgeCapacity(edgeLength)
            
            # Step 2: Set flow = 0 and local edge ID, set capacity by calling computeEdgeCapacity
            edgeFlow = 0
            localID = localEdgeID
            localEdgeID += 1
            
            # Cell ID is defined by startNode
            cellID = startNode.cellID
            
            newEdge = Edge(startNode, endNode, localID, OSMID, edgeLength, edgeCap, edgeFlow, cellID)
            
            self.edgeListByOSMID[OSMID] = newEdge
            
            self.edgeListByLocalID[localID] = newEdge
        
    def computeNodeCapacity(self, buildingType):
        # Step 1: capacity by building type list
        
        buildingCapacityKey = {
            "house": 5, "detached": 5, "semidetached_house": 8, "terrace": 10,
            "residential": 60, "apartments": 150, "dormitory": 400,
            "hut": 5, "bungalow": 6, "static_caravan": 4, "hotel": 200, "hostel": 80,

            # Education / research
            "school": 800, "college": 1200, "university": 2000, "kindergarten": 120, "research_institute": 300,

            # Healthcare
            "hospital": 600, "clinic": 40, "nursing_home": 150, "doctors": 30,

            # Civic / public
            "public": 300, "civic": 300, "townhall": 300, "courthouse": 300,
            "fire_station": 60, "police": 60, "library": 300, "community_centre": 500,

            # Worship (some mappers put function in building=*)
            "church": 250, "cathedral": 1500, "mosque": 400, "temple": 200, "synagogue": 200,

            # Commercial / retail / office
            "retail": 80, "commercial": 200, "office": 200, "supermarket": 200, "mall": 1000, "marketplace": 300,

            # Leisure / sports / event
            "theatre": 400, "sports_hall": 500, "stadium": 10000,

            # Industrial / storage
            "industrial": 100, "warehouse": 80, "hangar": 100, "barn": 0,
            
            # Transport / parking
            "train_station": 1000, "transportation": 400, "terminal": 500, "parking": 300, "garage": 30, "garages": 60,

            # Misc
            "shed": 0, "greenhouse": 0, "toilets": 0, "bridge": 0, "ruins": 0, "construction": 0,
            
            "shelter": 200, "place_of_worship": 250,
        }
        
        #Step 2: Return the associated capcity with matching
        if not buildingType:
            return 100
        return buildingCapacityKey.get(str(buildingType), 100)

    def computeEdgeCapacity(self, edgeLen):
        """Return physical-link storage at the configured jam density."""
        length = max(0.0, float(edgeLen))
        return float(
            length * self.effectiveWalkwayWidthM * self.jamDensityPedPerM2
        )
            
    def computeNodeCapSum(self):
        self.nodeCapSum = 0
        for _, node in self.nodeListByLocalID.items():
            self.nodeCapSum += float(node.nodeCap)
        return self.nodeCapSum
       
    def assignGenerationNode(self):
        # Step 1: (Randomly) sample a numerical placement
        
        # Step 2: Assign specific node through np.random.choice
        if self._node_pool.size == 0:
            self._node_pool = np.array(list(self.nodeListByLocalID.values()), dtype = object)
        chosenNode = np.random.choice(self._node_pool)
        #print("generation node is: ", chosenNode)
        return chosenNode
    
    def assignTerminationNode(self, startNode):
        # Step 1: Sum the capacity of all building and normalize (building's capacity over capacity sum)
        
        # Step 2: Assign specific node through np.random.choice
        if self._node_pool.size == 0:
            self._node_pool = np.array(list(self.nodeListByLocalID.values()), dtype = object)
        eligible = np.asarray(
            [node for node in self._node_pool if node is not startNode],
            dtype=object,
        )
        if eligible.size == 0:
            raise RuntimeError("A termination node requires at least two distinct network nodes")
        chosenNode = np.random.choice(eligible)
        
        #print("termination node is: ", chosenNode)
        return chosenNode

    def networkDistanceToTarget(self, startNode, terminationNode):
        """Return exact OSM-network distance using one reverse search per target.

        Shelter comparison repeatedly asks many pedestrian origins about the
        same small set of shelter destinations. Running a separate Dijkstra
        search for every origin is mathematically redundant. On a directed
        graph, one search from the destination on the reversed graph gives the
        same origin-to-destination distances for every origin.
        """
        if startNode is None or terminationNode is None:
            return float("inf")
        src = int(startNode.OSMID)
        dst = int(terminationNode.OSMID)
        if src == dst:
            return 0.0
        distances, _ = self._routing_tree(dst)
        return float(distances.get(src, float("inf")))

    def _routing_tree(self, target_osmid: int) -> tuple[dict, dict]:
        """Cache one exact reverse-search tree for every shelter destination."""
        dst = int(target_osmid)
        cached = self._routing_tree_to_target_cache.get(dst)
        if cached is None:
            if dst not in self.locationDrive:
                return {}, {}
            cached = routing_tree_to_target(
                self.locationDrive,
                dst,
                weight="length",
            )
            self._routing_tree_to_target_cache[dst] = cached
            self._distance_to_target_cache[dst] = cached[0]
        return cached
    
    """Helper function to look up edges based on node pairs (start, end)"""
    def buildEdgeIndices(self):
        self.idx_uv_to_edges.clear()
        self.idx_uv_best.clear()
        self.incident_edges_by_osmid.clear()
        
        for edge in self.edgeListByLocalID.values():
            u = int(edge.startNode.OSMID)
            v = int(edge.endNode.OSMID)
            
            self.idx_uv_to_edges[(u, v)].append(edge)
            self.incident_edges_by_osmid[u].append(edge)
            if v != u:
                self.incident_edges_by_osmid[v].append(edge)
            
            best = self.idx_uv_best.get((u, v))
            if (best is None) or (float(edge.edgeLen) < float(best.edgeLen)):
                self.idx_uv_best[(u, v)] = edge

        for osmid, edges in self.incident_edges_by_osmid.items():
            edges.sort(
                key=lambda edge: (
                    int(
                        edge.endNode.OSMID
                        if int(edge.startNode.OSMID) == int(osmid)
                        else edge.startNode.OSMID
                    ),
                    float(edge.edgeLen),
                    int(edge.edgeID),
                )
            )

    def incidentEdges(self, node):
        """Return deterministic incident road-edge choices for one node."""
        if node is None:
            return ()
        if not self.incident_edges_by_osmid:
            self.buildEdgeIndices()
        unique = []
        seen = set()
        for edge in self.incident_edges_by_osmid.get(int(node.OSMID), ()):
            low, high = sorted((int(edge.startNode.OSMID), int(edge.endNode.OSMID)))
            key = (low, high, str(edge.OSMID))
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return tuple(unique)
        
    """Still working and need check"""
    def shortestPath(self, startNode, terminationNode):
        """
        Use OSMnx routing to get a node-id path, then find the corresponding Node objects of the nodes in that list, 
        and the Edges connecting those Nodes, form a Route object and assign to the Ped Agent.
        """
        
        if startNode is None or terminationNode is None:
            return None
        
        src = int(startNode.OSMID)
        dst = int(terminationNode.OSMID)
        key = (src, dst)
        pathContainer = self._shortest_path_osmid_cache.get(key)
        if pathContainer is None:
            _, next_hop = self._routing_tree(dst)
            path = [src]
            visited = {src}
            while path[-1] != dst:
                successor = next_hop.get(path[-1])
                if successor is None or successor in visited:
                    path = []
                    break
                path.append(successor)
                visited.add(successor)
            pathContainer = tuple(path)
            if pathContainer:
                self._shortest_path_osmid_cache[key] = pathContainer
        if not pathContainer:
            # A route request must never silently substitute a different origin
            # or destination. Callers may explicitly sample a new pair if desired.
            return None

        routeNodes = []
        
        # Step 3: Map OSM node IDs to node objects in the existing list
        for nid in pathContainer:
            node = self.nodeListByOSMID.get(int(nid))
            if node is None:
                return None
                #raise KeyError(f"OSM node {nid} not found in nodeListByOSMID")
            routeNodes.append(node)
         
        # build edge indices if needed
        if not self.idx_uv_best:
            self.buildEdgeIndices()
            
        # Step 4: Map each (u,v) pair to your Edge object
        routeEdges = []
        for u, v in zip(pathContainer[:-1], pathContainer[1:]):
            u, v = int(u), int(v)
            # Fast path: use precomputed "best" edge
            edge_obj = self.idx_uv_best.get((u, v))
            if edge_obj is None:
                edge_obj = self.idx_uv_best.get((v, u))
            
            if edge_obj is None:
                return None
            routeEdges.append(edge_obj)
        
        # Step 5: Declare new route object and return
        
        newRoute = Route(
            startNode = startNode,
            endNode = terminationNode, 
            nodeRemained = routeNodes,
            edgeRemained = routeEdges
        )
        return newRoute

    def assignEvacuationRoute(self, guidance, assignedShelter):
        
        startPoint = guidance.nodeMapped
        destination = assignedShelter.nodeMapped
        
        return self.shortestPath(startPoint, destination)
