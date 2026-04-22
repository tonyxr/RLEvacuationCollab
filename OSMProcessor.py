#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

7/29: Created structure
8/1: Version 1.0 done, all functions implemented, pending testing
"""

import osmnx as OSM
import csv
from shapely import geometry
import os
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import copy
import hashlib
import json
import networkx as nx


class OSMProcessor:
    _location_cache = {}
    _network_feature_cache = {}
    _building_feature_cache = {}
    _default_overpass_endpoints = (
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://overpass.openstreetmap.ru/api",
    )

    def __init__(self, address, verbose: bool = False):
        
        self.locationDrive = None
        
        self.networkFeature = None
        
        self.nodeList = []
        
        self.edgeList = []
        
        self.address = str(address) if address else "State College, PA, USA"
        #self.address = str(address) if address else "State College, PA, USA"
        self.verbose = bool(verbose)
                
        self.interStreetCount = {}
        
        self.buildingNodes = {}
        
        self.intersectionNodes = {}
        
        self.tags = {'amenity':True, 'building':True, 'Assembly point':True, 'Office':True, 'Shop':True, 'Sport':True}
        
        self.mapStat = {}
        
        self.intersectionCount = 0
        
        self.G_proj = None
        
           
    def _graph_cache_path(self):
        key = hashlib.sha1(self.address.encode("utf-8")).hexdigest()[:12]
        safe_addr = "".join(ch if ch.isalnum() else "_" for ch in self.address).strip("_")
        safe_addr = safe_addr[:80] if safe_addr else "address"
        os.makedirs(OSM.settings.cache_folder, exist_ok = True)
        return os.path.join(OSM.settings.cache_folder, f"graph_walk_{safe_addr}_{key}.graphml")
    
    def _try_graph_from_place(self, overpass_url):
        old_overpass_url = OSM.settings.overpass_url
        old_requests_timeout = OSM.settings.requests_timeout
        try:
            OSM.settings.overpass_url = overpass_url
            # Keep responsiveness; each endpoint may still retry internally.
            OSM.settings.requests_timeout = min(int(old_requests_timeout), 90)
            G = OSM.graph.graph_from_place(
                self.address,
                network_type = "walk",
                truncate_by_edge = False,
            )
            return G
        finally:
            OSM.settings.overpass_url = old_overpass_url
            OSM.settings.requests_timeout = old_requests_timeout
    
    def _load_location_graph(self):
        graph_cache_path = self._graph_cache_path()
        if os.path.exists(graph_cache_path):
            if self.verbose:
                print(f"[OSM] Loading cached graphml: {graph_cache_path}")
            return OSM.io.load_graphml(graph_cache_path)
        
        errors = []
        for overpass_url in self._default_overpass_endpoints:
            try:
                if self.verbose:
                    print(f"[OSM] Attempting Overpass endpoint: {overpass_url}")
                G = self._try_graph_from_place(overpass_url)
                OSM.io.save_graphml(G, graph_cache_path)
                if self.verbose:
                    print(f"[OSM] Saved graphml cache: {graph_cache_path}")
                return G
            except Exception as exc:
                errors.append(f"{overpass_url} -> {type(exc).__name__}: {exc}")
                if self.verbose:
                    print(f"[OSM] Failed endpoint {overpass_url}: {exc}")
        
        if os.path.exists(graph_cache_path):
            # Defensive fallback in case write completed despite transient exception.
            return OSM.io.load_graphml(graph_cache_path)
        
        # Last-resort offline fallback: reconstruct a graph from raw cached Overpass
        # response JSON files created by prior successful runs in this repository.
        raw_cache_graph = self._load_graph_from_raw_overpass_cache()
        if raw_cache_graph is not None:
            OSM.io.save_graphml(raw_cache_graph, graph_cache_path)
            if self.verbose:
                print(f"[OSM] Reconstructed graph from raw cache and saved: {graph_cache_path}")
            return raw_cache_graph
        
        detail = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(
            f"Unable to fetch OSM road graph for '{self.address}'. "
            f"Tried endpoints: {', '.join(self._default_overpass_endpoints)}. "
            f"Errors: {detail}. "
            "If running in a restricted network, pre-warm cache by running once with internet access."
        )
    
    def _load_graph_from_raw_overpass_cache(self):
        cache_dir = OSM.settings.cache_folder
        if not os.path.isdir(cache_dir):
            return None
        
        response_jsons = []
        for name in sorted(os.listdir(cache_dir)):
            if not name.endswith(".json"):
                continue
            p = os.path.join(cache_dir, name)
            try:
                with open(p, "r", encoding = "utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "elements" in data:
                    response_jsons.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "elements" in item:
                            response_jsons.append(item)
            except Exception:
                continue
        
        if not response_jsons:
            return None
        
        try:
            G = OSM.graph._create_graph(response_jsons, bidirectional = True)
            if len(G.nodes) == 0:
                return None
            return G
        except Exception:
            return None


    """Getter Functions"""
    
    def getNodeList(self):
        return self.nodeList
        
    def getEdgeList(self):
        return self.edgeList
    
    def getBuildingNodes(self):
        return self.buildingNodes
    
    def getInterStreetCount(self):
        return self.interStreetCount
    
    """Main Functions"""
    
    """Extracts the corresponding map data package according to the input address string, the result dataset is spatial and includes only location coordinates of nodes and edges"""
    # By default simplify = True, retain_all = False, dist_type = "bbox", custom_filter = None
    def setLocationDrive(self):
        if self.address in self._location_cache:
            self.locationDrive = copy.deepcopy(self._location_cache[self.address])
            self.mapStat = OSM.stats.basic_stats(self.locationDrive)
            return
        
        self.locationDrive = self._load_location_graph()
        # Graphs reconstructed from raw cached responses may not carry the
        # street_count node attribute expected by OSMnx stats helpers.
        if not all("street_count" in data for _, data in self.locationDrive.nodes(data = True)):
            street_count = OSM.stats.count_streets_per_node(self.locationDrive)
            nx.set_node_attributes(self.locationDrive, street_count, name = "street_count")
        self.locationDrive = OSM.routing.add_edge_speeds(self.locationDrive, fallback = 6.5)
        
        OSM.distance.add_edge_lengths(self.locationDrive)
        self.mapStat = OSM.stats.basic_stats(self.locationDrive)
        self._location_cache[self.address] = copy.deepcopy(self.locationDrive)
    
    """This function extracts the necessary buildings, land use, amenity, and road information"""
    def setNetworkFeature(self):
        cache_key = (self.address, tuple(sorted(self.tags.items())), 1000)
        if cache_key in self._network_feature_cache:
            self.networkFeature = self._network_feature_cache[cache_key].copy()
            return
        self.networkFeature = OSM.features.features_from_address(self.address, self.tags, dist = 1000)
        self._network_feature_cache[cache_key] = self.networkFeature.copy()
    
    """This function extracts the node and edge sets as separate Python Lists from the LocationDrive"""
    def setNodeEdgeSets(self):
        self.nodeList = list(self.locationDrive.nodes(data = True))
        if self.verbose:
            for i in range(min(9, len(self.nodeList))):      
                print(self.nodeList[i])
        
        self.edgeList = list(self.locationDrive.edges(data = True))
        if self.verbose:
            for i in range(min(9, len(self.edgeList))):      
                print(self.edgeList[i])
    
    def setIntersectionStreetCount(self, min_streets = 3):
        # Step 1: Get a dictionary of number of street connections by each node, labeled by node ID
        # Step 1: number of street connections by node id (dict)
        counts = OSM.stats.streets_per_node(self.locationDrive)

        intersections = [nid for nid, c in counts.items() if int(c) > int(min_streets)]

        # Step 3: keep dict + separate count
        self.interStreetCount = counts                   
        self.intersectionCount = len(intersections)
    
    def setBuildingOnly(self, max_dist_m = 100):
        # Step 1: download building footprints
        
        if self.locationDrive is None:
            raise RuntimeError("Call setLocationDrive() before setBuildingOnly().")
        
        tags = {"building": True, "amenity": True}
        cache_key = (self.address, "building+amenity", "place")
        if cache_key in self._building_feature_cache:
            buildings = self._building_feature_cache[cache_key].copy()
        else:
            # IMPORTANT: query the same place-scale footprint as the road graph.
            # Using features_from_address(..., dist=1000) can under-sample large cities
            # and concentrate candidates near the geocoder centroid.
            try:
                buildings = OSM.features.features_from_place(self.address, tags)
            except Exception:
                # Fallback for environments/providers where place lookup is unavailable.
                try:
                    buildings = OSM.features.features_from_address(self.address, tags, dist = 1000)
                except Exception:
                    # Offline or endpoint issues: continue without building/amenity stamps.
                    buildings = gpd.GeoDataFrame(geometry = [], crs = "EPSG:4326")
            self._building_feature_cache[cache_key] = buildings.copy()
        
        if buildings.empty:
            for nid in self.locationDrive.nodes:
                self.locationDrive.nodes[nid]['building_type'] = None
                self.locationDrive.nodes[nid]['amenity_type'] = None
                
            self.nodeList = list(self.locationDrive.nodes(data = True))
            print("Stamped building_type: 0 (no buildings found)")
            print("Stamped amenity_type: 0 (no amenities found)")
            return
        
        b_3857 = buildings.to_crs(3857)
        b_3857 = b_3857.copy()
        b_3857['centroid'] = b_3857.geometry.centroid
        if 'building' not in b_3857.columns:
            b_3857['building'] = None
        if 'amenity' not in b_3857.columns:
            b_3857['amenity'] = None
            
        b_ctr = gpd.GeoDataFrame(
            {'building': b_3857['building'], 'amenity': b_3857['amenity']},
            geometry=b_3857['centroid'],
            crs=3857
        )        
        
        nodes_any = OSM.graph_to_gdfs(self.locationDrive, nodes = True, edges = False)
        nodes_gdf = nodes_any[0] if isinstance(nodes_any, tuple) else nodes_any
        if nodes_gdf.crs is None:
            nodes_gdf.set_crs(4326, inplace = True)
        nodes_3857 = nodes_gdf.to_crs(3857)
        
        if 'geometry' not in nodes_3857.columns or nodes_3857.geometry.isnull().any():
            nodes_3857 = nodes_3857.copy()
            nodes_3857['geometry'] = gpd.points_from_xy(nodes_3857['x'], nodes_3857['y'], crs=nodes_3857.crs)
        
        joined = gpd.sjoin_nearest(
                nodes_3857,
                b_ctr,
                how = 'left',
                distance_col = 'dist_m'
            )
        
        if max_dist_m is not None:
            too_far = joined['dist_m'] > float(max_dist_m)
            joined.loc[too_far, 'building'] = None
            joined.loc[too_far, 'amenity'] = None
        
        for nid, btype, atype in zip(joined.index, joined['building'], joined['amenity']):
            self.locationDrive.nodes[nid]['building_type'] = (str(btype) if pd.notna(btype) else None)
            self.locationDrive.nodes[nid]['amenity_type'] = (str(atype) if pd.notna(atype) else None)

        self.nodeList = list(self.locationDrive.nodes(data=True))
        
        if self.verbose:
            for i in range(min(9, len(self.nodeList))):
                print(self.nodeList[i])

        n_with = sum(1 for _, d in self.nodeList if d.get('building_type') is not None)
        print(f"Stamped building_type on nodes: {n_with} / {self.locationDrive.number_of_nodes()}")
        n_with_amenity = sum(1 for _, d in self.nodeList if d.get('amenity_type') is not None)
        print(f"Stamped amenity_type on nodes: {n_with_amenity} / {self.locationDrive.number_of_nodes()}")
        
        self.buildingNodes = {int(nid): data for nid, data in self.locationDrive.nodes(data=True)
                              if (data.get('building_type') is not None) or (data.get('amenity_type') is not None)}
        
        """
        building_nodes_with_types = set()
        
        for idx, row in buildings.iterrows():
            building_type = row.get('building') # Get the building type
            if building_type: # Only proceed if a building type is present
                geometry = row['geometry']
                if geometry.geom_type == 'Polygon':
                    # Extract nodes from the exterior of the polygon
                    for x, y in geometry.exterior.coords:
                        building_nodes_with_types.add(((x, y), building_type))
                elif geometry.geom_type == 'MultiPolygon':
                    for polygon in geometry.geoms:
                        for x, y in polygon.exterior.coords:
                            building_nodes_with_types.add(((x, y), building_type))
        
        print("building node list: ", building_nodes_with_types)
        """
    def setIntersectionOnly(self, tolerance = 15.0, min_streets = 3):
        #Step 1: Identify intersections and combine nearby sub-nodes, intersections
        self.G_proj = OSM.projection.project_graph(self.locationDrive)
        
        rawIntersections = OSM.consolidate_intersections(self.G_proj, tolerance = tolerance, rebuild_graph = True, dead_ends = False, reconnect_edges = True)
                
        counts = OSM.stats.streets_per_node(rawIntersections)
        # Step 2: convert format and match
        
        if self.verbose:
            print("checkpoint 1")

        nodes_any = OSM.graph_to_gdfs(rawIntersections, nodes=True, edges=False)
        nodes_gdf = nodes_any[0] if isinstance(nodes_any, tuple) else nodes_any
        if 'geometry' not in nodes_gdf.columns or nodes_gdf.geometry.isnull().any():
            nodes_gdf = nodes_gdf.copy()
            nodes_gdf['geometry'] = gpd.points_from_xy(nodes_gdf['x'], nodes_gdf['y'], crs=nodes_gdf.crs)
        nodes_wgs84 = nodes_gdf.to_crs(epsg = 4326)
        
        valid_ids = [node for node, count in counts.items() if count > min_streets and node in nodes_wgs84.index]
        
        if not valid_ids:
            self.intersectionNodes = []
            self.interStreetCount = {}
            print("intersection nodes 0 (none above threshold)")
            return
        if self.verbose:
            print("checkpoint 2")

        sub = nodes_gdf.loc[valid_ids]
        self.intersectionNodes = [(int(idx), geom) for idx, geom in zip(sub.index, sub.geometry)]
        self.interStreetCount = {int(n): int(counts[n]) for n in valid_ids}
        
        print(f"intersection nodes {len(self.intersectionNodes)}")
    
    """!!! No longer needed, deprecated !!!"""
    # This function checks 
    def getGuidanceCan(self):
        # Get the number of intersection candidates with enough degrees/street connections, 
        # to check if input guidnace candidate volume exceeds the eligible intersections in the network
        self.intersectionCount = OSM.stats.intersection_count(self.locationDrive, min_streets = 5)
        
        return self.intersectionCount
    
    def getShelterCanVol(self):
        return len(self.buildingNodes)