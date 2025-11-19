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


class OSMProcessor:
    def __init__(self, address):
        
        self.locationDrive = None
        
        self.networkFeature = None
        
        self.nodeList = []
        
        self.edgeList = []
        
        self.address = str(address) if address else "State College, PA, USA"
        #self.address = str(address) if address else "State College, PA, USA"
                
        self.interStreetCount = {}
        
        self.buildingNodes = {}
        
        self.intersectionNodes = {}
        
        self.tags = {'amenity':True, 'building':True, 'Assembly point':True, 'Office':True, 'Shop':True, 'Sport':True}
        
        self.mapStat = {}
        
        self.intersectionCount = 0
        
        self.G_proj = None

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
        self.locationDrive = OSM.graph.graph_from_place(self.address, network_type = "walk", truncate_by_edge = False)
        self.locationDrive = OSM.routing.add_edge_speeds(self.locationDrive, fallback = 6.5)
        
        #OSM.plot.plot_graph(self.locationDrive)
        
        #self.locationDrive = OSM.simplification.consolidate_intersections(self.locationDrive, tolerance = 5)
        
        OSM.distance.add_edge_lengths(self.locationDrive)
        self.mapStat = OSM.stats.basic_stats(self.locationDrive)
    
    """This function extracts the necessary buildings, land use, amenity, and road information"""
    def setNetworkFeature(self):
        self.networkFeature = OSM.features.features_from_address(self.address, self.tags, dist = 1000)
    
    """This function extracts the node and edge sets as separate Python Lists from the LocationDrive"""
    def setNodeEdgeSets(self):
        self.nodeList = list(self.locationDrive.nodes(data = True))
        for i in range(min(9, len(self.nodeList))):      
            print(self.nodeList[i])
        
        self.edgeList = list(self.locationDrive.edges(data = True))
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
        
        tags = {"building": True}
        buildings = OSM.features.features_from_address(self.address, tags, dist = 1000)
        
        if buildings.empty:
            for nid in self.locationDrive.nodes:
                self.locationDrive.nodes[nid]['building_type'] = None
                
            self.nodeList = list(self.locationDrive.nodes(data = True))
            print("Stamped building_type: 0 (no buildings found)")
            return
        
        b_3857 = buildings.to_crs(3857)
        b_3857 = b_3857.copy()
        b_3857['centroid'] = b_3857.geometry.centroid
        b_ctr = gpd.GeoDataFrame(
            {'building': b_3857['building']},
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
            joined.loc[joined['dist_m'] > float(max_dist_m), 'building'] = None
        
        for nid, btype in zip(joined.index, joined['building']):
            self.locationDrive.nodes[nid]['building_type'] = (str(btype) if pd.notna(btype) else None)

        self.nodeList = list(self.locationDrive.nodes(data=True))
        
        for i in range(min(9, len(self.nodeList))):
            print(self.nodeList[i])

        n_with = sum(1 for _, d in self.nodeList if d.get('building_type') is not None)
        print(f"Stamped building_type on nodes: {n_with} / {self.locationDrive.number_of_nodes()}")
        
        self.buildingNodes = {int(nid): data for nid, data in self.locationDrive.nodes(data=True)
                              if data.get('building_type') is not None}
        
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