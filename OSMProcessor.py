#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

7/29: Created structure
8/1: Version 1.0 done, all functions implemented, pending testing
"""

import osmnx as OSM
import os
import geopandas as gpd
import pandas as pd
import copy
import hashlib
import json
import networkx as nx
import math

from ShelterTypes import SHELTER_OSM_TAGS


class OSMProcessor:
    _location_cache = {}
    _network_feature_cache = {}
    _building_feature_cache = {}
    _default_overpass_endpoints = (
        "https://overpass-api.de/api",
        "https://overpass.kumi.systems/api",
        "https://overpass.openstreetmap.ru/api",
    )

    def __init__(
        self,
        address,
        *,
        query_mode: str = "place",
        center_point=None,
        radius_m=None,
        verbose: bool = False,
    ):
        
        self.locationDrive = None
        
        self.networkFeature = None
        
        self.nodeList = []
        
        self.edgeList = []
        
        self.address = str(address) if address else "State College, PA, USA"
        self.query_mode = str(query_mode).strip().lower()
        if self.query_mode not in {"place", "point"}:
            raise ValueError("query_mode must be either 'place' or 'point'")
        self.center_point = None
        self.radius_m = None
        if self.query_mode == "point":
            if center_point is None or len(center_point) != 2 or radius_m is None:
                raise ValueError("point queries require center_point=(lat, lon) and radius_m")
            lat, lon = float(center_point[0]), float(center_point[1])
            radius = float(radius_m)
            if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
                raise ValueError("center latitude must lie in [-90, 90]")
            if not math.isfinite(lon) or not -180.0 <= lon <= 180.0:
                raise ValueError("center longitude must lie in [-180, 180]")
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("radius_m must be finite and positive")
            self.center_point = (lat, lon)
            self.radius_m = radius
        self.verbose = bool(verbose)
                
        self.interStreetCount = {}
        
        self.buildingNodes = {}
        
        self.intersectionNodes = {}
        
        self.tags = {'amenity':True, 'building':True, 'Assembly point':True, 'Office':True, 'Shop':True, 'Sport':True}
        
        self.mapStat = {}
        
        self.intersectionCount = 0
        
        self.G_proj = None
        self._graph_provenance = None
        self.intersectionConsolidation = {
            "enabled": False,
            "tolerance_m": None,
            "nodes_before": None,
            "nodes_after": None,
            "edges_before": None,
            "edges_after": None,
            "nodes_consolidated": 0,
            "zero_length_edges_removed": 0,
            "cache_reused": False,
        }
        self._consolidated_graph_cache = None
        
           
    def query_spec(self):
        return {
            "address": self.address,
            "query_mode": self.query_mode,
            "center": None if self.center_point is None else list(self.center_point),
            "radius_m": self.radius_m,
            "network_type": "walk",
        }

    def query_key(self):
        return json.dumps(self.query_spec(), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def graph_provenance(self):
        if self.locationDrive is None:
            raise RuntimeError("Call setLocationDrive() before requesting provenance")
        if self._graph_provenance is None:
            cache_path = os.path.abspath(self._graph_cache_path())
            self._graph_provenance = {
                "source": "OpenStreetMap",
                "license": "Open Database License (ODbL)",
                "query_spec": self.query_spec(),
                "query_sha256": hashlib.sha256(self.query_key().encode("utf-8")).hexdigest(),
                "graph_cache_path": cache_path,
                "graph_cache_sha256": (
                    self._sha256_file(cache_path) if os.path.exists(cache_path) else None
                ),
                "consolidated_graph_cache_path": self._consolidated_graph_cache,
                "consolidated_graph_cache_sha256": (
                    self._sha256_file(self._consolidated_graph_cache)
                    if self._consolidated_graph_cache
                    and os.path.exists(self._consolidated_graph_cache)
                    else None
                ),
                "graph_nodes": int(self.locationDrive.number_of_nodes()),
                "graph_edges": int(self.locationDrive.number_of_edges()),
                "stamped_building_or_amenity_nodes": int(len(self.buildingNodes)),
                "intersection_consolidation": dict(self.intersectionConsolidation),
            }
        return dict(self._graph_provenance)

    def _graph_cache_path(self):
        # Preserve the established single-place cache name. Point/radius
        # profiles hash the complete query so two study extents in one city
        # cannot collide.
        identity = self.address if self.query_mode == "place" else self.query_key()
        key = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        safe_addr = "".join(ch if ch.isalnum() else "_" for ch in self.address).strip("_")
        safe_addr = safe_addr[:80] if safe_addr else "address"
        os.makedirs(OSM.settings.cache_folder, exist_ok = True)
        return os.path.join(OSM.settings.cache_folder, f"graph_walk_{safe_addr}_{key}.graphml")

    def _consolidated_cache_paths(self, tolerance):
        contract = {
            "algorithm": "osmnx_topological_consolidation_v1",
            "query": self.query_spec(),
            "tolerance_m": float(tolerance),
            "dead_ends": False,
            "reconnect_edges": True,
        }
        key = hashlib.sha1(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        raw_path = self._graph_cache_path()
        stem, _ = os.path.splitext(raw_path)
        graph_path = f"{stem}_consolidated_{key}.graphml"
        return graph_path, f"{graph_path}.json", contract

    @staticmethod
    def _write_json_atomic(path, payload):
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        os.replace(temporary, path)
    
    def _try_graph_from_place(self, overpass_url):
        # OSMnx renamed these settings in 2.0.  Supporting both names keeps
        # cached/reproducible experiments runnable across the declared 1.x/2.x
        # environments without silently ignoring the requested endpoint.
        url_setting = (
            "overpass_url"
            if hasattr(OSM.settings, "overpass_url")
            else "overpass_endpoint"
        )
        timeout_setting = (
            "requests_timeout"
            if hasattr(OSM.settings, "requests_timeout")
            else "timeout"
        )
        old_overpass_url = getattr(OSM.settings, url_setting)
        old_requests_timeout = getattr(OSM.settings, timeout_setting)
        try:
            setattr(OSM.settings, url_setting, overpass_url)
            # Keep responsiveness; each endpoint may still retry internally.
            setattr(
                OSM.settings,
                timeout_setting,
                min(int(old_requests_timeout), 90),
            )
            if self.query_mode == "point":
                G = OSM.graph.graph_from_point(
                    self.center_point,
                    dist=self.radius_m,
                    dist_type="bbox",
                    network_type="walk",
                    truncate_by_edge=False,
                )
            else:
                G = OSM.graph.graph_from_place(
                    self.address,
                    network_type="walk",
                    truncate_by_edge=False,
                )
            return G
        finally:
            setattr(OSM.settings, url_setting, old_overpass_url)
            setattr(OSM.settings, timeout_setting, old_requests_timeout)
    
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
        
        detail = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(
            f"Unable to fetch OSM road graph for {self.query_spec()}. "
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
        cache_key = self.query_key()
        if cache_key in self._location_cache:
            self.locationDrive = copy.deepcopy(self._location_cache[cache_key])
            self.mapStat = OSM.stats.basic_stats(self.locationDrive)
            return
        
        self.locationDrive = self._load_location_graph()
        # Graphs reconstructed from raw cached responses may not carry the
        # street_count node attribute expected by OSMnx stats helpers.
        if not all("street_count" in data for _, data in self.locationDrive.nodes(data = True)):
            street_count = OSM.stats.count_streets_per_node(self.locationDrive)
            nx.set_node_attributes(self.locationDrive, street_count, name = "street_count")
        # OSMnx 2.x moved speed helpers under ``routing``; 1.x exposes the
        # same operation at top level (and under ``speed`` in later 1.x).
        if hasattr(OSM, "routing"):
            add_edge_speeds = OSM.routing.add_edge_speeds
        elif hasattr(OSM, "speed"):
            add_edge_speeds = OSM.speed.add_edge_speeds
        else:
            add_edge_speeds = OSM.add_edge_speeds
        self.locationDrive = add_edge_speeds(self.locationDrive, fallback=6.5)
        
        OSM.distance.add_edge_lengths(self.locationDrive)
        self.mapStat = OSM.stats.basic_stats(self.locationDrive)
        self._location_cache[cache_key] = copy.deepcopy(self.locationDrive)
    
    """This function extracts the necessary buildings, land use, amenity, and road information"""
    def setNetworkFeature(self):
        cache_key = (self.query_key(), tuple(sorted(self.tags.items())))
        if cache_key in self._network_feature_cache:
            self.networkFeature = self._network_feature_cache[cache_key].copy()
            return
        if self.query_mode == "point":
            self.networkFeature = OSM.features.features_from_point(
                self.center_point,
                self.tags,
                dist=self.radius_m,
            )
        else:
            self.networkFeature = OSM.features.features_from_place(self.address, self.tags)
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

    def consolidateIntersections(self, tolerance=5.0):
        """Replace OSM junction-node clusters with one routable intersection.

        OSM frequently represents divided roads and large junctions with
        several nearby nodes.  OSMnx's topology rebuild preserves reconnecting
        edges while replacing each buffered cluster with a single centroid
        node.  The rebuilt graph is projected back to its original geographic
        CRS because the simulator's coordinate conversion expects longitude
        and latitude.
        """
        if self.locationDrive is None:
            raise RuntimeError("Call setLocationDrive() before consolidating intersections")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("intersection consolidation tolerance must be finite and positive")

        graph = self.locationDrive
        nodes_before = int(graph.number_of_nodes())
        edges_before = int(graph.number_of_edges())
        raw_cache_path = self._graph_cache_path()
        graph_cache_path, metadata_cache_path, cache_contract = (
            self._consolidated_cache_paths(tolerance)
        )
        raw_cache_sha256 = (
            self._sha256_file(raw_cache_path)
            if os.path.exists(raw_cache_path)
            else None
        )
        if (
            raw_cache_sha256 is not None
            and os.path.exists(graph_cache_path)
            and os.path.exists(metadata_cache_path)
        ):
            try:
                with open(metadata_cache_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                valid_metadata = bool(
                    metadata.get("schema_version") == 1
                    and metadata.get("contract") == cache_contract
                    and metadata.get("raw_graph_sha256") == raw_cache_sha256
                    and int(metadata["diagnostics"]["nodes_before"])
                    == nodes_before
                    and int(metadata["diagnostics"]["edges_before"])
                    == edges_before
                )
                if valid_metadata:
                    rebuilt = OSM.io.load_graphml(graph_cache_path)
                    if rebuilt.number_of_nodes() == 0 or rebuilt.number_of_edges() == 0:
                        raise ValueError("cached consolidated graph is empty")
                    if not all(
                        float(data.get("length", 0.0)) > 0.0
                        for *_, data in rebuilt.edges(data=True)
                    ):
                        raise ValueError(
                            "cached consolidated graph has a non-positive edge"
                        )
                    diagnostics = dict(metadata["diagnostics"])
                    diagnostics["cache_reused"] = True
                    self.locationDrive = rebuilt
                    self.G_proj = None
                    self.nodeList = list(rebuilt.nodes(data=True))
                    self.edgeList = list(rebuilt.edges(data=True))
                    self.mapStat = OSM.stats.basic_stats(rebuilt)
                    self.intersectionConsolidation = diagnostics
                    self._consolidated_graph_cache = os.path.abspath(
                        graph_cache_path
                    )
                    self._graph_provenance = None
                    print(
                        "[INTERSECTION CONSOLIDATION CACHE] "
                        f"tolerance_m={tolerance:g} "
                        f"nodes={nodes_before}->{rebuilt.number_of_nodes()} "
                        f"edges={edges_before}->{rebuilt.number_of_edges()}",
                        flush=True,
                    )
                    return dict(self.intersectionConsolidation)
            except Exception as exc:
                if self.verbose:
                    print(
                        "[INTERSECTION CONSOLIDATION CACHE] Ignoring invalid "
                        f"cache {graph_cache_path}: {exc}",
                        flush=True,
                    )
        original_crs = graph.graph.get("crs", "EPSG:4326")
        projected = OSM.projection.project_graph(graph)
        consolidate = (
            OSM.consolidate_intersections
            if hasattr(OSM, "consolidate_intersections")
            else OSM.simplification.consolidate_intersections
        )
        rebuilt = consolidate(
            projected,
            tolerance=tolerance,
            rebuild_graph=True,
            dead_ends=False,
            reconnect_edges=True,
        )
        if rebuilt.number_of_nodes() == 0 or rebuilt.number_of_edges() == 0:
            raise RuntimeError("Intersection consolidation produced an empty road graph")
        rebuilt = OSM.projection.project_graph(rebuilt, to_crs=original_crs)

        zero_length_edges = [
            (u, v, key)
            for u, v, key, data in rebuilt.edges(keys=True, data=True)
            if float(data.get("length", 0.0)) <= 0.0
        ]
        rebuilt.remove_edges_from(zero_length_edges)
        rebuilt.remove_nodes_from(tuple(nx.isolates(rebuilt)))

        # Consolidation changes topology, so street counts must be recomputed
        # on the graph that is actually routed and simulated.
        street_count = OSM.stats.count_streets_per_node(rebuilt)
        nx.set_node_attributes(rebuilt, street_count, name="street_count")
        if not all(float(data.get("length", 0.0)) > 0.0 for *_, data in rebuilt.edges(data=True)):
            raise RuntimeError("Intersection consolidation retained a non-positive edge length")

        self.locationDrive = rebuilt
        # ``locationDrive`` is the authoritative rebuilt geographic graph.
        # Avoid retaining the pre-consolidation projected graph as apparently
        # matching state; it can be projected again on demand if ever needed.
        self.G_proj = None
        self.nodeList = list(rebuilt.nodes(data=True))
        self.edgeList = list(rebuilt.edges(data=True))
        self.mapStat = OSM.stats.basic_stats(rebuilt)
        self.intersectionConsolidation = {
            "enabled": True,
            "tolerance_m": tolerance,
            "nodes_before": nodes_before,
            "nodes_after": int(rebuilt.number_of_nodes()),
            "edges_before": edges_before,
            "edges_after": int(rebuilt.number_of_edges()),
            "nodes_consolidated": max(0, nodes_before - int(rebuilt.number_of_nodes())),
            "zero_length_edges_removed": len(zero_length_edges),
            "cache_reused": False,
        }
        if raw_cache_sha256 is not None:
            temporary_graph = f"{graph_cache_path}.tmp.{os.getpid()}"
            try:
                OSM.io.save_graphml(rebuilt, temporary_graph)
                os.replace(temporary_graph, graph_cache_path)
                self._write_json_atomic(
                    metadata_cache_path,
                    {
                        "schema_version": 1,
                        "contract": cache_contract,
                        "raw_graph_sha256": raw_cache_sha256,
                        "diagnostics": dict(self.intersectionConsolidation),
                    },
                )
                self._consolidated_graph_cache = os.path.abspath(
                    graph_cache_path
                )
            finally:
                if os.path.exists(temporary_graph):
                    os.remove(temporary_graph)
        self._graph_provenance = None
        print(
            "[INTERSECTION CONSOLIDATION] "
            f"tolerance_m={tolerance:g} nodes={nodes_before}->{rebuilt.number_of_nodes()} "
            f"edges={edges_before}->{rebuilt.number_of_edges()}",
            flush=True,
        )
        return dict(self.intersectionConsolidation)
    
    def setBuildingOnly(self, max_dist_m = 100):
        # Step 1: download building footprints
        
        if self.locationDrive is None:
            raise RuntimeError("Call setLocationDrive() before setBuildingOnly().")
        
        tags = SHELTER_OSM_TAGS
        cache_key = (
            self.query_key(),
            "shelter-candidates-v1",
            tuple((key, tuple(values)) for key, values in sorted(tags.items())),
        )
        if cache_key in self._building_feature_cache:
            buildings = self._building_feature_cache[cache_key].copy()
        else:
            try:
                if self.query_mode == "point":
                    buildings = OSM.features.features_from_point(
                        self.center_point,
                        tags,
                        dist=self.radius_m,
                    )
                else:
                    buildings = OSM.features.features_from_place(self.address, tags)
            except Exception:
                # A road-only graph is still useful for map prewarming, but Core
                # will fail loudly before an experiment if no shelter candidates
                # can be derived.  Never substitute a different geographic query.
                buildings = gpd.GeoDataFrame(geometry = [], crs = "EPSG:4326")
            self._building_feature_cache[cache_key] = buildings.copy()
        
        if buildings.empty:
            for nid in self.locationDrive.nodes:
                self.locationDrive.nodes[nid]['building_type'] = None
                self.locationDrive.nodes[nid]['amenity_type'] = None
                
            self.nodeList = list(self.locationDrive.nodes(data = True))
            self._graph_provenance = None
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
        self._graph_provenance = None
        
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
        """Compute intersection diagnostics on the active consolidated graph.

        This historical method no longer builds a disconnected side graph.
        ``consolidateIntersections`` must be called first when topology
        consolidation is desired.
        """
        if self.locationDrive is None:
            raise RuntimeError("Call setLocationDrive() before identifying intersections")
        counts = OSM.stats.streets_per_node(self.locationDrive)
        nodes_any = OSM.graph_to_gdfs(self.locationDrive, nodes=True, edges=False)
        nodes_gdf = nodes_any[0] if isinstance(nodes_any, tuple) else nodes_any
        if 'geometry' not in nodes_gdf.columns or nodes_gdf.geometry.isnull().any():
            nodes_gdf = nodes_gdf.copy()
            nodes_gdf['geometry'] = gpd.points_from_xy(nodes_gdf['x'], nodes_gdf['y'], crs=nodes_gdf.crs)
        valid_ids = [
            node
            for node, count in counts.items()
            if int(count) >= int(min_streets) and node in nodes_gdf.index
        ]
        
        if not valid_ids:
            self.intersectionNodes = []
            self.interStreetCount = {}
            print("intersection nodes 0 (none above threshold)")
            return
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
