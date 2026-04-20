#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

7/28: Structure created, writen all import dependencies, variables
8/6: Worked on initSimulator function
8/7: Finished readInputCSV function
"""

import os
import osmnx as OSM
import csv
import numpy as np
import torch

#import torch
#import torch.nn as nn
#import torch.optim as optim

"""Import other main Simulator modules"""
from OSMProcessor import OSMProcessor
from MapDatabase import MapDS
from HazardDatabase import HazardDS
from GuidanceDatabase import GuidanceDS
from ShelterDatabase import ShelterDS
from PedestrianDatabase import PedDS
from CAProcessor import CellTracker
from SocialForce import ForceProcessor

"""Import RL components"""
from RLBridge import RLBridge
from TrainingLogger import trainingLog

"""Helpers of RL, help track training progress and timestep progress"""
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

#from reporter import log

import time
import math

"""Timer: efficiency debugging (how much step each step take)"""
class Timer:
    def __init__(self): self.t0 = time.perf_counter()
    def lap(self, label):
        t = time.perf_counter()
        dt = t - self.t0
        self.t0 = t
        print(f"[TIMER] {label}: {dt:.3f}s")

class Core:
    def __init__(self, machine):
        
        """Input Parameters"""
        # Indicate the total duration of timestep
        self.stopTime = 0
        
        # Indicate the input-defined, initial (and max) volume of pedestrians in the network
        self.pedVol = 0
        
        # Indicate the input-defined, total volume of hazards we will simulate
        self.hazardVol = 0
        
        # Indicate the input-defined, ideal traveling speed of evacuees
        self.maxSpeed = 0
        
        # Input defined, indicate how many shelter candidate building nodes we will sample
        self.shelterCanVol = 0
        
        # Input defined, indicate how many guidance point candidates intersection nodes we will sample
        self.guidanceCanVol = 0
        
        # Input defined, indicate the mean and variance values [int, int] of speed reduction value of hazards
        self.hazardSpeedReduct = []
        
        # Indicate the spreading probability of hazards
        self.hazardSpreadRate = []
        
        # Input defined, indicate the mean and variance values [int, int] of casualty rate of hazards
        self.hazardCasualtyRate = []
        
        # Input defined, indicate the initial volume of shelters we deploy in the network at t = 0
        self.initShelterVol = 0
        
        # Input defined, indicate the initial volume of guidance points we deploy in the network at t = 0
        self.initGuidanceVol = 0
        
        # Input defined, indicate the address name we used to extract needed map data, ex. Seattle, Washington, USA
        self.address = ""
        
        # indicates the number of X-axis cuts for cells
        self.cellX = 0
        
        # indicates the number of Y-axis cuts for cells
        self.cellY = 0
        
        """RL parameters, will tune later"""
        # contrainer for learning rate used by the RL model
        self.learningRate = 0
        # Exploration vs. Exploitation prob
        self.explorationRate = 0
        # name of the optimizer used
        self.optimizer = "Adam"
        
        """Pointers to main processing modules"""
        # Pointer for the NetworkDatabase
        self.mapDS = None
        
        # Pointer for the PedestrianDatabase
        self.pedDS = None
        
        # Pointer for the HazardDatabase
        self.hazardDS = None
        
        # Pointer for the CellularAutomataTracker
        self.cellTracker = None
        
        # Pointer for the GuidanceDatabase
        self.guidanceDS = None
        
        # Pointer for the ShelterDatabase
        self.shelterDS = None
        
        # Pointer for the SocialForceProcessor
        self.forceTracker = None
        
        # Pointer to handler of all SOSM extracter functions
        self.OSMProcessor = None
        
        self.rl = None
        
        self.logger = None
        self.run_dir = None
        
        """Other Parameters, for auto-execution purposes"""
        self.machine = machine
        
        self.logName = ""
        
        self.currReplication = 0
        self.verbose = False
        self.profile_timing = False
        self.optimize_guidance = False

    """Getter Functions"""
    
    def getStopTime(self):
        return self.stopTime

    def getPedVol(self):
        return self.pedVol
    
    def getHazardVol(self):
        return self.hazardVol 
    
    def getMaxSpeed(self):
        return self.maxSpeed
    
    def getShelterCanVol(self):
        return self.shelterCanVol
    
    def getGuidanceCanVol(self):
        return self.guidanceCanVol
    
    def getSpeedReduct(self):
        return self.hazardSpeedReduct
    
    def getCasualtyRate(self):
        return self.hazardCasualtyRate
    
    def getInitShelterVol(self):
        return self.initShelterVol
    
    def getInitGuidanceVol(self):
        return self.initGuidanceVol
    
    """Handles all functionality at t = 0"""
    """Still need to add input parameters to function calls"""
    def initSimulator(self, replication, machine, config_overrides = None, phase: str = "train", train_mode: bool = True):
        # read all input data
        self.readInputCSV()
        
        # runtime override for experiment scripts
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                    
        if not self.optimize_guidance:
            self.guidanceCanVol = 0
            self.initGuidanceVol = 0
        
        # For automated model excution once uploaded to a cloud-based computing platform
        self.run_dir = os.path.join("runs", str(phase), f"rep_{int(replication):03d}_{machine}")
        os.makedirs(self.run_dir, exist_ok = True)
        
        # Call OSMProcessor to get relevant map data
        self.OSMProcessor = OSMProcessor(self.address, verbose = self.verbose)
        
        # Call relevant OSMProcessor functions in order
        # Extract all relevant map data and setup the node, edges, intersection, buildings dataset
        # locationDrive is the overall container of all map data
        self.OSMProcessor.setLocationDrive()
        print("Network geometry extracted")
        # NOTE: OSMProcessor.setNetworkFeature() is intentionally skipped here:
        # it is not consumed downstream in the simulation path and is very expensive.
        # formally establish the node and edge set
        self.OSMProcessor.setNodeEdgeSets()
        print("Nodes and edge sets extracted")
        # get intersection and building sets ready
        self.OSMProcessor.setIntersectionStreetCount()
        self.OSMProcessor.setBuildingOnly()
        self.OSMProcessor.setIntersectionOnly()
        
        # check if have enough shelter and guidance candidates in the network
        """
        networkGuidanceCanVol = self.OSMProcessor.getGuidanceCan()
        print("Detected guidance can volume: ", networkGuidanceCanVol)
        networkShelterCanVol = self.OSMProcessor.getShelterCanVol()
        print("Detected shelter can volume: ", networkShelterCanVol)

        if self.guidanceCanVol >= networkGuidanceCanVol:
            self.guidanceCanVol = networkGuidanceCanVol
            print("Not enough guidance candidates in the network, volume of guidance candidate adjusted to network!")

        if self.shelterCanVol >= networkShelterCanVol:
            self.shelterCanVol = networkShelterCanVol
            print("Not enough shelter candidates in the network, volume of shelter candidate adjusted to network!")
        """
        
        # Initialize instances of processing modules
        self.mapDS = MapDS(self.OSMProcessor.nodeList, self.OSMProcessor.edgeList, self.address, self.OSMProcessor.locationDrive)
        self.pedDS = PedDS(self.pedVol)
        self.hazardDS = HazardDS(self.hazardVol, self.hazardCasualtyRate, self.hazardSpreadRate, self.hazardSpeedReduct)
        self.cellTracker = CellTracker(self.cellX, self.cellY)
        self.forceTracker = ForceProcessor()
        self.shelterDS = ShelterDS(self.shelterCanVol, self.initShelterVol)
        self.guidanceDS = GuidanceDS(self.guidanceCanVol, self.initGuidanceVol)
        
        # Call initializing functions here, functions needed at t = 0 (follow old model, with additions)
        self.mapDS.computeConvertUnit()
        self.mapDS.boundarySetter()
        
        self.hazardDS.setCellTracker(self.cellTracker)

        
        x_vals = []
        y_vals = []
        for _, data in self.OSMProcessor.nodeList:
            lon = float(data["x"])
            lat = float(data["y"])
            x_m, y_m = self.mapDS.coordToMeters(lon, lat)
            x_vals.append(float(x_m))
            y_vals.append(float(y_m))
        
        def _axis_stats(vals):
            if not vals:
                return 0.0, 1.0
            arr = np.asarray(vals, dtype = float)
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if not math.isfinite(vmin):
                vmin = 0.0
            if not math.isfinite(vmax):
                vmax = vmin + 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            return vmin, vmax

        xMin, xMax = _axis_stats(x_vals)
        yMin, yMax = _axis_stats(y_vals)
        xLength = float(xMax - xMin)
        yLength = float(yMax - yMin)

        # Build adaptive cell boundaries from actual node distribution in meter space.
        # Use observed axis min/max instead of global bbox anchors so partitions
        # match the true occupied road-network extent.
        def _adaptive_edges(vals, bins, axis_min, axis_max):
            bins = int(bins)
            if bins <= 0:
                raise ValueError("bins must be > 0")
            if not vals:
                step = (float(axis_max) - float(axis_min)) / bins
                return [float(axis_min) + i * step for i in range(bins)] + [float(axis_max)]
            
            arr = np.asarray(vals, dtype = float)
            q = np.linspace(0.0, 1.0, bins + 1)
            edges = np.quantile(arr, q).astype(float)
            edges[0] = float(axis_min)
            edges[-1] = float(axis_max)
            for idx in range(1, len(edges)):
                if (not math.isfinite(edges[idx])) or edges[idx] <= edges[idx - 1]:
                    edges[idx] = min(float(axis_max), edges[idx - 1] + 1e-6)
            edges[-1] = max(edges[-1], edges[-2] + 1e-6)
            return list(edges)
        
        
        x_edges = _adaptive_edges(x_vals, int(self.cellX), xMin, xMax)
        y_edges = _adaptive_edges(y_vals, int(self.cellY), yMin, yMax)

        print("network X span (occupied): ", xLength)
        print("network Y span (occupied): ", yLength)
        print("network X range (occupied): ", (xMin, xMax))
        print("network Y range (occupied): ", (yMin, yMax))
        
        self.cellTracker.initialCut(xLength, yLength, xEdges = x_edges, yEdges = y_edges)
        
        N = int(self.cellX * self.cellY)
        def _init_vec(name):
            if getattr(self.cellTracker, name, None) is None:
                setattr(self.cellTracker, name, np.zeros(N, dtype=float))
        
        for wire in [
            "countByCell",
            "avgVelocityByCell",
            "heatByCell",
            "smokeByCell",
            "dangerLevelByCell",
            "shelterFulfillByCell",
            "guidanceInterByCell",   # or guidanceByCell — whichever you chose in (B)
            "wellnessPenaltyByCell",
        ]:
            _init_vec(wire)

        self.mapDS.nodeInit(self.cellTracker)
        self.mapDS.edgeInit(self.cellTracker)
        self.mapDS.buildEdgeIndices()
        self.mapDS.computeNodeCapSum()
        
        self.forceTracker.setupCellTracker(self.cellTracker)
        
        self.guidanceDS.guidanceCanList = self.mapDS.guidanceCanList
        print(f"Guidance candidates detected: {len(self.guidanceDS.guidanceCanList)}")
        self.shelterDS.shelterCanList = self.mapDS.shelterCanList
        print(f"Shelter candidates detected: {len(self.shelterDS.shelterCanList)}")

        self.guidanceDS.pointPerCell(self.cellTracker, self.cellX, self.cellY)
        if self.optimize_guidance:
            self.guidanceDS.initGuidance()
            if self.verbose:
                print("Guidance Points list: ", self.guidanceDS.guidanceList)
        else:
            print("Guidance optimization disabled: skipping initial guidance installation.")
                
        self.shelterDS.shelterPerCell(self.cellTracker, self.cellX, self.cellY)
        self.shelterDS.initShelter()
        if self.verbose:
            print("Shelters list: ", self.shelterDS.shelterList)
        
        self.hazardDS.initHazard(self.mapDS, self.cellTracker)
        
        self.pedDS.initPedestrianAgent(self.mapDS, self.cellTracker, self.maxSpeed)
        
        self.guidanceDS.guidanceByOSMID = {gu.nodeMapped.OSMID: gu for gu in self.guidanceDS.guidanceList.values()}  
        self.shelterDS.shelterByOSMID   = {sh.nodeMapped.OSMID: sh for sh in self.shelterDS.shelterList.values()}    
        
        self.pedDS.checkReady(mapDS = self.mapDS,
                              cellTracker = self.cellTracker,
                              maxSpeed = self.maxSpeed,
                              hazardDS = self.hazardDS,
                              shelterDS = self.shelterDS,
                              guidanceDS = self.guidanceDS,
                              forceTracker = self.forceTracker)

        rl_lr = float(self.learningRate) if self.learningRate > 0 else 3e-4
        if rl_lr > 1e-2:
            rl_lr = 1e-3
            print("[RL CONFIG] Input learning rate too high; clamped to 1e-3 for PPO stability.")
        self.rl = RLBridge(
            self,
            mode = "full",
            train_mode = train_mode,
            lr = rl_lr,
            gamma = 0.995,
            lam = 0.97,
            clip_eps = 0.15,
            epochs = 8,
            minibatch_size = 8,
            entropy_coef = 0.003,
            value_coef = 0.7,
            target_kl = 0.03,
            value_clip_eps = 0.2,
            shelter_action_interval = 5,
        )
        self.logger = trainingLog(run_dir = self.run_dir, window = 100, use_tensorboard = False)
        
        self.simulationEnumerator()
        
    """Main Functions"""
    """Read the input-defined parameter values accordingly from a CSV file"""
    def readInputCSV(self):
        print("Input parameters are as follows: ")
        line = 1
        with open('RLEvacuationParameter.csv') as csvfile:
            parameterReader = csv.reader(csvfile)
            parameterList = list(parameterReader)
            for row in parameterList:
                # primary model parameters
                if line == 2:
                    self.stopTime = int(row[0])
                    print("Stop Time is: ", self.stopTime)
                    self.address = str(row[1])
                    print("Input address is: ", self.address)
                    self.maxSpeed = int(row[2])
                    print("Pedestrian Max Speed is: ", self.maxSpeed)
                    self.pedVol = int(row[3])
                    print("Pedestrian Volume is: ", self.pedVol)
                    self.hazardVol = int(row[4])
                    print("Hazard volume is: ", self.hazardVol)
                    self.cellX = int(row[5])
                    print("Number of X-axis cell cut is: ", self.cellX)
                    self.cellY = int(row[6])
                    print("Number of Y-axis cell cut is: ", self.cellY)
                # guidance and shelter parameters
                elif line == 4:
                    self.guidanceCanVol = int(row[0])
                    print("Guidance Candidate vol is: ", self.guidanceCanVol)
                    self.shelterCanVol = int(row[1])
                    print("Shelter Candidate vol is: ", self.shelterCanVol)
                    self.initShelterVol = int(row[2])
                    print("Initial Shelter vol is: ", self.initShelterVol)
                    self.initGuidanceVol = int(row[3])
                    print("Initial Guidance vol is: ", self.initGuidanceVol)
                # hazard parameters
                elif line == 6:
                    self.hazardCasualtyRate = [int(row[0]), int(row[1])]
                    print("Hazard casualty mean and variance rates are: ", self.hazardCasualtyRate)
                    self.hazardSpreadRate = [int(row[2]), int(row[3])]
                    print("Hazard spread mean and variance rates are: ", self.hazardSpreadRate)
                    self.hazardSpeedReduct = [int(row[4]), int(row[5])]
                    print("Hazard speed reduction mean and variance rates are: ", self.hazardSpeedReduct)
                # RL parameters
                elif line == 8:
                    self.learningRate = float(row[0])
                    print("RL model learning rate is: ", self.learningRate)
                    self.explorationRate = float(row[1])
                    print("RL model exploration rate is: ", self.explorationRate)
                    self.optimizer = str(row[2])
                    print("RL model optimizier is: ", self.optimizer)
            
                line += 1
                
    """Controls the main execution of functions in the Simulator, all functionalities for t >= 1"""
    """Still need to add input parameters to function calls"""
    def simulationEnumerator(self):
        iterator = range(1, self.stopTime)
        
        # for output, visualized progress bar
        if _HAS_TQDM:
            iterator = tqdm(iterator, desc = "Sim timesteps", ncols = 120)
        
        for time in iterator: 
            if not _HAS_TQDM:
                print("Current timestep is: ", time)
                
            tmr = Timer() if self.profile_timing else None
                
            self.pedDS.startDocument()
            if tmr is not None: tmr.lap("startDocument")

            self.hazardDS.spreadUpdate()
            if tmr is not None: tmr.lap("hazard.spreadUpdate")

            self.hazardDS.heatUpdate()
            if tmr is not None: tmr.lap("hazard.heatUpdate")

            self.hazardDS.smokeUpdate()
            if tmr is not None: tmr.lap("hazard.smokeUpdate")

            # === PED/GU/SH LOOKUPS ===
            self.pedDS.loadGuShLookup(self.guidanceDS.guidanceByOSMID, self.shelterDS.shelterByOSMID)
            if tmr is not None: tmr.lap("ped.loadGuShLookup")

            # === PEDESTRIAN INTERACTIONS ===
            self.pedDS.pedestrianHazardInteraction()
            if tmr is not None: tmr.lap("ped.hazardInteraction")

            self.pedDS.interPedestrianInteraction()
            if tmr is not None: tmr.lap("ped.interPedInteraction")

            self.pedDS.pedestrianNetworkInteraction()
            if tmr is not None: tmr.lap("ped.networkInteraction")

            # === CELL UPDATE ===
            self.cellTracker.cellUpdate(pedDS=self.pedDS, forceTracker=self.forceTracker)
            if tmr is not None: tmr.lap("cellTracker.cellUpdate")

            # === RL ===
            rl_out = self.rl.step()
            if tmr is not None: tmr.lap("rl.step")

            # === DOCU/LOG ===
            self.pedDS.docuStatus()
            if tmr is not None: tmr.lap("ped.docuStatus")
            
            cumuResult = self.pedDS.result
            metrics = {
                "arrival": cumuResult.get("arrival", 0),
                "casualty": cumuResult.get("casualty", 0),
                "evacuated": cumuResult.get("evacuated", 0),
                "guided": cumuResult.get("guided", 0),
                "affected": cumuResult.get("affected", 0),
                "added_shelters": rl_out.get("added_shelters", 0),
            }
            
            self.logger.log_step(t=time, reward=float(rl_out["reward"]), metrics=metrics)
            
            if getattr(self.rl, "debug", False) and (time % int(getattr(self.rl, "print_every", 1)) == 0):
                print(f"[CORE] t={time} | reward={rl_out['reward']:.3f} | "
                      f"arr={metrics['arrival']} cas={metrics['casualty']} evac={metrics['evacuated']} "
                      f"guided={metrics['guided']} affected={metrics['affected']} | "
                      f"added_sh={metrics['added_shelters']}")
            if tmr is not None: tmr.lap("logger.log_step")
       
            for name in ["countByCell","avgVelocityByCell","heatByCell","smokeByCell",
                "dangerLevelByCell","shelterFulfillByCell","guidanceInterByCell","wellnessPenaltyByCell"]:
                arr = getattr(self.cellTracker, name, None)
                if arr is None or len(np.asarray(arr).reshape(-1)) != (self.cellX * self.cellY):
                    print(f"[WIRE CHECK] {name} missing or wrong length")
        if self.rl is not None and hasattr(self.rl, "end_episode"):
            self.rl.end_episode()
            if hasattr(self.shelterDS, "remainingCandidateCount"):
                print(f"[SHELTER POOL] remaining_candidates={self.shelterDS.remainingCandidateCount()} active_shelters={len(self.shelterDS.shelterList)}")
            
        if self.logger is not None:
            self.logger.plot_png("reward_curve.png")
            self.logger.plot_metrics_png(out_name = "core_metrics_curve.png", metric_cols = ["casualty", "evacuated", "arrival"])
            self.logger.close()
    