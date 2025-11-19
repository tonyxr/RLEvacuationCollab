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
    def initSimulator(self, replication, machine):
        # read all input data
        self.readInputCSV()
        # For automated model excution once uploaded to a cloud-based computing platform
        self.run_dir = "run" + str(replication) + str(machine)
        
        # Call OSMProcessor to get relevant map data
        self.OSMProcessor = OSMProcessor(self.address)
        
        # Call relevant OSMProcessor functions in order
        # Extract all relevant map data and setup the node, edges, intersection, buildings dataset
        # locationDrive is the overall container of all map data
        self.OSMProcessor.setLocationDrive()
        print("Network geometry extracted")
        # features include nodes, edges and such
        self.OSMProcessor.setNetworkFeature()
        print("Network features extracted")
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

        x0, x1, y0, y1 = self.mapDS.boundMeters  
        xLength = abs(float(x1) - float(x0))
        yLength = abs(float(y1) - float(y0))
        
        print("network X length: ", xLength)
        print("network Y length: ", yLength)
        
        self.cellTracker.initialCut(xLength, yLength)
        
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
        print("Guidance candidates list: ", self.guidanceDS.guidanceCanList)
        self.shelterDS.shelterCanList = self.mapDS.shelterCanList
        print("Shelter candidates list: ", self.shelterDS.shelterCanList)

        self.guidanceDS.pointPerCell(self.cellTracker, self.cellX, self.cellY)
        self.guidanceDS.initGuidance()
        print("Initial guidance installations completed!")
        print("Guidance Points list: ", self.guidanceDS.guidanceList)
                
        self.shelterDS.shelterPerCell(self.cellTracker, self.cellX, self.cellY)
        self.shelterDS.initShelter()
        print("Initial shelter installations completed!")
        print("Shelters list: ", self.shelterDS.shelterList)
        
        self.hazardDS.initHazard(self.mapDS, self.cellTracker)
        print("Hazard generation completed!")
        
        self.pedDS.initPedestrianAgent(self.mapDS, self.cellTracker, self.maxSpeed)
        print("Pedestrian generation completed!")
        
        self.guidanceDS.guidanceByOSMID = {gu.nodeMapped.OSMID: gu for gu in self.guidanceDS.guidanceList.values()}  
        self.shelterDS.shelterByOSMID   = {sh.nodeMapped.OSMID: sh for sh in self.shelterDS.shelterList.values()}    
        
        self.pedDS.checkReady(mapDS = self.mapDS,
                              cellTracker = self.cellTracker,
                              maxSpeed = self.maxSpeed,
                              hazardDS = self.hazardDS,
                              shelterDS = self.shelterDS,
                              guidanceDS = self.guidanceDS,
                              forceTracker = self.forceTracker)
        print("Wire competency checked!")

        """!!! Currently stuck here !!!"""
        self.rl = RLBridge(self, mode = "simple")
        self.logger = trainingLog(run_dir = self.run_dir, window = 100, use_tensorboard = True)
        
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
                
            tmr = Timer()
                
            self.pedDS.startDocument()
            tmr.lap("startDocument")

            self.hazardDS.spreadUpdate()
            tmr.lap("hazard.spreadUpdate")

            self.hazardDS.heatUpdate()
            tmr.lap("hazard.heatUpdate")

            self.hazardDS.smokeUpdate()
            tmr.lap("hazard.smokeUpdate")

            # === PED/GU/SH LOOKUPS ===
            self.pedDS.loadGuShLookup(self.guidanceDS.guidanceByOSMID, self.shelterDS.shelterByOSMID)
            tmr.lap("ped.loadGuShLookup")

            # === PEDESTRIAN INTERACTIONS ===
            self.pedDS.pedestrianHazardInteraction()
            tmr.lap("ped.hazardInteraction")

            self.pedDS.interPedestrianInteraction()
            tmr.lap("ped.interPedInteraction")

            self.pedDS.pedestrianNetworkInteraction()
            tmr.lap("ped.networkInteraction")

            # === CELL UPDATE ===
            self.cellTracker.cellUpdate(pedDS=self.pedDS, forceTracker=self.forceTracker)
            tmr.lap("cellTracker.cellUpdate")

            # === RL ===
            rl_out = self.rl.step()
            tmr.lap("rl.step")

            # === DOCU/LOG ===
            self.pedDS.docuStatus()
            tmr.lap("ped.docuStatus")
            
            cumuResult = self.pedDS.result
            metrics = {
                "arrival": cumuResult.get("arrival", 0),
                "casualty": cumuResult.get("casualty", 0),
                "evacuated": cumuResult.get("evacuated", 0),
                "guided": cumuResult.get("guided", 0),
                "affected": cumuResult.get("affected", 0),
                "added_shelters": rl_out.get("added_shelters", 0),
                "added_guidances": rl_out.get("added_guidances", 0)
            }
            
            self.logger.log_step(t=time, reward=float(rl_out["reward"]), metrics=metrics)
            
            if getattr(self.rl, "debug", False) and (time % int(getattr(self.rl, "print_every", 1)) == 0):
                print(f"[CORE] t={time} | reward={rl_out['reward']:.3f} | "
                      f"arr={metrics['arrival']} cas={metrics['casualty']} evac={metrics['evacuated']} "
                      f"guided={metrics['guided']} affected={metrics['affected']} | "
                      f"added_sh={metrics['added_shelters']} added_gu={metrics['added_guidances']}")
            tmr.lap("logger.log_step")
       
            for name in ["countByCell","avgVelocityByCell","heatByCell","smokeByCell",
                "dangerLevelByCell","shelterFulfillByCell","guidanceInterByCell","wellnessPenaltyByCell"]:
                arr = getattr(self.cellTracker, name, None)
                if arr is None or len(np.asarray(arr).reshape(-1)) != (self.cellX * self.cellY):
                    print(f"[WIRE CHECK] {name} missing or wrong length")
        if self.rl is not None and hasattr(self.rl, "end_episode"):
            self.rl.end_episode()
    