#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:16:55 2025

@author: Xiaoru Shi
"""

import torch

from RLBridge import RLBridge
from Core import Core

def Script():
    replications = 5
    machine = "a"
    
    core = Core(machine)
    
    for r in range(replications):
        print(f"\n=== Starting replication {r+1}/{replications} ===")
        
        core.initSimulator(replications, machine)
        
if __name__ == "__main__":
    Script()