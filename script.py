#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:16:55 2025

@author: Xiaoru Shi
"""

import torch

from RLBridge import RLBridge
from Core import Core

def Script(machine):
    core = Core(machine)
    
    rl = RLBridge(core,
                  mode = "simple",
                  reward_interval = 5,
                  debug = True)
    
    num_episodes = 50
    
    for ep in range(num_episodes):
        core.reset()
        done = False
        
        while not done:
            core.step()
            done = core.is_done()
            
        rl.end_episode()
        print(r"Episode {ep} finished")
    
if __name__ == '__main__':
    print("Enter the number of replications")
    replications = 5
    print("Enter machine index")
    machine = "a"
    simulator = Script(machine)
    for i in range(0, replications):
        simulator.initSimulator(i, machine)