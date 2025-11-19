#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:28:04 2025

@author: Xiaoru Shi
"""

import os
import csv
from collections import deque
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import matplotlib.pyplot as plt

class trainingLog:
    def __init__(self, run_dir: str = "runs/default", window: int = 100, use_tensorboard: bool = True):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok = True)
        
        self.window = int(window)
        self.recent_rewards = deque(maxlen=self.window)
        
        self.csv_path = os.path.join(self.run_dir, "progress.csv")
        
        self._csv_new_file = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline = "")
        self.csv_writer = csv.writer(self.csv_file)
        
        if self._csv_new_file:
            self.csv_writer.writerow([
                "timestep",
                "reward",
                "arrival",
                "casualty",
                "evacuated",
                "guided",
                "affected",
                "added_shelters",
                "added_guidances",
                "reward_ma_window"
            ])
            self.csv_file.flush()
        
        self.tb = None
        if use_tensorboard:
            try: 
                self.tb = SummaryWriter(self.run_dir)
            except Exception:
                self.tb = None
                
    def moving_avg(self) -> float:
        if not self.recent_rewards:
            return 0.0
        
        return sum(self.recent_rewards) / len(self.recent_rewards)
                
    def log_step(self, t: int, reward: float, metrics: Dict[str, float]):
        
        """
        the metric is to contain:
            "arrival"
            "casualty"
            "evacuated"
            "affected"
            "added_shelters"
            "added_guidances"
        """
        
        arrival          = int(metrics.get("arrival", 0))         
        casualty         = int(metrics.get("casualty", 0))
        evacuated        = int(metrics.get("evacuated", 0))
        guided           = int(metrics.get("guided", 0))          
        affected         = int(metrics.get("affected", 0))
        added_shelters   = int(metrics.get("added_shelters", 0))
        added_guidances  = int(metrics.get("added_guidances", 0))
        
        self.recent_rewards.append(float(reward))
        reward_ma = self.moving_avg() 
        
        row = [
            int(t),                # timestep
            float(reward),         # reward
            arrival,
            casualty,
            evacuated,
            guided,
            affected,
            added_shelters,
            added_guidances,
            float(reward_ma),      # moving average window
        ]
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        if self.tb is not None:
            self.tb.add_scalar("reward/instant", float(reward), global_step=t)
            self.tb.add_scalar("reward/moving_avg", float(reward_ma), global_step=t)

            self.tb.add_scalar("ped/arrival", arrival, global_step=t)
            self.tb.add_scalar("ped/casualty", casualty, global_step=t)
            self.tb.add_scalar("ped/evacuated", evacuated, global_step=t)
            self.tb.add_scalar("ped/guided", guided, global_step=t)
            self.tb.add_scalar("ped/affected", affected, global_step=t)

            self.tb.add_scalar("actions/added_shelters", added_shelters, global_step=t)
            self.tb.add_scalar("actions/added_guidances", added_guidances, global_step=t)
    
    def close(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        
        if self.tb is not None:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass
            
    def plot_png(self, out_name: str = "reward_curve.png"):
        
        if not os.path.exists(self.csv_path):
            return
        
        df = pd.read_csv(self.csv_path)
        if "reward" not in df.columns or "timestep" not in df.columns:
            return
        
        if "reward_ma_window" in df.columns:
            df_ma = df["reward_ma_window"]
        else:
            N = max(1, self.window)
            df_ma = df["reward"].rolling(window = N, min_periods = 1).mean()
        
        plt.figure()
        plt.plot(df["timestep"], df["reward"], label="reward")
        plt.plot(df["timestep"], df_ma, label=f"reward_ma (w={self.window})")
        plt.xlabel("timestep")
        plt.ylabel("reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, out_name))
        plt.close()
        