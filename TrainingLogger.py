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

import pandas as pd

CSV_COLUMNS = [
    "timestep",
    "reward",
    "reward_raw",
    "arrival",
    "casualty",
    "evacuated",
    "guided",
    "affected",
    "added_shelters",
    "mean_evacuation_time",
    "total_shelter_capacity",
    "shelter_utilization",
    "reward_ma_window",
]

class trainingLog:
    def __init__(self, run_dir: str = "runs/default", window: int = 100, use_tensorboard: bool = False):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok = True)
        
        self.window = int(window)
        self.recent_rewards = deque(maxlen=self.window)
        self.csv_path = os.path.join(self.run_dir, "progress.csv")
        
        self._ensure_csv_schema()
        
        self._csv_new_file = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline = "")
        self.csv_writer = csv.writer(self.csv_file)
        
        if self._csv_new_file:
            self.csv_writer.writerow(CSV_COLUMNS)
            self.csv_file.flush()
        
        self.tb = None
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(self.run_dir)
            
    def _ensure_csv_schema(self):
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, "r", newline = "") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
        except Exception:
            return

        if header == CSV_COLUMNS:
            return

        legacy_path = f"{self.csv_path}.legacy"
        idx = 1
        while os.path.exists(legacy_path):
            idx += 1
            legacy_path = f"{self.csv_path}.legacy{idx}"
        os.replace(self.csv_path, legacy_path)
        print(
            f"[trainingLog] Detected CSV schema mismatch. Archived legacy file to {legacy_path} "
            f"and starting a fresh progress.csv with schema v2.",
            flush=True,
        )
                
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
        """
        
        arrival          = int(metrics.get("arrival", 0))         
        casualty         = int(metrics.get("casualty", 0))
        evacuated        = int(metrics.get("evacuated", 0))
        guided           = int(metrics.get("guided", 0))
        affected         = int(metrics.get("affected", 0))
        added_shelters   = int(metrics.get("added_shelters", 0))
        mean_evacuation_time = float(metrics.get("mean_evacuation_time", 0.0))
        total_shelter_capacity = float(metrics.get("total_shelter_capacity", 0.0))
        shelter_utilization = float(metrics.get("shelter_utilization", 0.0))
        reward_raw = float(metrics.get("reward_raw", reward))
        
        self.recent_rewards.append(float(reward))
        reward_ma = self.moving_avg() 
        
        row = [
            int(t),                # timestep
            float(reward),         # normalized reward for RL convergence curve
            reward_raw,            # raw reward before normalization
            arrival,
            casualty,
            evacuated,
            guided,
            affected,
            added_shelters,
            mean_evacuation_time,
            total_shelter_capacity,
            shelter_utilization,
            float(reward_ma),      # moving average window
        ]
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        if self.tb is not None:
            self.tb.add_scalar("reward/instant", float(reward), global_step=t)
            self.tb.add_scalar("reward/moving_avg", float(reward_ma), global_step=t)
            self.tb.add_scalar("reward/raw", reward_raw, global_step=t)

            self.tb.add_scalar("ped/arrival", arrival, global_step=t)
            self.tb.add_scalar("ped/casualty", casualty, global_step=t)
            self.tb.add_scalar("ped/evacuated", evacuated, global_step=t)
            self.tb.add_scalar("ped/affected", affected, global_step=t)

            self.tb.add_scalar("actions/added_shelters", added_shelters, global_step=t)
            self.tb.add_scalar("ped/mean_evacuation_time", mean_evacuation_time, global_step=t)
            self.tb.add_scalar("shelter/total_capacity", total_shelter_capacity, global_step=t)
            self.tb.add_scalar("shelter/utilization", shelter_utilization, global_step=t)
    
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
        import matplotlib.pyplot as plt
        
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
        plt.plot(df["timestep"], df["reward"], label="reward_norm")
        plt.plot(df["timestep"], df_ma, label=f"reward_norm_ma (w={self.window})")
        if "reward_raw" in df.columns:
            plt.plot(df["timestep"], df["reward_raw"], label="reward_raw", alpha=0.35)
        plt.xlabel("timestep")
        plt.ylabel("reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, out_name))
        plt.close()
        
    def plot_metrics_png(self, out_name: str = "core_metrics_curve.png", metric_cols = None):
        import matplotlib.pyplot as plt
        if metric_cols is None:
            metric_cols = ["casualty", "evacuated", "arrival"]
        
        if not os.path.exists(self.csv_path):
            return
        
        df = pd.read_csv(self.csv_path)
        if "timestep" not in df.columns:
            return
        
        use_cols = [c for c in metric_cols if c in df.columns]
        if not use_cols:
            return
        
        plt.figure()
        for c in use_cols:
            plt.plot(df["timestep"], df[c], label=c)
        
        plt.xlabel("timestep")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, out_name))
        plt.close()
        