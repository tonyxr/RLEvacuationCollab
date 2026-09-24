#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 14:28:04 2025

@author: Xiaoru Shi
"""

import os
import csv
from collections import deque
from typing import Dict

CSV_COLUMNS = [
    "timestep",
    "reward",
    "episode_return",
    "reward_safe",
    "reward_casualty",
    "reward_evacuation_time",
    "reward_hazard_exposure",
    "reward_risk_time",
    "reward_shelter_service",
    "new_safe_completions",
    "new_casualties",
    "attributed_shelter_service",
    "risk_weighted_person_time",
    "active_person_time",
    "hazard_exposure_person_time",
    "decision_made",
    "selected_candidate",
    "heuristic_candidate",
    "selected_cell",
    "heuristic_cell",
    "completed_action",
    "feasible_cells",
    "feasible_candidates",
    "remaining_deployments",
    "arrival",
    "casualty",
    "evacuated",
    "unfinished",
    "active_remaining",
    "affected",
    "panic_eligible_first_exposures",
    "panic_onsets",
    "active_panicked",
    "realized_panic_onset_rate",
    "panic_herd_choices",
    "panic_random_choices",
    "added_shelters",
    "capacity_added",
    "rerouted_population",
    "mean_evacuation_time",
    "mean_safe_completion_time",
    "safe_completion_rate",
    "shelter_evacuation_rate",
    "total_shelter_capacity",
    "shelter_utilization",
    "mean_effective_speed_m_per_minute",
    "mean_congestion_speed_ratio",
    "minimum_congestion_speed_ratio",
    "maximum_link_density_ped_per_m2",
    "occupied_physical_links",
    "congested_population",
    "congestion_substeps",
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
            f"and starting a fresh progress.csv with schema v6.",
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
        unfinished       = int(metrics.get("unfinished", 0))
        active_remaining = int(metrics.get("active_remaining", 0))
        affected         = int(metrics.get("affected", 0))
        added_shelters   = int(metrics.get("added_shelters", 0))
        mean_evacuation_time = float(metrics.get("mean_evacuation_time", 0.0))
        mean_safe_completion_time = float(metrics.get("mean_safe_completion_time", 0.0))
        total_shelter_capacity = float(metrics.get("total_shelter_capacity", 0.0))
        safe_completion_rate = float(metrics.get("safe_completion_rate", 0.0))
        shelter_evacuation_rate = float(metrics.get("shelter_evacuation_rate", 0.0))
        shelter_utilization = float(metrics.get("shelter_utilization", 0.0))
        
        self.recent_rewards.append(float(reward))
        reward_ma = self.moving_avg() 
        
        row = [
            int(t),                # timestep
            float(reward),
            float(metrics.get("episode_return", 0.0)),
            float(metrics.get("reward_safe", 0.0)),
            float(metrics.get("reward_casualty", 0.0)),
            float(metrics.get("reward_evacuation_time", 0.0)),
            float(metrics.get("reward_hazard_exposure", 0.0)),
            float(metrics.get("reward_risk_time", 0.0)),
            float(metrics.get("reward_shelter_service", 0.0)),
            int(metrics.get("new_safe_completions", 0)),
            int(metrics.get("new_casualties", 0)),
            int(metrics.get("attributed_shelter_service", 0)),
            float(metrics.get("risk_weighted_person_time", 0.0)),
            float(metrics.get("active_person_time", 0.0)),
            float(metrics.get("hazard_exposure_person_time", 0.0)),
            int(metrics.get("decision_made", 0)),
            int(metrics.get("selected_candidate", -1)),
            int(metrics.get("heuristic_candidate", -1)),
            int(metrics.get("selected_cell", -1)),
            int(metrics.get("heuristic_cell", -1)),
            int(metrics.get("completed_action", -1)),
            int(metrics.get("feasible_cells", 0)),
            int(metrics.get("feasible_candidates", 0)),
            int(metrics.get("remaining_deployments", 0)),
            arrival,
            casualty,
            evacuated,
            unfinished,
            active_remaining,
            affected,
            int(metrics.get("panic_eligible_first_exposures", 0)),
            int(metrics.get("panic_onsets", 0)),
            int(metrics.get("active_panicked", 0)),
            float(metrics.get("realized_panic_onset_rate", 0.0)),
            int(metrics.get("panic_herd_choices", 0)),
            int(metrics.get("panic_random_choices", 0)),
            added_shelters,
            float(metrics.get("capacity_added", 0.0)),
            int(metrics.get("rerouted_population", 0)),
            mean_evacuation_time,
            mean_safe_completion_time,
            safe_completion_rate,
            shelter_evacuation_rate,
            total_shelter_capacity,
            shelter_utilization,
            float(metrics.get("mean_effective_speed_m_per_minute", 0.0)),
            float(metrics.get("mean_congestion_speed_ratio", 1.0)),
            float(metrics.get("minimum_congestion_speed_ratio", 1.0)),
            float(metrics.get("maximum_link_density_ped_per_m2", 0.0)),
            int(metrics.get("occupied_physical_links", 0)),
            int(metrics.get("congested_population", 0)),
            int(metrics.get("congestion_substeps", 1)),
            float(reward_ma),
        ]
        
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        if self.tb is not None:
            self.tb.add_scalar("reward/instant", float(reward), global_step=t)
            self.tb.add_scalar("reward/moving_avg", float(reward_ma), global_step=t)
            self.tb.add_scalar("reward/episode_return", float(metrics.get("episode_return", 0.0)), global_step=t)
            self.tb.add_scalar("reward/safe", float(metrics.get("reward_safe", 0.0)), global_step=t)
            self.tb.add_scalar("reward/casualty", float(metrics.get("reward_casualty", 0.0)), global_step=t)
            self.tb.add_scalar("reward/evacuation_time", float(metrics.get("reward_evacuation_time", 0.0)), global_step=t)
            self.tb.add_scalar("reward/hazard_exposure", float(metrics.get("reward_hazard_exposure", 0.0)), global_step=t)
            self.tb.add_scalar("reward/risk_time", float(metrics.get("reward_risk_time", 0.0)), global_step=t)
            self.tb.add_scalar("reward/shelter_service", float(metrics.get("reward_shelter_service", 0.0)), global_step=t)

            self.tb.add_scalar("ped/arrival", arrival, global_step=t)
            self.tb.add_scalar("ped/casualty", casualty, global_step=t)
            self.tb.add_scalar("ped/evacuated", evacuated, global_step=t)
            self.tb.add_scalar("ped/affected", affected, global_step=t)
            self.tb.add_scalar("ped/panic_onsets", float(metrics.get("panic_onsets", 0.0)), global_step=t)
            self.tb.add_scalar("ped/realized_panic_onset_rate", float(metrics.get("realized_panic_onset_rate", 0.0)), global_step=t)

            self.tb.add_scalar("actions/added_shelters", added_shelters, global_step=t)
            self.tb.add_scalar("ped/mean_evacuation_time", mean_evacuation_time, global_step=t)
            self.tb.add_scalar("ped/mean_safe_completion_time", mean_safe_completion_time, global_step=t)
            self.tb.add_scalar("ped/safe_completion_rate", safe_completion_rate, global_step=t)
            self.tb.add_scalar("ped/shelter_evacuation_rate", shelter_evacuation_rate, global_step=t)
            self.tb.add_scalar("shelter/total_capacity", total_shelter_capacity, global_step=t)
            self.tb.add_scalar("shelter/utilization", shelter_utilization, global_step=t)
            self.tb.add_scalar(
                "congestion/mean_effective_speed_m_per_minute",
                float(metrics.get("mean_effective_speed_m_per_minute", 0.0)),
                global_step=t,
            )
            self.tb.add_scalar(
                "congestion/mean_speed_ratio",
                float(metrics.get("mean_congestion_speed_ratio", 1.0)),
                global_step=t,
            )
            self.tb.add_scalar(
                "congestion/max_density_ped_per_m2",
                float(metrics.get("maximum_link_density_ped_per_m2", 0.0)),
                global_step=t,
            )
    
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
        import pandas as pd
        
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
        plt.xlabel("timestep")
        plt.ylabel("reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, out_name))
        plt.close()

    def plot_metrics_png(self, out_name: str = "core_metrics_curve.png", metric_cols = None):
        import matplotlib.pyplot as plt
        import pandas as pd
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
