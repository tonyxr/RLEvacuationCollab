#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:16:55 2025

@author: Xiaoru Shi
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from Core import Core

def _aggregate_phase_curves(phase: str, metrics = None, out_name: str = "summary_metrics.png"):
    if metrics is None:
        metrics = ["casualty", "evacuated", "arrival"]
    
    files = sorted(glob.glob(os.path.join("runs", phase, "rep_*", "progress.csv")))
    if not files:
        return None
    
    merged = []
    for f in files:
        df = pd.read_csv(f)
        keep = ["timestep"] + [m for m in metrics if m in df.columns]
        if len(keep) <= 1:
            continue
        merged.append(df[keep].copy())
    
    if not merged:
        return None
    
    cat = pd.concat(merged, ignore_index=True)
    mean_df = cat.groupby("timestep", as_index=False).mean(numeric_only=True)
    
    out_dir = os.path.join("runs", phase)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "summary_metrics.csv")
    mean_df.to_csv(out_csv, index=False)
    
    plt.figure()
    for m in metrics:
        if m in mean_df.columns:
            plt.plot(mean_df["timestep"], mean_df[m], label=m)
    plt.xlabel("timestep")
    plt.ylabel("mean count across replications")
    plt.title(f"{phase.upper()} mean curves")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(out_dir, out_name)
    plt.savefig(out_png)
    plt.close()
    return out_png


def Script():
    machine = "a"
    train_replications = 100
    eval_replications = 20
    
    # user-requested fixed scenario
    overrides = {
        "address": "Reading, PA, USA",
        "pedVol": 2000,
    }
    
    for r in range(1, train_replications + 1):
        print(f"\n=== TRAIN replication {r}/{train_replications} ===")
        core = Core(machine)
        core.initSimulator(replication = r, machine = machine, config_overrides = overrides, phase = "train", train_mode = True)
    
    for r in range(1, eval_replications + 1):
        print(f"\n=== EVAL replication {r}/{eval_replications} ===")
        core = Core(machine)
        core.initSimulator(replication = r, machine = machine, config_overrides = overrides, phase = "eval", train_mode = False)
    
    train_png = _aggregate_phase_curves("train", out_name = "training_log_graph.png")
    eval_png = _aggregate_phase_curves("eval", out_name = "evaluation_log_graph.png")
    
    print("\n=== Completed ===")
    print("Training summary graph:", train_png)
    print("Evaluation summary graph:", eval_png)
        
if __name__ == "__main__":
    Script()