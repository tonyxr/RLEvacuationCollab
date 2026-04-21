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
import numpy as np

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

def _aggregate_strategy_curves(base_phase: str, strategy: str, metrics = None):
    phase = os.path.join(base_phase, strategy)
    return _aggregate_phase_curves(phase, metrics = metrics, out_name = f"{strategy}_summary_metrics.png")

def _episode_summary(base_phase: str, strategy: str):
    files = sorted(glob.glob(os.path.join("runs", base_phase, strategy, "rep_*", "progress.csv")))
    rows = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            continue
        last = df.iloc[-1]
        rows.append({
            "strategy": strategy,
            "replication": os.path.basename(os.path.dirname(f)),
            "final_reward_ma": float(last.get("reward_ma_window", np.nan)),
            "final_casualty": float(last.get("casualty", np.nan)),
            "final_evacuated": float(last.get("evacuated", np.nan)),
            "final_arrival": float(last.get("arrival", np.nan)),
            "mean_evacuation_time": float(last.get("mean_evacuation_time", np.nan)),
        })
    if not rows:
        return None
    return pd.DataFrame(rows)

def _run_replications(machine: str, replications: int, phase: str, strategy: str,
                      train_mode: bool, overrides: dict):
    for r in range(1, replications + 1):
        print(f"\n=== {phase.upper()} [{strategy}] replication {r}/{replications} ===")
        core = Core(machine)
        core.initSimulator(
            replication = r,
            machine = machine,
            config_overrides = overrides,
            phase = phase,
            train_mode = train_mode,
            deployment_strategy = strategy,
            run_tag = strategy,
        )

def Script():
    machine = "a"
    train_replications = 30
    eval_replications = 12
    
    # user-requested fixed scenario
    overrides = {
    }
    
    # (a) RL convergence diagnostics (reward logs + trajectory graph)
    _run_replications(machine, train_replications, "train", "rl", True, overrides)
    train_png = _aggregate_strategy_curves("train", "rl", metrics = ["reward", "reward_ma_window", "casualty", "evacuated"])

    # (b) RL temporal shelters vs initial shelters only
    _run_replications(machine, eval_replications, "eval_b", "rl", False, overrides)
    _run_replications(machine, eval_replications, "eval_b", "none", False, overrides)
    _aggregate_strategy_curves("eval_b", "rl", metrics = ["casualty", "evacuated", "mean_evacuation_time"])
    _aggregate_strategy_curves("eval_b", "none", metrics = ["casualty", "evacuated", "mean_evacuation_time"])

    # (c) Strategy comparison RL vs random vs heuristic
    _run_replications(machine, eval_replications, "eval_c", "rl", False, overrides)
    _run_replications(machine, eval_replications, "eval_c", "random", False, overrides)
    _run_replications(machine, eval_replications, "eval_c", "heuristic", False, overrides)
    _aggregate_strategy_curves("eval_c", "rl", metrics = ["casualty", "evacuated", "mean_evacuation_time"])
    _aggregate_strategy_curves("eval_c", "random", metrics = ["casualty", "evacuated", "mean_evacuation_time"])
    _aggregate_strategy_curves("eval_c", "heuristic", metrics = ["casualty", "evacuated", "mean_evacuation_time"])

    # Summaries for tables
    compare_tables = []
    for phase in ["eval_b", "eval_c"]:
        strategies = ["rl", "none"] if phase == "eval_b" else ["rl", "random", "heuristic"]
        frame_list = []
        for s in strategies:
            df = _episode_summary(phase, s)
            if df is not None:
                df["phase"] = phase
                frame_list.append(df)
        if frame_list:
            big = pd.concat(frame_list, ignore_index = True)
            phase_out = os.path.join("runs", phase, "strategy_summary_by_replication.csv")
            os.makedirs(os.path.dirname(phase_out), exist_ok = True)
            big.to_csv(phase_out, index = False)

            mean_df = big.groupby(["phase", "strategy"], as_index = False).mean(numeric_only = True)
            mean_out = os.path.join("runs", phase, "strategy_summary_mean.csv")
            mean_df.to_csv(mean_out, index = False)
            compare_tables.append(mean_out)

    # Simple convergence check output
    convergence_df = _episode_summary("train", "rl")
    if convergence_df is not None and not convergence_df.empty:
        early = convergence_df["final_reward_ma"].head(max(1, len(convergence_df)//3)).mean()
        late = convergence_df["final_reward_ma"].tail(max(1, len(convergence_df)//3)).mean()
        print(f"[CONVERGENCE] early_mean_reward_ma={early:.4f} late_mean_reward_ma={late:.4f}")
    
    print("\n=== Completed ===")
    print("Training summary graph:", train_png)
    for p in compare_tables:
        print("Comparison summary table:", p)
        
if __name__ == "__main__":
    Script()