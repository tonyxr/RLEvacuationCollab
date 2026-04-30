#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:16:55 2025

@author: Xiaoru Shi
"""

import os
import glob
import zipfile
import pandas as pd
import numpy as np

from Core import Core

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")

def _runs_path(*parts):
    return os.path.join(RUNS_ROOT, *parts)

def _is_colab_runtime() -> bool:
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False
    
def _collect_existing_paths(paths):
    seen = set()
    existing = []
    missing = []
    for path in paths:
        if not path:
            continue
        abspath = os.path.abspath(path)
        if abspath in seen:
            continue
        seen.add(abspath)
        if os.path.exists(abspath):
            existing.append(abspath)
        else:
            missing.append(abspath)
    return existing, missing

def _build_outputs_bundle(paths, bundle_name: str = "colab_outputs_bundle.zip"):
    downloadables, missing = _collect_existing_paths(paths)
    bundle_path = _runs_path(bundle_name)
    os.makedirs(os.path.dirname(bundle_path), exist_ok = True)

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abspath in downloadables:
            arcname = os.path.relpath(abspath, PROJECT_ROOT)
            zf.write(abspath, arcname = arcname)
        manifest = [
            "# RLEvacuationCollab bundle manifest",
            f"project_root={PROJECT_ROOT}",
            f"included_files={len(downloadables)}",
            f"missing_files={len(missing)}",
            "",
            "[included]",
            *downloadables,
            "",
            "[missing]",
            *missing,
            "",
        ]
        zf.writestr("bundle_manifest.txt", "\n".join(manifest))

    return bundle_path, downloadables, missing

def _download_files_if_colab(paths):
    bundle_path, downloadables, missing = _build_outputs_bundle(paths)
    print("[OUTPUT] Bundle path:", bundle_path)
    if missing:
        print(f"[OUTPUT] Skipped {len(missing)} missing files while creating bundle.")
    if not _is_colab_runtime():
        print("[OUTPUT] Non-Colab runtime detected; bundle created for manual use.")
        return
    try:
        from google.colab import files
        from IPython import get_ipython
    except Exception as exc:
        print(f"[COLAB] Could not import google.colab.files: {exc}")
        return

    print("[COLAB] To download manually in a Colab code cell, run:")
    print("from google.colab import files")
    print(f"files.download(r'{bundle_path}')")

    ip = get_ipython()
    if ip is None or getattr(ip, "kernel", None) is None:
        print("[COLAB] No active IPython kernel; created a bundle instead of browser downloads.")
        return

    for abspath in downloadables:
        print(f"[COLAB] Download prompt for: {abspath}")
        try:
            files.download(abspath)
        except Exception as exc:
            print(f"[COLAB] Failed to trigger download for {abspath}: {exc}")

def _aggregate_phase_curves(phase: str, metrics = None, out_name: str = "summary_metrics.png"):
    if metrics is None:
        metrics = ["casualty", "evacuated", "arrival"]
    
    files = sorted(glob.glob(_runs_path(phase, "rep_*", "progress.csv")))
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
    
    out_dir = _runs_path(phase)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "summary_metrics.csv")
    mean_df.to_csv(out_csv, index=False)
    
    return out_csv

def _aggregate_strategy_curves(base_phase: str, strategy: str, metrics = None):
    phase = os.path.join(base_phase, strategy)
    return _aggregate_phase_curves(phase, metrics = metrics, out_name = f"{strategy}_summary_metrics.png")



def _plot_strategy_comparison_curves(base_phase: str, strategies, metrics = None, out_name: str = "strategy_comparison.png"):
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["casualty", "mean_evacuation_time", "total_shelter_capacity", "shelter_utilization"]

    curve_by_strategy = {}
    for strategy in strategies:
        csv_path = _runs_path(base_phase, strategy, "summary_metrics.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        keep = ["timestep"] + [m for m in metrics if m in df.columns]
        if len(keep) <= 1:
            continue
        curve_by_strategy[strategy] = df[keep].copy()

    if not curve_by_strategy:
        return None

    plot_metrics = [m for m in metrics if any(m in df.columns for df in curve_by_strategy.values())]
    if not plot_metrics:
        return None

    fig, axes = plt.subplots(len(plot_metrics), 1, figsize = (10, 4 * len(plot_metrics)), sharex = True)
    if len(plot_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, plot_metrics):
        for strategy, df in curve_by_strategy.items():
            if metric not in df.columns:
                continue
            ax.plot(df["timestep"], df[metric], label = strategy)
        ax.set_title(f"{metric} (mean over replications)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha = 0.3)
        ax.legend()

    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"Strategy comparison ({base_phase})", y = 1.02)
    fig.tight_layout()

    out_path = _runs_path(base_phase, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok = True)
    fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    return out_path

def _plot_pairwise_metric_groups(base_phase: str, groups, metrics, out_dir_name: str = "policy_pairwise_groups",
                                 x_min: int = 0, x_max: int = 240):
    import matplotlib.pyplot as plt

    out_dir = _runs_path(base_phase, out_dir_name)
    os.makedirs(out_dir, exist_ok = True)
    saved_paths = []

    for group_name, strategies in groups.items():
        curve_by_strategy = {}
        for strategy in strategies:
            csv_path = _runs_path(base_phase, strategy, "summary_metrics.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            keep = ["timestep"] + [m for m in metrics if m in df.columns]
            if len(keep) <= 1:
                continue
            curve_by_strategy[strategy] = df[keep].copy()

        if not curve_by_strategy:
            continue

        fig, axes = plt.subplots(len(metrics), 1, figsize = (10, 4 * len(metrics)), sharex = True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            for strategy, df in curve_by_strategy.items():
                if metric not in df.columns:
                    continue
                ax.plot(df["timestep"], df[metric], label = strategy)
            ax.set_title(f"{group_name}: {metric} (mean over 12 replications)")
            ax.set_ylabel(metric)
            ax.grid(True, alpha = 0.3)
            ax.legend()
            ax.set_xlim(x_min, x_max)

        axes[-1].set_xlabel("timestep")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{group_name}.png")
        fig.savefig(out_path, dpi = 150, bbox_inches = "tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths

def _episode_summary(base_phase: str, strategy: str):
    files = sorted(glob.glob(_runs_path(base_phase, strategy, "rep_*", "progress.csv")))
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

    # (b) Policy comparison: RL vs random vs heuristic vs all shelters installed at t=0
    compare_phase = "eval_compare"
    compare_strategies = ["rl", "random", "heuristic", "initial_only"]
    compare_pngs = {}
    for strategy in compare_strategies:
        _run_replications(machine, eval_replications, compare_phase, strategy, False, overrides)
        compare_pngs[strategy] = _aggregate_strategy_curves(
            compare_phase,
            strategy,
            metrics = ["casualty", "evacuated", "mean_evacuation_time", "shelter_utilization"],
        )
        
    compare_overlay_png = _plot_strategy_comparison_curves(
        compare_phase,
        compare_strategies,
        metrics = ["casualty", "evacuated", "mean_evacuation_time", "shelter_utilization"],
        out_name = "strategy_comparison_overlay.png",
    )
    
    pairwise_groups = {
        "group_1_rl_vs_one_time_installation": ["rl", "initial_only"],
        "group_2_rl_vs_random": ["rl", "random"],
        "group_3_rl_vs_heuristic": ["rl", "heuristic"],
        "group_4_all_policies": ["rl", "initial_only", "random", "heuristic"],
    }
    pairwise_group_pngs = _plot_pairwise_metric_groups(
        compare_phase,
        pairwise_groups,
        metrics = ["casualty", "evacuated", "mean_evacuation_time", "shelter_utilization"],
        out_dir_name = "policy_pairwise_groups",
        x_min = 0,
        x_max = 240,
    )

    # Summaries for tables
    compare_tables = []
    frame_list = []
    for strategy in compare_strategies:
        df = _episode_summary(compare_phase, strategy)
        if df is not None:
            df["phase"] = compare_phase
            frame_list.append(df)
    if frame_list:
        big = pd.concat(frame_list, ignore_index = True)
        phase_out = _runs_path(compare_phase, "strategy_summary_by_replication.csv")
        os.makedirs(os.path.dirname(phase_out), exist_ok = True)
        big.to_csv(phase_out, index = False)

        mean_df = big.groupby(["phase", "strategy"], as_index = False).mean(numeric_only = True)
        mean_out = _runs_path(compare_phase, "strategy_summary_mean.csv")
        mean_df.to_csv(mean_out, index = False)
        compare_tables.append(mean_out)

        perf_cols = ["strategy", "final_casualty", "final_evacuated", "mean_evacuation_time"]
        printable = mean_df[perf_cols].sort_values(by = "strategy").reset_index(drop = True)
        print("\n[POLICY COMPARISON] Mean performance over replications")
        print(printable.to_string(index = False))


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
    print("Comparison overlay graph:", compare_overlay_png)
        
    
    _download_files_if_colab([
        train_png,
        compare_pngs.get("rl"),
        compare_pngs.get("random"),
        compare_pngs.get("heuristic"),
        compare_pngs.get("initial_only"),
        _runs_path("train", "rl", "summary_metrics.csv"),
        _runs_path(compare_phase, "rl", "summary_metrics.csv"),
        _runs_path(compare_phase, "random", "summary_metrics.csv"),
        _runs_path(compare_phase, "heuristic", "summary_metrics.csv"),
        _runs_path(compare_phase, "initial_only", "summary_metrics.csv"),
        compare_overlay_png,
        *pairwise_group_pngs,
    ] + compare_tables)
        
if __name__ == "__main__":
    Script()