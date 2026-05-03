#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:16:55 2025

@author: Xiaoru Shi
"""

import os
import glob
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from Core import Core

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")

def _runs_path(*parts):
    return os.path.join(RUNS_ROOT, *parts)

def _save_svg_line_plot(series_map: dict, title: str, x_label: str, y_label: str, out_path: str, x_min: int, x_max: int):
    width, height = 1000, 420
    ml, mr, mt, mb = 70, 20, 40, 55
    pw, ph = width - ml - mr, height - mt - mb
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    all_y = []
    for _, ys, _ in series_map.values():
        all_y.extend([float(v) for v in ys if pd.notna(v)])
    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 0.05 * (y_max - y_min)
    y_min -= pad
    y_max += pad
    def sx(x): return ml + (float(x - x_min) / max(1e-9, (x_max - x_min))) * pw
    def sy(y): return mt + (1.0 - float(y - y_min) / max(1e-9, (y_max - y_min))) * ph
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    lines.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16">{title}</text>')
    lines.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>')
    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>')
    for i in range(6):
        xv = x_min + (x_max - x_min) * i / 5
        xpix = sx(xv)
        lines.append(f'<line x1="{xpix}" y1="{mt+ph}" x2="{xpix}" y2="{mt+ph+5}" stroke="#333"/>')
        lines.append(f'<text x="{xpix}" y="{mt+ph+20}" text-anchor="middle" font-size="11">{int(round(xv))}</text>')
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        ypix = sy(yv)
        lines.append(f'<line x1="{ml-5}" y1="{ypix}" x2="{ml}" y2="{ypix}" stroke="#333"/>')
        lines.append(f'<text x="{ml-8}" y="{ypix+4}" text-anchor="end" font-size="11">{yv:.2f}</text>')
    lines.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="12">{x_label}</text>')
    lines.append(f'<text x="18" y="{height/2}" transform="rotate(-90 18,{height/2})" text-anchor="middle" font-size="12">{y_label}</text>')
    for idx, (name, (xs, ys, _)) in enumerate(series_map.items()):
        pts = []
        for x, y in zip(xs, ys):
            if pd.isna(y):
                continue
            pts.append(f"{sx(x):.2f},{sy(y):.2f}")
        if pts:
            lines.append(f'<polyline fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2" points="{" ".join(pts)}"/>')
    lx, ly = ml + 10, mt + 10
    for idx, name in enumerate(series_map.keys()):
        c = colors[idx % len(colors)]
        y = ly + idx * 18
        lines.append(f'<line x1="{lx}" y1="{y}" x2="{lx+18}" y2="{y}" stroke="{c}" stroke-width="2"/>')
        lines.append(f'<text x="{lx+24}" y="{y+4}" font-size="12">{name}</text>')
    lines.append("</svg>")
    with open(out_path, "w", encoding = "utf-8") as fh:
        fh.write("\n".join(lines))

def _is_colab_runtime() -> bool:
    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False
    

def _print_graph_outputs(group_metric_pngs: dict):
    print("\n[GRAPH OUTPUTS] 4 groups x 4 graphs each (printed below)")
    ordered_group_names = sorted(group_metric_pngs.keys())
    for idx, group_name in enumerate(ordered_group_names, start = 1):
        metric_map = group_metric_pngs[group_name]
        print(f"\nGroup {idx}: {group_name}")
        for metric_name, path in metric_map.items():
            print(f"  - {metric_name}: {path}")
        print(f"  -> graph_count={len(metric_map)}")
    if not _is_colab_runtime():
        return
    try:
        from IPython.display import Image, SVG, display
    except Exception as exc:
        print(f"[GRAPH OUTPUTS] Could not import IPython display: {exc}")
        return
    
    for idx, group_name in enumerate(ordered_group_names, start = 1):
        print(f"\n=== Group {idx} inline display: {group_name} ===")
        for metric_name, path in group_metric_pngs[group_name].items():
            print(f"[{metric_name}] {path}")
            ext = os.path.splitext(path)[1].lower()
            if ext == ".svg":
                display(SVG(filename = path))
            else:
                display(Image(filename = path))
            
def _read_progress_csv(path: str):
    """Read progress logs robustly across schema revisions and malformed rows."""
    import csv

    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return pd.DataFrame()

        rows = []
        has_reward_raw = "reward_raw" in header
        if ("reward" in header) and (not has_reward_raw):
            reward_idx = header.index("reward")
        else:
            reward_idx = -1
        width = len(header)

        for row in reader:
            if not row:
                continue
            if len(row) == width + 1 and reward_idx >= 0:
                # Legacy header missing reward_raw, but row contains it.
                row = row[: reward_idx + 1] + row[reward_idx + 2 :]
            elif len(row) < width:
                row = row + [""] * (width - len(row))
            elif len(row) > width:
                row = row[:width]
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df

def _aggregate_phase_curves(phase: str, metrics = None, out_name: str = "summary_metrics.png"):
    if metrics is None:
        metrics = ["casualty", "evacuated", "arrival"]
    
    files = sorted(glob.glob(_runs_path(phase, "rep_*", "progress.csv")))
    if not files:
        return None
    
    merged = []
    for f in files:
        df = _read_progress_csv(f)
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
    
    plot_cols = [m for m in metrics if m in mean_df.columns]
    out_png = None
    if plot_cols:
        if plt is None:
            return out_csv
        fig, axes = plt.subplots(len(plot_cols), 1, figsize = (10, 4 * len(plot_cols)), sharex = True)
        if len(plot_cols) == 1:
            axes = [axes]
        for ax, metric in zip(axes, plot_cols):
            ax.plot(mean_df["timestep"], mean_df[metric], label = metric)
            ax.set_title(f"{metric} (mean over replications)")
            ax.set_ylabel(metric)
            ax.grid(True, alpha = 0.3)
            ax.legend()
        axes[-1].set_xlabel("timestep")
        fig.tight_layout()
        out_png = os.path.join(out_dir, out_name)
        fig.savefig(out_png, dpi = 150, bbox_inches = "tight")
        plt.close(fig)
    
    return out_png if out_png is not None else out_csv

def _aggregate_strategy_curves(base_phase: str, strategy: str, metrics = None):
    phase = os.path.join(base_phase, strategy)
    return _aggregate_phase_curves(phase, metrics = metrics, out_name = f"{strategy}_summary_metrics.png")

def _plot_strategy_comparison_curves(base_phase: str, strategies, metrics = None, out_name: str = "strategy_comparison.png"):
    if metrics is None:
        metrics = ["casualty", "mean_evacuation_time", "total_shelter_capacity", "shelter_utilization"]
    
    if plt is None:
        return None
    
    curve_by_strategy = {}
    for strategy in strategies:
        csv_path = _runs_path(base_phase, strategy, "summary_metrics.csv")
        if not os.path.exists(csv_path):
            continue
        df = _read_progress_csv(csv_path)
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
    out_dir = _runs_path(base_phase, out_dir_name)
    os.makedirs(out_dir, exist_ok = True)
    saved_paths = []
    grouped_metric_pngs = {}

    for group_name, strategies in groups.items():
        curve_by_strategy = {}
        for strategy in strategies:
            csv_path = _runs_path(base_phase, strategy, "summary_metrics.csv")
            if not os.path.exists(csv_path):
                continue
            df = _read_progress_csv(csv_path)
            keep = ["timestep"] + [m for m in metrics if m in df.columns]
            if len(keep) <= 1:
                continue
            curve_by_strategy[strategy] = df[keep].copy()

        if not curve_by_strategy:
            continue

        # Individual metric files: 4 graphs per group (SVG fallback avoids matplotlib dependency)
        metric_dir = os.path.join(out_dir, group_name)
        os.makedirs(metric_dir, exist_ok = True)
        grouped_metric_pngs[group_name] = {}
        for metric in metrics:
            series_map = {}
            for strategy, df in curve_by_strategy.items():
                if metric not in df.columns:
                    continue
                series_map[strategy] = (df["timestep"].to_numpy(), df[metric].to_numpy(), strategy)
            if not series_map:
                continue
            metric_path = os.path.join(metric_dir, f"{metric}.svg")
            _save_svg_line_plot(
                series_map,
                title = f"{group_name}: {metric} (mean over 12 replications)",
                x_label = "timestep",
                y_label = metric,
                out_path = metric_path,
                x_min = x_min,
                x_max = x_max,
            )
            saved_paths.append(metric_path)
            grouped_metric_pngs[group_name][metric] = metric_path

    return saved_paths, grouped_metric_pngs

def _episode_summary(base_phase: str, strategy: str):
    files = sorted(glob.glob(_runs_path(base_phase, strategy, "rep_*", "progress.csv")))
    rows = []
    for f in files:
        df = _read_progress_csv(f)
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
    pairwise_group_pngs, pairwise_group_metric_pngs = _plot_pairwise_metric_groups(
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
    
    expected_groups = 4
    expected_graphs_per_group = 4
    actual_groups = len(pairwise_group_metric_pngs)
    if actual_groups != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} groups, got {actual_groups}.")
    for group_name, metric_map in pairwise_group_metric_pngs.items():
        if len(metric_map) != expected_graphs_per_group:
            raise RuntimeError(
                f"Expected {expected_graphs_per_group} metric plots for {group_name}, got {len(metric_map)}."
            )
    _print_graph_outputs(pairwise_group_metric_pngs)
        
if __name__ == "__main__":
    Script()