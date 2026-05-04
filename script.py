#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:16:55 2025

@author: Xiaoru Shi
"""

import os
import glob
import random
import shutil
import zipfile
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from Core import Core
from RLBridge import set_torch_global_seed

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
        for metric_name in sorted(metric_map.keys()):
            path = metric_map[metric_name]
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
        for metric_name in sorted(group_metric_pngs[group_name].keys()):
            path = group_metric_pngs[group_name][metric_name]
            print(f"[{metric_name}] {path}")
            ext = os.path.splitext(path)[1].lower()
            if ext == ".svg":
                display(SVG(filename = path))
            else:
                display(Image(filename = path))
                
def _validate_graph_outputs(group_metric_pngs: dict, groups: dict, required_metrics):
    expected_group_names = set(groups.keys())
    actual_group_names = set(group_metric_pngs.keys())
    missing_groups = sorted(expected_group_names - actual_group_names)
    extra_groups = sorted(actual_group_names - expected_group_names)
    if missing_groups or extra_groups:
        raise RuntimeError(
            f"Graph group mismatch. missing={missing_groups} extra={extra_groups}"
        )

    actual_groups = len(group_metric_pngs)
    expected_groups = len(groups)
    if actual_groups != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} groups, got {actual_groups}.")

    for group_name, metric_map in group_metric_pngs.items():
        missing_metrics = [m for m in required_metrics if m not in metric_map]
        if missing_metrics:
            raise RuntimeError(f"Missing required metrics for {group_name}: {missing_metrics}")
        for metric_name, path in metric_map.items():
            if not os.path.exists(path):
                raise RuntimeError(f"Missing graph output for {group_name}/{metric_name}: {path}")
            if os.path.getsize(path) == 0:
                raise RuntimeError(f"Empty graph output for {group_name}/{metric_name}: {path}")
            if path.lower().endswith(".svg"):
                with open(path, "r", encoding = "utf-8") as fh:
                    svg_text = fh.read()
                if "<polyline" not in svg_text:
                    raise RuntimeError(f"SVG missing polyline for {group_name}/{metric_name}: {path}")
            
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

def _load_or_refresh_strategy_summary(base_phase: str, strategy: str, metrics):
    """Load per-strategy summary CSV, regenerating from replication logs when possible.

    This avoids plotting stale summary files when logs have changed.
    """
    strategy_phase = os.path.join(base_phase, strategy)
    csv_path = _runs_path(strategy_phase, "summary_metrics.csv")

    # If replication progress files exist, always recompute to stay in sync with logs.
    has_rep_logs = bool(glob.glob(_runs_path(strategy_phase, "rep_*", "progress.csv")))
    if has_rep_logs:
        _aggregate_phase_curves(strategy_phase, metrics = metrics, out_name = f"{strategy}_summary_metrics.png")

    if not os.path.exists(csv_path):
        return None

    df = _read_progress_csv(csv_path)
    keep = ["timestep"] + [m for m in metrics if m in df.columns]
    if len(keep) <= 1:
        return None
    return df[keep].copy()

def _plot_strategy_comparison_curves(base_phase: str, strategies, metrics = None, out_name: str = "strategy_comparison.png"):
    if metrics is None:
        metrics = ["casualty", "mean_evacuation_time", "total_shelter_capacity", "shelter_utilization"]
    
    if plt is None:
        return None
    
    curve_by_strategy = {}
    for strategy in strategies:
        df = _load_or_refresh_strategy_summary(base_phase, strategy, metrics)
        if df is None:
            continue
        curve_by_strategy[strategy] = df

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
            is_rl = str(strategy).strip().lower() == "rl"
            ax.plot(
                df["timestep"],
                df[metric],
                label = strategy,
                linewidth = 2.8 if is_rl else 1.8,
                alpha = 1.0 if is_rl else 0.85,
            )
        if metric == "shelter_utilization":
            ax.set_title("shelter_utilization = occupied_shelter_slots / installed_shelter_capacity (mean over replications)")
            final_vals = []
            for _strategy, _df in curve_by_strategy.items():
                if metric in _df.columns and len(_df[metric]) > 0:
                    final_vals.append(float(_df[metric].iloc[-1]))
            if final_vals and max(final_vals) <= 1e-9:
                ax.text(0.5, 1.02, "Note: all strategies are ~0.0, indicating shelters are not reached (or no capacity installed).", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="darkred")
        else:
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
            df = _load_or_refresh_strategy_summary(base_phase, strategy, metrics)
            if df is None:
                continue
            curve_by_strategy[strategy] = df
            
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
                title = f"{group_name}: {metric} (mean over replications)",
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
            "final_successful_evacuation_rate": float(last.get("successful_evacuation_rate", np.nan)),
            "mean_evacuation_time": float(last.get("mean_evacuation_time", np.nan)),
        })
    if not rows:
        return None
    return pd.DataFrame(rows)

def _plot_training_convergence(train_phase: str, out_name: str = "training_convergence_by_replication.png"):
    df = _episode_summary(train_phase, "rl")
    if df is None or df.empty:
        return None

    # `_episode_summary` exposes replication IDs under `replication` (e.g., "rep_3").
    # Keep plotting resilient by deriving a numeric order even if formatting varies.
    rep_series = df.get("replication", pd.Series(np.arange(1, len(df) + 1), index=df.index))
    rep_numeric = rep_series.astype(str).str.extract(r"(\d+)", expand=False)
    df = df.assign(rep=rep_numeric).sort_values(by="rep", na_position="last").reset_index(drop=True)
    fallback_rep = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=float)
    x = pd.to_numeric(df["rep"], errors="coerce").fillna(fallback_rep).to_numpy(dtype=float)
    y = df["final_reward_ma"].to_numpy(dtype=float)
    win = max(3, int(len(y) * 0.2))
    trend = pd.Series(y).rolling(win, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(x, y, marker="o", linewidth=1.5, alpha=0.75, label="final_reward_ma")
    ax.plot(x, trend, linewidth=2.5, label=f"rolling_mean(w={win})")
    ax.set_title(f"Training convergence ({train_phase})")
    ax.set_xlabel("replication")
    ax.set_ylabel("final_reward_ma")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = _runs_path(train_phase, out_name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out



def _export_launch_package(launch_id: str) -> str:
    launch_dir = _runs_path(launch_id)
    package_path = _runs_path(f"{launch_id}_package.zip")
    os.makedirs(os.path.dirname(package_path), exist_ok = True)
    with zipfile.ZipFile(package_path, "w", compression = zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(launch_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, _runs_path())
                zf.write(full, arcname = rel)
    return package_path


def _export_graph_assets_zip(source_root: str, out_zip_path: str, include_exts=(".png", ".svg", ".csv")) -> str:
    """Export graph/table artifacts from ``source_root`` into a compact zip file."""
    include_exts = tuple(ext.lower() for ext in include_exts)
    os.makedirs(os.path.dirname(out_zip_path), exist_ok = True)
    total = 0
    with zipfile.ZipFile(out_zip_path, "w", compression = zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_root):
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in include_exts:
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, source_root)
                zf.write(full, arcname = rel)
                total += 1
    print(f"[EXPORT] graph assets zipped: files={total} root={source_root} -> {out_zip_path}")
    return out_zip_path

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
    launch_seed = int.from_bytes(os.urandom(8), byteorder = "big") & 0x7FFFFFFF
    random.seed(launch_seed)
    np.random.seed(launch_seed)
    set_torch_global_seed(launch_seed)
    launch_id = f"launch_{launch_seed}"
    print(f"[SEED] experiment_launch_seed={launch_seed}")
    print(f"[RUN ROOT] {os.path.join(RUNS_ROOT, launch_id)}")
    
    machine = "a"
    train_replications = 20
    eval_replications = 12
    
    # user-requested fixed scenario
    overrides = {}
    
    # (a) RL convergence diagnostics (reward logs + trajectory graph)
    train_phase = os.path.join(launch_id, "train")
    _run_replications(machine, train_replications, train_phase, "rl", True, overrides)
    train_png = _aggregate_strategy_curves(train_phase, "rl", metrics = ["reward", "reward_ma_window", "casualty", "evacuated"])
    train_conv_png = _plot_training_convergence(train_phase)

    # (b) Policy comparison: RL vs random vs heuristic vs all shelters installed at t=0
    compare_phase = os.path.join(launch_id, "eval_compare")
    compare_strategies = ["rl", "initial_only", "random", "heuristic"]
    compare_pngs = {}
    for strategy in compare_strategies:
        _run_replications(machine, eval_replications, compare_phase, strategy, False, overrides)
        compare_pngs[strategy] = _aggregate_strategy_curves(
            compare_phase,
            strategy,
            metrics = ["shelter_utilization", "mean_evacuation_time", "evacuated", "casualty"],
        )
        
    compare_overlay_png = _plot_strategy_comparison_curves(
        compare_phase,
        compare_strategies,
        metrics = ["shelter_utilization", "mean_evacuation_time", "evacuated", "casualty"],
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
        metrics = ["shelter_utilization", "mean_evacuation_time", "evacuated", "casualty"],
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

        perf_cols = ["strategy", "final_casualty", "final_evacuated", "final_successful_evacuation_rate", "mean_evacuation_time"]
        printable = mean_df[perf_cols].sort_values(by = "strategy").reset_index(drop = True)
        print("\n[POLICY COMPARISON] Mean performance over replications")
        print(printable.to_string(index = False))


    # Simple convergence check output
    convergence_df = _episode_summary(train_phase, "rl")
    if convergence_df is not None and not convergence_df.empty:
        early = convergence_df["final_reward_ma"].head(max(1, len(convergence_df)//3)).mean()
        late = convergence_df["final_reward_ma"].tail(max(1, len(convergence_df)//3)).mean()
        print(f"[CONVERGENCE] early_mean_reward_ma={early:.4f} late_mean_reward_ma={late:.4f}")
    
    
    print("\n=== Completed ===")
    print("Training summary graph:", train_png)
    print("Training convergence graph:", train_conv_png)
    for p in compare_tables:
        print("Comparison summary table:", p)
    print("Comparison overlay graph:", compare_overlay_png)
    
    required_metrics = ["shelter_utilization", "mean_evacuation_time", "evacuated", "casualty"]
    _validate_graph_outputs(pairwise_group_metric_pngs, pairwise_groups, required_metrics)
    _print_graph_outputs(pairwise_group_metric_pngs)
    
    package_zip = _export_launch_package(launch_id)
    print("Download package (includes convergence graph):", package_zip)
    
    
    filtered_assets_zip = _runs_path(f"{launch_id}_graphs_and_tables.zip")
    _export_graph_assets_zip(_runs_path(launch_id), filtered_assets_zip)
    print("Download graph/table package:", filtered_assets_zip)
    
    
    # Keep a stable pointer for Colab/export snippets that package `runs/`.
    # This makes the newest launch discoverable at `runs/latest` while still
    # preserving unique launch folders for historical comparisons.
    latest_ptr = _runs_path("latest")
    try:
        if os.path.islink(latest_ptr) or os.path.isfile(latest_ptr):
            os.unlink(latest_ptr)
        elif os.path.isdir(latest_ptr):
            shutil.rmtree(latest_ptr)
        os.symlink(_runs_path(launch_id), latest_ptr)
        print(f"[RUN ROOT] latest -> {_runs_path(launch_id)}")
    except OSError:
        # Symlink may be unsupported on some environments; fallback to copy.
        if os.path.isdir(latest_ptr):
            shutil.rmtree(latest_ptr)
        shutil.copytree(_runs_path(launch_id), latest_ptr)
        print(f"[RUN ROOT] latest copied from {_runs_path(launch_id)}")
        
if __name__ == "__main__":
    Script()