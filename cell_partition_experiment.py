#!/usr/bin/env python3
"""Characterize equal-area and node-density-adaptive regional grids."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import networkx as nx
import numpy as np

from CellPartitioning import build_cell_partition, cell_rows


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "cell_partition_experiment.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "cell_partition_state_college_20260909"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _strict(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_strict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            args,
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def load_local_metre_graph(path: Path):
    """Load cached GraphML and reproduce MapDS's deterministic coordinates."""
    graph = nx.read_graphml(path)
    node_ids = list(graph.nodes)
    if not node_ids:
        raise ValueError("Cached graph contains no nodes")
    longitude = np.asarray(
        [float(graph.nodes[node_id]["x"]) for node_id in node_ids], dtype=float
    )
    latitude = np.asarray(
        [float(graph.nodes[node_id]["y"]) for node_id in node_ids], dtype=float
    )
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("Cached graph has non-finite node coordinates")
    metres_per_degree_y = 111_132.0
    metres_per_degree_x = 111_320.0 * math.cos(math.radians(float(latitude[0])))
    x = (longitude - float(np.min(longitude))) * metres_per_degree_x
    y = (float(np.max(latitude)) - latitude) * metres_per_degree_y
    positions = {
        node_id: (float(x[index]), float(y[index]))
        for index, node_id in enumerate(node_ids)
    }
    return graph, x, y, positions


def accuracy_design_matrix(config: dict) -> list[dict]:
    design = config["confirmatory_accuracy_design"]
    rows = []
    for city_id in design["cities"]:
        for population in design["population_levels"]:
            for side in design["accuracy_grid_levels"]:
                for mode in config["partition_modes"]:
                    rows.append(
                        {
                            "city_id": city_id,
                            "population": int(population),
                            "grid_side": int(side),
                            "cell_count": int(side) ** 2,
                            "partition_mode": str(mode),
                            "policy_seeds": int(design["policy_seeds"]),
                            "held_out_scenarios": int(
                                design["held_out_scenarios_per_cell"]
                            ),
                            "learned_policy_episodes": int(design["policy_seeds"])
                            * int(design["held_out_scenarios_per_cell"]),
                            "heuristic_episodes": int(
                                design["held_out_scenarios_per_cell"]
                            ),
                            "status": "pending_compute",
                        }
                    )
    return rows


def state_college_gate_matrix(config: dict) -> list[dict]:
    design = config["confirmatory_accuracy_design"]
    gate = design["state_college_gate"]
    rows = []
    for population in gate["population_levels"]:
        for mode in config["partition_modes"]:
            rows.append(
                {
                    "city_id": "state_college_pa",
                    "population": int(population),
                    "grid_side": int(gate["grid_level"]),
                    "cell_count": int(gate["grid_level"]) ** 2,
                    "partition_mode": str(mode),
                    "policy_seeds": int(gate["policy_seeds"]),
                    "held_out_scenarios": int(
                        gate["held_out_scenarios_per_cell"]
                    ),
                    "learned_policy_episodes": int(gate["policy_seeds"])
                    * int(gate["held_out_scenarios_per_cell"]),
                    "heuristic_episodes": int(
                        gate["held_out_scenarios_per_cell"]
                    ),
                    "status": "required_before_multicity",
                }
            )
    return rows


def characterize(config: dict, output_dir: Path) -> dict:
    study = config["state_college_characterization"]
    graph_path = (PROJECT_ROOT / study["graph_cache_path"]).resolve()
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Registered cached graph is missing: {graph_path}. "
            "Run the State College map preparation first."
        )
    graph, x, y, positions = load_local_metre_graph(graph_path)
    modes = [str(mode) for mode in config["partition_modes"]]
    levels = [int(side) for side in study["grid_levels"]]
    minimum = float(config["partition_contract"]["minimum_axis_width_fraction"])
    summaries: list[dict] = []
    cells: list[dict] = []
    partitions: dict[tuple[str, int], dict] = {}

    for side in levels:
        for mode in modes:
            partition = build_cell_partition(
                x,
                y,
                side,
                side,
                mode,
                min_width_fraction=minimum,
            )
            partitions[(mode, side)] = partition
            diagnostic = {
                key: value
                for key, value in partition["diagnostics"].items()
                if key not in {"x_edges_m", "y_edges_m"}
            }
            diagnostic.update(
                {
                    "city_id": study["city_id"],
                    "grid_side": side,
                    "partition_mode": mode,
                    "minimum_axis_width_fraction": minimum,
                    "cell_area_min_median_ratio": float(
                        diagnostic["cell_area_min_m2"]
                        / diagnostic["cell_area_median_m2"]
                    ),
                }
            )
            summaries.append(diagnostic)

            counts, _, _ = np.histogram2d(
                x,
                y,
                bins=(partition["x_edges"], partition["y_edges"]),
            )
            for row in cell_rows(partition):
                xi = int(row["cell_x"])
                yi = int(row["cell_y"])
                count = int(counts[xi, yi])
                row.update(
                    {
                        "city_id": study["city_id"],
                        "road_node_count": count,
                        "road_node_density_per_km2": float(
                            count / float(row["area_m2"]) * 1_000_000.0
                        ),
                    }
                )
                cells.append(row)

    by_key = {(row["partition_mode"], row["grid_side"]): row for row in summaries}
    comparisons = []
    for side in levels:
        equal = by_key[("equal_area", side)]
        adaptive = by_key[("node_density_adaptive", side)]
        comparisons.append(
            {
                "city_id": study["city_id"],
                "grid_side": side,
                "cell_count": side * side,
                "equal_area_node_count_cv": equal["node_count_cv"],
                "adaptive_node_count_cv": adaptive["node_count_cv"],
                "node_count_cv_reduction_fraction": float(
                    1.0 - adaptive["node_count_cv"] / equal["node_count_cv"]
                ),
                "equal_area_node_count_gini": equal["node_count_gini"],
                "adaptive_node_count_gini": adaptive["node_count_gini"],
                "equal_area_empty_cell_fraction": equal["empty_cell_fraction"],
                "adaptive_empty_cell_fraction": adaptive["empty_cell_fraction"],
                "adaptive_area_max_min_ratio": adaptive["cell_area_max_min_ratio"],
            }
        )

    _write_csv(output_dir / "partition_summary.csv", summaries)
    _write_csv(output_dir / "partition_cells.csv", cells)
    _write_csv(output_dir / "partition_mode_comparison.csv", comparisons)
    full_accuracy_matrix = accuracy_design_matrix(config)
    gate_matrix = state_college_gate_matrix(config)
    _write_csv(output_dir / "planned_accuracy_design_matrix.csv", full_accuracy_matrix)
    _write_csv(output_dir / "state_college_gate_matrix.csv", gate_matrix)
    figure_paths = generate_figure(
        graph,
        x,
        y,
        positions,
        partitions,
        summaries,
        illustration_grid=int(study["illustration_grid"]),
        output_dir=output_dir,
    )
    return {
        "graph_cache_path": str(graph_path),
        "graph_cache_sha256": _sha256(graph_path),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "partition_summary_path": str(output_dir / "partition_summary.csv"),
        "partition_cells_path": str(output_dir / "partition_cells.csv"),
        "partition_mode_comparison_path": str(
            output_dir / "partition_mode_comparison.csv"
        ),
        "planned_accuracy_design_matrix_path": str(
            output_dir / "planned_accuracy_design_matrix.csv"
        ),
        "state_college_gate_matrix_path": str(
            output_dir / "state_college_gate_matrix.csv"
        ),
        "figure_paths": figure_paths,
        "design_cells": len(summaries),
        "per_cell_rows": len(cells),
        "planned_accuracy_design_cells": len(full_accuracy_matrix),
        "state_college_gate_cells": len(gate_matrix),
        "all_partition_node_counts_match_graph": bool(
            all(
                int(row["road_node_count"]) == int(graph.number_of_nodes())
                for row in summaries
            )
        ),
        "adaptive_node_count_cv_lower_at_every_grid_level": bool(
            all(
                float(row["adaptive_node_count_cv"])
                < float(row["equal_area_node_count_cv"])
                for row in comparisons
            )
        ),
    }


def generate_figure(
    graph,
    x,
    y,
    positions,
    partitions,
    summaries,
    *,
    illustration_grid: int,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )
    edge_segments = []
    for source, target in graph.edges():
        if source in positions and target in positions:
            edge_segments.append([positions[source], positions[target]])

    selected = {
        mode: partitions[(mode, int(illustration_grid))]
        for mode in ("equal_area", "node_density_adaptive")
    }
    selected_counts = {
        mode: np.histogram2d(
            x,
            y,
            bins=(partition["x_edges"], partition["y_edges"]),
        )[0]
        for mode, partition in selected.items()
    }
    maximum_count = max(float(np.max(values)) for values in selected_counts.values())
    norm = LogNorm(vmin=1.0, vmax=max(1.0, maximum_count))
    fig = plt.figure(figsize=(10.8, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 0.8))
    map_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    lower_axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    titles = {
        "equal_area": "(a) Equal-area 8×8 partition",
        "node_density_adaptive": "(b) Node-density-adaptive 8×8 partition",
    }
    mesh = None
    for ax, mode in zip(map_axes, ("equal_area", "node_density_adaptive")):
        partition = selected[mode]
        mesh = ax.pcolormesh(
            partition["x_edges"],
            partition["y_edges"],
            np.ma.masked_less_equal(selected_counts[mode].T, 0.0),
            cmap="viridis",
            norm=norm,
            shading="flat",
            alpha=0.84,
        )
        ax.add_collection(
            LineCollection(
                edge_segments,
                colors="#222222",
                linewidths=0.18,
                alpha=0.42,
                rasterized=True,
            )
        )
        for boundary in partition["x_edges"]:
            ax.axvline(boundary, color="white", linewidth=0.55, alpha=0.95)
        for boundary in partition["y_edges"]:
            ax.axhline(boundary, color="white", linewidth=0.55, alpha=0.95)
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_ylim(float(np.max(y)), float(np.min(y)))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles[mode], loc="left", fontweight="bold")
        ax.set_xlabel("metres east of western network edge")
        ax.set_ylabel("metres south of northern network edge")
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=map_axes, shrink=0.82, pad=0.02)
        colorbar.set_label("OSM road nodes per cell (log scale)")

    colors = {"equal_area": "#0072B2", "node_density_adaptive": "#D55E00"}
    labels = {
        "equal_area": "Equal area",
        "node_density_adaptive": "Node-density adaptive",
    }
    for mode in ("equal_area", "node_density_adaptive"):
        rows = sorted(
            (row for row in summaries if row["partition_mode"] == mode),
            key=lambda row: int(row["grid_side"]),
        )
        cell_counts = [int(row["cell_count"]) for row in rows]
        lower_axes[0].plot(
            cell_counts,
            [float(row["node_count_cv"]) for row in rows],
            marker="o",
            color=colors[mode],
            label=labels[mode],
        )
        lower_axes[1].plot(
            cell_counts,
            [float(row["empty_cell_fraction"]) for row in rows],
            marker="o",
            color=colors[mode],
            label=labels[mode],
        )
    lower_axes[0].set_title("(c) Road-node imbalance", loc="left", fontweight="bold")
    lower_axes[0].set_ylabel("CV of nodes per cell (lower is balanced)")
    lower_axes[1].set_title("(d) Empty-cell prevalence", loc="left", fontweight="bold")
    lower_axes[1].set_ylabel("Fraction of cells with zero road nodes")
    for ax in lower_axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("regional cell count, n²")
        ax.grid(True, which="both", color="#D9D9D9", linewidth=0.6)
        ax.legend(frameon=False)
    fig.suptitle(
        "State College regional discretization: geometry and infrastructure balance",
        fontsize=12,
        fontweight="bold",
    )
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"F8a_partition_geometry_and_node_balance.{suffix}"
        fig.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def write_report(config: dict, output_dir: Path, artifacts: dict) -> Path:
    rows = []
    with (output_dir / "partition_mode_comparison.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    illustration = next(
        row
        for row in rows
        if int(row["grid_side"])
        == int(config["state_college_characterization"]["illustration_grid"])
    )
    reduction = 100.0 * float(illustration["node_count_cv_reduction_fraction"])
    report_path = output_dir / "CELL_PARTITION_RESULTS.md"
    content = f"""# Cell-partition characterization — State College

## Material Passport

- Material type: Experiment Result
- Material ID: `{config['suite_id']}`
- Status: `ANALYZED`
- Evidence tier: deterministic map/preprocessing characterization
- Input graph SHA-256: `{artifacts['graph_cache_sha256']}`
- Claim boundary: no evacuation-accuracy or population-density claim is made

## Result

Both partition modes produced exactly n² rectangular cells with identical
row-major action indexing. At the {illustration['grid_side']}×{illustration['grid_side']}
illustration grid, the node-density-adaptive partition changed the coefficient
of variation of road nodes per cell from
{float(illustration['equal_area_node_count_cv']):.3f} to
{float(illustration['adaptive_node_count_cv']):.3f} ({reduction:.1f}% reduction).
Its cell-area max/min ratio was
{float(illustration['adaptive_area_max_min_ratio']):.2f}, confirming that the
geometric resolution is nonuniform.

This is a deterministic characterization of one cached State College road
network. It verifies the intended discretization mechanism but does not show
that adaptive cells improve casualty or evacuation-time outcomes. That claim
requires the registered separately-trained, paired held-out experiment.

## Artifacts

- `partition_summary.csv`: one row per grid-size/mode condition
- `partition_cells.csv`: cell boundaries, areas, road-node counts, and densities
- `partition_mode_comparison.csv`: paired geometry/balance contrasts
- `planned_accuracy_design_matrix.csv`: all 300 held-out design cells
- `state_college_gate_matrix.csv`: six required gate cells before expansion
- `F8a_partition_geometry_and_node_balance.png` and `.svg`: journal figure
- `manifest.json`: environment, code state, provenance, and artifact hashes
"""
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, report_path)
    return report_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported cell-partition experiment schema")

    started_wall = time.perf_counter()
    started = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "status": "running",
        "started_utc": started.isoformat(),
        "design_path": str(config_path),
        "design_sha256": _sha256(config_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        "git": {
            "commit": _git_value("git", "rev-parse", "HEAD"),
            "status": _git_value("git", "status", "--short"),
        },
        "artifacts": {},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    artifacts = characterize(config, output_dir)
    report_path = write_report(config, output_dir, artifacts)
    artifacts["report_path"] = str(report_path)
    characterization = config["state_college_characterization"]
    accuracy = config["confirmatory_accuracy_design"]
    expected_counts = {
        "partition_design_cells": len(config["partition_modes"])
        * len(characterization["grid_levels"]),
        "per_cell_rows": len(config["partition_modes"])
        * sum(int(side) ** 2 for side in characterization["grid_levels"]),
        "accuracy_design_cells": len(accuracy["cities"])
        * len(accuracy["population_levels"])
        * len(accuracy["accuracy_grid_levels"])
        * len(config["partition_modes"]),
        "state_college_gate_cells": len(
            accuracy["state_college_gate"]["population_levels"]
        )
        * len(config["partition_modes"]),
    }
    observed_counts = {
        "partition_design_cells": int(artifacts["design_cells"]),
        "per_cell_rows": int(artifacts["per_cell_rows"]),
        "accuracy_design_cells": int(artifacts["planned_accuracy_design_cells"]),
        "state_college_gate_cells": int(artifacts["state_college_gate_cells"]),
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Partition experiment count mismatch: expected={expected_counts}, "
            f"observed={observed_counts}"
        )
    if not artifacts["all_partition_node_counts_match_graph"]:
        raise RuntimeError("At least one partition failed to conserve all graph nodes")
    manifest["artifacts"] = artifacts
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_time_s"] = float(time.perf_counter() - started_wall)
    manifest["artifact_sha256"] = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest["audit"] = {
        "status": "passed",
        "artifact_hashes_recorded": True,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "all_partition_node_counts_match_graph": bool(
            artifacts["all_partition_node_counts_match_graph"]
        ),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
