#!/usr/bin/env python3
"""Generate the lean v61 paper-figure package from completed artifacts.

The script is deliberately read-only with respect to the simulator and learner.
It consumes the completed five-city visualization run and the matched State
College evaluation run, verifies their source artifacts, and writes figures,
plot-ready tables, captions, and a checksum manifest.

Main-paper figures
------------------
1. Five-city dynamic execution atlas (one fixed-seed episode per city).
2. State College matched performance, objective decomposition, and online
   decision-latency comparison (20 common-random-number tapes).

Supplement
----------
3. Simulator-graded actor-fit diagnostics over the 20 policy updates.

Internal audit (not paper-ready)
--------------------------------
4. Stage-E random-reference reward calibration against its declared zero line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


DEFAULT_C1_RUN = Path(
    "runs/c1_five_city_visuals_v61_seed2646201033_20261018"
)
DEFAULT_SC_RUN = Path(
    "runs/state_college_2500_regime_v61_origin_grouped_E20_P80_eval20_seed6110_20261015_r2"
)
DEFAULT_OUTPUT = Path("publication_graphs/v61_paper_evidence_20261018")

CITY_ORDER = (
    "state_college_pa",
    "spokane_wa",
    "seattle_wa",
    "chicago_il",
    "malibu_ca",
)
CITY_LABELS = {
    "state_college_pa": "State College",
    "spokane_wa": "Spokane",
    "seattle_wa": "Seattle",
    "chicago_il": "Chicago",
    "malibu_ca": "Malibu–Santa Monica Bay",
}

# t=15 is included because the hazard fields change mainly before minute 60;
# t=180 retains the final crowd state and all eight numbered installations.
ATLAS_TIMES = (0, 15, 60, 180)

OKABE_ITO = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#111111",
    "grey": "#7A7A7A",
}

HAZARD_COLORS = (
    (1.0, 1.0, 1.0, 0.0),  # 0: unaffected, transparent over roads
    (0.996, 0.878, 0.545, 0.68),
    (0.957, 0.522, 0.188, 0.70),
    (0.839, 0.153, 0.157, 0.72),
    (0.502, 0.055, 0.224, 0.76),
    (0.102, 0.102, 0.102, 0.80),
)
HAZARD_CMAP = ListedColormap(HAZARD_COLORS, name="evacuation_danger_state")
HAZARD_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), HAZARD_CMAP.N)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _verify_file(path: Path, expected_sha256: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return actual


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
        }
    )


def _visualization_dir(c1_root: Path, city_id: str) -> Path:
    matches = sorted((c1_root / "episodes" / city_id / "rl").glob(
        "rep_*_c1/evacuation_visualization"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one visualization directory for {city_id}; found {matches}"
        )
    return matches[0]


def _verify_c1_sources(c1_root: Path) -> Dict[str, Mapping[str, object]]:
    top_manifest_path = c1_root / "experiment_manifest.json"
    top = json.loads(top_manifest_path.read_text())
    if top.get("status") != "complete":
        raise RuntimeError(f"C1 run is not complete: {top.get('status')}")
    if tuple(top.get("completed_cities", ())) != CITY_ORDER:
        raise RuntimeError("C1 completed-city order does not match the paper contract")

    verified: Dict[str, Mapping[str, object]] = {}
    for city_id in CITY_ORDER:
        viz = _visualization_dir(c1_root, city_id)
        manifest_path = viz / "visualization_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 5:
            raise RuntimeError(f"{city_id} visualization schema is not v5")
        if tuple(manifest.get("grid_shape", ())) != (16, 16):
            raise RuntimeError(f"{city_id} does not use the required 16x16 grid")
        if int(manifest.get("decision_epoch_count", -1)) != 8:
            raise RuntimeError(f"{city_id} does not contain eight decisions")
        for artifact_name, entry in manifest["artifacts"].items():
            artifact_path = Path(str(entry["path"]))
            # Manifests contain absolute source paths. Resolve by filename inside
            # the selected run if the workspace has moved.
            if not artifact_path.is_file():
                artifact_path = viz / artifact_name
            _verify_file(artifact_path, str(entry["sha256"]))
        verified[city_id] = {"directory": str(viz.resolve()), "manifest": manifest}
    return verified


def _load_atlas_city(viz: Path) -> Dict[str, pd.DataFrame]:
    return {
        "milestone_cells": pd.read_csv(viz / "milestone_cells.csv"),
        "milestone_pedestrians": pd.read_csv(viz / "milestone_pedestrians.csv"),
        "milestone_shelters": pd.read_csv(viz / "milestone_shelters.csv"),
        "milestone_hazards": pd.read_csv(viz / "milestone_hazards.csv"),
        "milestone_outcomes": pd.read_csv(viz / "milestone_outcomes.csv"),
        "decision_cells": pd.read_csv(viz / "decision_epoch_cells.csv"),
        "decision_pedestrians": pd.read_csv(viz / "decision_epoch_pedestrians.csv.gz"),
        "decision_hazards": pd.read_csv(viz / "decision_epoch_hazards.csv"),
        "decisions": pd.read_csv(viz / "deployment_decisions.csv"),
        "roads": pd.read_csv(viz / "osm_road_segments.csv"),
    }


def _snapshot(data: Mapping[str, pd.DataFrame], time_minute: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if time_minute in (0, 60, 180):
        cells = data["milestone_cells"].query("simulation_time == @time_minute").copy()
        pedestrians = data["milestone_pedestrians"].query(
            "simulation_time == @time_minute"
        ).copy()
        hazards = data["milestone_hazards"].query(
            "simulation_time == @time_minute"
        ).copy()
    else:
        cells = data["decision_cells"].query("simulation_time == @time_minute").copy()
        pedestrians = data["decision_pedestrians"].query(
            "simulation_time == @time_minute"
        ).copy()
        hazards = data["decision_hazards"].query(
            "simulation_time == @time_minute"
        ).copy()
    if len(cells) != 256:
        raise RuntimeError(f"Expected 256 cells at t={time_minute}; found {len(cells)}")
    return cells, pedestrians, hazards


def _nice_scale_length(span_metres: float) -> float:
    target = max(1.0, 0.18 * span_metres)
    exponent = 10.0 ** math.floor(math.log10(target))
    candidates = [1.0, 2.0, 5.0, 10.0]
    feasible = [c * exponent for c in candidates if c * exponent <= target]
    return max(feasible) if feasible else exponent


def _scale_label(length_m: float) -> str:
    if length_m >= 1000:
        return f"{length_m / 1000:g} km"
    return f"{length_m:g} m"


def _label_offsets(
    points: Sequence[Tuple[float, float]],
    span: float,
    bounds: Tuple[float, float, float, float],
) -> Sequence[Tuple[int, int]]:
    """Place numbered shelter callouts without collisions at print scale."""
    choices = (
        (7, 7),
        (-8, 7),
        (8, -8),
        (-9, -8),
        (13, 0),
        (-13, 0),
        (0, 13),
        (0, -13),
        (14, 9),
        (-14, 9),
        (14, -9),
        (-14, -9),
        (19, 0),
        (-19, 0),
        (0, 19),
        (0, -19),
    )
    x_min, x_max, y_min, y_max = bounds
    # Each atlas panel is approximately 105 points across at the final 6.9-inch
    # figure width. Work in that display-coordinate approximation because map
    # distances alone do not predict whether two 8-point callouts collide.
    points_per_map_unit = 105.0 / span
    display_width = (x_max - x_min) * points_per_map_unit
    display_height = (y_max - y_min) * points_per_map_unit
    edge_margin = 7.0
    minimum_separation = 15.0
    offsets = []
    label_centres: list[Tuple[float, float]] = []
    for x, y in points:
        anchor_x = (x - x_min) * points_per_map_unit
        anchor_y = (y_max - y) * points_per_map_unit
        valid_candidates = []
        for preference, (dx, dy) in enumerate(choices):
            centre_x = anchor_x + dx
            centre_y = anchor_y + dy
            if not (
                edge_margin <= centre_x <= display_width - edge_margin
                and edge_margin <= centre_y <= display_height - edge_margin
            ):
                continue
            nearest = min(
                (
                    math.hypot(centre_x - prior_x, centre_y - prior_y)
                    for prior_x, prior_y in label_centres
                ),
                default=float("inf"),
            )
            valid_candidates.append((nearest, -preference, dx, dy, centre_x, centre_y))
            if nearest >= minimum_separation:
                break
        if valid_candidates:
            _nearest, _preference, dx, dy, centre_x, centre_y = max(
                valid_candidates
            )
        else:
            # All preferred centres are near a border. Pick the least-outward
            # option deterministically and still include it in collision state.
            dx, dy = min(
                choices,
                key=lambda offset: (
                    max(0.0, edge_margin - (anchor_x + offset[0]))
                    + max(0.0, anchor_x + offset[0] - (display_width - edge_margin))
                    + max(0.0, edge_margin - (anchor_y + offset[1]))
                    + max(0.0, anchor_y + offset[1] - (display_height - edge_margin))
                ),
            )
            centre_x = anchor_x + dx
            centre_y = anchor_y + dy
        offsets.append((int(dx), int(dy)))
        label_centres.append((centre_x, centre_y))
    return offsets


def _plot_atlas_panel(
    ax: plt.Axes,
    *,
    data: Mapping[str, pd.DataFrame],
    time_minute: int,
    first_column: bool,
) -> Dict[str, float]:
    cells, pedestrians, hazards = _snapshot(data, time_minute)
    roads = data["roads"]
    decisions = data["decisions"].query("simulation_time <= @time_minute").copy()
    initial = data["milestone_shelters"].query(
        "simulation_time == 0 and initial_shelter == 1"
    ).drop_duplicates("shelter_id")

    x_min = float(cells["x_min_m"].min())
    x_max = float(cells["x_max_m"].max())
    y_min = float(cells["y_min_m"].min())
    y_max = float(cells["y_max_m"].max())
    span = max(x_max - x_min, y_max - y_min)

    segments = np.stack(
        [
            roads[["start_x_m", "start_y_m"]].to_numpy(dtype=float),
            roads[["end_x_m", "end_y_m"]].to_numpy(dtype=float),
        ],
        axis=1,
    )
    roads_collection = LineCollection(
        segments,
        colors="#B8B8B8",
        linewidths=0.16,
        alpha=0.58,
        zorder=1,
        rasterized=True,
    )
    ax.add_collection(roads_collection)

    patches = []
    facecolors = []
    for row in cells.itertuples(index=False):
        state = int(np.clip(round(float(row.hazard_state)), 0, 5))
        if state <= 0:
            continue
        patches.append(
            Rectangle(
                (float(row.x_min_m), float(row.y_min_m)),
                float(row.x_max_m - row.x_min_m),
                float(row.y_max_m - row.y_min_m),
            )
        )
        facecolors.append(HAZARD_COLORS[state])
    if patches:
        ax.add_collection(
            PatchCollection(
                patches,
                facecolors=facecolors,
                edgecolors=(0.25, 0.25, 0.25, 0.32),
                linewidths=0.18,
                zorder=2,
            )
        )

    if not pedestrians.empty:
        sizes = 4.2 + 1.25 * np.sqrt(
            pedestrians["group_size"].to_numpy(dtype=float)
        )
        ax.scatter(
            pedestrians["x_m"],
            pedestrians["y_m"],
            s=sizes,
            c=OKABE_ITO["blue"],
            alpha=0.83,
            edgecolors="white",
            linewidths=0.12,
            zorder=4,
            rasterized=True,
        )

    if not hazards.empty:
        ax.scatter(
            hazards["x_m"],
            hazards["y_m"],
            marker="X",
            s=23,
            c=OKABE_ITO["black"],
            edgecolors="white",
            linewidths=0.45,
            zorder=7,
        )

    if not initial.empty:
        ax.scatter(
            initial["x_m"],
            initial["y_m"],
            marker="^",
            s=29,
            c="white",
            edgecolors=OKABE_ITO["black"],
            linewidths=0.65,
            zorder=8,
        )

    if not decisions.empty:
        xy = list(
            zip(
                decisions["candidate_x_m"].astype(float),
                decisions["candidate_y_m"].astype(float),
            )
        )
        offsets = _label_offsets(xy, span, (x_min, x_max, y_min, y_max))
        ax.scatter(
            decisions["candidate_x_m"],
            decisions["candidate_y_m"],
            marker="o",
            s=18,
            c=OKABE_ITO["green"],
            edgecolors="white",
            linewidths=0.55,
            zorder=8,
        )
        for row, offset in zip(decisions.itertuples(index=False), offsets):
            ax.annotate(
                str(int(row.decision_index)),
                xy=(float(row.candidate_x_m), float(row.candidate_y_m)),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold",
                color="white",
                bbox={
                    "boxstyle": "circle,pad=0.15",
                    "facecolor": OKABE_ITO["green"],
                    "edgecolor": "white",
                    "linewidth": 0.45,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": OKABE_ITO["green"],
                    "linewidth": 0.45,
                    "shrinkA": 1.5,
                    "shrinkB": 1.5,
                },
                zorder=9,
                annotation_clip=True,
            )

    active = float(pedestrians["group_size"].sum()) if not pedestrians.empty else 0.0
    ax.text(
        0.985,
        0.025,
        f"active {active:,.0f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        zorder=10,
    )

    if first_column:
        scale_m = _nice_scale_length(x_max - x_min)
        ax.add_artist(
            AnchoredSizeBar(
                ax.transData,
                scale_m,
                _scale_label(scale_m),
                "lower left",
                pad=0.22,
                color="#222222",
                frameon=True,
                size_vertical=max(1.0, span * 0.0018),
                fontproperties={"size": 8.0},
                borderpad=0.25,
                sep=1.5,
            )
        )

    # A small common-fraction margin prevents border candidates, hazard sources,
    # and their numbered callouts from being clipped while preserving scale.
    x_pad = 0.015 * (x_max - x_min)
    y_pad = 0.015 * (y_max - y_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)  # y is southward; render north-up
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#BDBDBD")
        spine.set_linewidth(0.35)

    affected_cells = int((cells["hazard_state"] > 0).sum())
    maximum_state = int(cells["hazard_state"].max())
    return {
        "simulation_time": int(time_minute),
        "active_population": active,
        "affected_cells": affected_cells,
        "maximum_hazard_state": maximum_state,
        "installed_dynamic_shelters": int(len(decisions)),
    }


def _save_figure(fig: plt.Figure, stem: Path, *, png_dpi: int = 600) -> Sequence[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    # Preserve the declared physical canvas so journal-width fonts remain the
    # sizes specified above. ``bbox_inches='tight'`` can silently expand a
    # multi-panel figure beyond the intended two-column width.
    fig.savefig(png, dpi=png_dpi, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    return (png, pdf)


def generate_c1_atlas(c1_root: Path, out_dir: Path) -> Tuple[Sequence[Path], Path]:
    fig, axes = plt.subplots(
        len(CITY_ORDER),
        len(ATLAS_TIMES),
        figsize=(6.9, 9.25),
        gridspec_kw={"hspace": 0.06, "wspace": 0.035},
    )
    panel_rows = []
    for row_index, city_id in enumerate(CITY_ORDER):
        viz = _visualization_dir(c1_root, city_id)
        data = _load_atlas_city(viz)
        for col_index, time_minute in enumerate(ATLAS_TIMES):
            ax = axes[row_index, col_index]
            metrics = _plot_atlas_panel(
                ax,
                data=data,
                time_minute=time_minute,
                first_column=col_index == 0,
            )
            panel_rows.append({"city_id": city_id, **metrics})
            if row_index == 0:
                ax.set_title(f"{time_minute} min", pad=3.0, fontweight="bold")
        axes[row_index, 0].text(
            -0.12,
            0.5,
            CITY_LABELS[city_id],
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=8.0,
            fontweight="bold" if row_index == 0 else "normal",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OKABE_ITO["blue"],
            markeredgecolor="white",
            markersize=5,
            label="active cohort (≤10 people)",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=OKABE_ITO["black"],
            markeredgecolor="white",
            markersize=5.5,
            label="hazard source",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="white",
            markeredgecolor=OKABE_ITO["black"],
            markersize=5.5,
            label="initial shelter",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OKABE_ITO["green"],
            markeredgecolor="white",
            markersize=5.5,
            label="sequential shelter (number = order)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.49, 0.037),
        columnspacing=1.5,
        handletextpad=0.5,
    )
    cax = fig.add_axes((0.18, 0.018, 0.64, 0.011))
    colorbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=HAZARD_CMAP,
        norm=HAZARD_NORM,
        boundaries=np.arange(-0.5, 6.5, 1.0),
        ticks=np.arange(0, 6),
        spacing="uniform",
        orientation="horizontal",
    )
    colorbar.set_label("Cellular-automata danger state", labelpad=1.0)
    colorbar.ax.tick_params(length=2, pad=1)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.98, bottom=0.09)

    table_path = out_dir / "data" / "c1_atlas_panel_summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(panel_rows).to_csv(table_path, index=False)
    paths = _save_figure(fig, out_dir / "main" / "figure_1_five_city_dynamic_atlas")
    plt.close(fig)
    return paths, table_path


def _paired_by_seed(
    evaluation: pd.DataFrame,
    left_strategy: str,
    right_strategy: str,
    metric: str,
) -> pd.DataFrame:
    keys = ["scenario_seed"]
    left = (
        evaluation.query("deployment_strategy == @left_strategy")[keys + [metric]]
        .rename(columns={metric: "left"})
        .drop_duplicates(keys)
    )
    right = (
        evaluation.query("deployment_strategy == @right_strategy")[keys + [metric]]
        .rename(columns={metric: "right"})
        .drop_duplicates(keys)
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    if len(paired) != 20:
        raise RuntimeError(
            f"Expected 20 paired tapes for {left_strategy} vs {right_strategy}; "
            f"found {len(paired)}"
        )
    return paired


def _bootstrap_mean_ci(
    values: Iterable[float],
    *,
    rng: np.random.Generator,
    draws: int = 50_000,
) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Bootstrap input must be a finite nonempty vector")
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _mean_ci(values: Iterable[float], rng: np.random.Generator) -> Tuple[float, float, float]:
    return _bootstrap_mean_ci(values, rng=rng)


def generate_state_college_main(
    state_college_root: Path,
    out_dir: Path,
) -> Tuple[Sequence[Path], Sequence[Path]]:
    evaluation_path = state_college_root / "evaluation_episode_summary.csv"
    evaluation = pd.read_csv(evaluation_path)
    required_strategies = {
        "rl",
        "objective_greedy",
        "rl_forecast_ablated",
        "belief_rollout",
    }
    missing = required_strategies - set(evaluation["deployment_strategy"].unique())
    if missing:
        raise RuntimeError(f"Evaluation is missing required strategies: {sorted(missing)}")
    if any(
        len(evaluation.query("deployment_strategy == @strategy")) != 20
        for strategy in required_strategies
    ):
        raise RuntimeError("Each plotted strategy must contain exactly 20 evaluation tapes")

    rng = np.random.default_rng(20261018)
    effect_specs = (
        ("objective_greedy", "Obj.-greedy"),
        ("rl_forecast_ablated", "No-forecast RL"),
        ("belief_rollout", "Belief rollout"),
    )
    effect_rows = []
    for benchmark, label in effect_specs:
        paired = _paired_by_seed(
            evaluation,
            "rl",
            benchmark,
            "objective_normalized_cost",
        )
        # Cost improvement is benchmark minus RL; positive favors trained RL.
        differences = paired["right"] - paired["left"]
        mean, low, high = _bootstrap_mean_ci(differences, rng=rng)
        effect_rows.append(
            {
                "comparison": label,
                "benchmark_strategy": benchmark,
                "paired_tapes": len(paired),
                "mean_cost_improvement_of_rl": mean,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "rl_win_rate": float((differences > 0).mean()),
                "tie_rate": float((differences == 0).mean()),
            }
        )
    effect_table = pd.DataFrame(effect_rows)

    denominator = float(
        evaluation["initial_population"].iloc[0]
        * evaluation["horizon_minutes"].iloc[0]
    )
    component_specs = (
        (
            "Active and remaining time",
            lambda frame: 0.5
            * frame["realized_plain_time_to_safety"].to_numpy(dtype=float)
            / denominator,
        ),
        (
            "Hazard exposure",
            lambda frame: 2.0
            * frame["realized_hazard_exposure_time_to_safety"].to_numpy(dtype=float)
            / denominator,
        ),
        (
            "Expected casualty",
            lambda frame: frame["objective_expected_casualty_person_time"].to_numpy(
                dtype=float
            )
            / denominator,
        ),
        (
            "Clearance tail",
            lambda frame: frame["objective_clearance_tail_person_time"].to_numpy(
                dtype=float
            )
            / denominator,
        ),
    )
    strategy_order = (
        ("objective_greedy", "Obj.-\ngreedy"),
        ("rl_forecast_ablated", "No-forecast\nRL"),
        ("rl", "RL"),
        ("belief_rollout", "Belief\nrollout"),
    )
    decomposition_rows = []
    for strategy, label in strategy_order:
        frame = evaluation.query("deployment_strategy == @strategy")
        for component, calculator in component_specs:
            values = calculator(frame)
            decomposition_rows.append(
                {
                    "strategy": strategy,
                    "strategy_label": label,
                    "component": component,
                    "mean_normalized_contribution": float(np.mean(values)),
                }
            )
        reconstructed = sum(
            row["mean_normalized_contribution"]
            for row in decomposition_rows
            if row["strategy"] == strategy
        )
        reported = float(frame["objective_normalized_cost"].mean())
        if not math.isclose(reconstructed, reported, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(
                f"Objective decomposition failed for {strategy}: "
                f"{reconstructed} != {reported}"
            )
    decomposition_table = pd.DataFrame(decomposition_rows)

    latency_rows = []
    for strategy, label in (
        ("objective_greedy", "Objective-greedy"),
        ("rl", "Trained RL"),
        ("belief_rollout", "Belief rollout"),
    ):
        frame = evaluation.query("deployment_strategy == @strategy")
        cost_mean, cost_low, cost_high = _mean_ci(
            frame["objective_normalized_cost"], rng
        )
        selection = frame["mean_policy_selection_latency_ms"].to_numpy(dtype=float)
        latency_rows.append(
            {
                "strategy": strategy,
                "strategy_label": label,
                "mean_objective_normalized_cost": cost_mean,
                "bootstrap_95_ci_low": cost_low,
                "bootstrap_95_ci_high": cost_high,
                "median_policy_selection_latency_ms": float(np.median(selection)),
                "mean_policy_selection_latency_ms": float(np.mean(selection)),
                "latency_q25_ms": float(np.quantile(selection, 0.25)),
                "latency_q75_ms": float(np.quantile(selection, 0.75)),
            }
        )
    latency_table = pd.DataFrame(latency_rows)

    headroom_panel = (
        evaluation.query(
            "deployment_strategy in ['objective_greedy', 'rl', 'belief_rollout']"
        )
        .pivot(
            index="scenario_seed",
            columns="deployment_strategy",
            values="objective_normalized_cost",
        )
        .sort_index()
    )
    if len(headroom_panel) != 20 or headroom_panel.isna().any().any():
        raise RuntimeError("Headroom bootstrap requires 20 complete paired tapes")
    greedy_minus_rl = (
        headroom_panel["objective_greedy"] - headroom_panel["rl"]
    ).to_numpy(dtype=float)
    greedy_minus_rollout = (
        headroom_panel["objective_greedy"] - headroom_panel["belief_rollout"]
    ).to_numpy(dtype=float)
    headroom_share = float(greedy_minus_rl.mean() / greedy_minus_rollout.mean())
    headroom_rng = np.random.default_rng(20261018)
    headroom_indices = headroom_rng.integers(
        0, len(headroom_panel), size=(50_000, len(headroom_panel))
    )
    bootstrap_denominator = greedy_minus_rollout[headroom_indices].mean(axis=1)
    if np.any(bootstrap_denominator <= 0.0):
        raise RuntimeError("Headroom bootstrap produced a nonpositive denominator")
    bootstrap_headroom = (
        greedy_minus_rl[headroom_indices].mean(axis=1) / bootstrap_denominator
    )
    headroom_low, headroom_high = np.quantile(
        bootstrap_headroom, (0.025, 0.975)
    )
    headroom_table = pd.DataFrame(
        [
            {
                "paired_tapes": len(headroom_panel),
                "rl_share_of_greedy_to_rollout_gap": headroom_share,
                "paired_bootstrap_95_ci_low": float(headroom_low),
                "paired_bootstrap_95_ci_high": float(headroom_high),
                "bootstrap_draws": len(bootstrap_headroom),
            }
        ]
    )

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    effect_path = data_dir / "state_college_paired_cost_effects.csv"
    decomposition_path = data_dir / "state_college_objective_decomposition.csv"
    latency_path = data_dir / "state_college_latency_frontier.csv"
    headroom_path = data_dir / "state_college_headroom_summary.csv"
    effect_table.to_csv(effect_path, index=False)
    decomposition_table.to_csv(decomposition_path, index=False)
    latency_table.to_csv(latency_path, index=False)
    headroom_table.to_csv(headroom_path, index=False)

    fig = plt.figure(figsize=(6.9, 5.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :]),
    )

    # A: paired CRN effects relative to the trained policy.
    ax = axes[0]
    plot_effect = effect_table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_effect))
    colors = [
        OKABE_ITO["orange"] if value < 0 else OKABE_ITO["blue"]
        for value in plot_effect["mean_cost_improvement_of_rl"]
    ]
    means = plot_effect["mean_cost_improvement_of_rl"].to_numpy()
    lows = plot_effect["bootstrap_95_ci_low"].to_numpy()
    highs = plot_effect["bootstrap_95_ci_high"].to_numpy()
    for mean_value, low_value, high_value, y_value, color in zip(
        means, lows, highs, y, colors
    ):
        ax.errorbar(
            mean_value,
            y_value,
            xerr=np.asarray(
                [[mean_value - low_value], [high_value - mean_value]]
            ),
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.8,
            elinewidth=1.3,
            capsize=2.5,
            zorder=3,
        )
    ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, plot_effect["comparison"])
    ax.set_xlabel("Benchmark cost − RL cost\n(positive favors RL)")
    ax.set_title("a  Paired cost effects", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)

    # B: exact additive objective decomposition.
    ax = axes[1]
    component_colors = {
        "Active and remaining time": OKABE_ITO["sky"],
        "Hazard exposure": OKABE_ITO["orange"],
        "Expected casualty": OKABE_ITO["vermillion"],
        "Clearance tail": OKABE_ITO["purple"],
    }
    component_legend_labels = {
        "Active and remaining time": "Active/remaining",
        "Hazard exposure": "Hazard exposure",
        "Expected casualty": "Expected casualty",
        "Clearance tail": "Clearance tail",
    }
    x = np.arange(len(strategy_order))
    bottom = np.zeros(len(strategy_order), dtype=float)
    for component, _ in component_specs:
        values = []
        for strategy, _label in strategy_order:
            row = decomposition_table.query(
                "strategy == @strategy and component == @component"
            )
            values.append(float(row["mean_normalized_contribution"].iloc[0]))
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.72,
            color=component_colors[component],
            edgecolor="white",
            linewidth=0.4,
            label=component_legend_labels[component],
        )
        bottom += np.asarray(values)
    ax.set_xticks(
        x,
        [label for _strategy, label in strategy_order],
        rotation=0,
        ha="center",
        fontsize=8.0,
    )
    ax.set_ylabel("Mean normalized cost")
    ax.set_title("b  Cost decomposition", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_ylim(0.0, 0.38)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        fontsize=8.0,
        columnspacing=0.8,
        handlelength=1.2,
    )

    # C: quality/online-compute frontier.
    ax = axes[2]
    frontier_colors = {
        "objective_greedy": OKABE_ITO["orange"],
        "rl": OKABE_ITO["blue"],
        "belief_rollout": OKABE_ITO["green"],
    }
    for row in latency_table.itertuples(index=False):
        x_value = float(row.median_policy_selection_latency_ms)
        y_value = float(row.mean_objective_normalized_cost)
        ax.errorbar(
            x_value,
            y_value,
            yerr=np.asarray(
                [
                    [y_value - float(row.bootstrap_95_ci_low)],
                    [float(row.bootstrap_95_ci_high) - y_value],
                ]
            ),
            fmt="o",
            color=frontier_colors[row.strategy],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=6.2,
            capsize=2.0,
            zorder=3,
        )
        offsets = {
            "objective_greedy": (6, 7),
            "rl": (6, -14),
            "belief_rollout": (-74, 7),
        }
        plot_labels = {
            "objective_greedy": "Obj.-greedy",
            "rl": "RL",
            "belief_rollout": "Belief rollout",
        }
        ax.annotate(
            plot_labels[row.strategy],
            (x_value, y_value),
            xytext=offsets[row.strategy],
            textcoords="offset points",
            fontsize=8.0,
            color=frontier_colors[row.strategy],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Selection latency (ms, log)")
    ax.set_ylabel("Normalized cost\n(lower is better)")
    ax.set_title("c  Cost vs latency", loc="left", fontweight="bold")
    ax.grid(True, which="both", color="#E5E5E5", linewidth=0.6)

    greedy_cost = float(
        latency_table.query("strategy == 'objective_greedy'")[
            "mean_objective_normalized_cost"
        ].iloc[0]
    )
    rl_cost = float(
        latency_table.query("strategy == 'rl'")["mean_objective_normalized_cost"].iloc[0]
    )
    rollout_cost = float(
        latency_table.query("strategy == 'belief_rollout'")[
            "mean_objective_normalized_cost"
        ].iloc[0]
    )
    plotted_headroom_share = (greedy_cost - rl_cost) / (greedy_cost - rollout_cost)
    if not math.isclose(
        plotted_headroom_share, headroom_share, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("Headroom point estimates disagree")
    rl_latency = float(
        latency_table.query("strategy == 'rl'")[
            "median_policy_selection_latency_ms"
        ].iloc[0]
    )
    rollout_latency = float(
        latency_table.query("strategy == 'belief_rollout'")[
            "median_policy_selection_latency_ms"
        ].iloc[0]
    )
    ax.text(
        0.39,
        0.97,
        f"RL share of greedy→rollout gap: {headroom_share:.1%}\n"
        f"paired 95% CI [{headroom_low:.1%}, {headroom_high:.1%}]\n"
        f"Rollout/RL latency: {rollout_latency / rl_latency:,.0f}×",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9, "pad": 2.0},
    )

    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(
        left=0.145,
        right=0.985,
        top=0.95,
        bottom=0.10,
        hspace=0.76,
        wspace=0.50,
    )
    paths = _save_figure(
        fig,
        out_dir / "main" / "figure_2_state_college_performance_and_latency",
    )
    plt.close(fig)
    return paths, (effect_path, decomposition_path, latency_path, headroom_path)


def generate_actor_diagnostics(
    state_college_root: Path,
    out_dir: Path,
) -> Tuple[Sequence[Path], Path]:
    diagnostics_path = (
        state_college_root / "policies" / "policy_001" / "learner_diagnostics.csv"
    )
    diagnostics = pd.read_csv(diagnostics_path)
    updates = diagnostics.query(
        "optimizer_updated > 0 and rollout_pi_actor_enabled > 0 "
        "and rollout_pi_physical_branch_seconds > 0"
    ).copy()
    if len(updates) != 20:
        raise RuntimeError(f"Expected 20 policy updates; found {len(updates)}")
    updates = updates.sort_values(["policy_iterations_completed", "episode"]).reset_index(
        drop=True
    )
    updates["policy_update"] = np.arange(1, len(updates) + 1)

    columns = [
        "policy_update",
        "episode",
        "rollout_pi_weighted_pairwise_accuracy_before",
        "rollout_pi_weighted_pairwise_accuracy_after",
        "rollout_pi_branched_top1_before",
        "rollout_pi_branched_top1_after",
        "rollout_pi_target_kl_before",
        "rollout_pi_target_kl_after",
        "rollout_pi_rollout_changed_base_best_fraction",
        "rollout_pi_physical_branch_seconds",
        "rollout_pi_paired_contrast_standard_error",
        "rollout_pi_within_state_value_sd",
    ]
    output_table = updates[columns].copy()
    table_path = out_dir / "data" / "actor_fit_diagnostics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(table_path, index=False)

    x = updates["policy_update"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.8), sharex=True)
    panels = (
        (
            axes[0, 0],
            "rollout_pi_weighted_pairwise_accuracy_before",
            "rollout_pi_weighted_pairwise_accuracy_after",
            "Weighted pairwise accuracy",
        ),
        (
            axes[0, 1],
            "rollout_pi_branched_top1_before",
            "rollout_pi_branched_top1_after",
            "Branched top-choice accuracy",
        ),
    )
    for ax, before_column, after_column, title in panels:
        ax.plot(
            x,
            updates[before_column],
            color="#888888",
            marker="o",
            markersize=3.2,
            label="before update",
        )
        ax.plot(
            x,
            updates[after_column],
            color=OKABE_ITO["blue"],
            marker="o",
            markersize=3.2,
            label="after update",
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Fraction correct")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, color="#E5E5E5", linewidth=0.6)

    ax = axes[1, 0]
    ax.plot(
        x,
        updates["rollout_pi_target_kl_before"],
        color="#888888",
        marker="o",
        markersize=3.2,
        label="before update",
    )
    ax.plot(
        x,
        updates["rollout_pi_target_kl_after"],
        color=OKABE_ITO["green"],
        marker="o",
        markersize=3.2,
        label="after update",
    )
    ax.axhline(0.1, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=0.8, label="target limit")
    ax.set_ylabel("KL(target || actor)")
    ax.set_xlabel("Policy update")
    ax.set_title("Target-distribution fit", loc="left", fontweight="bold")
    ax.grid(True, color="#E5E5E5", linewidth=0.6)

    ax = axes[1, 1]
    ax.plot(
        x,
        updates["rollout_pi_rollout_changed_base_best_fraction"],
        color=OKABE_ITO["purple"],
        marker="o",
        markersize=3.2,
        label="rollout changes base-best",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Decision-state fraction")
    ax.set_xlabel("Policy update")
    ax.set_title("Headroom in graded panels", loc="left", fontweight="bold")
    ax.grid(True, color="#E5E5E5", linewidth=0.6)
    time_ax = ax.twinx()
    time_ax.plot(
        x,
        updates["rollout_pi_physical_branch_seconds"],
        color=OKABE_ITO["orange"],
        alpha=0.65,
        linewidth=0.9,
        label="branch time",
    )
    time_ax.set_ylabel("Branch simulation (s)", color=OKABE_ITO["orange"])
    time_ax.tick_params(axis="y", colors=OKABE_ITO["orange"])

    axes[0, 0].legend(loc="lower right", frameon=False)
    axes[1, 0].legend(loc="upper right", frameon=False, fontsize=8.0)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    time_ax.spines["top"].set_visible(False)
    fig.text(
        0.5,
        0.018,
        "One policy seed; diagnostics are measured on the simulator-graded training panels, not held-out performance.",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.92, top=0.96, bottom=0.15, hspace=0.38, wspace=0.35)
    paths = _save_figure(
        fig,
        out_dir / "supplement" / "figure_s1_actor_fit_diagnostics",
    )
    plt.close(fig)
    return paths, table_path


def generate_reward_calibration_audit(
    state_college_root: Path,
    out_dir: Path,
) -> Tuple[Sequence[Path], Path, Mapping[str, float]]:
    training = pd.read_csv(state_college_root / "training_episode_summary.csv")
    stage_e = training.query("rollout_pi_actor_enabled == 0").copy()
    if len(stage_e) != 20:
        # The first 20 episodes are the declared E stage in this run. Keep the
        # fail-closed behavior if the logged stage field becomes ambiguous.
        stage_e = training.sort_values("episode").head(20).copy()
    returns = stage_e["episode_return"].to_numpy(dtype=float)
    if len(returns) != 20 or not np.isfinite(returns).all():
        raise RuntimeError("Stage-E calibration requires 20 finite returns")
    mean = float(np.mean(returns))
    se = float(np.std(returns, ddof=1) / math.sqrt(len(returns)))
    low = mean - 1.96 * se
    high = mean + 1.96 * se

    episode_column = "episode" if "episode" in stage_e.columns else "replication"
    audit_table = stage_e[
        [episode_column, "episode_return", "reward_accounting_gap"]
    ].copy()
    audit_table = audit_table.rename(columns={episode_column: "episode"})
    table_path = out_dir / "audit" / "stage_e_reward_calibration.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    audit_table.to_csv(table_path, index=False)

    rng = np.random.default_rng(20261018)
    jitter = rng.uniform(-0.07, 0.07, size=len(returns))
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.scatter(
        jitter,
        returns,
        color=OKABE_ITO["sky"],
        edgecolors="white",
        linewidths=0.5,
        s=28,
        alpha=0.9,
        label="Stage-E episodes",
    )
    ax.errorbar(
        [0.23],
        [mean],
        yerr=[[mean - low], [high - mean]],
        fmt="o",
        color=OKABE_ITO["vermillion"],
        markeredgecolor="white",
        capsize=4,
        markersize=6,
        label="mean ± normal 95% CI",
    )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9, label="declared zero line")
    ax.set_xlim(-0.18, 0.40)
    ax.set_xticks([])
    ax.set_ylabel("Episode return relative to random reference")
    ax.set_title("Internal audit: Stage-E reward calibration", loc="left", fontweight="bold")
    ax.text(
        0.98,
        0.97,
        f"mean {mean:+.3f}\n95% CI [{low:+.3f}, {high:+.3f}]\n"
        "zero line not calibrated",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.94, "pad": 2.5},
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8.0)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    paths = _save_figure(
        fig,
        out_dir / "audit" / "figure_q1_stage_e_reward_calibration",
    )
    plt.close(fig)
    return paths, table_path, {
        "episodes": int(len(returns)),
        "mean_return": mean,
        "standard_error": se,
        "normal_95_ci_low": low,
        "normal_95_ci_high": high,
        "zero_inside_interval": bool(low <= 0.0 <= high),
    }


def _write_captions(out_dir: Path, calibration: Mapping[str, float]) -> Path:
    path = out_dir / "CAPTIONS_AND_USE.md"
    text = f"""# v61 paper figures: captions and permitted claims

## Figure 1 — Five-city dynamic execution atlas

Dynamic execution of the latest State-College-trained policy on five north-up OpenStreetMap walking networks. Rows show the training city and four zero-shot cities; columns show 0, 15, 60, and 180 minutes from one fixed-seed episode per city (2,500 represented pedestrians, 16×16 population-adaptive cells, two initial shelters, and eight sequential installations at 15-minute intervals). Blue dots are active pedestrian cohorts grouped by initial node, each representing at most ten people; cell colour is the cellular-automata danger state, × marks hazard sources, triangles mark initial shelters, and numbered circles identify sequential shelter-installation order. Seattle's 180-minute panel is an absorbing snapshot after completion at minute 179. The single episodes demonstrate end-to-end portability, evolving crowd and hazard state, and sequenced deployment; they do not establish cross-city policy superiority or statistical generalization. Road data © OpenStreetMap contributors, ODbL.

## Figure 2 — State College matched performance and latency

Performance and online computation on 20 held-out State College common-random-number tapes for one frozen trained checkpoint. (a) Paired normalized-cost differences, defined as benchmark cost minus trained-RL cost; points are paired means and intervals are pointwise, unadjusted percentile-bootstrap 95% confidence intervals over the 20 tapes. Objective-greedy is the prespecified primary comparator; the ablation and rollout contrasts are secondary. (b) Exact additive decomposition of the mean normalized objective using the registered weights (active and remaining time 0.5, hazard exposure 2.0, expected-casualty person-time as logged, and clearance tail 0.25). (c) Mean normalized cost versus median policy-selection latency; vertical intervals are bootstrap 95% intervals over tapes. The reported headroom fraction is (J_objective-greedy − J_RL)/(J_objective-greedy − J_belief-rollout), with a paired scenario-bootstrap 95% interval. Latency is policy scoring and selection time on the recorded local hardware; it excludes offline training and simulation time outside policy selection. The evidence is conditional on one trained policy seed. This checkpoint did not meet the registered training-convergence criterion, although its frozen held-out evaluation remains valid. It supports an objective-cost advantage over objective-greedy and a lower online selection latency than belief rollout. Objective-greedy itself is faster than RL, whereas belief rollout remains better in cost but orders of magnitude slower. The forecast ablation is statistically inconclusive, so this figure does not establish a forecast-driven foresight mechanism.

## Figure S1 — Simulator-graded actor-fit diagnostics

Actor fit across the 20 policy-improvement updates from one State College training seed. Panels show weighted pairwise ranking accuracy, branched top-choice accuracy, target-to-actor KL before and after each categorical update, the fraction of graded states in which rollout changes the greedy base choice, and physical branch-simulation time. The checkpoint did not meet the registered training-convergence criterion. These are in-sample optimization diagnostics, not a held-out convergence or multi-seed reproducibility result.

## Internal audit Q1 — not a paper figure

The current Stage-E random-reference return is {calibration['mean_return']:+.3f} ± {calibration['standard_error']:.3f} SE over {int(calibration['episodes'])} episodes; its normal 95% interval [{calibration['normal_95_ci_low']:+.3f}, {calibration['normal_95_ci_high']:+.3f}] excludes zero. Therefore the present v61 run cannot support the claimed zero-mean random-reference calibration. The graph is retained under `audit/` to make that evidence gap explicit and must not be presented as validation of Contribution 3.

## Evidence still required

- At least three independently trained policy seeds before calling the training curve reproducible or converged.
- A matched v61 random-reference calibration run before publishing a zero-line reward claim.
- A controlled State College regime-family or state-level forward-looking audit before claiming that the forecast channel causes the trained policy's advantage. The current no-forecast paired effect includes zero.
"""
    path.write_text(text)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-run", type=Path, default=DEFAULT_C1_RUN)
    parser.add_argument("--state-college-run", type=Path, default=DEFAULT_SC_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    c1_root = (project_root / args.c1_run).resolve() if not args.c1_run.is_absolute() else args.c1_run.resolve()
    state_college_root = (
        (project_root / args.state_college_run).resolve()
        if not args.state_college_run.is_absolute()
        else args.state_college_run.resolve()
    )
    out_dir = (
        (project_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    verified_cities = _verify_c1_sources(c1_root)
    required_sc_sources = (
        state_college_root / "experiment_manifest.json",
        state_college_root / "evaluation_episode_summary.csv",
        state_college_root / "training_episode_summary.csv",
        state_college_root / "policies" / "policy_001" / "learner_diagnostics.csv",
        state_college_root / "policies" / "policy_001" / "regional_policy.pt",
    )
    source_hashes = {str(path.resolve()): _verify_file(path) for path in required_sc_sources}

    atlas_outputs, atlas_table = generate_c1_atlas(c1_root, out_dir)
    state_outputs, state_tables = generate_state_college_main(state_college_root, out_dir)
    actor_outputs, actor_table = generate_actor_diagnostics(state_college_root, out_dir)
    audit_outputs, audit_table, calibration = generate_reward_calibration_audit(
        state_college_root, out_dir
    )
    captions = _write_captions(out_dir, calibration)

    outputs = [
        *atlas_outputs,
        atlas_table,
        *state_outputs,
        *state_tables,
        *actor_outputs,
        actor_table,
        *audit_outputs,
        audit_table,
        captions,
    ]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source_runs": {
            "contribution_1": {
                "path": str(c1_root),
                "experiment_manifest_sha256": _sha256(c1_root / "experiment_manifest.json"),
                "verified_city_artifact_sets": verified_cities,
            },
            "state_college_v61": {
                "path": str(state_college_root),
                "source_sha256": source_hashes,
                "policy_seeds": 1,
                "evaluation_tapes": 20,
            },
        },
        "figures": {
            "F1": {
                "title": "Five-city dynamic execution atlas",
                "status": "paper_ready_descriptive",
                "claim": "pipeline portability and dynamic closed-loop execution",
                "limits": [
                    "one fixed-seed episode per city",
                    "not cross-city policy-superiority evidence",
                ],
                "artifacts": [str(path.resolve()) for path in atlas_outputs],
            },
            "F2": {
                "title": "State College matched performance and latency",
                "status": "paper_ready_conditional_on_one_checkpoint",
                "claim": "paired objective advantage and lower online selection latency than belief rollout",
                "limits": [
                    "one trained policy seed",
                    "checkpoint did not meet the registered training-convergence criterion",
                    "forecast ablation is null",
                    "expected-casualty intensity does not improve",
                ],
                "artifacts": [str(path.resolve()) for path in state_outputs],
            },
            "FS1": {
                "title": "Actor-fit diagnostics",
                "status": "supplement_descriptive",
                "claim": "categorical actor fits simulator-graded panels",
                "limits": [
                    "one seed",
                    "checkpoint did not meet the registered training-convergence criterion",
                    "training panels, not held-out convergence",
                ],
                "artifacts": [str(path.resolve()) for path in actor_outputs],
            },
            "Q1": {
                "title": "Stage-E reward calibration audit",
                "status": "internal_audit_not_paper_ready",
                "claim": "current random-reference zero-line claim fails",
                "statistics": calibration,
                "artifacts": [str(path.resolve()) for path in audit_outputs],
            },
        },
        "outputs": {
            _rel_or_abs(path, out_dir): {"sha256": _sha256(path)} for path in outputs
        },
        "prohibited_inferences": [
            "cross-city policy superiority from Figure 1",
            "multi-seed convergence from Figure S1",
            "forecast-driven foresight from the current null ablation",
            "zero-mean reward calibration from the current Stage-E run",
        ],
    }
    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Generated {len(outputs)} artifacts plus manifest at {out_dir}")
    for path in outputs:
        print(path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
