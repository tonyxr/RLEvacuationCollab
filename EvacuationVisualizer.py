#!/usr/bin/env python3
"""Read-only, publication-oriented milestone maps for evacuation episodes.

The renderer uses the exact OpenStreetMap road graph already loaded by the
simulator.  It never queries the network, changes an agent, or participates in
the policy interface.  Every plotted layer is also exported as a tabular file
so a paper figure can be regenerated and audited independently of simulation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


SCHEMA_VERSION = 4


def resolve_milestones(spec, horizon: int) -> tuple[int, ...]:
    """Return sorted, unique milestone boundaries within ``[0, horizon]``.

    ``"quartiles"`` creates five evenly spaced snapshots, matching the visual
    grammar of the supplied reference figure without assuming its 120-minute
    horizon.  A comma-separated string or integer iterable requests explicit
    simulator times.  Time zero and the terminal horizon are always retained.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("Visualization horizon must be at least one timestep")

    if spec is None or str(spec).strip().lower() in {"", "quartiles", "five"}:
        requested = np.rint(np.linspace(0, horizon, 5)).astype(int).tolist()
    elif isinstance(spec, str):
        try:
            requested = [int(item.strip()) for item in spec.split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(
                "Visualization milestones must be 'quartiles' or comma-separated integers"
            ) from exc
    else:
        requested = [int(item) for item in spec]

    out_of_range = [value for value in requested if value < 0 or value > horizon]
    if out_of_range:
        raise ValueError(
            f"Visualization milestones must fall in [0, {horizon}], got {out_of_range}"
        )
    return tuple(sorted(set([0, horizon, *requested])))


@dataclass(frozen=True)
class MilestoneSnapshot:
    simulation_time: int
    pedestrians: tuple[dict, ...]
    shelters: tuple[dict, ...]
    cells: tuple[dict, ...]
    outcome: dict
    latest_decision: Optional[dict]
    absorbing_after_terminal: bool = False


class EvacuationVisualizer:
    """Capture and render non-interventional OSM milestone snapshots."""

    def __init__(
        self,
        core,
        *,
        milestones="quartiles",
        output_dir: Optional[str] = None,
        render_individual_snapshots: bool = True,
        render_vector_outputs: bool = True,
    ):
        self.core = core
        self.horizon = max(1, int(core.stopTime) - 1)
        self.time_step_minutes = float(getattr(core, "timeStepMinutes", 1.0))
        self.milestones = resolve_milestones(milestones, self.horizon)
        self.output_dir = os.path.abspath(
            output_dir or os.path.join(core.run_dir, "evacuation_visualization")
        )
        self.render_individual_snapshots = bool(render_individual_snapshots)
        self.render_vector_outputs = bool(render_vector_outputs)
        os.makedirs(self.output_dir, exist_ok=True)

        all_initialized = frozenset(int(key) for key in core.shelterDS.shelterList)
        self.initial_shelter_ids = frozenset(
            getattr(core, "baseline_initial_shelter_ids", all_initialized)
        )
        self.static_predeployment_shelter_ids = frozenset(
            getattr(core, "static_predeployment_shelter_ids", frozenset())
        )
        self.known_shelter_ids = set(all_initialized)
        self.decisions: list[dict] = []
        self.snapshots: list[MilestoneSnapshot] = []
        self.decision_snapshots: list[MilestoneSnapshot] = []
        self._captured_times: set[int] = set()
        self._finalized = False
        self.terminal_time: Optional[int] = None

        self.road_segments = self._road_segments()
        self.x_limits, self.y_limits = self._map_limits()
        for shelter_id in sorted(self.static_predeployment_shelter_ids):
            shelter = core.shelterDS.shelterList[shelter_id]
            self.decisions.append(
                self._decision_row(
                    shelter=shelter,
                    simulation_time=0,
                    decision_type="static_predeployment",
                    selected_cell=-1,
                    heuristic_cell=-1,
                    rerouted_population=0,
                )
            )

    def _decision_row(
        self,
        *,
        shelter,
        simulation_time: int,
        decision_type: str,
        selected_cell: int,
        heuristic_cell: int,
        rerouted_population: int,
    ) -> dict:
        cell_i, cell_j = shelter.cellLocated
        node = shelter.nodeMapped
        return {
            "decision_index": len(self.decisions) + 1,
            "simulation_time": int(simulation_time),
            "strategy": str(self.core.rl.deployment_strategy),
            "decision_type": str(decision_type),
            "selected_cell": int(selected_cell),
            "heuristic_cell": int(heuristic_cell),
            "shelter_id": int(shelter.shelterID),
            "candidate_osm_node_id": str(getattr(node, "OSMID", "")),
            "candidate_x_m": float(node.nodeX),
            "candidate_y_m": float(node.nodeY),
            "candidate_cell_i": int(cell_i),
            "candidate_cell_j": int(cell_j),
            "capacity_added": float(getattr(shelter, "shelterCap", 0.0)),
            "rerouted_population": int(rerouted_population),
        }

    def _road_segments(self) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
        segments = []
        seen = set()
        for edge in self.core.mapDS.edgeListByLocalID.values():
            start = (
                float(edge.startNode.nodeX),
                float(edge.startNode.nodeY),
            )
            end = (
                float(edge.endNode.nodeX),
                float(edge.endNode.nodeY),
            )
            if not all(np.isfinite((*start, *end))) or start == end:
                continue
            # Directed OSM graphs commonly contain the same physical segment in
            # both directions.  Deduplication keeps the basemap visually honest.
            key = tuple(sorted((start, end)))
            if key in seen:
                continue
            seen.add(key)
            segments.append((start, end))
        return tuple(segments)

    def _map_limits(self) -> tuple[tuple[float, float], tuple[float, float]]:
        tracker = self.core.cellTracker
        x_edges = np.asarray(tracker.xEdges, dtype=float)
        y_edges = np.asarray(tracker.yEdges, dtype=float)
        if x_edges.size < 2 or y_edges.size < 2:
            raise RuntimeError("Cell boundaries must be initialized before visualization")
        x_span = max(1.0, float(x_edges[-1] - x_edges[0]))
        y_span = max(1.0, float(y_edges[-1] - y_edges[0]))
        padding = 0.02
        return (
            (float(x_edges[0] - padding * x_span), float(x_edges[-1] + padding * x_span)),
            (float(y_edges[0] - padding * y_span), float(y_edges[-1] + padding * y_span)),
        )

    @staticmethod
    def _group_size(pedestrian) -> int:
        return max(1, int(getattr(pedestrian, "group_size", 1)))

    def _pedestrian_rows(self, simulation_time: int) -> tuple[dict, ...]:
        rows = []
        for pedestrian in self.core.pedDS.pedAgentList.values():
            if bool(getattr(pedestrian, "terminated", False)):
                continue
            cell = getattr(pedestrian, "currCell", None)
            cell_i = int(cell[0]) if cell is not None else -1
            cell_j = int(cell[1]) if cell is not None else -1
            row = {
                "simulation_time": int(simulation_time),
                "agent_id": str(getattr(pedestrian, "agentID", "")),
                "x_m": float(getattr(pedestrian, "lastX", np.nan)),
                "y_m": float(getattr(pedestrian, "lastY", np.nan)),
                "cell_i": cell_i,
                "cell_j": cell_j,
                "group_size": self._group_size(pedestrian),
                "speed_m_per_minute": float(getattr(pedestrian, "currSpeed", 0.0)),
                "affected": int(bool(getattr(pedestrian, "affected", False))),
            }
            if np.isfinite(row["x_m"]) and np.isfinite(row["y_m"]):
                rows.append(row)
        return tuple(rows)

    def _shelter_rows(self, simulation_time: int) -> tuple[dict, ...]:
        rows = []
        for shelter_id, shelter in sorted(self.core.shelterDS.shelterList.items()):
            node = shelter.nodeMapped
            cell = shelter.cellLocated
            rows.append(
                {
                    "simulation_time": int(simulation_time),
                    "shelter_id": int(shelter_id),
                    "osm_node_id": str(getattr(node, "OSMID", "")),
                    "x_m": float(node.nodeX),
                    "y_m": float(node.nodeY),
                    "cell_i": int(cell[0]),
                    "cell_j": int(cell[1]),
                    "capacity": float(getattr(shelter, "shelterCap", 0.0)),
                    "flow": float(getattr(shelter, "shelterFlow", 0.0)),
                    "status": int(getattr(shelter, "status", 0)),
                    "initial_shelter": int(int(shelter_id) in self.initial_shelter_ids),
                    "absorbing_after_terminal": 0,
                    "deployment_mode": (
                        "initial"
                        if int(shelter_id) in self.initial_shelter_ids
                        else "static_predeployment"
                        if int(shelter_id) in self.static_predeployment_shelter_ids
                        else "dynamic"
                    ),
                }
            )
        return tuple(rows)

    def _cell_rows(self, simulation_time: int, pedestrians: tuple[dict, ...]) -> tuple[dict, ...]:
        tracker = self.core.cellTracker
        active = np.zeros(int(self.core.cellX * self.core.cellY), dtype=int)
        for row in pedestrians:
            i, j = int(row["cell_i"]), int(row["cell_j"])
            if 0 <= i < int(self.core.cellX) and 0 <= j < int(self.core.cellY):
                active[i * int(self.core.cellY) + j] += int(row["group_size"])
        danger = np.asarray(
            getattr(tracker, "dangerLevelByCell", np.zeros_like(active, dtype=float)),
            dtype=float,
        ).reshape(-1)
        rows = []
        for i in range(int(self.core.cellX)):
            for j in range(int(self.core.cellY)):
                index = i * int(self.core.cellY) + j
                cell = tracker.cellList[(i, j)]
                rows.append(
                    {
                        "simulation_time": int(simulation_time),
                        "cell_index": int(index),
                        "cell_i": int(i),
                        "cell_j": int(j),
                        "x_min_m": float(tracker.xEdges[i]),
                        "x_max_m": float(tracker.xEdges[i + 1]),
                        "y_min_m": float(tracker.yEdges[j]),
                        "y_max_m": float(tracker.yEdges[j + 1]),
                        "hazard_state": int(getattr(cell, "impactedLevel", 0)),
                        "danger": float(np.clip(danger[index], 0.0, 1.0)),
                        "active_population": int(active[index]),
                        "absorbing_after_terminal": 0,
                    }
                )
        return tuple(rows)

    def _outcome(self, simulation_time: int, pedestrians: tuple[dict, ...]) -> dict:
        result = self.core.pedDS.result
        return {
            "simulation_time": int(simulation_time),
            "active_population": int(sum(row["group_size"] for row in pedestrians)),
            "safe_completed": int(result.get("arrival", 0)) + int(result.get("evacuated", 0)),
            "ordinary_arrivals": int(result.get("arrival", 0)),
            "shelter_evacuated": int(result.get("evacuated", 0)),
            "casualties": int(result.get("casualty", 0)),
            "affected_cumulative": int(result.get("affected", 0)),
            "absorbing_after_terminal": 0,
            "terminal_time": "",
        }

    def _capture_decision_epoch(self, simulation_time: int) -> None:
        """Capture the full spatial state immediately after a deployment.

        Decision maps use the same visual variables as milestone maps: one
        point per active pedestrian and the continuous normalized danger of
        every cell.  Retaining the coordinates also makes each plotted point
        independently auditable instead of reconstructing people from cell
        counts.
        """
        pedestrians = self._pedestrian_rows(simulation_time)
        self.decision_snapshots.append(
            MilestoneSnapshot(
                simulation_time=int(simulation_time),
                pedestrians=pedestrians,
                shelters=self._shelter_rows(simulation_time),
                cells=self._cell_rows(simulation_time, pedestrians),
                outcome=self._outcome(simulation_time, pedestrians),
                latest_decision=dict(self.decisions[-1]),
                absorbing_after_terminal=False,
            )
        )

    def observe(self, simulation_time: int, decision_output: Optional[dict] = None) -> bool:
        """Record a decision, then capture state if this boundary is a milestone."""
        if self._finalized:
            raise RuntimeError("Cannot observe an episode after visualization finalization")
        simulation_time = int(simulation_time)
        if decision_output and int(decision_output.get("decision_made", 0)):
            current_ids = set(int(key) for key in self.core.shelterDS.shelterList)
            newly_installed = sorted(current_ids.difference(self.known_shelter_ids))
            self.known_shelter_ids.update(current_ids)
            if len(newly_installed) != 1:
                raise RuntimeError(
                    "Every regional decision must install exactly one shelter candidate"
                )
            shelter = self.core.shelterDS.shelterList[newly_installed[0]]
            self.decisions.append(
                self._decision_row(
                    shelter=shelter,
                    simulation_time=simulation_time,
                    decision_type="dynamic",
                    selected_cell=int(decision_output.get("selected_cell", -1)),
                    heuristic_cell=int(decision_output.get("heuristic_cell", -1)),
                    rerouted_population=int(
                        decision_output.get("rerouted_population", 0)
                    ),
                )
            )
            self._capture_decision_epoch(simulation_time)
        if simulation_time in self.milestones:
            return self.capture(simulation_time)
        return False

    def capture(self, simulation_time: int, *, force: bool = False) -> bool:
        simulation_time = int(simulation_time)
        if simulation_time in self._captured_times:
            return False
        if not force and simulation_time not in self.milestones:
            return False
        pedestrians = self._pedestrian_rows(simulation_time)
        shelters = self._shelter_rows(simulation_time)
        cells = self._cell_rows(simulation_time, pedestrians)
        latest_decision = next(
            (decision for decision in reversed(self.decisions) if decision["simulation_time"] <= simulation_time),
            None,
        )
        self.snapshots.append(
            MilestoneSnapshot(
                simulation_time=simulation_time,
                pedestrians=pedestrians,
                shelters=shelters,
                cells=cells,
                outcome=self._outcome(simulation_time, pedestrians),
                latest_decision=None if latest_decision is None else dict(latest_decision),
                absorbing_after_terminal=False,
            )
        )
        self._captured_times.add(simulation_time)
        if (
            simulation_time == 0
            and self.static_predeployment_shelter_ids
            and not self.decision_snapshots
        ):
            snapshot = self.snapshots[-1]
            self.decision_snapshots.append(
                MilestoneSnapshot(
                    simulation_time=0,
                    pedestrians=snapshot.pedestrians,
                    shelters=snapshot.shelters,
                    cells=snapshot.cells,
                    outcome=dict(snapshot.outcome),
                    latest_decision=None,
                    absorbing_after_terminal=False,
                )
            )
        return True

    @staticmethod
    def _write_rows(path: str, rows: Iterable[dict], fieldnames: tuple[str, ...]) -> None:
        temporary = f"{path}.tmp"
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    @staticmethod
    def _write_gzip_rows(
        path: str,
        rows: Iterable[dict],
        fieldnames: tuple[str, ...],
    ) -> None:
        """Atomically stream a potentially large audit table to gzip CSV."""
        temporary = f"{path}.tmp"
        with gzip.open(
            temporary,
            "wt",
            newline="",
            encoding="utf-8",
            compresslevel=6,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _render_axis(
        self,
        axis,
        snapshot: MilestoneSnapshot,
        *,
        compact: bool,
    ) -> None:
        import matplotlib
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Rectangle

        danger_map = matplotlib.colormaps["cividis"]

        if self.road_segments:
            axis.add_collection(
                LineCollection(
                    self.road_segments,
                    colors="#787878",
                    linewidths=0.38 if compact else 0.55,
                    alpha=0.55,
                    zorder=1,
                )
            )

        for cell in snapshot.cells:
            danger_level = float(np.clip(cell["danger"], 0.0, 1.0))
            axis.add_patch(
                Rectangle(
                    (cell["x_min_m"], cell["y_min_m"]),
                    cell["x_max_m"] - cell["x_min_m"],
                    cell["y_max_m"] - cell["y_min_m"],
                    facecolor=danger_map(danger_level),
                    edgecolor=(1.0, 1.0, 1.0, 0.28),
                    linewidth=0.22,
                    alpha=0.58,
                    zorder=0,
                )
            )

        if snapshot.latest_decision is not None:
            selected = int(snapshot.latest_decision["selected_cell"])
            if selected >= 0:
                i, j = divmod(selected, int(self.core.cellY))
                tracker = self.core.cellTracker
                axis.add_patch(
                    Rectangle(
                        (tracker.xEdges[i], tracker.yEdges[j]),
                        tracker.xEdges[i + 1] - tracker.xEdges[i],
                        tracker.yEdges[j + 1] - tracker.yEdges[j],
                        facecolor="none",
                        edgecolor="#f1b514",
                        linewidth=1.8 if compact else 2.4,
                        linestyle="--",
                        zorder=5,
                    )
                )

        if snapshot.pedestrians:
            x = np.asarray([row["x_m"] for row in snapshot.pedestrians], dtype=float)
            y = np.asarray([row["y_m"] for row in snapshot.pedestrians], dtype=float)
            axis.scatter(
                x,
                y,
                s=4.5 if compact else 7.0,
                c="#d7191c",
                edgecolors="white",
                linewidths=0.10 if compact else 0.16,
                alpha=0.86,
                zorder=4,
                rasterized=True,
            )

        if snapshot.shelters:
            initial = [row for row in snapshot.shelters if row["deployment_mode"] == "initial"]
            static = [
                row
                for row in snapshot.shelters
                if row["deployment_mode"] == "static_predeployment"
            ]
            dynamic = [row for row in snapshot.shelters if row["deployment_mode"] == "dynamic"]
            if initial:
                axis.scatter(
                    [row["x_m"] for row in initial], [row["y_m"] for row in initial],
                    marker="^", s=34 if compact else 48, c="#2166ac", edgecolors="white",
                    linewidths=0.6, zorder=6,
                )
            if dynamic:
                axis.scatter(
                    [row["x_m"] for row in dynamic], [row["y_m"] for row in dynamic],
                    marker="*", s=48 if compact else 70, c="#00a6ca", edgecolors="#08306b",
                    linewidths=0.45, zorder=6,
                )
            if static:
                axis.scatter(
                    [row["x_m"] for row in static], [row["y_m"] for row in static],
                    marker="s", s=38 if compact else 55, c="#7b3294", edgecolors="white",
                    linewidths=0.55, zorder=6,
                )

        for hazard in self.core.hazardDS.hazardList.values():
            node = getattr(hazard, "sourceNode", None)
            if node is not None:
                axis.scatter(
                    [float(node.nodeX)], [float(node.nodeY)], marker="X",
                    s=25 if compact else 38, c="#6a040f", edgecolors="white",
                    linewidths=0.45, zorder=7,
                )

        axis.set_xlim(*self.x_limits)
        # Local y is distance south from the map's northern boundary.
        axis.set_ylim(self.y_limits[1], self.y_limits[0])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        outcome = snapshot.outcome
        elapsed_minutes = float(snapshot.simulation_time) * self.time_step_minutes
        axis.set_title(
            f"step {snapshot.simulation_time} ({elapsed_minutes:g} min)\n"
            f"active {outcome['active_population']} | safe {outcome['safe_completed']} | "
            f"casualty {outcome['casualties']}"
            + (
                f"\nabsorbing after terminal step {self.terminal_time}"
                if snapshot.absorbing_after_terminal
                else ""
            ),
            fontsize=8.2 if compact else 10.0,
        )

    @staticmethod
    def _add_danger_colorbar(fig, axes, *, compact: bool) -> None:
        """Add the common fixed danger scale used by every map panel."""
        import matplotlib
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        scalar = ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=1.0),
            cmap=matplotlib.colormaps["cividis"],
        )
        scalar.set_array([])
        colorbar = fig.colorbar(
            scalar,
            ax=axes,
            fraction=0.022 if compact else 0.046,
            pad=0.012 if compact else 0.035,
            aspect=28,
        )
        colorbar.set_label("Normalized cell danger", fontsize=8 if compact else 9)
        colorbar.ax.tick_params(labelsize=7 if compact else 8)

    def _render(self) -> list[str]:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        outputs = []
        ordered = sorted(
            (
                snapshot
                for snapshot in self.snapshots
                if snapshot.simulation_time in self.milestones
            ),
            key=lambda item: item.simulation_time,
        )
        if not ordered:
            ordered = sorted(self.snapshots, key=lambda item: item.simulation_time)
        if self.render_individual_snapshots:
            for snapshot in ordered:
                fig, axis = plt.subplots(figsize=(7.2, 7.0))
                fig.subplots_adjust(left=0.04, right=0.96, bottom=0.045, top=0.90)
                self._render_axis(axis, snapshot, compact=False)
                self._add_danger_colorbar(fig, axis, compact=False)
                fig.suptitle(
                    f"{self.core.address} — {self.core.rl.deployment_strategy.upper()} evacuation",
                    fontsize=12,
                )
                fig.text(
                    0.99, 0.012, "Road network © OpenStreetMap contributors (ODbL)",
                    ha="right", va="bottom", fontsize=7, color="#555555",
                )
                path = os.path.join(
                    self.output_dir,
                    f"milestone_t{snapshot.simulation_time:04d}.png",
                )
                fig.savefig(path, dpi=220, facecolor="white")
                plt.close(fig)
                outputs.append(path)

        columns = len(ordered)
        fig, axes = plt.subplots(
            1,
            columns,
            figsize=(3.35 * columns, 4.25),
            squeeze=False,
        )
        fig.subplots_adjust(left=0.01, right=0.98, bottom=0.16, top=0.73, wspace=0.08)
        for axis, snapshot in zip(axes[0], ordered):
            self._render_axis(axis, snapshot, compact=True)
        self._add_danger_colorbar(fig, list(axes[0]), compact=True)
        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#d7191c", label="Active pedestrian", markersize=6),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#2166ac", label="Initial shelter", markersize=7),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#00a6ca", label="Sequential shelter", markersize=8),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#7b3294", label="Static t=0 shelter", markersize=7),
            Line2D([0], [0], color="#f1b514", linestyle="--", label="Latest priority cell"),
            Line2D([0], [0], marker="X", color="none", markerfacecolor="#6a040f", label="Hazard source", markersize=6),
        ]
        fig.suptitle(
            f"Evacuation milestones: {self.core.address} — {self.core.rl.deployment_strategy.upper()}",
            fontsize=12,
        )
        fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, fontsize=8)
        fig.text(
            0.99, 0.012, "Road network © OpenStreetMap contributors (ODbL)",
            ha="right", va="bottom", fontsize=7, color="#555555",
        )
        png_path = os.path.join(self.output_dir, "evacuation_milestones.png")
        fig.savefig(png_path, dpi=240, facecolor="white")
        outputs.append(png_path)
        if self.render_vector_outputs:
            svg_path = os.path.join(self.output_dir, "evacuation_milestones.svg")
            fig.savefig(svg_path, facecolor="white")
            outputs.append(svg_path)
        plt.close(fig)
        return outputs

    def _render_deployment_sequence(self) -> list[str]:
        """Map every exact implemented candidate and its decision order."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D

        fig, axis = plt.subplots(figsize=(7.4, 7.1))
        if self.road_segments:
            axis.add_collection(
                LineCollection(
                    self.road_segments,
                    colors="#858585",
                    linewidths=0.5,
                    alpha=0.58,
                    zorder=1,
                )
            )
        initial = [
            shelter
            for shelter_id, shelter in sorted(self.core.shelterDS.shelterList.items())
            if int(shelter_id) in self.initial_shelter_ids
        ]
        if initial:
            axis.scatter(
                [shelter.nodeMapped.nodeX for shelter in initial],
                [shelter.nodeMapped.nodeY for shelter in initial],
                marker="^",
                s=62,
                c="#2166ac",
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )
        if self.decisions:
            denominator = max(1, len(self.decisions) - 1)
            for decision in self.decisions:
                order = int(decision["decision_index"])
                color = plt.cm.viridis((order - 1) / denominator)
                marker = "s" if decision["decision_type"] == "static_predeployment" else "o"
                axis.scatter(
                    [decision["candidate_x_m"]],
                    [decision["candidate_y_m"]],
                    marker=marker,
                    s=86,
                    c=[color],
                    edgecolors="#111111",
                    linewidths=0.75,
                    zorder=5,
                )
                axis.annotate(
                    str(order),
                    (decision["candidate_x_m"], decision["candidate_y_m"]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                    zorder=6,
                )
        axis.set_xlim(*self.x_limits)
        axis.set_ylim(self.y_limits[1], self.y_limits[0])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        strategy = str(self.core.rl.deployment_strategy)
        timing = (
            "all additional sites at t=0"
            if strategy in {"none", "initial_only", "static_greedy", "rl_precommit"}
            else "numbers show decision order"
        )
        axis.set_title(
            f"{self.core.address} — {strategy.upper()} candidate implementations\n"
            f"{timing}; {len(self.decisions)} additional shelters"
        )
        legend = [
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#2166ac", label="Common initial shelter", markersize=8),
        ]
        if strategy in {"none", "initial_only", "static_greedy", "rl_precommit"}:
            legend.append(
                Line2D([0], [0], marker="s", color="none", markerfacecolor="#35b779", markeredgecolor="#111111", label="Static predeployment", markersize=7)
            )
        else:
            legend.append(
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#35b779", markeredgecolor="#111111", label="Sequential implementation", markersize=7)
            )
        axis.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=7.5)
        fig.text(
            0.99,
            0.012,
            "Road network © OpenStreetMap contributors (ODbL)",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#555555",
        )
        fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.97))
        paths = []
        suffixes = ("png", "svg") if self.render_vector_outputs else ("png",)
        for suffix in suffixes:
            path = os.path.join(self.output_dir, f"deployment_sequence.{suffix}")
            fig.savefig(path, dpi=240 if suffix == "png" else None, facecolor="white")
            paths.append(path)
        plt.close(fig)
        return paths

    def _render_decision_epochs(self) -> list[str]:
        """Render the regional state at every decision, not only milestones."""
        if not self.decision_snapshots:
            return []
        import math
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        static = str(self.core.rl.deployment_strategy) in {
            "none",
            "initial_only",
            "static_greedy",
            "rl_precommit",
        }
        count = len(self.decision_snapshots)
        columns = min(5, count)
        rows = int(math.ceil(count / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.25 * columns, 4.20 * rows),
            squeeze=False,
        )
        fig.subplots_adjust(left=0.01, right=0.98, bottom=0.15, top=0.73, wspace=0.08)
        for index, (axis, snapshot) in enumerate(
            zip(axes.flat, self.decision_snapshots),
            start=1,
        ):
            self._render_axis(
                axis,
                snapshot,
                compact=True,
            )
            if static:
                axis.set_title(
                    f"Static decision, step 0 (0 min)\n"
                    f"{len(self.decisions)} additional candidates installed simultaneously\n"
                    f"active {snapshot.outcome['active_population']}",
                    fontsize=8.2,
                )
                continue
            decision = self.decisions[index - 1]
            axis.scatter(
                [decision["candidate_x_m"]],
                [decision["candidate_y_m"]],
                s=118,
                marker="o",
                facecolors="none",
                edgecolors="#f1b514",
                linewidths=1.8,
                zorder=8,
            )
            axis.set_title(
                f"Decision {index}, step {decision['simulation_time']} "
                f"({float(decision['simulation_time']) * self.time_step_minutes:g} min)\n"
                f"priority cell {decision['selected_cell']} → OSM {decision['candidate_osm_node_id']}\n"
                f"active {snapshot.outcome['active_population']}",
                fontsize=8.2,
            )
        for axis in list(axes.flat)[count:]:
            axis.axis("off")
        self._add_danger_colorbar(
            fig,
            list(axes.flat)[:count],
            compact=True,
        )
        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#d7191c", markeredgecolor="white", label="Active pedestrian", markersize=6),
            Line2D([0], [0], color="#f1b514", linestyle="--", label="Selected priority cell"),
            Line2D([0], [0], marker="o", color="none", markeredgecolor="#f1b514", markerfacecolor="none", label="Implemented candidate", markersize=8),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#2166ac", label="Common initial shelter", markersize=7),
        ]
        fig.suptitle(
            f"Decision-epoch regional priorities: {self.core.address} — "
            f"{self.core.rl.deployment_strategy.upper()}",
            fontsize=12,
        )
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=7.5,
        )
        fig.text(
            0.99,
            0.008,
            "Road network © OpenStreetMap contributors (ODbL)",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#555555",
        )
        paths = []
        suffixes = ("png", "svg") if self.render_vector_outputs else ("png",)
        for suffix in suffixes:
            path = os.path.join(self.output_dir, f"decision_epochs.{suffix}")
            fig.savefig(path, dpi=240 if suffix == "png" else None, facecolor="white")
            paths.append(path)
        plt.close(fig)
        return paths

    def _append_absorbing_milestones(self, terminal_time: int) -> None:
        """Continue a completed episode with explicit absorbing MDP states.

        The simulator is not advanced after all pedestrians are classified.
        Instead, requested post-terminal milestones repeat the true empty
        terminal state, freeze shelter/hazard layers, and carry an explicit
        absorbing-state flag.  No pedestrian position is synthesized.
        """
        terminal = next(
            snapshot
            for snapshot in self.snapshots
            if snapshot.simulation_time == int(terminal_time)
        )
        if int(terminal.outcome["active_population"]) != 0:
            return
        for milestone in self.milestones:
            if milestone <= terminal_time or milestone in self._captured_times:
                continue
            shelters = tuple(
                {
                    **row,
                    "simulation_time": int(milestone),
                    "absorbing_after_terminal": 1,
                }
                for row in terminal.shelters
            )
            cells = tuple(
                {
                    **row,
                    "simulation_time": int(milestone),
                    "absorbing_after_terminal": 1,
                }
                for row in terminal.cells
            )
            outcome = {
                **terminal.outcome,
                "simulation_time": int(milestone),
                "active_population": 0,
                "absorbing_after_terminal": 1,
                "terminal_time": int(terminal_time),
            }
            self.snapshots.append(
                MilestoneSnapshot(
                    simulation_time=int(milestone),
                    pedestrians=tuple(),
                    shelters=shelters,
                    cells=cells,
                    outcome=outcome,
                    latest_decision=(
                        None
                        if terminal.latest_decision is None
                        else dict(terminal.latest_decision)
                    ),
                    absorbing_after_terminal=True,
                )
            )
            self._captured_times.add(int(milestone))

    def finalize(self, *, terminal_time: Optional[int] = None) -> dict:
        """Write source tables, render figures, and return an artifact manifest."""
        if self._finalized:
            raise RuntimeError("Visualization was already finalized")
        if terminal_time is not None:
            self.terminal_time = int(terminal_time)
            if self.terminal_time < 0 or self.terminal_time > self.horizon:
                raise ValueError("terminal_time must fall within the visualization horizon")
            if self.terminal_time not in self._captured_times:
                self.capture(self.terminal_time, force=True)
            self._append_absorbing_milestones(self.terminal_time)
        if not self.snapshots:
            self.capture(0, force=True)

        pedestrian_path = os.path.join(self.output_dir, "milestone_pedestrians.csv")
        shelter_path = os.path.join(self.output_dir, "milestone_shelters.csv")
        cell_path = os.path.join(self.output_dir, "milestone_cells.csv")
        decision_path = os.path.join(self.output_dir, "deployment_decisions.csv")
        road_path = os.path.join(self.output_dir, "osm_road_segments.csv")
        outcome_path = os.path.join(self.output_dir, "milestone_outcomes.csv")
        decision_cell_path = os.path.join(self.output_dir, "decision_epoch_cells.csv")
        decision_pedestrian_path = os.path.join(
            self.output_dir,
            "decision_epoch_pedestrians.csv.gz",
        )
        self._write_rows(
            pedestrian_path,
            (row for snapshot in self.snapshots for row in snapshot.pedestrians),
            (
                "simulation_time", "agent_id", "x_m", "y_m", "cell_i", "cell_j",
                "group_size", "speed_m_per_minute", "affected",
            ),
        )
        self._write_rows(
            shelter_path,
            (row for snapshot in self.snapshots for row in snapshot.shelters),
            (
                "simulation_time", "shelter_id", "osm_node_id", "x_m", "y_m",
                "cell_i", "cell_j", "capacity", "flow", "status", "initial_shelter",
                "deployment_mode", "absorbing_after_terminal",
            ),
        )
        self._write_rows(
            cell_path,
            (row for snapshot in self.snapshots for row in snapshot.cells),
            (
                "simulation_time", "cell_index", "cell_i", "cell_j", "x_min_m",
                "x_max_m", "y_min_m", "y_max_m", "hazard_state", "danger",
                "active_population",
                "absorbing_after_terminal",
            ),
        )
        self._write_rows(
            outcome_path,
            (snapshot.outcome for snapshot in self.snapshots),
            (
                "simulation_time", "active_population", "safe_completed",
                "ordinary_arrivals", "shelter_evacuated", "casualties",
                "affected_cumulative", "absorbing_after_terminal", "terminal_time",
            ),
        )
        self._write_rows(
            decision_path,
            self.decisions,
            (
                "decision_index", "simulation_time", "strategy", "decision_type",
                "selected_cell", "heuristic_cell", "shelter_id",
                "candidate_osm_node_id", "candidate_x_m", "candidate_y_m",
                "candidate_cell_i", "candidate_cell_j", "capacity_added",
                "rerouted_population",
            ),
        )
        self._write_rows(
            road_path,
            (
                {
                    "segment_id": index,
                    "start_x_m": segment[0][0],
                    "start_y_m": segment[0][1],
                    "end_x_m": segment[1][0],
                    "end_y_m": segment[1][1],
                }
                for index, segment in enumerate(self.road_segments)
            ),
            ("segment_id", "start_x_m", "start_y_m", "end_x_m", "end_y_m"),
        )
        decision_cell_rows = []
        static_strategy = str(self.core.rl.deployment_strategy) in {
            "none",
            "initial_only",
            "static_greedy",
            "rl_precommit",
        }
        for state_index, snapshot in enumerate(self.decision_snapshots):
            decision = (
                None
                if static_strategy or state_index >= len(self.decisions)
                else self.decisions[state_index]
            )
            for cell in snapshot.cells:
                decision_cell_rows.append(
                    {
                        "decision_index": 0 if decision is None else decision["decision_index"],
                        "simulation_time": snapshot.simulation_time,
                        "selected_cell": -1 if decision is None else decision["selected_cell"],
                        "candidate_osm_node_id": "" if decision is None else decision["candidate_osm_node_id"],
                        **cell,
                    }
                )
        self._write_rows(
            decision_cell_path,
            decision_cell_rows,
            (
                "decision_index", "simulation_time", "selected_cell",
                "candidate_osm_node_id", "cell_index", "cell_i", "cell_j",
                "x_min_m", "x_max_m", "y_min_m", "y_max_m", "hazard_state",
                "danger", "active_population", "absorbing_after_terminal",
            ),
        )
        def decision_pedestrian_rows():
            for state_index, snapshot in enumerate(self.decision_snapshots):
                decision = (
                    None
                    if static_strategy or state_index >= len(self.decisions)
                    else self.decisions[state_index]
                )
                for pedestrian in snapshot.pedestrians:
                    yield {
                        "decision_index": 0 if decision is None else decision["decision_index"],
                        "selected_cell": -1 if decision is None else decision["selected_cell"],
                        "candidate_osm_node_id": "" if decision is None else decision["candidate_osm_node_id"],
                        **pedestrian,
                    }

        self._write_gzip_rows(
            decision_pedestrian_path,
            decision_pedestrian_rows(),
            (
                "decision_index", "simulation_time", "selected_cell",
                "candidate_osm_node_id", "agent_id", "x_m", "y_m", "cell_i",
                "cell_j", "group_size", "speed_m_per_minute", "affected",
            ),
        )
        render_paths = [
            *self._render(),
            *self._render_deployment_sequence(),
            *self._render_decision_epochs(),
        ]

        artifact_paths = [
            pedestrian_path,
            shelter_path,
            cell_path,
            decision_path,
            road_path,
            outcome_path,
            decision_cell_path,
            decision_pedestrian_path,
            *render_paths,
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": "OpenStreetMap road graph loaded by the simulator",
            "attribution": "© OpenStreetMap contributors; Open Database License (ODbL)",
            "address": str(self.core.address),
            "city_id": str(getattr(self.core, "cityID", "single_city")),
            "map_spec": {
                "query_mode": str(getattr(self.core, "mapQueryMode", "place")),
                "center": (
                    None
                    if getattr(self.core, "mapCenterLat", None) is None
                    or getattr(self.core, "mapCenterLon", None) is None
                    else [
                        float(self.core.mapCenterLat),
                        float(self.core.mapCenterLon),
                    ]
                ),
                "radius_m": (
                    None
                    if getattr(self.core, "mapRadiusM", None) is None
                    else float(self.core.mapRadiusM)
                ),
                "network_type": "walk",
            },
            "grid_shape": [int(self.core.cellX), int(self.core.cellY)],
            "cell_partition": {
                **dict(getattr(self.core, "cellPartitionDiagnostics", {}) or {}),
                "mode": str(
                    getattr(
                        self.core,
                        "cellPartitionMode",
                        "node_density_adaptive",
                    )
                ),
                "minimum_axis_width_fraction": float(
                    getattr(self.core, "cellPartitionMinWidthFraction", 1e-4)
                ),
            },
            "strategy": str(self.core.rl.deployment_strategy),
            "scenario_seed": int(self.core.scenario_seed),
            "policy_seed": int(self.core.policy_seed),
            "horizon": int(self.horizon),
            "time_step_minutes": self.time_step_minutes,
            "horizon_minutes": float(self.horizon) * self.time_step_minutes,
            "shelter_action_interval_timesteps": int(
                getattr(
                    self.core.rl,
                    "shelter_action_interval",
                    getattr(self.core, "shelterActionInterval", 2),
                )
            ),
            "shelter_action_interval_minutes": (
                int(
                    getattr(
                        self.core.rl,
                        "shelter_action_interval",
                        getattr(self.core, "shelterActionInterval", 2),
                    )
                )
                * self.time_step_minutes
            ),
            "speed_units": "metres_per_minute",
            "free_flow_speed_m_per_minute": float(self.core.maxSpeed),
            "congestion": (
                None
                if getattr(self.core, "congestionModel", None) is None
                else self.core.congestionModel.contract()
            ),
            "requested_milestones": list(self.milestones),
            "captured_milestones": sorted(self._captured_times),
            "decision_epoch_count": len(self.decision_snapshots),
            "terminal_time": self.terminal_time,
            "post_terminal_rule": (
                "requested post-terminal panels are explicitly flagged absorbing states; "
                "no pedestrian positions are synthesized"
            ),
            "individual_snapshot_figures": self.render_individual_snapshots,
            "vector_figures": self.render_vector_outputs,
            "coordinate_system": "local metres; x eastward, y southward; north-up rendering",
            "layers": [
                "OSM road graph", "continuous normalized cell-danger heatmap",
                "one red marker per active pedestrian agent",
                "initial and dynamically installed shelters", "latest selected priority cell",
                "static predeployments", "exact numbered candidate implementation sequence",
                "hazard sources",
            ],
            "visual_encoding": {
                "cell_danger": {
                    "source_field": "dangerLevelByCell",
                    "scale": [0.0, 1.0],
                    "normalization": "fixed across all policies and timesteps",
                    "colormap": "cividis",
                },
                "active_pedestrians": {
                    "marker": "red circle",
                    "unit": "one marker per active pedestrian agent",
                    "terminated_agents_plotted": False,
                },
            },
            "non_interventional": True,
            "artifacts": {
                os.path.basename(path): {
                    "path": path,
                    "sha256": self._sha256(path),
                }
                for path in artifact_paths
            },
        }
        manifest_path = os.path.join(self.output_dir, "visualization_manifest.json")
        temporary = f"{manifest_path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(temporary, manifest_path)
        manifest["manifest_path"] = manifest_path
        self._finalized = True
        return manifest
