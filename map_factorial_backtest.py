#!/usr/bin/env python3
"""Generate matched shelter-decision and evacuation-progress map evidence.

The full design crosses five cities, five population levels, four sampled OSM
shelter-candidate levels, and one to five hazards.  RL, the active-population
heuristic, and static predeployment use the same scenario seed within each
condition. Dynamic policies share the administrator-facing exact-candidate
interface; static placement is explicitly labeled as an anticipative
t=0 comparator.

The maximum episode horizon is 60 one-minute transitions (``stopTime=61``).
Dynamic policies may add one shelter every two transitions. Progress
montages use seven prespecified time boundaries, while a separate numbered map
shows every exact shelter candidate implementation at every decision point.
Episodes that exhaust the pedestrian population early show their true terminal
state and do not fabricate later pedestrian locations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

from CityProfiles import DEFAULT_CITY_PROFILE_PATH, load_city_suite
from multicity_backtest import RUNS_ROOT, _city_overrides
from population_candidate_backtest import (
    DEFAULT_SOURCE_LAUNCH,
    _checkpoint_paths,
    _condition_seed,
    _sha256,
    _source_training_horizons,
    _strict,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DESIGN_PATH = PROJECT_ROOT / "config" / "map_visualization_experiment.json"


@dataclass(frozen=True)
class MapVisualizationDesign:
    experiment_id: str
    population_levels: tuple[int, ...]
    shelter_candidate_levels: tuple[int, ...]
    hazard_count_levels: tuple[int, ...]
    initial_shelters: int
    maximum_additional_shelters: int
    shelter_action_interval: int
    horizon_timesteps: int
    progress_milestones: tuple[int, ...]
    strategies: tuple[str, ...]
    replications_per_cell: int
    early_termination_rule: str
    source_path: Path
    source_sha256: str


def load_map_design(path: str | Path = DEFAULT_DESIGN_PATH) -> MapVisualizationDesign:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "population_levels",
        "shelter_candidate_levels",
        "hazard_count_levels",
        "initial_shelters",
        "maximum_additional_shelters",
        "shelter_action_interval",
        "horizon_timesteps",
        "progress_milestones",
        "strategies",
        "visualization_replications_per_cell",
        "early_termination_rule",
    }
    if set(payload) != required:
        raise ValueError("Map-visualization experiment fields do not match schema")
    if int(payload["schema_version"]) != 2:
        raise ValueError("Unsupported map-visualization schema version")
    populations = tuple(int(value) for value in payload["population_levels"])
    candidates = tuple(int(value) for value in payload["shelter_candidate_levels"])
    hazards = tuple(int(value) for value in payload["hazard_count_levels"])
    milestones = tuple(int(value) for value in payload["progress_milestones"])
    strategies = tuple(str(value).strip().lower() for value in payload["strategies"])
    horizon = int(payload["horizon_timesteps"])
    initial = int(payload["initial_shelters"])
    maximum_additions = int(payload["maximum_additional_shelters"])
    action_interval = int(payload["shelter_action_interval"])
    if populations != (10000, 20000, 30000, 40000, 50000):
        raise ValueError("The map experiment requires the five declared population levels")
    if candidates != (5, 10, 15, 20):
        raise ValueError("Shelter-candidate levels must be 5, 10, 15, and 20")
    if hazards != (1, 2, 3, 4, 5):
        raise ValueError("Hazard counts must be 1 through 5")
    if initial <= 0 or initial >= min(candidates):
        raise ValueError("Initial shelters must be positive and below every candidate level")
    if maximum_additions <= 0:
        raise ValueError("maximum_additional_shelters must be positive")
    if action_interval != 2:
        raise ValueError("The map experiment requires a two-transition action interval")
    if horizon != 60:
        raise ValueError("The map experiment requires exactly 60 transitions")
    if milestones != tuple(sorted(set(milestones))) or milestones[0] != 0 or milestones[-1] != horizon:
        raise ValueError(
            f"Progress milestones must be unique, ordered, and span 0 to {horizon}"
        )
    if strategies != ("rl", "heuristic", "initial_only"):
        raise ValueError("Map comparisons require RL, heuristic, and static predeployment")
    replications = int(payload["visualization_replications_per_cell"])
    if replications <= 0:
        raise ValueError("visualization_replications_per_cell must be positive")
    return MapVisualizationDesign(
        experiment_id=str(payload["experiment_id"]),
        population_levels=populations,
        shelter_candidate_levels=candidates,
        hazard_count_levels=hazards,
        initial_shelters=initial,
        maximum_additional_shelters=maximum_additions,
        shelter_action_interval=action_interval,
        horizon_timesteps=horizon,
        progress_milestones=milestones,
        strategies=strategies,
        replications_per_cell=replications,
        early_termination_rule=str(payload["early_termination_rule"]),
        source_path=resolved,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_map_execution_plan(
    *,
    cities,
    design: MapVisualizationDesign,
) -> dict:
    factor_cells = (
        len(cities)
        * len(design.population_levels)
        * len(design.shelter_candidate_levels)
        * len(design.hazard_count_levels)
        * design.replications_per_cell
    )
    requested_pedestrian_trajectories = (
        len(cities)
        * sum(design.population_levels)
        * len(design.shelter_candidate_levels)
        * len(design.hazard_count_levels)
        * design.replications_per_cell
        * len(design.strategies)
    )
    return {
        "city_ids": [city.city_id for city in cities],
        "population_levels": list(design.population_levels),
        "shelter_candidate_levels": list(design.shelter_candidate_levels),
        "hazard_count_levels": list(design.hazard_count_levels),
        "initial_shelters": design.initial_shelters,
        "maximum_additional_shelters": design.maximum_additional_shelters,
        "shelter_action_interval": design.shelter_action_interval,
        "horizon_timesteps": design.horizon_timesteps,
        "simulator_stop_time": design.horizon_timesteps + 1,
        "progress_milestones": list(design.progress_milestones),
        "strategies": list(design.strategies),
        "replications_per_cell": design.replications_per_cell,
        "factor_cells": factor_cells,
        "episodes": factor_cells * len(design.strategies),
        "requested_pedestrian_trajectories": requested_pedestrian_trajectories,
        "maximum_person_transitions": (
            requested_pedestrian_trajectories * design.horizon_timesteps
        ),
        "decision_sequence_maps": factor_cells,
        "decision_epoch_comparison_maps": factor_cells,
        "progress_comparison_maps": factor_cells,
        "congestion_diagnostic_graphs": factor_cells,
        "total_comparison_and_diagnostic_graphs": factor_cells * 4,
    }


def _source_training_installation_budgets(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> dict[str, int]:
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    budgets = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Source manifest lacks overrides for city {city_id!r}")
        # Zero is the documented legacy behavior: installation count is limited
        # only by decision windows and candidate availability.
        budgets[city_id] = int(overrides.get("maxAdditionalShelters", 0))
    return budgets


def _source_training_action_intervals(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> dict[str, int]:
    """Recover the action cadence, treating absent values as legacy five-step runs."""
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    intervals = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Source manifest lacks overrides for city {city_id!r}")
        intervals[city_id] = int(overrides.get("shelterActionInterval", 5))
    return intervals


def _map_contract(overrides: Mapping) -> dict:
    center = None
    if overrides.get("mapCenterLat") is not None and overrides.get("mapCenterLon") is not None:
        center = [
            float(overrides["mapCenterLat"]),
            float(overrides["mapCenterLon"]),
        ]
    return {
        "query_mode": str(overrides.get("mapQueryMode", "place")),
        "center": center,
        "radius_m": (
            None
            if overrides.get("mapRadiusM") is None
            else float(overrides["mapRadiusM"])
        ),
        "grid": [int(overrides.get("cellX", 0)), int(overrides.get("cellY", 0))],
    }


def _source_training_map_contracts(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> dict[str, dict]:
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    contracts = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Source manifest lacks overrides for city {city_id!r}")
        contracts[city_id] = _map_contract(overrides)
    return contracts


def _congestion_contract(overrides: Mapping) -> dict:
    """Normalize the simulator fields that change pedestrian transition laws.

    A manifest without ``congestionEnabled`` is a pre-congestion legacy run,
    not an implicit use of the new default.  Treating it as disabled prevents
    old checkpoints from being presented as confirmatory evidence under a
    dynamics model on which they were never trained.
    """
    enabled = bool(overrides.get("congestionEnabled", False))
    return {
        "time_step_minutes": float(overrides.get("timeStepMinutes", 1.0)),
        "enabled": enabled,
        "model": "weidmann_physical_link_v1" if enabled else "none",
        "effective_width_m": (
            float(overrides.get("congestionEffectiveWidthM", 3.0))
            if enabled else None
        ),
        "jam_density_ped_per_m2": (
            float(overrides.get("congestionJamDensityPedPerM2", 5.4))
            if enabled else None
        ),
        "shape": (
            float(overrides.get("congestionShape", 1.913))
            if enabled else None
        ),
        "minimum_speed_ratio": (
            float(overrides.get("congestionMinimumSpeedRatio", 0.05))
            if enabled else None
        ),
        "integration_substep_seconds": (
            float(overrides.get("congestionSubstepSeconds", 10.0))
            if enabled else None
        ),
    }


def _source_training_congestion_contracts(
    source_manifest: Mapping,
    city_ids: Sequence[str],
) -> dict[str, dict]:
    effective = source_manifest.get("effective_overrides_by_city")
    if not isinstance(effective, Mapping):
        raise ValueError(
            "Source experiment manifest has no effective_overrides_by_city mapping"
        )
    contracts = {}
    for city_id in city_ids:
        overrides = effective.get(city_id)
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Source manifest lacks overrides for city {city_id!r}")
        contracts[city_id] = _congestion_contract(overrides)
    return contracts


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_strict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {str(key) for row in rows for key in row}
    preferred = (
        "condition_id",
        "city_id",
        "city_scale_rank",
        "population_level",
        "shelter_candidate_level",
        "hazard_count",
        "visualization_replication",
        "deployment_strategy",
        "scenario_seed",
        "policy_seed",
        "horizon_timesteps",
        "shelter_action_interval_timesteps",
        "actual_candidate_count",
        "condition_available",
        "objective_episode_return",
        "safe_completed",
        "casualty",
        "unfinished",
        "simulation_runtime_s",
        "visualization_manifest",
        "deployment_sequence_comparison",
        "decision_epoch_comparison",
        "evacuation_progress_comparison",
        "pedestrian_congestion_diagnostics",
        "rl_heuristic_sequence_identical",
    )
    fieldnames = [field for field in preferred if field in fields]
    fieldnames.extend(sorted(fields.difference(fieldnames)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _identity(row: Mapping) -> tuple:
    return (
        str(row["city_id"]),
        int(float(row["population_level"])),
        int(float(row["shelter_candidate_level"])),
        int(float(row["hazard_count"])),
        int(float(row["visualization_replication"])),
        str(row["deployment_strategy"]),
    )


def _condition_id(city_id: str, population: int, candidates: int, hazards: int, replication: int) -> str:
    return (
        f"{city_id}__population_{population}__candidates_{candidates}__"
        f"hazards_{hazards}__rep_{replication:02d}"
    )


def _manifest_artifact(row: Mapping, filename: str) -> Path:
    manifest_path = Path(str(row["visualization_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest.get("artifacts", {}).get(filename)
    if not isinstance(artifact, Mapping):
        raise KeyError(f"Visualization manifest lacks {filename}: {manifest_path}")
    path = Path(str(artifact["path"]))
    if not path.exists() or _sha256(path) != str(artifact["sha256"]):
        raise RuntimeError(f"Visualization artifact failed checksum validation: {path}")
    return path


def _compose_policy_comparison(
    rows: Sequence[Mapping],
    *,
    filename: str,
    output_path: Path,
    title: str,
    vertical: bool,
    strategies: Sequence[str] = ("rl", "heuristic", "initial_only"),
    footer: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    by_strategy = {str(row["deployment_strategy"]): row for row in rows}
    order = tuple(strategies)
    if not order or any(strategy not in by_strategy for strategy in order):
        raise ValueError(f"Comparison lacks one or more requested strategies: {order}")
    if vertical:
        fig, axes = plt.subplots(
            len(order),
            1,
            figsize=(18.0, 3.8 * len(order)),
            squeeze=False,
        )
        axes_now = axes[:, 0]
    else:
        fig, axes = plt.subplots(
            1,
            len(order),
            figsize=(5.35 * len(order), 5.5),
            squeeze=False,
        )
        axes_now = axes[0]
    for axis, strategy in zip(axes_now, order):
        source = _manifest_artifact(by_strategy[strategy], filename)
        axis.imshow(mpimg.imread(source))
        axis.axis("off")
    fig.suptitle(title, fontsize=13)
    if footer:
        fig.text(
            0.5,
            0.012,
            footer,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#444444",
        )
    fig.tight_layout(rect=(0, 0.035 if footer else 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=220, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output_path)
    return output_path


def _compose_congestion_diagnostics(
    rows: Sequence[Mapping],
    *,
    output_path: Path,
    title: str,
) -> Path:
    """Plot auditable movement diagnostics from each policy's progress table."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = {"rl": "#0072B2", "heuristic": "#E69F00", "initial_only": "#009E73"}
    labels = {"rl": "RL", "heuristic": "Heuristic", "initial_only": "Static"}
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharex=True)
    jam_density = None
    for row in rows:
        strategy = str(row["deployment_strategy"])
        manifest_path = Path(str(row["visualization_manifest"]))
        progress_path = manifest_path.parent.parent / "progress.csv"
        progress = _read_csv(progress_path)
        if not progress:
            raise RuntimeError(f"Congestion diagnostic lacks progress data: {progress_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest.get("congestion") or {}
        jam_density = float(contract.get("jam_density_ped_per_m2", 5.4))
        time_step = float(manifest.get("time_step_minutes", 1.0))
        elapsed = [float(item["timestep"]) * time_step for item in progress]
        axes[0].plot(
            elapsed,
            [float(item["mean_congestion_speed_ratio"]) for item in progress],
            color=colors[strategy],
            linewidth=1.5,
            label=labels[strategy],
        )
        axes[1].plot(
            elapsed,
            [float(item["minimum_congestion_speed_ratio"]) for item in progress],
            color=colors[strategy],
            linewidth=1.5,
        )
        axes[2].plot(
            elapsed,
            [float(item["maximum_link_density_ped_per_m2"]) for item in progress],
            color=colors[strategy],
            linewidth=1.5,
        )
    axes[0].axhline(
        1.0, color="#666666", linewidth=1.0, linestyle="--",
        label="uncongested multiplier 1.0",
    )
    if jam_density is not None:
        axes[2].axhline(
            jam_density, color="#666666", linewidth=1.0, linestyle="--",
            label=f"jam reference {jam_density:g} ped/m²",
        )
    panel_specs = (
        ("Population-mean congestion effect", "Mean speed ratio"),
        ("Most congested occupied link", "Minimum speed ratio"),
        ("Peak physical-link density", "Pedestrians per m²"),
    )
    for axis, (panel_title, ylabel) in zip(axes, panel_specs):
        axis.set_title(panel_title)
        axis.set_xlabel("Elapsed time (min)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Pedestrian congestion diagnostics — {title}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=220, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output_path)
    return output_path


def _compose_condition(rows: Sequence[Mapping], figure_root: Path) -> tuple[Path, Path, Path, Path]:
    reference = rows[0]
    condition_id = str(reference["condition_id"])
    city_label = str(reference.get("address") or reference["city_id"])
    title = (
        f"{city_label} | population {int(float(reference['population_level'])):,} | "
        f"candidates {reference['shelter_candidate_level']} | hazards {reference['hazard_count']}"
    )
    output_dir = figure_root / str(reference["city_id"]) / condition_id
    sequence = _compose_policy_comparison(
        rows,
        filename="deployment_sequence.png",
        output_path=output_dir / "shelter_decision_sequence_comparison.png",
        title=f"Exact shelter candidate implementations — {title}",
        vertical=False,
    )
    decisions = _compose_policy_comparison(
        rows,
        filename="decision_epochs.png",
        output_path=output_dir / "priority_decision_epoch_comparison.png",
        title=f"Regional priority at every shelter decision — {title}",
        vertical=True,
        strategies=("rl", "heuristic"),
        footer=(
            "Static predeployment has no online decision epochs; its five "
            "simultaneous t=0 implementations are shown in the shelter-sequence figure."
        ),
    )
    progress = _compose_policy_comparison(
        rows,
        filename="evacuation_milestones.png",
        output_path=output_dir / "evacuation_progress_comparison.png",
        title=(
            "Evacuation progress over the "
            f"{int(float(reference['horizon_timesteps']))}-minute horizon — {title}"
        ),
        vertical=True,
    )
    congestion = _compose_congestion_diagnostics(
        rows,
        output_path=output_dir / "pedestrian_congestion_diagnostics.png",
        title=title,
    )
    return sequence, decisions, progress, congestion


def _decision_sequence(row: Mapping) -> tuple[tuple[str, str], ...]:
    path = _manifest_artifact(row, "deployment_decisions.csv")
    return tuple(
        (record["selected_cell"], record["candidate_osm_node_id"])
        for record in _read_csv(path)
    )


def _completion_audit(
    *,
    rows: Sequence[Mapping],
    design: MapVisualizationDesign,
    plan: Mapping,
    figure_paths: Sequence[Path],
) -> dict:
    """Validate scientific and artifact invariants before marking completion.

    Only fields persisted in the evaluation table are used, so resumed and
    uninterrupted executions receive the same audit.
    """

    expected_policies = set(design.strategies)
    by_condition: dict[str, list[Mapping]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition_id"]), []).append(row)

    policy_sets_complete = True
    dynamic_observation_parity = True
    common_hazard_paths = True
    outcome_accounting_balanced = True
    dynamic_budget_respected = True
    identical_dynamic_sequences = 0
    for condition_rows in by_condition.values():
        policy_sets_complete &= {
            str(row["deployment_strategy"]) for row in condition_rows
        } == expected_policies
        dynamic = [
            row
            for row in condition_rows
            if str(row["deployment_strategy"]) in {"rl", "heuristic"}
        ]
        dynamic_observation_parity &= (
            len(dynamic) == 2
            and len({str(row["initial_observation_digest"]) for row in dynamic}) == 1
        )
        common_hazard_paths &= (
            len(condition_rows) == len(expected_policies)
            and len({str(row["hazard_trajectory_digest"]) for row in condition_rows}) == 1
        )
        if len(dynamic) == 2:
            identical_dynamic_sequences += int(
                str(dynamic[0].get("rl_heuristic_sequence_identical", "")).lower()
                in {"true", "1"}
            )
        for row in condition_rows:
            population = int(float(row["initial_population"]))
            classified = (
                int(float(row["safe_completed"]))
                + int(float(row["casualty"]))
                + int(float(row["unfinished"]))
            )
            outcome_accounting_balanced &= classified == population
            if str(row["deployment_strategy"]) in {"rl", "heuristic"}:
                expected_additions = min(
                    int(design.maximum_additional_shelters),
                    max(
                        0,
                        int(float(row["actual_candidate_count"]))
                        - int(design.initial_shelters),
                    ),
                )
                dynamic_budget_respected &= (
                    int(float(row["deployments_made"])) == expected_additions
                )

    expected_figures = int(plan["total_comparison_and_diagnostic_graphs"])
    observed_figures = len(tuple(figure_paths))
    audit = {
        "expected_rows": int(plan["episodes"]),
        "observed_rows": len(rows),
        "expected_conditions": int(plan["factor_cells"]),
        "observed_conditions": len(by_condition),
        "expected_comparison_and_diagnostic_figures": expected_figures,
        "observed_comparison_and_diagnostic_figures": observed_figures,
        "policy_sets_complete": bool(policy_sets_complete),
        "dynamic_initial_observation_parity": bool(dynamic_observation_parity),
        "common_hazard_trajectory_within_condition": bool(common_hazard_paths),
        "outcome_accounting_balanced": bool(outcome_accounting_balanced),
        "dynamic_installation_budget_respected": bool(dynamic_budget_respected),
        "rl_heuristic_identical_sequence_conditions": int(
            identical_dynamic_sequences
        ),
    }
    audit["passed"] = bool(
        audit["observed_rows"] == audit["expected_rows"]
        and audit["observed_conditions"] == audit["expected_conditions"]
        and audit["observed_comparison_and_diagnostic_figures"]
        == audit["expected_comparison_and_diagnostic_figures"]
        and policy_sets_complete
        and dynamic_observation_parity
        and common_hazard_paths
        and outcome_accounting_balanced
        and dynamic_budget_respected
    )
    return audit


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-config", type=Path, default=DEFAULT_DESIGN_PATH)
    parser.add_argument("--city-profiles", default=DEFAULT_CITY_PROFILE_PATH)
    parser.add_argument("--source-launch-dir", type=Path, default=DEFAULT_SOURCE_LAUNCH)
    parser.add_argument("--launch-id", default="factorial_maps_seed_20260908")
    parser.add_argument("--launch-seed", type=int, default=20260908)
    parser.add_argument("--policy-replication", type=int, default=1)
    parser.add_argument("--cities", default="all")
    parser.add_argument("--machine", default="map_factorial")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-population", type=int, default=10000)
    parser.add_argument("--pilot-candidates", type=int, default=10)
    parser.add_argument("--pilot-hazards", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    design = load_map_design(args.design_config)
    city_suite = load_city_suite(args.city_profiles)
    requested = None if args.cities.strip().lower() == "all" else args.cities.split(",")
    cities = city_suite.select(requested)
    if args.pilot:
        cities = cities[:1]
        for name, value, levels in (
            ("pilot population", args.pilot_population, design.population_levels),
            ("pilot candidates", args.pilot_candidates, design.shelter_candidate_levels),
            ("pilot hazards", args.pilot_hazards, design.hazard_count_levels),
        ):
            if value not in levels:
                raise ValueError(f"{name} must be one of {list(levels)}")
        design = MapVisualizationDesign(
            **{
                **design.__dict__,
                "population_levels": (args.pilot_population,),
                "shelter_candidate_levels": (args.pilot_candidates,),
                "hazard_count_levels": (args.pilot_hazards,),
            }
        )
    plan = build_map_execution_plan(cities=cities, design=design)
    source_launch = args.source_launch_dir.expanduser().resolve()
    source_manifest_path = source_launch / "experiment_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise ValueError("Source policy launch must be complete")
    checkpoint = _checkpoint_paths(source_launch, args.policy_replication)[
        args.policy_replication
    ]
    source_horizons = _source_training_horizons(
        source_manifest, [city.city_id for city in cities]
    )
    source_installation_budgets = _source_training_installation_budgets(
        source_manifest,
        [city.city_id for city in cities],
    )
    source_action_intervals = _source_training_action_intervals(
        source_manifest,
        [city.city_id for city in cities],
    )
    source_map_contracts = _source_training_map_contracts(
        source_manifest,
        [city.city_id for city in cities],
    )
    source_congestion_contracts = _source_training_congestion_contracts(
        source_manifest,
        [city.city_id for city in cities],
    )
    expected_congestion_contracts = {
        city.city_id: _congestion_contract(
            _city_overrides(city_suite, city, {}, "stochastic")
        )
        for city in cities
    }
    expected_map_contracts = {
        city.city_id: _map_contract(
            _city_overrides(city_suite, city, {}, "stochastic")
        )
        for city in cities
    }
    horizon_mismatches = {
        city_id: horizon
        for city_id, horizon in source_horizons.items()
        if horizon != design.horizon_timesteps
    }
    installation_budget_mismatches = {
        city_id: budget
        for city_id, budget in source_installation_budgets.items()
        if budget != design.maximum_additional_shelters
    }
    action_interval_mismatches = {
        city_id: interval
        for city_id, interval in source_action_intervals.items()
        if interval != design.shelter_action_interval
    }
    map_contract_mismatches = {
        city_id: {
            "source": source_map_contracts[city_id],
            "evaluation": expected_map_contracts[city_id],
        }
        for city_id in source_map_contracts
        if source_map_contracts[city_id] != expected_map_contracts[city_id]
    }
    congestion_contract_mismatches = {
        city_id: {
            "source": source_congestion_contracts[city_id],
            "evaluation": expected_congestion_contracts[city_id],
        }
        for city_id in source_congestion_contracts
        if source_congestion_contracts[city_id]
        != expected_congestion_contracts[city_id]
    }
    source_policy_count = int(source_manifest.get("policy_replicates", 0))
    source_converged = bool(
        source_manifest.get("training_convergence", {}).get(
            "all_policies_converged", False
        )
    )
    checkpoint_compatibility = {
        "training_horizon_matches": not bool(horizon_mismatches),
        "installation_budget_matches": not bool(installation_budget_mismatches),
        "action_interval_matches": not bool(action_interval_mismatches),
        "map_contract_matches": not bool(map_contract_mismatches),
        "congestion_contract_matches": not bool(congestion_contract_mismatches),
        "source_training_converged": source_converged,
    }
    checkpoint_compatibility["ready_for_confirmatory_maps"] = all(
        checkpoint_compatibility.values()
    )
    plan.update(
        {
            "mode": "pilot" if args.pilot else "full_visualization_matrix",
            "source_launch": str(source_launch),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_training_horizon_by_city": source_horizons,
            "source_training_horizon_mismatches": horizon_mismatches,
            "source_training_installation_budget_by_city": source_installation_budgets,
            "source_training_installation_budget_mismatches": installation_budget_mismatches,
            "source_training_action_interval_by_city": source_action_intervals,
            "source_training_action_interval_mismatches": action_interval_mismatches,
            "source_training_map_contract_by_city": source_map_contracts,
            "evaluation_map_contract_by_city": expected_map_contracts,
            "source_training_map_contract_mismatches": map_contract_mismatches,
            "source_training_congestion_contract_by_city": source_congestion_contracts,
            "evaluation_congestion_contract_by_city": expected_congestion_contracts,
            "source_training_congestion_contract_mismatches": congestion_contract_mismatches,
            "source_policy_replicates": source_policy_count,
            "source_training_converged": source_converged,
            "checkpoint_compatibility": checkpoint_compatibility,
            "checkpoint": str(checkpoint),
            "checkpoint_available": checkpoint.is_file(),
        }
    )
    launch_id = f"{args.launch_id}_pilot" if args.pilot and not args.launch_id.endswith("_pilot") else args.launch_id
    launch_dir = Path(RUNS_ROOT) / launch_id
    figure_root = PROJECT_ROOT / "publication_graphs" / "factorial_maps" / launch_id
    launch_dir.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    table_path = launch_dir / "map_factorial_evaluation.csv"
    index_path = figure_root / "map_factorial_figure_index.csv"
    manifest_path = launch_dir / "map_factorial_manifest.json"
    pilot_limitations = []
    if args.pilot and horizon_mismatches:
        pilot_limitations.append(
            "Pilot checkpoint was trained at a different horizon; maps validate the pipeline and are not confirmatory policy evidence."
        )
    if args.pilot and installation_budget_mismatches:
        pilot_limitations.append(
            "Pilot checkpoint predates the fixed five-addition resource budget; confirmatory training must use the evaluation budget."
        )
    if args.pilot and action_interval_mismatches:
        pilot_limitations.append(
            "Pilot checkpoint used a different shelter-decision cadence; maps validate mechanics only and are not policy-performance evidence."
        )
    if args.pilot and map_contract_mismatches:
        pilot_limitations.append(
            "Pilot checkpoint was trained on different OSM footprints; maps validate mechanics only and are not confirmatory transfer evidence."
        )
    if args.pilot and congestion_contract_mismatches:
        pilot_limitations.append(
            "Pilot checkpoint predates the pedestrian link-congestion transition law; maps validate mechanics only and are not policy-performance evidence."
        )
    manifest = {
        "schema_version": 1,
        "status": "planned" if args.dry_run else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "launch_id": launch_id,
        "launch_seed": int(args.launch_seed),
        "design": {
            "experiment_id": design.experiment_id,
            "path": str(design.source_path),
            "sha256": design.source_sha256,
            "early_termination_rule": design.early_termination_rule,
        },
        "execution_plan": plan,
        "scientific_contract": {
            "dynamic_interface_parity": "RL and heuristic share observation, exact-candidate mask, action timing, budget, and executor",
            "fixed_resource_budget": f"at most {design.maximum_additional_shelters} additional shelters, independent of candidate-pool size",
            "decision_cadence": (
                "one regional shelter decision every "
                f"{design.shelter_action_interval} one-minute transitions"
            ),
            "static_policy_role": "anticipative comparator; additional shelters are predeployed at t=0",
            "common_random_numbers": True,
            "candidate_identity": "OSM node id plus local coordinates and decision order",
            "progress_rule": "prespecified milestones plus true early terminal state",
            "pedestrian_motion": expected_congestion_contracts,
        },
        "limitations": pilot_limitations,
        "artifacts": {
            "evaluation_table": str(table_path),
            "figure_index": str(index_path),
            "figure_root": str(figure_root),
        },
    }
    _write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(f"[MAP FACTORIAL PLAN] artifact={manifest_path}")
        return 0
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.policy_replication > source_policy_count:
        raise ValueError("Requested policy replication is absent from the source launch")
    if not args.pilot:
        checkpoint_issues = []
        if horizon_mismatches:
            checkpoint_issues.append(f"{design.horizon_timesteps}-transition training horizon")
        if installation_budget_mismatches:
            checkpoint_issues.append(
                f"fixed {design.maximum_additional_shelters}-addition resource budget"
            )
        if action_interval_mismatches:
            checkpoint_issues.append(
                f"{design.shelter_action_interval}-transition shelter-action interval"
            )
        if map_contract_mismatches:
            checkpoint_issues.append("exact enlarged OSM study footprints")
        if congestion_contract_mismatches:
            checkpoint_issues.append("exact pedestrian link-congestion contract")
        if not source_converged:
            checkpoint_issues.append("passed source-training convergence audit")
        if checkpoint_issues:
            raise ValueError(
                "Full map evidence rejected the source policy; required: "
                + "; ".join(checkpoint_issues)
            )
    if table_path.exists() and not args.resume:
        raise FileExistsError(f"Use --resume or a new --launch-id: {table_path}")

    from backtest import _run_episode

    rows = _read_csv(table_path) if args.resume else []
    completed = {_identity(row) for row in rows}
    base_overrides = {
        city.city_id: _city_overrides(city_suite, city, {}, "stochastic")
        for city in cities
    }
    for overrides in base_overrides.values():
        if source_manifest.get("learning_rate") is not None:
            overrides["learningRate"] = float(source_manifest["learning_rate"])
        if source_manifest.get("ppo_rollout_episodes") is not None:
            overrides["ppoRolloutEpisodes"] = int(
                source_manifest["ppo_rollout_episodes"]
            )
    condition_number = 0
    for city in cities:
        for population_index, population in enumerate(design.population_levels, start=1):
            for candidate_index, candidates in enumerate(design.shelter_candidate_levels, start=1):
                for hazard_index, hazards in enumerate(design.hazard_count_levels, start=1):
                    for visual_replication in range(1, design.replications_per_cell + 1):
                        condition_number += 1
                        condition_id = _condition_id(
                            city.city_id,
                            population,
                            candidates,
                            hazards,
                            visual_replication,
                        )
                        scenario_seed = _condition_seed(
                            args.launch_seed,
                            city.scale_rank,
                            population_index,
                            candidate_index,
                            visual_replication,
                            810 + hazard_index,
                        )
                        policy_seed = _condition_seed(
                            args.launch_seed,
                            city.scale_rank,
                            population_index,
                            candidate_index,
                            visual_replication,
                            910 + hazard_index,
                        )
                        for strategy in design.strategies:
                            identity = (
                                city.city_id,
                                population,
                                candidates,
                                hazards,
                                visual_replication,
                                strategy,
                            )
                            if identity in completed:
                                continue
                            overrides = dict(base_overrides[city.city_id])
                            overrides.update(
                                {
                                    "pedVol": population,
                                    "shelterCanVol": candidates,
                                    "initShelterVol": design.initial_shelters,
                                    "maxAdditionalShelters": design.maximum_additional_shelters,
                                    "shelterActionInterval": design.shelter_action_interval,
                                    "hazardVol": hazards,
                                    "stopTime": design.horizon_timesteps + 1,
                                }
                            )
                            print(
                                f"[MAP FACTORIAL] {condition_number}/{plan['factor_cells']} "
                                f"city={city.city_id} population={population} "
                                f"candidates={candidates} hazards={hazards} strategy={strategy}",
                                flush=True,
                            )
                            started = time.perf_counter()
                            result = _run_episode(
                                replication=condition_number,
                                machine=args.machine,
                                phase=str(
                                    launch_dir
                                    / "episodes"
                                    / city.city_id
                                    / condition_id
                                    / strategy
                                ),
                                strategy=strategy,
                                train_mode=False,
                                scenario_seed=scenario_seed,
                                policy_seed=policy_seed,
                                checkpoint_path=str(checkpoint),
                                diagnostics_path=str(launch_dir / "unused_diagnostics.csv"),
                                overrides=overrides,
                                visualization_enabled=True,
                                visualization_milestones=",".join(
                                    str(value) for value in design.progress_milestones
                                ),
                                visualization_individual_snapshots=False,
                                visualization_vector_outputs=False,
                            )
                            actual_candidates = int(result["active_shelters"]) + int(
                                result["remaining_candidates"]
                            )
                            result.update(
                                {
                                    "condition_id": condition_id,
                                    "city_id": city.city_id,
                                    "city_scale_rank": city.scale_rank,
                                    "population_level": population,
                                    "shelter_candidate_level": candidates,
                                    "hazard_count": hazards,
                                    "visualization_replication": visual_replication,
                                    "horizon_timesteps": design.horizon_timesteps,
                                    "shelter_action_interval_timesteps": (
                                        design.shelter_action_interval
                                    ),
                                    "stop_time_config": design.horizon_timesteps + 1,
                                    "actual_candidate_count": actual_candidates,
                                    "condition_available": actual_candidates == candidates,
                                    "simulation_runtime_s": time.perf_counter() - started,
                                    "source_policy_replication": args.policy_replication,
                                }
                            )
                            rows.append(result)
                            completed.add(identity)
                            _write_csv(table_path, rows)

                        condition_rows = [
                            row
                            for row in rows
                            if str(row.get("condition_id")) == condition_id
                        ]
                        if len(condition_rows) == len(design.strategies):
                            dynamic = [
                                row
                                for row in condition_rows
                                if row["deployment_strategy"] in {"rl", "heuristic"}
                            ]
                            if len(dynamic) != 2:
                                raise RuntimeError("Condition lacks both dynamic policies")
                            if str(dynamic[0]["initial_observation_digest"]) != str(
                                dynamic[1]["initial_observation_digest"]
                            ):
                                raise RuntimeError(
                                    f"RL/heuristic observation parity failed: {condition_id}"
                                )
                            rl_row = next(
                                row
                                for row in dynamic
                                if row["deployment_strategy"] == "rl"
                            )
                            heuristic_row = next(
                                row
                                for row in dynamic
                                if row["deployment_strategy"] == "heuristic"
                            )
                            identical_sequence = (
                                _decision_sequence(rl_row)
                                == _decision_sequence(heuristic_row)
                            )
                            sequence, decisions, progress, congestion = _compose_condition(
                                condition_rows, figure_root
                            )
                            for row in condition_rows:
                                row["deployment_sequence_comparison"] = str(sequence)
                                row["decision_epoch_comparison"] = str(decisions)
                                row["evacuation_progress_comparison"] = str(progress)
                                row["pedestrian_congestion_diagnostics"] = str(congestion)
                                row["rl_heuristic_sequence_identical"] = identical_sequence
                            _write_csv(table_path, rows)
                            _write_csv(index_path, rows)

    unavailable = sorted(
        {
            str(row["condition_id"])
            for row in rows
            if str(row.get("condition_available", "")).lower() not in {"true", "1"}
        }
    )
    expected_rows = int(plan["episodes"])
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Map-factorial row count incomplete: expected={expected_rows}, observed={len(rows)}"
        )
    figure_paths = sorted(
        path for path in figure_root.rglob("*.png") if path.is_file()
    )
    completion_audit = _completion_audit(
        rows=rows,
        design=design,
        plan=plan,
        figure_paths=figure_paths,
    )
    if not completion_audit["passed"]:
        manifest.update(
            {
                "status": "audit_failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "completion_audit": completion_audit,
            }
        )
        _write_json(manifest_path, manifest)
        _write_json(figure_root / "map_factorial_manifest.json", manifest)
        raise RuntimeError(
            "Map-factorial completion audit failed; inspect completion_audit in "
            f"{manifest_path}"
        )
    manifest.update(
        {
            "status": (
                "pilot_complete"
                if args.pilot
                else "complete_with_unavailable_conditions"
                if unavailable
                else "complete"
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": len(rows),
            "completion_audit": completion_audit,
            "unavailable_candidate_conditions": unavailable,
            "artifact_hashes": {
                "evaluation_table": _sha256(table_path),
                "figure_index": _sha256(index_path),
                "comparison_figures": {
                    str(path.relative_to(figure_root)): _sha256(path)
                    for path in figure_paths
                },
                "checkpoint": _sha256(checkpoint),
            },
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(figure_root / "map_factorial_manifest.json", manifest)
    print(f"[MAP FACTORIAL COMPLETE] artifact={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
