#!/usr/bin/env python3
"""Controlled extreme-policy and computational sensitivity experiments.

This experiment deliberately separates policy/mechanism evidence from full
evacuation-simulator timing.  The controlled contextual cases use the
production neural architecture; road-network preprocessing uses the five
registered cached OSM graphs; legacy full-simulator episode timing is retained
as a clearly labeled incomplete observational sample.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn

from CellPartitioning import build_cell_partition
from DecisionInterface import (
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_SLICE,
    INFRA_FEATURE_SLICE,
    PED_FEATURE_SLICE,
)
from GNN import fit_gnn, grid_edge_index
from or_journal_experiments import _policy


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "extreme_sensitivity_experiment.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "extreme_sensitivity_full_20260909"


@dataclass(frozen=True)
class CitySpec:
    city_id: str
    display_name: str
    radius_m: float
    scale_rank: int
    graph_path: Path


def _strict(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_strict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _seed(base: int, *parts: int) -> int:
    return int(
        np.random.SeedSequence([int(base), *(int(part) for part in parts)]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def load_city_specs(
    config: Mapping,
    *,
    require_cached_graph: bool = False,
) -> tuple[CitySpec, ...]:
    profile_path = PROJECT_ROOT / config["city_complexity"]["profile_path"]
    profiles = _read_json(profile_path)
    cities = []
    for value in profiles["cities"]:
        query = json.dumps(
            {
                "address": value["address"],
                "query_mode": "point",
                "center": [value["center_lat"], value["center_lon"]],
                "radius_m": float(value["radius_m"]),
                "network_type": "walk",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        safe_address = "".join(
            character if character.isalnum() else "_"
            for character in value["address"]
        ).strip("_")
        graph_path = PROJECT_ROOT / "cache" / f"graph_walk_{safe_address}_{key}.graphml"
        if require_cached_graph and not graph_path.exists():
            raise FileNotFoundError(f"Registered graph cache is missing: {graph_path}")
        cities.append(
            CitySpec(
                city_id=str(value["city_id"]),
                display_name=str(value["display_name"]),
                radius_m=float(value["radius_m"]),
                scale_rank=int(value["scale_rank"]),
                graph_path=graph_path.resolve(),
            )
        )
    return tuple(cities)


def sensitivity_conditions(config: Mapping, cities: Sequence[CitySpec]) -> list[dict]:
    base = config["base_condition"]
    by_city = {city.city_id: city for city in cities}
    conditions: dict[tuple[int, str, int], dict] = {}

    def add(population: int, city_id: str, side: int, axis: str, order: int, label: str):
        key = (int(population), str(city_id), int(side))
        row = conditions.setdefault(
            key,
            {
                "population": int(population),
                "city_id": str(city_id),
                "city": by_city[str(city_id)],
                "grid_side": int(side),
                "axes": [],
            },
        )
        row["axes"].append({"axis": axis, "order": int(order), "label": str(label)})

    for order, population in enumerate(config["population_levels"], start=1):
        add(
            int(population),
            str(base["city_id"]),
            int(base["grid_side"]),
            "population",
            order,
            f"{int(population) // 1000}k",
        )
    for order, city in enumerate(cities, start=1):
        add(
            int(base["population"]),
            city.city_id,
            int(base["grid_side"]),
            "city_complexity",
            order,
            city.display_name.split(",")[0],
        )
    for order, side in enumerate(config["grid_levels"], start=1):
        add(
            int(base["population"]),
            str(base["city_id"]),
            int(side),
            "cell_dimensionality",
            order,
            f"{int(side)}×{int(side)}",
        )
    output = []
    for condition_id, row in enumerate(conditions.values(), start=1):
        output.append({**row, "condition_id": int(condition_id)})
    return output


_DISTANCE_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _physical_distances(side: int, radius_m: float) -> np.ndarray:
    key = (int(side), float(radius_m))
    cached = _DISTANCE_CACHE.get(key)
    if cached is not None:
        return cached
    coordinates = np.asarray(
        [(i, j) for i in range(int(side)) for j in range(int(side))], dtype=float
    )
    grid_distance = np.abs(
        coordinates[:, None, :] - coordinates[None, :, :]
    ).sum(axis=2)
    cell_step_m = 2.0 * float(radius_m) / float(max(1, int(side) - 1))
    cached = grid_distance * cell_step_m
    _DISTANCE_CACHE[key] = cached
    return cached


def extreme_case(
    rng: np.random.Generator,
    variant: str,
    *,
    side: int,
    population: int,
    city: CitySpec,
    service_range_m: float,
) -> dict:
    if int(side) < 3 or int(side) % 2 == 0:
        raise ValueError("Extreme grids must have an odd side of at least three")
    center_i = int(side) // 2
    center_j = int(side) // 2
    center = center_i * int(side) + center_j
    ring = np.asarray(
        [
            (center_i + di) * int(side) + (center_j + dj)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if not (di == 0 and dj == 0)
        ],
        dtype=int,
    )
    cells = int(side) ** 2
    if variant == "asymmetric_ring":
        hot_position = int(rng.integers(0, ring.size))
        weights = np.full(ring.size, 0.45 / (ring.size - 1), dtype=float)
        weights[hot_position] = 0.55
        weights += rng.normal(0.0, 0.003, ring.size)
        weights = np.clip(weights, 0.001, None)
        weights /= weights.sum()
    else:
        weights = rng.dirichlet(np.full(ring.size, 45.0))
    counts = rng.multinomial(int(population), weights)
    active = np.zeros(cells, dtype=np.float32)
    active[ring] = (counts / float(population)).astype(np.float32)

    danger = np.full(cells, 0.65, dtype=np.float32)
    danger[ring] = rng.uniform(0.20, 0.34, ring.size)
    danger[center] = float(rng.uniform(0.01, 0.05))
    if variant == "dangerous_center":
        danger[center] = float(rng.uniform(0.88, 0.98))
        danger[ring] = rng.uniform(0.03, 0.14, ring.size)
    elif variant == "asymmetric_ring":
        danger[ring] = rng.uniform(0.10, 0.22, ring.size)

    mask = np.zeros(cells, dtype=np.bool_)
    mask[ring] = True
    mask[center] = variant != "center_unavailable"
    distances = _physical_distances(int(side), float(city.radius_m))
    accessibility = np.exp(-distances / float(service_range_m))
    service = active.astype(float) @ accessibility
    utility = service - 0.45 * danger.astype(float)
    utility[~mask] = -np.inf
    oracle = int(np.argmax(utility))
    feasible = np.flatnonzero(mask)
    heuristic = int(feasible[np.argmax(active[feasible])])

    static_start = time.perf_counter_ns()
    expected_active = np.zeros(cells, dtype=float)
    expected_active[ring] = 1.0 / float(ring.size)
    expected_service = expected_active @ accessibility
    expected_danger = np.full(cells, 0.22, dtype=float)
    static_utility = expected_service - 0.45 * expected_danger
    static_utility[~mask] = -np.inf
    static_action = int(np.argmax(static_utility))
    static_planning_ms = (time.perf_counter_ns() - static_start) / 1_000_000.0

    cell_features = np.zeros((cells, len(CELL_FEATURE_NAMES)), dtype=np.float32)
    cell_features[:, 0] = active
    cell_features[:, 1] = np.where(active > 0.0, 0.35 + 0.65 * danger, 0.0)
    cell_features[:, 2] = danger
    global_features = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    return {
        "variant": str(variant),
        "population": int(population),
        "city_id": city.city_id,
        "city_scale_rank": int(city.scale_rank),
        "map_radius_m": float(city.radius_m),
        "grid_side": int(side),
        "center": int(center),
        "ring": ring,
        "active": active,
        "danger": danger,
        "mask": mask,
        "cell_features": cell_features,
        "global_features": global_features,
        "utility": utility.astype(np.float32),
        "oracle": int(oracle),
        "heuristic": int(heuristic),
        "static": int(static_action),
        "static_planning_ms": float(static_planning_ms),
    }


def case_batch(cases: Sequence[Mapping], side: int):
    cells = int(side) ** 2
    features = np.stack([case["cell_features"] for case in cases])
    globals_ = np.stack([case["global_features"] for case in cases])
    flattened = torch.as_tensor(
        features.reshape(-1, len(CELL_FEATURE_NAMES)),
        dtype=torch.float32,
    )
    batch = torch.arange(len(cases), dtype=torch.long).repeat_interleave(cells)
    graph = fit_gnn(
        flattened[:, PED_FEATURE_SLICE],
        flattened[:, HAZARD_FEATURE_SLICE],
        flattened[:, INFRA_FEATURE_SLICE],
        x_global=torch.as_tensor(globals_, dtype=torch.float32),
        edge_index=torch.as_tensor(grid_edge_index(side, side), dtype=torch.long),
        batch=batch,
    )
    masks = torch.as_tensor(np.stack([case["mask"] for case in cases]), dtype=torch.bool)
    utility = torch.as_tensor(
        np.stack([case["utility"] for case in cases]), dtype=torch.float32
    )
    return graph, masks, utility


def train_policies(
    config: Mapping,
    cities: Sequence[CitySpec],
    output_dir: Path,
    *,
    quick: bool,
    base_seed: int,
):
    training = config["policy_training"]
    policy_seeds = int(
        training["quick_policy_seeds"] if quick else training["policy_seeds"]
    )
    updates = int(
        training["quick_updates_per_seed_grid"]
        if quick
        else training["updates_per_seed_grid"]
    )
    batch_size = int(
        training["quick_batch_size"] if quick else training["batch_size"]
    )
    variants = tuple(config["extreme_variants"])
    probabilities = np.asarray(
        [training["variant_probabilities"][variant] for variant in variants],
        dtype=float,
    )
    probabilities /= probabilities.sum()
    service_range = float(config["city_complexity"]["service_range_m"])
    actors: dict[tuple[int, int], object] = {}
    update_rows: list[dict] = []
    seed_rows: list[dict] = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for side in config["grid_levels"]:
        side = int(side)
        for policy_replication in range(1, policy_seeds + 1):
            policy_seed = _seed(base_seed, 101, side, policy_replication)
            rng = np.random.default_rng(policy_seed)
            torch.manual_seed(policy_seed)
            actor = _policy(side, init_seed=policy_seed)
            actor.train()
            optimizer = torch.optim.AdamW(
                actor.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            seed_start = time.perf_counter()
            for update in range(1, updates + 1):
                update_start = time.perf_counter_ns()
                sampled_variants = rng.choice(
                    np.asarray(variants), size=batch_size, p=probabilities
                )
                sampled_populations = rng.choice(
                    np.asarray(config["population_levels"], dtype=int), size=batch_size
                )
                sampled_city_indices = rng.integers(0, len(cities), size=batch_size)
                cases = [
                    extreme_case(
                        rng,
                        str(sampled_variants[index]),
                        side=side,
                        population=int(sampled_populations[index]),
                        city=cities[int(sampled_city_indices[index])],
                        service_range_m=service_range,
                    )
                    for index in range(batch_size)
                ]
                graph, mask, utility = case_batch(cases, side)
                construction_ms = (
                    time.perf_counter_ns() - update_start
                ) / 1_000_000.0

                optimization_start = time.perf_counter_ns()
                logits, values = actor(graph)
                masked_logits = logits.masked_fill(~mask, -1e9)
                distribution = torch.distributions.Categorical(logits=masked_logits)
                actions = distribution.sample()
                rewards = utility.gather(1, actions.unsqueeze(1)).squeeze(1)
                advantages = rewards - values.detach()
                policy_loss = -(distribution.log_prob(actions) * advantages).mean()
                value_loss = 0.5 * torch.mean((values - rewards) ** 2)
                entropy = distribution.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Sensitivity policy loss became non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                optimizer.step()
                optimization_ms = (
                    time.perf_counter_ns() - optimization_start
                ) / 1_000_000.0
                with torch.inference_mode():
                    deterministic = masked_logits.argmax(dim=1)
                    deterministic_reward = utility.gather(
                        1, deterministic.unsqueeze(1)
                    ).squeeze(1)
                update_rows.append(
                    {
                        "grid_side": side,
                        "cell_count": side * side,
                        "policy_replication": policy_replication,
                        "policy_seed": policy_seed,
                        "update": update,
                        "batch_size": batch_size,
                        "case_and_graph_construction_ms": construction_ms,
                        "optimizer_update_ms": optimization_ms,
                        "total_update_ms": construction_ms + optimization_ms,
                        "sampled_utility": float(rewards.mean().item()),
                        "deterministic_utility": float(
                            deterministic_reward.mean().item()
                        ),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.item()),
                        "gradient_norm": float(gradient_norm.item()),
                    }
                )
            training_wall_s = float(time.perf_counter() - seed_start)
            actor.eval()
            actors[(side, policy_replication)] = actor
            checkpoint_path = (
                checkpoint_dir
                / f"grid_{side:02d}_policy_{policy_replication:03d}.pt"
            )
            torch.save(
                {
                    "schema_version": 1,
                    "experiment": config["suite_id"],
                    "grid_side": side,
                    "policy_replication": policy_replication,
                    "policy_seed": policy_seed,
                    "updates": updates,
                    "state_dict": actor.state_dict(),
                },
                checkpoint_path,
            )
            subset = [
                row
                for row in update_rows
                if int(row["grid_side"]) == side
                and int(row["policy_replication"]) == policy_replication
            ]
            seed_rows.append(
                {
                    "grid_side": side,
                    "cell_count": side * side,
                    "policy_replication": policy_replication,
                    "policy_seed": policy_seed,
                    "updates": updates,
                    "batch_size": batch_size,
                    "training_wall_s": training_wall_s,
                    "mean_total_update_ms": float(
                        np.mean([row["total_update_ms"] for row in subset])
                    ),
                    "mean_case_and_graph_construction_ms": float(
                        np.mean(
                            [row["case_and_graph_construction_ms"] for row in subset]
                        )
                    ),
                    "mean_optimizer_update_ms": float(
                        np.mean([row["optimizer_update_ms"] for row in subset])
                    ),
                    "final_20_update_deterministic_utility": float(
                        np.mean(
                            [
                                row["deterministic_utility"]
                                for row in subset[-min(20, len(subset)) :]
                            ]
                        )
                    ),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
    _write_csv(output_dir / "training_updates.csv", update_rows)
    _write_csv(output_dir / "training_time_by_policy.csv", seed_rows)
    return actors, update_rows, seed_rows


def evaluate_policies(
    config: Mapping,
    cities: Sequence[CitySpec],
    conditions: Sequence[Mapping],
    actors: Mapping,
    output_dir: Path,
    *,
    quick: bool,
    base_seed: int,
):
    training = config["policy_training"]
    evaluation = config["evaluation"]
    policy_seeds = int(
        training["quick_policy_seeds"] if quick else training["policy_seeds"]
    )
    scenarios = int(
        evaluation["quick_held_out_scenarios_per_variant_condition"]
        if quick
        else evaluation["held_out_scenarios_per_variant_condition"]
    )
    service_range = float(config["city_complexity"]["service_range_m"])
    variants = tuple(config["extreme_variants"])
    rows: list[dict] = []

    for side in config["grid_levels"]:
        side = int(side)
        warm_case = extreme_case(
            np.random.default_rng(_seed(base_seed, 300, side)),
            "safe_center",
            side=side,
            population=int(config["base_condition"]["population"]),
            city=next(
                city
                for city in cities
                if city.city_id == config["base_condition"]["city_id"]
            ),
            service_range_m=service_range,
        )
        warm_graph, warm_mask, _ = case_batch([warm_case], side)
        for policy_replication in range(1, policy_seeds + 1):
            actor = actors[(side, policy_replication)]
            for _ in range(int(evaluation["inference_warmups_per_grid"])):
                with torch.inference_mode():
                    actor(warm_graph)[0].masked_fill(~warm_mask, -torch.inf).argmax(dim=1)

    for condition in conditions:
        side = int(condition["grid_side"])
        city = condition["city"]
        for variant_index, variant in enumerate(variants):
            for scenario in range(1, scenarios + 1):
                scenario_seed = _seed(
                    base_seed,
                    401,
                    int(condition["condition_id"]),
                    variant_index,
                    scenario,
                )
                case_start = time.perf_counter_ns()
                case = extreme_case(
                    np.random.default_rng(scenario_seed),
                    variant,
                    side=side,
                    population=int(condition["population"]),
                    city=city,
                    service_range_m=service_range,
                )
                case_generation_ms = (
                    time.perf_counter_ns() - case_start
                ) / 1_000_000.0
                graph_start = time.perf_counter_ns()
                graph, mask, _ = case_batch([case], side)
                graph_construction_ms = (
                    time.perf_counter_ns() - graph_start
                ) / 1_000_000.0
                utility_values = np.asarray(case["utility"], dtype=float)
                oracle_utility = float(utility_values[int(case["oracle"])])

                for policy_replication in range(1, policy_seeds + 1):
                    actor = actors[(side, policy_replication)]
                    selection_start = time.perf_counter_ns()
                    with torch.inference_mode():
                        logits, _ = actor(graph)
                        rl_action = int(
                            logits.masked_fill(~mask, -torch.inf).argmax(dim=1).item()
                        )
                    rl_selection_ms = (
                        time.perf_counter_ns() - selection_start
                    ) / 1_000_000.0

                    strategy_actions = [("rl", rl_action, rl_selection_ms)]
                    for strategy in (
                        "population_heuristic",
                        "static_expected_demand",
                        "oracle",
                    ):
                        selection_start = time.perf_counter_ns()
                        if strategy == "population_heuristic":
                            feasible = np.flatnonzero(case["mask"])
                            action = int(
                                feasible[
                                    np.argmax(np.asarray(case["active"])[feasible])
                                ]
                            )
                        elif strategy == "static_expected_demand":
                            action = int(case["static"])
                        else:
                            action = int(np.argmax(utility_values))
                        selection_ms = (
                            time.perf_counter_ns() - selection_start
                        ) / 1_000_000.0
                        strategy_actions.append((strategy, action, selection_ms))

                    for strategy, action, selection_ms in strategy_actions:
                        representation_ms = (
                            graph_construction_ms if strategy == "rl" else 0.0
                        )
                        rows.append(
                            {
                                "condition_id": int(condition["condition_id"]),
                                "sensitivity_axes": ";".join(
                                    item["axis"] for item in condition["axes"]
                                ),
                                "population": int(condition["population"]),
                                "city_id": city.city_id,
                                "city_scale_rank": int(city.scale_rank),
                                "map_radius_m": float(city.radius_m),
                                "grid_side": side,
                                "cell_count": side * side,
                                "variant": variant,
                                "scenario": scenario,
                                "scenario_seed": scenario_seed,
                                "policy_replication": policy_replication,
                                "strategy": strategy,
                                "selected_cell": action,
                                "center_selected": int(action == int(case["center"])),
                                "oracle_selected": int(action == int(case["oracle"])),
                                "decision_utility": float(utility_values[action]),
                                "oracle_regret": float(
                                    oracle_utility - utility_values[action]
                                ),
                                "case_generation_ms": case_generation_ms,
                                "representation_construction_ms": representation_ms,
                                "selection_ms": selection_ms,
                                "policy_evaluation_ms": representation_ms + selection_ms,
                                "total_decision_pipeline_ms": (
                                    case_generation_ms + representation_ms + selection_ms
                                ),
                                "static_offline_planning_ms": float(
                                    case["static_planning_ms"]
                                ),
                            }
                        )
    _write_csv(output_dir / "extreme_sensitivity_evaluation.csv", rows)
    return rows


def _expanded_rows(rows: Sequence[Mapping], conditions: Sequence[Mapping]):
    axes = {
        int(condition["condition_id"]): tuple(condition["axes"])
        for condition in conditions
    }
    for row in rows:
        for axis in axes[int(row["condition_id"])]:
            yield {
                **row,
                "sensitivity_axis": axis["axis"],
                "level_order": int(axis["order"]),
                "level_label": axis["label"],
            }


def summarize_outcomes(rows: Sequence[Mapping], conditions: Sequence[Mapping]) -> list[dict]:
    groups: dict[tuple, list[Mapping]] = {}
    for row in _expanded_rows(rows, conditions):
        key = (
            row["sensitivity_axis"],
            int(row["level_order"]),
            row["level_label"],
            row["variant"],
            row["strategy"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, subset in sorted(groups.items()):
        utility = np.asarray([float(row["decision_utility"]) for row in subset])
        regret = np.asarray([float(row["oracle_regret"]) for row in subset])
        output.append(
            {
                "sensitivity_axis": key[0],
                "level_order": key[1],
                "level_label": key[2],
                "variant": key[3],
                "strategy": key[4],
                "observations": len(subset),
                "mean_decision_utility": float(np.mean(utility)),
                "mean_oracle_regret": float(np.mean(regret)),
                "median_oracle_regret": float(np.median(regret)),
                "oracle_selection_rate": float(
                    np.mean([float(row["oracle_selected"]) for row in subset])
                ),
                "center_selection_rate": float(
                    np.mean([float(row["center_selected"]) for row in subset])
                ),
            }
        )
    return output


_SIGN_MATRIX_CACHE: dict[int, np.ndarray] = {}


def _exact_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    count = int(values.size)
    signs = _SIGN_MATRIX_CACHE.get(count)
    if signs is None:
        assignments = np.arange(1 << count, dtype=np.uint32)[:, None]
        bits = (assignments >> np.arange(count, dtype=np.uint32)) & 1
        signs = (2.0 * bits.astype(np.float64)) - 1.0
        _SIGN_MATRIX_CACHE[count] = signs
    randomized = np.abs((signs @ values) / float(count))
    return float(np.mean(randomized >= observed - 1e-15))


def paired_comparisons(rows: Sequence[Mapping], conditions: Sequence[Mapping]) -> list[dict]:
    expanded = list(_expanded_rows(rows, conditions))
    groups: dict[tuple, list[Mapping]] = {}
    for row in expanded:
        key = (
            row["sensitivity_axis"],
            int(row["level_order"]),
            row["level_label"],
            row["variant"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, subset in sorted(groups.items()):
        by_key = {
            (
                int(row["policy_replication"]),
                int(row["scenario"]),
                str(row["strategy"]),
            ): row
            for row in subset
        }
        policies = sorted({int(row["policy_replication"]) for row in subset})
        scenarios = sorted({int(row["scenario"]) for row in subset})
        for baseline in ("population_heuristic", "static_expected_demand"):
            differences = []
            cluster_means = []
            wins = []
            for policy in policies:
                cluster = []
                for scenario in scenarios:
                    rl = by_key[(policy, scenario, "rl")]
                    comparison = by_key[(policy, scenario, baseline)]
                    difference = float(rl["decision_utility"]) - float(
                        comparison["decision_utility"]
                    )
                    differences.append(difference)
                    cluster.append(difference)
                    wins.append(float(difference > 0.0))
                cluster_means.append(float(np.mean(cluster)))
            clusters = np.asarray(cluster_means, dtype=float)
            standard_error = (
                float(np.std(clusters, ddof=1) / math.sqrt(clusters.size))
                if clusters.size > 1
                else 0.0
            )
            output.append(
                {
                    "sensitivity_axis": key[0],
                    "level_order": key[1],
                    "level_label": key[2],
                    "variant": key[3],
                    "baseline": baseline,
                    "policy_seed_clusters": len(policies),
                    "scenarios_per_policy_seed": len(scenarios),
                    "paired_observations": len(differences),
                    "mean_rl_utility_improvement": float(np.mean(differences)),
                    "cluster_normal_95_ci_low": float(
                        np.mean(clusters) - 1.96 * standard_error
                    ),
                    "cluster_normal_95_ci_high": float(
                        np.mean(clusters) + 1.96 * standard_error
                    ),
                    "rl_utility_win_rate": float(np.mean(wins)),
                    "exact_sign_randomization_p": _exact_sign_p(clusters),
                }
            )

    ordered = sorted(range(len(output)), key=lambda index: output[index]["exact_sign_randomization_p"])
    running = 0.0
    total = len(output)
    for rank, index in enumerate(ordered, start=1):
        adjusted = min(
            1.0,
            (total - rank + 1) * float(output[index]["exact_sign_randomization_p"]),
        )
        running = max(running, adjusted)
        output[index]["holm_adjusted_p"] = float(running)
    return output


def summarize_evaluation_timing(
    rows: Sequence[Mapping], conditions: Sequence[Mapping]
) -> list[dict]:
    groups: dict[tuple, list[Mapping]] = {}
    for row in _expanded_rows(rows, conditions):
        key = (
            row["sensitivity_axis"],
            int(row["level_order"]),
            row["level_label"],
            row["strategy"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, subset in sorted(groups.items()):
        evaluation = np.asarray(
            [float(row["policy_evaluation_ms"]) for row in subset], dtype=float
        )
        pipeline = np.asarray(
            [float(row["total_decision_pipeline_ms"]) for row in subset], dtype=float
        )
        selection = np.asarray(
            [float(row["selection_ms"]) for row in subset], dtype=float
        )
        representation = np.asarray(
            [float(row["representation_construction_ms"]) for row in subset],
            dtype=float,
        )
        static_planning = np.asarray(
            [float(row["static_offline_planning_ms"]) for row in subset], dtype=float
        )
        output.append(
            {
                "sensitivity_axis": key[0],
                "level_order": key[1],
                "level_label": key[2],
                "strategy": key[3],
                "observations": len(subset),
                "population": int(subset[0]["population"]),
                "city_id": str(subset[0]["city_id"]),
                "city_scale_rank": int(subset[0]["city_scale_rank"]),
                "grid_side": int(subset[0]["grid_side"]),
                "cell_count": int(subset[0]["cell_count"]),
                "median_policy_evaluation_ms": float(np.median(evaluation)),
                "p95_policy_evaluation_ms": float(np.quantile(evaluation, 0.95)),
                "median_total_decision_pipeline_ms": float(np.median(pipeline)),
                "p95_total_decision_pipeline_ms": float(np.quantile(pipeline, 0.95)),
                "median_selection_ms": float(np.median(selection)),
                "p95_selection_ms": float(np.quantile(selection, 0.95)),
                "median_representation_construction_ms": float(
                    np.median(representation)
                ),
                "median_static_offline_planning_ms": (
                    float(np.median(static_planning))
                    if key[3] == "static_expected_demand"
                    else 0.0
                ),
            }
        )
    return output


def _local_metre_coordinates(graph) -> tuple[np.ndarray, np.ndarray]:
    node_ids = list(graph.nodes)
    longitude = np.asarray(
        [float(graph.nodes[node]["x"]) for node in node_ids], dtype=float
    )
    latitude = np.asarray(
        [float(graph.nodes[node]["y"]) for node in node_ids], dtype=float
    )
    x_unit = 111_320.0 * math.cos(math.radians(float(latitude[0])))
    x = (longitude - float(np.min(longitude))) * x_unit
    y = (float(np.max(latitude)) - latitude) * 111_132.0
    return x, y


def benchmark_state_aggregation(
    config: Mapping,
    cities: Sequence[CitySpec],
    output_dir: Path,
    *,
    quick: bool,
    base_seed: int,
):
    repetitions = int(
        config["state_aggregation_benchmark"][
            "quick_repetitions" if quick else "repetitions"
        ]
    )
    modes = tuple(config["state_aggregation_benchmark"]["partition_modes"])
    graph_rows = []
    partition_rows = []
    aggregation_rows = []
    for city_index, city in enumerate(cities):
        load_start = time.perf_counter()
        graph = nx.read_graphml(city.graph_path)
        load_seconds = float(time.perf_counter() - load_start)
        x, y = _local_metre_coordinates(graph)
        graph_rows.append(
            {
                "city_id": city.city_id,
                "display_name": city.display_name,
                "city_scale_rank": city.scale_rank,
                "map_radius_m": city.radius_m,
                "graph_nodes": int(graph.number_of_nodes()),
                "graph_edges": int(graph.number_of_edges()),
                "graph_file_bytes": int(city.graph_path.stat().st_size),
                "graph_load_s": load_seconds,
                "graph_cache_path": str(city.graph_path),
                "graph_cache_sha256": _sha256(city.graph_path),
            }
        )
        for side in config["grid_levels"]:
            side = int(side)
            for mode in modes:
                partition_start = time.perf_counter_ns()
                partition = build_cell_partition(x, y, side, side, mode)
                partition_ms = (
                    time.perf_counter_ns() - partition_start
                ) / 1_000_000.0
                partition_rows.append(
                    {
                        "city_id": city.city_id,
                        "city_scale_rank": city.scale_rank,
                        "map_radius_m": city.radius_m,
                        "graph_nodes": int(graph.number_of_nodes()),
                        "graph_edges": int(graph.number_of_edges()),
                        "grid_side": side,
                        "cell_count": side * side,
                        "partition_mode": mode,
                        "partition_build_ms": partition_ms,
                    }
                )
                x_edges = np.asarray(partition["x_edges"], dtype=float)
                y_edges = np.asarray(partition["y_edges"], dtype=float)
                x_cell = np.clip(
                    np.searchsorted(x_edges, x, side="right") - 1, 0, side - 1
                )
                y_cell = np.clip(
                    np.searchsorted(y_edges, y, side="right") - 1, 0, side - 1
                )
                node_cell = x_cell * side + y_cell
                for population in config["population_levels"]:
                    population = int(population)
                    for repetition in range(1, repetitions + 1):
                        # Deliberately omit partition mode from the seed so the
                        # sampled road nodes are paired across both partitions.
                        rng = np.random.default_rng(
                            _seed(
                                base_seed,
                                501,
                                city_index,
                                side,
                                population,
                                repetition,
                            )
                        )
                        aggregation_start = time.perf_counter_ns()
                        node_indices = rng.integers(
                            0, node_cell.size, size=population
                        )
                        counts = np.bincount(
                            node_cell[node_indices], minlength=side * side
                        )
                        aggregation_ms = (
                            time.perf_counter_ns() - aggregation_start
                        ) / 1_000_000.0
                        if int(np.sum(counts)) != population:
                            raise RuntimeError("State aggregation lost pedestrians")
                        aggregation_rows.append(
                            {
                                "city_id": city.city_id,
                                "city_scale_rank": city.scale_rank,
                                "map_radius_m": city.radius_m,
                                "graph_nodes": int(graph.number_of_nodes()),
                                "graph_edges": int(graph.number_of_edges()),
                                "population": population,
                                "grid_side": side,
                                "cell_count": side * side,
                                "partition_mode": mode,
                                "repetition": repetition,
                                "state_aggregation_ms": aggregation_ms,
                                "scope": (
                                    "node-sampled position assignment plus bincount"
                                ),
                            }
                        )
        del graph
    _write_csv(output_dir / "city_graph_complexity.csv", graph_rows)
    _write_csv(output_dir / "partition_construction_time.csv", partition_rows)
    _write_csv(output_dir / "state_aggregation_time.csv", aggregation_rows)
    return graph_rows, partition_rows, aggregation_rows


def observational_simulator_timing(config: Mapping, output_dir: Path):
    source = PROJECT_ROOT / config["observational_simulator_timing"]["source_launch"]
    rows = []
    if source.exists():
        for summary_path in sorted(source.rglob("episode_summary.json")):
            metadata_path = summary_path.with_name("run_metadata.json")
            if not metadata_path.exists():
                continue
            summary = _read_json(summary_path)
            metadata = _read_json(metadata_path)
            elapsed = float(summary_path.stat().st_mtime - metadata_path.stat().st_mtime)
            if elapsed <= 0.0:
                continue
            rows.append(
                {
                    "source_launch": str(source),
                    "source_status": "incomplete_legacy_model_v8",
                    "city_id": summary.get("city_id"),
                    "initial_population": int(summary.get("initial_population", 0)),
                    "grid_side": int(metadata.get("grid_shape", [0, 0])[0]),
                    "cell_count": int(np.prod(metadata.get("grid_shape", [0, 0]))),
                    "replication": int(summary_path.parent.name.split("_")[1]),
                    "optimizer_updated": float(summary.get("optimizer_updated", 0.0)),
                    "episode_elapsed_s_reconstructed": elapsed,
                    "timing_definition": (
                        "episode_summary mtime minus run_metadata mtime"
                    ),
                }
            )
    if not rows:
        rows = [
            {
                "source_launch": str(source),
                "source_status": "unavailable",
                "city_id": "unavailable",
                "initial_population": 0,
                "grid_side": 0,
                "cell_count": 0,
                "replication": 0,
                "optimizer_updated": 0.0,
                "episode_elapsed_s_reconstructed": 0.0,
                "timing_definition": "no usable source episode timestamps",
            }
        ]
    _write_csv(output_dir / "observed_simulator_training_episode_time.csv", rows)
    return rows


def _group_training_summary(seed_rows: Sequence[Mapping]) -> list[dict]:
    output = []
    for side in sorted({int(row["grid_side"]) for row in seed_rows}):
        subset = [row for row in seed_rows if int(row["grid_side"]) == side]
        times = np.asarray([float(row["training_wall_s"]) for row in subset])
        output.append(
            {
                "grid_side": side,
                "cell_count": side * side,
                "policy_seeds": len(subset),
                "updates": int(subset[0]["updates"]),
                "batch_size": int(subset[0]["batch_size"]),
                "median_training_wall_s": float(np.median(times)),
                "minimum_training_wall_s": float(np.min(times)),
                "maximum_training_wall_s": float(np.max(times)),
                "median_mean_total_update_ms": float(
                    np.median([float(row["mean_total_update_ms"]) for row in subset])
                ),
                "median_mean_case_and_graph_construction_ms": float(
                    np.median(
                        [
                            float(row["mean_case_and_graph_construction_ms"])
                            for row in subset
                        ]
                    )
                ),
                "median_mean_optimizer_update_ms": float(
                    np.median(
                        [float(row["mean_optimizer_update_ms"]) for row in subset]
                    )
                ),
            }
        )
    return output


def _group_state_summary(rows: Sequence[Mapping]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        key = (
            str(row["city_id"]),
            int(row["city_scale_rank"]),
            int(row["population"]),
            int(row["grid_side"]),
            int(row["cell_count"]),
            str(row["partition_mode"]),
        )
        groups.setdefault(key, []).append(float(row["state_aggregation_ms"]))
    return [
        {
            "city_id": key[0],
            "city_scale_rank": key[1],
            "population": key[2],
            "grid_side": key[3],
            "cell_count": key[4],
            "partition_mode": key[5],
            "repetitions": len(values),
            "median_state_aggregation_ms": float(np.median(values)),
            "p95_state_aggregation_ms": float(np.quantile(values, 0.95)),
        }
        for key, values in sorted(groups.items())
    ]


def _group_observed_simulator(rows: Sequence[Mapping]) -> list[dict]:
    valid = [row for row in rows if float(row["episode_elapsed_s_reconstructed"]) > 0.0]
    output = []
    for city in sorted({str(row["city_id"]) for row in valid}):
        subset = [row for row in valid if str(row["city_id"]) == city]
        values = np.asarray(
            [float(row["episode_elapsed_s_reconstructed"]) for row in subset]
        )
        output.append(
            {
                "city_id": city,
                "episodes": len(subset),
                "initial_population": int(subset[0]["initial_population"]),
                "grid_side": int(subset[0]["grid_side"]),
                "median_episode_elapsed_s_reconstructed": float(np.median(values)),
                "p95_episode_elapsed_s_reconstructed": float(np.quantile(values, 0.95)),
                "minimum_episode_elapsed_s_reconstructed": float(np.min(values)),
                "maximum_episode_elapsed_s_reconstructed": float(np.max(values)),
                "evidence_status": "incomplete_legacy_observational_timing",
            }
        )
    return output


def training_time_by_sensitivity_scenario(
    config: Mapping,
    conditions: Sequence[Mapping],
    training_rows: Sequence[Mapping],
    state_rows: Sequence[Mapping],
    evaluation_timing_rows: Sequence[Mapping],
) -> list[dict]:
    """Join the timing components without pretending they share one clock.

    Population and city are inputs to policies trained jointly at a fixed grid,
    so their rows repeat the applicable measured training distribution.  Grid
    levels use their separately trained actors.  Environment state aggregation
    is retained as a separate deploy/training input-construction component.
    """
    training_by_side = {
        int(row["grid_side"]): row for row in training_rows
    }
    state_by_key = {
        (
            str(row["city_id"]),
            int(row["population"]),
            int(row["grid_side"]),
            str(row["partition_mode"]),
        ): row
        for row in state_rows
    }
    timing_by_key = {
        (
            str(row["sensitivity_axis"]),
            int(row["level_order"]),
            str(row["strategy"]),
        ): row
        for row in evaluation_timing_rows
    }
    output = []
    for condition in conditions:
        training = training_by_side[int(condition["grid_side"])]
        for axis in condition["axes"]:
            static_timing = timing_by_key[
                (str(axis["axis"]), int(axis["order"]), "static_expected_demand")
            ]
            for mode in config["state_aggregation_benchmark"]["partition_modes"]:
                state = state_by_key[
                    (
                        str(condition["city_id"]),
                        int(condition["population"]),
                        int(condition["grid_side"]),
                        str(mode),
                    )
                ]
                shared = str(axis["axis"]) in ("population", "city_complexity")
                output.append(
                    {
                        "sensitivity_axis": str(axis["axis"]),
                        "level_order": int(axis["order"]),
                        "level_label": str(axis["label"]),
                        "population": int(condition["population"]),
                        "city_id": str(condition["city_id"]),
                        "grid_side": int(condition["grid_side"]),
                        "cell_count": int(condition["grid_side"]) ** 2,
                        "partition_mode": str(mode),
                        "rl_training_wall_s_median": float(
                            training["median_training_wall_s"]
                        ),
                        "rl_training_wall_s_minimum": float(
                            training["minimum_training_wall_s"]
                        ),
                        "rl_training_wall_s_maximum": float(
                            training["maximum_training_wall_s"]
                        ),
                        "rl_training_updates": int(
                            training["updates"]
                        ),
                        "rl_policy_seeds": int(
                            training["policy_seeds"]
                        ),
                        "heuristic_training_wall_s": 0.0,
                        "static_training_wall_s": 0.0,
                        "static_offline_plan_ms_median": float(
                            static_timing["median_static_offline_planning_ms"]
                        ),
                        "state_aggregation_ms_median": float(
                            state["median_state_aggregation_ms"]
                        ),
                        "state_aggregation_ms_p95": float(
                            state["p95_state_aggregation_ms"]
                        ),
                        "training_scope": (
                            "shared cross-population/city actor at fixed grid"
                            if shared
                            else "separate actor for this grid resolution"
                        ),
                    }
                )
    return output


def _scaling_fit(
    component: str,
    x_name: str,
    x: Sequence[float],
    y: Sequence[float],
    scope: str,
) -> dict:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    valid = (x_values > 0.0) & (y_values > 0.0)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 3:
        raise ValueError(f"At least three positive observations required for {component}")
    log_x = np.log(x_values)
    log_y = np.log(y_values)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    log_prediction = intercept + exponent * log_x
    log_residual = log_y - log_prediction
    log_total = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    log_r2 = 1.0 - float(np.sum(log_residual**2)) / log_total if log_total > 0 else 1.0
    linear_slope, linear_intercept = np.polyfit(x_values, y_values, 1)
    linear_prediction = linear_intercept + linear_slope * x_values
    linear_total = float(np.sum((y_values - float(np.mean(y_values))) ** 2))
    linear_r2 = (
        1.0 - float(np.sum((y_values - linear_prediction) ** 2)) / linear_total
        if linear_total > 0
        else 1.0
    )
    degrees = int(x_values.size - 2)
    exponent_se = math.sqrt(
        float(np.sum(log_residual**2))
        / float(degrees)
        / float(np.sum((log_x - float(np.mean(log_x))) ** 2))
    )
    # All registered scaling fits use five levels (df=3); retain a safe normal
    # fallback should this utility be reused with a different level count.
    critical = 3.182 if degrees == 3 else 1.96
    return {
        "component": component,
        "predictor": x_name,
        "observations": int(x_values.size),
        "minimum_x": float(np.min(x_values)),
        "maximum_x": float(np.max(x_values)),
        "power_law_exponent": float(exponent),
        "power_law_exponent_95_ci_low": float(exponent - critical * exponent_se),
        "power_law_exponent_95_ci_high": float(exponent + critical * exponent_se),
        "log_log_r_squared": float(log_r2),
        "linear_r_squared": float(linear_r2),
        "descriptive_better_fit": "power_law" if log_r2 >= linear_r2 else "linear",
        "scope": scope,
    }


def computational_scaling_models(
    config: Mapping,
    training_rows: Sequence[Mapping],
    evaluation_timing_rows: Sequence[Mapping],
    state_rows: Sequence[Mapping],
    graph_rows: Sequence[Mapping],
    partition_rows: Sequence[Mapping],
) -> list[dict]:
    base = config["base_condition"]
    output = [
        _scaling_fit(
            "controlled_rl_training_wall",
            "cell_count",
            [float(row["cell_count"]) for row in training_rows],
            [float(row["median_training_wall_s"]) for row in training_rows],
            (
                f"{int(training_rows[0]['updates'])}-update actor training; "
                "excludes evacuation simulation"
            ),
        )
    ]
    rl_cell = sorted(
        (
            row
            for row in evaluation_timing_rows
            if row["sensitivity_axis"] == "cell_dimensionality"
            and row["strategy"] == "rl"
        ),
        key=lambda row: int(row["cell_count"]),
    )
    output.append(
        _scaling_fit(
            "rl_policy_evaluation",
            "cell_count",
            [float(row["cell_count"]) for row in rl_cell],
            [float(row["median_policy_evaluation_ms"]) for row in rl_cell],
            "state tensor/graph construction plus deterministic actor forward pass",
        )
    )
    for mode in config["state_aggregation_benchmark"]["partition_modes"]:
        population_rows = sorted(
            (
                row
                for row in state_rows
                if row["city_id"] == base["city_id"]
                and int(row["grid_side"]) == int(base["grid_side"])
                and row["partition_mode"] == mode
            ),
            key=lambda row: int(row["population"]),
        )
        output.append(
            _scaling_fit(
                f"state_aggregation_{mode}",
                "population",
                [float(row["population"]) for row in population_rows],
                [float(row["median_state_aggregation_ms"]) for row in population_rows],
                "Spokane at 9x9; road-node sampling, cell assignment, and bincount",
            )
        )
        cell_rows = sorted(
            (
                row
                for row in state_rows
                if row["city_id"] == base["city_id"]
                and int(row["population"]) == int(base["population"])
                and row["partition_mode"] == mode
            ),
            key=lambda row: int(row["cell_count"]),
        )
        output.append(
            _scaling_fit(
                f"state_aggregation_{mode}",
                "cell_count",
                [float(row["cell_count"]) for row in cell_rows],
                [float(row["median_state_aggregation_ms"]) for row in cell_rows],
                "Spokane at 30k; road-node sampling, cell assignment, and bincount",
            )
        )
        partition_city = sorted(
            (
                row
                for row in partition_rows
                if int(row["grid_side"]) == int(base["grid_side"])
                and row["partition_mode"] == mode
            ),
            key=lambda row: int(row["graph_nodes"]),
        )
        output.append(
            _scaling_fit(
                f"partition_construction_{mode}",
                "graph_nodes",
                [float(row["graph_nodes"]) for row in partition_city],
                [float(row["partition_build_ms"]) for row in partition_city],
                "five cached OSM graphs at 9x9",
            )
        )
    graph_sorted = sorted(graph_rows, key=lambda row: int(row["graph_nodes"]))
    output.append(
        _scaling_fit(
            "cached_graph_load",
            "graph_nodes",
            [float(row["graph_nodes"]) for row in graph_sorted],
            [float(row["graph_load_s"]) for row in graph_sorted],
            "five cached OSM GraphML files; one observed load each",
        )
    )
    return output


def _save_figure(fig, output_dir: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / "figures" / f"{stem}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    return paths


def generate_figures(output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
        }
    )
    paths = []
    comparisons = _read_csv(output_dir / "extreme_paired_comparisons.csv")
    axes_order = ("population", "city_complexity", "cell_dimensionality")
    baselines = ("population_heuristic", "static_expected_demand")
    variants = ("safe_center", "dangerous_center", "center_unavailable", "asymmetric_ring")
    values = np.asarray(
        [float(row["mean_rl_utility_improvement"]) for row in comparisons]
    )
    limit = max(0.01, float(np.max(np.abs(values))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, panels = plt.subplots(3, 2, figsize=(11.2, 10.0), constrained_layout=True)
    image = None
    for row_index, axis_name in enumerate(axes_order):
        for column_index, baseline in enumerate(baselines):
            ax = panels[row_index, column_index]
            subset = [
                row
                for row in comparisons
                if row["sensitivity_axis"] == axis_name and row["baseline"] == baseline
            ]
            levels = sorted(
                {(int(row["level_order"]), row["level_label"]) for row in subset}
            )
            matrix = np.full((len(variants), len(levels)), np.nan)
            adjusted_p = np.full((len(variants), len(levels)), np.nan)
            for row in subset:
                i = variants.index(row["variant"])
                j = [item[0] for item in levels].index(int(row["level_order"]))
                matrix[i, j] = float(row["mean_rl_utility_improvement"])
                adjusted_p[i, j] = float(row["holm_adjusted_p"])
            image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
            ax.set_xticks(range(len(levels)), [item[1] for item in levels], rotation=25, ha="right")
            ax.set_yticks(range(len(variants)), [item.replace("_", " ") for item in variants])
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    marker = "*" if adjusted_p[i, j] < 0.05 else ""
                    ax.text(
                        j,
                        i,
                        f"{value:+.3f}{marker}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
            baseline_label = "heuristic" if baseline == "population_heuristic" else "static"
            axis_label = axis_name.replace("_", " ")
            ax.set_title(
                f"({chr(97 + row_index * 2 + column_index)}) {axis_label}: RL − {baseline_label}",
                loc="left",
                fontweight="bold",
            )
            ax.set_xlabel(axis_label)
            ax.set_ylabel("extreme-case variant")
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(panels.flat), shrink=0.72, pad=0.015)
        colorbar.set_label("Mean paired decision-utility difference")
    fig.suptitle(
        "Extreme-case policy sensitivity across demand, city scale, and grid resolution",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(0.5, 0.002, "* Holm-adjusted p < 0.05 across 120 registered contrasts", ha="center", fontsize=7)
    paths.extend(_save_figure(fig, output_dir, "F13_extreme_policy_sensitivity"))
    plt.close(fig)

    training = _read_csv(output_dir / "training_time_summary.csv")
    state = _read_csv(output_dir / "state_aggregation_time_summary.csv")
    graph = _read_csv(output_dir / "city_graph_complexity.csv")
    observed = _read_csv(output_dir / "observed_simulator_training_time_summary.csv")
    fig, panels = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    cells = np.asarray([int(row["cell_count"]) for row in training])
    medians = np.asarray([float(row["median_training_wall_s"]) for row in training])
    low = np.asarray([float(row["minimum_training_wall_s"]) for row in training])
    high = np.asarray([float(row["maximum_training_wall_s"]) for row in training])
    panels[0, 0].errorbar(cells, medians, yerr=(medians - low, high - medians), marker="o", capsize=3)
    panels[0, 0].set_title("(a) Controlled RL training by grid", loc="left", fontweight="bold")
    panels[0, 0].set_xlabel("cell count, n²")
    panels[0, 0].set_ylabel("wall time per policy seed (s)")
    panels[0, 0].set_xscale("log", base=2)
    panels[0, 0].set_xticks(cells, [str(value) for value in cells])
    panels[0, 0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    panels[0, 1].plot(
        cells,
        [float(row["median_mean_case_and_graph_construction_ms"]) for row in training],
        marker="o",
        label="case + graph construction",
    )
    panels[0, 1].plot(
        cells,
        [float(row["median_mean_optimizer_update_ms"]) for row in training],
        marker="s",
        label="actor/value update",
    )
    panels[0, 1].set_title("(b) Training-update decomposition", loc="left", fontweight="bold")
    panels[0, 1].set_xlabel("cell count, n²")
    panels[0, 1].set_ylabel("mean time per update (ms)")
    panels[0, 1].set_xscale("log", base=2)
    panels[0, 1].set_xticks(cells, [str(value) for value in cells])
    panels[0, 1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    panels[0, 1].legend(frameon=False)

    base_grid = 9
    for city_id in sorted({row["city_id"] for row in state}):
        subset = sorted(
            (
                row
                for row in state
                if row["city_id"] == city_id and int(row["grid_side"]) == base_grid
                and row["partition_mode"] == "node_density_adaptive"
            ),
            key=lambda row: int(row["population"]),
        )
        panels[1, 0].plot(
            [int(row["population"]) for row in subset],
            [float(row["median_state_aggregation_ms"]) for row in subset],
            marker="o",
            label=city_id.replace("_", " "),
        )
    panels[1, 0].set_title("(c) Population-state aggregation (adaptive)", loc="left", fontweight="bold")
    panels[1, 0].set_xlabel("pedestrian population")
    panels[1, 0].set_ylabel("median aggregation time (ms)")
    panels[1, 0].legend(frameon=False, ncol=2, fontsize=6.8)

    city_order = sorted(graph, key=lambda row: int(row["city_scale_rank"]))
    observed_by_city = {row["city_id"]: row for row in observed}
    x_positions = np.arange(len(city_order))
    observed_medians = [
        float(observed_by_city[row["city_id"]]["median_episode_elapsed_s_reconstructed"])
        if row["city_id"] in observed_by_city
        else np.nan
        for row in city_order
    ]
    panels[1, 1].bar(x_positions, observed_medians, color="#7A7A7A")
    panels[1, 1].set_xticks(
        x_positions,
        [row["display_name"].split(",")[0] for row in city_order],
        rotation=25,
        ha="right",
    )
    panels[1, 1].set_title(
        "(d) Observed 50k simulator training episodes*",
        loc="left",
        fontweight="bold",
    )
    panels[1, 1].set_ylabel("reconstructed episode wall time (s)")
    panels[1, 1].text(
        0.01,
        0.98,
        "*21 completed episodes; incomplete legacy v8 launch",
        transform=panels[1, 1].transAxes,
        va="top",
        fontsize=7,
    )
    for ax in panels.flat:
        ax.grid(True, color="#D9D9D9", linewidth=0.6, axis="y")
    fig.suptitle(
        "Training and state-construction time: measured components and evidence scope",
        fontsize=12,
        fontweight="bold",
    )
    paths.extend(_save_figure(fig, output_dir, "F14_training_and_state_scaling"))
    plt.close(fig)

    timing = _read_csv(output_dir / "policy_evaluation_time_summary.csv")
    fig, panels = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    strategy_styles = {
        "rl": ("o", "RL"),
        "population_heuristic": ("s", "Heuristic"),
        "static_expected_demand": ("^", "Static"),
        "oracle": ("x", "Oracle reference"),
    }
    for ax, axis_name, title in zip(
        panels,
        ("population", "city_complexity", "cell_dimensionality"),
        ("population", "city complexity", "cell dimensionality"),
    ):
        for strategy, (marker, label) in strategy_styles.items():
            subset = sorted(
                (
                    row
                    for row in timing
                    if row["sensitivity_axis"] == axis_name
                    and row["strategy"] == strategy
                ),
                key=lambda row: int(row["level_order"]),
            )
            ax.plot(
                [int(row["level_order"]) for row in subset],
                [float(row["median_policy_evaluation_ms"]) for row in subset],
                marker=marker,
                label=label,
            )
        levels = sorted(
            {
                (int(row["level_order"]), row["level_label"])
                for row in timing
                if row["sensitivity_axis"] == axis_name
            }
        )
        ax.set_xticks([item[0] for item in levels], [item[1] for item in levels], rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_title(f"{title}", loc="left", fontweight="bold")
        ax.set_xlabel(title)
        ax.set_ylabel("median policy evaluation time (ms)")
        ax.grid(True, which="both", color="#D9D9D9", linewidth=0.6)
    panels[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Online policy-evaluation time for every extreme sensitivity scenario",
        fontsize=12,
        fontweight="bold",
    )
    paths.extend(_save_figure(fig, output_dir, "F15_policy_evaluation_latency"))
    plt.close(fig)

    partition = _read_csv(output_dir / "partition_construction_time.csv")
    base_city = "spokane_wa"
    base_population = 30000
    base_grid = 9
    mode_styles = {
        "equal_area": ("o", "Equal area"),
        "node_density_adaptive": ("s", "Node-density adaptive"),
    }
    fig, panels = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for mode, (marker, label) in mode_styles.items():
        subset = sorted(
            (
                row
                for row in state
                if row["city_id"] == base_city
                and int(row["grid_side"]) == base_grid
                and row["partition_mode"] == mode
            ),
            key=lambda row: int(row["population"]),
        )
        panels[0].plot(
            [int(row["population"]) for row in subset],
            [float(row["median_state_aggregation_ms"]) for row in subset],
            marker=marker,
            label=label,
        )
    panels[0].set_title("(a) Population", loc="left", fontweight="bold")
    panels[0].set_xlabel("pedestrian population")
    panels[0].set_ylabel("median state aggregation (ms)")

    graph_by_city = {row["city_id"]: row for row in graph}
    city_order = sorted(graph, key=lambda row: int(row["city_scale_rank"]))
    for mode, (marker, label) in mode_styles.items():
        subset = {
            row["city_id"]: row
            for row in partition
            if int(row["grid_side"]) == base_grid
            and row["partition_mode"] == mode
        }
        panels[1].plot(
            range(1, len(city_order) + 1),
            [float(subset[row["city_id"]]["partition_build_ms"]) for row in city_order],
            marker=marker,
            label=label,
        )
    panels[1].set_xticks(
        range(1, len(city_order) + 1),
        [row["display_name"].split(",")[0] for row in city_order],
        rotation=25,
        ha="right",
    )
    panels[1].set_title("(b) City graph complexity", loc="left", fontweight="bold")
    panels[1].set_xlabel("city (increasing registered scale)")
    panels[1].set_ylabel("9×9 partition construction (ms)")

    for mode, (marker, label) in mode_styles.items():
        subset = sorted(
            (
                row
                for row in state
                if row["city_id"] == base_city
                and int(row["population"]) == base_population
                and row["partition_mode"] == mode
            ),
            key=lambda row: int(row["cell_count"]),
        )
        panels[2].plot(
            [int(row["cell_count"]) for row in subset],
            [float(row["median_state_aggregation_ms"]) for row in subset],
            marker=marker,
            label=label,
        )
    panels[2].set_title("(c) Cell dimensionality", loc="left", fontweight="bold")
    panels[2].set_xlabel("cell count, n²")
    panels[2].set_ylabel("median state aggregation (ms)")
    panels[2].set_xscale("log", base=2)
    cell_ticks = sorted(
        {
            int(row["cell_count"])
            for row in state
            if row["city_id"] == base_city
            and int(row["population"]) == base_population
        }
    )
    panels[2].set_xticks(cell_ticks, [str(value) for value in cell_ticks])
    panels[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for ax in panels:
        ax.grid(True, color="#D9D9D9", linewidth=0.6, axis="y")
    panels[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Preprocessing sensitivity under both registered cell-partition options",
        fontsize=12,
        fontweight="bold",
    )
    paths.extend(
        _save_figure(fig, output_dir, "F16_partition_mode_preprocessing_scaling")
    )
    plt.close(fig)
    return paths


def write_report(config: Mapping, output_dir: Path, manifest: Mapping) -> Path:
    comparisons = _read_csv(output_dir / "extreme_paired_comparisons.csv")
    training = _read_csv(output_dir / "training_time_summary.csv")
    timing = _read_csv(output_dir / "policy_evaluation_time_summary.csv")
    scaling = _read_csv(output_dir / "computational_scaling_models.csv")
    observed = _read_csv(output_dir / "observed_simulator_training_time_summary.csv")
    base = config["base_condition"]
    base_rows = [
        row
        for row in comparisons
        if row["sensitivity_axis"] == "population"
        and row["level_label"] == f"{int(base['population']) // 1000}k"
    ]
    baseline_names = {
        "population_heuristic": "heuristic",
        "static_expected_demand": "static",
    }
    base_lines = []
    for variant in config["extreme_variants"]:
        for baseline in ("population_heuristic", "static_expected_demand"):
            row = next(
                value
                for value in base_rows
                if value["variant"] == variant and value["baseline"] == baseline
            )
            base_lines.append(
                "| "
                + " | ".join(
                    (
                        variant.replace("_", " "),
                        baseline_names[baseline],
                        f"{float(row['mean_rl_utility_improvement']):+.4f}",
                        (
                            f"[{float(row['cluster_normal_95_ci_low']):+.4f}, "
                            f"{float(row['cluster_normal_95_ci_high']):+.4f}]"
                        ),
                        f"{float(row['rl_utility_win_rate']):.3f}",
                        f"{float(row['holm_adjusted_p']):.4g}",
                    )
                )
                + " |"
            )
    base_table = "\n".join(base_lines)
    significant_count = sum(float(row["holm_adjusted_p"]) < 0.05 for row in comparisons)
    smallest = min(training, key=lambda row: int(row["cell_count"]))
    largest = max(training, key=lambda row: int(row["cell_count"]))
    base_timing = {
        row["strategy"]: row
        for row in timing
        if row["sensitivity_axis"] == "population"
        and row["level_label"] == f"{int(base['population']) // 1000}k"
    }
    training_scaling = next(
        row for row in scaling if row["component"] == "controlled_rl_training_wall"
    )
    inference_scaling = next(
        row for row in scaling if row["component"] == "rl_policy_evaluation"
    )
    population_scaling = next(
        row
        for row in scaling
        if row["component"] == "state_aggregation_node_density_adaptive"
        and row["predictor"] == "population"
    )
    observed_count = sum(int(row["episodes"]) for row in observed)
    report = output_dir / "EXTREME_SENSITIVITY_RESULTS.md"
    content = f"""# Extreme-policy sensitivity and runtime results

## Material Passport

- Material type: Experiment Result
- Material ID: `{config['suite_id']}`
- Verification Status: ANALYZED
- Controlled policy seeds: {manifest['audit']['policy_seeds']}
- Held-out scenarios per variant-condition: {manifest['audit']['scenarios']}
- Registered paired contrasts: {len(comparisons)}; Holm-significant: {significant_count}
- Partition implementations timed: equal-area and node-density-adaptive
- Claim boundary: controlled decision mechanism and computational timing, not city evacuation efficacy

## Policy sensitivity

At the registered base condition ({base['population']:,} pedestrians,
`{base['city_id']}`, {base['grid_side']}×{base['grid_side']} cells):

| Extreme variant | Comparator | Mean RL difference | Policy-seed 95% CI | Win rate | Holm p |
|---|---:|---:|---:|---:|---:|
{base_table}

The complete table contains all {len(comparisons)} registered RL contrasts:
five population levels, five city-scale levels, five grid resolutions, four
extreme variants, and two comparators. Common scenario seeds are used within
every policy/comparator pair; inference is clustered at the independent policy
seed rather than treating scenario replications as independent learned models.

## Training time

Controlled actor training used one CPU thread and the same number of updates
for each grid. Median wall time per policy seed was
{float(smallest['median_training_wall_s']):.2f} s at {smallest['cell_count']}
cells and {float(largest['median_training_wall_s']):.2f} s at
{largest['cell_count']} cells. This includes controlled case/tensor construction
and actor/value optimization but excludes evacuation simulation. Across the
five registered cell counts, the descriptive log-log exponent was
{float(training_scaling['power_law_exponent']):.2f} (95% CI
{float(training_scaling['power_law_exponent_95_ci_low']):.2f} to
{float(training_scaling['power_law_exponent_95_ci_high']):.2f}). Heuristic and
static policies require no learned-parameter training; the static plan's
separate offline optimization time is reported in the scenario timing table.

One actor is jointly trained across all five populations and cities at each
grid resolution. Therefore increasing population or switching cities does not
trigger another policy-training run at fixed grid size. Their measured cost
appears in state construction: for the primary adaptive partition, the
population-scaling exponent was
{float(population_scaling['power_law_exponent']):.2f} over 10,000--50,000
pedestrians. The scenario timing table joins these separate components without
adding them into a fictitious end-to-end training clock.

## Online policy-evaluation time

At the 30,000-person Spokane 9×9 base condition, median online evaluation was
{float(base_timing['rl']['median_policy_evaluation_ms']):.3f} ms for RL,
{float(base_timing['population_heuristic']['median_policy_evaluation_ms']):.6f}
ms for the population heuristic, and
{float(base_timing['static_expected_demand']['median_policy_evaluation_ms']):.6f}
ms for static lookup. RL includes state-graph representation construction plus
the deterministic neural forward pass; the heuristic is recomputed from the
realized active-population vector; static time is online lookup and its offline
planning time is recorded separately. The descriptive RL inference exponent
over 25--169 cells was {float(inference_scaling['power_law_exponent']):.2f}.

The separate full-simulator timing table contains {observed_count} completed
50,000-person, 8×8 RL training episodes from an incomplete legacy v8 launch.
Those reconstructed episode intervals are descriptive only and are not mixed
with current-version controlled timings.

## Interpretation

Population and city scale enter the fixed-size regional policy as normalized
features, so neural evaluation time should remain approximately flat at fixed
grid size. Population affects the measured state-aggregation kernel, while
grid dimensionality affects graph construction and neural computation. Actual
end-to-end simulator time also includes routing, congestion, hazard evolution,
and movement and must be reported from completed current-version city runs.
The controlled decision utility is not a casualty count or evacuation-time
estimate; city-level safety claims must be made only from the separately
registered full simulator experiment.

## Design provenance

The 20-update development pilot was used only to validate the pipeline. Before
examining confirmatory held-out results, the design was amended to 500 updates
and 16 policy seeds. The calibration and pilot paths and the amendment reason
are frozen in the experiment configuration. Both cell-partition implementations
are timed here; their evacuation-quality comparison remains the separate
registered partition-mode experiment.
"""
    temporary = report.with_suffix(report.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(temporary, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(config_path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported extreme-sensitivity experiment schema")
    torch.set_num_threads(int(config["evaluation"]["inference_torch_threads"]))
    cities = load_city_specs(config, require_cached_graph=True)
    conditions = sensitivity_conditions(config, cities)
    started = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "status": "running",
        "quick": bool(args.quick),
        "seed": int(args.seed),
        "started_utc": started.isoformat(),
        "design_path": str(config_path),
        "design_sha256": _sha256(config_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "networkx": nx.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "git": {
            "commit": _git_value("git", "rev-parse", "HEAD"),
            "status": _git_value("git", "status", "--short"),
        },
        "artifacts": {},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    actors, update_rows, seed_rows = train_policies(
        config, cities, output_dir, quick=args.quick, base_seed=args.seed
    )
    evaluation_rows = evaluate_policies(
        config,
        cities,
        conditions,
        actors,
        output_dir,
        quick=args.quick,
        base_seed=args.seed,
    )
    outcome_summary = summarize_outcomes(evaluation_rows, conditions)
    comparisons = paired_comparisons(evaluation_rows, conditions)
    evaluation_timing = summarize_evaluation_timing(evaluation_rows, conditions)
    training_summary = _group_training_summary(seed_rows)
    graph_rows, partition_rows, aggregation_rows = benchmark_state_aggregation(
        config, cities, output_dir, quick=args.quick, base_seed=args.seed
    )
    state_summary = _group_state_summary(aggregation_rows)
    observed_rows = observational_simulator_timing(config, output_dir)
    observed_summary = _group_observed_simulator(observed_rows)
    training_scenarios = training_time_by_sensitivity_scenario(
        config,
        conditions,
        training_summary,
        state_summary,
        evaluation_timing,
    )
    scaling_models = computational_scaling_models(
        config,
        training_summary,
        evaluation_timing,
        state_summary,
        graph_rows,
        partition_rows,
    )
    _write_csv(output_dir / "extreme_outcome_summary.csv", outcome_summary)
    _write_csv(output_dir / "extreme_paired_comparisons.csv", comparisons)
    _write_csv(output_dir / "policy_evaluation_time_summary.csv", evaluation_timing)
    _write_csv(output_dir / "training_time_summary.csv", training_summary)
    _write_csv(
        output_dir / "training_time_by_sensitivity_scenario.csv",
        training_scenarios,
    )
    _write_csv(output_dir / "state_aggregation_time_summary.csv", state_summary)
    _write_csv(output_dir / "computational_scaling_models.csv", scaling_models)
    _write_csv(
        output_dir / "observed_simulator_training_time_summary.csv",
        observed_summary,
    )
    figure_paths = generate_figures(output_dir)

    training_design = config["policy_training"]
    evaluation_design = config["evaluation"]
    policy_seeds = int(
        training_design["quick_policy_seeds"]
        if args.quick
        else training_design["policy_seeds"]
    )
    updates = int(
        training_design["quick_updates_per_seed_grid"]
        if args.quick
        else training_design["updates_per_seed_grid"]
    )
    scenarios = int(
        evaluation_design["quick_held_out_scenarios_per_variant_condition"]
        if args.quick
        else evaluation_design["held_out_scenarios_per_variant_condition"]
    )
    repetitions = int(
        config["state_aggregation_benchmark"][
            "quick_repetitions" if args.quick else "repetitions"
        ]
    )
    expected = {
        "training_update_rows": len(config["grid_levels"]) * policy_seeds * updates,
        "training_policy_rows": len(config["grid_levels"]) * policy_seeds,
        "evaluation_rows": len(conditions)
        * len(config["extreme_variants"])
        * scenarios
        * policy_seeds
        * len(config["strategies"]),
        "paired_comparison_rows": 3
        * 5
        * len(config["extreme_variants"])
        * 2,
        "state_aggregation_rows": len(cities)
        * len(config["grid_levels"])
        * len(config["population_levels"])
        * repetitions
        * len(config["state_aggregation_benchmark"]["partition_modes"]),
        "partition_construction_rows": len(cities)
        * len(config["grid_levels"])
        * len(config["state_aggregation_benchmark"]["partition_modes"]),
        "training_sensitivity_rows": 3
        * 5
        * len(config["state_aggregation_benchmark"]["partition_modes"]),
        "computational_scaling_model_rows": 9,
    }
    observed = {
        "training_update_rows": len(update_rows),
        "training_policy_rows": len(seed_rows),
        "evaluation_rows": len(evaluation_rows),
        "paired_comparison_rows": len(comparisons),
        "state_aggregation_rows": len(aggregation_rows),
        "partition_construction_rows": len(partition_rows),
        "training_sensitivity_rows": len(training_scenarios),
        "computational_scaling_model_rows": len(scaling_models),
    }
    if observed != expected:
        raise RuntimeError(f"Sensitivity count mismatch: expected={expected}, observed={observed}")
    if not all(
        math.isfinite(float(row["total_update_ms"]))
        and float(row["total_update_ms"]) > 0.0
        for row in update_rows
    ):
        raise RuntimeError("Invalid training timing value")
    if not all(
        math.isfinite(float(row["policy_evaluation_ms"]))
        and float(row["policy_evaluation_ms"]) >= 0.0
        for row in evaluation_rows
    ):
        raise RuntimeError("Invalid evaluation timing value")

    manifest["audit"] = {
        "status": "passed",
        "expected_counts": expected,
        "observed_counts": observed,
        "policy_seeds": policy_seeds,
        "updates_per_seed_grid": updates,
        "scenarios": scenarios,
        "state_aggregation_repetitions": repetitions,
        "all_training_timings_finite_positive": True,
        "all_evaluation_timings_finite_nonnegative": True,
    }
    manifest["artifacts"] = {
        "training_updates": str(output_dir / "training_updates.csv"),
        "training_time_by_policy": str(output_dir / "training_time_by_policy.csv"),
        "training_time_by_sensitivity_scenario": str(
            output_dir / "training_time_by_sensitivity_scenario.csv"
        ),
        "evaluation_raw": str(output_dir / "extreme_sensitivity_evaluation.csv"),
        "outcome_summary": str(output_dir / "extreme_outcome_summary.csv"),
        "paired_comparisons": str(output_dir / "extreme_paired_comparisons.csv"),
        "evaluation_time_summary": str(output_dir / "policy_evaluation_time_summary.csv"),
        "state_aggregation_raw": str(output_dir / "state_aggregation_time.csv"),
        "state_aggregation_summary": str(output_dir / "state_aggregation_time_summary.csv"),
        "city_graph_complexity": str(output_dir / "city_graph_complexity.csv"),
        "partition_construction_time": str(output_dir / "partition_construction_time.csv"),
        "computational_scaling_models": str(
            output_dir / "computational_scaling_models.csv"
        ),
        "observational_simulator_timing": str(
            output_dir / "observed_simulator_training_episode_time.csv"
        ),
        "figures": figure_paths,
    }
    report_path = write_report(config, output_dir, manifest)
    manifest["artifacts"]["report"] = str(report_path)
    manifest["status"] = "pilot_complete" if args.quick else "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_time_s"] = float(time.perf_counter() - wall_start)
    manifest["artifact_sha256"] = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
