#!/usr/bin/env python3
"""Reproducible computational experiments for the OR journal study.

This driver deliberately separates two evidence tiers:

* ``latency`` benchmarks the production regional actor and the implemented
  active-population heuristic under controlled action-space growth.
* ``extreme`` runs the requested populated-ring/empty-center mechanism test as
  a one-decision contextual bandit using the production policy architecture.

Neither tier is mislabeled as a five-city evacuation outcome experiment.  The
city-level confirmatory design is frozen in the companion JSON protocol and is
executed by the existing backtest runners after the State College runtime gate.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from DecisionInterface import (
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    HAZARD_FEATURE_SLICE,
    INFRA_FEATURE_NAMES,
    INFRA_FEATURE_SLICE,
    PED_FEATURE_NAMES,
    PED_FEATURE_SLICE,
    ActivePopulationHeuristic,
    OutcomeSnapshot,
    RegionalObservation,
)
from GNN import EvacPolicy, fit_gnn, grid_edge_index


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "or_journal_experiment_suite.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "or_journal_mechanism_latency_v1"


def _strict(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
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


def _policy(side: int, *, init_seed: int | None = None) -> EvacPolicy:
    """Instantiate the exact architecture used by RLBridge."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(
            20260909 + int(side) if init_seed is None else int(init_seed)
        )
        result = EvacPolicy(
            d_ped=len(PED_FEATURE_NAMES),
            d_hazard=len(HAZARD_FEATURE_NAMES),
            d_infra=len(INFRA_FEATURE_NAMES),
            d_global=len(GLOBAL_FEATURE_NAMES),
            d_candidate=len(CANDIDATE_FEATURE_NAMES),
            verbose=False,
        )
    result.eval()
    return result


def _synthetic_observation(
    side: int,
    candidate_count: int,
    rng: np.random.Generator,
) -> RegionalObservation:
    cells = int(side) ** 2
    population = 50_000
    active = rng.multinomial(population, np.full(cells, 1.0 / cells)).astype(np.float32)
    speed = np.clip(rng.normal(45.0, 10.0, cells), 0.0, 75.0).astype(np.float32)
    danger = rng.beta(2.0, 5.0, cells).astype(np.float32)
    candidate_cells = rng.integers(
        0,
        cells,
        size=max(1, int(candidate_count)),
        dtype=np.int64,
    )
    candidate_capacities = rng.uniform(
        100.0,
        500.0,
        candidate_cells.size,
    ).astype(np.float32)
    candidate_by_cell = np.bincount(candidate_cells, minlength=cells).astype(np.float32)
    deployable = np.zeros(cells, dtype=np.float32)
    for cell, candidate_capacity in zip(candidate_cells, candidate_capacities):
        deployable[cell] = max(deployable[cell], candidate_capacity)
    capacity = rng.uniform(0.0, 300.0, cells).astype(np.float32)
    nodes = rng.integers(1, 30, cells).astype(np.float32)
    return RegionalObservation(
        decision_index=0,
        simulation_time=1,
        horizon=60,
        initial_population=population,
        remaining_deployments=5,
        maximum_deployments=5,
        maximum_speed=75.0,
        active_by_cell=active,
        mean_speed_by_cell=speed,
        danger_by_cell=danger,
        remaining_capacity_by_cell=capacity,
        deployable_capacity_by_cell=deployable,
        candidate_count_by_cell=candidate_by_cell,
        action_mask=np.ones(candidate_cells.size, dtype=bool),
        outcome=OutcomeSnapshot(
            safe_completed=0,
            casualties=0,
            shelter_evacuated=0,
            ordinary_arrivals=0,
            active_population=population,
            risk_mass=float(np.dot(active, 1.0 + danger)),
        ),
        candidate_osm_node_ids=tuple(
            f"synthetic-{index}" for index in range(candidate_cells.size)
        ),
        candidate_cell_indices=candidate_cells,
        candidate_capacities=candidate_capacities,
        candidate_east_positions=rng.uniform(0.0, 1.0, candidate_cells.size),
        candidate_north_positions=rng.uniform(0.0, 1.0, candidate_cells.size),
        network_node_count_by_cell=nodes,
    )


def _graph_from_observation(observation: RegionalObservation, side: int):
    cells, global_features = observation.policy_features()
    tensor = torch.as_tensor(cells, dtype=torch.float32)
    return fit_gnn(
        tensor[:, PED_FEATURE_SLICE],
        tensor[:, HAZARD_FEATURE_SLICE],
        tensor[:, INFRA_FEATURE_SLICE],
        x_global=torch.as_tensor(global_features, dtype=torch.float32).unsqueeze(0),
        edge_index=torch.as_tensor(grid_edge_index(side, side), dtype=torch.long),
        candidate_cell_index=torch.as_tensor(
            observation.candidate_cell_indices,
            dtype=torch.long,
        ),
        candidate_features=torch.as_tensor(
            observation.candidate_features(),
            dtype=torch.float32,
        ),
    )


class _FlatCandidateActor(nn.Module):
    """Candidate-level reference representation for computational scaling.

    This actor is a benchmark implementation, not a trained evacuation policy.
    It applies a shared MLP to one local/global/candidate feature vector per site and
    therefore exposes the memory and scoring cost of a candidate-level action
    set without confounding the comparison with an inefficient Python loop.
    """

    def __init__(self):
        super().__init__()
        feature_count = (
            len(CELL_FEATURE_NAMES)
            + len(GLOBAL_FEATURE_NAMES)
            + len(CANDIDATE_FEATURE_NAMES)
        )
        self.network = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        return self.network(candidate_features).squeeze(-1)


def _measure(
    operation: Callable[[], object],
    *,
    warmups: int,
    repetitions: int,
) -> np.ndarray:
    for _ in range(int(warmups)):
        operation()
    samples = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        start = time.perf_counter_ns()
        operation()
        samples[index] = (time.perf_counter_ns() - start) / 1_000_000.0
    if not np.isfinite(samples).all() or np.any(samples <= 0.0):
        raise RuntimeError("Latency benchmark produced invalid durations")
    return samples


def _timing_rows(
    *,
    family: str,
    method: str,
    samples: np.ndarray,
    action_count: int,
    grid_side: int,
    candidate_count: int,
    directed_edges: int,
    parameter_count: int,
    model_bytes: int,
    input_bytes: int,
) -> list[dict]:
    return [
        {
            "benchmark_family": family,
            "method": method,
            "repetition": int(index + 1),
            "latency_ms": float(value),
            "action_count": int(action_count),
            "grid_side": int(grid_side),
            "cell_count": int(grid_side) ** 2,
            "candidate_count": int(candidate_count),
            "directed_grid_edges": int(directed_edges),
            "parameter_count": int(parameter_count),
            "model_storage_bytes": int(model_bytes),
            "input_storage_bytes": int(input_bytes),
        }
        for index, value in enumerate(samples)
    ]


def run_latency(config: Mapping, output_dir: Path, *, quick: bool, seed: int) -> dict:
    torch.set_num_threads(int(config["latency_benchmark"]["torch_threads"]))
    warmups = int(
        config["latency_benchmark"][
            "quick_warmup_repetitions" if quick else "warmup_repetitions"
        ]
    )
    repetitions = int(
        config["latency_benchmark"][
            "quick_measured_repetitions" if quick else "measured_repetitions"
        ]
    )
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 11]))
    heuristic = ActivePopulationHeuristic()
    rows: list[dict] = []

    for side in config["grid_levels"]:
        side = int(side)
        observation = _synthetic_observation(side, side * side, rng)
        graph = _graph_from_observation(observation, side)
        actor = _policy(side)
        mask = torch.as_tensor(observation.action_mask, dtype=torch.bool).unsqueeze(0)
        parameter_count = sum(parameter.numel() for parameter in actor.parameters())
        model_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in actor.parameters()
        )
        edge_count = int(graph.edge_index.shape[1])
        input_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (graph.x_ped, graph.x_hazard, graph.x_infra, graph.x_global, graph.edge_index)
        )

        def rl_operation():
            with torch.inference_mode():
                logits, _ = actor(graph)
                return int(logits.masked_fill(~mask, -torch.inf).argmax(dim=1).item())

        def heuristic_operation():
            return int(heuristic.select(observation).action_index)

        rows.extend(
            _timing_rows(
                family="grid_dimensionality",
                method="hierarchical_rl",
                samples=_measure(rl_operation, warmups=warmups, repetitions=repetitions),
                action_count=candidate_count,
                grid_side=side,
                candidate_count=side * side,
                directed_edges=edge_count,
                parameter_count=parameter_count,
                model_bytes=model_bytes,
                input_bytes=input_bytes,
            )
        )
        rows.extend(
            _timing_rows(
                family="grid_dimensionality",
                method="population_heuristic",
                samples=_measure(
                    heuristic_operation,
                    warmups=warmups,
                    repetitions=repetitions,
                ),
                action_count=side * side,
                grid_side=side,
                candidate_count=side * side,
                directed_edges=edge_count,
                parameter_count=0,
                model_bytes=0,
                input_bytes=int(
                    observation.active_by_cell.nbytes + observation.action_mask.nbytes
                ),
            )
        )

    side = 8
    actor = _policy(side)
    parameter_count = sum(parameter.numel() for parameter in actor.parameters())
    model_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in actor.parameters()
    )
    flat_actor = _FlatCandidateActor().eval()
    flat_parameter_count = sum(parameter.numel() for parameter in flat_actor.parameters())
    flat_model_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in flat_actor.parameters()
    )
    for candidate_count in config["candidate_levels"]:
        candidate_count = int(candidate_count)
        observation = _synthetic_observation(side, candidate_count, rng)
        graph = _graph_from_observation(observation, side)
        mask = torch.as_tensor(observation.action_mask, dtype=torch.bool).unsqueeze(0)
        flat_feature_count = (
            len(CELL_FEATURE_NAMES)
            + len(GLOBAL_FEATURE_NAMES)
            + len(CANDIDATE_FEATURE_NAMES)
        )
        candidate_features = torch.as_tensor(
            rng.normal(size=(candidate_count, flat_feature_count)), dtype=torch.float32
        )

        def contextual_candidate_operation():
            with torch.inference_mode():
                logits, _ = actor(graph)
                return int(logits.masked_fill(~mask, -torch.inf).argmax(dim=1).item())

        def flat_operation():
            with torch.inference_mode():
                return int(flat_actor(candidate_features).argmax().item())

        edge_count = int(graph.edge_index.shape[1])
        contextual_input_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                graph.x_ped,
                graph.x_hazard,
                graph.x_infra,
                graph.x_global,
                graph.edge_index,
                graph.candidate_cell_index,
                graph.candidate_features,
            )
        )
        rows.extend(
            _timing_rows(
                family="candidate_action_space",
                method="hierarchical_rl",
                samples=_measure(
                    contextual_candidate_operation,
                    warmups=warmups,
                    repetitions=repetitions,
                ),
                action_count=side * side,
                grid_side=side,
                candidate_count=candidate_count,
                directed_edges=edge_count,
                parameter_count=parameter_count,
                model_bytes=model_bytes,
                input_bytes=contextual_input_bytes,
            )
        )
        rows.extend(
            _timing_rows(
                family="candidate_action_space",
                method="flat_candidate_actor",
                samples=_measure(flat_operation, warmups=warmups, repetitions=repetitions),
                action_count=candidate_count,
                grid_side=side,
                candidate_count=candidate_count,
                directed_edges=0,
                parameter_count=flat_parameter_count,
                model_bytes=flat_model_bytes,
                input_bytes=int(candidate_features.numel() * candidate_features.element_size()),
            )
        )

    raw_path = output_dir / "latency_raw.csv"
    _write_csv(raw_path, rows)
    summary = summarize_latency(rows)
    summary_path = output_dir / "latency_summary.csv"
    _write_csv(summary_path, summary)
    scaling = latency_scaling_models(summary)
    scaling_path = output_dir / "latency_scaling_models.json"
    _write_json(scaling_path, scaling)
    return {
        "rows": len(rows),
        "warmups": warmups,
        "measured_repetitions": repetitions,
        "raw_path": str(raw_path),
        "summary_path": str(summary_path),
        "scaling_path": str(scaling_path),
    }


def summarize_latency(rows: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple, list[float]] = {}
    exemplar: dict[tuple, Mapping] = {}
    for row in rows:
        key = (
            row["benchmark_family"],
            row["method"],
            int(row["grid_side"]),
            int(row["candidate_count"]),
        )
        grouped.setdefault(key, []).append(float(row["latency_ms"]))
        exemplar[key] = row
    output = []
    for key, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        row = exemplar[key]
        output.append(
            {
                "benchmark_family": key[0],
                "method": key[1],
                "grid_side": key[2],
                "cell_count": int(row["cell_count"]),
                "candidate_count": key[3],
                "action_count": int(row["action_count"]),
                "directed_grid_edges": int(row["directed_grid_edges"]),
                "parameter_count": int(row["parameter_count"]),
                "model_storage_bytes": int(row["model_storage_bytes"]),
                "input_storage_bytes": int(row["input_storage_bytes"]),
                "repetitions": int(array.size),
                "median_ms": float(np.median(array)),
                "mean_ms": float(array.mean()),
                "standard_deviation_ms": float(array.std(ddof=1)),
                "p05_ms": float(np.quantile(array, 0.05)),
                "p95_ms": float(np.quantile(array, 0.95)),
            }
        )
    return output


def latency_scaling_models(summary: Sequence[Mapping]) -> dict:
    models = []
    for family in sorted({str(row["benchmark_family"]) for row in summary}):
        for method in sorted(
            {str(row["method"]) for row in summary if row["benchmark_family"] == family}
        ):
            subset = [
                row
                for row in summary
                if row["benchmark_family"] == family and row["method"] == method
            ]
            x_name = "cell_count" if family == "grid_dimensionality" else "candidate_count"
            x = np.asarray([float(row[x_name]) for row in subset], dtype=float)
            y = np.asarray([float(row["median_ms"]) for row in subset], dtype=float)
            coefficients = np.polyfit(np.log(x), np.log(y), 1)
            fitted = np.polyval(coefficients, np.log(x))
            residual = np.log(y) - fitted
            total = np.log(y) - np.log(y).mean()
            r_squared = 1.0 - float(np.dot(residual, residual) / max(np.dot(total, total), 1e-15))
            models.append(
                {
                    "benchmark_family": family,
                    "method": method,
                    "scale_variable": x_name,
                    "log_log_slope": float(coefficients[0]),
                    "log_intercept": float(coefficients[1]),
                    "r_squared": r_squared,
                    "observations": len(subset),
                    "interpretation": "median_latency_ms = exp(intercept) * scale^slope",
                }
            )
    return {"models": models}


EXTREME_VARIANTS = (
    "safe_center",
    "dangerous_center",
    "center_unavailable",
    "asymmetric_ring",
)


def _extreme_case(
    rng: np.random.Generator,
    variant: str,
    *,
    side: int = 9,
) -> dict[str, np.ndarray | int | str]:
    if variant not in EXTREME_VARIANTS:
        raise ValueError(f"Unknown extreme-case variant: {variant}")
    center_i = side // 2
    center_j = side // 2
    center = center_i * side + center_j
    ring = np.asarray(
        [
            (center_i + di) * side + (center_j + dj)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if not (di == 0 and dj == 0)
        ],
        dtype=int,
    )
    cells = side * side
    active = np.zeros(cells, dtype=np.float32)
    if variant == "asymmetric_ring":
        hot_position = int(rng.integers(0, ring.size))
        weights = np.full(ring.size, 0.45 / (ring.size - 1), dtype=float)
        weights[hot_position] = 0.55
        weights += rng.normal(0.0, 0.003, ring.size)
        weights = np.clip(weights, 0.001, None)
        weights /= weights.sum()
    else:
        weights = rng.dirichlet(np.full(ring.size, 45.0))
    active[ring] = weights.astype(np.float32)

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

    coordinates = np.asarray([(i, j) for i in range(side) for j in range(side)], dtype=float)
    distances = np.abs(coordinates[:, None, :] - coordinates[None, :, :]).sum(axis=2)
    service = np.asarray(
        [float(np.dot(active, np.exp(-distances[:, action] / 1.30))) for action in range(cells)]
    )
    utility = service - 0.45 * danger.astype(float)
    utility[~mask] = -np.inf
    oracle = int(np.argmax(utility))
    feasible = np.flatnonzero(mask)
    heuristic = int(feasible[np.argmax(active[feasible])])

    cell_features = np.zeros((cells, len(CELL_FEATURE_NAMES)), dtype=np.float32)
    cell_features[:, 0] = active
    cell_features[:, 1] = np.where(active > 0.0, 0.35 + 0.65 * danger, 0.0)
    cell_features[:, 2] = danger
    cell_features[mask, 3] = 0.20
    global_features = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    return {
        "variant": variant,
        "center": center,
        "ring": ring,
        "active": active,
        "danger": danger,
        "mask": mask,
        "cell_features": cell_features,
        "global_features": global_features,
        "utility": utility.astype(np.float32),
        "oracle": oracle,
        "heuristic": heuristic,
    }


def _extreme_batch(
    rng: np.random.Generator,
    variants: Sequence[str],
    *,
    side: int,
) -> tuple[object, torch.Tensor, torch.Tensor, list[dict]]:
    cases = [_extreme_case(rng, variant, side=side) for variant in variants]
    cells = side * side
    feature_array = np.stack([case["cell_features"] for case in cases])
    global_array = np.stack([case["global_features"] for case in cases])
    flattened = torch.as_tensor(
        feature_array.reshape(-1, len(CELL_FEATURE_NAMES)),
        dtype=torch.float32,
    )
    batch = torch.arange(len(cases), dtype=torch.long).repeat_interleave(cells)
    graph = fit_gnn(
        flattened[:, PED_FEATURE_SLICE],
        flattened[:, HAZARD_FEATURE_SLICE],
        flattened[:, INFRA_FEATURE_SLICE],
        x_global=torch.as_tensor(global_array, dtype=torch.float32),
        edge_index=torch.as_tensor(grid_edge_index(side, side), dtype=torch.long),
        batch=batch,
    )
    mask = torch.as_tensor(np.stack([case["mask"] for case in cases]), dtype=torch.bool)
    utility = torch.as_tensor(np.stack([case["utility"] for case in cases]), dtype=torch.float32)
    return graph, mask, utility, cases


def _sample_training_variants(rng: np.random.Generator, size: int) -> list[str]:
    return list(
        rng.choice(
            np.asarray(EXTREME_VARIANTS),
            size=int(size),
            p=np.asarray([0.55, 0.20, 0.15, 0.10]),
        )
    )


def run_extreme(config: Mapping, output_dir: Path, *, quick: bool, seed: int) -> dict:
    design = config["extreme_ring_design"]
    side = int(design["grid"])
    policy_seeds = int(design["quick_policy_seeds"] if quick else design["policy_seeds"])
    updates = int(design["quick_training_updates"] if quick else design["training_updates"])
    batch_size = int(design["quick_batch_size"] if quick else design["batch_size"])
    scenarios = int(
        design[
            "quick_held_out_scenarios_per_variant"
            if quick
            else "held_out_scenarios_per_variant"
        ]
    )
    training_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    checkpoint_dir = output_dir / "extreme_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for policy_replication in range(1, policy_seeds + 1):
        policy_seed = int(
            np.random.SeedSequence([int(seed), 101, policy_replication]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        rng = np.random.default_rng(policy_seed)
        torch.manual_seed(policy_seed)
        actor = _policy(side, init_seed=policy_seed)
        actor.train()
        optimizer = torch.optim.AdamW(actor.parameters(), lr=8e-4, weight_decay=1e-4)
        for update in range(1, updates + 1):
            batch_variants = _sample_training_variants(rng, batch_size)
            graph, mask, utility, cases = _extreme_batch(
                rng, batch_variants, side=side
            )
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
                raise FloatingPointError("Extreme-case policy loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                deterministic = masked_logits.argmax(dim=1)
                deterministic_rewards = utility.gather(
                    1, deterministic.unsqueeze(1)
                ).squeeze(1)
                centers = torch.as_tensor(
                    [int(case["center"]) for case in cases], dtype=torch.long
                )
            training_rows.append(
                {
                    "policy_replication": policy_replication,
                    "policy_seed": policy_seed,
                    "update": update,
                    "batch_size": batch_size,
                    "sampled_reward": float(rewards.mean().item()),
                    "deterministic_reward": float(deterministic_rewards.mean().item()),
                    "deterministic_center_rate": float(
                        (deterministic == centers).to(torch.float32).mean().item()
                    ),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "gradient_norm": float(gradient_norm.item()),
                }
            )

        checkpoint_path = checkpoint_dir / f"policy_{policy_replication:03d}.pt"
        torch.save(
            {
                "schema_version": 1,
                "experiment": "extreme_ring_contextual_bandit",
                "policy_seed": policy_seed,
                "updates": updates,
                "state_dict": actor.state_dict(),
            },
            checkpoint_path,
        )
        actor.eval()
        for variant_index, variant in enumerate(EXTREME_VARIANTS):
            evaluation_seed = int(
                np.random.SeedSequence(
                    [int(seed), 202, variant_index, policy_replication]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            evaluation_rng = np.random.default_rng(evaluation_seed)
            for scenario in range(1, scenarios + 1):
                graph, mask, utility, cases = _extreme_batch(
                    evaluation_rng, [variant], side=side
                )
                case = cases[0]
                with torch.inference_mode():
                    logits, _ = actor(graph)
                    rl_action = int(
                        logits.masked_fill(~mask, -torch.inf).argmax(dim=1).item()
                    )
                oracle_action = int(case["oracle"])
                heuristic_action = int(case["heuristic"])
                utility_values = np.asarray(case["utility"], dtype=float)
                oracle_utility = float(utility_values[oracle_action])
                for strategy, action in (
                    ("rl", rl_action),
                    ("population_heuristic", heuristic_action),
                    ("oracle", oracle_action),
                ):
                    evaluation_rows.append(
                        {
                            "policy_replication": policy_replication,
                            "policy_seed": policy_seed,
                            "variant": variant,
                            "scenario": scenario,
                            "strategy": strategy,
                            "selected_cell": action,
                            "selected_row": action // side,
                            "selected_column": action % side,
                            "center_cell": int(case["center"]),
                            "center_selected": int(action == int(case["center"])),
                            "decision_utility": float(utility_values[action]),
                            "oracle_utility": oracle_utility,
                            "oracle_regret": float(oracle_utility - utility_values[action]),
                            "oracle_selected": int(action == oracle_action),
                            "heuristic_selected": int(action == heuristic_action),
                        }
                    )

    training_path = output_dir / "extreme_training.csv"
    evaluation_path = output_dir / "extreme_evaluation.csv"
    _write_csv(training_path, training_rows)
    _write_csv(evaluation_path, evaluation_rows)
    summary = summarize_extreme(evaluation_rows)
    summary_path = output_dir / "extreme_summary.csv"
    _write_csv(summary_path, summary)
    comparison_path = output_dir / "extreme_paired_comparison.csv"
    _write_csv(comparison_path, paired_extreme_comparisons(evaluation_rows))
    canonical = {
        variant: _strict(_extreme_case(np.random.default_rng(9090 + index), variant, side=side))
        for index, variant in enumerate(EXTREME_VARIANTS)
    }
    canonical_path = output_dir / "extreme_canonical_cases.json"
    _write_json(canonical_path, canonical)
    return {
        "policy_seeds": policy_seeds,
        "training_updates_per_seed": updates,
        "held_out_scenarios_per_variant_seed": scenarios,
        "training_path": str(training_path),
        "evaluation_path": str(evaluation_path),
        "summary_path": str(summary_path),
        "paired_comparison_path": str(comparison_path),
        "canonical_path": str(canonical_path),
    }


def summarize_extreme(rows: Sequence[Mapping]) -> list[dict]:
    groups: dict[tuple[str, str], list[Mapping]] = {}
    for row in rows:
        groups.setdefault((str(row["variant"]), str(row["strategy"])), []).append(row)
    output = []
    for (variant, strategy), subset in sorted(groups.items()):
        utility = np.asarray([float(row["decision_utility"]) for row in subset])
        regret = np.asarray([float(row["oracle_regret"]) for row in subset])
        center = np.asarray([float(row["center_selected"]) for row in subset])
        oracle = np.asarray([float(row["oracle_selected"]) for row in subset])
        seed_means = []
        for policy in sorted({int(row["policy_replication"]) for row in subset}):
            values = np.asarray(
                [
                    float(row["decision_utility"])
                    for row in subset
                    if int(row["policy_replication"]) == policy
                ]
            )
            seed_means.append(float(values.mean()))
        seed_array = np.asarray(seed_means, dtype=float)
        half_width = (
            1.96 * float(seed_array.std(ddof=1)) / math.sqrt(seed_array.size)
            if seed_array.size > 1
            else 0.0
        )
        output.append(
            {
                "variant": variant,
                "strategy": strategy,
                "observations": len(subset),
                "policy_seed_clusters": int(seed_array.size),
                "mean_decision_utility": float(utility.mean()),
                "utility_cluster_95_ci_low": float(seed_array.mean() - half_width),
                "utility_cluster_95_ci_high": float(seed_array.mean() + half_width),
                "mean_oracle_regret": float(regret.mean()),
                "median_oracle_regret": float(np.median(regret)),
                "center_selection_rate": float(center.mean()),
                "oracle_selection_rate": float(oracle.mean()),
            }
        )
    return output


def _exact_sign_randomization_pvalue(cluster_means: np.ndarray) -> float:
    values = np.asarray(cluster_means, dtype=float)
    observed = abs(float(values.mean()))
    if values.size == 0:
        return float("nan")
    exceedances = 0
    assignments = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        assignments += 1
        permuted = abs(float(np.mean(values * np.asarray(signs))))
        exceedances += int(permuted >= observed - 1e-15)
    return float(exceedances / assignments)


def paired_extreme_comparisons(rows: Sequence[Mapping]) -> list[dict]:
    """Compare RL and heuristic on identical held-out contextual cases."""
    output = []
    for variant in EXTREME_VARIANTS:
        subset = [row for row in rows if str(row["variant"]) == variant]
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
        utility_differences = []
        regret_reductions = []
        center_differences = []
        policy_utility_means = []
        for policy in policies:
            policy_differences = []
            for scenario in scenarios:
                rl = by_key[(policy, scenario, "rl")]
                heuristic = by_key[(policy, scenario, "population_heuristic")]
                utility_difference = float(rl["decision_utility"]) - float(
                    heuristic["decision_utility"]
                )
                utility_differences.append(utility_difference)
                policy_differences.append(utility_difference)
                regret_reductions.append(
                    float(heuristic["oracle_regret"]) - float(rl["oracle_regret"])
                )
                center_differences.append(
                    float(rl["center_selected"])
                    - float(heuristic["center_selected"])
                )
            policy_utility_means.append(float(np.mean(policy_differences)))
        cluster_means = np.asarray(policy_utility_means, dtype=float)
        standard_error = (
            float(cluster_means.std(ddof=1) / math.sqrt(cluster_means.size))
            if cluster_means.size > 1
            else 0.0
        )
        output.append(
            {
                "variant": variant,
                "policy_seed_clusters": len(policies),
                "scenarios_per_policy_seed": len(scenarios),
                "paired_observations": len(utility_differences),
                "mean_rl_utility_improvement": float(np.mean(utility_differences)),
                "cluster_normal_95_ci_low": float(
                    cluster_means.mean() - 1.96 * standard_error
                ),
                "cluster_normal_95_ci_high": float(
                    cluster_means.mean() + 1.96 * standard_error
                ),
                "two_sided_exact_sign_randomization_p": _exact_sign_randomization_pvalue(
                    cluster_means
                ),
                "rl_utility_win_rate": float(
                    np.mean(np.asarray(utility_differences) > 0.0)
                ),
                "mean_rl_regret_reduction": float(np.mean(regret_reductions)),
                "mean_center_selection_rate_difference": float(
                    np.mean(center_differences)
                ),
                "inference_unit": "policy_seed",
            }
        )
    return output


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    paths: list[str] = []

    latency_path = output_dir / "latency_summary.csv"
    if latency_path.exists():
        rows = _load_csv(latency_path)
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
        grid_rows = [row for row in rows if row["benchmark_family"] == "grid_dimensionality"]
        for method, label, color in (
            ("hierarchical_rl", "Exact-site contextual actor", "#0072B2"),
            ("population_heuristic", "Population heuristic", "#D55E00"),
        ):
            subset = sorted(
                [row for row in grid_rows if row["method"] == method],
                key=lambda row: int(row["cell_count"]),
            )
            x = np.asarray([int(row["cell_count"]) for row in subset])
            median = np.asarray([float(row["median_ms"]) for row in subset])
            p05 = np.asarray([float(row["p05_ms"]) for row in subset])
            p95 = np.asarray([float(row["p95_ms"]) for row in subset])
            axes[0].plot(x, median, marker="o", label=label, color=color)
            axes[0].fill_between(x, p05, p95, alpha=0.16, color=color)
        axes[0].set(
            xscale="log",
            yscale="log",
            xlabel="Regional cells",
            ylabel="Decision latency (ms)",
            title="(a) Grid dimensionality",
        )
        axes[0].legend(frameon=False)

        candidate_rows = [
            row for row in rows if row["benchmark_family"] == "candidate_action_space"
        ]
        for method, label, color in (
            ("hierarchical_rl", "Exact-site contextual actor", "#0072B2"),
            ("flat_candidate_actor", "Context-free candidate MLP", "#009E73"),
        ):
            subset = sorted(
                [row for row in candidate_rows if row["method"] == method],
                key=lambda row: int(row["candidate_count"]),
            )
            axes[1].plot(
                [int(row["candidate_count"]) for row in subset],
                [float(row["median_ms"]) for row in subset],
                marker="o",
                label=label,
                color=color,
            )
        axes[1].set(
            xscale="log",
            yscale="log",
            xlabel="Candidate sites",
            ylabel="Decision latency (ms)",
            title="(b) Candidate action growth",
        )
        axes[1].legend(frameon=False)

        for method, label, color in (
            ("hierarchical_rl", "Contextual exact-site input", "#0072B2"),
            ("flat_candidate_actor", "Context-free input", "#009E73"),
        ):
            subset = sorted(
                [row for row in candidate_rows if row["method"] == method],
                key=lambda row: int(row["candidate_count"]),
            )
            axes[2].plot(
                [int(row["candidate_count"]) for row in subset],
                [float(row["input_storage_bytes"]) / 1024.0 for row in subset],
                marker="o",
                label=label,
                color=color,
            )
        axes[2].set(
            xscale="log",
            yscale="log",
            xlabel="Candidate sites",
            ylabel="Input representation (KiB)",
            title="(c) Action input storage",
        )
        axes[2].legend(frameon=False)
        for axis in axes:
            axis.grid(alpha=0.25, which="both")
        fig.tight_layout()
        paths.extend(_save_figure(fig, output_dir, "F6_F7_computational_scaling"))
        plt.close(fig)

    extreme_path = output_dir / "extreme_summary.csv"
    if extreme_path.exists():
        rows = _load_csv(extreme_path)
        variants = list(EXTREME_VARIANTS)
        methods = ["rl", "population_heuristic"]
        labels = {"rl": "RL", "population_heuristic": "Heuristic"}
        colors = {"rl": "#0072B2", "population_heuristic": "#D55E00"}
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
        width = 0.36
        x = np.arange(len(variants))
        for offset, method in enumerate(methods):
            subset = {
                row["variant"]: row for row in rows if row["strategy"] == method
            }
            axes[0].bar(
                x + (offset - 0.5) * width,
                [float(subset[variant]["center_selection_rate"]) for variant in variants],
                width,
                label=labels[method],
                color=colors[method],
            )
            axes[1].bar(
                x + (offset - 0.5) * width,
                [float(subset[variant]["mean_oracle_regret"]) for variant in variants],
                width,
                label=labels[method],
                color=colors[method],
            )
        pretty = ["Safe\ncenter", "Dangerous\ncenter", "Center\nunavailable", "Asymmetric\nring"]
        axes[0].set(
            xticks=x,
            xticklabels=pretty,
            ylim=(0.0, 1.05),
            ylabel="Center-selection rate",
            title="(a) Learned contextual response",
        )
        axes[1].set(
            xticks=x,
            xticklabels=pretty,
            ylabel="Mean regret from oracle",
            title="(b) Decision quality",
        )
        axes[0].legend(frameon=False)
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        paths.extend(_save_figure(fig, output_dir, "F9_extreme_ring_performance"))
        plt.close(fig)

    canonical_path = output_dir / "extreme_canonical_cases.json"
    if canonical_path.exists():
        canonical = _read_json(canonical_path)
        fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.6), sharex=True, sharey=True)
        for column, variant in enumerate(EXTREME_VARIANTS):
            case = canonical[variant]
            active = np.asarray(case["active"], dtype=float).reshape(9, 9)
            danger = np.asarray(case["danger"], dtype=float).reshape(9, 9)
            for row_index, (values, cmap, label) in enumerate(
                ((active, "Reds", "Population share"), (danger, "magma", "Danger"))
            ):
                image = axes[row_index, column].imshow(
                    values, origin="lower", cmap=cmap, vmin=0.0
                )
                axes[row_index, column].scatter(
                    [4], [4], marker="s", s=55, facecolors="none", edgecolors="#00FFFF", linewidths=1.5
                )
                oracle = int(case["oracle"])
                heuristic = int(case["heuristic"])
                axes[row_index, column].scatter(
                    [oracle % 9], [oracle // 9], marker="*", s=75, color="#56B4E9", label="Oracle"
                )
                axes[row_index, column].scatter(
                    [heuristic % 9], [heuristic // 9], marker="x", s=50, color="#F0E442", label="Heuristic"
                )
                if column == 0:
                    axes[row_index, column].set_ylabel(label)
                if row_index == 0:
                    axes[row_index, column].set_title(variant.replace("_", " ").title())
                axes[row_index, column].set_xticks([])
                axes[row_index, column].set_yticks([])
                fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.03)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle("Controlled populated-ring / empty-center cases", y=1.01)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        paths.extend(_save_figure(fig, output_dir, "F9_extreme_ring_design"))
        plt.close(fig)

    training_path = output_dir / "extreme_training.csv"
    if training_path.exists():
        rows = _load_csv(training_path)
        policies = sorted({int(row["policy_replication"]) for row in rows})
        updates = sorted({int(row["update"]) for row in rows})
        reward = np.asarray(
            [
                [
                    float(
                        next(
                            row["deterministic_reward"]
                            for row in rows
                            if int(row["policy_replication"]) == policy
                            and int(row["update"]) == update
                        )
                    )
                    for update in updates
                ]
                for policy in policies
            ]
        )
        center = np.asarray(
            [
                [
                    float(
                        next(
                            row["deterministic_center_rate"]
                            for row in rows
                            if int(row["policy_replication"]) == policy
                            and int(row["update"]) == update
                        )
                    )
                    for update in updates
                ]
                for policy in policies
            ]
        )
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))
        for axis, values, ylabel, title in (
            (axes[0], reward, "Deterministic utility", "(a) Decision utility"),
            (axes[1], center, "Center-selection rate", "(b) Behavioral shift"),
        ):
            mean = values.mean(axis=0)
            standard_error = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
            axis.plot(updates, mean, color="#0072B2")
            axis.fill_between(
                updates,
                mean - 1.96 * standard_error,
                mean + 1.96 * standard_error,
                color="#0072B2",
                alpha=0.18,
            )
            axis.set(xlabel="Training update", ylabel=ylabel, title=title)
            axis.grid(alpha=0.25)
        fig.tight_layout()
        paths.extend(_save_figure(fig, output_dir, "F4_extreme_training_diagnostics"))
        plt.close(fig)

    readiness = [
        ("F1", "City-population effects", False, "Awaiting confirmatory city runs"),
        ("F2", "Primary paired forest", False, "Awaiting confirmatory city runs"),
        ("F3", "Sequential timing ablation", False, "Awaiting rl_precommit city runs"),
        ("F4", "Training diagnostics", training_path.exists(), "Controlled stress test complete"),
        ("F5", "Dynamic map sequence", False, "Awaiting confirmatory map episode"),
        ("F6", "Grid latency", latency_path.exists(), "Computational tier complete"),
        ("F7", "Candidate action scaling", latency_path.exists(), "Computational tier complete"),
        ("F8", "Grid accuracy tradeoff", False, "Awaiting grid-specific training"),
        ("F9", "Extreme ring mechanism", extreme_path.exists(), "Mechanism tier complete"),
        ("F10", "Runtime decomposition", False, "Awaiting instrumented city runs"),
        ("F11", "Robustness heatmap", False, "Awaiting robustness trials"),
        ("F12", "Completion audit", True, "Generated from current evidence state"),
    ]
    readiness_rows = [
        {
            "figure_id": figure_id,
            "figure_title": title,
            "status": "complete" if complete else "pending",
            "reason": reason,
        }
        for figure_id, title, complete, reason in readiness
    ]
    _write_csv(output_dir / "figure_readiness.csv", readiness_rows)
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    values = np.asarray([[1.0 if row[2] else 0.0 for row in readiness]])
    ax.imshow(values, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#D9D9D9", "#009E73"]), vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(readiness)), [row[0] for row in readiness])
    ax.set_yticks([0], ["Evidence status"])
    for index, (_, _, complete, _) in enumerate(readiness):
        ax.text(index, 0, "Complete" if complete else "Pending", ha="center", va="center", rotation=90, color="white" if complete else "#333333", fontsize=8)
    ax.set_title("Publication-figure readiness: completed evidence is not backfilled")
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    paths.extend(_save_figure(fig, output_dir, "F12_figure_readiness"))
    plt.close(fig)
    return paths


def _design_counts(config: Mapping) -> dict:
    cities = len(config["cities"])
    populations = len(config["population_levels"])
    capacity_regimes = len(
        config["confirmatory_design"].get("capacity_regimes", {"default": None})
    )
    seeds = int(config["confirmatory_design"]["policy_seeds"])
    scenarios = int(
        config["confirmatory_design"]["held_out_scenarios_per_city_population"]
    )
    dynamic_learned = 2 * seeds
    nonlearned = 4
    return {
        "training_city_episodes": (
            cities
            * seeds
            * int(config["confirmatory_design"]["training_episodes_per_city"])
        ),
        "primary_rl_heuristic_evaluation_episodes": cities
        * populations
        * capacity_regimes
        * scenarios
        * (seeds + 1),
        "all_strategy_evaluation_episodes": cities
        * populations
        * capacity_regimes
        * scenarios
        * (dynamic_learned + nonlearned),
        "city_population_capacity_cells": cities * populations * capacity_regimes,
    }


def confirmatory_design_matrix(config: Mapping) -> list[dict]:
    seeds = int(config["confirmatory_design"]["policy_seeds"])
    scenarios = int(
        config["confirmatory_design"]["held_out_scenarios_per_city_population"]
    )
    regimes = config["confirmatory_design"].get(
        "capacity_regimes", {"default": "Single registered capacity regime."}
    )
    rows = []
    for city_index, city_id in enumerate(config["cities"], start=1):
        for population in config["population_levels"]:
            for regime, definition in regimes.items():
                rows.append(
                    {
                        "city_id": city_id,
                        "city_design_index": city_index,
                        "population": int(population),
                        "capacity_regime": str(regime),
                        "capacity_regime_definition": str(definition),
                        "policy_seeds": seeds,
                        "held_out_scenarios": scenarios,
                        "primary_rl_heuristic_episodes": scenarios * (seeds + 1),
                        "all_strategy_episodes": scenarios * (2 * seeds + 4),
                        "common_random_numbers": True,
                        "status": "pending_city_execution",
                    }
                )
    return rows


def audit_existing(config: Mapping, output_dir: Path) -> dict:
    """Recompute summaries, validate row counts, and refresh artifact hashes."""
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") not in {"complete", "pilot_complete"}:
        raise RuntimeError("Only a completed experiment directory can be audited")
    quick = bool(manifest.get("quick", False))
    repetitions = int(
        config["latency_benchmark"][
            "quick_measured_repetitions" if quick else "measured_repetitions"
        ]
    )
    extreme = config["extreme_ring_design"]
    policies = int(extreme["quick_policy_seeds"] if quick else extreme["policy_seeds"])
    updates = int(
        extreme["quick_training_updates"] if quick else extreme["training_updates"]
    )
    scenarios = int(
        extreme[
            "quick_held_out_scenarios_per_variant"
            if quick
            else "held_out_scenarios_per_variant"
        ]
    )
    latency_rows = _load_csv(output_dir / "latency_raw.csv")
    extreme_training = _load_csv(output_dir / "extreme_training.csv")
    extreme_evaluation = _load_csv(output_dir / "extreme_evaluation.csv")
    expected = {
        "latency_rows": 2
        * (len(config["grid_levels"]) + len(config["candidate_levels"]))
        * repetitions,
        "extreme_training_rows": policies * updates,
        "extreme_evaluation_rows": policies
        * len(EXTREME_VARIANTS)
        * scenarios
        * 3,
        "checkpoint_count": policies,
    }
    observed = {
        "latency_rows": len(latency_rows),
        "extreme_training_rows": len(extreme_training),
        "extreme_evaluation_rows": len(extreme_evaluation),
        "checkpoint_count": len(list((output_dir / "extreme_checkpoints").glob("policy_*.pt"))),
    }
    if observed != expected:
        raise RuntimeError(f"Audit count mismatch: expected={expected}, observed={observed}")
    if not all(
        math.isfinite(float(row["latency_ms"])) and float(row["latency_ms"]) > 0.0
        for row in latency_rows
    ):
        raise RuntimeError("Non-finite or non-positive latency found")
    if not all(
        math.isfinite(float(row[field]))
        for row in extreme_training
        for field in (
            "sampled_reward",
            "deterministic_reward",
            "policy_loss",
            "value_loss",
            "entropy",
            "gradient_norm",
        )
    ):
        raise RuntimeError("Non-finite extreme-case training diagnostic found")

    _write_csv(output_dir / "latency_summary.csv", summarize_latency(latency_rows))
    _write_json(
        output_dir / "latency_scaling_models.json",
        latency_scaling_models(summarize_latency(latency_rows)),
    )
    _write_csv(output_dir / "extreme_summary.csv", summarize_extreme(extreme_evaluation))
    paired_path = output_dir / "extreme_paired_comparison.csv"
    _write_csv(paired_path, paired_extreme_comparisons(extreme_evaluation))
    _write_csv(
        output_dir / "confirmatory_design_matrix.csv",
        confirmatory_design_matrix(config),
    )
    figure_paths = generate_figures(output_dir)
    if isinstance(manifest.get("artifacts", {}).get("extreme"), dict):
        manifest["artifacts"]["extreme"]["paired_comparison_path"] = str(paired_path)
    environment = manifest.setdefault("environment", {})
    if "torch_threads" in environment and "torch_threads_at_process_start" not in environment:
        environment["torch_threads_at_process_start"] = int(
            environment.pop("torch_threads")
        )
    environment["latency_torch_threads_configured"] = int(
        config["latency_benchmark"]["torch_threads"]
    )
    manifest["audit"] = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "expected_counts": expected,
        "observed_counts": observed,
        "all_latency_values_finite_positive": True,
        "all_training_diagnostics_finite": True,
        "figures_regenerated": figure_paths,
    }
    manifest["design_counts"] = _design_counts(config)
    artifact_hashes = {}
    for path in output_dir.rglob("*"):
        if path.is_file() and path != manifest_path:
            artifact_hashes[str(path.relative_to(output_dir))] = _sha256(path)
    manifest["artifact_sha256"] = artifact_hashes
    _write_json(manifest_path, manifest)
    return manifest["audit"]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "latency", "extreme", "tractable", "figures", "audit"),
        default="tractable",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = _read_json(args.config.resolve())
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported OR-journal experiment schema")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "audit":
        result = audit_existing(config, output_dir)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    _write_csv(
        output_dir / "confirmatory_design_matrix.csv",
        confirmatory_design_matrix(config),
    )
    started = datetime.now(timezone.utc)
    manifest = {
        "schema_version": 1,
        "suite_id": config["suite_id"],
        "status": "running",
        "mode": args.mode,
        "quick": bool(args.quick),
        "seed": int(args.seed),
        "started_utc": started.isoformat(),
        "design_path": str(args.config.resolve()),
        "design_sha256": _sha256(args.config.resolve()),
        "design_counts": _design_counts(config),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads_at_start": torch.get_num_threads(),
            "latency_torch_threads_configured": int(
                config["latency_benchmark"]["torch_threads"]
            ),
        },
        "git": {
            "commit": _git_value("git", "rev-parse", "HEAD"),
            "status": _git_value("git", "status", "--short"),
        },
        "artifacts": {},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest["design_counts"], indent=2), flush=True)
    if args.mode == "dry-run":
        manifest["status"] = "dry_run_complete"
    else:
        if args.mode in {"latency", "tractable"}:
            manifest["artifacts"]["latency"] = run_latency(
                config, output_dir, quick=args.quick, seed=args.seed
            )
        if args.mode in {"extreme", "tractable"}:
            manifest["artifacts"]["extreme"] = run_extreme(
                config, output_dir, quick=args.quick, seed=args.seed
            )
        if args.mode in {"figures", "tractable", "latency", "extreme"}:
            manifest["artifacts"]["figures"] = generate_figures(output_dir)
        manifest["status"] = "pilot_complete" if args.quick else "complete"

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["wall_time_s"] = float(
        (datetime.fromisoformat(manifest["completed_utc"]) - started).total_seconds()
    )
    artifact_hashes = {}
    for path in output_dir.rglob("*"):
        if path.is_file() and path != manifest_path:
            artifact_hashes[str(path.relative_to(output_dir))] = _sha256(path)
    manifest["artifact_sha256"] = artifact_hashes
    _write_json(manifest_path, manifest)
    print(f"Experiment status: {manifest['status']}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
