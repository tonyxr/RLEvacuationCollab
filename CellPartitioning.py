#!/usr/bin/env python3
"""Deterministic rectangular partitions for regional shelter decisions.

Both modes retain an ``nx`` by ``ny`` tensor and the same four-neighbour cell
topology.  The only difference is the placement of the projected-metre axis
boundaries:

``equal_area``
    Equal-width intervals on both axes, hence equal-area rectangles.

``node_density_adaptive``
    Empirical road-node quantiles on each axis.  Dense coordinate ranges get
    narrower intervals while the number and indexing of cells stay fixed.

``population_density_adaptive``
    Weighted quantiles of the scenario's realized initial pedestrian origins.
    Each represented person contributes one unit of mass, so dense population
    ranges receive narrower cells without relying on a city-specific census
    layer.  The outer boundaries still cover the complete road network.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


EQUAL_AREA = "equal_area"
NODE_DENSITY_ADAPTIVE = "node_density_adaptive"
POPULATION_DENSITY_ADAPTIVE = "population_density_adaptive"
PARTITION_MODES = (
    EQUAL_AREA,
    NODE_DENSITY_ADAPTIVE,
    POPULATION_DENSITY_ADAPTIVE,
)


def normalize_partition_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "uniform": EQUAL_AREA,
        "equal": EQUAL_AREA,
        "adaptive": NODE_DENSITY_ADAPTIVE,
        "node_adaptive": NODE_DENSITY_ADAPTIVE,
        "population_adaptive": POPULATION_DENSITY_ADAPTIVE,
        "pedestrian_density_adaptive": POPULATION_DENSITY_ADAPTIVE,
    }
    value = aliases.get(value, value)
    if value not in PARTITION_MODES:
        raise ValueError(
            f"cellPartitionMode must be one of {PARTITION_MODES}; got {mode!r}"
        )
    return value


def _finite_axis(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} coordinates must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} coordinates must all be finite")
    return array


def _axis_edges(
    values: Sequence[float],
    bins: int,
    mode: str,
    *,
    min_width_fraction: float,
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    array = _finite_axis(values, "axis")
    bins = int(bins)
    if bins <= 0:
        raise ValueError("bins must be positive")
    axis_min = float(np.min(array))
    axis_max = float(np.max(array))
    if axis_max <= axis_min:
        axis_max = axis_min + 1.0
    span = axis_max - axis_min

    if mode == EQUAL_AREA:
        return np.linspace(axis_min, axis_max, bins + 1, dtype=float)

    minimum = float(min_width_fraction)
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("min_width_fraction must be finite and non-negative")
    if minimum * bins >= 1.0:
        raise ValueError("min_width_fraction * bins must be less than one")

    probabilities = np.linspace(0.0, 1.0, bins + 1)
    if mode == POPULATION_DENSITY_ADAPTIVE:
        mass = _finite_weights(weights, array.size)
        order = np.argsort(array, kind="mergesort")
        ordered_values = array[order]
        ordered_mass = mass[order]
        positive = ordered_mass > 0.0
        ordered_values = ordered_values[positive]
        ordered_mass = ordered_mass[positive]
        cumulative = np.cumsum(ordered_mass)
        # Inverting the right-continuous empirical CDF makes each marginal
        # strip contain approximately equal represented population.  Linear
        # interpolation avoids unnecessary duplicate edges at large origins.
        cumulative_midpoints = (cumulative - 0.5 * ordered_mass) / cumulative[-1]
        raw = np.interp(
            probabilities,
            cumulative_midpoints,
            ordered_values,
            left=axis_min,
            right=axis_max,
        ).astype(float)
    else:
        raw = np.quantile(array, probabilities).astype(float)
    raw[0] = axis_min
    raw[-1] = axis_max
    min_width = span * minimum
    edges = np.empty(bins + 1, dtype=float)
    edges[0] = axis_min
    edges[-1] = axis_max
    for index in range(1, bins):
        lower = edges[index - 1] + min_width
        upper = axis_max - (bins - index) * min_width
        edges[index] = min(max(float(raw[index]), lower), upper)
    if not np.all(np.diff(edges) > 0.0):
        raise RuntimeError("Adaptive partition construction produced invalid edges")
    return edges


def _finite_weights(
    values: Sequence[float] | None,
    expected_size: int,
) -> np.ndarray:
    if values is None:
        raise ValueError(
            "population_density_adaptive requires one population weight per node"
        )
    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.size != int(expected_size):
        raise ValueError("population weights must match the coordinate count")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("population weights must be finite and non-negative")
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("population weights must contain positive mass")
    return weights


def sample_population_origins(
    node_count: int,
    total_population: int,
    maximum_group_size: int,
    seed: int,
) -> dict:
    """Draw a reproducible origin plan before the population-adaptive cut.

    The returned node mass is the exact represented population that will be
    initialized at every sampled road node.  Sampling from its own stream
    prevents shelter and hazard initialization from changing the population
    grid merely because their implementation consumes a different RNG draw.
    """
    node_count = int(node_count)
    total_population = int(total_population)
    maximum_group_size = int(maximum_group_size)
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if total_population <= 0:
        raise ValueError("total_population must be positive")
    if maximum_group_size <= 0:
        raise ValueError("maximum_group_size must be positive")

    group_sizes = []
    remaining = total_population
    while remaining > 0:
        size = min(maximum_group_size, remaining)
        group_sizes.append(int(size))
        remaining -= size
    rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
    origin_indices = rng.integers(0, node_count, size=len(group_sizes), endpoint=False)
    node_population = np.zeros(node_count, dtype=float)
    np.add.at(node_population, origin_indices, np.asarray(group_sizes, dtype=float))
    if int(round(float(node_population.sum()))) != total_population:
        raise RuntimeError("Population-origin plan does not conserve represented people")
    return {
        "origin_indices": [int(index) for index in origin_indices],
        "group_sizes": group_sizes,
        "node_population": node_population,
    }


def _gini(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0 or float(np.sum(flat)) <= 0.0:
        return 0.0
    ordered = np.sort(flat)
    index = np.arange(1, ordered.size + 1, dtype=float)
    return float(
        np.sum((2.0 * index - ordered.size - 1.0) * ordered)
        / (ordered.size * np.sum(ordered))
    )


def partition_diagnostics(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    *,
    mode: str,
    population_weights: Sequence[float] | None = None,
) -> dict:
    """Summarize geometry and road-node balance without changing the model."""
    x = _finite_axis(x_values, "x")
    y = _finite_axis(y_values, "y")
    if x.size != y.size:
        raise ValueError("x and y must contain the same number of nodes")
    x_boundaries = np.asarray(x_edges, dtype=float)
    y_boundaries = np.asarray(y_edges, dtype=float)
    if (
        x_boundaries.size < 2
        or y_boundaries.size < 2
        or not np.isfinite(x_boundaries).all()
        or not np.isfinite(y_boundaries).all()
        or not np.all(np.diff(x_boundaries) > 0.0)
        or not np.all(np.diff(y_boundaries) > 0.0)
    ):
        raise ValueError("Partition edges must be finite and strictly increasing")

    counts, _, _ = np.histogram2d(x, y, bins=(x_boundaries, y_boundaries))
    population_counts = None
    if population_weights is not None:
        population_mass = _finite_weights(population_weights, x.size)
        population_counts, _, _ = np.histogram2d(
            x,
            y,
            bins=(x_boundaries, y_boundaries),
            weights=population_mass,
        )
    areas = np.outer(np.diff(x_boundaries), np.diff(y_boundaries))
    mean_count = float(np.mean(counts))
    mean_area = float(np.mean(areas))
    positive = counts > 0.0
    densities = np.divide(counts, areas, out=np.zeros_like(counts), where=areas > 0.0)
    log_area_density_correlation = None
    log_areas = np.log(areas[positive])
    log_densities = np.log(densities[positive])
    if (
        int(np.sum(positive)) >= 2
        and float(np.std(log_areas)) > 0.0
        and float(np.std(log_densities)) > 0.0
    ):
        correlation = float(
            np.corrcoef(log_areas, log_densities)[0, 1]
        )
        if math.isfinite(correlation):
            log_area_density_correlation = correlation

    edge_payload = {
        "mode": normalize_partition_mode(mode),
        "x_edges_m": [float(value) for value in x_boundaries],
        "y_edges_m": [float(value) for value in y_boundaries],
    }
    edge_digest = hashlib.sha256(
        json.dumps(edge_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "mode": normalize_partition_mode(mode),
        "grid_shape": [int(counts.shape[0]), int(counts.shape[1])],
        "cell_count": int(counts.size),
        "road_node_count": int(x.size),
        "empty_cell_fraction": float(np.mean(counts == 0.0)),
        "node_count_mean": mean_count,
        "node_count_std": float(np.std(counts)),
        "node_count_cv": float(np.std(counts) / mean_count) if mean_count > 0.0 else 0.0,
        "node_count_gini": _gini(counts),
        "node_count_min": int(np.min(counts)),
        "node_count_median": float(np.median(counts)),
        "node_count_max": int(np.max(counts)),
        "cell_area_mean_m2": mean_area,
        "cell_area_min_m2": float(np.min(areas)),
        "cell_area_median_m2": float(np.median(areas)),
        "cell_area_max_m2": float(np.max(areas)),
        "cell_area_cv": float(np.std(areas) / mean_area) if mean_area > 0.0 else 0.0,
        "cell_area_max_min_ratio": float(np.max(areas) / np.min(areas)),
        "log_area_node_density_correlation": log_area_density_correlation,
        "partition_edge_sha256": edge_digest,
        "x_edges_m": edge_payload["x_edges_m"],
        "y_edges_m": edge_payload["y_edges_m"],
    }
    if population_counts is not None:
        mean_population = float(np.mean(population_counts))
        population_densities = np.divide(
            population_counts,
            areas,
            out=np.zeros_like(population_counts),
            where=areas > 0.0,
        )
        occupied = population_counts > 0.0
        population_area_correlation = None
        if (
            int(np.sum(occupied)) >= 2
            and float(np.std(np.log(areas[occupied]))) > 0.0
            and float(np.std(np.log(population_densities[occupied]))) > 0.0
        ):
            value = float(
                np.corrcoef(
                    np.log(areas[occupied]),
                    np.log(population_densities[occupied]),
                )[0, 1]
            )
            if math.isfinite(value):
                population_area_correlation = value
        result.update(
            {
                "population_mass_total": float(np.sum(population_counts)),
                "population_count_mean": mean_population,
                "population_count_std": float(np.std(population_counts)),
                "population_count_cv": (
                    float(np.std(population_counts) / mean_population)
                    if mean_population > 0.0
                    else 0.0
                ),
                "population_count_gini": _gini(population_counts),
                "population_count_min": float(np.min(population_counts)),
                "population_count_median": float(np.median(population_counts)),
                "population_count_max": float(np.max(population_counts)),
                "empty_population_cell_fraction": float(
                    np.mean(population_counts == 0.0)
                ),
                "log_area_population_density_correlation": (
                    population_area_correlation
                ),
            }
        )
    return result


def build_cell_partition(
    x_values: Sequence[float],
    y_values: Sequence[float],
    nx: int,
    ny: int,
    mode: str,
    *,
    min_width_fraction: float = 1e-4,
    population_weights: Sequence[float] | None = None,
) -> dict:
    """Return deterministic edges and diagnostics for an ``nx`` by ``ny`` grid."""
    canonical_mode = normalize_partition_mode(mode)
    x = _finite_axis(x_values, "x")
    y = _finite_axis(y_values, "y")
    if x.size != y.size:
        raise ValueError("x and y must contain the same number of nodes")
    weights = None
    if canonical_mode == POPULATION_DENSITY_ADAPTIVE:
        weights = _finite_weights(population_weights, x.size)
    x_edges = _axis_edges(
        x,
        int(nx),
        canonical_mode,
        min_width_fraction=min_width_fraction,
        weights=weights,
    )
    y_edges = _axis_edges(
        y,
        int(ny),
        canonical_mode,
        min_width_fraction=min_width_fraction,
        weights=weights,
    )
    diagnostics = partition_diagnostics(
        x,
        y,
        x_edges,
        y_edges,
        mode=canonical_mode,
        population_weights=weights,
    )
    return {
        "mode": canonical_mode,
        "x_edges": [float(value) for value in x_edges],
        "y_edges": [float(value) for value in y_edges],
        "diagnostics": diagnostics,
    }


def cell_rows(partition: Mapping) -> list[dict]:
    """Materialize per-cell geometry for audit tables and re-plotting."""
    diagnostics = partition["diagnostics"]
    x_edges = np.asarray(partition["x_edges"], dtype=float)
    y_edges = np.asarray(partition["y_edges"], dtype=float)
    # The caller can attach counts after locating the observed nodes.
    rows = []
    for x_index in range(x_edges.size - 1):
        for y_index in range(y_edges.size - 1):
            rows.append(
                {
                    "partition_mode": str(partition["mode"]),
                    "grid_nx": int(x_edges.size - 1),
                    "grid_ny": int(y_edges.size - 1),
                    "cell_x": int(x_index),
                    "cell_y": int(y_index),
                    "cell_index": int(x_index * (y_edges.size - 1) + y_index),
                    "x_left_m": float(x_edges[x_index]),
                    "x_right_m": float(x_edges[x_index + 1]),
                    "y_lower_m": float(y_edges[y_index]),
                    "y_upper_m": float(y_edges[y_index + 1]),
                    "area_m2": float(
                        (x_edges[x_index + 1] - x_edges[x_index])
                        * (y_edges[y_index + 1] - y_edges[y_index])
                    ),
                    "partition_edge_sha256": diagnostics["partition_edge_sha256"],
                }
            )
    return rows
