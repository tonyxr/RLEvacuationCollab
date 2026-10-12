import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from CAProcessor import CellTracker, authoritative_grid_shape
from CellPartitioning import (
    EQUAL_AREA,
    NODE_DENSITY_ADAPTIVE,
    POPULATION_DENSITY_ADAPTIVE,
    build_cell_partition,
    normalize_partition_mode,
    sample_population_origins,
)
from cell_partition_experiment import accuracy_design_matrix, state_college_gate_matrix


class CellPartitioningTests(unittest.TestCase):
    def test_registered_design_contains_both_modes_and_state_college_gate(self):
        path = Path(__file__).resolve().parents[1] / "config" / "cell_partition_experiment.json"
        with path.open("r", encoding="utf-8") as handle:
            design = json.load(handle)
        self.assertEqual(design["partition_modes"], [EQUAL_AREA, NODE_DENSITY_ADAPTIVE])
        gate = design["confirmatory_accuracy_design"]["state_college_gate"]
        self.assertEqual(gate["grid_level"], 8)
        self.assertEqual(gate["planned_training_episodes"], 720)
        matrix = accuracy_design_matrix(design)
        self.assertEqual(len(matrix), 300)
        self.assertEqual(
            sum(row["learned_policy_episodes"] + row["heuristic_episodes"] for row in matrix),
            27000,
        )
        self.assertEqual(len(state_college_gate_matrix(design)), 6)

    def test_equal_area_has_equal_projected_geometry(self):
        x = np.linspace(100.0, 900.0, 101)
        y = np.linspace(50.0, 450.0, 101)
        partition = build_cell_partition(x, y, 4, 5, EQUAL_AREA)
        self.assertEqual(partition["diagnostics"]["grid_shape"], [4, 5])
        self.assertEqual(partition["diagnostics"]["cell_count"], 20)
        np.testing.assert_allclose(np.diff(partition["x_edges"]), 200.0)
        np.testing.assert_allclose(np.diff(partition["y_edges"]), 80.0)
        self.assertAlmostEqual(partition["diagnostics"]["cell_area_cv"], 0.0)

    def test_adaptive_grid_refines_a_dense_core_and_balances_nodes(self):
        rng = np.random.default_rng(20260909)
        samples = 50_000
        dense_x = rng.normal(500.0, 45.0, int(samples * 0.8))
        sparse_x = rng.uniform(0.0, 1000.0, int(samples * 0.2))
        dense_y = rng.normal(500.0, 45.0, int(samples * 0.8))
        sparse_y = rng.uniform(0.0, 1000.0, int(samples * 0.2))
        x = np.clip(np.concatenate((dense_x, sparse_x)), 0.0, 1000.0)
        # Shuffle independently so the joint sample is not artificially diagonal.
        y = np.clip(np.concatenate((dense_y, sparse_y)), 0.0, 1000.0)
        rng.shuffle(y)

        equal = build_cell_partition(x, y, 8, 8, EQUAL_AREA)
        adaptive = build_cell_partition(x, y, 8, 8, NODE_DENSITY_ADAPTIVE)
        adaptive_widths = np.diff(adaptive["x_edges"])
        center_width = adaptive_widths[len(adaptive_widths) // 2]
        self.assertLess(center_width, float(np.mean(adaptive_widths)))
        self.assertLess(
            adaptive["diagnostics"]["node_count_cv"],
            equal["diagnostics"]["node_count_cv"],
        )
        self.assertGreater(adaptive["diagnostics"]["cell_area_max_min_ratio"], 1.0)

    def test_modes_have_same_tensor_size_and_distinct_provenance(self):
        x = np.arange(100, dtype=float)
        y = np.square(np.linspace(0.0, 1.0, 100))
        equal = build_cell_partition(x, y, 6, 6, EQUAL_AREA)
        adaptive = build_cell_partition(x, y, 6, 6, NODE_DENSITY_ADAPTIVE)
        self.assertEqual(equal["diagnostics"]["cell_count"], 36)
        self.assertEqual(adaptive["diagnostics"]["cell_count"], 36)
        self.assertNotEqual(
            equal["diagnostics"]["partition_edge_sha256"],
            adaptive["diagnostics"]["partition_edge_sha256"],
        )

    def test_population_adaptive_grid_uses_people_not_road_node_count(self):
        x = np.linspace(0.0, 1000.0, 1001)
        y = np.linspace(0.0, 1000.0, 1001)
        population = np.ones_like(x)
        population[(x >= 450.0) & (x <= 550.0)] = 100.0
        population[(y >= 450.0) & (y <= 550.0)] *= 100.0

        node_adaptive = build_cell_partition(
            x, y, 8, 8, NODE_DENSITY_ADAPTIVE
        )
        population_adaptive = build_cell_partition(
            x,
            y,
            8,
            8,
            POPULATION_DENSITY_ADAPTIVE,
            population_weights=population,
        )
        node_center_width = np.diff(node_adaptive["x_edges"])[4]
        population_center_width = np.diff(population_adaptive["x_edges"])[4]
        self.assertLess(population_center_width, node_center_width)
        self.assertAlmostEqual(
            population_adaptive["diagnostics"]["population_mass_total"],
            float(population.sum()),
        )
        self.assertGreater(
            population_adaptive["diagnostics"]["cell_area_max_min_ratio"],
            1.0,
        )

    def test_population_origin_plan_is_reproducible_and_conserves_people(self):
        first = sample_population_origins(17, 103, 10, 20260930)
        second = sample_population_origins(17, 103, 10, 20260930)
        self.assertEqual(first["origin_indices"], second["origin_indices"])
        self.assertEqual(first["group_sizes"], [10] * 10 + [3])
        np.testing.assert_array_equal(
            first["node_population"], second["node_population"]
        )
        self.assertEqual(int(first["node_population"].sum()), 103)
        self.assertEqual(len(first["origin_indices"]), 11)

    def test_population_adaptive_requires_valid_population_weights(self):
        coordinates = np.arange(10, dtype=float)
        with self.assertRaisesRegex(ValueError, "population weight"):
            build_cell_partition(
                coordinates,
                coordinates,
                2,
                2,
                POPULATION_DENSITY_ADAPTIVE,
            )

    def test_mode_aliases_are_canonical_and_unknown_modes_fail(self):
        self.assertEqual(normalize_partition_mode("uniform"), EQUAL_AREA)
        self.assertEqual(normalize_partition_mode("adaptive"), NODE_DENSITY_ADAPTIVE)
        self.assertEqual(
            normalize_partition_mode("population_adaptive"),
            POPULATION_DENSITY_ADAPTIVE,
        )
        with self.assertRaises(ValueError):
            normalize_partition_mode("hexagonal")

    def test_cell_tracker_rejects_malformed_custom_edges(self):
        tracker = CellTracker(2, 2)
        with self.assertRaises(ValueError):
            tracker.initialCut(2.0, 2.0, xEdges=[0.0, 1.0], yEdges=[0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            tracker.initialCut(
                2.0,
                2.0,
                xEdges=[0.0, 1.0, 1.0],
                yEdges=[0.0, 1.0, 2.0],
            )

    def test_rectangular_grid_has_one_flattening_and_neighbor_contract(self):
        tracker = CellTracker(2, 3)
        tracker.initialCut(20.0, 30.0)
        self.assertEqual(tuple(tracker.cell_ids()), (
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        ))
        self.assertEqual(
            [tracker.flatten_cell(cell) for cell in tracker.cell_ids()],
            list(range(6)),
        )
        self.assertEqual(tracker.unflatten_cell(4), (1, 1))
        self.assertEqual(set(tracker.getNeighborCells((0, 1))), {
            (0, 0), (0, 2), (1, 0), (1, 1), (1, 2),
        })
        tracker.setCellState((1, 2), 5)
        self.assertEqual(tracker.state_vector().tolist(), [0, 0, 0, 0, 0, 5])
        self.assertEqual(tracker.assert_grid_coherent(), (2, 3))

    def test_same_cell_count_with_different_shape_is_rejected(self):
        tracker = CellTracker(2, 8)
        tracker.initialCut(20.0, 80.0)
        core = SimpleNamespace(
            cellX=4,
            cellY=4,
            cellTracker=tracker,
            hazardDS=None,
        )
        with self.assertRaisesRegex(RuntimeError, "configured=.*tracker"):
            authoritative_grid_shape(core)

    def test_hazard_store_must_share_the_runtime_tracker_instance(self):
        tracker = CellTracker(2, 3)
        tracker.initialCut(20.0, 30.0)
        other = CellTracker(2, 3)
        other.initialCut(20.0, 30.0)
        core = SimpleNamespace(
            cellX=2,
            cellY=3,
            cellTracker=tracker,
            hazardDS=SimpleNamespace(cellTracker=other),
        )
        with self.assertRaisesRegex(RuntimeError, "same CellTracker"):
            authoritative_grid_shape(core)


if __name__ == "__main__":
    unittest.main()
