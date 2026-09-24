import json
from pathlib import Path
import unittest

import numpy as np

from CAProcessor import CellTracker
from CellPartitioning import (
    EQUAL_AREA,
    NODE_DENSITY_ADAPTIVE,
    build_cell_partition,
    normalize_partition_mode,
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

    def test_mode_aliases_are_canonical_and_unknown_modes_fail(self):
        self.assertEqual(normalize_partition_mode("uniform"), EQUAL_AREA)
        self.assertEqual(normalize_partition_mode("adaptive"), NODE_DENSITY_ADAPTIVE)
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


if __name__ == "__main__":
    unittest.main()
