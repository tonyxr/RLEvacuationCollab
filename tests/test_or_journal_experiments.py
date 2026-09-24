import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from ShelterDatabase import ShelterDS
from or_journal_experiments import (
    DEFAULT_CONFIG,
    _design_counts,
    _extreme_case,
    _read_json,
    confirmatory_design_matrix,
    latency_scaling_models,
    summarize_latency,
)


class ORJournalDesignTests(unittest.TestCase):
    def test_frozen_design_has_five_cities_and_five_population_levels(self):
        design = _read_json(Path(DEFAULT_CONFIG))
        self.assertEqual(len(design["cities"]), 5)
        self.assertEqual(design["population_levels"], [10000, 20000, 30000, 40000, 50000])
        counts = _design_counts(design)
        self.assertEqual(counts["city_population_capacity_cells"], 50)
        self.assertGreater(counts["all_strategy_evaluation_episodes"], 0)
        matrix = confirmatory_design_matrix(design)
        self.assertEqual(len(matrix), 50)
        self.assertEqual(
            sum(row["all_strategy_episodes"] for row in matrix),
            counts["all_strategy_evaluation_episodes"],
        )

    def test_safe_empty_center_is_oracle_but_not_heuristic(self):
        case = _extreme_case(np.random.default_rng(17), "safe_center")
        center = int(case["center"])
        self.assertEqual(float(case["active"][center]), 0.0)
        self.assertEqual(int(case["oracle"]), center)
        self.assertNotEqual(int(case["heuristic"]), center)

    def test_danger_and_feasibility_change_the_extreme_case_oracle(self):
        dangerous = _extreme_case(np.random.default_rng(23), "dangerous_center")
        unavailable = _extreme_case(np.random.default_rng(29), "center_unavailable")
        self.assertNotEqual(int(dangerous["oracle"]), int(dangerous["center"]))
        self.assertFalse(bool(unavailable["mask"][int(unavailable["center"])]))
        self.assertNotEqual(int(unavailable["oracle"]), int(unavailable["center"]))

    def test_latency_summary_and_log_log_model_are_finite(self):
        rows = []
        for scale, latency in ((4, 0.2), (16, 0.8), (64, 3.2)):
            for repetition in range(1, 4):
                rows.append(
                    {
                        "benchmark_family": "grid_dimensionality",
                        "method": "hierarchical_rl",
                        "grid_side": int(np.sqrt(scale)),
                        "cell_count": scale,
                        "candidate_count": scale,
                        "action_count": scale,
                        "directed_grid_edges": scale * 3,
                        "parameter_count": 10,
                        "model_storage_bytes": 40,
                        "input_storage_bytes": scale * 28,
                        "repetition": repetition,
                        "latency_ms": latency,
                    }
                )
        summary = summarize_latency(rows)
        model = latency_scaling_models(summary)["models"][0]
        self.assertAlmostEqual(model["log_log_slope"], 1.0, places=6)
        self.assertTrue(np.isfinite(model["r_squared"]))


class StaticGreedyTests(unittest.TestCase):
    @staticmethod
    def _node(identifier, capacity):
        return SimpleNamespace(
            OSMID=identifier,
            nodeCap=capacity,
            nodeX=0.0,
            nodeY=0.0,
        )

    def test_expected_demand_greedy_uses_marginal_coverage(self):
        database = ShelterDS(candidateVol=0, initVol=0)
        database.shelterCanByCell = [
            [[self._node("low-demand-large", 100)]],
            [[self._node("high-demand", 60)]],
        ]
        database.shelterByCell = [[[]], [[]]]
        installed = database.predeployStaticDemandGreedy(
            demand_by_cell=[10.0, 55.0],
            additions=1,
            cellTracker=object(),
        )
        self.assertEqual(installed, [0])
        self.assertEqual(database.shelterList[0].nodeMapped.OSMID, "high-demand")


if __name__ == "__main__":
    unittest.main()
