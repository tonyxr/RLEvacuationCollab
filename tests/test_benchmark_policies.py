import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from DecisionInterface import (
    AccessibilityDeficitHeuristic,
    HazardWeightedDemandHeuristic,
    OutcomeSnapshot,
    RegionalObservation,
    RiskTimeReductionHeuristic,
)
from RLBridge import RLBridge


def observation(*, active, danger, remaining_capacity, mask):
    active = np.asarray(active, dtype=np.float32)
    cell_count = int(active.size)
    return RegionalObservation(
        decision_index=0,
        simulation_time=1,
        horizon=10,
        initial_population=max(1, int(active.sum())),
        remaining_deployments=2,
        maximum_deployments=2,
        maximum_speed=10.0,
        active_by_cell=active,
        mean_speed_by_cell=np.zeros(cell_count, dtype=np.float32),
        danger_by_cell=np.asarray(danger, dtype=np.float32),
        remaining_capacity_by_cell=np.asarray(remaining_capacity, dtype=np.float32),
        deployable_capacity_by_cell=np.ones(cell_count, dtype=np.float32),
        candidate_count_by_cell=np.ones(cell_count, dtype=np.float32),
        action_mask=np.asarray(mask, dtype=bool),
        outcome=OutcomeSnapshot(
            safe_completed=0,
            casualties=0,
            shelter_evacuated=0,
            ordinary_arrivals=0,
            active_population=int(active.sum()),
            risk_mass=float(np.sum(active * (1.0 + np.asarray(danger)))),
        ),
    )


class BenchmarkPolicyTests(unittest.TestCase):
    def test_risk_reduction_score_and_mask(self):
        state = observation(
            active=(10, 8, 1, 1),
            danger=(0, 0, 0, 0),
            remaining_capacity=(0, 0, 0, 0),
            mask=(True, False, True, True),
        )
        state = replace(
            state,
            candidate_risk_time_reduction=np.asarray(
                (0.1, 1.0, 0.8, 0.2), dtype=np.float32
            ),
        )
        decision = RiskTimeReductionHeuristic().select(state)
        self.assertEqual(decision.strategy, "risk_reduction")
        self.assertEqual(decision.action_index, 2)

    def test_hazard_weighted_score_and_mask(self):
        state = observation(
            active=(10, 8, 1, 1),
            danger=(0.0, 0.5, 0.0, 1.0),
            remaining_capacity=(0, 0, 0, 0),
            mask=(True, True, True, False),
        )
        self.assertEqual(
            HazardWeightedDemandHeuristic().select(state).action_index,
            1,
        )

    def test_hazard_weighted_exact_tie_uses_lowest_cell_id(self):
        state = observation(
            active=(10, 5, 0, 0),
            danger=(0.0, 1.0, 0.0, 0.0),
            remaining_capacity=(0, 0, 0, 0),
            mask=(True, True, True, True),
        )
        self.assertEqual(
            HazardWeightedDemandHeuristic().select(state).action_index,
            0,
        )

    def test_accessibility_deficit_uses_shortfall_and_physical_distance(self):
        state = observation(
            active=(0, 8, 0, 7),
            danger=(0, 0, 0, 0),
            remaining_capacity=(100, 0, 0, 0),
            mask=(False, True, False, True),
        )
        centers = ((0, 0), (0, 1), (1, 0), (1, 1))
        self.assertEqual(
            AccessibilityDeficitHeuristic(centers).select(state).action_index,
            3,
        )

    def test_accessibility_deficit_all_served_tie_uses_lowest_feasible_id(self):
        state = observation(
            active=(0, 2, 0, 3),
            danger=(0, 0, 0, 0),
            remaining_capacity=(0, 2, 0, 3),
            mask=(False, True, False, True),
        )
        centers = ((0, 0), (0, 1), (1, 0), (1, 1))
        self.assertEqual(
            AccessibilityDeficitHeuristic(centers).select(state).action_index,
            1,
        )

    def test_accessibility_deficit_rejects_wrong_geometry(self):
        state = observation(
            active=(1, 1, 1, 1),
            danger=(0, 0, 0, 0),
            remaining_capacity=(0, 0, 0, 0),
            mask=(True, True, True, True),
        )
        with self.assertRaisesRegex(ValueError, "same number of cells"):
            AccessibilityDeficitHeuristic(((0, 0),)).select(state)

    def test_bridge_dispatches_dynamic_benchmark_strategies(self):
        candidate_grid = [[[], []], [[], []]]
        candidate_id = 10
        for i in range(2):
            for j in range(2):
                candidate_grid[i][j].append(
                    SimpleNamespace(
                        OSMID=candidate_id,
                        nodeCap=20,
                        nodeX=float(i),
                        nodeY=float(j),
                    )
                )
                candidate_id += 1
        core = SimpleNamespace(
            cellX=2,
            cellY=2,
            maxSpeed=10.0,
            stopTime=11,
            address="test",
            pedDS=SimpleNamespace(
                pedAgentList={
                    1: SimpleNamespace(group_size=20, terminated=False),
                },
                result={"arrival": 0, "evacuated": 0, "casualty": 0},
            ),
            shelterDS=SimpleNamespace(
                shelterList={},
                shelterByCell=[[[], []], [[], []]],
                shelterCanByCell=candidate_grid,
            ),
            cellTracker=SimpleNamespace(
                xEdges=np.asarray((0.0, 1.0, 2.0)),
                yEdges=np.asarray((0.0, 1.0, 2.0)),
            ),
        )
        hazard_state = observation(
            active=(10, 8, 1, 1),
            danger=(0.0, 0.5, 0.0, 0.0),
            remaining_capacity=(0, 0, 0, 0),
            mask=(True, True, True, True),
        )
        hazard_bridge = RLBridge(
            core,
            deployment_strategy="hazard_weighted",
            train_mode=False,
            target_active_shelters=2,
        )
        hazard_decision = hazard_bridge._select_action(hazard_state)
        self.assertEqual(hazard_decision.strategy, "hazard_weighted")
        self.assertEqual(hazard_decision.action_index, 1)

        accessibility_state = observation(
            active=(0, 8, 0, 7),
            danger=(0, 0, 0, 0),
            remaining_capacity=(100, 0, 0, 0),
            mask=(False, True, False, True),
        )
        accessibility_bridge = RLBridge(
            core,
            deployment_strategy="accessibility_deficit",
            train_mode=False,
            target_active_shelters=2,
        )
        accessibility_decision = accessibility_bridge._select_action(accessibility_state)
        self.assertEqual(accessibility_decision.strategy, "accessibility_deficit")
        self.assertEqual(accessibility_decision.action_index, 3)

        risk_state = replace(
            hazard_state,
            candidate_risk_time_reduction=np.asarray(
                (0.1, 0.3, 0.9, 0.2), dtype=np.float32
            ),
        )
        risk_bridge = RLBridge(
            core,
            deployment_strategy="risk_reduction",
            train_mode=False,
            target_active_shelters=2,
        )
        risk_decision = risk_bridge._select_action(risk_state)
        self.assertEqual(risk_decision.strategy, "risk_reduction")
        self.assertEqual(risk_decision.action_index, 2)


if __name__ == "__main__":
    unittest.main()
