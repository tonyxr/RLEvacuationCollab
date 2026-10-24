"""v64 burned-candidate retirement and defer fallback tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import CounterfactualBranch as CB
from DecisionInterface import (
    AccessibilityDeficitHeuristic,
    ActivePopulationHeuristic,
    ForecastGreedyHeuristic,
    HazardWeightedDemandHeuristic,
    ObjectiveMatchedGreedyHeuristic,
    RegionalObservationBuilder,
    RiskTimeReductionHeuristic,
)
from RLBridge import RLBridge
from ShelterDatabase import ShelterDS
from nmcc_testbed import build


def _node(osmid, cell):
    return SimpleNamespace(
        OSMID=int(osmid), nodeCap=100, nodeX=float(cell[0]), nodeY=float(cell[1])
    )


class _Tracker:
    def __init__(self):
        self.state = {(0, 0): 4, (1, 0): 0}

    def getCellState(self, cell):
        return self.state[tuple(cell)]


def _defer_only_observation():
    return SimpleNamespace(
        action_mask=np.asarray([True, False, False], dtype=bool),
        defer_action_index=0,
        number_of_actions=3,
        number_of_cells=2,
    )


class V64CandidateRetirementTests(unittest.TestCase):
    def test_retirement_is_idempotent_and_preserves_identity(self):
        burned = _node(11, (0, 0))
        safe = _node(22, (1, 0))
        shelters = ShelterDS(2, 0)
        shelters.currentSimulationTime = 7
        shelters.shelterCanByCell = [[[burned]], [[safe]]]
        shelters.shelterByCell = [[[]], [[]]]
        self.assertEqual(shelters.retireOverrunCandidates(_Tracker()), 1)
        self.assertEqual(shelters.shelterCanByCell[0][0], [])
        self.assertEqual(shelters.remainingCandidateCount(), 1)
        self.assertEqual(shelters.retiredCandidates, [(7, (0, 0), "11")])
        self.assertEqual(shelters.retireOverrunCandidates(_Tracker()), 0)

    def test_retired_cell_cannot_be_installed(self):
        shelters = ShelterDS(1, 0)
        shelters.shelterCanByCell = [[[_node(11, (0, 0))]], [[]]]
        shelters.shelterByCell = [[[]], [[]]]
        tracker = _Tracker()
        shelters.retireOverrunCandidates(tracker)
        self.assertIsNone(shelters.newShelter({"cell": (0, 0)}, tracker))

    def test_builder_defers_at_deadline_when_every_install_is_burned(self):
        core = build(stop_time=10, population=20, group_size=10, candidate_count=8)
        core.actionSpace = "region_defer"
        builder = RegionalObservationBuilder(
            core, initial_population=20, horizon=9, maximum_deployments=1
        )
        candidate_cells = {
            (i, j)
            for i, row in enumerate(core.shelterDS.shelterCanByCell)
            for j, candidates in enumerate(row)
            if candidates
        }
        for cell in candidate_cells:
            core.cellTracker.setCellState(cell, 4)
            core.cellTracker.dangerLevelByCell[
                core.cellTracker.flatten_cell(cell)
            ] = 0.8
        closure = core.pedDS.resolveHazardShelterClosures(simulation_time=0)
        self.assertGreater(closure["retired_candidates"], 0)
        observation = builder.build(
            decision_index=0, simulation_time=8, remaining_deployments=1
        )
        self.assertTrue(observation.action_mask[0])
        self.assertFalse(observation.action_mask[1:].any())
        self.assertTrue(np.all(observation.candidate_capacities[1:] == 0.0))

    def test_every_simple_heuristic_returns_the_only_feasible_defer(self):
        observation = _defer_only_observation()
        policies = (
            ActivePopulationHeuristic(),
            RiskTimeReductionHeuristic(),
            ForecastGreedyHeuristic(),
            HazardWeightedDemandHeuristic(),
            AccessibilityDeficitHeuristic(np.zeros((2, 2))),
        )
        for policy in policies:
            self.assertEqual(policy.select(observation).action_index, 0)
        objective = ObjectiveMatchedGreedyHeuristic()
        self.assertEqual(
            objective.select(
                observation,
                candidate_values=np.asarray([-np.inf, -np.inf, -np.inf]),
            ).action_index,
            0,
        )

    def test_rollout_grader_skips_defer_only_state(self):
        bridge = object.__new__(RLBridge)
        self.assertIsNone(
            RLBridge._collect_rollout_policy_improvement(
                bridge,
                _defer_only_observation(),
                SimpleNamespace(),
                SimpleNamespace(action_index=0),
                simulation_time=8,
            )
        )

    def test_digest_covers_retirement_ledger(self):
        core = build(population=20, group_size=10)
        before = CB.live_state_digest(core)
        core.shelterDS.retiredCandidateCount += 1
        self.assertNotEqual(CB.live_state_digest(core), before)


if __name__ == "__main__":
    unittest.main()
