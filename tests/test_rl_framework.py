import csv
import gzip
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

from CAProcessor import CellTracker
from Core import Core
from DecisionInterface import (
    AccessibilityDeficitHeuristic,
    ActivePopulationHeuristic,
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    INFRA_FEATURE_NAMES,
    PED_FEATURE_NAMES,
    HazardWeightedDemandHeuristic,
    OutcomeSnapshot,
    RegionalObservation,
    RegionalObservationBuilder,
    RegionalShelterExecutor,
    UniformRegionalPolicy,
)
from EvacuationVisualizer import EvacuationVisualizer, resolve_milestones
from GNN import EvacPolicy, fit_gnn, grid_edge_index
from HazardDatabase import HazardDS
from NetworkOptimization import distances_to_target, routing_tree_to_target
from PedestrianDatabase import PedDS
from RewardProcessor import RewardProcessor
from RLBridge import RLBridge
from Shelter import Shelter
from ShelterDatabase import ShelterDS
from TrainingLogger import CSV_COLUMNS, trainingLog
from backtest import (
    _benchmark_analysis,
    _evaluation_metric,
    _paired_analysis,
    _paired_randomization_pvalue,
    _performance_assessment,
    _json_dump,
    _training_convergence,
    _verify_matched_interface,
    _write_paper_table,
)


def make_observation(
    active=(4, 8, 3, 1),
    mask=(True, True, True, False),
    danger=(0.1, 0.2, 0.3, 0.4),
    remaining_capacity=(0, 5, 0, 0),
    **outcome,
):
    active_array = np.asarray(active, dtype=np.float32)
    population = int(active_array.sum()) + int(outcome.get("safe_completed", 0)) + int(
        outcome.get("casualties", 0)
    )
    return RegionalObservation(
        decision_index=0,
        simulation_time=1,
        horizon=10,
        initial_population=max(1, population),
        remaining_deployments=2,
        maximum_deployments=2,
        maximum_speed=10.0,
        active_by_cell=active_array,
        mean_speed_by_cell=np.asarray((2, 3, 4, 5), dtype=np.float32),
        danger_by_cell=np.asarray(danger, dtype=np.float32),
        remaining_capacity_by_cell=np.asarray(remaining_capacity, dtype=np.float32),
        deployable_capacity_by_cell=np.asarray((10, 20, 30, 0), dtype=np.float32),
        candidate_count_by_cell=np.asarray((1, 2, 1, 0), dtype=np.float32),
        action_mask=np.asarray(mask, dtype=bool),
        outcome=OutcomeSnapshot(
            safe_completed=int(outcome.get("safe_completed", 0)),
            casualties=int(outcome.get("casualties", 0)),
            shelter_evacuated=int(outcome.get("shelter_evacuated", 0)),
            ordinary_arrivals=int(outcome.get("ordinary_arrivals", 0)),
            active_population=int(active_array.sum()),
            risk_mass=float(
                np.sum(active_array * (1.0 + np.asarray(danger, dtype=np.float32)))
            ),
        ),
    )


class RewardTests(unittest.TestCase):
    def test_exact_paper_equation(self):
        before = OutcomeSnapshot(2, 1, 2, 0, 7, 9.0)
        after = OutcomeSnapshot(4, 2, 3, 1, 4, 5.0)
        reward = RewardProcessor(casualty_weight=3.0).evaluate(
            before=before,
            after=after,
            active_person_time=12.0,
            hazard_exposure_person_time=8.0,
            initial_population=10,
            horizon=10,
        )
        self.assertAlmostEqual(reward.safe_completion_reward, 0.2)
        self.assertAlmostEqual(reward.casualty_penalty, -0.3)
        self.assertAlmostEqual(reward.evacuation_time_penalty, -0.12)
        self.assertAlmostEqual(reward.hazard_exposure_penalty, -0.08)
        self.assertAlmostEqual(reward.risk_time_penalty, -0.2)
        self.assertAlmostEqual(reward.total, -0.3)

    def test_safe_arrivals_are_rewarded_without_site_specific_shaping(self):
        before = OutcomeSnapshot(2, 0, 1, 1, 8, 8.0)
        after = OutcomeSnapshot(4, 0, 3, 1, 6, 6.0)
        reward = RewardProcessor().evaluate(
            before=before,
            after=after,
            active_person_time=0.0,
            hazard_exposure_person_time=0.0,
            initial_population=10,
            horizon=10,
        )
        self.assertAlmostEqual(reward.safe_completion_reward, 0.2)
        self.assertAlmostEqual(reward.shelter_service_reward, 0.0)
        self.assertAlmostEqual(reward.total, 0.2)

    def test_casualty_weight_is_strictly_above_maximum_avoided_exposure(self):
        with self.assertRaises(ValueError):
            RewardProcessor(casualty_weight=2.0)
        self.assertEqual(RewardProcessor().casualty_weight, 3.0)

    def test_outcomes_must_be_monotone(self):
        processor = RewardProcessor()
        with self.assertRaisesRegex(ValueError, "monotone"):
            processor.evaluate(
                before=OutcomeSnapshot(2, 0, 2, 0, 1, 1.0),
                after=OutcomeSnapshot(1, 0, 1, 0, 2, 2.0),
                active_person_time=0.0,
                hazard_exposure_person_time=0.0,
                initial_population=3,
                horizon=5,
            )


class HazardCouplingTests(unittest.TestCase):
    def test_percent_mean_and_variance_are_converted_to_probabilities(self):
        hazards = HazardDS(1, [10, 0], [1, 0], [20, 0])
        self.assertEqual(hazards.sampleProbability(hazards.casualtyRate), 0.1)
        self.assertEqual(hazards.sampleProbability(hazards.spreadRate), 0.01)
        self.assertEqual(hazards.sampleProbability(hazards.speedReduct), 0.2)
        with self.assertRaises(ValueError):
            HazardDS(1, [101, 0], [1, 0], [20, 0])
        with self.assertRaises(ValueError):
            HazardDS(1, [10, -1], [1, 0], [20, 0])

    def test_hazard_state_is_the_bounded_decision_danger(self):
        tracker = CellTracker(2, 1)
        tracker.initialCut(2.0, 1.0)
        tracker.setCellState((0, 0), 1)
        tracker.setCellState((1, 0), 5)
        tracker.cellUpdate()
        np.testing.assert_allclose(tracker.dangerLevelByCell, (0.2, 1.0))

    def test_overlapping_hazard_effects_use_configured_rates(self):
        store = PedDS(1)
        store.hazardDS = SimpleNamespace(
            hazardList={
                0: SimpleNamespace(
                    active=True,
                    impactedCells=[(0, 0)],
                    speedReduct=0.1,
                    casualtyRate=0.1,
                ),
                1: SimpleNamespace(
                    active=True,
                    impactedCells=[[0, 0]],
                    speedReduct=0.2,
                    casualtyRate=0.2,
                ),
            }
        )
        speed_reduction, casualty_probability, exposed = store._hazard_effects((0, 0), 5)
        self.assertTrue(exposed)
        self.assertAlmostEqual(speed_reduction, 0.28)
        expected_one_minute = 1.0 - (
            (1.0 - 0.1) ** (1.0 / 60.0)
            * (1.0 - 0.2) ** (1.0 / 60.0)
        )
        self.assertAlmostEqual(casualty_probability, expected_one_minute)
        self.assertAlmostEqual(1.0 - (1.0 - casualty_probability) ** 60, 0.28)

    def test_casualty_probability_is_timestep_invariant_and_level_gated(self):
        def one_step(minutes, level):
            store = PedDS(1)
            store.timeStepMinutes = float(minutes)
            store.casualty_reference_exposure_minutes = 60.0
            store.hazardDS = SimpleNamespace(
                hazardList={
                    0: SimpleNamespace(
                        active=True,
                        impactedCells=[(0, 0)],
                        speedReduct=0.0,
                        casualtyRate=0.1,
                    )
                }
            )
            return store._hazard_effects((0, 0), level)[1]

        self.assertEqual(one_step(1.0, 3), 0.0)
        one_minute = one_step(1.0, 5)
        half_minute = one_step(0.5, 5)
        self.assertAlmostEqual(1.0 - (1.0 - one_minute) ** 60, 0.1)
        self.assertAlmostEqual(1.0 - (1.0 - half_minute) ** 120, 0.1)

    def test_hazard_rng_is_independent_of_global_numpy_consumption(self):
        first = HazardDS(
            0,
            [10, 4],
            [1, 2],
            [20, 5],
            rng=np.random.default_rng(71),
        )
        second = HazardDS(
            0,
            [10, 4],
            [1, 2],
            [20, 5],
            rng=np.random.default_rng(71),
        )
        first_draw = first.sampleProbability(first.casualtyRate, first.rng)
        np.random.random(1000)
        second_draw = second.sampleProbability(second.casualtyRate, second.rng)
        self.assertEqual(first_draw, second_draw)

    def test_forecast_is_higher_downwind_than_upwind(self):
        tracker = CellTracker(3, 1)
        tracker.initialCut(3.0, 1.0)
        tracker.setCellState((1, 0), 5)
        tracker.cellUpdate()
        hazards = HazardDS(
            0,
            [10, 0],
            [50, 0],
            [20, 0],
            wind_speed_m_per_minute=1.0,
            wind_direction_degrees=0.0,
            wind_influence=1.0,
        )
        hazards.setCellTracker(tracker)
        hazards.hazardList = {
            0: SimpleNamespace(active=True, spreadRate=0.5),
        }
        forecast = hazards.forecast_danger_by_cell(1)
        self.assertGreater(float(forecast[2]), float(forecast[0]))

    def test_pedestrian_hazard_shock_is_keyed_by_scenario_time_and_person(self):
        store = PedDS(0)
        store.set_hazard_random_seed(90210)
        store.currTime = 7
        expected = store._hazard_uniform(12)
        np.random.random(1000)
        self.assertEqual(store._hazard_uniform(12), expected)
        self.assertNotEqual(store._hazard_uniform(13), expected)
        store.currTime = 8
        self.assertNotEqual(store._hazard_uniform(12), expected)

    def test_hazard_lifecycle_removes_lethality_at_expiry(self):
        tracker = CellTracker(1, 1)
        tracker.initialCut(1.0, 1.0)
        tracker.setCellState((0, 0), 5)
        hazards = HazardDS(0, [10, 0], [1, 0], [20, 0])
        hazards.setCellTracker(tracker)
        hazard = SimpleNamespace(
            active=True,
            age=0,
            lifespan=1,
            impactedCells=[(0, 0)],
        )
        hazards.hazardList = {0: hazard}
        self.assertEqual(hazards.terminateHazard(), 1)
        self.assertFalse(hazard.active)
        self.assertEqual(tracker.getCellState((0, 0)), 2)


class SharedDecisionInterfaceTests(unittest.TestCase):
    def test_active_population_benchmark_uses_only_feasible_cells(self):
        observation = make_observation(active=(100, 8, 3, 1), mask=(False, True, True, False))
        decision = ActivePopulationHeuristic().select(observation)
        self.assertEqual(decision.action_index, 1)

    def test_active_population_ties_use_cell_id(self):
        observation = make_observation(active=(4, 8, 8, 1))
        self.assertEqual(ActivePopulationHeuristic().select(observation).action_index, 1)

    def test_hazard_weighted_demand_can_prioritize_smaller_exposed_population(self):
        observation = make_observation(
            active=(10, 8, 1, 1),
            danger=(0.0, 0.5, 0.0, 0.0),
        )
        decision = HazardWeightedDemandHeuristic().select(observation)
        self.assertEqual(decision.action_index, 1)

    def test_hazard_weighted_demand_obeys_feasibility_mask(self):
        observation = make_observation(
            active=(10, 8, 1, 1),
            danger=(0.0, 1.0, 0.0, 0.0),
            mask=(True, False, True, False),
        )
        decision = HazardWeightedDemandHeuristic().select(observation)
        self.assertEqual(decision.action_index, 0)

    def test_accessibility_deficit_balances_capacity_shortfall_and_distance(self):
        observation = make_observation(
            active=(0, 8, 0, 7),
            remaining_capacity=(100, 0, 0, 0),
            mask=(False, True, False, True),
        )
        centers = ((0, 0), (0, 1), (1, 0), (1, 1))
        decision = AccessibilityDeficitHeuristic(centers).select(observation)
        self.assertEqual(decision.action_index, 3)

    def test_accessibility_deficit_accounts_for_local_remaining_capacity(self):
        observation = make_observation(
            active=(0, 10, 0, 8),
            remaining_capacity=(100, 10, 0, 0),
            mask=(False, True, False, True),
        )
        centers = ((0, 0), (0, 1), (1, 0), (1, 1))
        decision = AccessibilityDeficitHeuristic(centers).select(observation)
        self.assertEqual(decision.action_index, 3)

    def test_random_and_heuristic_consume_the_same_observation_type_and_space(self):
        observation = make_observation()
        heuristic_action = ActivePopulationHeuristic().select(observation).action_index
        hazard_action = HazardWeightedDemandHeuristic().select(observation).action_index
        accessibility_action = AccessibilityDeficitHeuristic(
            ((0, 0), (0, 1), (1, 0), (1, 1))
        ).select(observation).action_index
        random_action = UniformRegionalPolicy(7).select(observation).action_index
        self.assertTrue(observation.action_mask[heuristic_action])
        self.assertTrue(observation.action_mask[hazard_action])
        self.assertTrue(observation.action_mask[accessibility_action])
        self.assertTrue(observation.action_mask[random_action])
        self.assertLess(heuristic_action, observation.number_of_cells)
        self.assertLess(hazard_action, observation.number_of_cells)
        self.assertLess(accessibility_action, observation.number_of_cells)
        self.assertLess(random_action, observation.number_of_cells)

    def test_features_have_fixed_scales_and_shapes(self):
        cells, global_features = make_observation().policy_features()
        candidates = make_observation().candidate_features()
        self.assertEqual(cells.shape, (4, len(CELL_FEATURE_NAMES)))
        self.assertEqual(global_features.shape, (len(GLOBAL_FEATURE_NAMES),))
        self.assertEqual(candidates.shape, (4, len(CANDIDATE_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(cells).all())
        self.assertTrue(np.isfinite(global_features).all())
        self.assertTrue(np.logical_and(cells >= 0.0, cells <= 1.0).all())
        self.assertTrue(np.logical_and(global_features >= 0.0, global_features <= 1.0).all())
        self.assertTrue(np.logical_and(candidates >= 0.0, candidates <= 1.0).all())

    def test_features_expose_mobility_delay_and_available_capacity(self):
        observation = make_observation(
            active=(4, 8, 3, 1),
            remaining_capacity=(0, 5, 0, 0),
        )
        cells, _ = observation.policy_features()
        np.testing.assert_allclose(
            cells[:, CELL_FEATURE_NAMES.index("mobility_delay_fraction")],
            (0.8, 0.7, 0.6, 0.5),
        )
        np.testing.assert_allclose(
            cells[:, CELL_FEATURE_NAMES.index("remaining_shelter_capacity_fraction")],
            (0.0, 0.3125, 0.0, 0.0),
        )


class FakeCellTracker:
    def __init__(self):
        self.dangerLevelByCell = np.zeros(4, dtype=float)
        self.added = []

    def addShelter(self, cell, shelter):
        self.added.append((tuple(cell), shelter))


class FakePedestrianStore:
    def __init__(self):
        self.pedAgentList = {
            0: SimpleNamespace(currCell=(0, 0), currSpeed=2.0, terminated=False, group_size=1),
            1: SimpleNamespace(currCell=(0, 1), currSpeed=4.0, terminated=False, group_size=1),
        }
        self.result = {"arrival": 0, "evacuated": 0, "casualty": 0}

    def reroute_to_new_shelter_if_closer(self, shelter):
        return len(self.pedAgentList)


def make_core():
    shelter_store = ShelterDS(candidateVol=4, initVol=0)
    shelter_store.shelterByCell = [[[], []], [[], []]]
    shelter_store.shelterCanByCell = [[[], []], [[], []]]
    node_id = 100
    capacities = ((10, 20), (30, 40))
    for i in range(2):
        for j in range(2):
            capacity = capacities[i][j]
            shelter_store.shelterCanByCell[i][j].append(
                SimpleNamespace(OSMID=node_id, nodeCap=capacity, nodeX=float(i), nodeY=float(j))
            )
            node_id += 1
    return SimpleNamespace(
        cellX=2,
        cellY=2,
        maxSpeed=10.0,
        stopTime=11,
        maximumShelterForecastDanger=0.6,
        address="test",
        pedDS=FakePedestrianStore(),
        shelterDS=shelter_store,
        cellTracker=FakeCellTracker(),
    )


class ObservationAndExecutionTests(unittest.TestCase):
    def test_builder_reads_authoritative_pedestrian_and_shelter_stores(self):
        core = make_core()
        core.hazardVol = 3
        core.panicRate = 0.5
        core.shelterCapacityToken = 1
        builder = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        )
        observation = builder.build(decision_index=0, simulation_time=1, remaining_deployments=2)
        np.testing.assert_array_equal(observation.active_by_cell, (1, 1, 0, 0))
        np.testing.assert_array_equal(observation.deployable_capacity_by_cell, (10, 20, 30, 40))
        np.testing.assert_array_equal(observation.action_mask, (True, True, True, True))
        np.testing.assert_allclose(observation.candidate_east_positions, (0, 0, 1, 1))
        np.testing.assert_allclose(observation.candidate_north_positions, (0, 1, 0, 1))
        _, global_features = observation.policy_features()
        self.assertAlmostEqual(
            float(global_features[GLOBAL_FEATURE_NAMES.index("hazard_instance_fraction")]),
            0.75,
        )
        self.assertAlmostEqual(
            float(global_features[GLOBAL_FEATURE_NAMES.index("configured_panic_fraction")]),
            0.5,
        )
        self.assertAlmostEqual(
            float(global_features[GLOBAL_FEATURE_NAMES.index("deployment_capacity_coverage_fraction")]),
            1.0,
        )
        self.assertEqual(
            observation.candidate_features().shape,
            (4, len(CANDIDATE_FEATURE_NAMES)),
        )

    def test_equal_capacity_token_is_visible_and_enforced_for_every_cell(self):
        core = make_core()
        core.shelterDS.deploymentCapacityToken = 25
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        np.testing.assert_array_equal(
            observation.deployable_capacity_by_cell,
            (0, 0, 25, 25),
        )
        np.testing.assert_array_equal(
            observation.action_mask,
            (False, False, True, True),
        )
        first_receipt = RegionalShelterExecutor(core).execute(
            observation,
            SimpleNamespace(action_index=3),
        )
        self.assertEqual(first_receipt.capacity_added, 25.0)
        self.assertEqual(
            core.shelterDS.shelterList[first_receipt.shelter_id].shelterCap,
            25,
        )

        second_core = make_core()
        second_core.shelterDS.deploymentCapacityToken = 25
        second_observation = RegionalObservationBuilder(
            second_core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        second_receipt = RegionalShelterExecutor(second_core).execute(
            second_observation,
            SimpleNamespace(action_index=2),
        )
        self.assertEqual(second_receipt.capacity_added, 25.0)
        self.assertNotEqual(first_receipt.executed_cell, second_receipt.executed_cell)

    def test_cell_with_multiple_equal_capacity_sites_still_yields_one_slot(self):
        """Under the cell-priority action space a cell contributes exactly one
        action slot no matter how many raw candidates it holds. When two
        candidates tie on capacity, the slot exposes whichever one the shared
        deterministic rule (``ShelterDatabase._candidate_index``: max
        capacity, then OSM identifier) would install -- here the original
        OSMID "100" beats the appended OSMID "999" on the identifier
        tie-break."""
        core = make_core()
        core.shelterDS.shelterCanByCell[0][0].append(
            SimpleNamespace(OSMID=999, nodeCap=10, nodeX=0.25, nodeY=0.75)
        )
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        self.assertEqual(observation.number_of_actions, 4)
        self.assertEqual(observation.candidate_osm_node_ids[0], "100")
        self.assertEqual(observation.candidate_east_positions[0], 0.0)
        self.assertEqual(observation.candidate_north_positions[0], 0.0)

    def test_builder_reads_dynamic_shelter_fulfillment(self):
        core = make_core()
        node = SimpleNamespace(OSMID=77, nodeCap=10, nodeX=0.0, nodeY=0.0)
        shelter = Shelter(0, node, (0, 0), 10, 6, 0)
        core.shelterDS.shelterList = {0: shelter}
        core.shelterDS.shelterByCell[0][0].append(shelter)
        builder = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        )
        observation = builder.build(
            decision_index=0,
            simulation_time=1,
            remaining_deployments=2,
        )
        self.assertAlmostEqual(observation.remaining_capacity_by_cell[0], 4.0)
        self.assertAlmostEqual(observation.shelter_utilization_by_cell[0], 0.6)

    def test_builder_aggregates_routes_and_wellness_without_person_nodes(self):
        core = make_core()
        target = SimpleNamespace(cellID=(1, 1), nodeX=1.0, nodeY=1.0)
        core.pedDS.pedAgentList[0].routeFollowing = SimpleNamespace(
            edgeRemained=[SimpleNamespace(edgeLen=100.0)],
            endNode=target,
        )
        core.pedDS.pedAgentList[0].affected = True
        core.pedDS.pedAgentList[0].panicked = False
        core.pedDS.pedAgentList[1].affected = True
        core.pedDS.pedAgentList[1].panicked = True
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        self.assertEqual(observation.exposed_wellness_by_cell[0], 1.0)
        self.assertEqual(observation.panicked_wellness_by_cell[1], 1.0)
        self.assertEqual(observation.long_route_share_by_cell[0], 1.0)
        self.assertEqual(observation.route_edge_index.shape, (2, 2))
        self.assertEqual(
            set(map(tuple, observation.route_edge_index.T.tolist())),
            {(0, 3), (3, 0)},
        )
        np.testing.assert_allclose(observation.route_edge_weight, (0.5, 0.5))

    def test_forecast_unsafe_candidates_are_removed_from_every_policy_mask(self):
        core = make_core()
        core.cellTracker.dangerLevelByCell[1] = 0.8
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        np.testing.assert_array_equal(
            observation.action_mask,
            (True, False, True, True),
        )

    def test_active_population_tie_uses_stable_high_capacity_candidate_order(self):
        core = make_core()
        extra = SimpleNamespace(OSMID=999, nodeCap=50, nodeX=0.0, nodeY=0.0)
        core.shelterDS.shelterCanByCell[0][0].append(extra)
        builder = RegionalObservationBuilder(core, initial_population=2, horizon=10, maximum_deployments=2)
        observation = builder.build(decision_index=0, simulation_time=1, remaining_deployments=2)
        decision = ActivePopulationHeuristic().select(observation)
        receipt = RegionalShelterExecutor(core).execute(observation, decision)
        self.assertEqual(decision.action_index, 0)
        self.assertEqual(receipt.capacity_added, 50.0)
        self.assertEqual(receipt.executed_candidate, decision.action_index)
        self.assertEqual(receipt.executed_cell, 0)
        self.assertEqual(core.cellTracker.added[0][0], (0, 0))

    def test_executor_always_installs_the_shared_deterministic_rule_winner(self):
        """Choosing a cell can never choose a specific building: whichever
        site the shared deterministic rule resolves for that cell (here the
        higher-capacity appended candidate, OSMID 999) is what gets
        installed, and the observation already predicted it before the
        action was taken."""
        core = make_core()
        extra = SimpleNamespace(OSMID=999, nodeCap=50, nodeX=0.0, nodeY=0.0)
        core.shelterDS.shelterCanByCell[0][0].append(extra)
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        self.assertEqual(observation.number_of_actions, 4)
        self.assertEqual(observation.candidate_osm_node_ids[0], "999")
        receipt = RegionalShelterExecutor(core).execute(
            observation,
            SimpleNamespace(action_index=0),
        )
        self.assertEqual(receipt.executed_cell, 0)
        self.assertEqual(receipt.candidate_osm_node_id, "999")
        self.assertEqual(receipt.capacity_added, 50.0)

    def test_executor_rejects_a_site_that_diverges_from_the_prediction(self):
        """Fail-closed invariant: if whatever installs the cell ever returns a
        building other than the one the observation predicted for that slot,
        the executor raises instead of returning a silently-wrong receipt."""
        core = make_core()
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        wrong_node = SimpleNamespace(OSMID=123456, nodeCap=999.0, nodeX=0.0, nodeY=0.0)

        def divergent_new_shelter(action, cellTracker):
            del cellTracker
            sid = core.shelterDS.allocID()
            shelter = Shelter(sid, wrong_node, tuple(action["cell"]), 999, 0, 0)
            core.shelterDS.shelterList[sid] = shelter
            return sid

        core.shelterDS.newShelter = divergent_new_shelter
        with self.assertRaisesRegex(RuntimeError, "diverged"):
            RegionalShelterExecutor(core).execute(
                observation,
                SimpleNamespace(action_index=0),
            )

    def test_empty_cell_yields_an_infeasible_placeholder_slot(self):
        """A cell with no remaining candidate keeps its slot in the fixed,
        cell-indexed action table (so the table size never changes across an
        episode) but the slot is masked infeasible and carries zero
        capacity."""
        core = make_core()
        core.shelterDS.shelterCanByCell[1][1] = []
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        empty_cell_index = 1 * int(core.cellY) + 1
        self.assertEqual(observation.number_of_actions, 4)
        self.assertFalse(bool(observation.action_mask[empty_cell_index]))
        self.assertEqual(observation.candidate_capacities[empty_cell_index], 0.0)
        self.assertTrue(
            observation.candidate_osm_node_ids[empty_cell_index].startswith("empty-cell-")
        )

    def test_every_benchmark_policy_and_rl_resolve_the_identical_site_per_cell(self):
        """RL and every heuristic benchmark share the exact same lower-level
        site-selection rule: whichever cell a policy names, the building
        that gets installed is determined entirely by
        ``ShelterDatabase._candidate_index`` and never by which policy chose
        the cell. This checks the observation's predicted site for whichever
        cell each policy picks against an independent call into that shared
        rule."""
        core = make_core()
        core.shelterDS.shelterCanByCell[0][0].append(
            SimpleNamespace(OSMID=999, nodeCap=15, nodeX=0.1, nodeY=0.1)
        )
        observation = RegionalObservationBuilder(
            core,
            initial_population=2,
            horizon=10,
            maximum_deployments=2,
        ).build(decision_index=0, simulation_time=1, remaining_deployments=2)
        cell_centers = np.column_stack(
            [observation.region_east_positions, observation.region_north_positions]
        )
        policies = (
            ActivePopulationHeuristic(),
            HazardWeightedDemandHeuristic(),
            AccessibilityDeficitHeuristic(cell_centers),
            UniformRegionalPolicy(seed=0),
        )
        for policy in policies:
            decision = policy.select(observation)
            action = int(decision.action_index)
            cell_index = int(observation.candidate_cell_indices[action])
            cell = divmod(cell_index, int(core.cellY))
            expected = core.shelterDS.previewShelterCandidate(cell, core.cellTracker)
            expected_osm_id = str(getattr(expected, "OSMID", ""))
            self.assertEqual(
                observation.candidate_osm_node_ids[action],
                expected_osm_id,
                msg=f"policy {policy.name!r} disagreed with the shared site rule",
            )

    def test_reverse_target_search_matches_directed_shortest_path_distance(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 2, length=3.0)
        graph.add_edge(2, 3, length=4.0)
        to_three = distances_to_target(graph, 3)
        to_one = distances_to_target(graph, 1)
        self.assertEqual(to_three[1], 7.0)
        self.assertNotIn(3, to_one)

    def test_reverse_routing_tree_returns_deterministic_next_hops(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 2, length=3.0)
        graph.add_edge(2, 4, length=4.0)
        graph.add_edge(1, 3, length=3.0)
        graph.add_edge(3, 4, length=4.0)
        distances, next_hop = routing_tree_to_target(graph, 4)
        self.assertEqual(distances[1], 7.0)
        self.assertEqual(next_hop[1], 2)
        self.assertEqual(next_hop[2], 4)


class PolicyTests(unittest.TestCase):
    def test_behavior_policy_has_no_dropout_and_entropy_has_common_scale(self):
        policy = EvacPolicy(2, 1, 2, d_global=1, d_candidate=1)
        self.assertFalse(
            any(isinstance(module, torch.nn.Dropout) for module in policy.modules())
        )
        for action_count in (2, 5):
            logits = torch.zeros(1, action_count)
            distribution = torch.distributions.Categorical(logits=logits)
            mask = torch.ones(1, action_count, dtype=torch.bool)
            entropy = RLBridge._normalized_categorical_entropy(
                distribution,
                mask,
            )
            torch.testing.assert_close(entropy, torch.ones(1))
        one_action = torch.distributions.Categorical(logits=torch.zeros(1, 1))
        torch.testing.assert_close(
            RLBridge._normalized_categorical_entropy(
                one_action,
                torch.ones(1, 1, dtype=torch.bool),
            ),
            torch.zeros(1),
        )

    def test_policy_emits_one_logit_per_exact_candidate(self):
        policy = EvacPolicy(2, 1, 1, d_global=3)
        graph = fit_gnn(
            torch.randn(4, 2),
            torch.randn(4, 1),
            torch.randn(4, 1),
            torch.randn(3),
            candidate_cell_index=torch.tensor([0, 0, 1, 2, 3]),
            candidate_features=torch.rand(5, 1),
        )
        logits, values = policy(graph)
        self.assertEqual(tuple(logits.shape), (1, 5))
        self.assertEqual(tuple(values.shape), (1,))

    def test_factorized_critic_components_sum_to_scalar_value(self):
        policy = EvacPolicy(2, 1, 1, d_global=3, d_momentum=9)
        graph = fit_gnn(
            torch.randn(4, 2),
            torch.randn(4, 1),
            torch.randn(4, 1),
            torch.randn(3),
        )
        _, scalar, components, _, state = policy.forward_recurrent(
            graph,
            None,
            torch.zeros(9),
        )
        self.assertEqual(tuple(components.shape), (1, 4))
        self.assertEqual(tuple(state[0].shape), (1, policy.temporal_dim))
        torch.testing.assert_close(scalar, components.sum(dim=-1))

    def test_nmcc_heads_are_factored_bounded_and_action_local(self):
        torch.manual_seed(103)
        policy = EvacPolicy(
            2,
            1,
            1,
            d_global=3,
            d_candidate=2,
            d_momentum=9,
            nmcc_ensemble_size=3,
        )
        common = dict(
            x_ped=torch.randn(4, 2),
            x_hazard=torch.randn(4, 1),
            x_infra=torch.randn(4, 1),
            x_global=torch.randn(3),
            candidate_cell_index=torch.tensor([0, 1, 3]),
        )
        first = fit_gnn(
            **common,
            candidate_features=torch.zeros(3, 2),
        )
        second = fit_gnn(
            **common,
            candidate_features=torch.ones(3, 2),
        )
        first_output = policy.forward_nmcc_recurrent(
            first,
            None,
            torch.zeros(9),
        )
        second_output = policy.forward_nmcc_recurrent(
            second,
            None,
            torch.zeros(9),
        )
        self.assertEqual(tuple(first_output.natural_outcomes.shape), (1, 6))
        self.assertEqual(
            tuple(first_output.causal_outcome_samples.shape),
            (1, 3, 3, 6),
        )
        self.assertEqual(
            tuple(first_output.causal_component_mean.shape),
            (1, 3, 4),
        )
        self.assertEqual(tuple(first_output.robust_causal_score.shape), (1, 3))
        self.assertTrue(torch.all(first_output.natural_outcomes >= 0.0))
        self.assertTrue(torch.all(first_output.natural_outcomes <= 1.0))
        active_fraction = common["x_ped"][:, 0].sum().clamp(0.0, 1.0)
        torch.testing.assert_close(
            first_output.natural_outcomes[0, [0, 1, 4]].sum(),
            active_fraction,
        )
        self.assertLessEqual(
            float(first_output.natural_outcomes[0, 5]),
            float(first_output.natural_outcomes[0, 4]) + 1e-7,
        )
        self.assertTrue(torch.all(first_output.causal_outcome_samples >= -1.0))
        self.assertTrue(torch.all(first_output.causal_outcome_samples <= 1.0))
        # Candidate/site features may alter intervention effects, but cannot
        # leak into the action-independent natural-momentum forecast.
        torch.testing.assert_close(
            first_output.natural_outcomes,
            second_output.natural_outcomes,
        )
        self.assertFalse(
            torch.allclose(
                first_output.causal_outcome_samples,
                second_output.causal_outcome_samples,
            )
        )

    def test_nmcc_actor_guidance_is_detached_from_the_world_model(self):
        rl = object.__new__(RLBridge)
        logits = torch.zeros(1, 3, requires_grad=True)
        scores = torch.tensor([[0.1, 0.3, -0.2]], requires_grad=True)
        guided, _ = rl._apply_nmcc_guidance(
            logits,
            scores,
            torch.ones(1, 3, dtype=torch.bool),
            weight=0.5,
        )
        guided.sum().backward()
        self.assertIsNone(scores.grad)
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))

    def test_lstm_carries_observation_history_between_decisions(self):
        torch.manual_seed(37)
        policy = EvacPolicy(2, 1, 1, d_global=3, d_momentum=9)
        with torch.no_grad():
            policy.temporal_actor_context.weight.fill_(0.05)
            policy.actor_cell.weight.fill_(0.05)
        graph = fit_gnn(
            torch.randn(4, 2),
            torch.randn(4, 1),
            torch.randn(4, 1),
            torch.randn(3),
        )
        first_logits, _, _, _, state = policy.forward_recurrent(
            graph,
            None,
            torch.zeros(9),
        )
        second_logits, _, _, _, _ = policy.forward_recurrent(
            graph,
            state,
            torch.ones(9),
        )
        self.assertFalse(torch.allclose(first_logits, second_logits))

    def test_one_candidate_per_cell_fixture_outputs_four_logits(self):
        torch.manual_seed(3)
        policy = EvacPolicy(2, 1, 3, d_global=5)
        batch_size = 3
        graph = fit_gnn(
            torch.randn(batch_size * 4, 2),
            torch.randn(batch_size * 4, 1),
            torch.randn(batch_size * 4, 3),
            torch.randn(batch_size, 5),
            batch=torch.arange(batch_size).repeat_interleave(4),
        )
        logits, values = policy(graph)
        self.assertEqual(tuple(logits.shape), (batch_size, 4))
        self.assertEqual(tuple(values.shape), (batch_size,))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(values).all())

    def test_local_cell_has_direct_gradient_path(self):
        torch.manual_seed(9)
        policy = EvacPolicy(
            2,
            1,
            3,
            d_global=5,
        )
        with torch.no_grad():
            policy.global_actor_context.weight.zero_()
            policy.global_actor_context.bias.zero_()
            policy.actor_cell.weight.fill_(0.1)
        x_ped = torch.column_stack((torch.zeros(4), torch.randn(4))).requires_grad_()
        graph = fit_gnn(
            x_ped,
            torch.randn(4, 1),
            torch.randn(4, 3),
            torch.randn(5),
        )
        logits, _ = policy(graph)
        logits[0, 2].backward()
        self.assertGreater(float(x_ped.grad[2].abs().sum()), 0.0)
        torch.testing.assert_close(x_ped.grad[[0, 1, 3]], torch.zeros(3, 2))

    def test_zero_residual_initialization_exactly_ranks_active_population(self):
        torch.manual_seed(12)
        policy = EvacPolicy(
            2,
            1,
            3,
            d_global=5,
        )
        active = torch.tensor([[0.1, 0.0], [0.4, 0.0], [0.2, 0.0], [0.0, 0.0]])
        graph = fit_gnn(
            active,
            torch.zeros(4, 1),
            torch.zeros(4, 3),
            torch.zeros(5),
            edge_index=torch.as_tensor(grid_edge_index(2, 2)),
        )
        with torch.no_grad():
            logits, _ = policy(graph)
        self.assertEqual(int(torch.argmax(logits, dim=-1).item()), 1)
        torch.testing.assert_close(
            logits[0],
            policy.HEURISTIC_PRIOR_SCALE * active[:, 0] / active[:, 0].max(),
        )

    def test_learned_residual_stays_within_its_registered_bound(self):
        policy = EvacPolicy(
            2,
            1,
            3,
            d_global=5,
        )
        with torch.no_grad():
            policy.actor_cell.weight.fill_(1e6)
            policy.actor_cell.bias.fill_(1e6)
        graph = fit_gnn(
            torch.randn(4, 2),
            torch.randn(4, 1),
            torch.randn(4, 3),
            torch.randn(5),
            edge_index=torch.as_tensor(grid_edge_index(2, 2)),
        )
        with torch.no_grad():
            _, _, residual = policy.forward_with_residual(graph)
        self.assertTrue(
            torch.all(torch.abs(residual) <= policy.RESIDUAL_LOGIT_BOUND)
        )

    def test_spatial_scorer_has_neighbor_gradient_path(self):
        torch.manual_seed(19)
        policy = EvacPolicy(
            2,
            1,
            3,
            d_global=5,
        )
        with torch.no_grad():
            policy.global_actor_context.weight.zero_()
            policy.global_actor_context.bias.zero_()
            policy.actor_cell.weight.fill_(0.1)
        x_ped = torch.column_stack((torch.zeros(9), torch.randn(9))).requires_grad_()
        graph = fit_gnn(
            x_ped,
            torch.randn(9, 1),
            torch.randn(9, 3),
            torch.randn(5),
            edge_index=torch.as_tensor(grid_edge_index(3, 3)),
        )
        logits, _ = policy(graph)
        logits[0, 0].backward()
        self.assertGreater(float(x_ped.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(x_ped.grad[[1, 3, 2, 4, 6]].abs().sum()), 0.0)
        torch.testing.assert_close(x_ped.grad[8], torch.zeros(2))

    def test_one_policy_accepts_different_region_counts_and_a_mixed_batch(self):
        policy = EvacPolicy(2, 1, 2, d_global=3, d_candidate=2)
        for node_count in (4, 6):
            graph = fit_gnn(
                torch.randn(node_count, 2),
                torch.randn(node_count, 1),
                torch.randn(node_count, 2),
                torch.randn(3),
                candidate_cell_index=torch.tensor([0, 1, node_count - 1]),
                candidate_features=torch.randn(3, 2),
            )
            logits, value = policy(graph)
            self.assertEqual(tuple(logits.shape), (1, 3))
            self.assertEqual(tuple(value.shape), (1,))

        counts = (4, 6)
        batch = torch.tensor([0] * counts[0] + [1] * counts[1], dtype=torch.long)
        graph = fit_gnn(
            torch.randn(sum(counts), 2),
            torch.randn(sum(counts), 1),
            torch.randn(sum(counts), 2),
            torch.randn(2, 3),
            batch=batch,
            candidate_cell_index=torch.tensor([[0, 1, 3], [0, 2, 5]]),
            candidate_features=torch.randn(2, 3, 2),
        )
        logits, value = policy(graph)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(value.shape), (2,))

    def test_route_aggregation_preserves_assigned_population_magnitude(self):
        policy = EvacPolicy(2, 1, 2, d_global=1)
        layer = policy.message_layers[0]
        features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        edges = torch.tensor([[0], [1]], dtype=torch.long)
        low = layer._aggregate(
            features,
            edges,
            torch.tensor([0.1]),
            normalize=False,
        )
        high = layer._aggregate(
            features,
            edges,
            torch.tensor([0.9]),
            normalize=False,
        )
        torch.testing.assert_close(high[1], 9.0 * low[1])

    def test_candidate_features_can_learn_to_reverse_the_demand_prior(self):
        """The architecture can improve on, rather than copy, the heuristic."""
        torch.manual_seed(41)
        policy = EvacPolicy(
            1,
            1,
            1,
            d_global=1,
            d_candidate=2,
            embed_dim=8,
            message_layers=1,
        )
        graph = fit_gnn(
            torch.tensor([[1.0], [0.2]]),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(1),
            candidate_cell_index=torch.tensor([0, 1]),
            candidate_features=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
        target = torch.tensor([1])
        policy.train()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = policy(graph)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            logits, _ = policy(graph)
        self.assertEqual(int(torch.argmax(logits, dim=1).item()), 1)
        self.assertGreater(float(logits[0, 1] - logits[0, 0]), 0.5)


class BridgeTests(unittest.TestCase):
    def test_checkpoint_rejects_a_different_cell_partition_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            training_core = make_core()
            training_core.cellPartitionMode = "equal_area"
            training_core.cellPartitionMinWidthFraction = 1e-4
            bridge = RLBridge(
                training_core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=1,
                checkpoint_path=checkpoint,
                policy_seed=43,
            )
            bridge._save_checkpoint()

            evaluation_core = make_core()
            evaluation_core.cellPartitionMode = "node_density_adaptive"
            evaluation_core.cellPartitionMinWidthFraction = 1e-4
            with self.assertRaisesRegex(RuntimeError, "decision interface"):
                RLBridge(
                    evaluation_core,
                    train_mode=False,
                    deployment_strategy="rl",
                    target_active_shelters=1,
                    checkpoint_path=checkpoint,
                    policy_seed=43,
                )

    def test_learned_precommit_uses_full_budget_before_first_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            diagnostics = os.path.join(directory, "ppo.csv")
            training_core = make_core()
            trained = RLBridge(
                training_core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=41,
                epochs=1,
                minibatch_size=2,
                rollout_episodes=1,
            )
            trained.step(simulation_time=1)
            del training_core.pedDS.pedAgentList[0]
            training_core.pedDS.result["evacuated"] = 1
            trained.step(simulation_time=6)
            training_core.pedDS.pedAgentList.clear()
            training_core.pedDS.result["evacuated"] = 2
            trained.step(simulation_time=10, is_terminal=True)
            trained.end_episode()

            evaluation_core = make_core()
            precommitted = RLBridge(
                evaluation_core,
                train_mode=False,
                deployment_strategy="rl_precommit",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=41,
            )
            receipts = precommitted.precommit_all()
            self.assertEqual(len(receipts), 2)
            self.assertEqual(precommitted.deployments_made, 2)
            self.assertEqual(precommitted.remaining_deployments, 0)
            self.assertIsNotNone(precommitted.precommit_initial_observation_digest)
            first_boundary = precommitted.step(simulation_time=1)
            self.assertEqual(first_boundary["decision_made"], 0)

    def test_reward_belongs_to_interval_after_executed_action(self):
        core = make_core()
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="heuristic",
            target_active_shelters=2,
            shelter_action_interval=5,
        )
        first = bridge.step(simulation_time=1)
        self.assertEqual(first["decision_made"], 1)
        self.assertEqual(first["reward"], 0.0)

        del core.pedDS.pedAgentList[0]
        core.pedDS.result["evacuated"] = 1
        second = bridge.step(simulation_time=6)
        self.assertAlmostEqual(second["reward_safe"], 0.5)
        self.assertAlmostEqual(second["reward_evacuation_time"], -0.25)
        self.assertAlmostEqual(second["reward_hazard_exposure"], 0.0)
        self.assertAlmostEqual(second["reward_risk_time"], -0.25)
        self.assertAlmostEqual(second["reward"], 0.25)
        self.assertEqual(second["completed_action"], 0)

        core.pedDS.pedAgentList.clear()
        core.pedDS.result["evacuated"] = 2
        terminal = bridge.step(simulation_time=10, is_terminal=True)
        self.assertEqual(terminal["completed_action"], 1)
        diagnostics = bridge.end_episode()
        self.assertAlmostEqual(diagnostics["episode_return"], 0.75)
        # The policy-level objective covers the whole t=0..10 episode, while
        # the action return begins at the first decision boundary t=1.
        self.assertAlmostEqual(diagnostics["objective_episode_return"], 0.65)
        self.assertAlmostEqual(
            diagnostics["objective_risk_weighted_person_time"], 7.0
        )

    def test_static_policy_has_action_count_invariant_episode_objective(self):
        core = make_core()
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="initial_only",
            target_active_shelters=0,
            shelter_action_interval=5,
        )
        bridge.step(simulation_time=1)
        del core.pedDS.pedAgentList[0]
        core.pedDS.result["evacuated"] = 1
        bridge.step(simulation_time=6)
        core.pedDS.pedAgentList.clear()
        core.pedDS.result["evacuated"] = 2
        bridge.step(simulation_time=10, is_terminal=True)
        diagnostics = bridge.end_episode()

        self.assertEqual(diagnostics["episode_return"], 0.0)
        self.assertAlmostEqual(diagnostics["objective_episode_return"], 0.65)
        self.assertAlmostEqual(
            diagnostics["objective_safe_completion_reward"], 1.0
        )
        self.assertAlmostEqual(
            diagnostics["objective_risk_time_penalty"], -0.35
        )

    def test_last_action_credit_continues_to_true_terminal(self):
        core = make_core()
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="heuristic",
            target_active_shelters=1,
            shelter_action_interval=5,
        )
        bridge.step(simulation_time=1)
        still_open = bridge.step(simulation_time=6)
        self.assertEqual(still_open["completed_action"], -1)
        self.assertEqual(still_open["episode_return"], 0.0)
        terminal = bridge.step(simulation_time=10, is_terminal=True)
        self.assertEqual(terminal["completed_action"], 0)
        diagnostics = bridge.end_episode()
        self.assertAlmostEqual(diagnostics["episode_return"], -0.9)
        self.assertAlmostEqual(diagnostics["post_action_objective_return"], -0.9)
        self.assertAlmostEqual(diagnostics["reward_accounting_gap"], 0.0)

    def test_casualties_after_budget_exhaustion_are_not_censored(self):
        core = make_core()
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="heuristic",
            target_active_shelters=1,
            shelter_action_interval=5,
        )
        bridge.step(simulation_time=1)
        still_open = bridge.step(simulation_time=6)
        self.assertEqual(still_open["completed_action"], -1)
        core.pedDS.pedAgentList.clear()
        core.pedDS.result["casualty"] = 2
        terminal = bridge.step(simulation_time=10, is_terminal=True)
        self.assertEqual(terminal["completed_action"], 0)
        self.assertAlmostEqual(terminal["reward_casualty"], -3.0)
        diagnostics = bridge.end_episode()
        self.assertAlmostEqual(diagnostics["casualty_penalty"], -3.0)
        self.assertAlmostEqual(diagnostics["reward_accounting_gap_casualty"], 0.0)

    def test_recurrent_rollout_caches_all_predecision_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            core = make_core()
            bridge = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "ppo.csv"),
                rollout_episodes=2,
            )
            for simulation_time in range(1, 10):
                bridge.step(simulation_time=simulation_time)
            bridge.step(simulation_time=10, is_terminal=True)
            self.assertEqual(len(bridge.traj), 2)
            self.assertEqual(
                [len(transition.observation_history) for transition in bridge.traj],
                [1, 5],
            )
            self.assertEqual(float(bridge.traj[-1].done.item()), 1.0)
            result = bridge.end_episode()
            self.assertAlmostEqual(result["reward_accounting_gap"], 0.0)

    def test_synthetic_ppo_episode_updates_and_reloads_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            diagnostics = os.path.join(directory, "ppo.csv")
            core = make_core()
            bridge = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=13,
                epochs=2,
                minibatch_size=2,
                rollout_episodes=1,
            )
            bridge.step(simulation_time=1)
            del core.pedDS.pedAgentList[0]
            core.pedDS.result["evacuated"] = 1
            bridge.step(simulation_time=6)
            core.pedDS.pedAgentList.clear()
            core.pedDS.result["evacuated"] = 2
            bridge.step(simulation_time=10, is_terminal=True)
            result = bridge.end_episode()
            self.assertGreater(result["gradient_norm"], 0.0)
            self.assertTrue(np.isfinite(result["approximate_kl"]))
            self.assertGreaterEqual(result["entropy"], 0.0)
            self.assertLessEqual(result["entropy"], 1.0 + 1e-6)
            self.assertIn("learning_rate_before_update", result)
            self.assertLessEqual(result["learning_rate"], bridge.lr)
            self.assertTrue(os.path.exists(checkpoint))
            self.assertTrue(os.path.exists(diagnostics))

            reloaded = RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=13,
                epochs=2,
                minibatch_size=2,
                rollout_episodes=1,
            )
            self.assertEqual(reloaded.episodes_completed, 1)
            self.assertEqual(
                reloaded.actor_optimizer_updates,
                bridge.actor_optimizer_updates,
            )
            self.assertEqual(
                reloaded.critic_optimizer_updates,
                bridge.critic_optimizer_updates,
            )
            self.assertEqual(
                reloaded.actor_rollout_updates,
                bridge.actor_rollout_updates,
            )
            self.assertEqual(
                reloaded.actor_return_baselines,
                bridge.actor_return_baselines,
            )
            self.assertEqual(
                reloaded.actor_optimizer.state_dict()["param_groups"],
                bridge.actor_optimizer.state_dict()["param_groups"],
            )
            self.assertEqual(
                reloaded.critic_optimizer.state_dict()["param_groups"],
                bridge.critic_optimizer.state_dict()["param_groups"],
            )

    def test_all_unsafe_training_episode_is_recorded_without_fake_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=1,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "ppo.csv"),
                rollout_episodes=2,
            )
            bridge.observation_builder._candidate_action_mask = (
                lambda remaining, forecast, records: np.zeros(
                    bridge.num_candidate_actions,
                    dtype=np.bool_,
                )
            )
            bridge.step(simulation_time=1)
            bridge.step(simulation_time=10, is_terminal=True)
            result = bridge.end_episode()
            self.assertEqual(result["training_episode_has_decision"], 0.0)
            self.assertEqual(result["optimizer_updated"], 0.0)
            self.assertEqual(result["rollout_episodes_pending"], 0.0)
            self.assertEqual(bridge.episodes_completed, 1)
            self.assertEqual(len(bridge.rollout_traj), 0)

    def test_missed_safe_action_is_retried_and_interval_uses_actual_deployment_time(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=1,
                shelter_action_interval=5,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "ppo.csv"),
            )
            self.assertTrue(bridge._decision_due(1))
            self.assertTrue(bridge._decision_due(2))
            bridge.last_deployment_time = 2
            self.assertFalse(bridge._decision_due(6))
            self.assertTrue(bridge._decision_due(7))

    def test_evaluation_only_checkpoint_cannot_resume_training(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            bridge = RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=1,
                checkpoint_path=checkpoint,
                policy_seed=17,
            )
            bridge._save_checkpoint()
            try:
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            payload["training_resume_allowed"] = False
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "evaluation-only"):
                RLBridge(
                    make_core(),
                    train_mode=True,
                    deployment_strategy="rl",
                    target_active_shelters=1,
                    checkpoint_path=checkpoint,
                    policy_seed=17,
                )

    def test_evaluation_ignores_training_only_signature_fields_but_resume_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            diagnostics = os.path.join(directory, "ppo.csv")
            core = make_core()
            trained = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=31,
                lr=3e-4,
                epochs=2,
                minibatch_size=2,
                rollout_episodes=1,
            )
            trained.step(simulation_time=1)
            del core.pedDS.pedAgentList[0]
            core.pedDS.result["evacuated"] = 1
            trained.step(simulation_time=6)
            core.pedDS.pedAgentList.clear()
            core.pedDS.result["evacuated"] = 2
            trained.step(simulation_time=10, is_terminal=True)
            trained.end_episode()

            evaluated = RLBridge(
                make_core(),
                train_mode=False,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=31,
                lr=1e-3,
                epochs=3,
                minibatch_size=4,
                rollout_episodes=2,
            )
            self.assertEqual(evaluated.episodes_completed, 1)

            with self.assertRaisesRegex(RuntimeError, "incompatible checkpoint"):
                RLBridge(
                    make_core(),
                    train_mode=False,
                    deployment_strategy="rl",
                    target_active_shelters=2,
                    shelter_action_interval=2,
                    checkpoint_path=checkpoint,
                    diagnostics_path=diagnostics,
                    policy_seed=31,
                    lr=1e-3,
                    epochs=3,
                    minibatch_size=4,
                    rollout_episodes=2,
                )

            with self.assertRaisesRegex(RuntimeError, "exact PPO resume configuration"):
                RLBridge(
                    make_core(),
                    train_mode=True,
                    deployment_strategy="rl",
                    target_active_shelters=2,
                    shelter_action_interval=5,
                    checkpoint_path=checkpoint,
                    diagnostics_path=diagnostics,
                    policy_seed=31,
                    lr=1e-3,
                    epochs=3,
                    minibatch_size=4,
                    rollout_episodes=2,
                )

    def test_partial_rollout_is_checkpointed_and_updated_only_when_complete(self):
        def complete_episode(bridge, core):
            bridge.step(simulation_time=1)
            del core.pedDS.pedAgentList[0]
            core.pedDS.result["evacuated"] = 1
            bridge.step(simulation_time=6)
            core.pedDS.pedAgentList.clear()
            core.pedDS.result["evacuated"] = 2
            bridge.step(simulation_time=10, is_terminal=True)
            return bridge.end_episode()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            diagnostics = os.path.join(directory, "ppo.csv")
            first_core = make_core()
            first = RLBridge(
                first_core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=29,
                epochs=2,
                minibatch_size=4,
                rollout_episodes=2,
            )
            first_result = complete_episode(first, first_core)
            self.assertEqual(first_result["optimizer_updated"], 0.0)
            self.assertEqual(first_result["rollout_episodes_pending"], 1.0)
            self.assertEqual(first.optimizer_updates, 0)
            self.assertEqual(len(first.rollout_traj), 2)

            second_core = make_core()
            resumed = RLBridge(
                second_core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                shelter_action_interval=5,
                checkpoint_path=checkpoint,
                diagnostics_path=diagnostics,
                policy_seed=29,
                epochs=2,
                minibatch_size=4,
                rollout_episodes=2,
            )
            self.assertEqual(resumed.episodes_completed, 1)
            self.assertEqual(resumed.rollout_episode_count, 1)
            self.assertEqual(len(resumed.rollout_traj), 2)
            second_result = complete_episode(resumed, second_core)
            self.assertEqual(second_result["optimizer_updated"], 1.0)
            self.assertGreater(second_result["gradient_norm"], 0.0)
            self.assertEqual(second_result["transitions"], 4.0)
            self.assertEqual(second_result["rollout_episodes_pending"], 0.0)
            self.assertEqual(len(resumed.rollout_traj), 0)


class AccountingAndDiagnosticsTests(unittest.TestCase):
    def test_candidate_pool_is_decoupled_from_explicit_installation_budget(self):
        self.assertEqual(
            Core._shelter_target(
                available_candidates=20,
                initial_shelters=2,
                decision_windows=24,
                maximum_additions=5,
            ),
            7,
        )
        self.assertEqual(
            Core._shelter_target(
                available_candidates=5,
                initial_shelters=2,
                decision_windows=24,
                maximum_additions=5,
            ),
            5,
        )

    def test_core_defaults_to_equal_capacity_shelter_tokens(self):
        core = Core("local")
        self.assertGreater(core.shelterCapacityToken, 0)
        self.assertEqual(
            Core._shelter_target(
                available_candidates=20,
                initial_shelters=2,
                decision_windows=24,
                maximum_additions=0,
            ),
            20,
        )

    def test_configuration_rejects_unknown_overrides_and_off_policy_epsilon(self):
        core = Core("local")
        with self.assertRaisesRegex(KeyError, "Unknown configuration"):
            core._apply_configuration_overrides({"typo_parameter": 1})
        core.address = "Test City"
        core.stopTime = 10
        core.pedVol = 10
        core.hazardVol = 1
        core.maxSpeed = 2.0
        core.cellX = 2
        core.cellY = 2
        core.shelterCanVol = 4
        core.initShelterVol = 1
        core.learningRate = 3e-4
        core.explorationRate = 0.5
        core.optimizer = "AdamW"
        core.hazardEvolutionMode = "stochastic"
        with self.assertRaisesRegex(ValueError, "explorationRate must be 0"):
            core._validate_effective_configuration()
        core.explorationRate = 0.0
        core._validate_effective_configuration()
        self.assertEqual(core.criticLearningRate, core.learningRate)
        self.assertEqual(core.actorPpoEpochs, 1)
        self.assertEqual(core.criticPpoEpochs, 4)

        core.congestionEffectiveWidthM = 0.0
        with self.assertRaisesRegex(ValueError, "congestionEffectiveWidthM"):
            core._validate_effective_configuration()
        core.congestionEffectiveWidthM = 3.0
        core.congestionMinimumSpeedRatio = 1.0
        with self.assertRaisesRegex(ValueError, "congestionMinimumSpeedRatio"):
            core._validate_effective_configuration()

    def test_initial_shelter_is_a_reachable_route_target_and_receives_flow(self):
        store = PedDS(1)
        start = SimpleNamespace(OSMID=1, nodeX=0.0, nodeY=0.0)
        shelter_node = SimpleNamespace(OSMID=2, nodeX=1.0, nodeY=0.0)
        route = SimpleNamespace(startNode=start, endNode=shelter_node, edgeRemained=[])
        store.mapDS = SimpleNamespace(shortestPath=lambda source, target: route)
        shelter = Shelter(0, shelter_node, (0, 0), 5, 0, 0)
        shelter_store = ShelterDS(1, 1)
        shelter_store.shelterList = {0: shelter}
        shelter_store.shelterByOSMID = {2: shelter}
        store.shelterDS = shelter_store
        store.cellTracker = SimpleNamespace(locateCell=lambda x, y: (0, 0))
        pedestrian = SimpleNamespace(
            agentID=0,
            terminated=False,
            group_size=1,
            currNode=start,
            atNode=True,
            edge_dest_node=None,
            lastX=0.0,
            lastY=0.0,
            currCell=(0, 0),
            routeFollowing=None,
        )
        store.pedAgentList[0] = pedestrian

        self.assertEqual(store.route_active_to_nearest_shelter(), 1)
        self.assertIs(pedestrian.routeFollowing.endNode, shelter_node)
        store.loadShelterLookup(shelter_store.shelterByOSMID)
        store.arrive_node(pedestrian, shelter_node)
        self.assertEqual(shelter.shelterFlow, 1)
        self.assertTrue(pedestrian.evacuated)

    def test_full_target_shelter_blocks_without_false_safe_arrival(self):
        store = PedDS(1)
        shelter_node = SimpleNamespace(OSMID=2, nodeX=1.0, nodeY=0.0)
        shelter = Shelter(0, shelter_node, (0, 0), 1, 1, 0)
        shelter_store = ShelterDS(1, 1)
        shelter_store.shelterList = {0: shelter}
        shelter_store.shelterByOSMID = {2: shelter}
        store.shelterDS = shelter_store
        store.cellTracker = SimpleNamespace(locateCell=lambda x, y: (0, 0))
        store.loadShelterLookup(shelter_store.shelterByOSMID)
        pedestrian = SimpleNamespace(
            agentID=0,
            terminated=False,
            group_size=1,
            currNode=shelter_node,
            atNode=True,
            edge_dest_node=None,
            lastX=1.0,
            lastY=0.0,
            currCell=(0, 0),
            routeFollowing=SimpleNamespace(endNode=shelter_node),
        )
        store.pedAgentList[0] = pedestrian

        self.assertFalse(store.arrive_node(pedestrian, shelter_node))
        self.assertFalse(pedestrian.terminated)
        self.assertIn(0, store.pedAgentList)
        self.assertEqual(store.result["arrival"], 0)
        self.assertEqual(store.result["evacuated"], 0)

    def test_shelter_admission_respects_group_size_atomically(self):
        node = SimpleNamespace(OSMID=2)
        shelter = Shelter(0, node, (0, 0), 5, 3, 0)
        shelter_store = ShelterDS(1, 1)
        group = SimpleNamespace(group_size=3)
        self.assertEqual(shelter_store.updateShelterFlow(group, shelter), 1)
        self.assertEqual(shelter.shelterFlow, 3)
        self.assertEqual(shelter.status, 0)
        pair = SimpleNamespace(group_size=2)
        self.assertEqual(shelter_store.updateShelterFlow(pair, shelter), 0)
        self.assertEqual(shelter.shelterFlow, 5)
        self.assertEqual(shelter.status, 1)

    def test_unfinished_is_not_counted_as_arrival(self):
        store = PedDS(1)
        pedestrian = SimpleNamespace(agentID=0, terminated=False, group_size=1)
        store.pedAgentList[0] = pedestrian
        self.assertEqual(store.finalize_remaining_pedestrians("Unfinished"), 1)
        store.docuStatus()
        self.assertEqual(store.result["unfinished"], 1)
        self.assertEqual(store.result["arrival"], 0)

    def test_logger_emits_version_four_decision_and_congestion_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = trainingLog(directory)
            logger.log_step(1, 0.5, {"decision_made": 1, "selected_cell": 2})
            logger.close()
            with open(os.path.join(directory, "progress.csv"), newline="") as handle:
                reader = csv.DictReader(handle)
                row = next(reader)
            self.assertEqual(reader.fieldnames, CSV_COLUMNS)
            self.assertEqual(row["selected_cell"], "2")
            self.assertEqual(row["mean_congestion_speed_ratio"], "1.0")

    def test_osm_milestone_visualization_exports_auditable_data_and_figures(self):
        self.assertEqual(resolve_milestones("quartiles", 12), (0, 3, 6, 9, 12))
        with tempfile.TemporaryDirectory() as directory:
            core = make_core()
            core.run_dir = directory
            core.scenario_seed = 101
            core.policy_seed = 202
            core.rl = SimpleNamespace(deployment_strategy="heuristic")
            core.cellTracker.xEdges = [0.0, 1.0, 2.0]
            core.cellTracker.yEdges = [0.0, 1.0, 2.0]
            core.cellTracker.cellList = {
                (i, j): SimpleNamespace(impactedLevel=int(i == 0 and j == 0))
                for i in range(2)
                for j in range(2)
            }
            core.cellTracker.dangerLevelByCell = np.asarray(
                (0.10, 0.25, 0.50, 0.90),
                dtype=float,
            )
            core.pedDS.pedAgentList[0].agentID = 0
            core.pedDS.pedAgentList[0].lastX = 0.25
            core.pedDS.pedAgentList[0].lastY = 0.25
            core.pedDS.pedAgentList[0].affected = False
            core.pedDS.pedAgentList[1].agentID = 1
            core.pedDS.pedAgentList[1].lastX = 0.25
            core.pedDS.pedAgentList[1].lastY = 1.25
            core.pedDS.pedAgentList[1].affected = True
            first_node = SimpleNamespace(OSMID=10, nodeX=0.1, nodeY=0.1)
            second_node = SimpleNamespace(OSMID=11, nodeX=1.9, nodeY=1.9)
            core.mapDS = SimpleNamespace(
                edgeListByLocalID={
                    0: SimpleNamespace(startNode=first_node, endNode=second_node),
                }
            )
            initial = Shelter(0, first_node, (0, 0), 20, 0, 0)
            core.shelterDS.shelterList = {0: initial}
            core.shelterDS.shelterByCell[0][0].append(initial)
            core.shelterDS.nextID = 1
            core.hazardDS = SimpleNamespace(
                hazardList={0: SimpleNamespace(sourceNode=first_node)}
            )

            visualizer = EvacuationVisualizer(
                core,
                milestones="0,5,10",
                output_dir=os.path.join(directory, "visualization"),
            )
            self.assertTrue(visualizer.observe(0))
            core.shelterDS.newShelter({"cell": (0, 0)}, core.cellTracker)
            self.assertTrue(
                visualizer.observe(
                    5,
                    {
                        "decision_made": 1,
                        "selected_cell": 0,
                        "heuristic_cell": 0,
                        "capacity_added": 10,
                        "rerouted_population": 1,
                    },
                )
            )
            manifest = visualizer.finalize(terminal_time=5)
            self.assertEqual(manifest["captured_milestones"], [0, 5])
            self.assertTrue(manifest["non_interventional"])
            for filename in (
                "milestone_pedestrians.csv",
                "milestone_shelters.csv",
                "milestone_cells.csv",
                "milestone_outcomes.csv",
                "decision_epoch_cells.csv",
                "decision_epoch_pedestrians.csv.gz",
                "deployment_decisions.csv",
                "osm_road_segments.csv",
                "evacuation_milestones.png",
                "evacuation_milestones.svg",
                "deployment_sequence.png",
                "deployment_sequence.svg",
                "decision_epochs.png",
                "decision_epochs.svg",
                "visualization_manifest.json",
            ):
                self.assertTrue(
                    os.path.exists(os.path.join(directory, "visualization", filename))
                )
            with open(
                os.path.join(directory, "visualization", "deployment_decisions.csv"),
                newline="",
            ) as handle:
                decisions = list(csv.DictReader(handle))
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["decision_type"], "dynamic")
            self.assertEqual(
                decisions[0]["candidate_osm_node_id"],
                str(core.shelterDS.shelterList[1].nodeMapped.OSMID),
            )
            with gzip.open(
                os.path.join(
                    directory,
                    "visualization",
                    "decision_epoch_pedestrians.csv.gz",
                ),
                "rt",
                newline="",
                encoding="utf-8",
            ) as handle:
                decision_pedestrians = list(csv.DictReader(handle))
            self.assertEqual(len(decision_pedestrians), 2)
            self.assertEqual(
                {row["agent_id"] for row in decision_pedestrians},
                {"0", "1"},
            )
            self.assertEqual(
                {row["decision_index"] for row in decision_pedestrians},
                {"1"},
            )
            with open(
                os.path.join(directory, "visualization", "decision_epoch_cells.csv"),
                newline="",
            ) as handle:
                decision_cells = list(csv.DictReader(handle))
            self.assertEqual(
                [float(row["danger"]) for row in decision_cells],
                [0.10, 0.25, 0.50, 0.90],
            )
            self.assertEqual(manifest["schema_version"], 4)
            self.assertEqual(manifest["speed_units"], "metres_per_minute")
            self.assertEqual(
                manifest["visual_encoding"]["active_pedestrians"]["unit"],
                "one marker per active pedestrian agent",
            )

    def test_static_predeployments_are_logged_as_time_zero_candidate_choices(self):
        with tempfile.TemporaryDirectory() as directory:
            core = make_core()
            core.run_dir = directory
            core.scenario_seed = 101
            core.policy_seed = 202
            core.rl = SimpleNamespace(deployment_strategy="initial_only")
            core.cellTracker.xEdges = [0.0, 1.0, 2.0]
            core.cellTracker.yEdges = [0.0, 1.0, 2.0]
            core.cellTracker.cellList = {
                (i, j): SimpleNamespace(impactedLevel=0)
                for i in range(2)
                for j in range(2)
            }
            first_node = SimpleNamespace(OSMID=10, nodeX=0.1, nodeY=0.1)
            second_node = SimpleNamespace(OSMID=11, nodeX=1.9, nodeY=1.9)
            core.mapDS = SimpleNamespace(
                edgeListByLocalID={
                    0: SimpleNamespace(startNode=first_node, endNode=second_node),
                }
            )
            initial = Shelter(0, first_node, (0, 0), 20, 0, 0)
            static = Shelter(1, second_node, (1, 1), 30, 0, 0)
            core.shelterDS.shelterList = {0: initial, 1: static}
            core.baseline_initial_shelter_ids = frozenset({0})
            core.static_predeployment_shelter_ids = frozenset({1})
            core.hazardDS = SimpleNamespace(hazardList={})
            for pedestrian in core.pedDS.pedAgentList.values():
                pedestrian.agentID = int(id(pedestrian))
                pedestrian.lastX = 0.25
                pedestrian.lastY = 0.25
                pedestrian.affected = False

            visualizer = EvacuationVisualizer(
                core,
                milestones="0,10",
                output_dir=os.path.join(directory, "visualization"),
                render_individual_snapshots=False,
                render_vector_outputs=False,
            )
            self.assertTrue(visualizer.observe(0))
            manifest = visualizer.finalize(terminal_time=0)
            self.assertFalse(manifest["individual_snapshot_figures"])
            self.assertFalse(
                os.path.exists(
                    os.path.join(directory, "visualization", "milestone_t0000.png")
                )
            )
            with open(
                os.path.join(directory, "visualization", "deployment_decisions.csv"),
                newline="",
            ) as handle:
                decisions = list(csv.DictReader(handle))
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["decision_type"], "static_predeployment")
            self.assertEqual(decisions[0]["simulation_time"], "0")
            self.assertEqual(decisions[0]["candidate_osm_node_id"], "11")

            with open(
                os.path.join(directory, "visualization", "milestone_shelters.csv"),
                newline="",
            ) as handle:
                shelters = {row["shelter_id"]: row for row in csv.DictReader(handle)}
            self.assertEqual(shelters["0"]["deployment_mode"], "initial")
            self.assertEqual(shelters["1"]["deployment_mode"], "static_predeployment")
            self.assertFalse(
                os.path.exists(
                    os.path.join(directory, "visualization", "evacuation_milestones.svg")
                )
            )

    def test_early_completion_is_extended_only_as_labeled_absorbing_states(self):
        with tempfile.TemporaryDirectory() as directory:
            core = make_core()
            core.run_dir = directory
            core.scenario_seed = 101
            core.policy_seed = 202
            core.rl = SimpleNamespace(deployment_strategy="heuristic")
            core.cellTracker.xEdges = [0.0, 1.0, 2.0]
            core.cellTracker.yEdges = [0.0, 1.0, 2.0]
            core.cellTracker.cellList = {
                (i, j): SimpleNamespace(impactedLevel=0)
                for i in range(2)
                for j in range(2)
            }
            first_node = SimpleNamespace(OSMID=10, nodeX=0.1, nodeY=0.1)
            second_node = SimpleNamespace(OSMID=11, nodeX=1.9, nodeY=1.9)
            core.mapDS = SimpleNamespace(
                edgeListByLocalID={
                    0: SimpleNamespace(startNode=first_node, endNode=second_node),
                }
            )
            core.shelterDS.shelterList = {
                0: Shelter(0, first_node, (0, 0), 20, 0, 0)
            }
            core.hazardDS = SimpleNamespace(hazardList={})
            for pedestrian in core.pedDS.pedAgentList.values():
                pedestrian.terminated = True
            core.pedDS.result = {
                "arrival": 0,
                "evacuated": 2,
                "casualty": 0,
                "affected": 0,
            }

            visualizer = EvacuationVisualizer(
                core,
                milestones="0,5,10",
                output_dir=os.path.join(directory, "visualization"),
                render_individual_snapshots=False,
                render_vector_outputs=False,
            )
            visualizer.observe(0)
            manifest = visualizer.finalize(terminal_time=5)
            self.assertEqual(manifest["terminal_time"], 5)
            self.assertEqual(manifest["captured_milestones"], [0, 5, 10])
            with open(
                os.path.join(directory, "visualization", "milestone_outcomes.csv"),
                newline="",
            ) as handle:
                outcomes = {
                    int(row["simulation_time"]): row for row in csv.DictReader(handle)
                }
            self.assertEqual(outcomes[10]["active_population"], "0")
            self.assertEqual(outcomes[10]["absorbing_after_terminal"], "1")
            self.assertEqual(outcomes[10]["terminal_time"], "5")

    def test_backtest_checks_observation_parity_and_paired_statistics(self):
        rows = []
        for replication in range(1, 5):
            for strategy, reward, casualty in (("rl", 1.0, 0), ("heuristic", 0.5, 1)):
                rows.append(
                    {
                        "replication": replication,
                        "deployment_strategy": strategy,
                        "scenario_seed": 100 + replication,
                        "policy_seed": 7,
                        "episode_return": reward,
                        "safe_completed": 10,
                        "shelter_evacuated": 10,
                        "arrival": 0,
                        "casualty": casualty,
                        "unfinished": 0,
                        "restricted_mean_time_to_safety": 5.0,
                        "mean_safe_completion_time": 5.0,
                        "mean_evacuation_time": 5.0,
                        "normalized_risk_weighted_person_time": 0.2,
                        "risk_weighted_person_time": 20.0,
                        "decisions": 2,
                        "deployments_made": 2,
                        "maximum_dynamic_deployments": 2,
                        "total_shelter_capacity": 1000.0,
                        "active_shelters": 2,
                        "initial_observation_digest": f"same-{replication}",
                        "random_stream_seeds": {
                            "initialization": 100 + replication,
                            "hazard_evolution": 200 + replication,
                            "pedestrian_hazard_outcomes": 300 + replication,
                        },
                        "hazard_trajectory_digest": f"hazard-{replication}",
                    }
                )
        verification = _verify_matched_interface(rows, ("rl", "heuristic"))
        self.assertTrue(verification["verified"])
        analysis = _paired_analysis(rows, launch_seed=3, draws=200)
        return_row = next(row for row in analysis if row["metric"] == "episode_return")
        casualty_row = next(row for row in analysis if row["metric"] == "casualty")
        self.assertGreater(return_row["mean_rl_improvement"], 0.0)
        self.assertGreater(casualty_row["mean_rl_improvement"], 0.0)

        unequal_capacity = [dict(row) for row in rows]
        unequal_capacity[0]["total_shelter_capacity"] = 999.0
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            _verify_matched_interface(
                unequal_capacity,
                ("rl", "heuristic"),
            )

    def test_benchmark_analysis_uses_full_episode_objective_for_every_comparator(self):
        self.assertEqual(
            _evaluation_metric(
                {"episode_return": 99.0, "objective_episode_return": 2.0},
                "episode_return",
            ),
            2.0,
        )
        self.assertAlmostEqual(
            _evaluation_metric(
                {
                    "normalized_risk_weighted_person_time": 0.0,
                    "objective_risk_weighted_person_time": 40.0,
                    "initial_population": 10,
                    "horizon_transitions": 8,
                },
                "normalized_risk_weighted_person_time",
            ),
            0.5,
        )
        rows = []
        common = {
            "safe_completed": 8,
            "casualty": 1,
            "unfinished": 1,
            "restricted_mean_time_to_safety": 4.0,
            "normalized_risk_weighted_person_time": 0.2,
        }
        for replication in (1, 2):
            for policy_replication, objective in ((1, 1.0), (2, 1.2)):
                rows.append(
                    {
                        **common,
                        "replication": replication,
                        "policy_replication": policy_replication,
                        "deployment_strategy": "rl",
                        "episode_return": 0.0,
                        "objective_episode_return": objective,
                    }
                )
            for strategy, objective in (("heuristic", 0.5), ("random", 0.2)):
                rows.append(
                    {
                        **common,
                        "replication": replication,
                        "policy_replication": 0,
                        "deployment_strategy": strategy,
                        "episode_return": 0.0,
                        "objective_episode_return": objective,
                    }
                )
        analysis = _benchmark_analysis(
            rows,
            launch_seed=5,
            draws=100,
            benchmark_strategies=("heuristic", "random"),
        )
        self.assertEqual(len(analysis), 12)
        primary = next(
            row
            for row in analysis
            if row["benchmark_strategy"] == "heuristic"
            and row["metric"] == "episode_return"
        )
        self.assertEqual(primary["comparison_role"], "primary")
        self.assertAlmostEqual(primary["rl_mean"], 1.1)
        self.assertAlmostEqual(primary["benchmark_mean"], 0.5)
        self.assertAlmostEqual(primary["mean_rl_improvement"], 0.6)

    def test_policy_level_randomization_is_exact_for_five_training_seeds(self):
        value = _paired_randomization_pvalue(
            np.ones(5, dtype=float),
            np.random.default_rng(3),
            draws=20_000,
        )
        self.assertEqual(value, 0.0625)

    def test_convergence_audit_requires_finite_stationary_policy_seeds(self):
        rows = []
        for policy in (1, 2):
            for episode in range(1, 101):
                rows.append(
                    {
                        "policy_replication": policy,
                        "replication": episode,
                        "episode_return": -0.5,
                        "entropy": 1.0,
                        "approximate_kl": 0.01,
                        "gradient_norm": 0.2,
                        "policy_loss": 0.1,
                        "value_loss": 0.1,
                    }
                )
        audit = _training_convergence(
            rows,
            minimum_episodes=100,
            window_fraction=0.2,
            trend_threshold=0.5,
            shift_threshold=0.5,
            target_kl=0.03,
        )
        self.assertTrue(audit["all_policies_converged"])
        rows[-1]["gradient_norm"] = float("nan")
        failed = _training_convergence(
            rows,
            minimum_episodes=100,
            window_fraction=0.2,
            trend_threshold=0.5,
            shift_threshold=0.5,
            target_kl=0.03,
        )
        self.assertFalse(failed["all_policies_converged"])

    def test_performance_assessment_never_claims_superiority_when_interval_crosses_zero(self):
        inconclusive = _performance_assessment(
            [
                {
                    "metric": "episode_return",
                    "mean_rl_improvement": 0.1,
                    "bootstrap_95_ci_low": -0.1,
                    "bootstrap_95_ci_high": 0.3,
                    "policy_replications": 5,
                    "scenario_replications": 50,
                }
            ]
        )
        self.assertEqual(inconclusive["status"], "inconclusive")

    def test_performance_assessment_requires_independent_replication(self):
        assessment = _performance_assessment(
            [
                {
                    "metric": "episode_return",
                    "mean_rl_improvement": 0.1,
                    "bootstrap_95_ci_low": 0.1,
                    "bootstrap_95_ci_high": 0.1,
                    "policy_replications": 1,
                    "scenario_replications": 1,
                }
            ]
        )
        self.assertEqual(assessment["status"], "descriptive_only_insufficient_replication")
        self.assertFalse(assessment["inferentially_eligible"])

    def test_performance_assessment_supports_preregistered_fixed_policy_scope(self):
        assessment = _performance_assessment(
            [
                {
                    "metric": "episode_return",
                    "mean_rl_improvement": 0.1,
                    "bootstrap_95_ci_low": 0.05,
                    "bootstrap_95_ci_high": 0.15,
                    "policy_replications": 1,
                    "scenario_replications": 25,
                    "scenarios_per_city": 5,
                }
            ],
            allow_fixed_policy=True,
        )
        self.assertEqual(assessment["status"], "rl_superior")
        self.assertTrue(assessment["inferentially_eligible"])
        self.assertEqual(
            assessment["inference_scope"],
            "conditional_on_one_fixed_trained_policy",
        )

    def test_fixed_policy_paper_table_marks_policy_seed_p_value_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "paired.md")
            _write_paper_table(
                path,
                [
                    {
                        "metric": "episode_return",
                        "rl_mean": 0.2,
                        "heuristic_mean": 0.1,
                        "mean_rl_improvement": 0.1,
                        "bootstrap_95_ci_low": 0.05,
                        "bootstrap_95_ci_high": 0.15,
                        "two_sided_randomization_p": None,
                        "inferentially_eligible": True,
                        "inference_scope": "conditional_on_one_fixed_trained_policy",
                    }
                ],
            )
            with open(path, encoding="utf-8") as handle:
                rendered = handle.read()
        self.assertIn("| NA |", rendered)
        self.assertIn("one frozen policy", rendered)

    def test_json_artifacts_are_strict_when_diagnostics_are_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "audit.json")
            _json_dump(path, {"nan": float("nan"), "infinity": float("inf")})
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            self.assertEqual(json.loads(raw), {"infinity": None, "nan": None})


if __name__ == "__main__":
    unittest.main()
