import csv
import gzip
import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F

import CounterfactualBranch as CB
import NMCCPolicyImprovement as NPI
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
    estimate_remaining_hazard_weighted_travel_time,
    estimate_remaining_time_to_safety,
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
    def test_random_site_advantage_is_signed_and_capacity_centered(self):
        snapshot = OutcomeSnapshot(0, 0, 0, 0, 10, 10.0)
        processor = RewardProcessor()
        better = processor.evaluate_random_site_advantage(
            start=snapshot,
            action_after=snapshot,
            random_plain_time=600.0,
            action_plain_time=500.0,
            random_hazard_exposure=300.0,
            action_hazard_exposure=200.0,
            initial_population=10,
            horizon_minutes=100.0,
        )
        worse = processor.evaluate_random_site_advantage(
            start=snapshot,
            action_after=snapshot,
            random_plain_time=600.0,
            action_plain_time=700.0,
            random_hazard_exposure=300.0,
            action_hazard_exposure=400.0,
            initial_population=10,
            horizon_minutes=100.0,
        )
        self.assertAlmostEqual(better.total, 0.2)
        self.assertAlmostEqual(worse.total, -0.2)

    def test_wait_action_reward_adds_plain_and_hazard_exposure_reductions(self):
        snapshot = OutcomeSnapshot(0, 0, 0, 0, 10, 10.0)
        reward = RewardProcessor().evaluate_counterfactual(
            start=snapshot,
            action_after=snapshot,
            plain_time_wait=600.0,
            plain_time_action=500.0,
            hazard_exposure_wait=300.0,
            hazard_exposure_action=150.0,
            initial_population=10,
            horizon_minutes=120.0,
        )
        self.assertEqual(reward.raw_plain_time_reduction, 100.0)
        self.assertEqual(reward.raw_hazard_exposure_reduction, 150.0)
        self.assertEqual(reward.raw_remaining_risk_time_reduction, 250.0)
        self.assertAlmostEqual(reward.total, 250.0 / 1200.0)

    def test_casualty_and_panic_have_no_reward_coefficients(self):
        start = OutcomeSnapshot(0, 0, 0, 0, 10, 10.0)
        after = OutcomeSnapshot(0, 3, 0, 0, 7, 7.0)
        reward = RewardProcessor().evaluate_counterfactual(
            start=start,
            action_after=after,
            plain_time_wait=500.0,
            plain_time_action=500.0,
            hazard_exposure_wait=100.0,
            hazard_exposure_action=100.0,
            initial_population=10,
            horizon_minutes=120.0,
        )
        self.assertEqual(reward.new_casualties, 3)
        self.assertEqual(reward.casualty_penalty, 0.0)
        self.assertEqual(reward.total, 0.0)

    def test_positive_remaining_risk_time_reduction_equation(self):
        before = OutcomeSnapshot(2, 1, 2, 0, 7, 9.0)
        after = OutcomeSnapshot(2, 1, 2, 0, 7, 9.0)
        reward = RewardProcessor().evaluate(
            before=before,
            after=after,
            remaining_risk_time_before=80.0,
            remaining_risk_time_after=40.0,
            initial_population=10,
            horizon=10,
        )
        self.assertAlmostEqual(reward.safe_completion_reward, 0.0)
        self.assertAlmostEqual(reward.casualty_penalty, 0.0)
        self.assertAlmostEqual(reward.evacuation_time_penalty, 0.0)
        self.assertAlmostEqual(reward.hazard_exposure_penalty, 0.4)
        self.assertAlmostEqual(reward.terminal_residual_penalty, 0.0)
        self.assertAlmostEqual(reward.risk_time_penalty, 0.4)
        self.assertAlmostEqual(reward.risk_weighted_person_time, 40.0)
        self.assertAlmostEqual(reward.total, 0.4)
        np.testing.assert_allclose(reward.component_vector(), (0.4,))

    def test_route_worsening_is_zero_not_negative_environmental_reward(self):
        snapshot = OutcomeSnapshot(0, 0, 0, 0, 2, 2.0)
        reward = RewardProcessor().evaluate(
            before=snapshot,
            after=snapshot,
            remaining_risk_time_before=20.0,
            remaining_risk_time_after=30.0,
            initial_population=2,
            horizon=10,
        )
        self.assertEqual(reward.raw_remaining_risk_time_reduction, -10.0)
        self.assertEqual(reward.remaining_risk_time_reduction, 0.0)
        self.assertEqual(reward.total, 0.0)

    def test_safe_arrivals_have_no_separate_bonus_or_site_shaping(self):
        before = OutcomeSnapshot(2, 0, 1, 1, 8, 8.0)
        after = OutcomeSnapshot(4, 0, 3, 1, 6, 6.0)
        reward = RewardProcessor().evaluate(
            before=before,
            after=after,
            remaining_risk_time_before=40.0,
            remaining_risk_time_after=40.0,
            initial_population=10,
            horizon=10,
        )
        self.assertAlmostEqual(reward.safe_completion_reward, 0.0)
        self.assertAlmostEqual(reward.shelter_service_reward, 0.0)
        self.assertAlmostEqual(reward.total, 0.0)

    def test_casualties_are_reported_but_not_rewarded(self):
        reward = RewardProcessor().evaluate(
            before=OutcomeSnapshot(0, 0, 0, 0, 3, 3.0),
            after=OutcomeSnapshot(0, 2, 0, 0, 1, 1.0),
            remaining_risk_time_before=15.0,
            remaining_risk_time_after=15.0,
            initial_population=3,
            horizon=5,
        )
        self.assertEqual(reward.new_casualties, 2)
        self.assertEqual(reward.casualty_penalty, 0.0)
        self.assertEqual(reward.total, 0.0)
        self.assertEqual(RewardProcessor().evacuation_time_weight, 0.0)

    def test_outcomes_must_be_monotone(self):
        processor = RewardProcessor()
        with self.assertRaisesRegex(ValueError, "monotone"):
            processor.evaluate(
                before=OutcomeSnapshot(2, 0, 2, 0, 1, 1.0),
                after=OutcomeSnapshot(1, 0, 1, 0, 2, 2.0),
                remaining_risk_time_before=10.0,
                remaining_risk_time_after=10.0,
                initial_population=3,
                horizon=5,
            )

    def test_pedestrians_without_reserved_shelter_receive_full_horizon_time(self):
        core = make_core()
        core.cellTracker.dangerLevelByCell[:] = (0.5, 1.0, 0.0, 0.0)
        debt = estimate_remaining_hazard_weighted_travel_time(
            core,
            episode_horizon=10,
        )
        self.assertAlmostEqual(debt, 35.0)


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

    def test_forecast_distribution_exposes_momentum_probability_and_uncertainty(self):
        tracker = CellTracker(2, 1)
        tracker.initialCut(200.0, 100.0)
        tracker.setCellState((0, 0), 5)
        tracker.cellUpdate()
        hazards = HazardDS(
            1,
            [0, 0],
            [50, 0],
            [0, 0],
            rng=np.random.default_rng(17),
            wind_speed_m_per_minute=20.0,
            wind_direction_degrees=0.0,
        )
        hazards.cell_state_evolution_mode = "stochastic"
        hazards.hazardList = {
            0: SimpleNamespace(active=True, spreadRate=0.5)
        }
        hazards.setCellTracker(tracker)
        rng_before = repr(hazards.rng.bit_generator.state)
        forecast = hazards.forecast_distribution_by_cell(5)
        self.assertEqual(rng_before, repr(hazards.rng.bit_generator.state))
        self.assertEqual(
            set(forecast),
            {
                "expected",
                "momentum",
                "escalation_probability",
                "uncertainty",
            },
        )
        for values in forecast.values():
            self.assertEqual(values.shape, (2,))
            self.assertTrue(np.isfinite(values).all())
        self.assertGreater(float(forecast["expected"][1]), 0.0)
        self.assertGreater(float(forecast["momentum"][1]), 0.0)
        self.assertGreater(float(forecast["escalation_probability"][1]), 0.0)
        self.assertGreaterEqual(float(forecast["uncertainty"][1]), 0.0)

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
        # A 100 m cell makes the calibrated physical front take three minutes
        # downwind and four minutes upwind at the declared spread speed.
        tracker.initialCut(300.0, 100.0)
        tracker.setCellState((1, 0), 1)
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
            0: SimpleNamespace(
                active=True,
                spreadRate=0.5,
                sourceCell=(1, 0),
                impactedCells=[(1, 0)],
                cellImpactStep={},
                step=0,
                age=0,
                lifespan=60,
            ),
        }
        forecast = hazards.forecast_danger_by_cell(3)
        self.assertGreater(float(forecast[2]), float(forecast[0]))

    def test_north_wind_uses_physical_north_not_southward_map_y(self):
        tracker = CellTracker(1, 3)
        tracker.initialCut(1.0, 3.0)
        hazards = HazardDS(
            0,
            [10, 0],
            [50, 0],
            [20, 0],
            wind_speed_m_per_minute=1.0,
            wind_direction_degrees=90.0,
            wind_influence=1.0,
        )
        hazards.setCellTracker(tracker)
        northward = hazards.directional_spread_multiplier((0, 1), (0, 0))
        southward = hazards.directional_spread_multiplier((0, 1), (0, 2))
        self.assertGreater(northward, southward)

    def test_deterministic_forecast_matches_rectangular_runtime_ca(self):
        def model():
            tracker = CellTracker(2, 3)
            tracker.initialCut(2.0, 3.0)
            tracker.setCellState((0, 1), 1)
            tracker.cellUpdate()
            hazards = HazardDS(0, [10, 0], [50, 0], [20, 0])
            hazards.setCellTracker(tracker)
            hazards.hazardList = {
                0: SimpleNamespace(
                    active=True,
                    spreadRate=0.5,
                    sourceCell=(0, 1),
                    impactedCells=[(0, 1)],
                    cellImpactStep={},
                    step=0,
                    age=0,
                    lifespan=60,
                )
            }
            return tracker, hazards

        forecast_tracker, forecast_hazards = model()
        forecast = forecast_hazards.forecast_danger_by_cell(6)
        self.assertEqual(forecast_hazards.hazardList[0].step, 0)
        self.assertEqual(forecast_tracker.getCellState((0, 1)), 1)

        runtime_tracker, runtime_hazards = model()
        for _ in range(6):
            runtime_hazards.spreadUpdate()
            runtime_hazards.terminateHazard()
        np.testing.assert_allclose(
            forecast,
            runtime_tracker.state_vector(normalized=True),
        )

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
        signed_cell = CELL_FEATURE_NAMES.index("danger_spread_momentum")
        unsigned_cells = np.delete(cells, signed_cell, axis=1)
        self.assertTrue(
            np.logical_and(unsigned_cells >= 0.0, unsigned_cells <= 1.0).all()
        )
        self.assertTrue(
            np.logical_and(cells[:, signed_cell] >= -1.0, cells[:, signed_cell] <= 1.0).all()
        )
        signed_global = {
            GLOBAL_FEATURE_NAMES.index("wind_east_velocity_fraction"),
            GLOBAL_FEATURE_NAMES.index("wind_north_velocity_fraction"),
            GLOBAL_FEATURE_NAMES.index("forecast_hazard_growth_fraction"),
        }
        for index, value in enumerate(global_features):
            lower = -1.0 if index in signed_global else 0.0
            self.assertGreaterEqual(float(value), lower)
            self.assertLessEqual(float(value), 1.0)
        signed_candidate = CANDIDATE_FEATURE_NAMES.index(
            "exact_assignment_gain_fraction"
        )
        unsigned_candidates = np.delete(candidates, signed_candidate, axis=1)
        self.assertTrue(
            np.logical_and(
                unsigned_candidates >= 0.0,
                unsigned_candidates <= 1.0,
            ).all()
        )
        self.assertTrue(
            np.logical_and(
                candidates[:, signed_candidate] >= -2.0,
                candidates[:, signed_candidate] <= 2.0,
            ).all()
        )

    def test_exact_assignment_gain_preserves_bad_shelter_signal(self):
        observation = replace(
            make_observation(),
            candidate_assignment_gain=np.asarray(
                (0.2, -0.4, 0.0, 0.0), dtype=np.float32
            ),
        )
        feature_index = CANDIDATE_FEATURE_NAMES.index(
            "exact_assignment_gain_fraction"
        )
        self.assertAlmostEqual(
            float(observation.candidate_features()[1, feature_index]),
            -0.4,
            places=6,
        )

    def test_features_expose_congestion_delay_and_available_capacity(self):
        observation = make_observation(
            active=(4, 8, 3, 1),
            remaining_capacity=(0, 5, 0, 0),
        )
        cells, _ = observation.policy_features()
        np.testing.assert_allclose(
            cells[:, CELL_FEATURE_NAMES.index("remaining_shelter_capacity_fraction")],
            (0.0, 0.3125, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            cells[:, CELL_FEATURE_NAMES.index("congestion_delay_fraction")],
            (0.0, 0.0, 0.0, 0.0),
        )

    def test_policy_contract_removes_redundant_features_and_uses_wind_vector(self):
        self.assertNotIn("assigned_population_fraction", PED_FEATURE_NAMES)
        self.assertNotIn("mobility_delay_fraction", PED_FEATURE_NAMES)
        self.assertNotIn("outgoing_assignment_flow_fraction", PED_FEATURE_NAMES)
        self.assertNotIn("incoming_assignment_flow_fraction", PED_FEATURE_NAMES)
        self.assertNotIn("wind_speed_fraction", GLOBAL_FEATURE_NAMES)
        self.assertIn("wind_east_velocity_fraction", GLOBAL_FEATURE_NAMES)
        self.assertIn("wind_north_velocity_fraction", GLOBAL_FEATURE_NAMES)

    def test_policy_observes_the_spatial_clearance_tail_it_is_optimized_on(self):
        observation = replace(
            make_observation(active=(4, 8, 3, 1)),
            slow_tail_population_by_cell=np.asarray(
                (0.0, 2.0, 0.0, 0.0), dtype=np.float32
            ),
            slow_tail_time_by_cell=np.asarray(
                (0.0, 20.0, 0.0, 0.0), dtype=np.float32
            ),
        )
        cells, global_features = observation.policy_features()
        population_scale = float(observation.initial_population)
        route_scale = float(observation.horizon) * observation.time_step_minutes
        np.testing.assert_allclose(
            cells[
                :,
                CELL_FEATURE_NAMES.index("slow_tail_population_fraction"),
            ],
            (0.0, 2.0 / population_scale, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            cells[:, CELL_FEATURE_NAMES.index("slow_tail_time_fraction")],
            (0.0, 20.0 / (population_scale * route_scale), 0.0, 0.0),
        )
        self.assertAlmostEqual(
            float(
                global_features[
                    GLOBAL_FEATURE_NAMES.index("slow_tail_population_fraction")
                ]
            ),
            2.0 / population_scale,
        )
        self.assertAlmostEqual(
            float(
                global_features[
                    GLOBAL_FEATURE_NAMES.index("slow_tail_time_fraction")
                ]
            ),
            20.0 / (population_scale * route_scale),
        )


class FakeCellTracker:
    def __init__(self):
        self.cellXNum = 2
        self.cellYNum = 2
        self.xEdges = np.asarray((0.0, 1.0, 2.0), dtype=float)
        self.yEdges = np.asarray((0.0, 1.0, 2.0), dtype=float)
        self.dangerLevelByCell = np.zeros(4, dtype=float)
        self.countByCell = np.asarray((1, 1, 0, 0), dtype=float)
        self.added = []

    @property
    def number_of_cells(self):
        return self.cellXNum * self.cellYNum

    def flatten_cell(self, cell):
        return int(cell[0]) * self.cellYNum + int(cell[1])

    def unflatten_cell(self, index):
        return divmod(int(index), self.cellYNum)

    def addShelter(self, cell, shelter):
        self.added.append((tuple(cell), shelter))

    def cellUpdate(self, pedDS=None, forceTracker=None):
        del forceTracker
        counts = np.zeros(4, dtype=float)
        for pedestrian in getattr(pedDS, "pedAgentList", {}).values():
            if not bool(getattr(pedestrian, "terminated", False)):
                i, j = pedestrian.currCell
                counts[int(i) * 2 + int(j)] += int(
                    getattr(pedestrian, "group_size", 1)
                )
        self.countByCell = counts


class FakePedestrianStore:
    def __init__(self):
        self.pedAgentList = {
            0: SimpleNamespace(agentID=0, currCell=(0, 0), currSpeed=2.0, terminated=False, group_size=1),
            1: SimpleNamespace(agentID=1, currCell=(0, 1), currSpeed=4.0, terminated=False, group_size=1),
        }
        self.result = {"arrival": 0, "evacuated": 0, "casualty": 0}
        self.currTime = 0

    def reroute_active_to_nearest_shelter(self, new_shelter=None):
        del new_shelter
        return len(self.pedAgentList)

    @staticmethod
    def _route_anchor_node(pedestrian):
        return getattr(pedestrian, "currNode", None)

    def startDocument(self):
        return None

    def loadShelterLookup(self, lookup):
        del lookup

    def pedestrianHazardInteraction(self):
        return None

    def interPedestrianInteraction(self):
        return None

    def pedestrianNetworkInteraction(self):
        return None

    def docuStatus(self):
        self.currTime += 1


class FakeHazardStore:
    def spreadUpdate(self):
        return None

    def heatUpdate(self):
        return None

    def smokeUpdate(self):
        return None

    def terminateHazard(self):
        return None


class FakeMapStore:
    """Minimal exact-routing contract for zero-horizon reward tests."""

    def __init__(self, nodes):
        self.nodeListByLocalID = {
            index: node for index, node in enumerate(nodes)
        }
        self.edgeListByLocalID = {}

    def _routing_tree(self, target_osmid):
        del target_osmid
        distances = {
            int(node.OSMID): 10_000.0
            for node in self.nodeListByLocalID.values()
        }
        return distances, {}

    @staticmethod
    def shortestPath(start, target):
        return SimpleNamespace(
            edgeRemained=[
                SimpleNamespace(
                    edgeLen=10_000.0,
                    cellID=getattr(start, "cellID", (0, 0)),
                )
            ],
            endNode=target,
        )


def make_core():
    shelter_store = ShelterDS(candidateVol=4, initVol=0)
    shelter_store.shelterByCell = [[[], []], [[], []]]
    shelter_store.shelterCanByCell = [[[], []], [[], []]]
    node_id = 100
    capacities = ((10, 20), (30, 40))
    candidate_nodes = []
    for i in range(2):
        for j in range(2):
            capacity = capacities[i][j]
            node = SimpleNamespace(
                OSMID=node_id,
                nodeCap=capacity,
                nodeX=float(i),
                nodeY=float(j),
                cellID=(i, j),
            )
            shelter_store.shelterCanByCell[i][j].append(node)
            candidate_nodes.append(node)
            node_id += 1
    pedestrian_store = FakePedestrianStore()
    pedestrian_nodes = [
        SimpleNamespace(OSMID=1, nodeX=0.0, nodeY=0.0, cellID=(0, 0)),
        SimpleNamespace(OSMID=2, nodeX=0.0, nodeY=1.0, cellID=(0, 1)),
    ]
    for pedestrian, node in zip(
        pedestrian_store.pedAgentList.values(),
        pedestrian_nodes,
    ):
        pedestrian.currNode = node
        pedestrian.atNode = True
    return SimpleNamespace(
        cellX=2,
        cellY=2,
        maxSpeed=10.0,
        stopTime=11,
        maximumShelterForecastDanger=0.6,
        address="test",
        timeStepMinutes=1.0,
        pedDS=pedestrian_store,
        mapDS=FakeMapStore(pedestrian_nodes + candidate_nodes),
        hazardDS=FakeHazardStore(),
        shelterDS=shelter_store,
        cellTracker=FakeCellTracker(),
        forceTracker=None,
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
        np.testing.assert_allclose(observation.candidate_east_positions, (0, 0, 0.5, 0.5))
        np.testing.assert_allclose(observation.candidate_north_positions, (1, 0.5, 1, 0.5))
        _, global_features = observation.policy_features()
        self.assertAlmostEqual(
            float(global_features[GLOBAL_FEATURE_NAMES.index("hazard_footprint_fraction")]),
            0.0,
        )
        self.assertAlmostEqual(
            float(global_features[GLOBAL_FEATURE_NAMES.index("wind_east_velocity_fraction")]),
            0.0,
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
        self.assertEqual(observation.candidate_north_positions[0], 1.0)

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

    def test_forecast_danger_is_observed_without_censoring_the_action(self):
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
            (True, True, True, True),
        )
        self.assertGreater(float(observation.candidate_forecast_danger[1]), 0.0)

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
        self.assertEqual(tuple(components.shape), (1, 1))
        self.assertEqual(tuple(state[0].shape), (1, policy.temporal_dim))
        torch.testing.assert_close(scalar, components.sum(dim=-1))

    def test_staged_dynamics_and_continuation_heads_are_compact(self):
        torch.manual_seed(103)
        policy = EvacPolicy(
            2,
            1,
            1,
            d_global=3,
            d_candidate=2,
            d_momentum=9,
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
        self.assertEqual(tuple(first_output.natural_outcomes.shape), (1, 3, 9))
        self.assertTrue(torch.all(first_output.natural_outcomes >= -1.0))
        self.assertTrue(torch.all(first_output.natural_outcomes <= 1.0))
        self.assertFalse(
            any(
                name.startswith(("causal_outcome_heads.", "improvement_"))
                for name, _ in policy.named_parameters()
            )
        )

    def test_nmcc_actor_and_continuation_fast_paths_match_full_forward(self):
        torch.manual_seed(107)
        policy = EvacPolicy(
            2,
            1,
            1,
            d_global=3,
            d_candidate=2,
            d_momentum=9,
        )
        graph = fit_gnn(
            torch.randn(4, 2),
            torch.randn(4, 1),
            torch.randn(4, 1),
            torch.randn(3),
            candidate_cell_index=torch.tensor([0, 1, 3]),
            candidate_features=torch.randn(3, 2),
        )
        momentum = torch.randn(9)
        full = policy.forward_nmcc_recurrent(
            graph,
            None,
            momentum,
        )
        (
            actor_logits,
            prior_logits,
            learned_residual,
            actor_state,
        ) = policy.forward_nmcc_actor_recurrent(
            graph,
            None,
            momentum,
        )
        continuation, continuation_state = (
            policy.forward_nmcc_continuation_recurrent(
                graph,
                None,
                momentum,
            )
        )
        torch.testing.assert_close(actor_logits, full.logits)
        torch.testing.assert_close(prior_logits, full.prior_logits)
        torch.testing.assert_close(learned_residual, full.learned_residual)
        torch.testing.assert_close(
            continuation,
            full.nmcc_continuation_value,
        )
        for fast, expected in zip(actor_state, full.recurrent_state):
            torch.testing.assert_close(fast, expected)
        for fast, expected in zip(continuation_state, full.recurrent_state):
            torch.testing.assert_close(fast, expected)

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
            actor_prior="active_population",
            actor_prior_scale=1.0,
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
            actor_prior="active_population",
            actor_prior_scale=1.0,
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

    def test_zero_residual_initialization_exactly_ranks_objective_greedy(self):
        policy = EvacPolicy(
            2,
            1,
            3,
            d_global=5,
            d_candidate=5,
            actor_prior="objective_greedy",
            actor_prior_scale=1.0,
            actor_prior_feature_index=4,
        )
        candidate_features = torch.zeros(4, 5)
        candidate_features[:, 4] = torch.tensor((0.2, 0.7, 0.4, 0.1))
        graph = fit_gnn(
            torch.zeros(4, 2),
            torch.zeros(4, 1),
            torch.zeros(4, 3),
            torch.zeros(5),
            edge_index=torch.as_tensor(grid_edge_index(2, 2)),
            candidate_cell_index=torch.arange(4),
            candidate_features=candidate_features,
        )
        with torch.no_grad():
            output = policy.forward_nmcc_recurrent(graph)
        self.assertEqual(int(torch.argmax(output.logits, dim=-1).item()), 1)
        torch.testing.assert_close(output.learned_residual, torch.zeros(1, 4))

    def test_learned_residual_is_the_unbounded_linear_actor_readout(self):
        policy = EvacPolicy(
            2,
            1,
            3,
            d_global=5,
            actor_prior="active_population",
            actor_prior_scale=1.0,
        )
        with torch.no_grad():
            policy.actor_cell.weight.zero_()
            policy.actor_cell.bias.fill_(5.5)
        graph = fit_gnn(
            torch.randn(4, 2),
            torch.randn(4, 1),
            torch.randn(4, 3),
            torch.randn(5),
            edge_index=torch.as_tensor(grid_edge_index(2, 2)),
        )
        with torch.no_grad():
            _, _, residual = policy.forward_with_residual(graph)
        torch.testing.assert_close(residual, torch.full_like(residual, 5.5))

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
        self.assertGreater(float(x_ped.grad[[1, 3, 4]].abs().sum()), 0.0)
        # With the CA's Moore topology and two message layers, even the
        # opposite diagonal is reachable from cell zero.
        self.assertGreater(float(x_ped.grad[8].abs().sum()), 0.0)

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
    def test_categorical_rollout_target_is_conservative_and_keeps_ungraded_mass(self):
        prior = np.asarray((0.40, 0.30, 0.20, 0.10), dtype=np.float64)
        target = NPI.improvement_target(
            prior,
            np.ones(4, dtype=bool),
            np.asarray((0, 2), dtype=np.int64),
            np.asarray((0.2, 0.5), dtype=np.float64),
            epsilon=0.05,
            eta_min=0.03,
        )
        self.assertLessEqual(target.kl_to_old, 0.05 + 1e-8)
        self.assertAlmostEqual(target.target[1], prior[1], places=10)
        self.assertAlmostEqual(target.target[3], prior[3], places=10)
        self.assertGreater(target.target[2], prior[2])

    def test_compact_momentum_separates_traffic_relief_from_hazard_growth(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            shelter_action_interval=5,
        )
        previous = make_observation(active=(4, 4, 0, 0), danger=(0.1, 0.1, 0.0, 0.0))
        current = make_observation(active=(3, 3, 0, 0), danger=(0.4, 0.3, 0.0, 0.0))
        current = replace(current, simulation_time=6)
        previous.congestion_speed_ratio_by_cell[:] = (0.4, 0.5, 1.0, 1.0)
        current.congestion_speed_ratio_by_cell[:] = (0.8, 0.9, 1.0, 1.0)
        previous.mean_route_time_by_cell[:] = (8.0, 7.0, 0.0, 0.0)
        current.mean_route_time_by_cell[:] = (4.0, 3.0, 0.0, 0.0)
        previous.forecast_danger_by_cell[:] = (0.1, 0.1, 0.0, 0.0)
        current.forecast_danger_by_cell[:] = (0.5, 0.4, 0.0, 0.0)
        values = bridge._momentum_array(previous, current)
        self.assertEqual(values.shape, (4,))
        self.assertGreater(values[0], 0.0)
        self.assertGreater(values[1], 0.0)
        self.assertGreater(values[2], 0.0)
        self.assertLess(values[3], 0.0)

    def test_nmcc_backward_targets_span_all_eight_installations(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=8,
            gamma=1.0,
        )
        rewards = torch.arange(1.0, 9.0).reshape(8, 1)
        values = torch.zeros_like(rewards)
        dones = torch.zeros(8)
        dones[-1] = 1.0
        durations = torch.ones(8)
        actor_targets, critic_targets = bridge._component_credit_targets(
            rewards,
            values,
            dones,
            durations,
        )
        expected = torch.tensor(
            [[36.0], [35.0], [33.0], [30.0], [26.0], [21.0], [15.0], [8.0]]
        )
        torch.testing.assert_close(actor_targets, expected)
        torch.testing.assert_close(critic_targets, expected)

    def test_learning_and_categorical_exploration_schedules_decay_end_to_end(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            actor_lr=1e-3,
            critic_lr=1e-3,
            entropy_coef=0.01,
            entropy_coef_end=5e-4,
            exploration_decay_updates=12,
            action_temperature_start=1.5,
            action_temperature_end=0.35,
            learning_rate_schedule="cosine",
            lr_warmup_updates=0,
            lr_decay_updates=12,
            actor_lr_min_fraction=0.1,
            critic_lr_min_fraction=0.1,
            nmcc_policy_improvement=True,
            exploration_rate_start=0.3,
            exploration_rate_end=0.02,
            actor_prior="none",
            actor_prior_scale=0.0,
        )

        self.assertAlmostEqual(bridge._current_entropy_coef(), 0.01)
        self.assertAlmostEqual(bridge._current_action_temperature(), 1.5)
        self.assertAlmostEqual(bridge._current_exploration_rate(), 0.3)
        actor_lr, critic_lr = bridge._apply_learning_rate_schedule()
        self.assertAlmostEqual(actor_lr, 1e-3)
        self.assertAlmostEqual(critic_lr, 1e-3)

        bridge.actor_rollout_updates = 6
        bridge.rollout_updates_completed = 6
        self.assertAlmostEqual(bridge._current_entropy_coef(), 0.00525)
        self.assertAlmostEqual(bridge._current_action_temperature(), 0.925)
        self.assertAlmostEqual(bridge._current_exploration_rate(), 0.16)
        actor_lr, critic_lr = bridge._apply_learning_rate_schedule()
        self.assertAlmostEqual(actor_lr, 5.5e-4)
        self.assertAlmostEqual(critic_lr, 5.5e-4)

        bridge.actor_rollout_updates = 12
        bridge.rollout_updates_completed = 12
        self.assertAlmostEqual(bridge._current_entropy_coef(), 5e-4)
        self.assertAlmostEqual(bridge._current_action_temperature(), 0.35)
        self.assertAlmostEqual(bridge._current_exploration_rate(), 0.02)
        actor_lr, critic_lr = bridge._apply_learning_rate_schedule()
        self.assertAlmostEqual(actor_lr, 1e-4)
        self.assertAlmostEqual(critic_lr, 1e-4)
        self.assertAlmostEqual(bridge.actor_optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(bridge.critic_optimizer.param_groups[0]["lr"], 1e-4)

    def test_evaluation_policy_has_no_training_exploration_mixture(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            nmcc_policy_improvement=True,
            exploration_rate_start=0.4,
            exploration_rate_end=0.1,
            action_temperature_start=1.5,
            action_temperature_end=0.8,
            actor_prior="none",
            actor_prior_scale=0.0,
        )
        bridge.train_mode = False
        self.assertEqual(bridge._current_exploration_rate(), 0.0)

    def test_each_curriculum_stage_resets_and_honors_longer_decay_horizon(self):
        common = dict(
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            actor_lr=1e-3,
            critic_lr=2e-3,
            learning_rate_schedule="cosine",
            lr_warmup_updates=0,
            lr_decay_updates=12,
            actor_lr_min_fraction=0.1,
            critic_lr_min_fraction=0.2,
            nmcc_policy_improvement=True,
            exploration_rate_start=0.3,
            exploration_rate_end=0.02,
            exploration_schedule="exponential_half_life",
            exploration_decay_updates=3,
            actor_prior="none",
            actor_prior_scale=0.0,
            training_stage_total_updates=6,
        )
        start = RLBridge(
            make_core(),
            training_stage_id="P",
            training_stage_update_index=0,
            **common,
        )
        end = RLBridge(
            make_core(),
            training_stage_id="P",
            training_stage_update_index=5,
            **common,
        )
        self.assertAlmostEqual(start._current_exploration_rate(), 0.3)
        self.assertAlmostEqual(end._current_exploration_rate(), 0.02)
        self.assertAlmostEqual(start._apply_learning_rate_schedule()[0], 1e-3)
        self.assertAlmostEqual(start._apply_learning_rate_schedule()[1], 2e-3)
        cosine = 0.5 * (1.0 + math.cos(math.pi * 5.0 / 12.0))
        self.assertAlmostEqual(
            end._apply_learning_rate_schedule()[0],
            1e-3 * (0.1 + 0.9 * cosine),
        )
        self.assertAlmostEqual(
            end._apply_learning_rate_schedule()[1],
            2e-3 * (0.2 + 0.8 * cosine),
        )

    def test_loaded_optimizer_rate_resets_at_new_curriculum_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.pt")
            common = dict(
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                checkpoint_path=checkpoint,
                actor_lr=1e-3,
                critic_lr=2e-3,
                learning_rate_schedule="cosine",
                lr_warmup_updates=0,
                lr_decay_updates=5,
                actor_lr_min_fraction=0.1,
                critic_lr_min_fraction=0.2,
                nmcc_policy_improvement=True,
                training_stage_total_updates=6,
            )
            finished = RLBridge(
                make_core(),
                training_stage_id="P",
                training_stage_update_index=5,
                **common,
            )
            finished._apply_learning_rate_schedule()
            self.assertAlmostEqual(
                finished.actor_optimizer.param_groups[0]["lr"], 1e-4
            )
            finished._save_checkpoint()

            reset = RLBridge(
                make_core(),
                training_stage_id="P",
                training_stage_update_index=0,
                **common,
            )
            self.assertAlmostEqual(
                reset.actor_optimizer.param_groups[0]["lr"], 1e-3
            )
            self.assertAlmostEqual(
                reset.critic_optimizer.param_groups[0]["lr"], 2e-3
            )

    def test_policy_stage_uses_simulator_rollouts_not_learned_continuation(self):
        common = dict(
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            nmcc_policy_improvement=True,
            nmcc_pi_continuation_enabled=True,
            nmcc_pi_branch_horizon=5,
            shelter_action_interval=5,
            nmcc_natural_loss_coef=1.0,
            representation_mode="staged_transfer",
        )
        environment = RLBridge(
            make_core(), training_stage_id="E", **common
        )
        policy_improvement = RLBridge(
            make_core(), training_stage_id="P", **common
        )
        self.assertTrue(environment._stage_contract().train_dynamics)
        self.assertFalse(environment._stage_contract().train_actor)
        self.assertTrue(policy_improvement._stage_contract().train_actor)
        self.assertFalse(policy_improvement._stage_contract().use_continuation)
        self.assertFalse(policy_improvement._stage_contract().train_value)
        self.assertEqual(
            {
                name.split(".", 1)[0]
                for name, _ in policy_improvement.actor_named_parameters
            },
            set(policy_improvement.policy.ACTOR_HEAD_MODULES),
        )
        self.assertTrue(
            {
                name
                for name, _ in policy_improvement.representation_named_parameters
            }
            <= {name for name, _ in policy_improvement.critic_named_parameters}
        )
        self.assertFalse(
            {name for name, _ in policy_improvement.actor_named_parameters}
            & {name for name, _ in policy_improvement.critic_named_parameters}
        )
        metadata = policy_improvement._model_signature()["learner_hyperparameters"]
        self.assertEqual(
            metadata["actor_credit_target"],
            "kl_constrained_categorical_rollout_policy_improvement",
        )
        self.assertEqual(
            metadata["actor_baseline"],
            "objective_greedy_score_plus_learned_residual",
        )
        self.assertEqual(
            metadata["critic_target"],
            "not_used_by_rollout_policy_improvement",
        )

    def test_uniform_site_nmcc_credit_does_not_change_on_policy_critic_target(self):
        returns = torch.tensor([0.8, 0.5, 0.2])
        component_returns = returns.reshape(-1, 1)
        advantages = torch.tensor(
            [
                [0.4, -0.1],
                [-0.3, 0.2],
                [0.1, -0.5],
            ]
        )
        actions = torch.tensor([0, 1, 1])
        normalized, raw, critic, critic_components = (
            RLBridge._nmcc_actor_critic_targets(
                returns,
                component_returns,
                advantages,
                actions,
            )
        )
        torch.testing.assert_close(raw, torch.tensor([0.4, 0.2, -0.5]))
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(
            float(normalized.std(unbiased=False)), 1.0, places=6
        )
        torch.testing.assert_close(critic, returns)
        torch.testing.assert_close(critic_components, component_returns)

    def test_rollout_actor_has_no_learned_endpoint_grader(self):
        self.assertFalse(
            hasattr(RLBridge, "_random_reference_branch_advantage")
        )

    def test_complete_debt_keeps_panicked_but_does_not_price_realized_draws(self):
        core = make_core()
        core.pedDS.currTime = 7
        core.pedDS.pedAgentList[0].panicked = True
        core.pedDS.result["casualty"] = 2
        estimate = estimate_remaining_time_to_safety(
            core,
            episode_horizon=20,
            pedestrian_ids={0},
            include_panicked=True,
            include_casualties=True,
        )
        expected = 1.0 * 13.0 * float(core.timeStepMinutes)
        self.assertAlmostEqual(estimate.plain_person_minutes, expected)
        self.assertGreaterEqual(estimate.hazard_exposure_person_minutes, 0.0)

    def test_single_update_stage_uses_starting_rates(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            training_stage_id="P",
            training_stage_update_index=0,
            training_stage_total_updates=1,
            actor_lr=1e-3,
            critic_lr=2e-3,
            learning_rate_schedule="cosine",
            actor_lr_min_fraction=0.1,
            critic_lr_min_fraction=0.2,
            nmcc_policy_improvement=True,
            exploration_rate_start=0.3,
            exploration_rate_end=0.02,
            actor_prior="none",
            actor_prior_scale=0.0,
        )
        self.assertAlmostEqual(bridge._current_exploration_rate(), 0.3)
        self.assertAlmostEqual(bridge._apply_learning_rate_schedule()[0], 1e-3)
        self.assertAlmostEqual(bridge._apply_learning_rate_schedule()[1], 2e-3)

    def test_direct_controller_uses_neural_logits_not_prior_or_gate(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            actor_prior="none",
            actor_prior_scale=0.0,
        )
        actor_logits = torch.tensor([[0.1, 0.7, -0.2, 0.3]])
        step = SimpleNamespace(
            logits=actor_logits,
            prior_logits=torch.tensor([[9.0, -9.0, -9.0, -9.0]]),
        )
        mask = torch.tensor([True, True, True, True])
        selected = bridge._controller_logits(step, mask)
        torch.testing.assert_close(selected, actor_logits)
        self.assertEqual(int(selected.argmax(dim=-1).item()), 1)

    def test_clean_nmcc_improvement_is_enabled_without_auxiliary_controller(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            nmcc_policy_improvement=True,
        )
        self.assertTrue(bridge.nmcc_policy_improvement)
        self.assertFalse(bridge.policy.auxiliary_models_enabled)
        with self.assertRaisesRegex(ValueError, "alternative estimators"):
            RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                counterfactual_credit=True,
                nmcc_policy_improvement=True,
            )
        residual = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            nmcc_policy_improvement=True,
            actor_prior="objective_greedy",
            actor_prior_scale=1.0,
        )
        self.assertEqual(residual.actor_prior, "objective_greedy")

    def test_exact_nmcc_credit_keeps_negative_and_positive_action_effects(self):
        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            counterfactual_credit=True,
            counterfactual_weight=1.0,
            nmcc_joint_counterfactual_weight=1.0,
            nmcc_natural_loss_coef=0.0,
            actor_prior="none",
            actor_prior_scale=0.0,
        )
        transitions = [
            SimpleNamespace(
                action=torch.tensor([0]),
                counterfactual_advantage=torch.tensor([-0.2]),
                counterfactual_components=torch.tensor([-0.2]),
                counterfactual_reference_action=torch.tensor([1]),
                causal_outcome_target=torch.zeros(6),
            ),
            SimpleNamespace(
                action=torch.tensor([1]),
                counterfactual_advantage=torch.tensor([0.1]),
                counterfactual_components=torch.tensor([0.1]),
                counterfactual_reference_action=torch.tensor([0]),
                causal_outcome_target=torch.zeros(6),
            ),
        ]
        optimized, diagnostics = bridge._blend_counterfactual_advantages(
            torch.tensor([0.4, -0.4]), transitions
        )
        self.assertLess(float(optimized[0]), 0.0)
        self.assertGreater(float(optimized[1]), 0.0)
        self.assertEqual(diagnostics["counterfactual_fraction"], 1.0)
        self.assertEqual(diagnostics["counterfactual_negative_fraction"], 0.5)
        self.assertFalse(bridge.policy.auxiliary_models_enabled)

    def test_uniform_learning_reward_scale_amplifies_signal_only(self):
        def make_bridge(scale):
            return RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                counterfactual_credit=True,
                counterfactual_weight=1.0,
                nmcc_joint_counterfactual_weight=1.0,
                learning_reward_scale=scale,
                advantage_scale_floor=0.05,
                actor_prior="none",
                actor_prior_scale=0.0,
            )

        def transitions(scale):
            return [
                SimpleNamespace(
                    action=torch.tensor([0]),
                    counterfactual_advantage=torch.tensor([-0.02 * scale]),
                    counterfactual_components=torch.tensor([-0.02 * scale]),
                    counterfactual_reference_action=torch.tensor([1]),
                    causal_outcome_target=torch.zeros(6),
                ),
                SimpleNamespace(
                    action=torch.tensor([1]),
                    counterfactual_advantage=torch.tensor([0.01 * scale]),
                    counterfactual_components=torch.tensor([0.01 * scale]),
                    counterfactual_reference_action=torch.tensor([0]),
                    causal_outcome_target=torch.zeros(6),
                ),
            ]

        base = make_bridge(1.0)
        scaled = make_bridge(10.0)
        base_credit, base_diagnostics = base._blend_counterfactual_advantages(
            torch.zeros(2), transitions(1.0)
        )
        scaled_credit, scaled_diagnostics = scaled._blend_counterfactual_advantages(
            torch.zeros(2), transitions(10.0)
        )

        self.assertAlmostEqual(
            scaled_diagnostics["counterfactual_advantage_sd"],
            10.0 * base_diagnostics["counterfactual_advantage_sd"],
        )
        torch.testing.assert_close(scaled_credit, base_credit)
        self.assertAlmostEqual(
            scaled_diagnostics[
                "counterfactual_remaining_risk_time_difference_mean"
            ],
            base_diagnostics[
                "counterfactual_remaining_risk_time_difference_mean"
            ],
            places=5,
        )

    def test_exact_nmcc_has_no_missing_branch_or_partial_blend_fallback(self):
        with self.assertRaisesRegex(ValueError, "partial fallback blending"):
            RLBridge(
                make_core(),
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=2,
                counterfactual_credit=True,
                counterfactual_weight=0.5,
                nmcc_joint_counterfactual_weight=1.0,
            )

        bridge = RLBridge(
            make_core(),
            train_mode=True,
            deployment_strategy="rl",
            target_active_shelters=2,
            counterfactual_credit=True,
            counterfactual_weight=1.0,
            nmcc_joint_counterfactual_weight=1.0,
            actor_prior="none",
            actor_prior_scale=0.0,
        )
        transitions = [
            SimpleNamespace(
                action=torch.tensor([0]),
                counterfactual_advantage=torch.tensor([0.1]),
                counterfactual_components=torch.tensor([0.1]),
                counterfactual_reference_action=torch.tensor([1]),
                causal_outcome_target=torch.zeros(6),
            ),
            SimpleNamespace(
                action=torch.tensor([1]),
                counterfactual_advantage=None,
                counterfactual_components=torch.tensor([0.0]),
                counterfactual_reference_action=torch.tensor([0]),
                causal_outcome_target=torch.zeros(6),
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "no same-noise"):
            bridge._blend_counterfactual_advantages(
                torch.tensor([0.4, -0.4]), transitions
            )

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

    def test_initial_only_commits_at_time_zero_and_executes_on_common_schedule(self):
        core = make_core()
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="initial_only",
            target_active_shelters=2,
            shelter_action_interval=5,
            first_decision_time=5,
        )

        def stable_initial_ranking(observation):
            values = np.full(observation.number_of_actions, -np.inf)
            feasible = np.flatnonzero(observation.action_mask)
            values[feasible] = -feasible.astype(float)
            return values

        bridge._objective_greedy_values = stable_initial_ranking
        schedule = bridge.precommit_static_schedule()
        self.assertEqual(schedule, (("100", 0), ("101", 1)))
        self.assertEqual(bridge.deployments_made, 0)
        self.assertEqual(bridge.deployment_times, [])
        self.assertEqual(core.shelterDS.shelterList, {})
        self.assertEqual(core.cellTracker.added, [])

        # Changing the scorer after commitment cannot change the open-loop
        # execution order because no later observation is used to re-rank it.
        bridge._objective_greedy_values = lambda observation: np.arange(
            observation.number_of_actions,
            dtype=float,
        )
        self.assertEqual(bridge.step(simulation_time=1)["decision_made"], 0)
        first = bridge.step(simulation_time=5)
        second = bridge.step(simulation_time=10)
        self.assertEqual(first["candidate_osm_node_id"], "100")
        self.assertEqual(second["candidate_osm_node_id"], "101")
        self.assertEqual(bridge.deployment_times, [5, 10])
        self.assertEqual(bridge.remaining_deployments, 0)

    def test_benchmark_objective_does_not_masquerade_as_learning_reward(self):
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
        self.assertTrue(np.isnan(first["objective_episode_score"]))

        del core.pedDS.pedAgentList[0]
        core.pedDS.result["evacuated"] = 1
        second = bridge.step(simulation_time=6)
        self.assertAlmostEqual(second["reward_safe"], 0.0)
        self.assertEqual(second["reward_evacuation_time"], 0.0)
        self.assertAlmostEqual(
            second["reward"],
            second["reward_evacuation_time"]
            + second["reward_hazard_exposure"],
        )
        self.assertAlmostEqual(second["reward_risk_time"], second["reward"])
        self.assertEqual(second["completed_action"], 0)
        self.assertTrue(np.isnan(second["objective_episode_score"]))

        core.pedDS.pedAgentList.clear()
        core.pedDS.result["evacuated"] = 2
        terminal = bridge.step(simulation_time=10, is_terminal=True)
        self.assertEqual(terminal["completed_action"], 1)
        diagnostics = bridge.end_episode()
        self.assertEqual(diagnostics["episode_return"], 0.0)
        self.assertAlmostEqual(
            diagnostics["episode_return"], diagnostics["forward_learning_reward"]
        )
        self.assertGreater(diagnostics["objective_episode_score"], 0.0)
        self.assertGreater(diagnostics["objective_normalized_cost"], 0.0)
        self.assertGreater(
            diagnostics["realized_hazard_weighted_time_to_safety"], 0.0
        )

    def test_static_policy_with_no_installation_has_zero_installation_reward(self):
        core = make_core()
        core.cellTracker.dangerLevelByCell[:] = 1.0
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
        self.assertGreater(diagnostics["objective_episode_score"], 0.0)
        self.assertGreater(diagnostics["objective_normalized_cost"], 0.0)

    def test_terminal_progress_closes_the_forecast_debt_transition(self):
        core = make_core()
        core.cellTracker.dangerLevelByCell[:] = 1.0
        core.maximumShelterForecastDanger = 1.0
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
        self.assertTrue(np.isfinite(diagnostics["episode_return"]))
        self.assertAlmostEqual(
            diagnostics["episode_return"], diagnostics["forward_learning_reward"]
        )
        self.assertGreater(diagnostics["objective_episode_score"], 0.0)
        self.assertGreater(
            diagnostics["objective_terminal_residual_person_time"], 0.0
        )
        self.assertAlmostEqual(diagnostics["reward_accounting_gap"], 0.0)

    def test_casualties_after_budget_exhaustion_are_reported_not_rewarded(self):
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
        self.assertAlmostEqual(terminal["reward_casualty"], 0.0)
        diagnostics = bridge.end_episode()
        self.assertEqual(bridge.objective_latest_outcome.casualties, 2)
        self.assertGreaterEqual(
            diagnostics["objective_expected_casualty_person_time"], 0.0
        )

    def test_recurrent_rollout_uses_one_observation_per_decision_epoch(self):
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
                [1, 1],
            )
            self.assertEqual(bridge.observation_frames_seen, 2)
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
            self.assertEqual(result["optimizer_updated"], 1.0)
            self.assertGreaterEqual(result["gradient_norm"], 0.0)
            self.assertTrue(np.isfinite(result["approximate_kl"]))
            self.assertGreaterEqual(result["behavior_policy_entropy"], 0.0)
            self.assertLessEqual(result["behavior_policy_entropy"], 1.0 + 1e-6)
            self.assertIn("actor_learning_rate_before_update", result)
            self.assertLessEqual(result["learning_rate"], bridge.lr)
            self.assertTrue(os.path.exists(checkpoint))
            self.assertTrue(os.path.exists(diagnostics))
            with open(diagnostics, newline="") as handle:
                header = next(csv.reader(handle))
            self.assertIn("approximate_kl", header)
            self.assertIn("value_loss", header)
            self.assertNotIn("rollout_pi_target_coverage", header)

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

    def test_population_matched_decisions_use_absolute_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            core = make_core()
            core.requirePopulationMatchedShelterCapacity = True
            bridge = RLBridge(
                core,
                train_mode=True,
                deployment_strategy="rl",
                target_active_shelters=3,
                shelter_action_interval=5,
                first_decision_time=5,
                checkpoint_path=os.path.join(directory, "policy.pt"),
                diagnostics_path=os.path.join(directory, "ppo.csv"),
            )
            self.assertFalse(bridge._decision_due(4))
            self.assertTrue(bridge._decision_due(5))
            bridge.deployments_made = 1
            bridge.last_deployment_time = 7
            self.assertFalse(bridge._decision_due(9))
            self.assertTrue(bridge._decision_due(10))

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
            self.assertGreaterEqual(second_result["gradient_norm"], 0.0)
            self.assertEqual(second_result["transitions"], 4.0)
            self.assertEqual(second_result["rollout_episodes_pending"], 0.0)
            self.assertEqual(len(resumed.rollout_traj), 0)


class AccountingAndDiagnosticsTests(unittest.TestCase):
    def test_initial_only_starts_with_the_common_initial_shelter_count(self):
        self.assertEqual(
            Core._initial_shelter_volume("initial_only", 2, 10),
            2,
        )
        self.assertEqual(Core._initial_shelter_volume("none", 2, 10), 10)
        for strategy in (
            "rl",
            "objective_greedy",
            "heuristic",
            "hazard_weighted",
            "accessibility_deficit",
            "random",
            "static_greedy",
            "rl_precommit",
            "initial_only",
        ):
            self.assertEqual(
                Core._initial_shelter_volume(strategy, 2, 10),
                2,
            )

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
        core._apply_configuration_overrides(
            {
                "entropyCoefficientStart": 0.01,
                "entropyCoefficientEnd": 0.0005,
                "explorationDecayUpdates": 12,
                "actionTemperatureStart": 1.5,
                "actionTemperatureEnd": 0.35,
            }
        )
        core._validate_effective_configuration()
        self.assertEqual(core.criticLearningRate, core.learningRate)
        self.assertEqual(core.actorPpoEpochs, 1)
        self.assertEqual(core.criticPpoEpochs, 4)
        exploration = core._effective_configuration()["exploration_schedule"]
        self.assertEqual(exploration["entropy_start"], 0.01)
        self.assertEqual(exploration["entropy_end"], 0.0005)
        self.assertEqual(exploration["temperature_start"], 1.5)
        self.assertEqual(exploration["temperature_end"], 0.35)
        self.assertEqual(exploration["decay_updates"], 12)

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
            for strategy, reward, casualty in (
                ("rl", 1.0, 0),
                ("objective_greedy", 0.5, 1),
            ):
                rows.append(
                    {
                        "replication": replication,
                        "deployment_strategy": strategy,
                        "scenario_seed": 100 + replication,
                        "policy_seed": 7,
                        "objective_normalized_cost": 1.0 - reward / 10.0,
                        "safe_completed": 10,
                        "shelter_evacuated": 10,
                        "arrival": 0,
                        "casualty": casualty,
                        "expected_casualties": float(casualty),
                        "unfinished": 0,
                        "restricted_mean_time_to_safety": 5.0,
                        "mean_safe_completion_time": 5.0,
                        "mean_evacuation_time": 5.0,
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
        verification = _verify_matched_interface(
            rows,
            ("rl", "objective_greedy"),
        )
        self.assertTrue(verification["verified"])
        serialized_rows = [dict(row) for row in rows]
        for row in serialized_rows:
            row["random_stream_seeds"] = repr(row["random_stream_seeds"])
        self.assertTrue(
            _verify_matched_interface(
                serialized_rows,
                ("rl", "objective_greedy"),
            )["verified"]
        )
        analysis = _paired_analysis(rows, launch_seed=3, draws=200)
        objective_row = next(
            row for row in analysis
            if row["metric"] == "objective_normalized_cost"
        )
        casualty_row = next(row for row in analysis if row["metric"] == "casualty")
        self.assertGreater(objective_row["mean_rl_improvement"], 0.0)
        self.assertGreater(casualty_row["mean_rl_improvement"], 0.0)
        self.assertEqual(
            objective_row["inference_scope"],
            "conditional_on_one_fixed_trained_policy",
        )
        self.assertIsNone(objective_row["two_sided_randomization_p"])

        unequal_capacity = [dict(row) for row in rows]
        unequal_capacity[0]["total_shelter_capacity"] = 999.0
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            _verify_matched_interface(
                unequal_capacity,
                ("rl", "objective_greedy"),
            )

    def test_benchmark_analysis_uses_full_episode_objective_for_every_comparator(self):
        self.assertEqual(
            _evaluation_metric(
                {"objective_normalized_cost": 2.0},
                "objective_normalized_cost",
            ),
            2.0,
        )
        rows = []
        common = {
            "safe_completed": 8,
            "expected_casualties": 1.0,
            "casualty": 1,
            "unfinished": 1,
            "restricted_mean_time_to_safety": 4.0,
        }
        for replication in (1, 2):
            for policy_replication, objective in ((1, 0.8), (2, 0.6)):
                rows.append(
                    {
                        **common,
                        "replication": replication,
                        "policy_replication": policy_replication,
                        "deployment_strategy": "rl",
                        "objective_normalized_cost": objective,
                    }
                )
            for strategy, objective in (
                ("objective_greedy", 1.0),
                ("random", 1.2),
            ):
                rows.append(
                    {
                        **common,
                        "replication": replication,
                        "policy_replication": 0,
                        "deployment_strategy": strategy,
                        "objective_normalized_cost": objective,
                    }
                )
        analysis = _benchmark_analysis(
            rows,
            launch_seed=5,
            draws=100,
            benchmark_strategies=("objective_greedy", "random"),
        )
        self.assertEqual(len(analysis), 12)
        primary = next(
            row
            for row in analysis
            if row["benchmark_strategy"] == "objective_greedy"
            and row["metric"] == "objective_normalized_cost"
        )
        self.assertEqual(
            primary["comparison_role"],
            "primary_myopic_comparator",
        )
        self.assertAlmostEqual(primary["rl_mean"], 0.7)
        self.assertAlmostEqual(primary["benchmark_mean"], 1.0)
        self.assertAlmostEqual(primary["mean_rl_improvement"], 0.3)

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
                        "behavior_policy_entropy": 1.0,
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

    def test_rollout_pi_convergence_has_no_retired_gate_or_kl_limit(self):
        rows = []
        for episode in range(1, 13):
            update = episode % 2 == 0
            rows.append(
                {
                    "policy_replication": 1,
                    "replication": episode,
                    "episode_return": 0.25,
                    "behavior_policy_entropy": 0.8,
                    "rollout_pi_actor_kl": 0.2 if update else 0.0,
                    "actor_gradient_norm": 0.2 if update else 0.0,
                    "policy_loss": 0.4 if update else 0.0,
                    "value_loss": 0.0,
                    "optimizer_updated": float(update),
                    "rollout_episodes_pending": 0.0,
                    "rollout_pi_actor_enabled": float(update),
                    "rollout_pi_target_kl_before": 1.0 if update else 0.0,
                    "rollout_pi_target_kl_after": 0.5 if update else 0.0,
                    "actor_policy_entropy_after": 0.7 if update else 0.0,
                    "rollout_pi_target_coverage": 1.0 if update else 0.0,
                }
            )
        audit = _training_convergence(
            rows,
            minimum_episodes=12,
            window_fraction=0.25,
            trend_threshold=0.5,
            shift_threshold=0.5,
            target_kl=0.03,
        )
        self.assertTrue(audit["all_policies_converged"])
        policy = audit["policies"][0]
        self.assertEqual(
            policy["convergence_mode"],
            "rollout_policy_iteration_tail_stationarity",
        )
        self.assertTrue(policy["rollout_pi_fit_diagnostics_finite"])
        self.assertEqual(policy["rollout_pi_target_kl_improvement_fraction"], 1.0)

    def test_performance_assessment_never_claims_superiority_when_interval_crosses_zero(self):
        inconclusive = _performance_assessment(
            [
                {
                    "metric": "objective_normalized_cost",
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
                    "metric": "objective_normalized_cost",
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
                    "metric": "objective_normalized_cost",
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
                        "benchmark_mean": 0.1,
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
