import time
import unittest
from types import SimpleNamespace

import numpy as np

from DecisionInterface import (
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    MOMENTUM_FEATURE_NAMES,
    PED_FEATURE_NAMES,
    OutcomeSnapshot,
    RegionalObservation,
    RegionalObservationBuilder,
)
from RLBridge import RLBridge
from RolloutPolicyIteration import momentum_array
from nmcc_testbed import TestbedCore


def _observation(**overrides):
    values = dict(
        decision_index=0,
        simulation_time=4,
        horizon=20,
        initial_population=100,
        remaining_deployments=1,
        maximum_deployments=1,
        maximum_speed=10.0,
        active_by_cell=np.asarray((20.0,), dtype=np.float32),
        mean_speed_by_cell=np.asarray((2.0,), dtype=np.float32),
        danger_by_cell=np.asarray((0.2,), dtype=np.float32),
        remaining_capacity_by_cell=np.asarray((0.0,), dtype=np.float32),
        deployable_capacity_by_cell=np.asarray((10.0,), dtype=np.float32),
        candidate_count_by_cell=np.asarray((1.0,), dtype=np.float32),
        action_mask=np.asarray((True,), dtype=np.bool_),
        outcome=OutcomeSnapshot(80, 0, 0, 80, 20, 24.0),
        susceptible_population_by_cell=np.asarray((20.0,), dtype=np.float32),
        threatened_population_by_cell=np.asarray((12.0,), dtype=np.float32),
        front_arrival_probability_by_cell=np.asarray((0.6,), dtype=np.float32),
        front_lead_time_fraction_by_cell=np.asarray((0.4,), dtype=np.float32),
        candidate_threat_capture_fraction=np.asarray((0.06,), dtype=np.float32),
        candidate_front_approach_fraction=np.asarray((1.0,), dtype=np.float32),
        candidate_front_arrival_probability=np.asarray((0.6,), dtype=np.float32),
        candidate_front_lead_time_fraction=np.asarray((0.4,), dtype=np.float32),
        threatened_population_fraction=0.12,
    )
    values.update(overrides)
    return RegionalObservation(**values)


class V64FrontFeatureTests(unittest.TestCase):
    def test_schema_widths_and_tensor_shapes(self):
        observation = _observation()
        cells, global_features = observation.policy_features()
        candidates = observation.candidate_features()
        self.assertEqual(len(PED_FEATURE_NAMES), 12)
        self.assertEqual(len(HAZARD_FEATURE_NAMES), 12)
        self.assertEqual(len(CELL_FEATURE_NAMES), 38)
        self.assertEqual(len(GLOBAL_FEATURE_NAMES), 19)
        self.assertEqual(len(MOMENTUM_FEATURE_NAMES), 7)
        self.assertEqual(len(CANDIDATE_FEATURE_NAMES), 21)
        self.assertEqual(cells.shape, (1, 38))
        self.assertEqual(global_features.shape, (19,))
        self.assertEqual(candidates.shape, (1, 21))

    def test_hand_cdf_threat_and_candidate_capture_arithmetic(self):
        cdf = np.asarray(((0.0,), (0.1,), (0.3,), (0.5,), (0.6,)))

        class HazardStore:
            hazardList = {
                0: SimpleNamespace(
                    active=True,
                    sourceNode=SimpleNamespace(nodeX=100.0, nodeY=0.0),
                )
            }

            @staticmethod
            def belief_overrun_first_passage_by_cell(*args, **kwargs):
                del args, kwargs
                return {
                    "crossing_cdf_by_step": cdf,
                    "conditional_expected_first_crossing_steps": np.asarray((2.0,)),
                }

        builder = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        builder.core = SimpleNamespace(
            shelterActionInterval=1,
            nmccFrontForecastIntervals=4,
            forecastEnsembleSize=8,
            panicDangerThreshold=3,
            maxSpeed=10.0,
            timeStepMinutes=1.0,
            pedDS=SimpleNamespace(panic_danger_threshold=3),
            hazardDS=HazardStore(),
            hazard_regime=None,
        )
        builder.nx = builder.ny = builder.number_of_cells = 1
        builder.horizon = 10
        builder.initial_population = 100
        builder._region_centers_m = lambda: np.asarray(((0.0, 0.0),))
        builder._site_region_distances_m = lambda node: np.asarray((20.0,))
        arrivals = (((4, 20),),)
        front, arrival, lead, susceptible, threatened, window = builder._front_layer(
            np.asarray((0.0,), dtype=np.float32), 0, arrivals
        )
        capture, approach, candidate_arrival, candidate_lead = (
            builder._candidate_front_features(
                (("site", 0, 10.0, 0.0, 0.0),),
                (SimpleNamespace(OSMID=1, nodeX=20.0, nodeY=0.0),),
                front,
                arrivals,
                lead,
                window,
            )
        )
        np.testing.assert_allclose(susceptible, (20.0,))
        np.testing.assert_allclose(threatened, (12.0,))
        np.testing.assert_allclose(arrival, (0.6,))
        np.testing.assert_allclose(capture, (0.06,), atol=1e-7)
        np.testing.assert_allclose(approach, (1.0,))
        np.testing.assert_allclose(candidate_arrival, (0.6,))
        np.testing.assert_allclose(candidate_lead, lead)

    def test_already_crossed_cell_has_full_threat_and_zero_lead(self):
        builder = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        builder.core = SimpleNamespace(
            shelterActionInterval=1,
            nmccFrontForecastIntervals=2,
            forecastEnsembleSize=8,
            panicDangerThreshold=3,
            maxSpeed=10.0,
            timeStepMinutes=1.0,
            pedDS=SimpleNamespace(panic_danger_threshold=3),
            hazardDS=SimpleNamespace(
                belief_overrun_first_passage_by_cell=lambda *args, **kwargs: {
                    "crossing_cdf_by_step": np.ones((3, 1)),
                    "conditional_expected_first_crossing_steps": np.zeros(1),
                }
            ),
            hazard_regime=None,
        )
        builder.nx = builder.ny = builder.number_of_cells = 1
        builder.horizon = 10
        _, arrival, lead, _, threatened, _ = builder._front_layer(
            np.asarray((0.8,), dtype=np.float32), 0, (((2, 7),),)
        )
        np.testing.assert_allclose(arrival, (1.0,))
        np.testing.assert_allclose(lead, (0.0,))
        np.testing.assert_allclose(threatened, (7.0,))

    def test_front_seed_depends_on_public_state_not_hidden_tape(self):
        builder = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        builder.nx = builder.ny = 1
        danger = np.asarray((0.25,), dtype=np.float32)
        builder.core = SimpleNamespace(
            hazard_regime=SimpleNamespace(public={"wind_warning": True}, hidden="a")
        )
        first = builder._belief_forecast_seed(danger, 5)
        builder.core.hazard_regime.hidden = "different-realized-tape"
        second = builder._belief_forecast_seed(danger, 5)
        self.assertEqual(first, second)

    def test_forecast_ablation_resets_front_forecasts_only(self):
        observation = _observation()
        ablated = RLBridge._without_forecast_information(observation)
        np.testing.assert_allclose(
            ablated.susceptible_population_by_cell,
            observation.susceptible_population_by_cell,
        )
        np.testing.assert_allclose(ablated.threatened_population_by_cell, 0.0)
        np.testing.assert_allclose(ablated.front_arrival_probability_by_cell, 0.0)
        np.testing.assert_allclose(ablated.front_lead_time_fraction_by_cell, 1.0)
        np.testing.assert_allclose(ablated.candidate_threat_capture_fraction, 0.0)
        np.testing.assert_allclose(ablated.candidate_front_approach_fraction, 0.5)
        np.testing.assert_allclose(ablated.candidate_front_arrival_probability, 0.0)
        np.testing.assert_allclose(ablated.candidate_front_lead_time_fraction, 1.0)
        self.assertEqual(ablated.threatened_population_fraction, 0.0)

    def test_front_feature_cost_is_below_one_second_at_16_by_16(self):
        core = TestbedCore(
            grid=16,
            cell_x=16,
            cell_y=16,
            population=32,
            group_size=4,
            hazard_count=1,
            stop_time=80,
            candidate_count=20,
        )
        builder = RegionalObservationBuilder(
            core,
            initial_population=32,
            horizon=80,
            maximum_deployments=2,
        )
        started = time.perf_counter()
        observation = builder.build(
            decision_index=0,
            simulation_time=0,
            remaining_deployments=2,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(observation.number_of_cells, 256)
        self.assertLess(elapsed, 1.0)

    def test_momentum_has_seventh_threat_reduction_component(self):
        previous = _observation(
            simulation_time=4,
            threatened_population_fraction=0.20,
        )
        current = _observation(
            simulation_time=8,
            threatened_population_fraction=0.12,
        )
        values = momentum_array(
            previous,
            current,
            initial_population=100,
            shelter_action_interval=4,
        )
        self.assertEqual(values.shape, (7,))
        self.assertAlmostEqual(float(values[-1]), 0.08, places=6)

    def test_distance_cache_is_finite_falls_back_and_keys_routing_version(self):
        nodes = (
            SimpleNamespace(OSMID=10, nodeX=0.0, nodeY=0.0),
            SimpleNamespace(OSMID=20, nodeX=10.0, nodeY=0.0),
        )
        calls = []

        def routing_tree(target):
            calls.append(target)
            return {10: 7.0}, {}

        map_store = SimpleNamespace(routing_version=0, _routing_tree=routing_tree)
        tracker = SimpleNamespace(
            getCellCenter=lambda cell: ((0.0, 0.0), (10.0, 0.0))[cell[1]]
        )
        builder = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        builder.core = SimpleNamespace(mapDS=map_store, cellTracker=tracker)
        builder.nx, builder.ny, builder.number_of_cells = 1, 2, 2
        builder.region_east_positions = np.asarray((0.0, 1.0))
        builder.region_north_positions = np.asarray((0.0, 0.0))
        builder._region_road_nodes = ((nodes[0],), (nodes[1],))
        builder._site_region_distance_cache = {}
        site = SimpleNamespace(OSMID=99, nodeX=20.0, nodeY=0.0)
        first = builder._site_region_distances_m(site)
        second = builder._site_region_distances_m(site)
        np.testing.assert_allclose(first, (7.0, 10.0))
        self.assertTrue(np.isfinite(first).all())
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        map_store.routing_version = 1
        third = builder._site_region_distances_m(site)
        np.testing.assert_allclose(third, first)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
