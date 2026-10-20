"""Focused contract tests for forward-looking shelter-overrun observations."""

import unittest
from types import SimpleNamespace

import numpy as np

from DecisionInterface import (
    CELL_FEATURE_NAMES,
    RegionalObservation,
    RegionalObservationBuilder,
    OutcomeSnapshot,
)
from Shelter import Shelter


def _observation(**overrides):
    cell_count = 2
    values = dict(
        decision_index=1,
        simulation_time=5,
        horizon=20,
        initial_population=100,
        remaining_deployments=2,
        maximum_deployments=4,
        maximum_speed=10.0,
        active_by_cell=np.asarray((25.0, 35.0), dtype=np.float32),
        mean_speed_by_cell=np.asarray((2.0, 3.0), dtype=np.float32),
        danger_by_cell=np.asarray((0.4, 0.6), dtype=np.float32),
        remaining_capacity_by_cell=np.asarray((3.0, 0.0), dtype=np.float32),
        deployable_capacity_by_cell=np.asarray((30.0, 30.0), dtype=np.float32),
        candidate_count_by_cell=np.ones(cell_count, dtype=np.float32),
        action_mask=np.ones(cell_count, dtype=np.bool_),
        outcome=OutcomeSnapshot(
            safe_completed=40,
            casualties=0,
            shelter_evacuated=40,
            ordinary_arrivals=0,
            active_population=60,
            risk_mass=90.0,
        ),
        operational_shelter_capacity_by_cell=np.asarray(
            (10.0, 20.0), dtype=np.float32
        ),
        sheltered_population_by_cell=np.asarray((4.0, 12.0), dtype=np.float32),
        enroute_reserved_population_by_cell=np.asarray(
            (3.0, 5.0), dtype=np.float32
        ),
        forecast_overrun_probability_by_cell=np.asarray(
            (0.25, 0.75), dtype=np.float32
        ),
        # Cell zero can be the next generic crossing, while cell one is the
        # next crossing among cells that actually hold an operational shelter.
        forecast_next_shelter_overrun_probability_by_cell=np.asarray(
            (0.0, 0.8), dtype=np.float32
        ),
        forecast_overrun_lead_time_fraction_by_cell=np.asarray(
            (0.2, 0.4), dtype=np.float32
        ),
        forecast_reevacuation_population_by_cell=np.asarray(
            (1.0, 9.0), dtype=np.float32
        ),
        forecast_enroute_reroute_population_by_cell=np.asarray(
            (0.75, 3.75), dtype=np.float32
        ),
    )
    values.update(overrides)
    return RegionalObservation(**values)


class OverrunObservationContractTests(unittest.TestCase):
    def test_cell_tensor_identifies_the_next_operational_shelter(self):
        observation = _observation()
        cell_features, _ = observation.policy_features()

        next_index = CELL_FEATURE_NAMES.index(
            "forecast_next_shelter_overrun_probability"
        )
        np.testing.assert_allclose(cell_features[:, next_index], (0.0, 0.8))

        operational_index = CELL_FEATURE_NAMES.index(
            "operational_shelter_capacity_fraction"
        )
        sheltered_index = CELL_FEATURE_NAMES.index(
            "sheltered_population_fraction"
        )
        enroute_index = CELL_FEATURE_NAMES.index(
            "enroute_reserved_population_fraction"
        )
        np.testing.assert_allclose(
            cell_features[:, operational_index], (0.10, 0.20)
        )
        np.testing.assert_allclose(
            cell_features[:, sheltered_index], (0.04, 0.12)
        )
        np.testing.assert_allclose(
            cell_features[:, enroute_index], (0.03, 0.05)
        )

    def test_regional_capacity_separates_residents_reservations_and_failed_sites(self):
        open_shelter = Shelter(
            0, SimpleNamespace(), (0, 0), cap=10, flow=4, status=Shelter.OPEN
        )
        full_shelter = Shelter(
            1, SimpleNamespace(), (0, 1), cap=5, flow=5, status=Shelter.FULL
        )
        failed_shelter = Shelter(
            2, SimpleNamespace(), (0, 1), cap=20, flow=0, status=Shelter.OVERRUN
        )
        reservations = {0: 3, 1: 0, 2: 7}
        shelter_store = SimpleNamespace(
            shelterByCell=[[[open_shelter], [full_shelter, failed_shelter]]],
            shelterCanByCell=[[[], []]],
            reservedPopulation=lambda shelter: reservations[shelter.shelterID],
            availableCapacity=lambda shelter: max(
                0,
                shelter.shelterCap
                - shelter.shelterFlow
                - reservations[shelter.shelterID],
            ),
        )
        builder = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        builder.number_of_cells = 2
        builder.nx = 1
        builder.ny = 2
        builder.core = SimpleNamespace(shelterDS=shelter_store)

        (
            remaining,
            operational,
            sheltered,
            enroute,
            utilization,
            _deployable,
            _candidate_count,
            _preview,
        ) = builder._regional_capacity()

        np.testing.assert_allclose(remaining, (3.0, 0.0))
        np.testing.assert_allclose(operational, (10.0, 5.0))
        np.testing.assert_allclose(sheltered, (4.0, 5.0))
        np.testing.assert_allclose(enroute, (3.0, 0.0))
        np.testing.assert_allclose(utilization, (0.7, 1.0))

    def test_candidate_response_is_capacity_timing_and_survival_bounded(self):
        builder = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        builder.initial_population = 100
        builder.region_east_positions = np.asarray((0.0, 1.0))
        builder.region_north_positions = np.asarray((0.0, 0.0))
        records = (
            ("safe-near", 0, 20.0, 0.0, 0.0),
            ("unsafe", 1, 20.0, 1.0, 0.0),
        )
        response = builder._candidate_reevacuation_response(
            records,
            np.asarray((10.0, 0.0)),
            np.asarray((0.0, 1.0)),
            np.asarray((0.0, 1.0)),
        )
        self.assertAlmostEqual(float(response[0]), 0.10, places=6)
        self.assertEqual(float(response[1]), 0.0)
        self.assertTrue(np.logical_and(response >= 0.0, response <= 1.0).all())

    def test_belief_quadrature_seed_does_not_read_hidden_tape_or_scenario_seed(self):
        public = {
            "wind_forecast_direction": 45.0,
            "wind_forecast_speed": 12.0,
            "wind_warning": True,
            "time_to_change": 3,
            "closure_warning_edges": (5,),
        }
        left_regime = SimpleNamespace(public=dict(public), scheduled=["hidden-a"])
        right_regime = SimpleNamespace(public=dict(public), scheduled=["hidden-b"])
        left = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        right = RegionalObservationBuilder.__new__(RegionalObservationBuilder)
        for builder, regime, scenario_seed in (
            (left, left_regime, 11),
            (right, right_regime, 999_999),
        ):
            builder.nx = 1
            builder.ny = 2
            builder.core = SimpleNamespace(
                hazard_regime=regime,
                scenario_seed=scenario_seed,
                random_stream_seeds={"hazard_regime": scenario_seed + 7},
            )
        danger = np.asarray((0.2, 0.6), dtype=np.float32)
        self.assertEqual(
            left._belief_forecast_seed(danger, 8),
            right._belief_forecast_seed(danger, 8),
        )
        right_regime.public["wind_forecast_speed"] = 13.0
        self.assertNotEqual(
            left._belief_forecast_seed(danger, 8),
            right._belief_forecast_seed(danger, 8),
        )


if __name__ == "__main__":
    unittest.main()
