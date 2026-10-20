import copy
import types
import unittest
from types import SimpleNamespace

import numpy as np

import CounterfactualBranch as CB
from HazardDatabase import HazardDS
from tests.test_v53r_regime_crn import regime_core


class _TinyTracker:
    number_of_cells = 4
    grid_shape = (1, 4)

    @staticmethod
    def state_vector(*, max_state, normalized):
        assert max_state == 5
        assert normalized
        return np.asarray([0.8, 0.4, 0.2, 0.2], dtype=np.float32)

    @staticmethod
    def unflatten_cell(index):
        return (0, int(index))

    @staticmethod
    def assert_grid_coherent(*, require_initialized):
        del require_initialized


def _tiny_hazard():
    hazard = HazardDS(
        0,
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        wind_speed_m_per_minute=7.0,
        wind_direction_degrees=25.0,
        time_step_minutes=2.0,
    )
    hazard.setCellTracker(_TinyTracker())
    trajectories = {
        1: np.asarray([0.8, 0.8, 0.8, 0.2], dtype=np.float32),
        2: np.asarray([0.8, 0.8, 0.8, 0.4], dtype=np.float32),
        3: np.asarray([0.8, 0.8, 0.8, 0.6], dtype=np.float32),
    }

    def forecast_states(
        _self,
        horizon_steps,
        future_ignitions=(),
        *,
        configuration_by_step=None,
        return_trajectory=False,
    ):
        del future_ignitions, configuration_by_step
        values = np.rint(
            np.stack(
                [trajectories[lead] for lead in range(1, int(horizon_steps) + 1)]
            )
            * 5.0
        ).astype(np.int16)
        return values if return_trajectory else values[-1]

    hazard._deterministic_forecast_states = types.MethodType(
        forecast_states, hazard
    )
    return hazard


class HazardOverrunFirstPassageTests(unittest.TestCase):
    def test_first_passage_semantics_and_next_cell_probability(self):
        hazard = _tiny_hazard()
        settings_before = (
            hazard.wind_speed_m_per_minute,
            hazard.wind_direction_degrees,
            hazard.intensity_multiplier,
        )

        result = hazard.belief_overrun_first_passage_by_cell(
            3,
            4,
            ensemble_size=7,
        )

        np.testing.assert_array_equal(
            result["currently_overrun"], [1.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_array_equal(
            result["crossing_probability"], [1.0, 1.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            result["future_crossing_probability"], [0.0, 1.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            result["expected_first_crossing_steps"], [0.0, 1.0, 1.0, 4.0]
        )
        np.testing.assert_array_equal(
            result["expected_first_crossing_minutes"], [0.0, 2.0, 2.0, 8.0]
        )
        np.testing.assert_array_equal(
            result["conditional_expected_first_crossing_steps"],
            [0.0, 1.0, 1.0, 4.0],
        )
        np.testing.assert_array_equal(
            result["next_overrun_probability"], [0.0, 0.5, 0.5, 0.0]
        )
        np.testing.assert_array_equal(
            result["next_priority_overrun_probability"],
            result["next_overrun_probability"],
        )
        np.testing.assert_array_equal(result["next_overrun_rank"], [0, 1, 2, 0])
        np.testing.assert_array_equal(
            result["crossing_cdf_by_step"],
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ],
        )
        self.assertEqual(
            settings_before,
            (
                hazard.wind_speed_m_per_minute,
                hazard.wind_direction_degrees,
                hazard.intensity_multiplier,
            ),
        )

    def test_next_priority_crossing_is_ranked_within_operational_subset(self):
        hazard = _tiny_hazard()
        result = hazard.belief_overrun_first_passage_by_cell(
            3,
            4,
            ensemble_size=3,
            next_event_mask=(False, False, True, False),
        )
        # Cell 1 crosses at the same earliest time but has no operational
        # shelter. It must not split or erase the probability assigned to the
        # first threatened shelter cell (cell 2).
        np.testing.assert_array_equal(
            result["next_priority_overrun_probability"],
            [0.0, 0.0, 1.0, 0.0],
        )
        with self.assertRaisesRegex(ValueError, "next_event_mask"):
            hazard.belief_overrun_first_passage_by_cell(
                3,
                4,
                next_event_mask=(True, False),
            )

    def test_endpoint_distribution_applies_regime_changes_at_their_step(self):
        hazard = _tiny_hazard()
        captured = {}

        def forecast_states(
            _self,
            horizon_steps,
            future_ignitions=(),
            *,
            configuration_by_step=None,
            return_trajectory=False,
        ):
            captured["configurations"] = dict(configuration_by_step)
            captured["ignitions"] = tuple(future_ignitions)
            trajectory = np.asarray(
                [
                    [3, 2, 1, 1],
                    [3, 2, 2, 1],
                    [3, 3, 2, 1],
                    [4, 3, 2, 1],
                ],
                dtype=np.int16,
            )[: int(horizon_steps)]
            return trajectory if return_trajectory else trajectory[-1]

        hazard._deterministic_forecast_states = types.MethodType(
            forecast_states, hazard
        )
        particle = SimpleNamespace(
            current_time=10,
            wind_direction_degrees=0.0,
            base_wind_speed_m_per_minute=10.0,
            wind_speed_multiplier=1.0,
            intensity_multiplier=1.0,
            scheduled=[
                {
                    "time": 12,
                    "kind": "wind",
                    "payload": {
                        "direction_delta": 90.0,
                        "speed_multiplier": 1.5,
                    },
                },
                {
                    "time": 13,
                    "kind": "intensity",
                    "payload": {"multiplier": 1.6},
                },
                {
                    "time": 14,
                    "kind": "ignition",
                    "payload": {"cell_index": 3},
                },
            ],
        )

        result = hazard.forecast_distribution_by_cell(4, particles=(particle,))

        self.assertEqual(
            captured["configurations"],
            {
                1: (10.0, 0.0, 1.0),
                2: (15.0, 90.0, 1.0),
                3: (15.0, 90.0, 1.6),
                4: (15.0, 90.0, 1.6),
            },
        )
        self.assertEqual(captured["ignitions"], ((4, (0, 3)),))
        np.testing.assert_allclose(result["expected"], [0.8, 0.6, 0.4, 0.2])
        np.testing.assert_allclose(result["momentum"], [-0.2, 0.0, 0.0, 0.0])

    def test_same_public_belief_ignores_different_hidden_future_tapes(self):
        core = regime_core(6291)
        live = core.hazard_regime
        live.current_time = 7
        live._update_public(7)
        twin = copy.deepcopy(live)
        twin.intensity_multiplier = (
            1.6 if float(live.intensity_multiplier) != 1.6 else 0.7
        )
        twin.scheduled.append(
            {
                "time": 9,
                "kind": "intensity",
                "payload": {"multiplier": 1.6},
            }
        )
        twin.scheduled.sort(key=lambda event: (event["time"], event["kind"]))
        self.assertEqual(live.public, twin.public)
        self.assertNotEqual(live.scheduled, twin.scheduled)

        hazard_rng_before = copy.deepcopy(core.hazardDS.rng.bit_generator.state)
        live_tape_before = copy.deepcopy(live.scheduled)
        core.hazardDS.set_regime(live)
        left = core.hazardDS.belief_overrun_first_passage_by_cell(
            4,
            3,
            ensemble_size=5,
            rng=np.random.default_rng(902),
        )
        matching_particles = live.belief_particles(
            5, np.random.default_rng(902)
        )
        endpoint = core.hazardDS.forecast_distribution_by_cell(
            4, matching_particles
        )
        for key in (
            "expected",
            "momentum",
            "escalation_probability",
            "uncertainty",
        ):
            np.testing.assert_allclose(left[key], endpoint[key], atol=0.0, rtol=0.0)
        core.hazardDS.set_regime(twin)
        right = core.hazardDS.belief_overrun_first_passage_by_cell(
            4,
            3,
            ensemble_size=5,
            rng=np.random.default_rng(902),
        )

        self.assertEqual(live.scheduled, live_tape_before)
        self.assertEqual(core.hazardDS.rng.bit_generator.state, hazard_rng_before)
        for key in left:
            np.testing.assert_allclose(left[key], right[key], atol=0.0, rtol=0.0)

    def test_belief_particle_sampling_cannot_reopen_the_live_map(self):
        core = regime_core(6292)
        edge_id = int(next(iter(core.mapDS.edgeListByLocalID)))
        core.mapDS.close_edge(edge_id, 19)
        core.hazard_regime.active_closures[edge_id] = 19
        closures_before = copy.deepcopy(core.mapDS.closed_edges)
        routing_version_before = int(core.mapDS.routing_version)
        digest_before = CB.live_state_digest(core)

        particles = core.hazard_regime.belief_particles(
            3, np.random.default_rng(903)
        )

        self.assertEqual(len(particles), 3)
        self.assertEqual(core.mapDS.closed_edges, closures_before)
        self.assertEqual(core.mapDS.routing_version, routing_version_before)
        self.assertEqual(CB.live_state_digest(core), digest_before)

    def test_active_closure_reopening_time_is_resampled_not_leaked(self):
        left_core = regime_core(6293)
        right_core = regime_core(6293)
        edge_id = int(next(iter(left_core.mapDS.edgeListByLocalID)))
        for core, hidden_end in ((left_core, 12), (right_core, 31)):
            core.hazard_regime.current_time = 7
            core.mapDS.close_edge(edge_id, hidden_end)
            core.hazard_regime.active_closures[edge_id] = hidden_end
            core.hazard_regime._update_public(7)
        self.assertEqual(
            left_core.hazard_regime.public,
            right_core.hazard_regime.public,
        )
        self.assertNotEqual(
            left_core.hazard_regime.active_closures,
            right_core.hazard_regime.active_closures,
        )

        left_particles = left_core.hazard_regime.belief_particles(
            5, np.random.default_rng(904)
        )
        right_particles = right_core.hazard_regime.belief_particles(
            5, np.random.default_rng(904)
        )

        self.assertEqual(
            [particle.active_closures for particle in left_particles],
            [particle.active_closures for particle in right_particles],
        )
        self.assertEqual(
            [particle.scheduled for particle in left_particles],
            [particle.scheduled for particle in right_particles],
        )

    def test_invalid_forecast_contract_is_rejected(self):
        hazard = _tiny_hazard()
        with self.assertRaisesRegex(ValueError, "horizon_steps"):
            hazard.belief_overrun_first_passage_by_cell(-1, 4)
        with self.assertRaisesRegex(ValueError, "danger_threshold_level"):
            hazard.belief_overrun_first_passage_by_cell(3, 0)
        with self.assertRaisesRegex(ValueError, "lead_time_quantile"):
            hazard.belief_overrun_first_passage_by_cell(
                3, 4, lead_time_quantile=1.1
            )
        with self.assertRaisesRegex(ValueError, "ensemble_size"):
            hazard.belief_overrun_first_passage_by_cell(3, 4, ensemble_size=0)


if __name__ == "__main__":
    unittest.main()
