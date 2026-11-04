"""v65 S1: the vectorized mean-field forecast kernel is bit-identical.

Every comparison is ``np.array_equal`` (bitwise), never ``allclose``.  The
reference kernel is the unchanged v64 loop, kept as
``HazardDS._stochastic_forecast_danger_reference``.
"""

from __future__ import annotations

import dataclasses
import os
import unittest

import numpy as np

import CounterfactualBranch as CB
import HazardDatabase as HD
from tests.v65_testbed import (
    PAPER_SPREAD,
    STRESS_SPREAD,
    advance,
    builder_for,
    full_state_digest,
    regime_testbed,
)

PAPER_SCALE = os.environ.get("V65_PAPER_SCALE", "") == "1"


def _with_kernel(hazard_store, kernel):
    hazard_store.configure_forecast_engine(
        kernel=kernel,
        memo_enabled=hazard_store.forecast_memo_enabled,
        memo_verify_fraction=hazard_store.forecast_memo_verify_fraction,
    )


def _observation_differences(left, right):
    differing = []
    for field in dataclasses.fields(left):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                differing.append(field.name)
        elif a != b:
            differing.append(field.name)
    return differing


class V65ForecastKernelTests(unittest.TestCase):
    CASES = (
        (6501, PAPER_SPREAD, (6, 18, 40)),
        (6502, STRESS_SPREAD, (6, 25, 50)),
    )

    def test_static_trajectories_and_endpoints_are_bitwise_equal(self):
        for seed, spread, times in self.CASES:
            core = regime_testbed(seed=seed, spread=spread)
            hazard_store = core.hazardDS
            now = 0
            for target in times:
                advance(core, target - now)
                now = target
                for horizon in (0, 1, 7, 20):
                    for trajectory in (True, False):
                        _with_kernel(hazard_store, "reference")
                        expected = hazard_store._stochastic_forecast_danger(
                            horizon, return_trajectory=trajectory
                        )
                        _with_kernel(hazard_store, "vectorized")
                        observed = hazard_store._stochastic_forecast_danger(
                            horizon, return_trajectory=trajectory
                        )
                        with self.subTest(
                            seed=seed, t=now, horizon=horizon, trajectory=trajectory
                        ):
                            self.assertEqual(expected.dtype, observed.dtype)
                            self.assertTrue(np.array_equal(expected, observed))

    def test_latent_ignitions_and_configuration_schedule_are_bitwise_equal(self):
        core = regime_testbed(seed=6503)
        advance(core, 12)
        hazard_store = core.hazardDS
        tracker = core.cellTracker
        ignitions = ((2, tracker.unflatten_cell(5)), (9, tracker.unflatten_cell(40)))
        schedule = {
            offset: (
                16.67 * (1.5 if offset > 6 else 0.6),
                (45.0 * offset) % 360.0,
                (0.7, 1.0, 1.6)[offset % 3],
            )
            for offset in range(1, 16)
        }
        results = {}
        states = {}
        for kernel in ("reference", "vectorized"):
            _with_kernel(hazard_store, kernel)
            before = (
                hazard_store.wind_speed_m_per_minute,
                hazard_store.wind_direction_degrees,
                hazard_store.intensity_multiplier,
            )
            results[kernel] = hazard_store._stochastic_forecast_danger(
                15,
                ignitions,
                configuration_by_step=schedule,
                return_trajectory=True,
            )
            # The kernel leaves the last configuration applied, exactly like
            # the reference; callers restore it.
            states[kernel] = (
                hazard_store.wind_speed_m_per_minute,
                hazard_store.wind_direction_degrees,
                hazard_store.intensity_multiplier,
            )
            hazard_store.configure_wind(
                speed_m_per_minute=before[0],
                direction_degrees=before[1],
                influence=hazard_store.wind_influence,
                time_step_minutes=hazard_store.time_step_minutes,
            )
            hazard_store.set_intensity(before[2])
        self.assertTrue(np.array_equal(results["reference"], results["vectorized"]))
        self.assertEqual(states["reference"], states["vectorized"])

    def test_belief_first_passage_outputs_are_bitwise_equal(self):
        for seed, spread, times in self.CASES:
            core = regime_testbed(seed=seed, spread=spread)
            hazard_store = core.hazardDS
            now = 0
            mask = np.zeros(core.cellTracker.number_of_cells, dtype=bool)
            mask[::5] = True
            for target in times:
                advance(core, target - now)
                now = target
                for level, horizon in ((4.0, 20), (3.0, 40)):
                    outputs = {}
                    for kernel in ("reference", "vectorized"):
                        _with_kernel(hazard_store, kernel)
                        rng = np.random.default_rng(np.random.SeedSequence([seed, now, 63]))
                        outputs[kernel] = (
                            hazard_store.belief_overrun_first_passage_by_cell(
                                horizon,
                                level,
                                ensemble_size=8,
                                rng=rng,
                                next_event_mask=mask,
                            ),
                            rng.bit_generator.state,
                        )
                    expected, expected_rng = outputs["reference"]
                    observed, observed_rng = outputs["vectorized"]
                    self.assertEqual(set(expected), set(observed))
                    self.assertEqual(expected_rng, observed_rng)
                    for key in expected:
                        with self.subTest(seed=seed, t=now, level=level, key=key):
                            self.assertTrue(
                                np.array_equal(
                                    np.asarray(expected[key]), np.asarray(observed[key])
                                )
                            )

    def test_full_observation_is_bitwise_equal(self):
        for seed, spread, times in self.CASES:
            core = regime_testbed(seed=seed, spread=spread)
            builder = builder_for(core)
            now = 0
            for target in (10, 20, 30, 50):
                advance(core, target - now)
                now = target
                kwargs = dict(
                    decision_index=target // 10 - 1,
                    simulation_time=now,
                    remaining_deployments=8,
                )
                _with_kernel(core.hazardDS, "reference")
                expected = builder.build(**kwargs)
                _with_kernel(core.hazardDS, "vectorized")
                observed = builder.build(**kwargs)
                with self.subTest(seed=seed, t=now):
                    self.assertEqual(_observation_differences(expected, observed), [])

    def test_forecast_has_no_side_effect_on_the_live_simulator(self):
        core = regime_testbed(seed=6504)
        advance(core, 15)
        builder = builder_for(core)
        before = full_state_digest(core)
        _with_kernel(core.hazardDS, "vectorized")
        builder.build(decision_index=0, simulation_time=15, remaining_deployments=8)
        core.hazardDS.belief_overrun_first_passage_by_cell(
            30, 3.0, ensemble_size=8, rng=np.random.default_rng(3)
        )
        self.assertEqual(full_state_digest(core), before)

    def test_engine_configuration_is_validated_and_recorded_for_workers(self):
        core = regime_testbed(seed=6505)
        with self.assertRaises(ValueError):
            core.hazardDS.configure_forecast_engine(kernel="numba")
        with self.assertRaises(ValueError):
            core.hazardDS.configure_forecast_engine(memo_verify_fraction=1.5)
        snapshot = CB.capture(core, label="v65-kernel-contract")
        portable = CB._portable_branch_core(core, snapshot)
        self.assertEqual(portable.hazardDS.forecast_kernel, "vectorized")
        core.hazardForecastKernel = "reference"
        with self.assertRaisesRegex(RuntimeError, "forecast kernel"):
            CB._portable_branch_core(core, snapshot)

    def test_kernel_caches_survive_restore_deep_copies(self):
        core = regime_testbed(seed=6506)
        advance(core, 8)
        HD._FORECAST_GEOMETRY_CACHE.clear()
        snapshot = CB.capture(core, label="v65-cache")
        core.hazardDS._stochastic_forecast_danger(5)
        self.assertEqual(len(HD._FORECAST_GEOMETRY_CACHE), 1)
        CB.restore(core, snapshot)
        core.hazardDS._stochastic_forecast_danger(5)
        self.assertEqual(len(HD._FORECAST_GEOMETRY_CACHE), 1)

    @unittest.skipUnless(PAPER_SCALE, "set V65_PAPER_SCALE=1 for the 16x16, 2,500-person check")
    def test_paper_scale_first_passage_is_bitwise_equal(self):
        core = regime_testbed(
            seed=7101, population=2500, grid=24, cells=16, stop_time=181, interval=15
        )
        advance(core, 30)
        builder = builder_for(core)
        kwargs = dict(decision_index=1, simulation_time=30, remaining_deployments=7)
        _with_kernel(core.hazardDS, "reference")
        expected = builder.build(**kwargs)
        _with_kernel(core.hazardDS, "vectorized")
        observed = builder.build(**kwargs)
        self.assertEqual(_observation_differences(expected, observed), [])


if __name__ == "__main__":
    unittest.main()
