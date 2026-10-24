import unittest

import numpy as np

from CounterfactualBranch import CounterfactualBrancher
from DecisionInterface import (
    BENCHMARK_POLICY_CONTRACTS,
    OutcomeSnapshot,
    RegionalObservation,
    RegionalObservationBuilder,
    public_objective_kwargs,
    public_regional_objective_values,
)
from RLBridge import RLBridge
from backtest import DYNAMIC_STRATEGIES as SINGLE_CITY_DYNAMIC
from backtest import STATIC_STRATEGIES as SINGLE_CITY_STATIC
from multicity_backtest import DYNAMIC_STRATEGIES as MULTICITY_DYNAMIC
from multicity_backtest import STATIC_STRATEGIES as MULTICITY_STATIC
from nmcc_testbed import TestbedCore


def _observation(**overrides):
    active = np.asarray(overrides.pop("active_by_cell", (20.0, 20.0)), dtype=np.float32)
    values = dict(
        decision_index=0,
        simulation_time=5,
        horizon=20,
        initial_population=int(active.sum()),
        remaining_deployments=2,
        maximum_deployments=2,
        maximum_speed=10.0,
        active_by_cell=active,
        mean_speed_by_cell=np.asarray((2.0, 2.0), dtype=np.float32),
        danger_by_cell=np.asarray((0.2, 0.2), dtype=np.float32),
        remaining_capacity_by_cell=np.zeros(2, dtype=np.float32),
        deployable_capacity_by_cell=np.asarray((10.0, 10.0), dtype=np.float32),
        candidate_count_by_cell=np.ones(2, dtype=np.float32),
        action_mask=np.ones(2, dtype=np.bool_),
        outcome=OutcomeSnapshot(0, 0, 0, 0, int(active.sum()), 48.0),
        candidate_osm_node_ids=("a", "b"),
        candidate_cell_indices=np.asarray((0, 1), dtype=np.int64),
        candidate_capacities=np.asarray((10.0, 10.0), dtype=np.float32),
        remaining_evacuation_debt_by_cell=np.asarray((0.4, 0.4), dtype=np.float32),
        slow_tail_time_by_cell=np.asarray((5.0, 5.0), dtype=np.float32),
        candidate_forecast_danger=np.asarray((0.2, 0.2), dtype=np.float32),
        candidate_forecast_overrun_probability=np.zeros(2, dtype=np.float32),
        candidate_forecast_reevacuation_response=np.zeros(2, dtype=np.float32),
        candidate_threat_capture_fraction=np.zeros(2, dtype=np.float32),
        configured_panic_fraction=0.5,
    )
    values.update(overrides)
    return RegionalObservation(**values)


def _legacy_reference(
    observation,
    *,
    clearance_tail_weight,
    casualty_priority_kappa,
):
    values = np.full(observation.number_of_actions, -np.inf, dtype=np.float64)
    feasible = np.flatnonzero(observation.action_mask)
    defer = int(observation.defer_action_index)
    feasible = feasible[feasible != defer]
    if feasible.size == 0:
        return values
    cells = np.asarray(observation.candidate_cell_indices[feasible], dtype=np.int64)
    population = float(max(1, int(observation.initial_population)))
    active = np.asarray(observation.active_by_cell[cells], dtype=np.float64)
    capacities = np.asarray(observation.candidate_capacities[feasible], dtype=np.float64)
    coverage = np.minimum(active, capacities) / np.maximum(active, 1.0)
    debt = np.asarray(
        observation.remaining_evacuation_debt_by_cell[cells], dtype=np.float64
    )
    horizon_minutes = max(
        float(observation.time_step_minutes),
        float(observation.horizon) * float(observation.time_step_minutes),
    )
    tail = (
        max(0.0, float(clearance_tail_weight))
        * np.asarray(observation.slow_tail_time_by_cell[cells], dtype=np.float64)
        / (population * horizon_minutes)
    )
    danger = np.maximum(
        np.asarray(observation.danger_by_cell[cells], dtype=np.float64),
        np.asarray(observation.candidate_forecast_danger[feasible], dtype=np.float64),
    )
    time_remaining = max(
        0.0,
        float(observation.horizon - observation.simulation_time)
        / float(max(1, observation.horizon)),
    )
    expected_harm = (
        max(0.0, float(casualty_priority_kappa))
        * (active / population)
        * danger
        * time_remaining
    )
    displaced_access = np.asarray(
        observation.candidate_forecast_reevacuation_response[feasible],
        dtype=np.float64,
    )
    scores = coverage * (debt + tail + expected_harm) + displaced_access
    if not np.isfinite(scores).all() or np.any(scores < 0.0):
        raise FloatingPointError
    scale = max(float(np.max(scores)), 1e-12)
    values[feasible] = scores / scale
    return values


class V64PublicObjectiveTests(unittest.TestCase):
    def test_legacy_variant_is_bitwise_v63_for_twenty_random_states(self):
        rng = np.random.default_rng(20261024)
        for _ in range(20):
            active = rng.integers(1, 40, size=2).astype(np.float32)
            observation = _observation(
                active_by_cell=active,
                initial_population=int(active.sum()),
                outcome=OutcomeSnapshot(0, 0, 0, 0, int(active.sum()), 0.0),
                candidate_capacities=rng.uniform(1.0, 30.0, size=2).astype(np.float32),
                remaining_evacuation_debt_by_cell=rng.uniform(0.0, 1.0, size=2).astype(np.float32),
                slow_tail_time_by_cell=rng.uniform(0.0, 100.0, size=2).astype(np.float32),
                danger_by_cell=rng.uniform(0.0, 1.0, size=2).astype(np.float32),
                candidate_forecast_danger=rng.uniform(0.0, 1.0, size=2).astype(np.float32),
                candidate_forecast_reevacuation_response=rng.uniform(0.0, 1.0, size=2).astype(np.float32),
            )
            kwargs = dict(
                clearance_tail_weight=float(rng.uniform(0.0, 2.0)),
                casualty_priority_kappa=float(rng.uniform(1.0, 4.0)),
            )
            expected = _legacy_reference(observation, **kwargs)
            actual = public_regional_objective_values(
                observation, variant="legacy", **kwargs
            )
            self.assertTrue(np.array_equal(actual, expected))

    def test_front_aware_penalizes_site_overrun_survival(self):
        observation = _observation(
            candidate_forecast_overrun_probability=np.asarray((0.0, 0.9), dtype=np.float32)
        )
        front = public_regional_objective_values(
            observation,
            clearance_tail_weight=0.0,
            casualty_priority_kappa=2.0,
            variant="front_aware",
        )
        legacy = public_regional_objective_values(
            observation,
            clearance_tail_weight=0.0,
            casualty_priority_kappa=2.0,
            variant="legacy",
        )
        self.assertGreater(front[0], front[1])
        self.assertEqual(legacy[0], legacy[1])

    def test_front_aware_rewards_threat_capture(self):
        observation = _observation(
            candidate_threat_capture_fraction=np.asarray((0.0, 0.1), dtype=np.float32)
        )
        front = public_regional_objective_values(
            observation,
            clearance_tail_weight=0.0,
            casualty_priority_kappa=2.0,
            variant="front_aware",
        )
        legacy = public_regional_objective_values(
            observation,
            clearance_tail_weight=0.0,
            casualty_priority_kappa=2.0,
            variant="legacy",
        )
        self.assertGreater(front[1], front[0])
        self.assertEqual(legacy[0], legacy[1])

    def test_front_aware_does_not_seek_danger(self):
        observation = _observation(
            danger_by_cell=np.asarray((0.0, 0.8), dtype=np.float32),
            candidate_forecast_danger=np.asarray((0.0, 0.8), dtype=np.float32),
        )
        front = public_regional_objective_values(
            observation,
            clearance_tail_weight=0.0,
            casualty_priority_kappa=2.0,
            variant="front_aware",
        )
        legacy = public_regional_objective_values(
            observation,
            clearance_tail_weight=0.0,
            casualty_priority_kappa=2.0,
            variant="legacy",
        )
        self.assertEqual(front[0], front[1])
        self.assertGreater(legacy[1], legacy[0])

    def test_brancher_prior_consistency_and_legacy_strategy_argmax(self):
        core = TestbedCore(
            grid=6,
            cell_x=4,
            cell_y=4,
            population=20,
            group_size=2,
            stop_time=20,
            candidate_count=20,
        )
        core.publicObjectiveVariant = "front_aware"
        observation = RegionalObservationBuilder(
            core,
            initial_population=20,
            horizon=20,
            maximum_deployments=2,
        ).build(
            decision_index=0,
            simulation_time=0,
            remaining_deployments=2,
        )
        brancher = CounterfactualBrancher(
            core,
            horizon=0,
            reward_model=core.reward_model(),
            initial_population=20,
            episode_horizon=20,
        )
        enriched = brancher.enrich_observation(observation)
        expected = public_regional_objective_values(
            enriched, **public_objective_kwargs(core)
        )
        np.testing.assert_allclose(
            enriched.candidate_public_objective_score,
            np.where(np.isfinite(expected), expected, 0.0),
        )
        legacy_kwargs = public_objective_kwargs(core)
        legacy_kwargs["variant"] = "legacy"
        legacy = public_regional_objective_values(enriched, **legacy_kwargs)
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy="objective_greedy_legacy",
            target_active_shelters=2,
        )
        self.assertEqual(
            bridge._select_action(enriched).action_index,
            int(np.nanargmax(legacy)),
        )

    def test_strategy_contracts_cover_single_and_multicity_runners(self):
        strategies = set(
            SINGLE_CITY_DYNAMIC
            + SINGLE_CITY_STATIC
            + MULTICITY_DYNAMIC
            + MULTICITY_STATIC
        )
        self.assertEqual(strategies - set(BENCHMARK_POLICY_CONTRACTS), set())


if __name__ == "__main__":
    unittest.main()
