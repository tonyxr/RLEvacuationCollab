#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch-free tests for the NMCC-PI target, branch valuation and cell selection.

The simulator-level tests run the real hazard, casualty, panic, movement and
shelter dynamics on the synthetic testbed.  The properties pinned here are the
ones the corrected design depends on: the improvement target moves by exactly
the requested amount and only toward higher exact value; branching never
disturbs the live episode; every cell at a state is compared under the same
future tape; and that tape is independent of the live episode's own future.
"""

import unittest

import numpy as np

import CounterfactualBranch as CB
import NMCCPolicyImprovement as PI
import nmcc_testbed
from DecisionInterface import RegionalObservationBuilder, RegionalShelterExecutor


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.array([True, True, True, False, True])
        self.pi = np.array([0.4, 0.3, 0.2, 0.0, 0.1])
        self.actions = np.array([0, 1, 2, 4])
        self.values = np.array([0.10, 0.30, 0.20, 0.05])

    def test_target_reaches_the_requested_kl(self):
        for epsilon in (0.05, 0.2, 0.6):
            target = PI.improvement_target(
                self.pi, self.mask, self.actions, self.values, epsilon=epsilon, eta_min=1e-4
            )
            self.assertAlmostEqual(target.kl_to_old, epsilon, places=4)
            self.assertAlmostEqual(float(target.target.sum()), 1.0, places=12)
            self.assertEqual(float(target.target[3]), 0.0)

    def test_target_moves_only_toward_higher_value(self):
        """The likelihood ratio q/pi_old must be monotone in the exact value."""
        target = PI.improvement_target(
            self.pi, self.mask, self.actions, self.values, epsilon=0.3, eta_min=1e-4
        )
        ratio = target.target[self.actions] / self.pi[self.actions]
        order = np.argsort(self.values)
        self.assertTrue(np.all(np.diff(ratio[order]) > 0))
        self.assertEqual(target.best_action, 1)

    def test_equal_values_leave_the_policy_unchanged(self):
        """No evidence, no movement."""
        target = PI.improvement_target(
            self.pi, self.mask, self.actions, np.full(4, 0.2), epsilon=0.3, eta_min=0.01
        )
        np.testing.assert_allclose(target.target, self.pi / self.pi.sum(), atol=1e-12)
        self.assertAlmostEqual(target.kl_to_old, 0.0, places=12)

    def test_temperature_floor_stops_noise_becoming_a_confident_label(self):
        """A difference far below eta_min must not produce a large move."""
        values = np.array([0.2000, 0.2001, 0.2000, 0.2000])
        target = PI.improvement_target(
            self.pi, self.mask, self.actions, values, epsilon=0.5, eta_min=0.03
        )
        self.assertEqual(target.eta, 0.03)
        self.assertLess(target.kl_to_old, 1e-5)

    def test_unbranched_cells_keep_their_probability(self):
        target = PI.improvement_target(
            self.pi, self.mask, np.array([0, 1]), np.array([0.1, 0.3]), epsilon=0.2, eta_min=1e-4
        )
        self.assertAlmostEqual(float(target.target[2]), 0.2, places=12)
        self.assertAlmostEqual(float(target.target[4]), 0.1, places=12)
        self.assertAlmostEqual(float(target.target[0] + target.target[1]), 0.7, places=12)

    def test_advantage_is_within_state_and_policy_centered(self):
        """V_wait and any other cell-independent quantity must cancel exactly."""
        shifted = self.values + 12.5
        a = PI.improvement_target(self.pi, self.mask, self.actions, self.values, epsilon=0.3, eta_min=1e-4)
        b = PI.improvement_target(self.pi, self.mask, self.actions, shifted, epsilon=0.3, eta_min=1e-4)
        np.testing.assert_allclose(a.target, b.target, atol=1e-10)
        weights = self.pi[self.actions] / self.pi[self.actions].sum()
        self.assertAlmostEqual(float(np.dot(weights, a.advantage[self.actions])), 0.0, places=12)

    def test_score_target_orders_only_exactly_branched_candidates(self):
        target = PI.score_ranking_target(
            self.mask,
            np.array([0, 1, 4]),
            np.array([0.1, 0.4, 0.2]),
            temperature=0.05,
        )
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)
        self.assertEqual(int(np.argmax(target)), 1)
        self.assertEqual(float(target[2]), 0.0)
        self.assertEqual(float(target[3]), 0.0)


class SelectionTests(unittest.TestCase):
    def test_wait_defers_without_consuming_a_deployment(self):
        clock = PI.EpisodeClock(
            horizon=30,
            interval=10,
            budget=3,
            population=100,
            t=1,
            deployed=0,
            next_decision_time=1,
        )
        clock.defer_deployment(clock.t)
        self.assertEqual(clock.deployed, 0)
        self.assertFalse(clock.is_decision(2))
        self.assertTrue(clock.is_decision(11))

    def test_early_decisions_are_exhaustive(self):
        feasible = np.arange(15)
        chosen = PI.select_branch_actions(
            feasible, np.full(15, 1 / 15), decision_index=0, exhaustive_decisions=2,
            max_branches=4, rng=np.random.default_rng(0),
        )
        np.testing.assert_array_equal(chosen, feasible)

    def test_later_decisions_keep_top_cells_executed_action_and_support(self):
        feasible = np.arange(15)
        probs = np.linspace(1, 2, 15)
        probs /= probs.sum()
        chosen = PI.select_branch_actions(
            feasible, probs, decision_index=3, exhaustive_decisions=2, max_branches=6,
            must_include=(0,), rng=np.random.default_rng(1),
        )
        self.assertEqual(chosen.size, 6)
        self.assertIn(0, chosen)
        self.assertIn(14, chosen)  # most probable cell


class BranchValuationTests(unittest.TestCase):
    def build(self):
        core = nmcc_testbed.build(
            grid=10, cell_x=4, cell_y=4, population=160, stop_time=31, spread_rate=(4, 2),
            casualty_rate=(40, 9), panic_rate=0.5, hazard_count=2, scenario_seed=77,
            spacing_m=150, candidate_count=10, shelter_capacity_token=40,
        )
        builder = RegionalObservationBuilder(core, initial_population=160, horizon=30, maximum_deployments=3)
        executor = RegionalShelterExecutor(core)
        clock = PI.EpisodeClock(horizon=30, interval=10, budget=3, population=160)
        PI.step_and_score(core, clock, PI.RewardProcessor())
        obs = builder.build(decision_index=0, simulation_time=clock.t, remaining_deployments=3)
        valuer = PI.BranchValuer(core, builder=builder, executor=executor, tapes=2)
        return core, obs, clock, valuer

    def test_candidate_benefit_weights_the_same_time_saving_by_urgency(self):
        _, _, _, valuer = self.build()
        builder = valuer.builder
        zeros = np.zeros(builder.number_of_cells, dtype=np.float32)
        ones = np.ones(builder.number_of_cells, dtype=np.float32)
        calm = builder._candidate_operational_features(
            zeros,
            zeros,
            builder._candidate_records,
            builder._candidate_nodes,
        )[-1]
        urgent = builder._candidate_operational_features(
            zeros,
            ones,
            builder._candidate_records,
            builder._candidate_nodes,
        )[-1]
        useful = calm > 0.0
        self.assertTrue(bool(useful.any()))
        self.assertTrue(bool(np.all(urgent[useful] >= calm[useful])))
        self.assertTrue(bool(np.any(urgent[useful] > calm[useful])))

    def test_valuation_does_not_disturb_the_live_episode(self):
        core, obs, clock, valuer = self.build()
        before = CB.live_state_digest(core)
        shelters = len(core.shelterDS.shelterList)
        valuer.value(obs, clock, np.flatnonzero(obs.action_mask)[:3], episode_seed=77)
        self.assertEqual(CB.live_state_digest(core), before)
        self.assertEqual(len(core.shelterDS.shelterList), shelters)
        self.assertEqual(clock.deployed, 0)

    def test_valuation_is_reproducible(self):
        core, obs, clock, valuer = self.build()
        actions = np.flatnonzero(obs.action_mask)[:3]
        a = valuer.value(obs, clock, actions, episode_seed=77)
        b = valuer.value(obs, clock, actions, episode_seed=77)
        np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(a.wait_values, b.wait_values)
        np.testing.assert_array_equal(a.outcomes, b.outcomes)
        np.testing.assert_array_equal(a.wait_outcomes, b.wait_outcomes)
        self.assertEqual(a.outcomes.shape, (2, len(actions), 6))
        self.assertEqual(a.paired_outcome_effects().shape, (2, len(actions), 6))

    def test_every_cell_shares_the_tape_so_wait_is_independent_of_the_subset(self):
        """CRN across cells: the WAIT reference cannot depend on which cells were branched."""
        core, obs, clock, valuer = self.build()
        feasible = np.flatnonzero(obs.action_mask)
        a = valuer.value(obs, clock, feasible[:2], episode_seed=77)
        b = valuer.value(obs, clock, feasible[1:4], episode_seed=77)
        np.testing.assert_array_equal(a.wait_values, b.wait_values)
        shared = int(feasible[1])
        np.testing.assert_array_equal(
            a.values[:, list(a.actions).index(shared)], b.values[:, list(b.actions).index(shared)]
        )

    def test_first_decision_uses_terminal_horizon_not_the_short_branch_window(self):
        core, obs, clock, valuer = self.build()
        action = int(np.flatnonzero(obs.action_mask)[0])
        short = PI.BranchValuer(
            core,
            builder=valuer.builder,
            executor=valuer.executor,
            tapes=1,
            branch_horizon=1,
            full_horizon_decisions=0,
        ).value(obs, clock, [action], episode_seed=77)
        full = PI.BranchValuer(
            core,
            builder=valuer.builder,
            executor=valuer.executor,
            tapes=1,
            branch_horizon=1,
            full_horizon_decisions=1,
        ).value(obs, clock, [action], episode_seed=77)
        # Outcome coordinate 2 is accumulated active person-time divided by
        # population*horizon. A terminal continuation must account for more
        # than a one-step label at the first, high-leverage intervention.
        self.assertGreater(full.outcomes[0, 0, 2], short.outcomes[0, 0, 2])

    def test_tapes_are_not_the_live_future(self):
        """The target must be implementable: branches may not see the real future.

        The factual continuation of the live episode under the base policy is
        computed separately; a branch value equal to it on every tape would mean
        the tape leaked the episode's own noise.
        """
        core, obs, clock, valuer = self.build()
        action = int(PI.risk_reduction_policy(obs))
        values = valuer.value(obs, clock, [action], episode_seed=77)

        base = CB.capture(core)
        live = clock.copy()
        valuer._install(obs, action)
        live.deployed += 1
        factual = valuer._continue(live)
        CB.restore(core, base)

        self.assertFalse(np.allclose(values.values[:, 0], factual))
        self.assertNotEqual(PI.tape_seed(77, 0, 0), PI.tape_seed(77, 0, 1))

    def test_unknown_base_policy_is_rejected(self):
        core, obs, clock, valuer = self.build()
        with self.assertRaises(ValueError):
            PI.BranchValuer(core, builder=valuer.builder, executor=valuer.executor, base_policy="oracle")


if __name__ == "__main__":
    unittest.main()
