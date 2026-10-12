#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-0 validity tests for natural-momentum counterfactual branching.

These tests are deliberately torch-free.  They exercise the real hazard
cellular automaton, the real keyed casualty and panic draws, the real social
force and congestion models and the real shelter installer on a synthetic grid
city, so a regression in snapshot fidelity is caught by the dynamics
themselves rather than by a mock.

The properties under test are the ones NMCC names as prerequisites: if any of
them fails, every causal effect measured through the branch machinery is an
artifact, and training on those labels would be worse than the noisy gradients
it set out to replace.
"""

import unittest
import subprocess
import sys
import time

import numpy as np

import CounterfactualBranch as CB
import nmcc_testbed
from CAProcessor import authoritative_grid_shape
from DecisionInterface import RegionalObservationBuilder
from RewardProcessor import RewardProcessor


def build_core(**kwargs):
    defaults = dict(
        grid=10,
        cell_x=4,
        cell_y=4,
        population=120,
        stop_time=40,
        spread_rate=(8, 4),
        casualty_rate=(25, 10),
        hazard_count=2,
        scenario_seed=4242,
    )
    defaults.update(kwargs)
    return nmcc_testbed.build(**defaults)


def brancher_for(core, horizon=5):
    return CB.CounterfactualBrancher(
        core,
        horizon=horizon,
        reward_model=RewardProcessor(),
        initial_population=int(core.initial_population),
        episode_horizon=int(core.stopTime),
    )


class SnapshotFidelityTests(unittest.TestCase):
    def test_worker_protocol_read_has_a_bounded_deadline(self):
        process = subprocess.Popen(
            (sys.executable, "-c", "import time; time.sleep(5)"),
            stdout=subprocess.PIPE,
        )
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                CB._read_worker_message(
                    process,
                    deadline=time.monotonic() + 0.05,
                    context="deadline test",
                )
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            process.terminate()
            process.wait(timeout=5.0)
            process.stdout.close()

    def test_uniform_reference_rollouts_are_complete_and_restore_live_state(self):
        core = build_core(
            population=40,
            stop_time=12,
            hazard_count=1,
            candidate_count=20,
            shelter_capacity_token=20,
        )
        core.shelterActionInterval = 3
        runner = brancher_for(core, horizon=3)
        snapshot = CB.capture(core, label="uniform-reference-test")
        digest = CB.live_state_digest(core)
        costs = runner.random_policy_costs(
            snapshot=snapshot,
            future_seeds=(101, 101, 203, 307),
            decision_index=0,
            remaining_deployments=2,
            maximum_deployments=2,
        )
        self.assertEqual(len(costs), 4)
        self.assertAlmostEqual(costs[0], costs[1], places=12)
        self.assertTrue(np.isfinite(costs).all())
        self.assertTrue(all(cost > 0.0 for cost in costs))
        self.assertEqual(CB.live_state_digest(core), digest)

    def test_rectangular_grid_stays_coherent_through_episode_and_restore(self):
        core = build_core(
            grid=6,
            cell_x=2,
            cell_y=3,
            population=30,
            stop_time=12,
            hazard_count=1,
            hazard_mode="deterministic",
            candidate_count=24,
        )
        self.assertEqual(authoritative_grid_shape(core), (2, 3))
        builder = RegionalObservationBuilder(
            core,
            initial_population=30,
            horizon=12,
            maximum_deployments=2,
        )
        initial = builder.build(
            decision_index=0,
            simulation_time=0,
            remaining_deployments=2,
        )
        self.assertEqual(initial.number_of_cells, 6)
        self.assertEqual(initial.number_of_actions, 6)
        self.assertEqual(initial.danger_by_cell.shape, (6,))
        self.assertEqual(initial.forecast_danger_by_cell.shape, (6,))

        snapshot = CB.capture(core, label="rectangular-grid")
        for _ in range(4):
            CB.advance_one_timestep(core)
            self.assertEqual(authoritative_grid_shape(core), (2, 3))
        later = builder.build(
            decision_index=1,
            simulation_time=4,
            remaining_deployments=1,
        )
        self.assertEqual(later.danger_by_cell.shape, (6,))

        CB.restore(core, snapshot)
        self.assertEqual(authoritative_grid_shape(core), (2, 3))
        np.testing.assert_allclose(
            core.cellTracker.dangerLevelByCell,
            initial.danger_by_cell,
        )

    def test_restored_snapshot_replays_bitwise_identically(self):
        """The prerequisite: a restored state must reproduce the same future.

        If any mutable simulator state escapes the snapshot, two replays of the
        same factual trajectory diverge, and a paired comparison would be
        measuring that leak rather than the intervention.
        """
        core = build_core()
        for _ in range(6):
            CB.advance_one_timestep(core)
        digest = CB.assert_replay_is_bitwise_identical(core, steps=8)
        self.assertEqual(len(digest), 64)

    def test_snapshot_is_reusable_across_many_branches(self):
        """One snapshot must seed any number of branches identically.

        ``restore`` copies on the way out as well as in; without that the first
        branch would mutate the stored state and silently invalidate every
        later comparison drawn from the same decision.
        """
        core = build_core()
        for _ in range(6):
            CB.advance_one_timestep(core)
        base = CB.capture(core)
        digests = []
        for _ in range(3):
            CB.restore(core, base)
            for _ in range(4):
                CB.advance_one_timestep(core)
            digests.append(CB.live_state_digest(core))
        self.assertEqual(len(set(digests)), 1)

    def test_a_branch_does_not_disturb_the_live_episode(self):
        """Asking a counterfactual must not change the world that asked it."""
        core = build_core()
        for _ in range(6):
            CB.advance_one_timestep(core)
        before = CB.live_state_digest(core)
        shelters_before = len(core.shelterDS.shelterList)

        runner = brancher_for(core)
        runner.paired_effect(core.feasible_cells()[0])

        self.assertEqual(CB.live_state_digest(core), before)
        self.assertEqual(len(core.shelterDS.shelterList), shelters_before)

    def test_full_candidate_table_uses_one_shared_wait_and_restores_live_state(self):
        core = build_core()
        for _ in range(6):
            CB.advance_one_timestep(core)
        before = CB.live_state_digest(core)
        cells = core.feasible_cells()[:4]
        effects = brancher_for(core).candidate_effects(cells)

        self.assertEqual([effect.cell_index for effect in effects], list(cells))
        self.assertTrue(all(effect.waited is effects[0].waited for effect in effects))
        self.assertEqual(CB.live_state_digest(core), before)

    def test_parallel_multi_future_tables_equal_serial_tables_exactly(self):
        core = build_core(population=160)
        for _ in range(6):
            CB.advance_one_timestep(core)
        before = CB.live_state_digest(core)
        base = CB.capture(core)
        cells = core.feasible_cells()[:4]
        seeds = (1_103, 2_107)
        endpoint_spec = CB.EndpointObservationSpec(
            initial_population=int(core.initial_population),
            episode_horizon=int(core.stopTime),
            maximum_deployments=4,
            decision_index=1,
            remaining_deployments=3,
        )

        serial = CB.CounterfactualBrancher(
            core,
            horizon=5,
            reward_model=RewardProcessor(),
            initial_population=int(core.initial_population),
            episode_horizon=int(core.stopTime),
            workers=1,
        ).candidate_effects_many_futures(
            cells,
            future_seeds=seeds,
            snapshot=base,
            endpoint_spec=endpoint_spec,
        )
        parallel = CB.CounterfactualBrancher(
            core,
            horizon=5,
            reward_model=RewardProcessor(),
            initial_population=int(core.initial_population),
            episode_horizon=int(core.stopTime),
            workers=2,
        ).candidate_effects_many_futures(
            cells,
            future_seeds=seeds,
            snapshot=base,
            endpoint_spec=endpoint_spec,
        )

        self.assertEqual(CB.live_state_digest(core), before)
        for serial_table, parallel_table in zip(serial, parallel):
            self.assertEqual(
                [effect.cell_index for effect in serial_table],
                [effect.cell_index for effect in parallel_table],
            )
            self.assertEqual(
                serial_table[0].waited.end_digest,
                parallel_table[0].waited.end_digest,
            )
            for serial_effect, parallel_effect in zip(
                serial_table,
                parallel_table,
            ):
                self.assertEqual(
                    serial_effect.acted.end_digest,
                    parallel_effect.acted.end_digest,
                )
                self.assertAlmostEqual(
                    serial_effect.reward_difference,
                    parallel_effect.reward_difference,
                    places=12,
                )
                np.testing.assert_array_equal(
                    serial_effect.acted.endpoint_payload.active_by_cell,
                    parallel_effect.acted.endpoint_payload.active_by_cell,
                )

    def test_parallel_workers_are_reused_without_changing_exact_results(self):
        core = build_core(population=40, grid=6, cell_x=3, cell_y=3)
        for _ in range(3):
            CB.advance_one_timestep(core)
        base = CB.capture(core)
        cells = core.feasible_cells()[:2]
        brancher = CB.CounterfactualBrancher(
            core,
            horizon=2,
            reward_model=RewardProcessor(),
            initial_population=int(core.initial_population),
            episode_horizon=int(core.stopTime),
            workers=2,
        )
        try:
            first = brancher.candidate_effects_many_futures(
                cells,
                future_seeds=(101,),
                snapshot=base,
            )
            process_ids = tuple(
                process.pid for process in brancher._worker_pool.processes
            )
            completed_after_first = int(
                brancher._worker_pool.batches_completed
            )
            second = brancher.candidate_effects_many_futures(
                cells,
                future_seeds=(101,),
                snapshot=base,
            )
            self.assertEqual(
                process_ids,
                tuple(process.pid for process in brancher._worker_pool.processes),
            )
            self.assertEqual(
                int(brancher._worker_pool.batches_completed),
                completed_after_first + 1,
            )
            for first_effect, second_effect in zip(first[0], second[0]):
                self.assertEqual(
                    first_effect.waited.end_digest,
                    second_effect.waited.end_digest,
                )
                self.assertEqual(
                    first_effect.acted.end_digest,
                    second_effect.acted.end_digest,
                )
        finally:
            brancher.close()

    def test_parallel_workers_persist_across_episode_branchers(self):
        """Fresh episode objects must reuse one process-level OpenMP pool."""
        CB.shutdown_persistent_branch_pools()
        first_core = build_core(population=40, grid=6, cell_x=3, cell_y=3)
        second_core = build_core(
            population=40,
            grid=6,
            cell_x=3,
            cell_y=3,
            scenario_seed=4243,
        )
        first = CB.CounterfactualBrancher(
            first_core,
            horizon=2,
            reward_model=RewardProcessor(),
            initial_population=int(first_core.initial_population),
            episode_horizon=int(first_core.stopTime),
            workers=2,
        )
        second = CB.CounterfactualBrancher(
            second_core,
            horizon=2,
            reward_model=RewardProcessor(),
            initial_population=int(second_core.initial_population),
            episode_horizon=int(second_core.stopTime),
            workers=2,
        )
        try:
            first.candidate_effects_many_futures(
                first_core.feasible_cells()[:2],
                future_seeds=(101,),
                snapshot=CB.capture(first_core),
            )
            pool = first._worker_pool
            process_ids = tuple(process.pid for process in pool.processes)
            completed_before = int(pool.batches_completed)
            first.close()

            second.candidate_effects_many_futures(
                second_core.feasible_cells()[:2],
                future_seeds=(103,),
                snapshot=CB.capture(second_core),
            )
            self.assertIs(second._worker_pool, pool)
            self.assertEqual(
                tuple(process.pid for process in second._worker_pool.processes),
                process_ids,
            )
            self.assertEqual(
                int(second._worker_pool.batches_completed),
                completed_before + 1,
            )
        finally:
            first.close()
            second.close()
            CB.shutdown_persistent_branch_pools()

    def test_four_worker_pool_is_not_restarted_by_episode_cleanup(self):
        CB.shutdown_persistent_branch_pools()
        process_ids = None
        batches_completed = None
        try:
            for episode_seed in range(5):
                core = build_core(
                    population=32,
                    grid=6,
                    cell_x=3,
                    cell_y=3,
                    stop_time=10,
                    scenario_seed=8_100 + episode_seed,
                )
                brancher = CB.CounterfactualBrancher(
                    core,
                    horizon=2,
                    reward_model=RewardProcessor(),
                    initial_population=int(core.initial_population),
                    episode_horizon=int(core.stopTime),
                    workers=4,
                )
                brancher.candidate_effects_many_futures(
                    core.feasible_cells()[:2],
                    future_seeds=(201, 203),
                    snapshot=CB.capture(core),
                )
                current_ids = tuple(
                    process.pid for process in brancher._worker_pool.processes
                )
                if process_ids is None:
                    process_ids = current_ids
                    batches_completed = int(
                        brancher._worker_pool.batches_completed
                    )
                else:
                    self.assertEqual(current_ids, process_ids)
                    self.assertEqual(
                        int(brancher._worker_pool.batches_completed),
                        batches_completed + episode_seed,
                    )
                brancher.close()
        finally:
            CB.shutdown_persistent_branch_pools()

    def test_installed_reference_contrast_matches_full_table_action_difference(self):
        core = build_core(population=120)
        for _ in range(4):
            CB.advance_one_timestep(core)
        before = CB.live_state_digest(core)
        base = CB.capture(core)
        cells = core.feasible_cells()[:3]
        seeds = (7_019, 7_021)
        brancher = CB.CounterfactualBrancher(
            core,
            horizon=4,
            reward_model=RewardProcessor(),
            initial_population=int(core.initial_population),
            episode_horizon=int(core.stopTime),
            workers=1,
        )
        full = brancher.candidate_effects_many_futures(
            cells,
            future_seeds=seeds,
            snapshot=base,
        )
        direct = brancher.candidate_effects_many_futures(
            cells[1:],
            future_seeds=seeds,
            snapshot=base,
            reference_cell=int(cells[0]),
        )

        self.assertEqual(CB.live_state_digest(core), before)
        for full_table, direct_table in zip(full, direct):
            self.assertEqual(
                direct_table[0].waited.end_digest,
                full_table[0].acted.end_digest,
            )
            for full_effect, direct_effect in zip(full_table[1:], direct_table):
                self.assertEqual(
                    direct_effect.acted.end_digest,
                    full_effect.acted.end_digest,
                )
                expected = (
                    full_effect.acted.hazard_weighted_time_to_safety_cost
                    - full_table[0].acted.hazard_weighted_time_to_safety_cost
                )
                observed = (
                    direct_effect.acted.hazard_weighted_time_to_safety_cost
                    - direct_effect.waited.hazard_weighted_time_to_safety_cost
                )
                self.assertAlmostEqual(observed, expected, places=9)

    def test_branch_order_does_not_change_either_branch(self):
        """Branch-order invariance.

        Running ``WAIT`` first and the deployment second must produce exactly
        the outcomes of the opposite order.  If it does not, the two branches
        are sharing state through some channel the snapshot does not cover, and
        the difference between them is not a causal effect.
        """
        core = build_core()
        for _ in range(6):
            CB.advance_one_timestep(core)
        runner = brancher_for(core)
        cell = core.feasible_cells()[0]
        base = CB.capture(core)

        act_first = runner.paired_effect(cell, snapshot=base, wait_first=False)
        wait_first = runner.paired_effect(cell, snapshot=base, wait_first=True)

        self.assertEqual(act_first.acted.end_digest, wait_first.acted.end_digest)
        self.assertEqual(act_first.waited.end_digest, wait_first.waited.end_digest)
        self.assertAlmostEqual(
            act_first.reward_difference, wait_first.reward_difference, places=12
        )

    def test_hazard_is_action_independent(self):
        """Shelters must not perturb the fire.

        Hazard is the dominant exogenous process.  If installing a shelter
        changed it, the WAIT branch would ride a different fire and the paired
        difference would confound the intervention with the hazard.
        """
        core = build_core()
        for _ in range(6):
            CB.advance_one_timestep(core)
        CB.assert_hazard_is_action_independent(
            core, steps=6, cell_index=core.feasible_cells()[0]
        )


class OutcomeFidelityTests(unittest.TestCase):
    def test_branch_outcomes_match_the_observation_builder(self):
        """The fast outcome read must agree with the authoritative one.

        ``outcome_snapshot`` skips the candidate feature block, which costs
        ``O(cells * pedestrians)`` and is not needed to score a branch.  This
        pins it against ``RegionalObservationBuilder`` so the shortcut cannot
        drift away from the quantity the reward is actually defined on.
        """
        core = build_core()
        builder = RegionalObservationBuilder(
            core,
            initial_population=int(core.initial_population),
            horizon=int(core.stopTime),
            maximum_deployments=4,
        )
        for step in range(1, 9):
            CB.advance_one_timestep(core)
            observation = builder.build(
                decision_index=0,
                simulation_time=step,
                remaining_deployments=4,
            )
            fast = CB.outcome_snapshot(core)
            self.assertEqual(
                fast.safe_completed,
                observation.outcome.safe_completed,
                msg=f"safe_completed drifted at step {step}",
            )
            self.assertEqual(fast.casualties, observation.outcome.casualties)
            self.assertEqual(
                fast.active_population, observation.outcome.active_population
            )
            self.assertAlmostEqual(
                fast.risk_mass, observation.outcome.risk_mass, places=4
            )


class CausalSignalTests(unittest.TestCase):
    def test_zero_horizon_array_evaluator_matches_transactional_branches(self):
        """Fast full-action labels must equal literal install/reroute twins."""
        core = build_core(population=200)
        for _ in range(8):
            CB.advance_one_timestep(core)
        runner = brancher_for(core, horizon=0)
        base = CB.capture(core)
        CB.restore(core, base)
        literal_wait = runner._run_branch(
            label="waited", install_cell=None, steps=0
        )
        cells = core.feasible_cells()[:4]
        literal_actions = {}
        for cell in cells:
            CB.restore(core, base)
            literal_actions[int(cell)] = runner._run_branch(
                label="acted", install_cell=int(cell), steps=0
            )
        CB.restore(core, base)
        fast = runner.candidate_effects(cells, snapshot=base)

        for effect in fast:
            literal = literal_actions[effect.cell_index]
            self.assertAlmostEqual(
                effect.waited.remaining_plain_person_time,
                literal_wait.remaining_plain_person_time,
                places=9,
            )
            self.assertAlmostEqual(
                effect.waited.remaining_hazard_exposure_person_time,
                literal_wait.remaining_hazard_exposure_person_time,
                places=9,
            )
            self.assertAlmostEqual(
                effect.acted.remaining_plain_person_time,
                literal.remaining_plain_person_time,
                places=9,
            )
            self.assertAlmostEqual(
                effect.acted.remaining_hazard_exposure_person_time,
                literal.remaining_hazard_exposure_person_time,
                places=9,
            )

    def test_zero_horizon_wait_allows_all_existing_shelters_to_be_full(self):
        core = build_core(population=120)
        for _ in range(6):
            CB.advance_one_timestep(core)
        for pedestrian in tuple(core.pedDS.pedAgentList.values()):
            core.shelterDS.releaseReservation(pedestrian)
        for shelter in core.shelterDS.shelterList.values():
            shelter.shelterFlow = shelter.shelterCap
            shelter.status = 1
        core.shelterDS.assertCapacityInvariant()
        runner = brancher_for(core, horizon=0)
        cell = core.feasible_cells()[0]
        base = CB.capture(core)
        CB.restore(core, base)
        literal_wait = runner._run_branch(
            label="waited", install_cell=None, steps=0
        )
        CB.restore(core, base)
        literal_action = runner._run_branch(
            label="acted", install_cell=cell, steps=0
        )
        CB.restore(core, base)
        fast = runner.paired_effect(cell, snapshot=base)
        self.assertAlmostEqual(
            fast.waited.hazard_weighted_time_to_safety_cost,
            literal_wait.hazard_weighted_time_to_safety_cost,
            places=9,
        )
        self.assertAlmostEqual(
            fast.acted.hazard_weighted_time_to_safety_cost,
            literal_action.hazard_weighted_time_to_safety_cost,
            places=9,
        )

    def test_reward_twins_price_casualty_intensity_before_realized_draws(self):
        core = build_core(
            population=120,
            panic_rate=1.0,
            casualty_rate=(100, 0),
        )
        for _ in range(8):
            CB.advance_one_timestep(core)
        runner = brancher_for(core, horizon=6)
        effect = runner.paired_effect(core.feasible_cells()[0])

        self.assertGreater(effect.acted.expected_casualty_person_time, 0.0)
        self.assertGreater(effect.waited.expected_casualty_person_time, 0.0)
        self.assertGreaterEqual(
            effect.acted.end_outcome.casualties,
            effect.acted.start_outcome.casualties,
        )
        self.assertGreaterEqual(
            effect.waited.end_outcome.casualties,
            effect.waited.start_outcome.casualties,
        )

    def test_exposure_removal_uses_matched_future_noise(self):
        """Dynamic reward may vary, but each pair must share one noise tape."""
        core = build_core(population=200)
        for _ in range(8):
            CB.advance_one_timestep(core)
        runner = brancher_for(core, horizon=6)
        cell = core.feasible_cells()[1]
        base = CB.capture(core)

        acted, waited = [], []
        for tape in range(6):
            seed = 1_000 + 37 * tape
            CB.restore(core, base)
            runner._reseed_exogenous_streams(seed)
            waited.append(
                runner._run_branch(label="waited", install_cell=None, steps=6)
                .discounted_return()
            )
            CB.restore(core, base)
            runner._reseed_exogenous_streams(seed)
            acted.append(
                runner._run_branch(label="acted", install_cell=cell, steps=6)
                .discounted_return()
            )
        CB.restore(core, base)

        acted = np.asarray(acted)
        waited = np.asarray(waited)
        paired = acted - waited

        self.assertGreater(float(acted.std(ddof=1)), 0.0)
        self.assertGreater(float(waited.std(ddof=1)), 0.0)
        self.assertTrue(np.isfinite(paired).all())
        self.assertLessEqual(
            float(paired.std(ddof=1)),
            float(acted.std(ddof=1) + waited.std(ddof=1)) + 1e-12,
        )

    def test_pairing_does_not_change_the_estimated_effect(self):
        """Variance reduction must not buy itself with bias."""
        core = build_core(population=200)
        for _ in range(8):
            CB.advance_one_timestep(core)
        runner = brancher_for(core, horizon=5)
        cell = core.feasible_cells()[1]
        base = CB.capture(core)

        common = [
            runner.paired_effect(cell, snapshot=base, common_noise=True).reward_difference
            for _ in range(4)
        ]
        independent = [
            runner.paired_effect(
                cell,
                snapshot=base,
                common_noise=False,
                independent_noise_seed=91 + index,
            ).reward_difference
            for index in range(8)
        ]
        spread = float(np.std(independent, ddof=1))
        self.assertLessEqual(
            abs(float(np.mean(common)) - float(np.mean(independent))),
            4.0 * spread + 1e-9,
            msg="paired and unpaired estimators disagree on the mean effect",
        )


class AdvantageTests(unittest.TestCase):
    def test_counterfactual_advantage_matches_its_definition(self):
        """``A_CF`` must be the reward difference plus a bootstrapped tail."""
        acted = CB.BranchResult(label="acted", rewards=[0.4, 0.2], steps=2)
        waited = CB.BranchResult(label="waited", rewards=[0.1, 0.1], steps=2)
        effect = CB.PairedEffect(cell_index=3, horizon=2, acted=acted, waited=waited)

        undiscounted = CB.counterfactual_advantage(effect)
        self.assertAlmostEqual(undiscounted, 0.4, places=12)

        with_tail = CB.counterfactual_advantage(
            effect,
            gamma=0.5,
            acted_bootstrap_value=2.0,
            waited_bootstrap_value=1.0,
            intervention_cost=0.05,
        )
        expected = (0.4 - 0.1) + 0.5 * (0.2 - 0.1) + (0.5 ** 2) * (2.0 - 1.0) - 0.05
        self.assertAlmostEqual(with_tail, expected, places=12)

    def test_wait_baseline_is_constant_across_cells(self):
        """The control variate must not depend on the action it scores.

        This is the property that keeps the policy-gradient direction intact:
        a baseline that varied with the chosen cell would bias the update
        rather than merely de-noising it.
        """
        core = build_core(population=160)
        for _ in range(8):
            CB.advance_one_timestep(core)
        runner = brancher_for(core, horizon=5)
        base = CB.capture(core)
        cells = core.feasible_cells()[:3]

        wait_returns = []
        for cell in cells:
            effect = runner.paired_effect(cell, snapshot=base)
            wait_returns.append(round(effect.waited.discounted_return(), 12))
        self.assertEqual(len(set(wait_returns)), 1)


if __name__ == "__main__":
    unittest.main()
