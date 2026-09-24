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

import numpy as np

import CounterfactualBranch as CB
import nmcc_testbed
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
    def test_common_noise_reduces_effect_variance(self):
        """The whole point: pairing must collapse the difference variance.

        Both arms reuse the identical acted simulations, so the comparison
        isolates the pairing and nothing else.
        """
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
        unpaired = acted  # PPO's baseline is constant in the action

        self.assertGreater(
            unpaired.std(ddof=1),
            0.0,
            msg="the scenario is degenerate: the unpaired estimator has no variance",
        )
        self.assertLess(
            paired.std(ddof=1),
            0.5 * unpaired.std(ddof=1),
            msg="common random numbers failed to reduce effect variance",
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
