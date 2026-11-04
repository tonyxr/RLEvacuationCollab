"""v65 S3: belief-trajectory memo shared by candidates of one future."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import CounterfactualBranch as CB
import HazardDatabase as HD
import RolloutPolicyIteration as RPI
from tests.v65_testbed import (
    advance,
    builder_for,
    hazard_digest,
    regime_testbed,
)


def _first_passage(hazard_store, *, seed, level=3.0, horizon=30, mask=None):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 64]))
    result = hazard_store.belief_overrun_first_passage_by_cell(
        horizon, level, ensemble_size=8, rng=rng, next_event_mask=mask
    )
    return result, rng.bit_generator.state


def _assert_same(test, left, right):
    test.assertEqual(set(left), set(right))
    for key in left:
        test.assertTrue(
            np.array_equal(np.asarray(left[key]), np.asarray(right[key])), key
        )


def _decision_state(seed=6521, minute=10):
    core = regime_testbed(seed=seed)
    advance(core, minute)
    builder = builder_for(core)
    observation = builder.build(
        decision_index=0, simulation_time=minute, remaining_deployments=8
    )
    cells = [
        int(observation.candidate_cell_indices[action])
        for action in np.flatnonzero(observation.action_mask)
        if action != observation.defer_action_index
    ][:3]
    return core, cells


def _rollout_costs(core, cells, *, workers, memo, chunking="tape_affine"):
    core.forecastMemoEnabled = bool(memo)
    core.branchTaskChunking = chunking
    core.hazardDS.configure_forecast_engine(
        kernel="vectorized", memo_enabled=bool(memo), memo_verify_fraction=0.0
    )
    snapshot = CB.capture(core, label="v65-memo-test")
    brancher = CB.CounterfactualBrancher(
        core,
        horizon=0,
        reward_model=None,
        initial_population=int(core.initial_population),
        episode_horizon=int(core.stopTime) - 1,
        workers=workers,
    )
    try:
        costs = brancher.rollout_policy_costs(
            cells,
            snapshot=snapshot,
            future_seeds=(101, 202),
            decision_index=0,
            remaining_deployments=8,
            maximum_deployments=8,
            frozen_policy=RPI.ObjectiveGreedyContinuationPolicy(),
            mode="belief",
        )
    finally:
        brancher.close()
    return costs, CB.live_state_digest(core)


class V65ForecastMemoTests(unittest.TestCase):
    def setUp(self):
        HD.clear_belief_forecast_memo()
        HD.set_belief_forecast_memo_scope(None)

    def test_no_scope_means_no_memo(self):
        core = regime_testbed(seed=6520)
        advance(core, 12)
        _first_passage(core.hazardDS, seed=1)
        _first_passage(core.hazardDS, seed=1)
        self.assertEqual(HD.belief_forecast_memo_statistics()["entries"], 0)

    def test_hit_is_identical_to_recomputation_for_any_mask(self):
        core = regime_testbed(seed=6522)
        advance(core, 12)
        hazard_store = core.hazardDS
        hazard_store.configure_forecast_engine(memo_verify_fraction=1.0)
        cells = core.cellTracker.number_of_cells
        masks = (None, np.arange(cells) % 3 == 0, np.arange(cells) % 7 == 1)
        unscoped = [_first_passage(hazard_store, seed=9, level=4.0, mask=m) for m in masks]
        before = HD.belief_forecast_memo_statistics()
        with HD.belief_forecast_memo_scope(("batch", 0)):
            scoped = [
                _first_passage(hazard_store, seed=9, level=4.0, mask=m) for m in masks
            ]
        after = HD.belief_forecast_memo_statistics()
        self.assertEqual(after["misses"] - before["misses"], 1)
        self.assertEqual(after["hits"] - before["hits"], 2)
        self.assertEqual(after["verified"] - before["verified"], 2)
        for (expected, expected_rng), (observed, observed_rng) in zip(unscoped, scoped):
            _assert_same(self, expected, observed)
            self.assertEqual(expected_rng, observed_rng)

    def test_different_scope_or_state_does_not_hit(self):
        core = regime_testbed(seed=6523)
        advance(core, 12)
        before = HD.belief_forecast_memo_statistics()["hits"]
        with HD.belief_forecast_memo_scope(("batch", 0)):
            _first_passage(core.hazardDS, seed=3)
        with HD.belief_forecast_memo_scope(("batch", 1)):
            _first_passage(core.hazardDS, seed=3)
        with HD.belief_forecast_memo_scope(("batch", 0)):
            _first_passage(core.hazardDS, seed=4)  # different forecast seed
            advance(core, 1)
            _first_passage(core.hazardDS, seed=3)  # different hazard state
        self.assertEqual(HD.belief_forecast_memo_statistics()["hits"], before)

    def test_verification_detects_a_corrupted_entry(self):
        core = regime_testbed(seed=6524)
        advance(core, 12)
        core.hazardDS.configure_forecast_engine(memo_verify_fraction=1.0)
        with HD.belief_forecast_memo_scope(("batch", 0)):
            _first_passage(core.hazardDS, seed=5)
            key = next(iter(HD._BELIEF_TRAJECTORY_MEMO))
            samples, state = HD._BELIEF_TRAJECTORY_MEMO[key]
            corrupted = [np.array(sample, copy=True) for sample in samples]
            corrupted[0][-1, 0] += 0.2
            HD._BELIEF_TRAJECTORY_MEMO[key] = (tuple(corrupted), state)
            with self.assertRaisesRegex(RuntimeError, "not action-independent"):
                _first_passage(core.hazardDS, seed=5)

    def test_memo_switch_disables_storage(self):
        core = regime_testbed(seed=6525)
        advance(core, 12)
        core.hazardDS.configure_forecast_engine(memo_enabled=False)
        with HD.belief_forecast_memo_scope(("batch", 0)):
            _first_passage(core.hazardDS, seed=6)
            _first_passage(core.hazardDS, seed=6)
        self.assertEqual(HD.belief_forecast_memo_statistics()["entries"], 0)

    def test_hazard_is_action_independent_within_one_future(self):
        core, cells = _decision_state(seed=6526)
        snapshot = CB.capture(core, label="v65-independence")
        brancher = CB.CounterfactualBrancher(
            core,
            horizon=0,
            reward_model=None,
            initial_population=int(core.initial_population),
            episode_horizon=int(core.stopTime) - 1,
        )
        traces = []
        for cell in cells:
            CB.restore(core, snapshot)
            brancher._reseed_exogenous_streams(777)
            particle = core.hazard_regime.belief_particles(1, np.random.default_rng(777))[0]
            particle.cell_tracker = core.cellTracker
            particle.map_ds = core.mapDS
            core.hazard_regime = particle
            core.hazardDS.set_regime(particle)
            _, cell_y = CB.authoritative_grid_shape(core)
            shelter_id = core.shelterDS.newShelter({"cell": divmod(cell, cell_y)}, core.cellTracker)
            shelter = core.shelterDS.shelterList[shelter_id]
            core.cellTracker.addShelter(divmod(cell, cell_y), shelter)
            core.pedDS.reroute_active_to_nearest_shelter(new_shelter=shelter)
            trace = []
            for _ in range(int(core.stopTime) - 1 - 10):
                CB.advance_one_timestep(core)
                trace.append(hazard_digest(core))
            traces.append(trace)
        CB.restore(core, snapshot)
        self.assertGreaterEqual(len(cells), 2)
        for trace in traces[1:]:
            self.assertEqual(trace, traces[0])

    def test_local_rollout_costs_identical_with_memo_and_memo_hits(self):
        core, cells = _decision_state(seed=6527)
        reference, digest_reference = _rollout_costs(core, cells, workers=1, memo=False)
        before = HD.belief_forecast_memo_statistics()
        memoized, digest_memoized = _rollout_costs(core, cells, workers=1, memo=True)
        after = HD.belief_forecast_memo_statistics()
        self.assertTrue(np.array_equal(reference, memoized))
        self.assertEqual(digest_reference, digest_memoized)
        self.assertGreater(after["hits"] - before["hits"], 0)
        self.assertEqual(after["entries"], 0)  # cleared after the batch

    def test_pooled_rollout_costs_identical_across_memo_and_chunking(self):
        core, cells = _decision_state(seed=6528)
        reference, _ = _rollout_costs(
            core, cells, workers=2, memo=False, chunking="round_robin"
        )
        memoized, _ = _rollout_costs(core, cells, workers=2, memo=True)
        local, _ = _rollout_costs(core, cells, workers=1, memo=False)
        self.assertTrue(np.array_equal(reference, memoized))
        self.assertTrue(np.array_equal(reference, local))

    def test_tape_affine_chunks_cover_every_task_once_in_tape_blocks(self):
        tasks = [
            SimpleNamespace(tape_index=tape, action_position=position)
            for tape in range(4)
            for position in range(15)
        ]
        for workers in (1, 2, 3, 4, 7, 8):
            chunks = CB._branch_task_chunks(tasks, workers, "tape_affine")
            flat = [task for chunk in chunks for task in chunk]
            self.assertEqual(sorted(map(id, flat)), sorted(map(id, tasks)))
            sizes = [len(chunk) for chunk in chunks]
            self.assertLessEqual(max(sizes) - min(sizes), 1)
            tapes = [task.tape_index for task in flat]
            self.assertEqual(tapes, sorted(tapes))
        legacy = CB._branch_task_chunks(tasks, 4, "round_robin")
        self.assertEqual(legacy, [tuple(tasks[i::4]) for i in range(4)])
        with self.assertRaises(ValueError):
            CB._branch_task_chunks(tasks, 4, "random")


if __name__ == "__main__":
    unittest.main()
