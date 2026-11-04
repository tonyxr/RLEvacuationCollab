"""v65 S2: build the full observation only where it is consumed.

Between decision epochs ``RLBridge.step`` reads three fields of the regional
observation (``active_by_cell``, ``danger_by_cell``, ``outcome``).  The
ledger snapshot computes exactly those, with the helpers ``build()`` uses.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import CounterfactualBranch as CB
from DecisionInterface import (
    LedgerObservation,
    PolicyDecision,
    RegionalShelterExecutor,
)
from tests.v65_testbed import (
    advance,
    builder_for,
    full_state_digest,
    regime_testbed,
)

try:  # The RLBridge replay needs the real learner stack.
    import torch

    REAL_TORCH = isinstance(getattr(torch, "__version__", None), str)
except Exception:  # pragma: no cover - torch missing entirely
    REAL_TORCH = False


def _greedy_install(observation):
    feasible = [
        action
        for action in np.flatnonzero(observation.action_mask)
        if action != observation.defer_action_index
    ]
    if not feasible:
        return int(observation.defer_action_index)
    scores = [
        float(observation.active_by_cell[int(observation.candidate_cell_indices[a])])
        for a in feasible
    ]
    return int(feasible[int(np.argmax(scores))])


class V65LedgerSnapshotTests(unittest.TestCase):
    def test_ledger_fields_equal_full_build_every_minute(self):
        core = regime_testbed(seed=6511)
        builder = builder_for(core)
        executor = RegionalShelterExecutor(core)
        remaining = 8
        decision_index = 0
        for minute in range(1, 60):
            advance(core, 1)
            ledger = builder.ledger_snapshot(simulation_time=minute)
            full = builder.build(
                decision_index=decision_index,
                simulation_time=minute,
                remaining_deployments=remaining,
            )
            with self.subTest(minute=minute):
                self.assertIsInstance(ledger, LedgerObservation)
                self.assertEqual(ledger.active_by_cell.dtype, full.active_by_cell.dtype)
                self.assertTrue(np.array_equal(ledger.active_by_cell, full.active_by_cell))
                self.assertTrue(np.array_equal(ledger.danger_by_cell, full.danger_by_cell))
                self.assertEqual(ledger.outcome, full.outcome)
                self.assertEqual(
                    ledger.outcome.hazard_exposure_mass,
                    full.outcome.hazard_exposure_mass,
                )
            if minute % 10 == 0 and remaining > 0 and full.has_feasible_action:
                receipt = executor.execute(
                    full,
                    PolicyDecision(action_index=_greedy_install(full), strategy="v65-test"),
                )
                remaining -= int(bool(receipt.installed))
                decision_index += 1

    def test_ledger_snapshot_has_no_side_effect(self):
        core = regime_testbed(seed=6512)
        advance(core, 17)
        builder = builder_for(core)
        before = full_state_digest(core)
        builder.ledger_snapshot(simulation_time=17)
        self.assertEqual(full_state_digest(core), before)

    def test_skipping_non_decision_builds_leaves_the_trajectory_unchanged(self):
        """Two copies, same seed and decisions; one builds every minute."""
        cores = {name: regime_testbed(seed=6513) for name in ("every", "gated")}
        builders = {name: builder_for(core) for name, core in cores.items()}
        executors = {name: RegionalShelterExecutor(core) for name, core in cores.items()}
        remaining = {name: 8 for name in cores}
        decisions = {name: 0 for name in cores}
        builds = {name: 0 for name in cores}
        ledger_totals = {name: [0.0, 0.0] for name in cores}
        horizon = int(cores["every"].stopTime) - 1
        for minute in range(1, horizon + 1):
            observations = {}
            for name, core in cores.items():
                advance(core, 1)
                due = minute % 10 == 0 and remaining[name] > 0
                if name == "every" or minute == 1 or due or minute == horizon:
                    observation = builders[name].build(
                        decision_index=decisions[name],
                        simulation_time=minute,
                        remaining_deployments=remaining[name],
                    )
                    builds[name] += 1
                else:
                    observation = builders[name].ledger_snapshot(simulation_time=minute)
                ledger_totals[name][0] += float(observation.outcome.active_population)
                ledger_totals[name][1] += float(observation.outcome.hazard_exposure_mass)
                observations[name] = (observation, due)
            for name in cores:
                observation, due = observations[name]
                if due and observation.has_feasible_action:
                    receipt = executors[name].execute(
                        observation,
                        PolicyDecision(
                            action_index=_greedy_install(observation),
                            strategy="v65-test",
                        ),
                    )
                    remaining[name] -= int(bool(receipt.installed))
                    decisions[name] += 1
            with self.subTest(minute=minute):
                self.assertEqual(
                    full_state_digest(cores["every"]), full_state_digest(cores["gated"])
                )
        self.assertEqual(ledger_totals["every"], ledger_totals["gated"])
        self.assertEqual(decisions["every"], decisions["gated"])
        self.assertEqual(builds["every"], horizon)
        self.assertLessEqual(builds["gated"], 2 + horizon // 10)

    @unittest.skipUnless(REAL_TORCH, "RLBridge replay requires torch")
    def test_bridge_episode_is_identical_with_and_without_every_minute_builds(self):
        from RLBridge import RLBridge
        import nmcc_testbed

        ignored = {
            "observation_build_latency_ms",
            "policy_selection_latency_ms",
            "deployment_execution_latency_ms",
            "end_to_end_deployment_latency_ms",
            "observation_built",
            "feasible_cells",
            "feasible_candidates",
        }
        for strategy, train_mode in (("objective_greedy", False), ("rl", True)):
            runs = {}
            for every_minute in (True, False):
                core = nmcc_testbed.build(
                    grid=8,
                    cell_x=4,
                    cell_y=4,
                    population=120,
                    stop_time=31,
                    hazard_count=2,
                    scenario_seed=6514,
                )
                core.observationEveryMinute = every_minute
                with tempfile.TemporaryDirectory() as directory:
                    bridge = RLBridge(
                        core,
                        train_mode=train_mode,
                        deployment_strategy=strategy,
                        target_active_shelters=len(core.shelterDS.shelterList) + 3,
                        shelter_action_interval=3,
                        first_decision_time=1,
                        policy_seed=13,
                        rollout_episodes=1,
                        checkpoint_path=os.path.join(directory, "policy.pt"),
                        diagnostics_path=os.path.join(directory, "diagnostics.csv"),
                    )
                    rows, digests = [], []
                    for minute in range(1, int(core.stopTime)):
                        CB.advance_one_timestep(core)
                        result = bridge.step(
                            simulation_time=minute,
                            is_terminal=minute == int(core.stopTime) - 1,
                        )
                        rows.append(
                            (
                                {k: v for k, v in result.items() if k not in ignored},
                                int(result["observation_built"]),
                                int(result["feasible_cells"]),
                                int(result["feasible_candidates"]),
                            )
                        )
                        digests.append(CB.live_state_digest(core))
                        if bridge.episode_done:
                            break
                    runs[every_minute] = dict(
                        rows=rows,
                        digests=digests,
                        objective=(
                            bridge.objective_active_person_time,
                            bridge.objective_hazard_exposure_person_time,
                            bridge.objective_expected_casualty_person_time,
                        ),
                        deployment_times=list(bridge.deployment_times),
                        builds=int(bridge.full_observations_built),
                    )
            every, gated = runs[True], runs[False]
            with self.subTest(strategy=strategy):
                self.assertEqual(every["digests"], gated["digests"])
                self.assertEqual(every["objective"], gated["objective"])
                self.assertEqual(every["deployment_times"], gated["deployment_times"])
                self.assertEqual(
                    [row[0] for row in every["rows"]], [row[0] for row in gated["rows"]]
                )
                for every_row, gated_row in zip(every["rows"], gated["rows"]):
                    if gated_row[1]:
                        self.assertEqual(every_row[2:], gated_row[2:])
                self.assertEqual(every["builds"], len(every["rows"]))
                self.assertLess(gated["builds"], every["builds"])


if __name__ == "__main__":
    unittest.main()
