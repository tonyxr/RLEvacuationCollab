"""v65.1: the comparator plug-in path is inert and exact.

``deployment_strategy='external'`` with the reference plug-in
``comparators.public_score_greedy:PublicScoreGreedy`` must reproduce
``objective_greedy`` bit for bit, and malformed plug-ins must fail closed.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import CounterfactualBranch as CB  # noqa: E402
import nmcc_testbed
from DecisionInterface import (
    PolicyDecision,
    load_external_policy,
    validated_external_decision,
)


def _episode(strategy, *, plugin="comparators.public_score_greedy:PublicScoreGreedy"):
    from RLBridge import RLBridge

    core = nmcc_testbed.build(
        grid=8, cell_x=4, cell_y=4, population=120, stop_time=31,
        hazard_count=2, scenario_seed=6551,
    )
    core.observationEveryMinute = False
    core.externalPolicy = plugin
    core.externalPolicyParams = {}
    with tempfile.TemporaryDirectory() as directory:
        bridge = RLBridge(
            core,
            train_mode=False,
            deployment_strategy=strategy,
            target_active_shelters=len(core.shelterDS.shelterList) + 3,
            shelter_action_interval=3,
            first_decision_time=1,
            policy_seed=13,
            rollout_episodes=1,
            checkpoint_path=os.path.join(directory, "policy.pt"),
            diagnostics_path=os.path.join(directory, "diagnostics.csv"),
        )
        digests, rows = [], []
        for minute in range(1, int(core.stopTime)):
            CB.advance_one_timestep(core)
            result = bridge.step(
                simulation_time=minute, is_terminal=minute == int(core.stopTime) - 1
            )
            rows.append(
                {
                    key: value
                    for key, value in result.items()
                    if not key.endswith("latency_ms")
                }
            )
            digests.append(CB.live_state_digest(core))
            if bridge.episode_done:
                break
        return dict(
            digests=digests,
            rows=rows,
            objective=(
                bridge.objective_active_person_time,
                bridge.objective_hazard_exposure_person_time,
                bridge.objective_expected_casualty_person_time,
            ),
            deployment_times=list(bridge.deployment_times),
        )


class _Plugin:
    INFORMATION_CONTRACT = {
        "role": "test", "label": "test", "information": "test", "timing": "test",
    }

    def __init__(self, *, core, params):
        self.params = params

    def select(self, observation, deterministic=True):
        return PolicyDecision(action_index=int(self.params["action"]), strategy="x")


class _NoContract:
    def __init__(self, *, core, params):
        pass

    def select(self, observation, deterministic=True):
        return None


class V651ExternalComparatorTests(unittest.TestCase):
    def test_reference_plugin_reproduces_objective_greedy_bitwise(self):
        reference = _episode("objective_greedy")
        plugin = _episode("external")
        self.assertGreaterEqual(len(reference["deployment_times"]), 1)
        self.assertEqual(reference["digests"], plugin["digests"])
        self.assertEqual(reference["objective"], plugin["objective"])
        self.assertEqual(reference["deployment_times"], plugin["deployment_times"])
        self.assertEqual(len(reference["rows"]), len(plugin["rows"]))
        for left, right in zip(reference["rows"], plugin["rows"]):
            self.assertEqual(set(left), set(right))
            for key in left:
                a, b = left[key], right[key]
                same = (
                    a == b
                    or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))
                )
                self.assertTrue(same, (key, a, b))

    def test_missing_or_malformed_specification_fails_closed(self):
        for spec in ("", "comparators.public_score_greedy", ":PublicScoreGreedy"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                load_external_policy(SimpleNamespace(externalPolicy=spec, externalPolicyParams={}))
        with self.assertRaises(AttributeError):
            load_external_policy(SimpleNamespace(
                externalPolicy="comparators.public_score_greedy:Missing",
                externalPolicyParams={},
            ))
        with self.assertRaises(TypeError):
            load_external_policy(SimpleNamespace(
                externalPolicy=f"{__name__}:_NoContract", externalPolicyParams={},
            ))
        with self.assertRaises(TypeError):
            load_external_policy(SimpleNamespace(
                externalPolicy=f"{__name__}:_Plugin", externalPolicyParams=[1],
            ))
        with self.assertRaises(ValueError):
            load_external_policy(SimpleNamespace(
                externalPolicy="comparators.public_score_greedy:PublicScoreGreedy",
                externalPolicyParams={"unexpected": 1},
            ))

    def test_infeasible_plugin_action_is_rejected(self):
        observation = SimpleNamespace(action_mask=np.array([True, False, True]))
        self.assertEqual(
            validated_external_decision(
                PolicyDecision(action_index=2, strategy="x"), observation
            ).action_index,
            2,
        )
        for action in (1, 3, -1):
            with self.subTest(action=action), self.assertRaises(RuntimeError):
                validated_external_decision(
                    PolicyDecision(action_index=action, strategy="x"), observation
                )
        with self.assertRaises(TypeError):
            validated_external_decision(2, observation)

    def test_plugin_receives_parameters(self):
        policy = load_external_policy(SimpleNamespace(
            externalPolicy=f"{__name__}:_Plugin", externalPolicyParams={"action": 0},
        ))
        self.assertEqual(policy.params, {"action": 0})


if __name__ == "__main__":
    unittest.main()
