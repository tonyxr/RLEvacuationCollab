"""v64 casualty-pricing contract tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import CounterfactualBranch as CB
from Core import Core
from TrainingCurriculum import LEARNER_STAGE_OVERRIDES
from nmcc_testbed import build


class V64CasualtyPriceTests(unittest.TestCase):
    def test_harm_monotone_price_dominates_remaining_living_burden(self):
        core = build(stop_time=61)
        core.casualtyPriceRule = "harm_monotone"
        core.activePersonTimeWeight = 0.5
        core.hazardExposureTimeWeight = 2.0
        horizon = 60
        for kappa in (1.0, 2.0):
            core.casualtyPriorityKappa = kappa
            for time_index in range(horizon):
                price = CB.casualty_loss_weight(core, time_index)
                for danger in np.linspace(0.0, 1.0, 6):
                    burden = (0.5 + 2.0 * danger) * (
                        (horizon - time_index) + horizon
                    )
                    self.assertGreaterEqual(price, burden - 1e-9)
            if kappa == 1.0:
                expected = 2.5 * ((horizon - 17) + horizon)
                self.assertEqual(CB.casualty_loss_weight(core, 17), expected)

    def test_legacy_price_is_bitwise_v63_formula(self):
        core = build(stop_time=61)
        core.casualtyPriceRule = "legacy"
        for kappa in (1.0, 2.0, 3.25):
            core.casualtyPriorityKappa = kappa
            for time_index in range(60):
                self.assertEqual(
                    CB.casualty_loss_weight(core, time_index),
                    float(kappa * 2.0 * (60 - time_index)),
                )

    def test_config_validation_curriculum_and_portable_contract(self):
        core = Core("test")
        core.casualtyPriceRule = " OTHER "
        with self.assertRaisesRegex(ValueError, "legacy or harm_monotone"):
            core._validate_effective_configuration()
        self.assertIn("casualtyPriceRule", LEARNER_STAGE_OVERRIDES)
        self.assertIn(
            "casualtyPriceRule", CB._PORTABLE_REQUIRED_SIMULATION_ATTRIBUTES
        )
        self.assertIn("casualtyPriceRule", CB._PORTABLE_CORE_ATTRIBUTES)

    def test_missing_rule_retains_legacy_default(self):
        core = SimpleNamespace(
            stopTime=61,
            pedDS=SimpleNamespace(currTime=11),
            casualtyPriorityKappa=2.0,
        )
        self.assertEqual(CB.casualty_loss_weight(core), 2.0 * 2.0 * 49.0)


if __name__ == "__main__":
    unittest.main()
