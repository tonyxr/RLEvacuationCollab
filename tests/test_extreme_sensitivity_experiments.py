import json
import unittest

import numpy as np

from DecisionInterface import CELL_FEATURE_NAMES
from extreme_sensitivity_experiments import (
    _exact_sign_p,
    extreme_case,
    load_city_specs,
    sensitivity_conditions,
)


class ExtremeSensitivityExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("config/extreme_sensitivity_experiment.json", encoding="utf-8") as handle:
            cls.config = json.load(handle)
        cls.cities = load_city_specs(cls.config)
        cls.spokane = next(city for city in cls.cities if city.city_id == "spokane_wa")

    def test_registered_one_factor_design_has_thirteen_unique_conditions(self):
        conditions = sensitivity_conditions(self.config, self.cities)
        self.assertEqual(len(conditions), 13)
        memberships = sum(len(condition["axes"]) for condition in conditions)
        self.assertEqual(memberships, 15)
        base = [condition for condition in conditions if len(condition["axes"]) == 3]
        self.assertEqual(len(base), 1)

    def test_all_extreme_variants_have_empty_center_and_occupied_ring(self):
        for variant in self.config["extreme_variants"]:
            with self.subTest(variant=variant):
                case = extreme_case(
                    np.random.default_rng(17),
                    variant,
                    side=9,
                    population=30_000,
                    city=self.spokane,
                    service_range_m=1_300.0,
                )
                self.assertEqual(float(case["active"][case["center"]]), 0.0)
                self.assertTrue(np.all(case["active"][case["ring"]] > 0.0))
                self.assertAlmostEqual(float(np.sum(case["active"])), 1.0, places=6)
                self.assertEqual(
                    case["cell_features"].shape,
                    (81, len(CELL_FEATURE_NAMES)),
                )
                self.assertEqual(case["mask"].shape, (81,))

    def test_center_unavailable_is_enforced_and_other_centers_are_candidates(self):
        unavailable = extreme_case(
            np.random.default_rng(21),
            "center_unavailable",
            side=9,
            population=30_000,
            city=self.spokane,
            service_range_m=1_300.0,
        )
        self.assertFalse(bool(unavailable["mask"][unavailable["center"]]))
        for variant in ("safe_center", "dangerous_center", "asymmetric_ring"):
            case = extreme_case(
                np.random.default_rng(21),
                variant,
                side=9,
                population=30_000,
                city=self.spokane,
                service_range_m=1_300.0,
            )
            self.assertTrue(bool(case["mask"][case["center"]]))

    def test_static_plan_does_not_use_realized_danger(self):
        safe = extreme_case(
            np.random.default_rng(31),
            "safe_center",
            side=9,
            population=30_000,
            city=self.spokane,
            service_range_m=1_300.0,
        )
        dangerous = extreme_case(
            np.random.default_rng(31),
            "dangerous_center",
            side=9,
            population=30_000,
            city=self.spokane,
            service_range_m=1_300.0,
        )
        self.assertEqual(safe["static"], dangerous["static"])

    def test_sixteen_seeds_can_resolve_registered_holm_family(self):
        minimum_two_sided = _exact_sign_p(np.ones(16, dtype=float))
        self.assertAlmostEqual(minimum_two_sided, 2.0 / (2.0**16))
        self.assertLess(120.0 * minimum_two_sided, 0.05)


if __name__ == "__main__":
    unittest.main()
