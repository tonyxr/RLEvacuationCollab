"""Pure tests of the evaluation stress-scenario constructors (no simulator)."""

import math
import unittest

import numpy as np

from HazardDatabase import HazardDS
from StressScenarios import (
    StressScenarioSpec,
    bearing_degrees,
    clustered_origin_plan,
    downwind_front_speed,
    expand_stress_families,
    out_of_support_keys,
    paired_dose_slope,
    population_centroid,
    threat_source_index,
    training_support,
)


class _CenterTracker:
    def __init__(self, centers):
        self.centers = centers

    def getCellCenter(self, cell):
        return self.centers[tuple(cell)]


def _grid_nodes(side=41, spacing=100.0):
    axis = np.arange(side, dtype=np.float64) * spacing
    xs, ys = np.meshgrid(axis, axis)
    return np.column_stack((xs.ravel(), ys.ravel()))


class GeometryTests(unittest.TestCase):
    def test_bearing_matches_hazard_wind_convention(self):
        # Map y grows southward. A target north-east of the source.
        source, target = (0.0, 1000.0), (1000.0, 0.0)
        self.assertAlmostEqual(bearing_degrees(source, target), 45.0)
        self.assertAlmostEqual(bearing_degrees((0, 0), (0, 500)), 270.0)
        hazard = HazardDS(1, (10, 4), (1, 2), (10, 5), wind_speed_m_per_minute=30.0,
                          wind_direction_degrees=bearing_degrees(source, target))
        hazard.cellTracker = _CenterTracker({(0, 0): source, (1, 1): target})
        downwind = hazard.directional_spread_multiplier((0, 0), (1, 1))
        upwind = hazard.directional_spread_multiplier((1, 1), (0, 0))
        self.assertGreater(downwind, 1.0)
        self.assertLess(upwind, 1.0)
        self.assertAlmostEqual(
            downwind * hazard.spread_speed_m_per_minute,
            downwind_front_speed(hazard.spread_speed_m_per_minute, 30.0),
        )

    def test_zero_wind_front_speed_is_the_isotropic_speed(self):
        self.assertEqual(downwind_front_speed(33.3, 0.0), 33.3)


class SpecTests(unittest.TestCase):
    def test_invalid_specifications_fail(self):
        with self.assertRaisesRegex(ValueError, "sum to at most 1"):
            StressScenarioSpec(cluster_shares=(0.7, 0.5))
        with self.assertRaisesRegex(ValueError, "stressThreatenedCluster"):
            StressScenarioSpec(cluster_shares=(0.5,), threatened_cluster=1)
        with self.assertRaisesRegex(ValueError, "stressThreatArrivalMinutes"):
            StressScenarioSpec(cluster_shares=(0.5,), threatened_cluster=0)
        with self.assertRaisesRegex(ValueError, "requires stressThreatenedCluster"):
            StressScenarioSpec(wind_aim="threatened_cluster")
        with self.assertRaisesRegex(ValueError, "stressWindAim"):
            StressScenarioSpec(wind_aim="downhill")
        self.assertFalse(StressScenarioSpec().active)


class OriginPlanTests(unittest.TestCase):
    def setUp(self):
        self.xy = _grid_nodes()
        self.spec = StressScenarioSpec(
            cluster_shares=(0.6, 0.25), cluster_radius_m=250.0,
            cluster_min_separation_m=1500.0,
        )

    def test_conserves_people_and_places_shares_in_compact_separated_clusters(self):
        plan = clustered_origin_plan(self.xy, 2500, 1, self.spec, seed=17)
        self.assertEqual(int(plan["node_population"].sum()), 2500)
        self.assertEqual(len(plan["origin_indices"]), 2500)
        anchors = plan["anchor_indices"]
        self.assertEqual(len(anchors), 2)
        self.assertGreaterEqual(
            float(np.hypot(*(self.xy[anchors[0]] - self.xy[anchors[1]]))), 1500.0
        )
        origins = self.xy[np.asarray(plan["origin_indices"])]
        for share, anchor in zip(self.spec.cluster_shares, anchors):
            near = np.hypot(*(origins - self.xy[anchor]).T) <= 250.0
            self.assertGreaterEqual(near.mean(), share - 1e-9)

    def test_same_seed_same_plan_and_different_seed_different_plan(self):
        first = clustered_origin_plan(self.xy, 500, 1, self.spec, seed=3)
        second = clustered_origin_plan(self.xy, 500, 1, self.spec, seed=3)
        third = clustered_origin_plan(self.xy, 500, 1, self.spec, seed=4)
        self.assertEqual(first["origin_indices"], second["origin_indices"])
        self.assertNotEqual(first["origin_indices"], third["origin_indices"])

    def test_grouped_population_is_conserved(self):
        plan = clustered_origin_plan(self.xy, 1003, 4, self.spec, seed=9)
        self.assertEqual(sum(plan["group_sizes"]), 1003)
        self.assertEqual(int(plan["node_population"].sum()), 1003)

    def test_population_centroid_moves_toward_the_heavier_cluster(self):
        plan = clustered_origin_plan(self.xy, 2500, 1, self.spec, seed=17)
        centroid = population_centroid(self.xy, plan["node_population"])
        heavy, light = (self.xy[index] for index in plan["anchor_indices"])
        self.assertLess(np.hypot(*(centroid - heavy)), np.hypot(*(centroid - light)))


class ThreatSourceTests(unittest.TestCase):
    def test_source_lies_in_the_distance_band_and_away_from_the_safe_cluster(self):
        xy = _grid_nodes()
        threatened, safe = 20 * 41 + 10, 20 * 41 + 30
        source = threat_source_index(
            xy, threatened, 800.0, seed=5, other_anchor_indices=(safe,)
        )
        distance = float(np.hypot(*(xy[source] - xy[threatened])))
        self.assertLessEqual(abs(distance - 800.0), 80.0)
        self.assertGreater(float(np.hypot(*(xy[source] - xy[safe]))), distance)


class EvaluationSpecTests(unittest.TestCase):
    BASE = {"pedVol": 2500, "initShelterVol": 2, "maxAdditionalShelters": 8,
            "shelterCapacitySurplus": 200, "panicRate": 0.5}

    def test_doses_share_tapes_and_derive_capacity(self):
        spec = {"stress_families": {
            "congestion_cluster": {
                "dose_key": "pedVol", "doses": [1000, 2500, 5000],
                "tuning_dose": 2500, "capacity_token_from_population": True,
                "overrides": {"stressClusterShares": [0.5]},
            },
            "panic_cliff": {"dose_key": "panicRate", "doses": [0.25, 0.75],
                            "tuning_dose": 0.25, "overrides": {}},
        }}
        entries = expand_stress_families(spec, self.BASE)
        self.assertEqual([entry["name"] for entry in entries],
                         ["congestion_cluster@1000", "congestion_cluster@2500",
                          "congestion_cluster@5000", "panic_cliff@0.25", "panic_cliff@0.75"])
        self.assertEqual(len({entry["seed_index"] for entry in entries[:3]}), 1)
        self.assertNotEqual(entries[0]["seed_index"], entries[3]["seed_index"])
        self.assertEqual([entry["overrides"]["shelterCapacityToken"] for entry in entries[:3]],
                         [120, 270, 520])
        self.assertEqual(sum(entry["is_tuning_dose"] for entry in entries[:3]), 1)

    def test_training_support_flags_extrapolation(self):
        curriculum = {
            "learner_overrides": {"panicRate": 0.5},
            "stages": [{"stage_id": "P", "variants": [{"overrides": {"pedVol": 2500}}]}],
        }
        support = training_support(curriculum, stage_ids=("P",))
        self.assertEqual(out_of_support_keys({"pedVol": 2500, "panicRate": 0.5}, support), [])
        self.assertEqual(
            out_of_support_keys({"pedVol": 5000, "hazardWindSpeedMPerMinute": 0.0,
                                 "stressScenarioFamily": "x"}, support),
            ["pedVol"],
        )
        self.assertEqual(
            out_of_support_keys({"hazardWindSpeedMPerMinute": 33.3}, support),
            ["hazardWindSpeedMPerMinute"],
        )

    def test_paired_dose_slope_recovers_a_common_trend(self):
        gaps = {tape: {0.0: 0.01 * tape, 1.0: 0.01 * tape + 0.05, 2.0: 0.01 * tape + 0.10}
                for tape in range(6)}
        slope, se, count = paired_dose_slope(gaps)
        self.assertAlmostEqual(slope, 0.05)
        self.assertAlmostEqual(se, 0.0)
        self.assertEqual(count, 6)


if __name__ == "__main__":
    unittest.main()
