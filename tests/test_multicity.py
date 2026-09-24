import tempfile
import unittest
from unittest import mock

import networkx as nx

from CityProfiles import load_city_suite
from Core import Core
from MapDatabase import MapDS
from OSMProcessor import OSMProcessor
from ShelterTypes import canonical_shelter_site_type
from multicity_backtest import (
    DYNAMIC_STRATEGIES,
    _city_overrides,
    _curriculum_is_monotone_extension,
    _equal_city_training_blocks,
    _learning_assessment,
    _manifest_command_fields,
    _manifest_time_fields,
    _parse_args,
    _stratified_paired_analysis,
    _scale_trend_diagnostics,
    _preflight_maps,
    _validate_resume_rows,
    _validate_shared_interface,
    balanced_city_schedule,
    balanced_rollout_episodes,
)


class CityProfileTests(unittest.TestCase):
    def test_all_dynamic_benchmarks_are_registered(self):
        self.assertTrue(
            {
                "risk_reduction",
                "heuristic",
                "hazard_weighted",
                "accessibility_deficit",
                "random",
            }.issubset(
                DYNAMIC_STRATEGIES
            )
        )

    def test_default_suite_is_strictly_increasing_and_has_shared_action_space(self):
        suite = load_city_suite()
        self.assertEqual(len(suite.cities), 5)
        self.assertEqual(
            [city.city_id for city in suite.cities],
            [
                "malibu_ca",
                "state_college_pa",
                "spokane_wa",
                "seattle_wa",
                "chicago_il",
            ],
        )
        self.assertIn("Southern California", suite.selection_basis)
        self.assertIn("north and west Los Angeles", suite.selection_basis)
        self.assertIn("not mislabeled as the San Francisco Bay Area", suite.selection_basis)
        self.assertEqual(
            [city.census_2020_population for city in suite.cities],
            sorted(city.census_2020_population for city in suite.cities),
        )
        self.assertEqual(suite.cities[0].center_lat, 33.98)
        self.assertEqual(suite.cities[0].center_lon, -118.6)
        self.assertEqual(suite.cities[0].radius_m, 35000.0)
        overrides = {
            city.city_id: _city_overrides(suite, city, {}, "stochastic")
            for city in suite.cities
        }
        self.assertEqual(_validate_shared_interface(overrides), (8, 8))
        self.assertTrue(suite.common_experiment["congestionEnabled"])
        self.assertEqual(suite.common_experiment["timeStepMinutes"], 1.0)
        self.assertEqual(
            suite.common_experiment["congestionJamDensityPedPerM2"], 5.4
        )
        self.assertEqual(suite.common_experiment["congestionSubstepSeconds"], 10.0)
        self.assertTrue(suite.common_experiment["socialForceEnabled"])
        self.assertTrue(suite.common_experiment["intersectionConsolidationEnabled"])
        self.assertEqual(suite.common_experiment["panicHerdProbability"], 0.5)
        self.assertEqual(suite.common_experiment["shelterActionInterval"], 10)
        self.assertEqual(
            suite.common_experiment["hazardCasualtyReferenceMinutes"], 60.0
        )

    def test_balanced_schedule_is_exact_and_reproducible(self):
        cities = load_city_suite().cities
        first = balanced_city_schedule(cities, 7, 19)
        second = balanced_city_schedule(cities, 7, 19)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 35)
        for city in cities:
            self.assertEqual(sum(item.city_id == city.city_id for item in first), 7)
        for start in range(0, len(first), len(cities)):
            self.assertEqual(
                {city.city_id for city in first[start:start + len(cities)]},
                {city.city_id for city in cities},
            )

    def test_rollout_contains_only_complete_city_blocks(self):
        self.assertEqual(balanced_rollout_episodes(1), 8)
        self.assertEqual(balanced_rollout_episodes(4), 8)
        self.assertEqual(balanced_rollout_episodes(5), 10)
        self.assertEqual(balanced_rollout_episodes(7), 14)
        for city_count in range(1, 9):
            rollout = balanced_rollout_episodes(city_count)
            self.assertGreaterEqual(rollout, 8)
            self.assertEqual(rollout % city_count, 0)

    def test_pooled_learning_rate_is_frozen_and_validated(self):
        self.assertEqual(_parse_args(["--train-only"]).learning_rate, 3e-4)
        with self.assertRaises(SystemExit):
            _parse_args(["--train-only", "--learning-rate", "0"])

    def test_resume_preserves_original_manifest_start_time(self):
        first = _manifest_time_fields(None)
        resumed = _manifest_time_fields({"started_utc": first["started_utc"]})
        self.assertEqual(resumed["started_utc"], first["started_utc"])
        self.assertIn("resumed_utc", resumed)

    def test_resume_preserves_original_manifest_command(self):
        original = ["python", "multicity_backtest.py", "--train-only"]
        resumed = [*original, "--resume"]
        fields = _manifest_command_fields({"command": original}, resumed)
        self.assertEqual(fields["command"], original)
        self.assertEqual(fields["resume_command"], resumed)

    def test_training_plot_finalizer_uses_headless_backend(self):
        from backtest import _plot_outputs

        rows = [
            {
                "replication": episode,
                "policy_replication": 1,
                "episode_return": 0.01 * episode,
                "safe_completion_reward": 0.5,
                "casualty_penalty": -0.01,
                "risk_time_penalty": -0.2,
                "entropy": 0.9,
                "approximate_kl": 0.01 if episode == 5 else 0.0,
                "optimizer_updated": 1.0 if episode == 5 else 0.0,
                "heuristic_agreement_rate": 0.4,
            }
            for episode in range(1, 6)
        ]
        with tempfile.TemporaryDirectory() as directory:
            outputs = _plot_outputs(directory, rows, [])
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                with open(output, "rb") as handle:
                    self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")

    def test_training_policy_shards_preserve_declared_replication_indices(self):
        default = _parse_args(["--train-only", "--policy-replicates", "4"])
        self.assertEqual((1, 2, 3, 4), default.training_policy_indices)
        shard = _parse_args(
            [
                "--train-only",
                "--policy-replicates",
                "8",
                "--training-policy-indices",
                "2,7",
            ]
        )
        self.assertEqual((2, 7), shard.training_policy_indices)
        with self.assertRaises(SystemExit):
            _parse_args(
                [
                    "--policy-replicates",
                    "8",
                    "--training-policy-indices",
                    "2",
                ]
            )
        with self.assertRaises(SystemExit):
            _parse_args(
                [
                    "--train-only",
                    "--policy-replicates",
                    "8",
                    "--training-policy-indices",
                    "2,9",
                ]
            )

    def test_cli_override_cannot_silently_change_map_identity(self):
        suite = load_city_suite()
        city = suite.cities[0]
        with self.assertRaisesRegex(ValueError, "profile file"):
            _city_overrides(suite, city, {"mapRadiusM": 9999}, "stochastic")
        override = _city_overrides(suite, city, {"pedVol": 25}, "deterministic")
        self.assertEqual(override["pedVol"], 25)
        self.assertEqual(override["cityID"], city.city_id)
        self.assertEqual(override["hazardEvolutionMode"], "deterministic")

    def test_pooled_interface_requires_one_partition_contract(self):
        suite = load_city_suite()
        cities = suite.cities[:2]
        overrides = {
            city.city_id: _city_overrides(
                suite,
                city,
                {"cellPartitionMode": "equal_area"},
                "stochastic",
            )
            for city in cities
        }
        self.assertEqual(_validate_shared_interface(overrides), (8, 8))
        overrides[cities[1].city_id]["cellPartitionMode"] = "node_density_adaptive"
        with self.assertRaisesRegex(ValueError, "partition mode"):
            _validate_shared_interface(overrides)

    def test_resume_rows_must_follow_preregistered_city_schedule(self):
        cities = load_city_suite().cities[:2]
        schedule = balanced_city_schedule(cities, 2, 7)
        rows = [
            {
                "policy_replication": 1,
                "replication": index,
                "city_id": city.city_id,
            }
            for index, city in enumerate(schedule, start=1)
        ]
        _validate_resume_rows(rows, schedule, 1)
        rows[1]["city_id"] = rows[0]["city_id"]
        with self.assertRaisesRegex(ValueError, "city schedule"):
            _validate_resume_rows(rows, schedule, 1)

    def test_curriculum_continuation_only_accepts_a_monotone_extension(self):
        stage = {
            "stage_id": "S1",
            "label": "stationary",
            "episodes_per_city": 120,
            "variants": [
                {
                    "variant_id": "nominal",
                    "weight": 1,
                    "overrides": {"pedVol": 2500, "hazardVol": 3},
                }
            ],
        }
        previous = {
            "schema_version": 1,
            "curriculum_id": "stationary_v1",
            "description": "fixed contract",
            "episodes_per_city": 120,
            "stages": [stage],
            "source_path": "initial.json",
            "source_sha256": "old",
        }
        extended_stage = dict(stage, episodes_per_city=160)
        extended = dict(
            previous,
            episodes_per_city=160,
            stages=[extended_stage],
            source_path="extended.json",
            source_sha256="new",
        )
        self.assertTrue(_curriculum_is_monotone_extension(previous, extended))

        changed_population = dict(extended_stage)
        changed_population["variants"] = [
            {
                "variant_id": "nominal",
                "weight": 1,
                "overrides": {"pedVol": 2000, "hazardVol": 3},
            }
        ]
        self.assertFalse(
            _curriculum_is_monotone_extension(
                previous,
                dict(extended, stages=[changed_population]),
            )
        )
        self.assertFalse(
            _curriculum_is_monotone_extension(
                previous,
                dict(extended, episodes_per_city=80, stages=[dict(stage, episodes_per_city=80)]),
            )
        )

    def test_convergence_blocks_remove_city_scale_level_effects(self):
        cities = load_city_suite().cities[:2]
        rows = []
        order = (cities[1], cities[0], cities[0], cities[1])
        for replication, city in enumerate(order, 1):
            baseline = 10.0 if city.city_id == cities[1].city_id else 0.0
            rows.append(
                {
                    "replication": replication,
                    "policy_replication": 1,
                    "city_id": city.city_id,
                    "episode_return": baseline + 0.5,
                    "entropy": 1.0,
                    "optimizer_updated": float(replication % 2 == 0),
                    "approximate_kl": 0.01,
                    "gradient_norm": 0.1,
                    "policy_loss": 0.0,
                    "value_loss": 0.1,
                    "rollout_episodes_pending": 0.0,
                }
            )
        blocks = _equal_city_training_blocks(rows, cities)
        self.assertEqual(len(blocks), 2)
        self.assertEqual([row["episode_return"] for row in blocks], [5.5, 5.5])
        self.assertTrue(all(row["optimizer_updated"] == 1.0 for row in blocks))


class MapSpecificationTests(unittest.TestCase):
    def test_amenity_type_controls_candidate_capacity_when_building_is_generic(self):
        class CellTracker:
            @staticmethod
            def locateCell(x_m, y_m):
                return (0, 0)

        raw_nodes = [
            (
                101,
                {
                    "x": -75.0,
                    "y": 40.0,
                    "street_count": 1,
                    "building_type": "yes",
                    "amenity_type": "library",
                },
            )
        ]
        database = MapDS(raw_nodes, [], "Test City", nx.MultiDiGraph())
        database.nodeInit(CellTracker())
        node = database.nodeListByLocalID[0]
        self.assertEqual(canonical_shelter_site_type("yes", "library"), "library")
        self.assertEqual(node.buildingType, "library")
        self.assertEqual(node.nodeCap, 300)
        self.assertIs(database.shelterCanList[0], node)

    def test_semicolon_tag_uses_first_recognized_function(self):
        self.assertEqual(
            canonical_shelter_site_type("yes", "cafe;community_centre"),
            "community_centre",
        )

    def test_point_query_dispatches_exact_center_and_radius(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 2)
        processor = OSMProcessor(
            "Test City",
            query_mode="point",
            center_point=(40.0, -75.0),
            radius_m=1234,
        )
        with mock.patch("OSMProcessor.OSM.graph.graph_from_point", return_value=graph) as call:
            output = processor._try_graph_from_place("https://example.invalid/api")
        self.assertIs(output, graph)
        call.assert_called_once_with(
            (40.0, -75.0),
            dist=1234.0,
            dist_type="bbox",
            network_type="walk",
            truncate_by_edge=False,
        )

    def test_point_cache_identity_includes_radius(self):
        first = OSMProcessor(
            "Test City", query_mode="point", center_point=(40.0, -75.0), radius_m=1000
        )
        second = OSMProcessor(
            "Test City", query_mode="point", center_point=(40.0, -75.0), radius_m=2000
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("OSMProcessor.OSM.settings.cache_folder", directory):
                self.assertNotEqual(first._graph_cache_path(), second._graph_cache_path())

    def test_core_rejects_incomplete_point_specification(self):
        core = Core("test")
        core.address = "Test City"
        core.cityID = "test_city"
        core.mapQueryMode = "point"
        core.mapCenterLat = 40.0
        core.mapCenterLon = None
        core.mapRadiusM = 1000
        core.stopTime = 10
        core.pedVol = 10
        core.hazardVol = 1
        core.maxSpeed = 10
        core.cellX = 2
        core.cellY = 2
        core.shelterCanVol = 4
        core.initShelterVol = 1
        core.learningRate = 3e-4
        core.explorationRate = 0
        core.optimizer = "AdamW"
        with self.assertRaisesRegex(ValueError, "point map queries require"):
            core._validate_effective_configuration()

    def test_preflight_is_complete_only_after_every_requested_city(self):
        graph = nx.MultiDiGraph()
        graph.add_edge(1, 2)

        class FakeProcessor:
            def __init__(self, *args, **kwargs):
                self.locationDrive = graph
                self.buildingNodes = {1: {}}

            def setLocationDrive(self):
                return None

            def setNodeEdgeSets(self):
                return None

            def setIntersectionStreetCount(self):
                return None

            def setBuildingOnly(self):
                return None

            def _graph_cache_path(self):
                return "/tmp/fake.graphml"

            def graph_provenance(self):
                return {"graph_nodes": 2, "graph_edges": 1}

        cities = load_city_suite().cities[:2]
        with tempfile.TemporaryDirectory() as directory:
            output = f"{directory}/preflight.json"
            with mock.patch("OSMProcessor.OSMProcessor", FakeProcessor):
                result = _preflight_maps(cities, output)
        self.assertTrue(result["complete"])
        self.assertTrue(result["all_ready"])
        self.assertEqual(result["requested_cities"], [city.city_id for city in cities])
        self.assertEqual(len(result["results"]), 2)


class StratifiedAnalysisTests(unittest.TestCase):
    @staticmethod
    def _row(city_id, city_scenario, global_replication, policy, strategy, delta):
        is_rl = strategy == "rl"
        return {
            "city_id": city_id,
            "city_scenario_replication": city_scenario,
            "replication": global_replication,
            "policy_replication": policy,
            "deployment_strategy": strategy,
            "episode_return": 1.0 + (delta if is_rl else 0.0),
            "safe_completed": 50.0 + (10.0 * delta if is_rl else 0.0),
            "casualty": 5.0 - (10.0 * delta if is_rl else 0.0),
            "unfinished": 10.0 - (10.0 * delta if is_rl else 0.0),
            "restricted_mean_time_to_safety": 20.0 - (delta if is_rl else 0.0),
            "normalized_risk_weighted_person_time": 0.5 - (delta if is_rl else 0.0),
        }

    def test_macro_analysis_weights_cities_equally(self):
        cities = load_city_suite().cities[:2]
        rows = []
        global_replication = 0
        for city, delta in zip(cities, (0.1, 0.3)):
            for scenario in (1, 2):
                global_replication += 1
                rows.append(self._row(city.city_id, scenario, global_replication, 0, "heuristic", 0))
                for policy in (1, 2):
                    rows.append(self._row(city.city_id, scenario, global_replication, policy, "rl", delta))
        analysis = _stratified_paired_analysis(rows, cities, 73, 200)
        primary = next(row for row in analysis if row["metric"] == "episode_return")
        self.assertAlmostEqual(primary["mean_rl_improvement"], 0.2)
        self.assertAlmostEqual(primary["bootstrap_95_ci_low"], 0.2)
        self.assertAlmostEqual(primary["bootstrap_95_ci_high"], 0.2)
        self.assertEqual(primary["cities"], 2)
        self.assertEqual(primary["scenarios_per_city"], 2)
        self.assertTrue(primary["inferentially_eligible"])
        per_city = []
        for city, effect in zip(cities, (0.1, 0.3)):
            per_city.append(
                {
                    "scope": "city",
                    "city_id": city.city_id,
                    "metric": "episode_return",
                    "mean_rl_improvement": effect,
                }
            )
        trend = _scale_trend_diagnostics(per_city, cities)[0]
        self.assertAlmostEqual(trend["slope_per_scale_rank"], 0.2)
        self.assertGreater(trend["pearson_correlation_with_scale_rank"], 0.99)

    def test_macro_superiority_is_disabled_without_within_city_replication(self):
        cities = load_city_suite().cities[:2]
        rows = []
        for global_replication, city in enumerate(cities, start=1):
            rows.append(self._row(city.city_id, 1, global_replication, 0, "heuristic", 0))
            for policy in (1, 2):
                rows.append(
                    self._row(city.city_id, 1, global_replication, policy, "rl", 0.2)
                )
        analysis = _stratified_paired_analysis(rows, cities, 73, 200)
        primary = next(row for row in analysis if row["metric"] == "episode_return")
        self.assertFalse(primary["inferentially_eligible"])
        self.assertFalse(primary["superiority_ci_excludes_zero"])

    def test_single_policy_macro_inference_is_conditional_on_fixed_checkpoint(self):
        cities = load_city_suite().cities[:2]
        rows = []
        global_replication = 0
        for city in cities:
            for scenario in (1, 2, 3):
                global_replication += 1
                rows.append(
                    self._row(
                        city.city_id,
                        scenario,
                        global_replication,
                        0,
                        "heuristic",
                        0,
                    )
                )
                rows.append(
                    self._row(
                        city.city_id,
                        scenario,
                        global_replication,
                        1,
                        "rl",
                        0.2,
                    )
                )
        analysis = _stratified_paired_analysis(rows, cities, 73, 200)
        primary = next(row for row in analysis if row["metric"] == "episode_return")
        self.assertTrue(primary["inferentially_eligible"])
        self.assertEqual(
            primary["inference_scope"],
            "conditional_on_one_fixed_trained_policy",
        )
        self.assertIsNone(primary["two_sided_randomization_p"])
        self.assertTrue(primary["superiority_ci_excludes_zero"])

    def test_learning_assessment_does_not_confuse_a_positive_macro_with_robustness(self):
        analysis = [
            {
                "scope": "macro_all_cities",
                "metric": "episode_return",
                "mean_rl_improvement": 0.1,
            },
            {
                "scope": "macro_all_cities",
                "metric": "casualty",
                "mean_rl_improvement": -0.01,
            },
            {
                "scope": "city",
                "city_id": "small",
                "metric": "episode_return",
                "mean_rl_improvement": 0.2,
            },
            {
                "scope": "city",
                "city_id": "large",
                "metric": "episode_return",
                "mean_rl_improvement": -0.1,
            },
            {
                "scope": "city",
                "city_id": "small",
                "metric": "casualty",
                "mean_rl_improvement": 0.0,
            },
            {
                "scope": "city",
                "city_id": "large",
                "metric": "casualty",
                "mean_rl_improvement": -0.02,
            },
        ]
        result = _learning_assessment(
            {"all_policies_converged": True},
            {"inferentially_eligible": False},
            analysis,
        )
        self.assertEqual(result["status"], "descriptively_promising_not_confirmed")
        self.assertFalse(result["qualifies_as_learning_well_cross_city"])
        self.assertFalse(result["all_city_return_point_estimates_nonnegative"])
        self.assertFalse(result["aggregate_casualty_point_estimate_nonworsening"])


if __name__ == "__main__":
    unittest.main()
