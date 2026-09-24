import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import networkx as nx

import Guidance as guidance_module
import GuidanceDatabase as guidance_database_module
import OSMProcessor as osm_module
from Edge import Edge
from Guidance import Guidance
from GuidanceDatabase import GuidanceDS
from MapDatabase import MapDS
from OSMProcessor import OSMProcessor
from PedestrianDatabase import PedDS
from PolicyCache import PolicyCache, canonical_contract_key
from SocialForce import ForceProcessor
from factorial_backtest import (
    _latency_pairs,
    execution_plan,
    load_design,
    validate_source_training_manifest,
)


class _CellTracker:
    def __init__(self, state=0, heat=0.0, smoke=0.0):
        self.cell = SimpleNamespace(
            impactedLevel=int(state),
            heat=float(heat),
            smoke=float(smoke),
        )

    def getCellState(self, _cell):
        return int(self.cell.impactedLevel)

    def getCell(self, _i, _j):
        return self.cell

    def locateCell(self, _x, _y):
        return (0, 0)


class GuidanceDeprecationTests(unittest.TestCase):
    def test_guidance_modules_and_types_are_explicitly_deprecated(self):
        self.assertTrue(guidance_module.DEPRECATED)
        self.assertTrue(guidance_database_module.DEPRECATED)
        self.assertTrue(Guidance.DEPRECATED)
        self.assertTrue(GuidanceDS.DEPRECATED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Guidance(1, None, (0, 0), None, 0, 0, 0)
            GuidanceDS(0, 0)
        self.assertEqual(len(caught), 2)
        self.assertTrue(all(item.category is DeprecationWarning for item in caught))

    def test_map_initialization_never_promotes_intersections_to_guidance_candidates(self):
        raw_nodes = [
            (10, {"x": -77.0, "y": 40.0, "street_count": 8}),
            (11, {"x": -76.999, "y": 40.0, "street_count": 8}),
        ]
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(raw_nodes)
        graph.add_edge(10, 11, length=100.0, osmid=99)
        store = MapDS(raw_nodes, list(graph.edges(data=True)), "synthetic", graph)
        store.nodeInit(_CellTracker())
        self.assertEqual(store.guidanceCanList, {})
        self.assertEqual(store.guidanceList, {})


class SocialForceTests(unittest.TestCase):
    def test_equations_19_to_22_apply_safe_self_force_and_hazard_impact(self):
        processor = ForceProcessor()
        safe = SimpleNamespace(impactedLevel=1, heat=0.0, smoke=0.0)
        speed, self_force, impact = processor.compute(60.0, safe, max_speed=100.0)
        self.assertAlmostEqual(self_force, 1.0)
        self.assertAlmostEqual(impact, 0.0)
        self.assertAlmostEqual(speed, 63.0)

        hazardous = SimpleNamespace(impactedLevel=3, heat=40.0, smoke=150.0)
        speed, self_force, impact = processor.compute(60.0, hazardous, max_speed=100.0)
        self.assertAlmostEqual(self_force, 0.0)
        self.assertAlmostEqual(impact, 0.5)
        self.assertAlmostEqual(speed, 45.0)

    def test_pedestrian_transition_uses_social_force_before_network_movement(self):
        store = PedDS(1)
        tracker = _CellTracker(state=1)
        store.cellTracker = tracker
        store.maxSpeed = 100.0
        pedestrian = SimpleNamespace(
            agentID=0,
            group_size=1,
            terminated=False,
            currCell=(0, 0),
            currSpeed=60.0,
            social_force_speed=60.0,
            affected=False,
            distance_travelled_m=0.0,
        )
        store.pedAgentList[0] = pedestrian
        store.pedestrianHazardInteraction()
        self.assertAlmostEqual(pedestrian.currSpeed, 63.0)
        self.assertAlmostEqual(pedestrian.social_self_force, 1.0)
        self.assertEqual(store.socialForcePersonSteps, 1.0)


class PanicBehaviorTests(unittest.TestCase):
    def _pedestrian(self, node, agent_id=0):
        return SimpleNamespace(
            agentID=agent_id,
            group_size=1,
            terminated=False,
            panicked=False,
            panic_onset_time=None,
            panic_decision_count=0,
            panic_next_edge=None,
            currCell=(0, 0),
            currNode=node,
            currEdge=None,
            atNode=True,
            routeFollowing=object(),
        )

    def test_onset_is_first_exposure_gated_and_permanent(self):
        node = SimpleNamespace(OSMID=1)
        store = PedDS(1)
        store.cellTracker = _CellTracker(state=2)
        store.configure_panic(rate=1.0, random_seed=7)
        pedestrian = self._pedestrian(node)
        store.pedAgentList[0] = pedestrian
        self.assertEqual(store.updatePanicStates(), 0)
        store.cellTracker.cell.impactedLevel = 3
        self.assertEqual(store.updatePanicStates(), 1)
        self.assertTrue(pedestrian.panicked)
        self.assertIsNone(pedestrian.routeFollowing)
        store.cellTracker.cell.impactedLevel = 0
        self.assertEqual(store.updatePanicStates(), 0)
        self.assertTrue(pedestrian.panicked)

    def test_non_susceptible_person_is_not_retried_each_timestep(self):
        node = SimpleNamespace(OSMID=1)
        store = PedDS(1)
        store.cellTracker = _CellTracker(state=3)
        store.configure_panic(rate=0.5, random_seed=7)
        pedestrian = self._pedestrian(node)
        store.pedAgentList[0] = pedestrian
        with mock.patch.object(
            store,
            "_panic_susceptibility_uniform",
            return_value=0.9,
        ) as draw:
            self.assertEqual(store.updatePanicStates(), 0)
            self.assertEqual(store.updatePanicStates(), 0)
        self.assertEqual(draw.call_count, 1)
        self.assertTrue(pedestrian.panic_eligibility_evaluated)
        self.assertFalse(pedestrian.panicked)
        self.assertEqual(store._step["panic_eligible_first_exposure"], 1)

    def test_latent_susceptibility_is_independent_of_exposure_time(self):
        first = PedDS(1)
        delayed = PedDS(1)
        first.configure_panic(rate=0.5, random_seed=20260915)
        delayed.configure_panic(rate=0.5, random_seed=20260915)
        first.currTime = 1
        delayed.currTime = 47
        self.assertEqual(
            first._panic_susceptibility_uniform(1234),
            delayed._panic_susceptibility_uniform(1234),
        )
        self.assertNotEqual(
            first._panic_uniform(1234, 0, 1),
            delayed._panic_uniform(1234, 0, 1),
        )

    def test_realized_onset_and_herding_rates_match_configured_probabilities(self):
        population = 5000
        node = SimpleNamespace(OSMID=1, nodeX=0.0, nodeY=0.0)
        left = SimpleNamespace(OSMID=2, nodeX=-1.0, nodeY=0.0)
        right = SimpleNamespace(OSMID=3, nodeX=1.0, nodeY=0.0)
        edge_left = Edge(node, left, 1, 101, 1.0, 1.0, 0, (0, 0))
        edge_right = Edge(node, right, 2, 102, 1.0, 1.0, 0, (0, 0))
        store = PedDS(population)
        store.cellTracker = _CellTracker(state=3)
        store.mapDS = SimpleNamespace(
            incidentEdges=lambda _node: (edge_left, edge_right)
        )
        store.configure_panic(
            rate=0.3,
            herd_probability=0.5,
            random_seed=20260915,
        )
        for agent_id in range(population):
            store.pedAgentList[agent_id] = self._pedestrian(node, agent_id)
        onsets = store.updatePanicStates()
        self.assertAlmostEqual(onsets / population, 0.3, delta=0.025)
        for pedestrian in store.pedAgentList.values():
            if pedestrian.panicked:
                store._choose_panic_edge(
                    pedestrian,
                    {
                        store._edge_load_key(edge_left): 2,
                        store._edge_load_key(edge_right): 9,
                    },
                )
        choices = (
            store._step["panic_herd_choices"]
            + store._step["panic_random_choices"]
        )
        self.assertEqual(choices, onsets)
        self.assertAlmostEqual(
            store._step["panic_herd_choices"] / choices,
            0.5,
            delta=0.04,
        )

    def test_herd_choice_uses_greatest_frozen_edge_occupancy(self):
        center = SimpleNamespace(OSMID=1, nodeX=0.0, nodeY=0.0)
        left = SimpleNamespace(OSMID=2, nodeX=-1.0, nodeY=0.0)
        right = SimpleNamespace(OSMID=3, nodeX=1.0, nodeY=0.0)
        edge_left = Edge(center, left, 1, 101, 1.0, 1.0, 0, (0, 0))
        edge_right = Edge(center, right, 2, 102, 1.0, 1.0, 0, (0, 0))
        store = PedDS(1)
        store.mapDS = SimpleNamespace(incidentEdges=lambda _node: (edge_left, edge_right))
        store.configure_panic(rate=1.0, herd_probability=1.0, random_seed=11)
        pedestrian = self._pedestrian(center)
        pedestrian.panicked = True
        chosen = store._choose_panic_edge(
            pedestrian,
            {store._edge_load_key(edge_left): 2, store._edge_load_key(edge_right): 9},
        )
        self.assertIs(chosen, edge_right)
        self.assertEqual(store._step["panic_herd_choices"], 1)

    def test_nonzero_panic_rejects_cohort_approximation(self):
        with self.assertRaisesRegex(ValueError, "individual pedestrians"):
            PedDS(10, maximum_group_size=5).configure_panic(rate=0.1)


class IntersectionConsolidationTests(unittest.TestCase):
    def test_rebuilt_topology_becomes_active_and_zero_length_connectors_are_removed(self):
        original = nx.MultiDiGraph()
        original.graph["crs"] = "EPSG:4326"
        original.add_node(1, x=0.0, y=0.0)
        original.add_node(2, x=0.00001, y=0.0)
        original.add_edge(1, 2, key=0, length=1.0)
        rebuilt = nx.MultiDiGraph()
        rebuilt.graph["crs"] = "EPSG:4326"
        rebuilt.add_node(10, x=0.0, y=0.0)
        rebuilt.add_node(20, x=0.001, y=0.0)
        rebuilt.add_node(30, x=0.002, y=0.0)
        rebuilt.add_edge(10, 20, key=0, length=100.0)
        rebuilt.add_edge(20, 20, key=0, length=0.0)

        processor = OSMProcessor("synthetic")
        processor.locationDrive = original
        with (
            mock.patch.object(
                osm_module.OSM.projection,
                "project_graph",
                side_effect=[original, rebuilt],
            ),
            mock.patch.object(
                osm_module.OSM,
                "consolidate_intersections",
                return_value=rebuilt,
                create=True,
            ),
            mock.patch.object(
                osm_module.OSM.stats,
                "count_streets_per_node",
                return_value={10: 1, 20: 1},
            ),
            mock.patch.object(
                osm_module.OSM.stats,
                "basic_stats",
                return_value={"n": 2, "m": 1},
            ),
        ):
            result = processor.consolidateIntersections(5.0)
        self.assertIs(processor.locationDrive, rebuilt)
        self.assertEqual(rebuilt.number_of_nodes(), 2)
        self.assertEqual(rebuilt.number_of_edges(), 1)
        self.assertEqual(result["zero_length_edges_removed"], 1)
        self.assertTrue(result["enabled"])

    def test_consolidated_topology_cache_is_reused_only_for_matching_raw_graph(self):
        original = nx.MultiDiGraph()
        original.graph["crs"] = "EPSG:4326"
        original.add_node(1, x=0.0, y=0.0)
        original.add_node(2, x=0.001, y=0.0)
        original.add_edge(1, 2, key=0, length=100.0)
        rebuilt = nx.MultiDiGraph()
        rebuilt.graph["crs"] = "EPSG:4326"
        rebuilt.add_node(10, x=0.0, y=0.0)
        rebuilt.add_node(20, x=0.001, y=0.0)
        rebuilt.add_edge(10, 20, key=0, length=100.0)

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.graphml"
            raw_path.write_bytes(b"stable raw graph")

            def save_graph(_graph, path):
                Path(path).write_bytes(b"consolidated graph")

            first = OSMProcessor("synthetic")
            first.locationDrive = original
            second = OSMProcessor("synthetic")
            second.locationDrive = original.copy()
            with (
                mock.patch.object(
                    OSMProcessor,
                    "_graph_cache_path",
                    return_value=str(raw_path),
                ),
                mock.patch.object(
                    osm_module.OSM.projection,
                    "project_graph",
                    side_effect=[original, rebuilt],
                ),
                mock.patch.object(
                    osm_module.OSM,
                    "consolidate_intersections",
                    return_value=rebuilt,
                    create=True,
                ) as consolidate,
                mock.patch.object(
                    osm_module.OSM.io,
                    "save_graphml",
                    side_effect=save_graph,
                ),
                mock.patch.object(
                    osm_module.OSM.io,
                    "load_graphml",
                    return_value=rebuilt,
                ) as load_graph,
                mock.patch.object(
                    osm_module.OSM.stats,
                    "count_streets_per_node",
                    return_value={10: 1, 20: 1},
                ),
                mock.patch.object(
                    osm_module.OSM.stats,
                    "basic_stats",
                    return_value={"n": 2, "m": 1},
                ),
            ):
                created = first.consolidateIntersections(5.0)
                reused = second.consolidateIntersections(5.0)

            self.assertFalse(created["cache_reused"])
            self.assertTrue(reused["cache_reused"])
            self.assertEqual(consolidate.call_count, 1)
            self.assertEqual(load_graph.call_count, 1)


class ExperimentInfrastructureTests(unittest.TestCase):
    def test_factorial_design_is_exact_and_counts_all_rl_replicates(self):
        design = load_design()
        plan = execution_plan(5, 8, design)
        self.assertEqual(plan["factor_cells"], 625)
        self.assertEqual(plan["episodes_per_scenario_replication"], 13)
        self.assertEqual(plan["total_episodes"], 81250)
        fixed_policy_plan = execution_plan(5, 1, design)
        self.assertEqual(fixed_policy_plan["episodes_per_scenario_replication"], 6)
        self.assertEqual(fixed_policy_plan["total_episodes"], 37500)

    def test_policy_cache_round_trip_is_content_addressed_and_checksummed(self):
        contract = {"city_profile": "abc", "curriculum": {"episodes": 120}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pt"
            destination = root / "restored.pt"
            source.write_bytes(b"trained-policy")
            cache = PolicyCache(root / "cache")
            stored = cache.store(
                contract,
                source,
                metadata={"training_episode_count": 600, "training_rows": []},
            )
            self.assertIsNone(
                cache.restore(
                    contract,
                    root / "wrong.pt",
                    required_metadata={"training_episode_count": 601},
                )
            )
            restored = cache.restore(
                contract,
                destination,
                required_metadata={"training_episode_count": 600},
            )
            self.assertEqual(stored["cache_key"], canonical_contract_key(contract))
            self.assertEqual(restored["checkpoint_sha256"], stored["checkpoint_sha256"])
            self.assertEqual(restored["metadata"]["training_episode_count"], 600)
            self.assertEqual(destination.read_bytes(), b"trained-policy")

    def test_latency_comparison_pairs_rl_with_same_scenario_heuristic(self):
        common = {
            "city_id": "malibu_ca",
            "population_level": 5000,
            "hazard_count": 1,
            "panic_level": 0.1,
            "scenario_replication": 1,
        }
        rows = [
            {**common, "deployment_strategy": "heuristic", "policy_replication": 0,
             "mean_end_to_end_deployment_latency_ms": 2.0},
            {**common, "deployment_strategy": "random", "policy_replication": 0,
             "mean_end_to_end_deployment_latency_ms": 4.0},
            {**common, "deployment_strategy": "rl", "policy_replication": 3,
             "mean_end_to_end_deployment_latency_ms": 5.0},
        ]
        pairs = _latency_pairs(rows)
        self.assertEqual([row["heuristic_strategy"] for row in pairs], ["heuristic", "random"])
        pair = pairs[0]
        self.assertEqual(pair["rl_minus_heuristic_latency_ms"], 3.0)
        self.assertEqual(pair["rl_to_heuristic_latency_ratio"], 2.5)

    def test_factorial_source_gate_requires_complete_converged_training(self):
        with self.assertRaisesRegex(ValueError, "not complete"):
            validate_source_training_manifest({"status": "running"})
        complete = {
            "status": "complete",
            "training_convergence": {"all_policies_converged": False},
        }
        with self.assertRaisesRegex(ValueError, "convergence"):
            validate_source_training_manifest(complete)
        validate_source_training_manifest(complete, allow_nonconverged=True)


if __name__ == "__main__":
    unittest.main()
