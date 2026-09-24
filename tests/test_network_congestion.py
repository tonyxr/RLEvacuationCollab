import unittest
from types import SimpleNamespace

from NetworkCongestion import PedestrianCongestionModel
from PedestrianDatabase import PedDS
from Route import Route


def _node(osmid):
    return SimpleNamespace(OSMID=osmid)


def _edge(start, end, edge_id, osmid=900, length=10.0):
    return SimpleNamespace(
        startNode=start,
        endNode=end,
        edgeID=edge_id,
        OSMID=osmid,
        edgeLen=length,
        edgeFlow=-1,
        congestionOccupancy=-1,
        congestionDensityPedPerM2=-1.0,
        congestionSpeedRatio=-1.0,
    )


def _ped(agent_id, edge, size=1, *, at_node=True):
    return SimpleNamespace(
        agentID=agent_id,
        group_size=size,
        terminated=False,
        atNode=at_node,
        currEdge=None if at_node else edge,
        routeFollowing=SimpleNamespace(edgeRemained=[edge]),
    )


class PedestrianCongestionTests(unittest.TestCase):
    def setUp(self):
        self.model = PedestrianCongestionModel(
            effective_width_m=3.0,
            jam_density_ped_per_m2=5.4,
            shape=1.913,
            minimum_speed_ratio=0.05,
        )

    def test_weidmann_speed_density_curve_is_bounded_and_monotone(self):
        densities = (0.0, 0.25, 1.0, 2.0, 4.0, 5.4, 7.0)
        ratios = [self.model.speed_ratio(value) for value in densities]
        self.assertEqual(ratios[0], 1.0)
        self.assertTrue(all(0.05 <= value <= 1.0 for value in ratios))
        self.assertTrue(all(a >= b for a, b in zip(ratios, ratios[1:])))
        self.assertEqual(ratios[-1], 0.05)

    def test_counterflow_shares_one_physical_link_density(self):
        node_a, node_b = _node(1), _node(2)
        forward = _edge(node_a, node_b, 10)
        reverse = _edge(node_b, node_a, 11)
        pedestrians = [
            _ped(1, forward, size=30),
            _ped(2, reverse, size=30),
        ]

        ratios, states, metrics = self.model.snapshot(
            pedestrians, (forward, reverse)
        )

        self.assertEqual(len(states), 1)
        self.assertAlmostEqual(states[0].density_ped_per_m2, 2.0)
        self.assertEqual(states[0].occupancy, 60)
        self.assertEqual(forward.edgeFlow, 30)
        self.assertEqual(reverse.edgeFlow, 30)
        self.assertEqual(forward.congestionOccupancy, 60)
        self.assertEqual(reverse.congestionOccupancy, 60)
        self.assertAlmostEqual(ratios[id(pedestrians[0])], states[0].speed_ratio)
        self.assertAlmostEqual(ratios[id(pedestrians[1])], states[0].speed_ratio)
        self.assertEqual(metrics["congested_population"], 60)

    def test_snapshot_is_invariant_to_pedestrian_iteration_order(self):
        node_a, node_b = _node(1), _node(2)
        edge = _edge(node_a, node_b, 10, length=20.0)
        pedestrians = [_ped(index, edge, size=index + 1) for index in range(5)]

        ratios_a, states_a, metrics_a = self.model.snapshot(pedestrians, (edge,))
        by_agent_a = {
            ped.agentID: ratios_a[id(ped)] for ped in pedestrians
        }
        ratios_b, states_b, metrics_b = self.model.snapshot(
            tuple(reversed(pedestrians)), (edge,)
        )
        by_agent_b = {
            ped.agentID: ratios_b[id(ped)] for ped in pedestrians
        }

        self.assertEqual(by_agent_a, by_agent_b)
        self.assertEqual(states_a, states_b)
        self.assertEqual(metrics_a, metrics_b)

    def test_disabled_model_preserves_free_flow_speed(self):
        disabled = PedestrianCongestionModel(enabled=False)
        self.assertEqual(disabled.speed_ratio(5.4), 1.0)

    def test_network_movement_uses_the_link_density_speed(self):
        start = SimpleNamespace(OSMID=1, nodeX=0.0, nodeY=0.0)
        end = SimpleNamespace(OSMID=2, nodeX=10.0, nodeY=0.0)
        edge = _edge(start, end, 1, length=10.0)
        route = Route(start, end, [], [edge])
        pedestrian = SimpleNamespace(
            agentID=1,
            group_size=30,
            terminated=False,
            atNode=True,
            currNode=start,
            currEdge=None,
            currCell=(0, 0),
            routeFollowing=route,
            lastX=0.0,
            lastY=0.0,
            currSpeed=4.0,
            affected=False,
            edge_dest_node=None,
        )
        store = PedDS(30)
        store.maxSpeed = 4.0
        store.pedAgentList = {1: pedestrian}
        store.mapDS = SimpleNamespace(edgeListByLocalID={1: edge})
        store.cellTracker = SimpleNamespace(locateCell=lambda x, y: (0, 0))
        store.congestionModel = self.model

        store.pedestrianNetworkInteraction()

        expected_ratio = self.model.speed_ratio(1.0)
        self.assertAlmostEqual(pedestrian.lastX, 4.0 * expected_ratio)
        self.assertAlmostEqual(pedestrian.currSpeed, 4.0 * expected_ratio)
        self.assertAlmostEqual(
            store.lastCongestionMetrics["mean_congestion_speed_ratio"],
            expected_ratio,
        )
        self.assertEqual(store.lastCongestionMetrics["congestion_substeps"], 6)
        self.assertEqual(
            self.model.contract()["synchronization"],
            "occupancy_frozen_at_each_internal_substep",
        )

    def test_invalid_physical_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            PedestrianCongestionModel(effective_width_m=0.0)
        with self.assertRaises(ValueError):
            PedestrianCongestionModel(jam_density_ped_per_m2=float("nan"))
        with self.assertRaises(ValueError):
            PedestrianCongestionModel(minimum_speed_ratio=1.0)
        with self.assertRaises(ValueError):
            PedestrianCongestionModel(integration_substep_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
