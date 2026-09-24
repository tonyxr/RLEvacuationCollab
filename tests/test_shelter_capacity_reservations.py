import unittest
from types import SimpleNamespace

import CounterfactualBranch as CB
import nmcc_testbed
from DecisionInterface import RegionalObservationBuilder
from PedestrianDatabase import PedDS
from Shelter import Shelter
from ShelterDatabase import ShelterDS


def node(osmid, x):
    return SimpleNamespace(OSMID=osmid, nodeX=float(x), nodeY=0.0)


def pedestrian(agent_id, start, population=1, danger_cell=(0, 0)):
    return SimpleNamespace(
        agentID=int(agent_id),
        group_size=int(population),
        terminated=False,
        panicked=False,
        affected=False,
        evacuated=False,
        arrival=False,
        currNode=start,
        currEdge=None,
        currCell=danger_cell,
        atNode=True,
        edge_dest_node=None,
        edge_remain=0.0,
        lastX=float(start.nodeX),
        lastY=float(start.nodeY),
        currSpeed=10.0,
        distance_travelled_m=0.0,
        routeFollowing=None,
    )


class MapStub:
    @staticmethod
    def networkDistanceToTarget(start, target):
        return abs(float(start.nodeX) - float(target.nodeX))

    @staticmethod
    def shortestPath(start, target):
        return SimpleNamespace(startNode=start, endNode=target, edgeRemained=[])


class CellStub:
    def __init__(self, danger=0):
        self.danger = int(danger)

    @staticmethod
    def locateCell(_x, _y):
        return (0, 0)

    def getCellState(self, _cell):
        return self.danger


def stores(*shelters):
    shelter_store = ShelterDS(len(shelters), len(shelters))
    shelter_store.shelterList = {
        int(shelter.shelterID): shelter for shelter in shelters
    }
    shelter_store.shelterByOSMID = {
        shelter.nodeMapped.OSMID: shelter for shelter in shelters
    }
    pedestrian_store = PedDS(0)
    pedestrian_store.mapDS = MapStub()
    pedestrian_store.cellTracker = CellStub()
    pedestrian_store.shelterDS = shelter_store
    pedestrian_store.loadShelterLookup(shelter_store.shelterByOSMID)
    return pedestrian_store, shelter_store


class ReservationLedgerTests(unittest.TestCase):
    def test_failed_reassignment_is_transactional_and_never_overbooks(self):
        first = Shelter(0, node(10, 0), (0, 0), 1, 0, 0)
        second = Shelter(1, node(11, 10), (0, 0), 1, 0, 0)
        _, shelter_store = stores(first, second)
        one = pedestrian(1, node(1, 2))
        two = pedestrian(2, node(2, 3))

        self.assertEqual(shelter_store.reserveShelter(one, first), 1)
        self.assertEqual(shelter_store.reserveShelter(two, second), 1)
        self.assertEqual(shelter_store.reserveShelter(two, first), 0)
        self.assertEqual(shelter_store.reservationFor(two), (second, 1))
        self.assertEqual(shelter_store.availableCapacity(first), 0)
        self.assertEqual(shelter_store.availableCapacity(second), 0)
        shelter_store.assertCapacityInvariant()

    def test_initial_routing_stops_exactly_at_committed_capacity(self):
        shelter = Shelter(0, node(10, 100), (0, 0), 2, 0, 0)
        pedestrian_store, shelter_store = stores(shelter)
        agents = [pedestrian(index, node(index, index)) for index in range(3)]
        pedestrian_store.pedAgentList = {
            agent.agentID: agent for agent in agents
        }

        self.assertEqual(pedestrian_store.route_active_to_nearest_shelter(), 2)
        self.assertEqual(shelter_store.reservedPopulation(shelter), 2)
        self.assertEqual(sum(agent.routeFollowing is not None for agent in agents), 2)
        shelter_store.assertCapacityInvariant()

    def test_walk_in_cannot_take_a_reserved_slot(self):
        target_node = node(10, 100)
        shelter = Shelter(0, target_node, (0, 0), 1, 0, 0)
        pedestrian_store, shelter_store = stores(shelter)
        assigned = pedestrian(1, node(1, 0))
        assigned.routeFollowing = MapStub.shortestPath(assigned.currNode, target_node)
        self.assertEqual(shelter_store.reserveShelter(assigned, shelter), 1)
        walk_in = pedestrian(2, target_node)
        walk_in.panicked = True
        pedestrian_store.pedAgentList = {1: assigned, 2: walk_in}

        self.assertTrue(pedestrian_store.arrive_node(walk_in, target_node))
        self.assertEqual(shelter.shelterFlow, 0)
        self.assertFalse(walk_in.terminated)
        self.assertFalse(pedestrian_store.arrive_node(assigned, target_node))
        self.assertEqual(shelter.shelterFlow, 1)
        self.assertTrue(assigned.terminated)
        self.assertEqual(shelter_store.reservedPopulation(shelter), 0)

    def test_cohort_commitment_clamps_after_casualties(self):
        target_node = node(10, 100)
        shelter = Shelter(0, target_node, (0, 0), 8, 0, 0)
        pedestrian_store, shelter_store = stores(shelter)
        cohort = pedestrian(1, node(1, 0), population=8)
        cohort.routeFollowing = MapStub.shortestPath(cohort.currNode, target_node)
        pedestrian_store.pedAgentList = {1: cohort}

        too_small = Shelter(1, node(11, 90), (0, 0), 7, 0, 0)
        shelter_store.shelterList[1] = too_small
        shelter_store.shelterByOSMID[11] = too_small
        self.assertEqual(shelter_store.reserveShelter(cohort, too_small), 0)
        self.assertEqual(shelter_store.reserveShelter(cohort, shelter), 8)
        cohort.group_size = 3
        self.assertEqual(
            shelter_store.reconcileReservations([cohort]),
            5,
        )
        self.assertEqual(shelter_store.reservationFor(cohort), (shelter, 3))
        self.assertEqual(shelter_store.availableCapacity(shelter), 5)
        self.assertFalse(pedestrian_store.arrive_node(cohort, target_node))
        self.assertEqual(shelter.shelterFlow, 3)
        self.assertEqual(shelter_store.reservedPopulation(shelter), 0)

    def test_panic_and_termination_release_capacity(self):
        target_node = node(10, 100)
        shelter = Shelter(0, target_node, (0, 0), 2, 0, 0)
        pedestrian_store, shelter_store = stores(shelter)
        first = pedestrian(1, node(1, 0))
        second = pedestrian(2, node(2, 0))
        for agent in (first, second):
            agent.routeFollowing = MapStub.shortestPath(agent.currNode, target_node)
            shelter_store.reserveShelter(agent, shelter)
        pedestrian_store.pedAgentList = {1: first, 2: second}

        first.panicked = True
        first.routeFollowing = None
        shelter_store.reconcileReservations(pedestrian_store.pedAgentList.values())
        self.assertEqual(shelter_store.availableCapacity(shelter), 1)
        pedestrian_store.terminatePedestrianAgent(second, "Unfinished")
        self.assertEqual(shelter_store.availableCapacity(shelter), 2)

    def test_damaged_target_releases_and_reassigns_in_one_reconciliation(self):
        damaged = Shelter(0, node(10, 100), (0, 0), 1, 0, 0)
        alternate = Shelter(1, node(11, 80), (0, 0), 1, 0, 0)
        pedestrian_store, shelter_store = stores(damaged, alternate)
        agent = pedestrian(1, node(1, 0))
        agent.routeFollowing = MapStub.shortestPath(agent.currNode, damaged.nodeMapped)
        pedestrian_store.pedAgentList = {1: agent}
        shelter_store.reserveShelter(agent, damaged)

        damaged.status = 2
        shelter_store.reconcileReservations([agent])
        self.assertTrue(pedestrian_store.reroute_if_target_shelter_unavailable(agent))
        self.assertIs(agent.routeFollowing.endNode, alternate.nodeMapped)
        self.assertEqual(shelter_store.reservationFor(agent), (alternate, 1))

    def test_new_shelter_capacity_goes_to_largest_risk_time_reduction(self):
        old = Shelter(0, node(10, 100), (0, 0), 3, 0, 0)
        new = Shelter(1, node(11, 0), (0, 0), 1, 0, 0)
        pedestrian_store, shelter_store = stores(old, new)
        pedestrian_store.cellTracker = CellStub(danger=5)
        agents = [
            pedestrian(1, node(1, 10)),
            pedestrian(2, node(2, 40)),
            pedestrian(3, node(3, 80)),
        ]
        for agent in agents:
            agent.routeFollowing = MapStub.shortestPath(agent.currNode, old.nodeMapped)
            shelter_store.reserveShelter(agent, old)
        pedestrian_store.pedAgentList = {
            agent.agentID: agent for agent in agents
        }

        self.assertEqual(pedestrian_store.reroute_to_new_shelter_if_closer(new), 1)
        self.assertEqual(shelter_store.reservationFor(agents[0]), (new, 1))
        self.assertEqual(shelter_store.reservationFor(agents[1]), (old, 1))
        self.assertEqual(shelter_store.reservationFor(agents[2]), (old, 1))
        shelter_store.assertCapacityInvariant()


class ReservationIntegrationTests(unittest.TestCase):
    def test_observation_exposes_only_uncommitted_capacity(self):
        core = nmcc_testbed.build(
            grid=6,
            cell_x=3,
            cell_y=3,
            population=8,
            shelter_capacity_token=8,
            candidate_count=8,
            panic_rate=0.0,
        )
        builder = RegionalObservationBuilder(
            core,
            initial_population=core.initial_population,
            horizon=core.stopTime,
            maximum_deployments=2,
        )
        remaining, utilization, _, _, _ = builder._regional_capacity()
        expected_remaining = sum(
            core.shelterDS.availableCapacity(shelter)
            for shelter in core.shelterDS.shelterList.values()
        )
        by_cell = {}
        for shelter in core.shelterDS.shelterList.values():
            capacity, committed = by_cell.get(shelter.cellLocated, (0, 0))
            by_cell[shelter.cellLocated] = (
                capacity + shelter.shelterCap,
                committed
                + shelter.shelterFlow
                + core.shelterDS.reservedPopulation(shelter),
            )
        expected_utilization_sum = sum(
            committed / capacity for capacity, committed in by_cell.values()
        )
        self.assertEqual(int(remaining.sum()), expected_remaining)
        self.assertAlmostEqual(
            float(utilization.sum()),
            expected_utilization_sum,
            places=6,
        )

    def test_live_dynamics_preserve_capacity_invariant(self):
        core = nmcc_testbed.build(
            grid=8,
            cell_x=4,
            cell_y=4,
            population=40,
            shelter_capacity_token=8,
            candidate_count=16,
            panic_rate=0.25,
            stop_time=20,
        )
        for step in range(10):
            if step in (1, 4):
                cell = core.feasible_cells()[0]
                shelter_id = core.shelterDS.newShelter(
                    {"cell": divmod(cell, core.cellY)}, core.cellTracker
                )
                shelter = core.shelterDS.shelterList[shelter_id]
                core.pedDS.reroute_to_new_shelter_if_closer(shelter)
            CB.advance_one_timestep(core)
            core.shelterDS.assertCapacityInvariant()
            for shelter in core.shelterDS.shelterList.values():
                committed = (
                    shelter.shelterFlow
                    + core.shelterDS.reservedPopulation(shelter)
                )
                self.assertLessEqual(committed, shelter.shelterCap)

    def test_counterfactual_digest_includes_reservations(self):
        core = nmcc_testbed.build(
            grid=6,
            cell_x=3,
            cell_y=3,
            population=8,
            shelter_capacity_token=8,
            candidate_count=8,
            panic_rate=0.0,
        )
        before = CB.live_state_digest(core)
        agent = next(iter(core.pedDS.pedAgentList.values()))
        core.shelterDS.releaseReservation(agent)
        self.assertNotEqual(CB.live_state_digest(core), before)


if __name__ == "__main__":
    unittest.main()
