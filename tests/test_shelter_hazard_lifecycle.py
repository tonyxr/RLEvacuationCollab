"""Behavioral specification for shelters overrun by a hazard.

These tests deliberately exercise the lifecycle through public simulator
operations: pedestrians are routed and admitted normally, the tracked danger
level changes, and ``PedDS.resolveHazardShelterClosures()`` performs the state
transition.  The intended contract is:

* danger level 4 is the inclusive closure threshold used in these tests;
* status 2 means permanently damaged (a later fall in danger cannot reopen it);
* reservations to a newly damaged shelter are released and rational active
  cohorts are rerouted to an operational shelter;
* admitted cohorts are restored to the active population without losing or
  duplicating represented people, including a cohort that was partly admitted;
* shelter evacuation and safe-completion ledgers describe the currently safe
  stock, so re-evacuation creates a legitimate negative completion delta.

The tests target the single
``PedDS.resolveHazardShelterClosures()`` lifecycle transaction and do not
prescribe its internal occupant-registry representation.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from DecisionInterface import OutcomeSnapshot
from PedestrianDatabase import PedDS
from RewardProcessor import RewardProcessor
from Shelter import Shelter
from ShelterDatabase import ShelterDS


def _node(osmid, x, cell):
    return SimpleNamespace(
        OSMID=int(osmid),
        nodeX=float(x),
        nodeY=0.0,
        nodeCap=100,
        cell=tuple(cell),
    )


class _MapStub:
    def __init__(self, *nodes):
        self.nodeListByOSMID = {int(node.OSMID): node for node in nodes}
        self.routing_version = 0

    @staticmethod
    def networkDistanceToTarget(start, target):
        return abs(float(start.nodeX) - float(target.nodeX))

    @staticmethod
    def shortestPath(start, target):
        return SimpleNamespace(
            startNode=start,
            endNode=target,
            edgeRemained=[],
        )


class _CellTrackerStub:
    def __init__(self):
        self.cellXNum = 3
        self.cellYNum = 1
        self.dangerLevelByCell = np.zeros((3, 1), dtype=np.int64)

    @staticmethod
    def locateCell(x, _y):
        x = float(x)
        if x < 0.0:
            return (2, 0)
        return (0, 0) if x < 15.0 else (1, 0)

    def getCellState(self, cell):
        ci, cj = (int(value) for value in cell)
        return int(self.dangerLevelByCell[ci, cj])

    def set_danger(self, cell, level):
        ci, cj = (int(value) for value in cell)
        self.dangerLevelByCell[ci, cj] = int(level)


def _pedestrian(agent_id, start, target, population):
    return SimpleNamespace(
        agentID=int(agent_id),
        group_id=int(agent_id) + 1,
        group_size=int(population),
        origin_node_osmid=int(start.OSMID),
        terminated=False,
        panicked=False,
        panic_eligibility_evaluated=False,
        panic_onset_time=None,
        panic_decision_count=0,
        panic_next_edge=None,
        affected=False,
        casualty=False,
        evacuated=False,
        arrival=False,
        unfinished=False,
        currNode=start,
        currEdge=None,
        currCell=tuple(start.cell),
        atNode=True,
        edge_dest_node=None,
        edge_dest_mode=None,
        edge_remain=0.0,
        edge_vec=None,
        lastX=float(start.nodeX),
        lastY=float(start.nodeY),
        currSpeed=10.0,
        desired_speed=10.0,
        social_force_speed=10.0,
        social_self_force=0.0,
        social_impact_force=0.0,
        speed_danger_level=None,
        distance_travelled_m=0.0,
        routeFollowing=SimpleNamespace(
            startNode=start,
            endNode=target,
            edgeRemained=[],
        ),
    )


def _world(population, threatened_capacity, *, maximum_group_size=None):
    origin = _node(1, -10.0, (2, 0))
    threatened_node = _node(10, 10.0, (0, 0))
    alternate_node = _node(20, 20.0, (1, 0))
    threatened = Shelter(
        0, threatened_node, (0, 0), int(threatened_capacity), 0, 0
    )
    alternate = Shelter(1, alternate_node, (1, 0), 100, 0, 0)

    shelters = ShelterDS(2, 2)
    shelters.shelterList = {0: threatened, 1: alternate}
    shelters.shelterByOSMID = {
        threatened_node.OSMID: threatened,
        alternate_node.OSMID: alternate,
    }

    cells = _CellTrackerStub()
    store = PedDS(
        int(population),
        maximum_group_size=int(maximum_group_size or max(1, population)),
    )
    store.mapDS = _MapStub(origin, threatened_node, alternate_node)
    store.cellTracker = cells
    store.shelterDS = shelters
    store.loadShelterLookup(shelters.shelterByOSMID)
    store.maxSpeed = 10.0
    store.timeStepMinutes = 1.0
    store.shelter_damage_danger_threshold = 4
    store._next_agent_id = 1
    return store, shelters, cells, origin, threatened, alternate


def _resolve_closures(test_case, store):
    resolver = getattr(store, "resolveHazardShelterClosures", None)
    test_case.assertTrue(
        callable(resolver),
        "PedDS must expose resolveHazardShelterClosures() as the single "
        "hazard-to-shelter lifecycle transition",
    )
    return resolver()


def _commit_step(store):
    store.docuStatus()


def _active_population(store):
    return sum(
        int(pedestrian.group_size)
        for pedestrian in store.pedAgentList.values()
        if not bool(getattr(pedestrian, "terminated", False))
    )


class ShelterHazardLifecycleTests(unittest.TestCase):
    def test_site_installed_in_overrun_cell_consumes_candidate_but_is_unusable(self):
        cells = _CellTrackerStub()
        cells.set_danger((0, 0), 4)
        candidate = _node(33, 10.0, (0, 0))
        shelters = ShelterDS(1, 0)
        shelters.currentSimulationTime = 9
        shelters.shelterCanByCell = [[[candidate]], [[]], [[]]]
        shelters.shelterByCell = [[[]], [[]], [[]]]

        shelter_id = shelters.newShelter({"cell": (0, 0)}, cells)

        self.assertEqual(shelter_id, 0)
        self.assertEqual(shelters.remainingCandidateCount(), 0)
        installed = shelters.shelterList[shelter_id]
        self.assertEqual(installed.status, 2)
        self.assertEqual(installed.overrunTime, 9)
        self.assertEqual(shelters.availableCapacity(installed), 0)
        self.assertEqual(shelters.hazardOverrunCount, 1)

    def test_closure_occurs_at_the_inclusive_danger_threshold(self):
        store, _shelters, cells, _origin, threatened, alternate = _world(0, 10)

        cells.set_danger(threatened.cellLocated, 3)
        _resolve_closures(self, store)
        self.assertEqual(threatened.status, 0)

        cells.set_danger(threatened.cellLocated, 4)
        _resolve_closures(self, store)
        self.assertEqual(threatened.status, 2)
        self.assertEqual(threatened.shelterFlow, 0)
        self.assertEqual(alternate.status, 0)

    def test_closure_records_the_transition_boundary_not_the_prior_step(self):
        store, _shelters, cells, _origin, threatened, _alternate = _world(0, 10)
        cells.set_danger(threatened.cellLocated, 4)

        store.resolveHazardShelterClosures(simulation_time=7)

        self.assertEqual(threatened.overrunTime, 7)
        self.assertEqual(store.shelterDS.currentSimulationTime, 7)

    def test_damaged_status_is_permanent_when_danger_recedes(self):
        store, shelters, cells, _origin, threatened, _alternate = _world(0, 10)
        cells.set_danger(threatened.cellLocated, 5)
        _resolve_closures(self, store)
        self.assertEqual(threatened.status, 2)

        cells.set_danger(threatened.cellLocated, 0)
        _resolve_closures(self, store)
        shelters.shelterStatusUpdate()

        self.assertEqual(threatened.status, 2)
        self.assertEqual(shelters.availableCapacity(threatened), 0)

    def test_closure_releases_reservations_and_reroutes_active_cohorts(self):
        store, shelters, cells, origin, threatened, alternate = _world(2, 10)
        cohort = _pedestrian(0, origin, threatened.nodeMapped, 2)
        store.pedAgentList[cohort.agentID] = cohort
        self.assertEqual(shelters.reserveShelter(cohort, threatened), 2)

        cells.set_danger(threatened.cellLocated, 4)
        _resolve_closures(self, store)

        self.assertEqual(threatened.status, 2)
        self.assertEqual(shelters.reservedPopulation(threatened), 0)
        self.assertEqual(shelters.reservationFor(cohort), (alternate, 2))
        self.assertIs(cohort.routeFollowing.endNode, alternate.nodeMapped)
        shelters.assertCapacityInvariant()

    def test_full_grouped_cohort_is_reactivated_without_changing_mass(self):
        store, shelters, cells, origin, threatened, alternate = _world(4, 4)
        cohort = _pedestrian(0, origin, threatened.nodeMapped, 4)
        store.pedAgentList[cohort.agentID] = cohort
        self.assertEqual(shelters.reserveShelter(cohort, threatened), 4)

        self.assertFalse(store.arrive_node(cohort, threatened.nodeMapped))
        _commit_step(store)
        self.assertEqual(_active_population(store), 0)
        self.assertEqual(store.result["evacuated"], 4)
        self.assertEqual(store.safeCompletionCount, 4)
        self.assertEqual(threatened.shelterFlow, 4)

        store.startDocument()
        cells.set_danger(threatened.cellLocated, 4)
        _resolve_closures(self, store)
        _commit_step(store)

        active = list(store.pedAgentList.values())
        self.assertEqual([pedestrian.group_size for pedestrian in active], [4])
        self.assertEqual(_active_population(store), 4)
        self.assertEqual(store.result["evacuated"], 0)
        self.assertEqual(store.safeCompletionCount, 0)
        self.assertEqual(store.safeCompletionTimeSum, 0.0)
        self.assertEqual(dict(store.safeCompletionTimeHistogram), {})
        self.assertEqual(threatened.shelterFlow, 0)
        self.assertEqual(threatened.status, 2)
        self.assertEqual(shelters.reservationFor(active[0]), (alternate, 4))
        self.assertIs(active[0].routeFollowing.endNode, alternate.nodeMapped)
        shelters.assertCapacityInvariant()

    def test_partial_group_admission_reactivates_only_the_admitted_fragment(self):
        store, shelters, cells, origin, threatened, alternate = _world(8, 5)
        cohort = _pedestrian(0, origin, threatened.nodeMapped, 8)
        store.pedAgentList[cohort.agentID] = cohort

        # Five people enter; the remaining three stay active and are routed to
        # the alternate shelter by the normal arrival path.
        self.assertTrue(store.arrive_node(cohort, threatened.nodeMapped))
        _commit_step(store)
        self.assertEqual(cohort.group_size, 3)
        self.assertEqual(_active_population(store), 3)
        self.assertEqual(store.result["evacuated"], 5)
        self.assertEqual(store.safeCompletionCount, 5)
        self.assertEqual(threatened.shelterFlow, 5)
        self.assertEqual(shelters.reservationFor(cohort), (alternate, 3))

        store.startDocument()
        cells.set_danger(threatened.cellLocated, 5)
        _resolve_closures(self, store)
        _commit_step(store)

        active = list(store.pedAgentList.values())
        self.assertEqual(
            sorted(int(pedestrian.group_size) for pedestrian in active),
            [3, 5],
        )
        self.assertEqual(_active_population(store), 8)
        self.assertEqual(store.result["evacuated"], 0)
        self.assertEqual(store.safeCompletionCount, 0)
        self.assertEqual(store.safeCompletionTimeSum, 0.0)
        self.assertEqual(dict(store.safeCompletionTimeHistogram), {})
        self.assertEqual(threatened.shelterFlow, 0)
        self.assertEqual(threatened.status, 2)
        for active_cohort in active:
            self.assertEqual(
                shelters.reservationFor(active_cohort),
                (alternate, int(active_cohort.group_size)),
            )
            self.assertIs(
                active_cohort.routeFollowing.endNode,
                alternate.nodeMapped,
            )
        shelters.assertCapacityInvariant()

    def test_no_alternative_after_closure_keeps_residents_active_and_unassigned(self):
        store, shelters, cells, origin, threatened, alternate = _world(4, 4)
        alternate.status = 2
        cohort = _pedestrian(0, origin, threatened.nodeMapped, 4)
        store.pedAgentList[cohort.agentID] = cohort
        self.assertEqual(shelters.reserveShelter(cohort, threatened), 4)
        self.assertFalse(store.arrive_node(cohort, threatened.nodeMapped))
        _commit_step(store)

        store.startDocument()
        cells.set_danger(threatened.cellLocated, 4)
        _resolve_closures(self, store)
        _commit_step(store)

        active = list(store.pedAgentList.values())
        self.assertEqual(_active_population(store), 4)
        self.assertEqual(len(active), 1)
        self.assertIsNone(active[0].routeFollowing)
        self.assertEqual(shelters.reservationFor(active[0]), (None, 0))
        self.assertEqual(store.result["evacuated"], 0)

    def test_readmission_records_only_the_later_durable_completion(self):
        store, shelters, cells, origin, threatened, alternate = _world(4, 4)
        cohort = _pedestrian(0, origin, threatened.nodeMapped, 4)
        store.pedAgentList[cohort.agentID] = cohort
        shelters.reserveShelter(cohort, threatened)
        self.assertFalse(store.arrive_node(cohort, threatened.nodeMapped))
        _commit_step(store)

        store.startDocument()
        cells.set_danger(threatened.cellLocated, 4)
        _resolve_closures(self, store)
        _commit_step(store)
        reactivated = next(iter(store.pedAgentList.values()))
        first_completion_time = 1.0
        self.assertNotIn(first_completion_time, store.safeCompletionTimeHistogram)

        self.assertFalse(store.arrive_node(reactivated, alternate.nodeMapped))
        _commit_step(store)

        self.assertEqual(store.result["evacuated"], 4)
        self.assertEqual(store.safeCompletionCount, 4)
        self.assertEqual(sum(store.safeCompletionTimeHistogram.values()), 4)
        self.assertEqual(store.cumulativeShelterAdmissions, 8)
        self.assertEqual(store.cumulativeForcedReevacuated, 4)
        self.assertGreater(store.mean_safe_completion_time(), first_completion_time)
        shelters.assertCapacityInvariant()
        store.assertShelterOccupancyInvariant()

    def test_reward_allows_safe_stock_to_fall_during_reevacuation(self):
        # Safe completions are a stock once shelters can fail.  A closure can
        # therefore move people from safe back to active without invalidating
        # an otherwise well-defined matched branch reward.
        start = OutcomeSnapshot(5, 0, 5, 0, 3, 3.0)
        action_after = OutcomeSnapshot(2, 0, 2, 0, 6, 6.0)

        reward = RewardProcessor().evaluate_counterfactual(
            start=start,
            action_after=action_after,
            plain_time_wait=60.0,
            plain_time_action=60.0,
            hazard_exposure_wait=20.0,
            hazard_exposure_action=20.0,
            initial_population=8,
            horizon_minutes=10.0,
            signed=True,
        )

        self.assertEqual(reward.new_safe_completions, -3)
        self.assertEqual(reward.new_casualties, 0)
        self.assertEqual(reward.total, 0.0)


class PopulationMatchedAssignmentInvariantTests(unittest.TestCase):
    @staticmethod
    def _shelters(*entries):
        store = ShelterDS(len(entries), len(entries))
        store.shelterList = {
            index: SimpleNamespace(
                shelterID=index,
                shelterCap=int(capacity),
                status=int(status),
            )
            for index, (capacity, status) in enumerate(entries)
        }
        return store

    def test_unassigned_population_fails_with_intact_network_and_capacity(self):
        shelters = self._shelters((6, Shelter.OPEN), (4, Shelter.FULL))

        with self.assertRaisesRegex(
            RuntimeError,
            "sufficient operational network and capacity",
        ):
            shelters.assertPopulationMatchedAssignment(2, 10)

    def test_hazard_overrun_permits_a_physical_assignment_shortfall(self):
        shelters = self._shelters(
            (10, Shelter.OPEN),
            (4, Shelter.OVERRUN),
        )
        shelters.hazardOverrunCount = 1

        shelters.assertPopulationMatchedAssignment(4, 10)

    def test_road_disruption_permits_an_unreachable_population_shortfall(self):
        shelters = self._shelters((10, Shelter.OPEN))

        shelters.assertPopulationMatchedAssignment(
            3,
            10,
            network_disruption_active=True,
        )

    def test_missing_final_measurement_never_passes_as_a_physical_shortfall(self):
        shelters = self._shelters((10, Shelter.OPEN))
        shelters.hazardOverrunCount = 1

        with self.assertRaisesRegex(RuntimeError, "did not record"):
            shelters.assertPopulationMatchedAssignment(None, 10)


if __name__ == "__main__":
    unittest.main()
