import unittest
from types import SimpleNamespace

from PedestrianDatabase import PedDS
from Shelter import Shelter
from ShelterDatabase import ShelterDS


class _MapStub:
    def __init__(self):
        self.node = SimpleNamespace(OSMID=1, nodeX=0.0, nodeY=0.0)

    def assignGenerationNode(self):
        return self.node


class _CellStub:
    def locateCell(self, _x, _y):
        return (0, 0)

    def getCellState(self, _cell):
        return 5


class PopulationCohortTests(unittest.TestCase):
    def test_initialization_preserves_exact_represented_population(self):
        store = PedDS(11, maximum_group_size=4)
        store.initPedestrianAgent(_MapStub(), _CellStub(), 64.0)
        self.assertEqual(len(store.pedAgentList), 3)
        self.assertEqual(
            [agent.group_size for agent in store.pedAgentList.values()],
            [4, 4, 3],
        )
        self.assertEqual(store.remaining_active_count(), 11)

    def test_partial_cohort_casualties_preserve_survivors(self):
        store = PedDS(10, maximum_group_size=10)
        pedestrian = SimpleNamespace(
            agentID=0,
            group_size=10,
            terminated=False,
            currCell=(0, 0),
            currSpeed=64.0,
            desired_speed=64.0,
            affected=False,
        )
        store.pedAgentList[0] = pedestrian
        store.cellTracker = _CellStub()
        hazard = SimpleNamespace(
            active=True,
            impactedCells=[(0, 0)],
            speedReduct=0.0,
            casualtyRate=0.5,
        )
        store.hazardDS = SimpleNamespace(hazardList={0: hazard})
        store.maxSpeed = 64.0
        store._hazard_casualty_count = lambda *_args: 3

        store.pedestrianHazardInteraction()

        self.assertEqual(store._step["casualty"], 3)
        self.assertEqual(pedestrian.group_size, 7)
        self.assertFalse(pedestrian.terminated)
        self.assertEqual(store.remaining_active_count(), 7)
        self.assertEqual(store._step["affected"], 7)

    def test_partial_shelter_admission_splits_cohort_accounting(self):
        node = SimpleNamespace(OSMID=7, nodeX=0.0, nodeY=0.0)
        shelter = Shelter(0, node, (0, 0), 5, 0, 0)
        shelters = ShelterDS(1, 1)
        shelters.shelterList[0] = shelter
        shelters.shelterByOSMID[node.OSMID] = shelter
        pedestrian = SimpleNamespace(
            agentID=0,
            group_size=8,
            terminated=False,
            currNode=node,
            currEdge=None,
            currCell=(0, 0),
            lastX=0.0,
            lastY=0.0,
            atNode=True,
            routeFollowing=SimpleNamespace(endNode=node),
        )
        store = PedDS(8, maximum_group_size=8)
        store.cellTracker = _CellStub()
        store.shelterDS = shelters
        store.shelter_osmid_map = shelters.shelterByOSMID
        store.pedAgentList[0] = pedestrian

        self.assertFalse(store.arrive_node(pedestrian, node))
        self.assertEqual(shelter.shelterFlow, 5)
        self.assertEqual(shelter.status, 1)
        self.assertEqual(store._step["evacuated"], 5)
        self.assertEqual(pedestrian.group_size, 3)
        self.assertFalse(pedestrian.terminated)
        self.assertEqual(store.remaining_active_count(), 3)

    def test_individual_hazard_draw_retains_original_counter_contract(self):
        store = PedDS(1)
        store.currTime = 4
        store.set_hazard_random_seed(123)
        expected = int(store._hazard_uniform(9) < 0.25)
        self.assertEqual(store._hazard_casualty_count(9, 1, 0.25), expected)


if __name__ == "__main__":
    unittest.main()
