"""v64 reversal-consistent shelter-stay ledger tests."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import CounterfactualBranch as CB
from tests.test_shelter_hazard_lifecycle import _pedestrian, _world


def _flattenable_world(population=10):
    store, shelters, cells, origin, threatened, alternate = _world(
        population, population
    )
    cells.dangerLevelByCell = np.zeros((3, 1), dtype=np.float64)
    cells.flatten_cell = lambda cell: int(cell[0]) * cells.cellYNum + int(cell[1])
    return store, shelters, cells, origin, threatened, alternate


def _admit(store, shelters, origin, shelter, population=10, agent_id=0):
    cohort = _pedestrian(agent_id, origin, shelter.nodeMapped, population)
    store.pedAgentList[cohort.agentID] = cohort
    shelters.reserveShelter(cohort, shelter)
    if store.arrive_node(cohort, shelter.nodeMapped):
        raise AssertionError("fixture admission unexpectedly continued movement")
    return cohort


class V64SettledStayLedgerTests(unittest.TestCase):
    def test_reversal_settles_exact_plain_and_exposure_stay(self):
        store, shelters, cells, origin, threatened, _alternate = _flattenable_world()
        cells.dangerLevelByCell[0, 0] = 0.4
        cohort = _admit(store, shelters, origin, threatened)
        steps = 3
        for _ in range(steps):
            store.docuStatus()
            store.startDocument()

        cells.dangerLevelByCell[0, 0] = 4.0
        store.resolveHazardShelterClosures()
        self.assertAlmostEqual(store.cumulativeSettledPlainPersonMinutes, 30.0)
        self.assertAlmostEqual(store.cumulativeSettledExposurePersonMinutes, 12.0)

        core = SimpleNamespace(
            cellTracker=cells,
            cellX=3,
            cellY=1,
            pedDS=store,
            timeStepMinutes=1.0,
            stopTime=61,
            casualtyPriorityKappa=2.0,
            casualtyPriceRule="legacy",
        )
        plain, exposure, _ = CB.controllable_time_mass(core, {cohort.agentID})
        store.stepSettledPlainPersonMinutes = 0.0
        store.stepSettledExposurePersonMinutes = 0.0
        base_plain, base_exposure, _ = CB.controllable_time_mass(
            core, {cohort.agentID}
        )
        self.assertAlmostEqual(plain - base_plain, 30.0)
        self.assertAlmostEqual(exposure - base_exposure, 12.0)

    def test_operational_shelter_never_settles_shadow_time(self):
        store, shelters, cells, origin, threatened, _alternate = _flattenable_world()
        cells.dangerLevelByCell[0, 0] = 0.4
        _admit(store, shelters, origin, threatened)
        for _ in range(4):
            store.docuStatus()
            store.startDocument()
        self.assertEqual(store.cumulativeSettledPlainPersonMinutes, 0.0)
        self.assertEqual(store.cumulativeSettledExposurePersonMinutes, 0.0)

    def test_readmission_starts_a_fresh_shadow_ledger(self):
        store, shelters, cells, origin, threatened, alternate = _flattenable_world()
        cells.dangerLevelByCell[0, 0] = 0.2
        cohort = _admit(store, shelters, origin, threatened)
        store.docuStatus()
        store.startDocument()
        cells.dangerLevelByCell[0, 0] = 4.0
        store.resolveHazardShelterClosures()
        first = store.cumulativeSettledPlainPersonMinutes

        cells.dangerLevelByCell[1, 0] = 0.6
        shelters.releaseReservation(cohort)
        shelters.reserveShelter(cohort, alternate)
        store.arrive_node(cohort, alternate.nodeMapped)
        store.docuStatus()
        store.startDocument()
        cells.dangerLevelByCell[1, 0] = 4.0
        store.resolveHazardShelterClosures()
        self.assertAlmostEqual(first, 10.0)
        self.assertAlmostEqual(store.cumulativeSettledPlainPersonMinutes, 20.0)
        self.assertAlmostEqual(store.cumulativeSettledExposurePersonMinutes, 8.0)

    def test_digest_covers_step_and_resident_shadow_state(self):
        core = __import__("nmcc_testbed").build(population=20, group_size=10)
        first = CB.live_state_digest(core)
        core.pedDS.stepSettledPlainPersonMinutes = 1.0
        self.assertNotEqual(CB.live_state_digest(core), first)

        core.pedDS.stepSettledPlainPersonMinutes = 0.0
        shelter = next(iter(core.shelterDS.shelterList.values()))
        pedestrian = next(iter(core.pedDS.pedAgentList.values()))
        core.pedDS._register_shelter_occupant(
            pedestrian, shelter, int(pedestrian.group_size)
        )
        before_shadow = CB.live_state_digest(core)
        record = next(iter(core.pedDS.shelteredOccupants[shelter.shelterID].values()))
        record["shadow_plain_person_minutes"] = 5.0
        self.assertNotEqual(CB.live_state_digest(core), before_shadow)


if __name__ == "__main__":
    unittest.main()
