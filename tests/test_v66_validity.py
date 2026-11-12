"""v66 validity options: unit tests (standard library unittest; torch optional).

Run from the v66 clone root:
    .venv/bin/python -m unittest tests/test_v66_validity.py -v

Every test is local to one function.  End-to-end equality with v65.1 is
checked by panel v66_identity (Stage 0 of the run manual), not here.
"""
from __future__ import annotations

import math
import statistics
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ShelterDatabase as shelter_module  # noqa: E402
import PedestrianDatabase as pedestrian_module  # noqa: E402
import DecisionInterface as decision_module  # noqa: E402

try:
    import torch  # noqa: F401
    import GNN as gnn_module
except Exception:  # pragma: no cover - torch missing
    gnn_module = None


def _grid_database(mode: str, nx: int = 16, ny: int = 16, empty=()):
    """A ShelterDatabase with one fake candidate per occupied cell."""
    db = shelter_module.ShelterDS.__new__(shelter_module.ShelterDS)
    db.candidateSamplingMode = mode
    grid = []
    for i in range(nx):
        column = []
        for j in range(ny):
            if (i, j) in empty:
                column.append([])
            else:
                column.append([SimpleNamespace(nodeX=100.0 * i + 50.0, nodeY=100.0 * j + 50.0)])
        grid.append(column)
    db.shelterCanByCell = grid
    return db


class CandidateOrder(unittest.TestCase):
    def test_legacy_is_row_major(self):
        db = _grid_database("legacy_row_major")
        order = db._candidate_cell_order(16, 16)
        self.assertEqual(order, [(i, j) for i in range(16) for j in range(16)])
        # The v65.1 artifact: the first 20 cells lie in x strips 0 and 1.
        self.assertEqual(sorted({i for i, _ in order[:20]}), [0, 1])

    def test_maximin_spreads_the_first_twenty(self):
        db = _grid_database("spatial_maximin")
        order = db._candidate_cell_order(16, 16)
        strips = {i for i, _ in order[:20]}
        rows = {j for _, j in order[:20]}
        self.assertGreaterEqual(len(strips), 10)
        self.assertGreaterEqual(len(rows), 10)
        self.assertEqual(min(strips), 0)
        self.assertEqual(max(strips), 15)

    def test_maximin_is_a_permutation_and_deterministic(self):
        empty = {(0, 0), (3, 7), (15, 15)}
        db = _grid_database("spatial_maximin", empty=empty)
        first = db._candidate_cell_order(16, 16)
        second = db._candidate_cell_order(16, 16)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), [(i, j) for i in range(16) for j in range(16)])
        # Empty cells come last, after every occupied cell.
        self.assertEqual(set(first[-len(empty):]), empty)

    def test_unknown_mode_raises(self):
        db = _grid_database("west_first")
        with self.assertRaises(ValueError):
            db._candidate_cell_order(16, 16)


def _pedestrian_database(mean, seed=12345, shape=1.5, maximum=90.0):
    db = pedestrian_module.PedDS.__new__(pedestrian_module.PedDS)
    db.currTime = 0
    db.configure_departure(mean_minutes=mean, shape=shape, max_minutes=maximum, random_seed=seed)
    return db


class DepartureDelay(unittest.TestCase):
    def test_disabled_means_immediate(self):
        db = _pedestrian_database(0.0)
        self.assertEqual({db._sample_departure_time(k) for k in range(500)}, {0})
        self.assertTrue(db.has_departed(SimpleNamespace(departure_time=50)))

    def test_mean_and_bounds(self):
        db = _pedestrian_database(20.0)
        draws = [db._sample_departure_time(k) for k in range(20000)]
        self.assertAlmostEqual(statistics.fmean(draws), 20.0, delta=0.5)
        self.assertGreaterEqual(min(draws), 0)
        self.assertLessEqual(max(draws), 90)

    def test_truncation(self):
        db = _pedestrian_database(60.0, maximum=30.0)
        self.assertLessEqual(max(db._sample_departure_time(k) for k in range(5000)), 30)

    def test_common_random_numbers(self):
        a = _pedestrian_database(20.0, seed=7)
        b = _pedestrian_database(20.0, seed=7)
        c = _pedestrian_database(20.0, seed=8)
        draws_a = [a._sample_departure_time(k) for k in range(1000)]
        self.assertEqual(draws_a, [b._sample_departure_time(k) for k in range(1000)])
        self.assertNotEqual(draws_a, [c._sample_departure_time(k) for k in range(1000)])

    def test_has_departed(self):
        db = _pedestrian_database(20.0)
        waiting = SimpleNamespace(departure_time=10, affected=False, panicked=False)
        db.currTime = 9
        self.assertFalse(db.has_departed(waiting))
        db.currTime = 10
        self.assertTrue(db.has_departed(waiting))
        db.currTime = 0
        self.assertTrue(db.has_departed(SimpleNamespace(departure_time=10, affected=True, panicked=False)))
        self.assertTrue(db.has_departed(SimpleNamespace(departure_time=10, affected=False, panicked=True)))

    def test_invalid_parameters_raise(self):
        db = pedestrian_module.PedDS.__new__(pedestrian_module.PedDS)
        for kwargs in ({"mean_minutes": -1.0}, {"shape": 0.0}, {"max_minutes": float("nan")}):
            with self.assertRaises(ValueError):
                db.configure_departure(**kwargs)

    def test_decision_interface_wait(self):
        wait = decision_module._departure_wait_minutes
        self.assertEqual(wait(SimpleNamespace(departure_time=12), 4, 1.0), 8.0)
        self.assertEqual(wait(SimpleNamespace(departure_time=3), 4, 1.0), 0.0)
        self.assertEqual(wait(SimpleNamespace(departure_time=12, affected=True), 4, 1.0), 0.0)
        self.assertEqual(wait(SimpleNamespace(departure_time=12, panicked=True), 4, 1.0), 0.0)
        self.assertEqual(wait(SimpleNamespace(), 4, 1.0), 0.0)


@unittest.skipIf(gnn_module is None, "torch is not importable")
class DeferPrior(unittest.TestCase):
    def test_zero_reproduces_v651_tie_rule(self):
        prior = torch.zeros(1, 6)
        out = gnn_module.apply_defer_prior(prior, "zero")
        self.assertEqual(float(out[0, 0]), 0.0)
        # All scores tie: the argmax is slot 0, which is defer (finding T1-4).
        self.assertEqual(int(out.argmax(dim=1)), 0)

    def test_below_sites_breaks_ties_like_the_index_rule(self):
        prior = torch.zeros(1, 6)
        out = gnn_module.apply_defer_prior(prior, "below_sites")
        self.assertEqual(float(out[0, 0]), -1.0)
        self.assertEqual(int(out.argmax(dim=1)), 1)

    def test_below_sites_keeps_site_logits(self):
        prior = torch.tensor([[5.0, 0.3, -1.2, 2.0], [0.0, 1.0, 1.0, 1.0]])
        out = gnn_module.apply_defer_prior(prior, "below_sites")
        self.assertTrue(torch.equal(out[:, 1:], prior[:, 1:]))
        self.assertAlmostEqual(float(out[0, 0]), -2.2, places=6)
        self.assertAlmostEqual(float(out[1, 0]), 0.0, places=6)
        self.assertTrue(torch.equal(prior[:, 0], torch.tensor([5.0, 0.0])))  # input untouched

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            gnn_module.apply_defer_prior(torch.zeros(1, 3), "above_sites")


class ConfigurationContract(unittest.TestCase):
    def test_defer_mode_is_a_core_field_with_v651_default(self):
        import NMCCPIConfig

        fields = {attribute: (keyword, default) for attribute, keyword, _, default in NMCCPIConfig.NMCC_PI_ALL_CORE_FIELDS}
        self.assertEqual(fields["actorPriorDeferMode"], ("actor_prior_defer_mode", "zero"))

    def test_curriculum_accepts_the_v66_keys(self):
        import TrainingCurriculum

        keys = {
            "candidateSamplingMode", "initialShelterMode", "plannerShortlistObservation",
            "departureDelayMeanMinutes", "departureDelayShape", "departureDelayMaxMinutes",
            "actorPriorDeferMode",
        }
        containers = [
            value for value in vars(TrainingCurriculum).values()
            if isinstance(value, (set, frozenset)) and "hazardWindSpeedMPerMinute" in value
        ]
        self.assertTrue(containers, "learner-override whitelist not found")
        self.assertTrue(keys <= set(containers[0]))

    def test_core_defaults_reproduce_v651(self):
        source = (ROOT / "Core.py").read_text()  # importing Core needs osmnx
        for line in (
            'self.candidateSamplingMode = "legacy_row_major"',
            'self.initialShelterMode = "legacy_row_major"',
            'self.plannerShortlistObservation = "raw"',
            "self.departureDelayMeanMinutes = 0.0",
        ):
            self.assertIn(line, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
