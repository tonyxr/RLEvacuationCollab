"""Forward-looking evacuation orders (package ``recourse``).

Physics on the regime testbed with the site-loss hook: an order closes the
targeted shelter at its true danger through the simulator's own transaction;
the forecast rule orders right after the ignition becomes public; the
stopping rule grades later orders as costlier; deciding never mutates the live
simulator. Wiring: decisions taken at the live boundary execute in the next
minute's live closure phase only, never in branches or in the post-episode
audit. Evaluator: arms without recourse keep their keys, so the stored
site_loss episodes are reused.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import CounterfactualBranch as CB  # noqa: E402
import PedestrianDatabase  # noqa: E402
import panel_eval as PE  # noqa: E402
from recourse import ShelterRecourseController, execute_evacuation_orders  # noqa: E402
from recourse import runtime as RT  # noqa: E402
from recourse.rollout import NEVER, branch_step, evaluate_stopping  # noqa: E402
from scenario_hooks import InitialSiteLoss  # noqa: E402
from v65_testbed import full_state_digest, regime_testbed  # noqa: E402

SEED = 6602
CTRL = "recourse.controller:ShelterRecourseController"
HOOK = "scenario_hooks.initial_site_loss:InitialSiteLoss"


def _site_loss_core(targets=1, ignition=5, stop=91):
    core = regime_testbed(seed=SEED, stop_time=stop)
    applied = InitialSiteLoss(targets=targets, ignition_minute=ignition).apply(core)
    return core, applied


def _drive(core, controller, minutes):
    """Live-loop order: step t executes the orders decided at boundary t - 1."""
    queue, executed = [], []
    for t in range(1, minutes + 1):
        executed += branch_step(core, order_cells=queue)
        queue = controller.decide(core, t) if controller is not None else []
    return executed


class OrderTransactionTests(unittest.TestCase):
    def test_order_closes_the_target_at_its_true_danger_before_overrun(self):
        core, applied = _site_loss_core()
        target = applied["targeted_shelter_ids"][0]
        shelter = core.shelterDS.shelterList[target]
        cell = tuple(shelter.cellLocated)
        _drive(core, None, 19)
        occupants = int(shelter.shelterFlow)
        reserved = int(core.shelterDS.reservedPopulation(shelter))
        danger = int(core.cellTracker.getCellState(cell))
        self.assertLess(danger, 4)
        self.assertGreater(occupants + reserved, 0)
        records = branch_step(core, order_cells=[cell])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record["shelter_id"], record["minute"]), (target, 20))
        self.assertEqual(record["occupants_evacuated"], occupants)
        self.assertEqual(record["reservations_released"], reserved)
        self.assertTrue(record["ordered_before_overrun"])
        live = core.shelterDS.shelterList[target]
        self.assertEqual(int(live.status), 2)
        self.assertEqual(int(live.overrunTime), 20)
        self.assertLess(int(live.overrunDangerLevel), 4)
        self.assertEqual(int(live.shelterFlow), 0)
        self.assertEqual(int(core.shelterDS.reservedPopulation(live)), 0)
        _drive(core, None, 20)  # through minute 40: the region reaches level 4 at 34
        self.assertEqual(int(core.shelterDS.shelterList[target].overrunTime), 20)
        self.assertEqual(int(core.shelterDS.shelterList[target].forcedReevacuatedPopulation), occupants)

    def test_unordered_branch_step_equals_the_engine_step(self):
        left, _ = _site_loss_core()
        right = copy.deepcopy(left)
        for _ in range(15):
            CB.advance_one_timestep(left)
            branch_step(right)
        self.assertEqual(full_state_digest(left), full_state_digest(right))

    def test_ordering_an_empty_or_closed_region_is_a_no_op(self):
        core, applied = _site_loss_core()
        self.assertEqual(execute_evacuation_orders(core, [], simulation_time=1), [])
        self.assertEqual(execute_evacuation_orders(core, [(99, 99)], simulation_time=1), [])


class ControllerTests(unittest.TestCase):
    def test_forecast_rule_orders_as_soon_as_the_ignition_is_public(self):
        for ignition in (5, 35):
            core, applied = _site_loss_core(ignition=ignition)
            controller = ShelterRecourseController(rule="forecast")
            executed = _drive(core, controller, ignition + 35)
            self.assertEqual([r["minute"] for r in executed], [ignition + 1], ignition)
            self.assertEqual([r["shelter_id"] for r in executed], applied["targeted_shelter_ids"])
            order = controller.episode.orders[0]
            self.assertEqual(order["decided_minute"], ignition)
            self.assertGreaterEqual(order["probability"], 0.5)
            self.assertAlmostEqual(order["expected_crossing_minutes"], 29.0)

    def test_a_shorter_lead_orders_later(self):
        core, _ = _site_loss_core(ignition=5)
        executed = _drive(core, ShelterRecourseController(rule="forecast", lead_minutes=15), 40)
        self.assertEqual([r["minute"] for r in executed], [21])

    def test_both_targets_are_ordered(self):
        core, applied = _site_loss_core(targets=2, ignition=35)
        executed = _drive(core, ShelterRecourseController(rule="forecast"), 60)
        self.assertEqual(sorted(r["shelter_id"] for r in executed), sorted(applied["targeted_shelter_ids"]))
        self.assertEqual({r["minute"] for r in executed}, {36})

    def test_deciding_never_mutates_the_live_simulator(self):
        for rule in ("forecast", "rollout"):
            core, _ = _site_loss_core()
            _drive(core, None, 5)
            controller = ShelterRecourseController(rule=rule, rollout_futures=2)
            before = full_state_digest(core)
            cells = controller.decide(core, 5)
            self.assertTrue(cells, rule)
            self.assertEqual(full_state_digest(core), before, rule)

    def test_stopping_rule_grades_delay_as_costlier_and_orders_now(self):
        core, applied = _site_loss_core(ignition=35)
        _drive(core, None, 35)
        cell = tuple(core.shelterDS.shelterList[applied["targeted_shelter_ids"][0]].cellLocated)
        result = evaluate_stopping(core, [cell], offsets=(0, 5, 10, 15, 20, 25), futures=2,
                                   horizon_minutes=45, seed=11)
        self.assertEqual(result.offsets, (0, 5, 10, 15, 20, 25, NEVER))
        self.assertEqual(result.best, 0)
        self.assertLess(result.mean_cost[0], result.mean_cost[NEVER])
        again = evaluate_stopping(core, [cell], offsets=(0, 5, 10, 15, 20, 25), futures=2,
                                  horizon_minutes=45, seed=11)
        self.assertEqual(result.mean_cost, again.mean_cost)

    def test_parameters_are_validated_and_forecast_keys_omit_rollout_settings(self):
        for bad in ({"rule": "x"}, {"threshold": 0.0}, {"lead_minutes": 0}, {"check_interval": 0},
                    {"rollout_offsets": (5, 10)}, {"rollout_offsets": (0, 50)}):
            with self.assertRaises(ValueError, msg=bad):
                ShelterRecourseController(**bad)
        forecast = ShelterRecourseController(rule="forecast").params()
        self.assertNotIn("rollout_futures", forecast)
        self.assertIn("rollout_futures", ShelterRecourseController(rule="rollout").params())

    def test_no_order_without_people_to_protect(self):
        core, _ = _site_loss_core()
        _drive(core, None, 5)
        controller = ShelterRecourseController(rule="forecast", min_people=10 ** 6)
        self.assertEqual(controller.decide(core, 5), [])


class _Bridge:
    def step(self, *, simulation_time, is_terminal=False):
        return {}


class LiveWiringTests(unittest.TestCase):
    """The runtime on a testbed core driven in the live loop's call order."""

    def _loop_class(self, core, minutes, branch_calls):
        base = type(core)

        class Loop(base):
            def _advance_exogenous_hazard(self, sim_time, timer=None):
                regime = self.hazard_regime
                regime.advance(int(sim_time), self.hazardDS, self.mapDS)
                self.hazardDS.spreadUpdate()
                self.hazardDS.heatUpdate()
                self.hazardDS.smokeUpdate()
                self.hazardDS.terminateHazard()

            def simulationEnumerator(self):
                for t in range(1, minutes + 1):
                    self.pedDS.startDocument()
                    self._advance_exogenous_hazard(t)
                    self.pedDS.resolveHazardShelterClosures(simulation_time=t)
                    self.pedDS.loadShelterLookup(self.shelterDS.shelterByOSMID)
                    self.pedDS.pedestrianHazardInteraction()
                    self.pedDS.interPedestrianInteraction()
                    self.pedDS.pedestrianNetworkInteraction()
                    self.cellTracker.cellUpdate(pedDS=self.pedDS, forceTracker=self.forceTracker)
                    self.pedDS.docuStatus()
                    if t in branch_calls:  # a planner branch between steps: never live
                        clone = copy.deepcopy(self.pedDS)
                        clone.resolveHazardShelterClosures(simulation_time=t + 1)
                    self.rl.step(simulation_time=t, is_terminal=(t == minutes))
                self._advance_exogenous_hazard(minutes + 1)  # post-episode hazard audit

        return Loop

    def test_orders_execute_in_the_next_live_closure_phase_only(self):
        core, applied = _site_loss_core()
        Loop = self._loop_class(core, 40, branch_calls={5, 6})
        core.__class__ = Loop
        core.rl = _Bridge()
        saved = {
            "loop": Loop.simulationEnumerator, "hazard": Loop._advance_exogenous_hazard,
            "resolve": PedestrianDatabase.PedDS.resolveHazardShelterClosures, "step": _Bridge.step,
        }
        try:
            RT.install(Loop, _Bridge, PedestrianDatabase.PedDS)
            RT.install(Loop, _Bridge, PedestrianDatabase.PedDS)  # idempotent
            RT.LIVE.reset(ShelterRecourseController(rule="forecast"))
            core.simulationEnumerator()
            result = RT.LIVE.result()
            self.assertIsNone(RT.LIVE.core)
        finally:
            Loop.simulationEnumerator = saved["loop"]
            Loop._advance_exogenous_hazard = saved["hazard"]
            PedestrianDatabase.PedDS.resolveHazardShelterClosures = saved["resolve"]
            _Bridge.step = saved["step"]
            RT.LIVE.reset(None)
        self.assertEqual([(r["shelter_id"], r["minute"]) for r in result["executed"]],
                         [(applied["targeted_shelter_ids"][0], 6)])
        self.assertEqual(result["false_alarm_regions"], 0)
        self.assertEqual(int(core.shelterDS.shelterList[applied["targeted_shelter_ids"][0]].overrunTime), 6)

    def test_scenario_runtime_records_live_closures_with_the_true_danger(self):
        core = regime_testbed(seed=SEED, stop_time=91)  # the scenario runtime applies the hook
        Loop = self._loop_class(core, 40, branch_calls={10})
        core.__class__ = Loop
        core.rl = _Bridge()
        import ShelterDatabase as SD

        saved = (Loop.simulationEnumerator, Loop._advance_exogenous_hazard,
                 PedestrianDatabase.PedDS.resolveHazardShelterClosures, _Bridge.step,
                 SD.ShelterDS.closeHazardOverrunShelters)
        try:
            PE.install_scenario_runtime(Loop)
            RT.install(Loop, _Bridge, PedestrianDatabase.PedDS)
            PE.SCENARIO.reset({"hook": HOOK, "params": {"targets": 1, "ignition_minute": 5}})
            RT.LIVE.reset(ShelterRecourseController(rule="forecast"))
            core.simulationEnumerator()
            closures = list(PE.SCENARIO.closures)
            audit = PE.SCENARIO.result()["audit"]
        finally:
            (Loop.simulationEnumerator, Loop._advance_exogenous_hazard,
             PedestrianDatabase.PedDS.resolveHazardShelterClosures, _Bridge.step,
             SD.ShelterDS.closeHazardOverrunShelters) = saved
            PE.SCENARIO.reset(None)
            RT.LIVE.reset(None)
        self.assertEqual(len(closures), 1)
        self.assertTrue(closures[0]["ordered"])
        self.assertEqual(closures[0]["minute"], 6)
        self.assertLess(closures[0]["danger"], 4)
        self.assertEqual(audit["first_target_overrun_minute"], 6)  # closed by the order, not the fire
        self.assertFalse(audit["targets_overrun_as_designed"])


class RecoursePanelTests(unittest.TestCase):
    def test_off_arms_reproduce_the_site_loss_arms_exactly(self):
        site_loss = json.loads((ROOT / "config/panels/site_loss.json").read_text())
        recourse = json.loads((ROOT / "config/panels/site_loss_recourse.json").read_text())
        rollout = json.loads((ROOT / "config/panels/site_loss_recourse_rollout.json").read_text())
        key_fields = ("strategy", "policies", "scenario", "overrides", "checkpoint", "replications", "cities")
        reference = {a["arm"]: {k: a.get(k) for k in key_fields} for a in site_loss["arms"]}
        for spec in (recourse, rollout):
            for field in ("campaign_dir", "cities", "training"):
                self.assertEqual(spec[field], site_loss[field], field)
            for arm in spec["arms"]:
                name, mode = arm["arm"].rsplit("__", 1)
                expected = dict(reference[name])
                if spec is rollout and arm["labels"]["policy"] == "rl":
                    expected["policies"] = [1]
                self.assertEqual({k: arm.get(k) for k in key_fields}, expected, arm["arm"])
                self.assertEqual(arm["recourse"] is None, mode == "off")
                if arm["recourse"] is not None:
                    PE.recourse_controller(arm["recourse"])
        self.assertEqual(len(recourse["arms"]), 75)
        self.assertEqual(len(rollout["arms"]), 18)

    def test_recourse_enters_only_the_keys_of_its_own_jobs(self):
        import test_v651_panel_eval as T

        fixture = T.PanelEvaluatorEquivalenceTests
        fixture.setUpClass()
        try:
            scenario = {"hook": HOOK, "params": {"targets": 1, "ignition_minute": 5}}
            plain = [{"arm": "g", "strategy": "objective_greedy", "scenario": scenario}]
            mixed = [
                {"arm": "g__off", "strategy": "objective_greedy", "scenario": scenario, "recourse": None},
                {"arm": "g__f30", "strategy": "objective_greedy", "scenario": scenario,
                 "recourse": {"controller": CTRL, "params": {"rule": "forecast"}}},
                {"arm": "g__f15", "strategy": "objective_greedy", "scenario": scenario,
                 "recourse": {"controller": CTRL, "params": {"rule": "forecast", "lead_minutes": 15}}},
            ]

            def jobs(arms):
                spec = fixture._spec(fixture, arms=arms)
                cli = PE.build_parser().parse_args(["plan", "--spec", str(spec)])
                return PE.prepare(cli)[3]

            base = jobs(plain)
            mixed_jobs = jobs(mixed)
        finally:
            fixture.tearDownClass()
        by_arm = {}
        for job in mixed_jobs:
            by_arm.setdefault(job.arm, []).append(job)
        self.assertEqual([j.key for j in by_arm["g__off"]], [j.key for j in base])
        self.assertTrue(set(j.key for j in by_arm["g__f30"]).isdisjoint(j.key for j in base))
        self.assertTrue(set(j.key for j in by_arm["g__f30"]).isdisjoint(j.key for j in by_arm["g__f15"]))
        job = by_arm["g__f30"][0]
        self.assertEqual(job.recourse["params"]["lead_minutes"], 30)
        self.assertIsNotNone(job.recourse_source_sha256)

    def test_controller_validation(self):
        with self.assertRaises(ValueError):
            PE.recourse_controller({"controller": "os:system", "params": {}})
        with self.assertRaises(ValueError):
            PE.recourse_controller({"controller": CTRL, "params": {"rule": "nope"}})
        with self.assertRaises(TypeError):
            PE.recourse_controller({"controller": CTRL, "params": {"unknown": 1}})


if __name__ == "__main__":
    unittest.main()


class RecourseResultsTests(unittest.TestCase):
    """The analysis script on a synthetic collected panel with known answers."""

    VALUE = {"rl": 0.050, "objective_greedy": 0.040, "objective_greedy_legacy": 0.040,
             "static_greedy": 0.030, "heuristic": 0.020}

    def _frame(self, *, break_identity=False, miss=False):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(1)
        rows = []
        for city in ("state_college_pa", "spokane_wa"):
            for tape in range(1, 7):
                base_cost = 0.35 + 0.1 * rng.random()
                for policy, value in self.VALUE.items():
                    for seed in ((1, 2, 3) if policy == "rl" else (0,)):
                        for cell in ("base", "t1_m05", "t1_m35", "t2_m05", "t2_m35"):
                            noise = 0.001 * rng.standard_normal()
                            for mode in ("off", "fL30", "fL15"):
                                stressed = cell != "base"
                                targets = [] if not stressed else ([0] if cell.startswith("t1") else [0, 1])
                                designed = 34 if cell.endswith("m05") else 64
                                ordered = stressed and mode != "off"
                                cost = base_cost + (0.2 if stressed else 0.0) + noise
                                if ordered:
                                    cost -= value * (1.0 if mode == "fL30" else 0.5)
                                if ordered:
                                    minute = (6 if cell.endswith("m05") else 36) + (0 if mode == "fL30" else 15)
                                    events = [{"shelter_id": t, "time": minute, "danger_level": 1,
                                               "forced_reevacuated_population": 5} for t in targets]
                                else:
                                    events = [{"shelter_id": t, "time": designed, "danger_level": 4,
                                               "forced_reevacuated_population": 60} for t in targets]
                                if miss and ordered and cell == "t1_m05" and tape == 2 and policy == "heuristic":
                                    events = []
                                n = 2500.0
                                row = {
                                    "job_key": f"{city}-{tape}-{policy}-{seed}-{cell}-{mode}",
                                    "label_policy": policy, "label_cell": cell, "label_recourse": mode,
                                    "city_id": city, "city_scenario_replication": tape, "policy_replication": seed,
                                    "initial_population": n, "active_person_time_weight": 0.5,
                                    "hazard_exposure_time_weight": 2.0,
                                    "objective_active_person_time": cost * n * 180 / 0.5,
                                    "realized_terminal_plain_time": 0.0, "objective_hazard_exposure_person_time": 0.0,
                                    "realized_terminal_hazard_exposure_time": 0.0,
                                    "objective_expected_casualty_person_time": 0.0,
                                    "objective_normalized_cost": cost, "expected_casualties": 20.0,
                                    "unfinished": 150.0, "safe_completed": 2350.0,
                                    "forced_reevacuated": 0.0 if ordered else 60.0 * len(targets),
                                    "shelter_overrun_events": json.dumps(events),
                                    "scenario_targeted_shelter_ids": json.dumps(targets) if stressed else "",
                                }
                                if mode != "off":
                                    row.update({
                                        "recourse_closed_shelters": len(targets) if ordered else 0,
                                        "recourse_order_minutes": json.dumps([e["time"] for e in events] if ordered else []),
                                        "recourse_order_dangers": json.dumps([1] * len(targets) if ordered else []),
                                        "recourse_occupants_evacuated": 5 * len(targets) if ordered else 0,
                                        "recourse_reservations_released": 40 * len(targets) if ordered else 0,
                                        "recourse_false_alarm_regions": 0, "recourse_wall_s": 0.5,
                                        "recourse_evaluations": 0,
                                    })
                                rows.append(row)
        frame = pd.DataFrame(rows)
        if break_identity:
            index = frame.index[(frame.label_recourse == "fL30") & (frame.label_cell == "base")][0]
            frame.loc[index, "unfinished"] = 151.0
        return frame

    def _run(self, frame, *extra):
        import tempfile
        from unittest import mock

        import site_loss_recourse_results as R

        campaign = Path(tempfile.mkdtemp(prefix="recourse_results_"))
        (campaign / "panels" / "site_loss_recourse").mkdir(parents=True)
        frame.to_csv(campaign / "panels" / "site_loss_recourse" / "evaluation_episode_summary.csv", index=False)
        with mock.patch("builtins.print"):
            code = R.main(["--campaign", str(campaign), *extra])
        return code, campaign / "recourse_results"

    def test_value_of_recourse_and_checks(self):
        import pandas as pd

        code, out = self._run(self._frame(), "--check")
        self.assertEqual(code, 0)
        values = pd.read_csv(out / "recourse_value.csv")
        macro = values[(values.metric == "objective_normalized_cost") & (values.scope == "macro")]
        get = lambda m, c, p: float(macro[(macro["mode"] == m) & (macro.cell == c) & (macro.policy == p)].value_mean.iloc[0])  # noqa: E731
        self.assertAlmostEqual(get("fL30", "t1_m35", "rl"), 0.050, places=6)
        self.assertAlmostEqual(get("fL15", "t2_m05", "heuristic"), 0.010, places=6)
        self.assertAlmostEqual(get("fL30", "base", "objective_greedy"), 0.0, places=9)
        status = pd.read_csv(out / "target_status.csv")
        self.assertEqual(set(status[status["mode"] != "off"].status), {"ordered"})
        self.assertEqual(set(status[status["mode"] == "off"].status), {"delivered"})
        identity = pd.read_csv(out / "zero_order_identity.csv")
        self.assertEqual(len(identity), 2 * 2 * 6 * 7)  # both modes, base cell, all tapes and policy seeds
        self.assertTrue((identity.differing_columns == 0).all())
        contrasts = pd.read_csv(out / "recourse_contrasts.csv")
        row = contrasts[(contrasts["mode"] == "fL30") & (contrasts.left == "objective_greedy")
                        & (contrasts.right == "rl") & (contrasts.cell == "t1_m35")]
        self.assertAlmostEqual(float(row.change_vs_off_mean.iloc[0]), 0.010, places=6)

    def test_check_fails_on_identity_break_or_missed_target(self):
        self.assertEqual(self._run(self._frame(break_identity=True), "--check")[0], 1)
        self.assertEqual(self._run(self._frame(miss=True), "--check")[0], 1)
