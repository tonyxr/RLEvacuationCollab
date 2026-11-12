"""Initial-site-loss corner case (scenario_hooks.initial_site_loss) and the
panel evaluator's scenario hooks.

Physics: on the regime testbed the targeted initial shelters fail exactly at
``ignition + 30 - 1`` minutes (the ignition minute is the first burning
minute), never in the unhooked control, and the hook draws no random number.
Evaluator: specs without hooks keep their exact pre-hook episode keys; hook
parameters and source enter the key of hooked jobs only; the runtime applies
the hook once before the first transition and records closures of the
episode's own shelter store only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]

import ShelterDatabase as SD  # noqa: E402
import panel_eval as PE  # noqa: E402
from scenario_hooks import InitialSiteLoss  # noqa: E402
from v65_testbed import advance, regime_testbed  # noqa: E402

HOOK = "scenario_hooks.initial_site_loss:InitialSiteLoss"


def _closure_log():
    log = []
    original = SD.ShelterDS.closeHazardOverrunShelters

    def wrapped(self, cellTracker, simulation_time=None):
        transitions = original(self, cellTracker, simulation_time=simulation_time)
        for shelter, released, danger in transitions:
            log.append({"shelter_id": int(shelter.shelterID), "minute": int(self.currentSimulationTime),
                        "released_population": int(released), "danger": int(danger)})
        return transitions

    return log, mock.patch.object(SD.ShelterDS, "closeHazardOverrunShelters", wrapped)


class InitialSiteLossPhysicsTests(unittest.TestCase):
    SEED = 6602

    def _run(self, hook: InitialSiteLoss | None, minutes: int = 70):
        core = regime_testbed(seed=self.SEED, stop_time=91)
        log, patch = _closure_log()
        before_rng = copy.deepcopy(core.hazard_regime.rng.bit_generator.state)
        before_events = copy.deepcopy(core.hazard_regime.scheduled)
        applied = None if hook is None else hook.apply(core)
        self.assertEqual(core.hazard_regime.rng.bit_generator.state, before_rng)
        if applied is not None:
            added = [e for e in core.hazard_regime.scheduled if e not in before_events]
            self.assertEqual(len(added), len(applied["ignited_cells"]))
            self.assertTrue(all(e["kind"] == "ignition" and e["time"] == hook.ignition_minute for e in added))
        with patch:
            advance(core, minutes)
        return core, applied, log

    def test_one_target_fails_exactly_when_designed(self):
        hook = InitialSiteLoss(targets=1, ignition_minute=5)
        core, applied, log = self._run(hook)
        self.assertEqual(applied["expected_overrun_minute"], 34.0)
        audit = hook.audit(core, applied, log)
        self.assertTrue(audit["all_targets_overrun"])
        self.assertTrue(audit["targets_overrun_as_designed"])
        self.assertEqual(audit["first_target_overrun_minute"], 34)
        target = applied["targeted_shelter_ids"][0]
        self.assertEqual([c["minute"] for c in log if c["shelter_id"] == target], [34])

    def test_two_targets_and_a_late_ignition(self):
        hook = InitialSiteLoss(targets=2, ignition_minute=35)
        core, applied, log = self._run(hook, minutes=80)
        self.assertEqual(len(applied["targeted_shelter_ids"]), 2)
        audit = hook.audit(core, applied, log)
        self.assertTrue(audit["targets_overrun_as_designed"], audit)
        self.assertEqual(set(audit["overrun_minute_by_shelter"].values()), {64})

    def test_control_without_the_hook_loses_no_initial_shelter(self):
        core, _, log = self._run(None, minutes=80)
        initial = InitialSiteLoss.initial_shelter_ids(core)
        self.assertFalse([c for c in log if c["shelter_id"] in initial])

    def test_regime_events_are_identical_with_and_without_the_hook(self):
        plain, _, _ = self._run(None)
        hooked, applied, _ = self._run(InitialSiteLoss(targets=1, ignition_minute=5))
        added = {("ignition", applied["ignited_cell_indices"][0], 5)}
        events = lambda core: sorted(  # noqa: E731
            (e["kind"], json.dumps(e["payload"], sort_keys=True), e["time"]) for e in core.hazard_regime.scheduled
            if (e["kind"], e["payload"].get("cell_index"), e["time"]) not in added
        )
        self.assertEqual(events(plain), events(hooked))
        self.assertEqual(plain.hazard_regime.rng.bit_generator.state, hooked.hazard_regime.rng.bit_generator.state)

    def test_same_region_targets_are_reported(self):
        core = regime_testbed(seed=self.SEED, stop_time=91)
        first, second = InitialSiteLoss.initial_shelter_ids(core)[:2]
        shelters = core.shelterDS.shelterList
        shelters[second].cellLocated = shelters[first].cellLocated
        applied = InitialSiteLoss(targets=1, ignition_minute=5).apply(core)
        self.assertEqual(applied["targeted_shelter_ids"], [first])
        self.assertEqual(applied["shelters_in_ignited_cells"], sorted([first, second]))
        self.assertEqual(len(applied["ignited_cells"]), 1)

    def test_invalid_constructions_fail_closed(self):
        with self.assertRaises(ValueError):
            InitialSiteLoss(targets=3)
        with self.assertRaises(ValueError):
            InitialSiteLoss(ignition_minute=0)
        core = regime_testbed(seed=self.SEED, stop_time=40)
        with self.assertRaisesRegex(RuntimeError, "after the"):
            InitialSiteLoss(ignition_minute=15).apply(core)
        core = regime_testbed(seed=self.SEED, stop_time=91)
        core.hazard_regime.enabled = False
        with self.assertRaisesRegex(RuntimeError, "hazard regime"):
            InitialSiteLoss().apply(core)


# ---------------------------------------------------------------------------
class ScenarioRuntimeTests(unittest.TestCase):
    def test_runtime_applies_once_audits_and_ignores_branch_stores(self):
        core = regime_testbed(seed=6602, stop_time=91)

        class Loop(type(core)):
            def simulationEnumerator(self):
                clone = copy.deepcopy(self.shelterDS)  # stands in for a planner branch store
                clone.closeHazardOverrunShelters(self.cellTracker, simulation_time=0)
                advance(self, 60)

        core.__class__ = Loop
        original_close = SD.ShelterDS.closeHazardOverrunShelters
        try:
            PE.install_scenario_runtime(Loop)
            PE.install_scenario_runtime(Loop)  # idempotent
            PE.SCENARIO.reset({"hook": HOOK, "params": {"targets": 1, "ignition_minute": 5}})
            core.simulationEnumerator()
            result = PE.SCENARIO.result()
        finally:
            SD.ShelterDS.closeHazardOverrunShelters = original_close
            PE.SCENARIO.reset(None)
        self.assertEqual(result["applied"]["expected_overrun_minute"], 34.0)
        self.assertTrue(result["audit"]["targets_overrun_as_designed"])
        self.assertEqual([c["minute"] for c in result["closures"]], [34])

    def test_declared_but_unapplied_scenario_is_an_error(self):
        PE.SCENARIO.reset({"hook": HOOK, "params": {}})
        try:
            with self.assertRaisesRegex(RuntimeError, "never applied"):
                PE.SCENARIO.result()
        finally:
            PE.SCENARIO.reset(None)

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            PE.scenario_hook({"hook": "os:system", "params": {}})
        with self.assertRaises(ValueError):
            PE.scenario_hook({"hook": HOOK, "params": {"targets": 5}})
        with self.assertRaises(TypeError):
            PE.scenario_hook({"hook": HOOK, "params": {"unknown": 1}})
        spec = json.loads((ROOT / "config" / "panels" / "site_loss.json").read_text())
        self.assertEqual(len(spec["arms"]), 25)
        for arm in spec["arms"]:
            if arm["scenario"] is not None:
                PE.scenario_hook(arm["scenario"])


# ---------------------------------------------------------------------------
def _frozen_panel_eval():
    """The panel evaluator as frozen (before scenario hooks), loaded from git."""
    for ref in ("v65.1-freeze", "HEAD~1", "HEAD"):
        source = subprocess.run(["git", "show", f"{ref}:scripts/panel_eval.py"], cwd=ROOT,
                                capture_output=True, text=True)
        if source.returncode == 0 and "scenario_hook" not in source.stdout:
            break
    else:
        return None
    path = Path(tempfile.mkdtemp(prefix="frozen_panel_eval_")) / "panel_eval_frozen.py"
    path.write_text(source.stdout.replace(
        "ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(ROOT)!r})"))
    spec = importlib.util.spec_from_file_location("panel_eval_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScenarioKeyTests(unittest.TestCase):
    """Keys on the multicity fixture of test_v651_panel_eval."""

    @classmethod
    def setUpClass(cls):
        import test_v651_panel_eval as T

        cls.T = T
        cls.fixture = T.PanelEvaluatorEquivalenceTests
        cls.fixture.setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def _spec(self, **extra):
        return self.fixture._spec(self.fixture, **extra)

    def _jobs(self, module, spec_path):
        cli = module.build_parser().parse_args(["plan", "--spec", str(spec_path)])
        return module.prepare(cli)[3]

    def test_specs_without_hooks_keep_their_exact_keys(self):
        frozen = _frozen_panel_eval()
        if frozen is None:
            self.skipTest("frozen panel_eval not available from git")
        spec = self._spec()
        self.assertEqual([j.key for j in self._jobs(frozen, spec)], [j.key for j in self._jobs(PE, spec)])

    def test_hook_parameters_and_source_enter_hooked_keys_only(self):
        arms = [
            {"arm": "greedy_base", "strategy": "objective_greedy", "scenario": None},
            {"arm": "greedy_plain", "strategy": "objective_greedy"},
            {"arm": "greedy_t1_m05", "strategy": "objective_greedy",
             "scenario": {"hook": HOOK, "params": {"targets": 1, "ignition_minute": 5}}},
            {"arm": "greedy_t1_m35", "strategy": "objective_greedy",
             "scenario": {"hook": HOOK, "params": {"targets": 1, "ignition_minute": 35}}},
        ]
        jobs = self._jobs(PE, self._spec(arms=arms))
        by_arm = {}
        for job in jobs:
            by_arm.setdefault(job.arm, []).append(job)
        base = [j.key for j in by_arm["greedy_base"]]
        self.assertEqual(base, [j.key for j in by_arm["greedy_plain"]])
        self.assertFalse(set(base) & {j.key for j in by_arm["greedy_t1_m05"]})
        self.assertFalse({j.key for j in by_arm["greedy_t1_m05"]} & {j.key for j in by_arm["greedy_t1_m35"]})
        self.assertTrue(all(j.scenario["params"] == {"targets": 1, "ignition_minute": 5}
                            for j in by_arm["greedy_t1_m05"]))
        self.assertEqual([j.scenario_seed for j in by_arm["greedy_base"]],
                         [j.scenario_seed for j in by_arm["greedy_t1_m05"]])
        with mock.patch.object(PE, "_scenario_source_sha", lambda: "0" * 64):
            changed = self._jobs(PE, self._spec(arms=arms))
        self.assertEqual({j.key for j in changed if j.arm == "greedy_base"}, set(base))
        self.assertFalse({j.key for j in changed if j.arm == "greedy_t1_m05"}
                         & {j.key for j in by_arm["greedy_t1_m05"]})


if __name__ == "__main__":
    unittest.main()


class SiteLossResultsTests(unittest.TestCase):
    """The analysis script on a synthetic collected panel with known answers."""

    def _frame(self, broken: bool = False, preempt: bool = True, inconsistent: bool = False):
        """Synthetic panel with known penalties. With ``preempt``, the tape's own fire
        overruns initial shelter 1 at minute 30 on State College tape 1 in every
        episode (base and stressed), as on the real pilot."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        penalty = {"rl": 0.010, "objective_greedy": 0.012, "objective_greedy_legacy": 0.020,
                   "static_greedy": 0.040, "heuristic": 0.030}
        rows = []
        for city in ("state_college_pa", "spokane_wa"):
            for tape in range(1, 21):
                base_cost = 0.35 + 0.1 * rng.random()
                natural = preempt and city == "state_college_pa" and tape == 1
                for policy, extra in penalty.items():
                    seeds = (1, 2, 3) if policy == "rl" else (0,)
                    for seed in seeds:
                        for cell in ("base", "t1_m05", "t1_m35", "t2_m05", "t2_m35"):
                            stressed = cell != "base"
                            scale = 2.0 if cell.startswith("t2") else 1.0
                            cost = base_cost + (extra * scale if stressed else 0.0) + 0.001 * rng.standard_normal()
                            events = []
                            if natural and not (inconsistent and cell == "base" and policy == "heuristic"):
                                events.append({"shelter_id": 1, "time": 30, "danger_level": 4,
                                               "forced_reevacuated_population": 5})
                            row = {"label_policy": policy, "label_cell": cell, "city_id": city,
                                   "city_scenario_replication": tape, "policy_replication": seed,
                                   "objective_normalized_cost": cost, "unfinished": 150 + (40 * scale if stressed else 0),
                                   "safe_completed": 2350 - (40 * scale if stressed else 0),
                                   "expected_casualties": 15.0, "overrun_shelters": float(len(events)),
                                   "scenario_targeted_shelter_ids": "", "scenario_expected_overrun_minute": ""}
                            if stressed:
                                minute = 34 if cell.endswith("m05") else 64
                                targets = [0] if cell.startswith("t1") else [0, 1]
                                for target in targets:
                                    if any(e["shelter_id"] == target for e in events):
                                        continue
                                    if broken and cell == "t1_m05" and tape == 3 and city == "spokane_wa":
                                        continue
                                    events.append({"shelter_id": target, "time": minute, "danger_level": 4,
                                                   "forced_reevacuated_population": 60})
                                row.update({"scenario_targeted_shelter_ids": json.dumps(targets),
                                            "scenario_expected_overrun_minute": float(minute),
                                            "scenario_released_population_at_targets": 280 * len(targets),
                                            "scenario_other_overruns": 0})
                            row["shelter_overrun_events"] = json.dumps(events)
                            rows.append(row)
        return pd.DataFrame(rows)

    def _run(self, frame, *extra):
        import site_loss_results as R

        campaign = Path(tempfile.mkdtemp(prefix="site_loss_results_"))
        target = campaign / "panels" / "site_loss"
        target.mkdir(parents=True)
        frame.to_csv(target / "evaluation_episode_summary.csv", index=False)
        with mock.patch("builtins.print"):
            code = R.main(["--campaign", str(campaign), *extra])
        return code, campaign / "site_loss_results"

    def test_penalties_contrasts_and_treatment_delivery(self):
        import pandas as pd

        code, out = self._run(self._frame(), "--check")
        self.assertEqual(code, 0)
        penalty = pd.read_csv(out / "site_loss_penalty.csv")
        macro = penalty[(penalty["sample"] == "all_tapes") & (penalty.metric == "objective_normalized_cost")
                        & (penalty.scope == "macro")]
        value = lambda p, c: float(macro[(macro.policy == p) & (macro.cell == c)].penalty_mean.iloc[0])  # noqa: E731
        self.assertAlmostEqual(value("static_greedy", "t1_m05"), 0.040, places=3)
        self.assertAlmostEqual(value("rl", "t2_m35"), 0.020, places=3)
        contrasts = pd.read_csv(out / "penalty_contrasts.csv")
        adaptivity = contrasts[(contrasts["sample"] == "all_tapes")
                               & (contrasts.metric == "objective_normalized_cost") & (contrasts.scope == "macro")
                               & (contrasts.left == "static_greedy") & (contrasts.right == "rl")
                               & (contrasts.cell == "t1_m05")]
        self.assertAlmostEqual(float(adaptivity.difference_mean.iloc[0]), 0.030, places=3)
        self.assertTrue((out / "site_loss_penalty.png").is_file())
        check = pd.read_csv(out / "manipulation_check.csv")
        stressed = check[check.cell != "base"]
        self.assertTrue((stressed.share_missed == 0).all())
        # State College tape 1 loses initial shelter 1 at minute 30 in every episode:
        # pre-empted in both two-site cells, a non-target overrun in the one-site cells.
        tapes = pd.read_csv(out / "tape_delivery.csv")
        preempted = tapes[tapes.status == "pre-empted"]
        self.assertEqual(sorted(preempted.cell), ["t2_m05", "t2_m35"])
        self.assertTrue((preempted.city_id == "state_college_pa").all() and (preempted.tape == 1).all()
                        and (preempted.target == 1).all() and (preempted.failure_minute == 30).all())
        t2 = stressed[stressed.cell == "t2_m05"].set_index("policy")
        self.assertAlmostEqual(float(t2.loc["objective_greedy", "incremental_sites_lost_mean"]), 2 - 1 / 40)
        full = penalty[(penalty["sample"] == "full_dose") & (penalty.metric == "objective_normalized_cost")
                       & (penalty.scope == "macro") & (penalty.policy == "objective_greedy")].set_index("cell")
        self.assertEqual(int(full.loc["t2_m05", "tapes"]), 39)
        self.assertEqual(int(full.loc["t1_m05", "tapes"]), 40)

    def test_check_fails_when_a_target_survives(self):
        code, out = self._run(self._frame(broken=True), "--check")
        self.assertEqual(code, 1)
        import pandas as pd

        missed = pd.read_csv(out / "target_delivery.csv").query("status == 'missed'")
        self.assertEqual(len(missed), 7)  # 5 policies, RL with 3 seeds, on one tape
        self.assertTrue((missed.cell == "t1_m05").all())

    def test_early_failure_without_the_same_base_failure_is_missed(self):
        # One policy's base episode lacks the natural overrun: its early target failure
        # is no longer explained by the tape, and the tape becomes inconsistent.
        import pandas as pd

        code, out = self._run(self._frame(inconsistent=True), "--check")
        self.assertEqual(code, 1)
        tapes = pd.read_csv(out / "tape_delivery.csv")
        self.assertEqual(sorted(tapes[tapes.status == "inconsistent"].cell), ["t2_m05", "t2_m35"])


class SiteLossIdentityGateTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict]):
        columns = sorted({k for r in rows for k in r})
        import csv

        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_base_rows_must_match_the_reference_except_paths(self):
        import site_loss_identity as G

        tmp = Path(tempfile.mkdtemp(prefix="site_loss_identity_"))
        provenance = lambda root: json.dumps({"graph_cache_path": f"{root}/cache/g.graphml",  # noqa: E731
                                              "graph_cache_sha256": "abc", "consolidated_graph_cache_sha256": "def"})
        reference, site = [], []
        for rep in range(1, 4):
            common = {"deployment_strategy": "objective_greedy", "policy_replication": 0, "city_id": "spokane_wa",
                      "city_scenario_replication": rep, "objective_normalized_cost": 0.3 + rep / 100}
            reference.append({**common, "map_provenance": provenance("/main"), "panel_wall_s": 9})
            site.append({**common, "map_provenance": provenance("/worktree"), "label_cell": "base",
                         "panel_wall_s": 11, "scenario_hook": ""})
            site.append({**common, "objective_normalized_cost": 0.9, "label_cell": "t1_m05",
                         "map_provenance": provenance("/worktree")})
        self._write(tmp / "ref.csv", reference)
        self._write(tmp / "site.csv", site)
        with mock.patch("builtins.print"):
            self.assertEqual(G.main(["--site-loss", str(tmp / "site.csv"), "--reference", str(tmp / "ref.csv"),
                                     "--min-common", "3"]), 0)
        site[0]["objective_normalized_cost"] = 0.31000001
        self._write(tmp / "site.csv", site)
        with mock.patch("builtins.print"):
            self.assertEqual(G.main(["--site-loss", str(tmp / "site.csv"), "--reference", str(tmp / "ref.csv"),
                                     "--min-common", "3"]), 1)
        shutil.rmtree(tmp, ignore_errors=True)
