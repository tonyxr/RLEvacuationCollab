"""v65.1 results script: every exhibit renders from a synthetic campaign.

The campaign is generated from the production panel specs (arms, cities,
replications, labels), so a renamed arm or label in a spec breaks this test
instead of the final figure run.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paper_results as PR  # noqa: E402

PANELS = ROOT / "config" / "panels"
EFFECT = {"rl": -0.03, "rl_early_search": -0.035, "belief_rollout": -0.04, "objective_greedy": 0.0,
          "regime_oracle_greedy": -0.01, "hindsight_rollout": -0.06, "static_greedy": 0.02,
          "initial_only": 0.03, "random": 0.08}


def _parse_reps(value):
    if isinstance(value, int):
        return list(range(1, value + 1))
    return list(range(int(value[0]), int(value[1]) + 1))


def _progress(path: Path, population: int, strength: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep", "panic_eligible_first_exposures", "evacuated", "current_sheltered_population"])
        for t in range(1, 181):
            writer.writerow([t, int(population * 0.3 * (1 - np.exp(-t / (40 + 100 * strength)))),
                             int(population * 0.2 * t / 180), int(population * 0.7 * (1 - np.exp(-t / 30)))])


def make_campaign(root: Path, panel_ids) -> Path:
    campaign = root / "campaign"
    rng = np.random.default_rng(1)
    for panel_id in panel_ids:
        spec = json.loads((PANELS / f"{panel_id}.json").read_text())
        rows = []
        for city in spec["cities"]:
            for rep in _parse_reps(spec["replications"]):
                tape = 0.4 + 0.1 * rng.standard_normal()  # common to every arm: momentum
                for arm in spec["arms"]:
                    if "replications" in arm and rep not in _parse_reps(arm["replications"]):
                        continue
                    strategy = arm.get("strategy", arm["arm"])
                    for policy in arm.get("policies", [0]):
                        artifacts = root / "store" / f"{panel_id}_{city}_{rep}_{arm['arm']}_{policy}"
                        _progress(artifacts / strategy / f"rep_{rep:03d}_local" / "progress.csv", 2500,
                                  -EFFECT.get(strategy, 0.0) * 10)
                        effect = EFFECT.get(strategy, 0.01)
                        labels = arm.get("labels", {})
                        if "interval_min" in labels:
                            effect *= labels["interval_min"] / 15.0
                        cost = tape + effect + 0.004 * rng.standard_normal()
                        row = {
                            "arm": arm["arm"], "deployment_strategy": strategy, "policy_replication": policy,
                            "city_id": city, "city_scenario_replication": rep, PR.COST: cost,
                            "episode_artifacts_dir": str(artifacts), "episode_wall_time_s": 60 + 300 * (strategy in PR.RL_ARMS),
                            "unfinished": 300 + 1000 * effect, "active_panicked_at_horizon": 100,
                            "expected_casualties": 5 + 50 * effect, "panic_eligible_first_exposures": 700,
                            "clearance_t90_minutes": 90, "clearance_time_minutes": 150, "overrun_shelters": int(strategy == "objective_greedy_legacy"),
                            "forced_reevacuated": 10, "deferred_decisions": 0, "mean_pedestrian_distance_m": 800,
                            "congested_person_minute_share": 0.1, "unassigned_rational_after_final_installation": 0,
                            "mean_end_to_end_deployment_latency_ms": 5 if strategy != "belief_rollout" else 300000,
                            "initial_population": 2500, "free_flow_speed_m_per_second": 1.34,
                            "captured_planner_matrices": "", "captured_link_density": "",
                        }
                        for key, value in {**spec.get("labels", {}), **labels}.items():
                            row[f"label_{key}"] = value
                        if strategy == "belief_rollout":
                            matrices = [{"strategy": "belief_rollout", "decision_index": d, "cells": [1, 2, 3, None],
                                         "matrix": (tape + rng.normal(0, 0.05, (8, 1)) + rng.normal(0, 0.01, (1, 4))).tolist()}
                                        for d in range(8)]
                            (artifacts / "planner_matrices.json").write_text(json.dumps(matrices))
                            row["captured_planner_matrices"] = str(artifacts / "planner_matrices.json")
                        if spec.get("capture", {}).get("link_density"):
                            counts = np.histogram(rng.gamma(1.2, 0.4, 5000), bins=PR_EDGES)[0].tolist()
                            (artifacts / "link_density.json").write_text(json.dumps({
                                "density_edges": PR_EDGES, "person_substeps": counts, "link_substeps": counts,
                                "jam_density": 5.4, "shape": 1.913, "minimum_speed_ratio": 0.05,
                                "effective_width_m": 3.0, "substeps": 100}))
                            row["captured_link_density"] = str(artifacts / "link_density.json")
                        if spec["panel_id"] == "plugin_check":
                            row[PR.COST] = round(tape, 6)
                        rows.append(row)
        directory = campaign / "panels" / spec["panel_id"]
        directory.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with (directory / "evaluation_episode_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (directory / "panel_manifest.json").write_text(json.dumps({"tuned_values": [
            {"tuning_panel": "tune_deadline_main", "parameter": "deadlineGreedyThreshold", "selected_value": 0.4}]}))
    return campaign


PR_EDGES = [round(0.1 * i, 2) for i in range(61)]


class PaperResultsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="paper_results_"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_every_exhibit_renders_from_the_production_specs(self):
        campaign = make_campaign(self.root, ["main", "gate1_calm", "gate1_compound", "capacity_sweep",
                                             "kappa_sweep", "zero_shot", "learning_curve", "sim_checks",
                                             "plugin_check", "g2_seed1"])
        (self.root / "shard1").mkdir()
        (self.root / "shard1" / "experiment_manifest.json").write_text(json.dumps({
            "launch_seed": 6500, "cell_observation_features": ["a", "b"], "candidate_observation_features": ["c"],
            "global_observation_features": ["d"], "momentum_observation_features": ["e"],
            "training_curriculum": {"curriculum_id": "x", "source_sha256": "y",
                                    "stages": [{"variants": [{"overrides": {"pedVol": 2500}}]}]},
            "policy_dynamics_source_sha256": {"Core.py": "z"}, "effective_overrides_by_city": {"state_college_pa": {}}}))
        shard = self.root / "shard1" / "policies" / "policy_001"
        shard.mkdir(parents=True, exist_ok=True)
        with (shard / "learner_diagnostics.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode", "optimizer_updated", "rollout_pi_actor_enabled",
                             "rollout_pi_weighted_pairwise_accuracy_before", "rollout_pi_weighted_pairwise_accuracy_after",
                             "rollout_pi_branched_top1_before", "rollout_pi_target_kl_after"])
            for episode in range(1, 41):
                writer.writerow([episode, 1, 1, 0.5 + episode / 100, 0.7, 0.3, 0.05])
        (self.root / "city_build_profile.json").write_text(json.dumps({"cities": [
            {"city_id": "state_college_pa", "graph_nodes": 100, "graph_edges": 200, "build_s": 12.0, "peak_rss_mb": 900.0}]}))
        out = self.root / "results"
        code = PR.main(["--campaign", str(campaign), "--training", str(self.root / "shard1"),
                        "--out", str(out), "--city-profile", str(self.root / "city_build_profile.json"), "--check"])
        status = json.loads((out / "exhibit_status.json").read_text())
        failed = [s for s in status["status"] if s["status"] != "ok"]
        self.assertEqual(failed, [], json.dumps(failed, indent=1)[:4000])
        self.assertEqual(code, 0)
        hypotheses = json.loads((out / "hypotheses_main.json").read_text())
        self.assertGreater(hypotheses["H1"]["estimate"], 0.02)
        self.assertTrue(hypotheses["H1"]["pass"])
        for name in ("fig5_forest.png", "fig6_mechanism.png", "fig7_capacity_sweep.png", "fig8_gate1.png",
                     "fig9_momentum_variance.png", "fig10_learning_curves.png", "fig13_zero_shot.png",
                     "fig3_fundamental_diagram.png", "fig4_cumulative_safe.png", "fig11_planner_share_vs_latency.png",
                     "fig12_severity_vs_gain.png"):
            self.assertTrue((out / name).is_file(), name)

    def test_audit_flags_a_missing_column(self):
        campaign = make_campaign(self.root, ["main"])
        path = campaign / "panels" / "main" / "evaluation_episode_summary.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.pop("overrun_shelters")
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        audit = PR.audit_columns(PR.Inputs(campaign, [], self.root / "o", None))
        main = next(a for a in audit if a["panel"] == "main")
        self.assertEqual(main["status"], "FAIL")
        self.assertIn("overrun_shelters", main["missing"])

    def test_paired_statistics(self):
        rows = []
        for rep in range(1, 11):
            rows.append({"arm": "g", "city_id": "a", "city_scenario_replication": rep, "policy_replication": 0, PR.COST: 1.0 + rep})
            for policy in (1, 2):
                rows.append({"arm": "rl", "city_id": "a", "city_scenario_replication": rep,
                             "policy_replication": policy, PR.COST: 0.9 + rep + 0.02 * policy})
        diffs = PR.paired(rows, "g", "rl")
        self.assertTrue(np.allclose(diffs["a"], 0.07))
        self.assertAlmostEqual(PR.per_seed_effects(rows, "g", "rl")[1], 0.08)
        self.assertAlmostEqual(PR.per_seed_effects(rows, "g", "rl")[2], 0.06)
        low, high = PR.seed_tape_bootstrap(rows, "g", "rl", draws=500)
        self.assertLessEqual(low, 0.07 + 1e-9)
        self.assertGreaterEqual(high, 0.07 - 1e-9)
        self.assertAlmostEqual(PR.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(PR.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertAlmostEqual(PR.spearman([1, 2, 2, 3], [1, 3, 2, 4]), 0.9486832980505139, places=9)
        adjusted = PR.holm({"x": 0.01, "y": 0.04, "z": 0.03})
        self.assertAlmostEqual(adjusted["x"], 0.03)
        self.assertAlmostEqual(adjusted["z"], 0.06)
        self.assertAlmostEqual(adjusted["y"], 0.06)


if __name__ == "__main__":
    unittest.main()
