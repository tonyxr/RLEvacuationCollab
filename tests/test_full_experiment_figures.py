import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ExperimentSuite import load_experiment_suite
from generate_full_experiment_figures import (
    _plot_ablation,
    _plot_regime_heatmap,
    _plot_robustness,
    _plot_scalability,
    _plot_population_candidate_stress,
    _plot_transfer,
    balanced_training_blocks,
    hierarchical_bootstrap,
    main,
    paired_difference_matrices,
    training_update_averages,
    validate_result_table,
)


class ExperimentSuiteTests(unittest.TestCase):
    def test_versioned_suite_has_expected_factorial(self):
        suite = load_experiment_suite()
        self.assertEqual(suite.policy_seeds, 8)
        self.assertEqual(suite.train_episodes_per_city, 120)
        self.assertEqual(len(suite.factor_cells), 18)
        self.assertEqual(suite.evaluation_scenarios_per_city, 90)
        self.assertEqual([f"E{index}" for index in range(7)], sorted(suite.experiments))
        self.assertEqual(13, len(suite.figure_families))
        self.assertEqual(
            (10000, 20000, 30000, 40000, 50000), suite.population_levels
        )
        self.assertEqual((25, 50, 75, 100, 125), suite.shelter_candidate_levels)
        self.assertEqual(60, suite.scale_horizon_timesteps)
        self.assertEqual(2, suite.scale_shelter_action_interval)
        self.assertTrue(
            {
                "risk_reduction",
                "heuristic",
                "hazard_weighted",
                "accessibility_deficit",
                "random",
                "initial_only",
            }.issubset(suite.experiments["E2"]["design"]["policies"])
        )

    def test_table_validation_fails_closed_on_missing_columns(self):
        suite = load_experiment_suite()
        spec = suite.output_tables["evaluation_summary"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.csv"
            path.write_text("city_id,episode_return\na,1.0\n", encoding="utf-8")
            status = validate_result_table(path, spec)
        self.assertFalse(status["valid"])
        self.assertIn("deployment_strategy", status["missing_columns"])


class StatisticalAggregationTests(unittest.TestCase):
    @staticmethod
    def _training_row(policy, replication, city, rank, value):
        return {
            "policy_replication": policy,
            "replication": replication,
            "city_id": city,
            "city_scale_rank": rank,
            "episode_return": value,
            "safe_completion_reward": value + 1.0,
            "casualty_penalty": -0.1,
            "risk_time_penalty": -0.9,
            "entropy": 1.5,
            "heuristic_agreement_rate": 0.5,
        }

    def test_training_blocks_keep_policy_seeds_separate(self):
        rows = []
        for policy in (1, 2):
            rows.extend(
                [
                    self._training_row(policy, 1, "small", 1, policy + 0.1),
                    self._training_row(policy, 2, "large", 2, policy + 0.3),
                    self._training_row(policy, 3, "large", 2, policy + 0.5),
                    self._training_row(policy, 4, "small", 1, policy + 0.7),
                ]
            )
        blocks = balanced_training_blocks(rows)
        self.assertEqual(4, len(blocks))
        self.assertEqual({1, 2}, {row["policy_replication"] for row in blocks})
        self.assertAlmostEqual(1.2, blocks[0]["episode_return"])
        self.assertAlmostEqual(1.6, blocks[1]["episode_return"])
        self.assertAlmostEqual(2.2, blocks[2]["episode_return"])

    def test_two_way_bootstrap_preserves_constant_paired_effect(self):
        rows = []
        for city_rank, city in enumerate(("small", "large"), start=1):
            for scenario in range(1, 5):
                common = {
                    "city_id": city,
                    "city_scale_rank": city_rank,
                    "city_scenario_replication": scenario,
                    "scenario_seed": 1000 * city_rank + scenario,
                    "initial_population": 100,
                    "safe_completed": 80,
                    "casualty": 1,
                    "unfinished": 19,
                    "restricted_mean_time_to_safety": 10,
                    "normalized_risk_weighted_person_time": 0.2,
                }
                rows.append(
                    {
                        **common,
                        "deployment_strategy": "heuristic",
                        "policy_replication": 0,
                        "episode_return": scenario / 10,
                    }
                )
                for policy in (1, 2, 3):
                    rows.append(
                        {
                            **common,
                            "deployment_strategy": "rl",
                            "policy_replication": policy,
                            "episode_return": scenario / 10 + 0.25,
                        }
                    )
        matrices = paired_difference_matrices(
            rows, "rl", "heuristic", "episode_return"
        )
        self.assertEqual((3, 4), matrices["small"].shape)
        estimate, lower, upper = hierarchical_bootstrap(
            matrices, seed=123, draws=500
        )
        self.assertAlmostEqual(0.25, estimate)
        self.assertAlmostEqual(0.25, lower)
        self.assertAlmostEqual(0.25, upper)

    def test_ppo_update_outcomes_average_every_contributing_episode(self):
        training = [
            {
                **self._training_row(1, episode, "small", 1, float(episode)),
                "decisions": 2,
            }
            for episode in range(1, 5)
        ]
        diagnostics = [
            {
                "policy_replication": 1,
                "episode": 2,
                "optimizer_updates": 8,
                "optimizer_updated": 1,
                "epochs_completed": 8,
                "approximate_kl": 0.01,
                "clip_fraction": 0.1,
                "explained_variance": 0.2,
                "policy_loss": -0.01,
                "value_loss": 0.3,
                "gradient_norm": 0.4,
                "update_entropy": 1.5,
            },
            {
                "policy_replication": 1,
                "episode": 4,
                "optimizer_updates": 16,
                "optimizer_updated": 1,
                "epochs_completed": 8,
                "approximate_kl": 0.02,
                "clip_fraction": 0.2,
                "explained_variance": 0.4,
                "policy_loss": -0.02,
                "value_loss": 0.2,
                "gradient_norm": 0.3,
                "update_entropy": 1.2,
            },
        ]
        updates = training_update_averages(training, diagnostics)
        self.assertEqual(2, len(updates))
        self.assertEqual(2, updates[0]["contributing_episodes"])
        self.assertAlmostEqual(1.5, updates[0]["mean_episode_return"])
        self.assertAlmostEqual(3.5, updates[1]["mean_episode_return"])

    def test_ppo_update_alignment_supports_continuation_episode_offset(self):
        training = [
            {
                **self._training_row(1, episode, "small", 1, float(episode)),
                "decisions": 2,
            }
            for episode in range(1, 5)
        ]
        diagnostics = []
        for source_episode in range(101, 105):
            diagnostics.append(
                {
                    "policy_replication": 1,
                    "episode": source_episode,
                    "optimizer_updates": 8 if source_episode < 104 else 16,
                    "optimizer_updated": int(source_episode in (102, 104)),
                    "epochs_completed": 8 if source_episode in (102, 104) else 0,
                    "approximate_kl": 0.01,
                    "clip_fraction": 0.1,
                    "explained_variance": 0.2,
                    "policy_loss": -0.01,
                    "value_loss": 0.3,
                    "gradient_norm": 0.4,
                    "update_entropy": 1.5,
                }
            )

        updates = training_update_averages(training, diagnostics)

        self.assertEqual(2, len(updates))
        self.assertEqual(100, updates[0]["episode_index_offset"])
        self.assertEqual(102, updates[0]["source_episode_end"])
        self.assertEqual(2, updates[0]["episode_end"])
        self.assertAlmostEqual(1.5, updates[0]["mean_episode_return"])
        self.assertAlmostEqual(3.5, updates[1]["mean_episode_return"])


class FigureRenderingTests(unittest.TestCase):
    @staticmethod
    def _quick_save(fig, output_dir, stem):
        path = Path(output_dir) / f"{stem}.png"
        fig.savefig(path, dpi=45, facecolor="white")
        return [path]

    @staticmethod
    def _base(city, rank, scenario, strategy, policy, value):
        return {
            "policy_replication": policy,
            "city_id": city,
            "city_scale_rank": rank,
            "city_scenario_replication": scenario,
            "scenario_seed": 10000 * rank + scenario,
            "deployment_strategy": strategy,
            "episode_return": value,
            "objective_episode_return": value,
            "initial_population": 100,
            "safe_completed": 70 + int(10 * value),
            "casualty": max(0, 3 - int(2 * value)),
            "unfinished": 27,
            "restricted_mean_time_to_safety": 12 - value,
            "normalized_risk_weighted_person_time": 0.5 - value / 10,
        }

    def _paired_rows(self, extra=None):
        rows = []
        for rank, city in enumerate(("small", "large"), start=1):
            for scenario in (1, 2):
                common_extra = dict(extra or {})
                rows.append(
                    {
                        **self._base(city, rank, scenario, "heuristic", 0, 0.1 * scenario),
                        **common_extra,
                    }
                )
                for policy in (1, 2):
                    rows.append(
                        {
                            **self._base(city, rank, scenario, "rl", policy, 0.2 + 0.1 * scenario),
                            **common_extra,
                        }
                    )
        return rows

    def test_optional_figure_families_render_from_declared_schemas(self):
        suite = load_experiment_suite()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with mock.patch(
                "generate_full_experiment_figures._save_figure",
                side_effect=self._quick_save,
            ):
                regime = []
                scenario_offset = 0
                for cell in suite.factor_cells:
                    cell_rows = self._paired_rows(cell)
                    for row in cell_rows:
                        row["city_scenario_replication"] += scenario_offset
                        row["scenario_seed"] += scenario_offset
                    regime.extend(cell_rows)
                    scenario_offset += 10
                paths, _ = _plot_regime_heatmap(
                    regime, suite, output, seed=1, draws=100
                )
                self.assertTrue(all(path.exists() for path in paths))

                scalability = []
                for method in ("hierarchical_region", "flat_candidate"):
                    for candidates in (64, 256):
                        for replication in range(3):
                            scalability.append(
                                {
                                    "city_id": "small",
                                    "scenario_seed": replication,
                                    "method": method,
                                    "graph_nodes": 1000,
                                    "candidate_count": candidates,
                                    "action_count": 64 if method == "hierarchical_region" else candidates,
                                    "decision_latency_ms": (1 if method == "hierarchical_region" else 2) * candidates,
                                    "episode_return": 0.2,
                                }
                            )
                paths, _ = _plot_scalability(
                    scalability, output, seed=2, draws=100
                )
                self.assertTrue(all(path.exists() for path in paths))

                ablation = []
                for rank, city in enumerate(("small", "large"), start=1):
                    for scenario in (1, 2):
                        for policy in (1, 2):
                            for variant, value in (
                                ("full_model", 0.5),
                                ("no_risk_person_time", 0.3),
                            ):
                                row = self._base(city, rank, scenario, "rl", policy, value)
                                row["ablation"] = variant
                                ablation.append(row)
                paths, _ = _plot_ablation(
                    ablation, suite, output, seed=3, draws=100
                )
                self.assertTrue(all(path.exists() for path in paths))

                transfer = []
                for row in self._paired_rows():
                    row["held_out_city"] = row.pop("city_id")
                    transfer.append(row)
                    if row["deployment_strategy"] == "rl":
                        pooled = dict(row)
                        pooled["deployment_strategy"] = "pooled_all_cities"
                        transfer.append(pooled)
                        row["deployment_strategy"] = "leave_one_city_out"
                paths, _ = _plot_transfer(transfer, output, seed=4, draws=100)
                self.assertTrue(all(path.exists() for path in paths))

                robust = []
                for perturbation, level in (("demand_shift", "low"), ("demand_shift", "high")):
                    robust.extend(self._paired_rows({"perturbation": perturbation, "level": level}))
                paths, _ = _plot_robustness(robust, output, seed=5, draws=100)
                self.assertTrue(all(path.exists() for path in paths))

                scale_rows = self._paired_rows(
                    {
                        "population_level": 10000,
                        "shelter_candidate_level": 25,
                        "horizon_timesteps": 60,
                        "simulation_runtime_s": 1.0,
                    }
                )
                paths, _ = _plot_population_candidate_stress(
                    scale_rows, suite, output, seed=6, draws=100
                )
                self.assertTrue(all(path.exists() for path in paths))

    def test_partial_main_writes_readiness_instead_of_fabricating_missing_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            launch = Path(temporary) / "launch"
            output = Path(temporary) / "figures"
            launch.mkdir()
            training_rows = []
            for policy in (1, 2):
                training_rows.extend(
                    [
                        StatisticalAggregationTests._training_row(policy, 1, "small", 1, 0.1),
                        StatisticalAggregationTests._training_row(policy, 2, "large", 2, 0.2),
                    ]
                )
            evaluation_rows = self._paired_rows()

            def write(path, rows):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
                    writer.writeheader()
                    writer.writerows(rows)

            write(launch / "training_episode_summary.csv", training_rows)
            write(launch / "evaluation_episode_summary.csv", evaluation_rows)
            with mock.patch(
                "generate_full_experiment_figures._save_figure",
                side_effect=self._quick_save,
            ):
                result = main(
                    [
                        "--launch-dir",
                        str(launch),
                        "--output-dir",
                        str(output),
                        "--bootstrap-draws",
                        "100",
                        "--skip-maps",
                    ]
                )
            self.assertEqual(0, result)
            readiness = (output / "full_suite_readiness.json").read_text(encoding="utf-8")
            self.assertIn('"complete": false', readiness)
            self.assertIn("Missing valid regime_evaluation table", readiness)
            self.assertTrue((output / "03_primary_policy_performance.png").exists())


if __name__ == "__main__":
    unittest.main()
