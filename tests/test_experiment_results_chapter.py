import csv
import json
from pathlib import Path
import tempfile
import unittest

from generate_experiment_results_chapter import (
    _simpson_status,
    audit_core_launch,
    equal_city_strategy_summary,
    fallacy_scan,
)
from reproduce_multicity_sample import EXACT_FIELDS, FLOAT_FIELDS, compare_rows


class ExperimentResultsChapterTests(unittest.TestCase):
    def test_equal_city_summary_does_not_weight_large_city_more(self):
        rows = []
        for city, population, safe in (
            ("small", 100, 100),
            ("large", 1000, 0),
        ):
            rows.append(
                {
                    "city_id": city,
                    "deployment_strategy": "rl",
                    "initial_population": population,
                    "objective_episode_return": safe / population,
                    "safe_completed": safe,
                    "casualty": 0,
                    "unfinished": population - safe,
                    "restricted_mean_time_to_safety": 30,
                    "normalized_risk_weighted_person_time": 0.5,
                }
            )
        summary = equal_city_strategy_summary(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["safe"], 0.5)
        self.assertAlmostEqual(summary[0]["objective"], 0.5)

    def test_fallacy_scan_has_all_eleven_registered_types(self):
        paired = [
            {
                "scope": "macro_all_cities",
                "city_id": "ALL",
                "metric": "episode_return",
                "mean_rl_improvement": 0.1,
            },
            {
                "scope": "city",
                "city_id": "a",
                "metric": "episode_return",
                "mean_rl_improvement": 0.2,
            },
            {
                "scope": "city",
                "city_id": "b",
                "metric": "episode_return",
                "mean_rl_improvement": -0.1,
            },
        ]
        scan = fallacy_scan(paired)
        self.assertEqual(len(scan), 11)
        self.assertEqual(_simpson_status(paired)[0], "CAUTION")

    def test_reproduction_comparison_ignores_timing_and_checks_scientific_fields(self):
        original = {field: "7" for field in EXACT_FIELDS}
        rerun = {field: 7 for field in EXACT_FIELDS}
        original.update({field: "0.25" for field in FLOAT_FIELDS})
        rerun.update({field: 0.25 + 1e-12 for field in FLOAT_FIELDS})
        original["episode_wall_time_s"] = "20"
        rerun["episode_wall_time_s"] = 200
        comparisons = compare_rows(original, rerun, absolute_tolerance=1e-10)
        self.assertEqual(len(comparisons), len(EXACT_FIELDS) + len(FLOAT_FIELDS))
        self.assertTrue(all(row["status"] == "MATCH" for row in comparisons))
        self.assertNotIn("episode_wall_time_s", {row["field"] for row in comparisons})

    def test_core_audit_uses_city_ids_and_strategy_multiplicity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "status": "complete",
                "city_ids": ["a"],
                "eval_replications_per_city": 1,
                "policy_replicates": 1,
                "train_episodes_per_city": 1,
                "strategies": ["rl", "heuristic"],
            }
            (root / "experiment_manifest.json").write_text(json.dumps(manifest))
            (root / "training_convergence_diagnostics.json").write_text(
                json.dumps({"all_policies_converged": True, "policies": []})
            )
            with (root / "training_episode_summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["replication"])
                writer.writeheader()
                writer.writerow({"replication": 1})
            evaluation = []
            for strategy in ("rl", "heuristic"):
                evaluation.append(
                    {
                        "deployment_strategy": strategy,
                        "objective_episode_return": -0.5,
                        "safe_completed": 4,
                        "casualty": 1,
                        "unfinished": 5,
                        "initial_population": 10,
                    }
                )
            with (root / "evaluation_episode_summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(evaluation[0]))
                writer.writeheader()
                writer.writerows(evaluation)
            with (root / "paired_comparison_by_city.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scope", "metric"])
                writer.writeheader()
                writer.writerow({"scope": "macro_all_cities", "metric": "episode_return"})
            result = audit_core_launch(root)
            self.assertEqual(result["expected_training"], 1)
            self.assertEqual(result["expected_evaluation"], 2)
            self.assertTrue(all(result["checks"].values()))


if __name__ == "__main__":
    unittest.main()
