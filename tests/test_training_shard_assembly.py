import csv
import json
from pathlib import Path
import tempfile
import unittest

from assemble_training_shards import CONTRACT_FIELDS, validate_training_shards


class TrainingShardAssemblyTests(unittest.TestCase):
    def _write_shard(self, root: Path, policy_index: int, *, launch_seed=17) -> Path:
        shard = root / f"shard_{policy_index:03d}"
        policy_dir = shard / "policies" / f"policy_{policy_index:03d}"
        policy_dir.mkdir(parents=True)
        contract = {
            "schema_version": 1,
            "launch_seed": launch_seed,
            "city_profile_sha256": "city-hash",
            "city_ids": ["a", "b"],
            "shared_grid": [2, 2],
            "policy_replicates": 2,
            "eval_replications_per_city": 3,
            "ppo_rollout_episodes": 2,
            "learning_rate": 0.001,
            "effective_overrides_by_city": {"a": {}, "b": {}},
            "training_curriculum": {"source_sha256": "curriculum-hash"},
            "train_episodes_per_city": 1,
            "total_train_episodes_per_policy": 2,
            "training_city_schedule": ["a", "b"],
            "reward_equation": "r",
            "policy_design": {"pooled_across_cities": True},
            "cell_observation_features": ["demand"],
            "global_observation_features": ["time"],
        }
        self.assertEqual(set(CONTRACT_FIELDS), set(contract))
        manifest = {
            **contract,
            "status": "training_shard_complete",
            "started_utc": "2026-01-01T00:00:00+00:00",
            "training_policy_indices": [policy_index],
            "training_convergence": {
                "all_policies_converged": True,
                "definition": "test",
                "policies": [
                    {
                        "policy_replication": policy_index,
                        "converged": True,
                    }
                ],
            },
        }
        (shard / "experiment_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        with (shard / "training_episode_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["policy_replication", "replication"]
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"policy_replication": policy_index, "replication": 1},
                    {"policy_replication": policy_index, "replication": 2},
                ]
            )
        (policy_dir / "regional_policy.pt").write_bytes(
            f"checkpoint-{policy_index}".encode()
        )
        (policy_dir / "ppo_diagnostics.csv").write_text(
            "episode,optimizer_updated\n1,0\n2,1\n", encoding="utf-8"
        )
        return shard

    def test_complete_disjoint_shards_validate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shards = [self._write_shard(root, index) for index in (1, 2)]
            validated = validate_training_shards(shards)
        self.assertEqual(2, len(validated["records"]))
        self.assertEqual(
            [1, 2],
            [row["policy_replication"] for row in validated["convergence_rows"]],
        )

    def test_contract_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_shard(root, 1)
            second = self._write_shard(root, 2, launch_seed=18)
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                validate_training_shards([first, second])

    def test_incomplete_policy_coverage_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._write_shard(root, 1)
            with self.assertRaisesRegex(ValueError, "do not cover"):
                validate_training_shards([first])


if __name__ == "__main__":
    unittest.main()
