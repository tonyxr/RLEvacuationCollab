"""Regression coverage for exact, production-feasible identity checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.identity_check import (
    _same,
    checkpoint_tensor_differences,
    normalize_preregistered_training_metadata,
)


class IdentityCheckTests(unittest.TestCase):
    def test_only_preregistered_metadata_and_local_paths_are_normalized(self):
        reference = {
            "stage_overrides": "{'scientific': 7}",
            "map_provenance": "{'graph_cache_path': '/tmp/a.graphml', "
            "'consolidated_graph_cache_path': '/tmp/a_c.graphml', "
            "'graph_cache_sha256': 'abc', 'graph_nodes': 42}",
            "episode_return": "1.25",
        }
        candidate = {
            "stage_overrides": "{'scientific': 7, 'forecastMemoEnabled': True, "
            "'branchTaskChunking': 'contiguous'}",
            "map_provenance": "{'graph_cache_path': '/workspace/a.graphml', "
            "'consolidated_graph_cache_path': '/workspace/a_c.graphml', "
            "'graph_cache_sha256': 'abc', 'graph_nodes': 42}",
            "episode_return": "1.25",
        }
        left = normalize_preregistered_training_metadata(reference)
        right = normalize_preregistered_training_metadata(candidate)
        self.assertTrue(_same(left["stage_overrides"], right["stage_overrides"]))
        self.assertTrue(_same(left["map_provenance"], right["map_provenance"]))
        self.assertEqual(left["episode_return"], right["episode_return"])

        candidate["episode_return"] = "1.2500001"
        changed = normalize_preregistered_training_metadata(candidate)
        self.assertFalse(_same(left["episode_return"], changed["episode_return"]))

    def test_checkpoint_comparison_is_bitwise_and_ignores_non_tensor_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.pt"
            candidate = root / "candidate.pt"
            tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            torch.save({"policy": {"weight": tensor}, "note": "v64"}, reference)
            torch.save({"policy": {"weight": tensor.clone()}, "note": "v65"}, candidate)
            left, right, differences = checkpoint_tensor_differences(reference, candidate)
            self.assertEqual((left, right), (1, 1))
            self.assertEqual(differences, [])

            changed = tensor.clone()
            changed[0, 1] = torch.nextafter(changed[0, 1], torch.tensor(float("inf")))
            torch.save({"policy": {"weight": changed}}, candidate)
            _, _, differences = checkpoint_tensor_differences(reference, candidate)
            self.assertEqual(len(differences), 1)


if __name__ == "__main__":
    unittest.main()
