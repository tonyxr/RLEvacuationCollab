import unittest

from cohort_fidelity_backtest import summarize_fidelity


def _row(replication, group_size, episode_return, safe, casualty, risk):
    return {
        "scenario_replication": replication,
        "maximum_persons_per_agent": group_size,
        "initial_population": 100,
        "horizon_transitions": 10,
        "objective_episode_return": episode_return,
        "safe_completed": safe,
        "casualty": casualty,
        "objective_risk_weighted_person_time": risk,
    }


class CohortFidelityAnalysisTests(unittest.TestCase):
    def test_identical_paired_metrics_pass(self):
        rows = []
        for replication in (1, 2):
            rows.extend(
                (
                    _row(replication, 1, 0.5, 50, 2, 300),
                    _row(replication, 20, 0.5, 50, 2, 300),
                )
            )
        result = summarize_fidelity(rows, 20)
        self.assertTrue(result["passes_training_surrogate_gate"])

    def test_large_return_bias_fails(self):
        rows = [
            _row(1, 1, 0.5, 50, 2, 300),
            _row(1, 20, 0.0, 50, 2, 300),
        ]
        result = summarize_fidelity(rows, 20)
        self.assertFalse(result["passes_training_surrogate_gate"])


if __name__ == "__main__":
    unittest.main()
