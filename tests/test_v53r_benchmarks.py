import unittest

import numpy as np

from DecisionInterface import (
    BeliefRolloutPolicy,
    DeadlineGreedyHeuristic,
    HindsightRolloutPolicy,
    ObjectiveMatchedGreedyHeuristic,
)
from tests.test_rl_framework import make_observation


class V53RBenchmarkTests(unittest.TestCase):
    def test_deadline_negative_infinity_equals_objective_greedy(self):
        observation = make_observation()
        values = np.asarray((0.4, 0.8, 0.6, -np.inf))
        objective = ObjectiveMatchedGreedyHeuristic().select(
            observation, candidate_values=values
        )
        deadline = DeadlineGreedyHeuristic(-np.inf).select(
            observation, candidate_values=values
        )
        self.assertEqual(objective.action_index, deadline.action_index)

    def test_rollout_policies_minimize_the_supplied_complete_cost(self):
        observation = make_observation()
        belief = BeliefRolloutPolicy().select(
            observation, candidate_costs=(9.0, 7.0, 8.0, np.inf)
        )
        hindsight = HindsightRolloutPolicy().select(
            observation, candidate_costs=(8.0, 7.0, 5.0, np.inf)
        )
        self.assertEqual(belief.action_index, 1)
        self.assertEqual(hindsight.action_index, 2)
        self.assertLessEqual(5.0, 7.0)

    def test_hindsight_belief_and_myopic_order_in_expectation_on_eight_tapes(self):
        observation = make_observation()
        myopic_costs = []
        belief_costs = []
        hindsight_costs = []
        for tape in range(8):
            myopic = np.asarray((9.0, 7.8 + 0.1 * tape, 8.2, np.inf))
            belief = np.asarray((8.4, 7.2 + 0.05 * tape, 7.7, np.inf))
            hindsight = np.asarray((7.8, 7.1, 6.5 + 0.03 * tape, np.inf))
            myopic_costs.append(float(np.min(myopic[:3])))
            belief_action = BeliefRolloutPolicy().select(
                observation, candidate_costs=belief
            ).action_index
            hindsight_action = HindsightRolloutPolicy().select(
                observation, candidate_costs=hindsight
            ).action_index
            belief_costs.append(float(belief[belief_action]))
            hindsight_costs.append(float(hindsight[hindsight_action]))
        self.assertLessEqual(np.mean(hindsight_costs), np.mean(belief_costs))
        self.assertLessEqual(np.mean(belief_costs), np.mean(myopic_costs))


if __name__ == "__main__":
    unittest.main()
