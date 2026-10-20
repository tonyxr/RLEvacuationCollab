import unittest
from dataclasses import replace

import torch
import numpy as np

from GNN import EvacPolicy, fit_gnn, grid_edge_index
from Core import Core
from RLBridge import RLBridge
from tests.test_rl_framework import make_observation


class V53RLearnerTests(unittest.TestCase):
    def test_policy_stage_reports_the_critic_it_actually_trains(self):
        core = Core("local")
        core.trainMode = True
        core.trainingStageId = "P"
        core.nmccPolicyImprovement = True
        effective = core._effective_configuration()
        self.assertTrue(effective["training_stage"]["train_shared_critic"])
        self.assertEqual(
            effective["training_stage"]["critic_target"],
            (
                "complete_episode_smdp_monte_carlo_return_to_go_with_"
                "realized_regime_baseline"
            ),
        )
        self.assertEqual(
            effective["counterfactual_branching"]["comparison_horizon"],
            "complete_simulation_to_episode_horizon",
        )

    def test_actor_cannot_see_realized_tape_but_critic_can(self):
        torch.manual_seed(53)
        policy = EvacPolicy(
            2, 2, 2, d_global=3, d_candidate=2, d_momentum=4,
            d_critic_exogenous=8,
        )
        with torch.no_grad():
            for parameter in policy.critic_trunk.parameters():
                parameter.uniform_(-0.2, 0.2)
            for parameter in policy.critic_heads.parameters():
                parameter.uniform_(-0.2, 0.2)
        common = dict(
            x_ped=torch.rand(4, 2), x_hazard=torch.rand(4, 2),
            x_infra=torch.rand(4, 2), x_global=torch.rand(3),
            edge_index=torch.as_tensor(grid_edge_index(2, 2)),
            candidate_cell_index=torch.arange(4),
            candidate_features=torch.rand(4, 2),
        )
        zero = fit_gnn(**common, critic_exogenous=torch.zeros(8))
        one = fit_gnn(**common, critic_exogenous=torch.ones(8))
        actor_zero = policy.forward_nmcc_actor_recurrent(zero)[0]
        actor_one = policy.forward_nmcc_actor_recurrent(one)[0]
        torch.testing.assert_close(actor_zero, actor_one)
        value_zero = policy.forward_nmcc_continuation_recurrent(zero)[0]
        value_one = policy.forward_nmcc_continuation_recurrent(one)[0]
        self.assertFalse(torch.allclose(value_zero, value_one))

    def test_forecast_ablation_removes_only_predictive_information(self):
        observation = make_observation(danger=(0.2, 0.4, 0.1, 0.0))
        observation.forecast_danger_by_cell[:] = (0.8, 0.7, 0.6, 0.5)
        observation = replace(
            observation,
            operational_shelter_capacity_by_cell=np.asarray(
                (5.0, 0.0, 0.0, 0.0), dtype=np.float32
            ),
            sheltered_population_by_cell=np.asarray(
                (2.0, 0.0, 0.0, 0.0), dtype=np.float32
            ),
            remaining_capacity_by_cell=np.asarray(
                (3.0, 0.0, 0.0, 0.0), dtype=np.float32
            ),
            forecast_overrun_probability_by_cell=np.asarray(
                (0.8, 0.0, 0.0, 0.0), dtype=np.float32
            ),
            forecast_next_shelter_overrun_probability_by_cell=np.asarray(
                (1.0, 0.0, 0.0, 0.0), dtype=np.float32
            ),
            forecast_overrun_lead_time_fraction_by_cell=np.asarray(
                (0.2, 1.0, 1.0, 1.0), dtype=np.float32
            ),
            forecast_reevacuation_population_by_cell=np.asarray(
                (1.6, 0.0, 0.0, 0.0), dtype=np.float32
            ),
            candidate_forecast_overrun_probability=np.full(
                4, 0.5, dtype=np.float32
            ),
            candidate_forecast_next_overrun_probability=np.full(
                4, 0.25, dtype=np.float32
            ),
            candidate_forecast_overrun_lead_time_fraction=np.full(
                4, 0.5, dtype=np.float32
            ),
            candidate_forecast_reevacuation_response=np.full(
                4, 0.25, dtype=np.float32
            ),
            forecast_reevacuation_population_fraction=0.1,
            forecast_safe_capacity_coverage_fraction=0.2,
            next_shelter_overrun_probability=1.0,
            next_shelter_overrun_lead_time_fraction=0.2,
        )
        ablated = RLBridge._without_forecast_information(observation)
        np.testing.assert_array_equal(ablated.danger_by_cell, observation.danger_by_cell)
        np.testing.assert_array_equal(ablated.action_mask, observation.action_mask)
        self.assertFalse(ablated.forecast_danger_by_cell.any())
        self.assertFalse(ablated.forecast_ensemble_mean.any())
        self.assertFalse(ablated.forecast_overrun_probability_by_cell.any())
        self.assertFalse(ablated.forecast_reevacuation_population_by_cell.any())
        self.assertFalse(ablated.candidate_forecast_reevacuation_response.any())
        self.assertEqual(ablated.forecast_reevacuation_population_fraction, 0.0)
        self.assertAlmostEqual(
            ablated.forecast_safe_capacity_coverage_fraction,
            5.0 / observation.initial_population,
        )
        expected_score = RLBridge._without_forecast_information(
            replace(
                observation,
                candidate_public_objective_score=np.asarray(
                    (1.0, 0.0, 0.5, 0.25), dtype=np.float32
                ),
                candidate_forecast_danger=np.asarray(
                    (1.0, 0.9, 0.8, 0.7), dtype=np.float32
                ),
                candidate_forecast_reevacuation_response=np.asarray(
                    (0.7, 0.6, 0.5, 0.4), dtype=np.float32
                ),
            )
        )
        np.testing.assert_allclose(
            ablated.candidate_features(),
            expected_score.candidate_features(),
        )

    def test_missing_overrun_forecast_means_no_crossing_not_immediate_failure(self):
        observation = make_observation()
        np.testing.assert_array_equal(
            observation.forecast_overrun_lead_time_fraction_by_cell,
            np.ones(observation.number_of_cells, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
