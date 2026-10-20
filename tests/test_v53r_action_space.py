import unittest

import numpy as np

import nmcc_testbed
import CounterfactualBranch as CB
from DecisionInterface import RegionalObservationBuilder, RegionalShelterExecutor, PolicyDecision
from RewardProcessor import RewardProcessor


class V53RActionSpaceTests(unittest.TestCase):
    def _builder(self):
        core = nmcc_testbed.build(
            grid=6, cell_x=4, cell_y=4, population=80, stop_time=41,
            hazard_count=1, candidate_count=32, shelter_capacity_token=10,
        )
        core.actionSpace = "region_defer"
        core.hazardRegimeEnabled = False
        core.shelterActionInterval = 10
        return core, RegionalObservationBuilder(
            core, initial_population=80, horizon=40, maximum_deployments=2
        )

    def test_regions_plus_defer_and_deadline_mask(self):
        _, builder = self._builder()
        slack = builder.build(
            decision_index=0, simulation_time=10, remaining_deployments=2
        )
        deadline = builder.build(
            decision_index=1, simulation_time=20, remaining_deployments=2
        )
        self.assertEqual(slack.number_of_actions, 17)
        self.assertTrue(slack.action_mask[0])
        self.assertEqual(slack.deadline_slack, 1)
        self.assertFalse(deadline.action_mask[0])
        self.assertEqual(deadline.deadline_slack, 0)

    def test_terminal_transition_is_not_counted_as_a_deployment_epoch(self):
        _, builder = self._builder()
        final_epoch = builder.build(
            decision_index=2, simulation_time=30, remaining_deployments=1
        )
        self.assertEqual(final_epoch.remaining_epochs, 1)
        self.assertEqual(final_epoch.deadline_slack, 0)
        self.assertFalse(final_epoch.action_mask[0])

    def test_defer_is_masked_after_the_budget_is_exhausted(self):
        _, builder = self._builder()
        exhausted = builder.build(
            decision_index=3, simulation_time=30, remaining_deployments=0
        )
        self.assertFalse(exhausted.action_mask.any())

    def test_defer_is_not_misgraded_as_an_installation_in_cell_zero(self):
        core, builder = self._builder()
        observation = builder.build(
            decision_index=0, simulation_time=10, remaining_deployments=2
        )
        brancher = CB.CounterfactualBrancher(
            core,
            horizon=0,
            reward_model=RewardProcessor(),
            initial_population=80,
            episode_horizon=40,
            workers=1,
        )
        enriched = brancher.enrich_observation(observation)
        self.assertEqual(enriched.candidate_public_objective_score[0], 0.0)

    def test_defer_keeps_capacity_and_install_consumes_exactly_one_site(self):
        core, builder = self._builder()
        executor = RegionalShelterExecutor(core)
        observation = builder.build(
            decision_index=0, simulation_time=0, remaining_deployments=2
        )
        before = len(core.shelterDS.shelterList)
        receipt = executor.execute(observation, PolicyDecision(0, "test"))
        self.assertFalse(receipt.installed)
        self.assertEqual(len(core.shelterDS.shelterList), before)
        install = int(np.flatnonzero(observation.action_mask[1:])[0] + 1)
        receipt = executor.execute(observation, PolicyDecision(install, "test"))
        self.assertTrue(receipt.installed)
        self.assertEqual(len(core.shelterDS.shelterList), before + 1)


if __name__ == "__main__":
    unittest.main()
