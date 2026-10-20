"""v59/v60: actor targets and latent intensity use the actor's information.

The hazard regime draws its whole event tape when the episode starts, so a
"realized" branch future replays wind shifts, intensity switches and other
latent events the actor cannot observe. Training targets graded that way are
hindsight targets: they rank actions as if the future were known and therefore
undervalue deferral and hedging. v59 grades actor targets with belief futures
by default; realized grading survives only as an explicitly named ablation.
v60 filters current latent intensity from observed spread instead of copying it.
"""

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from Core import Core
from HazardDatabase import HazardRegimeConfig, HazardRegimeProcess
from RLBridge import RLBridge
from TrainingCurriculum import LEARNER_STAGE_OVERRIDES


def _regime(seed: int) -> HazardRegimeProcess:
    config = HazardRegimeConfig(
        enabled=True,
        wind_shift_rate_per_minute=0.05,
        intensity_switch_rate_per_minute=0.05,
        wind_min_dwell_minutes=5,
        forecast_lead_minutes=15,
        forecast_miss_probability=0.0,
        episode_horizon=180,
    )
    # No map or cell tracker: closures and ignitions are not drawn, which
    # leaves wind (public within the lead window) and intensity (latent).
    return HazardRegimeProcess(config, np.random.default_rng(seed), None, None)


class BeliefRolloutTargetTests(unittest.TestCase):
    def test_belief_futures_depend_only_on_public_information(self):
        decision_time = 40
        lead = 15
        base = _regime(11)
        base.current_time = decision_time
        base._update_public(decision_time)

        # Same public state, different hidden future.
        twin = copy.deepcopy(base)
        base.intensity_multiplier = 0.7
        twin.intensity_multiplier = 1.6
        changed = 0
        for event in twin.scheduled:
            event_time = int(event["time"])
            if event["kind"] == "intensity" and event_time >= decision_time:
                multiplier = float(event["payload"]["multiplier"])
                event["payload"]["multiplier"] = 0.7 if multiplier != 0.7 else 1.6
                changed += 1
            elif event["kind"] == "wind" and event_time > decision_time + lead:
                event["payload"]["direction_delta"] = -float(
                    event["payload"]["direction_delta"]
                )
                changed += 1
        self.assertGreater(changed, 0, "test tape has no latent future event")
        self.assertNotEqual(base.scheduled, twin.scheduled)
        twin._update_public(decision_time)
        self.assertEqual(base.public, twin.public)

        belief_base = base.belief_particles(4, np.random.default_rng(7))
        belief_twin = twin.belief_particles(4, np.random.default_rng(7))
        for left, right in zip(belief_base, belief_twin):
            self.assertEqual(left.scheduled, right.scheduled)
            self.assertEqual(left.intensity_multiplier, right.intensity_multiplier)

        # Every particle is conditioned on the public warning, without
        # copying its concealed realized-tape payload.  This test config has
        # no false alarms or misses, so the warning implies a wind event.
        warning_time = decision_time + int(base.public["time_to_change"])
        for particle in belief_base:
            warned = [
                event
                for event in particle.scheduled
                if event["kind"] == "wind"
                and int(event["time"]) == warning_time
                and bool(event["payload"].get("warning_visible", False))
            ]
            self.assertEqual(len(warned), 1)
            payload = warned[0]["payload"]
            forecast_direction = (
                particle.wind_direction_degrees
                + float(payload["direction_delta"])
            ) % 360.0
            forecast_speed = (
                particle.base_wind_speed_m_per_minute
                * float(payload["speed_multiplier"])
            )
            self.assertAlmostEqual(
                forecast_direction,
                float(base.public["wind_forecast_direction"]),
            )
            self.assertAlmostEqual(
                forecast_speed,
                float(base.public["wind_forecast_speed"]),
            )

    def test_belief_is_the_default_and_realized_is_an_explicit_ablation(self):
        self.assertEqual(RLBridge._rollout_target_future_mode(SimpleNamespace()), "belief")
        self.assertEqual(
            RLBridge._rollout_target_future_mode(
                SimpleNamespace(nmccPiTargetFutureMode="Realized")
            ),
            "realized",
        )
        with self.assertRaisesRegex(ValueError, "nmccPiTargetFutureMode"):
            RLBridge._rollout_target_future_mode(
                SimpleNamespace(nmccPiTargetFutureMode="oracle")
            )

    def test_spread_evidence_updates_the_latent_intensity_posterior(self):
        class LikelihoodModel:
            @staticmethod
            def oneClip(value):
                return float(np.clip(value, 0.0, 1.0))

            @staticmethod
            def directional_spread_probability(base_probability, _source, _target):
                return float(base_probability)

        exposures = {
            index: [(0.2, (0, 0), (0, 1))]
            for index in range(30)
        }
        prior = np.full(3, 1.0 / 3.0, dtype=np.float64)

        high = _regime(17)
        high.intensity_belief = prior.copy()
        high.observe_spread_transition(
            LikelihoodModel(), exposures, tuple(exposures)
        )

        low = _regime(18)
        low.intensity_belief = prior.copy()
        low.observe_spread_transition(LikelihoodModel(), exposures, ())

        high_mean = float(np.dot(high.intensity_states, high.intensity_belief))
        low_mean = float(np.dot(low.intensity_states, low.intensity_belief))
        self.assertGreater(high_mean, float(np.dot(high.intensity_states, prior)))
        self.assertLess(low_mean, float(np.dot(low.intensity_states, prior)))
        self.assertGreater(high_mean, low_mean)

    def test_configuration_reports_and_validates_the_target_information_set(self):
        self.assertIn("nmccPiTargetFutureMode", LEARNER_STAGE_OVERRIDES)
        core = Core("local")
        self.assertEqual(core.nmccPiTargetFutureMode, "belief")
        core.trainMode = True
        core.trainingStageId = "P"
        core.nmccPolicyImprovement = True
        effective = core._effective_configuration()
        self.assertEqual(effective["nmcc_pi_target_future_mode"], "belief")
        learner = effective["policy_improvement_learner"]
        self.assertEqual(learner["target_future_mode"], "belief")
        self.assertEqual(learner["continuation_target"], "belief_complete_simulator_cost")

        core.nmccPiTargetFutureMode = "realized"
        learner = core._effective_configuration()["policy_improvement_learner"]
        self.assertEqual(
            learner["continuation_target"],
            "hindsight_realized_complete_simulator_cost",
        )

    def test_invalid_target_information_set_is_rejected(self):
        core = Core("local")
        core.address = "Test City"
        core.stopTime = 10
        core.pedVol = 10
        core.hazardVol = 1
        core.maxSpeed = 2.0
        core.cellX = 2
        core.cellY = 2
        core.shelterCanVol = 4
        core.initShelterVol = 1
        core.learningRate = 3e-4
        core.explorationRate = 0.0
        core.optimizer = "AdamW"
        core.hazardEvolutionMode = "stochastic"
        core._validate_effective_configuration()
        core.nmccPiTargetFutureMode = "oracle"
        with self.assertRaisesRegex(ValueError, "nmccPiTargetFutureMode"):
            core._validate_effective_configuration()


if __name__ == "__main__":
    unittest.main()
