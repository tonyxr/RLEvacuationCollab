#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision-epoch learning bridge for administrator-selected shelter cells."""

import copy
import csv
from dataclasses import dataclass
import hashlib
import math
import os
import time
from typing import Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import CounterfactualBranch as CB
import NMCCPolicyImprovement as NPI
import RolloutPolicyIteration as RPI
from CAProcessor import authoritative_grid_shape
from DecisionInterface import (
    AccessibilityDeficitHeuristic,
    ActivePopulationHeuristic,
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    HAZARD_FEATURE_SLICE,
    INFRA_FEATURE_NAMES,
    INFRA_FEATURE_SLICE,
    MOMENTUM_FEATURE_NAMES,
    ObjectiveMatchedGreedyHeuristic,
    PED_FEATURE_NAMES,
    PED_FEATURE_SLICE,
    HazardWeightedDemandHeuristic,
    OutcomeSnapshot,
    PolicyDecision,
    RegionalActionReceipt,
    RegionalObservation,
    RegionalObservationBuilder,
    RegionalShelterExecutor,
    RiskTimeReductionHeuristic,
    RouteTimeSavingHeuristic,
    UniformRegionalPolicy,
    estimate_clearance_tail,
    estimate_remaining_time_to_safety,
)
from GNN import (
    ACTOR_PRIORS,
    DEFAULT_EMBED_DIM,
    DEFAULT_MESSAGE_LAYERS,
    DEFAULT_TEMPORAL_DIM,
    NMCC_OUTCOME_NAMES,
    VALUE_COMPONENT_NAMES,
    EvacPolicy,
    fit_gnn,
)
from RewardProcessor import (
    REWARD_COMPONENT_NAMES,
    RewardBreakdown,
    RewardProcessor,
)


MODEL_VERSION = 57
# --- NMCC policy improvement (NMCC-PI) --------------------------------------
# See NMCCPolicyImprovement.py and docs/NMCC_POLICY_IMPROVEMENT_20260921.md.
from NMCCPIConfig import (  # noqa: E402  (torch-free shared contract)
    DEFAULT_NMCC_PI_ACTOR_EPOCHS,
    DEFAULT_NMCC_PI_BRANCH_HORIZON,
    DEFAULT_NMCC_PI_CONTINUATION_ENABLED,
    DEFAULT_NMCC_PI_CONTINUATION_LOSS_COEF,
    DEFAULT_NMCC_PI_WORKERS,
    DEFAULT_NMCC_PI_EPSILON,
    DEFAULT_NMCC_PI_ETA_MIN,
    DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS,
    DEFAULT_NMCC_PI_FIT_TOLERANCE,
    DEFAULT_NMCC_PI_MAX_BRANCHES,
    DEFAULT_NMCC_PI_REPLAY_MAX_EPISODES,
    DEFAULT_NMCC_PI_REPLAY_EPOCHS,
    DEFAULT_NMCC_PI_VALIDATION_FRACTION,
    DEFAULT_NMCC_PI_TAPES,
    DEFAULT_REPRESENTATION_CLONE_COEF,
    DEFAULT_REPRESENTATION_KL_CAP,
    DEFAULT_REPRESENTATION_MODE,
    DEFAULT_EXPLORATION_RATE_END,
    DEFAULT_EXPLORATION_RATE_START,
    DEFAULT_EXPLORATION_SCHEDULE,
    DEFAULT_LR_SCHEDULE,
    DEFAULT_LR_WARMUP_UPDATES,
    DEFAULT_LR_DECAY_UPDATES,
    DEFAULT_ACTOR_LR_MIN_FRACTION,
    DEFAULT_CRITIC_LR_MIN_FRACTION,
    REPRESENTATION_MODES,
)
DEFAULT_ROLLOUT_EPISODES = 8
DEFAULT_ENTROPY_COEF = 5e-3
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_ACTOR_LEARNING_RATE = 1e-4
DEFAULT_ACTOR_EPOCHS = 1
DEFAULT_CRITIC_EPOCHS = 4
DEFAULT_ADVANTAGE_SCALE_FLOOR = 0.05
DEFAULT_CLIP_EPS = 0.10
DEFAULT_PPO_EPOCHS = 4
# Reporting threshold only. Direct PPO never rejects, rolls back, or resizes an
# update because this diagnostic threshold was crossed.
DEFAULT_TARGET_KL = 0.015
DEFAULT_RESIDUAL_PENALTY_COEF = 0.0
DEFAULT_NMCC_NATURAL_LOSS_COEF = 0.0
DEFAULT_ENTROPY_COEF_END = DEFAULT_ENTROPY_COEF
DEFAULT_EXPLORATION_DECAY_UPDATES = 64
DEFAULT_TEMPERATURE_START = 1.0
DEFAULT_TEMPERATURE_END = 1.0
NMCC_TRAINING_PHASES = (
    "environment_dynamics",
    "policy_improvement",
    "evaluation",
)


@dataclass(frozen=True)
class TrainingStageContract:
    phase: str
    train_dynamics: bool
    train_random_reference: bool
    train_actor: bool
    train_value: bool
    use_continuation: bool


STAGED_TRAINING_CONTRACTS = {
    "E": TrainingStageContract(
        "environment_dynamics", True, True, False, False, False
    ),
    "P": TrainingStageContract(
        "policy_improvement", False, False, True, False, False
    ),
}

@dataclass
class ObservationFrame:
    cell_features: torch.Tensor
    global_features: torch.Tensor
    route_edge_index: torch.Tensor
    route_edge_weight: torch.Tensor
    candidate_cell_index: torch.Tensor
    candidate_features: torch.Tensor
    action_mask: torch.Tensor
    momentum_features: torch.Tensor
    simulation_time: torch.Tensor
    reference_debt: torch.Tensor


@dataclass
class Transition:
    observation_history: tuple[ObservationFrame, ...]
    action: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor
    value_components: torch.Tensor
    reward: torch.Tensor
    reward_components: torch.Tensor
    interval_cost: torch.Tensor
    random_reference_value: torch.Tensor
    random_reference_mc_target: Optional[torch.Tensor]
    done: torch.Tensor
    elapsed_timesteps: torch.Tensor
    # NMCC counterfactual advantage for this decision, when an exact paired
    # policy-candidate branch was collected. ``None`` is invalid when exact
    # counterfactual credit is enabled.
    counterfactual_advantage: Optional[torch.Tensor] = None
    counterfactual_components: Optional[torch.Tensor] = None
    counterfactual_reference_action: Optional[torch.Tensor] = None
    natural_outcome_target: Optional[torch.Tensor] = None
    causal_outcome_target: Optional[torch.Tensor] = None
    # Rollout policy iteration: frozen trainable actor prior, its constrained
    # categorical improvement target, and the physically graded support.
    # Scheduled exploration selects behavior actions but never enters the
    # policy-improvement target.
    improvement_prior: Optional[torch.Tensor] = None
    improvement_target: Optional[torch.Tensor] = None
    improvement_exact_mask: Optional[torch.Tensor] = None
    improvement_advantage: Optional[torch.Tensor] = None
    improvement_eta_floor: Optional[torch.Tensor] = None
    improvement_base_action: Optional[torch.Tensor] = None


@dataclass
class ReferenceValueEpisode:
    """One complete fixed-policy sequence for random-reference replay."""

    frames: tuple[ObservationFrame, ...]
    targets: torch.Tensor


@dataclass
class ActorReplayEpisode:
    """One action sequence with simulator-graded evidence for recent replay."""

    frames: tuple[ObservationFrame, ...]
    target_scores: torch.Tensor
    exact_masks: torch.Tensor
    eta_floors: torch.Tensor


@dataclass
class ActorFitEpisode:
    """One recent sequence retargeted from the current frozen actor."""

    frames: tuple[ObservationFrame, ...]
    target_probabilities: torch.Tensor
    target_scores: torch.Tensor
    exact_masks: torch.Tensor


@dataclass
class PendingDecision:
    observation: RegionalObservation
    decision: PolicyDecision
    receipt: RegionalActionReceipt
    observation_history: tuple[ObservationFrame, ...]
    active_person_time: float = 0.0
    hazard_exposure_person_time: float = 0.0
    controllable_active_person_time: float = 0.0
    controllable_hazard_exposure_person_time: float = 0.0
    controllable_expected_casualty_person_time: float = 0.0
    random_reference_value_before: float = 0.0
    random_reference_mc_target: float = float("nan")
    elapsed_timesteps: int = 0
    # Return of an independently policy-sampled reference shelter from this
    # decision state under the same future noise, plus its endpoint critic.
    # These stay empty when exact counterfactual credit is disabled.
    reference_action: int = -1
    reference_cell: int = -1
    reference_return: float = float("nan")
    reference_component_return: Optional[np.ndarray] = None
    reference_bootstrap_components: Optional[np.ndarray] = None
    reference_outcome_target: Optional[np.ndarray] = None
    counterfactual_steps: int = 0
    counterfactual_active_person_time: float = 0.0
    counterfactual_exposure_person_time: float = 0.0
    counterfactual_elapsed_timesteps: int = 0
    counterfactual_previous_outcome: Optional[OutcomeSnapshot] = None
    counterfactual_discounted_components: Optional[np.ndarray] = None
    factual_component_return: Optional[np.ndarray] = None
    factual_outcome_target: Optional[np.ndarray] = None
    factual_bootstrap_components: Optional[np.ndarray] = None
    # Recurrent state as of this decision. Both branches share history up to
    # here, so scoring both branch endpoints from it keeps the two bootstrap
    # values comparable instead of conditioning them on different pasts.
    recurrent_state_at_decision: Optional[tuple] = None
    # NMCC-PI record collected before the executor installed anything.
    policy_improvement: Optional[dict] = None


@dataclass(frozen=True)
class ForwardRewardBreakdown:
    """One completed interval under random-reference potential shaping."""

    reward: float
    random_reference_value_before: float
    random_reference_value_after: float
    plain_interval_cost: float
    exposure_interval_cost: float
    expected_casualty_interval_cost: float
    terminal_plain_residual_cost: float
    terminal_exposure_residual_cost: float
    decision_discount: float
    terminal_clearance_tail_cost: float = 0.0

    @property
    def interval_cost(self) -> float:
        return float(
            self.plain_interval_cost
            + self.exposure_interval_cost
            + self.expected_casualty_interval_cost
            + self.terminal_plain_residual_cost
            + self.terminal_exposure_residual_cost
            + self.terminal_clearance_tail_cost
        )

    @property
    def reference_potential_reduction(self) -> float:
        return float(
            self.random_reference_value_before
            - self.decision_discount * self.random_reference_value_after
        )


@dataclass
class RecurrentPolicyStep:
    frame: ObservationFrame
    logits: torch.Tensor
    prior_logits: torch.Tensor
    value: torch.Tensor
    value_components: torch.Tensor
    natural_outcomes: torch.Tensor
    nmcc_continuation_value: torch.Tensor
    random_reference_value: torch.Tensor


class RLBridge:
    """Coordinate shelter-site policies through one operational interface.

    Policies act only at deployment epochs. A pending action earns outcomes and
    person-time observed until the next feasible deployment decision or terminal
    state. The benchmark and RL policy therefore share the observation schema,
    cell-indexed action mask, action space, deployment budget, and the shared
    deterministic site-selection rule that resolves each chosen cell to its
    installed building -- so every policy's action is a cell priority, never a
    choice among buildings within a cell.
    """

    def __init__(
        self,
        core,
        *,
        gamma: float = 1.0,
        clip_eps: float = DEFAULT_CLIP_EPS,
        lr: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_PPO_EPOCHS,
        actor_lr: float = DEFAULT_ACTOR_LEARNING_RATE,
        critic_lr: Optional[float] = None,
        actor_epochs: int = DEFAULT_ACTOR_EPOCHS,
        critic_epochs: Optional[int] = None,
        advantage_scale_floor: float = DEFAULT_ADVANTAGE_SCALE_FLOOR,
        learning_reward_scale: float = 1.0,
        minibatch_size: int = 32,
        entropy_coef: float = DEFAULT_ENTROPY_COEF,
        rollout_episodes: int = DEFAULT_ROLLOUT_EPISODES,
        shelter_action_interval: int = 10,
        temporal_dim: int = DEFAULT_TEMPORAL_DIM,
        residual_penalty_coef: float = DEFAULT_RESIDUAL_PENALTY_COEF,
        optimizer_name: str = "AdamW",
        train_mode: bool = True,
        deployment_strategy: str = "rl",
        target_active_shelters: int = 0,
        policy_seed: int = 0,
        checkpoint_path: Optional[str] = None,
        diagnostics_path: Optional[str] = None,
        debug: bool = False,
        counterfactual_credit: bool = False,
        counterfactual_horizon: Optional[int] = None,
        counterfactual_weight: float = 1.0,
        counterfactual_intervention_cost: float = 0.0,
        nmcc_natural_loss_coef: float = DEFAULT_NMCC_NATURAL_LOSS_COEF,
        entropy_coef_end: float = DEFAULT_ENTROPY_COEF_END,
        exploration_decay_updates: int = DEFAULT_EXPLORATION_DECAY_UPDATES,
        action_temperature_start: float = DEFAULT_TEMPERATURE_START,
        action_temperature_end: float = DEFAULT_TEMPERATURE_END,
        nmcc_joint_counterfactual_weight: float = 1.0,
        nmcc_policy_improvement: bool = False,
        nmcc_pi_epsilon: float = DEFAULT_NMCC_PI_EPSILON,
        nmcc_pi_eta_min: float = DEFAULT_NMCC_PI_ETA_MIN,
        nmcc_pi_tapes: int = DEFAULT_NMCC_PI_TAPES,
        nmcc_pi_exhaustive_decisions: int = DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS,
        nmcc_pi_max_branches: int = DEFAULT_NMCC_PI_MAX_BRANCHES,
        nmcc_pi_continuation_enabled: bool = DEFAULT_NMCC_PI_CONTINUATION_ENABLED,
        nmcc_pi_continuation_loss_coef: float = DEFAULT_NMCC_PI_CONTINUATION_LOSS_COEF,
        nmcc_pi_workers: int = DEFAULT_NMCC_PI_WORKERS,
        random_reference_rollouts: int = 4,
        nmcc_pi_actor_epochs: int = DEFAULT_NMCC_PI_ACTOR_EPOCHS,
        nmcc_pi_fit_tolerance: float = DEFAULT_NMCC_PI_FIT_TOLERANCE,
        nmcc_pi_branch_horizon: int = DEFAULT_NMCC_PI_BRANCH_HORIZON,
        nmcc_pi_replay_max_episodes: int = DEFAULT_NMCC_PI_REPLAY_MAX_EPISODES,
        nmcc_pi_replay_epochs: int = DEFAULT_NMCC_PI_REPLAY_EPOCHS,
        nmcc_pi_validation_fraction: float = DEFAULT_NMCC_PI_VALIDATION_FRACTION,
        exploration_rate_start: float = DEFAULT_EXPLORATION_RATE_START,
        exploration_rate_end: float = DEFAULT_EXPLORATION_RATE_END,
        exploration_schedule: str = DEFAULT_EXPLORATION_SCHEDULE,
        learning_rate_schedule: str = DEFAULT_LR_SCHEDULE,
        lr_warmup_updates: int = DEFAULT_LR_WARMUP_UPDATES,
        lr_decay_updates: int = DEFAULT_LR_DECAY_UPDATES,
        actor_lr_min_fraction: float = DEFAULT_ACTOR_LR_MIN_FRACTION,
        critic_lr_min_fraction: float = DEFAULT_CRITIC_LR_MIN_FRACTION,
        actor_prior: str = "none",
        actor_prior_scale: float = 0.0,
        representation_mode: str = DEFAULT_REPRESENTATION_MODE,
        representation_clone_coef: float = DEFAULT_REPRESENTATION_CLONE_COEF,
        representation_kl_cap: float = DEFAULT_REPRESENTATION_KL_CAP,
        first_decision_time: int = 1,
        training_stage_id: str = "standalone",
        training_stage_update_index: int = 0,
        training_stage_total_updates: int = 0,
    ):
        self.core = core
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = self._bounded_float("gamma", gamma, 0.0, 1.0, lower_open=True)
        self.clip_eps = self._bounded_float("clip_eps", clip_eps, 0.0, 1.0, lower_open=True)
        self.lr = self._positive_float("lr", lr)
        self.epochs = self._positive_int("epochs", epochs)
        self.actor_lr = self._positive_float("actor_lr", actor_lr)
        self.critic_lr = self._positive_float(
            "critic_lr", self.lr if critic_lr is None else critic_lr
        )
        self.actor_epochs = self._positive_int("actor_epochs", actor_epochs)
        self.critic_epochs = self._positive_int(
            "critic_epochs", self.epochs if critic_epochs is None else critic_epochs
        )
        self.advantage_scale_floor = self._positive_float(
            "advantage_scale_floor", advantage_scale_floor
        )
        # Purely condition the learning signal.  The physical objective and
        # all reported episode metrics remain in their original normalized
        # units; PPO returns, critic targets, and both sides of every exact
        # candidate-pair comparison receive this same positive multiplier.
        self.learning_reward_scale = self._positive_float(
            "learning_reward_scale", learning_reward_scale
        )
        self.minibatch_size = self._positive_int("minibatch_size", minibatch_size)
        self.entropy_coef = self._bounded_float("entropy_coef", entropy_coef, 0.0, float("inf"))
        self.rollout_episodes = self._positive_int("rollout_episodes", rollout_episodes)
        self.training_stage_id = str(training_stage_id).strip()
        self.training_stage_update_index = int(training_stage_update_index)
        self.training_stage_total_updates = int(training_stage_total_updates)
        if not self.training_stage_id:
            raise ValueError("training_stage_id must not be empty")
        if self.training_stage_update_index < 0:
            raise ValueError("training_stage_update_index must be non-negative")
        if self.training_stage_total_updates < 0:
            raise ValueError("training_stage_total_updates must be non-negative")
        if (
            self.training_stage_total_updates > 0
            and self.training_stage_update_index
            >= self.training_stage_total_updates
        ):
            raise ValueError(
                "training_stage_update_index must be smaller than "
                "training_stage_total_updates"
            )
        # --- Hybrid NMCC exact credit and learned causal-model controls -----
        # Off by default for legacy runs; production curricula enable it
        # explicitly and the checkpoint signature records the full contract.
        self.counterfactual_credit = bool(counterfactual_credit)
        self.counterfactual_weight = float(counterfactual_weight)
        if not 0.0 <= self.counterfactual_weight <= 1.0:
            raise ValueError("counterfactual_weight must lie in [0, 1]")
        self.counterfactual_intervention_cost = float(counterfactual_intervention_cost)
        if not np.isfinite(self.counterfactual_intervention_cost) or self.counterfactual_intervention_cost < 0.0:
            raise ValueError("counterfactual_intervention_cost must be finite and non-negative")
        self.nmcc_natural_loss_coef = self._bounded_float(
            "nmcc_natural_loss_coef", nmcc_natural_loss_coef, 0.0, float("inf")
        )
        self.entropy_coef_end = self._bounded_float(
            "entropy_coef_end", entropy_coef_end, 0.0, float("inf")
        )
        if self.entropy_coef_end > self.entropy_coef:
            raise ValueError("entropy_coef_end cannot exceed entropy_coef")
        self.exploration_decay_updates = self._positive_int(
            "exploration_decay_updates", exploration_decay_updates
        )
        self.action_temperature_start = self._positive_float(
            "action_temperature_start", action_temperature_start
        )
        self.action_temperature_end = self._positive_float(
            "action_temperature_end", action_temperature_end
        )
        if self.action_temperature_end > self.action_temperature_start:
            raise ValueError(
                "action_temperature_end cannot exceed action_temperature_start"
            )
        self.nmcc_joint_counterfactual_weight = self._bounded_float(
            "nmcc_joint_counterfactual_weight",
            nmcc_joint_counterfactual_weight,
            0.0,
            1.0,
        )
        if self.counterfactual_credit:
            if self.counterfactual_intervention_cost != 0.0:
                raise ValueError(
                    "Candidate-versus-candidate NMCC requires zero intervention "
                    "cost because both branches install one capacity-matched shelter"
                )
            if (
                self.counterfactual_weight != 1.0
                or self.nmcc_joint_counterfactual_weight != 1.0
            ):
                raise ValueError(
                    "Exact NMCC credit must fully replace the ordinary actor "
                    "advantage; partial fallback blending is not supported"
                )
            if self.nmcc_natural_loss_coef > 0.0:
                raise ValueError(
                    "Pairwise NMCC credit cannot also train the staged "
                    "environment-dynamics head"
                )
        # --- Simulator-graded rollout policy improvement -------------------
        # A greedy-plus-uniform panel is evaluated from one simulator snapshot
        # under shared future tapes. Complete rollout costs produce a
        # constrained categorical target; no critic, correction gate or
        # fallback supplies an action grade.
        self.nmcc_policy_improvement = bool(nmcc_policy_improvement)
        if self.nmcc_policy_improvement and self.counterfactual_credit:
            raise ValueError(
                "Clean NMCC policy improvement and sampled pairwise NMCC "
                "credit are alternative estimators; enable only policy improvement"
            )
        self.nmcc_pi_epsilon = self._positive_float("nmcc_pi_epsilon", nmcc_pi_epsilon)
        self.nmcc_pi_eta_min = self._positive_float("nmcc_pi_eta_min", nmcc_pi_eta_min)
        self.nmcc_pi_tapes = self._positive_int("nmcc_pi_tapes", nmcc_pi_tapes)
        if self.nmcc_policy_improvement and self.nmcc_pi_tapes < 2:
            raise ValueError(
                "Simulator-graded policy iteration requires at least two "
                "shared futures to measure paired uncertainty"
            )
        self.nmcc_pi_exhaustive_decisions = max(0, int(nmcc_pi_exhaustive_decisions))
        self.nmcc_pi_max_branches = int(nmcc_pi_max_branches)
        if self.nmcc_pi_max_branches == 1 or self.nmcc_pi_max_branches < 0:
            raise ValueError(
                "nmcc_pi_max_branches must be zero (exhaustive) or at least two"
            )
        self.nmcc_pi_continuation_enabled = bool(nmcc_pi_continuation_enabled)
        self.nmcc_pi_continuation_loss_coef = self._bounded_float(
            "nmcc_pi_continuation_loss_coef",
            nmcc_pi_continuation_loss_coef,
            0.0,
            float("inf"),
        )
        if self.nmcc_pi_continuation_enabled and not self.nmcc_policy_improvement:
            raise ValueError(
                "NMCC continuation value requires nmcc_policy_improvement"
            )
        if self.nmcc_pi_continuation_enabled and self.nmcc_pi_continuation_loss_coef <= 0.0:
            raise ValueError(
                "NMCC continuation value requires a positive continuation loss coefficient"
            )
        self.nmcc_pi_workers = self._positive_int(
            "nmcc_pi_workers",
            nmcc_pi_workers,
        )
        self.random_reference_rollouts = self._positive_int(
            "random_reference_rollouts",
            random_reference_rollouts,
        )
        # Physical branch evidence is retained for a short recurrent replay
        # window. Its categorical targets are rebuilt from the current actor
        # before every M-step, rather than replaying stale action probabilities.
        self.nmcc_pi_actor_epochs = self._positive_int(
            "nmcc_pi_actor_epochs", nmcc_pi_actor_epochs
        )
        self.nmcc_pi_fit_tolerance = self._bounded_float(
            "nmcc_pi_fit_tolerance", nmcc_pi_fit_tolerance, 0.0, 1.0, lower_open=True
        )
        self.nmcc_pi_branch_horizon = int(nmcc_pi_branch_horizon)
        if self.nmcc_pi_branch_horizon < 0:
            raise ValueError("nmcc_pi_branch_horizon must be non-negative")
        self.nmcc_pi_replay_max_episodes = self._positive_int(
            "nmcc_pi_replay_max_episodes", nmcc_pi_replay_max_episodes
        )
        self.nmcc_pi_replay_epochs = self._positive_int(
            "nmcc_pi_replay_epochs", nmcc_pi_replay_epochs
        )
        self.nmcc_pi_validation_fraction = self._bounded_float(
            "nmcc_pi_validation_fraction",
            nmcc_pi_validation_fraction,
            0.0,
            0.5,
            lower_open=True,
        )
        self.exploration_rate_start = self._bounded_float(
            "exploration_rate_start", exploration_rate_start, 0.0, 1.0
        )
        self.exploration_rate_end = self._bounded_float(
            "exploration_rate_end", exploration_rate_end, 0.0, 1.0
        )
        if self.exploration_rate_end > self.exploration_rate_start:
            raise ValueError("exploration_rate_end cannot exceed exploration_rate_start")
        self.exploration_schedule = str(exploration_schedule).strip().lower()
        if self.exploration_schedule not in {"half_cosine", "exponential_half_life"}:
            raise ValueError(
                "exploration_schedule must be 'half_cosine' or "
                "'exponential_half_life'"
            )
        self.learning_rate_schedule = str(learning_rate_schedule).strip().lower()
        if self.learning_rate_schedule not in {"constant", "cosine"}:
            raise ValueError("learning_rate_schedule must be 'constant' or 'cosine'")
        self.lr_warmup_updates = max(0, int(lr_warmup_updates))
        self.lr_decay_updates = self._positive_int(
            "lr_decay_updates", lr_decay_updates
        )
        self.actor_lr_min_fraction = self._bounded_float(
            "actor_lr_min_fraction", actor_lr_min_fraction, 0.0, 1.0,
            lower_open=True,
        )
        self.critic_lr_min_fraction = self._bounded_float(
            "critic_lr_min_fraction", critic_lr_min_fraction, 0.0, 1.0,
            lower_open=True,
        )
        self.actor_prior = str(actor_prior)
        if self.actor_prior not in ACTOR_PRIORS:
            raise ValueError(f"actor_prior must be one of {ACTOR_PRIORS}")
        self.actor_prior_scale = self._bounded_float(
            "actor_prior_scale", actor_prior_scale, 0.0, float("inf")
        )
        if self.actor_prior == "none" and self.actor_prior_scale != 0.0:
            raise ValueError("actor_prior_scale must be zero when actor_prior is none")
        if self.actor_prior != "none" and self.actor_prior_scale <= 0.0:
            raise ValueError("A configured actor prior requires a positive scale")
        # Exact NMCC remains the only action-improvement mechanism.  A natural
        # outcome head is instantiated solely when explicitly supervised for
        # environment-dynamics representation pretraining; it never supplies
        # candidate values, action logits, a gate, or a fallback decision.
        self.learned_nmcc_models_enabled = bool(
            self.nmcc_natural_loss_coef > 0.0
            or self.nmcc_pi_continuation_enabled
        )
        # Who trains the observation encoder and the episode LSTM.
        # ``actor_owned`` (legacy) gives them to the actor optimizer only, so
        # the critic, the NMCC world model and the intervention-value ensemble
        # fit shallow heads on a representation they can never shape.
        # ``shared_phasic`` also trains them in the critic/auxiliary pass, with
        # an exact policy-preservation term KL(pi_ref || pi) so the dense
        # supervised signals shape the features without moving the policy
        # outside its trust region (phasic policy gradient, Cobbe et al. 2021).
        self.representation_mode = str(representation_mode)
        if self.representation_mode not in REPRESENTATION_MODES:
            raise ValueError(f"representation_mode must be one of {REPRESENTATION_MODES}")
        self.representation_clone_coef = self._bounded_float(
            "representation_clone_coef", representation_clone_coef, 0.0, float("inf")
        )
        self.representation_kl_cap = self._positive_float(
            "representation_kl_cap", representation_kl_cap
        )
        self.improvement_records: list[dict] = []
        self.counterfactual_records: list[dict] = []
        self._counterfactual_brancher = None
        self._reward_brancher = None
        self._counterfactual_horizon_override = (
            None if counterfactual_horizon is None else int(counterfactual_horizon)
        )
        self.shelter_action_interval = self._positive_int(
            "shelter_action_interval", shelter_action_interval
        )
        if (
            self.nmcc_pi_continuation_enabled
            and self.nmcc_pi_branch_horizon != self.shelter_action_interval
        ):
            raise ValueError(
                "Forward-looking NMCC requires nmcc_pi_branch_horizon to equal "
                "the shelter decision interval"
            )
        self.temporal_dim = self._positive_int("temporal_dim", temporal_dim)
        self.residual_penalty_coef = self._bounded_float(
            "residual_penalty_coef", residual_penalty_coef, 0.0, 1.0
        )
        self.train_mode = bool(train_mode)
        self.debug = bool(debug)

        strategy = str(deployment_strategy).strip().lower()
        if strategy not in {
            "rl",
            "rl_precommit",
            "objective_greedy",
            "random",
            "heuristic",
            "risk_reduction",
            "route_saving",
            "hazard_weighted",
            "accessibility_deficit",
            "none",
            "initial_only",
            "static_greedy",
        }:
            raise ValueError(f"Unsupported deployment strategy: {deployment_strategy!r}")
        if self.train_mode and strategy != "rl":
            raise ValueError("Policy training requires deployment_strategy='rl'")
        self.deployment_strategy = strategy

        self.nx, self.ny = authoritative_grid_shape(core)
        self.num_cells = self.nx * self.ny
        if self.num_cells <= 0:
            raise ValueError("The regional context grid must contain at least one cell")
        self.d_ped = len(PED_FEATURE_NAMES)
        self.d_haz = len(HAZARD_FEATURE_NAMES)
        self.d_inf = len(INFRA_FEATURE_NAMES)
        self.d_global = len(GLOBAL_FEATURE_NAMES)
        self.d_momentum = len(MOMENTUM_FEATURE_NAMES)
        self.d_candidate = len(CANDIDATE_FEATURE_NAMES)

        result = getattr(core.pedDS, "result", {})
        active_initial = sum(
            max(1, int(getattr(pedestrian, "group_size", 1)))
            for pedestrian in core.pedDS.pedAgentList.values()
            if not bool(getattr(pedestrian, "terminated", False))
        )
        classified_initial = sum(
            max(0, int(result.get(key, 0)))
            for key in ("arrival", "evacuated", "casualty")
        )
        self.initial_population = int(active_initial + classified_initial)
        if self.initial_population <= 0:
            raise ValueError("Cannot create a decision process with zero initialized pedestrians")
        self.horizon = max(1, int(core.stopTime) - 1)
        self.horizon_minutes = (
            float(self.horizon)
            * max(1e-9, float(getattr(core, "timeStepMinutes", 1.0)))
        )

        initial_shelters = int(len(core.shelterDS.shelterList))
        target_shelters = max(initial_shelters, int(target_active_shelters))
        self.maximum_deployments = max(0, target_shelters - initial_shelters)
        self.deployments_made = 0
        self.decision_index = 0
        self.first_decision_time = self._positive_int(
            "first_decision_time", first_decision_time
        )

        self.observation_builder = RegionalObservationBuilder(
            core,
            initial_population=self.initial_population,
            horizon=self.horizon,
            maximum_deployments=self.maximum_deployments,
        )
        self.edge_index = torch.as_tensor(
            self.observation_builder.spatial_edge_index,
            dtype=torch.long,
            device=self.device,
        )
        self.num_candidate_actions = self.observation_builder.number_of_actions
        self.executor = RegionalShelterExecutor(core)
        self.reward_model = RewardProcessor()
        # Population outcomes remain independent audit metrics.  The optimized
        # objective itself is accumulated only from executed installations:
        # each action receives its immediate pre/post reduction in remaining
        # hazard-weighted route time.
        self.objective_initial_outcome = OutcomeSnapshot(
            safe_completed=int(result.get("arrival", 0)) + int(result.get("evacuated", 0)),
            casualties=int(result.get("casualty", 0)),
            shelter_evacuated=int(result.get("evacuated", 0)),
            ordinary_arrivals=int(result.get("arrival", 0)),
            active_population=int(active_initial),
            risk_mass=0.0,
        )
        self.objective_latest_outcome = self.objective_initial_outcome
        self.objective_last_time = 0
        self.objective_active_person_time = 0.0
        self.objective_hazard_exposure_person_time = 0.0
        self.objective_expected_casualty_person_time = 0.0
        self.heuristic_policy = ActivePopulationHeuristic()
        self.objective_greedy_policy = ObjectiveMatchedGreedyHeuristic()
        self.risk_reduction_policy = RiskTimeReductionHeuristic()
        self.route_saving_policy = RouteTimeSavingHeuristic()
        self.hazard_weighted_policy = HazardWeightedDemandHeuristic()
        self.accessibility_deficit_policy = AccessibilityDeficitHeuristic(
            self._cell_centers()
        )
        self.policy_seed = int(policy_seed)
        self.random_policy = UniformRegionalPolicy(self.policy_seed)
        self._objective_greedy_brancher = None

        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actor_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.critic_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.representation_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.actor_head_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.critic_head_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.random_reference_named_parameters: list[tuple[str, nn.Parameter]] = []
        if self.deployment_strategy in {"rl", "rl_precommit"}:
            if tuple(VALUE_COMPONENT_NAMES) != tuple(REWARD_COMPONENT_NAMES):
                raise RuntimeError("Actor-critic and reward component orders disagree")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.policy_seed)
                self.policy = EvacPolicy(
                    d_ped=self.d_ped,
                    d_hazard=self.d_haz,
                    d_infra=self.d_inf,
                    d_global=self.d_global,
                    d_candidate=self.d_candidate,
                    embed_dim=DEFAULT_EMBED_DIM,
                    message_layers=DEFAULT_MESSAGE_LAYERS,
                    d_momentum=self.d_momentum,
                    temporal_dim=self.temporal_dim,
                    verbose=False,
                    actor_prior=self.actor_prior,
                    actor_prior_scale=self.actor_prior_scale,
                    actor_prior_feature_index=(
                        CANDIDATE_FEATURE_NAMES.index(
                            "exact_assignment_gain_fraction"
                        )
                        if self.actor_prior == "objective_greedy"
                        else 0
                    ),
                    auxiliary_models_enabled=self.learned_nmcc_models_enabled,
                ).to(self.device)
            roles = self.policy.parameter_roles()
            self.representation_named_parameters = list(roles["representation"])
            self.actor_head_named_parameters = list(roles["actor_head"])
            self.critic_head_named_parameters = list(roles["critic_head"])
            self.random_reference_named_parameters = [
                (name, parameter)
                for name, parameter in self.critic_head_named_parameters
                if name.startswith("random_reference_trunk.")
                or name.startswith("random_reference_head.")
            ]
            if self.representation_mode == "staged_transfer":
                # Stage E owns the shared graph/LSTM representation. Stage P
                # fits the complete actor adapter over those frozen shared
                # features. The dynamics head is actor-independent, so action
                # learning cannot erase the environment representation.
                self.actor_named_parameters = list(
                    self.actor_head_named_parameters
                )
                self.critic_named_parameters = (
                    self.representation_named_parameters
                    + self.critic_head_named_parameters
                )
            else:
                self.actor_named_parameters = (
                    self.representation_named_parameters
                    + self.actor_head_named_parameters
                )
                self.critic_named_parameters = list(
                    self.critic_head_named_parameters
                )
            if self.representation_mode in {"shared_multitask", "shared_phasic"}:
                self.critic_named_parameters = (
                    self.representation_named_parameters + self.critic_named_parameters
                )
            head_names = {name for name, _ in self.actor_head_named_parameters}
            critic_head_names = {name for name, _ in self.critic_head_named_parameters}
            representation_names = {
                name for name, _ in self.representation_named_parameters
            }
            all_names = {name for name, _ in self.policy.named_parameters()}
            if (
                head_names & critic_head_names
                or head_names & representation_names
                or critic_head_names & representation_names
                or head_names | critic_head_names | representation_names != all_names
            ):
                raise RuntimeError("Actor and critic parameter ownership is not a partition")
            self.actor_optimizer = self._make_optimizer(
                optimizer_name,
                [parameter for _, parameter in self.actor_named_parameters],
                self.actor_lr,
            )
            self.critic_optimizer = self._make_optimizer(
                optimizer_name,
                [parameter for _, parameter in self.critic_named_parameters],
                self.critic_lr,
            )

        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.action_generator = torch.Generator(device=generator_device)
        self.action_generator.manual_seed(self.policy_seed + 1)
        self.optimization_generator = torch.Generator(device="cpu")
        self.optimization_generator.manual_seed(self.policy_seed + 2)

        default_checkpoint = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "runs",
            "checkpoints",
            f"{str(core.address).replace(' ', '_')}_regional_policy.pt",
        )
        self.checkpoint_path = str(checkpoint_path or default_checkpoint)
        default_diagnostics = os.path.join(
            os.path.dirname(self.checkpoint_path),
            "learner_diagnostics.csv",
        )
        self.diagnostics_path = str(diagnostics_path or default_diagnostics)
        self.episodes_completed = 0
        self.optimizer_updates = 0
        self.actor_optimizer_updates = 0
        self.critic_optimizer_updates = 0
        self.actor_rollout_updates = 0
        self.rollout_updates_completed = 0
        self.rollout_traj: list[Transition] = []
        self.rollout_episode_count = 0
        self.random_reference_replay: list[ReferenceValueEpisode] = []
        self.actor_replay: list[ActorReplayEpisode] = []
        if self.deployment_strategy in {"rl", "rl_precommit"}:
            if os.path.exists(self.checkpoint_path):
                self._load_checkpoint(require_training_state=self.train_mode)
            elif not self.train_mode:
                raise FileNotFoundError(
                    f"RL evaluation requires a trained checkpoint: {self.checkpoint_path}"
                )
            if self.train_mode:
                # Loading an optimizer also restores its last learning rate.
                # A curriculum stage owns an independent schedule, so reset
                # the loaded parameter groups immediately rather than waiting
                # until the first update of the new stage.
                self._apply_learning_rate_schedule()
                self.policy.train()
            else:
                self.policy.eval()

        self.pending: Optional[PendingDecision] = None
        self.traj: list[Transition] = []
        self.recurrent_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.previous_temporal_observation: Optional[RegionalObservation] = None
        self.last_deployment_time: Optional[int] = None
        self.observation_frames_seen = 0
        self.last_simulation_time: Optional[int] = None
        self.episode_done = False
        self.episode_return = 0.0
        self.episode_random_reference_initial_value = float("nan")
        self.episode_random_reference_cost = 0.0
        self.episode_random_site_advantage = 0.0
        self.episode_controllable_interval_cost = 0.0
        self.episode_reference_potential_reduction = 0.0
        self.episode_remaining_risk_time_before = 0.0
        self.episode_remaining_risk_time_after = 0.0
        self.episode_raw_remaining_risk_time_reduction = 0.0
        self.episode_remaining_risk_time_reduction = 0.0
        self.episode_plain_time_wait = 0.0
        self.episode_plain_time_action = 0.0
        self.episode_raw_plain_time_reduction = 0.0
        self.episode_hazard_exposure_wait = 0.0
        self.episode_hazard_exposure_action = 0.0
        self.episode_raw_hazard_exposure_reduction = 0.0
        self.action_agreements = 0
        self.action_comparisons = 0
        self.behavior_entropies: list[float] = []
        self.last_training_diagnostics: Dict[str, float] = {}
        self.initial_observation_digest: Optional[str] = None
        self.precommit_initial_observation_digest: Optional[str] = None
        self.static_precommit_schedule: tuple[tuple[str, int], ...] = tuple()
        self.deployment_times: list[int] = []
        self.deployment_latency_records: list[dict[str, float]] = []

    def _cell_centers(self) -> np.ndarray:
        """Return row-major cell centroids in projected simulator metres."""
        tracker = getattr(self.core, "cellTracker", None)
        x_edges = np.asarray(getattr(tracker, "xEdges", ()), dtype=np.float64)
        y_edges = np.asarray(getattr(tracker, "yEdges", ()), dtype=np.float64)
        if x_edges.shape == (self.nx + 1,) and y_edges.shape == (self.ny + 1,):
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        else:
            # Test doubles and legacy callers may not expose partition edges.
            # Unit-spaced grid coordinates preserve deterministic geometry.
            x_centers = np.arange(self.nx, dtype=np.float64)
            y_centers = np.arange(self.ny, dtype=np.float64)
        return np.asarray(
            [(x_centers[i], y_centers[j]) for i in range(self.nx) for j in range(self.ny)],
            dtype=np.float64,
        )

    def _accumulate_policy_objective(
        self,
        observation: RegionalObservation,
        simulation_time: int,
        *,
        terminal: bool,
    ) -> None:
        """Accumulate realised hazard-weighted time to safety.

        This is the action-count-invariant evaluation objective, not the
        centred learning reward. Active pedestrians (including panicked
        pedestrians) accrue one plain person-minute plus their local danger
        mass per minute. Casualty risk is charged through its conditional
        intensity before the Bernoulli draw, so a realized draw never creates
        or removes learning cost.
        """
        del terminal
        simulation_time = int(simulation_time)
        elapsed = simulation_time - int(self.objective_last_time)
        if elapsed <= 0:
            raise ValueError("Policy-objective time must increase strictly")
        elapsed_minutes = (
            float(elapsed)
            * max(1e-9, float(getattr(self.core, "timeStepMinutes", 1.0)))
        )
        plain_mass = float(observation.outcome.active_population)
        exposure_mass = float(observation.outcome.hazard_exposure_mass)
        expected_casualty_mass = CB.expected_casualty_step_cost(
            self.core,
            simulation_time=simulation_time,
        )
        self.objective_active_person_time += plain_mass * elapsed_minutes
        self.objective_hazard_exposure_person_time += (
            exposure_mass * elapsed_minutes
        )
        self.objective_expected_casualty_person_time += (
            expected_casualty_mass * elapsed_minutes
        )
        if self.pending is not None:
            self.pending.active_person_time += plain_mass * elapsed_minutes
            self.pending.hazard_exposure_person_time += (
                exposure_mass * elapsed_minutes
            )
            active_by_cell = np.asarray(
                observation.active_by_cell, dtype=np.float64
            )
            complete_exposure = float(
                np.sum(
                    active_by_cell
                    * np.asarray(observation.danger_by_cell, dtype=np.float64)
                )
            )
            self.pending.controllable_active_person_time += (
                float(
                    active_by_cell.sum()
                )
                * elapsed_minutes
            )
            self.pending.controllable_hazard_exposure_person_time += (
                complete_exposure * elapsed_minutes
            )
            self.pending.controllable_expected_casualty_person_time += (
                expected_casualty_mass * elapsed_minutes
            )
        self.objective_last_time = simulation_time
        self.objective_latest_outcome = observation.outcome

    def _accumulate_installation_reward(
        self,
        breakdown: RewardBreakdown,
    ) -> None:
        """Record the random-site contrast as a credit diagnostic only."""
        self.episode_random_site_advantage += float(breakdown.total)
        self.episode_remaining_risk_time_before += float(
            breakdown.remaining_risk_time_before
        )
        self.episode_remaining_risk_time_after += float(
            breakdown.remaining_risk_time_after
        )
        self.episode_raw_remaining_risk_time_reduction += float(
            breakdown.raw_remaining_risk_time_reduction
        )
        self.episode_remaining_risk_time_reduction += float(
            breakdown.remaining_risk_time_reduction
        )
        self.episode_plain_time_wait += float(breakdown.plain_time_wait)
        self.episode_plain_time_action += float(breakdown.plain_time_action)
        self.episode_raw_plain_time_reduction += float(
            breakdown.raw_plain_time_reduction
        )
        self.episode_hazard_exposure_wait += float(
            breakdown.hazard_exposure_wait
        )
        self.episode_hazard_exposure_action += float(
            breakdown.hazard_exposure_action
        )
        self.episode_raw_hazard_exposure_reduction += float(
            breakdown.raw_hazard_exposure_reduction
        )

    def _random_site_installation_reward(
        self,
        observation: RegionalObservation,
        selected_action: int,
    ) -> RewardBreakdown:
        """Compare the selected site with the uniform feasible-site mean.

        This is the exact immediate assignment/routing value. Every candidate
        adds the same capacity token, so the common capacity gain cancels.
        Decision-interval congestion, hazard evolution and downstream choice
        value are added by the forward-looking NMCC target, not hidden inside
        this environmental reward.
        """
        feasible = np.flatnonzero(np.asarray(observation.action_mask, dtype=bool))
        action = int(selected_action)
        if action not in set(int(value) for value in feasible):
            raise ValueError("Selected action is not feasible")
        cells = np.asarray(
            observation.candidate_cell_indices[feasible], dtype=np.int64
        )
        if np.unique(cells).size != cells.size:
            raise RuntimeError(
                "Random-site reward requires one unique site per feasible action"
            )
        effects = self._ensure_reward_brancher().candidate_effects(cells)
        if len(effects) != int(feasible.size):
            raise RuntimeError(
                "Random-site reward did not evaluate every feasible site"
            )
        selected_position = int(np.flatnonzero(feasible == action)[0])
        selected = effects[selected_position]
        return self.reward_model.evaluate_random_site_advantage(
            start=selected.acted.start_outcome,
            action_after=selected.acted.end_outcome,
            random_plain_time=float(
                np.mean(
                    [effect.acted.plain_time_to_safety_cost for effect in effects]
                )
            ),
            action_plain_time=selected.acted.plain_time_to_safety_cost,
            random_hazard_exposure=float(
                np.mean(
                    [
                        effect.acted.hazard_exposure_to_safety_cost
                        for effect in effects
                    ]
                )
            ),
            action_hazard_exposure=(
                selected.acted.hazard_exposure_to_safety_cost
            ),
            initial_population=self.initial_population,
            horizon_minutes=self.horizon_minutes,
        )

    @staticmethod
    def _positive_int(name: str, value) -> int:
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _make_optimizer(name: str, parameters, learning_rate: float):
        """Construct one optimizer for one non-overlapping parameter owner."""
        optimizer_key = str(name).strip().lower()
        parameters = list(parameters)
        if not parameters:
            raise ValueError("An optimizer parameter partition must not be empty")
        if optimizer_key == "adam":
            return torch.optim.Adam(parameters, lr=float(learning_rate))
        if optimizer_key == "rmsprop":
            return torch.optim.RMSprop(parameters, lr=float(learning_rate))
        return torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=1e-4,
        )

    @staticmethod
    def _positive_float(name: str, value) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _bounded_float(
        name: str,
        value,
        lower: float,
        upper: float,
        *,
        lower_open: bool = False,
    ) -> float:
        result = float(value)
        lower_ok = result > lower if lower_open else result >= lower
        if not np.isfinite(result) or not lower_ok or result > upper:
            bracket = "(" if lower_open else "["
            raise ValueError(f"{name} must be in {bracket}{lower}, {upper}]")
        return result

    def _stage_contract(self) -> TrainingStageContract:
        """Return the explicit curriculum contract for this episode.

        Evaluation always exposes the complete trained policy regardless of
        the training label stored in the checkpoint.
        """
        if not self.train_mode:
            return TrainingStageContract(
                "evaluation", False, False, False, False,
                bool(self.nmcc_pi_continuation_enabled),
            )
        stage = self.training_stage_id.upper()
        if stage in STAGED_TRAINING_CONTRACTS:
            return STAGED_TRAINING_CONTRACTS[stage]
        if stage in {"STANDALONE", "JOINT"}:
            return STAGED_TRAINING_CONTRACTS["P"]
        raise ValueError(
            "NMCC staged training requires stage E or P; "
            f"received {self.training_stage_id!r}"
        )

    def _current_nmcc_training_phase(self) -> str:
        return self._stage_contract().phase

    def _actor_training_enabled(self) -> bool:
        return self._stage_contract().train_actor

    def _effective_counterfactual_weight(self) -> float:
        return float(self.counterfactual_weight)

    def _exploration_schedule(self, start: float, end: float, half_life: int) -> float:
        """Decay independently inside the current curriculum stage.

        Curriculum runs use a normalized stage clock and hit both declared
        endpoints exactly.  Direct legacy callers that do not supply stage
        coordinates retain the historical actor-update half-life schedule.
        """
        if self.training_stage_total_updates == 1:
            return float(start)
        if self.training_stage_total_updates > 1:
            age = float(self.training_stage_update_index)
            final_age = float(self.training_stage_total_updates - 1)
        else:
            age = max(0.0, float(self.actor_rollout_updates))
            final_age = float("inf")
        duration = float(max(1, int(half_life)))
        if self.exploration_schedule == "exponential_half_life":
            remaining = 0.5 ** (age / duration)
            if math.isfinite(final_age):
                terminal = 0.5 ** (final_age / duration)
                if terminal < 1.0:
                    remaining = (remaining - terminal) / (1.0 - terminal)
                else:
                    remaining = 0.0
                remaining = float(np.clip(remaining, 0.0, 1.0))
            return float(end + (start - end) * remaining)
        denominator = final_age if math.isfinite(final_age) else duration
        fraction = min(1.0, age / max(1.0, denominator))
        blend = 0.5 * (1.0 - math.cos(math.pi * fraction))
        return float(start + blend * (end - start))

    def _current_entropy_coef(self) -> float:
        return self._exploration_schedule(
            self.entropy_coef,
            self.entropy_coef_end,
            self.exploration_decay_updates,
        )

    def _current_action_temperature(self) -> float:
        return self._exploration_schedule(
            self.action_temperature_start,
            self.action_temperature_end,
            self.exploration_decay_updates,
        )

    def _current_exploration_rate(self) -> float:
        """Uniform-mixture exploration probability for categorical sampling."""
        if not self.train_mode or not self.nmcc_policy_improvement:
            return 0.0
        return self._exploration_schedule(
            self.exploration_rate_start,
            self.exploration_rate_end,
            self.exploration_decay_updates,
        )

    def _scheduled_learning_rate(
        self,
        base: float,
        minimum_fraction: float,
        update_index: int,
    ) -> float:
        if self.learning_rate_schedule == "constant":
            return float(base)
        if self.training_stage_total_updates > 0:
            update = max(0, int(self.training_stage_update_index))
            schedule_updates = max(
                1,
                int(self.lr_decay_updates),
                int(self.training_stage_total_updates)
                - 1
                - int(self.lr_warmup_updates),
            )
        else:
            update = max(0, int(update_index))
            schedule_updates = max(1, int(self.lr_decay_updates))
        if self.lr_warmup_updates > 0 and update < self.lr_warmup_updates:
            return float(base) * float(update + 1) / float(self.lr_warmup_updates)
        age = max(0, update - self.lr_warmup_updates)
        progress = min(1.0, age / float(schedule_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = float(minimum_fraction) + (1.0 - float(minimum_fraction)) * cosine
        return float(base) * multiplier

    def _apply_learning_rate_schedule(self) -> tuple[float, float]:
        actor_rate = self._scheduled_learning_rate(
            self.actor_lr,
            self.actor_lr_min_fraction,
            int(self.rollout_updates_completed),
        )
        critic_rate = self._scheduled_learning_rate(
            self.critic_lr,
            self.critic_lr_min_fraction,
            int(self.rollout_updates_completed),
        )
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_rate
        return actor_rate, critic_rate

    @property
    def remaining_deployments(self) -> int:
        return max(0, self.maximum_deployments - self.deployments_made)

    def _model_signature(self) -> dict:
        return {
            "version": MODEL_VERSION,
            "grid_shape": (self.nx, self.ny),
            "cell_features": CELL_FEATURE_NAMES,
            "global_features": GLOBAL_FEATURE_NAMES,
            "candidate_features": CANDIDATE_FEATURE_NAMES,
            "action_space": "structurally_masked_cell_policy_shared_deterministic_site_rule",
            "architecture": (
                "relational_route_gnn_lstm_with_auxiliary_dynamics"
                if self.policy.auxiliary_models_enabled
                else "relational_route_gnn_lstm_actor"
            ),
            "graph_contract": {
                "region_count": "runtime_X_times_Y",
                "spatial_edges": (
                    "celltracker_moore_neighbors_union_road_crossings"
                ),
                "cell_index_order": "x_major_then_y_celltracker_flattening",
                "coordinate_system": (
                    "local_x_east_y_south_with_exposed_north_features_inverted"
                ),
                "route_edges": "bidirectional_current_assignment_flow",
                "route_aggregation": "population_share_weighted_sum",
                "message_layers": DEFAULT_MESSAGE_LAYERS,
                "pooling": "learned_attention_plus_maximum",
                "temporal_encoder": "episode_lstm_over_shelter_decision_epochs",
                "temporal_hidden_dimension": int(self.temporal_dim),
                "momentum_features": MOMENTUM_FEATURE_NAMES,
                "behavior_policy_dropout": False,
                "shelter_capacity_contract": {
                    "assignment": (
                        "conserved_atomic_reservations_with_strict_route_improvement"
                    ),
                    "mode": (
                        "equal_token"
                        if int(getattr(self.core, "shelterCapacityToken", 0)) > 0
                        else "site_specific_legacy"
                    ),
                    "capacity_value": "runtime_observation_not_model_architecture",
                },
                "checkpoint_transfer_across_resolutions": False,
            },
            "heuristic_prior_scale": float(self.actor_prior_scale),
            "actor_prior": {
                "kind": str(self.actor_prior),
                "scale": float(self.actor_prior_scale),
                "feature": {
                    "risk_time_reduction": None,
                    "route_time_saving": None,
                    "active_population": "relative_active_population",
                    "objective_greedy": "exact_assignment_gain_fraction",
                    "none": None,
                }[self.actor_prior],
            },
            "learner_contract": {
                "objective": (
                    "simulator_graded_residual_rollout_policy_iteration"
                    if self.nmcc_policy_improvement
                    else "direct_on_policy_clipped_ppo"
                ),
                "learning_reward_scale": float(self.learning_reward_scale),
                "actor_advantage": (
                    "exact_common_noise_candidate_minus_uniform_feasible_reference"
                    if self.counterfactual_credit
                    else (
                        "kl_constrained_categorical_target_from_complete_rollout_costs"
                        if self.nmcc_policy_improvement
                        else "complete_episode_return_minus_rollout_value"
                    )
                ),
                "critic_target": (
                    "not_used_by_rollout_policy_improvement"
                    if self.nmcc_policy_improvement
                    else "complete_episode_return"
                ),
                "representation_mode": str(self.representation_mode),
                "representation_clone_coef": float(self.representation_clone_coef),
                "representation_kl_cap": float(self.representation_kl_cap),
                "learning_rate_schedule": str(self.learning_rate_schedule),
                "schedule_scope": (
                    "each_curriculum_stage_resets_independently_and_honors_"
                    "configured_decay_horizon"
                ),
                "learning_rate_warmup_updates": int(self.lr_warmup_updates),
                "learning_rate_decay_updates": int(self.lr_decay_updates),
                "actor_lr_min_fraction": float(self.actor_lr_min_fraction),
                "critic_lr_min_fraction": float(self.critic_lr_min_fraction),
                "exploration": (
                    {
                        "kind": "smooth_uniform_mixture_categorical_sampling",
                        "start": float(self.exploration_rate_start),
                        "end": float(self.exploration_rate_end),
                        "temperature_start": float(
                            self.action_temperature_start
                        ),
                        "temperature_end": float(
                            self.action_temperature_end
                        ),
                        "schedule_updates": int(self.exploration_decay_updates),
                        "schedule": str(self.exploration_schedule),
                        "scope": "each_curriculum_stage_resets_independently",
                    }
                    if self.nmcc_policy_improvement
                    else {
                        "kind": "categorical_sampling_from_masked_policy_logits",
                        "entropy_coefficient_start": float(self.entropy_coef),
                        "entropy_coefficient_end": float(self.entropy_coef_end),
                        "temperature_start": float(self.action_temperature_start),
                        "temperature_end": float(self.action_temperature_end),
                        "schedule_updates": int(self.exploration_decay_updates),
                        "schedule": str(self.exploration_schedule),
                        "scope": "each_curriculum_stage_resets_independently",
                    }
                ),
            },
            "candidate_constraints": {
                "mask": "remaining_budget_and_positive_physical_capacity_only",
                "privileged_candidate_rank_feature": False,
            },
            "residual_correction": "unbounded_linear_actor_readout",
            "counterfactual_rollout_contract": {
                "enabled": bool(
                    self.counterfactual_credit or self.nmcc_policy_improvement
                ),
                "variant": (
                    "simulator_graded_residual_rollout_policy_iteration"
                    if self.nmcc_policy_improvement
                    else "exact_policy_sampled_candidate_pair_credit"
                ),
                "role": (
                    "full_horizon_common_noise_action_grading"
                    if self.nmcc_policy_improvement
                    else "credit_assignment_only_not_action_selection"
                ),
                "reference_action": (
                    "objective_greedy_contenders_plus_uniform_feasible_extras"
                    if self.nmcc_policy_improvement
                    else "independent_categorical_draw_from_same_behavior_policy"
                ),
                "comparison_horizon": (
                    "complete_simulation_to_episode_horizon"
                ),
                "independent_future_tapes": int(self.nmcc_pi_tapes),
                "parallel_physical_branch_workers": int(self.nmcc_pi_workers),
                "random_reference_rollouts_per_state": int(
                    self.random_reference_rollouts
                ),
                "legacy_branch_horizon_timesteps_ignored": int(
                    self.nmcc_pi_branch_horizon
                ),
                "counterfactual_weight": float(self.counterfactual_weight),
                "paired_identity": "same_state_same_future_complete_rollouts",
                "wait_role": "not_an_actor_fallback_or_training_target",
                "sampled_actions_per_state": int(self.nmcc_pi_max_branches),
                "exhaustive_early_decisions": int(
                    self.nmcc_pi_exhaustive_decisions
                ),
                "continuation_policy": "frozen_current_residual_actor",
                "dedicated_continuation_value": False,
                "random_reference_value": (
                    "direct_uniform_policy_monte_carlo_at_each_decision"
                ),
                "shared_actor_critic_continuation_value": False,
                "continuation_loss_coefficient": float(
                    self.nmcc_pi_continuation_loss_coef
                ),
                "continuation_target": "realized_complete_simulator_cost",
                "stage_contract": {
                    "P": "frozen_rollout_collection_then_categorical_policy_fit",
                },
                "environment_reward_branch_horizon": 0,
                "actor_update": (
                    "current_policy_rebased_recent_rollout_cross_entropy"
                ),
                "actor_replay_episodes": int(
                    self.nmcc_pi_replay_max_episodes
                ),
                "correction_gate": False,
                "fallback_controller": False,
            },
            "cell_partition": {
                "mode": str(
                    getattr(
                        self.core,
                        "cellPartitionMode",
                        "node_density_adaptive",
                    )
                ),
                "minimum_axis_width_fraction": float(
                    getattr(self.core, "cellPartitionMinWidthFraction", 1e-4)
                ),
            },
            "hazard_forecast": {
                "horizon_timesteps": int(self.shelter_action_interval),
                "wind_observation": "normalized_east_north_velocity_tuple",
                "wind_influence": float(
                    getattr(self.core, "hazardWindInfluence", 1.0)
                ),
                "distribution_summary": (
                    "expected_danger_escalation_probability_and_uncertainty_"
                    "from_current_front_momentum"
                ),
                "cellular_automaton_contract": (
                    "shared_runtime_celltracker_shape_flattening_moore_topology_"
                    "and_deterministic_transition_rules"
                ),
                "used_for_observation_only": True,
            },
            "decision_schedule": {
                "first_decision_timestep": int(self.first_decision_time),
                "shelter_action_interval_timesteps": int(
                    self.shelter_action_interval
                ),
                "transition_boundary": "next_executed_action_or_true_environment_terminal",
                "reward_measurement": (
                    "uniform_random_policy_value_minus_realized_complete_"
                    "interval_cost_minus_next_uniform_random_policy_value"
                ),
                "counterfactual_accounting": (
                    "same_state_equal_capacity_sampled_installed_site_contrasts"
                ),
            },
            "rollout_episodes": self.rollout_episodes,
            "reward_contract": {
                "evacuation_time_weight": (
                    self.reward_model.evacuation_time_weight
                ),
                "hazard_exposure_weight": (
                    self.reward_model.hazard_exposure_weight
                ),
                "forward_hazard_weighted_progress": (
                    "uniform_random_policy_cost_to_go_before_minus_discounted_"
                    "reference_value_after_minus_realized_complete_interval_cost"
                ),
                "plain_time_term": "active_person_time_plus_endpoint_plain_time",
                "hazard_exposure_term": (
                    "danger_weighted_active_person_time_plus_endpoint_exposure"
                ),
                "route_change_rule": (
                    "none_to_reserved_or_strictly_shorter_capacity_valid_destination"
                ),
                "unavailable_shelter_time_minutes": float(
                    self.horizon
                    * float(getattr(self.core, "timeStepMinutes", 1.0))
                ),
                "genuine_safe_completion": (
                    "ordinary_arrival_or_actual_shelter_admission_only"
                ),
                "casualty_role": (
                    "rao_blackwellized_expected_loss_intensity_in_the_same_"
                    "complete_person_minute_ledger"
                ),
                "unfinished_terminal_debt": True,
                "clearance_tail": {
                    "fraction": float(
                        getattr(self.core, "clearanceTailFraction", 0.10)
                    ),
                    "extra_weight": float(
                        getattr(self.core, "clearanceTailExtraWeight", 0.0)
                    ),
                    "optimized_statistic": (
                        "sum_of_restricted_completion_times_in_slowest_fraction"
                    ),
                    "reported_statistics": "T90_tail_mean_and_T100",
                },
                "natural_evacuation_progress_reward": (
                    "potential_based_and_telescoping_not_a_standalone_bonus"
                ),
                "capacity_installation_bonus": False,
                "site_specific_shaping": (
                    "uniform_random_reference_policy_potential"
                ),
                "panic_and_casualty_learning_role": (
                    "retained_in_complete_time_to_safety_ledger_and_common_"
                    "noise_differenced_between_actions"
                ),
                "evaluation_objective": (
                    "minimize_normalized_complete_ledger_cost_composed_of_"
                    "active_time_hazard_exposure_expected_casualty_and_clearance_tail"
                ),
                "factorized_critic_components": (
                    "not_used_by_rollout_policy_improvement"
                    if self.nmcc_policy_improvement
                    else REWARD_COMPONENT_NAMES
                ),
                "learning_signal_scale": float(self.learning_reward_scale),
                "reported_objective_scale": 1.0,
            },
            "training_environment": {
                "horizon_timesteps": int(self.horizon),
                "candidate_action_count": int(self.num_candidate_actions),
                "maximum_deployments": int(self.maximum_deployments),
                "time_step_minutes": float(
                    getattr(self.core, "timeStepMinutes", 1.0)
                ),
                "free_flow_speed_m_per_minute": float(self.core.maxSpeed),
                "congestion": (
                    None
                    if getattr(self.core, "congestionModel", None) is None
                    else self.core.congestionModel.contract()
                ),
                "social_force": (
                    None
                    if getattr(self.core, "forceTracker", None) is None
                    else self.core.forceTracker.contract()
                ),
                "panic_behavior": {
                    key: value
                    for key, value in (
                        self.core.pedDS.panic_contract()
                        if callable(getattr(self.core.pedDS, "panic_contract", None))
                        else {
                            "model": "persistent_first_exposure_susceptibility_v2",
                            "danger_level_threshold": 3,
                            "one_onset_trial_per_pedestrian": True,
                            "persistent_after_onset": True,
                            "herd_choice_probability": 0.5,
                            "random_choice_probability": 0.5,
                            "herd_rule": "incident physical edge with greatest frozen active occupancy",
                            "population_representation": "individual_required_when_rate_positive",
                        }
                    ).items()
                    if key
                    not in {
                        "rate_per_eligible_pedestrian_timestep",
                        "rate_among_first_exposed_pedestrians",
                    }
                },
                "intersection_consolidation": {
                    "enabled": bool(
                        getattr(self.core, "intersectionConsolidationEnabled", True)
                    ),
                    "tolerance_m": float(
                        getattr(self.core, "intersectionConsolidationToleranceM", 5.0)
                    ),
                },
            },
            "learner_hyperparameters": {
                "gamma": self.gamma,
                "actor_credit_target": (
                    "kl_constrained_categorical_rollout_policy_improvement"
                    if self.nmcc_policy_improvement
                    else "exact_common_noise_chosen_candidate_minus_policy_reference_candidate"
                    if self.counterfactual_credit
                    else "complete_episode_smdp_monte_carlo_return_to_go"
                ),
                "actor_baseline": (
                    "objective_greedy_score_plus_learned_residual"
                    if self.nmcc_policy_improvement
                    else "independent_behavior_policy_candidate"
                    if self.counterfactual_credit
                    else "rollout_state_value"
                ),
                "advantage_normalization": (
                    "none_categorical_target_retains_physical_cost_scale"
                    if self.nmcc_policy_improvement
                    else "zero_mean_unit_variance_with_machine_epsilon_only"
                ),
                "legacy_advantage_scale_floor_ignored": self.advantage_scale_floor,
                "learning_reward_scale": self.learning_reward_scale,
                "critic_target": (
                    "not_used_by_rollout_policy_improvement"
                    if self.nmcc_policy_improvement
                    else "complete_episode_smdp_monte_carlo_return_to_go"
                ),
                "clip_epsilon": (
                    None if self.nmcc_policy_improvement else self.clip_eps
                ),
                "actor_learning_rate": self.actor_lr,
                "critic_learning_rate": self.critic_lr,
                "actor_epochs": (
                    self.nmcc_pi_replay_epochs
                    if self.nmcc_policy_improvement
                    else self.actor_epochs
                ),
                "actor_replay_episodes": (
                    self.nmcc_pi_replay_max_episodes
                    if self.nmcc_policy_improvement
                    else 0
                ),
                "critic_epochs": self.critic_epochs,
                "minibatch_size": self.minibatch_size,
                "entropy_coefficient_start": self.entropy_coef,
                "entropy_coefficient_end": self.entropy_coef_end,
                "action_temperature_start": self.action_temperature_start,
                "action_temperature_end": self.action_temperature_end,
                "learning_rate_schedule": self.learning_rate_schedule,
                "learning_rate_warmup_updates": self.lr_warmup_updates,
                "learning_rate_decay_updates": self.lr_decay_updates,
                "actor_learning_rate_minimum_fraction": self.actor_lr_min_fraction,
                "critic_learning_rate_minimum_fraction": self.critic_lr_min_fraction,
                "residual_penalty_coefficient": self.residual_penalty_coef,
                "entropy_normalization": "log_feasible_action_count",
                "kl_control": (
                    "e_step_target_kl_constraint_and_diagnostic_actor_kl"
                    if self.nmcc_policy_improvement
                    else "diagnostic_only_for_direct_ppo"
                ),
                "critic_loss": (
                    "not_used_by_rollout_policy_improvement"
                    if self.nmcc_policy_improvement
                    else "raw_smooth_l1_monte_carlo_return"
                ),
                "critic_factorization": (
                    "not_used_by_rollout_policy_improvement"
                    if self.nmcc_policy_improvement
                    else "signed_component_heads"
                ),
                "optimizer_ownership": (
                    "actor_only_current_policy_rebased_recent_rollout_fit"
                    if self.nmcc_policy_improvement
                    else "disjoint_actor_and_critic_world_parameters"
                ),
                "sequence_sampling": (
                    "latest_whole_episode_recurrent_window"
                    if self.nmcc_policy_improvement
                    else "whole_episode_recurrent_minibatches"
                ),
            },
        }

    @staticmethod
    def _inference_signature(model_signature: Mapping) -> dict:
        """Return the checkpoint fields that determine policy inference.

        Learner batch and optimizer settings govern how a policy is trained or
        resumed, but they do not change the observation tensors, action
        semantics, network architecture, or policy weights used at evaluation.
        Keeping this contract separate lets a frozen checkpoint be evaluated
        under a new experiment horizon without weakening exact continuation
        checks for training.
        """
        if not isinstance(model_signature, Mapping):
            raise ValueError("checkpoint model signature must be a mapping")
        fields = (
            "version",
            "grid_shape",
            "cell_features",
            "global_features",
            "candidate_features",
            "action_space",
            "architecture",
            "graph_contract",
            "heuristic_prior_scale",
            "actor_prior",
            "candidate_constraints",
            "residual_correction",
            "cell_partition",
            "hazard_forecast",
            "decision_schedule",
        )
        missing = [field for field in fields if field not in model_signature]
        if missing:
            raise ValueError(
                f"checkpoint model signature is missing inference fields: {missing}"
            )
        return {field: model_signature[field] for field in fields}

    def _load_checkpoint(self, *, require_training_state: bool) -> None:
        try:
            # Training continuation requires optimizer and RNG objects in addition to
            # tensor weights. Only load explicit experiment-local checkpoints.
            try:
                payload = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                payload = torch.load(self.checkpoint_path, map_location=self.device)
            if not isinstance(payload, dict) or "policy_state_dict" not in payload:
                raise ValueError("checkpoint does not contain the versioned regional policy payload")
            saved_model_signature = payload.get("model_signature")
            current_model_signature = self._model_signature()
            saved_inference_signature = self._inference_signature(saved_model_signature)
            declared_inference_signature = payload.get(
                "inference_signature", saved_inference_signature
            )
            if declared_inference_signature != saved_inference_signature:
                raise ValueError(
                    "checkpoint inference signature is inconsistent with its model signature"
                )
            if require_training_state:
                if saved_model_signature != current_model_signature:
                    raise ValueError(
                        "checkpoint training signature does not match this exact PPO resume configuration"
                    )
            elif declared_inference_signature != self._inference_signature(
                current_model_signature
            ):
                raise ValueError(
                    "checkpoint inference signature does not match this decision interface"
                )
            if require_training_state:
                if payload.get("training_resume_allowed", True) is not True:
                    raise ValueError(
                        "checkpoint is marked evaluation-only and cannot resume PPO training"
                    )
                required = {
                    "actor_optimizer_state_dict",
                    "critic_optimizer_state_dict",
                    "action_generator_state",
                    "optimization_generator_state",
                    "rollout_transitions",
                    "rollout_episode_count",
                    "random_reference_replay",
                    "actor_replay",
                    "actor_optimizer_updates",
                    "critic_optimizer_updates",
                    "actor_rollout_updates",
                    "rollout_updates_completed",
                }
                missing = sorted(required.difference(payload))
                if missing:
                    raise ValueError(f"training checkpoint is missing state fields: {missing}")

            current = self.policy.state_dict()
            saved = payload["policy_state_dict"]
            if set(current) != set(saved):
                raise ValueError("checkpoint parameter names do not match")
            for name, tensor in current.items():
                saved_tensor = saved[name]
                if not torch.is_tensor(saved_tensor) or tuple(saved_tensor.shape) != tuple(tensor.shape):
                    raise ValueError(f"checkpoint parameter {name!r} has an incompatible shape")

            self.policy.load_state_dict(saved, strict=True)
            if require_training_state:
                self.actor_optimizer.load_state_dict(
                    payload["actor_optimizer_state_dict"]
                )
                self.critic_optimizer.load_state_dict(
                    payload["critic_optimizer_state_dict"]
                )
                self.action_generator.set_state(payload["action_generator_state"])
                self.optimization_generator.set_state(payload["optimization_generator_state"])
                self.rollout_traj = [
                    self._deserialize_transition(item)
                    for item in payload["rollout_transitions"]
                ]
                self.rollout_episode_count = int(payload["rollout_episode_count"])
                self._validate_rollout_state()
                self.random_reference_replay = [
                    self._deserialize_reference_episode(item)
                    for item in payload["random_reference_replay"]
                ]
                self.actor_replay = [
                    self._deserialize_actor_replay_episode(item)
                    for item in payload["actor_replay"]
                ]
            self.episodes_completed = int(payload.get("episodes_completed", 0))
            self.optimizer_updates = int(payload.get("optimizer_updates", 0))
            self.actor_optimizer_updates = int(
                payload.get("actor_optimizer_updates", 0)
            )
            self.critic_optimizer_updates = int(
                payload.get("critic_optimizer_updates", 0)
            )
            self.actor_rollout_updates = int(
                payload.get("actor_rollout_updates", 0)
            )
            self.rollout_updates_completed = int(
                payload.get("rollout_updates_completed", 0)
            )
        except Exception as exc:
            if self.train_mode:
                raise RuntimeError(
                    f"Refusing to resume from incompatible checkpoint {self.checkpoint_path}: {exc}"
                ) from exc
            raise RuntimeError(
                f"Refusing to evaluate incompatible checkpoint {self.checkpoint_path}: {exc}"
            ) from exc

    def _save_checkpoint(self) -> None:
        if (
            not self.train_mode
            or self.policy is None
            or self.actor_optimizer is None
            or self.critic_optimizer is None
        ):
            return
        directory = os.path.dirname(os.path.abspath(self.checkpoint_path))
        os.makedirs(directory, exist_ok=True)
        temporary_path = f"{self.checkpoint_path}.tmp"
        payload = {
            "training_resume_allowed": True,
            "model_signature": self._model_signature(),
            "inference_signature": self._inference_signature(self._model_signature()),
            "policy_state_dict": self.policy.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "episodes_completed": self.episodes_completed,
            "optimizer_updates": self.optimizer_updates,
            "actor_optimizer_updates": self.actor_optimizer_updates,
            "critic_optimizer_updates": self.critic_optimizer_updates,
            "actor_rollout_updates": self.actor_rollout_updates,
            "rollout_updates_completed": self.rollout_updates_completed,
            "policy_seed": self.policy_seed,
            "action_generator_state": self.action_generator.get_state(),
            "optimization_generator_state": self.optimization_generator.get_state(),
            "rollout_transitions": [
                self._serialize_transition(transition)
                for transition in self.rollout_traj
            ],
            "rollout_episode_count": self.rollout_episode_count,
            "random_reference_replay": [
                self._serialize_reference_episode(episode)
                for episode in self.random_reference_replay
            ],
            "actor_replay": [
                self._serialize_actor_replay_episode(episode)
                for episode in self.actor_replay
            ],
        }
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, self.checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    @staticmethod
    def _serialize_frame(frame: ObservationFrame) -> dict:
        return {
            name: getattr(frame, name).detach().cpu()
            for name in ObservationFrame.__dataclass_fields__
        }

    @staticmethod
    def _serialize_transition(transition: Transition) -> dict:
        payload = {}
        for name in Transition.__dataclass_fields__:
            if name == "observation_history":
                continue
            value = getattr(transition, name)
            payload[name] = None if value is None else value.detach().cpu()
        payload["observation_history"] = [
            RLBridge._serialize_frame(frame)
            for frame in transition.observation_history
        ]
        return payload

    @staticmethod
    def _serialize_reference_episode(episode: ReferenceValueEpisode) -> dict:
        return {
            "frames": [
                RLBridge._serialize_frame(frame) for frame in episode.frames
            ],
            "targets": episode.targets.detach().cpu(),
        }

    @staticmethod
    def _serialize_actor_replay_episode(episode: ActorReplayEpisode) -> dict:
        return {
            "frames": [
                RLBridge._serialize_frame(frame) for frame in episode.frames
            ],
            "target_scores": episode.target_scores.detach().cpu(),
            "exact_masks": episode.exact_masks.detach().cpu(),
            "eta_floors": episode.eta_floors.detach().cpu(),
        }

    def _deserialize_frame(
        self, payload: dict, *, target_device: Optional[torch.device] = None
    ) -> ObservationFrame:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint observation frame must be a dictionary")
        expected = set(ObservationFrame.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("checkpoint observation frame fields do not match")
        tensors = {}
        for name in expected:
            value = payload[name]
            if not torch.is_tensor(value):
                raise ValueError(f"checkpoint observation field {name!r} is not a tensor")
            tensors[name] = value.detach().to(
                self.device if target_device is None else target_device
            )
        return ObservationFrame(**tensors)

    def _deserialize_transition(
        self, payload: dict, *, target_device: Optional[torch.device] = None
    ) -> Transition:
        if not isinstance(payload, dict):
            raise ValueError("checkpoint rollout transition must be a dictionary")
        expected = set(Transition.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("checkpoint rollout transition fields do not match")
        tensors = {}
        for name in expected.difference({"observation_history"}):
            value = payload[name]
            if value is None and name in {
                "counterfactual_advantage",
                "counterfactual_components",
                "counterfactual_reference_action",
                "random_reference_mc_target",
                "natural_outcome_target",
                "causal_outcome_target",
                "improvement_prior",
                "improvement_target",
                "improvement_exact_mask",
                "improvement_advantage",
                "improvement_eta_floor",
                "improvement_base_action",
            }:
                tensors[name] = None
                continue
            if not torch.is_tensor(value):
                raise ValueError(f"checkpoint rollout field {name!r} is not a tensor")
            tensors[name] = value.detach().to(
                self.device if target_device is None else target_device
            )
        history_payload = payload["observation_history"]
        if not isinstance(history_payload, (list, tuple)) or not history_payload:
            raise ValueError("checkpoint transition requires a non-empty observation history")
        tensors["observation_history"] = tuple(
            self._deserialize_frame(item, target_device=target_device)
            for item in history_payload
        )
        return Transition(**tensors)

    def _deserialize_reference_episode(
        self, payload: dict
    ) -> ReferenceValueEpisode:
        if not isinstance(payload, dict) or set(payload) != {"frames", "targets"}:
            raise ValueError("checkpoint reference-value episode fields do not match")
        frames = tuple(
            self._deserialize_frame(item) for item in payload["frames"]
        )
        targets = payload["targets"]
        if (
            not frames
            or not torch.is_tensor(targets)
            or targets.ndim != 1
            or targets.numel() != len(frames)
            or not bool(torch.isfinite(targets).all())
        ):
            raise ValueError("checkpoint reference-value episode is invalid")
        return ReferenceValueEpisode(
            frames=frames,
            targets=targets.detach().to(self.device),
        )

    def _deserialize_actor_replay_episode(
        self, payload: dict
    ) -> ActorReplayEpisode:
        expected = {"frames", "target_scores", "exact_masks", "eta_floors"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("checkpoint actor-replay episode fields do not match")
        frames = tuple(
            self._deserialize_frame(item, target_device=torch.device("cpu"))
            for item in payload["frames"]
        )
        scores = payload["target_scores"]
        masks = payload["exact_masks"]
        floors = payload["eta_floors"]
        if (
            not frames
            or not torch.is_tensor(scores)
            or not torch.is_tensor(masks)
            or not torch.is_tensor(floors)
            or scores.shape != masks.shape
            or scores.ndim != 2
            or scores.shape[0] != len(frames)
            or floors.shape != (len(frames),)
            or not bool(torch.isfinite(scores).all())
            or not bool(torch.isfinite(floors).all())
            or bool(torch.any(floors <= 0.0))
        ):
            raise ValueError("checkpoint actor-replay episode is invalid")
        return ActorReplayEpisode(
            frames=frames,
            target_scores=scores.detach().cpu(),
            exact_masks=masks.detach().cpu(),
            eta_floors=floors.detach().cpu(),
        )

    def _validate_frame(self, frame: ObservationFrame) -> None:
        if frame.cell_features.shape != (self.num_cells, len(CELL_FEATURE_NAMES)):
            raise ValueError("checkpoint observation cell features have an incompatible shape")
        if frame.global_features.shape != (self.d_global,):
            raise ValueError("checkpoint observation global features have an incompatible shape")
        if frame.momentum_features.shape != (self.d_momentum,):
            raise ValueError("checkpoint momentum features have an incompatible shape")
        if frame.route_edge_index.ndim != 2 or frame.route_edge_index.shape[0] != 2:
            raise ValueError("checkpoint route edges have an incompatible shape")
        if frame.route_edge_weight.shape != (frame.route_edge_index.shape[1],):
            raise ValueError("checkpoint route-edge weights have an incompatible shape")
        if frame.route_edge_index.numel() and (
            int(frame.route_edge_index.min().item()) < 0
            or int(frame.route_edge_index.max().item()) >= self.num_cells
        ):
            raise ValueError("checkpoint route edge references an invalid region")
        if frame.candidate_cell_index.shape != (self.num_candidate_actions,):
            raise ValueError("checkpoint candidate-cell table has an incompatible shape")
        if frame.candidate_features.shape != (
            self.num_candidate_actions,
            self.d_candidate,
        ):
            raise ValueError("checkpoint candidate features have an incompatible shape")
        if frame.action_mask.shape != (self.num_candidate_actions,):
            raise ValueError("checkpoint observation action mask has an incompatible shape")
        if frame.simulation_time.numel() != 1:
            raise ValueError("checkpoint observation time must be scalar")
        if (
            frame.reference_debt.numel() != 1
            or not bool(torch.isfinite(frame.reference_debt).all())
            or float(frame.reference_debt.item()) < 0.0
        ):
            raise ValueError("checkpoint exact reference debt is invalid")
        for name in (
            "cell_features",
            "global_features",
            "route_edge_weight",
            "candidate_features",
            "momentum_features",
        ):
            if not bool(torch.isfinite(getattr(frame, name)).all()):
                raise ValueError(f"checkpoint observation {name!r} is non-finite")

    def _validate_rollout_state(self) -> None:
        if not 0 <= self.rollout_episode_count < self.rollout_episodes:
            raise ValueError("checkpoint rollout episode count is outside the update cycle")
        if self.rollout_episode_count == 0 and self.rollout_traj:
            raise ValueError("checkpoint has rollout transitions but zero rollout episodes")
        terminal_count = 0
        previous_time = None
        for transition in self.rollout_traj:
            if not transition.observation_history:
                raise ValueError("checkpoint transition has no recurrent observation history")
            for frame in transition.observation_history:
                self._validate_frame(frame)
                frame_time = int(frame.simulation_time.item())
                if previous_time is not None and frame_time <= previous_time:
                    raise ValueError("checkpoint recurrent observations are not time ordered")
                previous_time = frame_time
            action_mask = transition.observation_history[-1].action_mask
            action = int(transition.action.reshape(-1)[0].item())
            if (
                not 0 <= action < self.num_candidate_actions
                or not bool(action_mask[action])
            ):
                raise ValueError("checkpoint rollout action is infeasible")
            for name in (
                "log_probability",
                "value",
                "reward",
                "interval_cost",
                "random_reference_value",
            ):
                tensor = getattr(transition, name)
                if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"checkpoint rollout {name} is invalid")
            if float(transition.interval_cost.item()) < 0.0:
                raise ValueError("checkpoint rollout interval cost is negative")
            for name in ("value_components", "reward_components"):
                tensor = getattr(transition, name)
                if tensor.shape != (len(REWARD_COMPONENT_NAMES),) or not bool(
                    torch.isfinite(tensor).all()
                ):
                    raise ValueError(f"checkpoint rollout {name} is invalid")
            optional_shapes = {
                "random_reference_mc_target": (1,),
                "counterfactual_advantage": (1,),
                "counterfactual_components": (len(REWARD_COMPONENT_NAMES),),
                "counterfactual_reference_action": (1,),
                "natural_outcome_target": (len(MOMENTUM_FEATURE_NAMES),),
                "causal_outcome_target": (len(NMCC_OUTCOME_NAMES),),
            }
            for name, shape in optional_shapes.items():
                tensor = getattr(transition, name)
                if tensor is not None and (
                    tuple(tensor.shape) != shape
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError(f"checkpoint rollout {name} is invalid")
            if transition.counterfactual_reference_action is not None:
                reference_action = int(
                    transition.counterfactual_reference_action.item()
                )
                if (
                    not 0 <= reference_action < self.num_candidate_actions
                    or not bool(action_mask[reference_action])
                ):
                    raise ValueError(
                        "checkpoint rollout counterfactual reference is infeasible"
                    )
            if not torch.isclose(
                transition.value.reshape(()),
                transition.value_components.sum(),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("checkpoint scalar value disagrees with critic components")
            if not torch.isclose(
                transition.reward.reshape(()),
                transition.reward_components.sum(),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise ValueError("checkpoint scalar reward disagrees with reward components")
            if transition.done.numel() != 1 or float(transition.done.item()) not in (0.0, 1.0):
                raise ValueError("checkpoint rollout terminal flag is invalid")
            terminal_count += int(float(transition.done.item()) == 1.0)
            if float(transition.done.item()) == 1.0:
                previous_time = None
            if transition.elapsed_timesteps.numel() != 1 or int(transition.elapsed_timesteps.item()) <= 0:
                raise ValueError("checkpoint rollout duration is invalid")
        if terminal_count != self.rollout_episode_count:
            raise ValueError("checkpoint rollout does not contain one terminal per episode")
        if self.rollout_traj and float(self.rollout_traj[-1].done.item()) != 1.0:
            raise ValueError("checkpoint rollout must end at an episode boundary")

    @staticmethod
    def _safe_tensor(tensor: torch.Tensor, clamp: Optional[float] = None) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        if clamp is not None:
            tensor = torch.clamp(tensor, -float(clamp), float(clamp))
        return tensor

    def _safe_masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape != mask.shape:
            raise ValueError(
                f"logits and mask must share (batch, candidates), got {logits.shape} and {mask.shape}"
            )
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("Every decision must contain at least one feasible candidate action")
        logits = self._safe_tensor(logits, clamp=50.0)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    @staticmethod
    def _ppo_log_ratio(
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if new_logp.ndim != 1 or old_logp.ndim != 1 or new_logp.shape != old_logp.shape:
            raise ValueError(
                f"PPO log probabilities must be matching vectors, got {new_logp.shape} and {old_logp.shape}"
            )
        log_ratio = new_logp - old_logp
        if not torch.isfinite(log_ratio).all():
            raise FloatingPointError("PPO log-ratio contains non-finite values")
        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        return log_ratio, ratio


    @staticmethod
    def _normalized_categorical_entropy(
        distribution: torch.distributions.Categorical,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return entropy on a common zero-to-one feasible-action scale."""
        if action_mask.ndim != 2:
            raise ValueError("action_mask must have shape (batch, candidates)")
        feasible = action_mask.to(dtype=torch.float32).sum(dim=-1)
        raw_entropy = distribution.entropy()
        maximum = torch.log(feasible.clamp_min(1.0))
        return torch.where(
            feasible > 1.0,
            raw_entropy / maximum.clamp_min(1e-8),
            torch.zeros_like(raw_entropy),
        )

    def _policy_tensors(
        self,
        observation: RegionalObservation,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cell_features_np, global_features_np = observation.policy_features()
        candidate_features_np = observation.candidate_features()
        cell_features = torch.as_tensor(
            cell_features_np,
            dtype=torch.float32,
            device=self.device,
        )
        global_features = torch.as_tensor(
            global_features_np,
            dtype=torch.float32,
            device=self.device,
        )
        route_edge_index = torch.as_tensor(
            observation.route_edge_index,
            dtype=torch.long,
            device=self.device,
        )
        route_edge_weight = torch.as_tensor(
            observation.route_edge_weight,
            dtype=torch.float32,
            device=self.device,
        )
        candidate_cell_index = torch.as_tensor(
            observation.candidate_cell_indices,
            dtype=torch.long,
            device=self.device,
        )
        candidate_features = torch.as_tensor(
            candidate_features_np,
            dtype=torch.float32,
            device=self.device,
        )
        action_mask = torch.as_tensor(
            observation.action_mask,
            dtype=torch.bool,
            device=self.device,
        )
        return (
            cell_features,
            global_features,
            route_edge_index,
            route_edge_weight,
            candidate_cell_index,
            candidate_features,
            action_mask,
        )

    def _reference_debt_tensor(
        self,
        observation: RegionalObservation,
    ) -> torch.Tensor:
        value = float(
            np.asarray(
                observation.remaining_evacuation_debt_by_cell,
                dtype=np.float64,
            ).sum()
        )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Exact remaining evacuation debt must be non-negative")
        return torch.as_tensor(
            [value], dtype=torch.float32, device=self.device
        )

    def _graph(
        self,
        cell_features: torch.Tensor,
        global_features: torch.Tensor,
        *,
        batch: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        route_edge_index: Optional[torch.Tensor] = None,
        route_edge_weight: Optional[torch.Tensor] = None,
        candidate_cell_index: Optional[torch.Tensor] = None,
        candidate_features: Optional[torch.Tensor] = None,
    ):
        return fit_gnn(
            x_ped=cell_features[:, PED_FEATURE_SLICE],
            x_hazard=cell_features[:, HAZARD_FEATURE_SLICE],
            x_infra=cell_features[:, INFRA_FEATURE_SLICE],
            x_global=global_features,
            edge_index=self.edge_index if edge_index is None else edge_index,
            route_edge_index=route_edge_index,
            route_edge_weight=route_edge_weight,
            batch=batch,
            candidate_cell_index=candidate_cell_index,
            candidate_features=candidate_features,
        )

    def _graph_from_frame(self, frame: ObservationFrame):
        return self._graph(
            frame.cell_features.to(self.device),
            frame.global_features.to(self.device),
            route_edge_index=frame.route_edge_index.to(self.device),
            route_edge_weight=frame.route_edge_weight.to(self.device),
            candidate_cell_index=frame.candidate_cell_index.to(self.device),
            candidate_features=frame.candidate_features.to(self.device),
        )

    def _momentum_features(self, observation: RegionalObservation) -> torch.Tensor:
        """Return causal traffic/hazard velocities from the observation history."""
        values = self._momentum_array(
            self.previous_temporal_observation,
            observation,
        )
        if values.shape != (self.d_momentum,) or not np.isfinite(values).all():
            raise RuntimeError("Temporal momentum feature construction failed")
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _momentum_array(
        self,
        previous: Optional[RegionalObservation],
        current: RegionalObservation,
    ) -> np.ndarray:
        return RPI.momentum_array(
            previous,
            current,
            initial_population=self.initial_population,
            shelter_action_interval=self.shelter_action_interval,
        )

    def _momentum_features_between(
        self,
        previous: RegionalObservation,
        current: RegionalObservation,
    ) -> torch.Tensor:
        return torch.as_tensor(
            self._momentum_array(previous, current),
            dtype=torch.float32,
            device=self.device,
        )

    def _advance_recurrent_observation(
        self,
        observation: RegionalObservation,
    ) -> RecurrentPolicyStep:
        (
            cell_features,
            global_features,
            route_edge_index,
            route_edge_weight,
            candidate_cell_index,
            candidate_features,
            action_mask,
        ) = self._policy_tensors(observation)
        momentum_features = self._momentum_features(observation)
        frame = ObservationFrame(
            cell_features=cell_features.detach(),
            global_features=global_features.detach(),
            route_edge_index=route_edge_index.detach(),
            route_edge_weight=route_edge_weight.detach(),
            candidate_cell_index=candidate_cell_index.detach(),
            candidate_features=candidate_features.detach(),
            action_mask=action_mask.detach(),
            momentum_features=momentum_features.detach(),
            simulation_time=torch.as_tensor(
                [int(observation.simulation_time)],
                dtype=torch.long,
                device=self.device,
            ),
            reference_debt=self._reference_debt_tensor(observation).detach(),
        )
        graph = self._graph_from_frame(frame)
        with torch.no_grad():
            nmcc_output = self.policy.forward_nmcc_recurrent(
                graph,
                self.recurrent_state,
                momentum_features,
            )
            logits = nmcc_output.logits
        recurrent_state = nmcc_output.recurrent_state
        self.recurrent_state = tuple(state.detach() for state in recurrent_state)
        self.previous_temporal_observation = observation
        self.observation_frames_seen += 1
        return RecurrentPolicyStep(
            frame=frame,
            logits=logits,
            prior_logits=nmcc_output.prior_logits,
            value=nmcc_output.value,
            value_components=nmcc_output.value_components,
            natural_outcomes=nmcc_output.natural_outcomes,
            nmcc_continuation_value=nmcc_output.nmcc_continuation_value,
            random_reference_value=(
                nmcc_output.random_reference_value + frame.reference_debt
            ),
        )

    def _controller_logits(
        self,
        recurrent_step: "RecurrentPolicyStep",
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the neural actor's scores without a fallback controller."""
        return self._safe_masked_logits(
            recurrent_step.logits,
            action_mask.reshape(1, -1),
        )


    def _behavior_policy(
        self,
        recurrent_step: RecurrentPolicyStep,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Categorical]:
        """Return the scheduled categorical behavior policy.

        Temperature changes relative preferences; epsilon mixes in uniform
        support over feasible actions. Both decay smoothly and the exact mixed
        probability is stored for PPO rather than treating exploration as an
        off-policy side channel.
        """
        action_mask = recurrent_step.frame.action_mask
        if self.train_mode and self._stage_contract().train_random_reference:
            mask = action_mask.reshape(1, -1).to(torch.bool)
            probabilities = mask.to(recurrent_step.logits.dtype)
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            masked_logits = self._safe_masked_logits(
                torch.zeros_like(recurrent_step.logits),
                mask,
            )
            return (
                masked_logits,
                probabilities,
                torch.distributions.Categorical(probs=probabilities),
            )
        # Scheduled temperature and uniform mixing belong only to stochastic
        # data collection. Deterministic evaluation ranks and reports the raw
        # temperature-one actor fitted to the categorical rollout target.
        behavior_temperature = (
            self._current_action_temperature() if self.train_mode else 1.0
        )
        behavior_logits = self._controller_logits(
            recurrent_step,
            action_mask,
        ) / behavior_temperature
        masked_logits = self._safe_masked_logits(
            behavior_logits,
            action_mask.unsqueeze(0),
        )
        probabilities = self._exploratory_probabilities(
            masked_logits,
            action_mask.unsqueeze(0),
        )
        return (
            masked_logits,
            probabilities,
            torch.distributions.Categorical(probs=probabilities),
        )

    def _exploratory_probabilities(
        self,
        masked_logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the actor distribution with uniform feasible exploration."""
        mask = action_mask.to(torch.bool)
        logits = self._safe_masked_logits(masked_logits, mask)
        policy = torch.softmax(logits, dim=-1)
        epsilon = float(self._current_exploration_rate())
        if epsilon <= 0.0:
            return policy
        uniform = mask.to(policy.dtype)
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return (1.0 - epsilon) * policy + epsilon * uniform

    def _distribution_from_logits(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(
            probs=self._exploratory_probabilities(logits, action_mask)
        )

    def _sample_counterfactual_reference_action(
        self,
        recurrent_step: RecurrentPolicyStep,
    ) -> int:
        """Independently sample the NMCC reference shelter from behavior policy.

        Equality with the factual action is intentionally allowed. Excluding
        it would make the baseline depend on the chosen action and bias the
        policy gradient; an equal draw correctly produces zero location credit.
        """
        _, probabilities, _ = self._behavior_policy(recurrent_step)
        action = torch.multinomial(
            probabilities.detach(),
            num_samples=1,
            generator=self.action_generator,
        ).squeeze(-1)
        return int(action.item())

    def _select_rl(
        self,
        observation: RegionalObservation,
        *,
        deterministic: bool,
        recurrent_step: Optional[RecurrentPolicyStep] = None,
    ) -> PolicyDecision:
        if recurrent_step is None:
            recurrent_step = self._advance_recurrent_observation(observation)
        action_mask = recurrent_step.frame.action_mask
        masked_logits, behavior_probabilities, distribution = (
            self._behavior_policy(recurrent_step)
        )
        normalized_entropy = self._normalized_categorical_entropy(
            distribution,
            action_mask.unsqueeze(0),
        )
        self.behavior_entropies.append(
            float(normalized_entropy.detach().item())
        )
        if deterministic:
            action = torch.argmax(masked_logits, dim=-1)
        else:
            action = torch.multinomial(
                behavior_probabilities,
                num_samples=1,
                generator=self.action_generator,
            ).squeeze(-1)
        log_probability = distribution.log_prob(action)
        decision = PolicyDecision(
            action_index=int(action.item()),
            strategy="rl",
            log_probability=log_probability,
            value=recurrent_step.value,
            value_components=recurrent_step.value_components,
        )
        return decision

    def _select_action(
        self,
        observation: RegionalObservation,
        recurrent_step: Optional[RecurrentPolicyStep] = None,
    ) -> PolicyDecision:
        if self.deployment_strategy in {"rl", "rl_precommit"}:
            return self._select_rl(
                observation,
                deterministic=not self.train_mode,
                recurrent_step=recurrent_step,
            )
        if self.deployment_strategy == "objective_greedy":
            decision = self.objective_greedy_policy.select(
                observation,
                candidate_values=self._objective_greedy_values(observation),
                deterministic=True,
            )
        elif self.deployment_strategy == "heuristic":
            decision = self.heuristic_policy.select(observation, deterministic=True)
        elif self.deployment_strategy == "risk_reduction":
            decision = self.risk_reduction_policy.select(
                observation,
                deterministic=True,
            )
        elif self.deployment_strategy == "route_saving":
            decision = self.route_saving_policy.select(
                observation,
                deterministic=True,
            )
        elif self.deployment_strategy == "hazard_weighted":
            decision = self.hazard_weighted_policy.select(
                observation,
                deterministic=True,
            )
        elif self.deployment_strategy == "accessibility_deficit":
            decision = self.accessibility_deficit_policy.select(
                observation,
                deterministic=True,
            )
        elif self.deployment_strategy == "random":
            decision = self.random_policy.select(observation, deterministic=False)
        elif self.deployment_strategy == "initial_only":
            if len(self.static_precommit_schedule) != int(self.maximum_deployments):
                raise RuntimeError(
                    "initial_only requires a complete time-zero precommitment schedule"
                )
            if int(self.deployments_made) >= len(self.static_precommit_schedule):
                raise RuntimeError("initial_only exhausted its precommitted schedule")
            expected_osm_id, expected_cell = self.static_precommit_schedule[
                int(self.deployments_made)
            ]
            matches = [
                action
                for action in np.flatnonzero(observation.action_mask)
                if str(observation.candidate_osm_node_ids[int(action)])
                == str(expected_osm_id)
                and int(observation.candidate_cell_indices[int(action)])
                == int(expected_cell)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Precommitted shelter is not uniquely executable at its "
                    f"scheduled epoch: OSM={expected_osm_id!r}, cell={expected_cell}"
                )
            decision = PolicyDecision(
                action_index=int(matches[0]),
                strategy="initial_only",
            )
        else:
            raise RuntimeError(f"Strategy {self.deployment_strategy!r} cannot make dynamic decisions")
        return decision

    def _objective_greedy_values(
        self,
        observation: RegionalObservation,
    ) -> np.ndarray:
        """Exact complete-ledger myopic value of every feasible action.

        Every candidate is valued by the same positive full-episode score used
        for evaluation, applied to its post-installation cost-to-go. Rational
        active pedestrians receive capacity-valid route costs; panicked people
        and casualties retain unresolved time-to-safety debt; people already
        safe contribute zero. Past realized cost is common to every candidate
        and therefore need not be added to determine the ranking.

        Physical time does not advance and no continuation value is used, so
        held-out improvement over this benchmark is specifically evidence of
        useful foresight or state context beyond the complete-ledger immediate
        calculation.
        """
        feasible = np.flatnonzero(observation.action_mask)
        if feasible.size == 0:
            raise RuntimeError(
                "Objective-greedy scoring requires at least one feasible action"
            )
        cells = observation.candidate_cell_indices[feasible]
        if len(set(int(cell) for cell in cells)) != int(feasible.size):
            raise RuntimeError(
                "Objective-greedy scoring requires one unique candidate cell per action"
            )
        if self._objective_greedy_brancher is None:
            self._objective_greedy_brancher = CB.CounterfactualBrancher(
                self.core,
                horizon=0,
                reward_model=self.reward_model,
                initial_population=int(self.initial_population),
                episode_horizon=int(self.horizon),
            )
        before = CB.live_state_digest(self.core)
        effects = self._objective_greedy_brancher.candidate_effects(cells)
        after = CB.live_state_digest(self.core)
        if before != after:
            raise RuntimeError("Objective-greedy scoring mutated the live simulator")
        if len(effects) != int(feasible.size):
            raise RuntimeError(
                "Objective-greedy scoring did not return every feasible action"
            )
        denominator = float(self.initial_population) * float(self.horizon_minutes)
        if denominator <= 0.0:
            raise RuntimeError("Objective-greedy denominator must be positive")
        action_costs = np.asarray(
            [
                effect.acted.hazard_weighted_time_to_safety_cost
                for effect in effects
            ],
            dtype=np.float64,
        )
        values = np.full(observation.number_of_actions, -np.inf, dtype=np.float64)
        values[feasible] = 1.0 / (1.0 + action_costs / denominator)
        if not np.isfinite(values[feasible]).all():
            raise FloatingPointError(
                "Objective-greedy candidate values contain non-finite values"
            )
        return values

    @staticmethod
    def _observation_digest(observation: RegionalObservation) -> str:
        digest = hashlib.sha256()
        for array in (
            observation.active_by_cell,
            observation.mean_speed_by_cell,
            observation.congestion_speed_ratio_by_cell,
            observation.danger_by_cell,
            observation.remaining_capacity_by_cell,
            observation.shelter_utilization_by_cell,
            observation.deployable_capacity_by_cell,
            observation.candidate_count_by_cell,
            observation.network_node_count_by_cell,
            observation.mean_route_time_by_cell,
            observation.long_route_share_by_cell,
            observation.stable_wellness_by_cell,
            observation.exposed_wellness_by_cell,
            observation.panicked_wellness_by_cell,
            observation.forecast_danger_by_cell,
            observation.danger_momentum_by_cell,
            observation.forecast_escalation_probability_by_cell,
            observation.forecast_uncertainty_by_cell,
            observation.hazard_source_proximity_by_cell,
            observation.region_east_positions,
            observation.region_north_positions,
            observation.region_area_fractions,
            observation.spatial_edge_index,
            observation.route_edge_index,
            observation.route_edge_weight,
            observation.candidate_cell_indices,
            observation.candidate_capacities,
            observation.candidate_east_positions,
            observation.candidate_north_positions,
            observation.candidate_nearest_shelter_distances,
            observation.candidate_forecast_danger,
            observation.candidate_hazard_safety_margin,
            observation.candidate_reroutable_population,
            observation.candidate_risk_time_reduction,
            observation.remaining_evacuation_debt_by_cell,
            observation.candidate_assignment_gain,
            observation.action_mask.astype(np.uint8),
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        for candidate_id in observation.candidate_osm_node_ids:
            digest.update(candidate_id.encode("utf-8"))
            digest.update(b"\0")
        digest.update(
            np.asarray(
                [
                    observation.outcome.safe_completed,
                    observation.outcome.casualties,
                    observation.outcome.active_population,
                    observation.remaining_deployments,
                ],
                dtype=np.int64,
            ).tobytes()
        )
        digest.update(
            np.asarray(
                [
                    observation.network_load_share,
                    observation.time_step_minutes,
                    observation.wind_speed_fraction,
                    observation.wind_east_direction_fraction,
                    observation.wind_north_direction_fraction,
                    observation.hazard_spread_fraction,
                ],
                dtype=np.float64,
            ).tobytes()
        )
        return digest.hexdigest()

    def precommit_all(self) -> tuple[RegionalActionReceipt, ...]:
        """Use a frozen learned policy to commit its full shelter set at t=0.

        No simulator transition occurs between choices.  Candidate availability
        and installed capacity are updated after each choice, but pedestrian
        locations and the hazard state remain at their initial values.  This is
        the matched learned-selector comparator needed to isolate the value of
        sequential state feedback from the value of the neural scoring rule.
        """
        if self.train_mode or self.deployment_strategy != "rl_precommit":
            raise RuntimeError("precommit_all requires evaluation strategy='rl_precommit'")
        if self.pending is not None or self.last_simulation_time is not None:
            raise RuntimeError("RL precommitment must occur before the first transition")
        receipts = []
        while self.remaining_deployments > 0:
            observation_started = time.perf_counter()
            observation = self._enrich_policy_observation(
                self.observation_builder.build(
                    decision_index=self.decision_index,
                    simulation_time=0,
                    remaining_deployments=self.remaining_deployments,
                )
            )
            observation_ms = (time.perf_counter() - observation_started) * 1000.0
            if self.precommit_initial_observation_digest is None:
                self.precommit_initial_observation_digest = self._observation_digest(
                    observation
                )
            if not observation.has_feasible_action:
                break
            selection_started = time.perf_counter()
            decision = self._select_rl(
                observation,
                deterministic=True,
            )
            selection_ms = (time.perf_counter() - selection_started) * 1000.0
            reference = self.heuristic_policy.select(observation, deterministic=True)
            self.action_comparisons += 1
            self.action_agreements += int(
                int(decision.action_index) == int(reference.action_index)
            )
            reward = self._random_site_installation_reward(
                observation,
                int(decision.action_index),
            )
            execution_started = time.perf_counter()
            receipt = self.executor.execute(observation, decision)
            execution_ms = (time.perf_counter() - execution_started) * 1000.0
            self._accumulate_installation_reward(reward)
            self.deployment_latency_records.append(
                {
                    "observation_ms": observation_ms,
                    "policy_selection_ms": selection_ms,
                    "execution_ms": execution_ms,
                    "end_to_end_ms": observation_ms + selection_ms + execution_ms,
                }
            )
            receipts.append(receipt)
            self.deployments_made += 1
            self.last_deployment_time = 0
            self.deployment_times.append(0)
            self.decision_index += 1
        return tuple(receipts)

    def precommit_static_schedule(self) -> tuple[tuple[str, int], ...]:
        """Commit an open-loop shelter sequence from the exact time-zero state.

        The planner applies the objective-matched immediate rule repeatedly on
        a frozen physical state. Each planned installation updates only the
        planning copy's capacity reservations and routes, so later choices
        complement earlier choices without observing any future traffic or
        hazard realization. The live simulator and all random streams are
        restored before the first transition. Runtime execution must follow
        these exact candidate identities at the common decision epochs.
        """
        if self.train_mode or self.deployment_strategy != "initial_only":
            raise RuntimeError(
                "precommit_static_schedule requires evaluation strategy='initial_only'"
            )
        if self.pending is not None or self.last_simulation_time is not None:
            raise RuntimeError(
                "Static schedule precommitment must occur before the first transition"
            )
        if self.static_precommit_schedule:
            raise RuntimeError("Static shelter schedule was already committed")

        base = CB.capture(self.core, label="initial-only-static-plan")
        initial_final_unassigned = getattr(
            self.core,
            "finalInstallationUnassignedRationalPopulation",
            None,
        )
        before = CB.live_state_digest(self.core)
        schedule: list[tuple[str, int]] = []
        try:
            for planning_index in range(int(self.maximum_deployments)):
                observation = self._enrich_policy_observation(
                    self.observation_builder.build(
                        decision_index=planning_index,
                        simulation_time=0,
                        remaining_deployments=(
                            int(self.maximum_deployments) - planning_index
                        ),
                    )
                )
                if planning_index == 0:
                    self.precommit_initial_observation_digest = (
                        self._observation_digest(observation)
                    )
                if not observation.has_feasible_action:
                    raise RuntimeError(
                        "Static precommitment cannot fill the common shelter budget"
                    )
                decision = self.objective_greedy_policy.select(
                    observation,
                    candidate_values=self._objective_greedy_values(observation),
                    deterministic=True,
                )
                action = int(decision.action_index)
                candidate_osm_id = str(
                    observation.candidate_osm_node_ids[action]
                )
                cell_index = int(observation.candidate_cell_indices[action])
                self.executor.execute(observation, decision)
                schedule.append((candidate_osm_id, cell_index))
        finally:
            CB.restore(self.core, base)
            if hasattr(
                self.core,
                "finalInstallationUnassignedRationalPopulation",
            ):
                self.core.finalInstallationUnassignedRationalPopulation = (
                    initial_final_unassigned
                )

        if CB.live_state_digest(self.core) != before:
            raise RuntimeError(
                "Static precommitment failed to restore the live simulator"
            )
        if len(schedule) != int(self.maximum_deployments):
            raise RuntimeError(
                "Static precommitment did not produce the complete shelter schedule"
            )
        restored_observation = self._enrich_policy_observation(
            self.observation_builder.build(
                decision_index=0,
                simulation_time=0,
                remaining_deployments=int(self.maximum_deployments),
            )
        )
        if (
            self.precommit_initial_observation_digest
            != self._observation_digest(restored_observation)
        ):
            raise RuntimeError(
                "Static precommitment changed the time-zero observation"
            )
        self.static_precommit_schedule = tuple(schedule)
        return self.static_precommit_schedule

    def _decision_due(self, simulation_time: int) -> bool:
        # Population-matched experiments use an absolute timetable: with two
        # initial shelters and eight additions at a 15-minute interval, the
        # final capacity token must exist at t=120.  Legacy experiments retain
        # their historical retry-from-the-last-actual-installation behavior.
        if bool(
            getattr(
                self.core,
                "requirePopulationMatchedShelterCapacity",
                False,
            )
        ):
            scheduled_time = int(self.first_decision_time) + int(
                self.deployments_made
            ) * int(self.shelter_action_interval)
            return int(simulation_time) >= scheduled_time
        if self.last_deployment_time is None:
            return int(simulation_time) >= int(self.first_decision_time)
        return int(simulation_time) >= (
            int(self.last_deployment_time) + int(self.shelter_action_interval)
        )

    def _shelter_flow(self, shelter_id: int) -> int:
        shelters = getattr(getattr(self.core, "shelterDS", None), "shelterList", {})
        shelter = shelters.get(int(shelter_id)) if hasattr(shelters, "get") else None
        if shelter is None:
            raise RuntimeError(f"Installed shelter {shelter_id} is no longer available")
        flow = int(round(max(0.0, float(getattr(shelter, "shelterFlow", 0.0)))))
        return flow

    def _record_completed_interval(
        self,
        *,
        after: RegionalObservation,
        terminal: bool,
        random_reference_value_after: float = 0.0,
    ) -> ForwardRewardBreakdown:
        if self.pending is None:
            raise RuntimeError("No pending decision exists to complete")
        pending = self.pending
        terminal_residual = estimate_remaining_time_to_safety(
            self.core,
            episode_horizon=self.horizon,
            include_panicked=True,
            include_casualties=False,
        )
        realized_interval_cost = float(
            pending.controllable_active_person_time
            + pending.controllable_hazard_exposure_person_time
            + pending.controllable_expected_casualty_person_time
        )
        terminal_plain_residual_cost = float(
            terminal_residual.plain_person_minutes if terminal else 0.0
        )
        terminal_exposure_residual_cost = float(
            terminal_residual.hazard_exposure_person_minutes if terminal else 0.0
        )
        terminal_residual_cost = (
            terminal_plain_residual_cost + terminal_exposure_residual_cost
        )
        terminal_clearance_tail_cost = 0.0
        if terminal:
            tail_weight = float(
                getattr(self.core, "clearanceTailExtraWeight", 0.0)
            )
            if tail_weight > 0.0:
                terminal_clearance_tail_cost = float(
                    tail_weight
                    * estimate_clearance_tail(
                        self.core,
                        episode_horizon=self.horizon,
                        initial_population=self.initial_population,
                        remaining=terminal_residual,
                    ).tail_person_minutes
                )
        complete_interval_cost = (
            realized_interval_cost
            + terminal_residual_cost
            + terminal_clearance_tail_cost
        )
        decision_discount = self.gamma ** max(1, int(pending.elapsed_timesteps))
        reward_denominator = float(
            self.initial_population * self.horizon_minutes
        )
        normalized_interval_cost = complete_interval_cost / reward_denominator
        reference_before = float(pending.random_reference_value_before)
        reference_after = float(
            0.0 if terminal else random_reference_value_after
        )
        if self.policy is None:
            learning_reward = 0.0
        else:
            learning_reward = (
                reference_before
                - decision_discount * reference_after
                - normalized_interval_cost
            )
        learning_reward_scaled = learning_reward * self.learning_reward_scale
        if not np.isfinite(learning_reward):
            raise FloatingPointError("Random-reference learning reward is non-finite")
        forward_breakdown = ForwardRewardBreakdown(
            reward=float(learning_reward),
            random_reference_value_before=reference_before,
            random_reference_value_after=reference_after,
            plain_interval_cost=float(pending.controllable_active_person_time),
            exposure_interval_cost=float(
                pending.controllable_hazard_exposure_person_time
            ),
            expected_casualty_interval_cost=float(
                pending.controllable_expected_casualty_person_time
            ),
            terminal_plain_residual_cost=float(terminal_plain_residual_cost),
            terminal_exposure_residual_cost=float(
                terminal_exposure_residual_cost
            ),
            terminal_clearance_tail_cost=float(
                terminal_clearance_tail_cost
            ),
            decision_discount=float(decision_discount),
        )
        self.episode_return += float(learning_reward)
        if not np.isfinite(self.episode_random_reference_initial_value):
            self.episode_random_reference_initial_value = float(
                forward_breakdown.random_reference_value_before
            )
        self.episode_random_reference_cost += float(normalized_interval_cost)
        self.episode_controllable_interval_cost += complete_interval_cost
        self.episode_reference_potential_reduction += float(
            forward_breakdown.reference_potential_reduction
        )

        if self.train_mode:
            if any(
                tensor is None
                for tensor in (
                    pending.decision.log_probability,
                    pending.decision.value,
                    pending.decision.value_components,
                )
            ):
                raise RuntimeError("RL decision is missing policy-owned transition fields")
            if not pending.observation_history:
                raise RuntimeError("RL decision is missing its recurrent observation history")
            counterfactual_tensor = None
            counterfactual_components_tensor = None
            natural_outcome_tensor = None
            causal_outcome_tensor = None
            if self.counterfactual_credit and np.isfinite(
                pending.reference_return
            ):
                if (
                    pending.factual_component_return is None
                    or pending.factual_bootstrap_components is None
                    or pending.factual_outcome_target is None
                    or pending.reference_outcome_target is None
                ):
                    raise RuntimeError(
                        "NMCC transition closed before its matched factual target was frozen"
                    )
                counterfactual_value, counterfactual_components = (
                    self._counterfactual_advantage_for(
                    pending,
                    factual_components=pending.factual_component_return,
                    factual_bootstrap_components=(
                        pending.factual_bootstrap_components
                    ),
                    elapsed=max(1, int(pending.counterfactual_steps)),
                    )
                )
                if counterfactual_value is not None:
                    counterfactual_tensor = torch.as_tensor(
                        [counterfactual_value],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    counterfactual_components_tensor = torch.as_tensor(
                        counterfactual_components,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    natural_outcome_tensor = torch.as_tensor(
                        pending.reference_outcome_target,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    causal_outcome_tensor = torch.as_tensor(
                        pending.factual_outcome_target
                        - pending.reference_outcome_target,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.counterfactual_records.append(
                        {
                            "decision_index": int(self.decision_index),
                            "chosen_action": int(
                                pending.decision.action_index
                            ),
                            "chosen_cell": int(pending.receipt.executed_cell),
                            "reference_action": int(
                                pending.reference_action
                            ),
                            "reference_cell": int(pending.reference_cell),
                            "factual_return": float(
                                pending.factual_component_return.sum()
                            ),
                            "reference_return": float(
                                pending.reference_return
                            ),
                            "counterfactual_advantage": float(counterfactual_value),
                            "remaining_risk_time_reduction_difference": float(
                                counterfactual_components[0]
                                * self.initial_population
                                * self.horizon_minutes
                                / self.learning_reward_scale
                            ),
                            "casualties_avoided": float(
                                -(
                                    pending.factual_outcome_target[1]
                                    - pending.reference_outcome_target[1]
                                )
                                * self.initial_population
                            ),
                            "counterfactual_horizon": int(
                                pending.counterfactual_steps
                            ),
                        }
                    )
            if self._stage_contract().train_dynamics:
                # Candidate-conditioned Stage-E target: the observed change in
                # evacuation traffic and hazard over exactly one installation
                # interval.  These are the same six bounded momentum rates the
                # deployed recurrent policy observes, not simulator-private
                # latent state or a heuristic score.
                natural_outcome_tensor = torch.as_tensor(
                    self._momentum_array(pending.observation, after),
                    dtype=torch.float32,
                    device=self.device,
                )
            self.traj.append(
                Transition(
                    observation_history=pending.observation_history,
                    action=torch.as_tensor(
                        [pending.decision.action_index],
                        dtype=torch.long,
                        device=self.device,
                    ),
                    log_probability=pending.decision.log_probability.detach().reshape(1),
                    value=pending.decision.value.detach().reshape(1),
                    value_components=(
                        pending.decision.value_components.detach().reshape(-1)
                    ),
                    reward=torch.as_tensor(
                        [learning_reward_scaled],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    reward_components=torch.as_tensor(
                        [learning_reward_scaled],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    interval_cost=torch.as_tensor(
                        [normalized_interval_cost],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    random_reference_value=torch.as_tensor(
                        [reference_before],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    random_reference_mc_target=(
                        torch.as_tensor(
                            [pending.random_reference_mc_target],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        if np.isfinite(pending.random_reference_mc_target)
                        else None
                    ),
                    done=torch.as_tensor(
                        [float(terminal)],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    elapsed_timesteps=torch.as_tensor(
                        [max(1, pending.elapsed_timesteps)],
                        dtype=torch.long,
                        device=self.device,
                    ),
                    counterfactual_advantage=counterfactual_tensor,
                    counterfactual_components=counterfactual_components_tensor,
                    counterfactual_reference_action=(
                        torch.as_tensor(
                            [pending.reference_action],
                            dtype=torch.long,
                            device=self.device,
                        )
                        if counterfactual_tensor is not None
                        else None
                    ),
                    natural_outcome_target=natural_outcome_tensor,
                    causal_outcome_target=causal_outcome_tensor,
                    **self._improvement_transition_fields(pending),
                )
            )
        self.pending = None
        return forward_breakdown

    def _improvement_transition_fields(self, pending: "PendingDecision") -> dict:
        """Tensor fields for a transition's NMCC-PI record, or all ``None``."""
        record = pending.policy_improvement
        names = (
            "improvement_prior",
            "improvement_target",
            "improvement_exact_mask",
            "improvement_advantage",
            "improvement_eta_floor",
            "improvement_base_action",
        )
        if record is None:
            return {name: None for name in names}
        sources = (
            "prior",
            "target",
            "exact_mask",
            "exact_advantage",
            "target_eta_floor",
            "base_action",
        )
        return {
            name: (
                None
                if record.get(source) is None
                else torch.as_tensor(
                    np.asarray(record.get(source), dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            for name, source in zip(names, sources)
        }

    def step(self, *, simulation_time: int, is_terminal: bool = False) -> Dict[str, object]:
        """Observe one simulator boundary and possibly complete/select a decision."""
        if self.episode_done:
            result = self._empty_step_result()
            result["episode_return"] = float(self.episode_return)
            result["remaining_deployments"] = int(self.remaining_deployments)
            return result
        simulation_time = int(simulation_time)
        if self.last_simulation_time is not None and simulation_time <= self.last_simulation_time:
            raise ValueError("simulation_time must increase strictly")

        observation_started = time.perf_counter()
        observation = self.observation_builder.build(
            decision_index=self.decision_index,
            simulation_time=simulation_time,
            remaining_deployments=self.remaining_deployments,
        )
        observation_ms = (time.perf_counter() - observation_started) * 1000.0
        # Dynamic policies keep their agreed installation schedule even if
        # evacuation finishes early, so every comparison arm receives the
        # same ten-shelter budget. Once the dynamic budget is exhausted, an
        # empty network may terminate normally. The initial-only benchmark is
        # sequential too, but its complete site ordering was frozen from the
        # time-zero state.
        terminal = bool(is_terminal) or (
            observation.outcome.active_population <= 0
            and self.remaining_deployments <= 0
        )
        self._accumulate_policy_objective(
            observation,
            simulation_time,
            terminal=terminal,
        )
        if self.initial_observation_digest is None:
            self.initial_observation_digest = self._observation_digest(observation)
        if self.pending is not None and self.last_simulation_time is not None:
            elapsed = simulation_time - self.last_simulation_time
            self.pending.elapsed_timesteps += elapsed
        recurrent_step = None
        completed_action = -1
        dynamic_strategy = self.deployment_strategy in {
            "rl",
            "objective_greedy",
            "random",
            "heuristic",
            "risk_reduction",
            "route_saving",
            "hazard_weighted",
            "accessibility_deficit",
            "initial_only",
        }
        deployment_due = bool(
            not terminal
            and dynamic_strategy
            and self.remaining_deployments > 0
            and self._decision_due(simulation_time)
        )
        if deployment_due and bool(
            getattr(
                self.core,
                "requirePopulationMatchedShelterCapacity",
                False,
            )
        ):
            scheduled_time = int(self.first_decision_time) + int(
                self.deployments_made
            ) * int(self.shelter_action_interval)
            if int(simulation_time) != scheduled_time:
                raise RuntimeError(
                    "Population-matched shelter deployment missed its fixed "
                    f"decision time: expected t={scheduled_time}, observed "
                    f"t={simulation_time}"
                )
            if not observation.has_feasible_action:
                raise RuntimeError(
                    "Population-matched shelter deployment has no feasible "
                    f"candidate at scheduled time t={scheduled_time}"
                )
        decision_opportunity = bool(
            deployment_due and observation.has_feasible_action
        )
        if decision_opportunity and self.deployment_strategy in {
            "rl",
            "rl_precommit",
            "objective_greedy",
            "initial_only",
        }:
            observation = self._enrich_policy_observation(observation)
        # The policy acts only at decision epochs, so its recurrent state also
        # advances only at decision epochs.  A physical NMCC branch ending at
        # the next decision now receives exactly the same one-step temporal
        # update as the live policy instead of skipping fourteen hidden LSTM
        # updates that existed only on the factual trajectory.
        if (
            decision_opportunity
            and self.deployment_strategy in {"rl", "rl_precommit"}
        ):
            recurrent_step = self._advance_recurrent_observation(observation)
        decision_random_reference_value = float("nan")
        if (
            decision_opportunity
            and self.train_mode
            and self.nmcc_policy_improvement
            and self._stage_contract().train_actor
        ):
            # The reward definition is unchanged: V_mu is the complete
            # cost-to-go of uniform feasible siting.  Rollout policy iteration
            # evaluates it directly instead of asking an untrained value head
            # to supply the potential in the single policy-improvement stage.
            decision_random_reference_value = (
                self._uniform_random_reference_target()
            )
        # An action's SMDP transition ends at the next action actually taken,
        # or at the true physical terminal boundary. Exhausting the budget or
        # temporarily having no feasible site does not censor later outcomes.
        completed_reward = None
        if self.pending is not None and (terminal or decision_opportunity):
            completed_action = int(self.pending.decision.action_index)
            completed_reward = self._record_completed_interval(
                after=observation,
                terminal=terminal,
                random_reference_value_after=(
                    0.0
                    if terminal
                    else decision_random_reference_value
                    if np.isfinite(decision_random_reference_value)
                    else 0.0
                    if recurrent_step is None
                    else float(
                        recurrent_step.random_reference_value.detach().item()
                    )
                ),
            )

        selected_candidate = -1
        heuristic_candidate = -1
        selected_cell = -1
        heuristic_cell = -1
        receipt = None
        selection_ms = 0.0
        execution_ms = 0.0
        end_to_end_ms = 0.0
        if decision_opportunity:
            if self.pending is not None:
                raise RuntimeError(
                    "A new action became due before the previous SMDP transition closed"
                )

            random_reference_mc_target = float("nan")
            if np.isfinite(decision_random_reference_value):
                random_reference_mc_target = decision_random_reference_value
            elif self.train_mode and self._stage_contract().train_random_reference:
                random_reference_mc_target = (
                    self._uniform_random_reference_target()
                )

            selection_started = time.perf_counter()
            decision = self._select_action(observation, recurrent_step)
            selection_ms = (time.perf_counter() - selection_started) * 1000.0
            reference = self.heuristic_policy.select(observation, deterministic=True)
            heuristic_candidate = int(reference.action_index)
            selected_candidate = int(decision.action_index)
            heuristic_cell = int(
                observation.candidate_cell_indices[heuristic_candidate]
            )
            selected_cell = int(
                observation.candidate_cell_indices[selected_candidate]
            )
            self.action_comparisons += 1
            self.action_agreements += int(
                selected_candidate == heuristic_candidate
            )
            counterfactual_reference_action = -1
            counterfactual_reference_cell = -1
            reference_return = float("nan")
            reference_components = None
            reference_bootstrap_components = None
            reference_outcome_target = None
            if self.counterfactual_credit and self.train_mode:
                if recurrent_step is None:
                    raise RuntimeError(
                        "Exact candidate credit requires the behavior-policy state"
                    )
                counterfactual_reference_action = (
                    self._sample_counterfactual_reference_action(recurrent_step)
                )
                counterfactual_reference_cell = int(
                    observation.candidate_cell_indices[
                        counterfactual_reference_action
                    ]
                )
                # Before the factual shelter exists, run another shelter drawn
                # independently from the exact behavior distribution.
                (
                    reference_return,
                    reference_components,
                    reference_bootstrap_components,
                    reference_outcome_target,
                ) = self._collect_candidate_reference(
                    observation,
                    reference_action=counterfactual_reference_action,
                    simulation_time=simulation_time,
                )
            policy_improvement = None
            if (
                self.nmcc_policy_improvement
                and self.train_mode
                and recurrent_step is not None
                and self._stage_contract().train_actor
            ):
                # Also before the shelter exists: every branched cell, and the
                # executed one, is valued from the same pre-intervention state.
                policy_improvement = self._collect_rollout_policy_improvement(
                    observation,
                    recurrent_step,
                    decision,
                    simulation_time=simulation_time,
                )
            factual_effect = None
            factual_counterfactual_breakdown = None
            if self.counterfactual_credit and self.train_mode:
                factual_effect = self._ensure_brancher().paired_effect(
                    selected_cell
                )
                factual_counterfactual_breakdown = (
                    factual_effect.reward_breakdown(
                        self.reward_model,
                        initial_population=self.initial_population,
                        episode_horizon_minutes=self.horizon_minutes,
                    )
                )
            execution_started = time.perf_counter()
            receipt = self.executor.execute(observation, decision)
            execution_ms = (time.perf_counter() - execution_started) * 1000.0
            end_to_end_ms = observation_ms + selection_ms + execution_ms
            self.deployment_latency_records.append(
                {
                    "observation_ms": observation_ms,
                    "policy_selection_ms": selection_ms,
                    "execution_ms": execution_ms,
                    "end_to_end_ms": end_to_end_ms,
                }
            )
            self.deployments_made += 1
            self.last_deployment_time = simulation_time
            self.deployment_times.append(int(simulation_time))
            observation_history = (
                (recurrent_step.frame,)
                if self.train_mode and recurrent_step is not None
                else tuple()
            )
            if self.train_mode and not observation_history:
                raise RuntimeError("A recurrent RL action requires cached observations")
            self.pending = PendingDecision(
                observation=observation,
                decision=decision,
                receipt=receipt,
                observation_history=observation_history,
                random_reference_value_before=float(
                    decision_random_reference_value
                    if np.isfinite(decision_random_reference_value)
                    else 0.0
                    if recurrent_step is None
                    else recurrent_step.random_reference_value.detach().item()
                ),
                random_reference_mc_target=random_reference_mc_target,
                reference_action=counterfactual_reference_action,
                reference_cell=counterfactual_reference_cell,
                reference_return=reference_return,
                reference_component_return=reference_components,
                reference_bootstrap_components=(
                    reference_bootstrap_components
                ),
                reference_outcome_target=reference_outcome_target,
                counterfactual_steps=int(
                    0
                    if factual_effect is None
                    else max(
                        factual_effect.acted.steps,
                        factual_effect.waited.steps,
                    )
                ),
                factual_component_return=(
                    factual_counterfactual_breakdown.raw_component_vector()
                    * self.learning_reward_scale
                    if factual_effect is not None
                    else None
                ),
                factual_outcome_target=(
                    self._normalized_outcome_target(
                        before=factual_effect.acted.start_outcome,
                        after=factual_effect.acted.end_outcome,
                        active_person_time=(
                            factual_effect.acted.active_person_time
                        ),
                        hazard_exposure_person_time=(
                            factual_effect.acted.hazard_exposure_person_time
                        ),
                    )
                    if factual_effect is not None
                    else None
                ),
                factual_bootstrap_components=(
                    np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float64)
                    if factual_effect is not None
                    else None
                ),
                recurrent_state_at_decision=(
                    tuple(state.detach() for state in self.recurrent_state)
                    if self.counterfactual_credit and self.recurrent_state is not None
                    else None
                ),
                policy_improvement=policy_improvement,
            )
            self.decision_index += 1

        self.last_simulation_time = simulation_time
        self.episode_done = terminal
        result = self._empty_step_result()
        result.update(
            {
                "reward": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else float(completed_reward.reward)
                ),
                "reward_norm": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else float(completed_reward.reward)
                ),
                "reward_safe": 0.0,
                "reward_casualty": 0.0,
                "reward_evacuation_time": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else -float(completed_reward.plain_interval_cost)
                    / (self.initial_population * self.horizon_minutes)
                ),
                "reward_hazard_exposure": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else -float(completed_reward.exposure_interval_cost)
                    / (self.initial_population * self.horizon_minutes)
                ),
                "reward_reference_potential": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else float(completed_reward.reference_potential_reduction)
                ),
                "reward_terminal_residual": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else -float(
                        completed_reward.terminal_plain_residual_cost
                        + completed_reward.terminal_exposure_residual_cost
                    )
                    / (self.initial_population * self.horizon_minutes)
                ),
                "reward_terminal_clearance": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else -float(completed_reward.terminal_clearance_tail_cost)
                    / (self.initial_population * self.horizon_minutes)
                ),
                "reward_risk_time": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else float(completed_reward.reward)
                ),
                "reward_remaining_risk_time_reduction": (
                    0.0
                    if completed_reward is None or self.policy is None
                    else float(completed_reward.reward)
                ),
                "reward_shelter_service": 0.0,
                "new_safe_completions": 0,
                "new_casualties": 0,
                "attributed_shelter_service": 0,
                "remaining_risk_time_before": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.random_reference_value_before)
                ),
                "remaining_risk_time_after": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.random_reference_value_after)
                ),
                "raw_remaining_risk_time_reduction": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.reference_potential_reduction)
                ),
                "remaining_risk_time_reduction": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.reward)
                ),
                "plain_time_wait": 0.0,
                "plain_time_action": 0.0,
                "raw_plain_time_reduction": 0.0,
                "hazard_exposure_wait": 0.0,
                "hazard_exposure_action": 0.0,
                "raw_hazard_exposure_reduction": 0.0,
                "active_person_time": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.plain_interval_cost)
                ),
                "hazard_exposure_person_time": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.exposure_interval_cost)
                ),
                "terminal_residual_person_time": (
                    0.0
                    if completed_reward is None
                    else float(
                        completed_reward.terminal_plain_residual_cost
                        + completed_reward.terminal_exposure_residual_cost
                    )
                ),
                "clearance_tail_person_time": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.terminal_clearance_tail_cost)
                ),
                "risk_weighted_person_time": (
                    0.0
                    if completed_reward is None
                    else float(completed_reward.interval_cost)
                ),
                "decision_made": int(receipt is not None),
                "selected_candidate": selected_candidate,
                "heuristic_candidate": heuristic_candidate,
                "selected_cell": selected_cell,
                "heuristic_cell": heuristic_cell,
                "completed_action": completed_action,
                "added_shelters": int(receipt is not None),
                "installed_shelter_id": -1 if receipt is None else int(receipt.shelter_id),
                "candidate_osm_node_id": (
                    "" if receipt is None else str(receipt.candidate_osm_node_id)
                ),
                "candidate_x_m": (
                    float("nan") if receipt is None else float(receipt.candidate_x_m)
                ),
                "candidate_y_m": (
                    float("nan") if receipt is None else float(receipt.candidate_y_m)
                ),
                "candidate_cell_i": (
                    -1 if receipt is None else int(receipt.candidate_cell_i)
                ),
                "candidate_cell_j": (
                    -1 if receipt is None else int(receipt.candidate_cell_j)
                ),
                "capacity_added": 0.0 if receipt is None else float(receipt.capacity_added),
                "rerouted_population": 0 if receipt is None else int(receipt.rerouted_population),
                "feasible_candidates": int(observation.action_mask.sum()),
                "feasible_cells": int(
                    np.unique(
                        observation.candidate_cell_indices[observation.action_mask]
                    ).size
                ),
                "remaining_deployments": int(self.remaining_deployments),
                "episode_return": float(self.episode_return),
                "observation_build_latency_ms": (
                    observation_ms if receipt is not None else 0.0
                ),
                "policy_selection_latency_ms": selection_ms,
                "deployment_execution_latency_ms": execution_ms,
                "end_to_end_deployment_latency_ms": end_to_end_ms,
            }
        )
        if self.debug and receipt is not None:
            print(
                f"[REGIONAL DECISION] t={simulation_time} selected={selected_cell} "
                f"heuristic={heuristic_cell} reward={result['reward']:.6f} "
                f"safe={observation.outcome.safe_completed} casualty={observation.outcome.casualties}",
                flush=True,
            )
        return result

    @staticmethod
    def _empty_step_result() -> Dict[str, object]:
        return {
            "reward": 0.0,
            "reward_norm": 0.0,
            "reward_safe": 0.0,
            "reward_casualty": 0.0,
            "reward_evacuation_time": 0.0,
            "reward_hazard_exposure": 0.0,
            "reward_reference_potential": 0.0,
            "reward_terminal_residual": 0.0,
            "reward_terminal_clearance": 0.0,
            "reward_risk_time": 0.0,
            "reward_remaining_risk_time_reduction": 0.0,
            "reward_shelter_service": 0.0,
            "new_safe_completions": 0,
            "new_casualties": 0,
            "attributed_shelter_service": 0,
            "risk_weighted_person_time": 0.0,
            "active_person_time": 0.0,
            "hazard_exposure_person_time": 0.0,
            "terminal_residual_person_time": 0.0,
            "clearance_tail_person_time": 0.0,
            "remaining_risk_time_before": 0.0,
            "remaining_risk_time_after": 0.0,
            "raw_remaining_risk_time_reduction": 0.0,
            "remaining_risk_time_reduction": 0.0,
            "plain_time_wait": 0.0,
            "plain_time_action": 0.0,
            "raw_plain_time_reduction": 0.0,
            "hazard_exposure_wait": 0.0,
            "hazard_exposure_action": 0.0,
            "raw_hazard_exposure_reduction": 0.0,
            "decision_made": 0,
            "selected_candidate": -1,
            "heuristic_candidate": -1,
            "selected_cell": -1,
            "heuristic_cell": -1,
            "completed_action": -1,
            "added_shelters": 0,
            "installed_shelter_id": -1,
            "candidate_osm_node_id": "",
            "candidate_x_m": float("nan"),
            "candidate_y_m": float("nan"),
            "candidate_cell_i": -1,
            "candidate_cell_j": -1,
            "capacity_added": 0.0,
            "rerouted_population": 0,
            "feasible_cells": 0,
            "feasible_candidates": 0,
            "remaining_deployments": 0,
            "episode_return": 0.0,
            # The complete realised objective includes the terminal residual
            # and therefore does not exist at an intermediate simulator step.
            # It is populated exactly once by ``end_episode``.  Reusing the
            # accumulated learning reward here previously gave one column two
            # incompatible meanings and made progress plots misleading.
            "objective_episode_score": float("nan"),
            "observation_build_latency_ms": 0.0,
            "policy_selection_latency_ms": 0.0,
            "deployment_execution_latency_ms": 0.0,
            "end_to_end_deployment_latency_ms": 0.0,
        }

    def end_episode(self, *, finalize_rollout: bool = False) -> Dict[str, float]:
        if self.pending is not None:
            raise RuntimeError("Episode ended before the pending regional decision was finalized")
        remaining = estimate_remaining_time_to_safety(
            self.core,
            episode_horizon=self.horizon,
            include_panicked=True,
            include_casualties=False,
        )
        objective_terminal_plain = float(remaining.plain_person_minutes)
        objective_terminal_exposure = float(
            remaining.hazard_exposure_person_minutes
        )
        objective_plain = float(
            self.objective_active_person_time + objective_terminal_plain
        )
        objective_exposure = float(
            self.objective_hazard_exposure_person_time
            + objective_terminal_exposure
        )
        objective_expected_casualty = float(
            self.objective_expected_casualty_person_time
        )
        tail_weight = float(
            getattr(self.core, "clearanceTailExtraWeight", 0.0)
        )
        if tail_weight > 0.0:
            clearance_tail = estimate_clearance_tail(
                self.core,
                episode_horizon=self.horizon,
                initial_population=self.initial_population,
                remaining=remaining,
            )
            objective_clearance_tail = float(
                tail_weight * clearance_tail.tail_person_minutes
            )
            clearance_tail_mean = float(clearance_tail.tail_mean_minutes)
            clearance_t90 = float(clearance_tail.t90_minutes)
            clearance_time = float(clearance_tail.clearance_time_minutes)
        else:
            objective_clearance_tail = 0.0
            clearance_tail_mean = 0.0
            clearance_t90 = 0.0
            clearance_time = 0.0
        objective_mass = float(
            objective_plain
            + objective_exposure
            + objective_expected_casualty
            + objective_clearance_tail
        )
        objective_denominator = float(
            self.initial_population * self.horizon_minutes
        )
        if objective_denominator <= 0.0:
            raise RuntimeError("Realised objective denominator must be positive")
        objective_normalized_cost = objective_mass / objective_denominator
        # Evaluation is fundamentally a positive cost (lower is better).  For
        # consumers that require a higher-is-better return, expose its monotone
        # reciprocal rather than the old negated cost.  This score is reporting
        # only; the learner uses the explicit random-reference reward and
        # simulator-graded rollout targets above.
        objective_score = 1.0 / (1.0 + float(objective_normalized_cost))
        latency = {
            key: np.asarray(
                [record[key] for record in self.deployment_latency_records],
                dtype=float,
            )
            for key in (
                "observation_ms",
                "policy_selection_ms",
                "execution_ms",
                "end_to_end_ms",
            )
        }
        random_reference_initial_value = (
            0.0
            if not np.isfinite(self.episode_random_reference_initial_value)
            else float(self.episode_random_reference_initial_value)
        )
        random_reference_identity_return = (
            random_reference_initial_value
            - float(self.episode_random_reference_cost)
            if self.policy is not None
            else 0.0
        )
        reward_accounting_gap = float(
            self.episode_return - random_reference_identity_return
        )
        diagnostics = {
            "episode_return": float(self.episode_return),
            "learning_reward_scale": float(self.learning_reward_scale),
            "forward_learning_reward": float(self.episode_return),
            "random_site_advantage_diagnostic": float(
                self.episode_random_site_advantage
            ),
            "controllable_interval_cost": float(
                self.episode_controllable_interval_cost
            ),
            "random_reference_initial_value": random_reference_initial_value,
            "random_reference_realized_cost": float(
                self.episode_random_reference_cost
            ),
            "random_reference_identity_return": float(
                random_reference_identity_return
            ),
            "reference_potential_reduction": float(
                self.episode_reference_potential_reduction
            ),
            "remaining_risk_time_before": float(
                self.episode_remaining_risk_time_before
            ),
            "remaining_risk_time_after": float(
                self.episode_remaining_risk_time_after
            ),
            "raw_remaining_risk_time_reduction": float(
                self.episode_raw_remaining_risk_time_reduction
            ),
            "remaining_risk_time_reduction": float(
                self.episode_remaining_risk_time_reduction
            ),
            "plain_time_wait": float(self.episode_plain_time_wait),
            "plain_time_action": float(self.episode_plain_time_action),
            "raw_plain_time_reduction": float(
                self.episode_raw_plain_time_reduction
            ),
            "hazard_exposure_wait": float(
                self.episode_hazard_exposure_wait
            ),
            "hazard_exposure_action": float(
                self.episode_hazard_exposure_action
            ),
            "raw_hazard_exposure_reduction": float(
                self.episode_raw_hazard_exposure_reduction
            ),
            "realized_plain_time_to_safety": float(objective_plain),
            "realized_hazard_exposure_time_to_safety": float(
                objective_exposure
            ),
            "expected_casualty_person_time_cost": float(
                objective_expected_casualty
            ),
            "expected_casualties": float(
                getattr(
                    getattr(self.core, "pedDS", None),
                    "expectedCasualtyIntensityCumulative",
                    0.0,
                )
            ),
            "clearance_tail_person_time_cost": float(
                objective_clearance_tail
            ),
            "clearance_tail_mean_minutes": float(
                clearance_tail_mean
            ),
            "clearance_t90_minutes": float(clearance_t90),
            "clearance_time_minutes": float(clearance_time),
            "realized_terminal_plain_time": float(objective_terminal_plain),
            "realized_terminal_hazard_exposure_time": float(
                objective_terminal_exposure
            ),
            "realized_hazard_weighted_time_to_safety": float(objective_mass),
            "normalized_realized_hazard_weighted_time_to_safety": float(
                objective_normalized_cost
            ),
            "decisions": float(self.action_comparisons),
            "heuristic_agreement_rate": (
                float(self.action_agreements) / float(self.action_comparisons)
                if self.action_comparisons > 0
                else 0.0
            ),
            "behavior_policy_entropy": (
                float(np.mean(self.behavior_entropies))
                if self.behavior_entropies
                else 0.0
            ),
            "optimizer_updated": 0.0,
            "actor_policy_entropy_before": 0.0,
            "actor_policy_entropy_after": 0.0,
            "actor_policy_entropy_change": 0.0,
            "environment_dynamics_loss": 0.0,
            "random_reference_value_loss": 0.0,
            "random_reference_value_explained_variance": 0.0,
            "random_reference_value_bias": 0.0,
            "random_reference_replay_loss": 0.0,
            "random_reference_replay_explained_variance": 0.0,
            "random_reference_validation_loss": 0.0,
            "random_reference_validation_explained_variance": 0.0,
            "random_reference_replay_episodes": float(
                len(self.random_reference_replay)
            ),
            **{name: 0.0 for name in self.ROLLOUT_PI_DIAGNOSTIC_NAMES},
            **{name: 0.0 for name in self.LEARNER_FLOW_DIAGNOSTIC_NAMES},
            "rollout_pi_actor_enabled": float(
                self.nmcc_policy_improvement and self._actor_training_enabled()
            ),
            "policy_iterations_completed": float(self.actor_rollout_updates),
            "actor_gradient_steps": float(self.actor_optimizer_updates),
            "critic_gradient_steps": float(self.critic_optimizer_updates),
            "entropy_coefficient": float(self._current_entropy_coef()),
            "action_temperature": float(
                self._current_action_temperature()
            ),
            "exploration_rate": float(self._current_exploration_rate()),
            "training_stage_update_index": float(
                self.training_stage_update_index
            ),
            "training_stage_total_updates": float(
                self.training_stage_total_updates
            ),
            "training_stage_progress": float(
                self.training_stage_update_index
                / max(1, self.training_stage_total_updates - 1)
            ),
            "actor_gradient_norm": 0.0,
            "critic_gradient_norm": 0.0,
            "policy_loss": 0.0,
            "actor_residual_rms_before_update": 0.0,
            "actor_residual_rms_after_update": 0.0,
            "actor_residual_abs_max_after_update": 0.0,
            "critic_epochs_completed": 0.0,
            "actor_epochs_completed": 0.0,
            "epochs_completed": 0.0,
            "transitions": 0.0,
            "training_episode_has_decision": float(bool(self.traj)),
            "observation_frames_seen": float(self.observation_frames_seen),
            "reward_accounting_gap": reward_accounting_gap,
            "learning_rate": float(
                self.actor_optimizer.param_groups[0]["lr"]
                if self.actor_optimizer is not None
                else self.actor_lr
            ),
            "actor_learning_rate_before_update": float(
                self.actor_optimizer.param_groups[0]["lr"]
                if self.actor_optimizer is not None
                else self.actor_lr
            ),
            "actor_learning_rate": float(
                self.actor_optimizer.param_groups[0]["lr"]
                if self.actor_optimizer is not None
                else self.actor_lr
            ),
            "critic_learning_rate_before_update": float(
                self.critic_optimizer.param_groups[0]["lr"]
                if self.critic_optimizer is not None
                else self.critic_lr
            ),
            "critic_learning_rate": float(
                self.critic_optimizer.param_groups[0]["lr"]
                if self.critic_optimizer is not None
                else self.critic_lr
            ),
            "objective_episode_score": float(objective_score),
            "objective_normalized_cost": float(objective_normalized_cost),
            "objective_active_person_time": float(
                self.objective_active_person_time
            ),
            "objective_hazard_exposure_person_time": float(
                self.objective_hazard_exposure_person_time
            ),
            "objective_expected_casualty_person_time": float(
                self.objective_expected_casualty_person_time
            ),
            "objective_clearance_tail_person_time": float(
                objective_clearance_tail
            ),
            "objective_clearance_tail_mean_minutes": float(
                clearance_tail_mean
            ),
            "objective_clearance_t90_minutes": float(
                clearance_t90
            ),
            "objective_clearance_time_minutes": float(
                clearance_time
            ),
            "objective_terminal_residual_person_time": float(
                objective_terminal_plain + objective_terminal_exposure
            ),
            "objective_hazard_weighted_person_time_cost": float(
                objective_mass
            ),
            "precommit_decisions": float(
                self.deployments_made
                if self.deployment_strategy == "rl_precommit"
                else len(self.static_precommit_schedule)
                if self.deployment_strategy == "initial_only"
                else 0
            ),
            "deployment_latency_observations": float(
                len(self.deployment_latency_records)
            ),
            "mean_observation_build_latency_ms": float(
                np.mean(latency["observation_ms"])
                if latency["observation_ms"].size
                else 0.0
            ),
            "mean_policy_selection_latency_ms": float(
                np.mean(latency["policy_selection_ms"])
                if latency["policy_selection_ms"].size
                else 0.0
            ),
            "mean_deployment_execution_latency_ms": float(
                np.mean(latency["execution_ms"])
                if latency["execution_ms"].size
                else 0.0
            ),
            "mean_end_to_end_deployment_latency_ms": float(
                np.mean(latency["end_to_end_ms"])
                if latency["end_to_end_ms"].size
                else 0.0
            ),
            "p95_end_to_end_deployment_latency_ms": float(
                np.percentile(latency["end_to_end_ms"], 95)
                if latency["end_to_end_ms"].size
                else 0.0
            ),
            "maximum_end_to_end_deployment_latency_ms": float(
                np.max(latency["end_to_end_ms"])
                if latency["end_to_end_ms"].size
                else 0.0
            ),
        }
        if self.train_mode:
            # An episode with no structurally executable candidate can
            # legitimately produce no decision. It contributes outcome
            # diagnostics but no fabricated policy transition.
            if self.traj:
                self.rollout_traj.extend(self.traj)
                self.rollout_episode_count += 1
            if self.rollout_traj and (
                self.rollout_episode_count >= self.rollout_episodes
                or bool(finalize_rollout)
            ):
                diagnostics.update(self._run_learner_update(self.rollout_traj))
                diagnostics["optimizer_updated"] = 1.0
                diagnostics["rollout_flushed_at_campaign_end"] = float(
                    bool(finalize_rollout)
                    and self.rollout_episode_count < self.rollout_episodes
                )
                self.rollout_traj.clear()
                self.rollout_episode_count = 0
            self.episodes_completed += 1
            diagnostics["rollout_episodes_pending"] = float(self.rollout_episode_count)
            diagnostics["rollout_transitions_pending"] = float(len(self.rollout_traj))
            self._save_checkpoint()
            self._append_learner_diagnostics(diagnostics)
        self.traj.clear()
        self.recurrent_state = None
        self.previous_temporal_observation = None
        if self._counterfactual_brancher is not None:
            self._counterfactual_brancher.close()
        self.last_training_diagnostics = dict(diagnostics)
        return diagnostics

    @staticmethod
    def _group_episode_indices(
        transitions: Sequence[Transition],
    ) -> list[list[int]]:
        episodes: list[list[int]] = []
        current: list[int] = []
        for index, transition in enumerate(transitions):
            current.append(index)
            if float(transition.done.item()) == 1.0:
                episodes.append(current)
                current = []
        if current:
            raise RuntimeError("Recurrent learner batch ends inside an episode")
        if not episodes:
            raise RuntimeError("Recurrent learner batch contains no complete episode")
        return episodes

    def _iter_episode_minibatches(
        self,
        episodes: Sequence[Sequence[int]],
    ) -> Iterator[list[list[int]]]:
        """Yield whole-episode minibatches bounded by decision count.

        Episode integrity takes precedence over the nominal transition limit;
        a single long episode is never split because doing so would require a
        burn-in state and would weaken temporal credit assignment.
        """
        order = torch.randperm(
            len(episodes),
            generator=self.optimization_generator,
            device="cpu",
        ).tolist()
        batch: list[list[int]] = []
        transition_count = 0
        for episode_index in order:
            episode = list(episodes[episode_index])
            if batch and transition_count + len(episode) > self.minibatch_size:
                yield batch
                batch = []
                transition_count = 0
            batch.append(episode)
            transition_count += len(episode)
        if batch:
            yield batch

    def _evaluate_recurrent_sequences(
        self,
        transitions: Sequence[Transition],
        episode_groups: Sequence[Sequence[int]],
        *,
        action_temperature: float,
    ) -> tuple:
        """Replay complete decision histories in their original order."""
        flat_indices: list[int] = []
        logits = []
        scalar_values = []
        component_values = []
        residuals = []
        natural_outcomes = []
        continuation_values = []
        random_reference_values = []
        for episode in episode_groups:
            recurrent_state = None
            for transition_index in episode:
                transition = transitions[int(transition_index)]
                output = None
                for frame in transition.observation_history:
                    output = self.policy.forward_nmcc_recurrent(
                        self._graph_from_frame(frame),
                        recurrent_state,
                        frame.momentum_features.to(self.device),
                    )
                    recurrent_state = output.recurrent_state
                if output is None:
                    raise RuntimeError("A recurrent decision contains no observations")
                step_logits = output.logits / float(action_temperature)
                if step_logits.size(0) != 1:
                    raise RuntimeError("Stored recurrent frames must represent one episode")
                flat_indices.append(int(transition_index))
                logits.append(step_logits)
                scalar_values.append(output.value.reshape(1))
                component_values.append(output.value_components.reshape(1, -1))
                residuals.append(output.learned_residual)
                natural_outcomes.append(output.natural_outcomes)
                continuation_values.append(
                    output.nmcc_continuation_value.reshape(1)
                )
                random_reference_values.append(
                    output.random_reference_value.reshape(1)
                    + frame.reference_debt.to(self.device).reshape(1)
                )
        return (
            torch.as_tensor(flat_indices, dtype=torch.long, device=self.device),
            torch.cat(logits, dim=0),
            torch.cat(scalar_values, dim=0),
            torch.cat(component_values, dim=0),
            torch.cat(residuals, dim=0),
            torch.cat(natural_outcomes, dim=0),
            torch.cat(continuation_values, dim=0),
            torch.cat(random_reference_values, dim=0),
        )

    def _evaluate_nmcc_actor_sequences(
        self,
        transitions: Sequence[Transition],
        episode_groups: Sequence[Sequence[int]],
        *,
        action_temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay recurrent histories through only the NMCC actor path."""
        flat_indices: list[int] = []
        logits = []
        residuals = []
        for episode in episode_groups:
            recurrent_state = None
            for transition_index in episode:
                transition = transitions[int(transition_index)]
                output = None
                for frame in transition.observation_history:
                    output = self.policy.forward_nmcc_actor_recurrent(
                        self._graph_from_frame(frame),
                        recurrent_state,
                        frame.momentum_features.to(self.device),
                    )
                    recurrent_state = output[3]
                if output is None:
                    raise RuntimeError("A recurrent decision contains no observations")
                step_logits, _, learned_residual, _ = output
                step_logits = step_logits / float(action_temperature)
                if step_logits.size(0) != 1:
                    raise RuntimeError("Stored recurrent frames must represent one episode")
                flat_indices.append(int(transition_index))
                logits.append(step_logits)
                residuals.append(learned_residual)
        return (
            torch.as_tensor(flat_indices, dtype=torch.long, device=self.device),
            torch.cat(logits, dim=0),
            torch.cat(residuals, dim=0),
        )

    def _actor_distribution_diagnostics(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
        action_masks: torch.Tensor,
    ) -> dict[str, float]:
        """Measure the raw trainable actor once, without behavior transforms.

        Behavior entropy is logged when actions are sampled and includes the
        declared temperature and uniform exploration mixture.  These update
        diagnostics instead use the temperature-1 actor that the categorical
        M-step fits and deterministic evaluation ranks.
        """
        with torch.no_grad():
            selected, logits, residual = self._evaluate_nmcc_actor_sequences(
                transitions,
                episodes,
                action_temperature=1.0,
            )
            expected = torch.arange(len(transitions), device=self.device)
            if not torch.equal(selected, expected):
                raise RuntimeError("Actor diagnostic replay changed transition order")
            masked_logits = self._safe_masked_logits(logits, action_masks)
            distribution = torch.distributions.Categorical(logits=masked_logits)
            entropy = self._normalized_categorical_entropy(
                distribution,
                action_masks,
            ).mean()
            feasible_residual = residual[action_masks.to(torch.bool)]
            if feasible_residual.numel() == 0:
                raise RuntimeError("Actor diagnostics found no feasible residuals")
            residual_rms = torch.sqrt(feasible_residual.square().mean())
            residual_abs_max = feasible_residual.abs().max()
        return {
            "entropy": float(entropy.item()),
            "residual_rms": float(residual_rms.item()),
            "residual_abs_max": float(residual_abs_max.item()),
        }

    # -- NMCC policy improvement: exact within-state targets ---------------

    def _actor_prior_probabilities(
        self,
        recurrent_step: "RecurrentPolicyStep",
        action_mask: torch.Tensor,
    ) -> np.ndarray:
        """Trainable actor distribution used by both NMCC E- and M-steps.

        Temperature and uniform mixing are data-collection exploration. They
        must not enter the policy-improvement prior because the actor M-step
        fits the raw masked logits and deterministic evaluation ranks those
        same logits.
        """
        with torch.no_grad():
            mask = action_mask.reshape(1, -1).to(torch.bool)
            logits = self._controller_logits(recurrent_step, action_mask)
            probabilities = torch.softmax(
                self._safe_masked_logits(logits, mask), dim=-1
            ).reshape(-1)
        return probabilities.detach().cpu().numpy().astype(np.float64)

    def _collect_rollout_policy_improvement(
        self,
        observation: RegionalObservation,
        recurrent_step: "RecurrentPolicyStep",
        decision: PolicyDecision,
        *,
        simulation_time: int,
    ) -> dict:
        """Grade a strong-plus-uniform panel to the horizon under the frozen actor.

        Early high-impact decisions are exhaustive. Later panels always keep
        the objective-greedy and executed actions, add the strongest remaining
        base-score contenders, and reserve slots for uniform feasible extras.
        The current actor is copied once into the worker batch. Every tested
        cell starts from the same exact simulator snapshot, receives the same
        keyed future tapes, and uses that frozen actor for all later shelter
        decisions.  Consequently no learned critic or endpoint estimate ever
        grades the actor's own action target.
        """
        mask = np.asarray(observation.action_mask, dtype=bool)
        feasible = np.flatnonzero(mask)
        if feasible.size < 2:
            raise RuntimeError(
                "Rollout policy improvement requires two feasible actions"
            )
        chosen_action = int(decision.action_index)
        if chosen_action not in set(feasible.tolist()):
            raise RuntimeError("Executed rollout action is infeasible")

        base_action = int(
            feasible[
                int(
                    np.argmax(
                        np.asarray(
                            observation.candidate_assignment_gain,
                            dtype=np.float64,
                        )[feasible]
                    )
                )
            ]
        )
        episode_seed = int(getattr(self.core, "scenario_seed", self.policy_seed))
        future_seeds = tuple(
            NPI.tape_seed(
                episode_seed ^ 0x524F4C4C,
                int(self.deployments_made),
                tape_index,
            )
            for tape_index in range(int(self.nmcc_pi_tapes))
        )
        frozen_policy = RPI.FrozenResidualPolicy.capture(
            self.policy,
            self.edge_index,
            initial_population=self.initial_population,
            shelter_action_interval=self.shelter_action_interval,
            previous_observation=observation,
            recurrent_state=self.recurrent_state,
        )
        brancher = self._ensure_brancher()
        snapshot = CB.capture(self.core, label="rollout-policy-iteration")
        before = CB.live_state_digest(self.core)
        started = time.perf_counter()
        exhaustive = bool(
            int(self.deployments_made) < int(self.nmcc_pi_exhaustive_decisions)
            or int(self.nmcc_pi_max_branches) <= 0
            or feasible.size <= int(self.nmcc_pi_max_branches)
        )
        if exhaustive:
            exact_actions = feasible
            uniform_panel_actions = 0
        else:
            panel_seed = NPI.tape_seed(
                episode_seed ^ 0x50414E45,
                int(self.deployments_made),
                0,
            )
            exact_actions = RPI.select_rollout_panel(
                feasible,
                np.asarray(
                    observation.candidate_assignment_gain,
                    dtype=np.float64,
                )[feasible],
                max_branches=int(self.nmcc_pi_max_branches),
                must_include=(chosen_action, base_action),
                uniform_extras=2,
                rng=np.random.default_rng(panel_seed),
            )
            uniform_panel_actions = min(
                2,
                int(self.nmcc_pi_max_branches) - len({chosen_action, base_action}),
            )
        exact_cells = np.asarray(
            observation.candidate_cell_indices[exact_actions], dtype=np.int64
        )
        costs = brancher.rollout_policy_costs(
            exact_cells,
            snapshot=snapshot,
            future_seeds=future_seeds,
            decision_index=int(self.deployments_made),
            remaining_deployments=int(self.remaining_deployments),
            maximum_deployments=int(self.maximum_deployments),
            frozen_policy=frozen_policy,
        )
        branch_seconds = time.perf_counter() - started
        if CB.live_state_digest(self.core) != before:
            raise RuntimeError("Rollout grading mutated the live simulator")
        fitted = RPI.estimate_rollout_costs(costs)
        prior = self._actor_prior_probabilities(
            recurrent_step, recurrent_step.frame.action_mask
        )
        target_eta_floor = max(
            float(self.nmcc_pi_eta_min),
            float(fitted.paired_contrast_standard_error),
        )
        improvement = NPI.improvement_target(
            prior,
            mask,
            exact_actions,
            -fitted.mean_cost,
            epsilon=float(self.nmcc_pi_epsilon),
            eta_min=target_eta_floor,
        )
        exact_mask = np.zeros(self.num_candidate_actions, dtype=np.float32)
        exact_mask[exact_actions] = 1.0
        branch_best = int(exact_actions[int(np.argmin(fitted.mean_cost))])
        mean_tape_standard_error = float(np.mean(fitted.standard_error))
        record = {
            "prior": prior,
            "target": improvement.target,
            "exact_mask": exact_mask,
            "exact_advantage": improvement.advantage,
            "base_action": base_action,
            "branch_best_action": branch_best,
            "branched": int(exact_actions.size),
            "feasible": int(feasible.size),
            "within_state_value_sd": float(np.std(fitted.score)),
            "tape_value_standard_error": mean_tape_standard_error,
            "paired_contrast_standard_error": float(
                fitted.paired_contrast_standard_error
            ),
            "tape_count": int(self.nmcc_pi_tapes),
            "eligible": int(feasible.size),
            "uniform_panel_actions": int(uniform_panel_actions),
            "target_kl_to_prior": float(improvement.kl_to_old),
            "target_eta": float(improvement.eta),
            "target_eta_floor": float(target_eta_floor),
            "physical_branch_seconds": float(branch_seconds),
            "branch_workers": int(self.nmcc_pi_workers),
            "prior_argmax_is_branch_best": bool(
                int(feasible[int(np.argmax(prior[feasible]))]) == branch_best
            ),
            "target_argmax_is_branch_best": bool(
                int(feasible[int(np.argmax(improvement.target[feasible]))])
                == branch_best
            ),
            "simulation_time": int(simulation_time),
            "decision_index": int(self.deployments_made),
            "rollout_horizon_timesteps": max(
                0, int(self.horizon) - int(simulation_time)
            ),
            "rollout_changed_base_best": bool(branch_best != base_action),
            "actor_objective": "kl_constrained_categorical_rollout_improvement",
            "actor_reward_baseline": "complete_rollout_cost_with_shared_futures",
            "interval_cost_contract": "complete_ledger_to_episode_horizon",
            "continuation_used": True,
            "rollout_mean_cost_min": float(np.min(fitted.mean_cost)),
            "rollout_mean_cost_max": float(np.max(fitted.mean_cost)),
        }
        self.improvement_records.append(
            {
                key: value
                for key, value in record.items()
                if np.isscalar(value) and key not in {"target", "prior"}
            }
        )
        return record

    # -- Hybrid NMCC exact common-noise counterfactual branching -----------

    @property
    def counterfactual_horizon(self) -> int:
        """Branch length ``L``.

        Both sampled-pair credit and sampled policy improvement evolve their
        installed-action contrasts over this physical horizon.
        """
        if self._counterfactual_horizon_override is not None:
            return max(1, int(self._counterfactual_horizon_override))
        return max(1, int(self.shelter_action_interval))

    def _ensure_brancher(self):
        if self._counterfactual_brancher is None:
            branch_horizon = (
                int(self.nmcc_pi_branch_horizon)
                if self.nmcc_policy_improvement
                else int(self.counterfactual_horizon)
            )
            self._counterfactual_brancher = CB.CounterfactualBrancher(
                self.core,
                horizon=branch_horizon,
                reward_model=self.reward_model,
                initial_population=int(self.initial_population),
                episode_horizon=int(self.horizon),
                workers=(self.nmcc_pi_workers if self.nmcc_policy_improvement else 1),
            )
        return self._counterfactual_brancher

    def _ensure_reward_brancher(self):
        """Immediate branch used by reward and exact decision features."""
        if self._reward_brancher is None:
            self._reward_brancher = CB.CounterfactualBrancher(
                self.core,
                horizon=0,
                reward_model=self.reward_model,
                initial_population=int(self.initial_population),
                episode_horizon=int(self.horizon),
            )
        return self._reward_brancher

    def _enrich_policy_observation(
        self,
        observation: RegionalObservation,
    ) -> RegionalObservation:
        """Attach exact D(s) and zero-horizon assignment gain consistently."""
        return self._ensure_reward_brancher().enrich_observation(observation)

    def _uniform_random_reference_target(self) -> float:
        """Monte Carlo target for the fixed uniform-siting value ``V_mu``."""
        episode_seed = int(
            getattr(self.core, "scenario_seed", self.policy_seed)
        )
        seeds = tuple(
            NPI.tape_seed(
                episode_seed ^ 0x4D55434F,
                int(self.deployments_made),
                tape_index,
            )
            for tape_index in range(self.random_reference_rollouts)
        )
        snapshot = CB.capture(
            self.core, label="uniform-random-reference-value"
        )
        costs = self._ensure_brancher().random_policy_costs(
            snapshot=snapshot,
            future_seeds=seeds,
            decision_index=int(self.deployments_made),
            remaining_deployments=int(self.remaining_deployments),
            maximum_deployments=int(self.maximum_deployments),
        )
        target = float(np.mean(costs))
        if not np.isfinite(target) or target < 0.0:
            raise FloatingPointError(
                "Uniform-random reference target must be finite and non-negative"
            )
        return target

    def _readonly_observation_values(
        self,
        observation: RegionalObservation,
        *,
        recurrent_state: Optional[tuple] = None,
    ) -> tuple[float, np.ndarray]:
        """Critic value of an off-trajectory state.

        A reference-candidate branch ends somewhere the live episode never
        goes, so its value must be read without advancing the recurrent state
        or appending to the observation cache. The decision epoch's recurrent
        state supplies the shared pre-intervention history for both endpoints.
        """
        (
            cell_features,
            global_features,
            route_edge_index,
            route_edge_weight,
            candidate_cell_index,
            candidate_features,
            action_mask,
        ) = self._policy_tensors(observation)
        momentum_features = self._momentum_features(observation)
        frame = ObservationFrame(
            cell_features=cell_features.detach(),
            global_features=global_features.detach(),
            route_edge_index=route_edge_index.detach(),
            route_edge_weight=route_edge_weight.detach(),
            candidate_cell_index=candidate_cell_index.detach(),
            candidate_features=candidate_features.detach(),
            action_mask=action_mask.detach(),
            momentum_features=momentum_features.detach(),
            simulation_time=torch.as_tensor(
                [int(observation.simulation_time)],
                dtype=torch.long,
                device=self.device,
            ),
            reference_debt=self._reference_debt_tensor(observation).detach(),
        )
        graph = self._graph_from_frame(frame)
        context = self.recurrent_state if recurrent_state is None else recurrent_state
        with torch.no_grad():
            _, value, components, _, _ = self.policy.forward_recurrent(
                graph,
                context,
                momentum_features,
            )
        return (
            float(value.reshape(-1)[0].item()),
            components.reshape(-1, len(REWARD_COMPONENT_NAMES))[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64),
        )

    def _readonly_observation_value(
        self,
        observation: RegionalObservation,
        *,
        recurrent_state: Optional[tuple] = None,
    ) -> float:
        value, _ = self._readonly_observation_values(
            observation,
            recurrent_state=recurrent_state,
        )
        return value

    def _nmcc_endpoint_frame(
        self,
        observation: RegionalObservation,
        *,
        previous_observation: RegionalObservation,
    ) -> ObservationFrame:
        (
            cell_features,
            global_features,
            route_edge_index,
            route_edge_weight,
            candidate_cell_index,
            candidate_features,
            action_mask,
        ) = self._policy_tensors(observation)
        momentum = self._momentum_features_between(
            previous_observation,
            observation,
        )
        return ObservationFrame(
            cell_features=cell_features.detach(),
            global_features=global_features.detach(),
            route_edge_index=route_edge_index.detach(),
            route_edge_weight=route_edge_weight.detach(),
            candidate_cell_index=candidate_cell_index.detach(),
            candidate_features=candidate_features.detach(),
            action_mask=action_mask.detach(),
            momentum_features=momentum.detach(),
            simulation_time=torch.as_tensor(
                [int(observation.simulation_time)],
                dtype=torch.long,
                device=self.device,
            ),
            reference_debt=self._reference_debt_tensor(observation).detach(),
        )

    def _packed_graph_from_frames(
        self,
        frames: Sequence[ObservationFrame],
    ):
        """Pack independent branch endpoints for one continuation forward pass."""
        if not frames:
            raise ValueError("NMCC continuation batching requires endpoint frames")
        counts = [int(frame.cell_features.size(0)) for frame in frames]
        batch = torch.repeat_interleave(
            torch.arange(len(frames), device=self.device),
            torch.as_tensor(counts, dtype=torch.long, device=self.device),
        )
        route_indices = []
        route_weights = []
        offset = 0
        for frame, count in zip(frames, counts):
            route_indices.append(frame.route_edge_index.to(self.device) + offset)
            route_weights.append(frame.route_edge_weight.to(self.device))
            offset += count
        return self._graph(
            torch.cat([frame.cell_features.to(self.device) for frame in frames], dim=0),
            torch.stack(
                [frame.global_features.to(self.device).reshape(-1) for frame in frames],
                dim=0,
            ),
            batch=batch,
            edge_index=self.edge_index,
            route_edge_index=torch.cat(route_indices, dim=1),
            route_edge_weight=torch.cat(route_weights, dim=0),
            candidate_cell_index=torch.stack(
                [frame.candidate_cell_index.to(self.device) for frame in frames],
                dim=0,
            ),
            candidate_features=torch.stack(
                [frame.candidate_features.to(self.device) for frame in frames],
                dim=0,
            ),
        )

    def _readonly_nmcc_endpoint_values(
        self,
        observations: Sequence[RegionalObservation],
        *,
        previous_observation: RegionalObservation,
        recurrent_state: Optional[tuple],
    ) -> tuple[list[float], list[float]]:
        """Read frozen ``V_mu`` and learned ``W_pi`` at branch endpoints."""
        if not observations:
            return [], []
        frames = [
            self._nmcc_endpoint_frame(
                observation,
                previous_observation=previous_observation,
            )
            for observation in observations
        ]
        momentum = torch.stack(
            [frame.momentum_features.to(self.device) for frame in frames],
            dim=0,
        )
        packed_state = None
        if recurrent_state is not None:
            packed_state = tuple(
                state.to(self.device).repeat(len(frames), 1)
                for state in recurrent_state
            )
        with torch.no_grad():
            reference_values, policy_values, _ = (
                self.policy.forward_nmcc_endpoint_values_recurrent(
                self._packed_graph_from_frames(frames),
                packed_state,
                momentum,
            )
            )
        reference_values = reference_values.reshape(-1) + torch.cat(
            [
                frame.reference_debt.to(self.device).reshape(1)
                for frame in frames
            ]
        )
        policy_values = policy_values.reshape(-1)
        if (
            reference_values.numel() != len(observations)
            or policy_values.numel() != len(observations)
        ):
            raise RuntimeError("NMCC endpoint value batch changed endpoint count")
        return (
            [float(value) for value in reference_values.cpu().tolist()],
            [float(value) for value in policy_values.cpu().tolist()],
        )

    def _readonly_nmcc_continuation_values(
        self,
        observations: Sequence[RegionalObservation],
        *,
        previous_observation: RegionalObservation,
        recurrent_state: Optional[tuple],
    ) -> list[float]:
        """Compatibility view of the policy-savings continuation only."""
        _, values = self._readonly_nmcc_endpoint_values(
            observations,
            previous_observation=previous_observation,
            recurrent_state=recurrent_state,
        )
        return values

    def _readonly_nmcc_continuation_value(
        self,
        observation: RegionalObservation,
        *,
        previous_observation: RegionalObservation,
        recurrent_state: Optional[tuple],
    ) -> float:
        """Read one shared-critic continuation without mutating recurrence."""
        return self._readonly_nmcc_continuation_values(
            (observation,),
            previous_observation=previous_observation,
            recurrent_state=recurrent_state,
        )[0]

    def _normalized_outcome_target(
        self,
        *,
        before: OutcomeSnapshot,
        after: OutcomeSnapshot,
        active_person_time: float,
        hazard_exposure_person_time: float,
    ) -> np.ndarray:
        population = float(self.initial_population)
        horizon_mass = population * float(self.horizon)
        return np.asarray(
            (
                (after.safe_completed - before.safe_completed) / population,
                (after.casualties - before.casualties) / population,
                float(active_person_time) / horizon_mass,
                float(hazard_exposure_person_time) / horizon_mass,
                after.active_population / population,
                after.risk_mass / (2.0 * population),
            ),
            dtype=np.float32,
        )

    def _collect_candidate_reference(
        self,
        observation: RegionalObservation,
        *,
        reference_action: int,
        simulation_time: int,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Measure one sampled candidate against the matched WAIT trajectory."""
        del simulation_time
        action = int(reference_action)
        if not 0 <= action < observation.number_of_actions:
            raise ValueError("Counterfactual reference action is outside the action space")
        if not bool(observation.action_mask[action]):
            raise ValueError("Counterfactual reference action is not executable")
        cell = int(observation.candidate_cell_indices[action])
        effect = self._ensure_brancher().paired_effect(cell)
        breakdown = effect.reward_breakdown(
            self.reward_model,
            initial_population=self.initial_population,
            episode_horizon_minutes=self.horizon_minutes,
        )
        # Actor ranking uses the signed causal gain. Displayed environmental
        # return remains clipped at zero, but two harmful locations therefore
        # cannot collapse into the same learning target.
        reference_return = float(
            breakdown.raw_total * self.learning_reward_scale
        )
        reference_components = (
            breakdown.raw_component_vector() * self.learning_reward_scale
        )
        reference_outcome_target = self._normalized_outcome_target(
            before=effect.acted.start_outcome,
            after=effect.acted.end_outcome,
            active_person_time=effect.acted.active_person_time,
            hazard_exposure_person_time=(
                effect.acted.hazard_exposure_person_time
            ),
        )
        reference_bootstrap_components = np.zeros(
            len(REWARD_COMPONENT_NAMES),
            dtype=np.float64,
        )
        return (
            reference_return,
            reference_components,
            reference_bootstrap_components,
            reference_outcome_target,
        )

    def _counterfactual_advantage_for(
        self,
        pending: "PendingDecision",
        *,
        factual_components: np.ndarray,
        factual_bootstrap_components: np.ndarray,
        elapsed: int,
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        """Return chosen minus reference immediate installation reduction."""
        del factual_bootstrap_components, elapsed
        if not np.isfinite(pending.reference_return):
            return None, None
        if pending.reference_component_return is None:
            return None, None
        components = (
            np.asarray(factual_components, dtype=np.float64)
            - np.asarray(pending.reference_component_return, dtype=np.float64)
        )
        return float(components.sum()), components.astype(np.float32)

    def _blend_counterfactual_advantages(
        self,
        advantages: torch.Tensor,
        transitions: Sequence["Transition"],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Use exact chosen-versus-policy-reference effects as PPO credit.

        NMCC does not select, approve, or reject an action.  The policy first
        samples and executes its own action. An independent draw from that same
        categorical policy supplies the paired reference installation under
        identical future noise. The signed difference is a lower-variance,
        within-state ranking advantage. Dividing by a positive batch scale
        conditions the optimizer without changing any sign.
        """
        diagnostics = {
            "counterfactual_fraction": 0.0,
            "counterfactual_advantage_mean": 0.0,
            "counterfactual_advantage_sd": 0.0,
            "counterfactual_negative_fraction": 0.0,
            "counterfactual_positive_fraction": 0.0,
            "counterfactual_equal_action_fraction": 0.0,
            "counterfactual_remaining_risk_time_difference_mean": 0.0,
            "counterfactual_casualties_avoided_mean": 0.0,
            "mc_advantage_sd": float(advantages.std(unbiased=False).item()),
            "gae_advantage_sd": float(advantages.std(unbiased=False).item()),
        }
        if not self.counterfactual_credit:
            return advantages, diagnostics

        values = torch.full_like(advantages, float("nan"))
        components = torch.full(
            (advantages.numel(), len(REWARD_COMPONENT_NAMES)),
            float("nan"),
            dtype=advantages.dtype,
            device=advantages.device,
        )
        casualties_avoided = torch.full_like(advantages, float("nan"))
        equal_action = torch.zeros_like(advantages, dtype=torch.bool)
        for index, transition in enumerate(transitions):
            transition_advantage = getattr(
                transition,
                "counterfactual_advantage",
                None,
            )
            transition_components = getattr(
                transition,
                "counterfactual_components",
                None,
            )
            reference_action = getattr(
                transition,
                "counterfactual_reference_action",
                None,
            )
            causal_outcome = getattr(
                transition,
                "causal_outcome_target",
                None,
            )
            if transition_advantage is not None:
                values[index] = transition_advantage.reshape(-1)[0]
            if transition_components is not None:
                components[index] = transition_components.reshape(-1)
            if reference_action is not None:
                equal_action[index] = bool(
                    int(transition.action.reshape(-1)[0].item())
                    == int(reference_action.reshape(-1)[0].item())
                )
            if causal_outcome is not None:
                casualties_avoided[index] = (
                    -causal_outcome.reshape(-1)[1]
                    * self.initial_population
                )
        available = torch.isfinite(values)
        count = int(available.sum().item())
        if (
            count != int(advantages.numel())
            or not bool(torch.isfinite(components).all())
            or not bool(torch.isfinite(casualties_avoided).all())
            or any(
                getattr(
                    transition,
                    "counterfactual_reference_action",
                    None,
                )
                is None
                for transition in transitions
            )
        ):
            raise RuntimeError(
                "Exact NMCC is enabled, but at least one sampled action has no "
                "same-noise candidate-versus-candidate credit. Refusing to substitute "
                "ordinary PPO credit for a missing branch."
            )

        present = values[available]
        diagnostics["counterfactual_fraction"] = float(count) / float(advantages.numel())
        diagnostics["counterfactual_advantage_mean"] = float(present.mean().item())
        diagnostics["counterfactual_advantage_sd"] = float(
            present.std(unbiased=False).item()
        )
        diagnostics["counterfactual_negative_fraction"] = float(
            (present < 0.0).to(torch.float32).mean().item()
        )
        diagnostics["counterfactual_positive_fraction"] = float(
            (present > 0.0).to(torch.float32).mean().item()
        )
        diagnostics["counterfactual_equal_action_fraction"] = float(
            equal_action.to(torch.float32).mean().item()
        )
        diagnostics["counterfactual_remaining_risk_time_difference_mean"] = float(
            components[:, 0].mean().item()
            * self.initial_population
            * self.horizon_minutes
            / self.learning_reward_scale
        )
        diagnostics["counterfactual_casualties_avoided_mean"] = float(
            casualties_avoided.mean().item()
        )
        return self._standardize_advantages(values), diagnostics

    def _component_credit_targets(
        self,
        component_rewards: torch.Tensor,
        component_values: torch.Tensor,
        dones: torch.Tensor,
        durations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return complete realized return-to-go for actor and critic.

        Every rollout contains complete episodes, so PPO can use the actual
        semi-Markov outcome after each sampled action.  No heuristic,
        counterfactual model, or one-step value bootstrap replaces that
        experience.  Components remain separate only for diagnostics and the
        component critic; their sum is the scalar policy return.
        """
        total, component_count = component_rewards.shape
        actor_returns = torch.zeros_like(component_rewards)
        running_return = torch.zeros(
            component_count,
            dtype=component_rewards.dtype,
            device=component_rewards.device,
        )
        for index in reversed(range(total)):
            nonterminal = 1.0 - dones[index]
            decision_discount = self.gamma ** durations[index]
            running_return = (
                component_rewards[index]
                + decision_discount * nonterminal * running_return
            )
            actor_returns[index] = running_return
        return actor_returns, actor_returns.clone()

    def _cost_to_go_targets(
        self,
        interval_costs: torch.Tensor,
        dones: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        """Backward complete cost-to-go for the fixed random reference."""
        if interval_costs.ndim != 1 or not (
            dones.shape == interval_costs.shape == durations.shape
        ):
            raise ValueError("Cost-to-go inputs must be aligned vectors")
        targets = torch.zeros_like(interval_costs)
        running = interval_costs.new_zeros(())
        for index in reversed(range(interval_costs.numel())):
            running = interval_costs[index] + (
                self.gamma ** durations[index]
            ) * (1.0 - dones[index]) * running
            targets[index] = running
        return targets

    def _on_policy_advantages(
        self,
        returns: torch.Tensor,
        rollout_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized PPO advantages and their unmodified values."""
        if returns.ndim != 1 or rollout_values.shape != returns.shape:
            raise ValueError("PPO returns and rollout values must be aligned vectors")
        raw = returns - rollout_values
        return self._standardize_advantages(raw), raw

    @staticmethod
    def _standardize_advantages(raw: torch.Tensor) -> torch.Tensor:
        """Center PPO credit and give every nonconstant batch unit variance.

        A fixed scale floor used to attenuate legitimate small NMCC contrasts:
        in the v42 pilot the foresight-stage standard deviation was below that floor,
        shrinking an already sparse actor update by more than threefold.  PPO
        advantage normalization is dimensionless, so its only floor should be
        the numerical resolution of the tensor dtype.  A genuinely constant
        batch contains no ranking information and correctly maps to zero.
        """
        if raw.ndim != 1:
            raise ValueError("PPO advantages must be a vector")
        centered = raw - raw.mean()
        scale = raw.std(unbiased=False)
        epsilon = torch.finfo(raw.dtype).eps
        if float(scale.item()) <= float(epsilon):
            return torch.zeros_like(raw)
        return centered / scale

    @classmethod
    def _nmcc_actor_critic_targets(
        cls,
        actor_returns: torch.Tensor,
        component_actor_returns: torch.Tensor,
        improvement_advantage: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate sampled NMCC diagnostics from the critic target.

        ``improvement_advantage`` is centered within each physically branched
        set under the actor prior, whereas ``actor_returns`` is one
        realized trajectory under that policy. The
        the recent-window M-step consumes the complete branched table;
        this helper retains the executed-action trace and, critically, never
        manufactures a critic target through ``G-A``.
        """
        if actor_returns.ndim != 1 or actions.shape != actor_returns.shape:
            raise ValueError("NMCC actor returns and actions must be aligned vectors")
        if component_actor_returns.ndim != 2 or (
            component_actor_returns.shape[0] != actor_returns.shape[0]
        ):
            raise ValueError("NMCC component returns must align with scalar returns")
        if improvement_advantage.ndim != 2 or (
            improvement_advantage.shape[0] != actor_returns.shape[0]
        ):
            raise ValueError("NMCC advantage table must align with returns")
        row = torch.arange(actor_returns.numel(), device=actor_returns.device)
        raw_advantages = improvement_advantage[row, actions]
        return (
            cls._standardize_advantages(raw_advantages),
            raw_advantages,
            actor_returns,
            component_actor_returns,
        )

    def _set_optimizer_partition_trainable(
        self,
        *,
        actor: bool,
        critic: bool,
    ) -> None:
        """Enable exactly the parameters owned by the current pass.

        A tensor shared by both optimizers (the representation in
        ``shared_phasic`` mode) is trainable when either enabled pass owns it.
        """
        enabled = set()
        if actor:
            enabled.update(id(parameter) for _, parameter in self.actor_named_parameters)
        if critic:
            enabled.update(id(parameter) for _, parameter in self.critic_named_parameters)
        owned = self.actor_named_parameters + self.critic_named_parameters
        for _, parameter in owned:
            parameter.requires_grad_(id(parameter) in enabled)


    LEARNER_FLOW_DIAGNOSTIC_NAMES = (
        "representation_shared",
        "representation_policy_drift_kl",
        "representation_rollback",
        "representation_clone_kl",
        "critic_representation_gradient_norm",
        "critic_head_gradient_norm",
        "actor_representation_gradient_norm",
        "actor_head_gradient_norm",
        "actor_representation_gradient_share",
        "actor_readout_norm",
        "actor_temporal_readout_norm",
    )

    def _learner_flow_diagnostics(
        self,
        *,
        shared: bool,
        drift_kl: float,
        rollback: bool,
        clone_kls: Sequence[float],
        critic_representation: Sequence[float],
        critic_head: Sequence[float],
        actor_representation: Sequence[float],
        actor_head: Sequence[float],
    ) -> Dict[str, float]:
        """Where gradient actually went in this update, before clipping.

        ``actor_representation_gradient_share`` near zero means the policy
        loss is not reaching the encoder/LSTM (the zero-readout failure);
        ``critic_representation_gradient_norm`` is zero by construction unless
        the representation is shared.
        """

        def mean(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        policy = self.policy
        if policy is None:
            raise RuntimeError("Learner diagnostics require an RL policy")
        share = [
            float(r) / float(r + h) if (r + h) > 0.0 else 0.0
            for r, h in zip(actor_representation, actor_head)
        ]
        return {
            "representation_shared": float(shared),
            "representation_policy_drift_kl": float(drift_kl),
            "representation_rollback": float(rollback),
            "representation_clone_kl": mean(clone_kls),
            "critic_representation_gradient_norm": mean(critic_representation),
            "critic_head_gradient_norm": mean(critic_head),
            "actor_representation_gradient_norm": mean(actor_representation),
            "actor_head_gradient_norm": mean(actor_head),
            "actor_representation_gradient_share": mean(share),
            "actor_readout_norm": float(
                policy.actor_cell.weight.detach().norm().item()
            ),
            "actor_temporal_readout_norm": float(
                policy.temporal_actor_context.weight.detach().norm().item()
            ),
        }

    ROLLOUT_PI_DIAGNOSTIC_NAMES: tuple[str, ...] = (
        "rollout_pi_target_coverage",
        "rollout_pi_batch_action_support_coverage",
        "rollout_pi_candidate_grade_fraction",
        "rollout_pi_mean_graded_candidates",
        "rollout_pi_mean_feasible_candidates",
        "rollout_pi_actor_kl",
        "rollout_pi_target_kl_before",
        "rollout_pi_target_kl_after",
        "rollout_pi_weighted_pairwise_accuracy_before",
        "rollout_pi_weighted_pairwise_accuracy_after",
        "rollout_pi_fit_tolerance_reached",
        "rollout_pi_branched_top1_before",
        "rollout_pi_branched_top1_after",
        "rollout_pi_target_top1",
        "rollout_pi_branch_expected_advantage_before",
        "rollout_pi_branch_expected_advantage_after",
        "rollout_pi_branch_expected_advantage_gain",
        "rollout_pi_within_state_value_sd",
        "rollout_pi_tape_value_standard_error",
        "rollout_pi_paired_contrast_standard_error",
        "rollout_pi_rollout_signal_fraction",
        "rollout_pi_target_step_kl",
        "rollout_pi_target_eta",
        "rollout_pi_uniform_panel_actions",
        "rollout_pi_actor_replay_episodes",
        "rollout_pi_actor_replay_transitions",
        "rollout_pi_rollout_horizon_timesteps",
        "rollout_pi_physical_branch_seconds",
        "rollout_pi_branch_workers",
        "rollout_pi_rollout_changed_base_best_fraction",
        "environment_dynamics_explained_variance",
        *tuple(
            f"environment_dynamics_explained_variance_{name}"
            for name in MOMENTUM_FEATURE_NAMES
        ),
    )

    DIRECT_PPO_DIAGNOSTIC_NAMES: tuple[str, ...] = (
        "epochs_completed",
        "critic_epochs_completed",
        "actor_epochs_completed",
        "value_loss",
        "attempted_kl",
        "approximate_kl",
        "kl_rollback",
        "actor_update_accepted",
        "actor_update_rejected",
        "clip_fraction",
        "gradient_norm",
        "explained_variance",
        "actor_mc_return_sd",
        "actor_value_baseline_mean",
        "actor_raw_advantage_mean",
        "actor_raw_advantage_sd",
        "actor_advantage_sd",
        "actor_baseline_coverage",
        "critic_td_error_sd",
        "residual_rms",
        "learning_rate",
        "counterfactual_fraction",
        "counterfactual_advantage_mean",
        "counterfactual_advantage_sd",
        "counterfactual_negative_fraction",
        "counterfactual_positive_fraction",
        "counterfactual_equal_action_fraction",
        "counterfactual_remaining_risk_time_difference_mean",
        "counterfactual_casualties_avoided_mean",
        "mc_advantage_sd",
        "gae_advantage_sd",
        *(f"value_loss_{name}" for name in REWARD_COMPONENT_NAMES),
        *(f"explained_variance_{name}" for name in REWARD_COMPONENT_NAMES),
    )

    def _improvement_diagnostics(
        self,
        *,
        coverage: float,
        sampled_action_coverage: float,
        policy_kl_before: float,
        policy_kl_after: float,
        target_kl_before: float,
        target_kl_after: float,
        pairwise_accuracy_before: float,
        pairwise_accuracy_after: float,
        top1_before: float,
        top1_after: float,
        expected_advantage_before: float,
        expected_advantage_after: float,
        converged: bool,
    ) -> Dict[str, float]:
        """Summarize this rollout's NMCC-PI evidence and consume its records.

        The pair ``top1_before``/``top1_after`` reports whether the M-step
        moved the actor toward the best physically branched cell.
        """
        records = self.improvement_records
        self.improvement_records = []

        def mean(key: str) -> float:
            values = [float(item[key]) for item in records]
            return float(np.mean(values)) if values else float("nan")

        mean_graded = mean("branched")
        mean_feasible = mean("feasible")
        signal = mean("within_state_value_sd")
        noise = mean("paired_contrast_standard_error")
        signal_fraction = (
            float(signal / (signal + noise))
            if np.isfinite(signal)
            and np.isfinite(noise)
            and signal + noise > 0.0
            else 0.0
        )

        return {
            "rollout_pi_target_coverage": coverage,
            "rollout_pi_batch_action_support_coverage": float(
                sampled_action_coverage
            ),
            "rollout_pi_candidate_grade_fraction": (
                float(mean_graded / mean_feasible)
                if np.isfinite(mean_graded)
                and np.isfinite(mean_feasible)
                and mean_feasible > 0.0
                else 0.0
            ),
            "rollout_pi_mean_graded_candidates": mean_graded,
            "rollout_pi_mean_feasible_candidates": mean_feasible,
            "rollout_pi_actor_kl": float(policy_kl_after),
            "rollout_pi_target_kl_before": float(target_kl_before),
            "rollout_pi_target_kl_after": float(target_kl_after),
            "rollout_pi_weighted_pairwise_accuracy_before": float(
                pairwise_accuracy_before
            ),
            "rollout_pi_weighted_pairwise_accuracy_after": float(
                pairwise_accuracy_after
            ),
            "rollout_pi_fit_tolerance_reached": float(converged),
            "rollout_pi_branched_top1_before": float(top1_before),
            "rollout_pi_branched_top1_after": float(top1_after),
            "rollout_pi_target_top1": mean("target_argmax_is_branch_best"),
            "rollout_pi_branch_expected_advantage_before": float(
                expected_advantage_before
            ),
            "rollout_pi_branch_expected_advantage_after": float(
                expected_advantage_after
            ),
            "rollout_pi_branch_expected_advantage_gain": float(
                expected_advantage_after - expected_advantage_before
            ),
            "rollout_pi_within_state_value_sd": signal,
            "rollout_pi_tape_value_standard_error": mean(
                "tape_value_standard_error"
            ),
            "rollout_pi_paired_contrast_standard_error": noise,
            "rollout_pi_rollout_signal_fraction": signal_fraction,
            "rollout_pi_target_step_kl": mean("target_kl_to_prior"),
            "rollout_pi_target_eta": mean("target_eta"),
            "rollout_pi_uniform_panel_actions": mean("uniform_panel_actions"),
            "rollout_pi_actor_replay_episodes": float(len(self.actor_replay)),
            "rollout_pi_actor_replay_transitions": float(
                sum(len(episode.frames) for episode in self.actor_replay)
            ),
            "rollout_pi_rollout_horizon_timesteps": mean(
                "rollout_horizon_timesteps"
            ),
            "rollout_pi_physical_branch_seconds": mean("physical_branch_seconds"),
            "rollout_pi_branch_workers": mean("branch_workers"),
            "rollout_pi_rollout_changed_base_best_fraction": mean(
                "rollout_changed_base_best"
            ),
        }

    def _policy_log_probabilities(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
        action_masks: torch.Tensor,
        *,
        action_temperature: float,
    ) -> torch.Tensor:
        """Masked log-probabilities of the current policy, in transition order."""
        with torch.no_grad():
            selected, logits, _ = self._evaluate_nmcc_actor_sequences(
                transitions,
                episodes,
                action_temperature=action_temperature,
            )
            ordered = torch.empty_like(logits)
            ordered[selected] = logits
            return torch.log(
                self._exploratory_probabilities(ordered, action_masks).clamp_min(1e-30)
            )

    @staticmethod
    def _categorical_kl(
        reference_log_probabilities: torch.Tensor,
        log_probabilities: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Exact per-decision KL(reference || current) over feasible cells."""
        reference = reference_log_probabilities.exp()
        terms = reference * (reference_log_probabilities - log_probabilities)
        return torch.where(action_masks.to(torch.bool), terms, torch.zeros_like(terms)).sum(
            dim=-1
        )

    @staticmethod
    def _gradient_norm(named_parameters: Sequence[tuple[str, nn.Parameter]]) -> float:
        """Pre-clipping L2 norm of the gradients currently held by a role."""
        total = 0.0
        for _, parameter in named_parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().square().sum().item())
        return float(np.sqrt(total))

    @staticmethod
    def _branch_best_actions(advantage: torch.Tensor, exact_mask: torch.Tensor) -> torch.Tensor:
        """Index of the highest exact within-state value among branched cells."""
        return torch.where(
            exact_mask > 0.5,
            advantage,
            torch.full_like(advantage, float("-inf")),
        ).argmax(dim=-1)

    def _improvement_policy_statistics(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
        action_masks: torch.Tensor,
        improvement_available: torch.Tensor,
        improvement_prior: torch.Tensor,
        improvement_target: torch.Tensor,
        improvement_advantage: torch.Tensor,
        branch_best: torch.Tensor,
        improvement_exact_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Measure actor movement and quality on simulator-graded support."""
        with torch.no_grad():
            selected, logits, _ = self._evaluate_nmcc_actor_sequences(
                transitions,
                episodes,
                action_temperature=1.0,
            )
            expected = torch.arange(len(transitions), device=self.device)
            if not torch.equal(selected, expected):
                raise RuntimeError("Complete recurrent replay changed transition order")
            log_new = F.log_softmax(
                self._safe_masked_logits(logits, action_masks),
                dim=-1,
            )
            available = improvement_available
            if not bool(available.all()):
                raise RuntimeError(
                    "Rollout policy diagnostics require every transition target"
                )
            prior = improvement_prior
            target = improvement_target
            log_rows = log_new
            feasible_rows = action_masks.to(torch.bool)
            exact_rows = improvement_exact_mask.to(torch.bool)
            new_rows = log_rows.exp()
            policy_kl = torch.where(
                feasible_rows & (new_rows > 0.0),
                new_rows
                * (
                    log_rows
                    - torch.log(prior.clamp_min(1e-30))
                ),
                torch.zeros_like(new_rows),
            ).sum(dim=-1)
            top1 = (
                self._safe_masked_logits(log_rows, exact_rows).argmax(dim=-1)
                == branch_best
            ).to(torch.float32)
            exact_probability = torch.where(
                exact_rows,
                new_rows,
                torch.zeros_like(new_rows),
            )
            exact_probability = exact_probability / exact_probability.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-30)
            expected_advantage = (
                exact_probability * improvement_advantage
            ).sum(dim=-1)
            kl = policy_kl.mean()
            positive_target = target > 0.0
            target_fit_kl = torch.where(
                positive_target,
                target
                * (
                    torch.log(target.clamp_min(1e-30))
                    - log_rows
                ),
                torch.zeros_like(target),
            ).sum(dim=-1).mean()
        if not torch.isfinite(kl):
            raise FloatingPointError("Rollout-policy KL became non-finite")
        if not torch.isfinite(target_fit_kl):
            raise FloatingPointError("NMCC categorical target KL became non-finite")
        return {
            "kl": float(kl.item()),
            "target_fit_kl": float(target_fit_kl.item()),
            "top1": float(top1.mean().item()),
            "expected_advantage": float(expected_advantage.mean().item()),
        }

    def _full_rollout_policy_kl(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        *,
        action_temperature: float,
    ) -> float:
        """Measure retained policy change on the complete recurrent batch.

        NMCC transitions store the complete trainable actor prior, so their
        trust-region diagnostic is the exact ``KL(pi_new || pi_prior)``.
        Using the one sampled action estimator here previously made this value
        incomparable with entropy and clipping diagnostics from the full
        categorical policy.
        """
        with torch.no_grad():
            has_nmcc_prior = all(
                transition.improvement_prior is not None
                for transition in transitions
            )
            selected, logits, _ = self._evaluate_nmcc_actor_sequences(
                transitions,
                episodes,
                action_temperature=(1.0 if has_nmcc_prior else action_temperature),
            )
            expected = torch.arange(len(transitions), device=self.device)
            if not torch.equal(selected, expected):
                raise RuntimeError("Complete recurrent replay changed transition order")
            if has_nmcc_prior:
                prior = torch.stack(
                    [
                        transition.improvement_prior.to(self.device)
                        for transition in transitions
                    ]
                )
                current = torch.softmax(
                    self._safe_masked_logits(logits, action_masks), dim=-1
                )
                full_kl = torch.where(
                    action_masks.to(torch.bool) & (current > 0.0),
                    current
                    * (
                        torch.log(current.clamp_min(1e-30))
                        - torch.log(prior.clamp_min(1e-30))
                    ),
                    torch.zeros_like(current),
                ).sum(dim=-1).mean()
            else:
                distribution = self._distribution_from_logits(logits, action_masks)
                current_log_probability = distribution.log_prob(actions)
                full_log_ratio, full_ratio = self._ppo_log_ratio(
                    current_log_probability,
                    old_log_probabilities,
                )
                full_kl = torch.mean((full_ratio - 1.0) - full_log_ratio)
        if not torch.isfinite(full_kl):
            raise FloatingPointError("Full-rollout recurrent PPO KL became non-finite")
        return float(full_kl.item())

    @staticmethod
    def _explained_variance(target: torch.Tensor, prediction: torch.Tensor) -> float:
        variance = torch.var(target, unbiased=False)
        if float(variance.item()) <= 1e-12:
            return 0.0
        score = 1.0 - torch.var(
            target - prediction,
            unbiased=False,
        ) / variance
        return float(score.item())

    @staticmethod
    def _frame_on_cpu(frame: ObservationFrame) -> ObservationFrame:
        return ObservationFrame(
            **{
                name: getattr(frame, name).detach().cpu().clone()
                for name in ObservationFrame.__dataclass_fields__
            }
        )

    def _append_random_reference_replay(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
        targets: torch.Tensor,
    ) -> None:
        for episode in episodes:
            indices = [int(index) for index in episode]
            self.random_reference_replay.append(
                ReferenceValueEpisode(
                    frames=tuple(
                        self._frame_on_cpu(
                            transitions[index].observation_history[-1]
                        )
                        for index in indices
                    ),
                    targets=targets[indices].detach().cpu().clone(),
                )
            )
        maximum = int(self.nmcc_pi_replay_max_episodes)
        if len(self.random_reference_replay) > maximum:
            del self.random_reference_replay[:-maximum]

    def _append_actor_replay(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
    ) -> None:
        """Store one internally consistent frozen-policy iteration.

        A rollout cost is :math:`Q^{pi_k}(s,a)`: its later decisions were made
        by the actor frozen for the batch that produced it.  Costs from an older
        actor cannot be made on-policy merely by recomputing probabilities.
        Consequently actor evidence is replaced at every policy iteration,
        never mixed across continuation policies.  The reference-value replay
        remains separate because its uniform continuation policy is fixed.
        """
        current_iteration: list[ActorReplayEpisode] = []
        for episode in episodes:
            indices = [int(index) for index in episode]
            if any(
                transitions[index].improvement_advantage is None
                or transitions[index].improvement_exact_mask is None
                or transitions[index].improvement_eta_floor is None
                for index in indices
            ):
                raise RuntimeError(
                    "Policy improvement requires NMCC labels at every decision"
                )
            current_iteration.append(
                ActorReplayEpisode(
                    frames=tuple(
                        self._frame_on_cpu(
                            transitions[index].observation_history[-1]
                        )
                        for index in indices
                    ),
                    target_scores=torch.stack(
                        [
                            transitions[index]
                            .improvement_advantage.detach()
                            .cpu()
                            .clone()
                            for index in indices
                        ]
                    ),
                    exact_masks=torch.stack(
                        [
                            transitions[index].improvement_exact_mask.detach().cpu()
                            for index in indices
                        ]
                    ),
                    eta_floors=torch.stack(
                        [
                            transitions[index]
                            .improvement_eta_floor.detach()
                            .cpu()
                            .reshape(())
                            .clone()
                            for index in indices
                        ]
                    ),
                )
            )
        maximum = int(self.nmcc_pi_replay_max_episodes)
        self.actor_replay = current_iteration[-maximum:]

    def _actor_fit_episodes(self) -> list[ActorFitEpisode]:
        """Build targets for the current frozen-policy iteration.

        Every retained branch score was generated with this actor as its frozen
        continuation. Each episode is replayed in order so its recurrent state
        reflects all shelter actions represented in the saved observations.
        """
        result: list[ActorFitEpisode] = []
        with torch.no_grad():
            for replay in self.actor_replay:
                recurrent_state = None
                targets = []
                for frame, scores, exact_mask, eta_floor in zip(
                    replay.frames,
                    replay.target_scores,
                    replay.exact_masks,
                    replay.eta_floors,
                ):
                    logits, _, _, recurrent_state = (
                        self.policy.forward_nmcc_actor_recurrent(
                            self._graph_from_frame(frame),
                            recurrent_state,
                            frame.momentum_features.to(self.device),
                        )
                    )
                    feasible = frame.action_mask.detach().cpu().numpy().astype(bool)
                    prior = torch.softmax(
                        self._safe_masked_logits(
                            logits.reshape(1, -1),
                            frame.action_mask.to(self.device).reshape(1, -1),
                        ),
                        dim=-1,
                    ).reshape(-1).detach().cpu().numpy()
                    exact = np.flatnonzero(
                        exact_mask.detach().cpu().numpy().astype(bool)
                    )
                    target = NPI.improvement_target(
                        prior,
                        feasible,
                        exact,
                        scores.detach().cpu().numpy()[exact],
                        epsilon=float(self.nmcc_pi_epsilon),
                        eta_min=float(eta_floor.item()),
                    ).target
                    targets.append(torch.as_tensor(target, dtype=torch.float32))
                result.append(
                    ActorFitEpisode(
                        frames=replay.frames,
                        target_probabilities=torch.stack(targets),
                        target_scores=replay.target_scores,
                        exact_masks=replay.exact_masks,
                    )
                )
        return result

    def _actor_fit_objective(
        self,
        episodes: Sequence[ActorFitEpisode],
    ) -> tuple[torch.Tensor, float, float]:
        """Return ``KL(q || pi_theta)`` and branch-ranking diagnostics.

        ``q`` is the KL-constrained NMCC E-step target.  It covers the entire
        feasible action table: physically unrefined actions keep their frozen
        prior probability, so the M-step cannot silently suppress candidates
        for which it has no new evidence.  Pair ordering is reported only as a
        diagnostic and has no gradient authority.
        """
        losses = []
        correct_weight = 0.0
        comparison_weight = 0.0
        for episode in episodes:
            recurrent_state = None
            for frame, target_probability, target_scores, exact_mask in zip(
                episode.frames,
                episode.target_probabilities,
                episode.target_scores,
                episode.exact_masks,
            ):
                logits, _, _, recurrent_state = (
                    self.policy.forward_nmcc_actor_recurrent(
                        self._graph_from_frame(frame),
                        recurrent_state,
                        frame.momentum_features.to(self.device),
                    )
                )
                logits = logits.reshape(-1)
                feasible = frame.action_mask.to(self.device).reshape(-1).to(
                    torch.bool
                )
                log_probability = F.log_softmax(
                    self._safe_masked_logits(
                        logits.reshape(1, -1), feasible.reshape(1, -1)
                    ),
                    dim=-1,
                ).reshape(-1)
                target = target_probability.to(self.device).reshape(-1)
                if not bool(torch.isfinite(target).all()) or bool(
                    torch.any(target < 0.0)
                ):
                    raise RuntimeError(
                        "NMCC categorical target must be finite and non-negative"
                    )
                if not torch.isclose(
                    target.sum(), target.new_tensor(1.0), atol=1e-6, rtol=1e-6
                ):
                    raise RuntimeError("NMCC categorical target must sum to one")
                if bool(torch.any(target[~feasible] > 0.0)):
                    raise RuntimeError(
                        "NMCC categorical target assigns mass to an infeasible action"
                    )
                positive = target > 0.0
                categorical_kl = torch.sum(
                    target[positive]
                    * (
                        torch.log(target[positive])
                        - log_probability[positive]
                    )
                )
                # Deterministic deployment uses the actor's argmax. Average
                # categorical KL can be arbitrarily small while reversing two
                # nearly tied top cells, so it is not by itself a sufficient
                # fit criterion. Match the target's top-vs-runner-up log-odds
                # as a listwise margin. This uses the same categorical target,
                # introduces no pairwise-regression objective, and becomes
                # exactly zero once the fitted policy preserves the target's
                # deterministic choice with its intended confidence.
                exact = exact_mask.to(self.device).reshape(-1) > 0.5
                exact_actions = torch.nonzero(exact, as_tuple=False).reshape(-1)
                top_choice_margin = logits.new_zeros(())
                if exact_actions.numel() >= 2:
                    exact_target = target[exact_actions]
                    ordered_target, order = torch.sort(
                        exact_target, descending=True
                    )
                    best_action = exact_actions[order[0]]
                    other_actions = exact_actions[order[1:]]
                    target_log_odds = torch.log(
                        ordered_target[0].clamp_min(1e-30)
                        / ordered_target[1].clamp_min(1e-30)
                    ).detach().clamp(max=1.0)
                    fitted_log_odds = (
                        logits[best_action] - logits[other_actions].max()
                    )
                    top_choice_margin = torch.square(
                        torch.relu(target_log_odds - fitted_log_odds)
                    )
                losses.append(categorical_kl + 0.1 * top_choice_margin)
                actions = torch.nonzero(
                    exact,
                    as_tuple=False,
                ).reshape(-1)
                if actions.numel() < 2:
                    continue
                pairs = torch.combinations(actions, r=2)
                prediction_difference = (
                    logits[pairs[:, 0]] - logits[pairs[:, 1]]
                )
                target_score = target_scores.to(self.device).reshape(-1)
                target_difference = (
                    target_score[pairs[:, 0]] - target_score[pairs[:, 1]]
                )
                weights = torch.abs(target_difference)
                informative = weights > torch.finfo(weights.dtype).eps
                if not bool(informative.any()):
                    continue
                prediction_difference = prediction_difference[informative]
                target_difference = target_difference[informative]
                weights = weights[informative]
                with torch.no_grad():
                    matching = (
                        torch.sign(prediction_difference)
                        == torch.sign(target_difference)
                    )
                    correct_weight += float(weights[matching].sum().item())
                    comparison_weight += float(weights.sum().item())
        if not losses:
            return (
                torch.zeros((), dtype=torch.float32, device=self.device),
                0.0,
                0.0,
            )
        return (
            torch.stack(losses).mean(),
            correct_weight,
            comparison_weight,
        )

    def _fit_actor_iteration(
        self,
        fit_episodes: Sequence[ActorFitEpisode],
    ) -> dict[str, float]:
        """Fit one frozen-policy iteration to its categorical NMCC targets."""
        count = len(fit_episodes)
        if count == 0:
            return {
                "loss": 0.0,
                "pairwise_accuracy": 0.0,
                "episodes": 0.0,
                "updates": 0.0,
                "gradient_norm": 0.0,
                "head_gradient_norm": 0.0,
                "representation_gradient_norm": 0.0,
                "initial_loss": 0.0,
                "initial_pairwise_accuracy": 0.0,
            }
        training = list(fit_episodes)

        with torch.no_grad():
            initial_loss, initial_correct, initial_comparisons = (
                self._actor_fit_objective(training)
            )

        gradient_norms = []
        head_gradient_norms = []
        representation_gradient_norms = []
        updates = 0
        maximum_updates = int(self.nmcc_pi_replay_epochs)
        minimum_updates = min(16, maximum_updates)
        check_interval = 4
        for _ in range(maximum_updates):
            loss, _, _ = self._actor_fit_objective(training)
            if not torch.isfinite(loss):
                raise FloatingPointError("Actor NMCC M-step loss is non-finite")
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = self._gradient_norm(self.actor_named_parameters)
            head_gradient_norm = self._gradient_norm(
                self.actor_head_named_parameters
            )
            representation_gradient_norm = self._gradient_norm(
                self.representation_named_parameters
            )
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("Actor fit gradient is non-finite")
            self.actor_optimizer.step()
            gradient_norms.append(float(gradient_norm))
            head_gradient_norms.append(float(head_gradient_norm))
            representation_gradient_norms.append(
                float(representation_gradient_norm)
            )
            updates += 1
            if updates >= minimum_updates and updates % check_interval == 0:
                with torch.no_grad():
                    fitted_loss, _, _ = self._actor_fit_objective(training)
                if float(fitted_loss.item()) <= float(
                    self.nmcc_pi_fit_tolerance
                ):
                    break

        with torch.no_grad():
            train_loss, train_correct, train_count = self._actor_fit_objective(
                training
            )
        return {
            "loss": float(train_loss.item()),
            "pairwise_accuracy": float(train_correct / max(1, train_count)),
            "episodes": float(count),
            "updates": float(updates),
            "gradient_norm": float(np.mean(gradient_norms)),
            "head_gradient_norm": float(np.mean(head_gradient_norms)),
            "representation_gradient_norm": float(
                np.mean(representation_gradient_norms)
            ),
            "initial_loss": float(initial_loss.item()),
            "initial_pairwise_accuracy": float(
                initial_correct / max(1, initial_comparisons)
            ),
        }

    def _value_replay_predictions(
        self,
        episodes: Sequence[ReferenceValueEpisode],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        targets = []
        for episode in episodes:
            recurrent_state = None
            for frame, target in zip(episode.frames, episode.targets):
                prediction, _, recurrent_state = (
                    self.policy.forward_nmcc_endpoint_values_recurrent(
                        self._graph_from_frame(frame),
                        recurrent_state,
                        frame.momentum_features.to(self.device),
                    )
                )
                prediction = prediction + frame.reference_debt.to(self.device)
                predictions.append(prediction.reshape(1))
                targets.append(target.reshape(1).to(self.device))
        if not predictions:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return empty, empty
        return torch.cat(predictions), torch.cat(targets)

    def _fit_value_replay(
        self,
        replay: Sequence[ReferenceValueEpisode],
    ) -> dict[str, float]:
        """Fit the fixed uniform-policy value on retained Stage-E episodes."""
        count = len(replay)
        if count == 0:
            return {
                "loss": 0.0,
                "explained_variance": 0.0,
                "target_sd": 0.0,
                "validation_loss": 0.0,
                "validation_explained_variance": 0.0,
                "episodes": 0.0,
            }
        validation_count = (
            max(1, int(round(count * self.nmcc_pi_validation_fraction)))
            if count >= 5
            else 0
        )
        validation_indices = set(
            np.linspace(0, count - 1, validation_count, dtype=int).tolist()
            if validation_count
            else ()
        )
        training = [
            episode
            for index, episode in enumerate(replay)
            if index not in validation_indices
        ]
        validation = [
            episode
            for index, episode in enumerate(replay)
            if index in validation_indices
        ]
        if not training:
            training = list(replay)
            validation = []

        for _ in range(int(self.nmcc_pi_replay_epochs)):
            prediction, target = self._value_replay_predictions(training)
            loss = F.smooth_l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Random-reference replay loss is non-finite")
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_optimizer.step()
            self.critic_optimizer_updates += 1
            self.optimizer_updates += 1

        with torch.no_grad():
            train_prediction, train_target = self._value_replay_predictions(training)
            validation_prediction, validation_target = (
                self._value_replay_predictions(validation)
                if validation
                else (train_prediction, train_target)
            )
            train_loss = F.smooth_l1_loss(
                train_prediction, train_target
            )
            validation_loss = F.smooth_l1_loss(
                validation_prediction, validation_target
            )
        return {
            "loss": float(train_loss.item()),
            "explained_variance": self._explained_variance(
                train_target, train_prediction
            ),
            "target_sd": float(train_target.std(unbiased=False).item()),
            "validation_loss": float(validation_loss.item()),
            "validation_explained_variance": self._explained_variance(
                validation_target, validation_prediction
            ),
            "episodes": float(count),
        }

    def _fit_random_reference_replay(self) -> dict[str, float]:
        """Fit and calibrate the fixed uniform-policy cost-to-go in Stage E."""
        owned = self.actor_named_parameters + self.critic_named_parameters
        previous = {
            id(parameter): bool(parameter.requires_grad)
            for _, parameter in owned
        }
        reference_ids = {
            id(parameter) for _, parameter in self.random_reference_named_parameters
        }
        try:
            for _, parameter in owned:
                parameter.requires_grad_(id(parameter) in reference_ids)
            return self._fit_value_replay(
                self.random_reference_replay,
            )
        finally:
            for _, parameter in owned:
                parameter.requires_grad_(previous[id(parameter)])


    def _run_learner_update(self, transitions: list[Transition]) -> Dict[str, float]:
        """Run one stage-specific learner update.

        Stage E fits only the environment/reference-value models. Stage P fits
        one categorical actor iteration from complete common-future simulator
        costs; it does not update or consult a critic. Physical branch scores
        are retained only within the frozen continuation-policy iteration that
        generated them, while policy probabilities are rebuilt before fitting.
        """
        if not transitions:
            raise RuntimeError("Learner update batch is empty")
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise RuntimeError("Training requires both optimizer partitions")
        stage = self._stage_contract()
        training_phase = stage.phase
        actor_enabled = bool(stage.train_actor)
        dynamics_training_enabled = bool(
            stage.train_dynamics and self.learned_nmcc_models_enabled
        )
        critic_value_enabled = bool(
            stage.train_value or not self.nmcc_policy_improvement
        )
        episodes = self._group_episode_indices(transitions)
        total = len(transitions)

        rewards = torch.cat([item.reward.reshape(1) for item in transitions])
        component_rewards = torch.stack([item.reward_components for item in transitions])
        interval_costs = torch.cat(
            [item.interval_cost.reshape(1) for item in transitions]
        )
        sampled_reference_values = torch.cat(
            [item.random_reference_value.reshape(1) for item in transitions]
        )
        values = torch.cat([item.value.reshape(1) for item in transitions])
        component_values = torch.stack([item.value_components for item in transitions])
        dones = torch.cat([item.done.reshape(1) for item in transitions])
        durations = torch.cat(
            [item.elapsed_timesteps.reshape(1) for item in transitions]
        ).to(dtype=torch.float32)
        actions = torch.cat([item.action.reshape(1) for item in transitions])
        action_masks = torch.stack(
            [item.observation_history[-1].action_mask for item in transitions]
        )
        old_log_probabilities = torch.cat(
            [item.log_probability.reshape(1) for item in transitions]
        )
        component_count = len(REWARD_COMPONENT_NAMES)
        if component_rewards.shape != (total, component_count):
            raise RuntimeError("Reward component rollout has an incompatible shape")
        if component_values.shape != (total, component_count):
            raise RuntimeError("Critic component rollout has an incompatible shape")
        if action_masks.shape != (total, self.num_candidate_actions):
            raise RuntimeError("Stored action masks do not match the candidate action space")
        if not torch.allclose(rewards, component_rewards.sum(dim=-1), atol=1e-6, rtol=1e-6):
            raise RuntimeError("Scalar rollout rewards do not equal component rewards")
        if not torch.allclose(values, component_values.sum(dim=-1), atol=1e-5, rtol=1e-5):
            raise RuntimeError("Scalar rollout values do not equal component values")
        if not bool(torch.isfinite(interval_costs).all()) or bool(
            torch.any(interval_costs < 0.0)
        ):
            raise RuntimeError("Complete interval costs must be finite and non-negative")
        if not bool(torch.isfinite(sampled_reference_values).all()):
            raise RuntimeError("Random-reference rollout values must be finite")

        natural_targets = torch.zeros(
            total,
            self.d_momentum,
            dtype=torch.float32,
            device=self.device,
        )
        natural_available = torch.as_tensor(
            [item.natural_outcome_target is not None for item in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        if dynamics_training_enabled:
            for index, transition in enumerate(transitions):
                if bool(natural_available[index]):
                    target = transition.natural_outcome_target.reshape(-1)
                    if target.numel() != self.d_momentum:
                        raise RuntimeError(
                            "Stage E dynamics targets must match the momentum encoder"
                        )
                    natural_targets[index] = target
        pi_mode = bool(self.nmcc_policy_improvement)
        action_count = self.num_candidate_actions
        improvement_available = torch.as_tensor(
            [item.improvement_target is not None for item in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        improvement_prior = torch.zeros(
            total, action_count, dtype=torch.float32, device=self.device
        )
        improvement_target = torch.zeros_like(improvement_prior)
        improvement_exact_mask = torch.zeros_like(improvement_prior)
        improvement_advantage = torch.zeros_like(improvement_prior)
        for index, transition in enumerate(transitions):
            if not bool(improvement_available[index]):
                continue
            improvement_prior[index] = transition.improvement_prior
            improvement_target[index] = transition.improvement_target
            improvement_exact_mask[index] = transition.improvement_exact_mask
            improvement_advantage[index] = transition.improvement_advantage
        if dynamics_training_enabled and not bool(natural_available.all()):
            raise RuntimeError(
                f"{training_phase} requires one physical dynamics target per transition"
            )
        actor_fit_episodes: list[ActorFitEpisode] = []
        if pi_mode and actor_enabled:
            if not bool(improvement_available.all()):
                raise RuntimeError(
                    "Policy improvement requires NMCC contrasts for every decision"
                )
            self._append_actor_replay(transitions, episodes)
            actor_fit_episodes = self._actor_fit_episodes()
        improvement_branch_best = self._branch_best_actions(
            improvement_advantage, improvement_exact_mask
        )
        if pi_mode and bool(improvement_available.any()):
            prior_top1 = (
                self._safe_masked_logits(
                    torch.log(
                        improvement_prior[improvement_available].clamp_min(
                            1e-30
                        )
                    ),
                    improvement_exact_mask[improvement_available].to(torch.bool),
                ).argmax(dim=-1)
                == improvement_branch_best[improvement_available]
            ).to(torch.float32)
            improvement_top1_before = float(prior_top1.mean().item())
        else:
            improvement_top1_before = float("nan")

        with torch.no_grad():
            if stage.train_random_reference:
                if any(
                    item.random_reference_mc_target is None
                    for item in transitions
                ):
                    raise RuntimeError(
                        "Stage E requires a complete uniform-policy Monte Carlo "
                        "target at every decision"
                    )
                random_reference_targets = torch.cat(
                    [
                        item.random_reference_mc_target.reshape(1)
                        for item in transitions
                    ]
                )
                self._append_random_reference_replay(
                    transitions,
                    episodes,
                    random_reference_targets,
                )
            else:
                random_reference_targets = self._cost_to_go_targets(
                    interval_costs,
                    dones,
                    durations,
                )
            component_actor_returns, component_td_targets = self._component_credit_targets(
                component_rewards,
                component_values,
                dones,
                durations,
            )
            actor_returns = component_actor_returns.sum(dim=-1)
            counterfactual_diagnostics = {
                "counterfactual_fraction": 0.0,
                "counterfactual_advantage_mean": 0.0,
                "counterfactual_advantage_sd": 0.0,
                "counterfactual_negative_fraction": 0.0,
                "counterfactual_positive_fraction": 0.0,
                "counterfactual_equal_action_fraction": 0.0,
                "counterfactual_remaining_risk_time_difference_mean": 0.0,
                "counterfactual_casualties_avoided_mean": 0.0,
            }
            if pi_mode and actor_enabled:
                if not bool(improvement_available.all()):
                    raise RuntimeError(
                        "NMCC policy improvement requires a branch contrast "
                        "for every actor transition"
                    )
                row = torch.arange(total, device=self.device)
                if not bool(improvement_exact_mask[row, actions].all()):
                    raise RuntimeError(
                        "The executed action is absent from its simulator-graded support"
                    )
                # NMCC estimates an action contrast against a *uniform-site*
                # reference under matched exogenous futures. The M-step
                # consumes the full table; the executed-action trace below is
                # diagnostic only. It is not the on-policy identity
                # A_pi=Q_pi-V_pi and therefore must never be subtracted from
                # the sampled return to manufacture a critic target.  The
                # critic learns the on-policy savings-over-random return-to-go;
                # NMCC uses the same V_mu/W_pi decomposition for the actor.
                (
                    advantages,
                    raw_advantages,
                    td_targets,
                    component_td_targets,
                ) = self._nmcc_actor_critic_targets(
                    actor_returns,
                    component_actor_returns,
                    improvement_advantage,
                    actions,
                )
                exact_values = improvement_advantage[
                    improvement_exact_mask > 0.5
                ]
                counterfactual_diagnostics.update(
                    {
                        "counterfactual_fraction": 1.0,
                        "counterfactual_advantage_mean": float(
                            exact_values.mean().item()
                        ),
                        "counterfactual_advantage_sd": float(
                            exact_values.std(unbiased=False).item()
                        ),
                        "counterfactual_negative_fraction": float(
                            (exact_values < 0).float().mean().item()
                        ),
                        "counterfactual_positive_fraction": float(
                            (exact_values > 0).float().mean().item()
                        ),
                    }
                )
            else:
                td_targets = component_td_targets.sum(dim=-1)
                advantages, raw_advantages = self._on_policy_advantages(
                    actor_returns,
                    values,
                )
                advantages, counterfactual_diagnostics = (
                    self._blend_counterfactual_advantages(
                        advantages,
                        transitions,
                    )
                )
            baseline_coverage = 1.0
            raw_actor_return_sd = float(actor_returns.std(unbiased=False).item())
            counterfactual_diagnostics["mc_advantage_sd"] = raw_actor_return_sd
            # Retain the historical column as an explicit compatibility alias.
            counterfactual_diagnostics["gae_advantage_sd"] = raw_actor_return_sd

        update_entropy_coef = self._current_entropy_coef()
        update_temperature = self._current_action_temperature()
        self._apply_learning_rate_schedule()
        actor_lr_before = float(self.actor_optimizer.param_groups[0]["lr"])
        critic_lr_before = float(self.critic_optimizer.param_groups[0]["lr"])
        actor_diagnostics_before = self._actor_distribution_diagnostics(
            transitions,
            episodes,
            action_masks,
        )

        value_losses: list[float] = []
        component_loss_records: list[np.ndarray] = []
        critic_gradient_norms: list[float] = []
        natural_losses: list[float] = []
        critic_epochs_completed = 0
        shared_representation = self.representation_mode in {
            "shared_multitask",
            "shared_phasic",
        } or (
            self.representation_mode == "staged_transfer"
            and training_phase == "environment_dynamics"
        )
        representation_drift_kl = 0.0
        representation_rollback = False
        clone_kls: list[float] = []
        critic_representation_gradient_norms: list[float] = []
        critic_head_gradient_norms: list[float] = []
        reference_log_probabilities = None
        if self.representation_mode == "shared_phasic":
            # The policy the critic pass must not move: the one the rollout was
            # collected with (the replay reproduces it exactly).
            reference_log_probabilities = self._policy_log_probabilities(
                transitions,
                episodes,
                action_masks,
                action_temperature=update_temperature,
            )
        self._set_optimizer_partition_trainable(actor=False, critic=True)
        if not stage.train_random_reference:
            for _, parameter in self.random_reference_named_parameters:
                parameter.requires_grad_(False)
        if (
            self.representation_mode == "staged_transfer"
            and training_phase != "environment_dynamics"
        ):
            for _, parameter in self.representation_named_parameters:
                parameter.requires_grad_(False)
        critic_parameters = [parameter for _, parameter in self.critic_named_parameters]
        critic_snapshot: Dict[str, torch.Tensor] = {}
        critic_optimizer_snapshot: dict = {}
        critic_epoch_budget = (
            self.critic_epochs
            if critic_value_enabled
            or dynamics_training_enabled
            or reference_log_probabilities is not None
            else 0
        )
        for _ in range(critic_epoch_budget):
            if self.representation_mode == "shared_phasic":
                critic_snapshot = {
                    name: parameter.detach().clone()
                    for name, parameter in self.critic_named_parameters
                }
                critic_optimizer_snapshot = copy.deepcopy(self.critic_optimizer.state_dict())
            critic_epochs_completed += 1
            for episode_batch in self._iter_episode_minibatches(episodes):
                critic_outputs = self._evaluate_recurrent_sequences(
                    transitions,
                    episode_batch,
                    action_temperature=update_temperature,
                )
                (
                    selected,
                    _,
                    _,
                    components_now,
                    _,
                    natural_now,
                    _,
                    _,
                ) = critic_outputs
                zero = components_now.new_zeros(())
                if critic_value_enabled:
                    component_losses = F.smooth_l1_loss(
                        components_now,
                        component_td_targets[selected],
                        reduction="none",
                    )
                    value_loss = component_losses.mean()
                else:
                    component_losses = torch.zeros_like(components_now)
                    value_loss = zero
                natural_loss = zero
                available_batch = natural_available[selected]
                if dynamics_training_enabled and bool(available_batch.any()):
                    candidate_dynamics = natural_now[available_batch]
                    chosen = actions[selected][available_batch]
                    row = torch.arange(chosen.numel(), device=self.device)
                    predicted_natural = candidate_dynamics[row, chosen]
                    target_natural = natural_targets[selected][available_batch]
                    natural_loss = F.smooth_l1_loss(predicted_natural, target_natural)
                critic_loss = self.nmcc_natural_loss_coef * natural_loss
                if critic_value_enabled:
                    value_weight = (
                        self.nmcc_pi_continuation_loss_coef
                        if stage.use_continuation
                        else 1.0
                    )
                    critic_loss = critic_loss + value_weight * value_loss
                clone_kl = zero
                if reference_log_probabilities is not None:
                    batch_masks = action_masks[selected]
                    clone_kl = self._categorical_kl(
                        reference_log_probabilities[selected],
                        F.log_softmax(
                            self._safe_masked_logits(critic_outputs[1], batch_masks), dim=-1
                        ),
                        batch_masks,
                    ).mean()
                    critic_loss = critic_loss + self.representation_clone_coef * clone_kl
                if not torch.isfinite(critic_loss):
                    raise FloatingPointError(
                        f"Critic/world loss became non-finite in {training_phase}"
                    )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_head_gradient_norms.append(
                    self._gradient_norm(self.critic_head_named_parameters)
                )
                if shared_representation:
                    critic_representation_gradient_norms.append(
                        self._gradient_norm(self.representation_named_parameters)
                    )
                gradient_norm = nn.utils.clip_grad_norm_(critic_parameters, 0.5)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Critic/world gradient norm became non-finite")
                self.critic_optimizer.step()
                self.critic_optimizer_updates += 1
                self.optimizer_updates += 1
                value_losses.append(float(value_loss.detach().item()))
                component_loss_records.append(
                    component_losses.detach().mean(dim=0).cpu().numpy()
                )
                critic_gradient_norms.append(float(gradient_norm.item()))
                natural_losses.append(float(natural_loss.detach().item()))
                clone_kls.append(float(clone_kl.detach().item()))
            if reference_log_probabilities is not None:
                # Transactional policy-preservation check on the full batch: a
                # critic epoch that moved the policy more than the cap is undone
                # together with its optimizer moments.
                drift = float(
                    self._categorical_kl(
                        reference_log_probabilities,
                        self._policy_log_probabilities(
                            transitions,
                            episodes,
                            action_masks,
                            action_temperature=update_temperature,
                        ),
                        action_masks,
                    )
                    .mean()
                    .item()
                )
                if not np.isfinite(drift):
                    raise FloatingPointError("Critic-pass policy drift became non-finite")
                if drift > self.representation_kl_cap:
                    with torch.no_grad():
                        for name, parameter in self.critic_named_parameters:
                            parameter.copy_(critic_snapshot[name])
                    self.critic_optimizer.load_state_dict(critic_optimizer_snapshot)
                    representation_rollback = True
                    critic_epochs_completed -= 1
                    break
                representation_drift_kl = drift

        dynamics_metrics = {
            "loss": 0.0,
            "explained_variance": 0.0,
            **{
                f"explained_variance_{name}": 0.0
                for name in MOMENTUM_FEATURE_NAMES
            },
        }
        if dynamics_training_enabled:
            with torch.no_grad():
                dynamics_outputs = self._evaluate_recurrent_sequences(
                    transitions,
                    episodes,
                    action_temperature=update_temperature,
                )
                selected = dynamics_outputs[0]
                candidate_predictions = dynamics_outputs[5]
                chosen = actions[selected]
                row = torch.arange(chosen.numel(), device=self.device)
                selected_predictions = candidate_predictions[row, chosen]
                ordered_predictions = torch.empty_like(selected_predictions)
                ordered_predictions[selected] = selected_predictions
                target = natural_targets
                dynamics_metrics["loss"] = float(
                    F.smooth_l1_loss(ordered_predictions, target).item()
                )
                dynamics_metrics["explained_variance"] = (
                    self._explained_variance(
                        target.reshape(-1), ordered_predictions.reshape(-1)
                    )
                )
                for index, name in enumerate(MOMENTUM_FEATURE_NAMES):
                    dynamics_metrics[f"explained_variance_{name}"] = (
                        self._explained_variance(
                            target[:, index], ordered_predictions[:, index]
                        )
                    )

        random_reference_metrics = {
            "loss": 0.0,
            "explained_variance": 0.0,
            "target_sd": 0.0,
            "validation_loss": 0.0,
            "validation_explained_variance": 0.0,
            "episodes": float(len(self.random_reference_replay)),
        }
        if stage.train_random_reference:
            random_reference_metrics = self._fit_random_reference_replay()

        policy_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []
        actor_gradient_norms: list[float] = []
        residual_rms_values: list[float] = []
        attempted_kls: list[float] = []
        actor_epochs_completed = 0
        actor_steps_accepted = 0
        kl_rollback = False
        actor_representation_gradient_norms: list[float] = []
        actor_head_gradient_norms: list[float] = []
        self._set_optimizer_partition_trainable(actor=True, critic=False)
        actor_parameters = [parameter for _, parameter in self.actor_named_parameters]
        # Simulator grading supplies matched costs for every screened shelter.
        # This update fits only the current iteration's constrained categorical
        # target: there is no PPO actor pass and no stale actor-target replay.
        if pi_mode and actor_enabled and not bool(improvement_available.all()):
            raise RuntimeError(
                "Rollout policy iteration requires branch contrasts for "
                "every decision"
            )
        epoch_budget = self.actor_epochs
        improvement_policy_kl_before = float("nan")
        improvement_policy_kl_after = float("nan")
        improvement_target_kl_before = float("nan")
        improvement_target_kl_after = float("nan")
        improvement_top1_after = float("nan")
        improvement_expected_advantage_before = float("nan")
        improvement_expected_advantage_after = float("nan")
        improvement_converged = False
        actor_fit_metrics = {
            "loss": 0.0,
            "pairwise_accuracy": 0.0,
            "initial_loss": 0.0,
            "initial_pairwise_accuracy": 0.0,
            "episodes": float(len(actor_fit_episodes)),
            "updates": 0.0,
            "gradient_norm": 0.0,
            "head_gradient_norm": 0.0,
            "representation_gradient_norm": 0.0,
        }

        def policy_statistics() -> Dict[str, float]:
            return self._improvement_policy_statistics(
                transitions,
                episodes,
                action_masks,
                improvement_available,
                improvement_prior,
                improvement_target,
                improvement_advantage,
                improvement_branch_best,
                improvement_exact_mask,
            )

        # Statistics are descriptive only; they never accept, reject, clip or
        # roll back an actor update.
        if pi_mode and actor_enabled:
            before_statistics = policy_statistics()
            improvement_policy_kl_before = before_statistics["kl"]
            improvement_target_kl_before = before_statistics["target_fit_kl"]
            improvement_expected_advantage_before = before_statistics[
                "expected_advantage"
            ]

        if actor_enabled and pi_mode:
            actor_fit_metrics = self._fit_actor_iteration(actor_fit_episodes)
            actor_epochs_completed = int(actor_fit_metrics["updates"])
            actor_steps_accepted = actor_epochs_completed
            policy_losses.append(actor_fit_metrics["loss"])
            actor_gradient_norms.append(
                actor_fit_metrics["gradient_norm"]
            )
            actor_head_gradient_norms.append(
                actor_fit_metrics["head_gradient_norm"]
            )
            actor_representation_gradient_norms.append(
                actor_fit_metrics["representation_gradient_norm"]
            )
        elif actor_enabled:
            for _ in range(epoch_budget):
                epoch_policy_losses: list[float] = []
                epoch_entropies: list[float] = []
                epoch_clip_fractions: list[float] = []
                epoch_gradient_norms: list[float] = []
                epoch_residual_rms: list[float] = []
                epoch_steps = 0
                for episode_batch in self._iter_episode_minibatches(episodes):
                    (
                        selected,
                        logits,
                        _,
                        _,
                        learned_residual,
                        _,
                        _,
                        _,
                    ) = self._evaluate_recurrent_sequences(
                        transitions,
                        episode_batch,
                        action_temperature=update_temperature,
                    )
                    selected_masks = action_masks[selected]
                    logits = self._safe_masked_logits(logits, selected_masks)
                    distribution = self._distribution_from_logits(
                        logits, selected_masks
                    )
                    entropy = self._normalized_categorical_entropy(
                        distribution,
                        selected_masks,
                    ).mean()
                    new_log_probabilities = distribution.log_prob(
                        actions[selected]
                    )
                    _, ratio = self._ppo_log_ratio(
                        new_log_probabilities,
                        old_log_probabilities[selected],
                    )
                    advantage_batch = advantages[selected]
                    unclipped = ratio * advantage_batch
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    ) * advantage_batch
                    policy_loss = -torch.minimum(
                        unclipped, clipped
                    ).mean()
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > self.clip_eps).to(
                            torch.float32
                        )
                    )
                    residual_penalty = learned_residual.square().mean()
                    actor_loss = (
                        policy_loss
                        - update_entropy_coef * entropy
                        + self.residual_penalty_coef * residual_penalty
                    )
                    if not torch.isfinite(actor_loss):
                        raise FloatingPointError("Actor loss became non-finite")
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_representation_gradient_norms.append(
                        self._gradient_norm(self.representation_named_parameters)
                    )
                    actor_head_gradient_norms.append(
                        self._gradient_norm(self.actor_head_named_parameters)
                    )
                    gradient_norm = nn.utils.clip_grad_norm_(actor_parameters, 0.5)
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("Actor gradient norm became non-finite")
                    self.actor_optimizer.step()
                    epoch_steps += 1
                    epoch_policy_losses.append(float(policy_loss.detach().item()))
                    epoch_entropies.append(float(entropy.detach().item()))
                    epoch_clip_fractions.append(float(clip_fraction.detach().item()))
                    epoch_gradient_norms.append(float(gradient_norm.item()))
                    epoch_residual_rms.append(
                        float(torch.sqrt(residual_penalty.detach()).item())
                    )
                attempted_kl = self._full_rollout_policy_kl(
                    transitions,
                    episodes,
                    action_masks,
                    actions,
                    old_log_probabilities,
                    action_temperature=update_temperature,
                )
                attempted_kls.append(attempted_kl)
                actor_epochs_completed += 1
                actor_steps_accepted += epoch_steps
                policy_losses.extend(epoch_policy_losses)
                entropies.extend(epoch_entropies)
                clip_fractions.extend(epoch_clip_fractions)
                actor_gradient_norms.extend(epoch_gradient_norms)
                residual_rms_values.extend(epoch_residual_rms)

        if pi_mode and actor_enabled:
            final_statistics = policy_statistics()
            improvement_policy_kl_after = final_statistics["kl"]
            improvement_target_kl_after = final_statistics["target_fit_kl"]
            improvement_top1_after = final_statistics["top1"]
            improvement_expected_advantage_after = final_statistics[
                "expected_advantage"
            ]
            improvement_converged = bool(
                improvement_target_kl_after <= self.nmcc_pi_fit_tolerance
            )
        if actor_enabled:
            retained_kl = self._full_rollout_policy_kl(
                transitions,
                episodes,
                action_masks,
                actions,
                old_log_probabilities,
                action_temperature=update_temperature,
            )
        else:
            retained_kl = 0.0
        actor_update_accepted = actor_steps_accepted > 0
        if actor_update_accepted:
            self.actor_optimizer_updates += actor_steps_accepted
            self.actor_rollout_updates += 1
            self.optimizer_updates += actor_steps_accepted
        self._set_optimizer_partition_trainable(actor=True, critic=True)

        # Diagnostics must describe the deployed post-update model, not the
        # stale values captured before any critic or NMCC fit occurred.
        with torch.no_grad():
            diagnostic_outputs = self._evaluate_recurrent_sequences(
                transitions,
                episodes,
                action_temperature=update_temperature,
            )
            diagnostic_selected = diagnostic_outputs[0]
            post_component_values = torch.empty_like(component_values)
            post_component_values[diagnostic_selected] = diagnostic_outputs[3]
            post_values = post_component_values.sum(dim=-1)
            post_random_reference_values = torch.empty_like(
                sampled_reference_values
            )
            post_random_reference_values[diagnostic_selected] = (
                diagnostic_outputs[7]
            )
        actor_diagnostics_after = self._actor_distribution_diagnostics(
            transitions,
            episodes,
            action_masks,
        )

        self.rollout_updates_completed += 1
        component_loss_mean = (
            np.mean(np.stack(component_loss_records), axis=0)
            if component_loss_records
            else np.zeros(component_count, dtype=np.float64)
        )
        actor_lr_after = float(self.actor_optimizer.param_groups[0]["lr"])
        critic_lr_after = float(self.critic_optimizer.param_groups[0]["lr"])
        actor_gradient_norm = float(np.mean(actor_gradient_norms)) if actor_gradient_norms else 0.0
        critic_gradient_norm = float(np.mean(critic_gradient_norms)) if critic_gradient_norms else 0.0
        diagnostics = {
            **{name: 0.0 for name in self.ROLLOUT_PI_DIAGNOSTIC_NAMES},
            "transitions": float(total),
            "sequence_episodes": float(len(episodes)),
            "observation_frames": float(
                sum(len(item.observation_history) for item in transitions)
            ),
            "mean_observation_history": float(
                np.mean([len(item.observation_history) for item in transitions])
            ),
            "mean_credit_duration": float(durations.mean().item()),
            "maximum_credit_duration": float(durations.max().item()),
            "epochs_completed": float(critic_epochs_completed + actor_epochs_completed),
            "critic_epochs_completed": float(critic_epochs_completed),
            "actor_epochs_completed": float(actor_epochs_completed),
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "actor_policy_entropy_before": actor_diagnostics_before["entropy"],
            "actor_policy_entropy_after": actor_diagnostics_after["entropy"],
            "actor_policy_entropy_change": (
                actor_diagnostics_after["entropy"]
                - actor_diagnostics_before["entropy"]
            ),
            "attempted_kl": (
                retained_kl
                if pi_mode and actor_enabled
                else float(max(attempted_kls)) if attempted_kls else 0.0
            ),
            "approximate_kl": retained_kl,
            "kl_rollback": float(kl_rollback),
            "actor_update_accepted": float(actor_update_accepted),
            "actor_update_rejected": float(kl_rollback),
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "gradient_norm": actor_gradient_norm or critic_gradient_norm,
            "actor_gradient_norm": actor_gradient_norm,
            "critic_gradient_norm": critic_gradient_norm,
            "explained_variance": self._explained_variance(td_targets, post_values),
            "actor_mc_return_sd": raw_actor_return_sd,
            "actor_value_baseline_mean": float(post_values.mean().item()),
            "actor_raw_advantage_mean": float(raw_advantages.mean().item()),
            "actor_raw_advantage_sd": float(
                raw_advantages.std(unbiased=False).item()
            ),
            "actor_advantage_sd": float(advantages.std(unbiased=False).item()),
            "actor_baseline_coverage": baseline_coverage,
            "critic_td_error_sd": float(
                (td_targets - post_values).std(unbiased=False).item()
            ),
            "residual_rms": actor_diagnostics_after["residual_rms"],
            "actor_residual_rms_before_update": actor_diagnostics_before[
                "residual_rms"
            ],
            "actor_residual_rms_after_update": actor_diagnostics_after[
                "residual_rms"
            ],
            "actor_residual_abs_max_after_update": actor_diagnostics_after[
                "residual_abs_max"
            ],
            "actor_learning_rate_before_update": actor_lr_before,
            "learning_rate": actor_lr_after,
            "actor_learning_rate": actor_lr_after,
            "critic_learning_rate_before_update": critic_lr_before,
            "critic_learning_rate": critic_lr_after,
            "policy_iterations_completed": float(self.actor_rollout_updates),
            "actor_gradient_steps": float(self.actor_optimizer_updates),
            "critic_gradient_steps": float(self.critic_optimizer_updates),
            "entropy_coefficient": float(update_entropy_coef),
            "action_temperature": float(update_temperature),
            "exploration_rate": float(self._current_exploration_rate()),
            "rollout_pi_actor_enabled": float(actor_enabled and pi_mode),
            "environment_dynamics_loss": dynamics_metrics["loss"],
            "random_reference_value_loss": float(
                F.smooth_l1_loss(
                    post_random_reference_values,
                    random_reference_targets,
                ).item()
            ),
            "random_reference_value_explained_variance": (
                self._explained_variance(
                    random_reference_targets,
                    post_random_reference_values,
                )
            ),
            "random_reference_value_bias": float(
                (
                    post_random_reference_values
                    - random_reference_targets
                ).mean().item()
            ),
            "random_reference_replay_loss": random_reference_metrics["loss"],
            "random_reference_replay_explained_variance": (
                random_reference_metrics["explained_variance"]
            ),
            "random_reference_validation_loss": (
                random_reference_metrics["validation_loss"]
            ),
            "random_reference_validation_explained_variance": (
                random_reference_metrics["validation_explained_variance"]
            ),
            "random_reference_replay_episodes": (
                random_reference_metrics["episodes"]
            ),
            "environment_dynamics_explained_variance": dynamics_metrics[
                "explained_variance"
            ],
            **{
                f"environment_dynamics_explained_variance_{name}": dynamics_metrics[
                    f"explained_variance_{name}"
                ]
                for name in MOMENTUM_FEATURE_NAMES
            },
            **{
                key: float(value)
                for key, value in counterfactual_diagnostics.items()
            },
            **self._learner_flow_diagnostics(
                shared=shared_representation,
                drift_kl=representation_drift_kl,
                rollback=representation_rollback,
                clone_kls=clone_kls,
                critic_representation=critic_representation_gradient_norms,
                critic_head=critic_head_gradient_norms,
                actor_representation=actor_representation_gradient_norms,
                actor_head=actor_head_gradient_norms,
            ),
            **(
                self._improvement_diagnostics(
                    coverage=float(
                        improvement_available.to(torch.float32).mean().item()
                    ),
                    sampled_action_coverage=float(
                        (
                            improvement_exact_mask[improvement_available]
                            .to(torch.bool)
                            .any(dim=0)
                            & action_masks[improvement_available]
                            .to(torch.bool)
                            .any(dim=0)
                        )
                        .sum()
                        .item()
                        / max(
                            1,
                            int(
                                action_masks[improvement_available]
                                .to(torch.bool)
                                .any(dim=0)
                                .sum()
                                .item()
                            ),
                        )
                    ),
                    policy_kl_before=improvement_policy_kl_before,
                    policy_kl_after=improvement_policy_kl_after,
                    target_kl_before=improvement_target_kl_before,
                    target_kl_after=improvement_target_kl_after,
                    pairwise_accuracy_before=actor_fit_metrics[
                        "initial_pairwise_accuracy"
                    ],
                    pairwise_accuracy_after=actor_fit_metrics[
                        "pairwise_accuracy"
                    ],
                    top1_before=improvement_top1_before,
                    top1_after=improvement_top1_after,
                    expected_advantage_before=(
                        improvement_expected_advantage_before
                    ),
                    expected_advantage_after=(
                        improvement_expected_advantage_after
                    ),
                    converged=improvement_converged,
                )
                if self.nmcc_policy_improvement
                else {}
            ),
        }
        for component_index, component_name in enumerate(REWARD_COMPONENT_NAMES):
            diagnostics[f"value_loss_{component_name}"] = float(
                component_loss_mean[component_index]
            )
            diagnostics[f"explained_variance_{component_name}"] = self._explained_variance(
                component_td_targets[:, component_index],
                post_component_values[:, component_index],
            )
        if pi_mode:
            for legacy_name in (
                "epochs_completed",
                "critic_epochs_completed",
                "actor_epochs_completed",
                "value_loss",
                "attempted_kl",
                "approximate_kl",
                "kl_rollback",
                "actor_update_accepted",
                "actor_update_rejected",
                "clip_fraction",
                "gradient_norm",
                "explained_variance",
                "actor_mc_return_sd",
                "actor_value_baseline_mean",
                "actor_raw_advantage_mean",
                "actor_raw_advantage_sd",
                "actor_advantage_sd",
                "actor_baseline_coverage",
                "critic_td_error_sd",
                "residual_rms",
                "learning_rate",
                *tuple(counterfactual_diagnostics),
                *(f"value_loss_{name}" for name in REWARD_COMPONENT_NAMES),
                *(
                    f"explained_variance_{name}"
                    for name in REWARD_COMPONENT_NAMES
                ),
            ):
                diagnostics.pop(legacy_name, None)
        return diagnostics

    def _resolve_diagnostics_schema(self, columns: Sequence[str]) -> None:
        """Never append rows under a header with a different column set.

        A CSV written by an earlier model version keeps its own header; rows of
        the current schema go to a sibling file named after it instead of being
        silently shifted into the wrong columns.
        """
        path = self.diagnostics_path
        stem, extension = os.path.splitext(path)
        suffix = 0
        while os.path.exists(path):
            with open(path, "r", newline="") as handle:
                header = next(csv.reader(handle), None)
            if header is None or list(header) == list(columns):
                break
            suffix += 1
            path = f"{stem}_schema_v{MODEL_VERSION}" + (
                f"_{suffix}" if suffix > 1 else ""
            ) + (extension or ".csv")
        self.diagnostics_path = path

    def _append_learner_diagnostics(self, diagnostics: Dict[str, float]) -> None:
        directory = os.path.dirname(os.path.abspath(self.diagnostics_path))
        os.makedirs(directory, exist_ok=True)
        common_columns: tuple[str, ...] = (
            "episode",
            "optimizer_steps",
            "policy_iterations_completed",
            "actor_gradient_steps",
            "critic_gradient_steps",
            "episode_return",
            "random_reference_initial_value",
            "random_reference_realized_cost",
            "random_reference_identity_return",
            "reference_potential_reduction",
            "reward_accounting_gap",
            "controllable_interval_cost",
            "random_site_advantage_diagnostic",
            "remaining_risk_time_before",
            "remaining_risk_time_after",
            "raw_remaining_risk_time_reduction",
            "remaining_risk_time_reduction",
            "realized_plain_time_to_safety",
            "realized_hazard_exposure_time_to_safety",
            "expected_casualty_person_time_cost",
            "expected_casualties",
            "clearance_tail_person_time_cost",
            "clearance_tail_mean_minutes",
            "clearance_t90_minutes",
            "clearance_time_minutes",
            "realized_terminal_plain_time",
            "realized_terminal_hazard_exposure_time",
            "realized_hazard_weighted_time_to_safety",
            "normalized_realized_hazard_weighted_time_to_safety",
            "objective_episode_score",
            "objective_normalized_cost",
            "objective_active_person_time",
            "objective_hazard_exposure_person_time",
            "objective_expected_casualty_person_time",
            "objective_clearance_tail_person_time",
            "objective_terminal_residual_person_time",
            "objective_hazard_weighted_person_time_cost",
            "decisions",
            "training_episode_has_decision",
            "observation_frames_seen",
            "optimizer_updated",
            "rollout_episodes_pending",
            "rollout_transitions_pending",
            "rollout_flushed_at_campaign_end",
            "transitions",
            "sequence_episodes",
            "observation_frames",
            "mean_observation_history",
            "mean_credit_duration",
            "maximum_credit_duration",
            "policy_loss",
            "behavior_policy_entropy",
            "actor_policy_entropy_before",
            "actor_policy_entropy_after",
            "actor_policy_entropy_change",
            "actor_gradient_norm",
            "critic_gradient_norm",
            "actor_residual_rms_before_update",
            "actor_residual_rms_after_update",
            "actor_residual_abs_max_after_update",
            "actor_learning_rate_before_update",
            "actor_learning_rate",
            "critic_learning_rate_before_update",
            "critic_learning_rate",
            "entropy_coefficient",
            "action_temperature",
            "exploration_rate",
            "training_stage_update_index",
            "training_stage_total_updates",
            "training_stage_progress",
            "learning_reward_scale",
            "rollout_pi_actor_enabled",
            "environment_dynamics_loss",
            "random_reference_value_loss",
            "random_reference_value_explained_variance",
            "random_reference_value_bias",
            "random_reference_replay_loss",
            "random_reference_replay_explained_variance",
            "random_reference_validation_loss",
            "random_reference_validation_explained_variance",
            "random_reference_replay_episodes",
        )
        optimizer_columns = (
            self.ROLLOUT_PI_DIAGNOSTIC_NAMES
            if self.nmcc_policy_improvement
            else self.DIRECT_PPO_DIAGNOSTIC_NAMES
        )
        columns: tuple[str, ...] = (
            *common_columns,
            *optimizer_columns,
            *self.LEARNER_FLOW_DIAGNOSTIC_NAMES,
        )
        self._resolve_diagnostics_schema(columns)
        new_file = not os.path.exists(self.diagnostics_path)
        with open(self.diagnostics_path, "a", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=columns)
            if new_file:
                writer.writeheader()
            row = {column: diagnostics.get(column, 0.0) for column in columns}
            row["episode"] = self.episodes_completed
            row["optimizer_steps"] = self.optimizer_updates
            writer.writerow(row)
