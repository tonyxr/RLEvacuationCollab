#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision-epoch PPO bridge for administrator-selected shelter candidates."""

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
)
from GNN import (
    ACTOR_PRIORS,
    HEURISTIC_PRIOR_SCALE,
    DEFAULT_EMBED_DIM,
    DEFAULT_MESSAGE_LAYERS,
    DEFAULT_NMCC_ENSEMBLE_SIZE,
    DEFAULT_TEMPORAL_DIM,
    NMCC_OUTCOME_NAMES,
    VALUE_COMPONENT_NAMES,
    EvacPolicy,
    fit_gnn,
)
from RewardProcessor import (
    DEFAULT_SAFE_COMPLETION_WEIGHT,
    REWARD_COMPONENT_NAMES,
    RewardBreakdown,
    RewardProcessor,
)


MODEL_VERSION = 28
# --- NMCC policy improvement (NMCC-PI) --------------------------------------
# See NMCCPolicyImprovement.py and docs/NMCC_POLICY_IMPROVEMENT_20260921.md.
from NMCCPIConfig import (  # noqa: E402  (torch-free shared contract)
    DEFAULT_NMCC_PI_ACTOR_EPOCHS,
    DEFAULT_NMCC_PI_ACTOR_OBJECTIVE,
    DEFAULT_NMCC_PI_BRANCH_HORIZON,
    DEFAULT_NMCC_PI_EPSILON,
    DEFAULT_NMCC_PI_ETA_MIN,
    DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS,
    DEFAULT_NMCC_PI_FIT_TOLERANCE,
    DEFAULT_NMCC_PI_FULL_HORIZON_DECISIONS,
    DEFAULT_NMCC_PI_GATE_SPEARMAN,
    DEFAULT_NMCC_PI_GATE_UPDATES,
    DEFAULT_NMCC_PI_KL_CAP,
    DEFAULT_NMCC_PI_MAX_BRANCHES,
    DEFAULT_NMCC_PI_MODEL_UNCERTAINTY_PENALTY,
    DEFAULT_NMCC_PI_RANK_MARGIN,
    DEFAULT_NMCC_PI_RANK_MARGIN_COEF,
    DEFAULT_NMCC_PI_RANKING_TEMPERATURE,
    DEFAULT_NMCC_PI_REPLAY_MAX_EPISODES,
    DEFAULT_NMCC_PI_REPLAY_EPOCHS,
    DEFAULT_NMCC_PI_VALIDATION_FRACTION,
    DEFAULT_NMCC_PI_EARLY_STOPPING_PATIENCE,
    DEFAULT_NMCC_PI_MIN_VALIDATION_STATES,
    DEFAULT_NMCC_PI_VALIDATION_GAIN_Z,
    DEFAULT_NMCC_PI_REPLAY_REFIT,
    DEFAULT_NMCC_PI_TAPES,
    DEFAULT_NMCC_PI_VALUE_LOSS_COEF,
    DEFAULT_NMCC_PI_VALUE_SCALE,
    DEFAULT_REPRESENTATION_CLONE_COEF,
    DEFAULT_REPRESENTATION_KL_CAP,
    DEFAULT_REPRESENTATION_MODE,
    DEFAULT_EXPLORATION_RATE_END,
    DEFAULT_EXPLORATION_RATE_START,
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
DEFAULT_ACTOR_BASELINE_DECAY = 0.95
DEFAULT_ADVANTAGE_SCALE_FLOOR = 0.05
DEFAULT_CLIP_EPS = 0.10
DEFAULT_PPO_EPOCHS = 4
DEFAULT_TARGET_KL = 0.015
DEFAULT_RESIDUAL_PENALTY_COEF = 1e-3
DEFAULT_MINIMUM_LR_FRACTION = 0.10
DEFAULT_KL_LR_REDUCTION = 0.50
DEFAULT_KL_LR_GROWTH = 1.05
DEFAULT_NMCC_NATURAL_LOSS_COEF = 0.50
DEFAULT_NMCC_CAUSAL_LOSS_COEF = 1.00
DEFAULT_NMCC_DUELING_LOSS_COEF = 0.25
DEFAULT_NMCC_TEACHER_COEF = 0.20
DEFAULT_NMCC_TEACHER_DECAY_UPDATES = 40
DEFAULT_NMCC_GUIDANCE_MAX = 0.50
DEFAULT_NMCC_GUIDANCE_WARMUP_UPDATES = 2
DEFAULT_NMCC_GUIDANCE_RAMP_UPDATES = 16
DEFAULT_NMCC_UNCERTAINTY_PENALTY = 0.50
DEFAULT_ENTROPY_COEF_END = 1e-3
DEFAULT_EXPLORATION_DECAY_UPDATES = 64
DEFAULT_TEMPERATURE_START = 1.35
DEFAULT_TEMPERATURE_END = 1.0
NMCC_TRAINING_PHASES = (
    "natural_pretrain",
    "causal_pretrain",
    "controller_warmup",
    "joint_optimization",
)

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


@dataclass
class Transition:
    observation_history: tuple[ObservationFrame, ...]
    action: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor
    value_components: torch.Tensor
    reward: torch.Tensor
    reward_components: torch.Tensor
    done: torch.Tensor
    elapsed_timesteps: torch.Tensor
    # NMCC counterfactual advantage for this decision, when an exact paired
    # WAIT branch was collected. ``None`` keeps the complete MC actor target.
    counterfactual_advantage: Optional[torch.Tensor] = None
    counterfactual_components: Optional[torch.Tensor] = None
    natural_outcome_target: Optional[torch.Tensor] = None
    causal_outcome_target: Optional[torch.Tensor] = None
    # NMCC-PI: supervised candidate-score target (or legacy KL target), frozen
    # score policy, exact branch support/advantages, all-candidate factored
    # physical effects, and ensemble bootstrap weights.
    improvement_target: Optional[torch.Tensor] = None
    improvement_behavior: Optional[torch.Tensor] = None
    improvement_exact_mask: Optional[torch.Tensor] = None
    improvement_advantage: Optional[torch.Tensor] = None
    improvement_bootstrap: Optional[torch.Tensor] = None
    improvement_natural_outcome: Optional[torch.Tensor] = None
    improvement_outcome_effect: Optional[torch.Tensor] = None
    improvement_base_action: Optional[torch.Tensor] = None


@dataclass
class PendingDecision:
    observation: RegionalObservation
    decision: PolicyDecision
    receipt: RegionalActionReceipt
    observation_history: tuple[ObservationFrame, ...]
    active_person_time: float = 0.0
    hazard_exposure_person_time: float = 0.0
    elapsed_timesteps: int = 0
    # Return of the matched no-deployment branch taken from this decision's
    # state under the same structural noise, and the critic's value at that
    # branch's end.  Both stay NaN when counterfactual branching is off.
    wait_return: float = float("nan")
    wait_bootstrap_value: float = float("nan")
    wait_component_return: Optional[np.ndarray] = None
    wait_bootstrap_components: Optional[np.ndarray] = None
    wait_outcome_target: Optional[np.ndarray] = None
    wait_steps: int = 0
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


@dataclass
class RecurrentPolicyStep:
    frame: ObservationFrame
    logits: torch.Tensor
    prior_logits: torch.Tensor
    value: torch.Tensor
    value_components: torch.Tensor
    natural_outcomes: torch.Tensor
    causal_outcome_samples: torch.Tensor
    causal_component_mean: torch.Tensor
    causal_component_std: torch.Tensor
    robust_causal_score: torch.Tensor
    teacher_logits: torch.Tensor
    improvement_value_samples: Optional[torch.Tensor] = None


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
        actor_baseline_decay: float = DEFAULT_ACTOR_BASELINE_DECAY,
        advantage_scale_floor: float = DEFAULT_ADVANTAGE_SCALE_FLOOR,
        minibatch_size: int = 32,
        entropy_coef: float = DEFAULT_ENTROPY_COEF,
        target_kl: float = DEFAULT_TARGET_KL,
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
        nmcc_ensemble_size: int = DEFAULT_NMCC_ENSEMBLE_SIZE,
        nmcc_natural_loss_coef: float = DEFAULT_NMCC_NATURAL_LOSS_COEF,
        nmcc_causal_loss_coef: float = DEFAULT_NMCC_CAUSAL_LOSS_COEF,
        nmcc_dueling_loss_coef: float = DEFAULT_NMCC_DUELING_LOSS_COEF,
        nmcc_teacher_coef: float = DEFAULT_NMCC_TEACHER_COEF,
        nmcc_teacher_decay_updates: int = DEFAULT_NMCC_TEACHER_DECAY_UPDATES,
        nmcc_guidance_max: float = DEFAULT_NMCC_GUIDANCE_MAX,
        nmcc_guidance_warmup_updates: int = DEFAULT_NMCC_GUIDANCE_WARMUP_UPDATES,
        nmcc_guidance_ramp_updates: int = DEFAULT_NMCC_GUIDANCE_RAMP_UPDATES,
        nmcc_uncertainty_penalty: float = DEFAULT_NMCC_UNCERTAINTY_PENALTY,
        entropy_coef_end: float = DEFAULT_ENTROPY_COEF_END,
        exploration_decay_updates: int = DEFAULT_EXPLORATION_DECAY_UPDATES,
        action_temperature_start: float = DEFAULT_TEMPERATURE_START,
        action_temperature_end: float = DEFAULT_TEMPERATURE_END,
        nmcc_natural_pretrain_rollouts: int = 0,
        nmcc_causal_pretrain_rollouts: int = 0,
        nmcc_controller_warmup_rollouts: int = 0,
        nmcc_joint_counterfactual_weight: float = 1.0,
        nmcc_policy_improvement: bool = False,
        nmcc_pi_epsilon: float = DEFAULT_NMCC_PI_EPSILON,
        nmcc_pi_eta_min: float = DEFAULT_NMCC_PI_ETA_MIN,
        nmcc_pi_kl_cap: float = DEFAULT_NMCC_PI_KL_CAP,
        nmcc_pi_tapes: int = DEFAULT_NMCC_PI_TAPES,
        nmcc_pi_exhaustive_decisions: int = DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS,
        nmcc_pi_max_branches: int = DEFAULT_NMCC_PI_MAX_BRANCHES,
        nmcc_pi_base_policy: str = "risk_reduction",
        nmcc_pi_value_scale: float = DEFAULT_NMCC_PI_VALUE_SCALE,
        nmcc_pi_value_loss_coef: float = DEFAULT_NMCC_PI_VALUE_LOSS_COEF,
        nmcc_pi_model_fill: bool = False,
        nmcc_pi_gate_spearman: float = DEFAULT_NMCC_PI_GATE_SPEARMAN,
        nmcc_pi_gate_updates: int = DEFAULT_NMCC_PI_GATE_UPDATES,
        nmcc_pi_model_uncertainty_penalty: float = DEFAULT_NMCC_PI_MODEL_UNCERTAINTY_PENALTY,
        nmcc_pi_actor_epochs: int = DEFAULT_NMCC_PI_ACTOR_EPOCHS,
        nmcc_pi_fit_tolerance: float = DEFAULT_NMCC_PI_FIT_TOLERANCE,
        nmcc_pi_branch_horizon: int = DEFAULT_NMCC_PI_BRANCH_HORIZON,
        nmcc_pi_full_horizon_decisions: int = DEFAULT_NMCC_PI_FULL_HORIZON_DECISIONS,
        nmcc_pi_actor_objective: str = DEFAULT_NMCC_PI_ACTOR_OBJECTIVE,
        nmcc_pi_ranking_temperature: float = DEFAULT_NMCC_PI_RANKING_TEMPERATURE,
        nmcc_pi_rank_margin: float = DEFAULT_NMCC_PI_RANK_MARGIN,
        nmcc_pi_rank_margin_coef: float = DEFAULT_NMCC_PI_RANK_MARGIN_COEF,
        nmcc_pi_replay_max_episodes: int = DEFAULT_NMCC_PI_REPLAY_MAX_EPISODES,
        nmcc_pi_replay_epochs: int = DEFAULT_NMCC_PI_REPLAY_EPOCHS,
        nmcc_pi_validation_fraction: float = DEFAULT_NMCC_PI_VALIDATION_FRACTION,
        nmcc_pi_early_stopping_patience: int = DEFAULT_NMCC_PI_EARLY_STOPPING_PATIENCE,
        nmcc_pi_min_validation_states: int = DEFAULT_NMCC_PI_MIN_VALIDATION_STATES,
        nmcc_pi_validation_gain_z: float = DEFAULT_NMCC_PI_VALIDATION_GAIN_Z,
        nmcc_pi_replay_refit: bool = DEFAULT_NMCC_PI_REPLAY_REFIT,
        exploration_rate_start: float = DEFAULT_EXPLORATION_RATE_START,
        exploration_rate_end: float = DEFAULT_EXPLORATION_RATE_END,
        learning_rate_schedule: str = DEFAULT_LR_SCHEDULE,
        lr_warmup_updates: int = DEFAULT_LR_WARMUP_UPDATES,
        lr_decay_updates: int = DEFAULT_LR_DECAY_UPDATES,
        actor_lr_min_fraction: float = DEFAULT_ACTOR_LR_MIN_FRACTION,
        critic_lr_min_fraction: float = DEFAULT_CRITIC_LR_MIN_FRACTION,
        actor_prior: str = "active_population",
        actor_prior_scale: float = HEURISTIC_PRIOR_SCALE,
        representation_mode: str = DEFAULT_REPRESENTATION_MODE,
        representation_clone_coef: float = DEFAULT_REPRESENTATION_CLONE_COEF,
        representation_kl_cap: float = DEFAULT_REPRESENTATION_KL_CAP,
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
        self.actor_baseline_decay = self._bounded_float(
            "actor_baseline_decay", actor_baseline_decay, 0.0, 1.0
        )
        if self.actor_baseline_decay >= 1.0:
            raise ValueError("actor_baseline_decay must be less than 1")
        self.advantage_scale_floor = self._positive_float(
            "advantage_scale_floor", advantage_scale_floor
        )
        self.minibatch_size = self._positive_int("minibatch_size", minibatch_size)
        self.entropy_coef = self._bounded_float("entropy_coef", entropy_coef, 0.0, float("inf"))
        self.target_kl = self._positive_float("target_kl", target_kl)
        self.rollout_episodes = self._positive_int("rollout_episodes", rollout_episodes)
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
        self.nmcc_ensemble_size = self._positive_int(
            "nmcc_ensemble_size", nmcc_ensemble_size
        )
        self.nmcc_natural_loss_coef = self._bounded_float(
            "nmcc_natural_loss_coef", nmcc_natural_loss_coef, 0.0, float("inf")
        )
        self.nmcc_causal_loss_coef = self._bounded_float(
            "nmcc_causal_loss_coef", nmcc_causal_loss_coef, 0.0, float("inf")
        )
        self.nmcc_dueling_loss_coef = self._bounded_float(
            "nmcc_dueling_loss_coef", nmcc_dueling_loss_coef, 0.0, float("inf")
        )
        self.nmcc_teacher_coef = self._bounded_float(
            "nmcc_teacher_coef", nmcc_teacher_coef, 0.0, float("inf")
        )
        self.nmcc_teacher_decay_updates = self._positive_int(
            "nmcc_teacher_decay_updates", nmcc_teacher_decay_updates
        )
        self.nmcc_guidance_max = self._bounded_float(
            "nmcc_guidance_max", nmcc_guidance_max, 0.0, float("inf")
        )
        self.nmcc_guidance_warmup_updates = max(0, int(nmcc_guidance_warmup_updates))
        self.nmcc_guidance_ramp_updates = self._positive_int(
            "nmcc_guidance_ramp_updates", nmcc_guidance_ramp_updates
        )
        self.nmcc_uncertainty_penalty = self._bounded_float(
            "nmcc_uncertainty_penalty", nmcc_uncertainty_penalty, 0.0, float("inf")
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
        self.nmcc_natural_pretrain_rollouts = max(
            0, int(nmcc_natural_pretrain_rollouts)
        )
        self.nmcc_causal_pretrain_rollouts = max(
            0, int(nmcc_causal_pretrain_rollouts)
        )
        self.nmcc_controller_warmup_rollouts = max(
            0, int(nmcc_controller_warmup_rollouts)
        )
        self.nmcc_joint_counterfactual_weight = self._bounded_float(
            "nmcc_joint_counterfactual_weight",
            nmcc_joint_counterfactual_weight,
            0.0,
            1.0,
        )
        # --- NMCC policy improvement ----------------------------------------
        # Exact within-state targets from CRN-paired candidate branches (see
        # NMCCPolicyImprovement.py). Off by default; v25 fits a direct
        # candidate-score ranking, while ``kl_target`` retains the v24 ablation.
        self.nmcc_policy_improvement = bool(nmcc_policy_improvement)
        self.nmcc_pi_epsilon = self._positive_float("nmcc_pi_epsilon", nmcc_pi_epsilon)
        self.nmcc_pi_eta_min = self._positive_float("nmcc_pi_eta_min", nmcc_pi_eta_min)
        self.nmcc_pi_kl_cap = self._positive_float("nmcc_pi_kl_cap", nmcc_pi_kl_cap)
        if self.nmcc_pi_kl_cap < self.nmcc_pi_epsilon:
            # The trust region bounds the same KL direction as the E-step; a
            # cap below epsilon would forbid the policy from ever reaching its
            # own target.
            raise ValueError("nmcc_pi_kl_cap must be at least nmcc_pi_epsilon")
        self.nmcc_pi_tapes = self._positive_int("nmcc_pi_tapes", nmcc_pi_tapes)
        self.nmcc_pi_exhaustive_decisions = max(0, int(nmcc_pi_exhaustive_decisions))
        self.nmcc_pi_max_branches = self._positive_int(
            "nmcc_pi_max_branches", nmcc_pi_max_branches
        )
        if self.nmcc_pi_max_branches < 2:
            raise ValueError("nmcc_pi_max_branches must allow a within-state contrast")
        self.nmcc_pi_base_policy = str(nmcc_pi_base_policy)
        if self.nmcc_pi_base_policy not in NPI.BASE_POLICIES:
            raise ValueError(
                f"nmcc_pi_base_policy must be one of {sorted(NPI.BASE_POLICIES)}"
            )
        self.nmcc_pi_value_scale = self._positive_float(
            "nmcc_pi_value_scale", nmcc_pi_value_scale
        )
        self.nmcc_pi_value_loss_coef = self._bounded_float(
            "nmcc_pi_value_loss_coef", nmcc_pi_value_loss_coef, 0.0, float("inf")
        )
        self.nmcc_pi_model_fill = bool(nmcc_pi_model_fill)
        self.nmcc_pi_gate_spearman = self._bounded_float(
            "nmcc_pi_gate_spearman", nmcc_pi_gate_spearman, -1.0, 1.0
        )
        self.nmcc_pi_gate_updates = self._positive_int(
            "nmcc_pi_gate_updates", nmcc_pi_gate_updates
        )
        # The M-step is a supervised fit to a fixed target, not a surrogate
        # that is only valid near pi_old, so it gets its own epoch budget and
        # stops once the remaining KL(q || pi) falls below a fraction of the
        # KL the E-step asked for.
        self.nmcc_pi_actor_epochs = self._positive_int(
            "nmcc_pi_actor_epochs", nmcc_pi_actor_epochs
        )
        self.nmcc_pi_fit_tolerance = self._bounded_float(
            "nmcc_pi_fit_tolerance", nmcc_pi_fit_tolerance, 0.0, 1.0, lower_open=True
        )
        self.nmcc_pi_model_uncertainty_penalty = self._bounded_float(
            "nmcc_pi_model_uncertainty_penalty",
            nmcc_pi_model_uncertainty_penalty,
            0.0,
            float("inf"),
        )
        self.nmcc_pi_branch_horizon = self._positive_int(
            "nmcc_pi_branch_horizon", nmcc_pi_branch_horizon
        )
        self.nmcc_pi_full_horizon_decisions = max(
            0, int(nmcc_pi_full_horizon_decisions)
        )
        self.nmcc_pi_actor_objective = str(nmcc_pi_actor_objective).strip().lower()
        if self.nmcc_pi_actor_objective not in {"kl_target", "score_ranking", "value_lcb"}:
            raise ValueError(
                "nmcc_pi_actor_objective must be 'kl_target', 'score_ranking', or 'value_lcb'"
            )
        self.nmcc_pi_ranking_temperature = self._positive_float(
            "nmcc_pi_ranking_temperature", nmcc_pi_ranking_temperature
        )
        self.nmcc_pi_rank_margin = self._bounded_float(
            "nmcc_pi_rank_margin", nmcc_pi_rank_margin, 0.0, float("inf")
        )
        self.nmcc_pi_rank_margin_coef = self._bounded_float(
            "nmcc_pi_rank_margin_coef", nmcc_pi_rank_margin_coef, 0.0, float("inf")
        )
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
        self.nmcc_pi_early_stopping_patience = self._positive_int(
            "nmcc_pi_early_stopping_patience", nmcc_pi_early_stopping_patience
        )
        self.nmcc_pi_min_validation_states = self._positive_int(
            "nmcc_pi_min_validation_states", nmcc_pi_min_validation_states
        )
        self.nmcc_pi_validation_gain_z = self._bounded_float(
            "nmcc_pi_validation_gain_z", nmcc_pi_validation_gain_z, 0.0, float("inf")
        )
        self.nmcc_pi_replay_refit = bool(nmcc_pi_replay_refit)
        self.exploration_rate_start = self._bounded_float(
            "exploration_rate_start", exploration_rate_start, 0.0, 1.0
        )
        self.exploration_rate_end = self._bounded_float(
            "exploration_rate_end", exploration_rate_end, 0.0, 1.0
        )
        if self.exploration_rate_end > self.exploration_rate_start:
            raise ValueError("exploration_rate_end cannot exceed exploration_rate_start")
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
        if self.nmcc_pi_model_fill and not self.nmcc_policy_improvement:
            raise ValueError("nmcc_pi_model_fill requires nmcc_policy_improvement")
        if self.nmcc_total_staged_rollouts > 0 and not (
            self.counterfactual_credit or self.nmcc_policy_improvement
        ):
            raise ValueError(
                "NMCC staged rollouts require exact counterfactual credit or policy improvement"
            )
        self.actor_prior = str(actor_prior)
        if self.actor_prior not in ACTOR_PRIORS:
            raise ValueError(f"actor_prior must be one of {ACTOR_PRIORS}")
        if self.nmcc_policy_improvement and self.nmcc_pi_actor_objective == "value_lcb":
            expected_prior = {
                "risk_reduction": "risk_time_reduction",
                "route_saving": "route_time_saving",
                "active_population": "active_population",
            }[self.nmcc_pi_base_policy]
            if self.actor_prior != expected_prior:
                raise ValueError(
                    "value_lcb requires actor_prior and nmcc_pi_base_policy to "
                    f"describe the same base rule; expected {expected_prior!r}"
                )
        self.actor_prior_scale = self._bounded_float(
            "actor_prior_scale", actor_prior_scale, 0.0, float("inf")
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
        # Persistent episode-level exact-branch dataset.  Validation episodes
        # never enter the fitted intervention-value optimizer; they decide
        # whether the learned correction is safe to deploy.
        self.improvement_gate_history: list[float] = []
        self.improvement_records: list[dict] = []
        self.improvement_replay: list[dict] = []
        self.improvement_replay_seen = 0
        self.improvement_replay_next_id = 0
        self._branch_valuer = None
        self._improvement_rng = np.random.default_rng(
            np.random.SeedSequence([int(policy_seed) & 0xFFFFFFFF, 0x4E504931])
        )
        self.counterfactual_records: list[dict] = []
        self._counterfactual_brancher = None
        self._counterfactual_horizon_override = (
            None if counterfactual_horizon is None else int(counterfactual_horizon)
        )
        self.shelter_action_interval = self._positive_int(
            "shelter_action_interval", shelter_action_interval
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
            raise ValueError("PPO training requires deployment_strategy='rl'")
        self.deployment_strategy = strategy

        self.nx = int(core.cellX)
        self.ny = int(core.cellY)
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

        initial_shelters = int(len(core.shelterDS.shelterList))
        target_shelters = max(initial_shelters, int(target_active_shelters))
        self.maximum_deployments = max(0, target_shelters - initial_shelters)
        self.deployments_made = 0
        self.decision_index = 0
        self.first_decision_time = 1

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
        # The learning transitions below are intentionally attached only to
        # executed regional actions.  Evaluation, however, must also support
        # policies that make no online decisions (for example a static
        # predeployment).  Track the paper objective independently over every
        # elapsed simulator interval so policy-level performance never depends
        # on how many actions a strategy happens to take.
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
        self.action_objective_baseline: Optional[RewardBreakdown] = None
        self.heuristic_policy = ActivePopulationHeuristic()
        self.risk_reduction_policy = RiskTimeReductionHeuristic()
        self.route_saving_policy = RouteTimeSavingHeuristic()
        self.hazard_weighted_policy = HazardWeightedDemandHeuristic()
        self.accessibility_deficit_policy = AccessibilityDeficitHeuristic(
            self._cell_centers()
        )
        self.policy_seed = int(policy_seed)
        self.random_policy = UniformRegionalPolicy(self.policy_seed)

        self.policy = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actor_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.critic_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.representation_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.actor_head_named_parameters: list[tuple[str, nn.Parameter]] = []
        self.critic_head_named_parameters: list[tuple[str, nn.Parameter]] = []
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
                    nmcc_ensemble_size=self.nmcc_ensemble_size,
                    verbose=False,
                    actor_prior=self.actor_prior,
                    actor_prior_scale=self.actor_prior_scale,
                    actor_prior_feature_index=CANDIDATE_FEATURE_NAMES.index(
                        "risk_time_reduction_fraction"
                    ),
                ).to(self.device)
            roles = self.policy.parameter_roles()
            self.representation_named_parameters = list(roles["representation"])
            self.actor_head_named_parameters = list(roles["actor_head"])
            self.critic_head_named_parameters = list(roles["critic_head"])
            # The actor optimizer always owns the representation (the policy
            # must be able to shape its own features).  In shared_phasic mode
            # the critic optimizer owns it too: each pass keeps its own Adam
            # moments for the shared tensors, as in phasic policy gradient.
            self.actor_named_parameters = (
                self.representation_named_parameters + self.actor_head_named_parameters
            )
            self.critic_named_parameters = list(self.critic_head_named_parameters)
            if self.representation_mode == "shared_phasic":
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
            "ppo_diagnostics.csv",
        )
        self.diagnostics_path = str(diagnostics_path or default_diagnostics)
        self.episodes_completed = 0
        self.optimizer_updates = 0
        self.actor_optimizer_updates = 0
        self.critic_optimizer_updates = 0
        self.actor_rollout_updates = 0
        self.rollout_updates_completed = 0
        self.actor_return_baselines: dict[str, dict[str, float]] = {}
        self.counterfactual_return_baselines: dict[str, dict[str, float]] = {}
        self.rollout_traj: list[Transition] = []
        self.rollout_episode_count = 0
        if self.deployment_strategy in {"rl", "rl_precommit"}:
            if os.path.exists(self.checkpoint_path):
                self._load_checkpoint(require_training_state=self.train_mode)
            elif not self.train_mode:
                raise FileNotFoundError(
                    f"RL evaluation requires a trained checkpoint: {self.checkpoint_path}"
                )
            if self.train_mode:
                self.policy.train()
            else:
                self.policy.eval()

        self.pending: Optional[PendingDecision] = None
        self.traj: list[Transition] = []
        self.recurrent_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self.observation_cache: list[ObservationFrame] = []
        self.previous_temporal_observation: Optional[RegionalObservation] = None
        self.last_deployment_time: Optional[int] = None
        self.observation_frames_seen = 0
        self.last_simulation_time: Optional[int] = None
        self.episode_done = False
        self.episode_return = 0.0
        self.episode_safe_reward = 0.0
        self.episode_casualty_penalty = 0.0
        self.episode_evacuation_time_penalty = 0.0
        self.episode_hazard_exposure_penalty = 0.0
        self.action_agreements = 0
        self.action_comparisons = 0
        self.behavior_entropies: list[float] = []
        self.last_training_diagnostics: Dict[str, float] = {}
        self.initial_observation_digest: Optional[str] = None
        self.precommit_initial_observation_digest: Optional[str] = None
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
    ) -> RewardBreakdown:
        """Update the action-count-invariant episode objective.

        Risk mass uses the same right-endpoint convention as decision-interval
        rewards.  Safe completions and casualties are cumulative, so evaluating
        from the initialized outcome to the latest outcome gives a telescoping
        episode objective without double counting.
        """
        simulation_time = int(simulation_time)
        elapsed = simulation_time - int(self.objective_last_time)
        if elapsed <= 0:
            raise ValueError("Policy-objective time must increase strictly")
        risk_mass = float(observation.outcome.risk_mass)
        if not np.isfinite(risk_mass) or risk_mass < 0.0:
            raise ValueError("Observed risk mass must be finite and non-negative")
        active_mass = float(observation.outcome.active_population)
        exposure_mass = float(observation.outcome.hazard_exposure_mass)
        self.objective_active_person_time += active_mass * elapsed
        self.objective_hazard_exposure_person_time += exposure_mass * elapsed
        self.objective_last_time = simulation_time
        self.objective_latest_outcome = observation.outcome
        return self.reward_model.evaluate(
            before=self.objective_initial_outcome,
            after=self.objective_latest_outcome,
            active_person_time=self.objective_active_person_time,
            hazard_exposure_person_time=self.objective_hazard_exposure_person_time,
            initial_population=self.initial_population,
            horizon=self.horizon,
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

    @property
    def nmcc_total_staged_rollouts(self) -> int:
        return int(
            self.nmcc_natural_pretrain_rollouts
            + self.nmcc_causal_pretrain_rollouts
            + self.nmcc_controller_warmup_rollouts
        )

    def _current_nmcc_training_phase(self) -> str:
        update = int(self.rollout_updates_completed)
        natural_end = int(self.nmcc_natural_pretrain_rollouts)
        causal_end = natural_end + int(self.nmcc_causal_pretrain_rollouts)
        controller_end = causal_end + int(self.nmcc_controller_warmup_rollouts)
        if update < natural_end:
            return "natural_pretrain"
        if update < causal_end:
            return "causal_pretrain"
        if update < controller_end:
            return "controller_warmup"
        return "joint_optimization"

    def _actor_training_enabled(self) -> bool:
        return self._current_nmcc_training_phase() in {
            "controller_warmup",
            "joint_optimization",
        }

    def _effective_counterfactual_weight(self) -> float:
        if self._current_nmcc_training_phase() == "joint_optimization":
            return float(self.nmcc_joint_counterfactual_weight)
        return float(self.counterfactual_weight)

    def _linear_schedule(self, start: float, end: float, duration: int) -> float:
        fraction = min(
            1.0,
            max(
                0.0,
                self.actor_rollout_updates / float(max(1, int(duration))),
            ),
        )
        return float(start + fraction * (end - start))

    def _current_entropy_coef(self) -> float:
        return self._linear_schedule(
            self.entropy_coef,
            self.entropy_coef_end,
            self.exploration_decay_updates,
        )

    def _current_action_temperature(self) -> float:
        return self._linear_schedule(
            self.action_temperature_start,
            self.action_temperature_end,
            self.exploration_decay_updates,
        )

    def _current_exploration_rate(self) -> float:
        """Epsilon for score-policy exploration.

        ``actor_rollout_updates`` advances only when the controller is allowed
        to train, so system-identification phases retain broad exploration
        without prematurely consuming the schedule.  The fitted-value
        controller intentionally has no PPO actor updates, so its schedule is
        driven by completed replay fits after natural/causal pretraining.
        """
        if self.nmcc_policy_improvement and self.nmcc_pi_actor_objective == "value_lcb":
            system_updates = (
                self.nmcc_natural_pretrain_rollouts
                + self.nmcc_causal_pretrain_rollouts
            )
            update_count = max(
                0, int(self.rollout_updates_completed) - int(system_updates)
            )
            fraction = min(
                1.0,
                update_count / float(max(1, self.exploration_decay_updates)),
            )
            return float(
                self.exploration_rate_start
                + fraction * (
                    self.exploration_rate_end - self.exploration_rate_start
                )
            )
        return self._linear_schedule(
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
        update = max(0, int(update_index))
        if self.lr_warmup_updates > 0 and update < self.lr_warmup_updates:
            return float(base) * float(update + 1) / float(self.lr_warmup_updates)
        age = max(0, update - self.lr_warmup_updates)
        progress = min(1.0, age / float(max(1, self.lr_decay_updates)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = float(minimum_fraction) + (1.0 - float(minimum_fraction)) * cosine
        return float(base) * multiplier

    def _apply_learning_rate_schedule(self) -> tuple[float, float]:
        controller_start = (
            self.nmcc_natural_pretrain_rollouts
            + self.nmcc_causal_pretrain_rollouts
        )
        actor_age = max(0, int(self.rollout_updates_completed) - int(controller_start))
        actor_rate = self._scheduled_learning_rate(
            self.actor_lr, self.actor_lr_min_fraction, actor_age
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

    def _current_nmcc_guidance_weight(self) -> float:
        if not self.counterfactual_credit or not self._actor_training_enabled():
            return 0.0
        age = self.actor_rollout_updates - self.nmcc_guidance_warmup_updates
        if age <= 0:
            return 0.0
        return float(
            self.nmcc_guidance_max
            * min(1.0, age / float(self.nmcc_guidance_ramp_updates))
        )

    def _current_nmcc_teacher_coef(self) -> float:
        if not self.counterfactual_credit or not self._actor_training_enabled():
            return 0.0
        age = self.actor_rollout_updates - self.nmcc_guidance_warmup_updates
        if age <= 0:
            return 0.0
        fraction = min(
            1.0,
            age / float(max(1, self.nmcc_teacher_decay_updates)),
        )
        return float(self.nmcc_teacher_coef * (1.0 - fraction))

    @staticmethod
    def _masked_standardize_scores(
        scores: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        if scores.shape != action_mask.shape:
            raise ValueError("NMCC score and action-mask shapes disagree")
        mask = action_mask.to(dtype=scores.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (scores * mask).sum(dim=-1, keepdim=True) / count
        centered = (scores - mean) * mask
        variance = centered.square().sum(dim=-1, keepdim=True) / count
        return centered / torch.sqrt(variance + 1e-6)

    def _apply_nmcc_guidance(
        self,
        logits: torch.Tensor,
        robust_scores: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        weight: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        teacher_logits = self._masked_standardize_scores(
            # Actor/PPO and imitation gradients must not train the learned
            # simulator. The natural and residual heads are fitted only to
            # physical paired-branch targets below; this detach prevents a
            # self-confirming planner/actor feedback loop.
            robust_scores.detach(),
            action_mask,
        )
        guidance = (
            self._current_nmcc_guidance_weight()
            if weight is None
            else float(weight)
        )
        return logits + guidance * teacher_logits, teacher_logits

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
            "action_space": "masked_candidate_score_argmax_shared_deterministic_site_rule",
            "architecture": "resolution_flexible_relational_route_gnn_lstm_nmcc_v10",
            "graph_contract": {
                "region_count": "runtime_X_times_Y",
                "spatial_edges": "grid_adjacency_union_road_crossings",
                "route_edges": "bidirectional_current_assignment_flow",
                "route_aggregation": "population_share_weighted_sum",
                "message_layers": DEFAULT_MESSAGE_LAYERS,
                "pooling": "learned_attention_plus_maximum",
                "temporal_encoder": "episode_lstm_over_every_observed_simulator_boundary",
                "temporal_hidden_dimension": int(self.temporal_dim),
                "momentum_features": MOMENTUM_FEATURE_NAMES,
                "behavior_policy_dropout": False,
                "nmcc_natural_model_action_independent": True,
                "nmcc_natural_population_conservation": (
                    "hard_safe_casualty_active_simplex"
                ),
                "nmcc_natural_risk_bounds": (
                    "half_active_to_active_after_two_population_normalization"
                ),
                "nmcc_residual_localization": "candidate_region_and_site_features",
                "shelter_capacity_contract": {
                    "assignment": (
                        "conserved_en_route_reservations_priority_by_risk_time"
                    ),
                    "mode": (
                        "equal_token"
                        if int(getattr(self.core, "shelterCapacityToken", 0)) > 0
                        else "site_specific_legacy"
                    ),
                    "capacity_per_shelter": int(
                        getattr(self.core, "shelterCapacityToken", 0)
                    ),
                },
                "checkpoint_transfer_across_resolutions": False,
            },
            "heuristic_prior_scale": float(self.actor_prior_scale),
            "actor_prior": {
                "kind": str(self.actor_prior),
                "scale": float(self.actor_prior_scale),
                "feature": {
                    "risk_time_reduction": "risk_time_reduction_fraction",
                    "route_time_saving": "risk_time_reduction_fraction",
                    "active_population": "relative_active_population",
                    "none": None,
                }[self.actor_prior],
                "intervention_value_heads": int(self.nmcc_ensemble_size),
            },
            "learner_contract": {
                "representation_mode": str(self.representation_mode),
                "representation_clone_coef": float(self.representation_clone_coef),
                "representation_kl_cap": float(self.representation_kl_cap),
                "pi_trust_region": "kl_new_to_behavior_line_search_no_lr_ratchet",
                "learning_rate_schedule": str(self.learning_rate_schedule),
                "learning_rate_warmup_updates": int(self.lr_warmup_updates),
                "learning_rate_decay_updates": int(self.lr_decay_updates),
                "actor_lr_min_fraction": float(self.actor_lr_min_fraction),
                "critic_lr_min_fraction": float(self.critic_lr_min_fraction),
                "exploration": {
                    "kind": "epsilon_greedy_over_masked_candidate_scores",
                    "start": float(self.exploration_rate_start),
                    "end": float(self.exploration_rate_end),
                    "decay_updates": int(self.exploration_decay_updates),
                },
            },
            "nmcc_policy_improvement": {
                "enabled": bool(self.nmcc_policy_improvement),
                "variant": "persistent_full_horizon_fitted_intervention_value_lcb",
                "actor_objective": str(self.nmcc_pi_actor_objective),
                "value_target": "physical_advantage_exact_pairwise",
                "uncertainty_reference": "candidate_minus_fixed_base_action",
                "heldout_gate_score": "base_prior_level_plus_conservative_paired_advantage",
                "control_refit": "frozen_system_gnn_lstm_plus_wide_deep_intervention_heads",
                "base_policy": str(self.nmcc_pi_base_policy),
                "epsilon": float(self.nmcc_pi_epsilon),
                "eta_min": float(self.nmcc_pi_eta_min),
                "kl_cap": float(self.nmcc_pi_kl_cap),
                "tapes": int(self.nmcc_pi_tapes),
                "exhaustive_decisions": int(self.nmcc_pi_exhaustive_decisions),
                "max_branches": int(self.nmcc_pi_max_branches),
                "value_scale": float(self.nmcc_pi_value_scale),
                "model_fill": bool(self.nmcc_pi_model_fill),
                "gate_spearman": float(self.nmcc_pi_gate_spearman),
                "gate_updates": int(self.nmcc_pi_gate_updates),
                "actor_epochs": int(self.nmcc_pi_actor_epochs),
                "fit_tolerance": float(self.nmcc_pi_fit_tolerance),
                "branch_horizon": int(self.nmcc_pi_branch_horizon),
                "full_horizon_decisions": int(self.nmcc_pi_full_horizon_decisions),
                "ranking_temperature": float(self.nmcc_pi_ranking_temperature),
                "rank_margin": float(self.nmcc_pi_rank_margin),
                "rank_margin_coefficient": float(self.nmcc_pi_rank_margin_coef),
                "replay_max_episodes": int(self.nmcc_pi_replay_max_episodes),
                "replay_epochs": int(self.nmcc_pi_replay_epochs),
                "validation_fraction": float(self.nmcc_pi_validation_fraction),
                "early_stopping_patience": int(self.nmcc_pi_early_stopping_patience),
                "minimum_validation_states": int(self.nmcc_pi_min_validation_states),
                "validation_gain_z": float(self.nmcc_pi_validation_gain_z),
                "replay_refit": bool(self.nmcc_pi_replay_refit),
                "physical_outcome_supervision": NMCC_OUTCOME_NAMES,
            },
            "candidate_constraints": {
                "maximum_forecast_danger": float(
                    self.observation_builder.maximum_shelter_forecast_danger
                ),
                "require_operational_benefit": bool(
                    self.observation_builder.require_candidate_operational_benefit
                ),
                "minimum_reroutable_fraction": float(
                    self.observation_builder.minimum_candidate_reroutable_fraction
                ),
                "minimum_risk_time_reduction": float(
                    self.observation_builder.minimum_candidate_risk_time_reduction
                ),
                "minimum_hazard_safety_margin": float(
                    self.observation_builder.minimum_candidate_hazard_safety_margin
                ),
            },
            "residual_logit_bound": float(self.policy.residual_logit_bound),
            "nmcc_contract": {
                "enabled": bool(self.counterfactual_credit),
                "variant": "hybrid_exact_pair_plus_factored_world_model",
                "outcomes": NMCC_OUTCOME_NAMES,
                "ensemble_size": int(self.nmcc_ensemble_size),
                "counterfactual_horizon_timesteps": int(
                    self.counterfactual_horizon
                ),
                "counterfactual_weight": float(self.counterfactual_weight),
                "joint_counterfactual_weight": float(
                    self.nmcc_joint_counterfactual_weight
                ),
                "intervention_cost": float(self.counterfactual_intervention_cost),
                "staged_rollouts": {
                    "natural_pretrain": int(
                        self.nmcc_natural_pretrain_rollouts
                    ),
                    "causal_pretrain": int(self.nmcc_causal_pretrain_rollouts),
                    "controller_warmup": int(
                        self.nmcc_controller_warmup_rollouts
                    ),
                    "phase_order": NMCC_TRAINING_PHASES,
                },
                "guidance_max": float(self.nmcc_guidance_max),
                "guidance_warmup_updates": int(
                    self.nmcc_guidance_warmup_updates
                ),
                "guidance_ramp_updates": int(self.nmcc_guidance_ramp_updates),
                "uncertainty_penalty": float(self.nmcc_uncertainty_penalty),
                "wait_identity": "exact_zero_intervention_residual",
                "dueling_decomposition": "Q_equals_V_wait_plus_D",
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
                "wind_speed_m_per_minute": float(
                    getattr(self.core, "hazardWindSpeedMPerMinute", 0.0)
                ),
                "wind_direction_degrees": float(
                    getattr(self.core, "hazardWindDirectionDegrees", 0.0)
                ),
                "wind_influence": float(
                    getattr(self.core, "hazardWindInfluence", 1.0)
                ),
                "maximum_shelter_forecast_danger": float(
                    getattr(self.core, "maximumShelterForecastDanger", 0.6)
                ),
            },
            "decision_schedule": {
                "first_decision_timestep": int(self.first_decision_time),
                "shelter_action_interval_timesteps": int(
                    self.shelter_action_interval
                ),
                "transition_boundary": "next_executed_action_or_true_environment_terminal",
                "final_action_accounting": "continues_through_true_environment_terminal",
                "counterfactual_accounting": "matched_fixed_horizon_frozen_inside_full_transition",
            },
            "rollout_episodes": self.rollout_episodes,
            "reward_contract": {
                "safe_completion_weight": DEFAULT_SAFE_COMPLETION_WEIGHT,
                "casualty_weight": self.reward_model.casualty_weight,
                "evacuation_time_weight": (
                    self.reward_model.evacuation_time_weight
                ),
                "hazard_exposure_weight": (
                    self.reward_model.hazard_exposure_weight
                ),
                "site_specific_shaping": False,
                "factorized_critic_components": REWARD_COMPONENT_NAMES,
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
            "ppo_hyperparameters": {
                "gamma": self.gamma,
                "actor_credit_target": "complete_episode_smdp_monte_carlo_return_to_go",
                "actor_baseline": "lagged_regime_position_ema",
                "actor_baseline_decay": self.actor_baseline_decay,
                "advantage_scale_floor": self.advantage_scale_floor,
                "critic_target": "independent_one_step_smdp_td0",
                "clip_epsilon": self.clip_eps,
                "actor_learning_rate": self.actor_lr,
                "critic_learning_rate": self.critic_lr,
                "actor_epochs": self.actor_epochs,
                "critic_epochs": self.critic_epochs,
                "minibatch_size": self.minibatch_size,
                "entropy_coefficient_start": self.entropy_coef,
                "entropy_coefficient_end": self.entropy_coef_end,
                "exploration_decay_updates": self.exploration_decay_updates,
                "action_temperature_start": self.action_temperature_start,
                "action_temperature_end": self.action_temperature_end,
                "epsilon_greedy_start": self.exploration_rate_start,
                "epsilon_greedy_end": self.exploration_rate_end,
                "learning_rate_schedule": self.learning_rate_schedule,
                "learning_rate_warmup_updates": self.lr_warmup_updates,
                "learning_rate_decay_updates": self.lr_decay_updates,
                "actor_learning_rate_minimum_fraction": self.actor_lr_min_fraction,
                "critic_learning_rate_minimum_fraction": self.critic_lr_min_fraction,
                "target_kl": self.target_kl,
                "residual_penalty_coefficient": self.residual_penalty_coef,
                "entropy_normalization": "log_feasible_action_count",
                "kl_control": "transactional_full_rollout_reject_and_actor_lr_backoff",
                "minimum_learning_rate_fraction": DEFAULT_MINIMUM_LR_FRACTION,
                "kl_learning_rate_reduction": DEFAULT_KL_LR_REDUCTION,
                "kl_learning_rate_growth": DEFAULT_KL_LR_GROWTH,
                "critic_loss": "raw_smooth_l1_td0",
                "critic_factorization": "signed_component_heads",
                "optimizer_ownership": "disjoint_actor_and_critic_world_parameters",
                "sequence_sampling": "whole_episode_recurrent_minibatches",
                "nmcc_natural_loss_coefficient": self.nmcc_natural_loss_coef,
                "nmcc_causal_loss_coefficient": self.nmcc_causal_loss_coef,
                "nmcc_dueling_loss_coefficient": self.nmcc_dueling_loss_coef,
                "nmcc_teacher_coefficient": self.nmcc_teacher_coef,
                "nmcc_teacher_decay_updates": self.nmcc_teacher_decay_updates,
            },
        }

    @staticmethod
    def _inference_signature(model_signature: Mapping) -> dict:
        """Return the checkpoint fields that determine policy inference.

        PPO rollout and optimizer settings govern how a policy is trained or
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
            "nmcc_policy_improvement",
            "candidate_constraints",
            "residual_logit_bound",
            "nmcc_contract",
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
            # PPO continuation requires optimizer and RNG objects in addition to
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
                    "actor_optimizer_updates",
                    "critic_optimizer_updates",
                    "actor_rollout_updates",
                    "rollout_updates_completed",
                    "actor_return_baselines",
                    "counterfactual_return_baselines",
                    "improvement_replay",
                    "improvement_replay_seen",
                    "improvement_replay_next_id",
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
            self.actor_return_baselines = copy.deepcopy(
                payload.get("actor_return_baselines", {})
            )
            self.counterfactual_return_baselines = copy.deepcopy(
                payload.get("counterfactual_return_baselines", {})
            )
            self.improvement_gate_history = [
                float(value) for value in payload.get("improvement_gate_history", [])
            ]
            if require_training_state:
                self.improvement_replay = [
                    {
                        "episode_id": int(item["episode_id"]),
                        "transitions": tuple(
                            self._deserialize_transition(
                                value, target_device=torch.device("cpu")
                            )
                            for value in item["transitions"]
                        ),
                    }
                    for item in payload["improvement_replay"]
                ]
                self.improvement_replay_seen = int(payload["improvement_replay_seen"])
                self.improvement_replay_next_id = int(payload["improvement_replay_next_id"])
                self._validate_improvement_replay()
            if require_training_state and "improvement_generator_state" in payload:
                self._improvement_rng.bit_generator.state = payload[
                    "improvement_generator_state"
                ]
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
            "actor_return_baselines": copy.deepcopy(self.actor_return_baselines),
            "counterfactual_return_baselines": copy.deepcopy(
                self.counterfactual_return_baselines
            ),
            "policy_seed": self.policy_seed,
            "action_generator_state": self.action_generator.get_state(),
            "optimization_generator_state": self.optimization_generator.get_state(),
            "rollout_transitions": [
                self._serialize_transition(transition)
                for transition in self.rollout_traj
            ],
            "rollout_episode_count": self.rollout_episode_count,
            "improvement_gate_history": list(self.improvement_gate_history),
            "improvement_generator_state": self._improvement_rng.bit_generator.state,
            "improvement_replay": [
                {
                    "episode_id": int(item["episode_id"]),
                    "transitions": [
                        self._serialize_transition(value)
                        for value in item["transitions"]
                    ],
                }
                for item in self.improvement_replay
            ],
            "improvement_replay_seen": int(self.improvement_replay_seen),
            "improvement_replay_next_id": int(self.improvement_replay_next_id),
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
                "natural_outcome_target",
                "causal_outcome_target",
                "improvement_target",
                "improvement_behavior",
                "improvement_exact_mask",
                "improvement_advantage",
                "improvement_bootstrap",
                "improvement_natural_outcome",
                "improvement_outcome_effect",
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
            for name in ("log_probability", "value", "reward"):
                tensor = getattr(transition, name)
                if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"checkpoint rollout {name} is invalid")
            for name in ("value_components", "reward_components"):
                tensor = getattr(transition, name)
                if tensor.shape != (len(REWARD_COMPONENT_NAMES),) or not bool(
                    torch.isfinite(tensor).all()
                ):
                    raise ValueError(f"checkpoint rollout {name} is invalid")
            optional_shapes = {
                "counterfactual_advantage": (1,),
                "counterfactual_components": (len(REWARD_COMPONENT_NAMES),),
                "natural_outcome_target": (len(NMCC_OUTCOME_NAMES),),
                "causal_outcome_target": (len(NMCC_OUTCOME_NAMES),),
            }
            for name, shape in optional_shapes.items():
                tensor = getattr(transition, name)
                if tensor is not None and (
                    tuple(tensor.shape) != shape
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ValueError(f"checkpoint rollout {name} is invalid")
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
        """Compute administrator-interpretable rates from past and present data.

        Rates are scaled to one decision interval, so observations remain
        comparable if the simulator exposes substeps.  No future outcome or
        post-decision information is used.
        """
        current_time = int(observation.simulation_time)
        previous = self.previous_temporal_observation
        if previous is None or current_time <= int(previous.simulation_time):
            deltas = np.zeros(self.d_momentum - 1, dtype=np.float32)
        else:
            elapsed = float(current_time - int(previous.simulation_time))
            scale = float(self.shelter_action_interval) / elapsed
            population = float(self.initial_population)

            def burdens(item: RegionalObservation) -> tuple[float, float, float]:
                active = np.asarray(item.active_by_cell, dtype=np.float64)
                route_scale = max(
                    float(item.time_step_minutes),
                    float(item.horizon) * float(item.time_step_minutes),
                )
                route = float(
                    np.sum(active * np.clip(item.mean_route_time_by_cell / route_scale, 0.0, 1.0))
                    / population
                )
                long_route = float(
                    np.sum(active * item.long_route_share_by_cell) / population
                )
                forecast = float(
                    np.sum(active * item.forecast_danger_by_cell) / population
                )
                return route, long_route, forecast

            previous_route, previous_long, previous_forecast = burdens(previous)
            current_route, current_long, current_forecast = burdens(observation)
            deltas = np.asarray(
                (
                    (
                        observation.outcome.safe_completed
                        - previous.outcome.safe_completed
                    )
                    / population,
                    (
                        observation.outcome.casualties
                        - previous.outcome.casualties
                    )
                    / population,
                    (
                        previous.outcome.active_population
                        - observation.outcome.active_population
                    )
                    / population,
                    (
                        previous.outcome.hazard_exposure_mass
                        - observation.outcome.hazard_exposure_mass
                    )
                    / population,
                    previous_route - current_route,
                    previous_long - current_long,
                    previous_forecast - current_forecast,
                    float(previous.network_load_share)
                    - float(observation.network_load_share),
                ),
                dtype=np.float32,
            ) * scale
            deltas = np.clip(deltas, -1.0, 1.0)

        if self.last_deployment_time is None:
            since_deployment = 1.0
        else:
            since_deployment = np.clip(
                (current_time - int(self.last_deployment_time))
                / float(max(1, self.shelter_action_interval)),
                0.0,
                1.0,
            )
        values = np.concatenate(
            (deltas, np.asarray([since_deployment], dtype=np.float32))
        )
        if values.shape != (self.d_momentum,) or not np.isfinite(values).all():
            raise RuntimeError("Temporal momentum feature construction failed")
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _advance_recurrent_observation(
        self,
        observation: RegionalObservation,
        *,
        cache_for_training: bool,
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
        )
        graph = self._graph_from_frame(frame)
        with torch.no_grad():
            nmcc_output = self.policy.forward_nmcc_recurrent(
                graph,
                self.recurrent_state,
                momentum_features,
                uncertainty_penalty=self.nmcc_uncertainty_penalty,
            )
            logits, teacher_logits = self._apply_nmcc_guidance(
                nmcc_output.logits,
                nmcc_output.robust_causal_score,
                action_mask.unsqueeze(0),
            )
        recurrent_state = nmcc_output.recurrent_state
        self.recurrent_state = tuple(state.detach() for state in recurrent_state)
        self.previous_temporal_observation = observation
        self.observation_frames_seen += 1
        if cache_for_training:
            self.observation_cache.append(frame)
        return RecurrentPolicyStep(
            frame=frame,
            logits=logits,
            prior_logits=nmcc_output.prior_logits,
            value=nmcc_output.value,
            value_components=nmcc_output.value_components,
            natural_outcomes=nmcc_output.natural_outcomes,
            causal_outcome_samples=nmcc_output.causal_outcome_samples,
            causal_component_mean=nmcc_output.causal_component_mean,
            causal_component_std=nmcc_output.causal_component_std,
            robust_causal_score=nmcc_output.robust_causal_score,
            teacher_logits=teacher_logits,
            improvement_value_samples=nmcc_output.improvement_value_samples,
        )

    def _controller_logits(
        self,
        recurrent_step: "RecurrentPolicyStep",
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the score used by the deployed controller.

        In v28 the actor is an explicit conservative planner: the fixed,
        interpretable base-policy prior is corrected only by a candidate's
        predicted full-horizon intervention value.  The correction is disabled
        until episode-held-out paired outcomes show a positive lower-confidence
        gain over that same base policy.
        """
        if not (
            self.nmcc_policy_improvement
            and self.nmcc_pi_actor_objective == "value_lcb"
        ):
            return recurrent_step.logits
        logits = recurrent_step.prior_logits
        if (
            self.improvement_gate_passed
            and recurrent_step.improvement_value_samples is not None
        ):
            samples = recurrent_step.improvement_value_samples
            masked_prior = self._safe_masked_logits(
                logits,
                action_mask.reshape(1, -1),
            )
            base_action = masked_prior.argmax(dim=-1)
            conservative = self._base_relative_improvement_lcb(
                samples,
                base_action,
            )
            # Preserve the base policy's common score level, but replace its
            # candidate ordering with the learned physical advantage.  Adding
            # full Q-advantage to the varying base prior would count the
            # heuristic twice and can make a provably better candidate
            # unreachable even under a perfect value fit.
            base_level = logits.gather(1, base_action[:, None])
            logits = base_level + (
                float(self.nmcc_pi_value_scale)
                * conservative
                / float(self.nmcc_pi_ranking_temperature)
            )
        return self._safe_masked_logits(logits, action_mask.reshape(1, -1))

    def _base_relative_improvement_lcb(
        self,
        samples: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        """Conservative candidate values relative to the fixed base action.

        Exact NMCC labels identify only within-state value differences.  Each
        bootstrap head therefore has an arbitrary state-level offset.  Taking
        an ensemble standard deviation on raw values mistakes those
        unidentified offsets for action uncertainty and can reverse the LCB
        ranking.  Subtracting every member's prediction for the route-saving
        base action produces identified paired effects; the base correction is
        exactly zero and uncertainty measures disagreement about interventions.
        """
        if samples.ndim != 3:
            raise ValueError("intervention samples must have shape (batch, members, actions)")
        base = base_action.to(device=samples.device, dtype=torch.long).reshape(-1)
        if base.numel() != samples.size(0):
            raise ValueError("one base action is required for each intervention state")
        if bool(((base < 0) | (base >= samples.size(-1))).any()):
            raise ValueError("base action lies outside the intervention action space")
        reference = samples.gather(
            2,
            base[:, None, None].expand(-1, samples.size(1), 1),
        )
        paired_effect = samples - reference
        return paired_effect.mean(dim=1) - float(
            self.nmcc_pi_model_uncertainty_penalty
        ) * paired_effect.std(dim=1, unbiased=False)

    def _select_rl(
        self,
        observation: RegionalObservation,
        *,
        deterministic: bool,
        recurrent_step: Optional[RecurrentPolicyStep] = None,
    ) -> PolicyDecision:
        if recurrent_step is None:
            recurrent_step = self._advance_recurrent_observation(
                observation,
                cache_for_training=self.train_mode,
            )
        action_mask = recurrent_step.frame.action_mask
        behavior_logits = self._controller_logits(
            recurrent_step, action_mask
        ) / self._current_action_temperature()
        masked_logits = self._safe_masked_logits(
            behavior_logits,
            action_mask.unsqueeze(0),
        )
        score_probabilities = torch.softmax(masked_logits, dim=-1)
        feasible = action_mask.reshape(1, -1).to(dtype=score_probabilities.dtype)
        uniform = feasible / feasible.sum(dim=-1, keepdim=True).clamp_min(1.0)
        if self.nmcc_policy_improvement and self.nmcc_pi_actor_objective in {
            "score_ranking", "value_lcb"
        }:
            greedy = torch.zeros_like(score_probabilities)
            greedy.scatter_(1, masked_logits.argmax(dim=-1, keepdim=True), 1.0)
            epsilon = 0.0 if deterministic else self._current_exploration_rate()
            behavior_probabilities = (1.0 - epsilon) * greedy + epsilon * uniform
        else:
            behavior_probabilities = score_probabilities
        distribution = torch.distributions.Categorical(probs=behavior_probabilities)
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
        if self.deployment_strategy == "heuristic":
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
        else:
            raise RuntimeError(f"Strategy {self.deployment_strategy!r} cannot make dynamic decisions")
        return decision

    @staticmethod
    def _observation_digest(observation: RegionalObservation) -> str:
        digest = hashlib.sha256()
        for array in (
            observation.active_by_cell,
            observation.mean_speed_by_cell,
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
            observation = self.observation_builder.build(
                decision_index=self.decision_index,
                simulation_time=0,
                remaining_deployments=self.remaining_deployments,
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
            execution_started = time.perf_counter()
            receipt = self.executor.execute(observation, decision)
            execution_ms = (time.perf_counter() - execution_started) * 1000.0
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
            self.decision_index += 1
        return tuple(receipts)

    def _decision_due(self, simulation_time: int) -> bool:
        # A temporarily empty hard-safety mask must defer a deployment, not
        # permanently erase one of the policy's capacity tokens.  Once a
        # shelter is installed, its successor is spaced from the actual action
        # time so every controller receives the same number of opportunities.
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

    def _accumulate_counterfactual_factual(
        self,
        *,
        after: RegionalObservation,
        elapsed: int,
        terminal: bool,
    ) -> None:
        """Freeze the factual twin at exactly the WAIT-branch horizon.

        Global SMDP accounting may continue until the next executed action or
        the physical terminal boundary. NMCC must not compare that longer
        interval with a one-window WAIT branch, so its factual rewards and
        physical outcomes are accumulated separately and frozen at ``L``.
        """
        pending = self.pending
        if (
            pending is None
            or pending.wait_steps <= 0
            or pending.factual_component_return is not None
        ):
            return
        remaining = int(pending.wait_steps) - int(
            pending.counterfactual_elapsed_timesteps
        )
        if remaining <= 0:
            return
        used = min(int(elapsed), remaining)
        if used <= 0:
            return
        if int(elapsed) > remaining:
            raise RuntimeError(
                "NMCC factual horizon was crossed without an observation boundary"
            )
        previous = (
            pending.counterfactual_previous_outcome
            if pending.counterfactual_previous_outcome is not None
            else pending.observation.outcome
        )
        active_time = float(after.outcome.active_population) * used
        exposure_time = float(after.outcome.hazard_exposure_mass) * used
        step_breakdown = self.reward_model.evaluate(
            before=previous,
            after=after.outcome,
            active_person_time=active_time,
            hazard_exposure_person_time=exposure_time,
            initial_population=self.initial_population,
            horizon=self.horizon,
        )
        if pending.counterfactual_discounted_components is None:
            pending.counterfactual_discounted_components = np.zeros(
                len(REWARD_COMPONENT_NAMES),
                dtype=np.float64,
            )
        discount = float(self.gamma) ** int(
            pending.counterfactual_elapsed_timesteps
        )
        pending.counterfactual_discounted_components += (
            discount * step_breakdown.component_vector().astype(np.float64)
        )
        pending.counterfactual_active_person_time += active_time
        pending.counterfactual_exposure_person_time += exposure_time
        pending.counterfactual_elapsed_timesteps += used
        pending.counterfactual_previous_outcome = after.outcome

        reached_horizon = (
            pending.counterfactual_elapsed_timesteps >= pending.wait_steps
        )
        if terminal or reached_horizon:
            pending.factual_component_return = (
                pending.counterfactual_discounted_components.copy()
            )
            pending.factual_outcome_target = self._normalized_outcome_target(
                before=pending.observation.outcome,
                after=after.outcome,
                active_person_time=pending.counterfactual_active_person_time,
                hazard_exposure_person_time=(
                    pending.counterfactual_exposure_person_time
                ),
            )
            bootstrap = np.zeros(
                len(REWARD_COMPONENT_NAMES),
                dtype=np.float64,
            )
            if reached_horizon and not terminal:
                _, bootstrap = self._readonly_observation_values(
                    after,
                    recurrent_state=pending.recurrent_state_at_decision,
                )
            pending.factual_bootstrap_components = bootstrap

    def _record_completed_interval(
        self,
        *,
        after: RegionalObservation,
        terminal: bool,
    ) -> RewardBreakdown:
        if self.pending is None:
            raise RuntimeError("No pending decision exists to complete")
        pending = self.pending
        breakdown = self.reward_model.evaluate(
            before=pending.observation.outcome,
            after=after.outcome,
            active_person_time=pending.active_person_time,
            hazard_exposure_person_time=pending.hazard_exposure_person_time,
            initial_population=self.initial_population,
            horizon=self.horizon,
        )
        self.episode_return += breakdown.total
        self.episode_safe_reward += breakdown.safe_completion_reward
        self.episode_casualty_penalty += breakdown.casualty_penalty
        self.episode_evacuation_time_penalty += breakdown.evacuation_time_penalty
        self.episode_hazard_exposure_penalty += breakdown.hazard_exposure_penalty

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
            if self.counterfactual_credit and np.isfinite(pending.wait_return):
                if (
                    pending.factual_component_return is None
                    or pending.factual_bootstrap_components is None
                    or pending.factual_outcome_target is None
                    or pending.wait_outcome_target is None
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
                    elapsed=max(1, int(pending.wait_steps)),
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
                        pending.wait_outcome_target,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    causal_outcome_tensor = torch.as_tensor(
                        pending.factual_outcome_target
                        - pending.wait_outcome_target,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.counterfactual_records.append(
                        {
                            "decision_index": int(self.decision_index),
                            "cell": int(pending.receipt.executed_cell),
                            "factual_return": float(
                                pending.factual_component_return.sum()
                            ),
                            "wait_return": float(pending.wait_return),
                            "counterfactual_advantage": float(counterfactual_value),
                            "counterfactual_horizon": int(pending.wait_steps),
                        }
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
                        [breakdown.total],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    reward_components=torch.as_tensor(
                        breakdown.component_vector(),
                        dtype=torch.float32,
                        device=self.device,
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
                    natural_outcome_target=natural_outcome_tensor,
                    causal_outcome_target=causal_outcome_tensor,
                    **self._improvement_transition_fields(pending),
                )
            )
        self.pending = None
        return breakdown

    def _improvement_transition_fields(self, pending: "PendingDecision") -> dict:
        """Tensor fields for a transition's NMCC-PI record, or all ``None``."""
        record = pending.policy_improvement
        names = (
            "improvement_target",
            "improvement_behavior",
            "improvement_exact_mask",
            "improvement_advantage",
            "improvement_bootstrap",
            "improvement_natural_outcome",
            "improvement_outcome_effect",
            "improvement_base_action",
        )
        if record is None:
            return {name: None for name in names}
        sources = (
            "target",
            "behavior",
            "exact_mask",
            "exact_advantage",
            "bootstrap",
            "natural_outcome",
            "outcome_effect",
            "base_action",
        )
        return {
            name: torch.as_tensor(
                np.asarray(record[source], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
            for name, source in zip(names, sources)
        }

    def step(self, *, simulation_time: int, is_terminal: bool = False) -> Dict[str, object]:
        """Observe one simulator boundary and possibly complete/select a decision."""
        if self.episode_done:
            result = self._empty_step_result()
            result["episode_return"] = float(self.episode_return)
            result["objective_episode_return"] = float(
                self.reward_model.evaluate(
                    before=self.objective_initial_outcome,
                    after=self.objective_latest_outcome,
                    active_person_time=self.objective_active_person_time,
                    hazard_exposure_person_time=(
                        self.objective_hazard_exposure_person_time
                    ),
                    initial_population=self.initial_population,
                    horizon=self.horizon,
                ).total
            )
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
        objective_breakdown = self._accumulate_policy_objective(
            observation,
            simulation_time,
        )
        if self.initial_observation_digest is None:
            self.initial_observation_digest = self._observation_digest(observation)
        pending_elapsed = 0
        if self.pending is not None and self.last_simulation_time is not None:
            elapsed = simulation_time - self.last_simulation_time
            pending_elapsed = elapsed
            active_mass = float(observation.outcome.active_population)
            exposure_mass = float(observation.outcome.hazard_exposure_mass)
            self.pending.active_person_time += active_mass * elapsed
            self.pending.hazard_exposure_person_time += exposure_mass * elapsed
            self.pending.elapsed_timesteps += elapsed

        terminal = bool(is_terminal) or observation.outcome.active_population <= 0
        if self.pending is not None and pending_elapsed > 0:
            self._accumulate_counterfactual_factual(
                after=observation,
                elapsed=pending_elapsed,
                terminal=terminal,
            )
        recurrent_step = None
        if (
            not terminal
            and self.deployment_strategy in {"rl", "rl_precommit"}
        ):
            recurrent_step = self._advance_recurrent_observation(
                observation,
                cache_for_training=self.train_mode,
            )
        breakdown = None
        completed_action = -1
        dynamic_strategy = self.deployment_strategy in {
            "rl",
            "random",
            "heuristic",
            "risk_reduction",
            "route_saving",
            "hazard_weighted",
            "accessibility_deficit",
        }
        decision_opportunity = bool(
            not terminal
            and dynamic_strategy
            and self.remaining_deployments > 0
            and self._decision_due(simulation_time)
            and observation.has_feasible_action
        )
        # An action's SMDP transition ends at the next action actually taken,
        # or at the true physical terminal boundary. Exhausting the budget or
        # temporarily having no feasible site does not censor later outcomes.
        if self.pending is not None and (terminal or decision_opportunity):
            completed_action = int(self.pending.decision.action_index)
            breakdown = self._record_completed_interval(
                after=observation,
                terminal=terminal,
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

            selection_started = time.perf_counter()
            if self.action_objective_baseline is None:
                self.action_objective_baseline = objective_breakdown
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
            wait_return = float("nan")
            wait_bootstrap_value = float("nan")
            wait_components = None
            wait_bootstrap_components = None
            wait_outcome_target = None
            if self.counterfactual_credit and self.train_mode:
                # Before the shelter exists: a branch taken after the install
                # would not be a counterfactual of this decision at all.
                (
                    wait_return,
                    wait_bootstrap_value,
                    wait_components,
                    wait_bootstrap_components,
                    wait_outcome_target,
                ) = self._collect_wait_baseline(
                    observation,
                    simulation_time=simulation_time,
                )
            policy_improvement = None
            if (
                self.nmcc_policy_improvement
                and self.train_mode
                and recurrent_step is not None
            ):
                # Also before the shelter exists: every branched cell, and the
                # executed one, is valued from the same pre-intervention state.
                policy_improvement = self._collect_policy_improvement(
                    observation,
                    recurrent_step,
                    decision,
                    simulation_time=simulation_time,
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
            observation_history = (
                tuple(self.observation_cache)
                if self.train_mode
                else tuple()
            )
            if self.train_mode and not observation_history:
                raise RuntimeError("A recurrent RL action requires cached observations")
            self.pending = PendingDecision(
                observation=observation,
                decision=decision,
                receipt=receipt,
                observation_history=observation_history,
                wait_return=wait_return,
                wait_bootstrap_value=wait_bootstrap_value,
                wait_component_return=wait_components,
                wait_bootstrap_components=wait_bootstrap_components,
                wait_outcome_target=wait_outcome_target,
                wait_steps=(
                    int(self.counterfactual_horizon)
                    if self.counterfactual_credit and self.train_mode
                    else 0
                ),
                counterfactual_previous_outcome=observation.outcome,
                counterfactual_discounted_components=(
                    np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float64)
                    if self.counterfactual_credit and self.train_mode
                    else None
                ),
                recurrent_state_at_decision=(
                    tuple(state.detach() for state in self.recurrent_state)
                    if self.counterfactual_credit and self.recurrent_state is not None
                    else None
                ),
                policy_improvement=policy_improvement,
            )
            self.observation_cache.clear()
            self.decision_index += 1

        self.last_simulation_time = simulation_time
        self.episode_done = terminal
        result = self._empty_step_result()
        result.update(
            {
                "reward": 0.0 if breakdown is None else float(breakdown.total),
                "reward_norm": 0.0 if breakdown is None else float(breakdown.total),
                "reward_safe": 0.0 if breakdown is None else float(breakdown.safe_completion_reward),
                "reward_casualty": 0.0 if breakdown is None else float(breakdown.casualty_penalty),
                "reward_evacuation_time": (
                    0.0
                    if breakdown is None
                    else float(breakdown.evacuation_time_penalty)
                ),
                "reward_hazard_exposure": (
                    0.0
                    if breakdown is None
                    else float(breakdown.hazard_exposure_penalty)
                ),
                "reward_risk_time": 0.0 if breakdown is None else float(breakdown.risk_time_penalty),
                "reward_shelter_service": 0.0,
                "new_safe_completions": 0 if breakdown is None else int(breakdown.new_safe_completions),
                "new_casualties": 0 if breakdown is None else int(breakdown.new_casualties),
                "attributed_shelter_service": 0,
                "active_person_time": (
                    0.0 if breakdown is None else float(breakdown.active_person_time)
                ),
                "hazard_exposure_person_time": (
                    0.0
                    if breakdown is None
                    else float(breakdown.hazard_exposure_person_time)
                ),
                "risk_weighted_person_time": 0.0 if breakdown is None else float(breakdown.risk_weighted_person_time),
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
                "objective_episode_return": float(objective_breakdown.total),
                "observation_build_latency_ms": (
                    observation_ms if receipt is not None else 0.0
                ),
                "policy_selection_latency_ms": selection_ms,
                "deployment_execution_latency_ms": execution_ms,
                "end_to_end_deployment_latency_ms": end_to_end_ms,
            }
        )
        if self.debug and (receipt is not None or breakdown is not None):
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
            "reward_risk_time": 0.0,
            "reward_shelter_service": 0.0,
            "new_safe_completions": 0,
            "new_casualties": 0,
            "attributed_shelter_service": 0,
            "risk_weighted_person_time": 0.0,
            "active_person_time": 0.0,
            "hazard_exposure_person_time": 0.0,
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
            "objective_episode_return": 0.0,
            "observation_build_latency_ms": 0.0,
            "policy_selection_latency_ms": 0.0,
            "deployment_execution_latency_ms": 0.0,
            "end_to_end_deployment_latency_ms": 0.0,
        }

    def end_episode(self, *, finalize_rollout: bool = False) -> Dict[str, float]:
        if self.pending is not None:
            raise RuntimeError("Episode ended before the pending regional decision was finalized")
        objective = self.reward_model.evaluate(
            before=self.objective_initial_outcome,
            after=self.objective_latest_outcome,
            active_person_time=self.objective_active_person_time,
            hazard_exposure_person_time=self.objective_hazard_exposure_person_time,
            initial_population=self.initial_population,
            horizon=self.horizon,
        )
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
        if self.action_objective_baseline is None:
            post_action_objective = 0.0
            post_action_components = np.zeros(
                len(REWARD_COMPONENT_NAMES),
                dtype=np.float64,
            )
        else:
            post_action_objective = float(
                objective.total - self.action_objective_baseline.total
            )
            post_action_components = (
                objective.component_vector().astype(np.float64)
                - self.action_objective_baseline.component_vector().astype(np.float64)
            )
        reward_accounting_gap = float(self.episode_return - post_action_objective)
        if self.train_mode and self.traj and abs(reward_accounting_gap) > 1e-6:
            raise RuntimeError(
                "Post-action transition rewards do not reconcile to the full "
                f"objective tail (gap={reward_accounting_gap:.9g})"
            )
        diagnostics = {
            "episode_return": float(self.episode_return),
            "safe_completion_reward": float(self.episode_safe_reward),
            "casualty_penalty": float(self.episode_casualty_penalty),
            "evacuation_time_penalty": float(
                self.episode_evacuation_time_penalty
            ),
            "hazard_exposure_penalty": float(
                self.episode_hazard_exposure_penalty
            ),
            "risk_time_penalty": float(
                self.episode_evacuation_time_penalty
                + self.episode_hazard_exposure_penalty
            ),
            "shelter_service_reward": 0.0,
            "risk_weighted_person_time": float(
                -(
                    self.episode_evacuation_time_penalty
                    + self.episode_hazard_exposure_penalty
                )
                * self.initial_population
                * self.horizon
            ),
            "active_person_time": float(
                -self.episode_evacuation_time_penalty
                * self.initial_population
                * self.horizon
            ),
            "hazard_exposure_person_time": float(
                -self.episode_hazard_exposure_penalty
                * self.initial_population
                * self.horizon
            ),
            "normalized_risk_weighted_person_time": float(
                -self.episode_evacuation_time_penalty
                - self.episode_hazard_exposure_penalty
            ),
            "decisions": float(self.action_comparisons),
            "heuristic_agreement_rate": (
                float(self.action_agreements) / float(self.action_comparisons)
                if self.action_comparisons > 0
                else 0.0
            ),
            "entropy": (
                float(np.mean(self.behavior_entropies))
                if self.behavior_entropies
                else 0.0
            ),
            "optimizer_updated": 0.0,
            "update_entropy": 0.0,
            "nmcc_counterfactual_fraction": 0.0,
            "nmcc_counterfactual_advantage_mean": 0.0,
            "nmcc_counterfactual_advantage_sd": 0.0,
            "nmcc_gae_advantage_sd": 0.0,
            "nmcc_guidance_weight": float(
                self._current_nmcc_guidance_weight()
            ),
            "nmcc_teacher_coefficient": float(
                self._current_nmcc_teacher_coef()
            ),
            "nmcc_natural_loss": 0.0,
            "nmcc_causal_loss": 0.0,
            "nmcc_dueling_loss": 0.0,
            "nmcc_teacher_loss": 0.0,
            "nmcc_causal_uncertainty": 0.0,
            **{name: 0.0 for name in self.NMCC_PI_DIAGNOSTIC_NAMES},
            **{name: 0.0 for name in self.LEARNER_FLOW_DIAGNOSTIC_NAMES},
            "nmcc_training_phase_index": float(
                NMCC_TRAINING_PHASES.index(
                    self._current_nmcc_training_phase()
                )
            ),
            "nmcc_actor_enabled": float(self._actor_training_enabled()),
            "nmcc_causal_model_enabled": float(
                self._current_nmcc_training_phase() != "natural_pretrain"
            ),
            "nmcc_effective_counterfactual_weight": float(
                self._effective_counterfactual_weight()
            ),
            "nmcc_rollout_updates_completed": float(
                self.rollout_updates_completed
            ),
            "nmcc_actor_optimizer_updates": float(
                self.actor_optimizer_updates
            ),
            "actor_rollout_updates": float(self.actor_rollout_updates),
            "critic_optimizer_updates": float(self.critic_optimizer_updates),
            "entropy_coefficient": float(self._current_entropy_coef()),
            "action_temperature": float(
                self._current_action_temperature()
            ),
            "exploration_rate": float(self._current_exploration_rate()),
            "approximate_kl": 0.0,
            "attempted_kl": 0.0,
            "kl_rollback": 0.0,
            "actor_update_accepted": 0.0,
            "actor_update_rejected": 0.0,
            "clip_fraction": 0.0,
            "gradient_norm": 0.0,
            "actor_gradient_norm": 0.0,
            "critic_gradient_norm": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "explained_variance": 0.0,
            "actor_mc_return_sd": 0.0,
            "actor_advantage_sd": 0.0,
            "actor_baseline_coverage": 0.0,
            "critic_td_error_sd": 0.0,
            "critic_epochs_completed": 0.0,
            "actor_epochs_completed": 0.0,
            "epochs_completed": 0.0,
            "transitions": 0.0,
            "training_episode_has_decision": float(bool(self.traj)),
            "observation_frames_seen": float(self.observation_frames_seen),
            "post_action_objective_return": post_action_objective,
            "reward_accounting_gap": reward_accounting_gap,
            "post_action_safe_completion_reward": float(post_action_components[0]),
            "post_action_casualty_penalty": float(post_action_components[1]),
            "post_action_evacuation_time_penalty": float(post_action_components[2]),
            "post_action_hazard_exposure_penalty": float(post_action_components[3]),
            "reward_accounting_gap_safe_completion": float(
                self.episode_safe_reward - post_action_components[0]
            ),
            "reward_accounting_gap_casualty": float(
                self.episode_casualty_penalty - post_action_components[1]
            ),
            "reward_accounting_gap_evacuation_time": float(
                self.episode_evacuation_time_penalty - post_action_components[2]
            ),
            "reward_accounting_gap_hazard_exposure": float(
                self.episode_hazard_exposure_penalty - post_action_components[3]
            ),
            "learning_rate": float(
                self.actor_optimizer.param_groups[0]["lr"]
                if self.actor_optimizer is not None
                else self.actor_lr
            ),
            "learning_rate_before_update": float(
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
            "objective_episode_return": float(objective.total),
            "objective_safe_completion_reward": float(objective.safe_completion_reward),
            "objective_casualty_penalty": float(objective.casualty_penalty),
            "objective_evacuation_time_penalty": float(
                objective.evacuation_time_penalty
            ),
            "objective_hazard_exposure_penalty": float(
                objective.hazard_exposure_penalty
            ),
            "objective_risk_time_penalty": float(objective.risk_time_penalty),
            "objective_shelter_service_reward": 0.0,
            "objective_active_person_time": float(objective.active_person_time),
            "objective_hazard_exposure_person_time": float(
                objective.hazard_exposure_person_time
            ),
            "objective_risk_weighted_person_time": float(
                objective.risk_weighted_person_time
            ),
            "precommit_decisions": float(
                self.deployments_made
                if self.deployment_strategy == "rl_precommit"
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
            # An all-unsafe action mask can legitimately produce no decision.
            # Such an episode contributes outcome diagnostics but no fabricated
            # policy transition or terminal marker to the on-policy rollout.
            if self.traj:
                self.rollout_traj.extend(self.traj)
                self.rollout_episode_count += 1
            if self.rollout_traj and (
                self.rollout_episode_count >= self.rollout_episodes
                or bool(finalize_rollout)
            ):
                diagnostics.update(self._optimize_policy(self.rollout_traj))
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
            self._append_training_diagnostics(diagnostics)
        self.traj.clear()
        self.observation_cache.clear()
        self.recurrent_state = None
        self.previous_temporal_observation = None
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
            raise RuntimeError("Recurrent PPO rollout ends inside an episode")
        if not episodes:
            raise RuntimeError("Recurrent PPO rollout contains no complete episode")
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
        guidance_weight: float,
        action_temperature: float,
        include_improvement: bool = False,
    ) -> tuple:
        """Replay complete causal histories and return decision-time outputs.

        With ``include_improvement`` two elements are appended: the
        intervention-value ensemble samples, shape (decisions, members, actions),
        and the fixed base-prior logits, shape (decisions, actions).  Returning
        both is essential: held-out policy improvement must evaluate the exact
        composite score that deployment will use, rather than a surrogate head
        in isolation.  The default keeps the historical ten-element contract
        for every caller.
        """
        flat_indices: list[int] = []
        improvement_samples = []
        logits = []
        scalar_values = []
        component_values = []
        residuals = []
        natural_outcomes = []
        causal_outcome_samples = []
        causal_component_means = []
        causal_component_stds = []
        teacher_logits = []
        prior_logits = []
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
                        uncertainty_penalty=self.nmcc_uncertainty_penalty,
                    )
                    recurrent_state = output.recurrent_state
                if output is None:
                    raise RuntimeError("A recurrent decision contains no observations")
                step_logits, step_teacher = self._apply_nmcc_guidance(
                    output.logits,
                    output.robust_causal_score,
                    transition.observation_history[-1].action_mask.to(
                        self.device
                    ).unsqueeze(0),
                    weight=guidance_weight,
                )
                step_logits = step_logits / float(action_temperature)
                if step_logits.size(0) != 1:
                    raise RuntimeError("Stored recurrent frames must represent one episode")
                flat_indices.append(int(transition_index))
                logits.append(step_logits)
                scalar_values.append(output.value.reshape(1))
                component_values.append(output.value_components.reshape(1, -1))
                residuals.append(output.learned_residual)
                natural_outcomes.append(output.natural_outcomes)
                causal_outcome_samples.append(output.causal_outcome_samples)
                causal_component_means.append(output.causal_component_mean)
                causal_component_stds.append(output.causal_component_std)
                teacher_logits.append(step_teacher)
                if include_improvement:
                    if output.improvement_value_samples is None:
                        raise RuntimeError("Policy did not produce intervention-value samples")
                    improvement_samples.append(output.improvement_value_samples)
                    prior_logits.append(output.prior_logits)
        outputs = (
            torch.as_tensor(flat_indices, dtype=torch.long, device=self.device),
            torch.cat(logits, dim=0),
            torch.cat(scalar_values, dim=0),
            torch.cat(component_values, dim=0),
            torch.cat(residuals, dim=0),
            torch.cat(natural_outcomes, dim=0),
            torch.cat(causal_outcome_samples, dim=0),
            torch.cat(causal_component_means, dim=0),
            torch.cat(causal_component_stds, dim=0),
            torch.cat(teacher_logits, dim=0),
        )
        if include_improvement:
            return outputs + (
                torch.cat(improvement_samples, dim=0),
                torch.cat(prior_logits, dim=0),
            )
        return outputs

    # -- NMCC policy improvement: exact within-state targets ---------------

    @property
    def improvement_gate_passed(self) -> bool:
        """Whether held-out paired outcomes justify using the learned correction.

        Each history entry is a lower confidence bound on the exact return gain
        of the model-selected branch over the fixed base-policy branch.  Rank
        correlation is diagnostic only: a model may rank cells consistently
        yet still make the controller worse at the decision boundary.
        """
        history = self.improvement_gate_history[-int(self.nmcc_pi_gate_updates):]
        return (
            len(history) >= int(self.nmcc_pi_gate_updates)
            and bool(np.all(np.isfinite(history)))
            and bool(np.all(np.asarray(history, dtype=np.float64) > 0.0))
        )

    def _behavior_probabilities(
        self,
        recurrent_step: "RecurrentPolicyStep",
        action_mask: torch.Tensor,
    ) -> np.ndarray:
        """Frozen score-policy distribution used to center branch values.

        Exploration is an external epsilon-greedy data-collection schedule;
        the policy-improvement target is fitted to the actor's score ordering,
        not to the exploratory mixture.
        """
        with torch.no_grad():
            controller_logits = self._controller_logits(
                recurrent_step, action_mask
            )
            masked = self._safe_masked_logits(
                controller_logits / self._current_action_temperature(),
                action_mask.reshape(1, -1),
            )
            probabilities = torch.softmax(masked, dim=-1).reshape(-1)
        return probabilities.detach().cpu().numpy().astype(np.float64)

    def _ensure_branch_valuer(self):
        if self._branch_valuer is None:
            self._branch_valuer = NPI.BranchValuer(
                self.core,
                builder=self.observation_builder,
                executor=self.executor,
                reward_model=self.reward_model,
                base_policy=self.nmcc_pi_base_policy,
                tapes=self.nmcc_pi_tapes,
                branch_horizon=self.nmcc_pi_branch_horizon,
                full_horizon_decisions=self.nmcc_pi_full_horizon_decisions,
            )
        return self._branch_valuer

    def _collect_policy_improvement(
        self,
        observation: RegionalObservation,
        recurrent_step: "RecurrentPolicyStep",
        decision: PolicyDecision,
        *,
        simulation_time: int,
    ) -> dict:
        """Branch cells from this decision state and build the improvement target.

        Called before the executor installs anything.  ``BranchValuer`` restores
        the simulator exactly afterwards, so the live episode continues from the
        state it was in.  Every branched cell is compared under the same
        independent future tape, which pairs the comparison without letting any
        branch see the real episode's future.
        """
        mask = np.asarray(observation.action_mask, dtype=bool)
        behavior = self._behavior_probabilities(
            recurrent_step, recurrent_step.frame.action_mask
        )
        feasible = np.flatnonzero(mask)
        base_action = int(NPI.BASE_POLICIES[self.nmcc_pi_base_policy](observation))
        branch_actions = NPI.select_branch_actions(
            feasible,
            behavior,
            decision_index=int(self.deployments_made),
            exhaustive_decisions=int(self.nmcc_pi_exhaustive_decisions),
            max_branches=int(self.nmcc_pi_max_branches),
            must_include=(int(decision.action_index), base_action),
            rng=self._improvement_rng,
        )
        clock = NPI.EpisodeClock(
            horizon=int(self.horizon),
            interval=int(self.shelter_action_interval),
            budget=int(self.maximum_deployments),
            population=int(self.initial_population),
            first_decision=int(self.first_decision_time),
            t=int(simulation_time),
            deployed=int(self.deployments_made),
            previous=observation.outcome,
            next_decision_time=int(simulation_time),
        )
        values = self._ensure_branch_valuer().value(
            observation,
            clock,
            branch_actions,
            episode_seed=int(getattr(self.core, "scenario_seed", 0) or 0),
        )
        exact_actions = values.actions
        exact_values = values.mean_values()
        exact_natural_outcome = values.wait_outcomes.mean(axis=0)
        exact_outcome_effect = values.paired_outcome_effects().mean(axis=0)

        # Exact within-state advantage, centered on the behavior policy over the
        # branched cells.  This is what the intervention-value ensemble learns.
        exact_advantage = np.zeros(mask.size, dtype=np.float64)
        exact_mask = np.zeros(mask.size, dtype=bool)
        exact_mask[exact_actions] = True
        weights = behavior[exact_actions]
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(
            exact_actions.size, 1.0 / exact_actions.size
        )
        exact_advantage[exact_actions] = exact_values - float(np.dot(weights, exact_values))

        target_actions = exact_actions
        target_values = exact_values
        model_filled = 0
        if (
            self.nmcc_pi_model_fill
            and self.improvement_gate_passed
            and recurrent_step.improvement_value_samples is not None
        ):
            samples = recurrent_step.improvement_value_samples.detach().reshape(
                self.nmcc_ensemble_size, -1
            ).cpu().numpy().astype(np.float64) * float(self.nmcc_pi_value_scale)
            model_mean = samples.mean(axis=0)
            model_std = samples.std(axis=0)
            unbranched = np.setdiff1d(feasible, exact_actions)
            if unbranched.size:
                # Anchor the model to the exact values on the branched cells, so
                # only its within-state differences are used, then discount
                # every filled value by its ensemble disagreement.
                anchor_exact = float(np.dot(weights, exact_values))
                anchor_model = float(np.dot(weights, model_mean[exact_actions]))
                filled = (
                    anchor_exact
                    + (model_mean[unbranched] - anchor_model)
                    - self.nmcc_pi_model_uncertainty_penalty * model_std[unbranched]
                )
                target_actions = np.concatenate((exact_actions, unbranched))
                target_values = np.concatenate((exact_values, filled))
                model_filled = int(unbranched.size)

        if self.nmcc_pi_actor_objective in {"score_ranking", "value_lcb"}:
            score_target = NPI.score_ranking_target(
                mask,
                exact_actions,
                exact_values,
                temperature=float(self.nmcc_pi_ranking_temperature),
            )
            target = None
            target_vector = score_target
            target_eta = float(self.nmcc_pi_ranking_temperature)
            target_kl = float(
                np.sum(
                    np.where(
                        score_target > 0.0,
                        score_target
                        * (
                            np.log(np.clip(score_target, 1e-300, None))
                            - np.log(np.clip(behavior, 1e-300, None))
                        ),
                        0.0,
                    )
                )
            )
        else:
            target = NPI.improvement_target(
                behavior,
                mask,
                target_actions,
                target_values,
                epsilon=float(self.nmcc_pi_epsilon),
                eta_min=float(self.nmcc_pi_eta_min),
            )
            target_vector = target.target
            target_eta = float(target.eta)
            target_kl = float(target.kl_to_old)
        outcome_effect = np.zeros((mask.size, len(NMCC_OUTCOME_NAMES)), dtype=np.float32)
        outcome_effect[exact_actions] = exact_outcome_effect.astype(np.float32)
        bootstrap = self._improvement_rng.poisson(
            1.0, size=int(self.nmcc_ensemble_size)
        ).astype(np.float64)
        record = {
            "target": target_vector,
            "behavior": behavior,
            "exact_mask": exact_mask.astype(np.float64),
            "exact_advantage": exact_advantage,
            "bootstrap": bootstrap,
            "natural_outcome": exact_natural_outcome.astype(np.float32),
            "outcome_effect": outcome_effect,
            "eta": target_eta,
            "kl_to_behavior": target_kl,
            "branched": int(exact_actions.size),
            "model_filled": model_filled,
            "feasible": int(feasible.size),
            "branch_best_action": int(exact_actions[int(np.argmax(exact_values))]),
            "base_action": int(values.base_action),
            "within_state_value_sd": float(np.std(exact_values)),
            "behavior_argmax_is_branch_best": bool(
                int(feasible[int(np.argmax(behavior[feasible]))])
                == int(exact_actions[int(np.argmax(exact_values))])
            ),
            "simulation_time": int(simulation_time),
            "decision_index": int(self.deployments_made),
            "actor_objective": self.nmcc_pi_actor_objective,
        }
        self.improvement_records.append(
            {
                key: record[key]
                for key in (
                    "eta",
                    "kl_to_behavior",
                    "branched",
                    "model_filled",
                    "feasible",
                    "within_state_value_sd",
                    "behavior_argmax_is_branch_best",
                    "simulation_time",
                    "decision_index",
                )
            }
        )
        return record

    # -- Hybrid NMCC exact common-noise counterfactual branching -----------

    @property
    def counterfactual_horizon(self) -> int:
        """Branch length ``L``.

        Defaults to one deployment interval, which is the choice that makes the
        acted branch coincide with the trajectory the episode is about to live
        anyway.  Only the ``WAIT`` branch is then extra simulation, so exact
        paired credit assignment costs one extra interval per decision --
        roughly 2x episode cost, not the ``|cells|x`` or ``O(H)x`` a naive
        parallel-rollout scheme would pay.
        """
        if self._counterfactual_horizon_override is not None:
            return max(1, int(self._counterfactual_horizon_override))
        return max(1, int(self.shelter_action_interval))

    def _ensure_brancher(self):
        if self._counterfactual_brancher is None:
            self._counterfactual_brancher = CB.CounterfactualBrancher(
                self.core,
                horizon=self.counterfactual_horizon,
                reward_model=self.reward_model,
                initial_population=int(self.initial_population),
                episode_horizon=int(self.horizon),
            )
        return self._counterfactual_brancher

    def _readonly_observation_values(
        self,
        observation: RegionalObservation,
        *,
        recurrent_state: Optional[tuple] = None,
    ) -> tuple[float, np.ndarray]:
        """Critic value of an off-trajectory state.

        The ``WAIT`` branch ends somewhere the episode never goes, so its value
        must be read without advancing the recurrent state or appending to the
        observation cache -- otherwise asking the counterfactual would corrupt
        the factual trajectory it exists to explain.  The decision epoch's
        recurrent state supplies the context, which is the correct
        conditioning: both branches share the same history up to the decision.
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

    def _collect_wait_baseline(
        self,
        observation: RegionalObservation,
        *,
        simulation_time: int,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Run the matched no-deployment branch from this decision's state.

        Must be called *before* the executor installs anything, so the branch
        starts from the pre-intervention state.  ``CounterfactualBranch``
        restores the simulator exactly afterwards, and
        ``test_a_branch_does_not_disturb_the_live_episode`` pins that property.
        """
        brancher = self._ensure_brancher()
        brancher.horizon = self.counterfactual_horizon
        base = CB.capture(self.core, label=f"decision-{self.decision_index}")
        try:
            branch = brancher._run_branch(
                label="waited",
                install_cell=None,
                steps=self.counterfactual_horizon,
            )
            wait_return = float(branch.discounted_return(self.gamma))
            wait_components = branch.discounted_component_return(self.gamma)
            wait_outcome_target = branch.normalized_outcome_vector(
                initial_population=self.initial_population,
                episode_horizon=self.horizon,
            )
            wait_value = 0.0
            wait_bootstrap_components = np.zeros(
                len(REWARD_COMPONENT_NAMES),
                dtype=np.float64,
            )
            if not branch.terminal:
                wait_observation = self.observation_builder.build(
                    decision_index=self.decision_index,
                    simulation_time=int(simulation_time)
                    + int(self.counterfactual_horizon),
                    remaining_deployments=self.remaining_deployments,
                )
                wait_value, wait_bootstrap_components = (
                    self._readonly_observation_values(wait_observation)
                )
        finally:
            CB.restore(self.core, base)
        return (
            wait_return,
            wait_value,
            wait_components,
            wait_bootstrap_components,
            wait_outcome_target,
        )

    def _counterfactual_advantage_for(
        self,
        pending: "PendingDecision",
        *,
        factual_components: np.ndarray,
        factual_bootstrap_components: np.ndarray,
        elapsed: int,
    ) -> tuple[Optional[float], Optional[np.ndarray]]:
        """``A_CF = (R_a - R_0) + gamma^L (V(s_L^a) - V(s_L^0)) - c(a)``.

        The ``WAIT`` return is a control variate: it depends on the state and
        the noise but not on which cell was chosen, so subtracting it removes
        the shared natural trajectory without moving the policy gradient's
        expectation.  What remains is the part of the outcome this decision was
        responsible for.
        """
        if not np.isfinite(pending.wait_return):
            return None, None
        if pending.wait_component_return is None:
            return None, None
        wait_bootstrap = (
            np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float64)
            if pending.wait_bootstrap_components is None
            else np.asarray(pending.wait_bootstrap_components, dtype=np.float64)
        )
        discount = float(self.gamma) ** max(1, int(elapsed))
        components = (
            np.asarray(factual_components, dtype=np.float64)
            - np.asarray(pending.wait_component_return, dtype=np.float64)
            + discount
            * (
                np.asarray(factual_bootstrap_components, dtype=np.float64)
                - wait_bootstrap
            )
        )
        # The intervention cost is kept explicit and assigned to the time/cost
        # branch so the component vector still sums exactly to the actor target.
        components[2] -= self.counterfactual_intervention_cost
        return float(components.sum()), components.astype(np.float32)

    def _blend_counterfactual_advantages(
        self,
        advantages: torch.Tensor,
        transitions: Sequence["Transition"],
        baseline_keys: Optional[Sequence[str]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Optionally blend exact NMCC effects using only a lagged baseline.

        The current rollout is never centered against itself. That former
        operation could erase a consistently beneficial controller's signal.
        """
        diagnostics = {
            "counterfactual_fraction": 0.0,
            "counterfactual_advantage_mean": 0.0,
            "counterfactual_advantage_sd": 0.0,
            "mc_advantage_sd": float(advantages.std(unbiased=False).item()),
            "gae_advantage_sd": float(advantages.std(unbiased=False).item()),
        }
        effective_weight = self._effective_counterfactual_weight()
        if not self.counterfactual_credit:
            return advantages, diagnostics

        values = torch.full_like(advantages, float("nan"))
        for index, transition in enumerate(transitions):
            if transition.counterfactual_advantage is not None:
                values[index] = transition.counterfactual_advantage.reshape(-1)[0]
        available = torch.isfinite(values)
        count = int(available.sum().item())
        if count == 0:
            return advantages, diagnostics

        present = values[available]
        diagnostics["counterfactual_fraction"] = float(count) / float(advantages.numel())
        diagnostics["counterfactual_advantage_mean"] = float(present.mean().item())
        diagnostics["counterfactual_advantage_sd"] = float(
            present.std(unbiased=False).item()
        )
        if effective_weight <= 0.0:
            return advantages, diagnostics

        if baseline_keys is None:
            baseline_keys = [f"counterfactual_position={index}" for index in range(len(transitions))]
        if len(baseline_keys) != len(transitions):
            raise ValueError("Counterfactual baseline keys must align with transitions")
        normalized = torch.zeros_like(present)
        present_indices = torch.nonzero(available, as_tuple=False).flatten()
        for output_index, transition_index in enumerate(present_indices.tolist()):
            state = self.counterfactual_return_baselines.get(
                str(baseline_keys[transition_index])
            )
            if state is None:
                normalized[output_index] = present[output_index]
                continue
            scale = max(
                self.advantage_scale_floor,
                float(max(0.0, state["variance"])) ** 0.5,
            )
            normalized[output_index] = (
                present[output_index] - float(state["mean"])
            ) / scale

        blended = advantages.clone()
        weight = float(effective_weight)
        blended[available] = (
            (1.0 - weight) * advantages[available] + weight * normalized
        )
        return blended, diagnostics

    def _component_credit_targets(
        self,
        component_rewards: torch.Tensor,
        component_values: torch.Tensor,
        dones: torch.Tensor,
        durations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return complete actor MC returns and independent critic TD(0) labels.

        Both targets use the semi-Markov duration discount. Actor labels never
        contain a critic bootstrap; critic labels bootstrap exactly one
        finalized action interval from the frozen rollout value.
        """
        total, component_count = component_rewards.shape
        actor_returns = torch.zeros_like(component_rewards)
        critic_targets = torch.zeros_like(component_rewards)
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
            if index + 1 < total:
                next_value = component_values[index + 1]
            else:
                next_value = torch.zeros_like(running_return)
            critic_targets[index] = (
                component_rewards[index]
                + decision_discount * nonterminal * next_value
            )
        return actor_returns, critic_targets

    def _regime_position_keys(
        self,
        episodes: Sequence[Sequence[int]],
        transition_count: int,
    ) -> list[str]:
        """Build interpretable lagged-baseline strata for every decision."""
        keys = [""] * int(transition_count)
        regime = (
            f"city={str(getattr(self.core, 'cityID', 'single_city'))}"
            f"|population={int(self.initial_population)}"
            f"|hazards={int(getattr(self.core, 'hazardVol', 0))}"
        )
        for episode in episodes:
            for position, transition_index in enumerate(episode):
                keys[int(transition_index)] = f"{regime}|decision={position}"
        if any(not key for key in keys):
            raise RuntimeError("Every transition must receive a baseline stratum")
        return keys

    def _normalize_with_lagged_baseline(
        self,
        returns: torch.Tensor,
        keys: Sequence[str],
        store: Mapping[str, Mapping[str, float]],
    ) -> tuple[torch.Tensor, float]:
        """Center returns on prior-rollout statistics without batch leakage."""
        if returns.ndim != 1 or len(keys) != int(returns.numel()):
            raise ValueError("Lagged baseline inputs must be one-dimensional and aligned")
        normalized = torch.empty_like(returns)
        covered = 0
        for index, key in enumerate(keys):
            state = store.get(str(key))
            if state is None:
                normalized[index] = returns[index]
                continue
            covered += 1
            scale = max(
                self.advantage_scale_floor,
                float(max(0.0, state["variance"])) ** 0.5,
            )
            normalized[index] = (returns[index] - float(state["mean"])) / scale
        return normalized, float(covered) / float(max(1, len(keys)))

    def _update_lagged_baseline(
        self,
        store: dict[str, dict[str, float]],
        keys: Sequence[str],
        values: torch.Tensor,
    ) -> None:
        """Update an EMA mean/variance only after the rollout update ends."""
        decay = float(self.actor_baseline_decay)
        for key, tensor in zip(keys, values.detach().cpu()):
            value = float(tensor.item())
            state = store.get(str(key))
            if state is None:
                store[str(key)] = {"count": 1.0, "mean": value, "variance": 0.0}
                continue
            previous_mean = float(state["mean"])
            delta = value - previous_mean
            state["mean"] = decay * previous_mean + (1.0 - decay) * value
            state["variance"] = decay * (
                float(state["variance"]) + (1.0 - decay) * delta * delta
            )
            state["count"] = float(state.get("count", 0.0)) + 1.0

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

    def _improvement_value_loss(
        self,
        samples: torch.Tensor,
        available: torch.Tensor,
        behavior: torch.Tensor,
        exact_mask: torch.Tensor,
        advantage: torch.Tensor,
        bootstrap: torch.Tensor,
    ) -> torch.Tensor:
        """Bootstrap-weighted within-state regression of the value ensemble.

        Each member predicts a per-cell value; only its behavior-centered
        contrast over the exactly branched cells is compared with the exact
        advantage, so any cell-independent level (V_wait, the natural
        momentum of the scenario) is outside what the heads can be asked to
        explain.  Poisson(1) weights make the members disagree where the
        data are thin, which is the uncertainty the fill-in penalizes.
        """
        rows = available & (exact_mask.sum(dim=-1) >= 2)
        if not bool(rows.any()):
            return samples.new_zeros(())
        prediction = samples[rows]  # (rows, members, actions)
        mask = exact_mask[rows]
        weights = behavior[rows] * mask
        weight_total = weights.sum(dim=-1, keepdim=True)
        uniform = mask / mask.sum(dim=-1, keepdim=True)
        weights = torch.where(weight_total > 0, weights / weight_total.clamp_min(1e-30), uniform)
        centered = prediction - (weights.unsqueeze(1) * prediction).sum(dim=-1, keepdim=True)
        target = (
            advantage[rows] / float(self.nmcc_pi_value_scale)
        ).unsqueeze(1)
        error = (centered - target).square() * mask.unsqueeze(1)
        member_weight = bootstrap[rows].unsqueeze(-1)  # (rows, members, 1)
        denominator = (member_weight * mask.unsqueeze(1)).sum()
        if float(denominator.item()) <= 0.0:
            return samples.new_zeros(())
        return (member_weight * error).sum() / denominator

    def _improvement_pairwise_loss(
        self,
        samples: torch.Tensor,
        available: torch.Tensor,
        exact_mask: torch.Tensor,
        advantage: torch.Tensor,
        bootstrap: torch.Tensor,
    ) -> torch.Tensor:
        """Fit all exact within-state candidate differences directly.

        Scalar calibration is useful to the uncertainty bound, but the action
        depends only on ordering.  Pairwise smooth-L1 supervision gives every
        exact comparison equal opportunity to reach the full GNN/LSTM and is
        invariant to the unidentified state-level value offset.
        """
        rows = available & (exact_mask.sum(dim=-1) >= 2)
        if not bool(rows.any()):
            return samples.new_zeros(())
        prediction = samples[rows]
        target = advantage[rows] / float(self.nmcc_pi_value_scale)
        predicted_difference = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
        target_difference = target.unsqueeze(-1) - target.unsqueeze(-2)
        pair_mask = (
            exact_mask[rows].to(torch.bool).unsqueeze(-1)
            & exact_mask[rows].to(torch.bool).unsqueeze(-2)
        )
        upper = torch.triu(
            torch.ones(
                samples.size(-1),
                samples.size(-1),
                dtype=torch.bool,
                device=self.device,
            ),
            diagonal=1,
        )
        pair_mask = pair_mask & upper.unsqueeze(0)
        error = F.smooth_l1_loss(
            predicted_difference,
            target_difference.unsqueeze(1).expand_as(predicted_difference),
            reduction="none",
        )
        weights = bootstrap[rows].unsqueeze(-1).unsqueeze(-1)
        denominator = (weights * pair_mask.unsqueeze(1)).sum()
        if float(denominator.item()) <= 0.0:
            return samples.new_zeros(())
        return (error * weights * pair_mask.unsqueeze(1)).sum() / denominator

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
        "actor_rollbacks",
        "actor_trust_region_reached",
        "actor_line_search_fraction",
        "actor_reverse_kl",
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
        actor_rollbacks: int,
        line_search_fractions: Sequence[float] = (),
        trust_region_reached: bool = False,
        reverse_kl: float = float("nan"),
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
            "actor_rollbacks": float(actor_rollbacks),
            "actor_trust_region_reached": float(trust_region_reached),
            "actor_line_search_fraction": (
                float(np.mean(line_search_fractions)) if line_search_fractions else 1.0
            ),
            "actor_reverse_kl": float(reverse_kl),
        }

    NMCC_PI_DIAGNOSTIC_NAMES = (
        "nmcc_pi_coverage",
        "nmcc_pi_branched",
        "nmcc_pi_feasible",
        "nmcc_pi_model_filled",
        "nmcc_pi_mean_eta",
        "nmcc_pi_eta_at_floor",
        "nmcc_pi_target_kl_to_behavior",
        "nmcc_pi_requested_kl",
        "nmcc_pi_fit_kl_before",
        "nmcc_pi_fit_kl_after",
        "nmcc_pi_fit_converged",
        "nmcc_pi_top1_before",
        "nmcc_pi_top1_after",
        "nmcc_pi_gate_rank_corr",
        "nmcc_pi_gate_passed",
        "nmcc_pi_improvement_loss",
        "nmcc_pi_within_state_value_sd",
        "nmcc_pi_replay_episodes",
        "nmcc_pi_replay_training_states",
        "nmcc_pi_replay_training_loss",
        "nmcc_pi_replay_training_rank",
        "nmcc_pi_replay_training_top1",
        "nmcc_pi_validation_states",
        "nmcc_pi_validation_episodes",
        "nmcc_pi_validation_loss",
        "nmcc_pi_validation_gain",
        "nmcc_pi_validation_gain_se",
        "nmcc_pi_validation_gain_lower",
        "nmcc_pi_validation_rank",
        "nmcc_pi_validation_top1",
        "nmcc_pi_replay_epochs_completed",
    )

    def _improvement_diagnostics(
        self,
        *,
        coverage: float,
        requested_kl: float,
        fit_kl_before: float,
        fit_kl_after: float,
        top1_before: float,
        top1_after: float,
        gate_rank: float,
        converged: bool,
        value_losses: Sequence[float],
    ) -> Dict[str, float]:
        """Summarize this rollout's NMCC-PI evidence and consume its records.

        The pair ``top1_before``/``top1_after`` is the direct answer to
        "did the policy move, and toward the exactly better cell?": the rate
        at which the actor's most probable cell is the branch-best cell,
        before and after the M-step, on the same decisions.
        """
        records = self.improvement_records
        self.improvement_records = []

        def mean(key: str) -> float:
            values = [float(item[key]) for item in records]
            return float(np.mean(values)) if values else float("nan")

        eta_floor = [
            float(item["eta"]) <= float(self.nmcc_pi_eta_min) * (1.0 + 1e-9)
            for item in records
        ]
        return {
            "nmcc_pi_coverage": coverage,
            "nmcc_pi_branched": mean("branched"),
            "nmcc_pi_feasible": mean("feasible"),
            "nmcc_pi_model_filled": mean("model_filled"),
            "nmcc_pi_mean_eta": mean("eta"),
            "nmcc_pi_eta_at_floor": float(np.mean(eta_floor)) if eta_floor else float("nan"),
            "nmcc_pi_target_kl_to_behavior": mean("kl_to_behavior"),
            "nmcc_pi_requested_kl": float(requested_kl),
            "nmcc_pi_fit_kl_before": float(fit_kl_before),
            "nmcc_pi_fit_kl_after": float(fit_kl_after),
            "nmcc_pi_fit_converged": float(converged),
            "nmcc_pi_top1_before": float(top1_before),
            "nmcc_pi_top1_after": float(top1_after),
            "nmcc_pi_gate_rank_corr": float(gate_rank),
            "nmcc_pi_gate_passed": float(self.improvement_gate_passed),
            "nmcc_pi_improvement_loss": (
                float(np.mean(value_losses)) if value_losses else 0.0
            ),
            "nmcc_pi_within_state_value_sd": mean("within_state_value_sd"),
        }

    def _policy_log_probabilities(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
        action_masks: torch.Tensor,
        *,
        guidance_weight: float,
        action_temperature: float,
    ) -> torch.Tensor:
        """Masked log-probabilities of the current policy, in transition order."""
        with torch.no_grad():
            selected, logits, *_ = self._evaluate_recurrent_sequences(
                transitions,
                episodes,
                guidance_weight=guidance_weight,
                action_temperature=action_temperature,
            )
            ordered = torch.empty_like(logits)
            ordered[selected] = logits
            return F.log_softmax(self._safe_masked_logits(ordered, action_masks), dim=-1)

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

    def _line_search_to_trust_region(
        self,
        snapshot: Dict[str, torch.Tensor],
        cap: float,
        statistics,
        *,
        iterations: int = 8,
    ) -> float:
        """Scale the last parameter step so the trust-region KL ends <= ``cap``.

        Bisects alpha in [0, 1] for theta = snapshot + alpha (theta - snapshot).
        The KL is 0 at alpha = 0 and > cap at alpha = 1, so the returned
        fraction is within 2**-iterations of the boundary and always feasible.
        """
        current = {
            name: parameter.detach().clone() for name, parameter in self.actor_named_parameters
        }

        def place(alpha: float) -> None:
            with torch.no_grad():
                for name, parameter in self.actor_named_parameters:
                    parameter.copy_(snapshot[name] + alpha * (current[name] - snapshot[name]))

        low, high = 0.0, 1.0
        for _ in range(int(iterations)):
            middle = 0.5 * (low + high)
            place(middle)
            if statistics()["kl"] <= cap:
                low = middle
            else:
                high = middle
        place(low)
        return float(low)

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
        actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        improvement_available: torch.Tensor,
        improvement_targets: torch.Tensor,
        improvement_behavior: torch.Tensor,
        branch_best: torch.Tensor,
        improvement_exact_mask: torch.Tensor,
        *,
        guidance_weight: float,
        action_temperature: float,
    ) -> Dict[str, float]:
        """Exact trust-region and fit statistics of the current actor.

        ``kl`` is the trust-region quantity: exact KL(pi_new || pi_behavior),
        the direction the E-step bounds, on every decision that carries its
        behavior distribution, and the sampled estimator only on decisions
        that do not.  ``reverse_kl`` is KL(pi_behavior || pi_new), reported
        only.  ``target_kl`` is the remaining M-step residual KL(q || pi_new).
        """
        with torch.no_grad():
            selected, logits, *_ = self._evaluate_recurrent_sequences(
                transitions,
                episodes,
                guidance_weight=guidance_weight,
                action_temperature=action_temperature,
            )
            expected = torch.arange(len(transitions), device=self.device)
            if not torch.equal(selected, expected):
                raise RuntimeError("Complete recurrent replay changed transition order")
            log_new = F.log_softmax(self._safe_masked_logits(logits, action_masks), dim=-1)
            per_decision = torch.zeros(len(transitions), device=self.device)
            available = improvement_available
            if bool(available.any()):
                behavior = improvement_behavior[available]
                target = improvement_targets[available]
                log_rows = log_new[available]
                feasible_rows = action_masks[available].to(torch.bool)
                reverse_kl = torch.where(
                    behavior > 0,
                    behavior * (torch.log(behavior.clamp_min(1e-30)) - log_rows),
                    torch.zeros_like(behavior),
                ).sum(dim=-1)
                # The E-step bounds KL(q || pi_old); the trust region must bound
                # the same direction, KL(pi_new || pi_old), or a policy that fits
                # the target exactly is outside it (measured on the testbed:
                # KL(pi_old || q) = 0.87-0.91 at t = 1 when KL(q || pi_old) = 0.5).
                new_rows = log_rows.exp()
                behavior_kl = torch.where(
                    feasible_rows & (behavior > 0),
                    new_rows * (log_rows - torch.log(behavior.clamp_min(1e-30))),
                    torch.zeros_like(behavior),
                ).sum(dim=-1)
                if self.nmcc_pi_actor_objective == "score_ranking":
                    exact_rows = improvement_exact_mask[available].to(torch.bool)
                    exact_log_rows = F.log_softmax(
                        self._safe_masked_logits(log_rows, exact_rows), dim=-1
                    )
                    target_kl = torch.where(
                        target > 0,
                        target
                        * (
                            torch.log(target.clamp_min(1e-30))
                            - exact_log_rows
                        ),
                        torch.zeros_like(target),
                    ).sum(dim=-1)
                    top1 = (
                        self._safe_masked_logits(log_rows, exact_rows).argmax(dim=-1)
                        == branch_best[available]
                    ).to(torch.float32)
                else:
                    target_kl = torch.where(
                        target > 0,
                        target * (torch.log(target.clamp_min(1e-30)) - log_rows),
                        torch.zeros_like(target),
                    ).sum(dim=-1)
                    top1 = (
                        log_rows.argmax(dim=-1) == branch_best[available]
                    ).to(torch.float32)
                per_decision[available] = behavior_kl
                mean_target_kl = float(target_kl.mean().item())
                top1_rate = float(top1.mean().item())
                mean_reverse_kl = float(reverse_kl.mean().item())
            else:
                mean_target_kl = 0.0
                top1_rate = float("nan")
                mean_reverse_kl = 0.0
            missing = ~available
            if bool(missing.any()):
                log_ratio, ratio = self._ppo_log_ratio(
                    log_new[missing].gather(1, actions[missing].reshape(-1, 1)).squeeze(1),
                    old_log_probabilities[missing],
                )
                per_decision[missing] = (ratio - 1.0) - log_ratio
            kl = per_decision.mean()
        if not torch.isfinite(kl):
            raise FloatingPointError("NMCC-PI trust-region KL became non-finite")
        return {
            "kl": float(kl.item()),
            "target_kl": mean_target_kl,
            "top1": top1_rate,
            "reverse_kl": mean_reverse_kl,
        }

    def _full_rollout_policy_kl(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
        action_masks: torch.Tensor,
        actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        *,
        guidance_weight: float,
        action_temperature: float,
    ) -> float:
        """Measure the retained policy change on the complete recurrent batch."""
        with torch.no_grad():
            selected, logits, *_ = self._evaluate_recurrent_sequences(
                transitions,
                episodes,
                guidance_weight=guidance_weight,
                action_temperature=action_temperature,
            )
            expected = torch.arange(len(transitions), device=self.device)
            if not torch.equal(selected, expected):
                raise RuntimeError("Complete recurrent replay changed transition order")
            distribution = torch.distributions.Categorical(
                logits=self._safe_masked_logits(logits, action_masks)
            )
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

    def _append_improvement_replay(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
    ) -> None:
        """Retain complete labelled episodes with bounded reservoir sampling."""
        for indices in episodes:
            episode = tuple(
                self._deserialize_transition(
                    self._serialize_transition(transitions[int(index)]),
                    target_device=torch.device("cpu"),
                )
                for index in indices
            )
            if not any(item.improvement_advantage is not None for item in episode):
                continue
            record = {
                "episode_id": int(self.improvement_replay_next_id),
                "transitions": episode,
            }
            self.improvement_replay_next_id += 1
            self.improvement_replay_seen += 1
            if len(self.improvement_replay) < int(self.nmcc_pi_replay_max_episodes):
                self.improvement_replay.append(record)
                continue
            slot = int(self._improvement_rng.integers(0, self.improvement_replay_seen))
            if slot < int(self.nmcc_pi_replay_max_episodes):
                self.improvement_replay[slot] = record

    def _validate_improvement_replay(self) -> None:
        if len(self.improvement_replay) > int(self.nmcc_pi_replay_max_episodes):
            raise ValueError("checkpoint intervention replay exceeds its configured bound")
        identifiers = [int(item["episode_id"]) for item in self.improvement_replay]
        if len(identifiers) != len(set(identifiers)) or any(value < 0 for value in identifiers):
            raise ValueError("checkpoint intervention replay has invalid episode ids")
        for item in self.improvement_replay:
            transitions = list(item["transitions"])
            if not transitions:
                raise ValueError("checkpoint intervention replay contains an empty episode")
            groups = self._group_episode_indices(transitions)
            if len(groups) != 1:
                raise ValueError("checkpoint intervention replay split an episode")
            for transition in transitions:
                for frame in transition.observation_history:
                    self._validate_frame(frame)
                if transition.improvement_advantage is None:
                    continue
                if transition.improvement_base_action is None:
                    raise ValueError("checkpoint intervention label has no base action")
                if transition.improvement_advantage.shape != (self.num_candidate_actions,):
                    raise ValueError("checkpoint intervention label shape is incompatible")
        if self.improvement_replay_seen < len(self.improvement_replay):
            raise ValueError("checkpoint intervention replay seen-count is inconsistent")

    def _split_improvement_replay(
        self,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Deterministic episode-level fit/early-stop/gate split.

        A full episode belongs to one partition. The gate set is never used to
        fit parameters or choose an early-stopping epoch.
        """
        period = max(3, int(round(1.0 / float(self.nmcc_pi_validation_fraction))))
        ordered = sorted(self.improvement_replay, key=lambda item: item["episode_id"])
        validation = [item for item in ordered if int(item["episode_id"]) % period == 0]
        gate = [item for item in ordered if int(item["episode_id"]) % period == 1]
        training = [
            item
            for item in ordered
            if int(item["episode_id"]) % period not in {0, 1}
        ]
        return training, validation, gate

    @staticmethod
    def _flatten_replay_episodes(records: Sequence[dict]) -> tuple[list[Transition], list[list[int]]]:
        transitions: list[Transition] = []
        episodes: list[list[int]] = []
        for record in records:
            start = len(transitions)
            transitions.extend(record["transitions"])
            episodes.append(list(range(start, len(transitions))))
        return transitions, episodes

    def _improvement_replay_arrays(
        self, transitions: Sequence[Transition]
    ) -> tuple[torch.Tensor, ...]:
        available = torch.as_tensor(
            [item.improvement_advantage is not None for item in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        behavior = torch.zeros(
            len(transitions), self.num_candidate_actions,
            dtype=torch.float32, device=self.device,
        )
        exact_mask = torch.zeros_like(behavior)
        advantage = torch.zeros_like(behavior)
        bootstrap = torch.ones(
            len(transitions), self.nmcc_ensemble_size,
            dtype=torch.float32, device=self.device,
        )
        base_action = torch.full(
            (len(transitions),), -1, dtype=torch.long, device=self.device
        )
        for index, item in enumerate(transitions):
            if not bool(available[index]):
                continue
            behavior[index] = item.improvement_behavior.to(self.device)
            exact_mask[index] = item.improvement_exact_mask.to(self.device)
            advantage[index] = item.improvement_advantage.to(self.device)
            bootstrap[index] = item.improvement_bootstrap.to(self.device)
            if item.improvement_base_action is not None:
                base_action[index] = item.improvement_base_action.to(
                    device=self.device, dtype=torch.long
                ).reshape(-1)[0]
        return available, behavior, exact_mask, advantage, bootstrap, base_action

    def _improvement_validation_metrics(
        self,
        transitions: Sequence[Transition],
        episodes: Sequence[Sequence[int]],
    ) -> dict[str, float]:
        if not transitions:
            return {
                "loss": float("nan"), "states": 0.0, "gain": float("nan"),
                "episodes": 0.0,
                "gain_se": float("nan"), "gain_lower": float("nan"),
                "rank": float("nan"), "top1": float("nan"),
            }
        available, behavior, exact_mask, advantage, _, base_action = (
            self._improvement_replay_arrays(transitions)
        )
        with torch.no_grad():
            outputs = self._evaluate_recurrent_sequences(
                transitions,
                episodes,
                guidance_weight=0.0,
                action_temperature=1.0,
                include_improvement=True,
            )
            order = outputs[0]
            samples = torch.empty_like(outputs[10])
            samples[order] = outputs[10]
            priors = torch.empty_like(outputs[11])
            priors[order] = outputs[11]
            validation_bootstrap = torch.ones(
                len(transitions), self.nmcc_ensemble_size,
                dtype=torch.float32, device=self.device,
            )
            loss = self._improvement_value_loss(
                samples, available, behavior, exact_mask, advantage,
                validation_bootstrap,
            )
            conservative = self._base_relative_improvement_lcb(
                samples,
                base_action.clamp_min(0),
            )
            # This is exactly the deterministic deployment controller evaluated
            # prospectively with the gate open.  Validating the correction head
            # alone can reject a useful residual (or accept a harmful one)
            # because the operational action is selected by prior + correction.
            base_level = priors.gather(1, base_action.clamp_min(0)[:, None])
            controller_score = base_level + (
                float(self.nmcc_pi_value_scale)
                * conservative
                / float(self.nmcc_pi_ranking_temperature)
            )
            eligible = (
                available
                & (exact_mask.sum(dim=-1) >= 2)
                & (exact_mask.sum(dim=-1) == (behavior > 0.0).sum(dim=-1))
                & (base_action >= 0)
            )
            gains: list[float] = []
            gain_by_row: dict[int, float] = {}
            top1: list[float] = []
            for row in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
                if not bool(exact_mask[row, base_action[row]] > 0.5):
                    continue
                predicted = torch.where(
                    exact_mask[row] > 0.5,
                    controller_score[row],
                    torch.full_like(controller_score[row], float("-inf")),
                )
                chosen = int(predicted.argmax().item())
                exact_best = int(torch.where(
                    exact_mask[row] > 0.5,
                    advantage[row],
                    torch.full_like(advantage[row], float("-inf")),
                ).argmax().item())
                gain_value = float(
                    advantage[row, chosen].item()
                    - advantage[row, base_action[row]].item()
                )
                gains.append(gain_value)
                gain_by_row[row] = gain_value
                top1.append(float(chosen == exact_best))
        episode_gains = [
            float(np.mean([gain_by_row[row] for row in episode if row in gain_by_row]))
            for episode in episodes
            if any(row in gain_by_row for row in episode)
        ]
        state_count = len(gains)
        episode_count = len(episode_gains)
        gain = float(np.mean(episode_gains)) if episode_gains else float("nan")
        gain_se = (
            float(np.std(episode_gains, ddof=1) / np.sqrt(episode_count))
            if episode_count > 1 else float("inf")
        )
        gain_lower = (
            gain - float(self.nmcc_pi_validation_gain_z) * gain_se
            if np.isfinite(gain) and np.isfinite(gain_se) else float("nan")
        )
        rows = available.detach().cpu().numpy()
        prediction = controller_score.detach().cpu().numpy()
        rank = NPI.within_state_rank_agreement(
            prediction[rows],
            advantage.detach().cpu().numpy()[rows],
            exact_mask.detach().cpu().numpy()[rows],
        )
        return {
            "loss": float(loss.item()),
            "states": float(state_count),
            "episodes": float(episode_count),
            "gain": gain,
            "gain_se": gain_se,
            "gain_lower": gain_lower,
            "rank": float(rank),
            "top1": float(np.mean(top1)) if top1 else float("nan"),
        }

    def _fit_improvement_replay(self) -> dict[str, float]:
        """Refit the conservative intervention-value model on all retained labels."""
        training_records, validation_records, gate_records = (
            self._split_improvement_replay()
        )
        training, training_episodes = self._flatten_replay_episodes(training_records)
        validation, validation_episodes = self._flatten_replay_episodes(validation_records)
        gate, gate_episodes = self._flatten_replay_episodes(gate_records)
        if not training:
            return {
                "nmcc_pi_replay_episodes": float(len(self.improvement_replay)),
                "nmcc_pi_replay_training_states": 0.0,
                "nmcc_pi_validation_states": 0.0,
                "nmcc_pi_validation_episodes": 0.0,
                "nmcc_pi_validation_gain": float("nan"),
                "nmcc_pi_validation_gain_lower": float("nan"),
                "nmcc_pi_validation_rank": float("nan"),
                "nmcc_pi_validation_top1": float("nan"),
                "nmcc_pi_replay_epochs_completed": 0.0,
            }
        if self.nmcc_pi_replay_refit:
            self.policy.reset_improvement_model(seed=29_311)
        # System identification owns the relational encoder and LSTM.  The
        # fitted controller consumes those full GNN features but must not
        # rewrite them from a small exact-branch dataset; doing so destroyed
        # natural-dynamics generalization in the v25 review.  Refit only the
        # dedicated wide-and-deep intervention model here.
        fit_named = list(self.policy.improvement_named_parameters())
        seen = set()
        unique_fit_named = []
        for name, parameter in fit_named:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            unique_fit_named.append((name, parameter))
        fit_named = unique_fit_named
        owned = list(self.policy.named_parameters())
        original_trainable = {name: parameter.requires_grad for name, parameter in owned}
        fit_ids = {id(parameter) for _, parameter in fit_named}
        for _, parameter in owned:
            parameter.requires_grad_(id(parameter) in fit_ids)
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in fit_named],
            # Every fitted-policy iteration starts from a fresh candidate-value
            # model and an expanded dataset.  Reusing the late world-model
            # cosine floor here would increasingly underfit those refits.
            lr=float(self.critic_lr),
            weight_decay=1e-4,
        )
        available, behavior, exact_mask, advantage, bootstrap, _ = (
            self._improvement_replay_arrays(training)
        )
        best_metric = float("inf")
        best_state = None
        stale_epochs = 0
        epochs_completed = 0
        try:
            for _ in range(int(self.nmcc_pi_replay_epochs)):
                epochs_completed += 1
                for episode_batch in self._iter_episode_minibatches(training_episodes):
                    outputs = self._evaluate_recurrent_sequences(
                        training,
                        episode_batch,
                        guidance_weight=0.0,
                        action_temperature=1.0,
                        include_improvement=True,
                    )
                    selected = outputs[0]
                    value_loss = self._improvement_value_loss(
                        outputs[10], available[selected], behavior[selected],
                        exact_mask[selected], advantage[selected], bootstrap[selected],
                    )
                    pairwise_loss = self._improvement_pairwise_loss(
                        outputs[10], available[selected], exact_mask[selected],
                        advantage[selected], bootstrap[selected],
                    )
                    loss = (
                        value_loss
                        + float(self.nmcc_pi_rank_margin_coef) * pairwise_loss
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Intervention-value replay loss became non-finite")
                    if not loss.requires_grad:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        [parameter for _, parameter in fit_named], 0.5
                    )
                    optimizer.step()
                validation_metrics = self._improvement_validation_metrics(
                    validation, validation_episodes
                )
                metric = validation_metrics["loss"]
                if not np.isfinite(metric):
                    metric = float(loss.detach().item())
                if metric < best_metric - 1e-7:
                    best_metric = metric
                    best_state = {
                        name: parameter.detach().clone()
                        for name, parameter in fit_named
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(self.nmcc_pi_early_stopping_patience):
                        break
            if best_state is not None:
                with torch.no_grad():
                    for name, parameter in fit_named:
                        parameter.copy_(best_state[name])
            metrics = self._improvement_validation_metrics(gate, gate_episodes)
            training_metrics = self._improvement_validation_metrics(
                training, training_episodes
            )
        finally:
            for name, parameter in owned:
                parameter.requires_grad_(original_trainable[name])
        if (
            metrics["states"] >= float(self.nmcc_pi_min_validation_states)
            and np.isfinite(metrics["gain_lower"])
        ):
            self.improvement_gate_history.append(float(metrics["gain_lower"]))
        return {
            "nmcc_pi_replay_episodes": float(len(self.improvement_replay)),
            "nmcc_pi_replay_training_states": float(sum(
                item.improvement_advantage is not None for item in training
            )),
            "nmcc_pi_replay_training_loss": training_metrics["loss"],
            "nmcc_pi_replay_training_rank": training_metrics["rank"],
            "nmcc_pi_replay_training_top1": training_metrics["top1"],
            "nmcc_pi_validation_states": metrics["states"],
            "nmcc_pi_validation_episodes": metrics["episodes"],
            "nmcc_pi_validation_loss": metrics["loss"],
            "nmcc_pi_validation_gain": metrics["gain"],
            "nmcc_pi_validation_gain_se": metrics["gain_se"],
            "nmcc_pi_validation_gain_lower": metrics["gain_lower"],
            "nmcc_pi_validation_rank": metrics["rank"],
            "nmcc_pi_validation_top1": metrics["top1"],
            "nmcc_pi_replay_epochs_completed": float(epochs_completed),
        }

    def _optimize_policy(self, transitions: list[Transition]) -> Dict[str, float]:
        """Fit critic/world models, then attempt a transactional recurrent PPO step."""
        if not transitions:
            raise RuntimeError("PPO rollout batch is empty")
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise RuntimeError("Training requires both optimizer partitions")
        training_phase = self._current_nmcc_training_phase()
        actor_enabled = (
            training_phase in {"controller_warmup", "joint_optimization"}
            and self.nmcc_pi_actor_objective != "value_lcb"
        )
        causal_model_enabled = training_phase != "natural_pretrain"
        critic_value_enabled = training_phase in {"controller_warmup", "joint_optimization"}
        episodes = self._group_episode_indices(transitions)
        if self.nmcc_policy_improvement:
            self._append_improvement_replay(transitions, episodes)
        total = len(transitions)

        rewards = torch.cat([item.reward.reshape(1) for item in transitions])
        component_rewards = torch.stack([item.reward_components for item in transitions])
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

        nmcc_available = torch.as_tensor(
            [
                item.natural_outcome_target is not None
                and item.causal_outcome_target is not None
                and item.counterfactual_components is not None
                for item in transitions
            ],
            dtype=torch.bool,
            device=self.device,
        )
        natural_targets = torch.zeros(
            total,
            len(NMCC_OUTCOME_NAMES),
            dtype=torch.float32,
            device=self.device,
        )
        causal_targets = torch.zeros_like(natural_targets)
        for index, transition in enumerate(transitions):
            if bool(nmcc_available[index]):
                natural_targets[index] = transition.natural_outcome_target
                causal_targets[index] = transition.causal_outcome_target
        pi_mode = bool(self.nmcc_policy_improvement)
        action_count = self.num_candidate_actions
        improvement_available = torch.as_tensor(
            [item.improvement_target is not None for item in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        improvement_targets = torch.zeros(
            total, action_count, dtype=torch.float32, device=self.device
        )
        improvement_behavior = torch.zeros_like(improvement_targets)
        improvement_exact_mask = torch.zeros_like(improvement_targets)
        improvement_advantage = torch.zeros_like(improvement_targets)
        improvement_bootstrap = torch.zeros(
            total, int(self.nmcc_ensemble_size), dtype=torch.float32, device=self.device
        )
        improvement_natural_outcomes = torch.zeros(
            total,
            len(NMCC_OUTCOME_NAMES),
            dtype=torch.float32,
            device=self.device,
        )
        improvement_outcome_effects = torch.zeros(
            total,
            action_count,
            len(NMCC_OUTCOME_NAMES),
            dtype=torch.float32,
            device=self.device,
        )
        improvement_outcome_available = torch.as_tensor(
            [
                item.improvement_natural_outcome is not None
                and item.improvement_outcome_effect is not None
                for item in transitions
            ],
            dtype=torch.bool,
            device=self.device,
        )
        for index, transition in enumerate(transitions):
            if not bool(improvement_available[index]):
                continue
            improvement_targets[index] = transition.improvement_target
            improvement_behavior[index] = transition.improvement_behavior
            improvement_exact_mask[index] = transition.improvement_exact_mask
            improvement_advantage[index] = transition.improvement_advantage
            improvement_bootstrap[index] = transition.improvement_bootstrap
            if bool(improvement_outcome_available[index]):
                improvement_natural_outcomes[index] = (
                    transition.improvement_natural_outcome
                )
                improvement_outcome_effects[index] = (
                    transition.improvement_outcome_effect
                )
        natural_targets[improvement_outcome_available] = (
            improvement_natural_outcomes[improvement_outcome_available]
        )
        nmcc_available = nmcc_available | improvement_outcome_available
        if training_phase in {"natural_pretrain", "causal_pretrain"} and not bool(
            nmcc_available.all()
        ):
            raise RuntimeError(
                f"{training_phase} requires exact paired physical targets for every transition"
            )
        improvement_branch_best = self._branch_best_actions(
            improvement_advantage, improvement_exact_mask
        )
        if pi_mode and bool(improvement_available.any()):
            if torch.any(
                (improvement_targets > 0) & ~action_masks.to(torch.bool)
            ):
                raise RuntimeError("NMCC-PI target places mass on an infeasible cell")
            behavior_top1 = (
                self._safe_masked_logits(
                    torch.log(improvement_behavior.clamp_min(1e-30)), action_masks
                ).argmax(dim=-1)
                == improvement_branch_best
            )[improvement_available].to(torch.float32)
            improvement_top1_before = float(behavior_top1.mean().item())
        else:
            improvement_top1_before = float("nan")

        with torch.no_grad():
            component_actor_returns, component_td_targets = self._component_credit_targets(
                component_rewards,
                component_values,
                dones,
                durations,
            )
            actor_returns = component_actor_returns.sum(dim=-1)
            td_targets = component_td_targets.sum(dim=-1)
            baseline_keys = self._regime_position_keys(episodes, total)
            advantages, baseline_coverage = self._normalize_with_lagged_baseline(
                actor_returns,
                baseline_keys,
                self.actor_return_baselines,
            )
            advantages, counterfactual_diagnostics = self._blend_counterfactual_advantages(
                advantages,
                transitions,
                baseline_keys,
            )
            raw_actor_return_sd = float(actor_returns.std(unbiased=False).item())
            counterfactual_diagnostics["mc_advantage_sd"] = raw_actor_return_sd
            # Retain the historical column as an explicit compatibility alias.
            counterfactual_diagnostics["gae_advantage_sd"] = raw_actor_return_sd

        update_entropy_coef = self._current_entropy_coef()
        update_temperature = self._current_action_temperature()
        update_guidance_weight = self._current_nmcc_guidance_weight()
        update_teacher_coef = self._current_nmcc_teacher_coef()
        self._apply_learning_rate_schedule()
        actor_lr_before = float(self.actor_optimizer.param_groups[0]["lr"])
        critic_lr_before = float(self.critic_optimizer.param_groups[0]["lr"])

        value_losses: list[float] = []
        component_loss_records: list[np.ndarray] = []
        critic_gradient_norms: list[float] = []
        natural_losses: list[float] = []
        causal_losses: list[float] = []
        dueling_losses: list[float] = []
        causal_uncertainties: list[float] = []
        improvement_losses: list[float] = []
        improvement_gate_rank = float("nan")
        improvement_rows = (
            pi_mode
            and self.nmcc_pi_actor_objective != "value_lcb"
            and bool(
            (improvement_available & (improvement_exact_mask.sum(dim=-1) >= 2)).any()
            )
        )
        if improvement_rows:
            # Held-out ranking gate: score this rollout's exact branches with
            # the ensemble *before* it trains on them.
            with torch.no_grad():
                gate_outputs = self._evaluate_recurrent_sequences(
                    transitions,
                    episodes,
                    guidance_weight=update_guidance_weight,
                    action_temperature=update_temperature,
                    include_improvement=True,
                )
                gate_order = gate_outputs[0]
                gate_samples = torch.empty_like(gate_outputs[10])
                gate_samples[gate_order] = gate_outputs[10]
                gate_prediction = gate_samples.mean(dim=1)
            rows = improvement_available.cpu().numpy()
            improvement_gate_rank = NPI.within_state_rank_agreement(
                gate_prediction.cpu().numpy()[rows],
                improvement_advantage.cpu().numpy()[rows],
                improvement_exact_mask.cpu().numpy()[rows],
            )
            if np.isfinite(improvement_gate_rank):
                self.improvement_gate_history.append(float(improvement_gate_rank))
        critic_epochs_completed = 0
        shared_representation = self.representation_mode == "shared_phasic"
        representation_drift_kl = 0.0
        representation_rollback = False
        clone_kls: list[float] = []
        critic_representation_gradient_norms: list[float] = []
        critic_head_gradient_norms: list[float] = []
        reference_log_probabilities = None
        if shared_representation:
            # The policy the critic pass must not move: the one the rollout was
            # collected with (the replay reproduces it exactly).
            reference_log_probabilities = self._policy_log_probabilities(
                transitions,
                episodes,
                action_masks,
                guidance_weight=update_guidance_weight,
                action_temperature=update_temperature,
            )
        self._set_optimizer_partition_trainable(actor=False, critic=True)
        critic_parameters = [parameter for _, parameter in self.critic_named_parameters]
        critic_snapshot: Dict[str, torch.Tensor] = {}
        critic_optimizer_snapshot: dict = {}
        for _ in range(self.critic_epochs):
            if shared_representation:
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
                    guidance_weight=update_guidance_weight,
                    action_temperature=update_temperature,
                    include_improvement=improvement_rows,
                )
                (
                    selected,
                    _,
                    _,
                    components_now,
                    _,
                    natural_now,
                    causal_samples_now,
                    _,
                    causal_component_std_now,
                    _,
                ) = critic_outputs[:10]
                zero = components_now.new_zeros(())
                improvement_loss = zero
                if improvement_rows and causal_model_enabled:
                    improvement_loss = self._improvement_value_loss(
                        critic_outputs[10],
                        improvement_available[selected],
                        improvement_behavior[selected],
                        improvement_exact_mask[selected],
                        improvement_advantage[selected],
                        improvement_bootstrap[selected],
                    )
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
                causal_loss = zero
                dueling_loss = zero
                uncertainty_mean = zero
                available_batch = nmcc_available[selected]
                if bool(available_batch.any()):
                    predicted_natural = natural_now[available_batch]
                    target_natural = natural_targets[selected][available_batch]
                    natural_loss = F.smooth_l1_loss(predicted_natural, target_natural)
                pi_batch = improvement_outcome_available[selected]
                if causal_model_enabled and bool(pi_batch.any()):
                    exact = improvement_exact_mask[selected][pi_batch]
                    samples = causal_samples_now[pi_batch]
                    targets = improvement_outcome_effects[selected][pi_batch]
                    element_loss = F.smooth_l1_loss(
                        samples,
                        targets.unsqueeze(1).expand_as(samples),
                        reduction="none",
                    )
                    expanded_mask = exact[:, None, :, None]
                    denominator = (
                        exact.sum()
                        * samples.shape[1]
                        * samples.shape[-1]
                    ).clamp_min(1.0)
                    causal_loss = (element_loss * expanded_mask).sum() / denominator
                    predicted_effect = samples.mean(dim=1)
                    predicted_total = (
                        natural_now[pi_batch].unsqueeze(1) + predicted_effect
                    )
                    target_total = (
                        improvement_natural_outcomes[selected][pi_batch].unsqueeze(1)
                        + targets
                    )
                    component_error = F.smooth_l1_loss(
                        self.policy._outcomes_to_components(predicted_total),
                        self.policy._outcomes_to_components(target_total),
                        reduction="none",
                    )
                    dueling_loss = (
                        component_error * exact.unsqueeze(-1)
                    ).sum() / (exact.sum() * component_error.shape[-1]).clamp_min(1.0)
                    uncertainty_mean = (
                        causal_component_std_now[pi_batch]
                        * exact.unsqueeze(-1)
                    ).sum() / (
                        exact.sum().clamp_min(1.0)
                        * causal_component_std_now.shape[-1]
                    )
                legacy_batch = available_batch & ~pi_batch
                if causal_model_enabled and bool(legacy_batch.any()):
                    target_causal = causal_targets[selected][legacy_batch]
                    chosen_actions = actions[selected][legacy_batch]
                    row = torch.arange(chosen_actions.numel(), device=self.device)
                    selected_causal_samples = causal_samples_now[legacy_batch][
                        row, :, chosen_actions, :
                    ]
                    predicted_causal = selected_causal_samples.mean(dim=1)
                    legacy_causal = F.smooth_l1_loss(
                        selected_causal_samples,
                        target_causal.unsqueeze(1).expand_as(selected_causal_samples),
                    )
                    legacy_natural = natural_targets[selected][legacy_batch]
                    legacy_dueling = F.smooth_l1_loss(
                        self.policy._outcomes_to_components(natural_now[legacy_batch])
                        + self.policy._outcomes_to_components(predicted_causal),
                        self.policy._outcomes_to_components(legacy_natural + target_causal),
                    )
                    causal_loss = causal_loss + legacy_causal
                    dueling_loss = dueling_loss + legacy_dueling
                critic_loss = self.nmcc_natural_loss_coef * natural_loss
                if causal_model_enabled:
                    critic_loss = (
                        critic_loss
                        + self.nmcc_causal_loss_coef * causal_loss
                        + self.nmcc_dueling_loss_coef * dueling_loss
                    )
                if critic_value_enabled:
                    critic_loss = critic_loss + value_loss
                if causal_model_enabled:
                    critic_loss = (
                        critic_loss
                        + self.nmcc_pi_value_loss_coef * improvement_loss
                    )
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
                causal_losses.append(float(causal_loss.detach().item()))
                dueling_losses.append(float(dueling_loss.detach().item()))
                causal_uncertainties.append(float(uncertainty_mean.detach().item()))
                improvement_losses.append(float(improvement_loss.detach().item()))
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
                            guidance_weight=update_guidance_weight,
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

        policy_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []
        actor_gradient_norms: list[float] = []
        residual_rms_values: list[float] = []
        teacher_losses: list[float] = []
        attempted_kls: list[float] = []
        actor_epochs_completed = 0
        actor_steps_accepted = 0
        kl_rollback = False
        actor_rollbacks = 0
        line_search_fractions: list[float] = []
        trust_region_reached = False
        actor_representation_gradient_norms: list[float] = []
        actor_head_gradient_norms: list[float] = []
        self._set_optimizer_partition_trainable(actor=True, critic=False)
        actor_parameters = [parameter for _, parameter in self.actor_named_parameters]
        pi_actor = (
            pi_mode
            and self.nmcc_pi_actor_objective != "value_lcb"
            and bool(improvement_available.any())
        )
        trust_region_cap = self.nmcc_pi_kl_cap if pi_actor else self.target_kl
        epoch_budget = self.nmcc_pi_actor_epochs if pi_actor else self.actor_epochs
        improvement_requested_kl = 0.0
        improvement_fit_kl_before = float("nan")
        improvement_fit_kl_after = float("nan")
        improvement_top1_after = float("nan")
        improvement_converged = False
        improvement_reverse_kl = float("nan")

        def policy_statistics() -> Dict[str, float]:
            return self._improvement_policy_statistics(
                transitions,
                episodes,
                action_masks,
                actions,
                old_log_probabilities,
                improvement_available,
                improvement_targets,
                improvement_behavior,
                improvement_branch_best,
                improvement_exact_mask,
                guidance_weight=update_guidance_weight,
                action_temperature=update_temperature,
            )

        if actor_enabled and pi_actor:
            # The scheduled rate is set once per rollout. Trust-region line
            # search may shorten a step, but never mutates the future schedule.
            requested = torch.where(
                improvement_targets > 0,
                improvement_targets
                * (
                    torch.log(improvement_targets.clamp_min(1e-30))
                    - torch.log(improvement_behavior.clamp_min(1e-30))
                ),
                torch.zeros_like(improvement_targets),
            ).sum(dim=-1)
            improvement_requested_kl = float(requested[improvement_available].mean().item())
            improvement_fit_kl_before = policy_statistics()["target_kl"]
        if actor_enabled:
            for _ in range(epoch_budget):
                parameter_snapshot = {
                    name: parameter.detach().clone()
                    for name, parameter in self.actor_named_parameters
                }
                optimizer_snapshot = copy.deepcopy(self.actor_optimizer.state_dict())
                epoch_policy_losses: list[float] = []
                epoch_entropies: list[float] = []
                epoch_clip_fractions: list[float] = []
                epoch_gradient_norms: list[float] = []
                epoch_residual_rms: list[float] = []
                epoch_teacher_losses: list[float] = []
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
                        _,
                        teacher_logits_now,
                    ) = self._evaluate_recurrent_sequences(
                        transitions,
                        episode_batch,
                        guidance_weight=update_guidance_weight,
                        action_temperature=update_temperature,
                    )
                    selected_masks = action_masks[selected]
                    logits = self._safe_masked_logits(logits, selected_masks)
                    distribution = torch.distributions.Categorical(logits=logits)
                    new_log_probabilities = distribution.log_prob(actions[selected])
                    entropy = self._normalized_categorical_entropy(
                        distribution,
                        selected_masks,
                    ).mean()
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
                    surrogate = -torch.minimum(unclipped, clipped)
                    residual_penalty = learned_residual.square().mean()
                    teacher_loss = surrogate.new_zeros(())
                    rank_margin_loss = surrogate.new_zeros(())
                    if pi_actor:
                        target_rows = improvement_available[selected]
                        target_batch = improvement_targets[selected]
                        if self.nmcc_pi_actor_objective == "score_ranking":
                            # Listwise supervision compares only candidates
                            # evaluated from this same simulator snapshot.
                            exact_batch = improvement_exact_mask[selected].to(torch.bool)
                            fit = torch.zeros_like(surrogate)
                            if bool(target_rows.any()):
                                ranked_logits = self._safe_masked_logits(
                                    logits[target_rows], exact_batch[target_rows]
                                )
                                log_policy = F.log_softmax(ranked_logits, dim=-1)
                                ranked_target = target_batch[target_rows]
                                fit[target_rows] = torch.where(
                                    ranked_target > 0,
                                    ranked_target
                                    * (
                                        torch.log(ranked_target.clamp_min(1e-30))
                                        - log_policy
                                    ),
                                    torch.zeros_like(ranked_target),
                                ).sum(dim=-1)
                                best = improvement_branch_best[selected][target_rows]
                                best_score = logits[target_rows].gather(
                                    1, best.unsqueeze(1)
                                )
                                rivals = exact_batch[target_rows].clone()
                                rivals.scatter_(1, best.unsqueeze(1), False)
                                margin_error = F.relu(
                                    self.nmcc_pi_rank_margin
                                    - (best_score - logits[target_rows])
                                )
                                rank_margin_loss = (
                                    margin_error * rivals.to(margin_error.dtype)
                                ).sum() / rivals.sum().clamp_min(1)
                        else:
                            log_policy = F.log_softmax(logits, dim=-1)
                            fit = torch.where(
                                target_batch > 0,
                                target_batch
                                * (
                                    torch.log(target_batch.clamp_min(1e-30))
                                    - log_policy
                                ),
                                torch.zeros_like(target_batch),
                            ).sum(dim=-1)
                        per_decision = torch.where(target_rows, fit, surrogate)
                        policy_loss = per_decision.mean()
                    else:
                        policy_loss = surrogate.mean()
                    available_batch = nmcc_available[selected]
                    if not pi_actor and bool(available_batch.any()):
                        teacher_mask = selected_masks[available_batch]
                        teacher_distribution = torch.distributions.Categorical(
                            logits=self._safe_masked_logits(
                                teacher_logits_now[available_batch].detach(),
                                teacher_mask,
                            )
                        )
                        actor_log_probabilities = F.log_softmax(
                            self._safe_masked_logits(logits[available_batch], teacher_mask),
                            dim=-1,
                        )
                        teacher_loss = -(
                            teacher_distribution.probs * actor_log_probabilities
                        ).sum(dim=-1).mean()
                    if pi_actor:
                        # Exact paired supervision determines the direction;
                        # exploration belongs to data collection, not this fit.
                        actor_loss = (
                            policy_loss
                            + self.nmcc_pi_rank_margin_coef * rank_margin_loss
                            + self.residual_penalty_coef * residual_penalty
                        )
                    else:
                        actor_loss = (
                            policy_loss
                            - update_entropy_coef * entropy
                            + self.residual_penalty_coef * residual_penalty
                            + update_teacher_coef * teacher_loss
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
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1.0) > self.clip_eps).to(torch.float32)
                    )
                    epoch_policy_losses.append(float(policy_loss.detach().item()))
                    epoch_entropies.append(float(entropy.detach().item()))
                    epoch_clip_fractions.append(float(clip_fraction.detach().item()))
                    epoch_gradient_norms.append(float(gradient_norm.item()))
                    epoch_residual_rms.append(
                        float(torch.sqrt(residual_penalty.detach()).item())
                    )
                    epoch_teacher_losses.append(float(teacher_loss.detach().item()))
                if pi_actor:
                    epoch_statistics = policy_statistics()
                    attempted_kl = epoch_statistics["kl"]
                else:
                    epoch_statistics = None
                    attempted_kl = self._full_rollout_policy_kl(
                        transitions,
                        episodes,
                        action_masks,
                        actions,
                        old_log_probabilities,
                        guidance_weight=update_guidance_weight,
                        action_temperature=update_temperature,
                    )
                attempted_kls.append(attempted_kl)
                if pi_actor and attempted_kl > trust_region_cap:
                    # Supervised score fitting can still overshoot the policy
                    # trust region. Keep the direction and line-search the
                    # largest safe fraction of the parameter step.
                    step_fraction = self._line_search_to_trust_region(
                        parameter_snapshot, trust_region_cap, policy_statistics
                    )
                    line_search_fractions.append(step_fraction)
                    kl_rollback = step_fraction <= 0.0
                    actor_rollbacks += int(step_fraction <= 0.0)
                    if step_fraction > 0.0:
                        actor_epochs_completed += 1
                        actor_steps_accepted += epoch_steps
                        policy_losses.extend(epoch_policy_losses)
                        entropies.extend(epoch_entropies)
                        clip_fractions.extend(epoch_clip_fractions)
                        actor_gradient_norms.extend(epoch_gradient_norms)
                        residual_rms_values.extend(epoch_residual_rms)
                        teacher_losses.extend(epoch_teacher_losses)
                    trust_region_reached = True
                    break
                if attempted_kl > trust_region_cap:
                    with torch.no_grad():
                        for name, parameter in self.actor_named_parameters:
                            parameter.copy_(parameter_snapshot[name])
                    self.actor_optimizer.load_state_dict(optimizer_snapshot)
                    reduced_lr = max(
                        self.actor_lr * DEFAULT_MINIMUM_LR_FRACTION,
                        float(self.actor_optimizer.param_groups[0]["lr"])
                        * DEFAULT_KL_LR_REDUCTION,
                    )
                    for parameter_group in self.actor_optimizer.param_groups:
                        parameter_group["lr"] = reduced_lr
                    kl_rollback = True
                    actor_rollbacks += 1
                    # PPO's surrogate is only trusted near pi_old: stop at the
                    # first violation.
                    break
                actor_epochs_completed += 1
                actor_steps_accepted += epoch_steps
                policy_losses.extend(epoch_policy_losses)
                entropies.extend(epoch_entropies)
                clip_fractions.extend(epoch_clip_fractions)
                actor_gradient_norms.extend(epoch_gradient_norms)
                residual_rms_values.extend(epoch_residual_rms)
                teacher_losses.extend(epoch_teacher_losses)
                if epoch_statistics is not None and (
                    epoch_statistics["target_kl"]
                    <= self.nmcc_pi_fit_tolerance * improvement_requested_kl
                ):
                    improvement_converged = True
                    break

        if actor_enabled and pi_actor:
            final_statistics = policy_statistics()
            retained_kl = final_statistics["kl"]
            improvement_reverse_kl = final_statistics["reverse_kl"]
            improvement_fit_kl_after = final_statistics["target_kl"]
            improvement_top1_after = final_statistics["top1"]
        elif actor_enabled:
            retained_kl = self._full_rollout_policy_kl(
                transitions,
                episodes,
                action_masks,
                actions,
                old_log_probabilities,
                guidance_weight=update_guidance_weight,
                action_temperature=update_temperature,
            )
        else:
            retained_kl = 0.0
        actor_update_accepted = actor_steps_accepted > 0
        if actor_update_accepted:
            self.actor_optimizer_updates += actor_steps_accepted
            self.actor_rollout_updates += 1
            self.optimizer_updates += actor_steps_accepted
            if not pi_actor and not kl_rollback and retained_kl < 0.5 * trust_region_cap:
                grown_lr = min(
                    self.actor_lr,
                    float(self.actor_optimizer.param_groups[0]["lr"])
                    * DEFAULT_KL_LR_GROWTH,
                )
                for parameter_group in self.actor_optimizer.param_groups:
                    parameter_group["lr"] = grown_lr
        self._set_optimizer_partition_trainable(actor=True, critic=True)

        # Only now may this rollout influence the baseline used by future
        # actor updates. This ordering is the causal anti-censoring contract.
        self._update_lagged_baseline(
            self.actor_return_baselines,
            baseline_keys,
            actor_returns,
        )
        cf_values = torch.full_like(actor_returns, float("nan"))
        for index, transition in enumerate(transitions):
            if transition.counterfactual_advantage is not None:
                cf_values[index] = transition.counterfactual_advantage.reshape(-1)[0]
        cf_available = torch.isfinite(cf_values)
        if bool(cf_available.any()):
            self._update_lagged_baseline(
                self.counterfactual_return_baselines,
                [baseline_keys[index] for index in torch.nonzero(cf_available).flatten().tolist()],
                cf_values[cf_available],
            )

        replay_diagnostics: dict[str, float] = {}
        if (
            self.nmcc_policy_improvement
            and self.nmcc_pi_actor_objective == "value_lcb"
            and causal_model_enabled
        ):
            replay_diagnostics = self._fit_improvement_replay()

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
            **{name: 0.0 for name in self.NMCC_PI_DIAGNOSTIC_NAMES},
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
            "update_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "attempted_kl": float(max(attempted_kls)) if attempted_kls else 0.0,
            "approximate_kl": retained_kl,
            "kl_rollback": float(kl_rollback),
            "actor_update_accepted": float(actor_update_accepted),
            "actor_update_rejected": float(kl_rollback),
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "gradient_norm": actor_gradient_norm or critic_gradient_norm,
            "actor_gradient_norm": actor_gradient_norm,
            "critic_gradient_norm": critic_gradient_norm,
            "explained_variance": self._explained_variance(td_targets, values),
            "actor_mc_return_sd": raw_actor_return_sd,
            "actor_advantage_sd": float(advantages.std(unbiased=False).item()),
            "actor_baseline_coverage": baseline_coverage,
            "critic_td_error_sd": float(
                (td_targets - values).std(unbiased=False).item()
            ),
            "residual_rms": float(np.mean(residual_rms_values)) if residual_rms_values else 0.0,
            "learning_rate_before_update": actor_lr_before,
            "learning_rate": actor_lr_after,
            "actor_learning_rate": actor_lr_after,
            "critic_learning_rate_before_update": critic_lr_before,
            "critic_learning_rate": critic_lr_after,
            "actor_rollout_updates": float(self.actor_rollout_updates),
            "critic_optimizer_updates": float(self.critic_optimizer_updates),
            "entropy_coefficient": float(update_entropy_coef),
            "action_temperature": float(update_temperature),
            "exploration_rate": float(self._current_exploration_rate()),
            "nmcc_training_phase_index": float(NMCC_TRAINING_PHASES.index(training_phase)),
            "nmcc_actor_enabled": float(actor_enabled),
            "nmcc_causal_model_enabled": float(causal_model_enabled),
            "nmcc_effective_counterfactual_weight": float(
                self.nmcc_joint_counterfactual_weight
                if training_phase == "joint_optimization"
                else self.counterfactual_weight
            ),
            "nmcc_rollout_updates_completed": float(self.rollout_updates_completed),
            "nmcc_actor_optimizer_updates": float(self.actor_optimizer_updates),
            "nmcc_guidance_weight": float(update_guidance_weight),
            "nmcc_teacher_coefficient": float(update_teacher_coef),
            "nmcc_natural_loss": float(np.mean(natural_losses)) if natural_losses else 0.0,
            "nmcc_causal_loss": float(np.mean(causal_losses)) if causal_losses else 0.0,
            "nmcc_dueling_loss": float(np.mean(dueling_losses)) if dueling_losses else 0.0,
            "nmcc_teacher_loss": float(np.mean(teacher_losses)) if teacher_losses else 0.0,
            "nmcc_causal_uncertainty": (
                float(np.mean(causal_uncertainties)) if causal_uncertainties else 0.0
            ),
            **{
                f"nmcc_{key}": float(value)
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
                actor_rollbacks=actor_rollbacks,
                line_search_fractions=line_search_fractions,
                trust_region_reached=trust_region_reached,
                reverse_kl=improvement_reverse_kl,
            ),
            **self._improvement_diagnostics(
                coverage=float(improvement_available.to(torch.float32).mean().item()),
                requested_kl=improvement_requested_kl,
                fit_kl_before=improvement_fit_kl_before,
                fit_kl_after=improvement_fit_kl_after,
                top1_before=improvement_top1_before,
                top1_after=improvement_top1_after,
                gate_rank=improvement_gate_rank,
                converged=improvement_converged,
                value_losses=improvement_losses,
            ),
            **replay_diagnostics,
        }
        for component_index, component_name in enumerate(REWARD_COMPONENT_NAMES):
            diagnostics[f"value_loss_{component_name}"] = float(
                component_loss_mean[component_index]
            )
            diagnostics[f"explained_variance_{component_name}"] = self._explained_variance(
                component_td_targets[:, component_index],
                component_values[:, component_index],
            )
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

    def _append_training_diagnostics(self, diagnostics: Dict[str, float]) -> None:
        directory = os.path.dirname(os.path.abspath(self.diagnostics_path))
        os.makedirs(directory, exist_ok=True)
        columns: tuple[str, ...] = (
            "episode",
            "optimizer_updates",
            "nmcc_rollout_updates_completed",
            "nmcc_actor_optimizer_updates",
            "actor_rollout_updates",
            "critic_optimizer_updates",
            "episode_return",
            "safe_completion_reward",
            "casualty_penalty",
            "evacuation_time_penalty",
            "hazard_exposure_penalty",
            "risk_time_penalty",
            "shelter_service_reward",
            "risk_weighted_person_time",
            "active_person_time",
            "hazard_exposure_person_time",
            "normalized_risk_weighted_person_time",
            "decisions",
            "training_episode_has_decision",
            "observation_frames_seen",
            "post_action_objective_return",
            "reward_accounting_gap",
            "post_action_safe_completion_reward",
            "post_action_casualty_penalty",
            "post_action_evacuation_time_penalty",
            "post_action_hazard_exposure_penalty",
            "reward_accounting_gap_safe_completion",
            "reward_accounting_gap_casualty",
            "reward_accounting_gap_evacuation_time",
            "reward_accounting_gap_hazard_exposure",
            "heuristic_agreement_rate",
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
            "epochs_completed",
            "critic_epochs_completed",
            "actor_epochs_completed",
            "policy_loss",
            "value_loss",
            "entropy",
            "update_entropy",
            "attempted_kl",
            "approximate_kl",
            "kl_rollback",
            "actor_update_accepted",
            "actor_update_rejected",
            "clip_fraction",
            "gradient_norm",
            "actor_gradient_norm",
            "critic_gradient_norm",
            "explained_variance",
            "actor_mc_return_sd",
            "actor_advantage_sd",
            "actor_baseline_coverage",
            "critic_td_error_sd",
            "value_loss_safe_completion",
            "value_loss_casualty",
            "value_loss_evacuation_time",
            "value_loss_hazard_exposure",
            "explained_variance_safe_completion",
            "explained_variance_casualty",
            "explained_variance_evacuation_time",
            "explained_variance_hazard_exposure",
            "residual_rms",
            "learning_rate_before_update",
            "learning_rate",
            "actor_learning_rate",
            "critic_learning_rate_before_update",
            "critic_learning_rate",
            "entropy_coefficient",
            "action_temperature",
            "exploration_rate",
            "nmcc_counterfactual_fraction",
            "nmcc_training_phase_index",
            "nmcc_actor_enabled",
            "nmcc_causal_model_enabled",
            "nmcc_effective_counterfactual_weight",
            "nmcc_counterfactual_advantage_mean",
            "nmcc_counterfactual_advantage_sd",
            "nmcc_mc_advantage_sd",
            "nmcc_gae_advantage_sd",
            "nmcc_guidance_weight",
            "nmcc_teacher_coefficient",
            "nmcc_natural_loss",
            "nmcc_causal_loss",
            "nmcc_dueling_loss",
            "nmcc_teacher_loss",
            "nmcc_causal_uncertainty",
            *self.NMCC_PI_DIAGNOSTIC_NAMES,
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
            row["optimizer_updates"] = self.optimizer_updates
            writer.writerow(row)
