# -*- coding: utf-8 -*-
"""Torch-free configuration contract for NMCC policy improvement (NMCC-PI)
and the learner's representation-ownership mode.

Shared by Core (defaults, casting, recorded configuration), RLBridge
(constructor defaults) and TrainingCurriculum (the learner keys a curriculum
may set and must hold constant), so that none of them has to import torch to
agree on names, types or defaults.
"""

# Must equal GNN.HEURISTIC_PRIOR_SCALE; checked by tests/test_nmcc_pi_torch.py.
DEFAULT_ACTOR_PRIOR_SCALE = 1.0

DEFAULT_NMCC_PI_EPSILON = 0.5  # E-step KL(q || pi_old) per decision state
DEFAULT_NMCC_PI_ETA_MIN = 0.03  # temperature floor ~ single-tape contrast SE
DEFAULT_NMCC_PI_KL_CAP = 0.6  # M-step mean KL(pi_new || pi_old): the E-step direction, >= epsilon
DEFAULT_NMCC_PI_TAPES = 1
DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS = 2
DEFAULT_NMCC_PI_MAX_BRANCHES = 6
DEFAULT_NMCC_PI_VALUE_SCALE = 0.05  # return units per unit of head output
DEFAULT_NMCC_PI_VALUE_LOSS_COEF = 1.0
DEFAULT_NMCC_PI_GATE_SPEARMAN = 0.6
DEFAULT_NMCC_PI_GATE_UPDATES = 3
DEFAULT_NMCC_PI_MODEL_UNCERTAINTY_PENALTY = 1.0
DEFAULT_NMCC_PI_ACTOR_EPOCHS = 32
DEFAULT_NMCC_PI_FIT_TOLERANCE = 0.2
DEFAULT_NMCC_PI_BRANCH_HORIZON = 20
DEFAULT_NMCC_PI_FULL_HORIZON_DECISIONS = 2
DEFAULT_NMCC_PI_ACTOR_OBJECTIVE = "score_ranking"
DEFAULT_NMCC_PI_RANKING_TEMPERATURE = 0.05
DEFAULT_NMCC_PI_RANK_MARGIN = 0.25
DEFAULT_NMCC_PI_RANK_MARGIN_COEF = 0.5
DEFAULT_NMCC_PI_REPLAY_MAX_EPISODES = 256
DEFAULT_NMCC_PI_REPLAY_EPOCHS = 24
DEFAULT_NMCC_PI_VALIDATION_FRACTION = 0.20
DEFAULT_NMCC_PI_EARLY_STOPPING_PATIENCE = 4
DEFAULT_NMCC_PI_MIN_VALIDATION_STATES = 8
DEFAULT_NMCC_PI_VALIDATION_GAIN_Z = 2.0
DEFAULT_NMCC_PI_REPLAY_REFIT = True
DEFAULT_EXPLORATION_RATE_START = 0.30
DEFAULT_EXPLORATION_RATE_END = 0.02
DEFAULT_LR_SCHEDULE = "cosine"
DEFAULT_LR_WARMUP_UPDATES = 1
DEFAULT_LR_DECAY_UPDATES = 32
DEFAULT_ACTOR_LR_MIN_FRACTION = 0.10
DEFAULT_CRITIC_LR_MIN_FRACTION = 0.20

# Learner representation contract (see RLBridge._optimize_policy).
REPRESENTATION_MODES = ("actor_owned", "shared_phasic")
DEFAULT_REPRESENTATION_MODE = "actor_owned"  # legacy: encoder/LSTM trained by the actor only
DEFAULT_REPRESENTATION_CLONE_COEF = 1.0  # PPG-style policy-preservation weight
DEFAULT_REPRESENTATION_KL_CAP = 0.05  # max policy drift a critic/auxiliary pass may cause

# Core attribute -> (RLBridge keyword, type, default).  One table drives the
# Core defaults, casting, constructor wiring and the recorded effective
# configuration, so an NMCC-PI setting cannot be accepted by one layer and
# silently dropped by another.
NMCC_PI_CORE_FIELDS = (
    ("nmccPolicyImprovement", "nmcc_policy_improvement", bool, False),
    ("nmccPiEpsilon", "nmcc_pi_epsilon", float, DEFAULT_NMCC_PI_EPSILON),
    ("nmccPiEtaMin", "nmcc_pi_eta_min", float, DEFAULT_NMCC_PI_ETA_MIN),
    ("nmccPiKlCap", "nmcc_pi_kl_cap", float, DEFAULT_NMCC_PI_KL_CAP),
    ("nmccPiTapes", "nmcc_pi_tapes", int, DEFAULT_NMCC_PI_TAPES),
    (
        "nmccPiExhaustiveDecisions",
        "nmcc_pi_exhaustive_decisions",
        int,
        DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS,
    ),
    ("nmccPiMaxBranches", "nmcc_pi_max_branches", int, DEFAULT_NMCC_PI_MAX_BRANCHES),
    ("nmccPiBasePolicy", "nmcc_pi_base_policy", str, "risk_reduction"),
    ("nmccPiValueScale", "nmcc_pi_value_scale", float, DEFAULT_NMCC_PI_VALUE_SCALE),
    (
        "nmccPiValueLossCoefficient",
        "nmcc_pi_value_loss_coef",
        float,
        DEFAULT_NMCC_PI_VALUE_LOSS_COEF,
    ),
    ("nmccPiModelFill", "nmcc_pi_model_fill", bool, False),
    ("nmccPiGateSpearman", "nmcc_pi_gate_spearman", float, DEFAULT_NMCC_PI_GATE_SPEARMAN),
    ("nmccPiGateUpdates", "nmcc_pi_gate_updates", int, DEFAULT_NMCC_PI_GATE_UPDATES),
    (
        "nmccPiModelUncertaintyPenalty",
        "nmcc_pi_model_uncertainty_penalty",
        float,
        DEFAULT_NMCC_PI_MODEL_UNCERTAINTY_PENALTY,
    ),
    ("nmccPiActorEpochs", "nmcc_pi_actor_epochs", int, DEFAULT_NMCC_PI_ACTOR_EPOCHS),
    ("nmccPiFitTolerance", "nmcc_pi_fit_tolerance", float, DEFAULT_NMCC_PI_FIT_TOLERANCE),
    (
        "nmccPiBranchHorizon",
        "nmcc_pi_branch_horizon",
        int,
        DEFAULT_NMCC_PI_BRANCH_HORIZON,
    ),
    (
        "nmccPiFullHorizonDecisions",
        "nmcc_pi_full_horizon_decisions",
        int,
        DEFAULT_NMCC_PI_FULL_HORIZON_DECISIONS,
    ),
    (
        "nmccPiActorObjective",
        "nmcc_pi_actor_objective",
        str,
        DEFAULT_NMCC_PI_ACTOR_OBJECTIVE,
    ),
    (
        "nmccPiRankingTemperature",
        "nmcc_pi_ranking_temperature",
        float,
        DEFAULT_NMCC_PI_RANKING_TEMPERATURE,
    ),
    ("nmccPiRankMargin", "nmcc_pi_rank_margin", float, DEFAULT_NMCC_PI_RANK_MARGIN),
    (
        "nmccPiRankMarginCoefficient",
        "nmcc_pi_rank_margin_coef",
        float,
        DEFAULT_NMCC_PI_RANK_MARGIN_COEF,
    ),
    (
        "nmccPiReplayMaxEpisodes",
        "nmcc_pi_replay_max_episodes",
        int,
        DEFAULT_NMCC_PI_REPLAY_MAX_EPISODES,
    ),
    (
        "nmccPiReplayEpochs",
        "nmcc_pi_replay_epochs",
        int,
        DEFAULT_NMCC_PI_REPLAY_EPOCHS,
    ),
    (
        "nmccPiValidationFraction",
        "nmcc_pi_validation_fraction",
        float,
        DEFAULT_NMCC_PI_VALIDATION_FRACTION,
    ),
    (
        "nmccPiEarlyStoppingPatience",
        "nmcc_pi_early_stopping_patience",
        int,
        DEFAULT_NMCC_PI_EARLY_STOPPING_PATIENCE,
    ),
    (
        "nmccPiMinimumValidationStates",
        "nmcc_pi_min_validation_states",
        int,
        DEFAULT_NMCC_PI_MIN_VALIDATION_STATES,
    ),
    (
        "nmccPiValidationGainZ",
        "nmcc_pi_validation_gain_z",
        float,
        DEFAULT_NMCC_PI_VALIDATION_GAIN_Z,
    ),
    (
        "nmccPiReplayRefit",
        "nmcc_pi_replay_refit",
        bool,
        DEFAULT_NMCC_PI_REPLAY_REFIT,
    ),
    (
        "explorationRateStart",
        "exploration_rate_start",
        float,
        DEFAULT_EXPLORATION_RATE_START,
    ),
    (
        "explorationRateEnd",
        "exploration_rate_end",
        float,
        DEFAULT_EXPLORATION_RATE_END,
    ),
    ("learningRateSchedule", "learning_rate_schedule", str, DEFAULT_LR_SCHEDULE),
    ("learningRateWarmupUpdates", "lr_warmup_updates", int, DEFAULT_LR_WARMUP_UPDATES),
    ("learningRateDecayUpdates", "lr_decay_updates", int, DEFAULT_LR_DECAY_UPDATES),
    (
        "actorLearningRateMinimumFraction",
        "actor_lr_min_fraction",
        float,
        DEFAULT_ACTOR_LR_MIN_FRACTION,
    ),
    (
        "criticLearningRateMinimumFraction",
        "critic_lr_min_fraction",
        float,
        DEFAULT_CRITIC_LR_MIN_FRACTION,
    ),
    ("actorPrior", "actor_prior", str, "active_population"),
    ("actorPriorScale", "actor_prior_scale", float, DEFAULT_ACTOR_PRIOR_SCALE),
    ("representationMode", "representation_mode", str, DEFAULT_REPRESENTATION_MODE),
    (
        "representationCloneCoefficient",
        "representation_clone_coef",
        float,
        DEFAULT_REPRESENTATION_CLONE_COEF,
    ),
    ("representationKlCap", "representation_kl_cap", float, DEFAULT_REPRESENTATION_KL_CAP),
)
