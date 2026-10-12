# -*- coding: utf-8 -*-
"""Torch-free configuration contract for NMCC policy improvement (NMCC-PI)
and the learner's representation-ownership mode.

Shared by Core (defaults, casting, recorded configuration), RLBridge
(constructor defaults) and TrainingCurriculum (the learner keys a curriculum
may set and must hold constant), so that none of them has to import torch to
agree on names, types or defaults.
"""

# Historical fitted-policy curricula may still opt into this prior scale.
DEFAULT_ACTOR_PRIOR_SCALE = 1.0

DEFAULT_NMCC_PI_EPSILON = 0.05  # one conservative E-step KL(q || pi_old)
DEFAULT_NMCC_PI_ETA_MIN = 0.03  # temperature floor for near-tied costs
DEFAULT_NMCC_PI_TAPES = 4
# Full-horizon rollout branches dominate training time. A panel includes the
# objective-greedy contenders plus uniform feasible extras, so every cell has
# a nonzero grading chance. A value of zero means exhaustive evaluation.
DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS = 0
DEFAULT_NMCC_PI_MAX_BRANCHES = 6
# Recent physical branch evidence is replayed across policy iterations. The
# categorical probabilities are regenerated from the current actor each time,
# so the window remembers simulator outcomes without preserving stale policy
# targets.
DEFAULT_NMCC_PI_ACTOR_EPOCHS = 64
DEFAULT_NMCC_PI_FIT_TOLERANCE = 0.01
DEFAULT_NMCC_PI_BRANCH_HORIZON = 20
DEFAULT_NMCC_PI_REPLAY_MAX_EPISODES = 12
DEFAULT_NMCC_PI_REPLAY_EPOCHS = 24
DEFAULT_NMCC_PI_VALIDATION_FRACTION = 0.20
DEFAULT_NMCC_PI_CONTINUATION_ENABLED = False
DEFAULT_NMCC_PI_CONTINUATION_LOSS_COEF = 1.0
DEFAULT_NMCC_PI_WORKERS = 4
DEFAULT_RANDOM_REFERENCE_ROLLOUTS = 4
DEFAULT_EXPLORATION_RATE_START = 0.30
DEFAULT_EXPLORATION_RATE_END = 0.02
DEFAULT_EXPLORATION_SCHEDULE = "half_cosine"
DEFAULT_LR_SCHEDULE = "cosine"
DEFAULT_LR_WARMUP_UPDATES = 1
DEFAULT_LR_DECAY_UPDATES = 32
DEFAULT_ACTOR_LR_MIN_FRACTION = 0.10
DEFAULT_CRITIC_LR_MIN_FRACTION = 0.20

# Learner representation contract (see RLBridge._run_learner_update).
REPRESENTATION_MODES = (
    "actor_owned",
    "shared_multitask",
    "shared_phasic",
    "staged_transfer",
)
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
    ("nmccPiTapes", "nmcc_pi_tapes", int, DEFAULT_NMCC_PI_TAPES),
    (
        "nmccPiExhaustiveDecisions",
        "nmcc_pi_exhaustive_decisions",
        int,
        DEFAULT_NMCC_PI_EXHAUSTIVE_DECISIONS,
    ),
    ("nmccPiMaxBranches", "nmcc_pi_max_branches", int, DEFAULT_NMCC_PI_MAX_BRANCHES),
    ("nmccPiActorEpochs", "nmcc_pi_actor_epochs", int, DEFAULT_NMCC_PI_ACTOR_EPOCHS),
    ("nmccPiFitTolerance", "nmcc_pi_fit_tolerance", float, DEFAULT_NMCC_PI_FIT_TOLERANCE),
    (
        "nmccPiBranchHorizon",
        "nmcc_pi_branch_horizon",
        int,
        DEFAULT_NMCC_PI_BRANCH_HORIZON,
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
    ("actorPrior", "actor_prior", str, "none"),
    ("actorPriorScale", "actor_prior_scale", float, 0.0),
    ("representationMode", "representation_mode", str, DEFAULT_REPRESENTATION_MODE),
    (
        "representationCloneCoefficient",
        "representation_clone_coef",
        float,
        DEFAULT_REPRESENTATION_CLONE_COEF,
    ),
    ("representationKlCap", "representation_kl_cap", float, DEFAULT_REPRESENTATION_KL_CAP),
)

# V37 is an additive extension of the historical NMCC-PI contract. Keeping it
# separate prevents old registered curriculum hashes/contracts from silently
# acquiring fields that did not exist when those experiments were frozen.
NMCC_PI_CONTINUATION_CORE_FIELDS = (
    (
        "randomReferenceRollouts",
        "random_reference_rollouts",
        int,
        DEFAULT_RANDOM_REFERENCE_ROLLOUTS,
    ),
    (
        "nmccPiContinuationEnabled",
        "nmcc_pi_continuation_enabled",
        bool,
        DEFAULT_NMCC_PI_CONTINUATION_ENABLED,
    ),
    (
        "nmccPiContinuationLossCoefficient",
        "nmcc_pi_continuation_loss_coef",
        float,
        DEFAULT_NMCC_PI_CONTINUATION_LOSS_COEF,
    ),
    (
        "nmccPiWorkers",
        "nmcc_pi_workers",
        int,
        DEFAULT_NMCC_PI_WORKERS,
    ),
)

NMCC_PI_ALL_CORE_FIELDS = (
    NMCC_PI_CORE_FIELDS + NMCC_PI_CONTINUATION_CORE_FIELDS
)
