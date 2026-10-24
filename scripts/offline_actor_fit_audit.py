#!/usr/bin/env python3
"""Offline v64 front-feature representability audit.

The audit consumes the simulator-graded actor replay retained in a v64
checkpoint.  Scenario seeds are checkpointed alongside replay episodes so the
80/20 split is scenario-disjoint.  For each fit seed, a fresh actor is fitted
through ``RLBridge._fit_actor_iteration`` and evaluated on untouched held-out
episodes with either the full observation or the prespecified front-feature
ablation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DecisionInterface import (  # noqa: E402
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    INFRA_FEATURE_NAMES,
    MOMENTUM_FEATURE_NAMES,
    PED_FEATURE_NAMES,
)
from GNN import (  # noqa: E402
    DEFAULT_EMBED_DIM,
    DEFAULT_MESSAGE_LAYERS,
    EvacPolicy,
)
from RLBridge import (  # noqa: E402
    ActorReplayEpisode,
    MODEL_VERSION,
    ObservationFrame,
    RLBridge,
)


FRONT_CANDIDATE_FEATURES = (
    "threat_capture_fraction",
    "front_approach_fraction",
    "front_arrival_probability",
    "front_lead_time_fraction",
)
FRONT_CELL_FEATURES = (
    "susceptible_population_fraction",
    "threatened_population_fraction",
)


def _load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    signature = payload.get("model_signature")
    if not isinstance(signature, dict) or int(signature.get("version", -1)) != 64:
        raise ValueError("offline actor audit requires a model-v64 checkpoint")
    if int(MODEL_VERSION) != 64:
        raise RuntimeError("audit code and checkpoint model versions disagree")
    replay = payload.get("actor_replay")
    seeds = payload.get("actor_replay_scenario_seeds")
    if not isinstance(replay, list) or not replay:
        raise ValueError("checkpoint contains no actor replay")
    if (
        not isinstance(seeds, (list, tuple))
        or len(seeds) != len(replay)
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("checkpoint actor-replay scenario seeds are invalid")
    return payload


def _deserialize_frame(payload: dict) -> ObservationFrame:
    expected = set(ObservationFrame.__dataclass_fields__)
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("checkpoint observation-frame fields do not match v64")
    tensors = {}
    for name in expected:
        value = payload[name]
        if not torch.is_tensor(value):
            raise ValueError(f"observation field {name!r} is not a tensor")
        tensors[name] = value.detach().cpu().clone()
    frame = ObservationFrame(**tensors)
    if (
        frame.spatial_edge_index.ndim != 2
        or frame.spatial_edge_index.shape[0] != 2
    ):
        raise ValueError("observation spatial edges have an incompatible shape")
    if frame.spatial_edge_index.numel() and (
        int(frame.spatial_edge_index.min().item()) < 0
        or int(frame.spatial_edge_index.max().item())
        >= int(frame.cell_features.shape[0])
    ):
        raise ValueError("observation spatial edge references an invalid region")
    return frame


def _deserialize_replay(payload: dict) -> ActorReplayEpisode:
    expected = {
        "frames",
        "target_scores",
        "exact_masks",
        "eta_floors",
        "iteration",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("actor-replay episode fields do not match v64")
    frames = tuple(_deserialize_frame(frame) for frame in payload["frames"])
    scores = payload["target_scores"]
    masks = payload["exact_masks"]
    floors = payload["eta_floors"]
    iteration = payload["iteration"]
    if (
        not frames
        or not torch.is_tensor(scores)
        or not torch.is_tensor(masks)
        or not torch.is_tensor(floors)
        or scores.shape != masks.shape
        or scores.shape[0] != len(frames)
        or floors.shape != (len(frames),)
        or not isinstance(iteration, int)
        or isinstance(iteration, bool)
        or iteration < 0
    ):
        raise ValueError("actor-replay episode is invalid")
    return ActorReplayEpisode(
        frames=frames,
        target_scores=scores.detach().cpu().clone(),
        exact_masks=masks.detach().cpu().clone(),
        eta_floors=floors.detach().cpu().clone(),
        iteration=int(iteration),
    )


def _scenario_split(
    seeds: Sequence[int], *, train_fraction: float
) -> tuple[list[int], list[int]]:
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    unique = sorted(set(int(seed) for seed in seeds))
    if len(unique) < 2:
        raise ValueError("scenario-disjoint audit requires at least two scenario seeds")
    ranked = sorted(
        unique,
        key=lambda seed: hashlib.sha256(str(seed).encode("utf-8")).digest(),
    )
    train_count = min(
        len(ranked) - 1,
        max(1, int(round(float(train_fraction) * len(ranked)))),
    )
    training_seeds = set(ranked[:train_count])
    training = [index for index, seed in enumerate(seeds) if seed in training_seeds]
    held_out = [index for index, seed in enumerate(seeds) if seed not in training_seeds]
    return training, held_out


def _ablate_front(episode: ActorReplayEpisode) -> ActorReplayEpisode:
    candidate_indices = [
        CANDIDATE_FEATURE_NAMES.index(name)
        for name in FRONT_CANDIDATE_FEATURES
    ]
    cell_indices = [CELL_FEATURE_NAMES.index(name) for name in FRONT_CELL_FEATURES]
    frames = []
    for frame in episode.frames:
        cell_features = frame.cell_features.detach().cpu().clone()
        candidate_features = frame.candidate_features.detach().cpu().clone()
        cell_features[:, cell_indices] = 0.0
        candidate_features[:, candidate_indices] = 0.0
        frames.append(
            replace(
                frame,
                cell_features=cell_features,
                candidate_features=candidate_features,
            )
        )
    return replace(episode, frames=tuple(frames))


def _fresh_bridge(
    signature: dict,
    example: ActorReplayEpisode,
    *,
    fit_seed: int,
    learning_rate: float,
    epochs: int,
    epsilon: float,
    half_life: float,
    early_decisions: int,
    early_weight: float,
) -> RLBridge:
    torch.manual_seed(int(fit_seed))
    bridge = object.__new__(RLBridge)
    bridge.device = torch.device("cpu")
    bridge.edge_index = (
        example.frames[0].spatial_edge_index.detach().cpu().to(torch.long)
    )
    actor_prior = signature.get("actor_prior", {})
    prior_kind = str(actor_prior.get("kind", "none"))
    prior_scale = float(actor_prior.get("scale", 0.0))
    frame = example.frames[0]
    bridge.policy = EvacPolicy(
        d_ped=len(PED_FEATURE_NAMES),
        d_hazard=len(HAZARD_FEATURE_NAMES),
        d_infra=len(INFRA_FEATURE_NAMES),
        d_global=int(frame.global_features.numel()),
        d_candidate=len(CANDIDATE_FEATURE_NAMES),
        embed_dim=DEFAULT_EMBED_DIM,
        message_layers=DEFAULT_MESSAGE_LAYERS,
        d_momentum=len(MOMENTUM_FEATURE_NAMES),
        temporal_dim=int(
            signature["graph_contract"]["temporal_hidden_dimension"]
        ),
        actor_prior=prior_kind,
        actor_prior_scale=prior_scale,
        actor_prior_feature_index=(
            CANDIDATE_FEATURE_NAMES.index("public_regional_objective_score")
            if prior_kind == "objective_greedy"
            else 0
        ),
        auxiliary_models_enabled=(
            signature.get("architecture")
            == "relational_route_gnn_lstm_with_auxiliary_dynamics"
        ),
        d_critic_exogenous=int(frame.critic_exogenous.numel()),
    ).to(bridge.device)
    roles = bridge.policy.parameter_roles()
    bridge.representation_named_parameters = list(roles["representation"])
    bridge.actor_head_named_parameters = list(roles["actor_head"])
    bridge.actor_named_parameters = (
        bridge.representation_named_parameters + bridge.actor_head_named_parameters
    )
    bridge.actor_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in bridge.actor_named_parameters],
        lr=float(learning_rate),
    )
    bridge.nmcc_pi_replay_epochs = int(epochs)
    bridge.nmcc_pi_fit_tolerance = 1e-4
    bridge.nmcc_pi_epsilon = float(epsilon)
    bridge.nmcc_pi_replay_half_life_iterations = float(half_life)
    bridge.nmcc_pi_early_decisions = int(early_decisions)
    bridge.nmcc_pi_early_decision_weight = float(early_weight)
    return bridge


def _frame_weight(
    episode: ActorReplayEpisode,
    frame_index: int,
    *,
    now: int,
    half_life: float,
    early_decisions: int,
    early_weight: float,
) -> float:
    recency = (
        0.5 ** ((int(now) - int(episode.iteration)) / float(half_life))
        if float(half_life) > 0.0
        else 1.0
    )
    return recency * (
        float(early_weight) if frame_index < int(early_decisions) else 1.0
    )


def _evaluate(
    bridge: RLBridge,
    episodes: Sequence[ActorReplayEpisode],
    *,
    now: int,
    half_life: float,
    early_decisions: int,
    early_weight: float,
) -> dict[str, dict[str, float]]:
    totals = {
        "early_0_2": {
            "weight": 0.0,
            "top1": 0.0,
            "pair_correct": 0.0,
            "pair_weight": 0.0,
            "regret": 0.0,
        },
        "late_3_plus": {
            "weight": 0.0,
            "top1": 0.0,
            "pair_correct": 0.0,
            "pair_weight": 0.0,
            "regret": 0.0,
        },
    }
    bridge.policy.eval()
    with torch.no_grad():
        for episode in episodes:
            recurrent_state = None
            for decision_index, (frame, scores, exact_mask) in enumerate(
                zip(episode.frames, episode.target_scores, episode.exact_masks)
            ):
                logits, _, _, recurrent_state = (
                    bridge.policy.forward_nmcc_actor_recurrent(
                        bridge._graph_from_frame(frame),
                        recurrent_state,
                        frame.momentum_features,
                    )
                )
                logits = logits.reshape(-1).cpu()
                exact = exact_mask.reshape(-1).to(torch.bool).cpu()
                actions = torch.nonzero(exact, as_tuple=False).reshape(-1)
                if actions.numel() < 2:
                    continue
                target = scores.reshape(-1).cpu()
                chosen = int(actions[torch.argmax(logits[actions])].item())
                best = int(actions[torch.argmax(target[actions])].item())
                weight = _frame_weight(
                    episode,
                    decision_index,
                    now=now,
                    half_life=half_life,
                    early_decisions=early_decisions,
                    early_weight=early_weight,
                )
                group = totals[
                    "early_0_2" if decision_index < 3 else "late_3_plus"
                ]
                group["weight"] += weight
                group["top1"] += weight * float(chosen == best)
                group["regret"] += weight * float(target[best] - target[chosen])
                pairs = torch.combinations(actions, r=2)
                target_difference = target[pairs[:, 0]] - target[pairs[:, 1]]
                pair_weight = torch.abs(target_difference)
                informative = pair_weight > torch.finfo(pair_weight.dtype).eps
                if bool(informative.any()):
                    prediction_difference = (
                        logits[pairs[:, 0]] - logits[pairs[:, 1]]
                    )[informative]
                    target_difference = target_difference[informative]
                    pair_weight = pair_weight[informative]
                    matching = (
                        torch.sign(prediction_difference)
                        == torch.sign(target_difference)
                    )
                    group["pair_correct"] += weight * float(
                        pair_weight[matching].sum().item()
                    )
                    group["pair_weight"] += weight * float(
                        pair_weight.sum().item()
                    )
    result = {}
    for group_name, values in totals.items():
        result[group_name] = {
            "weighted_top1_accuracy": (
                values["top1"] / values["weight"] if values["weight"] else 0.0
            ),
            "weighted_pairwise_accuracy": (
                values["pair_correct"] / values["pair_weight"]
                if values["pair_weight"]
                else 0.0
            ),
            "weighted_mean_regret": (
                values["regret"] / values["weight"]
                if values["weight"]
                else 0.0
            ),
            "weight": values["weight"],
        }
    return result


def run_audit(args: argparse.Namespace) -> dict:
    payload = _load_checkpoint(Path(args.checkpoint))
    signature = payload["model_signature"]
    source = [_deserialize_replay(item) for item in payload["actor_replay"]]
    scenario_seeds = [int(seed) for seed in payload["actor_replay_scenario_seeds"]]
    training_indices, held_out_indices = _scenario_split(
        scenario_seeds,
        train_fraction=float(args.train_fraction),
    )
    now = max(int(episode.iteration) for episode in source)
    rows = []
    for fit_seed in args.fit_seeds:
        for variant in ("full", "no_front"):
            episodes = (
                source
                if variant == "full"
                else [_ablate_front(episode) for episode in source]
            )
            training = [episodes[index] for index in training_indices]
            held_out = [episodes[index] for index in held_out_indices]
            bridge = _fresh_bridge(
                signature,
                training[0],
                fit_seed=int(fit_seed),
                learning_rate=float(args.learning_rate),
                epochs=int(args.epochs),
                epsilon=float(args.epsilon),
                half_life=float(args.half_life),
                early_decisions=int(args.early_decisions),
                early_weight=float(args.early_weight),
            )
            bridge.actor_replay = training
            bridge.actor_rollout_updates = now
            fit_metrics = bridge._fit_actor_iteration(
                bridge._actor_fit_episodes()
            )
            evaluation = _evaluate(
                bridge,
                held_out,
                now=now,
                half_life=float(args.half_life),
                early_decisions=int(args.early_decisions),
                early_weight=float(args.early_weight),
            )
            rows.append(
                {
                    "fit_seed": int(fit_seed),
                    "variant": variant,
                    "fit": fit_metrics,
                    "held_out": evaluation,
                }
            )
    early_top1 = {
        (row["fit_seed"], row["variant"]): row["held_out"]["early_0_2"][
            "weighted_top1_accuracy"
        ]
        for row in rows
    }
    improvements = {
        str(fit_seed): early_top1[(fit_seed, "full")]
        - early_top1[(fit_seed, "no_front")]
        for fit_seed in args.fit_seeds
    }
    passed_seeds = sum(value >= 0.1 for value in improvements.values())
    return {
        "schema_version": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_version": int(signature["version"]),
        "replay_episodes": len(source),
        "training_episodes": len(training_indices),
        "held_out_episodes": len(held_out_indices),
        "training_scenario_seeds": sorted(
            {scenario_seeds[index] for index in training_indices}
        ),
        "held_out_scenario_seeds": sorted(
            {scenario_seeds[index] for index in held_out_indices}
        ),
        "front_candidate_features": list(FRONT_CANDIDATE_FEATURES),
        "front_cell_features": list(FRONT_CELL_FEATURES),
        "rows": rows,
        "gate": {
            "criterion": (
                "full minus no_front early top-1 >= 0.1 on at least two "
                "of three fit seeds"
            ),
            "early_top1_improvement_by_fit_seed": improvements,
            "passing_fit_seeds": passed_seeds,
            "passed": passed_seeds >= 2,
        },
    }


def _parse_fit_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise argparse.ArgumentTypeError("fit seeds must contain three distinct integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-seeds", type=_parse_fit_seeds, default=(1, 2, 3))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--half-life", type=float, default=5.0)
    parser.add_argument("--early-decisions", type=int, default=3)
    parser.add_argument("--early-weight", type=float, default=2.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.learning_rate <= 0.0 or not np.isfinite(args.learning_rate):
        raise ValueError("learning rate must be finite and positive")
    report = run_audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
