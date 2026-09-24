#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolution-flexible graph policy for sequential shelter deployment.

The policy consumes a packed batch of regional graphs.  Node count and graph
topology are runtime data: no trainable tensor is indexed by a fixed cell id.
Two relation types are deliberately retained because they have different
operational meanings:

* spatial edges carry road/network and neighboring-region context;
* route edges carry the current aggregate pedestrian-to-shelter assignments.

Candidate shelters are scored with one shared network using the contextual
embedding of their host region and their exact, administrator-visible site
features.  This keeps the architecture compact while allowing it to learn
beyond demand-only heuristic rankings.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RewardProcessor import (
    DEFAULT_CASUALTY_WEIGHT,
    DEFAULT_EVACUATION_TIME_WEIGHT,
    DEFAULT_HAZARD_EXPOSURE_WEIGHT,
    DEFAULT_SAFE_COMPLETION_WEIGHT,
)


HEURISTIC_PRIOR_SCALE = 1.0
# ``risk_time_reduction`` is the v28 operational prior. The older route-time
# name remains loadable only for historical experiment definitions.
ACTOR_PRIORS = (
    "active_population",
    "risk_time_reduction",
    "route_time_saving",
    "none",
)
RESIDUAL_LOGIT_BOUND = 4.0
DEFAULT_EMBED_DIM = 32
DEFAULT_MESSAGE_LAYERS = 2
DEFAULT_TEMPORAL_DIM = 96
DEFAULT_NMCC_ENSEMBLE_SIZE = 3
VALUE_COMPONENT_NAMES = (
    "safe_completion",
    "casualty",
    "evacuation_time",
    "hazard_exposure",
)
NMCC_OUTCOME_NAMES = (
    "safe_completions",
    "casualties",
    "active_person_time",
    "hazard_exposure_person_time",
    "final_active_population",
    "final_risk_mass",
)


@dataclass
class GNNInput:
    x_ped: torch.Tensor
    x_hazard: torch.Tensor
    x_infra: torch.Tensor
    x_global: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    route_edge_index: Optional[torch.Tensor] = None
    route_edge_weight: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None
    candidate_cell_index: Optional[torch.Tensor] = None
    candidate_features: Optional[torch.Tensor] = None


@dataclass
class NMCCPolicyOutput:
    """Complete hybrid-NMCC output for one recurrent observation.

    ``natural_outcomes`` predicts the normalized finite-horizon outcome under
    ``WAIT``. ``causal_outcome_samples`` are independently initialized residual
    heads for every candidate; their disagreement supplies the conservative
    optimization penalty. The first four outcome coordinates are transformed
    exactly into the registered reward components, preserving an auditable
    dueling decomposition ``Q = V_wait + D``.
    """

    logits: torch.Tensor
    prior_logits: torch.Tensor
    value: torch.Tensor
    value_components: torch.Tensor
    learned_residual: torch.Tensor
    recurrent_state: tuple[torch.Tensor, torch.Tensor]
    natural_outcomes: torch.Tensor
    natural_components: torch.Tensor
    causal_outcome_samples: torch.Tensor
    causal_component_samples: torch.Tensor
    causal_component_mean: torch.Tensor
    causal_component_std: torch.Tensor
    robust_causal_score: torch.Tensor
    teacher_logits: torch.Tensor
    # NMCC-PI intervention-value ensemble: one scalar per member and candidate,
    # predicting the within-state, policy-centered full-horizon advantage.
    # Only its within-state differences are identified.
    improvement_value_samples: Optional[torch.Tensor] = None


def _validated_batch(batch: Optional[torch.Tensor], node_count: int, device) -> torch.Tensor:
    if batch is None:
        return torch.zeros(node_count, dtype=torch.long, device=device)
    if batch.ndim != 1 or batch.numel() != node_count:
        raise ValueError(f"batch must have shape ({node_count},), got {tuple(batch.shape)}")
    batch = batch.to(device=device, dtype=torch.long)
    if batch.numel() == 0 or int(batch.min().item()) < 0:
        raise ValueError("batch graph ids must be non-negative and non-empty")
    graph_ids = torch.unique(batch, sorted=True)
    expected = torch.arange(graph_ids.numel(), device=batch.device)
    if not torch.equal(graph_ids, expected):
        raise ValueError("batch graph ids must be contiguous and start at zero")
    if batch.numel() > 1 and bool(torch.any(batch[1:] < batch[:-1])):
        raise ValueError("nodes must be grouped by graph in packed batches")
    return batch


def graph_node_counts(batch: torch.Tensor) -> torch.Tensor:
    """Return the runtime node count of every graph in a packed batch."""
    graph_count = int(batch.max().item()) + 1
    return torch.bincount(batch, minlength=graph_count)


def graph_node_offsets(counts: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=counts.device),
            torch.cumsum(counts[:-1], dim=0),
        )
    )


def batch_local_edge_index(
    edge_index: Optional[torch.Tensor],
    node_counts: Union[Sequence[int], torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Repeat one local edge template over graphs with matching node counts.

    This helper is intended for a fixed-resolution PPO minibatch.  Mixed-size
    graph batches should supply a fully packed global edge list directly.
    """
    if edge_index is None:
        return None
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    counts = torch.as_tensor(node_counts, dtype=torch.long, device=edges.device).reshape(-1)
    if counts.numel() == 0 or bool(torch.any(counts <= 0)):
        raise ValueError("node_counts must contain positive graph sizes")
    if edges.numel() == 0:
        return edges.new_empty((2, 0))
    template_nodes = int(counts[0].item())
    if bool(torch.any(counts != template_nodes)):
        raise ValueError("a local edge template requires equal node counts")
    if int(edges.min().item()) < 0 or int(edges.max().item()) >= template_nodes:
        raise ValueError("local edge template contains a node outside its graph")
    offsets = graph_node_offsets(counts)
    return torch.cat([edges + offset for offset in offsets], dim=1)


def _normalize_edge_index(
    edge_index: Optional[torch.Tensor],
    batch: torch.Tensor,
    *,
    name: str,
) -> Optional[torch.Tensor]:
    if edge_index is None:
        return None
    edges = torch.as_tensor(edge_index, dtype=torch.long, device=batch.device)
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError(f"{name} must have shape (2, edges)")
    if edges.numel() == 0:
        return edges.reshape(2, 0)
    node_count = int(batch.numel())
    counts = graph_node_counts(batch)
    graph_count = int(counts.numel())

    # Backward-compatible convenience: a template whose indices fit one graph
    # is repeated for an equal-sized packed batch.  New mixed-resolution code
    # should provide explicit packed indices.
    if graph_count > 1 and bool(torch.all(counts == counts[0])):
        local_count = int(counts[0].item())
        if int(edges.max().item()) < local_count:
            edges = batch_local_edge_index(edges, counts, device=batch.device)

    if int(edges.min().item()) < 0 or int(edges.max().item()) >= node_count:
        raise ValueError(f"{name} contains a node outside the packed batch")
    if not torch.equal(batch[edges[0]], batch[edges[1]]):
        raise ValueError(f"{name} cannot connect nodes from different graphs")
    return edges


def fit_gnn(
    x_ped: torch.Tensor,
    x_hazard: torch.Tensor,
    x_infra: torch.Tensor,
    x_global: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    route_edge_index: Optional[torch.Tensor] = None,
    route_edge_weight: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    candidate_cell_index: Optional[torch.Tensor] = None,
    candidate_features: Optional[torch.Tensor] = None,
) -> GNNInput:
    """Validate and package a packed, variable-size regional graph batch."""
    tensors = {"x_ped": x_ped, "x_hazard": x_hazard, "x_infra": x_infra}
    for name, tensor in tensors.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape (nodes, features), got {tuple(tensor.shape)}")
    node_count = int(x_ped.size(0))
    if node_count <= 0:
        raise ValueError("At least one region node is required")
    if x_hazard.size(0) != node_count or x_infra.size(0) != node_count:
        raise ValueError("All observation branches must contain the same number of nodes")

    batch = _validated_batch(batch, node_count, x_ped.device)
    counts = graph_node_counts(batch)
    graph_count = int(counts.numel())
    if x_global is not None:
        if x_global.ndim == 1:
            x_global = x_global.unsqueeze(0)
        if x_global.ndim != 2 or x_global.size(0) != graph_count:
            raise ValueError(
                f"x_global must have shape ({graph_count}, features), got {tuple(x_global.shape)}"
            )

    edge_index = _normalize_edge_index(edge_index, batch, name="edge_index")
    route_edge_index = _normalize_edge_index(
        route_edge_index,
        batch,
        name="route_edge_index",
    )
    if route_edge_weight is not None:
        if route_edge_index is None:
            raise ValueError("route_edge_weight requires route_edge_index")
        route_edge_weight = torch.as_tensor(
            route_edge_weight,
            dtype=x_ped.dtype,
            device=x_ped.device,
        ).reshape(-1)
        if route_edge_weight.numel() != route_edge_index.size(1):
            raise ValueError("route_edge_weight must contain one value per route edge")
        if not bool(torch.isfinite(route_edge_weight).all()) or bool(
            torch.any(route_edge_weight < 0.0)
        ):
            raise ValueError("route_edge_weight must be finite and non-negative")

    if (candidate_cell_index is None) != (candidate_features is None):
        raise ValueError("candidate_cell_index and candidate_features must be supplied together")
    if candidate_cell_index is not None:
        if candidate_cell_index.ndim == 1:
            candidate_cell_index = candidate_cell_index.unsqueeze(0)
        if candidate_features.ndim == 2:
            candidate_features = candidate_features.unsqueeze(0)
        if candidate_cell_index.ndim != 2:
            raise ValueError("candidate_cell_index must have shape (graphs, actions)")
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape (graphs, actions, features)")
        if candidate_cell_index.size(0) != graph_count:
            raise ValueError("candidate_cell_index must contain one row per graph")
        if candidate_features.shape[:2] != candidate_cell_index.shape:
            raise ValueError("candidate tensors must share graph and action dimensions")
        for graph_id, count in enumerate(counts.tolist()):
            local = candidate_cell_index[graph_id]
            if local.numel() and (
                int(local.min().item()) < 0 or int(local.max().item()) >= int(count)
            ):
                raise ValueError(
                    f"candidate_cell_index contains a region outside graph {graph_id}"
                )

    return GNNInput(
        x_ped=x_ped,
        x_hazard=x_hazard,
        x_infra=x_infra,
        x_global=x_global,
        edge_index=edge_index,
        route_edge_index=route_edge_index,
        route_edge_weight=route_edge_weight,
        batch=batch,
        candidate_cell_index=candidate_cell_index,
        candidate_features=candidate_features,
    )


def grid_edge_index(nx: int, ny: int) -> np.ndarray:
    """Return a directed four-neighbor edge list with shape ``(2, E)``."""
    nx = int(nx)
    ny = int(ny)
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Grid dimensions must be positive, got ({nx}, {ny})")
    edges = []
    for i in range(nx):
        for j in range(ny):
            u = i * ny + j
            if i + 1 < nx:
                v = (i + 1) * ny + j
                edges.extend(((u, v), (v, u)))
            if j + 1 < ny:
                v = i * ny + (j + 1)
                edges.extend(((u, v), (v, u)))
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


class _MLPBranch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RelationalMessageLayer(nn.Module):
    """One residual message layer with separate spatial and route relations."""

    def __init__(self, dimension: int):
        super().__init__()
        self.self_linear = nn.Linear(dimension, dimension)
        self.spatial_linear = nn.Linear(dimension, dimension, bias=False)
        self.route_linear = nn.Linear(dimension, dimension, bias=False)
        self.norm = nn.LayerNorm(dimension)

    @staticmethod
    def _aggregate(
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor] = None,
        *,
        normalize: bool,
    ) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros_like(x)
        source, target = edge_index[0], edge_index[1]
        if edge_weight is None:
            weights = torch.ones(source.numel(), dtype=x.dtype, device=x.device)
        else:
            weights = edge_weight.to(device=x.device, dtype=x.dtype).reshape(-1)
        messages = x[source] * weights.unsqueeze(-1)
        totals = torch.zeros_like(x)
        totals.index_add_(0, target, messages)
        if not normalize:
            return totals
        normalizer = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        normalizer.index_add_(0, target, weights)
        return totals / normalizer.clamp_min(1e-8).unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        spatial_edges: Optional[torch.Tensor],
        route_edges: Optional[torch.Tensor],
        route_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        spatial = self._aggregate(x, spatial_edges, normalize=True)
        # Route weights are fractions of initialized population. A weighted
        # mean would erase assignment magnitude whenever only one route enters
        # a region; a bounded weighted sum preserves the difference between a
        # small and a large assigned population.
        routes = self._aggregate(
            x,
            route_edges,
            route_weights,
            normalize=False,
        )
        update = F.gelu(
            self.self_linear(x)
            + self.spatial_linear(spatial)
            + self.route_linear(routes)
        )
        return self.norm(x + update)


class EvacPolicy(nn.Module):
    """Compact relational GNN actor-critic with variable region-node count.

    No constructor argument or trainable tensor depends on a grid dimension.
    The same policy instance can therefore evaluate different regional graph
    sizes when given valid packed inputs.
    """

    HEURISTIC_PRIOR_SCALE = HEURISTIC_PRIOR_SCALE
    RESIDUAL_LOGIT_BOUND = RESIDUAL_LOGIT_BOUND

    def __init__(
        self,
        d_ped: int,
        d_hazard: int,
        d_infra: int,
        d_global: int = 0,
        d_candidate: int = 1,
        embed_dim: int = DEFAULT_EMBED_DIM,
        message_layers: int = DEFAULT_MESSAGE_LAYERS,
        d_momentum: int = 0,
        temporal_dim: int = DEFAULT_TEMPORAL_DIM,
        residual_logit_bound: float = RESIDUAL_LOGIT_BOUND,
        nmcc_ensemble_size: int = DEFAULT_NMCC_ENSEMBLE_SIZE,
        verbose: bool = False,
        actor_prior: str = "active_population",
        actor_prior_scale: float = HEURISTIC_PRIOR_SCALE,
        actor_prior_feature_index: int = 7,
    ):
        super().__init__()
        dimensions = {
            "d_ped": d_ped,
            "d_hazard": d_hazard,
            "d_infra": d_infra,
            "d_candidate": d_candidate,
            "embed_dim": embed_dim,
            "message_layers": message_layers,
            "temporal_dim": temporal_dim,
            "nmcc_ensemble_size": nmcc_ensemble_size,
        }
        for name, value in dimensions.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        self.verbose = bool(verbose)
        self.d_ped = int(d_ped)
        self.d_hazard = int(d_hazard)
        self.d_infra = int(d_infra)
        self.d_global = int(d_global)
        self.d_candidate = int(d_candidate)
        self.d_momentum = int(d_momentum)
        self.temporal_dim = int(temporal_dim)
        self.residual_logit_bound = float(residual_logit_bound)
        self.nmcc_ensemble_size = int(nmcc_ensemble_size)
        if self.d_global < 0:
            raise ValueError("d_global must be non-negative")
        if self.d_momentum < 0:
            raise ValueError("d_momentum must be non-negative")
        if not np.isfinite(self.residual_logit_bound) or self.residual_logit_bound <= 0.0:
            raise ValueError("residual_logit_bound must be finite and positive")
        self.actor_prior = str(actor_prior)
        if self.actor_prior not in ACTOR_PRIORS:
            raise ValueError(f"actor_prior must be one of {ACTOR_PRIORS}, got {actor_prior!r}")
        self.actor_prior_scale = float(actor_prior_scale)
        if not np.isfinite(self.actor_prior_scale) or self.actor_prior_scale < 0.0:
            raise ValueError("actor_prior_scale must be finite and non-negative")
        self.actor_prior_feature_index = int(actor_prior_feature_index)
        if self.actor_prior in {"risk_time_reduction", "route_time_saving"} and not (
            0 <= self.actor_prior_feature_index < self.d_candidate
        ):
            raise ValueError("actor_prior_feature_index must index a candidate feature")

        branch_dim = int(embed_dim)
        self.ped_encoder = _MLPBranch(int(d_ped), branch_dim)
        self.hazard_encoder = _MLPBranch(int(d_hazard), branch_dim)
        self.infra_encoder = _MLPBranch(int(d_infra), branch_dim)
        model_dim = 3 * branch_dim
        self.fusion = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.message_layers = nn.ModuleList(
            _RelationalMessageLayer(model_dim)
            for _ in range(int(message_layers))
        )

        # A learned attentive summary captures distributed need; maximum
        # pooling preserves a rare urgent region. Both are size independent.
        self.pool_gate = nn.Linear(model_dim, 1)
        graph_dim = 2 * model_dim
        decision_context_dim = graph_dim + self.d_global
        actor_dim = max(64, 2 * int(embed_dim))
        temporal_input_dim = decision_context_dim + self.d_momentum
        self.temporal = nn.LSTMCell(temporal_input_dim, self.temporal_dim)
        # A positive forget bias favors retaining operational history at the
        # beginning of training without hard-coding a particular time scale.
        with torch.no_grad():
            self.temporal.bias_ih[
                self.temporal_dim : 2 * self.temporal_dim
            ].fill_(1.0)
        self.node_actor = nn.Sequential(
            nn.Linear(model_dim, actor_dim),
            nn.GELU(),
            nn.Linear(actor_dim, actor_dim),
        )
        self.candidate_actor = _MLPBranch(self.d_candidate, actor_dim)
        self.global_actor_context = nn.Linear(decision_context_dim, actor_dim)
        self.temporal_actor_context = nn.Linear(self.temporal_dim, actor_dim, bias=False)
        nn.init.zeros_(self.temporal_actor_context.weight)
        # Zero readouts make the untrained policy exactly the prior.  They cost
        # one Adam step per zero layer before the representation and the LSTM
        # receive actor gradient (the first step's representation gradient is
        # exactly zero), after which Adam's per-parameter normalization makes
        # the delay immaterial.  Measured in learner_flow_experiment.py: with
        # a 32-epoch M-step, zero and small-random readouts reach the same
        # regret; the binding constraint in v22/v23 was the actor step budget
        # (one epoch under a 0.015 KL target), not the initialization.
        self.actor_cell = nn.Linear(actor_dim, 1)
        nn.init.zeros_(self.actor_cell.weight)
        nn.init.zeros_(self.actor_cell.bias)
        critic_input_dim = decision_context_dim + self.temporal_dim
        self.critic_trunk = nn.Sequential(
            nn.Linear(critic_input_dim, 2 * actor_dim),
            nn.GELU(),
            nn.LayerNorm(2 * actor_dim),
            nn.Linear(2 * actor_dim, actor_dim),
            nn.GELU(),
            nn.LayerNorm(actor_dim),
        )
        self.critic_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(actor_dim, actor_dim),
                    nn.GELU(),
                    nn.Linear(actor_dim, 1),
                )
                for name in VALUE_COMPONENT_NAMES
            }
        )
        for head in self.critic_heads.values():
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        # NMCC's natural model and intervention residual are deliberately
        # separate heads. The natural model cannot consume a candidate action;
        # the residual ensemble must consume one. This prevents the much larger
        # natural trajectory from absorbing the sparse intervention effect.
        self.natural_outcome_head = nn.Sequential(
            nn.Linear(actor_dim, actor_dim),
            nn.GELU(),
            nn.Linear(actor_dim, len(NMCC_OUTCOME_NAMES)),
        )
        nn.init.zeros_(self.natural_outcome_head[-1].weight)
        nn.init.zeros_(self.natural_outcome_head[-1].bias)
        self.causal_outcome_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(actor_dim, actor_dim),
                nn.GELU(),
                nn.Linear(actor_dim, len(NMCC_OUTCOME_NAMES)),
            )
            for _ in range(self.nmcc_ensemble_size)
        )
        # Independent small initializations give the ensemble a real epistemic
        # disagreement signal without perturbing the established actor.
        for index, head in enumerate(self.causal_outcome_heads):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(17_101 + index)
            with torch.no_grad():
                head[-1].weight.copy_(
                    1e-3
                    * torch.randn(
                        head[-1].weight.shape,
                        generator=generator,
                        dtype=head[-1].weight.dtype,
                    )
                )
                head[-1].bias.zero_()

        # Intervention-value ensemble for NMCC policy improvement.  Separate
        # from the finite-horizon outcome heads above: those predict the
        # physical L-step residual used by the natural/causal decomposition;
        # these predict the controller-relevant quantity, each cell's
        # full-horizon value relative to the other cells at the same state.
        self.improvement_value_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(actor_dim, actor_dim),
                nn.GELU(),
                nn.Linear(actor_dim, 1),
            )
            for _ in range(self.nmcc_ensemble_size)
        )
        # Low-variance wide path.  Exact branch datasets are deliberately
        # small and expensive; a purely deep readout can memorize episode
        # embeddings before learning stable physical coefficients.  These
        # linear heads see the same administrator-interpretable local,
        # candidate, scenario and momentum features while the parallel deep
        # path retains the full relational GNN/LSTM capacity.
        improvement_direct_dim = (
            self.d_ped
            + self.d_hazard
            + self.d_infra
            + self.d_candidate
            + self.d_global
            + self.d_momentum
        )
        self.improvement_direct_heads = nn.ModuleList(
            nn.Linear(improvement_direct_dim, 1)
            for _ in range(self.nmcc_ensemble_size)
        )
        for index, head in enumerate(self.improvement_value_heads):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(29_311 + index)
            with torch.no_grad():
                for layer in (head[0], head[-1]):
                    layer.weight.copy_(
                        (0.05 if layer is head[0] else 1e-3)
                        * torch.randn(
                            layer.weight.shape,
                            generator=generator,
                            dtype=layer.weight.dtype,
                        )
                    )
                    layer.bias.zero_()

        # The fitted-policy-iteration value branch must not depend on the
        # streaming actor projections.  In v25 its heads consumed
        # ``candidate_hidden`` even though those projections were actor-owned
        # and therefore frozen during the critic/world-model pass.  These
        # dedicated modules let exact replay labels train a complete
        # graph/candidate/temporal value function while retaining the same full
        # relational GNN and LSTM representation.
        self.improvement_node = nn.Sequential(
            nn.Linear(model_dim, actor_dim),
            nn.GELU(),
            nn.Linear(actor_dim, actor_dim),
        )
        self.improvement_candidate = _MLPBranch(self.d_candidate, actor_dim)
        self.improvement_global_context = nn.Linear(decision_context_dim, actor_dim)
        self.improvement_temporal_context = nn.Linear(
            self.temporal_dim, actor_dim, bias=False
        )
        self.reset_improvement_model(seed=29_311)

        if self.verbose:
            print(
                f"[EvacPolicy] resolution-flexible model_dim={model_dim} "
                f"message_layers={len(self.message_layers)}"
            )

    # Module ownership used by the learner.  The representation is everything
    # that turns an observation history into region/decision embeddings; the
    # actor head turns them into cell logits; the critic head holds every
    # value, world-model and intervention-value readout.
    REPRESENTATION_MODULES = (
        "ped_encoder",
        "hazard_encoder",
        "infra_encoder",
        "fusion",
        "message_layers",
        "pool_gate",
        "temporal",
    )
    ACTOR_HEAD_MODULES = (
        "node_actor",
        "candidate_actor",
        "global_actor_context",
        "temporal_actor_context",
        "actor_cell",
    )
    CRITIC_HEAD_MODULES = (
        "critic_trunk",
        "critic_heads",
        "natural_outcome_head",
        "causal_outcome_heads",
        "improvement_node",
        "improvement_candidate",
        "improvement_global_context",
        "improvement_temporal_context",
        "improvement_value_heads",
        "improvement_direct_heads",
    )

    def improvement_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Parameters refitted on the persistent exact-branch dataset."""
        prefixes = (
            "improvement_node.",
            "improvement_candidate.",
            "improvement_global_context.",
            "improvement_temporal_context.",
            "improvement_value_heads.",
            "improvement_direct_heads.",
        )
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.startswith(prefixes)
        ]

    def reset_improvement_model(self, *, seed: int) -> None:
        """Reinitialize only the fitted intervention-value model.

        The shared GNN/LSTM retains system knowledge; the candidate-value
        projection and ensemble are refit against the complete retained label
        set so a small new rollout cannot overwrite earlier scenarios.
        """
        modules = (
            self.improvement_node,
            self.improvement_candidate,
            self.improvement_global_context,
            self.improvement_temporal_context,
            self.improvement_value_heads,
            self.improvement_direct_heads,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            visited = set()
            for root in modules:
                for module in root.modules():
                    if id(module) in visited:
                        continue
                    visited.add(id(module))
                    reset = getattr(module, "reset_parameters", None)
                    if callable(reset):
                        reset()
            # Keep initial value corrections conservative without making the
            # upstream fitted branch identical across ensemble members.
            for index, head in enumerate(self.improvement_value_heads):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed) + 101 * index)
                with torch.no_grad():
                    head[-1].weight.copy_(
                        1e-3
                        * torch.randn(
                            head[-1].weight.shape,
                            generator=generator,
                            dtype=head[-1].weight.dtype,
                        )
                    )
                    head[-1].bias.zero_()
            for index, head in enumerate(self.improvement_direct_heads):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed) + 211 * index)
                with torch.no_grad():
                    head.weight.copy_(
                        1e-3
                        * torch.randn(
                            head.weight.shape,
                            generator=generator,
                            dtype=head.weight.dtype,
                        )
                    )
                    head.bias.zero_()

    def parameter_roles(self) -> dict[str, list[tuple[str, nn.Parameter]]]:
        """Partition every parameter into representation / actor / critic."""
        roles: dict[str, list[tuple[str, nn.Parameter]]] = {
            "representation": [],
            "actor_head": [],
            "critic_head": [],
        }
        owners = (
            ("representation", self.REPRESENTATION_MODULES),
            ("actor_head", self.ACTOR_HEAD_MODULES),
            ("critic_head", self.CRITIC_HEAD_MODULES),
        )
        for name, parameter in self.named_parameters():
            module = name.split(".", 1)[0]
            matches = [role for role, modules in owners if module in modules]
            if len(matches) != 1:
                raise RuntimeError(f"Parameter {name!r} has no unique learner role")
            roles[matches[0]].append((name, parameter))
        return roles

    def _graph_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        graph_count = int(batch.max().item()) + 1
        scores = self.pool_gate(x).squeeze(-1)
        score_max = torch.full(
            (graph_count,),
            -torch.inf,
            dtype=x.dtype,
            device=x.device,
        )
        score_max.scatter_reduce_(0, batch, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - score_max[batch])
        weight_total = torch.zeros(graph_count, dtype=x.dtype, device=x.device)
        weight_total.index_add_(0, batch, weights)
        attentive = torch.zeros(
            (graph_count, x.size(-1)),
            dtype=x.dtype,
            device=x.device,
        )
        attentive.index_add_(0, batch, x * weights.unsqueeze(-1))
        attentive = attentive / weight_total.clamp_min(1e-8).unsqueeze(-1)

        maximum = torch.full_like(attentive, -torch.inf)
        maximum.scatter_reduce_(
            0,
            batch.unsqueeze(-1).expand_as(x),
            x,
            reduce="amax",
            include_self=True,
        )
        if not bool(torch.isfinite(maximum).all()):
            raise ValueError("Every graph must contain at least one finite region embedding")
        return torch.cat((attentive, maximum), dim=-1)

    @staticmethod
    def _candidate_indices(
        local_indices: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        counts = graph_node_counts(batch)
        offsets = graph_node_offsets(counts)
        return local_indices + offsets.unsqueeze(1)

    def initial_recurrent_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zero LSTM state for a new physical evacuation episode."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (batch_size, self.temporal_dim)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def _encode_graph(self, g: GNNInput):
        batch = _validated_batch(g.batch, int(g.x_ped.size(0)), g.x_ped.device)
        ped = F.gelu(self.ped_encoder(g.x_ped))
        hazard = F.gelu(self.hazard_encoder(g.x_hazard))
        infra = F.gelu(self.infra_encoder(g.x_infra))
        contextual = self.fusion(torch.cat((ped, hazard, infra), dim=-1))
        for layer in self.message_layers:
            contextual = layer(
                contextual,
                g.edge_index,
                g.route_edge_index,
                g.route_edge_weight,
            )

        graph_embed = self._graph_pool(contextual, batch)
        graph_count = int(graph_embed.size(0))
        if self.d_global > 0:
            if g.x_global is None:
                raise ValueError("x_global is required when d_global is positive")
            global_features = g.x_global.to(device=contextual.device, dtype=contextual.dtype)
            if global_features.shape != (graph_count, self.d_global):
                raise ValueError(
                    f"x_global must have shape {(graph_count, self.d_global)}, "
                    f"got {tuple(global_features.shape)}"
                )
            decision_context = torch.cat((graph_embed, global_features), dim=-1)
        else:
            decision_context = graph_embed

        counts = graph_node_counts(batch)
        if g.candidate_cell_index is None:
            if not bool(torch.all(counts == counts[0])):
                raise ValueError(
                    "candidate tensors are required for mixed-size graph batches"
                )
            action_count = int(counts[0].item())
            candidate_cell_index = torch.arange(
                action_count,
                dtype=torch.long,
                device=contextual.device,
            ).unsqueeze(0).expand(graph_count, -1)
            candidate_features = torch.zeros(
                graph_count,
                action_count,
                self.d_candidate,
                dtype=contextual.dtype,
                device=contextual.device,
            )
        else:
            candidate_cell_index = g.candidate_cell_index.to(
                device=contextual.device,
                dtype=torch.long,
            )
            candidate_features = g.candidate_features.to(
                device=contextual.device,
                dtype=contextual.dtype,
            )
            expected = (
                graph_count,
                candidate_cell_index.size(1),
                self.d_candidate,
            )
            if tuple(candidate_features.shape) != expected:
                raise ValueError(
                    f"candidate_features must have shape {expected}, got {tuple(candidate_features.shape)}"
                )

        global_candidate_index = self._candidate_indices(candidate_cell_index, batch)
        candidate_regions = contextual[global_candidate_index]
        return (
            batch,
            decision_context,
            candidate_regions,
            candidate_features,
            global_candidate_index,
        )

    @staticmethod
    def _outcomes_to_components(outcomes: torch.Tensor) -> torch.Tensor:
        """Map normalized physical outcomes to signed reward components."""
        return torch.stack(
            (
                DEFAULT_SAFE_COMPLETION_WEIGHT * outcomes[..., 0],
                -DEFAULT_CASUALTY_WEIGHT * outcomes[..., 1],
                -DEFAULT_EVACUATION_TIME_WEIGHT * outcomes[..., 2],
                -DEFAULT_HAZARD_EXPOSURE_WEIGHT * outcomes[..., 3],
            ),
            dim=-1,
        )

    def forward_nmcc_recurrent(
        self,
        g: GNNInput,
        recurrent_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        momentum_features: Optional[torch.Tensor] = None,
        *,
        causal_guidance_weight: float = 0.0,
        uncertainty_penalty: float = 1.0,
    ) -> NMCCPolicyOutput:
        """Evaluate one observation while advancing the episode LSTM state.

        Graph encoding remains resolution-independent.  The recurrent state is
        reset only by the caller at a true environmental episode boundary.
        Component critic values are signed and sum exactly to the scalar value
        used by PPO's actor advantage.
        """
        (
            batch,
            decision_context,
            candidate_regions,
            candidate_features,
            global_candidate_index,
        ) = self._encode_graph(g)
        graph_count = int(decision_context.size(0))
        if momentum_features is None:
            momentum = torch.zeros(
                (graph_count, self.d_momentum),
                dtype=decision_context.dtype,
                device=decision_context.device,
            )
        else:
            momentum = momentum_features.to(
                device=decision_context.device,
                dtype=decision_context.dtype,
            )
            if momentum.ndim == 1:
                momentum = momentum.unsqueeze(0)
            if momentum.shape != (graph_count, self.d_momentum):
                raise ValueError(
                    "momentum_features must have shape "
                    f"{(graph_count, self.d_momentum)}, got {tuple(momentum.shape)}"
                )
        if recurrent_state is None:
            recurrent_state = self.initial_recurrent_state(
                graph_count,
                device=decision_context.device,
                dtype=decision_context.dtype,
            )
        hidden, memory = recurrent_state
        expected_state = (graph_count, self.temporal_dim)
        if hidden.shape != expected_state or memory.shape != expected_state:
            raise ValueError(
                f"recurrent state must contain tensors shaped {expected_state}"
            )
        hidden, memory = self.temporal(
            torch.cat((decision_context, momentum), dim=-1),
            (hidden, memory),
        )

        candidate_hidden = (
            self.node_actor(candidate_regions)
            + self.candidate_actor(candidate_features)
            + F.gelu(self.global_actor_context(decision_context)).unsqueeze(1)
            + self.temporal_actor_context(hidden).unsqueeze(1)
        )
        learned_residual = self.residual_logit_bound * torch.tanh(
            self.actor_cell(F.gelu(candidate_hidden)).squeeze(-1)
        )

        active_population = g.x_ped[:, 0]
        candidate_active = active_population[global_candidate_index]
        active_maximum = candidate_active.amax(dim=1, keepdim=True)
        relative_active = torch.where(
            active_maximum > 0.0,
            candidate_active / active_maximum.clamp_min(1e-8),
            torch.zeros_like(candidate_active),
        )
        if self.actor_prior == "active_population":
            prior_feature = relative_active
        elif self.actor_prior in {"risk_time_reduction", "route_time_saving"}:
            saving = candidate_features[..., self.actor_prior_feature_index].clamp_min(0.0)
            saving_maximum = saving.amax(dim=1, keepdim=True)
            prior_feature = torch.where(
                saving_maximum > 0.0,
                saving / saving_maximum.clamp_min(1e-8),
                torch.zeros_like(saving),
            )
        else:
            prior_feature = torch.zeros_like(relative_active)
        base_logits = self.actor_prior_scale * prior_feature + learned_residual
        critic_hidden = self.critic_trunk(
            torch.cat((decision_context, hidden), dim=-1)
        )
        component_values = torch.stack(
            tuple(
                self.critic_heads[name](critic_hidden).squeeze(-1)
                for name in VALUE_COMPONENT_NAMES
            ),
            dim=-1,
        )
        value = component_values.sum(dim=-1)

        natural_raw = self.natural_outcome_head(critic_hidden)
        # Hard population conservation: over the branch horizon, every person
        # who is active at the decision is either newly safe, newly a casualty,
        # or still active. A softmax allocation enforces that identity exactly
        # instead of asking an auxiliary loss to discover it. Person-time and
        # exposure remain bounded independent forecasts. Final normalized risk
        # is constrained to [0.5 * final_active, final_active], matching
        # risk_mass = active * (1 + danger) divided by 2P.
        active_fraction = torch.zeros(
            graph_count,
            dtype=g.x_ped.dtype,
            device=g.x_ped.device,
        )
        active_fraction.index_add_(0, batch, g.x_ped[:, 0])
        active_fraction = active_fraction.clamp(0.0, 1.0)
        terminal_allocation = torch.softmax(
            natural_raw[:, (0, 1, 4)],
            dim=-1,
        ) * active_fraction.unsqueeze(-1)
        final_active = terminal_allocation[:, 2]
        final_risk = final_active * (
            0.5 + 0.5 * torch.sigmoid(natural_raw[:, 5])
        )
        natural_outcomes = torch.stack(
            (
                terminal_allocation[:, 0],
                terminal_allocation[:, 1],
                torch.sigmoid(natural_raw[:, 2]),
                torch.sigmoid(natural_raw[:, 3]),
                final_active,
                final_risk,
            ),
            dim=-1,
        )
        natural_components = self._outcomes_to_components(natural_outcomes)

        causal_outcome_samples = torch.stack(
            tuple(
                torch.tanh(head(F.gelu(candidate_hidden)))
                for head in self.causal_outcome_heads
            ),
            dim=1,
        )
        causal_component_samples = self._outcomes_to_components(
            causal_outcome_samples
        )
        causal_component_mean = causal_component_samples.mean(dim=1)
        causal_component_std = causal_component_samples.std(
            dim=1,
            unbiased=False,
        )
        causal_scalar_samples = causal_component_samples.sum(dim=-1)
        causal_scalar_mean = causal_scalar_samples.mean(dim=1)
        causal_scalar_std = causal_scalar_samples.std(dim=1, unbiased=False)
        robust_causal_score = causal_scalar_mean - float(uncertainty_penalty) * causal_scalar_std
        centered_score = robust_causal_score - robust_causal_score.mean(
            dim=1,
            keepdim=True,
        )
        score_scale = robust_causal_score.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)
        teacher_logits = centered_score / score_scale
        logits = base_logits + float(causal_guidance_weight) * teacher_logits
        improvement_hidden = F.gelu(
            self.improvement_node(candidate_regions)
            + self.improvement_candidate(candidate_features)
            + F.gelu(self.improvement_global_context(decision_context)).unsqueeze(1)
            + self.improvement_temporal_context(hidden).unsqueeze(1)
        )
        direct_features = torch.cat(
            (
                g.x_ped[global_candidate_index],
                g.x_hazard[global_candidate_index],
                g.x_infra[global_candidate_index],
                candidate_features,
                (
                    decision_context[:, -self.d_global :]
                    if self.d_global > 0
                    else decision_context.new_zeros((graph_count, 0))
                ).unsqueeze(1).expand(-1, candidate_features.size(1), -1),
                momentum.unsqueeze(1).expand(-1, candidate_features.size(1), -1),
            ),
            dim=-1,
        )
        improvement_value_samples = torch.stack(
            tuple(
                (
                    deep_head(improvement_hidden)
                    + direct_head(direct_features)
                ).squeeze(-1)
                for deep_head, direct_head in zip(
                    self.improvement_value_heads,
                    self.improvement_direct_heads,
                )
            ),
            dim=1,
        )

        if self.verbose:
            print(
                f"[EvacPolicy.forward] graphs={graph_count} nodes={g.x_ped.size(0)} "
                f"actions={logits.size(1)}"
            )
        return NMCCPolicyOutput(
            logits=logits,
            prior_logits=self.actor_prior_scale * prior_feature,
            value=value,
            value_components=component_values,
            learned_residual=learned_residual,
            recurrent_state=(hidden, memory),
            natural_outcomes=natural_outcomes,
            natural_components=natural_components,
            causal_outcome_samples=causal_outcome_samples,
            causal_component_samples=causal_component_samples,
            causal_component_mean=causal_component_mean,
            causal_component_std=causal_component_std,
            robust_causal_score=robust_causal_score,
            teacher_logits=teacher_logits,
            improvement_value_samples=improvement_value_samples,
        )

    def forward_recurrent(
        self,
        g: GNNInput,
        recurrent_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        momentum_features: Optional[torch.Tensor] = None,
    ):
        """Backward-compatible recurrent actor/critic view."""
        output = self.forward_nmcc_recurrent(
            g,
            recurrent_state,
            momentum_features,
        )
        return (
            output.logits,
            output.value,
            output.value_components,
            output.learned_residual,
            output.recurrent_state,
        )

    def forward_with_residual(self, g: GNNInput):
        logits, value, _, learned_residual, _ = self.forward_recurrent(g)
        return logits, value, learned_residual

    def forward_with_components(self, g: GNNInput):
        """Stateless one-frame convenience used by diagnostics and tests."""
        logits, value, components, _, _ = self.forward_recurrent(g)
        return logits, value, components

    def forward(self, g: GNNInput):
        logits, value, _ = self.forward_with_residual(g)
        return logits, value
