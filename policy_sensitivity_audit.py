#!/usr/bin/env python3
"""Counterfactual audit for a trained regional shelter-placement policy.

The audit does not substitute for held-out simulator evaluation.  It verifies
that a saved actor is numerically responsive to each dynamic state family that
the decision model claims to use, after removing the fixed active-population
prior from the learned residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from DecisionInterface import (
    CANDIDATE_FEATURE_NAMES,
    CELL_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    HAZARD_FEATURE_NAMES,
    HAZARD_FEATURE_SLICE,
    INFRA_FEATURE_NAMES,
    INFRA_FEATURE_SLICE,
    PED_FEATURE_NAMES,
    PED_FEATURE_SLICE,
)
from GNN import EvacPolicy, fit_gnn, grid_edge_index


AUDIT_SCHEMA_VERSION = 2
# Float32 round-off in a two-logit contrast is commonly around 1e-6.  Requiring
# two orders of magnitude more movement avoids labeling numerical noise as a
# behavioral response.
DEFAULT_RESPONSE_TOLERANCE = 1e-4


def _load_payload(path: Path) -> Mapping:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if "model_signature" not in payload or "policy_state_dict" not in payload:
        raise ValueError("checkpoint is missing its model signature or policy weights")
    return payload


def load_policy(path: str | Path) -> tuple[EvacPolicy, Mapping]:
    checkpoint = Path(path).expanduser().resolve()
    payload = _load_payload(checkpoint)
    signature = payload["model_signature"]
    if tuple(signature.get("cell_features", ())) != tuple(CELL_FEATURE_NAMES):
        raise ValueError("checkpoint cell features do not match the current decision interface")
    if tuple(signature.get("global_features", ())) != tuple(GLOBAL_FEATURE_NAMES):
        raise ValueError("checkpoint global features do not match the current decision interface")
    if tuple(signature.get("candidate_features", ())) != tuple(CANDIDATE_FEATURE_NAMES):
        raise ValueError("checkpoint candidate features do not match the current decision interface")
    policy = EvacPolicy(
        d_ped=len(PED_FEATURE_NAMES),
        d_hazard=len(HAZARD_FEATURE_NAMES),
        d_infra=len(INFRA_FEATURE_NAMES),
        d_global=len(GLOBAL_FEATURE_NAMES),
        d_candidate=len(CANDIDATE_FEATURE_NAMES),
        verbose=False,
    )
    policy.load_state_dict(payload["policy_state_dict"], strict=True)
    policy.eval()
    return policy, signature


def _policy_outputs(
    policy: EvacPolicy,
    grid_shape: tuple[int, int],
    cells: np.ndarray,
    global_features: np.ndarray,
    candidate_features: np.ndarray,
    *,
    route_edge_index: torch.Tensor | None = None,
    route_edge_weight: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = grid_shape
    cell_tensor = torch.as_tensor(cells, dtype=torch.float32)
    global_tensor = torch.as_tensor(global_features, dtype=torch.float32)
    edge_index = torch.as_tensor(grid_edge_index(nx, ny), dtype=torch.long)
    graph = fit_gnn(
        x_ped=cell_tensor[:, PED_FEATURE_SLICE],
        x_hazard=cell_tensor[:, HAZARD_FEATURE_SLICE],
        x_infra=cell_tensor[:, INFRA_FEATURE_SLICE],
        x_global=global_tensor,
        edge_index=edge_index,
        route_edge_index=route_edge_index,
        route_edge_weight=route_edge_weight,
        candidate_cell_index=torch.arange(cells.shape[0], dtype=torch.long),
        candidate_features=torch.as_tensor(candidate_features, dtype=torch.float32),
    )
    with torch.no_grad():
        logits, _ = policy(graph)
    logits_np = logits.squeeze(0).cpu().numpy().astype(np.float64)
    active = cells[:, 0].astype(np.float64)
    active_max = float(np.max(active))
    prior = (
        np.zeros_like(active)
        if active_max <= 0.0
        else float(policy.HEURISTIC_PRIOR_SCALE) * active / active_max
    )
    return logits_np, logits_np - prior


def _canonical_inputs(num_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.full((num_cells, len(CELL_FEATURE_NAMES)), 0.5, dtype=np.float64)
    cells[:, CELL_FEATURE_NAMES.index("active_population_fraction")] = 0.4
    cells[:, CELL_FEATURE_NAMES.index("mobility_delay_fraction")] = 0.25
    cells[:, CELL_FEATURE_NAMES.index("danger")] = 0.1
    cells[:, CELL_FEATURE_NAMES.index("remaining_shelter_capacity_fraction")] = 0.4
    global_features = np.full(len(GLOBAL_FEATURE_NAMES), 0.5, dtype=np.float64)
    candidate_features = np.full(
        (num_cells, len(CANDIDATE_FEATURE_NAMES)),
        0.5,
        dtype=np.float64,
    )
    return cells, global_features, candidate_features


def audit_policy(
    checkpoint_path: str | Path,
    *,
    response_tolerance: float = DEFAULT_RESPONSE_TOLERANCE,
) -> dict:
    tolerance = float(response_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("response_tolerance must be finite and positive")
    policy, signature = load_policy(checkpoint_path)
    grid_shape = tuple(int(value) for value in signature["grid_shape"])
    num_cells = int(grid_shape[0] * grid_shape[1])
    cells, global_features, candidate_features = _canonical_inputs(num_cells)
    probe, control = 0, num_cells - 1
    base_logits, base_residual = _policy_outputs(
        policy,
        grid_shape,
        cells,
        global_features,
        candidate_features,
    )

    local_results = {}
    base_total_contrast = float(base_logits[probe] - base_logits[control])
    base_residual_contrast = float(base_residual[probe] - base_residual[control])
    for name in CELL_FEATURE_NAMES:
        changed = cells.copy()
        index = CELL_FEATURE_NAMES.index(name)
        increment = 0.35 if changed[probe, index] <= 0.6 else -0.35
        changed[probe, index] = np.clip(changed[probe, index] + increment, 0.0, 1.0)
        logits, residual = _policy_outputs(
            policy,
            grid_shape,
            changed,
            global_features,
            candidate_features,
        )
        total_delta = float((logits[probe] - logits[control]) - base_total_contrast)
        residual_delta = float(
            (residual[probe] - residual[control]) - base_residual_contrast
        )
        local_results[name] = {
            "perturbation": float(changed[probe, index] - cells[probe, index]),
            "total_logit_contrast_delta": total_delta,
            "learned_residual_contrast_delta": residual_delta,
            "learned_response_detected": bool(abs(residual_delta) > tolerance),
        }

    # A global feature can only affect the ranking through its interaction with
    # heterogeneous local embeddings. Introduce fixed local asymmetry, then
    # compare the probe-control contrast before and after each global change.
    heterogeneous = cells.copy()
    heterogeneous[probe, CELL_FEATURE_NAMES.index("remaining_shelter_capacity_fraction")] = 0.2
    heterogeneous[control, CELL_FEATURE_NAMES.index("remaining_shelter_capacity_fraction")] = 0.8
    _, heterogeneous_residual = _policy_outputs(
        policy,
        grid_shape,
        heterogeneous,
        global_features,
        candidate_features,
    )
    heterogeneous_contrast = float(
        heterogeneous_residual[probe] - heterogeneous_residual[control]
    )
    global_results = {}
    for index, name in enumerate(GLOBAL_FEATURE_NAMES):
        changed_global = global_features.copy()
        changed_global[index] = np.clip(changed_global[index] + 0.4, 0.0, 1.0)
        _, residual = _policy_outputs(
            policy,
            grid_shape,
            heterogeneous,
            changed_global,
            candidate_features,
        )
        delta = float(
            (residual[probe] - residual[control]) - heterogeneous_contrast
        )
        global_results[name] = {
            "perturbation": float(changed_global[index] - global_features[index]),
            "learned_residual_contrast_delta": delta,
            "learned_response_detected": bool(abs(delta) > tolerance),
        }

    candidate_results = {}
    for index, name in enumerate(CANDIDATE_FEATURE_NAMES):
        changed_candidates = candidate_features.copy()
        increment = 0.35 if changed_candidates[probe, index] <= 0.6 else -0.35
        changed_candidates[probe, index] = np.clip(
            changed_candidates[probe, index] + increment,
            0.0,
            1.0,
        )
        _, residual = _policy_outputs(
            policy,
            grid_shape,
            cells,
            global_features,
            changed_candidates,
        )
        delta = float(
            (residual[probe] - residual[control]) - base_residual_contrast
        )
        candidate_results[name] = {
            "perturbation": float(
                changed_candidates[probe, index] - candidate_features[probe, index]
            ),
            "learned_residual_contrast_delta": delta,
            "learned_response_detected": bool(abs(delta) > tolerance),
        }

    route_edges = torch.tensor(
        [[probe, control], [control, probe]],
        dtype=torch.long,
    )
    route_weights = torch.tensor([0.8, 0.2], dtype=torch.float32)
    _, routed_residual = _policy_outputs(
        policy,
        grid_shape,
        cells,
        global_features,
        candidate_features,
        route_edge_index=route_edges,
        route_edge_weight=route_weights,
    )
    route_delta = float(
        (routed_residual[probe] - routed_residual[control]) - base_residual_contrast
    )

    def any_local_response(names) -> bool:
        return any(local_results[name]["learned_response_detected"] for name in names)

    population_response = abs(
        local_results["active_population_fraction"]["total_logit_contrast_delta"]
    ) > tolerance
    state_family_checks = {
        "dynamic_population": bool(population_response),
        "routes_mobility_and_wellness": any_local_response(
            tuple(name for name in PED_FEATURE_NAMES if name != "active_population_fraction")
        ),
        "hazard_and_forecast": any_local_response(HAZARD_FEATURE_NAMES),
        "physical_environment_and_shelters": any_local_response(INFRA_FEATURE_NAMES),
        "candidate_site_attributes": any(
            item["learned_response_detected"] for item in candidate_results.values()
        ),
        "global_operational_context": any(
            item["learned_response_detected"] for item in global_results.values()
        ),
        "route_assignment_relation": bool(abs(route_delta) > tolerance),
    }

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "model_version": int(signature["version"]),
        "architecture": signature["architecture"],
        "grid_shape": list(grid_shape),
        "response_tolerance": tolerance,
        "interpretation": (
            "Finite counterfactual logit responses establish numerical use of a state "
            "family; they do not establish causal correctness or held-out superiority."
        ),
        "local_feature_counterfactuals": local_results,
        "candidate_feature_counterfactuals": candidate_results,
        "global_feature_counterfactuals": global_results,
        "route_relation_counterfactual": {
            "learned_residual_contrast_delta": route_delta,
            "learned_response_detected": bool(abs(route_delta) > tolerance),
        },
        "state_family_checks": state_family_checks,
        "all_required_state_families_responsive": bool(all(state_family_checks.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a regional_policy.pt checkpoint")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument(
        "--response-tolerance",
        type=float,
        default=DEFAULT_RESPONSE_TOLERANCE,
    )
    args = parser.parse_args()
    result = audit_policy(args.checkpoint, response_tolerance=args.response_tolerance)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
