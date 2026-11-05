"""Strict no-recourse execution for time-zero static shelter schedules.

This evaluator-only bridge preserves the trained policy/dynamics fingerprint.
It changes only the two open-loop comparator arms: if the exact site committed
at time zero is unavailable at its scheduled epoch, the allocation is
forfeited without installing a substitute.  Later commitments retain their
original epochs.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from DecisionInterface import RegionalActionReceipt, RegionalObservation
from RLBridge import RLBridge


STRICT_STATIC_STRATEGIES = frozenset({"initial_only", "static_greedy"})


class StaticNoRecourseRLBridge(RLBridge):
    """RLBridge extension confined to strict static-comparator evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.deployment_strategy not in STRICT_STATIC_STRATEGIES:
            raise ValueError(
                "StaticNoRecourseRLBridge is only valid for initial_only or "
                "static_greedy"
            )
        self.static_precommit_capacities: tuple[float, ...] = tuple()
        self.static_precommit_site_metadata: tuple[dict[str, object], ...] = tuple()
        self.deployment_attempt_times: list[int] = []
        self.forfeited_deployment_allocations: list[dict[str, object]] = []
        self._forfeiture_context: Optional[dict[str, object]] = None
        self._check_forfeiture_due = False

    def precommit_static_schedule(self) -> tuple[tuple[str, int], ...]:
        candidate_metadata: dict[str, dict[str, object]] = {}
        for row in self.core.shelterDS.shelterCanByCell or ():
            for candidates in row:
                for candidate in candidates:
                    candidate_osm_id = str(getattr(candidate, "OSMID", ""))
                    capacity = float(
                        self.core.shelterDS.configuredShelterCapacity(candidate)
                    )
                    candidate_metadata[candidate_osm_id] = {
                        "candidate_osm_node_id": candidate_osm_id,
                        "candidate_x_m": float(getattr(candidate, "nodeX", float("nan"))),
                        "candidate_y_m": float(getattr(candidate, "nodeY", float("nan"))),
                        "planned_capacity": capacity,
                    }
        schedule = super().precommit_static_schedule()
        metadata = []
        for candidate_osm_id, cell_index in schedule:
            if str(candidate_osm_id) not in candidate_metadata:
                raise RuntimeError(
                    "Static precommitment metadata is missing candidate "
                    f"{candidate_osm_id!r}"
                )
            record = dict(candidate_metadata[str(candidate_osm_id)])
            record["candidate_cell_index"] = int(cell_index)
            metadata.append(record)
        self.static_precommit_site_metadata = tuple(metadata)
        self.static_precommit_capacities = tuple(
            float(record["planned_capacity"]) for record in metadata
        )
        return schedule

    def _enrich_policy_observation(
        self,
        observation: RegionalObservation,
    ) -> RegionalObservation:
        # The frozen site identity is already fixed.  When that site is gone,
        # no counterfactual candidate evaluation may create an implicit
        # substitute or try to install a physically absent building.
        if self._forfeiture_context is not None:
            return observation
        return super()._enrich_policy_observation(observation)

    def _build_with_committed_failure(
        self,
        original_build,
        *,
        decision_index: int,
        simulation_time: int,
        remaining_deployments: int,
    ) -> RegionalObservation:
        observation = original_build(
            decision_index=decision_index,
            simulation_time=simulation_time,
            remaining_deployments=remaining_deployments,
        )
        if not self._check_forfeiture_due:
            return observation
        schedule_index = int(self.deployments_made)
        if schedule_index >= len(self.static_precommit_schedule):
            return observation
        expected_osm_id, expected_cell = self.static_precommit_schedule[
            schedule_index
        ]
        exact_matches = [
            int(action)
            for action in np.flatnonzero(observation.action_mask)
            if str(observation.candidate_osm_node_ids[int(action)])
            == str(expected_osm_id)
            and int(observation.candidate_cell_indices[int(action)])
            == int(expected_cell)
        ]
        if len(exact_matches) > 1:
            raise RuntimeError(
                "Precommitted shelter identity is not unique at its scheduled "
                f"epoch: OSM={expected_osm_id!r}, cell={expected_cell}"
            )
        if exact_matches:
            return observation

        cell_slots = np.flatnonzero(
            np.asarray(observation.candidate_cell_indices, dtype=int)
            == int(expected_cell)
        )
        if cell_slots.size != 1:
            raise RuntimeError(
                "Precommitted shelter cell does not have one stable action slot: "
                f"cell={expected_cell}, slots={cell_slots.tolist()}"
            )
        action = int(cell_slots[0])
        metadata = dict(self.static_precommit_site_metadata[schedule_index])
        metadata.update(
            {
                "allocation_index": int(schedule_index),
                "simulation_time": int(simulation_time),
                "reason": "committed_site_unavailable",
                "synthetic_action_index": int(action),
            }
        )
        self._forfeiture_context = metadata

        action_mask = np.asarray(observation.action_mask, dtype=bool).copy()
        action_mask[action] = True
        candidate_osm_ids = np.asarray(
            observation.candidate_osm_node_ids,
            dtype=object,
        ).copy()
        candidate_osm_ids[action] = str(expected_osm_id)
        candidate_capacities = np.asarray(
            observation.candidate_capacities,
            dtype=np.float32,
        ).copy()
        candidate_capacities[action] = float(metadata["planned_capacity"])
        return replace(
            observation,
            action_mask=action_mask,
            candidate_osm_node_ids=candidate_osm_ids,
            candidate_capacities=candidate_capacities,
        )

    def _forfeiture_receipt(
        self,
        observation: RegionalObservation,
        decision,
    ) -> RegionalActionReceipt:
        if self._forfeiture_context is None:
            return self._original_executor_execute(observation, decision)
        context = self._forfeiture_context
        cell_index = int(context["candidate_cell_index"])
        cell_i, cell_j = divmod(cell_index, int(self.ny))
        return RegionalActionReceipt(
            observation_id=(observation.decision_index, observation.simulation_time),
            requested_candidate=int(decision.action_index),
            executed_candidate=-1,
            requested_cell=cell_index,
            executed_cell=-1,
            shelter_id=-1,
            candidate_osm_node_id=str(context["candidate_osm_node_id"]),
            candidate_x_m=float(context["candidate_x_m"]),
            candidate_y_m=float(context["candidate_y_m"]),
            candidate_cell_i=int(cell_i),
            candidate_cell_j=int(cell_j),
            capacity_added=0.0,
            rerouted_population=0,
            installed=False,
        )

    def step(self, *, simulation_time: int, is_terminal: bool = False) -> dict:
        before_deployments = int(self.deployments_made)
        self._forfeiture_context = None
        self._check_forfeiture_due = bool(
            not is_terminal
            and self.remaining_deployments > 0
            and self._decision_due(int(simulation_time))
        )
        original_build = self.observation_builder.build
        self._original_executor_execute = self.executor.execute

        def build_with_committed_failure(**kwargs):
            return self._build_with_committed_failure(original_build, **kwargs)

        self.observation_builder.build = build_with_committed_failure
        self.executor.execute = self._forfeiture_receipt
        try:
            result = super().step(
                simulation_time=simulation_time,
                is_terminal=is_terminal,
            )
        finally:
            self.observation_builder.build = original_build
            self.executor.execute = self._original_executor_execute

        if self._forfeiture_context is None:
            if int(self.deployments_made) > before_deployments:
                self.deployment_attempt_times.append(int(simulation_time))
            return result

        context = dict(self._forfeiture_context)
        synthetic_action = int(context.pop("synthetic_action_index"))
        heuristic_action = int(result.get("heuristic_candidate", -1))
        self.action_comparisons = max(0, int(self.action_comparisons) - 1)
        self.action_agreements = max(
            0,
            int(self.action_agreements)
            - int(synthetic_action == heuristic_action),
        )
        self.deferred_decisions = max(0, int(self.deferred_decisions) - 1)
        self.forfeited_deployment_allocations.append(context)
        self.deployment_attempt_times.append(int(simulation_time))
        self.deployments_made += 1
        self.tokens_remaining -= 1
        self.last_deployment_time = int(simulation_time)
        if (
            self.remaining_deployments == 0
            and bool(
                getattr(
                    self.core,
                    "requirePopulationMatchedShelterCapacity",
                    False,
                )
            )
        ):
            self.core.finalInstallationUnassignedRationalPopulation = int(
                self.core.pedDS.unassigned_rational_population()
            )

        result.update(
            {
                "deployment_allocation_forfeited": 1,
                "selected_candidate": -1,
                "selected_cell": int(context["candidate_cell_index"]),
                "added_shelters": 0,
                "installed_shelter_id": -1,
                "candidate_osm_node_id": str(
                    context["candidate_osm_node_id"]
                ),
                "candidate_x_m": float(context["candidate_x_m"]),
                "candidate_y_m": float(context["candidate_y_m"]),
                "capacity_added": 0.0,
                "rerouted_population": 0,
                "remaining_deployments": int(self.remaining_deployments),
            }
        )
        return result

