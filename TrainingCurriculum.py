#!/usr/bin/env python3
"""Validated, deterministic curricula for pooled evacuation PPO training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from NMCCPIConfig import NMCC_PI_CORE_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAINING_CURRICULUM_PATH = (
    PROJECT_ROOT / "config" / "staged_training_curriculum.json"
)

# Scenario fields may vary by stage. Learner fields are also accepted so a
# curriculum file can be a complete, executable NMCC specification, but the
# loader requires them to remain identical in every stage/variant because a
# resumed PPO checkpoint must retain one exact architecture and optimizer
# contract.
LEARNER_STAGE_OVERRIDES = frozenset(
    {
        "nmccEnabled",
        "nmccCounterfactualHorizon",
        "nmccCounterfactualWeight",
        "nmccJointCounterfactualWeight",
        "nmccInterventionCost",
        "nmccEnsembleSize",
        "nmccNaturalLossCoefficient",
        "nmccCausalLossCoefficient",
        "nmccDuelingLossCoefficient",
        "nmccTeacherCoefficient",
        "nmccTeacherDecayUpdates",
        "nmccGuidanceMaximum",
        "nmccGuidanceWarmupUpdates",
        "nmccGuidanceRampUpdates",
        "nmccUncertaintyPenalty",
        "entropyCoefficientEnd",
        "explorationDecayUpdates",
        "actionTemperatureStart",
        "actionTemperatureEnd",
        "nmccNaturalPretrainRollouts",
        "nmccCausalPretrainRollouts",
        "nmccControllerWarmupRollouts",
        "actorLearningRate",
        "criticLearningRate",
        "actorPpoEpochs",
        "criticPpoEpochs",
        "actorBaselineDecay",
        "advantageScaleFloor",
        "shelterCapacityToken",
        "ppoRolloutEpisodes",
        "requireCandidateOperationalBenefit",
        "minimumCandidateReroutableFraction",
        "minimumCandidateRouteTimeSaving",
        "minimumCandidateHazardSafetyMargin",
        "maximumShelterForecastDanger",
    }
    | {attribute for attribute, _, _, _ in NMCC_PI_CORE_FIELDS}
)
ALLOWED_STAGE_OVERRIDES = frozenset(
    {
        "pedVol",
        "pedestrianGroupSize",
        "hazardVol",
        "shelterCanVol",
        "initShelterVol",
        "maxAdditionalShelters",
        "hazardCasualtyRate",
        "hazardSpreadRate",
        "hazardSpeedReduct",
        "panicRate",
    }
) | LEARNER_STAGE_OVERRIDES


@dataclass(frozen=True)
class CurriculumVariant:
    variant_id: str
    weight: int
    overrides: Mapping[str, object]


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    label: str
    episodes_per_city: int
    variants: tuple[CurriculumVariant, ...]

    @property
    def variant_weight_total(self) -> int:
        return sum(variant.weight for variant in self.variants)


@dataclass(frozen=True)
class TrainingCurriculum:
    schema_version: int
    curriculum_id: str
    description: str
    stages: tuple[CurriculumStage, ...]
    source_path: Path
    source_sha256: str

    @property
    def episodes_per_city(self) -> int:
        return sum(stage.episodes_per_city for stage in self.stages)

    @property
    def learner_overrides(self) -> dict[str, object]:
        """Immutable learner and decision-interface contract for this run."""
        if not self.stages or not self.stages[0].variants:
            return {}
        overrides = self.stages[0].variants[0].overrides
        return {
            key: overrides[key]
            for key in sorted(LEARNER_STAGE_OVERRIDES)
            if key in overrides
        }

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "curriculum_id": self.curriculum_id,
            "description": self.description,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "episodes_per_city": self.episodes_per_city,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "label": stage.label,
                    "episodes_per_city": stage.episodes_per_city,
                    "variants": [
                        {
                            "variant_id": variant.variant_id,
                            "weight": variant.weight,
                            "overrides": dict(variant.overrides),
                        }
                        for variant in stage.variants
                    ],
                }
                for stage in self.stages
            ],
        }


@dataclass(frozen=True)
class CurriculumEpisode:
    city: object
    stage_index: int
    stage_id: str
    stage_label: str
    stage_replication: int
    stage_city_replication: int
    city_training_replication: int
    variant_id: str
    stage_overrides: Mapping[str, object]

    def manifest_dict(self) -> dict:
        return {
            "city_id": str(self.city.city_id),
            "stage_index": self.stage_index,
            "stage_id": self.stage_id,
            "stage_label": self.stage_label,
            "stage_replication": self.stage_replication,
            "stage_city_replication": self.stage_city_replication,
            "city_training_replication": self.city_training_replication,
            "variant_id": self.variant_id,
            "stage_overrides": dict(self.stage_overrides),
        }


def _nonempty(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _variant_cycle(stage: CurriculumStage) -> tuple[CurriculumVariant, ...]:
    """Return a deterministic smooth weighted cycle for one city.

    Smooth weighted round-robin interleaves a dominant nominal variant with
    perturbations instead of placing all nominal episodes first.  The complete
    cycle still realizes every declared integer weight exactly.
    """
    current = [0] * len(stage.variants)
    total = stage.variant_weight_total
    output = []
    for _ in range(total):
        for index, variant in enumerate(stage.variants):
            current[index] += variant.weight
        selected = max(
            range(len(stage.variants)),
            key=lambda index: (current[index], -index),
        )
        current[selected] -= total
        output.append(stage.variants[selected])
    return tuple(output)


def load_training_curriculum(
    path: str | Path = DEFAULT_TRAINING_CURRICULUM_PATH,
) -> TrainingCurriculum:
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    schema_version = int(payload.get("schema_version", -1))
    required = {"schema_version", "curriculum_id", "description", "stages"}
    if schema_version == 2:
        required.add("learner_overrides")
    if set(payload) != required:
        raise ValueError("Training curriculum top-level fields do not match schema")
    if schema_version not in {1, 2}:
        raise ValueError("Unsupported training curriculum schema_version")
    common_learner_overrides = (
        {} if schema_version == 1 else dict(payload["learner_overrides"])
    )
    unknown_learner = sorted(
        set(common_learner_overrides).difference(LEARNER_STAGE_OVERRIDES)
    )
    if unknown_learner:
        raise ValueError(
            f"Curriculum learner_overrides contains non-learner fields: {unknown_learner}"
        )
    stages = []
    learner_contract = None
    stage_ids = set()
    for raw_stage in payload["stages"]:
        if set(raw_stage) != {"stage_id", "label", "episodes_per_city", "variants"}:
            raise ValueError("Training curriculum stage fields do not match schema")
        stage_id = _nonempty(raw_stage["stage_id"], "stage_id")
        if stage_id in stage_ids:
            raise ValueError(f"Duplicate curriculum stage_id {stage_id!r}")
        stage_ids.add(stage_id)
        variants = []
        variant_ids = set()
        for raw_variant in raw_stage["variants"]:
            if set(raw_variant) != {"variant_id", "weight", "overrides"}:
                raise ValueError("Curriculum variant fields do not match schema")
            variant_id = _nonempty(raw_variant["variant_id"], "variant_id")
            if variant_id in variant_ids:
                raise ValueError(
                    f"Duplicate variant_id {variant_id!r} in stage {stage_id!r}"
                )
            variant_ids.add(variant_id)
            weight = int(raw_variant["weight"])
            if weight <= 0:
                raise ValueError("Curriculum variant weights must be positive")
            variant_overrides = dict(raw_variant["overrides"])
            conflicts = sorted(
                key for key in variant_overrides
                if key in common_learner_overrides
                and variant_overrides[key] != common_learner_overrides[key]
            )
            if conflicts:
                raise ValueError(
                    "Variant overrides conflict with the common learner contract: "
                    f"{conflicts}"
                )
            overrides = {**common_learner_overrides, **variant_overrides}
            unknown = sorted(set(overrides).difference(ALLOWED_STAGE_OVERRIDES))
            if unknown:
                raise ValueError(
                    f"Stage {stage_id!r} changes protected configuration fields: {unknown}"
                )
            if "pedVol" not in overrides:
                raise ValueError(
                    f"Every variant in stage {stage_id!r} must declare pedVol"
                )
            if int(overrides["pedVol"]) <= 0:
                raise ValueError("Curriculum pedVol must be positive")
            variant_learner_contract = {
                key: overrides[key]
                for key in sorted(LEARNER_STAGE_OVERRIDES)
                if key in overrides
            }
            if learner_contract is None:
                learner_contract = variant_learner_contract
            elif variant_learner_contract != learner_contract:
                raise ValueError(
                    "Learner-stage overrides must be identical in every "
                    "curriculum stage and variant"
                )
            variants.append(
                CurriculumVariant(
                    variant_id=variant_id,
                    weight=weight,
                    overrides=overrides,
                )
            )
        episodes_per_city = int(raw_stage["episodes_per_city"])
        if episodes_per_city <= 0 or not variants:
            raise ValueError("Every curriculum stage needs positive episodes and variants")
        stage = CurriculumStage(
            stage_id=stage_id,
            label=_nonempty(raw_stage["label"], f"{stage_id}.label"),
            episodes_per_city=episodes_per_city,
            variants=tuple(variants),
        )
        if episodes_per_city % stage.variant_weight_total != 0:
            raise ValueError(
                f"Stage {stage_id!r} episodes_per_city must be a multiple of its "
                "variant weight total"
            )
        stages.append(stage)
    if not stages:
        raise ValueError("Training curriculum requires at least one stage")
    return TrainingCurriculum(
        schema_version=schema_version,
        curriculum_id=_nonempty(payload["curriculum_id"], "curriculum_id"),
        description=_nonempty(payload["description"], "description"),
        stages=tuple(stages),
        source_path=resolved,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_curriculum_schedule(
    cities: Sequence[object],
    curriculum: TrainingCurriculum,
    *,
    launch_seed: int,
    rollout_episodes: int,
) -> tuple[CurriculumEpisode, ...]:
    cities = tuple(cities)
    if not cities:
        raise ValueError("Curriculum scheduling requires at least one city")
    rollout_episodes = int(rollout_episodes)
    if rollout_episodes <= 0 or rollout_episodes % len(cities) != 0:
        raise ValueError("rollout_episodes must contain complete city blocks")
    city_total = {str(city.city_id): 0 for city in cities}
    schedule = []
    for stage_index, stage in enumerate(curriculum.stages, start=1):
        stage_total = stage.episodes_per_city * len(cities)
        if stage_total % rollout_episodes != 0:
            raise ValueError(
                f"Stage {stage.stage_id!r} does not end on a PPO rollout boundary"
            )
        variant_cycle = _variant_cycle(stage)
        stage_city_count = {str(city.city_id): 0 for city in cities}
        stage_replication = 0
        for cycle in range(stage.episodes_per_city):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(launch_seed), 73, stage_index, cycle])
            )
            for city_index in rng.permutation(len(cities)):
                city = cities[int(city_index)]
                city_id = str(city.city_id)
                stage_city_count[city_id] += 1
                city_total[city_id] += 1
                stage_replication += 1
                variant = variant_cycle[
                    (stage_city_count[city_id] - 1) % len(variant_cycle)
                ]
                schedule.append(
                    CurriculumEpisode(
                        city=city,
                        stage_index=stage_index,
                        stage_id=stage.stage_id,
                        stage_label=stage.label,
                        stage_replication=stage_replication,
                        stage_city_replication=stage_city_count[city_id],
                        city_training_replication=city_total[city_id],
                        variant_id=variant.variant_id,
                        stage_overrides=dict(variant.overrides),
                    )
                )
    expected = curriculum.episodes_per_city
    if any(count != expected for count in city_total.values()):
        raise AssertionError(f"Internal curriculum city imbalance: {city_total}")
    return tuple(schedule)
