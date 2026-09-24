#!/usr/bin/env python3
"""Validated experiment and reporting specification for the E0--E6 study.

This module deliberately contains no simulator imports.  It defines the
statistical experiment contract that produces the paper figures and validates
that a versioned JSON specification is internally coherent before expensive
map-backed experiments are launched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_SUITE_PATH = PROJECT_ROOT / "config" / "full_experiment_suite.json"


@dataclass(frozen=True)
class OutputTableSpec:
    """One machine-readable result table required by the reporting pipeline."""

    table_id: str
    path: str
    experiment: str
    required_for_complete_suite: bool
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class FigureFamily:
    """One prespecified figure family and its source-table dependencies."""

    figure_id: str
    title: str
    experiments: tuple[str, ...]
    source_tables: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSuite:
    """Immutable, validated E0--E6 experiment specification."""

    schema_version: int
    suite_id: str
    description: str
    city_profiles: str
    policy_seeds: int
    train_episodes_per_city: int
    bootstrap_draws: int
    confidence_level: float
    primary_metric: str
    primary_comparison: str
    city_weighting: str
    factors: Mapping[str, tuple[str, ...]]
    replications_per_factor_cell: int
    population_levels: tuple[int, ...]
    shelter_candidate_levels: tuple[int, ...]
    scale_horizon_timesteps: int
    scale_shelter_action_interval: int
    scale_replications_per_cell: int
    experiments: Mapping[str, Mapping]
    output_tables: Mapping[str, OutputTableSpec]
    figure_families: tuple[FigureFamily, ...]
    source_path: Path
    source_sha256: str

    @property
    def factor_cells(self) -> tuple[dict[str, str], ...]:
        """Return the prespecified evaluation factorial in stable order."""
        names = tuple(self.factors)
        return tuple(
            dict(zip(names, values))
            for values in itertools.product(*(self.factors[name] for name in names))
        )

    @property
    def evaluation_scenarios_per_city(self) -> int:
        return len(self.factor_cells) * int(self.replications_per_factor_cell)

    def table_path(self, launch_dir: Path, table_id: str) -> Path:
        try:
            relative = self.output_tables[table_id].path
        except KeyError as error:
            raise KeyError(f"Unknown experiment output table {table_id!r}") from error
        return launch_dir / relative


def _nonempty_string(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _string_tuple(values: Iterable, name: str) -> tuple[str, ...]:
    result = tuple(_nonempty_string(value, name) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique, non-empty values")
    return result


def _parse_output_tables(value: Mapping) -> dict[str, OutputTableSpec]:
    result: dict[str, OutputTableSpec] = {}
    required_fields = {
        "path",
        "experiment",
        "required_for_complete_suite",
        "required_columns",
    }
    for table_id, raw in value.items():
        if set(raw) != required_fields:
            raise ValueError(
                f"Output table {table_id!r} fields do not match schema: {sorted(raw)}"
            )
        identifier = _nonempty_string(table_id, "table_id")
        path = _nonempty_string(raw["path"], f"{identifier}.path")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{identifier}.path must stay relative to the launch directory")
        result[identifier] = OutputTableSpec(
            table_id=identifier,
            path=path,
            experiment=_nonempty_string(raw["experiment"], f"{identifier}.experiment"),
            required_for_complete_suite=bool(raw["required_for_complete_suite"]),
            required_columns=_string_tuple(
                raw["required_columns"], f"{identifier}.required_columns"
            ),
        )
    return result


def _parse_figure_families(
    value: Iterable[Mapping], output_tables: Mapping[str, OutputTableSpec]
) -> tuple[FigureFamily, ...]:
    required_fields = {"figure_id", "title", "experiments", "source_tables"}
    result = []
    for raw in value:
        if set(raw) != required_fields:
            raise ValueError(f"Figure-family fields do not match schema: {sorted(raw)}")
        family = FigureFamily(
            figure_id=_nonempty_string(raw["figure_id"], "figure_id"),
            title=_nonempty_string(raw["title"], "figure title"),
            experiments=_string_tuple(raw["experiments"], "figure experiments"),
            source_tables=_string_tuple(raw["source_tables"], "figure source_tables"),
        )
        unknown = sorted(set(family.source_tables).difference(output_tables))
        if unknown:
            raise ValueError(
                f"Figure {family.figure_id!r} references unknown tables: {unknown}"
            )
        result.append(family)
    identifiers = [family.figure_id for family in result]
    if not result or len(identifiers) != len(set(identifiers)):
        raise ValueError("figure_families must have unique identifiers")
    return tuple(result)


def load_experiment_suite(
    path: str | Path = DEFAULT_EXPERIMENT_SUITE_PATH,
) -> ExperimentSuite:
    """Load and fully validate a versioned full-experiment specification."""
    resolved = Path(path).expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    required = {
        "schema_version",
        "suite_id",
        "description",
        "city_profiles",
        "confirmatory_analysis",
        "evaluation_factorial",
        "scale_stress_design",
        "experiments",
        "output_tables",
        "figure_families",
    }
    if set(payload) != required:
        raise ValueError(
            "Experiment-suite top-level fields do not match schema: "
            f"missing={sorted(required.difference(payload))}, "
            f"extra={sorted(set(payload).difference(required))}"
        )
    if int(payload["schema_version"]) != 2:
        raise ValueError("Unsupported experiment-suite schema_version")

    analysis = payload["confirmatory_analysis"]
    analysis_fields = {
        "policy_seeds",
        "train_episodes_per_city",
        "bootstrap_draws",
        "confidence_level",
        "primary_metric",
        "primary_comparison",
        "city_weighting",
    }
    if set(analysis) != analysis_fields:
        raise ValueError("confirmatory_analysis fields do not match schema")
    policy_seeds = int(analysis["policy_seeds"])
    train_episodes = int(analysis["train_episodes_per_city"])
    draws = int(analysis["bootstrap_draws"])
    confidence = float(analysis["confidence_level"])
    if policy_seeds < 5:
        raise ValueError("The confirmatory design requires at least five policy seeds")
    if train_episodes <= 0 or draws < 1000 or not 0.5 < confidence < 1.0:
        raise ValueError("Invalid training, bootstrap, or confidence specification")

    factorial = payload["evaluation_factorial"]
    if set(factorial) != {"factors", "replications_per_factor_cell"}:
        raise ValueError("evaluation_factorial fields do not match schema")
    factors = {
        _nonempty_string(name, "factor name"): _string_tuple(levels, str(name))
        for name, levels in factorial["factors"].items()
    }
    if set(factors) != {"capacity_regime", "hazard_regime", "demand_pattern"}:
        raise ValueError(
            "The primary factorial must contain capacity_regime, hazard_regime, "
            "and demand_pattern"
        )
    replications = int(factorial["replications_per_factor_cell"])
    if replications <= 0:
        raise ValueError("replications_per_factor_cell must be positive")

    scale = payload["scale_stress_design"]
    scale_fields = {
        "population_levels",
        "shelter_candidate_levels",
        "horizon_timesteps",
        "shelter_action_interval",
        "replications_per_cell",
    }
    if set(scale) != scale_fields:
        raise ValueError("scale_stress_design fields do not match schema")
    population_levels = tuple(int(value) for value in scale["population_levels"])
    candidate_levels = tuple(int(value) for value in scale["shelter_candidate_levels"])
    if (
        len(population_levels) != 5
        or len(candidate_levels) != 5
        or population_levels != tuple(sorted(set(population_levels)))
        or candidate_levels != tuple(sorted(set(candidate_levels)))
        or any(value <= 0 for value in (*population_levels, *candidate_levels))
    ):
        raise ValueError(
            "Scale stress requires exactly five strictly increasing positive "
            "population and shelter-candidate levels"
        )
    scale_horizon = int(scale["horizon_timesteps"])
    scale_action_interval = int(scale["shelter_action_interval"])
    scale_replications = int(scale["replications_per_cell"])
    if (
        scale_horizon != 60
        or scale_action_interval != 2
        or scale_replications <= 0
    ):
        raise ValueError(
            "Scale stress must use a 60-transition horizon, a two-transition "
            "shelter-action interval, and a positive replication count"
        )

    experiments = payload["experiments"]
    expected_experiments = {f"E{index}" for index in range(7)}
    if set(experiments) != expected_experiments:
        raise ValueError("experiments must define exactly E0 through E6")
    for experiment_id, experiment in experiments.items():
        if set(experiment) != {"title", "research_question", "design", "outputs"}:
            raise ValueError(f"{experiment_id} fields do not match schema")
        _nonempty_string(experiment["title"], f"{experiment_id}.title")
        _nonempty_string(experiment["research_question"], f"{experiment_id}.research_question")
        if not isinstance(experiment["design"], Mapping) or not experiment["design"]:
            raise ValueError(f"{experiment_id}.design must be a non-empty object")
        _string_tuple(experiment["outputs"], f"{experiment_id}.outputs")

    output_tables = _parse_output_tables(payload["output_tables"])
    for table in output_tables.values():
        if table.experiment not in expected_experiments:
            raise ValueError(
                f"Output table {table.table_id!r} has unknown experiment {table.experiment!r}"
            )
    declared_outputs = {
        output
        for experiment in experiments.values()
        for output in experiment["outputs"]
    }
    if declared_outputs != set(output_tables):
        raise ValueError(
            "Experiment outputs and output_tables differ: "
            f"declared_only={sorted(declared_outputs.difference(output_tables))}, "
            f"tables_only={sorted(set(output_tables).difference(declared_outputs))}"
        )
    figures = _parse_figure_families(payload["figure_families"], output_tables)
    for figure in figures:
        unknown = sorted(set(figure.experiments).difference(expected_experiments))
        if unknown:
            raise ValueError(f"Figure {figure.figure_id!r} has unknown experiments: {unknown}")

    return ExperimentSuite(
        schema_version=2,
        suite_id=_nonempty_string(payload["suite_id"], "suite_id"),
        description=_nonempty_string(payload["description"], "description"),
        city_profiles=_nonempty_string(payload["city_profiles"], "city_profiles"),
        policy_seeds=policy_seeds,
        train_episodes_per_city=train_episodes,
        bootstrap_draws=draws,
        confidence_level=confidence,
        primary_metric=_nonempty_string(analysis["primary_metric"], "primary_metric"),
        primary_comparison=_nonempty_string(
            analysis["primary_comparison"], "primary_comparison"
        ),
        city_weighting=_nonempty_string(analysis["city_weighting"], "city_weighting"),
        factors=factors,
        replications_per_factor_cell=replications,
        population_levels=population_levels,
        shelter_candidate_levels=candidate_levels,
        scale_horizon_timesteps=scale_horizon,
        scale_shelter_action_interval=scale_action_interval,
        scale_replications_per_cell=scale_replications,
        experiments=experiments,
        output_tables=output_tables,
        figure_families=figures,
        source_path=resolved,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
