"""Shared v65 test fixtures: regime-enabled testbeds and an independent digest."""

from __future__ import annotations

import hashlib

import numpy as np

import CounterfactualBranch as CB
import nmcc_testbed
from DecisionInterface import RegionalObservationBuilder
from HazardDatabase import HazardRegimeConfig, HazardRegimeProcess

PAPER_SPREAD = (1, 2)  # RLEvacuationParameter.csv: spread rate mean 1 %, variance 2
STRESS_SPREAD = (30, 25)  # nmcc_testbed default: fast-burning stress case


def regime_testbed(
    *,
    seed: int = 6501,
    population: int = 240,
    grid: int = 12,
    cells: int = 8,
    spacing_m: float = 250.0,
    spread=PAPER_SPREAD,
    stop_time: int = 61,
    interval: int = 10,
    hazard_count: int = 3,
):
    """Testbed with the regime process on, region+defer actions."""
    core = nmcc_testbed.build(
        grid=grid,
        cell_x=cells,
        cell_y=cells,
        population=population,
        group_size=10,
        stop_time=stop_time,
        hazard_count=hazard_count,
        candidate_count=20,
        shelter_capacity_token=300,
        scenario_seed=seed,
        spacing_m=spacing_m,
        spread_rate=spread,
    )
    core.actionSpace = "region_defer"
    core.shelterActionInterval = int(interval)
    regime = HazardRegimeProcess(
        HazardRegimeConfig(
            enabled=True,
            episode_horizon=max(1, stop_time - 1),
            wind_shift_rate_per_minute=0.03,
            intensity_switch_rate_per_minute=0.02,
            closure_rate_per_minute=0.01,
            secondary_ignition_rate_per_minute=0.02,
        ),
        np.random.default_rng(seed + 5),
        core.cellTracker,
        core.mapDS,
    )
    regime.base_wind_speed_m_per_minute = 16.67
    regime.reset(0)
    core.hazard_regime = regime
    core.hazardDS.set_regime(regime)
    core.hazardRegimeEnabled = True
    return core


def builder_for(core) -> RegionalObservationBuilder:
    return RegionalObservationBuilder(
        core,
        initial_population=int(core.initial_population),
        horizon=int(core.stopTime) - 1,
        maximum_deployments=8,
    )


def advance(core, minutes: int) -> None:
    for _ in range(int(minutes)):
        CB.advance_one_timestep(core)


def hazard_digest(core) -> str:
    """Everything a belief forecast reads, plus every random stream."""
    hazard_store = core.hazardDS
    digest = hashlib.sha256()
    digest.update(np.asarray(hazard_store._current_danger()).tobytes())
    for hazard_id, hazard in sorted(hazard_store.hazardList.items()):
        record = hazard_store._forecast_hazard_record(hazard)
        digest.update(repr(sorted((k, repr(v)) for k, v in record.items())).encode())
    regime = getattr(core, "hazard_regime", None)
    if regime is not None:
        digest.update(
            repr(sorted((k, repr(v)) for k, v in regime.public.items())).encode()
        )
        digest.update(repr(regime.rng.bit_generator.state).encode())
    digest.update(repr(hazard_store.rng.bit_generator.state).encode())
    digest.update(
        repr(
            (
                hazard_store.wind_speed_m_per_minute,
                hazard_store.wind_direction_degrees,
                hazard_store.wind_influence,
                hazard_store.time_step_minutes,
                hazard_store.intensity_multiplier,
            )
        ).encode()
    )
    return digest.hexdigest()


def full_state_digest(core) -> str:
    return CB.live_state_digest(core) + hazard_digest(core)
