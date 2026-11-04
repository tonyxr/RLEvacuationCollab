#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/7: Constructed the skeletons
9/9: implemented the spreadUpdate function
"""

import contextlib
import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Mapping, Optional, Sequence

import numpy as np
from Hazard import hazardAgent


@dataclass(frozen=True)
class HazardRegimeConfig:
    """Configuration for the action-independent exogenous hazard tape."""

    enabled: bool = False
    wind_shift_rate_per_minute: float = 0.0
    wind_shift_degrees: tuple[float, ...] = (-90.0, -45.0, 45.0, 90.0)
    wind_speed_multipliers: tuple[float, ...] = (0.6, 1.0, 1.5)
    wind_min_dwell_minutes: int = 20
    forecast_lead_minutes: int = 15
    forecast_miss_probability: float = 0.2
    forecast_false_alarm_rate_per_minute: float = 0.0
    intensity_switch_rate_per_minute: float = 0.0
    intensity_multipliers: tuple[float, ...] = (0.7, 1.0, 1.6)
    closure_rate_per_minute: float = 0.0
    closure_duration_minutes: tuple[int, int] = (20, 60)
    closure_warning_lead_minutes: int = 10
    secondary_ignition_rate_per_minute: float = 0.0
    secondary_ignition_window: tuple[int, int] = (10, 90)
    episode_horizon: int = 180

    def __post_init__(self):
        for name in (
            "wind_shift_rate_per_minute",
            "forecast_false_alarm_rate_per_minute",
            "intensity_switch_rate_per_minute",
            "closure_rate_per_minute",
            "secondary_ignition_rate_per_minute",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        miss = float(self.forecast_miss_probability)
        if not math.isfinite(miss) or not 0.0 <= miss <= 1.0:
            raise ValueError("forecast_miss_probability must lie in [0, 1]")
        if min(
            int(self.wind_min_dwell_minutes),
            int(self.forecast_lead_minutes),
            int(self.closure_warning_lead_minutes),
        ) < 0:
            raise ValueError("regime dwell and warning times must be non-negative")
        if int(self.episode_horizon) <= 0:
            raise ValueError("episode_horizon must be positive")
        for name in (
            "wind_shift_degrees",
            "wind_speed_multipliers",
            "intensity_multipliers",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must be a non-empty finite sequence")
            if "multipliers" in name and any(value <= 0.0 for value in values):
                raise ValueError(f"{name} values must be positive")
        for name in ("closure_duration_minutes", "secondary_ignition_window"):
            bounds = tuple(int(value) for value in getattr(self, name))
            if len(bounds) != 2 or bounds[0] < 0 or bounds[1] < bounds[0]:
                raise ValueError(f"{name} must be an ordered non-negative pair")


class _BeliefParticleMapView:
    """Read-only map inventory used while sampling a latent belief future.

    ``HazardRegimeProcess`` normally resets the physical map when it is
    constructed.  A belief particle needs the edge identifiers in order to
    sample possible future closures, but it must never mutate the live map.
    This detached view gives the sampler exactly that inventory and an empty
    closure ledger.
    """

    def __init__(self, map_ds):
        self.edgeListByLocalID = dict(
            getattr(map_ds, "edgeListByLocalID", {})
        )
        self.closed_edges = {}

    def reopen_edge(self, edge_id):
        self.closed_edges.pop(int(edge_id), None)


class HazardRegimeProcess:
    """Markov-modulated, action-independent hazard event tape."""

    def __init__(self, config, rng, cell_tracker, map_ds):
        self.config = (
            config
            if isinstance(config, HazardRegimeConfig)
            else HazardRegimeConfig(**config)
        )
        self.enabled = bool(self.config.enabled)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.cell_tracker = cell_tracker
        self.map_ds = map_ds
        self.base_wind_direction_degrees = 0.0
        self.base_wind_speed_m_per_minute = 0.0
        self.wind_direction_degrees = 0.0
        self.wind_speed_multiplier = 1.0
        self.intensity_multiplier = 1.0
        self.intensity_states = self._intensity_state_space()
        self.intensity_belief = self._initial_intensity_belief()
        self.intensity_belief_time = 0
        self.intensity_belief_updates = 0
        self.active_closures = {}
        self.scheduled = []
        self.public = {}
        self.current_time = 0
        self.reset(0)

    def _intensity_state_space(self) -> np.ndarray:
        """Return the latent-intensity support, including the known start state."""
        values = sorted(
            set(float(value) for value in self.config.intensity_multipliers)
            | {1.0}
        )
        return np.asarray(values, dtype=np.float64)

    def _initial_intensity_belief(self) -> np.ndarray:
        """The simulator and decision maker both know that episodes start at 1."""
        belief = np.zeros(len(self.intensity_states), dtype=np.float64)
        belief[int(np.argmin(np.abs(self.intensity_states - 1.0)))] = 1.0
        return belief

    def _intensity_mark_distribution(self) -> np.ndarray:
        """Distribution of the new regime mark conditional on a switch."""
        probabilities = np.zeros(len(self.intensity_states), dtype=np.float64)
        configured = tuple(float(value) for value in self.config.intensity_multipliers)
        for value in configured:
            index = int(np.argmin(np.abs(self.intensity_states - value)))
            probabilities[index] += 1.0 / float(len(configured))
        return probabilities

    def _write_intensity_belief_public(self) -> None:
        belief = np.asarray(self.intensity_belief, dtype=np.float64)
        states = np.asarray(self.intensity_states, dtype=np.float64)
        mean = float(np.dot(belief, states))
        variance = float(np.dot(belief, np.square(states - mean)))
        positive = belief[belief > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        maximum = math.log(max(1, len(belief)))
        self.public.update(
            {
                "intensity_belief_mean": mean,
                "intensity_belief_std": math.sqrt(max(0.0, variance)),
                "intensity_belief_entropy": (
                    0.0 if maximum <= 0.0 else entropy / maximum
                ),
            }
        )

    def _predict_intensity_belief_to(self, sim_time: int) -> None:
        """Apply the exact marked-Poisson transition kernel to the belief."""
        target_time = int(sim_time)
        elapsed = target_time - int(self.intensity_belief_time)
        if elapsed < 0:
            raise ValueError("intensity belief cannot move backward in time")
        if elapsed == 0:
            return
        rate = float(self.config.intensity_switch_rate_per_minute)
        stay_probability = math.exp(-rate * float(elapsed))
        mark_distribution = self._intensity_mark_distribution()
        self.intensity_belief = (
            stay_probability * np.asarray(self.intensity_belief, dtype=np.float64)
            + (1.0 - stay_probability) * mark_distribution
        )
        self.intensity_belief /= float(np.sum(self.intensity_belief))
        self.intensity_belief_time = target_time

    def observe_spread_transition(
        self,
        hazard_ds,
        exposures: Mapping[int, Sequence[tuple]],
        newly_impacted: Sequence[int],
    ) -> None:
        """Bayes-update latent intensity from the publicly observed fire front.

        ``exposures`` contains every susceptible target cell and the active
        source edges that could ignite it during this minute. This is the
        complete likelihood of the observable 0->1 CA spread transition. The
        realized latent multiplier is deliberately not read here.
        """
        if not self.enabled:
            return
        impacted = set(int(index) for index in newly_impacted)
        log_likelihood = np.zeros(len(self.intensity_states), dtype=np.float64)
        epsilon = 1e-12
        for state_index, multiplier in enumerate(self.intensity_states):
            total = 0.0
            for target_index, source_edges in exposures.items():
                probability_not_ignited = 1.0
                for base_rate, source_cell, target_cell in source_edges:
                    probability = hazard_ds.directional_spread_probability(
                        hazard_ds.oneClip(float(base_rate) * float(multiplier)),
                        source_cell,
                        target_cell,
                    )
                    probability_not_ignited *= 1.0 - float(probability)
                ignition_probability = float(
                    np.clip(1.0 - probability_not_ignited, epsilon, 1.0 - epsilon)
                )
                total += math.log(
                    ignition_probability
                    if int(target_index) in impacted
                    else 1.0 - ignition_probability
                )
            log_likelihood[state_index] = total
        if exposures:
            log_posterior = np.log(
                np.clip(self.intensity_belief, epsilon, 1.0)
            ) + log_likelihood
            log_posterior -= float(np.max(log_posterior))
            posterior = np.exp(log_posterior)
            normalizer = float(np.sum(posterior))
            if not math.isfinite(normalizer) or normalizer <= 0.0:
                raise RuntimeError("latent-intensity posterior failed to normalize")
            self.intensity_belief = posterior / normalizer
        self.intensity_belief_updates += 1
        self._write_intensity_belief_public()

    @staticmethod
    def _next_event_time(rng, current, rate):
        if rate <= 0.0:
            return math.inf
        return int(current + max(1, math.ceil(float(rng.exponential(1.0 / rate)))))

    def _draw_rate_events(
        self, kind, rate, start, end, payload_factory, *, min_dwell=0
    ):
        events = []
        cursor = int(start)
        while True:
            event_time = self._next_event_time(self.rng, cursor, float(rate))
            if event_time > int(end):
                break
            events.append(
                {
                    "time": event_time,
                    "kind": str(kind),
                    "payload": dict(payload_factory(event_time)),
                }
            )
            cursor = event_time + int(min_dwell)
        return events

    def reset(self, start_time=0):
        for edge_id in tuple(getattr(self.map_ds, "closed_edges", {})):
            self.map_ds.reopen_edge(edge_id)
        self.current_time = int(start_time)
        self.wind_direction_degrees = float(
            self.base_wind_direction_degrees
        ) % 360.0
        self.wind_speed_multiplier = 1.0
        self.intensity_multiplier = 1.0
        self.intensity_states = self._intensity_state_space()
        self.intensity_belief = self._initial_intensity_belief()
        self.intensity_belief_time = int(start_time)
        self.intensity_belief_updates = 0
        self.active_closures = {}
        self.public = {
            "wind_forecast_direction": self.wind_direction_degrees,
            "wind_forecast_speed": self.base_wind_speed_m_per_minute,
            "wind_warning": False,
            "closure_warning_edges": (),
            "time_to_change": -1,
        }
        self._write_intensity_belief_public()
        self.scheduled = []
        if not self.enabled:
            return
        cfg = self.config
        horizon = int(cfg.episode_horizon)

        def wind_payload(_):
            return {
                "direction_delta": float(self.rng.choice(cfg.wind_shift_degrees)),
                "speed_multiplier": float(
                    self.rng.choice(cfg.wind_speed_multipliers)
                ),
                "warning_visible": bool(
                    self.rng.random() >= cfg.forecast_miss_probability
                ),
            }

        self.scheduled.extend(
            self._draw_rate_events(
                "wind",
                cfg.wind_shift_rate_per_minute,
                start_time,
                horizon,
                wind_payload,
                min_dwell=cfg.wind_min_dwell_minutes,
            )
        )
        self.scheduled.extend(
            self._draw_rate_events(
                "intensity",
                cfg.intensity_switch_rate_per_minute,
                start_time,
                horizon,
                lambda _: {
                    "multiplier": float(
                        self.rng.choice(cfg.intensity_multipliers)
                    )
                },
            )
        )
        edge_ids = tuple(
            sorted(
                int(key)
                for key in getattr(self.map_ds, "edgeListByLocalID", {})
            )
        )
        if edge_ids:
            low, high = (int(v) for v in cfg.closure_duration_minutes)
            self.scheduled.extend(
                self._draw_rate_events(
                    "closure",
                    cfg.closure_rate_per_minute,
                    start_time,
                    horizon,
                    lambda _: {
                        "edge_id": int(self.rng.choice(edge_ids)),
                        "duration": int(self.rng.integers(low, high + 1)),
                    },
                )
            )
        ignition_low, ignition_high = (
            int(v) for v in cfg.secondary_ignition_window
        )
        ignition_start = max(int(start_time), ignition_low)
        ignition_end = min(horizon, ignition_high)
        cell_count = int(getattr(self.cell_tracker, "number_of_cells", 0))
        if cell_count > 0 and ignition_end >= ignition_start:
            self.scheduled.extend(
                self._draw_rate_events(
                    "ignition",
                    cfg.secondary_ignition_rate_per_minute,
                    ignition_start,
                    ignition_end,
                    lambda _: {
                        "cell_index": int(self.rng.integers(0, cell_count))
                    },
                )
            )
        self.scheduled.extend(
            self._draw_rate_events(
                "false_alarm",
                cfg.forecast_false_alarm_rate_per_minute,
                start_time,
                horizon,
                lambda _: {
                    "direction_delta": float(
                        self.rng.choice(cfg.wind_shift_degrees)
                    ),
                    "speed_multiplier": float(
                        self.rng.choice(cfg.wind_speed_multipliers)
                    ),
                },
            )
        )
        self.scheduled.sort(
            key=lambda event: (int(event["time"]), str(event["kind"]))
        )
        self._update_public(self.current_time)

    def _update_public(self, sim_time):
        cfg = self.config
        wind_candidates = []
        closure_edges = []
        for event in self.scheduled:
            delta = int(event["time"]) - int(sim_time)
            if (
                event["kind"] == "wind"
                and 0 <= delta <= int(cfg.forecast_lead_minutes)
                and bool(event["payload"].get("warning_visible", False))
            ):
                wind_candidates.append((delta, event))
            elif (
                event["kind"] == "false_alarm"
                and 0 <= delta <= int(cfg.forecast_lead_minutes)
            ):
                wind_candidates.append((delta, event))
            elif (
                event["kind"] == "closure"
                and 0 <= delta <= int(cfg.closure_warning_lead_minutes)
            ):
                closure_edges.append(int(event["payload"]["edge_id"]))
        if wind_candidates:
            delta, event = min(wind_candidates, key=lambda item: item[0])
            payload = event["payload"]
            direction = (
                self.wind_direction_degrees
                + float(payload["direction_delta"])
            ) % 360.0
            speed = self.base_wind_speed_m_per_minute * float(
                payload["speed_multiplier"]
            )
            warning = True
            time_to_change = int(delta)
        else:
            direction = self.wind_direction_degrees
            speed = (
                self.base_wind_speed_m_per_minute
                * self.wind_speed_multiplier
            )
            warning = False
            time_to_change = -1
        self.public = {
            "wind_forecast_direction": float(direction),
            "wind_forecast_speed": float(speed),
            "wind_warning": bool(warning),
            "closure_warning_edges": tuple(sorted(set(closure_edges))),
            "time_to_change": int(time_to_change),
        }
        self._write_intensity_belief_public()

    def advance(self, sim_time, hazard_ds, map_ds):
        self.current_time = int(sim_time)
        if not self.enabled:
            return
        self._predict_intensity_belief_to(sim_time)
        for edge_id, end_time in tuple(self.active_closures.items()):
            if int(sim_time) >= int(end_time):
                map_ds.reopen_edge(edge_id)
                self.active_closures.pop(edge_id, None)
        for event in self.scheduled:
            if int(event["time"]) != int(sim_time):
                continue
            payload = event["payload"]
            if event["kind"] == "wind":
                self.wind_direction_degrees = (
                    self.wind_direction_degrees
                    + float(payload["direction_delta"])
                ) % 360.0
                self.wind_speed_multiplier = float(
                    payload["speed_multiplier"]
                )
            elif event["kind"] == "intensity":
                self.intensity_multiplier = float(payload["multiplier"])
            elif event["kind"] == "closure":
                edge_id = int(payload["edge_id"])
                end_time = int(sim_time) + int(payload["duration"])
                end_time = max(
                    end_time, int(self.active_closures.get(edge_id, end_time))
                )
                self.active_closures[edge_id] = end_time
                map_ds.close_edge(edge_id, end_time)
            elif event["kind"] == "ignition":
                cell = self.cell_tracker.unflatten_cell(
                    int(payload["cell_index"])
                )
                hazard_ds.add_hazard_at(cell, int(sim_time))
        hazard_ds.configure_wind(
            speed_m_per_minute=(
                self.base_wind_speed_m_per_minute
                * self.wind_speed_multiplier
            ),
            direction_degrees=self.wind_direction_degrees,
            influence=hazard_ds.wind_influence,
            time_step_minutes=hazard_ds.time_step_minutes,
        )
        hazard_ds.set_intensity(self.intensity_multiplier)
        self._update_public(sim_time)

    def snapshot(self):
        return {
            "config": asdict(self.config),
            "enabled": self.enabled,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "base_wind_direction_degrees": self.base_wind_direction_degrees,
            "base_wind_speed_m_per_minute": self.base_wind_speed_m_per_minute,
            "wind_direction_degrees": self.wind_direction_degrees,
            "wind_speed_multiplier": self.wind_speed_multiplier,
            "intensity_multiplier": self.intensity_multiplier,
            "intensity_states": self.intensity_states.tolist(),
            "intensity_belief": self.intensity_belief.tolist(),
            "intensity_belief_time": self.intensity_belief_time,
            "intensity_belief_updates": self.intensity_belief_updates,
            "active_closures": copy.deepcopy(self.active_closures),
            "scheduled": copy.deepcopy(self.scheduled),
            "public": copy.deepcopy(self.public),
            "current_time": self.current_time,
        }

    def restore(self, state: Mapping):
        self.config = HazardRegimeConfig(**dict(state["config"]))
        self.enabled = bool(state["enabled"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        for name in (
            "base_wind_direction_degrees",
            "base_wind_speed_m_per_minute",
            "wind_direction_degrees",
            "wind_speed_multiplier",
            "intensity_multiplier",
            "current_time",
        ):
            setattr(self, name, copy.deepcopy(state[name]))
        self.intensity_states = np.asarray(
            state["intensity_states"], dtype=np.float64
        ).copy()
        self.intensity_belief = np.asarray(
            state["intensity_belief"], dtype=np.float64
        ).copy()
        self.intensity_belief_time = int(state["intensity_belief_time"])
        self.intensity_belief_updates = int(state["intensity_belief_updates"])
        self.active_closures = copy.deepcopy(state["active_closures"])
        self.scheduled = copy.deepcopy(state["scheduled"])
        self.public = copy.deepcopy(state["public"])
        if self.map_ds is not None and hasattr(self.map_ds, "restore_closures"):
            self.map_ds.restore_closures(self.active_closures)

    def belief_particles(self, k, rng):
        """Sample latent futures conditioned only on the public dashboard.

        The realized schedule is deliberately not copied.  Public wind
        warnings reveal a forecast time/vector but not whether the warning is
        a true event; public road warnings reveal edge identity but not exact
        closure time or duration.  Those concealed quantities are resampled
        below.  Consequently two regimes with the same current state and
        public signals induce the same particle distribution even when their
        hidden episode tapes differ.
        """
        count = int(k)
        if count <= 0:
            raise ValueError("belief particle count must be positive")

        public_wind_warning = bool(self.public.get("wind_warning", False))
        public_wind_delta = int(self.public.get("time_to_change", -1))
        public_wind_direction = float(
            self.public.get(
                "wind_forecast_direction", self.wind_direction_degrees
            )
        )
        public_wind_speed = max(
            0.0,
            float(
                self.public.get(
                    "wind_forecast_speed",
                    self.base_wind_speed_m_per_minute
                    * self.wind_speed_multiplier,
                )
            ),
        )
        warned_edges = tuple(
            sorted(
                {
                    int(edge_id)
                    for edge_id in self.public.get(
                        "closure_warning_edges", ()
                    )
                }
            )
        )
        # The road dashboard exposes which edges are closed, not the realized
        # event tape's exact reopening minute.  Condition particles on those
        # public identities and integrate over the unobserved residual life.
        active_edge_ids = tuple(
            sorted(
                int(edge_id)
                for edge_id in getattr(self.map_ds, "closed_edges", {})
            )
        )

        def sampled_active_closures() -> dict[int, int]:
            if not active_edge_ids:
                return {}
            duration_low, duration_high = (
                int(value)
                for value in self.config.closure_duration_minutes
            )
            durations = np.arange(
                max(1, duration_low), duration_high + 1, dtype=np.int64
            )
            if durations.size == 0:
                raise RuntimeError(
                    "An active road closure is inconsistent with a zero-duration "
                    "closure model"
                )
            # At an arbitrary observation time, longer closures are more
            # likely to be the one currently in force.  Sample that
            # length-biased duration, then a uniformly located elapsed minute;
            # the resulting residual is the correct equilibrium distribution
            # conditional on seeing a closure but not its hidden end time.
            probabilities = durations.astype(np.float64)
            probabilities /= float(np.sum(probabilities))
            sampled = {}
            for edge_id in active_edge_ids:
                duration = int(rng.choice(durations, p=probabilities))
                elapsed = int(rng.integers(0, duration))
                residual = max(1, duration - elapsed)
                sampled[int(edge_id)] = int(self.current_time + residual)
            return sampled

        def sampled_warning_events() -> list[dict]:
            events = []
            if public_wind_warning:
                if not (
                    0 <= public_wind_delta
                    <= int(self.config.forecast_lead_minutes)
                ):
                    raise RuntimeError(
                        "Public wind warning has an invalid lead time"
                    )
                visible_wind_rate = max(
                    0.0,
                    float(self.config.wind_shift_rate_per_minute)
                    * (1.0 - float(self.config.forecast_miss_probability)),
                )
                false_alarm_rate = max(
                    0.0,
                    float(self.config.forecast_false_alarm_rate_per_minute),
                )
                total_warning_rate = visible_wind_rate + false_alarm_rate
                true_probability = (
                    visible_wind_rate / total_warning_rate
                    if total_warning_rate > 0.0
                    else 0.5
                )
                direction_delta = (
                    public_wind_direction - self.wind_direction_degrees
                ) % 360.0
                if direction_delta > 180.0:
                    direction_delta -= 360.0
                base_speed = max(
                    float(self.base_wind_speed_m_per_minute), 1e-12
                )
                payload = {
                    "direction_delta": float(direction_delta),
                    "speed_multiplier": float(public_wind_speed / base_speed),
                }
                is_true = bool(rng.random() < true_probability)
                if is_true:
                    payload["warning_visible"] = True
                events.append(
                    {
                        "time": int(self.current_time + public_wind_delta),
                        "kind": "wind" if is_true else "false_alarm",
                        "payload": payload,
                    }
                )

            closure_lead = max(
                1, int(self.config.closure_warning_lead_minutes)
            )
            closure_rate = max(
                0.0, float(self.config.closure_rate_per_minute)
            )
            duration_low, duration_high = (
                int(value)
                for value in self.config.closure_duration_minutes
            )
            for edge_id in warned_edges:
                if closure_rate > 0.0:
                    truncation = 1.0 - math.exp(
                        -closure_rate * float(closure_lead)
                    )
                    draw = float(rng.random())
                    continuous_lead = -math.log(
                        max(1e-15, 1.0 - draw * truncation)
                    ) / closure_rate
                    event_lead = min(
                        closure_lead,
                        max(1, int(math.ceil(continuous_lead))),
                    )
                else:
                    event_lead = int(rng.integers(1, closure_lead + 1))
                events.append(
                    {
                        "time": int(self.current_time + event_lead),
                        "kind": "closure",
                        "payload": {
                            "edge_id": int(edge_id),
                            "duration": int(
                                rng.integers(
                                    duration_low, duration_high + 1
                                )
                            ),
                        },
                    }
                )
            return events

        particles = []
        particle_map = _BeliefParticleMapView(self.map_ds)
        for _ in range(count):
            seed = int(
                rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
            particle = HazardRegimeProcess(
                self.config,
                np.random.default_rng(seed),
                self.cell_tracker,
                particle_map,
            )
            particle.base_wind_direction_degrees = (
                self.base_wind_direction_degrees
            )
            particle.base_wind_speed_m_per_minute = (
                self.base_wind_speed_m_per_minute
            )
            particle.wind_direction_degrees = self.wind_direction_degrees
            particle.wind_speed_multiplier = self.wind_speed_multiplier
            particle.intensity_multiplier = float(
                rng.choice(self.intensity_states, p=self.intensity_belief)
            )
            particle.intensity_states = self.intensity_states.copy()
            particle.intensity_belief = self.intensity_belief.copy()
            particle.intensity_belief_time = int(self.intensity_belief_time)
            particle.intensity_belief_updates = int(self.intensity_belief_updates)
            particle.active_closures = sampled_active_closures()
            latent = []
            for event in particle.scheduled:
                event_time = int(event["time"])
                if event_time <= self.current_time:
                    continue
                kind = str(event["kind"])
                if kind == "false_alarm":
                    if (
                        event_time - self.current_time
                        <= int(self.config.forecast_lead_minutes)
                    ):
                        continue
                if kind == "wind":
                    # Absence of a warning is itself information during the
                    # configured lead window. A missed event may still occur.
                    within_lead = (
                        event_time - self.current_time
                        <= int(self.config.forecast_lead_minutes)
                    )
                    visible = bool(
                        event["payload"].get("warning_visible", False)
                    )
                    if within_lead and visible:
                        continue
                elif kind == "closure":
                    within_lead = (
                        event_time - self.current_time
                        <= int(self.config.closure_warning_lead_minutes)
                    )
                    if within_lead:
                        continue
                latent.append(copy.deepcopy(event))
            particle.scheduled = sorted(
                sampled_warning_events() + latent,
                key=lambda event: (event["time"], event["kind"]),
            )
            particle.current_time = self.current_time
            particle.public = copy.deepcopy(self.public)
            particles.append(particle)
        return particles

# ---------------------------------------------------------------------------
# v65 forecast engine: vectorized mean-field kernel (S1) and the belief
# trajectory memo shared by candidates of one common-random-number future (S3).
# ---------------------------------------------------------------------------

FORECAST_KERNELS = ("vectorized", "reference")
DEFAULT_FORECAST_KERNEL = "vectorized"

# Geometry, edge probabilities and severity caps depend only on the CA grid,
# the wind state and the hazard source.  They are cached at module level so
# that the caches survive ``CounterfactualBranch.restore`` deep copies.
_FORECAST_GEOMETRY_CACHE: dict = {}
_FORECAST_EDGE_PROBABILITY_CACHE: dict = {}
_FORECAST_SEVERITY_CAP_CACHE: dict = {}
_FORECAST_EDGE_PROBABILITY_CACHE_LIMIT = 512
_FORECAST_SEVERITY_CAP_CACHE_LIMIT = 4096
_FORECAST_GEOMETRY_CACHE_LIMIT = 16


class _ForecastGeometry:
    """Moore-neighbour incidence of one CA grid in decision flattening order.

    ``source_pad[t, c]`` is the c-th source cell whose Moore neighbourhood
    contains target ``t``, in ascending source index -- exactly the order in
    which the reference double loop multiplies ``1 - p_s q_e`` into the
    target.  ``edge_pad[t, c]`` is that edge's index into ``edge_source`` /
    ``edge_target``; ``-1`` marks padding.
    """

    __slots__ = (
        "key",
        "cell_count",
        "cells",
        "source_pad",
        "edge_pad",
        "edge_source",
        "edge_target",
        "max_in_degree",
    )


def _forecast_geometry(hazard_store) -> _ForecastGeometry:
    tracker = hazard_store.cellTracker
    cell_count = int(tracker.number_of_cells)
    cells = tuple(tracker.unflatten_cell(index) for index in range(cell_count))
    centers = np.asarray(
        [tracker.getCellCenter(cell) for cell in cells], dtype=np.float64
    ).reshape(cell_count, 2)
    key = (
        int(tracker.cellXNum),
        int(tracker.cellYNum),
        hashlib.sha256(np.ascontiguousarray(centers).tobytes()).hexdigest(),
    )
    geometry = _FORECAST_GEOMETRY_CACHE.get(key)
    if geometry is not None:
        return geometry
    incoming = [[] for _ in range(cell_count)]
    edge_source = []
    edge_target = []
    for source in range(cell_count):
        for target_cell in tracker.getNeighborCells(cells[source]):
            target = int(tracker.flatten_cell(target_cell))
            incoming[target].append(len(edge_source))
            edge_source.append(source)
            edge_target.append(target)
    max_in_degree = max((len(edges) for edges in incoming), default=0)
    source_pad = np.zeros((cell_count, max_in_degree), dtype=np.int64)
    edge_pad = np.full((cell_count, max_in_degree), -1, dtype=np.int64)
    for target, edges in enumerate(incoming):
        # Edges were appended while sources ascended, so this list is already
        # in the reference multiplication order.
        for column, edge in enumerate(edges):
            source_pad[target, column] = edge_source[edge]
            edge_pad[target, column] = edge
    geometry = _ForecastGeometry()
    geometry.key = key
    geometry.cell_count = cell_count
    geometry.cells = cells
    geometry.source_pad = source_pad
    geometry.edge_pad = edge_pad
    geometry.edge_source = np.asarray(edge_source, dtype=np.int64)
    geometry.edge_target = np.asarray(edge_target, dtype=np.int64)
    geometry.max_in_degree = int(max_in_degree)
    if len(_FORECAST_GEOMETRY_CACHE) >= _FORECAST_GEOMETRY_CACHE_LIMIT:
        _FORECAST_GEOMETRY_CACHE.clear()
    _FORECAST_GEOMETRY_CACHE[key] = geometry
    return geometry


def _forecast_edge_probabilities(
    hazard_store, geometry: _ForecastGeometry, base: float
) -> np.ndarray:
    """Padded ``(cells, max_in_degree)`` spread probabilities ``q_e``.

    On a cache miss every edge is evaluated by the authoritative scalar
    method :meth:`HazardDS.directional_spread_probability`, so the vector
    kernel cannot drift from the simulator's wind/distance rule.  Padding
    entries are exactly zero, giving a factor ``1 - p * 0 = 1.0``.
    """
    key = (
        geometry.key,
        float(base),
        float(hazard_store.wind_speed_m_per_minute),
        float(hazard_store.wind_direction_degrees),
        float(hazard_store.wind_influence),
        float(hazard_store.spread_speed_m_per_minute),
        float(hazard_store.time_step_minutes),
    )
    cached = _FORECAST_EDGE_PROBABILITY_CACHE.get(key)
    if cached is not None:
        return cached
    cells = geometry.cells
    edge_probability = np.empty(geometry.edge_source.size, dtype=np.float64)
    for edge in range(geometry.edge_source.size):
        edge_probability[edge] = hazard_store.directional_spread_probability(
            base,
            cells[int(geometry.edge_source[edge])],
            cells[int(geometry.edge_target[edge])],
        )
    padded = np.zeros(geometry.edge_pad.shape, dtype=np.float64)
    valid = geometry.edge_pad >= 0
    padded[valid] = edge_probability[geometry.edge_pad[valid]]
    padded.setflags(write=False)
    if len(_FORECAST_EDGE_PROBABILITY_CACHE) >= _FORECAST_EDGE_PROBABILITY_CACHE_LIMIT:
        _FORECAST_EDGE_PROBABILITY_CACHE.clear()
    _FORECAST_EDGE_PROBABILITY_CACHE[key] = padded
    return padded


def _forecast_severity_caps(
    hazard_store, geometry: _ForecastGeometry, source_cell, source_point
) -> np.ndarray:
    key = (
        geometry.key,
        source_cell,
        None if source_point is None else tuple(float(v) for v in source_point),
        float(hazard_store.severe_radius_m),
    )
    cached = _FORECAST_SEVERITY_CAP_CACHE.get(key)
    if cached is not None:
        return cached
    caps = np.asarray(
        [
            hazard_store._severity_cap(source_cell, source_point, cell)
            for cell in geometry.cells
        ],
        dtype=np.int64,
    )
    caps.setflags(write=False)
    if len(_FORECAST_SEVERITY_CAP_CACHE) >= _FORECAST_SEVERITY_CAP_CACHE_LIMIT:
        _FORECAST_SEVERITY_CAP_CACHE.clear()
    _FORECAST_SEVERITY_CAP_CACHE[key] = caps
    return caps


def _forecast_severity_levels(hazard_store, elapsed_minutes) -> np.ndarray:
    """Vector form of the threshold count inside ``HazardDS._severity_level``."""
    elapsed = np.maximum(0.0, np.asarray(elapsed_minutes, dtype=np.float64))
    level = np.ones(elapsed.shape, dtype=np.int64)
    for threshold in hazard_store.severity_transition_minutes:
        level += (elapsed + 1e-12 >= threshold).astype(np.int64)
    return level


class _MeanFieldRecord:
    """Array form of one mean-field hazard record.

    ``arrival_mass[k]`` holds the masses that arrived at step
    ``arrival_step[k]``; masses at or below ``1e-15`` are stored as exact
    zeros, matching the reference kernel which never records them.
    """

    __slots__ = (
        "active",
        "step",
        "age",
        "lifespan",
        "spread_rate",
        "legacy_front",
        "caps",
        "probability",
        "initial_mask",
        "initial_first_step",
        "arrival_mass",
        "arrival_step",
        "arrival_count",
        "touched",
        "frozen_value",
    )


def _mean_field_record(
    hazard_store, geometry: _ForecastGeometry, raw: Mapping, capacity: int
) -> _MeanFieldRecord:
    record = _MeanFieldRecord()
    record.active = bool(raw["active"])
    record.step = int(raw["step"])
    record.age = int(raw["age"])
    record.lifespan = int(raw["lifespan"])
    record.spread_rate = float(raw["spread_rate"])
    record.legacy_front = bool(raw.get("legacy_front", False))
    record.caps = _forecast_severity_caps(
        hazard_store, geometry, raw.get("source_cell"), raw.get("source_point")
    )
    cell_count = geometry.cell_count
    record.probability = np.zeros(cell_count, dtype=np.float64)
    record.initial_mask = np.zeros(cell_count, dtype=np.bool_)
    record.initial_first_step = np.zeros(cell_count, dtype=np.int64)
    for cell in raw["impacted"]:
        index = int(hazard_store.cellTracker.flatten_cell(cell))
        record.probability[index] = 1.0
        record.initial_mask[index] = True
        record.initial_first_step[index] = int(raw["first_impacted"][cell])
    rows = max(1, int(capacity))
    record.arrival_mass = np.zeros((rows, cell_count), dtype=np.float64)
    record.arrival_step = np.zeros(rows, dtype=np.int64)
    record.arrival_count = 0
    record.touched = record.initial_mask.copy()
    record.frozen_value = None
    return record


def _mean_field_record_value(hazard_store, record: _MeanFieldRecord) -> np.ndarray:
    """``sum_k mass_k * level_k`` per cell, as a left fold in arrival order.

    Row 0 is the initial footprint (mass 1.0); rows 1..K are step arrivals.
    ``np.cumsum`` along axis 0 is a sequential running sum, which reproduces
    the reference ``value += mass * level`` loop bit for bit (adding an exact
    0.0 for cells without an arrival in a row leaves the sum unchanged).
    """
    time_step = hazard_store.time_step_minutes
    caps = record.caps
    count = record.arrival_count
    rows = np.empty((count + 1, caps.size), dtype=np.float64)
    elapsed = np.maximum(0, record.step - record.initial_first_step) * time_step
    level = np.minimum(caps, _forecast_severity_levels(hazard_store, elapsed))
    if not record.active:
        level = np.minimum(2, level)
    rows[0] = np.where(record.initial_mask, 1.0, 0.0) * level.astype(np.float64)
    if count:
        elapsed = np.maximum(0, record.step - record.arrival_step[:count]) * time_step
        levels = np.minimum(
            caps[None, :],
            _forecast_severity_levels(hazard_store, elapsed)[:, None],
        )
        if not record.active:
            levels = np.minimum(2, levels)
        np.multiply(
            record.arrival_mass[:count], levels.astype(np.float64), out=rows[1:]
        )
    return np.cumsum(rows, axis=0)[-1]


def _aggregate_mean_field_records(
    hazard_store, baseline: np.ndarray, records
) -> np.ndarray:
    """Vector twin of ``HazardDS._aggregate_stochastic_forecast_records``."""
    expected = np.asarray(baseline, dtype=np.float64).copy()
    if records:
        touched = np.zeros(expected.shape, dtype=np.bool_)
        legacy = np.zeros(expected.shape, dtype=np.bool_)
        values = []
        for record in records:
            if record.active:
                values.append(_mean_field_record_value(hazard_store, record))
            else:
                # An inactive record neither spreads nor ages, so its value is
                # constant for the rest of the forecast.
                if record.frozen_value is None:
                    record.frozen_value = _mean_field_record_value(
                        hazard_store, record
                    )
                values.append(record.frozen_value)
            touched |= record.touched
            if record.legacy_front:
                legacy |= record.touched
        if touched.any():
            combined = np.maximum.reduce(values)
            expected[touched] = combined[touched]
            if legacy.any():
                expected[legacy] = np.maximum(expected[legacy], baseline[legacy])
    return np.clip(expected / 5.0, 0.0, 1.0).astype(np.float32)


def _stochastic_forecast_danger_vectorized(
    hazard_store,
    horizon_steps: int,
    future_ignitions=(),
    *,
    configuration_by_step: Optional[Mapping[int, tuple]] = None,
    return_trajectory: bool = False,
) -> np.ndarray:
    """Bit-identical vector implementation of the mean-field forecast.

    Same inputs, outputs and side effects (``configure_wind`` /
    ``set_intensity`` at the same offsets) as
    :meth:`HazardDS._stochastic_forecast_danger_reference`.
    """
    geometry = _forecast_geometry(hazard_store)
    horizon = max(0, int(horizon_steps))
    baseline = hazard_store._current_danger().astype(np.float64) * 5.0
    records = [
        _mean_field_record(
            hazard_store,
            geometry,
            hazard_store._forecast_hazard_record(hazard),
            horizon,
        )
        for hazard in hazard_store.hazardList.values()
    ]
    ignitions = {}
    for offset, cell in future_ignitions:
        offset = int(offset)
        if 1 <= offset <= horizon:
            ignitions.setdefault(offset, []).append(hazard_store._cell_tuple(cell))
    trajectory = []
    source_pad = geometry.source_pad
    for offset in range(1, horizon + 1):
        if configuration_by_step is not None:
            configuration = configuration_by_step.get(offset)
            if configuration is not None:
                speed, direction, intensity = configuration
                hazard_store.configure_wind(
                    speed_m_per_minute=float(speed),
                    direction_degrees=float(direction),
                    influence=hazard_store.wind_influence,
                    time_step_minutes=hazard_store.time_step_minutes,
                )
                hazard_store.set_intensity(float(intensity))
        for cell in ignitions.get(offset, ()):
            records.append(
                _mean_field_record(
                    hazard_store,
                    geometry,
                    hazard_store._latent_ignition_record(cell, horizon - offset + 1),
                    horizon,
                )
            )
        for record in records:
            if not record.active:
                continue
            record.step += 1
            base = hazard_store.oneClip(
                float(record.spread_rate)
                * float(hazard_store.intensity_multiplier)
            )
            edge_probability = _forecast_edge_probabilities(
                hazard_store, geometry, base
            )
            current = record.probability
            probability_not_arriving = np.ones_like(current)
            for column in range(geometry.max_in_degree):
                probability_not_arriving *= (
                    1.0 - current[source_pad[:, column]] * edge_probability[:, column]
                )
            new_mass = (1.0 - current) * (1.0 - probability_not_arriving)
            record.probability = np.clip(current + new_mass, 0.0, 1.0)
            arrived = new_mass > 1e-15
            if arrived.any():
                row = record.arrival_count
                record.arrival_mass[row] = np.where(arrived, new_mass, 0.0)
                record.arrival_step[row] = record.step
                record.arrival_count = row + 1
                record.touched |= arrived
        for record in records:
            if not record.active:
                continue
            record.age += 1
            if record.lifespan > 0 and record.age >= record.lifespan:
                # The reference kernel also stores a ``residual_levels`` map
                # here; the stochastic aggregator never reads it.
                record.active = False
        if return_trajectory:
            trajectory.append(
                _aggregate_mean_field_records(hazard_store, baseline, records)
            )
    if return_trajectory:
        if trajectory:
            return np.stack(trajectory, axis=0)
        return np.empty((0, geometry.cell_count), dtype=np.float32)
    return _aggregate_mean_field_records(hazard_store, baseline, records)


# -- S3: belief-trajectory memo ----------------------------------------------
#
# Within one graded decision every candidate on rollout tape ``k`` shares the
# same belief particle and hazard random stream, and shelter placement cannot
# alter hazard evolution (see ``CounterfactualBranch.HazardFrame``).  The two
# belief first-passage forecasts at a later decision are therefore identical
# across those candidates; only the post-processing mask differs.  Branch
# workers set a scope ``(batch_token, tape_index)`` around each task; the live
# simulator never sets one, so it never reads or writes the memo.

_BELIEF_TRAJECTORY_MEMO: dict = {}
_BELIEF_TRAJECTORY_MEMO_LIMIT = 96
_BELIEF_MEMO_SCOPE = None
_BELIEF_MEMO_STATISTICS = {"hits": 0, "misses": 0, "verified": 0}


def set_belief_forecast_memo_scope(scope):
    """Set the active memo scope and return the previous one."""
    global _BELIEF_MEMO_SCOPE
    previous = _BELIEF_MEMO_SCOPE
    _BELIEF_MEMO_SCOPE = scope
    return previous


@contextlib.contextmanager
def belief_forecast_memo_scope(scope):
    """Context manager used by branch tasks; ``None`` disables the memo."""
    previous = set_belief_forecast_memo_scope(scope)
    try:
        yield
    finally:
        set_belief_forecast_memo_scope(previous)


def clear_belief_forecast_memo() -> None:
    _BELIEF_TRAJECTORY_MEMO.clear()


def belief_forecast_memo_statistics() -> dict:
    return dict(_BELIEF_MEMO_STATISTICS, entries=len(_BELIEF_TRAJECTORY_MEMO))


def _canonical_repr(value) -> str:
    """Order-independent repr for the forecast fingerprint."""
    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{_canonical_repr(k)}:{_canonical_repr(v)}"
            for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
        ) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_repr(v) for v in value) + "]"
    if isinstance(value, np.ndarray):
        return repr(np.ascontiguousarray(value).tolist())
    return repr(value)


class HazardDS:
    def __init__(
        self,
        hazardNum,
        casualtyRate,
        spreadRate,
        speedReduct,
        *,
        rng=None,
        wind_speed_m_per_minute=0.0,
        wind_direction_degrees=0.0,
        wind_influence=1.0,
        time_step_minutes=1.0,
        spread_speed_m_per_minute=100.0 / 3.0,
        severity_transition_minutes=(6.0, 15.0, 30.0, 50.0),
        severe_radius_m=100.0,
    ):
        # List of active hazard agents
        self.hazardList = {}
        # input parameter: number of total hazards presents in the environment
        self.totalHazard = int(hazardNum)
        
        # hazard casualty probability (probability causing a pedestrian casualty) range (mu, sigma)
        self.casualtyRate = self._validate_percent_distribution(
            "hazard casualty rate", casualtyRate
        )
        # hazard spread probability range (mu, sigma)
        self.spreadRate = self._validate_percent_distribution(
            "hazard spread rate", spreadRate
        )
        # hazard speed reduction caused to pedestrians (mu, sigma)
        self.speedReduct = self._validate_percent_distribution(
            "hazard speed reduction", speedReduct
        )
        # container for CAProcessor
        self.cellTracker = None
        # Keep legacy stochastic evolution code and allow temporary deterministic mode.
        self.cell_state_evolution_mode = "deterministic"  # {"deterministic", "stochastic"}
        transitions = tuple(float(value) for value in severity_transition_minutes)
        if (
            len(transitions) != 4
            or not all(math.isfinite(value) and value > 0.0 for value in transitions)
            or any(right <= left for left, right in zip(transitions, transitions[1:]))
        ):
            raise ValueError(
                "severity_transition_minutes must contain four strictly "
                "increasing positive minute thresholds"
            )
        radius = float(severe_radius_m)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("severe_radius_m must be finite and non-negative")
        self.severity_transition_minutes = transitions
        self.severe_radius_m = radius
        self.spread_speed_m_per_minute = float(spread_speed_m_per_minute)
        if (
            not math.isfinite(self.spread_speed_m_per_minute)
            or self.spread_speed_m_per_minute <= 0.0
        ):
            raise ValueError("spread_speed_m_per_minute must be finite and positive")
        # Paper Equations 13--14 and Table 3 defaults.
        self.initial_heat_intensity = 200.0
        self.heat_spread_radius_m = 100.0
        self.heat_growth_rate = 0.03
        self.initial_smoke_intensity = 200.0
        self.smoke_spread_radius_m = 60.0
        self.smoke_growth_rate = 0.04
        self.configure_wind(
            speed_m_per_minute=wind_speed_m_per_minute,
            direction_degrees=wind_direction_degrees,
            influence=wind_influence,
            time_step_minutes=time_step_minutes,
        )
        # Hazard evolution owns its random stream. Policy-dependent pedestrian
        # events must never advance the exogenous hazard sequence in a matched
        # policy comparison.
        self.rng = rng if rng is not None else np.random.default_rng()
        self.intensity_multiplier = 1.0
        self.regime = None
        # v65 forecast engine.  The kernel choice cannot change any number
        # (the vectorized kernel is bit-identical); the memo is active only
        # inside a branch-worker scope.  ``configure_forecast_engine`` is the
        # single entry point used by Core and the testbed.
        self.forecast_kernel = DEFAULT_FORECAST_KERNEL
        self.forecast_memo_enabled = True
        self.forecast_memo_verify_fraction = 0.0

    def configure_wind(
        self,
        *,
        speed_m_per_minute,
        direction_degrees,
        influence=1.0,
        time_step_minutes=1.0,
    ):
        """Set the observable wind vector used by spread and forecasting.

        Direction is the direction toward which the hazard is transported,
        measured counter-clockwise from projected east. Zero speed recovers
        the previous isotropic spread process exactly.
        """
        speed = float(speed_m_per_minute)
        direction = float(direction_degrees)
        influence = float(influence)
        time_step = float(time_step_minutes)
        if not math.isfinite(speed) or speed < 0.0:
            raise ValueError("wind speed must be finite and non-negative")
        if not math.isfinite(direction):
            raise ValueError("wind direction must be finite")
        if not math.isfinite(influence) or influence < 0.0:
            raise ValueError("wind influence must be finite and non-negative")
        if not math.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step_minutes must be finite and positive")
        self.wind_speed_m_per_minute = speed
        self.wind_direction_degrees = direction % 360.0
        self.wind_influence = influence
        self.time_step_minutes = time_step

    def set_regime(self, regime: Optional[HazardRegimeProcess]) -> None:
        self.regime = regime

    def set_intensity(self, multiplier: float) -> None:
        value = float(multiplier)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "hazard intensity multiplier must be finite and positive"
            )
        self.intensity_multiplier = value

    def configure_forecast_engine(
        self,
        *,
        kernel: str = DEFAULT_FORECAST_KERNEL,
        memo_enabled: bool = True,
        memo_verify_fraction: float = 0.0,
    ) -> None:
        """Select the mean-field kernel and the branch-scope memo policy."""
        kernel = str(kernel).strip().lower()
        if kernel not in FORECAST_KERNELS:
            raise ValueError(
                f"forecast kernel must be one of {FORECAST_KERNELS}, got {kernel!r}"
            )
        fraction = float(memo_verify_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("forecast memo verify fraction must lie in [0, 1]")
        self.forecast_kernel = kernel
        self.forecast_memo_enabled = bool(memo_enabled)
        self.forecast_memo_verify_fraction = fraction

    def add_hazard_at(self, cell, sim_time) -> int:
        """Add a secondary ignition without consuming the hazard RNG stream."""
        if self.cellTracker is None:
            raise RuntimeError("Secondary ignition requires a CellTracker")
        cell = self._cell_tuple(cell)
        self.cellTracker.flatten_cell(cell)
        local_id = (
            0
            if not self.hazardList
            else max(int(key) for key in self.hazardList) + 1
        )
        hazard = hazardAgent(
            local_id,
            self.oneClip(float(self.speedReduct[0]) / 100.0),
            self.oneClip(float(self.spreadRate[0]) / 100.0),
            self.oneClip(float(self.casualtyRate[0]) / 100.0),
            None,
            0.0,
            0.0,
            [cell],
        )
        hazard.active = True
        hazard.age = 0
        hazard.lifespan = (
            max(
                1,
                int(getattr(self.regime.config, "episode_horizon", 180))
                - int(sim_time),
            )
            if self.regime is not None
            else 60
        )
        hazard.sourceCell = cell
        hazard.step = 0
        hazard.cellImpactStep = {cell: 0}
        hazard.cellFirstImpactStep = {cell: 0}
        self.hazardList[local_id] = hazard
        if int(self.cellTracker.getCellState(cell)) < 1:
            self.cellTracker.setCellState(cell, 1)
        return local_id
        
    """Helper functions"""
    @staticmethod
    def _validate_percent_distribution(name, values):
        """Validate a ``[mean, variance]`` pair expressed in percent units."""
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{name} must be a [mean_percent, variance_percent_squared] pair")
        mean, variance = float(values[0]), float(values[1])
        if not math.isfinite(mean) or not 0.0 <= mean <= 100.0:
            raise ValueError(f"{name} mean must be a finite percentage in [0, 100]")
        if not math.isfinite(variance) or variance < 0.0:
            raise ValueError(f"{name} variance must be finite and non-negative")
        return (mean, variance)

    @classmethod
    def sampleProbability(cls, percent_distribution, rng=None):
        """Sample a clipped probability from percent mean/variance inputs."""
        mean, variance = cls._validate_percent_distribution(
            "probability distribution", percent_distribution
        )
        normal = np.random.normal if rng is None else rng.normal
        sampled_percent = float(normal(loc=mean, scale=math.sqrt(variance)))
        return cls.oneClip(sampled_percent / 100.0)
    
    @staticmethod
    def oneClip(x):
        """
        clamp to [0,1], a helper for probablistic sampling
        """
        return max(0.0, min(1.0, float(x)))
    
    @staticmethod
    def distance(a, b):
        """
        Returns the Euclidean distance between 2 points (x1, y1) and (x2, y2)
        """
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def assert_grid_coherent(self, *, require_initialized=True):
        """Validate that every hazard cell belongs to the bound CA grid."""
        if self.cellTracker is None:
            raise RuntimeError("HazardDS has no CellTracker")
        validate = getattr(self.cellTracker, "assert_grid_coherent", None)
        if callable(validate):
            validate(require_initialized=require_initialized)
        if not require_initialized:
            return tuple(self.cellTracker.grid_shape)
        for hazard_id, hazard in self.hazardList.items():
            source = getattr(hazard, "sourceCell", None)
            if source is not None:
                try:
                    self.cellTracker.flatten_cell(source)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has source cell outside grid "
                        f"{self.cellTracker.grid_shape}: {source}"
                    ) from exc
            for cell in getattr(hazard, "impactedCells", ()):
                try:
                    self.cellTracker.flatten_cell(cell)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has impacted cell outside grid "
                        f"{self.cellTracker.grid_shape}: {cell}"
                    ) from exc
            for cell in getattr(hazard, "cellImpactStep", {}):
                try:
                    self.cellTracker.flatten_cell(cell)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has dwell-time cell outside grid "
                        f"{self.cellTracker.grid_shape}: {cell}"
                    ) from exc
            for cell in getattr(hazard, "cellFirstImpactStep", {}):
                try:
                    self.cellTracker.flatten_cell(cell)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Hazard {hazard_id} has first-impact cell outside grid "
                        f"{self.cellTracker.grid_shape}: {cell}"
                    ) from exc
        return tuple(self.cellTracker.grid_shape)

    @staticmethod
    def _cell_tuple(cell):
        return tuple(cell) if isinstance(cell, list) else cell

    def _current_danger(self) -> np.ndarray:
        """Read danger directly from CA state in decision flattening order."""
        self.assert_grid_coherent(require_initialized=True)
        return self.cellTracker.state_vector(max_state=5, normalized=True)

    def _cell_center(self, cell):
        if self.cellTracker is None:
            return None
        try:
            center = self.cellTracker.getCellCenter(cell)
            return float(center[0]), float(center[1])
        except Exception:
            return None

    def _hazard_source_point(self, hazard):
        """Return the exact ignition point when available, else cell centre."""
        source_node = getattr(hazard, "sourceNode", None)
        if source_node is not None:
            try:
                point = (float(source_node.nodeX), float(source_node.nodeY))
                if all(math.isfinite(value) for value in point):
                    return point
            except (AttributeError, TypeError, ValueError):
                pass
        source_cell = self._cell_tuple(getattr(hazard, "sourceCell", None))
        return None if source_cell is None else self._cell_center(source_cell)

    def _severity_cap(self, source_cell, source_point, cell) -> int:
        """Maximum danger state allowed by source-local physical proximity."""
        cell = self._cell_tuple(cell)
        source_cell = self._cell_tuple(source_cell)
        if source_cell is not None and cell == source_cell:
            return 5
        if source_point is None:
            return 3
        distance = self.cellTracker.minimumDistanceToCell(source_point, cell)
        return 4 if distance <= self.severe_radius_m + 1e-9 else 3

    def _severity_level(self, elapsed_minutes, cap, *, active=True) -> int:
        """Map elapsed impact time to a bounded, source-local danger level."""
        elapsed = max(0.0, float(elapsed_minutes))
        level = 1 + sum(
            elapsed + 1e-12 >= threshold
            for threshold in self.severity_transition_minutes
        )
        level = min(int(cap), int(level))
        return level if active else min(2, level)

    def _ensure_hazard_bookkeeping(self, hazard) -> None:
        """Normalize one hazard's clock and per-cell first-impact ledger."""
        if not hasattr(hazard, "step"):
            hazard.step = int(getattr(hazard, "age", 0))
        if not hasattr(hazard, "cellImpactStep"):
            hazard.cellImpactStep = {}
        if not hasattr(hazard, "cellFirstImpactStep"):
            hazard.cellFirstImpactStep = dict(hazard.cellImpactStep)
        impacted = []
        seen = set()
        for value in getattr(hazard, "impactedCells", ()):
            cell = self._cell_tuple(value)
            if cell not in seen:
                seen.add(cell)
                impacted.append(cell)
        hazard.impactedCells = impacted
        first = {
            self._cell_tuple(cell): int(step)
            for cell, step in getattr(hazard, "cellFirstImpactStep", {}).items()
        }
        entered = {
            self._cell_tuple(cell): int(step)
            for cell, step in getattr(hazard, "cellImpactStep", {}).items()
        }
        source_cell = self._cell_tuple(getattr(hazard, "sourceCell", None))
        for cell in impacted:
            default = 0 if cell == source_cell else int(hazard.step)
            first.setdefault(cell, int(entered.get(cell, default)))
            entered.setdefault(cell, first[cell])
        hazard.cellFirstImpactStep = first
        hazard.cellImpactStep = entered

    def _hazard_cell_level(self, hazard, cell) -> int:
        self._ensure_hazard_bookkeeping(hazard)
        cell = self._cell_tuple(cell)
        if not bool(getattr(hazard, "active", True)):
            residual = getattr(hazard, "residualLevelByCell", {})
            if cell in residual:
                return int(residual[cell])
        first_step = int(hazard.cellFirstImpactStep[cell])
        elapsed = (
            max(0, int(hazard.step) - first_step) * self.time_step_minutes
        )
        cap = self._severity_cap(
            getattr(hazard, "sourceCell", None),
            self._hazard_source_point(hazard),
            cell,
        )
        return self._severity_level(
            elapsed,
            cap,
            active=bool(getattr(hazard, "active", True)),
        )

    def _apply_source_local_severity(self) -> int:
        """Aggregate every hazard by maximum severity and update each cell once."""
        targets = {}
        for hazard in self.hazardList.values():
            self._ensure_hazard_bookkeeping(hazard)
            for cell in hazard.impactedCells:
                targets[cell] = max(
                    int(targets.get(cell, 0)),
                    self._hazard_cell_level(hazard, cell),
                )
        changes = 0
        for cell, target in targets.items():
            current = int(self.cellTracker.getCellState(cell))
            if current != int(target):
                self.cellTracker.setCellState(cell, int(target))
                changes += 1
        return changes

    def directional_spread_multiplier(self, source_cell, target_cell) -> float:
        """Compute wind's dimensionless multiplier of physical front speed."""
        if self.wind_speed_m_per_minute <= 0.0 or self.wind_influence <= 0.0:
            return 1.0
        source = self._cell_center(source_cell)
        target = self._cell_center(target_cell)
        if source is None or target is None:
            return 1.0
        dx = target[0] - source[0]
        # MapDS stores local y in metres southward. Convert it to a physical
        # northward component before comparing with the public wind vector,
        # whose positive second component is north.
        north = source[1] - target[1]
        norm = math.hypot(dx, north)
        if norm <= 0.0:
            return 1.0
        angle = math.radians(self.wind_direction_degrees)
        alignment = (
            (dx / norm) * math.cos(angle)
            + (north / norm) * math.sin(angle)
        )
        strength = (
            self.wind_influence
            * self.wind_speed_m_per_minute
            / self.spread_speed_m_per_minute
        )
        return float(np.clip(math.exp(strength * alignment), 0.25, 4.0))

    def _neighbor_distance_m(self, source_cell, target_cell) -> float:
        source = self._cell_center(source_cell)
        target = self._cell_center(target_cell)
        if source is None or target is None:
            raise RuntimeError("Hazard spread requires physical cell centers")
        distance = self.distance(source, target)
        if not math.isfinite(distance) or distance <= 0.0:
            raise RuntimeError("Neighboring hazard cells require positive distance")
        return float(distance)

    def _spread_travel_steps(self, source_cell, target_cell) -> int:
        distance = self._neighbor_distance_m(source_cell, target_cell)
        speed = (
            self.spread_speed_m_per_minute
            * self.directional_spread_multiplier(source_cell, target_cell)
        )
        return max(
            1,
            int(math.ceil(distance / (speed * self.time_step_minutes))),
        )

    def directional_spread_probability(self, base_probability, source_cell, target_cell) -> float:
        """Scale stochastic spread by physical distance and wind transport."""
        base = self.oneClip(base_probability)
        if base <= 0.0 or base >= 1.0:
            return base
        multiplier = self.directional_spread_multiplier(source_cell, target_cell)
        distance = self._neighbor_distance_m(source_cell, target_cell)
        reference_distance = self.spread_speed_m_per_minute * self.time_step_minutes
        exposure = multiplier * reference_distance / distance
        return self.oneClip(1.0 - (1.0 - base) ** exposure)

    def _forecast_record_level(self, record, cell) -> int:
        if not bool(record.get("active", True)) and cell in record.get(
            "residual_levels", {}
        ):
            return int(record["residual_levels"][cell])
        first_step = int(record["first_impacted"][cell])
        elapsed = (
            max(0, int(record["step"]) - first_step)
            * self.time_step_minutes
        )
        cap = self._severity_cap(
            record.get("source_cell"), record.get("source_point"), cell
        )
        return self._severity_level(
            elapsed, cap, active=bool(record.get("active", True))
        )

    def _aggregate_forecast_records(self, states, hazards):
        result = np.asarray(states, dtype=np.int16).copy()
        targets = {}
        for hazard in hazards:
            for cell in hazard["impacted"]:
                targets[cell] = max(
                    int(targets.get(cell, 0)),
                    self._forecast_record_level(hazard, cell),
                )
        for cell, target in targets.items():
            result[self.cellTracker.flatten_cell(cell)] = int(target)
        return result

    def _forecast_hazard_record(self, hazard):
        step = int(getattr(hazard, "step", getattr(hazard, "age", 0)))
        impacted = []
        seen = set()
        for value in getattr(hazard, "impactedCells", ()):
            cell = self._cell_tuple(value)
            if cell not in seen:
                seen.add(cell)
                impacted.append(cell)
        legacy_front = not impacted and self.cellTracker is not None
        if legacy_front:
            # Legacy tests and snapshots may omit per-hazard ownership. Recover
            # the observable front without changing the live simulator state.
            impacted = [
                cell
                for cell in self.cellTracker.cell_ids()
                if int(self.cellTracker.getCellState(cell)) > 0
            ]
        source_cell = self._cell_tuple(getattr(hazard, "sourceCell", None))
        entered = {
            self._cell_tuple(cell): int(value)
            for cell, value in getattr(hazard, "cellImpactStep", {}).items()
        }
        first_impacted = {
            self._cell_tuple(cell): int(value)
            for cell, value in getattr(
                hazard, "cellFirstImpactStep", entered
            ).items()
        }
        for cell in impacted:
            default = 0 if cell == source_cell else step
            first_impacted.setdefault(cell, int(entered.get(cell, default)))
        return {
            "active": bool(getattr(hazard, "active", True)),
            "step": step,
            "age": int(getattr(hazard, "age", 0)),
            "lifespan": int(getattr(hazard, "lifespan", 0)),
            "spread_rate": float(getattr(hazard, "spreadRate", 0.0)),
            "source_cell": source_cell,
            "source_point": self._hazard_source_point(hazard),
            "impacted": impacted,
            "first_impacted": first_impacted,
            "residual_levels": {
                self._cell_tuple(cell): int(value)
                for cell, value in getattr(
                    hazard, "residualLevelByCell", {}
                ).items()
            },
            "legacy_front": bool(legacy_front),
        }

    def _latent_ignition_record(self, cell, remaining_steps):
        cell = self._cell_tuple(cell)
        return {
            "active": True,
            "step": 0,
            "age": 0,
            # The forecast window is not the ignition's physical lifetime.
            # Keep it active through the endpoint; the real event receives its
            # episode-relative lifetime when HazardRegimeProcess advances.
            "lifespan": max(1, int(remaining_steps) + 1),
            "spread_rate": self.oneClip(float(self.spreadRate[0]) / 100.0),
            "source_cell": cell,
            "source_point": self._cell_center(cell),
            "impacted": [cell],
            "first_impacted": {cell: 0},
            "residual_levels": {},
            "legacy_front": False,
        }

    def _deterministic_forecast_states(
        self,
        horizon_steps: int,
        future_ignitions=(),
        *,
        configuration_by_step: Optional[Mapping[int, tuple]] = None,
        return_trajectory: bool = False,
    ) -> np.ndarray:
        """Advance a side-effect-free copy of the actual deterministic CA.

        This mirrors ``_spreadUpdateDeterministic`` followed by
        ``terminateHazard``.  It deliberately reuses CellTracker flattening
        and neighbor queries, so a rectangular grid cannot be reinterpreted by
        the forecast layer. ``configuration_by_step`` is used only by belief
        forecasts to apply sampled wind and intensity changes at their proper
        future minute; existing endpoint forecasts retain their static-current
        configuration exactly.
        """
        states = np.rint(self._current_danger() * 5.0).astype(np.int16)
        hazards = [
            self._forecast_hazard_record(hazard)
            for hazard in self.hazardList.values()
        ]
        horizon = max(0, int(horizon_steps))
        ignitions = {}
        for offset, cell in future_ignitions:
            offset = int(offset)
            if 1 <= offset <= horizon:
                ignitions.setdefault(offset, []).append(self._cell_tuple(cell))
        trajectory = []

        for offset in range(1, horizon + 1):
            if configuration_by_step is not None:
                configuration = configuration_by_step.get(offset)
                if configuration is not None:
                    speed, direction, intensity = configuration
                    self.configure_wind(
                        speed_m_per_minute=float(speed),
                        direction_degrees=float(direction),
                        influence=self.wind_influence,
                        time_step_minutes=self.time_step_minutes,
                    )
                    self.set_intensity(float(intensity))
            for cell in ignitions.get(offset, ()):
                hazards.append(
                    self._latent_ignition_record(cell, horizon - offset + 1)
                )
            for hazard in hazards:
                if not hazard["active"]:
                    continue
                hazard["step"] += 1
                fire_front = list(hazard["impacted"])
                for cell in fire_front:
                    hazard["first_impacted"].setdefault(
                        cell, hazard["step"]
                    )

                newly_impacted = []
                for cell in fire_front:
                    for neighbor in self.cellTracker.getNeighborCells(cell):
                        travel_steps = self._spread_travel_steps(cell, neighbor)
                        if (
                            hazard["step"]
                            - int(hazard["first_impacted"][cell])
                            >= travel_steps
                            and neighbor not in hazard["impacted"]
                        ):
                            hazard["first_impacted"][neighbor] = hazard["step"]
                            if neighbor not in newly_impacted:
                                newly_impacted.append(neighbor)
                for cell in newly_impacted:
                    if cell not in hazard["impacted"]:
                        hazard["impacted"].append(cell)

            states = self._aggregate_forecast_records(states, hazards)

            # Match the simulator's end-of-step hazard lifecycle exactly.
            for hazard in hazards:
                if not hazard["active"]:
                    continue
                hazard["age"] += 1
                if hazard["lifespan"] <= 0 or hazard["age"] < hazard["lifespan"]:
                    continue
                hazard["residual_levels"] = {
                    cell: min(2, self._forecast_record_level(hazard, cell))
                    for cell in hazard["impacted"]
                }
                hazard["active"] = False
            states = self._aggregate_forecast_records(states, hazards)
            if return_trajectory:
                trajectory.append(states.copy())
        if return_trajectory:
            if trajectory:
                return np.stack(trajectory, axis=0)
            return np.empty(
                (0, int(self.cellTracker.number_of_cells)), dtype=np.int16
            )
        return states

    def _aggregate_stochastic_forecast_records(
        self, baseline: np.ndarray, records: Sequence[Mapping]
    ) -> np.ndarray:
        """Map incremental mean-field arrival records to normalized danger."""
        expected = np.asarray(baseline, dtype=np.float64).copy()
        touched = set()
        legacy_floor = set()
        per_hazard = []
        for record in records:
            values = np.zeros_like(expected)
            for cell, arrivals in record["arrivals"].items():
                cap = self._severity_cap(
                    record.get("source_cell"),
                    record.get("source_point"),
                    cell,
                )
                value = 0.0
                for mass, first_step in arrivals:
                    elapsed = (
                        max(0, int(record["step"]) - int(first_step))
                        * self.time_step_minutes
                    )
                    value += float(mass) * self._severity_level(
                        elapsed,
                        cap,
                        active=bool(record["active"]),
                    )
                index = int(self.cellTracker.flatten_cell(cell))
                values[index] = value
                touched.add(index)
                if bool(record.get("legacy_front", False)):
                    legacy_floor.add(index)
            per_hazard.append(values)
        if per_hazard and touched:
            combined = np.maximum.reduce(per_hazard)
            indices = np.asarray(sorted(touched), dtype=np.int64)
            expected[indices] = combined[indices]
            if legacy_floor:
                indices = np.asarray(sorted(legacy_floor), dtype=np.int64)
                expected[indices] = np.maximum(
                    expected[indices], baseline[indices]
                )
        return np.clip(expected / 5.0, 0.0, 1.0).astype(np.float32)

    def _stochastic_forecast_danger(
        self,
        horizon_steps: int,
        future_ignitions=(),
        *,
        configuration_by_step: Optional[Mapping[int, tuple]] = None,
        return_trajectory: bool = False,
    ) -> np.ndarray:
        """Mean-field forecast; dispatches to the configured kernel.

        ``vectorized`` (default) and ``reference`` return bit-identical arrays
        and have identical side effects; ``reference`` is kept for audits and
        the equivalence tests.
        """
        kernel = getattr(self, "forecast_kernel", DEFAULT_FORECAST_KERNEL)
        if kernel == "reference":
            return self._stochastic_forecast_danger_reference(
                horizon_steps,
                future_ignitions,
                configuration_by_step=configuration_by_step,
                return_trajectory=return_trajectory,
            )
        return _stochastic_forecast_danger_vectorized(
            self,
            horizon_steps,
            future_ignitions,
            configuration_by_step=configuration_by_step,
            return_trajectory=return_trajectory,
        )

    def _stochastic_forecast_danger_reference(
        self,
        horizon_steps: int,
        future_ignitions=(),
        *,
        configuration_by_step: Optional[Mapping[int, tuple]] = None,
        return_trajectory: bool = False,
    ) -> np.ndarray:
        """Mean-field forecast of the stochastic front with exact severity ages.

        v65: retained unchanged as the reference implementation of
        :func:`_stochastic_forecast_danger_vectorized`.

        The approximation preserves the stochastic spatial arrival process but
        uses the same elapsed-minute severity rule and source-distance caps as
        runtime.  Per-hazard arrival probabilities are kept separate so a
        latent ignition has its own clock and cannot turn a remote cell into a
        fictitious level-five source.
        """
        horizon = max(0, int(horizon_steps))
        baseline = self._current_danger().astype(np.float64) * 5.0
        records = []

        def prepare(record):
            probability = np.zeros(
                int(self.cellTracker.number_of_cells), dtype=np.float64
            )
            arrivals = {}
            for cell in record["impacted"]:
                index = int(self.cellTracker.flatten_cell(cell))
                probability[index] = 1.0
                arrivals[cell] = [
                    (1.0, int(record["first_impacted"][cell]))
                ]
            record["probability"] = probability
            record["arrivals"] = arrivals
            return record

        for hazard in self.hazardList.values():
            records.append(prepare(self._forecast_hazard_record(hazard)))

        ignitions = {}
        for offset, cell in future_ignitions:
            offset = int(offset)
            if 1 <= offset <= horizon:
                ignitions.setdefault(offset, []).append(self._cell_tuple(cell))
        trajectory = []

        for offset in range(1, horizon + 1):
            if configuration_by_step is not None:
                configuration = configuration_by_step.get(offset)
                if configuration is not None:
                    speed, direction, intensity = configuration
                    self.configure_wind(
                        speed_m_per_minute=float(speed),
                        direction_degrees=float(direction),
                        influence=self.wind_influence,
                        time_step_minutes=self.time_step_minutes,
                    )
                    self.set_intensity(float(intensity))
            for cell in ignitions.get(offset, ()):
                records.append(
                    prepare(
                        self._latent_ignition_record(
                            cell, horizon - offset + 1
                        )
                    )
                )

            for record in records:
                if not record["active"]:
                    continue
                record["step"] += 1
                current = record["probability"]
                probability_not_arriving = np.ones_like(current)
                for source_index, source_probability in enumerate(current):
                    if source_probability <= 0.0:
                        continue
                    source_cell = self.cellTracker.unflatten_cell(source_index)
                    for target_cell in self.cellTracker.getNeighborCells(source_cell):
                        target_index = int(
                            self.cellTracker.flatten_cell(target_cell)
                        )
                        edge_probability = self.directional_spread_probability(
                            self.oneClip(
                                float(record["spread_rate"])
                                * float(self.intensity_multiplier)
                            ),
                            source_cell,
                            target_cell,
                        )
                        probability_not_arriving[target_index] *= (
                            1.0
                            - float(source_probability)
                            * float(edge_probability)
                        )
                new_mass = (1.0 - current) * (
                    1.0 - probability_not_arriving
                )
                record["probability"] = np.clip(
                    current + new_mass, 0.0, 1.0
                )
                for index in np.flatnonzero(new_mass > 1e-15):
                    cell = self.cellTracker.unflatten_cell(int(index))
                    record["arrivals"].setdefault(cell, []).append(
                        (float(new_mass[index]), int(record["step"]))
                    )

            for record in records:
                if not record["active"]:
                    continue
                record["age"] += 1
                if (
                    int(record["lifespan"]) > 0
                    and int(record["age"]) >= int(record["lifespan"])
                ):
                    residual = {}
                    for cell, arrivals in record["arrivals"].items():
                        cap = self._severity_cap(
                            record.get("source_cell"),
                            record.get("source_point"),
                            cell,
                        )
                        value = 0.0
                        for mass, first_step in arrivals:
                            elapsed = (
                                max(
                                    0,
                                    int(record["step"])
                                    - int(first_step),
                                )
                                * self.time_step_minutes
                            )
                            value += float(mass) * self._severity_level(
                                elapsed, cap, active=True
                            )
                        residual[cell] = min(2.0, value)
                    record["residual_levels"] = residual
                    record["active"] = False
            if return_trajectory:
                trajectory.append(
                    self._aggregate_stochastic_forecast_records(
                        baseline, records
                    )
                )

        if return_trajectory:
            if trajectory:
                return np.stack(trajectory, axis=0)
            return np.empty(
                (0, int(self.cellTracker.number_of_cells)), dtype=np.float32
            )
        return self._aggregate_stochastic_forecast_records(baseline, records)

    def forecast_danger_by_cell(
        self, horizon_steps: int, future_ignitions=()
    ) -> np.ndarray:
        """Forecast danger on the exact runtime CA grid without random draws."""
        if self.cellTracker is None:
            return np.empty(0, dtype=np.float32)
        horizon = max(0, int(horizon_steps))
        current = self._current_danger()
        if horizon == 0 or (
            not future_ignitions
            and not any(
                bool(getattr(hazard, "active", True))
                for hazard in self.hazardList.values()
            )
        ):
            return current.astype(np.float32)
        if str(self.cell_state_evolution_mode).lower() == "deterministic":
            states = self._deterministic_forecast_states(
                horizon, future_ignitions=future_ignitions
            )
            return np.clip(states.astype(np.float64) / 5.0, 0.0, 1.0).astype(
                np.float32
            )
        return self._stochastic_forecast_danger(
            horizon, future_ignitions=future_ignitions
        )

    def _regime_particle_forecast_trajectory(
        self,
        horizon_steps: int,
        particle: HazardRegimeProcess,
    ) -> np.ndarray:
        """Evolve one supplied regime future stepwise from the live CA state."""
        horizon = max(0, int(horizon_steps))
        node_count = int(self.cellTracker.number_of_cells)
        if horizon == 0:
            return np.empty((0, node_count), dtype=np.float32)

        current_time = int(particle.current_time)
        direction = float(particle.wind_direction_degrees)
        base_speed = float(particle.base_wind_speed_m_per_minute)
        speed = base_speed * float(particle.wind_speed_multiplier)
        intensity = float(particle.intensity_multiplier)
        configurations = {}
        ignitions = []
        scheduled = tuple(
            sorted(
                getattr(particle, "scheduled", ()),
                key=lambda item: (int(item["time"]), str(item["kind"])),
            )
        )
        event_index = 0
        while (
            event_index < len(scheduled)
            and int(scheduled[event_index]["time"]) <= current_time
        ):
            event_index += 1

        for lead in range(1, horizon + 1):
            endpoint = current_time + lead
            while (
                event_index < len(scheduled)
                and int(scheduled[event_index]["time"]) <= endpoint
            ):
                event = scheduled[event_index]
                payload = event["payload"]
                kind = str(event["kind"])
                if kind == "wind":
                    direction = (
                        direction + float(payload["direction_delta"])
                    ) % 360.0
                    speed = base_speed * float(payload["speed_multiplier"])
                elif kind == "intensity":
                    intensity = float(payload["multiplier"])
                elif kind == "ignition":
                    ignitions.append(
                        (
                            int(event["time"]) - current_time,
                            self.cellTracker.unflatten_cell(
                                int(payload["cell_index"])
                            ),
                        )
                    )
                event_index += 1
            configurations[lead] = (speed, direction, intensity)

        if str(self.cell_state_evolution_mode).lower() == "deterministic":
            states = self._deterministic_forecast_states(
                horizon,
                future_ignitions=tuple(ignitions),
                configuration_by_step=configurations,
                return_trajectory=True,
            )
            return np.clip(
                np.asarray(states, dtype=np.float64) / 5.0, 0.0, 1.0
            ).astype(np.float32)
        return self._stochastic_forecast_danger(
            horizon,
            future_ignitions=tuple(ignitions),
            configuration_by_step=configurations,
            return_trajectory=True,
        )

    def forecast_distribution_by_cell(
        self,
        horizon_steps: int,
        particles: Optional[Sequence[HazardRegimeProcess]] = None,
    ) -> dict[str, np.ndarray]:
        """Summarize the next-front distribution from current spread momentum.

        The one-step forecast supplies the local front velocity.  The longer
        forecast supplies the expected danger at the next decision.  For a
        presently unaffected fraction ``1-current``, the implied escalation
        probability is the fraction expected to become affected by the
        horizon.  Its normalized Bernoulli variance is exposed as uncertainty.
        No simulator random stream is consumed and no realized future is used.
        """
        if self.cellTracker is None:
            empty = np.empty(0, dtype=np.float32)
            return {
                "expected": empty,
                "momentum": empty,
                "escalation_probability": empty,
                "uncertainty": empty,
            }
        node_count = int(self.cellTracker.number_of_cells)
        current = self._current_danger()
        if particles is not None:
            particles = tuple(particles)
            if not particles:
                raise ValueError("forecast particles must be non-empty")
            original = (
                self.wind_speed_m_per_minute,
                self.wind_direction_degrees,
                self.wind_influence,
                self.time_step_minutes,
                self.intensity_multiplier,
            )
            forecasts = []
            one_step_forecasts = []
            try:
                for particle in particles:
                    trajectory = np.asarray(
                        self._regime_particle_forecast_trajectory(
                            max(1, int(horizon_steps)), particle
                        ),
                        dtype=np.float64,
                    )
                    one_step_forecasts.append(trajectory[0])
                    forecasts.append(trajectory[-1])
            finally:
                self.configure_wind(
                    speed_m_per_minute=original[0],
                    direction_degrees=original[1],
                    influence=original[2],
                    time_step_minutes=original[3],
                )
                self.set_intensity(original[4])
            samples = np.stack(forecasts, axis=0)
            expected = np.mean(samples, axis=0)
            one_step = np.mean(np.stack(one_step_forecasts, axis=0), axis=0)
            momentum = np.clip(one_step - current, -1.0, 1.0)
            escalation = np.mean(
                samples > current[None, :] + 1e-12, axis=0
            )
            uncertainty = np.clip(4.0 * np.var(samples, axis=0), 0.0, 1.0)
            return {
                "expected": expected.astype(np.float32),
                "momentum": momentum.astype(np.float32),
                "escalation_probability": escalation.astype(np.float32),
                "uncertainty": uncertainty.astype(np.float32),
            }
        one_step = np.asarray(
            self.forecast_danger_by_cell(1), dtype=np.float64
        ).reshape(node_count)
        expected = np.asarray(
            self.forecast_danger_by_cell(max(1, int(horizon_steps))),
            dtype=np.float64,
        ).reshape(node_count)
        momentum = np.clip(one_step - current, -1.0, 1.0)
        if str(self.cell_state_evolution_mode).lower() == "deterministic":
            escalation = (expected > current + 1e-12).astype(np.float64)
            uncertainty = np.zeros(node_count, dtype=np.float64)
        else:
            unaffected = np.maximum(1.0 - current, 1e-12)
            escalation = np.where(
                current < 1.0 - 1e-12,
                (expected - current) / unaffected,
                0.0,
            )
            escalation = np.clip(escalation, 0.0, 1.0)
            uncertainty = np.clip(
                4.0 * escalation * (1.0 - escalation), 0.0, 1.0
            )
        return {
            "expected": expected.astype(np.float32),
            "momentum": momentum.astype(np.float32),
            "escalation_probability": escalation.astype(np.float32),
            "uncertainty": uncertainty.astype(np.float32),
        }

    def _compute_belief_member_trajectories(
        self,
        horizon: int,
        count: int,
        rng: Optional[np.random.Generator],
        use_belief: bool,
    ) -> tuple[list, Optional[dict]]:
        """Simulate the forecast ensemble; the v64 member loop, unchanged.

        Returns the per-member danger trajectories (``(horizon, cells)``
        float64, or the current danger as one row when ``horizon == 0``) and
        the caller generator's state after particle sampling.
        """
        node_count = int(self.cellTracker.number_of_cells)
        current = np.asarray(self._current_danger(), dtype=np.float64).reshape(
            node_count
        )
        if use_belief:
            forecast_rng = rng if rng is not None else np.random.default_rng(0)
            particles = tuple(self.regime.belief_particles(count, forecast_rng))
        else:
            particles = (None,)
        original = (
            self.wind_speed_m_per_minute,
            self.wind_direction_degrees,
            self.wind_influence,
            self.time_step_minutes,
            self.intensity_multiplier,
        )
        samples = []
        try:
            for particle in particles:
                if particle is None:
                    if str(self.cell_state_evolution_mode).lower() == "deterministic":
                        raw = self._deterministic_forecast_states(
                            horizon, return_trajectory=True
                        )
                        trajectory = np.asarray(raw, dtype=np.float64) / 5.0
                    else:
                        trajectory = np.asarray(
                            self._stochastic_forecast_danger(
                                horizon, return_trajectory=True
                            ),
                            dtype=np.float64,
                        )
                else:
                    trajectory = np.asarray(
                        self._regime_particle_forecast_trajectory(
                            horizon, particle
                        ),
                        dtype=np.float64,
                    )
                if horizon == 0:
                    samples.append(current.reshape(1, node_count))
                    continue
                samples.append(trajectory)
        finally:
            self.configure_wind(
                speed_m_per_minute=original[0],
                direction_degrees=original[1],
                influence=original[2],
                time_step_minutes=original[3],
            )
            self.set_intensity(original[4])
        rng_state_after = (
            copy.deepcopy(rng.bit_generator.state)
            if use_belief and rng is not None
            else None
        )
        return samples, rng_state_after

    def _belief_memo_key(
        self,
        horizon: int,
        count: int,
        rng: Optional[np.random.Generator],
        use_belief: bool,
    ):
        """Return the memo key, or ``None`` when memoization is inactive.

        The key is ``(scope, fingerprint)``.  The scope (batch token, tape)
        is what makes reuse exact: within one tape the hazard and regime
        evolve identically for every candidate.  The fingerprint covers every
        input the ensemble reads, so a scope misuse can only cause a miss.
        """
        scope = _BELIEF_MEMO_SCOPE
        if (
            scope is None
            or not bool(getattr(self, "forecast_memo_enabled", True))
            or self.cellTracker is None
        ):
            return None
        digest = hashlib.sha256()

        def put(label: str, value) -> None:
            digest.update(label.encode("utf-8"))
            digest.update(_canonical_repr(value).encode("utf-8"))

        put("ensemble", (int(horizon), int(count), str(self.cell_state_evolution_mode)))
        put(
            "rng",
            rng.bit_generator.state if use_belief and rng is not None else None,
        )
        digest.update(
            np.ascontiguousarray(self._current_danger(), dtype="<f8").tobytes()
        )
        for hazard_id in sorted(self.hazardList, key=repr):
            put(
                "hazard",
                (hazard_id, self._forecast_hazard_record(self.hazardList[hazard_id])),
            )
        put(
            "physics",
            (
                self.wind_speed_m_per_minute,
                self.wind_direction_degrees,
                self.wind_influence,
                self.time_step_minutes,
                self.intensity_multiplier,
                self.spread_speed_m_per_minute,
                self.severity_transition_minutes,
                self.severe_radius_m,
                tuple(self.spreadRate),
            ),
        )
        if use_belief:
            regime = self.regime
            put(
                "regime",
                (
                    getattr(regime, "current_time", None),
                    getattr(regime, "public", {}),
                    getattr(regime, "wind_direction_degrees", None),
                    getattr(regime, "base_wind_direction_degrees", None),
                    getattr(regime, "base_wind_speed_m_per_minute", None),
                    getattr(regime, "wind_speed_multiplier", None),
                    getattr(regime, "intensity_multiplier", None),
                    getattr(regime, "intensity_states", None),
                    getattr(regime, "intensity_belief", None),
                    getattr(regime, "intensity_belief_time", None),
                    getattr(regime, "intensity_belief_updates", None),
                    sorted(
                        getattr(
                            getattr(regime, "map_ds", None), "closed_edges", {}
                        ).items(),
                        key=lambda item: repr(item[0]),
                    ),
                ),
            )
        return (scope, digest.hexdigest())

    def _belief_member_trajectories(
        self,
        horizon: int,
        count: int,
        rng: Optional[np.random.Generator],
    ) -> tuple[list, bool]:
        """Ensemble trajectories, memoized inside a branch-worker scope (S3)."""
        regime = self.regime
        use_belief = regime is not None and bool(getattr(regime, "enabled", False))
        key = self._belief_memo_key(horizon, count, rng, use_belief)
        if key is not None:
            cached = _BELIEF_TRAJECTORY_MEMO.get(key)
            if cached is not None:
                samples, rng_state_after = cached
                fraction = float(getattr(self, "forecast_memo_verify_fraction", 0.0))
                if fraction > 0.0 and int(key[1][:8], 16) < fraction * float(1 << 32):
                    probe = None if rng is None else copy.deepcopy(rng)
                    fresh, fresh_state = self._compute_belief_member_trajectories(
                        horizon, count, probe, use_belief
                    )
                    if (
                        len(fresh) != len(samples)
                        or any(
                            not np.array_equal(left, right)
                            for left, right in zip(fresh, samples)
                        )
                        or fresh_state != rng_state_after
                    ):
                        raise RuntimeError(
                            "Belief forecast memo hit differs from recomputation: "
                            f"scope={key[0]!r}. The hazard is not action-"
                            "independent within this branch scope."
                        )
                    _BELIEF_MEMO_STATISTICS["verified"] += 1
                if rng is not None and rng_state_after is not None:
                    rng.bit_generator.state = copy.deepcopy(rng_state_after)
                _BELIEF_MEMO_STATISTICS["hits"] += 1
                return list(samples), use_belief
            _BELIEF_MEMO_STATISTICS["misses"] += 1
        samples, rng_state_after = self._compute_belief_member_trajectories(
            horizon, count, rng, use_belief
        )
        if key is not None:
            frozen = []
            for sample in samples:
                sample = np.array(sample, dtype=np.float64, copy=True)
                sample.setflags(write=False)
                frozen.append(sample)
            while len(_BELIEF_TRAJECTORY_MEMO) >= _BELIEF_TRAJECTORY_MEMO_LIMIT:
                _BELIEF_TRAJECTORY_MEMO.pop(next(iter(_BELIEF_TRAJECTORY_MEMO)))
            _BELIEF_TRAJECTORY_MEMO[key] = (tuple(frozen), rng_state_after)
            samples = frozen
        return samples, use_belief

    def belief_overrun_first_passage_by_cell(
        self,
        horizon_steps: int,
        danger_threshold_level: float,
        *,
        ensemble_size: int = 8,
        rng: Optional[np.random.Generator] = None,
        lead_time_quantile: float = 0.5,
        next_event_mask: Optional[Sequence[bool]] = None,
    ) -> dict[str, np.ndarray]:
        """Forecast each cell's first crossing of a danger threshold.

        When a hazard regime is active, this method samples fresh futures from
        :meth:`HazardRegimeProcess.belief_particles`.  It never reads the
        episode's latent realized tape, so two states with the same public
        signals and the same forecast RNG have identical distributions.  The
        live hazard RNG and live CA state are not advanced.

        The result also contains the four endpoint keys returned by
        :meth:`forecast_distribution_by_cell` (``expected``, ``momentum``,
        ``escalation_probability`` and ``uncertainty``), computed from the
        same trajectories so observation construction need not forecast twice.

        First-passage times are measured in simulator steps.  A cell that is
        already at or above the threshold has lead time zero.  A member that
        does not cross within the forecast window is right-censored at
        ``horizon_steps + 1``.  ``next_overrun_probability`` excludes cells
        that have already crossed; ties for the earliest future crossing split
        one unit of probability equally.  Thus its sum is the ensemble
        probability that at least one currently-safe cell crosses within the
        horizon.

        ``next_event_mask`` optionally defines a subset such as cells holding
        operational shelters. ``next_priority_overrun_probability`` then
        identifies the first future crossing within that subset; an earlier
        crossing in an unrelated empty cell does not erase the shelter signal.

        The spatial stochastic CA forecast is mean-field.  Consequently the
        returned probabilities describe variation across public-belief regime
        futures (including latent ignitions), while each member's crossing is
        based on its expected cell danger trajectory.
        """
        if self.cellTracker is None:
            empty = np.empty(0, dtype=np.float32)
            return {
                "expected": empty,
                "momentum": empty,
                "escalation_probability": empty,
                "uncertainty": empty,
                "currently_overrun": empty,
                "crossing_probability": empty,
                "future_crossing_probability": empty,
                "expected_first_crossing_steps": empty,
                "conditional_expected_first_crossing_steps": empty,
                "quantile_first_crossing_steps": empty,
                "expected_first_crossing_minutes": empty,
                "quantile_first_crossing_minutes": empty,
                "next_overrun_probability": empty,
                "next_priority_overrun_probability": empty,
                "next_overrun_rank": np.empty(0, dtype=np.int32),
                "crossing_cdf_by_step": np.empty((0, 0), dtype=np.float32),
            }

        horizon = int(horizon_steps)
        if horizon < 0:
            raise ValueError("horizon_steps must be non-negative")
        threshold_level = float(danger_threshold_level)
        if not math.isfinite(threshold_level) or not 0.0 < threshold_level <= 5.0:
            raise ValueError("danger_threshold_level must lie in (0, 5]")
        quantile = float(lead_time_quantile)
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("lead_time_quantile must lie in [0, 1]")
        count = int(ensemble_size)
        if count <= 0:
            raise ValueError("ensemble_size must be positive")

        node_count = int(self.cellTracker.number_of_cells)
        priority_mask = (
            np.ones(node_count, dtype=np.bool_)
            if next_event_mask is None
            else np.asarray(next_event_mask, dtype=np.bool_).reshape(-1)
        )
        if priority_mask.shape != (node_count,):
            raise ValueError(
                "next_event_mask must contain one boolean per hazard cell"
            )
        threshold = threshold_level / 5.0
        current = np.asarray(self._current_danger(), dtype=np.float64).reshape(
            node_count
        )
        currently_overrun = current >= threshold - 1e-12

        # v65 (S3): member trajectories come from one helper that a branch
        # worker may memoize across candidates of the same future.  The
        # threshold, first-passage and mask post-processing below is
        # unchanged and runs on every call.
        trajectory_samples, use_belief = self._belief_member_trajectories(
            horizon, count, rng
        )
        censored = horizon + 1
        first_passage = np.full(
            (len(trajectory_samples), node_count), censored, dtype=np.int32
        )
        first_passage[:, currently_overrun] = 0
        if horizon > 0:
            for member_index, trajectory in enumerate(trajectory_samples):
                crossed_by_step = trajectory >= threshold - 1e-12
                has_crossing = np.any(crossed_by_step, axis=0)
                first_indices = np.argmax(crossed_by_step, axis=0)
                newly_crossed = (
                    (first_passage[member_index] == censored) & has_crossing
                )
                first_passage[member_index, newly_crossed] = (
                    first_indices[newly_crossed] + 1
                )

        endpoint_samples = np.stack(
            [sample[-1] for sample in trajectory_samples], axis=0
        )
        one_step_samples = np.stack(
            [sample[0] for sample in trajectory_samples], axis=0
        )
        expected_danger = np.mean(endpoint_samples, axis=0)
        one_step_danger = np.mean(one_step_samples, axis=0)
        danger_momentum = np.clip(one_step_danger - current, -1.0, 1.0)
        if use_belief:
            danger_escalation = np.mean(
                endpoint_samples > current[None, :] + 1e-12,
                axis=0,
            )
            danger_uncertainty = np.clip(
                4.0 * np.var(endpoint_samples, axis=0), 0.0, 1.0
            )
        elif str(self.cell_state_evolution_mode).lower() == "deterministic":
            danger_escalation = (
                expected_danger > current + 1e-12
            ).astype(np.float64)
            danger_uncertainty = np.zeros(node_count, dtype=np.float64)
        else:
            unaffected = np.maximum(1.0 - current, 1e-12)
            danger_escalation = np.where(
                current < 1.0 - 1e-12,
                (expected_danger - current) / unaffected,
                0.0,
            )
            danger_escalation = np.clip(danger_escalation, 0.0, 1.0)
            danger_uncertainty = np.clip(
                4.0 * danger_escalation * (1.0 - danger_escalation),
                0.0,
                1.0,
            )

        crossed = first_passage <= horizon
        future_crossed = (first_passage >= 1) & crossed
        crossing_probability = np.mean(crossed, axis=0)
        future_crossing_probability = np.mean(future_crossed, axis=0)
        crossing_cdf_by_step = np.stack(
            [
                np.mean(first_passage <= lead, axis=0)
                for lead in range(horizon + 1)
            ],
            axis=0,
        )
        expected_steps = np.mean(first_passage, axis=0)
        crossing_counts = np.sum(crossed, axis=0)
        conditional_expected = np.full(node_count, float(censored))
        available = crossing_counts > 0
        conditional_expected[available] = (
            np.sum(np.where(crossed, first_passage, 0), axis=0)[available]
            / crossing_counts[available]
        )
        try:
            quantile_steps = np.quantile(
                first_passage,
                quantile,
                axis=0,
                method="higher",
            ).astype(np.float64)
        except TypeError:  # NumPy < 1.22 used the older keyword.
            quantile_steps = np.quantile(
                first_passage,
                quantile,
                axis=0,
                interpolation="higher",
            ).astype(np.float64)

        next_probability = np.zeros(node_count, dtype=np.float64)
        next_priority_probability = np.zeros(node_count, dtype=np.float64)
        for member in first_passage:
            future = np.flatnonzero((member >= 1) & (member <= horizon))
            if future.size:
                earliest = int(np.min(member[future]))
                tied = future[member[future] == earliest]
                next_probability[tied] += 1.0 / float(tied.size)
            priority_future = np.flatnonzero(
                priority_mask & (member >= 1) & (member <= horizon)
            )
            if priority_future.size:
                earliest_priority = int(np.min(member[priority_future]))
                tied_priority = priority_future[
                    member[priority_future] == earliest_priority
                ]
                next_priority_probability[tied_priority] += (
                    1.0 / float(tied_priority.size)
                )
        next_probability /= float(len(trajectory_samples))
        next_priority_probability /= float(len(trajectory_samples))
        next_rank = np.zeros(node_count, dtype=np.int32)
        candidates = np.flatnonzero(next_probability > 0.0)
        if candidates.size:
            order = candidates[
                np.lexsort(
                    (
                        candidates,
                        expected_steps[candidates],
                        -next_probability[candidates],
                    )
                )
            ]
            next_rank[order] = np.arange(1, order.size + 1, dtype=np.int32)

        minutes_per_step = float(self.time_step_minutes)
        return {
            "expected": expected_danger.astype(np.float32),
            "momentum": danger_momentum.astype(np.float32),
            "escalation_probability": danger_escalation.astype(np.float32),
            "uncertainty": danger_uncertainty.astype(np.float32),
            "currently_overrun": currently_overrun.astype(np.float32),
            "crossing_probability": crossing_probability.astype(np.float32),
            "future_crossing_probability": (
                future_crossing_probability.astype(np.float32)
            ),
            "expected_first_crossing_steps": expected_steps.astype(np.float32),
            "conditional_expected_first_crossing_steps": (
                conditional_expected.astype(np.float32)
            ),
            "quantile_first_crossing_steps": quantile_steps.astype(np.float32),
            "expected_first_crossing_minutes": (
                expected_steps * minutes_per_step
            ).astype(np.float32),
            "quantile_first_crossing_minutes": (
                quantile_steps * minutes_per_step
            ).astype(np.float32),
            "next_overrun_probability": next_probability.astype(np.float32),
            "next_priority_overrun_probability": (
                next_priority_probability.astype(np.float32)
            ),
            "next_overrun_rank": next_rank,
            "crossing_cdf_by_step": crossing_cdf_by_step.astype(np.float32),
        }
    
    def setCellTracker(self, cellTracker):
        if cellTracker is None:
            raise ValueError("HazardDS requires a CellTracker")
        validate = getattr(cellTracker, "assert_grid_coherent", None)
        if callable(validate):
            validate(require_initialized=False)
        self.cellTracker = cellTracker
        if self.hazardList and getattr(cellTracker, "cellList", None):
            self.assert_grid_coherent(require_initialized=True)
        
    """Primary functions"""
    def initHazard(self, mapDS, cellTracker = None, lifespan = 60, source_nodes = None):
        """Install the episode's hazards.

        ``source_nodes`` (evaluation stress scenarios only) fixes the source
        road node of hazard ``i`` to ``source_nodes[i]``. The random draw is
        still consumed for every hazard so the global initialization stream,
        and therefore every later random draw, is unchanged.
        """
        # Episode initialization replaces the complete hazard realization.
        # Clear it before binding the new tracker so an earlier episode's
        # front cannot be validated against, or copied into, a new grid.
        self.hazardList.clear()
        self.setCellTracker(cellTracker)
        validate = getattr(self.cellTracker, "assert_grid_coherent", None)
        if callable(validate):
            validate(require_initialized=True)
        
        hid = 0
        
        for _ in range(self.totalHazard):
            localID = hid
            hid += 1
            
            # CSV inputs use percent means and percent-squared variances.
            localSpeedReduct = self.sampleProbability(self.speedReduct, self.rng)
            localCasualtyRate = self.sampleProbability(self.casualtyRate, self.rng)
            localSpreadRate = self.sampleProbability(self.spreadRate, self.rng)
            # define other parameters
            sourceNode = mapDS.assignGenerationNode()
            if (
                source_nodes is not None
                and localID < len(source_nodes)
                and source_nodes[localID] is not None
            ):
                sourceNode = source_nodes[localID]
            
            smokeLevel = 0.0
            heatLevel = 0.0
            impactedCells = []
            
            hazard = hazardAgent(localID, 
                                 localSpeedReduct, 
                                 localSpreadRate, 
                                 localCasualtyRate, 
                                 sourceNode, 
                                 smokeLevel, 
                                 heatLevel, 
                                 impactedCells)
        
            hazard.active = True
            hazard.age = 0
            hazard.lifespan = int(lifespan)
            hazard.sourceCell = None
            hazard.step = 0
            hazard.cellImpactStep = {}
            hazard.cellFirstImpactStep = {}
            
            # sample hazard source location
            if self.cellTracker is not None and sourceNode is not None:
                sourceX, sourceY = float(sourceNode.nodeX), float(sourceNode.nodeY)
                sourceCell = self.cellTracker.locateCell(sourceX, sourceY)
                hazard.sourceCell = sourceCell
                hazard.impactedCells.append(hazard.sourceCell)
                source_key = self._cell_tuple(hazard.sourceCell)
                hazard.cellImpactStep[source_key] = 0
                hazard.cellFirstImpactStep[source_key] = 0
                
                if int(self.cellTracker.getCellState(sourceCell)) < 1:
                    self.cellTracker.setCellState(hazard.sourceCell, 1)
                    
            self.hazardList[localID] = hazard
        self.assert_grid_coherent(require_initialized=True)
    
    def spreadUpdate(self):
        if self.cellTracker is not None:
            self.assert_grid_coherent(require_initialized=True)
        if str(self.cell_state_evolution_mode).lower() == "stochastic":
            return self._spreadUpdateStochastic()
        return self._spreadUpdateDeterministic()

    def _spreadUpdateStochastic(self):
        if self.cellTracker is None:
            return 0
        
        totalChange = 0
        pre_states = np.asarray(
            [
                int(
                    self.cellTracker.getCellState(
                        self.cellTracker.unflatten_cell(index)
                    )
                )
                for index in range(int(self.cellTracker.number_of_cells))
            ],
            dtype=np.int8,
        )
        exposures = {}
        for hazard in self.hazardList.values():
            if not bool(getattr(hazard, "active", True)):
                continue
            base_rate = float(getattr(hazard, "spreadRate", 0.0))
            for source_cell in list(hazard.impactedCells):
                for target_cell in self.cellTracker.getNeighborCells(source_cell):
                    target_index = int(self.cellTracker.flatten_cell(target_cell))
                    if int(pre_states[target_index]) != 0:
                        continue
                    exposures.setdefault(target_index, []).append(
                        (base_rate, tuple(source_cell), tuple(target_cell))
                    )
        
        # evaluate the spread and new impact of each hazard
        for hazard in list(self.hazardList.values()):
            # Ignore hazards that are no longer active (naturally ceased or
            # contained by effort). Severity aging itself is deterministic;
            # stochasticity remains in which cells the front reaches.
            if not bool(getattr(hazard, "active", True)):
                continue
            self._ensure_hazard_bookkeeping(hazard)
            hazard.step = int(hazard.step) + 1

            spread_rate = self.oneClip(
                float(hazard.spreadRate) * float(self.intensity_multiplier)
            )
            fireFront = list(hazard.impactedCells)
            impacted = set(fireFront)
            newlyImpacted = []

            for cell in fireFront:
                neighbors = self.cellTracker.getNeighborCells(cell)
                for neighbor in neighbors:
                    neighbor = self._cell_tuple(neighbor)
                    if neighbor in impacted:
                        continue
                    directional_rate = self.directional_spread_probability(
                        spread_rate,
                        cell,
                        neighbor,
                    )
                    if self.rng.random() < directional_rate:
                        impacted.add(neighbor)
                        newlyImpacted.append(neighbor)
                        if int(self.cellTracker.getCellState(neighbor)) == 0:
                            self.cellTracker.setCellState(neighbor, 1)
                            totalChange += 1

            for cell in newlyImpacted:
                hazard.impactedCells.append(cell)
                hazard.cellFirstImpactStep[cell] = int(hazard.step)
                hazard.cellImpactStep[cell] = int(hazard.step)

        # One cell update after all fronts prevents overlapping hazards from
        # escalating the same global danger state more than once per minute.
        totalChange += self._apply_source_local_severity()

        if self.regime is not None and bool(getattr(self.regime, "enabled", False)):
            newly_impacted = [
                target_index
                for target_index in exposures
                if int(
                    self.cellTracker.getCellState(
                        self.cellTracker.unflatten_cell(target_index)
                    )
                )
                > 0
            ]
            self.regime.observe_spread_transition(
                self,
                exposures,
                newly_impacted,
            )
                
        return totalChange
    
    
    def _spreadUpdateDeterministic(self):
        if self.cellTracker is None:
            return 0

        totalChange = 0
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue
            self._ensure_hazard_bookkeeping(hazard)
            hazard.step = int(hazard.step) + 1
            fireFront = list(getattr(hazard, "impactedCells", []))

            # Deterministic spread: wind changes each neighbor's arrival delay.
            newlyImpacted = []
            impacted = set(fireFront)
            for cell in fireFront:
                for nb in self.cellTracker.getNeighborCells(cell):
                    nb = self._cell_tuple(nb)
                    if nb in impacted:
                        continue
                    travel_steps = self._spread_travel_steps(cell, nb)
                    cell_key = self._cell_tuple(cell)
                    if (
                        hazard.step
                        - int(hazard.cellFirstImpactStep[cell_key])
                        >= travel_steps
                    ):
                        impacted.add(nb)
                        newlyImpacted.append(nb)
                        if int(self.cellTracker.getCellState(nb)) == 0:
                            self.cellTracker.setCellState(nb, 1)
                            totalChange += 1
            for nb in newlyImpacted:
                hazard.impactedCells.append(nb)
                hazard.cellImpactStep[nb] = int(hazard.step)
                hazard.cellFirstImpactStep[nb] = int(hazard.step)

        totalChange += self._apply_source_local_severity()
        return totalChange
            
        # Step 1: (If Statement, nested loop) If the cell is of State 0, iteratively check each neighboring cell to see if trigger Event 
        
        # Step 2: (Else If Statement) If the cell is of State 1, iteratively check and count the number of neighboring cells with State 3 or above
        
        # Step 2.1: Compute the overall SpreadRate = the SumSpreadRate = (\# of State 3 to 5 neighboring cells) * (SpreadRate of the impacting emergency)
        
        # Step 3: (Else Statement) If the cell is of State 2 or above, sample a random integer and compare it to the SpreadRate to determine if to evolve the cell state to the next level
        
    """Heat and Smoke update, following similar logic"""
    def heatUpdate(self):
        """Update cell heat using the paper's Gaussian space-time field."""
        if self.cellTracker is None:
            return 0
        self.assert_grid_coherent(require_initialized=True)

        updates = 0
        contributions = {cell_id: 0.0 for cell_id in self.cellTracker.cellList}
        for hazard in list(self.hazardList.values()):
            # ignore hazards that are no longer active
            if not hazard.active:
                continue
            sourceID = hazard.sourceCell
            if sourceID is None:
                continue
            
            sourceX, sourceY = self.cellTracker.getCellCenter(sourceID)
            age = max(0.0, float(getattr(hazard, "age", 0)))
            for cell in self.cellTracker.cellList:
                cellX, cellY = self.cellTracker.getCellCenter(cell)
                dis = self.distance((sourceX, sourceY), (cellX, cellY))
                newHeatLevel = self.initial_heat_intensity * math.exp(
                    -(dis * dis) / (2.0 * self.heat_spread_radius_m ** 2)
                ) * (1.0 + self.heat_growth_rate * age)
                cell_key = tuple(cell) if isinstance(cell, list) else cell
                contributions[cell_key] += float(newHeatLevel)
                updates += 1

        for cell_id, value in contributions.items():
            self.cellTracker.setHeat(cell_id, value)
                
        return updates
            
        # Step 1: Acquire the current timestep
        
        # Step 2: (for loop, if statement) For each cell $ce_i$, if $ce_i$ is impacted, then acquire its center location and the hazard instance impacting the cell. 
        # Otherwise, moved on to review the next cell
        
        # Step 3: Execute the equation according to the given equation and return the computed heatwave level
            
    def smokeUpdate(self):
        """Update cell smoke using the paper's Gaussian space-time field."""
        if self.cellTracker is None:
            return 0
        self.assert_grid_coherent(require_initialized=True)

        updates = 0
        contributions = {cell_id: 0.0 for cell_id in self.cellTracker.cellList}
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue
            sourceCell = hazard.sourceCell
            if sourceCell is None:
                continue
            
            sourceX, sourceY = self.cellTracker.getCellCenter(sourceCell)
            age = max(0.0, float(getattr(hazard, "age", 0)))
            for cell in self.cellTracker.cellList:
                cellX, cellY = self.cellTracker.getCellCenter(cell)
                dis = self.distance((sourceX, sourceY), (cellX, cellY))
                newSmokeLevel = self.initial_smoke_intensity * math.exp(
                    -(dis * dis) / (2.0 * self.smoke_spread_radius_m ** 2)
                ) * (1.0 + self.smoke_growth_rate * age)
                cell_key = tuple(cell) if isinstance(cell, list) else cell
                contributions[cell_key] += float(newSmokeLevel)
                updates += 1

        for cell_id, value in contributions.items():
            self.cellTracker.setSmoke(cell_id, value)
                
        return updates
        # Step 1: Acquire the current timestep
        
        # Step 2: (for loop, if statement) For each cell $ce_i$, if $ce_i$ is impacted, then acquire its center location and the hazard instance impacting the cell.
        # Otherwise, moved on to review the next cell.
        
        # Step 3: Execute the equation according to the given equation and return the computed smoke intensity level
            
    def terminateHazard(self):
        if self.cellTracker is not None:
            self.assert_grid_coherent(require_initialized=True)
        terminated = 0
        
        for hazard in list(self.hazardList.values()):
            if not getattr(hazard, "active", True):
                continue
            
            hazard.age = int(getattr(hazard, "age", 0)) + 1
            lifespan = int(getattr(hazard, "lifespan", 0))
            
            # check on if the hazard will cease to continue impact
            if lifespan > 0 and hazard.age >= lifespan:
                hazard.residualLevelByCell = {
                    self._cell_tuple(cell): min(
                        2, int(self.cellTracker.getCellState(cell))
                    )
                    for cell in getattr(hazard, "impactedCells", ())
                }
                hazard.active = False
                terminated += 1
        if terminated:
            # Inactive hazards retain at most their realized level-two residual;
            # any overlapping active source supplies its own level. Recomputing
            # the max avoids retaining a stale level five merely because some
            # other low-severity hazard is still present in the same cell.
            self._apply_source_local_severity()
        return terminated
