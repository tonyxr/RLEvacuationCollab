#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

8/7: Constructed the skeletons
9/9: implemented the spreadUpdate function
"""

import copy
from dataclasses import asdict, dataclass
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
        """Mean-field forecast of the stochastic front with exact severity ages.

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

        regime = self.regime
        use_belief = regime is not None and bool(getattr(regime, "enabled", False))
        if use_belief:
            forecast_rng = rng if rng is not None else np.random.default_rng(0)
            particles = tuple(regime.belief_particles(count, forecast_rng))
        else:
            particles = (None,)

        censored = horizon + 1
        first_passage = np.full(
            (len(particles), node_count), censored, dtype=np.int32
        )
        first_passage[:, currently_overrun] = 0
        original = (
            self.wind_speed_m_per_minute,
            self.wind_direction_degrees,
            self.wind_influence,
            self.time_step_minutes,
            self.intensity_multiplier,
        )
        trajectory_samples = []

        try:
            for member_index, particle in enumerate(particles):
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
                    trajectory_samples.append(current.reshape(1, node_count))
                    continue
                trajectory_samples.append(trajectory)
                crossed_by_step = trajectory >= threshold - 1e-12
                has_crossing = np.any(crossed_by_step, axis=0)
                first_indices = np.argmax(crossed_by_step, axis=0)
                newly_crossed = (
                    (first_passage[member_index] == censored) & has_crossing
                )
                first_passage[member_index, newly_crossed] = (
                    first_indices[newly_crossed] + 1
                )
        finally:
            self.configure_wind(
                speed_m_per_minute=original[0],
                direction_degrees=original[1],
                influence=original[2],
                time_step_minutes=original[3],
            )
            self.set_intensity(original[4])

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
        next_probability /= float(len(particles))
        next_priority_probability /= float(len(particles))
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
