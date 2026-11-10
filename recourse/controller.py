"""Forward-looking evacuation orders for shelters about to be overrun.

The decision maker of the frozen model can only open shelters. This controller
adds the missing recourse: at check times it may order the evacuation of a
region whose shelters are forecast to be overrun, while the region's danger is
still below the overrun level. It is used together with any placement policy
(the placement policy is unchanged), so the same order rule applies to every
policy on a tape and policy comparisons stay common-random-number comparisons.

Information. The controller reads only what a decision maker can see: the
public danger field, the public hazard-regime signals (through the belief
particles that the model's own forecasts use), and each shelter's status,
occupants and reservations. It never reads the latent hazard tape.

Two decision rules, both forward-looking:

``forecast``
    Order region ``c`` when the public first-passage forecast gives
    ``P(c reaches the overrun level within L minutes) >= theta`` and at least
    one person is inside or holding a reservation. ``L`` is the lead window
    (default 30 minutes, the observation's two-interval forecast horizon).

``rollout``
    For regions with ``P >= p_screen``, simulate ordering now, ordering at each
    later offset, and never ordering within the look-ahead, on ``M``
    public-belief futures with common random numbers (``recourse.rollout``);
    order now only if that has the lowest mean cost. This is a rollout policy
    for the optimal-stopping problem with base policy "never order", graded
    with the same objective ledger as placement branches.

Checks happen every ``check_interval`` minutes at the decision boundary (after
the placement step); orders execute in the closure phase of the next minute.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .transaction import open_shelters_by_cell, overrun_level, people_at

RULES = ("forecast", "rollout")


@dataclass(frozen=True)
class RecourseParams:
    rule: str = "forecast"
    lead_minutes: int = 30
    threshold: float = 0.5
    check_interval: int = 5
    ensemble_size: int = 8
    min_people: int = 1
    screen_probability: float = 0.05
    rollout_futures: int = 4
    rollout_horizon_minutes: int = 45
    rollout_offsets: tuple = (0, 5, 10, 15, 20, 25)

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}")
        if not 1 <= int(self.lead_minutes) <= 120:
            raise ValueError("lead_minutes must lie in [1, 120]")
        if not 0.0 < float(self.threshold) <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        if not 1 <= int(self.check_interval) <= 60:
            raise ValueError("check_interval must lie in [1, 60]")
        if int(self.ensemble_size) < 1 or int(self.rollout_futures) < 1:
            raise ValueError("ensemble_size and rollout_futures must be positive")
        if int(self.min_people) < 1:
            raise ValueError("min_people must be at least 1")
        if not 0.0 < float(self.screen_probability) <= 1.0:
            raise ValueError("screen_probability must lie in (0, 1]")
        if int(self.rollout_horizon_minutes) < 1:
            raise ValueError("rollout_horizon_minutes must be positive")
        offsets = tuple(int(o) for o in self.rollout_offsets)
        if not offsets or offsets[0] != 0 or sorted(set(offsets)) != list(offsets) or offsets[-1] >= int(
                self.rollout_horizon_minutes):
            raise ValueError("rollout_offsets must start at 0, increase, and end before the horizon")
        object.__setattr__(self, "rollout_offsets", offsets)


@dataclass
class _Episode:
    orders: list = field(default_factory=list)
    evaluations: list = field(default_factory=list)
    checks: int = 0
    forecasts: int = 0
    wall_s: float = 0.0
    ordered_cells: set = field(default_factory=set)
    max_danger_after_order: dict = field(default_factory=dict)


class ShelterRecourseController:
    """See the module docstring. ``params()`` enters the episode key."""

    def __init__(self, **params):
        self.p = RecourseParams(**params)
        self.reset()

    # -- identity ------------------------------------------------------
    def params(self) -> dict:
        values = asdict(self.p)
        values["rollout_offsets"] = list(values["rollout_offsets"])
        if self.p.rule == "forecast":  # rollout-only settings do not change a forecast episode
            for name in ("screen_probability", "rollout_futures", "rollout_horizon_minutes", "rollout_offsets"):
                values.pop(name)
        return values

    def reset(self) -> None:
        self.episode = _Episode()

    # -- decision ------------------------------------------------------
    def _seed(self, core, simulation_time: int, salt: int) -> int:
        base = int(getattr(core, "scenario_seed", 0))
        return int(np.random.SeedSequence([base & 0xFFFFFFFF, int(simulation_time), int(salt), 0x52454353])
                   .generate_state(1)[0])

    def _cell_index(self, core, cell) -> int:
        from CAProcessor import authoritative_grid_shape

        _, cell_y = authoritative_grid_shape(core)
        return int(cell[0]) * int(cell_y) + int(cell[1])

    def candidates(self, core) -> dict:
        """Open regions with at least ``min_people`` inside or holding a reservation."""
        out = {}
        for cell, shelters in open_shelters_by_cell(core).items():
            occupants = reserved = 0
            for shelter in shelters:
                o, r = people_at(core, shelter)
                occupants += o
                reserved += r
            if occupants + reserved >= int(self.p.min_people):
                out[cell] = {"shelter_ids": sorted(int(s.shelterID) for s in shelters),
                             "occupants": occupants, "reserved": reserved}
        return out

    def forecast(self, core, cells, simulation_time: int) -> dict:
        """Public first-passage forecast for the given regions."""
        step_minutes = max(1e-9, float(getattr(core, "timeStepMinutes", 1.0)))
        steps = max(1, int(round(self.p.lead_minutes / step_minutes)))
        number = int(core.cellTracker.number_of_cells)
        mask = np.zeros(number, dtype=bool)
        index = {cell: self._cell_index(core, cell) for cell in cells}
        mask[list(index.values())] = True
        result = core.hazardDS.belief_overrun_first_passage_by_cell(
            steps, float(overrun_level(core)), ensemble_size=int(self.p.ensemble_size),
            rng=np.random.default_rng(self._seed(core, simulation_time, 1)), next_event_mask=mask,
        )
        self.episode.forecasts += 1
        out = {}
        for cell, i in index.items():
            out[cell] = {
                "probability": float(result["future_crossing_probability"][i]),
                "already_crossed": bool(result["currently_overrun"][i]) if len(result["currently_overrun"]) else False,
                "expected_crossing_minutes": float(result["conditional_expected_first_crossing_steps"][i]) * step_minutes,
            }
        return out

    def decide(self, core, simulation_time: int) -> list:
        """Regions to order now (executed in the closure phase of the next minute)."""
        t = int(simulation_time)
        horizon = int(getattr(core, "stopTime", 0)) - 1
        self._track_ordered(core)
        if t % int(self.p.check_interval) != 0 or t >= horizon:
            return []
        started = time.perf_counter()
        try:
            self.episode.checks += 1
            candidates = self.candidates(core)
            if not candidates:
                return []
            forecast = self.forecast(core, list(candidates), t)
            danger = {cell: int(core.cellTracker.getCellState(cell)) for cell in candidates}
            level = overrun_level(core)
            threatened = [c for c in sorted(candidates)
                          if not forecast[c]["already_crossed"] and danger[c] < level]
            if self.p.rule == "forecast":
                chosen = [c for c in threatened if forecast[c]["probability"] >= float(self.p.threshold)]
                evaluations = {}
            else:
                screened = [c for c in threatened if forecast[c]["probability"] >= float(self.p.screen_probability)]
                chosen, evaluations = [], {}
                if screened:
                    from .rollout import evaluate_stopping

                    result = evaluate_stopping(
                        core, screened, offsets=self.p.rollout_offsets, futures=int(self.p.rollout_futures),
                        horizon_minutes=int(self.p.rollout_horizon_minutes), seed=self._seed(core, t, 2),
                    )
                    evaluations = result.as_record()
                    if result.best == 0:
                        chosen = list(screened)
                    self.episode.evaluations.append({"minute": t, "cells": [list(c) for c in screened],
                                                     **evaluations})
            for cell in chosen:
                self.episode.orders.append({
                    "decided_minute": t, "cell": list(cell), "rule": self.p.rule,
                    "probability": forecast[cell]["probability"],
                    "expected_crossing_minutes": forecast[cell]["expected_crossing_minutes"],
                    "danger_at_decision": danger[cell], **candidates[cell],
                })
                self.episode.ordered_cells.add(tuple(cell))
                self.episode.max_danger_after_order[tuple(cell)] = danger[cell]
            return [tuple(c) for c in chosen]
        finally:
            self.episode.wall_s += time.perf_counter() - started

    def _track_ordered(self, core) -> None:
        for cell in self.episode.ordered_cells:
            state = int(core.cellTracker.getCellState(cell))
            if state > self.episode.max_danger_after_order.get(cell, -1):
                self.episode.max_danger_after_order[cell] = state

    def summary(self, core, executed: list) -> dict:
        level = overrun_level(core)
        self._track_ordered(core)
        reached = {cell: self.episode.max_danger_after_order.get(cell, 0) >= level
                   for cell in self.episode.ordered_cells}
        return {
            "params": self.params(),
            "orders": list(self.episode.orders),
            "executed": list(executed),
            "evaluations": list(self.episode.evaluations),
            "checks": int(self.episode.checks),
            "forecasts": int(self.episode.forecasts),
            "wall_s": float(self.episode.wall_s),
            "ordered_regions_reaching_overrun": {f"{c[0]},{c[1]}": bool(v) for c, v in sorted(reached.items())},
            "false_alarm_regions": int(sum(1 for v in reached.values() if not v)),
        }
