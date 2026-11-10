"""Live-episode wiring for the recourse controller (evaluation layer, no frozen file edited).

The frozen loop ``Core.simulationEnumerator`` runs, every minute ``t``::

    startDocument -> _advance_exogenous_hazard(t) -> pedDS.resolveHazardShelterClosures(t)
    -> pedestrian interactions -> cellUpdate -> docuStatus -> rl.step(t)

Counterfactual branches (planner, early search, this package's own rollouts)
replay the same steps through ``CounterfactualBranch.advance_one_timestep``,
which never calls ``_advance_exogenous_hazard`` or ``rl.step``. The runtime
uses that to tell the live episode from a branch:

* ``Core._advance_exogenous_hazard`` marks minute ``t`` as the pending live step;
* the next ``PedDS.resolveHazardShelterClosures(simulation_time=t)`` on the live
  core's pedestrian store is the live closure phase: queued evacuation orders
  execute there (before the overrun check of the same minute), then the
  unmodified overrun transaction runs;
* ``RLBridge.step(t)`` on the live bridge is the decision boundary: after the
  placement step, the controller decides the orders for minute ``t + 1``.

The post-episode hazard audit calls ``_advance_exogenous_hazard`` without a
closure phase, so nothing executes there. ``LIVE.in_live_closure`` is also what
the panel's scenario runtime uses to record live closures only.
"""

from __future__ import annotations

from typing import Optional

from .transaction import execute_evacuation_orders

_MARK = "_recourse_runtime"


class LiveEpisode:
    def __init__(self):
        self.reset(None)

    def reset(self, controller) -> None:
        self.controller = controller
        self.core = None
        self.running = False
        self.pending_time: Optional[int] = None
        self.in_live_closure = False
        self.queue: list = []
        self.queue_time: Optional[int] = None
        self.executed: list = []
        self.summary: Optional[dict] = None
        if controller is not None:
            controller.reset()

    # -- episode -------------------------------------------------------
    def begin(self, core) -> None:
        self.core = core
        self.running = True
        self.pending_time = None
        self.queue, self.queue_time = [], None

    def end(self, core) -> None:
        if core is not self.core:
            return
        self.running = False
        self.pending_time = None
        if self.queue:
            raise RuntimeError(f"recourse orders {self.queue} were decided but never executed")
        if self.controller is not None and self.summary is None:
            self.summary = self.controller.summary(core, list(self.executed))
        self.core = None  # between episodes nothing is live

    def result(self) -> Optional[dict]:
        if self.controller is None:
            return None
        if self.summary is None:
            raise RuntimeError("recourse controller was declared but the episode never ran")
        return self.summary

    # -- hooks ---------------------------------------------------------
    def is_live_closure(self, ped, simulation_time) -> bool:
        core = self.core
        return (
            core is not None and self.running and self.pending_time is not None
            and simulation_time is not None and int(simulation_time) == self.pending_time
            and ped is getattr(core, "pedDS", None)
        )

    def decide(self, bridge, simulation_time: int, is_terminal: bool) -> None:
        core = self.core
        if (self.controller is None or core is None or not self.running or is_terminal
                or bridge is not getattr(core, "rl", None)):
            return
        if self.queue:
            raise RuntimeError("recourse orders from the previous boundary were not executed")
        cells = self.controller.decide(core, int(simulation_time))
        if cells:
            self.queue, self.queue_time = list(cells), int(simulation_time) + 1

    def execute(self, simulation_time: int, resolve) -> list:
        if not self.queue:
            return []
        if self.queue_time != int(simulation_time):
            raise RuntimeError(f"recourse orders queued for minute {self.queue_time} reached minute {simulation_time}")
        cells, self.queue, self.queue_time = self.queue, [], None
        records = execute_evacuation_orders(self.core, cells, simulation_time=int(simulation_time), resolve=resolve)
        self.executed.extend(records)
        return records


LIVE = LiveEpisode()


def install(core_class, rl_class, ped_class) -> None:
    """Wrap the four call sites (once per process; idempotent)."""
    loop = core_class.simulationEnumerator
    if not getattr(loop, _MARK, False):
        def wrapped_loop(self, *args, **kwargs):
            LIVE.begin(self)
            try:
                return loop(self, *args, **kwargs)
            finally:
                LIVE.end(self)

        setattr(wrapped_loop, _MARK, True)
        core_class.simulationEnumerator = wrapped_loop

    hazard = core_class._advance_exogenous_hazard
    if not getattr(hazard, _MARK, False):
        def wrapped_hazard(self, sim_time, timer=None):
            result = hazard(self, sim_time, timer)
            if self is LIVE.core and LIVE.running:
                LIVE.pending_time = int(sim_time)
            return result

        setattr(wrapped_hazard, _MARK, True)
        core_class._advance_exogenous_hazard = wrapped_hazard

    resolve = ped_class.resolveHazardShelterClosures
    if not getattr(resolve, _MARK, False):
        def wrapped_resolve(self, *, simulation_time=None):
            if not LIVE.is_live_closure(self, simulation_time):
                return resolve(self, simulation_time=simulation_time)
            LIVE.pending_time = None
            LIVE.in_live_closure = True
            try:
                records = LIVE.execute(int(simulation_time), resolve)
                out = resolve(self, simulation_time=simulation_time)
                if records and isinstance(out, dict):
                    out = dict(out)
                    out["closed_shelters"] = int(out.get("closed_shelters", 0)) + len(records)
                    out["reactivated_population"] = int(out.get("reactivated_population", 0)) + sum(
                        r["occupants_evacuated"] for r in records)
                    out["released_reservations"] = int(out.get("released_reservations", 0)) + sum(
                        r["reservations_released"] for r in records)
                return out
            finally:
                LIVE.in_live_closure = False

        setattr(wrapped_resolve, _MARK, True)
        ped_class.resolveHazardShelterClosures = wrapped_resolve

    step = rl_class.step
    if not getattr(step, _MARK, False):
        def wrapped_step(self, *, simulation_time, is_terminal=False):
            out = step(self, simulation_time=simulation_time, is_terminal=is_terminal)
            LIVE.decide(self, int(simulation_time), bool(is_terminal))
            return out

        setattr(wrapped_step, _MARK, True)
        rl_class.step = wrapped_step
