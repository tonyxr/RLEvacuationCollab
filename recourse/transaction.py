"""Execute pre-emptive evacuation orders through the simulator's own closure transaction.

An evacuation order for a region closes every open shelter in it *now*, while
its danger is still below the overrun level:

* the shelter stops admitting people and its reservations are released;
* its occupants are reactivated at the shelter and rerouted to the nearest
  reachable operational shelter with free capacity;
* their safe completions are reversed and the time they spent inside is
  charged exactly as after an overrun (the occupants are not safe until they
  reach another shelter);
* candidate sites in the region are retired, as for an overrun region.

All of this is what ``PedestrianDatabase.resolveHazardShelterClosures``
already does when a shelter's region reaches the overrun level. The order is
executed by calling that unmodified transaction with a cell-tracker view that
reports the overrun level for the ordered regions only, so the frozen
simulator code, its invariants and its cost accounting are reused verbatim.
After the transaction the shelter's recorded failure danger is set back to
the true danger at the order, so an ordered closure (danger below the overrun
level) is distinguishable from an overrun in ``shelter_overrun_events``.

The function must be called in the closure phase of a step: after
``startDocument`` and the hazard transition, before pedestrians move. That is
where the live loop resolves overruns, so the order's costs land in the same
step counters an overrun's would.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

OVERRUN_STATUS = 2

# shelter id -> true danger, set while an order transaction runs, so that
# observers of the closure transaction (the panel's scenario runtime) can tell
# an ordered closure from an overrun.
ORDER_IN_PROGRESS: dict[int, int] = {}


def _cell(value) -> tuple[int, int]:
    i, j = value
    return int(i), int(j)


class OrderedDangerView:
    """Cell tracker view: the overrun level for ordered cells, the truth elsewhere."""

    def __init__(self, tracker, cells: Iterable[tuple[int, int]], level: int):
        object.__setattr__(self, "_tracker", tracker)
        object.__setattr__(self, "_cells", frozenset(_cell(c) for c in cells))
        object.__setattr__(self, "_level", int(level))

    def getCellState(self, cell):  # noqa: N802 - simulator API name
        state = int(self._tracker.getCellState(cell))
        if _cell(cell) in self._cells:
            return max(state, self._level)
        return state

    def __getattr__(self, name):
        return getattr(self._tracker, name)

    def __setattr__(self, name, value):
        setattr(self._tracker, name, value)


def overrun_level(core) -> int:
    return int(getattr(core.pedDS, "shelter_damage_danger_threshold",
                       getattr(core, "shelterOverrunDangerLevel", 4)))


def open_shelters_by_cell(core) -> dict[tuple[int, int], list]:
    cells: dict[tuple[int, int], list] = {}
    for shelter in core.shelterDS.shelterList.values():
        if int(getattr(shelter, "status", 0)) == OVERRUN_STATUS:
            continue
        cells.setdefault(_cell(shelter.cellLocated), []).append(shelter)
    return cells


def people_at(core, shelter) -> tuple[int, int]:
    """(occupants, people holding a reservation) of one shelter."""
    occupants = int(getattr(shelter, "shelterFlow", 0))
    reserved = int(core.shelterDS.reservedPopulation(shelter))
    return occupants, reserved


def execute_evacuation_orders(
    core,
    cells: Iterable[tuple[int, int]],
    *,
    simulation_time: int,
    resolve: Optional[Callable] = None,
) -> list[dict]:
    """Close every open shelter in ``cells`` now; return one record per closed shelter.

    ``resolve`` is the unwrapped ``PedestrianDatabase.resolveHazardShelterClosures``
    (the runtime passes it to avoid re-entering its own wrapper); by default the
    bound method of ``core.pedDS`` is used.
    """
    ordered = sorted({_cell(c) for c in cells})
    if not ordered:
        return []
    by_cell = open_shelters_by_cell(core)
    targets = [s for c in ordered for s in by_cell.get(c, [])]
    if not targets:
        return []
    tracker = core.pedDS.cellTracker
    level = overrun_level(core)
    before = {}
    for shelter in targets:
        occupants, reserved = people_at(core, shelter)
        before[int(shelter.shelterID)] = {
            "danger": int(tracker.getCellState(shelter.cellLocated)),
            "occupants": occupants,
            "reserved": reserved,
            "forced_before": int(getattr(shelter, "forcedReevacuatedPopulation", 0)),
        }
    view = OrderedDangerView(tracker, ordered, level)
    ped = core.pedDS
    ORDER_IN_PROGRESS.clear()
    ORDER_IN_PROGRESS.update({sid: info["danger"] for sid, info in before.items()})
    ped.cellTracker = view
    try:
        if resolve is None:
            ped.resolveHazardShelterClosures(simulation_time=int(simulation_time))
        else:
            resolve(ped, simulation_time=int(simulation_time))
    finally:
        ped.cellTracker = tracker
        ORDER_IN_PROGRESS.clear()
    records = []
    for shelter in targets:
        sid = int(shelter.shelterID)
        live = core.shelterDS.shelterList.get(sid, shelter)
        if int(getattr(live, "status", 0)) != OVERRUN_STATUS:
            raise RuntimeError(f"evacuation order did not close shelter {sid}")
        info = before[sid]
        if info["danger"] < level:
            live.overrunDangerLevel = info["danger"]
        records.append({
            "shelter_id": sid,
            "cell": list(_cell(live.cellLocated)),
            "minute": int(simulation_time),
            "danger": info["danger"],
            "occupants_evacuated": int(getattr(live, "forcedReevacuatedPopulation", 0)) - info["forced_before"],
            "occupants_before": info["occupants"],
            "reservations_released": info["reserved"],
            "ordered_before_overrun": info["danger"] < level,
        })
    return records
