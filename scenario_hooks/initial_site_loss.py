"""Corner case: a spot fire in the region of one or two initial shelters.

Why a dedicated construction is needed
--------------------------------------
Danger is source-local in the frozen simulator
(``HazardDS._severity_cap``):

- a hazard's source region may reach danger level 5;
- regions within ``hazardSevereRadiusM`` (100 m) of the source point may
  reach level 4;
- every other region is capped at level 3.

A shelter fails at level 4 (``shelterOverrunDangerLevel``), so a spreading
front can never overrun one. Overrun requires an ignition inside, or within
100 m of, a shelter's region. Primary sources are three random road nodes
that burn for 60 minutes; secondary ignitions occur at about 0.004 per
minute in random regions. Overrun is therefore rare by construction: 0 to
0.05 overrun shelters per held-out episode in the G2 panel.

Construction
------------
At ``ignition_minute`` the region of each targeted initial shelter ignites
as a secondary ignition (event kind ``"ignition"``, handled by
``HazardDS.add_hazard_at``):

- adding the ignition draws no random number. Wind, intensity, closures,
  other ignitions and evacuee origins stay identical to the unstressed tape.
  The spot fire's own stochastic spread draws from the shared sequential
  hazard stream, so the primary fires' spread after the spot fire starts
  spreading is a new realization of the same process. Stressed and base
  episodes are therefore paired on everything but that, and comparisons
  between policies within one cell are exact common-random-number
  comparisons (the hazard does not depend on actions);
- the ignited region is the hazard's source region (cap 5), and the new
  hazard stays active to the horizon, so the region's danger follows
  elapsed burn time alone: level 4 is reached
  ``hazardSeverityTransitionMinutes[2]`` (30) minutes after ignition;
- the targeted shelter is therefore overrun at ``ignition_minute + 29``
  (the ignition minute is the region's first burning minute) in every tape
  and under every policy, and its occupants and reservation holders are
  released to re-evacuate (the existing closure transaction).

The ignition is latent like any secondary ignition: no policy sees it in
advance. Once burning, the region's danger, the belief overrun forecasts and
the forecast displacement are public, so an adaptive policy has the decision
epochs between ignition and failure to place capacity for the evacuees who
will be released. A precommitted plan cannot respond.

``targets=1`` ignites the first initial shelter's region; total capacity
falls from 3,000 to 2,700, still above the 2,500 people (the N-1 design).
``targets=2`` ignites the regions of both initial shelters; capacity falls
to 2,400, so at least 100 people cannot be sheltered by any policy. If both
initial shelters share one region, ``targets=1`` already overruns both; the
record says so (``shelters_in_ignited_cells``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCENARIO_ID = "initial_site_loss"


@dataclass(frozen=True)
class InitialSiteLoss:
    targets: int = 1
    ignition_minute: int = 5

    def __post_init__(self):
        if int(self.targets) not in (1, 2):
            raise ValueError("InitialSiteLoss.targets must be 1 or 2")
        if int(self.ignition_minute) < 1:
            raise ValueError(
                "InitialSiteLoss.ignition_minute must be >= 1 (events at t=0 "
                "are never applied: the transition loop starts at t=1)"
            )
        object.__setattr__(self, "targets", int(self.targets))
        object.__setattr__(self, "ignition_minute", int(self.ignition_minute))

    def params(self) -> dict:
        return asdict(self)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def initial_shelter_ids(core) -> list[int]:
        """Initial shelters in creation order (excludes static predeployments)."""
        baseline = getattr(core, "baseline_initial_shelter_ids", None)
        ids = baseline if baseline else core.shelterDS.shelterList.keys()
        return sorted(int(value) for value in ids)

    @staticmethod
    def overrun_lead_minutes(core) -> float:
        """Minutes from a region's ignition to the shelter-failure danger level."""
        threshold = int(getattr(core, "shelterOverrunDangerLevel", 4))
        transitions = tuple(float(v) for v in core.hazardDS.severity_transition_minutes)
        if not 2 <= threshold <= len(transitions) + 1:
            raise ValueError(f"shelterOverrunDangerLevel {threshold} has no severity transition")
        return transitions[threshold - 2]

    # -- hook --------------------------------------------------------------------
    def apply(self, core) -> dict:
        """Schedule the ignitions; call after initialization, before t = 1."""
        regime = getattr(core, "hazard_regime", None)
        if regime is None or not bool(getattr(regime, "enabled", False)):
            raise RuntimeError("InitialSiteLoss requires the hazard regime process")
        if int(getattr(regime, "current_time", 0)) >= self.ignition_minute:
            raise RuntimeError("InitialSiteLoss must be applied before its ignition minute")
        initial = self.initial_shelter_ids(core)
        if len(initial) < self.targets:
            raise RuntimeError(
                f"InitialSiteLoss needs {self.targets} initial shelters; the episode has {len(initial)}"
            )
        targeted = initial[: self.targets]
        shelters = core.shelterDS.shelterList
        cells: list[tuple[int, ...]] = []
        for shelter_id in targeted:
            cell = tuple(int(v) for v in shelters[shelter_id].cellLocated)
            if cell not in cells:
                cells.append(cell)
        lead = self.overrun_lead_minutes(core)
        # The ignition minute is the region's first burning minute, so the
        # failure level is reached ``lead - 1`` minutes after it (1-minute steps).
        step = float(getattr(core, "timeStepMinutes", 1.0) or 1.0)
        expected = self.ignition_minute + lead - step
        horizon = int(core.stopTime) - 1
        if expected > horizon:
            raise RuntimeError(
                f"ignition at minute {self.ignition_minute} overruns at {expected:g}, "
                f"after the {horizon}-minute horizon"
            )
        indices = [int(core.cellTracker.flatten_cell(cell)) for cell in cells]
        for index in indices:
            regime.scheduled.append({
                "time": int(self.ignition_minute),
                "kind": "ignition",
                "payload": {"cell_index": index},
            })
        regime.scheduled.sort(key=lambda event: (int(event["time"]), str(event["kind"])))
        regime._update_public(int(getattr(regime, "current_time", 0)))
        in_cells = sorted(
            int(sid) for sid, shelter in shelters.items()
            if tuple(int(v) for v in shelter.cellLocated) in cells
        )
        return {
            "scenario": SCENARIO_ID,
            **self.params(),
            "initial_shelter_ids": initial,
            "targeted_shelter_ids": targeted,
            "ignited_cells": [list(cell) for cell in cells],
            "ignited_cell_indices": indices,
            "shelters_in_ignited_cells": in_cells,
            "overrun_lead_minutes": lead,
            "expected_overrun_minute": expected,
        }

    def audit(self, core, applied: dict, closures: list[dict]) -> dict:
        """After the episode: did every targeted shelter fail when designed?"""
        shelters = core.shelterDS.shelterList
        outcome = {}
        for shelter_id in applied["shelters_in_ignited_cells"]:
            shelter = shelters.get(int(shelter_id))
            time = None if shelter is None else getattr(shelter, "overrunTime", None)
            outcome[str(shelter_id)] = None if time is None else int(time)
        targeted = [outcome.get(str(sid)) for sid in applied["targeted_shelter_ids"]]
        released = sum(
            int(c["released_population"]) for c in closures
            if int(c["shelter_id"]) in set(applied["shelters_in_ignited_cells"])
        )
        other = [c for c in closures if int(c["shelter_id"]) not in set(applied["shelters_in_ignited_cells"])]
        return {
            "overrun_minute_by_shelter": outcome,
            "all_targets_overrun": all(t is not None for t in targeted),
            "targets_overrun_as_designed": all(
                t is not None and abs(t - float(applied["expected_overrun_minute"])) <= 1.0
                for t in targeted
            ),
            "first_target_overrun_minute": min((t for t in targeted if t is not None), default=None),
            "released_population_at_targets": int(released),
            "other_overruns": len(other),
            "other_released_population": int(sum(int(c["released_population"]) for c in other)),
        }
