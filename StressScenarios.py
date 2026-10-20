"""Pre-registered evaluation stress scenarios.

Each stress family isolates one mechanism in which a one-interval snapshot rule
can be wrong and a policy with foresight can be right, and exposes one "dose"
knob that turns the mechanism up:

* ``aimed_wind_front``   wind carries the front toward the population mass;
                         dose = wind speed.
* ``threatened_cluster`` a small cluster lies downwind of the fire while a
                         larger cluster is safe; dose = nominal front arrival
                         time at the small cluster (a deadline).
* ``congestion_cluster`` half the population starts in one neighbourhood;
                         dose = population size.
* ``panic_cliff``        a wind-driven front approaches the population;
                         dose = panic rate (threshold losses).
* ``wind_shift_hedge``   a real (non-zero) wind shifts with imperfect public
                         warnings; dose = shift rate.

Everything here is exogenous scenario construction. It runs once at episode
initialization from its own seeded stream, does not depend on any policy, and
is identical for every strategy evaluated on a tape. The module is pure
NumPy so that it can be tested without the simulator.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

STRESS_WIND_AIMS = ("none", "random", "population_centroid", "threatened_cluster")
STRESS_SEED_TAG = 0x53545253  # "STRS"
LABEL_ONLY_KEYS = frozenset({"stressScenarioFamily"})

# Simulator defaults for keys a curriculum may omit. Used only to decide
# whether an evaluation scenario lies inside the training support.
CORE_DEFAULTS_FOR_SUPPORT = {
    "hazardWindSpeedMPerMinute": 0.0,
    "hazardWindDirectionDegrees": 0.0,
    "stressClusterShares": [],
    "stressClusterRadiusM": 300.0,
    "stressClusterMinSeparationM": 1500.0,
    "stressThreatenedCluster": -1,
    "stressThreatArrivalMinutes": 0.0,
    "stressWindAim": "none",
    "constructedDecoyDemand": False,
}


# ---------------------------------------------------------------------------
# Geometry (hazard convention: degrees counter-clockwise from projected east;
# map y increases southward, so north = -dy)
# ---------------------------------------------------------------------------

def bearing_degrees(source_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    """Transport direction that carries a front from ``source`` to ``target``."""
    dx = float(target_xy[0]) - float(source_xy[0])
    north = float(source_xy[1]) - float(target_xy[1])
    if dx == 0.0 and north == 0.0:
        raise ValueError("bearing requires two distinct points")
    return math.degrees(math.atan2(north, dx)) % 360.0


def downwind_front_speed(
    spread_speed_m_per_minute: float,
    wind_speed_m_per_minute: float,
    wind_influence: float = 1.0,
) -> float:
    """Nominal downwind front speed, matching ``HazardDS.directional_spread_multiplier``."""
    spread = float(spread_speed_m_per_minute)
    if not math.isfinite(spread) or spread <= 0.0:
        raise ValueError("spread speed must be finite and positive")
    wind = float(wind_speed_m_per_minute)
    influence = float(wind_influence)
    if wind <= 0.0 or influence <= 0.0:
        return spread
    multiplier = float(np.clip(math.exp(influence * wind / spread), 0.25, 4.0))
    return spread * multiplier


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StressScenarioSpec:
    family: str = "none"
    cluster_shares: tuple = ()
    cluster_radius_m: float = 300.0
    cluster_min_separation_m: float = 1500.0
    threatened_cluster: int = -1
    threat_arrival_minutes: float = 0.0
    wind_aim: str = "none"

    def __post_init__(self):
        shares = tuple(float(value) for value in self.cluster_shares)
        object.__setattr__(self, "cluster_shares", shares)
        object.__setattr__(self, "wind_aim", str(self.wind_aim).strip().lower())
        object.__setattr__(self, "family", str(self.family))
        for share in shares:
            if not math.isfinite(share) or not 0.0 < share < 1.0:
                raise ValueError("stressClusterShares entries must lie in (0, 1)")
        if sum(shares) > 1.0 + 1e-12:
            raise ValueError("stressClusterShares must sum to at most 1")
        radius = float(self.cluster_radius_m)
        separation = float(self.cluster_min_separation_m)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("stressClusterRadiusM must be finite and positive")
        if not math.isfinite(separation) or separation < 0.0:
            raise ValueError("stressClusterMinSeparationM must be finite and non-negative")
        threatened = int(self.threatened_cluster)
        if threatened < -1 or threatened >= len(shares):
            raise ValueError("stressThreatenedCluster must index a configured cluster or be -1")
        arrival = float(self.threat_arrival_minutes)
        if threatened >= 0 and (not math.isfinite(arrival) or arrival <= 0.0):
            raise ValueError("stressThreatArrivalMinutes must be positive for a threatened cluster")
        if self.wind_aim not in STRESS_WIND_AIMS:
            raise ValueError(f"stressWindAim must be one of {STRESS_WIND_AIMS}")
        if self.wind_aim == "threatened_cluster" and threatened < 0:
            raise ValueError("stressWindAim=threatened_cluster requires stressThreatenedCluster")

    @property
    def active(self) -> bool:
        return bool(self.cluster_shares) or self.threatened_cluster >= 0 or self.wind_aim != "none"

    @property
    def requires_origin_plan(self) -> bool:
        return bool(self.cluster_shares) or self.wind_aim in {
            "population_centroid", "threatened_cluster",
        }

    @classmethod
    def from_core(cls, core) -> "StressScenarioSpec":
        return cls(
            family=getattr(core, "stressScenarioFamily", "none"),
            cluster_shares=tuple(getattr(core, "stressClusterShares", ()) or ()),
            cluster_radius_m=getattr(core, "stressClusterRadiusM", 300.0),
            cluster_min_separation_m=getattr(core, "stressClusterMinSeparationM", 1500.0),
            threatened_cluster=getattr(core, "stressThreatenedCluster", -1),
            threat_arrival_minutes=getattr(core, "stressThreatArrivalMinutes", 0.0),
            wind_aim=getattr(core, "stressWindAim", "none"),
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["cluster_shares"] = list(self.cluster_shares)
        return payload


def stress_rng(seed: int) -> np.random.Generator:
    """Dedicated stream so the stress construction never shifts another stream."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed) & 0xFFFFFFFFFFFFFFFF, STRESS_SEED_TAG])
    )


# ---------------------------------------------------------------------------
# Population origins
# ---------------------------------------------------------------------------

def _valid_xy(node_xy) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(node_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("node_xy must have shape (nodes, 2)")
    valid = np.flatnonzero(np.isfinite(xy).all(axis=1))
    if valid.size == 0:
        raise ValueError("no road node has finite coordinates")
    return xy, valid


def choose_cluster_anchors(node_xy, count: int, min_separation_m: float, rng) -> np.ndarray:
    xy, valid = _valid_xy(node_xy)
    anchors: list[int] = []
    for index in rng.permutation(valid):
        point = xy[int(index)]
        if all(
            float(np.hypot(*(point - xy[anchor]))) >= float(min_separation_m)
            for anchor in anchors
        ):
            anchors.append(int(index))
            if len(anchors) == int(count):
                break
    if len(anchors) < int(count):
        raise RuntimeError(
            f"cannot place {count} population clusters at least "
            f"{min_separation_m:.0f} m apart on this road network"
        )
    return np.asarray(anchors, dtype=np.int64)


def clustered_origin_plan(
    node_xy,
    total_population: int,
    maximum_group_size: int,
    spec: StressScenarioSpec,
    seed: int,
) -> dict:
    """Origin plan with ``spec.cluster_shares`` of the groups in compact clusters.

    Returns the same keys as ``CellPartitioning.sample_population_origins``
    (``origin_indices``, ``group_sizes``, ``node_population``) plus the cluster
    anchors, so it can replace that plan before the population-adaptive cut.
    """
    total_population = int(total_population)
    maximum_group_size = int(maximum_group_size)
    if total_population <= 0 or maximum_group_size <= 0:
        raise ValueError("population and group size must be positive")
    xy, valid = _valid_xy(node_xy)
    rng = stress_rng(seed)
    group_sizes = []
    remaining = total_population
    while remaining > 0:
        size = min(maximum_group_size, remaining)
        group_sizes.append(int(size))
        remaining -= size
    groups = len(group_sizes)
    anchors = choose_cluster_anchors(
        xy, len(spec.cluster_shares), spec.cluster_min_separation_m, rng
    ) if spec.cluster_shares else np.zeros(0, dtype=np.int64)

    assignments: list[int] = []
    cluster_counts = []
    for share, anchor in zip(spec.cluster_shares, anchors):
        count = max(1, int(math.floor(share * groups)))
        distance = np.hypot(*(xy[valid] - xy[int(anchor)]).T)
        members = valid[distance <= float(spec.cluster_radius_m)]
        if members.size == 0:
            members = np.asarray([int(anchor)], dtype=np.int64)
        assignments.extend(int(value) for value in rng.choice(members, size=count))
        cluster_counts.append(count)
    if len(assignments) > groups:
        raise RuntimeError("cluster shares exceed the represented population")
    uniform_count = groups - len(assignments)
    assignments.extend(int(value) for value in rng.choice(valid, size=uniform_count))
    order = rng.permutation(groups)
    origin_indices = [assignments[int(index)] for index in order]

    node_population = np.zeros(xy.shape[0], dtype=np.float64)
    np.add.at(node_population, origin_indices, np.asarray(group_sizes, dtype=np.float64))
    if int(round(float(node_population.sum()))) != total_population:
        raise RuntimeError("clustered origin plan does not conserve represented people")
    return {
        "origin_indices": origin_indices,
        "group_sizes": group_sizes,
        "node_population": node_population,
        "anchor_indices": [int(value) for value in anchors],
        "cluster_group_counts": cluster_counts,
        "uniform_group_count": int(uniform_count),
    }


def population_centroid(node_xy, node_population) -> np.ndarray:
    xy = np.asarray(node_xy, dtype=np.float64)
    weights = np.asarray(node_population, dtype=np.float64)
    usable = np.isfinite(xy).all(axis=1) & (weights > 0.0)
    if not usable.any():
        raise ValueError("population centroid requires positive population")
    return np.average(xy[usable], axis=0, weights=weights[usable])


# ---------------------------------------------------------------------------
# Hazard source for the threatened-cluster family
# ---------------------------------------------------------------------------

def threat_source_index(
    node_xy,
    anchor_index: int,
    distance_m: float,
    seed: int,
    *,
    other_anchor_indices: Sequence[int] = (),
    band_fraction: float = 0.1,
) -> int:
    """Road node at ``distance_m`` from the threatened anchor.

    Preference order: inside the distance band and farther from every other
    cluster than from the threatened one; inside the band; closest distance.
    """
    xy, valid = _valid_xy(node_xy)
    distance_m = float(distance_m)
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("threat distance must be finite and positive")
    rng = stress_rng(seed ^ 0x5448524541)
    anchor_xy = xy[int(anchor_index)]
    distance = np.hypot(*(xy[valid] - anchor_xy).T)
    in_band = np.abs(distance - distance_m) <= float(band_fraction) * distance_m
    candidates = valid[in_band]
    if candidates.size and len(other_anchor_indices):
        far_from_others = np.ones(candidates.size, dtype=bool)
        own = np.hypot(*(xy[candidates] - anchor_xy).T)
        for other in other_anchor_indices:
            other_distance = np.hypot(*(xy[candidates] - xy[int(other)]).T)
            far_from_others &= other_distance > own
        if far_from_others.any():
            candidates = candidates[far_from_others]
    if candidates.size:
        return int(rng.choice(candidates))
    return int(valid[int(np.argmin(np.abs(distance - distance_m)))])


# ---------------------------------------------------------------------------
# Evaluation specification: dose grids, training support, dose response
# ---------------------------------------------------------------------------

def _support_value(value):
    if isinstance(value, list):
        return tuple(_support_value(item) for item in value)
    if isinstance(value, float):
        return round(value, 9)
    return value


def training_support(curriculum: Mapping[str, Any], stage_ids: Optional[Sequence[str]] = None) -> dict:
    """Values every configuration key takes in the policy-training stages."""
    support: dict[str, set] = {}
    for key, value in dict(curriculum.get("learner_overrides", {})).items():
        support.setdefault(key, set()).add(_support_value(value))
    stages = list(curriculum.get("stages", ()))
    selected = [
        stage for stage in stages
        if stage_ids is None or stage.get("stage_id") in set(stage_ids)
    ]
    for stage in selected:
        for variant in stage.get("variants", ()):
            for key, value in dict(variant.get("overrides", {})).items():
                support.setdefault(key, set()).add(_support_value(value))
    for key, value in CORE_DEFAULTS_FOR_SUPPORT.items():
        support.setdefault(key, set()).add(_support_value(value))
    return support


def out_of_support_keys(overrides: Mapping[str, Any], support: Mapping[str, set]) -> list:
    """Keys whose evaluation value never occurs in training (label keys excluded).

    A key the curriculum never sets and that has no recorded simulator default
    is reported as outside the support: absence of evidence is not coverage.
    """
    outside = []
    for key, value in overrides.items():
        if key in LABEL_ONLY_KEYS:
            continue
        if key not in support or _support_value(value) not in support[key]:
            outside.append(key)
    return sorted(outside)


def expand_stress_families(spec: Mapping[str, Any], base: Mapping[str, Any]) -> list:
    """Expand ``spec["stress_families"]`` into concrete evaluation scenarios.

    Every dose of one family uses the same ``seed_index``, so held-out tapes
    (hazard, regime and origin streams) are common across doses and the
    dose-response contrast is paired. ``tuning_key`` lets a family tune the
    deadline threshold once, at its pre-registered ``tuning_dose``.
    """
    entries = []
    families = dict(spec.get("stress_families", {}))
    for family_index, (family, definition) in enumerate(families.items()):
        dose_key = str(definition["dose_key"])
        doses = list(definition["doses"])
        if not doses:
            raise ValueError(f"stress family {family!r} has no doses")
        tuning_dose = definition.get("tuning_dose", doses[len(doses) // 2])
        if tuning_dose not in doses:
            raise ValueError(f"tuning_dose of {family!r} must be one of its doses")
        for dose in doses:
            overrides = copy.deepcopy(dict(definition.get("overrides", {})))
            overrides["stressScenarioFamily"] = family
            overrides[dose_key] = dose
            if definition.get("capacity_token_from_population"):
                merged = {**base, **overrides}
                shelters = int(merged["initShelterVol"]) + int(merged["maxAdditionalShelters"])
                total = int(merged["pedVol"]) + int(merged.get("shelterCapacitySurplus", 200))
                if total % shelters:
                    raise ValueError(
                        f"{family}: population {merged['pedVol']} + surplus does not "
                        f"divide evenly over {shelters} shelters"
                    )
                overrides["shelterCapacityToken"] = total // shelters
            entries.append({
                "name": f"{family}@{dose}",
                "family": family,
                "dose_key": dose_key,
                "dose": float(dose),
                "seed_index": 5000 + family_index,
                "tuning_key": family,
                "is_tuning_dose": dose == tuning_dose,
                "is_endpoint_dose": dose in (min(doses), max(doses)),
                "overrides": overrides,
                "mechanism": str(definition.get("mechanism", "")),
            })
    return entries


def paired_dose_slope(gap_by_tape_and_dose: Mapping[int, Mapping[float, float]]) -> tuple:
    """Mean and SE across tapes of the per-tape OLS slope of a gap on dose.

    Tapes are common across doses, so each tape contributes one slope and
    scenario difficulty cancels within the tape.
    """
    slopes = []
    for by_dose in gap_by_tape_and_dose.values():
        doses = np.asarray(sorted(by_dose), dtype=np.float64)
        if doses.size < 2 or np.ptp(doses) == 0.0:
            continue
        gaps = np.asarray([by_dose[dose] for dose in sorted(by_dose)], dtype=np.float64)
        centered = doses - doses.mean()
        slopes.append(float(np.dot(centered, gaps - gaps.mean()) / np.dot(centered, centered)))
    if not slopes:
        return math.nan, math.nan, 0
    values = np.asarray(slopes, dtype=np.float64)
    se = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else math.nan
    return float(values.mean()), se, int(values.size)
