#!/usr/bin/env python3
"""Validated, immutable city specifications for cross-city experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from typing import Iterable, Mapping

from CellPartitioning import normalize_partition_mode

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CITY_PROFILE_PATH = os.path.join(PROJECT_ROOT, "config", "city_profiles.json")
_CITY_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


@dataclass(frozen=True)
class CityProfile:
    """One reproducible OSM study area, ordered by an external scale measure."""

    city_id: str
    display_name: str
    address: str
    scale_rank: int
    census_2020_population: int
    center_lat: float
    center_lon: float
    radius_m: float
    census_source: str

    @classmethod
    def from_mapping(cls, value: Mapping) -> "CityProfile":
        required = set(cls.__dataclass_fields__)
        missing = sorted(required.difference(value))
        extra = sorted(set(value).difference(required))
        if missing or extra:
            raise ValueError(
                f"City profile fields do not match schema; missing={missing}, extra={extra}"
            )
        profile = cls(
            city_id=str(value["city_id"]).strip(),
            display_name=str(value["display_name"]).strip(),
            address=str(value["address"]).strip(),
            scale_rank=int(value["scale_rank"]),
            census_2020_population=int(value["census_2020_population"]),
            center_lat=float(value["center_lat"]),
            center_lon=float(value["center_lon"]),
            radius_m=float(value["radius_m"]),
            census_source=str(value["census_source"]).strip(),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not _CITY_ID.fullmatch(self.city_id):
            raise ValueError(f"Invalid city_id {self.city_id!r}")
        if not self.display_name or not self.address:
            raise ValueError(f"City {self.city_id!r} requires display_name and address")
        if self.scale_rank <= 0 or self.census_2020_population <= 0:
            raise ValueError(f"City {self.city_id!r} requires positive scale metadata")
        if not math.isfinite(self.center_lat) or not -90.0 <= self.center_lat <= 90.0:
            raise ValueError(f"Invalid latitude for {self.city_id!r}")
        if not math.isfinite(self.center_lon) or not -180.0 <= self.center_lon <= 180.0:
            raise ValueError(f"Invalid longitude for {self.city_id!r}")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError(f"Invalid study radius for {self.city_id!r}")
        if not self.census_source.startswith("https://www.census.gov/"):
            raise ValueError(f"City {self.city_id!r} requires an authoritative Census URL")

    def map_spec(self) -> dict:
        return {
            "city_id": self.city_id,
            "address": self.address,
            "query_mode": "point",
            "center": [self.center_lat, self.center_lon],
            "radius_m": self.radius_m,
            "network_type": "walk",
        }

    def core_overrides(self) -> dict:
        return {
            "cityID": self.city_id,
            "address": self.address,
            "mapQueryMode": "point",
            "mapCenterLat": self.center_lat,
            "mapCenterLon": self.center_lon,
            "mapRadiusM": self.radius_m,
        }

    def as_dict(self) -> dict:
        return {
            "city_id": self.city_id,
            "display_name": self.display_name,
            "address": self.address,
            "scale_rank": self.scale_rank,
            "census_2020_population": self.census_2020_population,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "radius_m": self.radius_m,
            "census_source": self.census_source,
        }


@dataclass(frozen=True)
class CitySuite:
    schema_version: int
    selection_basis: str
    common_experiment: dict
    cities: tuple[CityProfile, ...]
    source_path: str
    source_sha256: str

    def select(self, city_ids: Iterable[str] | None = None) -> tuple[CityProfile, ...]:
        if city_ids is None:
            return self.cities
        requested = tuple(str(value).strip().lower() for value in city_ids if str(value).strip())
        by_id = {city.city_id: city for city in self.cities}
        unknown = sorted(set(requested).difference(by_id))
        if unknown:
            raise KeyError(f"Unknown city profile(s): {unknown}")
        if len(requested) != len(set(requested)):
            raise ValueError("City selection contains duplicates")
        return tuple(by_id[city_id] for city_id in requested)


def _validate_common_experiment(value: Mapping) -> dict:
    integer_fields = {
        "cellX", "cellY", "stopTime", "pedVol", "hazardVol",
        "shelterCanVol", "initShelterVol", "maxAdditionalShelters",
        "shelterActionInterval", "shelterCapacityToken",
        "panicDangerThreshold",
        "pedestrianGroupSize",
    }
    float_fields = {
        "timeStepMinutes", "congestionEffectiveWidthM",
        "congestionJamDensityPedPerM2", "congestionShape",
        "congestionMinimumSpeedRatio", "congestionSubstepSeconds",
        "cellPartitionMinWidthFraction",
        "socialForceSelfCoefficient", "socialForceImpactCoefficient",
        "intersectionConsolidationToleranceM", "panicRate",
        "panicHerdProbability", "hazardCasualtyReferenceMinutes",
        "maximumShelterForecastDanger",
    }
    boolean_fields = {
        "congestionEnabled",
        "socialForceEnabled",
        "intersectionConsolidationEnabled",
    }
    string_fields = {"cellPartitionMode"}
    allowed = integer_fields | float_fields | boolean_fields | string_fields
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"Unsupported common experiment fields: {unknown}")
    result = {}
    for key, raw in value.items():
        if key in integer_fields:
            result[str(key)] = int(raw)
        elif key in float_fields:
            result[str(key)] = float(raw)
        elif key in string_fields:
            result[str(key)] = normalize_partition_mode(raw)
        else:
            if not isinstance(raw, bool):
                raise ValueError(f"{key} must be boolean")
            result[str(key)] = bool(raw)
    positive = (
        "cellX", "cellY", "stopTime", "pedVol", "shelterCanVol",
        "shelterActionInterval",
    )
    if any(result.get(key, 0) <= 0 for key in positive):
        raise ValueError(f"Common experiment fields {positive} must be positive")
    if result.get("hazardVol", 0) < 0:
        raise ValueError("hazardVol must be non-negative")
    if result.get("maxAdditionalShelters", 0) < 0:
        raise ValueError("maxAdditionalShelters must be non-negative")
    if result.get("shelterCapacityToken", 0) < 0:
        raise ValueError("shelterCapacityToken must be non-negative")
    safety_threshold = result.get("maximumShelterForecastDanger", 0.6)
    if not 0.0 <= safety_threshold <= 1.0:
        raise ValueError("maximumShelterForecastDanger must lie in [0, 1]")
    if not 0 <= result.get("initShelterVol", 0) <= result.get("shelterCanVol", 0):
        raise ValueError("initShelterVol must lie within the candidate budget")
    for key in (
        "timeStepMinutes", "congestionEffectiveWidthM",
        "congestionJamDensityPedPerM2", "congestionShape",
        "congestionSubstepSeconds", "cellPartitionMinWidthFraction",
        "intersectionConsolidationToleranceM", "hazardCasualtyReferenceMinutes",
    ):
        if key in result and (not math.isfinite(result[key]) or result[key] <= 0.0):
            raise ValueError(f"{key} must be finite and positive")
    minimum_ratio = result.get("congestionMinimumSpeedRatio")
    if minimum_ratio is not None and (
        not math.isfinite(minimum_ratio) or not 0.0 <= minimum_ratio < 1.0
    ):
        raise ValueError(
            "congestionMinimumSpeedRatio must be finite and in [0, 1)"
        )
    for key in (
        "socialForceSelfCoefficient",
        "socialForceImpactCoefficient",
        "panicRate",
        "panicHerdProbability",
    ):
        value_now = result.get(key)
        if value_now is not None and (
            not math.isfinite(value_now) or not 0.0 <= value_now <= 1.0
        ):
            raise ValueError(f"{key} must be finite and in [0, 1]")
    threshold = result.get("panicDangerThreshold")
    if threshold is not None and not 0 <= threshold <= 5:
        raise ValueError("panicDangerThreshold must lie in [0, 5]")
    width_fraction = result.get("cellPartitionMinWidthFraction")
    if width_fraction is not None and width_fraction * max(
        result.get("cellX", 1), result.get("cellY", 1)
    ) >= 1.0:
        raise ValueError(
            "cellPartitionMinWidthFraction must be smaller than "
            "1/max(cellX, cellY)"
        )
    return result


def load_city_suite(path: str = DEFAULT_CITY_PROFILE_PATH) -> CitySuite:
    resolved = os.path.abspath(path)
    with open(resolved, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    required = {"schema_version", "selection_basis", "common_experiment", "cities"}
    if set(payload) != required:
        raise ValueError(
            f"City suite top-level fields do not match schema: {sorted(payload)}"
        )
    if int(payload["schema_version"]) != 2:
        raise ValueError("Unsupported city profile schema_version")
    selection_basis = str(payload["selection_basis"]).strip()
    if not selection_basis:
        raise ValueError("selection_basis must not be empty")
    common = _validate_common_experiment(payload["common_experiment"])
    cities = tuple(CityProfile.from_mapping(item) for item in payload["cities"])
    if len(cities) < 2:
        raise ValueError("A cross-city suite requires at least two cities")
    ids = [city.city_id for city in cities]
    ranks = [city.scale_rank for city in cities]
    populations = [city.census_2020_population for city in cities]
    if len(ids) != len(set(ids)) or len(ranks) != len(set(ranks)):
        raise ValueError("City identifiers and scale ranks must be unique")
    if ranks != list(range(1, len(cities) + 1)):
        raise ValueError("Cities must be ordered by contiguous scale_rank starting at one")
    if populations != sorted(populations) or len(set(populations)) != len(populations):
        raise ValueError("Cities must have strictly increasing Census population")
    return CitySuite(
        schema_version=2,
        selection_basis=selection_basis,
        common_experiment=common,
        cities=cities,
        source_path=resolved,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
