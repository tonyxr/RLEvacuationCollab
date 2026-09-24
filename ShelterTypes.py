"""Canonical OSM tags eligible for the shared shelter-site optimizer."""

from __future__ import annotations

from collections.abc import Iterable

# Keep this list intentionally narrow: these uses plausibly provide a public,
# institutional, or assembly-space shelter. Querying every building in a large
# city is unnecessary and makes OSM acquisition fragile.
SHELTER_BUILDING_TYPES = frozenset(
    {
        "school",
        "college",
        "university",
        "hospital",
        "public",
        "civic",
        "community_centre",
        "stadium",
        "fire_station",
        "police",
        "library",
        "place_of_worship",
        "church",
        "cathedral",
        "mosque",
        "temple",
        "synagogue",
        "shelter",
    }
)

SHELTER_AMENITY_TYPES = frozenset(
    {
        "hospital",
        "school",
        "college",
        "university",
        "community_centre",
        "place_of_worship",
        "library",
        "fire_station",
        "police",
        "social_facility",
        "shelter",
    }
)

SHELTER_OSM_TAGS = {
    "building": sorted(SHELTER_BUILDING_TYPES),
    "amenity": sorted(SHELTER_AMENITY_TYPES),
}

SHELTER_CANDIDATE_TYPES = SHELTER_BUILDING_TYPES.union(SHELTER_AMENITY_TYPES)


def _tag_values(value) -> tuple[str, ...]:
    """Return normalized OSM tag values without treating missing data as tags."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip().lower() for part in value.split(";") if part.strip())
    if isinstance(value, Iterable):
        output = []
        for item in value:
            output.extend(_tag_values(item))
        return tuple(output)
    text = str(value).strip().lower()
    return (text,) if text else ()


def canonical_shelter_site_type(building_type, amenity_type) -> str | None:
    """Choose the functional shelter type used by eligibility and capacity.

    OSM commonly stores ``building=yes`` and places the actual use in
    ``amenity``.  Amenity therefore has precedence when it is eligible, with
    the building tag as the fallback.  Multi-valued semicolon/list tags are
    handled deterministically in their recorded order.
    """
    for value in _tag_values(amenity_type):
        if value in SHELTER_AMENITY_TYPES:
            return value
    for value in _tag_values(building_type):
        if value in SHELTER_BUILDING_TYPES:
            return value
    return None
