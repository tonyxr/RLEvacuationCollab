#!/usr/bin/env python3
"""Render OpenStreetMap context beside the experimental road skeleton.

The script uses the city definitions in ``config/city_profiles.json`` and the
presentation extents in ``config/city_network_figure_profiles.json``.  Normal
city panels use the same cached walking-network GraphML files consumed by the
simulator and repeat the configured intersection-consolidation step.  The
Greater Malibu/Santa Monica Bay panel uses a regional major-road OSM skeleton
so northern and western Los Angeles remain legible at paper scale.

Standard OpenStreetMap raster tiles are downloaded only when absent from the
local tile cache.  Every exported figure includes the required attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import osmnx as ox
from PIL import Image, ImageOps
import requests

from OSMProcessor import OSMProcessor


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = PROJECT_ROOT / "config" / "city_profiles.json"
DEFAULT_FIGURE_PROFILE = PROJECT_ROOT / "config" / "city_network_figure_profiles.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "publication_graphs" / "city_network_pairs"
DEFAULT_TILE_CACHE = PROJECT_ROOT / "cache" / "osm_tiles"
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_ATTRIBUTION = "© OpenStreetMap contributors · Open Database License (ODbL)"
WEB_MERCATOR_LIMIT = 20037508.342789244
TILE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--figure-profiles",
        type=Path,
        default=DEFAULT_FIGURE_PROFILE,
        help="Optional presentation-only city extent overrides.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-cache", type=Path, default=DEFAULT_TILE_CACHE)
    parser.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional city IDs; the default renders every configured city.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--tile-zoom",
        type=int,
        default=None,
        help="OpenStreetMap tile zoom; default selects a clear scale automatically.",
    )
    parser.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="Plot the raw cached OSM graph instead of experimental topology.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require every OpenStreetMap tile to be present in the local cache.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = x / WEB_MERCATOR_LIMIT * 180.0
    lat = math.degrees(2.0 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2.0)
    return lon, lat


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    lat = min(85.05112878, max(-85.05112878, lat))
    x = WEB_MERCATOR_LIMIT * lon / 180.0
    y = 6378137.0 * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def point_radius_bounds(center_lat: float, center_lon: float, radius_m: float):
    """Return an approximate point-centered bbox in Web Mercator coordinates."""
    lat_delta = radius_m / 111_320.0
    lon_delta = radius_m / (111_320.0 * math.cos(math.radians(center_lat)))
    west, south = lonlat_to_mercator(center_lon - lon_delta, center_lat - lat_delta)
    east, north = lonlat_to_mercator(center_lon + lon_delta, center_lat + lat_delta)
    return west, south, east, north


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = min(85.05112878, max(-85.05112878, lat))
    scale = 2**zoom
    x = int(math.floor((lon + 180.0) / 360.0 * scale))
    lat_rad = math.radians(lat)
    y = int(
        math.floor(
            (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
        )
    )
    return max(0, min(scale - 1, x)), max(0, min(scale - 1, y))


def tile_mercator_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    scale = 2**zoom
    width = 2.0 * WEB_MERCATOR_LIMIT / scale
    x_min = -WEB_MERCATOR_LIMIT + x * width
    x_max = x_min + width
    y_max = WEB_MERCATOR_LIMIT - y * width
    y_min = y_max - width
    return x_min, y_min, x_max, y_max


def graph_bounds(graph) -> tuple[float, float, float, float]:
    xs = np.fromiter((float(data["x"]) for _, data in graph.nodes(data=True)), float)
    ys = np.fromiter((float(data["y"]) for _, data in graph.nodes(data=True)), float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    # A small shared margin prevents the outermost roads from touching the frame.
    x_pad = max((x_max - x_min) * 0.025, 50.0)
    y_pad = max((y_max - y_min) * 0.025, 50.0)
    return x_min - x_pad, y_min - y_pad, x_max + x_pad, y_max + y_pad


def automatic_tile_zoom(bounds: tuple[float, float, float, float]) -> int:
    """Choose roughly six to eight source tiles across the longer dimension."""
    width = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    if width <= 0.0:
        return 13
    zoom = round(math.log2(7.0 * (2.0 * WEB_MERCATOR_LIMIT) / width))
    return max(0, min(17, int(zoom)))


def fetch_tile(
    session: requests.Session,
    cache_root: Path,
    zoom: int,
    x: int,
    y: int,
    *,
    offline: bool,
) -> Image.Image:
    path = cache_root / str(zoom) / str(x) / f"{y}.png"
    if path.exists():
        with Image.open(path) as image:
            return image.convert("RGB")
    if offline:
        raise FileNotFoundError(f"OpenStreetMap tile is not cached: {path}")

    url = OSM_TILE_URL.format(z=zoom, x=x, y=y)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    if image.size != (TILE_SIZE, TILE_SIZE):
        raise RuntimeError(f"Unexpected tile size {image.size} from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    # Keep a deliberately gentle request cadence for the public OSM tile service.
    time.sleep(0.10)
    return image


def basemap_for_bounds(
    bounds: tuple[float, float, float, float],
    zoom: int,
    session: requests.Session,
    cache_root: Path,
    *,
    offline: bool,
) -> tuple[Image.Image, tuple[float, float, float, float], int]:
    x_min, y_min, x_max, y_max = bounds
    lon_min, lat_min = mercator_to_lonlat(x_min, y_min)
    lon_max, lat_max = mercator_to_lonlat(x_max, y_max)
    tile_x_min, tile_y_max = lonlat_to_tile(lon_min, lat_min, zoom)
    tile_x_max, tile_y_min = lonlat_to_tile(lon_max, lat_max, zoom)
    tile_x_min, tile_x_max = sorted((tile_x_min, tile_x_max))
    tile_y_min, tile_y_max = sorted((tile_y_min, tile_y_max))

    cols = tile_x_max - tile_x_min + 1
    rows = tile_y_max - tile_y_min + 1
    canvas = Image.new("RGB", (cols * TILE_SIZE, rows * TILE_SIZE))
    requests_made = 0
    for row, tile_y in enumerate(range(tile_y_min, tile_y_max + 1)):
        for col, tile_x in enumerate(range(tile_x_min, tile_x_max + 1)):
            tile_path = cache_root / str(zoom) / str(tile_x) / f"{tile_y}.png"
            was_cached = tile_path.exists()
            tile = fetch_tile(
                session,
                cache_root,
                zoom,
                tile_x,
                tile_y,
                offline=offline,
            )
            canvas.paste(tile, (col * TILE_SIZE, row * TILE_SIZE))
            requests_made += int(not was_cached)

    # XYZ tile rows increase southward.  The mosaic's north edge therefore
    # comes from the first row and its south edge from the final row.
    west, _, _, north = tile_mercator_bounds(tile_x_min, tile_y_min, zoom)
    _, south, east, _ = tile_mercator_bounds(tile_x_max, tile_y_max, zoom)
    return canvas, (west, south, east, north), requests_made


def edge_segments(graph) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for u, v, data in graph.edges(data=True):
        geometry = data.get("geometry")
        if geometry is not None and hasattr(geometry, "coords"):
            coords = np.asarray(geometry.coords, dtype=float)
        else:
            coords = np.asarray(
                [
                    (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                    (graph.nodes[v]["x"], graph.nodes[v]["y"]),
                ],
                dtype=float,
            )
        if coords.shape[0] >= 2:
            segments.append(coords[:, :2])
    return segments


def regional_graph_cache_path(city: dict) -> Path:
    identity = json.dumps(
        {
            "center": [city["center_lat"], city["center_lon"]],
            "radius_m": city["radius_m"],
            "highway_filter": city["highway_filter"],
            "simplify": True,
            "retain_all": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    key = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return PROJECT_ROOT / "cache" / f"graph_regional_malibu_santa_monica_bay_{key}.graphml"


def load_regional_graph(city: dict, *, offline: bool):
    graph_path = regional_graph_cache_path(city)
    if graph_path.exists():
        graph = ox.load_graphml(graph_path)
    else:
        if offline:
            raise FileNotFoundError(f"Regional graph is not cached: {graph_path}")
        url_setting = "overpass_url" if hasattr(ox.settings, "overpass_url") else "overpass_endpoint"
        timeout_setting = (
            "requests_timeout" if hasattr(ox.settings, "requests_timeout") else "timeout"
        )
        old_url = getattr(ox.settings, url_setting)
        old_timeout = getattr(ox.settings, timeout_setting)
        errors = []
        try:
            setattr(ox.settings, timeout_setting, max(int(old_timeout), 600))
            for endpoint in OSMProcessor._default_overpass_endpoints:
                try:
                    setattr(ox.settings, url_setting, endpoint)
                    graph = ox.graph_from_point(
                        (float(city["center_lat"]), float(city["center_lon"])),
                        dist=float(city["radius_m"]),
                        dist_type="bbox",
                        network_type="drive",
                        simplify=True,
                        retain_all=True,
                        truncate_by_edge=False,
                        custom_filter=city["highway_filter"],
                    )
                    ox.save_graphml(graph, graph_path)
                    break
                except Exception as exc:
                    errors.append(f"{endpoint}: {type(exc).__name__}: {exc}")
            else:
                raise RuntimeError("Unable to fetch regional OSM graph; " + "; ".join(errors))
        finally:
            setattr(ox.settings, url_setting, old_url)
            setattr(ox.settings, timeout_setting, old_timeout)
    return ox.project_graph(graph, to_crs="EPSG:3857"), graph_path


def load_experimental_graph(
    city: dict,
    common: dict,
    *,
    skip_consolidation: bool,
    offline: bool,
):
    if city.get("graph_mode") == "regional_major_roads":
        graph, graph_path = load_regional_graph(city, offline=offline)
        return graph, graph_path, False, 0.0
    processor = OSMProcessor(
        city["address"],
        query_mode="point",
        center_point=(city["center_lat"], city["center_lon"]),
        radius_m=city["radius_m"],
        verbose=False,
    )
    graph_path = Path(processor._graph_cache_path())
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Cached experiment graph is missing for {city['city_id']}: {graph_path}"
        )
    processor.setLocationDrive()
    consolidation_enabled = bool(
        common.get("intersectionConsolidationEnabled", True)
    ) and not skip_consolidation
    tolerance = float(common.get("intersectionConsolidationToleranceM", 5.0))
    if consolidation_enabled:
        processor.consolidateIntersections(tolerance)
    graph = ox.project_graph(processor.locationDrive, to_crs="EPSG:3857")
    return graph, graph_path, consolidation_enabled, tolerance


def render_pair(
    city: dict,
    graph,
    graph_path: Path,
    output_dir: Path,
    session: requests.Session,
    tile_cache: Path,
    *,
    dpi: int,
    tile_zoom: int | None,
    offline: bool,
    consolidation_enabled: bool,
    consolidation_tolerance_m: float,
) -> dict:
    bounds = (
        point_radius_bounds(
            float(city["center_lat"]),
            float(city["center_lon"]),
            float(city["radius_m"]),
        )
        if city.get("graph_mode") == "regional_major_roads"
        else graph_bounds(graph)
    )
    resolved_tile_zoom = automatic_tile_zoom(bounds) if tile_zoom is None else tile_zoom
    basemap, tile_extent, downloaded_tiles = basemap_for_bounds(
        bounds,
        resolved_tile_zoom,
        session,
        tile_cache,
        offline=offline,
    )
    segments = edge_segments(graph)
    node_x = np.fromiter((float(d["x"]) for _, d in graph.nodes(data=True)), float)
    node_y = np.fromiter((float(d["y"]) for _, d in graph.nodes(data=True)), float)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.titlesize": 11.5,
            "figure.facecolor": "white",
        }
    )
    fig, (map_ax, graph_ax) = plt.subplots(1, 2, figsize=(12.8, 6.2))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.095, top=0.865, wspace=0.045)
    radius_km = float(city["radius_m"]) / 1000.0
    scope_label = city.get("scope_label", "experimental study area")
    distance_label = city.get("distance_label", f"{radius_km:g} km radius")
    fig.suptitle(
        f"{city['display_name']} - {scope_label} ({distance_label})",
        fontsize=15.5,
        fontweight="bold",
        y=0.955,
    )

    map_ax.imshow(
        basemap,
        # ``tile_extent`` follows the project-wide (xmin, ymin, xmax, ymax)
        # convention; Matplotlib expects (left, right, bottom, top).
        extent=(tile_extent[0], tile_extent[2], tile_extent[1], tile_extent[3]),
        origin="upper",
        interpolation="bilinear",
        zorder=0,
    )
    map_ax.set_title("(a) OpenStreetMap context", pad=9)
    graph_ax.set_facecolor("#f7f9fc")
    graph_ax.add_collection(
        LineCollection(
            segments,
            colors="#44586f",
            linewidths=0.32,
            alpha=0.68,
            capstyle="round",
            zorder=1,
            rasterized=True,
        )
    )
    node_size = 1.25 if len(node_x) < 2_000 else (0.70 if len(node_x) < 15_000 else 0.42)
    graph_ax.scatter(
        node_x,
        node_y,
        s=node_size,
        c="#df4c32",
        alpha=0.86,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )
    graph_ax.set_title(
        f"(b) {city.get('skeleton_title', 'Experimental walking-network skeleton')}",
        pad=9,
    )

    for axis in (map_ax, graph_ax):
        axis.set_xlim(bounds[0], bounds[2])
        axis.set_ylim(bounds[1], bounds[3])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#c6ced8")
            spine.set_linewidth(0.8)

    fig.text(
        0.025,
        0.035,
        OSM_ATTRIBUTION,
        fontsize=8.2,
        color="#4f5965",
        ha="left",
    )
    topology_note = (
        f"Skeleton: {graph.number_of_nodes():,} nodes · "
        f"{graph.number_of_edges():,} directed edges · major OSM road classes"
        if city.get("graph_mode") == "regional_major_roads"
        else (
            f"Skeleton: {graph.number_of_nodes():,} nodes · "
            f"{graph.number_of_edges():,} directed edges · "
            f"configured {consolidation_tolerance_m:g} m intersection consolidation"
        )
        if consolidation_enabled
        else (
            f"Skeleton: {graph.number_of_nodes():,} nodes · "
            f"{graph.number_of_edges():,} directed edges · raw cached walking graph"
        )
    )
    fig.text(0.985, 0.035, topology_note, fontsize=8.2, color="#4f5965", ha="right")

    output_path = output_dir / f"{city['city_id']}__osm_vs_skeleton.png"
    pdf_path = output_dir / f"{city['city_id']}__osm_vs_skeleton.pdf"
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {
        "city_id": city["city_id"],
        "display_name": city["display_name"],
        "scope_label": scope_label,
        "graph_mode": city.get("graph_mode", "experimental_walking_network"),
        "extent_note": city.get("extent_note"),
        "radius_m": float(city["radius_m"]),
        "graph_cache_path": str(graph_path.resolve()),
        "graph_cache_sha256": sha256_file(graph_path),
        "experimental_graph_nodes": int(graph.number_of_nodes()),
        "experimental_graph_directed_edges": int(graph.number_of_edges()),
        "intersection_consolidation": {
            "enabled": consolidation_enabled,
            "tolerance_m": consolidation_tolerance_m if consolidation_enabled else None,
        },
        "tile_provider": "OpenStreetMap Standard",
        "tile_zoom": resolved_tile_zoom,
        "tiles_downloaded_this_run": downloaded_tiles,
        "output_path": str(output_path.resolve()),
        "pdf_path": str(pdf_path.resolve()),
    }


def create_contact_sheet(image_paths: Iterable[Path], output_path: Path) -> None:
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if not images:
        return
    width = 1800
    resized = []
    for image in images:
        height = round(image.height * width / image.width)
        resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    gutter = 24
    canvas = Image.new(
        "RGB",
        (width, sum(image.height for image in resized) + gutter * (len(resized) - 1)),
        "white",
    )
    y = 0
    for image in resized:
        framed = ImageOps.expand(image, border=(0, 0, 0, 1), fill="#d5dbe3")
        canvas.paste(framed, (0, y))
        y += image.height + gutter
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> int:
    args = parse_args()
    if args.tile_zoom is not None and not 0 <= args.tile_zoom <= 19:
        raise ValueError("--tile-zoom must fall in [0, 19]")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    with args.profiles.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    figure_profiles = {"city_overrides": {}}
    if args.figure_profiles is not None and args.figure_profiles.exists():
        with args.figure_profiles.open("r", encoding="utf-8") as handle:
            figure_profiles = json.load(handle)
    overrides = figure_profiles.get("city_overrides", {})
    cities = [
        {**city, **overrides.get(city["city_id"], {})}
        for city in profiles["cities"]
    ]
    if args.cities:
        requested = set(args.cities)
        known = {city["city_id"] for city in cities}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown city IDs: {', '.join(unknown)}")
        cities = [city for city in cities if city["city_id"] in requested]

    output_dir = args.output_dir.resolve()
    tile_cache = args.tile_cache.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_cache.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str((PROJECT_ROOT / "cache").resolve())
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "RLEvacuationCollab-map-figure-generator/1.0 "
                "(academic research; OpenStreetMap-attributed static figures)"
            )
        }
    )

    records = []
    print(f"Rendering {len(cities)} city pairs to {output_dir}", flush=True)
    for index, city in enumerate(cities, start=1):
        print(f"[{index}/{len(cities)}] {city['display_name']}: loading graph", flush=True)
        graph, graph_path, consolidation_enabled, tolerance = load_experimental_graph(
            city,
            profiles.get("common_experiment", {}),
            skip_consolidation=args.skip_consolidation,
            offline=args.offline,
        )
        record = render_pair(
            city,
            graph,
            graph_path,
            output_dir,
            session,
            tile_cache,
            dpi=args.dpi,
            tile_zoom=args.tile_zoom,
            offline=args.offline,
            consolidation_enabled=consolidation_enabled,
            consolidation_tolerance_m=tolerance,
        )
        records.append(record)
        print(f"[{index}/{len(cities)}] wrote {record['output_path']}", flush=True)

    contact_sheet_path = output_dir / "all_cities__osm_vs_skeleton.png"
    create_contact_sheet(
        (Path(record["output_path"]) for record in records),
        contact_sheet_path,
    )
    manifest = {
        "schema_version": 1,
        "profile_path": str(args.profiles.resolve()),
        "profile_sha256": sha256_file(args.profiles.resolve()),
        "figure_profile_path": (
            str(args.figure_profiles.resolve())
            if args.figure_profiles is not None and args.figure_profiles.exists()
            else None
        ),
        "figure_profile_sha256": (
            sha256_file(args.figure_profiles.resolve())
            if args.figure_profiles is not None and args.figure_profiles.exists()
            else None
        ),
        "attribution": OSM_ATTRIBUTION,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "figures": records,
    }
    manifest_path = output_dir / "city_network_pair_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote contact sheet: {contact_sheet_path}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
