#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Take a base city layout from Stage 1 and add DeliveryBench-specific annotations:
- POI tags (e.g., restaurants, stores, rest areas)
- Bus routes and bus stations
- Roadside charging stations

This produces the DeliveryBench-ready map file:
  maps/<map_name>/progen_world_enriched.json

Inputs (per map directory):
- progen_world.json
- roads.json

Notes:
- This script may optionally merge extra building metadata (if configured), but it is not
  required for basic DeliveryBench usage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# =========================
# IO
# =========================
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
# Geometry helpers
# =========================
def dist(p: Dict[str, float], q: Dict[str, float]) -> float:
    return math.hypot(float(p["x"]) - float(q["x"]), float(p["y"]) - float(q["y"]))


def cumulative_lengths(points: Sequence[Dict[str, float]]) -> List[float]:
    acc = [0.0]
    for i in range(1, len(points)):
        acc.append(acc[-1] + dist(points[i - 1], points[i]))
    return acc


def heading_deg(p: Dict[str, float], q: Dict[str, float]) -> float:
    return math.degrees(math.atan2(float(q["y"]) - float(p["y"]), float(q["x"]) - float(p["x"])))


def unit_tangent(p: Dict[str, float], q: Dict[str, float]) -> Tuple[float, float]:
    dx, dy = (float(q["x"]) - float(p["x"])), (float(q["y"]) - float(p["y"]))
    L = math.hypot(dx, dy)
    if L == 0.0:
        return 1.0, 0.0
    return dx / L, dy / L


def left_normal(p: Dict[str, float], q: Dict[str, float]) -> Tuple[float, float]:
    tx, ty = unit_tangent(p, q)
    return -ty, tx


# =========================
# World helpers
# =========================
def collect_buildings(world: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        n
        for n in (world.get("nodes", []) or [])
        if isinstance(n, dict) and str(n.get("instance_name", "")).startswith("BP_Building")
    ]


def normalize_roads_list(roads_json: Any) -> List[Dict[str, Any]]:
    if isinstance(roads_json, dict) and "roads" in roads_json:
        roads_json = roads_json.get("roads")
    return list(roads_json or [])


def build_road_graph_from_roads(
    roads_json: Any, *, rounding_digits: int = 4
) -> Tuple[Dict[Tuple[float, float], List[Tuple[Tuple[float, float], float]]], Dict[Tuple[float, float], Dict[str, float]]]:
    """
    Build an undirected graph from `roads.json`.
    Nodes are road endpoints rounded to `rounding_digits` to de-duplicate.
    Edge weights are Euclidean distances (in meters).
    """
    graph: Dict[Tuple[float, float], List[Tuple[Tuple[float, float], float]]] = defaultdict(list)
    node_map: Dict[Tuple[float, float], Dict[str, float]] = {}
    seen: set[Tuple[float, float]] = set()

    def key(p: Dict[str, Any]) -> Tuple[float, float]:
        return (round(float(p["x"]), rounding_digits), round(float(p["y"]), rounding_digits))

    seg_cnt = 0
    for seg in normalize_roads_list(roads_json):
        s, e = seg.get("start"), seg.get("end")
        if not (isinstance(s, dict) and isinstance(e, dict)):
            continue

        u, v = key(s), key(e)
        if u == v:
            continue

        if u not in seen:
            node_map[u] = {"x": float(u[0]), "y": float(u[1])}
            seen.add(u)
        if v not in seen:
            node_map[v] = {"x": float(v[0]), "y": float(v[1])}
            seen.add(v)

        w = dist(node_map[u], node_map[v])
        if w <= 0.0:
            continue

        graph[u].append((v, w))
        graph[v].append((u, w))
        seg_cnt += 1

    print(f"[build_road_graph_from_roads] segments={seg_cnt}, nodes={len(node_map)}")
    return graph, node_map


def extract_segments_from_roads(roads_json: Any) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
    segments: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    for seg in normalize_roads_list(roads_json):
        s, e = seg.get("start"), seg.get("end")
        if not (isinstance(s, dict) and isinstance(e, dict)):
            continue
        p = {"x": float(s.get("x", 0.0)), "y": float(s.get("y", 0.0))}
        q = {"x": float(e.get("x", 0.0)), "y": float(e.get("y", 0.0))}
        if p["x"] == q["x"] and p["y"] == q["y"]:
            continue
        segments.append((p, q))
    return segments


# =========================
# Stable ID generator
# =========================
class IdFactory:
    def __init__(self, existing_ids: Optional[Iterable[str]] = None):
        self.counters: Dict[str, int] = defaultdict(int)
        self.existing = set(existing_ids or [])

    def new_id(self, instance_name: str) -> str:
        self.counters[instance_name] += 1
        seq = self.counters[instance_name] - 1

        for _ in range(100):
            rid = random.randint(0, 999)
            nid = f"GEN_{instance_name}_{seq}_{rid}"
            if nid not in self.existing:
                self.existing.add(nid)
                return nid

        # Extremely unlikely fallback.
        nid = f"GEN_{instance_name}_{seq}_{random.randint(1000, 9999)}"
        self.existing.add(nid)
        return nid


# =========================
# POI writers
# =========================
def add_point_poi(
    world: Dict[str, Any],
    id_factory: IdFactory,
    *,
    instance_name: str,
    poi_type: str,
    x: float,
    y: float,
    yaw: float = 0.0,
) -> str:
    nid = id_factory.new_id(instance_name)
    node = {
        "id": nid,
        "instance_name": instance_name,
        "properties": {
            "poi_type": poi_type,
            "location": {"x": float(x), "y": float(y), "z": 0.0},
            "orientation": {"pitch": 0.0, "yaw": float(yaw), "roll": 0.0},
        },
    }
    world.setdefault("nodes", []).append(node)
    return nid


# =========================
# Bus route + stations
# =========================
def generate_route_polyline(
    graph: Dict[Tuple[float, float], List[Tuple[Tuple[float, float], float]]],
    node_map: Dict[Tuple[float, float], Dict[str, float]],
    *,
    target_len_m: float = 30_000.0,
    max_steps: int = 6000,
) -> List[Dict[str, float]]:
    """
    Generate a polyline by walking along the road graph.
    Units: meters.
    """
    if not graph:
        return []

    cur = random.choice(list(graph.keys()))
    route_nodes = [cur]
    total = 0.0
    visited_edges: set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()

    steps = 0
    while total < target_len_m and steps < max_steps:
        steps += 1
        nbrs = graph.get(cur, [])
        if not nbrs:
            break

        unvisited, visited = [], []
        for v, w in nbrs:
            ek = (cur, v) if cur < v else (v, cur)
            (unvisited if ek not in visited_edges else visited).append((v, w, ek))

        cand = unvisited if unvisited else visited
        if not cand:
            break

        v, w, ek = random.choice(cand)
        visited_edges.add(ek)
        route_nodes.append(v)
        total += float(w)
        cur = v

    return [node_map[n] for n in route_nodes]


def place_bus_stops_along(
    polyline_m: Sequence[Dict[str, float]],
    *,
    num_stops: int = 6,
    min_gap_m: float = 50.0,
    spacing_jitter_frac: float = 0.2,
    lateral_offset_m: float = 3.0,
    scale_m_to_cm: float = 100.0,
) -> List[Dict[str, float]]:
    """
    Place bus stops along a road polyline.

    Returns a list of points:
      [{"x": <cm>, "y": <cm>, "yaw": <deg>}, ...]
    """
    if len(polyline_m) < 2 or num_stops <= 0:
        return []

    cum = cumulative_lengths(polyline_m)
    total = cum[-1]
    if total <= 0.0:
        return []

    base_step = total / (num_stops + 1)
    targets: List[float] = []

    for i in range(1, num_stops + 1):
        center = base_step * i
        jitter = random.uniform(-spacing_jitter_frac, spacing_jitter_frac) * base_step
        pos = max(0.0, min(total, center + jitter))

        if targets and (pos - targets[-1]) < min_gap_m:
            pos = min(targets[-1] + min_gap_m, total)

        if i < num_stops and (total - pos) < (num_stops - i) * min_gap_m:
            pos = max(pos - (min_gap_m / 2.0), 0.0)

        targets.append(pos)

    stops: List[Dict[str, float]] = []
    seg_idx = 0

    for tlen in targets:
        while seg_idx + 1 < len(cum) and cum[seg_idx + 1] < tlen:
            seg_idx += 1

        p = polyline_m[seg_idx]
        q = polyline_m[seg_idx + 1] if seg_idx + 1 < len(polyline_m) else p
        L = dist(p, q)
        if L <= 0.0:
            stops.append({"x": float(p["x"]) * scale_m_to_cm, "y": float(p["y"]) * scale_m_to_cm, "yaw": 0.0})
            continue

        alpha = max(0.0, min(1.0, (tlen - cum[seg_idx]) / L))
        cx = float(p["x"]) + (float(q["x"]) - float(p["x"])) * alpha
        cy = float(p["y"]) + (float(q["y"]) - float(p["y"])) * alpha

        side_sign = random.choice((+1, -1))
        nx, ny = left_normal(p, q)
        px = cx + nx * (lateral_offset_m * side_sign)
        py = cy + ny * (lateral_offset_m * side_sign)

        # Make the station face toward the road centerline.
        yaw = heading_deg(p, q) - side_sign * 180.0
        stops.append({"x": px * scale_m_to_cm, "y": py * scale_m_to_cm, "yaw": float(yaw)})

    return stops


# =========================
# Road chargers
# =========================
def add_chargers_along_roads(
    world: Dict[str, Any],
    id_factory: IdFactory,
    roads_json: Any,
    *,
    count: int = 10,
    offset_m: float = 2.0,
    scale_m_to_cm: float = 100.0,
) -> int:
    segments = extract_segments_from_roads(roads_json)
    if not segments or count <= 0:
        return 0

    placed = 0
    for _ in range(count):
        p, q = random.choice(segments)
        t = random.uniform(0.15, 0.85)
        cx = float(p["x"]) + (float(q["x"]) - float(p["x"])) * t
        cy = float(p["y"]) + (float(q["y"]) - float(p["y"])) * t

        side_sign = random.choice((+1, -1))
        nx, ny = left_normal(p, q)
        px = cx + nx * (offset_m * side_sign)
        py = cy + ny * (offset_m * side_sign)

        yaw = heading_deg(p, q)
        add_point_poi(
            world,
            id_factory,
            instance_name="POI_ChargingStation_Road",
            poi_type="charging_station",
            x=px * scale_m_to_cm,
            y=py * scale_m_to_cm,
            yaw=yaw,
        )
        placed += 1

    return placed


# =========================
# Optional building metadata merge (kept for backwards compatibility)
# =========================
def merge_bbox_into_world_inplace(
    *,
    world_obj: Dict[str, Any],
    bbox_obj: Dict[str, Any],
    key_in_world: str = "instance_name",
    bbox_root_key: str = "buildings",
    write_to: str = "properties.bbox",
    default_zero_for_buildings: bool = True,
) -> None:
    """
    Merge extra building metadata into each node by matching `instance_name`.
    (Historically this came from a bbox file.)
    """
    buildings_map = (bbox_obj.get(bbox_root_key, {}) or {}) if isinstance(bbox_obj, dict) else {}
    items = world_obj.get("nodes", []) if isinstance(world_obj, dict) else []

    def set_nested(holder: dict, dotted_key: str, value: Any) -> None:
        cur = holder
        *parents, last = dotted_key.split(".")
        for k in parents:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[last] = value

    hit = 0
    miss = 0
    for obj in items:
        if not isinstance(obj, dict):
            miss += 1
            continue

        name = obj.get(key_in_world)
        if not name:
            miss += 1
            continue

        info = buildings_map.get(name)
        if isinstance(info, dict) and ("bbox" in info):
            set_nested(obj, write_to, info["bbox"])
            hit += 1
        else:
            if default_zero_for_buildings and isinstance(name, str) and name.startswith("BP_Building"):
                set_nested(obj, write_to, {"x": 0, "y": 0, "z": 0})
            miss += 1

    print(f"[merge_bbox] matched={hit}, others={miss}")


# =========================
# Building tagging
# =========================
def tag_random_buildings(
    world: Dict[str, Any],
    *,
    k: int,
    label: str,
    exclude_ids: set,
) -> List[Dict[str, Any]]:
    buildings = collect_buildings(world)
    cand = [
        b
        for b in buildings
        if b.get("id") not in exclude_ids and (b.get("properties", {}) or {}).get("poi_type") is None
    ]
    if not cand:
        return []

    chosen = random.sample(cand, k=min(int(k), len(cand)))
    for b in chosen:
        b.setdefault("properties", {})["poi_type"] = label

    exclude_ids.update(str(b["id"]) for b in chosen if "id" in b)
    return chosen


# =========================
# Main enrichment entrypoint
# =========================
def enrich_world_in_place(
    *,
    world_json_path: Path,
    roads_json_path: Optional[Path],
    bbox_json_path: Optional[Path],
    output_path: Path,
    # building tags
    num_restaurants: int = 5,
    num_stores: int = 5,
    num_rest_areas: int = 2,
    num_hospitals: int = 1,
    num_car_rentals: int = 2,
    # bus
    bus_route_target_len_m: float = 30_000.0,
    num_bus_stops: int = 10,
    bus_min_gap_m: float = 50.0,
    bus_spacing_jitter_frac: float = 0.2,
    bus_offset_m: float = 3.0,
    # chargers
    road_chargers: int = 10,
    charger_offset_m: float = 2.0,
    # scaling
    scale_m_to_cm: float = 100.0,
    rng_seed: Optional[int] = None,
) -> None:
    if rng_seed is not None:
        random.seed(int(rng_seed))
    else:
        random.seed()

    world = load_json(world_json_path)
    if not isinstance(world, dict):
        raise ValueError(f"world_json_path must contain an object, got: {type(world)}")

    existing_ids = {
        str(n.get("id"))
        for n in (world.get("nodes", []) or [])
        if isinstance(n, dict) and ("id" in n) and (n.get("id") is not None)
    }
    id_factory = IdFactory(existing_ids=existing_ids)

    # 1) Tag buildings into POI categories (by setting properties.poi_type).
    exclude_ids: set = set()
    restaurants = tag_random_buildings(world, k=num_restaurants, label="restaurant", exclude_ids=exclude_ids)
    stores = tag_random_buildings(world, k=num_stores, label="store", exclude_ids=exclude_ids)
    rest_areas = tag_random_buildings(world, k=num_rest_areas, label="rest_area", exclude_ids=exclude_ids)
    hospitals = tag_random_buildings(world, k=num_hospitals, label="hospital", exclude_ids=exclude_ids)
    car_rentals = tag_random_buildings(world, k=num_car_rentals, label="car_rental", exclude_ids=exclude_ids)

    # 2) Roads data: road chargers + bus route graph.
    bus_routes: List[Dict[str, Any]] = []
    graph = None
    node_map = None
    roads_json = None

    if roads_json_path is not None:
        roads_json = load_json(roads_json_path)
        graph, node_map = build_road_graph_from_roads(roads_json)

        added_road_chargers = add_chargers_along_roads(
            world,
            id_factory,
            roads_json,
            count=road_chargers,
            offset_m=charger_offset_m,
            scale_m_to_cm=scale_m_to_cm,
        )
    else:
        added_road_chargers = 0

    # 3) Bus route + stations.
    if graph and node_map:
        polyline_m = generate_route_polyline(
            graph,
            node_map,
            target_len_m=bus_route_target_len_m,
            max_steps=6000,
        )
        print(f"Generated bus route polyline with {len(polyline_m)} points.")
        if len(polyline_m) >= 2:
            stops = place_bus_stops_along(
                polyline_m,
                num_stops=num_bus_stops,
                min_gap_m=bus_min_gap_m,
                spacing_jitter_frac=bus_spacing_jitter_frac,
                lateral_offset_m=bus_offset_m,
                scale_m_to_cm=scale_m_to_cm,
            )
            station_ids: List[str] = []
            for pt in stops:
                nid = add_point_poi(
                    world,
                    id_factory,
                    instance_name="POI_BusStation",
                    poi_type="bus_station",
                    x=float(pt["x"]),
                    y=float(pt["y"]),
                    yaw=float(pt["yaw"]),
                )
                station_ids.append(nid)

            bus_routes.append(
                {
                    "type": "bus_route",
                    "id": "route_bus_1",
                    "station_ids": station_ids,
                    # Keep the route polyline in meters (matching roads.json).
                    "path": polyline_m,
                }
            )
            world["bus_routes"] = bus_routes
    else:
        print("No roads graph provided for bus route; skip bus.")

    # 4) Optional extra building metadata.
    if bbox_json_path is not None:
        bbox_obj = load_json(bbox_json_path)
        if not isinstance(bbox_obj, dict):
            raise ValueError(f"bbox_json_path must contain an object, got: {type(bbox_obj)}")
        merge_bbox_into_world_inplace(world_obj=world, bbox_obj=bbox_obj)

    # 5) Save.
    save_json(world, output_path)

    print(
        "Enrichment summary: "
        f"restaurants={len(restaurants)}, stores={len(stores)}, rest_areas={len(rest_areas)}, "
        f"hospitals={len(hospitals)}, car_rentals={len(car_rentals)}, road_chargers={added_road_chargers}"
    )
    if bus_routes:
        print(f"bus_routes=1, stops={len(bus_routes[0]['station_ids'])}, route_points={len(bus_routes[0]['path'])}")
    else:
        print("bus_routes=0")
    print(f"Saved: {output_path}")


def _default_bbox_path(repo_root: Path) -> Optional[Path]:
    """
    Prefer `simworld/data/Bbox.json` (exists in this repo).
    """
    cand = repo_root / "simworld" / "data" / "Bbox.json"
    return cand if cand.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich a map with DeliveryBench-specific annotations.")
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=None,
        help="A directory under maps/ containing progen_world.json and roads.json.",
    )
    parser.add_argument("--world-json", type=Path, default=None, help="Path to progen_world.json")
    parser.add_argument("--roads-json", type=Path, default=None, help="Path to roads.json (optional)")
    parser.add_argument(
        "--bbox-json",
        type=Path,
        default=None,
        help="Optional extra building metadata file (legacy name: bbox).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output path (default: <map-dir>/progen_world_enriched.json)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic enrichment.")
    parser.add_argument("--scale-m-to-cm", type=float, default=100.0, help="Coordinate scale for POI placement.")

    # building tags
    parser.add_argument("--num-restaurants", type=int, default=5)
    parser.add_argument("--num-stores", type=int, default=5)
    parser.add_argument("--num-rest-areas", type=int, default=2)
    parser.add_argument("--num-hospitals", type=int, default=1)
    parser.add_argument("--num-car-rentals", type=int, default=2)

    # bus
    parser.add_argument("--bus-route-target-len-m", type=float, default=30_000.0)
    parser.add_argument("--num-bus-stops", type=int, default=10)
    parser.add_argument("--bus-min-gap-m", type=float, default=50.0)
    parser.add_argument("--bus-spacing-jitter-frac", type=float, default=0.2)
    parser.add_argument("--bus-offset-m", type=float, default=3.0)

    # chargers
    parser.add_argument("--road-chargers", type=int, default=10)
    parser.add_argument("--charger-offset-m", type=float, default=2.0)

    # safety
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.map_dir is not None:
        map_dir = args.map_dir
        world_json = map_dir / "progen_world.json"
        roads_json = map_dir / "roads.json"
        out_json = args.out_json or (map_dir / "progen_world_enriched.json")
    else:
        if args.world_json is None or args.out_json is None:
            parser.error("Either --map-dir or (--world-json and --out-json) must be provided.")
        world_json = args.world_json
        roads_json = args.roads_json
        out_json = args.out_json

    bbox_json = args.bbox_json
    if bbox_json is None:
        bbox_json = _default_bbox_path(repo_root)

    if not world_json.is_file():
        raise FileNotFoundError(f"Missing world json: {world_json}")
    if roads_json is not None and not roads_json.is_file():
        print(f"[warn] roads.json not found, will skip roads-based enrichment: {roads_json}")
        roads_json = None
    if bbox_json is not None and not bbox_json.is_file():
        # Not required; silently skip.
        bbox_json = None

    if out_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_json}. Pass --overwrite to replace it.")

    enrich_world_in_place(
        world_json_path=world_json,
        roads_json_path=roads_json,
        bbox_json_path=bbox_json,
        output_path=out_json,
        num_restaurants=args.num_restaurants,
        num_stores=args.num_stores,
        num_rest_areas=args.num_rest_areas,
        num_hospitals=args.num_hospitals,
        num_car_rentals=args.num_car_rentals,
        bus_route_target_len_m=args.bus_route_target_len_m,
        num_bus_stops=args.num_bus_stops,
        bus_min_gap_m=args.bus_min_gap_m,
        bus_spacing_jitter_frac=args.bus_spacing_jitter_frac,
        bus_offset_m=args.bus_offset_m,
        road_chargers=args.road_chargers,
        charger_offset_m=args.charger_offset_m,
        scale_m_to_cm=args.scale_m_to_cm,
        rng_seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

