#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless map rendering smoke test (no GUI / no interaction).

This script:
- auto-detects the DeliveryBench repo root
- loads `maps/<map_name>/roads.json` + `maps/<map_name>/progen_world_enriched.json`
- samples a few random orders (pickup=restaurant, dropoff=building)
- renders and saves a single global PNG (optionally also a local PNG)

It is designed for Linux/headless usage (CI / servers). No window will be shown.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _detect_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        # 1) Current directory is the repo root.
        if (p / "vlm_delivery").is_dir() and (p / "maps").is_dir() and (p / "simworld").is_dir():
            return p

        # 2) Repo root is a direct child of the current path (common in monorepos/workspaces).
        for name in ("DeliveryBench-release", "DeliveryBench", "Food-Delivery-Bench"):
            cand = p / name
            if (cand / "vlm_delivery").is_dir() and (cand / "maps").is_dir() and (cand / "simworld").is_dir():
                return cand
    raise RuntimeError("Cannot auto-detect DeliveryBench repo root (missing vlm_delivery/, maps/, simworld/).")


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _load_cfg(repo_root: Path) -> Dict[str, Any]:
    """
    Load config the same way the main runner does:
    - game_mechanics_config.json provides defaults (incl. map geometry like sidewalk_offset_cm)
    - experiment_config.json may override a subset
    """
    game_cfg_path = repo_root / "vlm_delivery" / "input" / "game_mechanics_config.json"
    exp_cfg_path = repo_root / "vlm_delivery" / "input" / "experiment_config.json"

    game_cfg: Dict[str, Any] = {}
    exp_cfg: Dict[str, Any] = {}
    if game_cfg_path.is_file():
        game_cfg = json.loads(game_cfg_path.read_text(encoding="utf-8")) or {}
    if exp_cfg_path.is_file():
        exp_cfg = json.loads(exp_cfg_path.read_text(encoding="utf-8")) or {}

    return _deep_merge_dicts(game_cfg, exp_cfg)


def _load_world_nodes(world_json: Path) -> List[Dict[str, Any]]:
    with world_json.open("r", encoding="utf-8") as f:
        data = json.load(f) or {}
    return list((data.get("nodes") or []))


def _xy_from_props(props: Dict[str, Any]) -> Tuple[float, float]:
    loc = props.get("location", {}) or {}
    return float(loc.get("x", 0.0)), float(loc.get("y", 0.0))


def _sample_restaurant_hints(nodes: List[Dict[str, Any]], k: int) -> List[Optional[Dict[str, float]]]:
    cands = [
        (n.get("properties", {}) or {})
        for n in nodes
        if ((n.get("properties", {}) or {}).get("poi_type") or (n.get("properties", {}) or {}).get("type") or "")
        .strip()
        .lower()
        == "restaurant"
    ]
    if not cands:
        return [None] * k
    out: List[Optional[Dict[str, float]]] = []
    for _ in range(k):
        props = random.choice(cands)
        x, y = _xy_from_props(props)
        out.append({"x": x, "y": y})
    return out


def _sample_building_hints(nodes: List[Dict[str, Any]], k: int) -> List[Optional[Dict[str, float]]]:
    cands = [
        (n.get("properties", {}) or {})
        for n in nodes
        if str(n.get("instance_name", "")).startswith("BP_Building")
    ]
    if not cands:
        return [None] * k
    out: List[Optional[Dict[str, float]]] = []
    for _ in range(k):
        props = random.choice(cands)
        x, y = _xy_from_props(props)
        out.append({"x": x, "y": y})
    return out


@dataclass
class _OrderStub:
    id: str
    pickup_node: Any
    dropoff_node: Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a DeliveryBench map to PNG (headless).")
    parser.add_argument("--map-name", type=str, default="small-city-11", help="Map directory name under maps/.")
    parser.add_argument("--num-orders", type=int, default=4, help="Number of random orders to place on the map.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for orders + agent placement).")
    parser.add_argument("--show-road-names", action="store_true", help="Render road name labels on the export images.")
    parser.add_argument("--save-local", action="store_true", help="Also save a local (zoomed-in) PNG.")
    parser.add_argument(
        "--plain-buildings",
        type=str,
        default="all",
        choices=["none", "all"],
        help='Whether to draw generic buildings as gray rectangles ("all" matches the original map test behavior).',
    )
    parser.add_argument(
        "--out-global",
        type=Path,
        default=None,
        help="Optional explicit output path for the global PNG (overrides --out-dir naming).",
    )
    parser.add_argument(
        "--out-local",
        type=Path,
        default=None,
        help="Optional explicit output path for the local PNG (overrides --out-dir naming).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo_root>/outputs/map_debug/).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist.")
    args = parser.parse_args()

    # Must be set before importing PyQt5.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    repo_root = _detect_repo_root()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "simworld"))

    from PyQt5.QtWidgets import QApplication  # noqa: E402

    from vlm_delivery.map.map import Map  # noqa: E402
    from vlm_delivery.map.map_exportor import MapExportor  # noqa: E402

    random.seed(int(args.seed))
    cfg = _load_cfg(repo_root)

    map_dir = repo_root / "maps" / str(args.map_name)
    roads_json = map_dir / "roads.json"
    world_json = map_dir / "progen_world_enriched.json"
    if not roads_json.is_file():
        raise FileNotFoundError(f"Missing roads.json: {roads_json}")
    if not world_json.is_file():
        raise FileNotFoundError(f"Missing progen_world_enriched.json: {world_json}")

    out_dir = args.out_dir or (repo_root / "outputs" / "map_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_global = Path(args.out_global) if args.out_global else (out_dir / f"{args.map_name}_global_{ts}.png")
    out_local = Path(args.out_local) if args.out_local else (out_dir / f"{args.map_name}_local_{ts}.png")

    if (out_global.exists() or (args.save_local and out_local.exists())) and not args.overwrite:
        raise FileExistsError("Output file already exists. Pass --overwrite to replace it.")

    # Qt app is required even for off-screen export.
    app = QApplication.instance() or QApplication([])

    # Build map.
    # IMPORTANT: pass map geometry config (e.g., sidewalk_offset_cm) to match the official viewer.
    m = Map(cfg=(cfg.get("map", {}) or {}))
    m.import_roads(str(roads_json))
    m.import_pois(str(world_json))

    # Create a few random orders (restaurant -> building).
    nodes = _load_world_nodes(world_json)
    pu_hints = _sample_restaurant_hints(nodes, int(args.num_orders))
    do_hints = _sample_building_hints(nodes, int(args.num_orders))

    orders_input: List[Dict[str, Any]] = []
    for i in range(int(args.num_orders)):
        orders_input.append(
            {
                "id": str(i + 1),
                "pickup_type": "restaurant",
                "dropoff_type": "building",
                "pickup_hint": pu_hints[i],
                "dropoff_hint": do_hints[i],
            }
        )

    order_meta = m.set_active_orders(orders_input, world_json_path=str(world_json))
    order_stubs = [
        _OrderStub(id=str(rec.get("id", "")), pickup_node=rec.get("pickup_node"), dropoff_node=rec.get("dropoff_node"))
        for rec in order_meta
    ]

    # Sample an agent position inside bbox.
    mnx, mxx, mny, mxy = m.bbox()
    pad = 1000.0
    ax = random.uniform(mnx + pad, mxx - pad)
    ay = random.uniform(mny + pad, mxy - pad)

    exportor = MapExportor(map_obj=m, world_json_path=str(world_json), show_road_names=bool(args.show_road_names))
    exportor.prepare_base()

    # Match the original map test: show all generic buildings as gray rectangles.
    if str(args.plain_buildings).lower() == "all":
        try:
            v = exportor.viewer
            pw_g = getattr(v, "_exp_pw_g", None)
            pw_l = getattr(v, "_exp_pw_l", None)
            if pw_g is not None:
                v._draw_all_building_boxes(pw_g)
            if pw_l is not None:
                v._draw_all_building_boxes(pw_l)
        except Exception:
            pass

    global_bytes, local_bytes = exportor.export(agent_xy=(ax, ay), orders=order_stubs)

    out_global.write_bytes(global_bytes)
    if args.save_local:
        out_local.write_bytes(local_bytes)

    # No GUI loop; exit cleanly.
    try:
        app.quit()
    except Exception:
        pass

    print(f"Saved: {out_global}")
    if args.save_local:
        print(f"Saved: {out_local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

