#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a new procedural city layout (roads/buildings/elements/routes) using SimWorld's
Python city generator, and export it in the same format as `maps/*` expects:

  - roads.json
  - buildings.json
  - elements.json
  - routes.json
  - progen_world.json

This script does NOT require Unreal Engine; it uses `simworld.citygen`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _ensure_import_paths(repo_root: Path) -> None:
    # The SimWorld python package lives under `<repo_root>/simworld/simworld`.
    # Adding `<repo_root>/simworld` makes `import simworld` work when running from source.
    simworld_src = repo_root / "simworld"
    if simworld_src.is_dir():
        sys.path.insert(0, str(simworld_src))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a procedural DeliveryBench map under maps/.")
    parser.add_argument("--map-name", type=str, required=True, help="Directory name under maps/, e.g. generated-city-35")
    parser.add_argument(
        "--maps-root",
        type=Path,
        default=None,
        help="Path to maps/ directory (default: <repo_root>/maps).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic generation.")
    parser.add_argument(
        "--num-segments",
        type=int,
        default=35,
        help="Target number of road segments to generate.",
    )
    parser.add_argument("--generate-element", action="store_true", help="Also generate street elements.")
    parser.add_argument("--generate-route", action="store_true", help="Also generate routes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the target map directory if it exists.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_import_paths(repo_root)

    from simworld.citygen.function_call import CityFunctionCall  # noqa: E402
    from simworld.config import Config  # noqa: E402

    maps_root = args.maps_root or (repo_root / "maps")
    out_dir = maps_root / args.map_name

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target already exists: {out_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    # Avoid noisy fallback warnings by pointing to the repo's bbox file explicitly.
    # When running from the DeliveryBench repo root, the relative path is: simworld/data/Bbox.json
    bbox_path = repo_root / "simworld" / "data" / "Bbox.json"
    if bbox_path.is_file():
        config.config.setdefault("citygen", {})["input_bounding_boxes"] = str(bbox_path)

    cfc = CityFunctionCall(
        config,
        seed=int(args.seed),
        num_segments=int(args.num_segments),
        generate_element=bool(args.generate_element),
        generate_route=bool(args.generate_route),
    )
    cfc.generate_city()
    cfc.export_city(str(out_dir))

    print(f"✅ Generated city layout: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

