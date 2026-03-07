#!/usr/bin/env python3
"""
Generate a full-domain Xi'an radiomap from Xi'an shapefile coverage.

Pipeline:
1) Infer Xi'an bounds from a shapefile (EPSG:3857 or reprojected to it).
2) Build a full-domain tile list on a meter-based grid.
3) Build L3 tile cache from catalog + generated tile list.
4) Run city-wide radiomap stitching with the generated assets.

Default settings prioritize practical runtime and memory:
- tile_size_m=1024, resolution_m=4 -> each tile remains 256x256 pixels.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import pyogrio
import yaml
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XIAN_SHP = PROJECT_ROOT / "data" / "l3_urban" / "shanxisheng" / "陕西省" / "processed_xian20221010_all.shp"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "l3_urban" / "xian" / "catalog" / "buildings_xian.parquet"
BUILD_CACHE_SCRIPT = PROJECT_ROOT / "branch_L3" / "zhangyue" / "tools" / "build_l3_tile_cache.py"
CITY_MAP_SCRIPT = PROJECT_ROOT / "scripts" / "generate_xian_city_radiomap.py"


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-domain Xi'an radiomap from shapefile bounds."
    )
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml")
    parser.add_argument("--xian-shp", type=str, default=str(DEFAULT_XIAN_SHP))
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG))
    parser.add_argument("--timestamp", type=str, default="2025-01-01T12:00:00")

    parser.add_argument("--tile-size-m", type=float, default=1024.0,
                        help="Tile size in meters. 256 is highest detail but much heavier.")
    parser.add_argument("--margin-m", type=float, default=0.0,
                        help="Extra margin around shapefile bounds in meters.")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Optional cap for debug/quick run.")

    parser.add_argument("--output-root", type=str, default="output/xian_full_from_shp")
    parser.add_argument("--output-prefix", type=str, default="xian_full_from_shp")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--skip-cache-build", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--save-components", action="store_true")
    parser.add_argument("--norad-id", action="append", default=None)
    return parser.parse_args()


def _infer_bounds_3857(shp_path: Path) -> Tuple[float, float, float, float]:
    info = pyogrio.read_info(str(shp_path))
    bounds = info.get("total_bounds")
    if not bounds:
        raise RuntimeError(f"Cannot read bounds from shapefile: {shp_path}")
    x0, y0, x1, y1 = bounds
    crs = info.get("crs") or "EPSG:4326"
    to_3857 = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    xx = [x0, x1, x0, x1]
    yy = [y0, y0, y1, y1]
    tx, ty = to_3857.transform(xx, yy)
    return min(tx), min(ty), max(tx), max(ty)


def _aligned_grid_bounds(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    step_m: float,
) -> Tuple[float, float, float, float]:
    gx0 = math.floor(x_min / step_m) * step_m
    gy0 = math.floor(y_min / step_m) * step_m
    gx1 = math.ceil(x_max / step_m) * step_m
    gy1 = math.ceil(y_max / step_m) * step_m
    return gx0, gy0, gx1, gy1


def _generate_tile_list_csv(
    out_csv: Path,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    step_m: float,
    max_tiles: int | None,
) -> Dict[str, float]:
    to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    nx = max(1, int(math.ceil((x_max - x_min) / step_m)))
    ny = max(1, int(math.ceil((y_max - y_min) / step_m)))
    total = nx * ny
    run_total = min(total, int(max_tiles)) if max_tiles is not None else total

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["origin_x", "origin_y", "origin_x_3857", "origin_y_3857"])
        stop = False
        for j in range(ny):
            for i in range(nx):
                if max_tiles is not None and written >= int(max_tiles):
                    stop = True
                    break
                ox_3857 = x_min + i * step_m
                oy_3857 = y_min + j * step_m
                lon, lat = to_4326.transform(ox_3857, oy_3857)
                writer.writerow(
                    [
                        f"{lon:.10f}",
                        f"{lat:.10f}",
                        f"{ox_3857:.3f}",
                        f"{oy_3857:.3f}",
                    ]
                )
                written += 1
            if stop:
                break

    return {
        "nx": float(nx),
        "ny": float(ny),
        "total_tiles": float(total),
        "run_tiles": float(run_total),
    }


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _run_cmd(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with return code {proc.returncode}")


def main() -> None:
    args = parse_args()

    cfg_path = _resolve_path(args.config)
    shp_path = _resolve_path(args.xian_shp)
    catalog_path = _resolve_path(args.catalog)
    out_root = _resolve_path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not shp_path.exists():
        raise FileNotFoundError(f"Xi'an shapefile not found: {shp_path}")
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    if args.tile_size_m <= 0:
        raise ValueError("--tile-size-m must be > 0")

    resolution_m = float(args.tile_size_m) / 256.0
    if resolution_m <= 0:
        raise ValueError("Invalid resolution computed from tile size")

    x0, y0, x1, y1 = _infer_bounds_3857(shp_path)
    margin = float(args.margin_m)
    x0 -= margin
    y0 -= margin
    x1 += margin
    y1 += margin
    gx0, gy0, gx1, gy1 = _aligned_grid_bounds(x0, y0, x1, y1, float(args.tile_size_m))

    tile_list_path = out_root / f"tile_list_xian_full_{int(args.tile_size_m)}m.csv"
    tile_stats = _generate_tile_list_csv(
        out_csv=tile_list_path,
        x_min=gx0,
        y_min=gy0,
        x_max=gx1,
        y_max=gy1,
        step_m=float(args.tile_size_m),
        max_tiles=args.max_tiles,
    )

    print("[xian-full] shapefile:", shp_path)
    print(f"[xian-full] bounds_3857: ({x0:.3f}, {y0:.3f}) -> ({x1:.3f}, {y1:.3f})")
    print(
        f"[xian-full] grid: nx={int(tile_stats['nx'])}, ny={int(tile_stats['ny'])}, "
        f"total={int(tile_stats['total_tiles'])}, run={int(tile_stats['run_tiles'])}"
    )
    print(f"[xian-full] tile_size={args.tile_size_m:.1f} m, resolution={resolution_m:.3f} m/px")
    print(f"[xian-full] tile_list: {tile_list_path}")

    tile_cache_root = out_root / f"tiles_full_{int(args.tile_size_m)}m"
    if args.skip_cache_build:
        print("[xian-full] skip cache build (requested)")
    else:
        if tile_cache_root.exists() and not args.overwrite_cache:
            print(f"[xian-full] cache exists, reusing: {tile_cache_root}")
        else:
            build_cmd = [
                sys.executable,
                str(BUILD_CACHE_SCRIPT),
                "--catalog", str(catalog_path),
                "--tile-list", str(tile_list_path),
                "--output-root", str(tile_cache_root),
                "--origin-crs", "EPSG:4326",
                "--tile-size-m", str(float(args.tile_size_m)),
                "--resolution-m", str(float(resolution_m)),
                "--height-field", "height_m",
                "--log-level", "INFO",
            ]
            _run_cmd(build_cmd)

    cfg = _load_yaml(cfg_path)
    l3_cfg = cfg.setdefault("layers", {}).setdefault("l3_urban", {})
    l3_cfg["tile_cache_root"] = str(tile_cache_root)
    l3_cfg["coverage_km"] = float(args.tile_size_m) / 1000.0
    l3_cfg["resolution_m"] = float(resolution_m)

    derived_cfg = out_root / "configs" / f"mission_xian_full_{int(args.tile_size_m)}m.yaml"
    _save_yaml(derived_cfg, cfg)

    map_cmd = [
        sys.executable,
        str(CITY_MAP_SCRIPT),
        "--config", str(derived_cfg),
        "--tile-list", str(tile_list_path),
        "--timestamp", str(args.timestamp),
        "--output-dir", str(out_root / "radiomap"),
        "--output-prefix", str(args.output_prefix),
        "--dpi", str(int(args.dpi)),
    ]
    if args.max_tiles is not None:
        map_cmd += ["--max-tiles", str(int(args.max_tiles))]
    if args.save_components:
        map_cmd.append("--save-components")
    if args.norad_id:
        for norad in args.norad_id:
            if str(norad).strip():
                map_cmd += ["--norad-id", str(norad).strip()]

    _run_cmd(map_cmd)

    print("=" * 78)
    print("Xi'an full-domain radiomap done")
    print(f"  tile_list   : {tile_list_path}")
    print(f"  tile_cache  : {tile_cache_root}")
    print(f"  config      : {derived_cfg}")
    print(f"  radiomap_dir: {out_root / 'radiomap'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
