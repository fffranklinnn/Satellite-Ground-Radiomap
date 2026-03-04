#!/usr/bin/env python3
"""
Generate one city-wide full-physics radiomap for Xi'an by stitching all L3 tiles.

Pipeline:
  1) Select satellite + compute L1 components once (macro baseline template)
  2) For each city tile: compute L2(local topo) + L3(urban) at 256x256
  3) Composite per tile = L1_template + L2_interp + L3
  4) Stitch all tiles to one large city map and export PNG + NPY
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import RadioMapAggregator
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.layers.base import LayerContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one city-wide Xi'an radiomap mosaic.")
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml")
    parser.add_argument("--tile-list", type=str, default="data/l3_urban/xian/tile_list_xian_60.csv")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="ISO timestamp, e.g. 2025-01-01T06:00:00")
    parser.add_argument("--output-dir", type=str, default="output/xian_city_radiomap")
    parser.add_argument("--output-prefix", type=str, default="xian_city_full")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--norad-id", action="append", default=None,
                        help="Limit candidate satellites to specific NORAD ID; can be repeated")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Optional debug cap for number of tiles")
    parser.add_argument("--save-components", action="store_true",
                        help="Also save city-wide L2 and L3 mosaics")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_timestamp(ts_str: Optional[str], config: Dict) -> datetime:
    if ts_str is None:
        ts_str = config.get("time", {}).get("start", "2025-01-01T00:00:00")
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _geo_ticks_for_mosaic(ax, rows: int, cols: int,
                          lon_west: float, lon_east: float,
                          lat_south: float, lat_north: float,
                          n_ticks: int = 6) -> None:
    x_ticks = np.linspace(0, cols - 1, n_ticks)
    y_ticks = np.linspace(0, rows - 1, n_ticks)
    lon_vals = np.linspace(lon_west, lon_east, n_ticks)
    lat_vals = np.linspace(lat_north, lat_south, n_ticks)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{v:.4f}°E" for v in lon_vals], fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.4f}°N" for v in lat_vals], fontsize=8)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)


def _tile_placement(origin_x: float,
                    origin_y: float,
                    x_to_idx: Dict[float, int],
                    y_to_idx: Dict[float, int],
                    ny: int,
                    tile_px: int) -> Tuple[int, int, int, int]:
    x_idx = x_to_idx[origin_x]
    y_idx = y_to_idx[origin_y]
    r0 = (ny - 1 - y_idx) * tile_px
    c0 = x_idx * tile_px
    r1 = r0 + tile_px
    c1 = c0 + tile_px
    return r0, r1, c0, c1


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)

    tile_list_path = Path(args.tile_list)
    if not tile_list_path.is_absolute():
        tile_list_path = root / tile_list_path

    timestamp = parse_timestamp(args.timestamp, config)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tile_list_path)
    if not {"origin_x", "origin_y"}.issubset(df.columns):
        raise ValueError("tile-list CSV must contain origin_x and origin_y columns.")

    if args.max_tiles is not None:
        df = df.head(int(args.max_tiles)).copy()

    # Build city grid definition from tile-list
    x_vals = np.sort(df["origin_x"].astype(float).unique())
    y_vals = np.sort(df["origin_y"].astype(float).unique())
    nx = len(x_vals)
    ny = len(y_vals)
    x_to_idx = {float(v): i for i, v in enumerate(x_vals)}
    y_to_idx = {float(v): i for i, v in enumerate(y_vals)}

    tile_px = 256
    rows = ny * tile_px
    cols = nx * tile_px

    x_step = float(np.median(np.diff(x_vals))) if nx > 1 else 0.0
    y_step = float(np.median(np.diff(y_vals))) if ny > 1 else 0.0

    lon_west = float(x_vals.min())
    lon_east = float(x_vals.max() + x_step)
    lat_south = float(y_vals.min())
    lat_north = float(y_vals.max() + y_step)

    layers_cfg = config["layers"]
    l1_cfg = layers_cfg["l1_macro"]
    l2_cfg = layers_cfg["l2_topo"]
    l3_cfg = layers_cfg["l3_urban"]

    city_center_lat = float((lat_south + lat_north) / 2.0)
    city_center_lon = float((lon_west + lon_east) / 2.0)

    l1 = L1MacroLayer(l1_cfg, city_center_lat, city_center_lon)
    l2 = L2TopoLayer(l2_cfg, city_center_lat, city_center_lon)
    l3 = L3UrbanLayer(l3_cfg, city_center_lat, city_center_lon)
    agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2, l3_layer=l3)

    target_norad_ids: Optional[List[str]] = None
    if args.norad_id:
        target_norad_ids = [str(x).strip() for x in args.norad_id if str(x).strip()]
        if target_norad_ids:
            l1.target_norad_ids = target_norad_ids

    # L1 once: macro baseline template for each 256m tile
    l1_comp = l1.compute_components(
        timestamp=timestamp,
        origin_lat=city_center_lat,
        origin_lon=city_center_lon,
        target_norad_ids=target_norad_ids,
    )
    sat_info = l1_comp["satellite"]

    l1_template = agg._interpolate_to_target(l1_comp["total"], l1.coverage_km).astype(np.float32)
    l1_template_mean = float(np.mean(l1_template))

    incident_dir = l3_cfg.get("incident_dir")
    if incident_dir is None:
        incident_dir = {
            "az_deg": sat_info["azimuth_deg"],
            "el_deg": sat_info["elevation_deg"],
        }

    ctx = LayerContext.from_any({
        "incident_dir": incident_dir,
        "satellite_azimuth_deg": sat_info["azimuth_deg"],
        "satellite_elevation_deg": sat_info["elevation_deg"],
        "satellite_slant_range_km": sat_info.get("slant_range_km"),
        "satellite_altitude_km": sat_info.get("alt_m", 0.0) / 1000.0,
        "target_norad_ids": target_norad_ids,
    })

    city_composite = np.zeros((rows, cols), dtype=np.float32)
    city_l3 = np.zeros((rows, cols), dtype=np.float32)
    city_l2 = np.zeros((rows, cols), dtype=np.float32)

    total_tiles = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        ox = float(row.origin_x)
        oy = float(row.origin_y)

        l3_tile = l3.compute(origin_lat=oy, origin_lon=ox, timestamp=timestamp, context=ctx)
        l2_tile = l2.compute(origin_lat=oy, origin_lon=ox, timestamp=timestamp, context=ctx)
        l2_interp = agg._interpolate_to_target(l2_tile, l2.coverage_km).astype(np.float32)

        comp_tile = l1_template + l2_interp + l3_tile

        r0, r1, c0, c1 = _tile_placement(ox, oy, x_to_idx, y_to_idx, ny, tile_px)
        city_composite[r0:r1, c0:c1] = comp_tile
        city_l3[r0:r1, c0:c1] = l3_tile
        city_l2[r0:r1, c0:c1] = l2_interp

        if idx == 1 or idx % 100 == 0 or idx == total_tiles:
            print(f"[city] processed {idx}/{total_tiles} tiles")

    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    base = f"{args.output_prefix}_{stamp}"

    np.save(out_dir / f"{base}_composite.npy", city_composite)
    if args.save_components:
        np.save(out_dir / f"{base}_l2.npy", city_l2)
        np.save(out_dir / f"{base}_l3.npy", city_l3)

    # publication-style city-wide figure
    fig, ax = plt.subplots(figsize=(14, 12))
    vmin = float(np.percentile(city_composite, 1))
    vmax = float(np.percentile(city_composite, 99))
    im = ax.imshow(city_composite, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Composite Loss (dB)", rotation=270, labelpad=18)

    _geo_ticks_for_mosaic(ax, rows, cols, lon_west, lon_east, lat_south, lat_north)
    ax.set_title(
        "Xi'an City-Wide Full-Physics Radiomap\n"
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} | NORAD {sat_info.get('norad_id','N/A')} "
        f"(az={sat_info.get('azimuth_deg', float('nan')):.2f}°, el={sat_info.get('elevation_deg', float('nan')):.2f}°)",
        fontsize=12,
        fontweight="bold",
    )

    stats = [
        f"mosaic={ny}x{nx} tiles ({rows}x{cols} px)",
        f"L1 template mean/std={l1_template_mean:.3f}/{float(np.std(l1_template)):.3f} dB",
        f"L2 city mean/std={float(np.mean(city_l2)):.3f}/{float(np.std(city_l2)):.3f} dB",
        f"L3 city mean/std={float(np.mean(city_l3)):.3f}/{float(np.std(city_l3)):.3f} dB",
        f"Composite mean/std={float(np.mean(city_composite)):.3f}/{float(np.std(city_composite)):.3f} dB",
        f"Composite min/max={float(np.min(city_composite)):.3f}/{float(np.max(city_composite)):.3f} dB",
    ]

    ax.text(
        0.012,
        0.012,
        "\n".join(stats),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.84),
    )

    plt.tight_layout()
    out_png = out_dir / f"{base}.png"
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    l2.close()

    print("=" * 76)
    print("Xi'an city-wide radiomap generated")
    print(f"  PNG      : {out_png}")
    print(f"  NPY      : {out_dir / (base + '_composite.npy')}")
    if args.save_components:
        print(f"  L2 NPY   : {out_dir / (base + '_l2.npy')}")
        print(f"  L3 NPY   : {out_dir / (base + '_l3.npy')}")
    print(f"  City extent: lon {lon_west:.6f}~{lon_east:.6f}, lat {lat_south:.6f}~{lat_north:.6f}")
    print(f"  Composite stats: mean={float(np.mean(city_composite)):.3f}, std={float(np.std(city_composite)):.3f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
