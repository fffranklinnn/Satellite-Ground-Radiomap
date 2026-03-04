#!/usr/bin/env python3
"""
Generate a Shaanxi-scale terrain radiomap by stitching L2 tiles.

Design goal:
- Show province-scale terrain impact (occlusion/diffraction loss) clearly.
- Keep runtime practical with explicit progress + ETA reporting.

Notes:
- This script focuses on L2 terrain loss to highlight topographic effects.
- Province extent is inferred from the local Shaanxi shapefile directory by default.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyogrio
import yaml
from pyproj import Transformer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.layers import L2TopoLayer
from src.layers.base import LayerContext


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHP_DIR = PROJECT_ROOT / "data" / "l3_urban" / "shanxisheng" / "陕西省"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate province-scale L2 terrain radiomap for Shaanxi."
    )
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml",
                        help="Base config path (used for L2 defaults)")
    parser.add_argument("--output-dir", type=str, default="output/shaanxi_radiomap",
                        help="Output directory")
    parser.add_argument("--output-prefix", type=str, default="shaanxi_l2_terrain",
                        help="Output filename prefix")
    parser.add_argument("--timestamp", type=str, default="2025-01-01T06:00:00",
                        help="Timestamp label for metadata/filename")

    parser.add_argument("--province-shp-dir", type=str, default=str(DEFAULT_SHP_DIR),
                        help="Directory containing Shaanxi shapefiles (*.shp)")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Optional override bbox as lon_min,lat_min,lon_max,lat_max")
    parser.add_argument("--bbox-margin-deg", type=float, default=0.05,
                        help="Margin added around inferred bbox (degrees)")

    parser.add_argument("--tile-step-km", type=float, default=25.6,
                        help="Tile step in km for province stitching")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Optional cap for debug")

    parser.add_argument("--frequency-ghz", type=float, default=None,
                        help="Override L2 frequency in GHz")
    parser.add_argument("--sat-elevation-deg", type=float, default=None,
                        help="Override satellite elevation for L2")
    parser.add_argument("--sat-azimuth-deg", type=float, default=None,
                        help="Override satellite azimuth for L2")

    parser.add_argument("--dpi", type=int, default=220,
                        help="PNG dpi")
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_bbox(text: str) -> Tuple[float, float, float, float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 4:
        raise ValueError("--bbox must be lon_min,lat_min,lon_max,lat_max")
    lon_min, lat_min, lon_max, lat_max = vals
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError("Invalid --bbox ordering.")
    return lon_min, lat_min, lon_max, lat_max


def _iter_shp_files(shp_dir: Path) -> Iterable[Path]:
    for p in sorted(shp_dir.glob("*.shp")):
        if p.is_file():
            yield p


def _infer_bbox_from_shp_dir(shp_dir: Path, margin_deg: float) -> Tuple[float, float, float, float, int]:
    lon_min = float("inf")
    lon_max = float("-inf")
    lat_min = float("inf")
    lat_max = float("-inf")
    count = 0

    for shp in _iter_shp_files(shp_dir):
        info = pyogrio.read_info(str(shp))
        bounds = info.get("total_bounds")
        if not bounds:
            continue
        crs = info.get("crs") or "EPSG:4326"
        x0, y0, x1, y1 = bounds
        t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs = [x0, x1, x0, x1]
        ys = [y0, y0, y1, y1]
        lons, lats = t.transform(xs, ys)
        lon_min = min(lon_min, *lons)
        lon_max = max(lon_max, *lons)
        lat_min = min(lat_min, *lats)
        lat_max = max(lat_max, *lats)
        count += 1

    if count == 0:
        raise FileNotFoundError(f"No shapefiles found in: {shp_dir}")

    lon_min -= margin_deg
    lon_max += margin_deg
    lat_min -= margin_deg
    lat_max += margin_deg
    return lon_min, lat_min, lon_max, lat_max, count


def _format_hms(seconds: float) -> str:
    sec = int(max(seconds, 0.0))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_progress(done: int, total: int, t0: float) -> None:
    elapsed = max(time.time() - t0, 1e-6)
    rate = done / elapsed
    remain = max(total - done, 0)
    eta = remain / rate if rate > 1e-12 else float("inf")
    eta_str = "N/A" if not np.isfinite(eta) else _format_hms(eta)
    pct = 100.0 * done / total if total > 0 else 100.0
    print(
        f"[progress] {done}/{total} ({pct:.1f}%) | "
        f"rate={rate:.3f} tiles/s | eta={eta_str}"
    )


def _mesh_geo_ticks(
    ax: plt.Axes,
    rows: int,
    cols: int,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    n_ticks: int = 6,
) -> None:
    x_ticks = np.linspace(0, cols - 1, n_ticks)
    y_ticks = np.linspace(0, rows - 1, n_ticks)
    lon_vals = np.linspace(lon_min, lon_max, n_ticks)
    lat_vals = np.linspace(lat_max, lat_min, n_ticks)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:.2f}°E" for x in lon_vals], fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}°N" for y in lat_vals], fontsize=8)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)


def main() -> None:
    args = parse_args()

    cfg_path = _resolve_path(args.config)
    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(cfg_path)

    l2_cfg = dict(cfg.get("layers", {}).get("l2_topo", {}))
    if args.frequency_ghz is not None:
        l2_cfg["frequency_ghz"] = float(args.frequency_ghz)
    if args.sat_elevation_deg is not None:
        l2_cfg["satellite_elevation_deg"] = float(args.sat_elevation_deg)
    if args.sat_azimuth_deg is not None:
        l2_cfg["satellite_azimuth_deg"] = float(args.sat_azimuth_deg)

    if args.bbox:
        lon_min, lat_min, lon_max, lat_max = _parse_bbox(args.bbox)
        shp_count = 0
    else:
        shp_dir = _resolve_path(args.province_shp_dir)
        lon_min, lat_min, lon_max, lat_max, shp_count = _infer_bbox_from_shp_dir(
            shp_dir, margin_deg=float(args.bbox_margin_deg)
        )

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    x_min, y_min = to_3857.transform(lon_min, lat_min)
    x_max, y_max = to_3857.transform(lon_max, lat_max)

    step_m = float(args.tile_step_km) * 1000.0
    if step_m <= 0:
        raise ValueError("--tile-step-km must be > 0")

    nx = max(1, int(math.ceil((x_max - x_min) / step_m)))
    ny = max(1, int(math.ceil((y_max - y_min) / step_m)))
    total_tiles = nx * ny
    if args.max_tiles is not None:
        total_tiles = min(total_tiles, int(args.max_tiles))

    print("[shaanxi] generating province terrain radiomap")
    print(f"[shaanxi] bbox (lon/lat): {lon_min:.4f}, {lat_min:.4f}, {lon_max:.4f}, {lat_max:.4f}")
    print(f"[shaanxi] inferred shapefiles: {shp_count}")
    print(f"[shaanxi] grid tiles: nx={nx}, ny={ny}, total={nx*ny}, run_total={total_tiles}")
    print(
        f"[shaanxi] L2 params: f={l2_cfg.get('frequency_ghz', 'NA')} GHz, "
        f"el={l2_cfg.get('satellite_elevation_deg', 'NA')} deg, "
        f"az={l2_cfg.get('satellite_azimuth_deg', 'NA')} deg"
    )

    origin_lat = float((lat_min + lat_max) * 0.5)
    origin_lon = float((lon_min + lon_max) * 0.5)
    l2 = L2TopoLayer(l2_cfg, origin_lat=origin_lat, origin_lon=origin_lon)

    tile_px = 256
    rows = ny * tile_px
    cols = nx * tile_px
    mosaic = np.full((rows, cols), np.nan, dtype=np.float32)

    ctx = LayerContext.from_any({
        "satellite_elevation_deg": float(l2_cfg.get("satellite_elevation_deg", 45.0)),
        "satellite_azimuth_deg": float(l2_cfg.get("satellite_azimuth_deg", 180.0)),
    })

    t0 = time.time()
    done = 0
    progress_interval = max(1, total_tiles // 40)

    stop = False
    for iy in range(ny):
        for ix in range(nx):
            if args.max_tiles is not None and done >= int(args.max_tiles):
                stop = True
                break

            x0 = x_min + ix * step_m
            y0 = y_min + iy * step_m
            lon0, lat0 = to_4326.transform(x0, y0)

            try:
                tile_loss = l2.compute(
                    origin_lat=float(lat0),
                    origin_lon=float(lon0),
                    timestamp=None,
                    context=ctx,
                )
            except Exception as exc:
                print(f"[warn] tile({iy},{ix}) skipped: {exc}")
                tile_loss = np.full((tile_px, tile_px), np.nan, dtype=np.float32)

            r0 = (ny - 1 - iy) * tile_px
            c0 = ix * tile_px
            mosaic[r0:r0 + tile_px, c0:c0 + tile_px] = tile_loss.astype(np.float32, copy=False)

            done += 1
            if done == 1 or done % progress_interval == 0 or done == total_tiles:
                _print_progress(done, total_tiles, t0)

        if stop:
            break

    elapsed = time.time() - t0
    tag = args.timestamp.replace(":", "").replace("-", "")
    base = f"{args.output_prefix}_{tag}"

    npy_path = out_dir / f"{base}_l2_loss.npy"
    np.save(npy_path, mosaic)

    valid = np.isfinite(mosaic)
    vmin = float(np.nanpercentile(mosaic, 1)) if np.any(valid) else 0.0
    vmax = float(np.nanpercentile(mosaic, 99)) if np.any(valid) else 1.0

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(
        mosaic,
        cmap="inferno",
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("L2 Terrain Loss (dB)", rotation=270, labelpad=18)

    _mesh_geo_ticks(ax, rows, cols, lon_min, lon_max, lat_min, lat_max, n_ticks=7)
    ax.set_title(
        "Shaanxi Province Terrain Radiomap (L2)\n"
        f"{args.timestamp} | tile_step={args.tile_step_km:.1f} km | "
        f"f={float(l2_cfg.get('frequency_ghz', 14.5)):.2f} GHz | "
        f"el={float(l2_cfg.get('satellite_elevation_deg', 45.0)):.1f} deg | "
        f"az={float(l2_cfg.get('satellite_azimuth_deg', 180.0)):.1f} deg",
        fontsize=12,
    )

    mean_val = float(np.nanmean(mosaic)) if np.any(valid) else float("nan")
    std_val = float(np.nanstd(mosaic)) if np.any(valid) else float("nan")
    min_val = float(np.nanmin(mosaic)) if np.any(valid) else float("nan")
    max_val = float(np.nanmax(mosaic)) if np.any(valid) else float("nan")
    valid_pct = float(np.mean(valid) * 100.0)

    text = (
        f"tiles_done={done}/{total_tiles}\n"
        f"shape={mosaic.shape[0]}x{mosaic.shape[1]}\n"
        f"valid={valid_pct:.1f}%\n"
        f"mean/std={mean_val:.3f}/{std_val:.3f} dB\n"
        f"min/max={min_val:.3f}/{max_val:.3f} dB\n"
        f"elapsed={elapsed:.2f}s"
    )
    ax.text(
        0.012,
        0.012,
        text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    png_path = out_dir / f"{base}_l2_loss.png"
    plt.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    meta = {
        "timestamp": args.timestamp,
        "bbox_lonlat": [lon_min, lat_min, lon_max, lat_max],
        "nx_tiles": nx,
        "ny_tiles": ny,
        "tiles_done": done,
        "tile_step_km": float(args.tile_step_km),
        "l2_frequency_ghz": float(l2_cfg.get("frequency_ghz", 14.5)),
        "satellite_elevation_deg": float(l2_cfg.get("satellite_elevation_deg", 45.0)),
        "satellite_azimuth_deg": float(l2_cfg.get("satellite_azimuth_deg", 180.0)),
        "output_npy": str(npy_path),
        "output_png": str(png_path),
        "elapsed_sec": round(elapsed, 3),
        "loss_stats": {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        },
    }
    meta_path = out_dir / f"{base}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    l2.close()

    print("=" * 88)
    print("Shaanxi terrain radiomap generated")
    print(f"  NPY   : {npy_path}")
    print(f"  PNG   : {png_path}")
    print(f"  META  : {meta_path}")
    print(f"  tiles : {done}/{total_tiles} | elapsed={elapsed:.2f}s")
    print("=" * 88)


if __name__ == "__main__":
    main()
