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
import concurrent.futures as cf
import math
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
from src.layers.l3_urban import compute_nlos_mask


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
    parser.add_argument("--top-k-sats", type=int, default=1,
                        help="Use top-K visible satellites by elevation at city center (default: 1)")
    parser.add_argument("--sat-workers", type=int, default=1,
                        help="Parallel workers for per-tile top-K satellite evaluation (default: 1)")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Optional debug cap for number of tiles")
    parser.add_argument("--save-components", action="store_true",
                        help="Also save city-wide L2 and L3 mosaics")
    parser.add_argument("--use-config-incident-dir", action="store_true",
                        help="Use layers.l3_urban.incident_dir from config (default: dynamic satellite az/el)")
    parser.add_argument("--legacy-l2-origin", action="store_true",
                        help="Use legacy L2 origin (tile origin as 25.6km SW corner). "
                             "Default uses center-aligned L2 patch for each tile.")
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


def _shift_origin_by_km_sw(origin_lat: float,
                           origin_lon: float,
                           shift_km_south: float,
                           shift_km_west: float) -> Tuple[float, float]:
    """
    Shift a SW-corner lat/lon by given south/west distances in km.

    Uses local spherical approximation; sufficient for city-scale offsets.
    """
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / max(111.0 * math.cos(math.radians(origin_lat)), 1e-6)
    out_lat = float(origin_lat - shift_km_south * lat_per_km)
    out_lon = float(origin_lon - shift_km_west * lon_per_km)
    return out_lat, out_lon


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

    top_k_sats = max(int(args.top_k_sats), 1)
    visible_sats = l1.get_visible_satellites(
        origin_lat=city_center_lat,
        origin_lon=city_center_lon,
        timestamp=timestamp,
        target_norad_ids=target_norad_ids,
        max_count=top_k_sats,
    )
    if not visible_sats:
        raise RuntimeError(
            "No visible satellite found at city center for this timestamp."
        )

    sat_bundles: List[Dict[str, object]] = []
    l1_template_means: List[float] = []
    for sat_rank, sat_pred in enumerate(visible_sats, start=1):
        norad_id = str(sat_pred["norad_id"])
        l1_comp = l1.compute_components(
            timestamp=timestamp,
            origin_lat=city_center_lat,
            origin_lon=city_center_lon,
            target_norad_ids=[norad_id],
        )
        sat_info = l1_comp["satellite"]
        l1_template = agg._interpolate_to_target(l1_comp["total"], l1.coverage_km).astype(np.float32)
        l1_template_means.append(float(np.mean(l1_template)))

        if args.use_config_incident_dir and l3_cfg.get("incident_dir") is not None:
            incident_dir = l3_cfg.get("incident_dir")
            incident_source = "config"
        else:
            incident_dir = {
                "az_deg": sat_info["azimuth_deg"],
                "el_deg": sat_info["elevation_deg"],
            }
            incident_source = "satellite"

        ctx_extras = {
            "incident_dir": incident_dir,
            "satellite_azimuth_deg": sat_info["azimuth_deg"],
            "satellite_elevation_deg": sat_info["elevation_deg"],
            "satellite_slant_range_km": sat_info.get("slant_range_km"),
            "satellite_altitude_km": sat_info.get("alt_m", 0.0) / 1000.0,
            "target_norad_ids": [norad_id],
        }
        sat_bundles.append({
            "rank": sat_rank,
            "norad_id": norad_id,
            "sat_info": sat_info,
            "l1_template": l1_template,
            "ctx_extras": ctx_extras,
        })

    lead_sat_info = sat_bundles[0]["sat_info"]

    city_composite = np.zeros((rows, cols), dtype=np.float32)
    city_l3 = np.zeros((rows, cols), dtype=np.float32)
    city_l2 = np.zeros((rows, cols), dtype=np.float32)

    # L2 is a 25.6 km patch; target city tile is 0.256 km.
    # To keep the tile physically aligned with the center-crop interpolation,
    # shift L2 patch SW so that the city tile center is near the L2 patch center.
    # This captures surrounding terrain context while avoiding the legacy 12.8 km misalignment.
    l2_shift_km = max((float(l2.coverage_km) - 0.256) * 0.5, 0.0)

    sat_workers = max(int(args.sat_workers), 1)
    use_sat_parallel = len(sat_bundles) > 1 and sat_workers > 1
    sat_executor: Optional[cf.ThreadPoolExecutor] = None

    if use_sat_parallel:
        sat_executor = cf.ThreadPoolExecutor(max_workers=min(sat_workers, len(sat_bundles)))

    def _compute_l3_tile(
        height_m: np.ndarray,
        occ: Optional[np.ndarray],
        incident_dir: Dict[str, float],
    ) -> np.ndarray:
        # Reuse already-loaded tile cache to avoid repeated disk IO in top-K loop.
        nlos_mask = compute_nlos_mask(height_m, incident_dir)
        l3_tile = np.zeros(height_m.shape, dtype=np.float32)
        l3_tile[nlos_mask] = float(l3.nlos_loss_db)
        if l3.occ_loss_db is not None:
            occ_mask = occ.astype(bool) if occ is not None else (height_m > 0)
            l3_tile[occ_mask] = np.maximum(l3_tile[occ_mask], float(l3.occ_loss_db))
        return l3_tile

    def _compute_l2_interp_from_dem(
        dem_patch: np.ndarray,
        sat_info: Dict[str, float],
    ) -> np.ndarray:
        sat_elevation_deg = float(sat_info.get("elevation_deg", l2.sat_elevation_deg))
        sat_azimuth_deg = float(sat_info.get("azimuth_deg", l2.sat_azimuth_deg))
        sat_slant_range_km = sat_info.get("slant_range_km")
        sat_altitude_km = float(sat_info.get("alt_m", 0.0) / 1000.0) if sat_info.get("alt_m") is not None else float(l2.satellite_altitude_km)
        if sat_slant_range_km is None:
            sat_slant_range_km = l2._estimate_slant_range_km(sat_elevation_deg, sat_altitude_km)

        occlusion_mask, excess_height_m, obstacle_distance_m = l2._compute_occlusion_vectorized(
            dem_patch,
            sat_elevation_deg=sat_elevation_deg,
            sat_azimuth_deg=sat_azimuth_deg,
            return_profile=True,
        )
        l2_tile = l2._apply_diffraction_loss(
            dem=dem_patch,
            mask=occlusion_mask,
            excess_height_m=excess_height_m,
            obstacle_distance_m=obstacle_distance_m,
            sat_slant_range_m=max(float(sat_slant_range_km) * 1000.0, 1.0),
        )
        return agg._interpolate_to_target(l2_tile, l2.coverage_km).astype(np.float32, copy=False)

    def _compute_candidate(
        sat_bundle: Dict[str, object],
        dem_patch: np.ndarray,
        l3_height: np.ndarray,
        l3_occ: Optional[np.ndarray],
        static_l3_tile: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if static_l3_tile is None:
            incident_dir = sat_bundle["ctx_extras"]["incident_dir"]
            l3_tile_local = _compute_l3_tile(
                height_m=l3_height,
                occ=l3_occ,
                incident_dir=incident_dir,
            )
        else:
            l3_tile_local = static_l3_tile

        l2_interp_local = _compute_l2_interp_from_dem(dem_patch, sat_bundle["sat_info"])
        l1_template_local = sat_bundle["l1_template"]
        comp_local = (l1_template_local + l2_interp_local + l3_tile_local).astype(np.float32, copy=False)
        return comp_local, l2_interp_local, l3_tile_local

    try:
        total_tiles = len(df)
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            ox = float(row.origin_x)
            oy = float(row.origin_y)
            if args.legacy_l2_origin:
                l2_origin_lat, l2_origin_lon = oy, ox
            else:
                l2_origin_lat, l2_origin_lon = _shift_origin_by_km_sw(
                    origin_lat=oy,
                    origin_lon=ox,
                    shift_km_south=l2_shift_km,
                    shift_km_west=l2_shift_km,
                )

            l2._validate_bounds(l2_origin_lat, l2_origin_lon)
            dem_patch = l2._load_dem_patch(l2_origin_lat, l2_origin_lon)
            dem_patch = np.asarray(dem_patch, dtype=np.float32)
            l3_height, l3_occ = l3._load_tile_cache({"lat": oy, "lon": ox})
            l3_height = np.asarray(l3_height, dtype=np.float32)
            l3_occ = None if l3_occ is None else np.asarray(l3_occ)

            if len(sat_bundles) == 1:
                only = sat_bundles[0]
                comp_tile, l2_interp, l3_tile = _compute_candidate(
                    sat_bundle=only,
                    dem_patch=dem_patch,
                    l3_height=l3_height,
                    l3_occ=l3_occ,
                    static_l3_tile=None,
                )
            else:
                static_l3_tile: Optional[np.ndarray] = None
                if args.use_config_incident_dir:
                    static_l3_tile = _compute_l3_tile(
                        height_m=l3_height,
                        occ=l3_occ,
                        incident_dir=sat_bundles[0]["ctx_extras"]["incident_dir"],
                    )

                if sat_executor is not None:
                    futures = [
                        sat_executor.submit(
                            _compute_candidate,
                            sat_bundle,
                            dem_patch,
                            l3_height,
                            l3_occ,
                            static_l3_tile,
                        )
                        for sat_bundle in sat_bundles
                    ]
                    sat_results = [f.result() for f in futures]
                else:
                    sat_results = [
                        _compute_candidate(
                            sat_bundle=sat_bundle,
                            dem_patch=dem_patch,
                            l3_height=l3_height,
                            l3_occ=l3_occ,
                            static_l3_tile=static_l3_tile,
                        )
                        for sat_bundle in sat_bundles
                    ]

                best_comp: Optional[np.ndarray] = None
                best_l2: Optional[np.ndarray] = None
                best_l3: Optional[np.ndarray] = None
                for comp_candidate, l2_candidate, l3_candidate in sat_results:
                    if best_comp is None:
                        best_comp = comp_candidate.copy()
                        best_l2 = l2_candidate.copy()
                        best_l3 = l3_candidate.copy()
                    else:
                        improved = comp_candidate < best_comp
                        best_comp = np.where(improved, comp_candidate, best_comp)
                        best_l2 = np.where(improved, l2_candidate, best_l2)
                        best_l3 = np.where(improved, l3_candidate, best_l3)

                assert best_comp is not None and best_l2 is not None and best_l3 is not None
                comp_tile = best_comp.astype(np.float32, copy=False)
                l2_interp = best_l2.astype(np.float32, copy=False)
                l3_tile = best_l3.astype(np.float32, copy=False)

            r0, r1, c0, c1 = _tile_placement(ox, oy, x_to_idx, y_to_idx, ny, tile_px)
            city_composite[r0:r1, c0:c1] = comp_tile
            city_l3[r0:r1, c0:c1] = l3_tile
            city_l2[r0:r1, c0:c1] = l2_interp

            if idx == 1 or idx % 100 == 0 or idx == total_tiles:
                print(f"[city] processed {idx}/{total_tiles} tiles")
    finally:
        if sat_executor is not None:
            sat_executor.shutdown(wait=True)

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
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"Top-{len(sat_bundles)} sats @ center, lead NORAD {lead_sat_info.get('norad_id','N/A')} "
        f"(az={lead_sat_info.get('azimuth_deg', float('nan')):.2f}°, el={lead_sat_info.get('elevation_deg', float('nan')):.2f}°)",
        fontsize=12,
        fontweight="bold",
    )

    stats = [
        f"mosaic={ny}x{nx} tiles ({rows}x{cols} px)",
        f"L1 template mean range={min(l1_template_means):.3f}~{max(l1_template_means):.3f} dB",
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
    print(f"  Top-K sats       : {len(sat_bundles)}")
    print("  Center sat list  : " + ", ".join(
        f"#{int(b['rank'])}:{b['norad_id']}(el={float(b['sat_info'].get('elevation_deg', float('nan'))):.2f})"
        for b in sat_bundles
    ))
    print(f"  L3 incident source: {incident_source}")
    print(f"  L2 origin mode    : {'legacy' if args.legacy_l2_origin else 'center-aligned'}")
    print(f"  Sat workers       : {sat_workers}")
    print(f"  Composite stats: mean={float(np.mean(city_composite)):.3f}, std={float(np.std(city_composite)):.3f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
