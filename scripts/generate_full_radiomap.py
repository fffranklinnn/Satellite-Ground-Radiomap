#!/usr/bin/env python3
"""
Generate one full-physics radiomap for a specific time/location.

This script computes and fuses:
  L1: FSPL + atmospheric + ionospheric + antenna gain/polarization
  L2: terrain occlusion/diffraction
  L3: urban NLoS/occupancy loss

Outputs:
  - publication-ready composite PNG
  - optional decomposition PNG
  - .npy arrays for composite and major components
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import RadioMapAggregator
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.layers.base import LayerContext
from src.layers.l3_urban import compute_nlos_mask
from src.utils import plot_full_radiomap_paper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one full-physics radiomap (single timestamp + single location)."
    )
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml",
                        help="Path to mission config YAML")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="ISO timestamp, e.g. 2025-01-01T06:00:00")
    parser.add_argument("--origin-lat", type=float, default=None,
                        help="Override origin latitude")
    parser.add_argument("--origin-lon", type=float, default=None,
                        help="Override origin longitude")
    parser.add_argument("--output-dir", type=str, default="output/full_radiomap",
                        help="Output directory")
    parser.add_argument("--output-prefix", type=str, default="full_radiomap",
                        help="Output filename prefix")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure DPI")
    parser.add_argument("--show-decomposition", action="store_true",
                        help="Also save L1/L2/L3/composite decomposition panel")
    parser.add_argument("--allow-missing-data", action="store_true",
                        help="Allow missing ionex/era5/dem/tile data with fallback where possible")
    parser.add_argument("--norad-id", action="append", default=None,
                        help="Limit candidate satellites to specific NORAD ID; can be repeated")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def parse_timestamp(ts_str: Optional[str], config: Dict) -> datetime:
    if ts_str is None:
        ts_str = config.get("time", {}).get("start", "2025-01-01T00:00:00")
    ts = datetime.fromisoformat(ts_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def check_required_data(project_root: Path,
                        config: Dict,
                        allow_missing: bool) -> None:
    layers_cfg = config.get("layers", {})
    l1_cfg = layers_cfg.get("l1_macro", {})
    l2_cfg = layers_cfg.get("l2_topo", {})
    l3_cfg = layers_cfg.get("l3_urban", {})

    checks = [
        ("TLE", resolve_path(project_root, l1_cfg.get("tle_file")), True),
        ("IONEX", resolve_path(project_root, l1_cfg.get("ionex_file")), False),
        ("ERA5", resolve_path(project_root, l1_cfg.get("era5_file")), False),
        ("DEM", resolve_path(project_root, l2_cfg.get("dem_file")), True),
        ("L3 Tile Cache", resolve_path(project_root, l3_cfg.get("tile_cache_root")), True),
    ]

    missing_required = []
    for label, path_obj, required in checks:
        if path_obj is None:
            if required:
                missing_required.append(f"{label}: <not configured>")
            else:
                print(f"[WARN] {label} is not configured; fallback may be used.")
            continue

        exists = path_obj.exists()
        if not exists and required:
            missing_required.append(f"{label}: {path_obj}")
        elif not exists:
            print(f"[WARN] {label} not found: {path_obj}; fallback may be used.")

    if missing_required and not allow_missing:
        details = "\n".join(missing_required)
        raise FileNotFoundError(
            "Missing required input data:\n"
            f"{details}\n"
            "Use --allow-missing-data to proceed with fallback behavior."
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config) if not Path(args.config).is_absolute() else Path(args.config)

    config = load_config(config_path)
    check_required_data(project_root, config, allow_missing=args.allow_missing_data)

    layers_cfg = config["layers"]
    l1_cfg = layers_cfg["l1_macro"]
    l2_cfg = layers_cfg["l2_topo"]
    l3_cfg = layers_cfg["l3_urban"]

    origin_cfg = config.get("origin", {})
    origin_lat = float(args.origin_lat if args.origin_lat is not None else origin_cfg.get("latitude", 34.3416))
    origin_lon = float(args.origin_lon if args.origin_lon is not None else origin_cfg.get("longitude", 108.9398))
    timestamp = parse_timestamp(args.timestamp, config)

    out_dir = (project_root / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l1_layer = L1MacroLayer(l1_cfg, origin_lat, origin_lon)
    l2_layer = L2TopoLayer(l2_cfg, origin_lat, origin_lon)
    l3_layer = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)

    target_norad_ids: Optional[List[str]] = None
    if args.norad_id:
        target_norad_ids = [str(x).strip() for x in args.norad_id if str(x).strip()]
        if target_norad_ids:
            l1_layer.target_norad_ids = target_norad_ids

    l1_components = l1_layer.compute_components(
        timestamp=timestamp,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        target_norad_ids=target_norad_ids,
    )
    sat_info = l1_components["satellite"]

    incident_dir = l3_cfg.get("incident_dir")
    if incident_dir is None:
        incident_dir = {
            "az_deg": sat_info["azimuth_deg"],
            "el_deg": sat_info["elevation_deg"],
        }

    context = LayerContext.from_any({
        "incident_dir": incident_dir,
        "satellite_azimuth_deg": sat_info["azimuth_deg"],
        "satellite_elevation_deg": sat_info["elevation_deg"],
        "satellite_slant_range_km": sat_info.get("slant_range_km"),
        "satellite_altitude_km": sat_info.get("alt_m", 0.0) / 1000.0,
        "target_norad_ids": target_norad_ids,
    })

    l2_map = l2_layer.compute(origin_lat, origin_lon, timestamp=timestamp, context=context)
    l3_map = l3_layer.compute(origin_lat, origin_lon, timestamp=timestamp, context=context)

    aggregator = RadioMapAggregator(l1_layer=l1_layer, l2_layer=l2_layer, l3_layer=l3_layer)
    l1_interp = aggregator._interpolate_to_target(l1_components["total"], l1_layer.coverage_km).astype(np.float32)
    l2_interp = aggregator._interpolate_to_target(l2_map, l2_layer.coverage_km).astype(np.float32)
    fspl_interp = aggregator._interpolate_to_target(l1_components["fspl"], l1_layer.coverage_km).astype(np.float32)
    atm_interp = aggregator._interpolate_to_target(l1_components["atm"], l1_layer.coverage_km).astype(np.float32)
    iono_interp = aggregator._interpolate_to_target(l1_components["iono"], l1_layer.coverage_km).astype(np.float32)

    composite = (l1_interp + l2_interp + l3_map).astype(np.float32)

    terrain_mask_l3 = aggregator._interpolate_to_target(
        (l2_map > 0).astype(np.float32),
        l2_layer.coverage_km,
    ) > 0.5

    height_m, occ = l3_layer._load_tile_cache({"lat": origin_lat, "lon": origin_lon})
    urban_nlos_mask = compute_nlos_mask(height_m, incident_dir)
    urban_occ_mask = occ.astype(bool) if occ is not None else (height_m > 0)

    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{args.output_prefix}_{stamp}"

    np.save(out_dir / f"{base_name}_composite.npy", composite)
    np.save(out_dir / f"{base_name}_l1.npy", l1_interp)
    np.save(out_dir / f"{base_name}_l2.npy", l2_interp)
    np.save(out_dir / f"{base_name}_l3.npy", l3_map)
    np.save(out_dir / f"{base_name}_fspl.npy", fspl_interp)
    np.save(out_dir / f"{base_name}_atm.npy", atm_interp)
    np.save(out_dir / f"{base_name}_iono.npy", iono_interp)
    np.save(out_dir / f"{base_name}_terrain_mask.npy", terrain_mask_l3.astype(np.uint8))
    np.save(out_dir / f"{base_name}_urban_nlos_mask.npy", urban_nlos_mask.astype(np.uint8))
    np.save(out_dir / f"{base_name}_urban_occ_mask.npy", urban_occ_mask.astype(np.uint8))

    main_png = out_dir / f"{base_name}.png"
    decomp_png = out_dir / f"{base_name}_decomposition.png"

    note_lines = {
        "NORAD": sat_info.get("norad_id", "N/A"),
        "sat_az_deg": f"{sat_info.get('azimuth_deg', float('nan')):.2f}",
        "sat_el_deg": f"{sat_info.get('elevation_deg', float('nan')):.2f}",
        "frequency_GHz": f"{l1_layer.frequency_ghz:.2f}",
    }

    plot_full_radiomap_paper(
        composite_map=composite,
        output_file=str(main_png),
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        coverage_km=0.256,
        timestamp=timestamp,
        title="Satellite-Ground Full-Physics Radiomap",
        l1_map=l1_interp,
        l2_map=l2_interp,
        l3_map=l3_map,
        iono_map=iono_interp,
        atm_map=atm_interp,
        terrain_mask=terrain_mask_l3,
        urban_nlos_mask=urban_nlos_mask,
        urban_occ_mask=urban_occ_mask,
        note_lines=note_lines,
        show_decomposition=args.show_decomposition,
        decomposition_output_file=str(decomp_png),
        dpi=args.dpi,
    )

    print("=" * 72)
    print("Full radiomap generated")
    print(f"  Timestamp: {timestamp.isoformat()}")
    print(f"  Origin   : ({origin_lat:.6f}N, {origin_lon:.6f}E)")
    print(f"  Satellite: NORAD {sat_info.get('norad_id', 'N/A')} | "
          f"az={sat_info.get('azimuth_deg', float('nan')):.2f} deg | "
          f"el={sat_info.get('elevation_deg', float('nan')):.2f} deg")
    print(f"  PNG      : {main_png}")
    if args.show_decomposition:
        print(f"  Decomp   : {decomp_png}")
    print(f"  NPY dir  : {out_dir}")
    print(f"  Component std (dB): L1={np.nanstd(l1_interp):.6f}, "
          f"L2={np.nanstd(l2_interp):.6f}, L3={np.nanstd(l3_map):.6f}")
    print(f"  Component mean (dB): L1={np.nanmean(l1_interp):.3f}, "
          f"L2={np.nanmean(l2_interp):.3f}, L3={np.nanmean(l3_map):.3f}")
    print(f"  Composite stats: mean={np.nanmean(composite):.3f} dB, "
          f"std={np.nanstd(composite):.3f} dB, "
          f"min={np.nanmin(composite):.3f} dB, max={np.nanmax(composite):.3f} dB")
    print("=" * 72)

    l2_layer.close()


if __name__ == "__main__":
    main()
