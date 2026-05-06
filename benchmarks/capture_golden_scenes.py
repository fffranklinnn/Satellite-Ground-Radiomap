#!/usr/bin/env python3
"""
Capture golden scenes for regression testing.

Saves L1-only, L1+L2, and L1+L2+L3 frame outputs with serialized arrays,
a deterministic manifest (no volatile fields), frame list, output file hashes,
fallback log, and satellite metadata.

Uses the FrameBuilder/BenchmarkRunner pipeline (not the legacy RadioMapAggregator)
so golden arrays are consistent with what run_regression.py produces.

Usage:
    python benchmarks/capture_golden_scenes.py [--config CONFIG] [--output OUTPUT_DIR]
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.context import GridSpec, FrameBuilder
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.context.multiscale_map import MultiScaleMap
from src.context.time_utils import parse_iso_utc
from src.products.manifest import collect_input_file_paths, _sha256_dir as _sha256_dir_manifest


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _config_hash(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialized).hexdigest()


def _repo_relative(path: str) -> str:
    """Return path relative to repo root (strip absolute prefix if present)."""
    p = Path(path)
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def capture(config_path: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cfg_hash = _config_hash(config)
    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]

    # Use the first time step; store as explicit UTC string
    raw_time = config["time"]["start"]
    frame_dt = parse_iso_utc(raw_time, strict=False)
    frame_time_utc = frame_dt.isoformat()

    # Build FrameBuilder from config
    l1_cfg = config["layers"]["l1_macro"]
    l2_cfg = config["layers"]["l2_topo"]
    l3_cfg = config["layers"].get("l3_urban", {})
    coarse_km = float(l1_cfg.get("coverage_km", 256.0))
    grid_size = int(l1_cfg.get("grid_size", 256))
    grid = GridSpec.from_legacy_args(origin_lat, origin_lon, coarse_km, grid_size, grid_size)
    fb = FrameBuilder(grid=grid)

    # Build a single frame (same satellite selection for all scenes)
    frame = fb.build(frame_dt)

    # --- L1 only ---
    print("Capturing L1-only scene...")
    l1 = L1MacroLayer(l1_cfg, origin_lat, origin_lon)
    entry = l1.propagate_entry(frame)
    # Collect fallbacks from L1 (constructor-time + per-frame)
    l1_fallbacks = list(l1.fallbacks_used)
    msm_l1 = MultiScaleMap.compose_legacy(frame_id=frame.frame_id, grid=frame.grid,
                                    entry=entry, terrain=None, urban=None)
    l1_map = msm_l1.composite_db
    np.save(output_dir / "l1_only.npy", l1_map)

    # Extract satellite metadata from L1 internals
    sat_meta: dict = {}
    if hasattr(l1, "_last_sat_meta"):
        sat_meta = l1._last_sat_meta or {}

    # --- L2 standalone (raw terrain loss) ---
    print("Capturing L2 standalone scene...")
    l2 = L2TopoLayer(l2_cfg, origin_lat, origin_lon)
    terrain = l2.propagate_terrain(frame, entry=entry)
    l2_raw = terrain.loss_db
    np.save(output_dir / "l2_raw.npy", l2_raw)

    # --- L1 + L2 composite ---
    print("Capturing L1+L2 scene...")
    msm_l1l2 = MultiScaleMap.compose_legacy(frame_id=frame.frame_id, grid=frame.grid,
                                      entry=entry, terrain=terrain, urban=None)
    composite_l1l2 = msm_l1l2.composite_db
    np.save(output_dir / "l1l2_composite.npy", composite_l1l2)
    np.save(output_dir / "l1l2_l1.npy", msm_l1l2.l1_db if msm_l1l2.l1_db is not None
            else np.zeros((256, 256), dtype=np.float32))
    np.save(output_dir / "l1l2_l2.npy", msm_l1l2.l2_db if msm_l1l2.l2_db is not None
            else np.zeros((256, 256), dtype=np.float32))

    # --- L1 + L2 + L3 composite ---
    print("Capturing L1+L2+L3 scene...")
    l3 = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)
    urban = l3.refine_urban(frame, entry=entry)
    msm_full = MultiScaleMap.compose_legacy(frame_id=frame.frame_id, grid=frame.grid,
                                      entry=entry, terrain=terrain, urban=urban)
    composite_full = msm_full.composite_db
    np.save(output_dir / "l1l2l3_composite.npy", composite_full)
    np.save(output_dir / "l1l2l3_l1.npy", msm_full.l1_db if msm_full.l1_db is not None
            else np.zeros((256, 256), dtype=np.float32))
    np.save(output_dir / "l1l2l3_l2.npy", msm_full.l2_db if msm_full.l2_db is not None
            else np.zeros((256, 256), dtype=np.float32))
    np.save(output_dir / "l1l2l3_l3.npy", msm_full.l3_db if msm_full.l3_db is not None
            else np.zeros((256, 256), dtype=np.float32))

    # --- Input file hashes (repo-relative paths) ---
    raw_input_paths = collect_input_file_paths(config)
    input_files = {}
    for key, val in raw_input_paths.items():
        p = Path(val)
        if p.is_dir():
            sha = _sha256_dir_manifest(str(p)) or "missing"
        else:
            sha = _sha256(p)
        input_files[key] = {"path": _repo_relative(str(p)), "sha256": sha}

    # --- Output file hashes ---
    output_files = {}
    for fname in [
        "l1_only.npy", "l2_raw.npy",
        "l1l2_composite.npy", "l1l2_l1.npy", "l1l2_l2.npy",
        "l1l2l3_composite.npy", "l1l2l3_l1.npy", "l1l2l3_l2.npy", "l1l2l3_l3.npy",
    ]:
        p = output_dir / fname
        output_files[fname] = {"sha256": _sha256(p)}

    # --- Spatial delta (L2 SW-corner vs center) ---
    coverage_km = l2_cfg.get("coverage_km", 25.6)
    half_km = coverage_km / 2.0
    delta_lat_deg = half_km / 111.0
    delta_lon_deg = half_km / (111.0 * math.cos(math.radians(origin_lat)))
    spatial_delta = {
        "description": (
            "L2 currently uses SW-corner origin convention. "
            "After the GridSpec fix, L2 will shift its DEM window by approximately "
            "half the coverage in both lat and lon directions."
        ),
        "l2_current_sw_corner_lat": origin_lat,
        "l2_current_sw_corner_lon": origin_lon,
        "l2_expected_center_lat_after_fix": round(origin_lat + delta_lat_deg, 6),
        "l2_expected_center_lon_after_fix": round(origin_lon + delta_lon_deg, 6),
        "delta_lat_deg": round(delta_lat_deg, 6),
        "delta_lon_deg": round(delta_lon_deg, 6),
        "coverage_km": coverage_km,
    }

    # --- Scene statistics ---
    def _stats(arr: np.ndarray) -> dict:
        return {
            "shape": list(arr.shape),
            "min_db": round(float(arr.min()), 4),
            "max_db": round(float(arr.max()), 4),
            "mean_db": round(float(arr.mean()), 4),
            "nonzero_pixels": int((arr != 0).sum()),
        }

    # --- Deterministic manifest (no capture_time, no elapsed_s, no absolute paths) ---
    manifest = {
        "schema_version": "1",
        "config_hash": cfg_hash,
        "config_path": _repo_relative(config_path),
        "frame_list": [frame_time_utc],
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "input_files": input_files,
        "output_files": output_files,
        "fallbacks_used": l1_fallbacks,
        "satellite_meta": sat_meta,
        "scenes": {
            "l1_only": _stats(l1_map),
            "l2_raw": _stats(l2_raw),
            "l1l2_composite": _stats(composite_l1l2),
            "l1l2_l2_cropped": _stats(msm_l1l2.l2_db if msm_l1l2.l2_db is not None
                                       else np.zeros((256, 256), dtype=np.float32)),
            "l1l2l3_composite": _stats(composite_full),
            "l1l2l3_l2_cropped": _stats(msm_full.l2_db if msm_full.l2_db is not None
                                         else np.zeros((256, 256), dtype=np.float32)),
            "l1l2l3_l3": _stats(msm_full.l3_db if msm_full.l3_db is not None
                                 else np.zeros((256, 256), dtype=np.float32)),
        },
        "spatial_delta": spatial_delta,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Frame list as separate artifact
    frame_list_path = output_dir / "frame_list.json"
    with open(frame_list_path, "w") as f:
        json.dump({"frames": [frame_time_utc]}, f, indent=2)

    # Config snapshot (repo-relative)
    config_snapshot_path = output_dir / "config_snapshot.yaml"
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nGolden scenes saved to: {output_dir}")
    print(f"Config hash: {cfg_hash}")
    print(f"\nScene statistics:")
    print(f"  L1-only:         {_stats(l1_map)['min_db']:.2f}~{_stats(l1_map)['max_db']:.2f} dB")
    print(f"  L2 raw (25.6km): {_stats(l2_raw)['min_db']:.2f}~{_stats(l2_raw)['max_db']:.2f} dB  nonzero={_stats(l2_raw)['nonzero_pixels']}")
    print(f"  L1+L2 composite: {_stats(composite_l1l2)['min_db']:.2f}~{_stats(composite_l1l2)['max_db']:.2f} dB")
    print(f"  L1+L2+L3:        {_stats(composite_full)['min_db']:.2f}~{_stats(composite_full)['max_db']:.2f} dB")
    print(f"\nFallbacks recorded: {len(l1_fallbacks)}")
    for fallback_msg in l1_fallbacks:
        print(f"  - {fallback_msg}")
    print(f"\nSpatial delta (L2 SW-corner bug):")
    print(f"  SW corner = ({origin_lat:.4f}N, {origin_lon:.4f}E)")
    print(f"  Center after fix = ({origin_lat + delta_lat_deg:.4f}N, {origin_lon + delta_lon_deg:.4f}E)")
    print(f"  Delta: +{delta_lat_deg:.4f} deg lat, +{delta_lon_deg:.4f} deg lon (~{half_km:.1f} km each)")


def main():
    parser = argparse.ArgumentParser(description="Capture golden scenes for regression testing")
    parser.add_argument("--config", default="configs/mission_config.yaml")
    parser.add_argument("--output", default="benchmarks/golden_scenes")
    args = parser.parse_args()
    capture(args.config, Path(args.output))


if __name__ == "__main__":
    main()
