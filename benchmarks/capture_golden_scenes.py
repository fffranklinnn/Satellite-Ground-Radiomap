#!/usr/bin/env python3
"""
Capture golden scenes for regression testing.

Saves L1-only, L1+L2, and L1+L2+L3 frame outputs with serialized arrays,
config snapshot, and metadata for use as regression baselines.

Usage:
    python benchmarks/capture_golden_scenes.py [--config CONFIG] [--output OUTPUT_DIR]
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext


def _hash_file(path: Path) -> str:
    """SHA-256 hash of a file, or 'missing' if not found."""
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_config(config: dict) -> str:
    """Deterministic SHA-256 of the config dict."""
    serialized = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialized).hexdigest()


def capture(config_path: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config_hash = _hash_config(config)
    capture_time = datetime.now(timezone.utc).isoformat()

    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]

    # Use the first time step as the golden frame
    frame_time = datetime.fromisoformat(config["time"]["start"])

    # Build LayerContext for L3 incident direction
    l3_cfg = config["layers"].get("l3_urban", {})
    incident_dir_cfg = l3_cfg.get("incident_dir")
    context = LayerContext.from_any({"incident_dir": incident_dir_cfg}) if incident_dir_cfg else None

    # --- L1 only ---
    print("Capturing L1-only scene...")
    l1_cfg = config["layers"]["l1_macro"]
    l1 = L1MacroLayer(l1_cfg, origin_lat, origin_lon)
    t0 = time.perf_counter()
    l1_map = l1.compute(origin_lat, origin_lon, frame_time, context)
    l1_elapsed = time.perf_counter() - t0
    np.save(output_dir / "l1_only.npy", l1_map)
    print(f"  L1 shape={l1_map.shape} min={l1_map.min():.2f} max={l1_map.max():.2f} ({l1_elapsed:.2f}s)")

    # --- L1 + L2 ---
    print("Capturing L1+L2 scene...")
    l2_cfg = config["layers"]["l2_topo"]
    l2 = L2TopoLayer(l2_cfg, origin_lat, origin_lon)
    agg_l1l2 = RadioMapAggregator(l1, l2, None)
    t0 = time.perf_counter()
    contrib_l1l2 = agg_l1l2.get_layer_contributions(origin_lat, origin_lon, frame_time, context)
    l1l2_elapsed = time.perf_counter() - t0
    composite_l1l2 = contrib_l1l2["composite"]
    np.save(output_dir / "l1l2_composite.npy", composite_l1l2)
    np.save(output_dir / "l1l2_l1.npy", contrib_l1l2.get("l1", np.array([])))
    np.save(output_dir / "l1l2_l2.npy", contrib_l1l2.get("l2", np.array([])))
    print(f"  L1+L2 composite shape={composite_l1l2.shape} min={composite_l1l2.min():.2f} max={composite_l1l2.max():.2f} ({l1l2_elapsed:.2f}s)")

    # --- L1 + L2 + L3 ---
    print("Capturing L1+L2+L3 scene...")
    l3 = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)
    agg_full = RadioMapAggregator(l1, l2, l3)
    t0 = time.perf_counter()
    contrib_full = agg_full.get_layer_contributions(origin_lat, origin_lon, frame_time, context)
    full_elapsed = time.perf_counter() - t0
    composite_full = contrib_full["composite"]
    np.save(output_dir / "l1l2l3_composite.npy", composite_full)
    np.save(output_dir / "l1l2l3_l1.npy", contrib_full.get("l1", np.array([])))
    np.save(output_dir / "l1l2l3_l2.npy", contrib_full.get("l2", np.array([])))
    np.save(output_dir / "l1l2l3_l3.npy", contrib_full.get("l3", np.array([])))
    print(f"  L1+L2+L3 composite shape={composite_full.shape} min={composite_full.min():.2f} max={composite_full.max():.2f} ({full_elapsed:.2f}s)")

    # --- Collect data file hashes ---
    data_files = {}
    for key in ["ionex_file", "era5_file", "tle_file"]:
        val = l1_cfg.get(key)
        if val:
            p = Path(val)
            data_files[key] = {"path": str(p), "sha256": _hash_file(p)}
    dem_path = Path(l2_cfg.get("dem_file", ""))
    data_files["dem_file"] = {"path": str(dem_path), "sha256": _hash_file(dem_path)}

    # --- Spatial delta note (L2 SW-corner vs center) ---
    # L2 currently interprets origin as SW corner; L1/L3 interpret as center.
    # For coverage_km=25.6, the center offset is approximately:
    #   delta_lat ≈ 25600 / 2 / 111000 ≈ 0.1153 degrees north
    #   delta_lon ≈ 25600 / 2 / (111000 * cos(lat)) degrees east
    import math
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
        "l2_current_sw_corner": {"lat": origin_lat, "lon": origin_lon},
        "l2_expected_center_after_fix": {
            "lat": origin_lat + delta_lat_deg,
            "lon": origin_lon + delta_lon_deg,
        },
        "delta_lat_deg": delta_lat_deg,
        "delta_lon_deg": delta_lon_deg,
        "coverage_km": coverage_km,
    }

    # --- Write manifest ---
    manifest = {
        "capture_time_utc": capture_time,
        "config_hash": config_hash,
        "config_path": str(Path(config_path).resolve()),
        "frame_time": str(frame_time),
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "data_files": data_files,
        "scenes": {
            "l1_only": {
                "file": "l1_only.npy",
                "shape": list(l1_map.shape),
                "min_db": float(l1_map.min()),
                "max_db": float(l1_map.max()),
                "mean_db": float(l1_map.mean()),
                "elapsed_s": l1_elapsed,
            },
            "l1l2_composite": {
                "file": "l1l2_composite.npy",
                "shape": list(composite_l1l2.shape),
                "min_db": float(composite_l1l2.min()),
                "max_db": float(composite_l1l2.max()),
                "mean_db": float(composite_l1l2.mean()),
                "elapsed_s": l1l2_elapsed,
            },
            "l1l2l3_composite": {
                "file": "l1l2l3_composite.npy",
                "shape": list(composite_full.shape),
                "min_db": float(composite_full.min()),
                "max_db": float(composite_full.max()),
                "mean_db": float(composite_full.mean()),
                "elapsed_s": full_elapsed,
            },
        },
        "spatial_delta_note": spatial_delta,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Also save the config snapshot
    config_snapshot_path = output_dir / "config_snapshot.yaml"
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nGolden scenes saved to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Config hash: {config_hash}")
    print(f"\nSpatial delta (L2 SW-corner bug):")
    print(f"  Current L2 SW corner = ({origin_lat:.4f}N, {origin_lon:.4f}E)")
    print(f"  Expected center after fix = ({origin_lat + delta_lat_deg:.4f}N, {origin_lon + delta_lon_deg:.4f}E)")
    print(f"  Delta: +{delta_lat_deg:.4f} deg lat, +{delta_lon_deg:.4f} deg lon (~{half_km:.1f} km each)")


def main():
    parser = argparse.ArgumentParser(description="Capture golden scenes for regression testing")
    parser.add_argument("--config", default="configs/mission_config.yaml")
    parser.add_argument("--output", default="benchmarks/golden_scenes")
    args = parser.parse_args()
    capture(args.config, Path(args.output))


if __name__ == "__main__":
    main()
