#!/usr/bin/env python3
"""
Generate multi-satellite long-timeseries full-physics radiomap frames.

Per-frame pipeline:
  1) Enumerate visible satellites at timestamp t
  2) Compute L1/L2/L3 maps for each satellite
  3) Fuse multi-satellite maps by selected strategy
  4) Export PNG + NPY + frame JSON and append manifest.jsonl

Fixed output structure:
  <output-dir>/png/*.png
  <output-dir>/npy/*.npy
  <output-dir>/frame_json/*.json
  <output-dir>/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.layers.base import LayerContext
from src.context import GridSpec, CoverageSpec, FrameBuilder
from src.context.multiscale_map import MultiScaleMap
from src.context.time_utils import parse_iso_utc  # shared strict UTC helper
from src.products.manifest import ProductManifest, collect_input_file_paths
from src.products.projectors import export_dataset
from src.pipeline.manifest_writer import ManifestWriter
from src.utils import plot_radio_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate long-timeseries radiomap frames with multi-satellite fusion "
            "(best-server or soft-combine)."
        )
    )
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml",
                        help="Path to mission config YAML")
    parser.add_argument("--start", type=str, default=None,
                        help="Start timestamp (ISO). Defaults to config.time.start")
    parser.add_argument("--end", type=str, default=None,
                        help="End timestamp (ISO). Defaults to config.time.end")
    parser.add_argument("--step-minutes", type=float, default=None,
                        help="Frame step in minutes. Defaults to config.time.step_hours * 60")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional frame cap for debug runs")
    parser.add_argument("--origin-lat", type=float, default=None,
                        help="Override origin latitude")
    parser.add_argument("--origin-lon", type=float, default=None,
                        help="Override origin longitude")

    parser.add_argument("--fusion-mode", type=str, choices=["best-server", "soft-combine"],
                        default="best-server",
                        help="Multi-satellite fusion strategy")
    parser.add_argument("--max-satellites", type=int, default=8,
                        help="Max visible satellites per frame (sorted by elevation)")
    parser.add_argument("--min-elevation-deg", type=float, default=None,
                        help="Minimum elevation threshold for visible-satellite filtering")
    parser.add_argument("--soft-min-power", type=float, default=1e-30,
                        help="Power floor for soft-combine denominator stability")
    parser.add_argument("--norad-id", action="append", default=None,
                        help="Restrict candidate satellites to NORAD ID; can be repeated")

    parser.add_argument("--output-dir", type=str, default="output/multisat_timeseries_radiomap",
                        help="Output root directory")
    parser.add_argument("--output-prefix", type=str, default="multisat_ts",
                        help="Frame output prefix")
    parser.add_argument("--dpi", type=int, default=180,
                        help="PNG DPI")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap for output PNG")
    parser.add_argument("--allow-missing-data", action="store_true",
                        help="Allow missing optional/required data where fallback exists")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def build_frame_builder_for_script(config: dict, origin_lat: float, origin_lon: float) -> FrameBuilder:
    """Build a FrameBuilder from config and resolved origin coordinates."""
    l1_cfg = config.get("layers", {}).get("l1_macro", {})
    coarse_km = float(l1_cfg.get("coverage_km", 256.0))
    grid_size = int(l1_cfg.get("grid_size", 256))
    grid = GridSpec.from_legacy_args(origin_lat, origin_lon, coarse_km, grid_size, grid_size)
    product_km = float(config.get("product", {}).get("coverage_km", 0.256))
    product_nx = int(config.get("product", {}).get("grid_size", 256))
    coverage = CoverageSpec.from_config(
        origin_lat=origin_lat, origin_lon=origin_lon,
        coarse_coverage_km=coarse_km, coarse_nx=grid_size, coarse_ny=grid_size,
        product_coverage_km=product_km, product_nx=product_nx, product_ny=product_nx,
    )
    return FrameBuilder(grid=grid, coverage=coverage)


def build_time_grid(start: datetime,
                    end: datetime,
                    step_minutes: float,
                    max_frames: Optional[int] = None) -> List[datetime]:
    if step_minutes <= 0:
        raise ValueError("--step-minutes must be > 0.")
    if end < start:
        raise ValueError("--end must be later than or equal to --start.")

    out: List[datetime] = []
    step = timedelta(minutes=float(step_minutes))
    t = start
    while t <= end:
        out.append(t)
        if max_frames is not None and len(out) >= int(max_frames):
            break
        t += step
    return out


def check_required_data(project_root: Path,
                        config: Dict[str, Any],
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

    missing_required: List[str] = []
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


def array_stats(arr: np.ndarray) -> Dict[str, Any]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p99": None,
        }
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "p01": float(np.percentile(valid, 1)),
        "p99": float(np.percentile(valid, 99)),
    }


def fmt_seconds(seconds: float) -> str:
    sec_i = int(max(seconds, 0.0))
    hh = sec_i // 3600
    mm = (sec_i % 3600) // 60
    ss = sec_i % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def frame_stamp(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compute_satellite_maps(
    l1_layer: L1MacroLayer,
    l2_layer: L2TopoLayer,
    l3_layer: L3UrbanLayer,
    frame_builder: FrameBuilder,
    timestamp: datetime,
    norad_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], "FrameContext"]:
    """
    Compute per-satellite maps using the FrameContext pipeline.

    Returns (l1_map, l2_map, l3_map, total_map, sat_info, frame).
    The frame is pre-bound with satellite geometry and can be reused for export.

    Uses SatelliteSelector to pre-bind satellite geometry before frame build,
    matching the canonical contract in main.py and BenchmarkRunner.
    """
    from src.planning.satellite_selector import SatelliteSelector
    tle_cfg = l1_layer.config.get("tle", {})
    tle_path = (tle_cfg.get("file") if isinstance(tle_cfg, dict) else None) or l1_layer.config.get("tle_file", "")
    selector = SatelliteSelector(str(tle_path), strict=False, min_elevation_deg=-90.0)
    sat_info = selector.select(
        timestamp=timestamp,
        center=(frame_builder.grid.center_lat, frame_builder.grid.center_lon),
        target_ids=[str(norad_id)],
        strict=False,
    )
    frame = frame_builder.build(timestamp, sat_info=sat_info)

    # L1: propagate_entry uses frame-bound satellite geometry
    entry = l1_layer.propagate_entry(frame)
    sat = {
        "norad_id": entry.norad_id or norad_id,
        "lat_deg": entry.sat_lat_deg,
        "lon_deg": entry.sat_lon_deg,
        "alt_m": entry.sat_alt_m,
        "azimuth_deg": float(entry.azimuth_deg[entry.native_grid.ny // 2, entry.native_grid.nx // 2]),
        "elevation_deg": float(entry.elevation_deg[entry.native_grid.ny // 2, entry.native_grid.nx // 2]),
        "slant_range_m": float(entry.slant_range_m[entry.native_grid.ny // 2, entry.native_grid.nx // 2]),
    }

    # L2: propagate_terrain uses frame.grid.sw_corner() — no manual extras injection
    terrain = l2_layer.propagate_terrain(frame, entry=entry)

    # L3: refine_urban derives incident_dir from entry state
    urban = l3_layer.refine_urban(frame, entry=entry)

    l1_map = entry.total_loss_db
    l2_map = terrain.loss_db
    l3_map = urban.urban_residual_db
    _cov = object.__getattribute__(frame, "coverage")
    _product_grid = _cov.product_grid if _cov is not None else object.__getattribute__(frame, "grid")
    from src.compose import project_to_product_grid
    projected = project_to_product_grid(
        product_grid=_product_grid, entry=entry, terrain=terrain, urban=urban,
        frame_id=frame.frame_id,
    )
    msm = MultiScaleMap.compose_projected(
        frame_id=frame.frame_id,
        product_grid=_product_grid,
        **projected,
    )
    total_map = msm.composite_db
    return l1_map, l2_map, l3_map, total_map, sat, frame


def satellite_metadata(sat_info: Dict[str, Any],
                       l1_layer: L1MacroLayer,
                       rank: int) -> Dict[str, Any]:
    return {
        "rank": int(rank),
        "norad_id": str(sat_info.get("norad_id", "")),
        "azimuth_deg": float(sat_info.get("azimuth_deg", float("nan"))),
        "elevation_deg": float(sat_info.get("elevation_deg", float("nan"))),
        "slant_range_km": float(sat_info.get("slant_range_m", float("nan"))) / 1000.0,
        "lat_deg": float(sat_info.get("lat_deg", float("nan"))),
        "lon_deg": float(sat_info.get("lon_deg", float("nan"))),
        "alt_km": float(sat_info.get("alt_m", 0.0) / 1000.0),
        "beam": {
            "peak_gain_dbi": float(l1_layer.peak_gain_db),
            "hpbw_az_deg": float(l1_layer.hpbw_az_deg),
            "hpbw_el_deg": float(l1_layer.hpbw_el_deg),
        },
    }


def make_output_dirs(output_root: Path) -> Dict[str, Path]:
    png_dir = output_root / "png"
    npy_dir = output_root / "npy"
    json_dir = output_root / "frame_json"
    output_root.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_root,
        "png": png_dir,
        "npy": npy_dir,
        "json": json_dir,
        "manifest": output_root / "manifest.jsonl",
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)
    check_required_data(project_root, config, allow_missing=args.allow_missing_data)

    layers_cfg = config["layers"]
    l1_cfg = layers_cfg["l1_macro"]
    l2_cfg = layers_cfg["l2_topo"]
    l3_cfg = layers_cfg["l3_urban"]

    origin_cfg = config.get("origin", {})
    origin_lat = float(args.origin_lat if args.origin_lat is not None else origin_cfg.get("latitude", 34.3416))
    origin_lon = float(args.origin_lon if args.origin_lon is not None else origin_cfg.get("longitude", 108.9398))

    time_cfg = config.get("time", {})
    start_text = args.start if args.start is not None else str(time_cfg.get("start", "2025-01-01T00:00:00"))
    end_text = args.end if args.end is not None else str(time_cfg.get("end", start_text))
    if args.step_minutes is not None:
        step_minutes = float(args.step_minutes)
    else:
        step_hours = float(time_cfg.get("step_hours", 1.0))
        step_minutes = step_hours * 60.0

    start_ts = parse_iso_utc(start_text)
    end_ts = parse_iso_utc(end_text)
    timestamps = build_time_grid(
        start=start_ts,
        end=end_ts,
        step_minutes=step_minutes,
        max_frames=args.max_frames,
    )
    if not timestamps:
        raise RuntimeError("No frames generated. Check time range and step.")

    target_norad_ids: Optional[List[str]] = None
    if args.norad_id:
        target_norad_ids = [str(x).strip() for x in args.norad_id if str(x).strip()]
        if not target_norad_ids:
            target_norad_ids = None

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = project_root / out_root
    out_dirs = make_output_dirs(out_root)

    l1_layer = L1MacroLayer(l1_cfg, origin_lat, origin_lon)
    l2_layer = L2TopoLayer(l2_cfg, origin_lat, origin_lon)
    l3_layer = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)
    frame_builder = build_frame_builder_for_script(config, origin_lat, origin_lon)
    input_file_paths = collect_input_file_paths(config)

    if target_norad_ids:
        l1_layer.target_norad_ids = target_norad_ids

    try:
        total_frames = len(timestamps)
        print("=" * 88)
        print("Multi-satellite time-series radiomap generation")
        print(f"  time range     : {timestamps[0].isoformat()} -> {timestamps[-1].isoformat()}")
        print(f"  frame count    : {total_frames}")
        print(f"  step_minutes   : {step_minutes}")
        print(f"  origin         : ({origin_lat:.6f}, {origin_lon:.6f})")
        print(f"  fusion_mode    : {args.fusion_mode}")
        print(f"  max_satellites : {args.max_satellites}")
        print(f"  min_elevation  : {args.min_elevation_deg if args.min_elevation_deg is not None else 'default L1'}")
        print(f"  norad_filter   : {target_norad_ids if target_norad_ids else 'none'}")
        print(f"  output_root    : {out_root}")
        print("=" * 88)

        # Precompute visible-satellite plan for predictable progress reporting.
        visible_plan: List[List[Dict[str, float]]] = []
        pre_t0 = time.time()
        visible_accum = 0
        for idx, ts in enumerate(timestamps, start=1):
            visible = l1_layer.get_visible_satellites(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                timestamp=ts,
                min_elevation_deg=args.min_elevation_deg,
                target_norad_ids=target_norad_ids,
                max_count=args.max_satellites,
            )
            visible_plan.append(visible)
            visible_accum += len(visible)
            if idx == 1 or idx % 10 == 0 or idx == total_frames:
                elapsed = time.time() - pre_t0
                rate = idx / max(elapsed, 1e-6)
                print(
                    f"[prepass] {idx}/{total_frames} frames | "
                    f"avg_visible={visible_accum / idx:.2f} | "
                    f"rate={rate:.2f}/s"
                )

        total_sat_tasks = int(sum(len(v) for v in visible_plan))
        done_sat_tasks = 0
        run_t0 = time.time()
        status_counts: Dict[str, int] = {}

        with ManifestWriter(out_dirs["manifest"]) as manifest_writer:
            for frame_idx, (ts, visible) in enumerate(zip(timestamps, visible_plan)):
                frame_t0 = time.time()
                stamp = frame_stamp(ts)
                base = f"{args.output_prefix}_{frame_idx:06d}_{stamp}"

                png_rel = Path("png") / f"{base}.png"
                npy_rel = Path("npy") / f"{base}.npy"
                json_rel = Path("frame_json") / f"{base}.json"

                visible_count = len(visible)
                print(
                    f"[frame {frame_idx + 1}/{total_frames}] {stamp} | "
                    f"visible={visible_count}"
                )

                sat_entries: List[Dict[str, Any]] = []
                sat_frames: List[Any] = []  # store pre-bound FrameContext per satellite
                sat_errors: List[Dict[str, Any]] = []
                status = "ok"
                l1_layer.clear_fallbacks()

                target_size = frame_builder.grid.nx
                default_l1 = np.full((target_size, target_size), l1_layer.NO_COVERAGE_LOSS_DB, dtype=np.float32)
                l1_fused = default_l1.copy()
                l2_fused = np.zeros((target_size, target_size), dtype=np.float32)
                l3_fused = np.zeros((target_size, target_size), dtype=np.float32)
                composite = default_l1.copy()

                if visible_count == 0:
                    status = "no_visible_satellite"
                else:
                    if args.fusion_mode == "best-server":
                        best_total: Optional[np.ndarray] = None
                        best_l1: Optional[np.ndarray] = None
                        best_l2: Optional[np.ndarray] = None
                        best_l3: Optional[np.ndarray] = None
                        best_idx: Optional[np.ndarray] = None
                    else:
                        power_sum = np.zeros((target_size, target_size), dtype=np.float64)
                        weighted_l1 = np.zeros((target_size, target_size), dtype=np.float64)
                        weighted_l2 = np.zeros((target_size, target_size), dtype=np.float64)
                        weighted_l3 = np.zeros((target_size, target_size), dtype=np.float64)
                        sat_power_totals: List[float] = []

                    for sat_rank, sat_pred in enumerate(visible, start=1):
                        sat_t0 = time.time()
                        norad_id = str(sat_pred.get("norad_id", "")).strip()
                        if not norad_id:
                            sat_errors.append({
                                "rank": sat_rank,
                                "norad_id": "",
                                "error": "missing NORAD ID in visible list",
                            })
                            done_sat_tasks += 1
                            continue

                        try:
                            l1_map, l2_map, l3_map, total_map, sat_info, sat_frame = compute_satellite_maps(
                                l1_layer=l1_layer,
                                l2_layer=l2_layer,
                                l3_layer=l3_layer,
                                frame_builder=frame_builder,
                                timestamp=ts,
                                norad_id=norad_id,
                            )
                            sat_meta = satellite_metadata(sat_info, l1_layer=l1_layer, rank=sat_rank)
                            sat_meta["compute_sec"] = round(time.time() - sat_t0, 3)
                            sat_meta["sat_total_mean_db"] = float(np.nanmean(total_map))
                            sat_entries.append(sat_meta)
                            sat_frames.append(sat_frame)

                            if args.fusion_mode == "best-server":
                                sat_idx = len(sat_entries) - 1
                                if best_total is None:
                                    best_total = total_map.copy()
                                    best_l1 = l1_map.copy()
                                    best_l2 = l2_map.copy()
                                    best_l3 = l3_map.copy()
                                    best_idx = np.full(total_map.shape, sat_idx, dtype=np.int32)
                                else:
                                    improved = total_map < best_total
                                    best_total = np.where(improved, total_map, best_total)
                                    best_l1 = np.where(improved, l1_map, best_l1)
                                    best_l2 = np.where(improved, l2_map, best_l2)
                                    best_l3 = np.where(improved, l3_map, best_l3)
                                    best_idx = np.where(improved, sat_idx, best_idx)
                            else:
                                power = np.power(10.0, -np.asarray(total_map, dtype=np.float64) / 10.0)
                                power_sum += power
                                weighted_l1 += power * l1_map
                                weighted_l2 += power * l2_map
                                weighted_l3 += power * l3_map
                                sat_power_totals.append(float(np.sum(power)))

                        except Exception as exc:
                            sat_errors.append({
                                "rank": sat_rank,
                                "norad_id": norad_id,
                                "error": str(exc),
                            })

                        done_sat_tasks += 1
                        if sat_rank == 1 or sat_rank == visible_count or sat_rank % 5 == 0:
                            print(
                                f"  [sat {sat_rank}/{visible_count}] norad={norad_id} "
                                f"| ok={len(sat_entries)} err={len(sat_errors)}"
                            )

                    if not sat_entries:
                        status = "satellite_compute_failed"
                    elif args.fusion_mode == "best-server":
                        assert best_total is not None and best_l1 is not None
                        assert best_l2 is not None and best_l3 is not None and best_idx is not None
                        composite = best_total.astype(np.float32, copy=False)
                        l1_fused = best_l1.astype(np.float32, copy=False)
                        l2_fused = best_l2.astype(np.float32, copy=False)
                        l3_fused = best_l3.astype(np.float32, copy=False)

                        counts = np.bincount(best_idx.ravel(), minlength=len(sat_entries))
                        total_pixels = int(best_idx.size)
                        for i, cnt in enumerate(counts.tolist()):
                            sat_entries[i]["best_server_pixel_count"] = int(cnt)
                            sat_entries[i]["best_server_pixel_share"] = float(cnt / max(total_pixels, 1))
                    else:
                        denom = np.maximum(power_sum, float(args.soft_min_power))
                        composite = (-10.0 * np.log10(denom)).astype(np.float32, copy=False)
                        l1_fused = (weighted_l1 / denom).astype(np.float32, copy=False)
                        l2_fused = (weighted_l2 / denom).astype(np.float32, copy=False)
                        l3_fused = (weighted_l3 / denom).astype(np.float32, copy=False)

                        total_power = float(np.sum(sat_power_totals))
                        for i, pwr in enumerate(sat_power_totals):
                            sat_entries[i]["soft_power_share"] = (
                                float(pwr / total_power) if total_power > 0.0 else 0.0
                            )

                png_path = out_dirs["png"] / f"{base}.png"
                frame_json_path = out_dirs["json"] / f"{base}.json"

                # Build per-frame manifest and export npy via export_dataset()
                frame_manifest = ProductManifest.build(
                    frame_id=base,
                    timestamp_utc=ts.isoformat(),
                    config=config,
                    data_snapshot_id=config.get("data_validation", {}).get("snapshot_id", ""),
                    input_files=input_file_paths,
                    hash_files=True,
                    fallbacks_used=l1_layer.fallbacks_used,
                )
                # Reuse the first satellite's pre-bound frame for export
                export_frame = sat_frames[0] if sat_frames else frame_builder.build(ts, frame_id=base)
                written, _ = export_dataset(
                    output_dir=out_dirs["npy"],
                    frame=export_frame,
                    product_types=["path_loss_map"],
                    multiscale=MultiScaleMap(
                        frame_id=base,
                        grid=frame_builder.grid,
                        composite_db=composite,
                        l1_db=l1_fused,
                        l2_db=l2_fused,
                        l3_db=l3_fused,
                    ),
                    manifest=frame_manifest,
                    prefix=f"{base}_",
                    manifest_writer=manifest_writer,
                )
                npy_path = Path(written["path_loss_map"])
                npy_rel = npy_path.relative_to(out_root)

                plot_radio_map(
                    loss_map=composite,
                    title=(
                        "Multi-Satellite Time-Series Radiomap\n"
                        f"{ts.strftime('%Y-%m-%d %H:%M:%S UTC')} | "
                        f"fusion={args.fusion_mode} | sats={len(sat_entries)}"
                    ),
                    output_file=str(png_path),
                    cmap=args.cmap,
                    show_colorbar=True,
                    show_stats=True,
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    coverage_km=0.256,
                    dpi=args.dpi,
                )

                frame_meta: Dict[str, Any] = {
                    "frame_index": int(frame_idx),
                    "timestamp_utc": ts.isoformat(),
                    "status": status,
                    "origin": {
                        "lat": origin_lat,
                        "lon": origin_lon,
                    },
                    "fusion": {
                        "mode": args.fusion_mode,
                        "max_satellites": int(args.max_satellites),
                        "min_elevation_deg": args.min_elevation_deg,
                        "soft_min_power": float(args.soft_min_power),
                    },
                    "satellite_count_visible": int(visible_count),
                    "satellite_count_used": int(len(sat_entries)),
                    "satellites": sat_entries,
                    "satellite_errors": sat_errors,
                    "layer_stats_db": {
                        "l1": array_stats(l1_fused),
                        "l2": array_stats(l2_fused),
                        "l3": array_stats(l3_fused),
                        "composite": array_stats(composite),
                    },
                    "artifacts": {
                        "png": str(png_rel),
                        "npy": str(npy_rel),
                        "frame_json": str(json_rel),
                    },
                    "runtime_sec": round(time.time() - frame_t0, 3),
                }
                frame_json_path.write_text(
                    json.dumps(frame_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                status_counts[status] = status_counts.get(status, 0) + 1
                done_frames = frame_idx + 1
                elapsed = max(time.time() - run_t0, 1e-6)
                frame_rate = done_frames / elapsed
                eta = (total_frames - done_frames) / max(frame_rate, 1e-12)

                sat_prog_text = "N/A"
                if total_sat_tasks > 0:
                    sat_pct = 100.0 * done_sat_tasks / total_sat_tasks
                    sat_prog_text = f"{done_sat_tasks}/{total_sat_tasks} ({sat_pct:.1f}%)"

                print(
                    f"[done {done_frames}/{total_frames}] status={status} "
                    f"| sats_used={len(sat_entries)}/{visible_count} "
                    f"| sat_prog={sat_prog_text} "
                    f"| frame_dt={frame_meta['runtime_sec']}s "
                    f"| eta={fmt_seconds(eta)}"
                )

        print("=" * 88)
        print("Generation finished")
        print(f"  output_root   : {out_root}")
        print(f"  manifest_jsonl: {out_dirs['manifest']}")
        print(f"  status_counts : {status_counts}")
        print(f"  elapsed       : {fmt_seconds(time.time() - run_t0)}")
        print("=" * 88)
    finally:
        l2_layer.close()


if __name__ == "__main__":
    main()
