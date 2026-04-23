#!/usr/bin/env python3
"""
Generate categorized showcase figures for SG-MRM capabilities.

Outputs are organized into:
  - terrain/     : DEM loading, terrain occlusion, L2 terrain loss
  - atmosphere/  : ERA5 IWV loading and atmospheric attenuation maps
  - radiomap/    : 1..K satellites at a fixed region/frequency
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.physics import atmospheric_loss, atmospheric_loss_era5
from src.engine import RadioMapAggregator
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.layers.base import LayerContext


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate showcase figures by category.")
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml")
    parser.add_argument("--timestamp", type=str, default="2025-01-01T06:00:00",
                        help="ISO timestamp")
    parser.add_argument("--origin-lat", type=float, default=None,
                        help="Override origin latitude")
    parser.add_argument("--origin-lon", type=float, default=None,
                        help="Override origin longitude")
    parser.add_argument("--frequency-ghz", type=float, default=None,
                        help="Override operating frequency (GHz)")

    parser.add_argument("--rain-rates", type=str, default="0,10,25",
                        help='Rain rates for atmosphere maps, e.g. "0,10,25"')
    parser.add_argument("--top-k-sats", type=int, default=3,
                        help="Number of visible satellites to render in radiomap category")
    parser.add_argument("--min-elevation-deg", type=float, default=10.0,
                        help="Satellite visibility filter")
    parser.add_argument("--terrain-elevation-deg", type=float, default=None,
                        help="Override terrain-demo elevation angle (deg)")
    parser.add_argument("--terrain-azimuth-deg", type=float, default=None,
                        help="Override terrain-demo azimuth angle (deg)")

    parser.add_argument("--output-root", type=str, default="output/feature_showcase",
                        help="Root folder containing terrain/atmosphere/radiomap")
    parser.add_argument("--output-tag", type=str, default="xian_demo",
                        help="Tag used in output filenames")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_ts(ts: str) -> datetime:
    from src.context.time_utils import parse_iso_utc
    return parse_iso_utc(ts, strict=False)


def _parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for part in text.split(","):
        token = part.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError("At least one value is required.")
    return vals


def _geo_ticks(ax: plt.Axes,
               nrows: int,
               ncols: int,
               origin_lat: float,
               origin_lon: float,
               coverage_km: float,
               nticks: int = 6) -> None:
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    lon_vals = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, nticks)
    lat_vals = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, nticks)
    x_ticks = np.linspace(0, ncols - 1, nticks)
    y_ticks = np.linspace(0, nrows - 1, nticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x:.3f}E" for x in lon_vals], fontsize=8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.3f}N" for y in lat_vals], fontsize=8)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)


def _save_map(arr: np.ndarray,
              out_path: Path,
              title: str,
              cbar_label: str,
              cmap: str,
              origin_lat: float,
              origin_lon: float,
              coverage_km: float,
              dpi: int,
              vmin: Optional[float] = None,
              vmax: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    vvmin = float(np.nanpercentile(arr, 1)) if vmin is None else vmin
    vvmax = float(np.nanpercentile(arr, 99)) if vmax is None else vmax
    im = ax.imshow(arr, origin="upper", cmap=cmap, interpolation="nearest", vmin=vvmin, vmax=vvmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=16)
    _geo_ticks(ax, arr.shape[0], arr.shape[1], origin_lat, origin_lon, coverage_km)
    ax.set_title(title, fontsize=12)

    stats = (
        f"mean={float(np.nanmean(arr)):.3f}\n"
        f"std={float(np.nanstd(arr)):.3f}\n"
        f"min={float(np.nanmin(arr)):.3f}\n"
        f"max={float(np.nanmax(arr)):.3f}"
    )
    ax.text(
        0.012, 0.012, stats,
        transform=ax.transAxes,
        fontsize=8,
        fontfamily="monospace",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def _save_panel(items: List[Tuple[str, np.ndarray, str]],
                out_path: Path,
                panel_title: str,
                origin_lat: float,
                origin_lon: float,
                coverage_km: float,
                dpi: int) -> None:
    n = len(items)
    if n == 0:
        return
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8.8 * cols, 6.8 * rows))
    if isinstance(axes, np.ndarray):
        axs = axes.ravel()
    else:
        axs = np.array([axes])

    for i, (title, arr, cmap) in enumerate(items):
        ax = axs[i]
        vmin = float(np.nanpercentile(arr, 1))
        vmax = float(np.nanpercentile(arr, 99))
        im = ax.imshow(arr, origin="upper", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        _geo_ticks(ax, arr.shape[0], arr.shape[1], origin_lat, origin_lon, coverage_km)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    fig.suptitle(panel_title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def _terrain_category(l2: L2TopoLayer,
                      ts: datetime,
                      origin_lat: float,
                      origin_lon: float,
                      sat_el_deg: float,
                      sat_az_deg: float,
                      out_dir: Path,
                      tag: str,
                      dpi: int) -> Dict[str, Any]:
    print("[1/3] Terrain category")
    out_dir.mkdir(parents=True, exist_ok=True)

    dem = l2._load_dem_patch(origin_lat, origin_lon)
    occ_mask, excess_h, obs_dist = l2._compute_occlusion_vectorized(
        dem,
        sat_elevation_deg=sat_el_deg,
        sat_azimuth_deg=sat_az_deg,
        return_profile=True,
    )
    l2_loss = l2.compute(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=ts,
        context=LayerContext.from_any({
            "satellite_elevation_deg": sat_el_deg,
            "satellite_azimuth_deg": sat_az_deg,
        }),
    )

    dem_path = out_dir / f"{tag}_terrain_dem_loaded.png"
    occ_path = out_dir / f"{tag}_terrain_occlusion_mask.png"
    loss_path = out_dir / f"{tag}_terrain_l2_loss.png"
    panel_path = out_dir / f"{tag}_terrain_panel.png"

    _save_map(dem, dem_path, "Terrain DEM Loaded (L2)", "Elevation (m)", "terrain",
              origin_lat, origin_lon, 25.6, dpi)
    _save_map(occ_mask.astype(np.float32), occ_path, "Terrain Occlusion Mask", "Occluded (1/0)", "gray",
              origin_lat, origin_lon, 25.6, dpi, vmin=0.0, vmax=1.0)
    _save_map(l2_loss, loss_path, "Terrain Diffraction Loss (L2)", "Loss (dB)", "inferno",
              origin_lat, origin_lon, 25.6, dpi)
    _save_panel(
        [
            ("DEM Elevation (m)", dem, "terrain"),
            ("Occlusion Mask", occ_mask.astype(np.float32), "gray"),
            ("L2 Diffraction Loss (dB)", l2_loss, "inferno"),
        ],
        panel_path,
        "Terrain Data Loading and L2 Computation",
        origin_lat,
        origin_lon,
        25.6,
        dpi,
    )

    np.save(out_dir / f"{tag}_terrain_dem.npy", dem.astype(np.float32))
    np.save(out_dir / f"{tag}_terrain_occlusion_mask.npy", occ_mask.astype(np.uint8))
    np.save(out_dir / f"{tag}_terrain_l2_loss.npy", l2_loss.astype(np.float32))

    return {
        "timestamp": ts.isoformat(),
        "satellite_elevation_deg": sat_el_deg,
        "satellite_azimuth_deg": sat_az_deg,
        "occlusion_ratio": float(np.mean(occ_mask)),
        "loss_mean_db": float(np.mean(l2_loss)),
        "loss_std_db": float(np.std(l2_loss)),
    }


def _atmosphere_category(l1: L1MacroLayer,
                         ts: datetime,
                         origin_lat: float,
                         origin_lon: float,
                         rain_rates: List[float],
                         out_dir: Path,
                         tag: str,
                         dpi: int) -> Dict[str, Any]:
    print("[2/3] Atmosphere category")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_comp = l1.compute_components(timestamp=ts, origin_lat=origin_lat, origin_lon=origin_lon)
    iwv = base_comp["iwv"].astype(np.float32)
    elevation = base_comp["elevation"].astype(np.float32)
    base_rain = float(rain_rates[0]) if rain_rates else 0.0

    iwv_path = out_dir / f"{tag}_atmosphere_era5_iwv.png"
    _save_map(
        iwv,
        iwv_path,
        "ERA5 IWV Loaded Map",
        "IWV (kg/m^2)",
        "Blues",
        origin_lat,
        origin_lon,
        256.0,
        dpi,
    )

    atm_maps: Dict[str, np.ndarray] = {}
    has_era5 = not np.isnan(iwv).all()
    for rr in rain_rates:
        if has_era5:
            atm = atmospheric_loss_era5(
                elevation_angle_deg=elevation,
                frequency_ghz=l1.frequency_ghz,
                iwv_kg_m2=iwv,
                rain_rate_mm_h=rr,
            ).astype(np.float32)
            suffix = "era5"
        else:
            atm = atmospheric_loss(
                elevation_angle_deg=elevation,
                frequency_ghz=l1.frequency_ghz,
                rain_rate_mm_h=rr,
            ).astype(np.float32)
            suffix = "fallback"
        key = f"r{str(rr).replace('.', 'p')}"
        atm_maps[key] = atm
        out_path = out_dir / f"{tag}_atmosphere_loss_{suffix}_{key}.png"
        _save_map(
            atm,
            out_path,
            f"Atmospheric Loss ({rr:.1f} mm/h)",
            "Loss (dB)",
            "viridis",
            origin_lat,
            origin_lon,
            256.0,
            dpi,
        )
        np.save(out_dir / f"{tag}_atmosphere_loss_{suffix}_{key}.npy", atm)

    if len(rain_rates) >= 2:
        base_key = f"r{str(base_rain).replace('.', 'p')}"
        base_atm = atm_maps[base_key]
        for rr in rain_rates[1:]:
            key = f"r{str(rr).replace('.', 'p')}"
            delta = atm_maps[key] - base_atm
            out_path = out_dir / f"{tag}_atmosphere_rain_delta_{key}_vs_{base_key}.png"
            _save_map(
                delta,
                out_path,
                f"Rain Impact Delta ({rr:.1f} - {base_rain:.1f} mm/h)",
                "Delta Loss (dB)",
                "magma",
                origin_lat,
                origin_lon,
                256.0,
                dpi,
            )
            np.save(out_dir / f"{tag}_atmosphere_rain_delta_{key}_vs_{base_key}.npy", delta.astype(np.float32))

    panel_items: List[Tuple[str, np.ndarray, str]] = [("ERA5 IWV (kg/m^2)", iwv, "Blues")]
    for rr in rain_rates[:3]:
        key = f"r{str(rr).replace('.', 'p')}"
        panel_items.append((f"Atm Loss {rr:.1f} mm/h", atm_maps[key], "viridis"))
    _save_panel(
        panel_items,
        out_dir / f"{tag}_atmosphere_panel.png",
        "Atmosphere Data Loading and Loss Module",
        origin_lat,
        origin_lon,
        256.0,
        dpi,
    )

    return {
        "timestamp": ts.isoformat(),
        "frequency_ghz": float(l1.frequency_ghz),
        "rain_rates": rain_rates,
        "era5_available": bool(has_era5),
        "iwv_mean": float(np.nanmean(iwv)),
        "iwv_std": float(np.nanstd(iwv)),
    }


def _radiomap_category(l1: L1MacroLayer,
                       l2: L2TopoLayer,
                       l3: Optional[L3UrbanLayer],
                       ts: datetime,
                       origin_lat: float,
                       origin_lon: float,
                       top_k_sats: int,
                       min_el_deg: float,
                       out_dir: Path,
                       tag: str,
                       dpi: int) -> Dict[str, Any]:
    print("[3/3] Radiomap category")
    out_dir.mkdir(parents=True, exist_ok=True)

    visible = l1.get_visible_satellites(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=ts,
        min_elevation_deg=min_el_deg,
        max_count=top_k_sats,
    )
    if not visible:
        raise RuntimeError("No visible satellites for requested timestamp/location.")

    aggregator = RadioMapAggregator(l1_layer=l1, l2_layer=l2, l3_layer=l3) if l3 is not None else RadioMapAggregator(l1_layer=l1, l2_layer=l2)

    sat_rows: List[Dict[str, Any]] = []
    panel_items: List[Tuple[str, np.ndarray, str]] = []
    for i, sat in enumerate(visible, start=1):
        norad = str(sat["norad_id"])
        print(f"  satellite {i}/{len(visible)}: NORAD {norad}")

        comp_l1 = l1.compute_components(
            timestamp=ts,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            target_norad_ids=[norad],
        )
        sat_info = comp_l1["satellite"]

        ctx = LayerContext.from_any({
            "incident_dir": {"az_deg": sat_info["azimuth_deg"], "el_deg": sat_info["elevation_deg"]},
            "satellite_azimuth_deg": sat_info["azimuth_deg"],
            "satellite_elevation_deg": sat_info["elevation_deg"],
            "satellite_slant_range_km": sat_info.get("slant_range_km"),
            "satellite_altitude_km": sat_info.get("alt_m", 0.0) / 1000.0,
            "target_norad_ids": [norad],
        })

        l2_map = l2.compute(origin_lat=origin_lat, origin_lon=origin_lon, timestamp=ts, context=ctx)
        l1_interp = aggregator._interpolate_to_target(comp_l1["total"], l1.coverage_km).astype(np.float32)
        l2_interp = aggregator._interpolate_to_target(l2_map, l2.coverage_km).astype(np.float32)

        l3_map = np.zeros((256, 256), dtype=np.float32)
        l3_ok = False
        if l3 is not None:
            try:
                l3_map = l3.compute(origin_lat=origin_lat, origin_lon=origin_lon, timestamp=ts, context=ctx).astype(np.float32)
                l3_ok = True
            except Exception as exc:
                print(f"    [warn] L3 unavailable for this origin; fallback to zeros ({exc})")

        composite = (l1_interp + l2_interp + l3_map).astype(np.float32)
        panel_items.append((f"NORAD {norad}", composite, "viridis"))

        base = f"{tag}_radiomap_sat{i:02d}_norad{norad}"
        png_path = out_dir / f"{base}.png"
        l1_png = out_dir / f"{base}_l1.png"
        l2_png = out_dir / f"{base}_l2.png"

        _save_map(
            composite,
            png_path,
            f"Composite Radiomap | NORAD {norad} | f={l1.frequency_ghz:.2f} GHz",
            "Loss (dB)",
            "viridis",
            origin_lat,
            origin_lon,
            0.256,
            dpi,
        )
        _save_map(
            l1_interp,
            l1_png,
            f"L1 Contribution | NORAD {norad}",
            "Loss (dB)",
            "plasma",
            origin_lat,
            origin_lon,
            0.256,
            dpi,
        )
        _save_map(
            l2_interp,
            l2_png,
            f"L2 Contribution | NORAD {norad}",
            "Loss (dB)",
            "inferno",
            origin_lat,
            origin_lon,
            0.256,
            dpi,
        )

        np.save(out_dir / f"{base}_composite.npy", composite)
        np.save(out_dir / f"{base}_l1.npy", l1_interp)
        np.save(out_dir / f"{base}_l2.npy", l2_interp)
        np.save(out_dir / f"{base}_l3.npy", l3_map)

        sat_rows.append({
            "rank": i,
            "norad_id": norad,
            "elevation_deg": float(sat_info["elevation_deg"]),
            "azimuth_deg": float(sat_info["azimuth_deg"]),
            "slant_range_km": float(sat_info["slant_range_km"]),
            "l3_used": l3_ok,
            "composite_mean_db": float(np.mean(composite)),
            "composite_std_db": float(np.std(composite)),
        })

    _save_panel(
        panel_items,
        out_dir / f"{tag}_radiomap_topk_panel.png",
        "Radiomap Comparison Across Visible Satellites",
        origin_lat,
        origin_lon,
        0.256,
        dpi,
    )

    with (out_dir / f"{tag}_radiomap_satellite_table.json").open("w", encoding="utf-8") as f:
        json.dump(sat_rows, f, ensure_ascii=False, indent=2)
    print(f"  saved: {out_dir / f'{tag}_radiomap_satellite_table.json'}")

    return {
        "timestamp": ts.isoformat(),
        "top_k_requested": top_k_sats,
        "satellites_rendered": len(sat_rows),
        "satellites": sat_rows,
    }


def main() -> None:
    args = parse_args()

    cfg_path = _resolve_path(args.config)
    cfg = _load_yaml(cfg_path)

    ts = _parse_ts(args.timestamp)
    rain_rates = _parse_float_list(args.rain_rates)
    top_k = max(1, int(args.top_k_sats))

    origin_cfg = cfg.get("origin", {})
    origin_lat = float(args.origin_lat if args.origin_lat is not None else origin_cfg.get("latitude", 34.3416))
    origin_lon = float(args.origin_lon if args.origin_lon is not None else origin_cfg.get("longitude", 108.9398))

    layers_cfg = cfg.get("layers", {})
    l1_cfg = dict(layers_cfg.get("l1_macro", {}))
    l2_cfg = dict(layers_cfg.get("l2_topo", {}))
    l3_cfg = dict(layers_cfg.get("l3_urban", {}))

    if args.frequency_ghz is not None:
        l1_cfg["frequency_ghz"] = float(args.frequency_ghz)
        l2_cfg["frequency_ghz"] = float(args.frequency_ghz)

    sat_el = float(args.terrain_elevation_deg if args.terrain_elevation_deg is not None
                   else l2_cfg.get("satellite_elevation_deg", 45.0))
    sat_az = float(args.terrain_azimuth_deg if args.terrain_azimuth_deg is not None
                   else l2_cfg.get("satellite_azimuth_deg", 180.0))

    out_root = _resolve_path(args.output_root)
    terrain_dir = out_root / "terrain"
    atmosphere_dir = out_root / "atmosphere"
    radiomap_dir = out_root / "radiomap"
    for d in (terrain_dir, atmosphere_dir, radiomap_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Generate Feature Showcase")
    print(f"  config      : {cfg_path}")
    print(f"  timestamp   : {ts.isoformat()}")
    print(f"  origin      : ({origin_lat:.6f}, {origin_lon:.6f})")
    print(f"  frequency   : {float(l1_cfg.get('frequency_ghz', np.nan)):.2f} GHz")
    print(f"  rain rates  : {rain_rates}")
    print(f"  top-k sats  : {top_k}")
    print(f"  output root : {out_root}")
    print("=" * 88)

    l1 = L1MacroLayer(l1_cfg, origin_lat, origin_lon)
    l2 = L2TopoLayer(l2_cfg, origin_lat, origin_lon)
    l3: Optional[L3UrbanLayer]
    try:
        l3 = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)
    except Exception as exc:
        print(f"[warn] L3 layer init failed, radiomap will use L1+L2 only ({exc})")
        l3 = None

    summary: Dict[str, Any] = {
        "tag": args.output_tag,
        "timestamp": ts.isoformat(),
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "frequency_ghz": float(l1.frequency_ghz),
    }

    summary["terrain"] = _terrain_category(
        l2=l2,
        ts=ts,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        sat_el_deg=sat_el,
        sat_az_deg=sat_az,
        out_dir=terrain_dir,
        tag=args.output_tag,
        dpi=args.dpi,
    )

    summary["atmosphere"] = _atmosphere_category(
        l1=l1,
        ts=ts,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        rain_rates=rain_rates,
        out_dir=atmosphere_dir,
        tag=args.output_tag,
        dpi=args.dpi,
    )

    summary["radiomap"] = _radiomap_category(
        l1=l1,
        l2=l2,
        l3=l3,
        ts=ts,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        top_k_sats=top_k,
        min_el_deg=float(args.min_elevation_deg),
        out_dir=radiomap_dir,
        tag=args.output_tag,
        dpi=args.dpi,
    )

    summary_path = out_root / f"{args.output_tag}_showcase_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {summary_path}")

    l2.close()

    print("=" * 88)
    print("Showcase generation finished")
    print(f"  terrain    : {terrain_dir}")
    print(f"  atmosphere : {atmosphere_dir}")
    print(f"  radiomap   : {radiomap_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
