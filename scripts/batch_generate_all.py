#!/usr/bin/env python3
"""
Batch generate radiomaps across all available data dimensions.

Variation axes: time, multi-day IONEX, frequency, rain rate, geographic
location, satellite incident angle, L3 tile coverage.

Usage:
    python scripts/batch_generate_all.py                        # run all tasks
    python scripts/batch_generate_all.py --tasks G1 G3 G9       # selected tasks
    python scripts/batch_generate_all.py --no-png --workers 8   # fast, parallel
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XIAN = (34.3416, 108.9398)

CITIES: Dict[str, Tuple[float, float]] = {
    "xian":     (34.3416, 108.9398),
    "beijing":  (39.9042, 116.4074),
    "shanghai": (31.2304, 121.4737),
    "chengdu":  (30.5728, 104.0668),
    "urumqi":   (43.8256,  87.6168),
    "lhasa":    (29.6500,  91.1000),
    "harbin":   (45.7500, 126.6500),
    "guangzhou": (23.1291, 113.2644),
}

FREQUENCIES_GHZ = [1.5, 3.0, 6.0, 10.0, 14.5, 26.5]
FREQ_NAMES      = ["L", "S", "C", "X", "Ku", "Ka"]

RAIN_RATES = [0, 5, 10, 25, 50, 100]

ELEVATIONS = [15, 30, 45, 60, 75]  # 90° excluded: zero horizontal component
AZIMUTHS   = [0, 45, 90, 135, 180, 225, 270, 315]

MULTIDAY_DOYS = list(range(1, 20))  # DOY 001-019

TLE_FILE   = "data/2025_0101.tle"
ERA5_FILE  = "data/l1_space/data/data_stream-oper_stepType-instant.nc"
IONEX_FILE = "data/l1_space/data/cddis_data_2025/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz"
DEM_FILE   = "data/l2_topo/全国DEM数据.tif"
TILE_CACHE = "data/l3_urban/xian/tiles_60"

OUTPUT_ROOT = PROJECT_ROOT / "output" / "batch"

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def make_l1_config(
    freq_ghz: float = 14.5,
    rain_rate: float = 0.0,
    ionex_file: str = IONEX_FILE,
    era5_file: str = ERA5_FILE,
    tle_file: str = TLE_FILE,
) -> dict:
    return {
        "grid_size": 256,
        "coverage_km": 256.0,
        "resolution_m": 1000.0,
        "frequency_ghz": freq_ghz,
        "satellite_altitude_km": 550.0,
        "tec": 10.0,
        "rain_rate_mm_h": rain_rate,
        "ionex_file": ionex_file,
        "era5_file": era5_file,
        "tle_file": tle_file,
    }


def make_l2_config(
    el_deg: float = 45.0,
    az_deg: float = 180.0,
    freq_ghz: float = 14.5,
) -> dict:
    return {
        "grid_size": 256,
        "coverage_km": 25.6,
        "resolution_m": 100.0,
        "dem_file": DEM_FILE,
        "frequency_ghz": freq_ghz,
        "satellite_elevation_deg": el_deg,
        "satellite_azimuth_deg": az_deg,
    }


def make_l3_config() -> dict:
    return {
        "grid_size": 256,
        "coverage_km": 0.256,
        "resolution_m": 1.0,
        "tile_cache_root": TILE_CACHE,
        "nlos_loss_db": 20.0,
        "occ_loss_db": 30.0,
    }


def ionex_path_for_doy(doy: int) -> str:
    return f"data/l1_space/data/cddis_data_2025/UPC0OPSRAP_2025{doy:03d}0000_01D_15M_GIM.INX.gz"

# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_result(
    arr: np.ndarray,
    npy_path: Path,
    save_png: bool = False,
    title: str = "",
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
    coverage_km: Optional[float] = None,
) -> dict:
    """Save .npy (and optional .png). Return metadata dict."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), arr)

    meta = {
        "file": str(npy_path.relative_to(PROJECT_ROOT)),
        "shape": list(arr.shape),
        "loss_mean": float(np.nanmean(arr)),
        "loss_std": float(np.nanstd(arr)),
        "loss_min": float(np.nanmin(arr)),
        "loss_max": float(np.nanmax(arr)),
    }

    if save_png:
        try:
            from src.utils.plotter import plot_radio_map
            png_path = npy_path.with_suffix(".png")
            plot_radio_map(
                arr, title=title, output_file=str(png_path),
                origin_lat=origin_lat, origin_lon=origin_lon,
                coverage_km=coverage_km, dpi=100,
            )
            meta["png"] = str(png_path.relative_to(PROJECT_ROOT))
        except Exception:
            pass  # PNG is optional

    return meta

# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Per-task progress tracker with elapsed time, ETA, and speed."""

    def __init__(self, task: str, total: int):
        self.task = task
        self.total = total
        self.t0 = time.time()

    def update(self, i: int, extra: str = ""):
        done = i + 1
        elapsed = time.time() - self.t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (self.total - done) / speed if speed > 0 else 0
        print(f"  [{self.task}] {done}/{self.total} ({done/self.total*100:.0f}%) "
              f"{elapsed:.1f}s | ETA {eta:.1f}s | {speed:.2f}/s  {extra}")

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0


# ---------------------------------------------------------------------------
# L1 caching (applied in main via monkey-patch)
# ---------------------------------------------------------------------------

_tle_cache: Dict[str, Any] = {}
_original_parse_tle_file = None  # set in _apply_patches()

def _cached_parse_tle_file(tle_file_path: str):
    """Drop-in replacement for l1_macro._parse_tle_file with caching."""
    if tle_file_path not in _tle_cache:
        _tle_cache[tle_file_path] = _original_parse_tle_file(tle_file_path)
    else:
        n = len(_tle_cache[tle_file_path][0])
        print(f"[L1] TLE cache hit ({n} entries): {Path(tle_file_path).name}")
    return _tle_cache[tle_file_path]


_sat_cache: Dict[tuple, Any] = {}
_original_select_best_satellite = None  # set in _apply_patches()

def _cached_select_best_satellite(self, origin_lat, origin_lon):
    """Wrapper around L1MacroLayer._select_best_satellite with caching."""
    key = (round(origin_lat, 4), round(origin_lon, 4), str(self._sim_time))
    if key not in _sat_cache:
        _sat_cache[key] = _original_select_best_satellite(self, origin_lat, origin_lon)
    else:
        info = _sat_cache[key][1]
        print(f"[L1] (cached) NORAD {info['norad_id']} | "
              f"el={info['elevation_deg']:.2f}deg | "
              f"alt={info['alt_m']/1e3:.1f} km")
    return _sat_cache[key]


def _apply_patches():
    """Apply monkey-patches for TLE parsing and satellite selection caching."""
    global _original_parse_tle_file, _original_select_best_satellite

    import src.layers.l1_macro as _l1_mod
    _original_parse_tle_file = _l1_mod._parse_tle_file
    _l1_mod._parse_tle_file = _cached_parse_tle_file

    _original_select_best_satellite = L1MacroLayer._select_best_satellite
    L1MacroLayer._select_best_satellite = _cached_select_best_satellite
    print("[batch] L1 caching patches applied (TLE parse + satellite selection)")

# ---------------------------------------------------------------------------
# G1: L1 hourly (Xi'an, Ku-band)
# ---------------------------------------------------------------------------

def run_g1(save_png: bool = False) -> List[dict]:
    print("\n=== G1: L1 hourly (Xi'an, Ku-band, 24h) ===")
    out = OUTPUT_ROOT / "G01_l1_hourly"
    catalog = []
    lat, lon = XIAN
    cfg = make_l1_config(freq_ghz=14.5)
    layer = L1MacroLayer(cfg, lat, lon)
    pt = ProgressTracker("G1", 24)

    for h in range(24):
        ts = datetime(2025, 1, 1, h, 0, 0, tzinfo=timezone.utc)
        loss = layer.compute(lat, lon, timestamp=ts)
        path = out / f"l1_h{h:02d}.npy"
        m = save_result(loss, path, save_png, f"L1 Ku 14.5GHz {h:02d}:00 UTC",
                        lat, lon, 256.0)
        m.update(task="G1", layer="L1", origin="xian",
                 frequency_ghz=14.5, hour=h, timestamp=str(ts))
        catalog.append(m)
        pt.update(h, f"h={h:02d}")

    return catalog

# ---------------------------------------------------------------------------
# G2: L1 multi-day (Xi'an, Ku-band, noon)
# ---------------------------------------------------------------------------

def run_g2(save_png: bool = False) -> List[dict]:
    print("\n=== G2: L1 multi-day (Xi'an, Ku-band, noon, 19 days) ===")
    out = OUTPUT_ROOT / "G02_l1_multiday"
    catalog = []
    lat, lon = XIAN
    pt = ProgressTracker("G2", len(MULTIDAY_DOYS))

    for idx, doy in enumerate(MULTIDAY_DOYS):
        ionex = ionex_path_for_doy(doy)
        if not (PROJECT_ROOT / ionex).exists():
            print(f"  skip DOY {doy:03d}: IONEX not found")
            continue
        cfg = make_l1_config(freq_ghz=14.5, ionex_file=ionex)
        layer = L1MacroLayer(cfg, lat, lon)
        ts = datetime(2025, 1, doy, 12, 0, 0, tzinfo=timezone.utc)
        loss = layer.compute(lat, lon, timestamp=ts)
        path = out / f"l1_doy{doy:03d}.npy"
        m = save_result(loss, path, save_png, f"L1 Ku DOY{doy:03d} 12:00 UTC",
                        lat, lon, 256.0)
        m.update(task="G2", layer="L1", origin="xian",
                 frequency_ghz=14.5, doy=doy, timestamp=str(ts))
        catalog.append(m)
        pt.update(idx, f"DOY={doy:03d}")

    return catalog

# ---------------------------------------------------------------------------
# G3: L1 frequency sweep (Xi'an, noon)
# ---------------------------------------------------------------------------

def run_g3(save_png: bool = False) -> List[dict]:
    print("\n=== G3: L1 frequency sweep (Xi'an, noon, 6 bands) ===")
    out = OUTPUT_ROOT / "G03_l1_freq_sweep"
    catalog = []
    lat, lon = XIAN
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    pt = ProgressTracker("G3", len(FREQUENCIES_GHZ))

    for idx, (freq, name) in enumerate(zip(FREQUENCIES_GHZ, FREQ_NAMES)):
        cfg = make_l1_config(freq_ghz=freq)
        layer = L1MacroLayer(cfg, lat, lon)
        loss = layer.compute(lat, lon, timestamp=ts)
        path = out / f"l1_{name}_{freq}GHz.npy"
        m = save_result(loss, path, save_png, f"L1 {name}-band {freq}GHz",
                        lat, lon, 256.0)
        m.update(task="G3", layer="L1", origin="xian",
                 frequency_ghz=freq, band=name, timestamp=str(ts))
        catalog.append(m)
        pt.update(idx, f"{name} {freq}GHz")

    return catalog

# ---------------------------------------------------------------------------
# G4: L1 rain sweep (Xi'an, Ku-band, noon)
# ---------------------------------------------------------------------------

def run_g4(save_png: bool = False) -> List[dict]:
    print("\n=== G4: L1 rain sweep (Xi'an, Ku-band, noon, 6 rates) ===")
    out = OUTPUT_ROOT / "G04_l1_rain_sweep"
    catalog = []
    lat, lon = XIAN
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    pt = ProgressTracker("G4", len(RAIN_RATES))

    for idx, rain in enumerate(RAIN_RATES):
        cfg = make_l1_config(freq_ghz=14.5, rain_rate=rain)
        layer = L1MacroLayer(cfg, lat, lon)
        loss = layer.compute(lat, lon, timestamp=ts)
        path = out / f"l1_rain{rain}mm.npy"
        m = save_result(loss, path, save_png, f"L1 Ku rain={rain}mm/h",
                        lat, lon, 256.0)
        m.update(task="G4", layer="L1", origin="xian",
                 frequency_ghz=14.5, rain_rate_mm_h=rain, timestamp=str(ts))
        catalog.append(m)
        pt.update(idx, f"rain={rain}")

    return catalog

# ---------------------------------------------------------------------------
# G5: L1 multi-city (Ku-band, noon)
# ---------------------------------------------------------------------------

def run_g5(save_png: bool = False) -> List[dict]:
    print("\n=== G5: L1 multi-city (Ku-band, noon, 8 cities) ===")
    out = OUTPUT_ROOT / "G05_l1_multi_city"
    catalog = []
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    pt = ProgressTracker("G5", len(CITIES))

    for idx, (city, (lat, lon)) in enumerate(CITIES.items()):
        cfg = make_l1_config(freq_ghz=14.5)
        layer = L1MacroLayer(cfg, lat, lon)
        loss = layer.compute(lat, lon, timestamp=ts)
        path = out / f"l1_{city}.npy"
        m = save_result(loss, path, save_png, f"L1 Ku {city}",
                        lat, lon, 256.0)
        m.update(task="G5", layer="L1", origin=city,
                 origin_lat=lat, origin_lon=lon,
                 frequency_ghz=14.5, timestamp=str(ts))
        catalog.append(m)
        pt.update(idx, city)

    return catalog

# ---------------------------------------------------------------------------
# G6: L2 angle sweep (Xi'an)
# ---------------------------------------------------------------------------

def run_g6(save_png: bool = False) -> List[dict]:
    print("\n=== G6: L2 angle sweep (Xi'an, 48 angles) ===")
    out = OUTPUT_ROOT / "G06_l2_angle_sweep"
    catalog = []
    lat, lon = XIAN
    total = len(ELEVATIONS) * len(AZIMUTHS)
    pt = ProgressTracker("G6", total)
    i = 0

    for el in ELEVATIONS:
        for az in AZIMUTHS:
            cfg = make_l2_config(el_deg=el, az_deg=az)
            layer = L2TopoLayer(cfg, lat, lon)
            loss = layer.compute(lat, lon)
            path = out / f"l2_el{el}_az{az}.npy"
            m = save_result(loss, path, save_png, f"L2 el={el} az={az}",
                            lat, lon, 25.6)
            m.update(task="G6", layer="L2", origin="xian",
                     elevation_deg=el, azimuth_deg=az)
            catalog.append(m)
            pt.update(i, f"el={el} az={az}")
            i += 1

    return catalog

# ---------------------------------------------------------------------------
# G7: L2 multi-city (el=45, az=180)
# ---------------------------------------------------------------------------

def run_g7(save_png: bool = False) -> List[dict]:
    print("\n=== G7: L2 multi-city (el=45, az=180, 8 cities) ===")
    out = OUTPUT_ROOT / "G07_l2_multi_city"
    catalog = []
    pt = ProgressTracker("G7", len(CITIES))

    for idx, (city, (lat, lon)) in enumerate(CITIES.items()):
        cfg = make_l2_config(el_deg=45, az_deg=180)
        layer = L2TopoLayer(cfg, lat, lon)
        loss = layer.compute(lat, lon)
        path = out / f"l2_{city}.npy"
        m = save_result(loss, path, save_png, f"L2 {city} el=45 az=180",
                        lat, lon, 25.6)
        m.update(task="G7", layer="L2", origin=city,
                 origin_lat=lat, origin_lon=lon,
                 elevation_deg=45, azimuth_deg=180)
        catalog.append(m)
        pt.update(idx, city)

    return catalog


# ---------------------------------------------------------------------------
# G8: L3 angle sweep (Xi'an center tile, 48 angles)
# ---------------------------------------------------------------------------

def run_g8(save_png: bool = False) -> List[dict]:
    print("\n=== G8: L3 angle sweep (Xi'an center tile, 48 angles) ===")
    out = OUTPUT_ROOT / "G08_l3_angle_sweep"
    catalog = []
    lat, lon = XIAN
    cfg = make_l3_config()
    layer = L3UrbanLayer(cfg, lat, lon)
    total = len(ELEVATIONS) * len(AZIMUTHS)
    pt = ProgressTracker("G8", total)
    i = 0

    for el in ELEVATIONS:
        for az in AZIMUTHS:
            ctx = LayerContext(incident_dir={"az_deg": az, "el_deg": el})
            loss = layer.compute(lat, lon, context=ctx)
            path = out / f"l3_el{el}_az{az}.npy"
            m = save_result(loss, path, save_png, f"L3 el={el} az={az}",
                            lat, lon, 0.256)
            m.update(task="G8", layer="L3", origin="xian",
                     elevation_deg=el, azimuth_deg=az)
            catalog.append(m)
            pt.update(i, f"el={el} az={az}")
            i += 1

    return catalog


# ---------------------------------------------------------------------------
# G9: L3 all tiles (az=180, el=45)
# ---------------------------------------------------------------------------

def _g9_worker(args):
    """Worker for parallel L3 tile generation."""
    tile_id, tile_cache, npy_dir, save_png = args
    try:
        cfg = {
            "grid_size": 256, "coverage_km": 0.256, "resolution_m": 1.0,
            "tile_cache_root": tile_cache,
            "nlos_loss_db": 20.0, "occ_loss_db": 30.0,
        }
        lat, lon = XIAN
        layer = L3UrbanLayer(cfg, lat, lon)
        ctx = LayerContext(
            incident_dir={"az_deg": 180, "el_deg": 45},
            extras={"tile_id": tile_id},
        )
        loss = layer.compute(lat, lon, context=ctx)
        path = Path(npy_dir) / f"{tile_id}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), loss)
        return {
            "task": "G9", "layer": "L3", "origin": "xian",
            "tile_id": tile_id, "elevation_deg": 45, "azimuth_deg": 180,
            "file": str(path),
            "loss_mean": float(np.nanmean(loss)),
            "loss_std": float(np.nanstd(loss)),
        }
    except Exception as e:
        return {"task": "G9", "tile_id": tile_id, "error": str(e)}


def run_g9(save_png: bool = False, workers: int = 1) -> List[dict]:
    print("\n=== G9: L3 all tiles (az=180, el=45, 1320 tiles) ===")
    out = OUTPUT_ROOT / "G09_l3_all_tiles"
    out.mkdir(parents=True, exist_ok=True)

    tile_dirs = sorted((PROJECT_ROOT / TILE_CACHE).iterdir())
    tile_ids = [d.name for d in tile_dirs if d.is_dir()]
    total = len(tile_ids)
    print(f"  Found {total} tiles")
    pt = ProgressTracker("G9", total)

    args_list = [(tid, TILE_CACHE, str(out), save_png) for tid in tile_ids]

    catalog = []
    if workers > 1:
        with Pool(workers) as pool:
            for i, m in enumerate(pool.imap_unordered(_g9_worker, args_list)):
                catalog.append(m)
                if (i + 1) % 100 == 0 or i + 1 == total:
                    pt.update(i)
    else:
        for i, a in enumerate(args_list):
            catalog.append(_g9_worker(a))
            if (i + 1) % 100 == 0 or i + 1 == total:
                pt.update(i)

    return catalog


# ---------------------------------------------------------------------------
# G10: Composite L1+L2 multi-city (Ku, noon)
# ---------------------------------------------------------------------------

def run_g10(save_png: bool = False) -> List[dict]:
    print("\n=== G10: Composite L1+L2 multi-city (Ku, noon, 8 cities) ===")
    out = OUTPUT_ROOT / "G10_composite_l1l2"
    catalog = []
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    pt = ProgressTracker("G10", len(CITIES))

    for idx, (city, (lat, lon)) in enumerate(CITIES.items()):
        l1 = L1MacroLayer(make_l1_config(freq_ghz=14.5), lat, lon)
        l2 = L2TopoLayer(make_l2_config(el_deg=45, az_deg=180), lat, lon)
        agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2)
        loss = agg.aggregate(lat, lon, timestamp=ts)
        path = out / f"composite_l1l2_{city}.npy"
        m = save_result(loss, path, save_png, f"L1+L2 {city}",
                        lat, lon, 0.256)
        m.update(task="G10", layer="L1+L2", origin=city,
                 origin_lat=lat, origin_lon=lon,
                 frequency_ghz=14.5, timestamp="2025-01-01T12:00Z")
        catalog.append(m)
        pt.update(idx, city)

    return catalog


# ---------------------------------------------------------------------------
# G11: Composite L1+L2+L3 hourly (Xi'an)
# ---------------------------------------------------------------------------

def run_g11(save_png: bool = False) -> List[dict]:
    print("\n=== G11: Composite L1+L2+L3 hourly (Xi'an, 24h) ===")
    out = OUTPUT_ROOT / "G11_composite_hourly"
    catalog = []
    lat, lon = XIAN
    pt = ProgressTracker("G11", 24)

    l1 = L1MacroLayer(make_l1_config(freq_ghz=14.5), lat, lon)
    l2 = L2TopoLayer(make_l2_config(el_deg=45, az_deg=180), lat, lon)
    l3 = L3UrbanLayer(make_l3_config(), lat, lon)
    agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2, l3_layer=l3)
    ctx = LayerContext(incident_dir={"az_deg": 180, "el_deg": 45})

    for h in range(24):
        ts = datetime(2025, 1, 1, h, 0, 0, tzinfo=timezone.utc)
        loss = agg.aggregate(lat, lon, timestamp=ts, context=ctx)
        path = out / f"composite_h{h:02d}.npy"
        m = save_result(loss, path, save_png, f"L1+L2+L3 {h:02d}:00 UTC",
                        lat, lon, 0.256)
        m.update(task="G11", layer="L1+L2+L3", origin="xian",
                 frequency_ghz=14.5, timestamp=ts.isoformat())
        catalog.append(m)
        pt.update(h, f"{h:02d}:00 UTC")

    return catalog


# ---------------------------------------------------------------------------
# G12: Composite L1+L2+L3 frequency sweep (Xi'an, noon)
# ---------------------------------------------------------------------------

def run_g12(save_png: bool = False) -> List[dict]:
    print("\n=== G12: Composite L1+L2+L3 freq sweep (Xi'an, noon) ===")
    out = OUTPUT_ROOT / "G12_composite_freq"
    catalog = []
    lat, lon = XIAN
    ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    ctx = LayerContext(incident_dir={"az_deg": 180, "el_deg": 45})
    pt = ProgressTracker("G12", len(FREQUENCIES_GHZ))

    for idx, (freq, fname) in enumerate(zip(FREQUENCIES_GHZ, FREQ_NAMES)):
        l1 = L1MacroLayer(make_l1_config(freq_ghz=freq), lat, lon)
        l2 = L2TopoLayer(make_l2_config(el_deg=45, az_deg=180, freq_ghz=freq),
                          lat, lon)
        l3 = L3UrbanLayer(make_l3_config(), lat, lon)
        agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2, l3_layer=l3)
        loss = agg.aggregate(lat, lon, timestamp=ts, context=ctx)
        path = out / f"composite_{fname}_{freq}GHz.npy"
        m = save_result(loss, path, save_png,
                        f"L1+L2+L3 {fname} {freq}GHz", lat, lon, 0.256)
        m.update(task="G12", layer="L1+L2+L3", origin="xian",
                 frequency_ghz=freq, timestamp="2025-01-01T12:00Z")
        catalog.append(m)
        pt.update(idx, f"{fname} {freq}GHz")

    return catalog


# ---------------------------------------------------------------------------
# G13: L1 multi-day x multi-freq (Xi'an, noon)
# ---------------------------------------------------------------------------

def run_g13(save_png: bool = False) -> List[dict]:
    print("\n=== G13: L1 multi-day x multi-freq (19 days x 6 freqs = 114) ===")
    out = OUTPUT_ROOT / "G13_l1_multiday_freq"
    catalog = []
    lat, lon = XIAN
    total = len(MULTIDAY_DOYS) * len(FREQUENCIES_GHZ)
    pt = ProgressTracker("G13", total)
    i = 0

    for doy in MULTIDAY_DOYS:
        ionex = ionex_path_for_doy(doy)
        ts = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        for freq, fname in zip(FREQUENCIES_GHZ, FREQ_NAMES):
            cfg = make_l1_config(freq_ghz=freq, ionex_file=ionex)
            layer = L1MacroLayer(cfg, lat, lon)
            loss = layer.compute(lat, lon, timestamp=ts)
            path = out / f"l1_doy{doy:03d}_{fname}_{freq}GHz.npy"
            m = save_result(loss, path, save_png,
                            f"L1 DOY{doy:03d} {fname} {freq}GHz",
                            lat, lon, 256.0)
            m.update(task="G13", layer="L1", origin="xian",
                     frequency_ghz=freq, ionex_doy=doy,
                     timestamp=ts.isoformat())
            catalog.append(m)
            pt.update(i, f"DOY{doy:03d} {fname}")
            i += 1

    return catalog


# ---------------------------------------------------------------------------
# G14: L3 all tiles x 6 angles
# ---------------------------------------------------------------------------

def _g14_worker(args):
    """Worker for parallel L3 tile x angle generation."""
    tile_id, el, az, tile_cache, npy_dir = args
    try:
        cfg = {
            "grid_size": 256, "coverage_km": 0.256, "resolution_m": 1.0,
            "tile_cache_root": tile_cache,
            "nlos_loss_db": 20.0, "occ_loss_db": 30.0,
        }
        lat, lon = XIAN
        layer = L3UrbanLayer(cfg, lat, lon)
        ctx = LayerContext(
            incident_dir={"az_deg": az, "el_deg": el},
            extras={"tile_id": tile_id},
        )
        loss = layer.compute(lat, lon, context=ctx)
        path = Path(npy_dir) / f"{tile_id}_el{el}_az{az}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), loss)
        return {
            "task": "G14", "layer": "L3", "origin": "xian",
            "tile_id": tile_id, "elevation_deg": el, "azimuth_deg": az,
            "file": str(path),
            "loss_mean": float(np.nanmean(loss)),
            "loss_std": float(np.nanstd(loss)),
        }
    except Exception as e:
        return {"task": "G14", "tile_id": tile_id,
                "elevation_deg": el, "azimuth_deg": az, "error": str(e)}


# Use a subset of angles for G14 to keep it tractable
G14_ANGLES = [(15, 180), (30, 180), (45, 180), (60, 180), (45, 90), (45, 270)]


def run_g14(save_png: bool = False, workers: int = 1) -> List[dict]:
    print(f"\n=== G14: L3 all tiles x {len(G14_ANGLES)} angles ===")
    out = OUTPUT_ROOT / "G14_l3_tiles_angles"
    out.mkdir(parents=True, exist_ok=True)

    tile_dirs = sorted((PROJECT_ROOT / TILE_CACHE).iterdir())
    tile_ids = [d.name for d in tile_dirs if d.is_dir()]
    total = len(tile_ids) * len(G14_ANGLES)
    print(f"  {len(tile_ids)} tiles x {len(G14_ANGLES)} angles = {total} maps")
    pt = ProgressTracker("G14", total)

    args_list = [
        (tid, el, az, TILE_CACHE, str(out))
        for tid in tile_ids
        for el, az in G14_ANGLES
    ]

    catalog = []
    if workers > 1:
        with Pool(workers) as pool:
            for i, m in enumerate(pool.imap_unordered(_g14_worker, args_list)):
                catalog.append(m)
                if (i + 1) % 500 == 0 or i + 1 == total:
                    pt.update(i)
    else:
        for i, a in enumerate(args_list):
            catalog.append(_g14_worker(a))
            if (i + 1) % 500 == 0 or i + 1 == total:
                pt.update(i)

    return catalog


# ---------------------------------------------------------------------------
# Task registry & main
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "G1":  lambda png, w: run_g1(png),
    "G2":  lambda png, w: run_g2(png),
    "G3":  lambda png, w: run_g3(png),
    "G4":  lambda png, w: run_g4(png),
    "G5":  lambda png, w: run_g5(png),
    "G6":  lambda png, w: run_g6(png),
    "G7":  lambda png, w: run_g7(png),
    "G8":  lambda png, w: run_g8(png),
    "G9":  lambda png, w: run_g9(png, w),
    "G10": lambda png, w: run_g10(png),
    "G11": lambda png, w: run_g11(png),
    "G12": lambda png, w: run_g12(png),
    "G13": lambda png, w: run_g13(png),
    "G14": lambda png, w: run_g14(png, w),
}

ALL_TASKS = list(TASK_REGISTRY.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate radiomaps from all available data.")
    parser.add_argument(
        "--tasks", nargs="*", default=ALL_TASKS, metavar="G",
        help=f"Tasks to run (default: all). Choices: {', '.join(ALL_TASKS)}")
    parser.add_argument(
        "--no-png", action="store_true",
        help="Skip PNG generation for faster execution")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for G9/G14 (default: 1)")
    args = parser.parse_args()

    save_png = not args.no_png
    tasks = [t.upper() for t in args.tasks]
    for t in tasks:
        if t not in TASK_REGISTRY:
            print(f"Unknown task: {t}. Valid: {', '.join(ALL_TASKS)}")
            sys.exit(1)

    # Apply L1 caching patches
    _apply_patches()

    print(f"\n=== Batch Radiomap Generation ===")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"PNG: {'yes' if save_png else 'no'}")
    print(f"Workers: {args.workers}")
    print(f"Output: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    full_catalog = []
    t0 = time.time()

    for task_name in tasks:
        task_t0 = time.time()
        try:
            entries = TASK_REGISTRY[task_name](save_png, args.workers)
            full_catalog.extend(entries)
            ok = sum(1 for e in entries if "error" not in e)
            err = sum(1 for e in entries if "error" in e)
            task_elapsed = time.time() - task_t0
            total_elapsed = time.time() - t0
            print(f"  -> {task_name}: {ok} ok, {err} errors ({task_elapsed:.1f}s) | "
                  f"Total so far: {len(full_catalog)} maps, {total_elapsed:.1f}s")
        except Exception as e:
            print(f"  -> {task_name}: FAILED — {e}")
            traceback.print_exc()

    # Write catalog
    catalog_path = OUTPUT_ROOT / "catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(full_catalog, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.time() - t0
    total_ok = sum(1 for e in full_catalog if "error" not in e)
    total_err = sum(1 for e in full_catalog if "error" in e)
    print(f"\n=== Done ===")
    print(f"Total radiomaps: {total_ok} ok, {total_err} errors")
    print(f"Catalog: {catalog_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
