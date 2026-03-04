"""Unit tests for layers.l1_macro module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.layers.l1_macro import L1MacroLayer

ROOT = Path(__file__).resolve().parents[2]
TLE_PATH = ROOT / "data" / "2025_0101.tle"
IONEX_PATH = ROOT / "data" / "l1_space" / "data" / "UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz"

BASE_CONFIG = {
    "grid_size": 256,
    "coverage_km": 256.0,
    "resolution_m": 1000.0,
    "frequency_ghz": 10.0,
    "tle_file": str(TLE_PATH),
}


def _ts() -> datetime:
    return datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)


def test_l1_initialization():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    assert layer.grid_size == 256
    assert layer.coverage_km == 256.0
    assert layer.frequency_ghz == 10.0
    assert len(layer.satellites) > 0


def test_l1_initialization_missing_tle_raises():
    with pytest.raises(ValueError):
        L1MacroLayer(
            {
                "grid_size": 256,
                "coverage_km": 256.0,
                "resolution_m": 1000.0,
                "frequency_ghz": 10.0,
            },
            origin_lat=39.9,
            origin_lon=116.4,
        )


def test_l1_initialization_missing_era5_is_none():
    cfg = {**BASE_CONFIG, "era5_file": "/nonexistent/era5.nc"}
    layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
    assert layer.era5 is None


def test_l1_compute_shape_and_finite():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    loss = layer.compute(timestamp=_ts())
    assert loss.shape == (256, 256)
    valid = loss[~np.isnan(loss)]
    assert valid.size > 0
    assert np.all(np.isfinite(valid))
    assert np.all(valid > 0.0)


def test_l1_compute_components_contract():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    comp = layer.compute_components(timestamp=_ts())

    required_keys = {
        "total",
        "fspl",
        "atm",
        "iono",
        "gain",
        "pol",
        "elevation",
        "azimuth",
        "slant_range_m",
        "tec",
        "iwv",
        "occlusion_mask",
        "satellite",
        "timestamp",
    }
    assert required_keys.issubset(comp.keys())
    assert comp["total"].shape == (256, 256)
    assert comp["satellite"]["norad_id"]
    assert comp["satellite"]["elevation_deg"] > -90.0


def test_l1_visible_satellites_sorted():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=34.3416, origin_lon=108.9398)
    visible = layer.get_visible_satellites(
        origin_lat=34.3416,
        origin_lon=108.9398,
        timestamp=_ts(),
        min_elevation_deg=5.0,
        max_count=10,
    )
    assert len(visible) > 0
    elevations = [row["elevation_deg"] for row in visible]
    assert elevations == sorted(elevations, reverse=True)


def test_l1_target_norad_filter_selects_requested_sat():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=34.3416, origin_lon=108.9398)
    visible = layer.get_visible_satellites(
        origin_lat=34.3416,
        origin_lon=108.9398,
        timestamp=_ts(),
        min_elevation_deg=5.0,
        max_count=1,
    )
    assert visible
    top_norad = visible[0]["norad_id"]

    comp = layer.compute_components(timestamp=_ts(), target_norad_ids=[top_norad])
    assert comp["satellite"]["norad_id"] == top_norad


def test_l1_initialization_with_ionex_when_available():
    if not IONEX_PATH.exists():
        pytest.skip("IONEX sample file not found in repository.")
    cfg = {**BASE_CONFIG, "ionex_file": str(IONEX_PATH)}
    layer = L1MacroLayer(cfg, origin_lat=39.9042, origin_lon=116.4074)
    assert layer.ionex is not None
