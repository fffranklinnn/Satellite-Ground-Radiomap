"""Unit tests for layers.l1_macro module."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.context.time_utils import StrictModeError

from src.layers.l1_macro import L1MacroLayer
from src.context import GridSpec, FrameBuilder

ROOT = Path(__file__).resolve().parents[2]
TLE_PATH = ROOT / "data" / "starlink-2025-tle" / "2025-01-03.tle"
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


def test_l1_fallbacks_used_empty_on_init():
    """fallbacks_used must be empty after initialization with valid data."""
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    assert layer.fallbacks_used == []


def test_l1_fallbacks_used_records_missing_ionex(tmp_path):
    """fallbacks_used must record a fallback when IONEX file is missing."""
    cfg = {**BASE_CONFIG, "ionex_file": str(tmp_path / "nonexistent.INX")}
    layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
    assert any("IONEX" in fb for fb in layer.fallbacks_used)


def test_l1_clear_fallbacks_resets_per_frame_only(tmp_path):
    """clear_fallbacks() must preserve constructor-time fallbacks and only clear per-frame ones."""
    cfg = {**BASE_CONFIG, "ionex_file": str(tmp_path / "nonexistent.INX")}
    layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
    constructor_fallbacks = list(layer.fallbacks_used)
    assert len(constructor_fallbacks) > 0, "Expected constructor-time IONEX fallback"
    layer.clear_fallbacks()
    # Constructor-time fallbacks must survive clear_fallbacks()
    assert layer.fallbacks_used == constructor_fallbacks


def test_l1_fallbacks_used_returns_copy():
    """fallbacks_used must return a copy, not the internal list."""
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    fb1 = layer.fallbacks_used
    fb1.append("injected")
    assert "injected" not in layer.fallbacks_used


def test_l1_propagate_entry_component_decomposition():
    """
    total_loss_db == fspl_db + atm_db + iono_db + pol_db - gain_db
    for all non-occluded pixels (task23 AC-5 integration assertion).
    """
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    # Use the same grid_size as BASE_CONFIG so shapes match
    grid = GridSpec.from_legacy_args(39.9, 116.4, 256.0, 256, 256)
    fb = FrameBuilder(grid=grid)
    frame = fb.build(datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc))

    entry = layer.propagate_entry(frame)

    # Only check pixels that are not occluded (valid coverage)
    valid = ~entry.occlusion_mask
    if not valid.any():
        pytest.skip("No visible satellite pixels at this timestamp/location.")

    reconstructed = (
        entry.fspl_db[valid]
        + entry.atm_db[valid]
        + entry.iono_db[valid]
        + entry.pol_db[valid]
        - entry.gain_db[valid]
    )
    np.testing.assert_allclose(
        entry.total_loss_db[valid],
        reconstructed,
        atol=1e-3,
        err_msg="total_loss_db != fspl + atm + iono + pol - gain for non-occluded pixels",
    )


# ---------------------------------------------------------------------------
# Strict-mode fallback tests (task20 / AC-6)
# ---------------------------------------------------------------------------

def test_l1_strict_mode_missing_ionex_raises_strict_mode_error(tmp_path):
    """Missing IONEX in strict mode must raise StrictModeError, not FileNotFoundError."""
    cfg = {
        **BASE_CONFIG,
        "strict_data": True,
        "ionex_file": str(tmp_path / "nonexistent.INX"),
    }
    with pytest.raises(StrictModeError):
        L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)


def test_l1_strict_mode_missing_era5_raises_strict_mode_error(tmp_path):
    """Unreadable ERA5 in strict mode must raise StrictModeError."""
    # Write a file that exists but is not valid NetCDF so load_era5 returns None
    bad_era5 = tmp_path / "bad.nc"
    bad_era5.write_bytes(b"not a netcdf file")
    cfg = {
        **BASE_CONFIG,
        "strict_data": True,
        "era5_file": str(bad_era5),
    }
    with pytest.raises(StrictModeError):
        L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)


def test_l1_strict_mode_no_timestamp_raises_strict_mode_error():
    """No timestamp in strict mode must raise StrictModeError."""
    cfg = {**BASE_CONFIG, "strict_data": True}
    layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
    with pytest.raises(StrictModeError):
        layer._resolve_sim_datetime(None)


def test_l1_non_strict_missing_ionex_does_not_raise(tmp_path):
    """Missing IONEX in non-strict mode must not raise; fallback is recorded."""
    cfg = {
        **BASE_CONFIG,
        "strict_data": False,
        "ionex_file": str(tmp_path / "nonexistent.INX"),
    }
    layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
    assert any("IONEX" in fb for fb in layer.fallbacks_used)


# ---------------------------------------------------------------------------
# L2 strict-mode tests (task20 / AC-6)
# ---------------------------------------------------------------------------

from src.layers.l2_topo import L2TopoLayer


def test_l2_strict_mode_missing_dem_raises_strict_mode_error(tmp_path):
    """Missing DEM in strict mode must raise StrictModeError."""
    cfg = {
        "dem_file": str(tmp_path / "nonexistent.tif"),
        "frequency_ghz": 14.5,
        "strict_data": True,
        "grid_size": 256,
        "coverage_km": 25.6,
        "resolution_m": 100.0,
    }
    layer = L2TopoLayer(cfg, origin_lat=34.3, origin_lon=108.9)
    with pytest.raises(StrictModeError):
        layer._open_dem()


def test_l2_non_strict_missing_dem_raises_file_not_found(tmp_path):
    """Missing DEM in non-strict mode raises FileNotFoundError (not StrictModeError)."""
    cfg = {
        "dem_file": str(tmp_path / "nonexistent.tif"),
        "frequency_ghz": 14.5,
        "strict_data": False,
        "grid_size": 256,
        "coverage_km": 25.6,
        "resolution_m": 100.0,
    }
    layer = L2TopoLayer(cfg, origin_lat=34.3, origin_lon=108.9)
    with pytest.raises(FileNotFoundError) as exc_info:
        layer._open_dem()
    assert not isinstance(exc_info.value, StrictModeError)


# ---------------------------------------------------------------------------
# L3 strict-mode tests (task20 / AC-6)
# ---------------------------------------------------------------------------

from src.layers.l3_urban import L3UrbanLayer


def test_l3_strict_mode_empty_tile_cache_raises_strict_mode_error(tmp_path):
    """Empty tile cache in strict mode must raise StrictModeError."""
    empty_cache = tmp_path / "tiles"
    empty_cache.mkdir()
    cfg = {
        "tile_cache_root": str(empty_cache),
        "nlos_loss_db": 20.0,
        "strict_data": True,
        "incident_dir": {"az_deg": 180.0, "el_deg": 45.0},
        "grid_size": 256,
        "coverage_km": 0.256,
        "resolution_m": 1.0,
    }
    layer = L3UrbanLayer(cfg, origin_lat=34.3, origin_lon=108.9)
    with pytest.raises(StrictModeError):
        layer._find_nearest_tile_id(108.9, 34.3)


def test_l3_non_strict_empty_tile_cache_raises_file_not_found(tmp_path):
    """Empty tile cache in non-strict mode raises FileNotFoundError."""
    empty_cache = tmp_path / "tiles"
    empty_cache.mkdir()
    cfg = {
        "tile_cache_root": str(empty_cache),
        "nlos_loss_db": 20.0,
        "strict_data": False,
        "incident_dir": {"az_deg": 180.0, "el_deg": 45.0},
        "grid_size": 256,
        "coverage_km": 0.256,
        "resolution_m": 1.0,
    }
    layer = L3UrbanLayer(cfg, origin_lat=34.3, origin_lon=108.9)
    with pytest.raises(FileNotFoundError) as exc_info:
        layer._find_nearest_tile_id(108.9, 34.3)
    assert not isinstance(exc_info.value, StrictModeError)
