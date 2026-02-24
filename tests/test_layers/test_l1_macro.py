"""Unit tests for layers.l1_macro module."""

import pytest
import numpy as np
from datetime import datetime
from src.layers.l1_macro import L1MacroLayer

IONEX_PATH = "data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz"

BASE_CONFIG = {
    'grid_size': 256,
    'coverage_km': 256.0,
    'resolution_m': 1000.0,
    'frequency_ghz': 10.0,
    'satellite_altitude_km': 550.0,
}


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_l1_initialization():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    assert layer.grid_size == 256
    assert layer.coverage_km == 256.0
    assert layer.frequency_ghz == 10.0
    assert layer.satellite_altitude_km == 550.0


def test_l1_initialization_with_ionex():
    cfg = {**BASE_CONFIG, 'ionex_file': IONEX_PATH}
    layer = L1MacroLayer(cfg, origin_lat=39.9042, origin_lon=116.4074)
    assert layer.ionex is not None


def test_l1_initialization_missing_era5_is_none():
    cfg = {**BASE_CONFIG, 'era5_file': '/nonexistent/era5.nc'}
    layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
    assert layer.era5 is None


# ---------------------------------------------------------------------------
# Compute — basic contract
# ---------------------------------------------------------------------------

def test_l1_compute_shape():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    loss = layer.compute(datetime(2025, 1, 1, 0, 0, 0))
    assert loss.shape == (256, 256)


def test_l1_compute_finite_values():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    loss = layer.compute(datetime(2025, 1, 1, 12, 0, 0))
    valid = loss[~np.isnan(loss)]
    assert len(valid) > 0
    assert np.all(np.isfinite(valid))


def test_l1_compute_positive_losses():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    loss = layer.compute(datetime(2025, 1, 1, 0, 0, 0))
    valid = loss[~np.isnan(loss)]
    assert np.all(valid > 0)


def test_l1_compute_reasonable_range():
    """At 10 GHz / 550 km LEO, FSPL alone is ~167 dB; total should be 165–220 dB."""
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    loss = layer.compute(datetime(2025, 1, 1, 0, 0, 0))
    valid = loss[~np.isnan(loss)]
    assert valid.min() > 150.0
    assert valid.max() < 250.0


# ---------------------------------------------------------------------------
# Compute — with real IONEX data
# ---------------------------------------------------------------------------

def test_l1_compute_with_ionex_shape():
    cfg = {**BASE_CONFIG, 'ionex_file': IONEX_PATH}
    layer = L1MacroLayer(cfg, origin_lat=39.9042, origin_lon=116.4074)
    loss = layer.compute(datetime(2025, 1, 1, 0, 0, 0))
    assert loss.shape == (256, 256)


def test_l1_compute_with_ionex_spatial_variation():
    """With real TEC, the loss map should not be perfectly uniform."""
    cfg = {**BASE_CONFIG, 'ionex_file': IONEX_PATH, 'frequency_ghz': 1.5}
    layer = L1MacroLayer(cfg, origin_lat=39.9042, origin_lon=116.4074)
    loss = layer.compute(datetime(2025, 1, 1, 6, 0, 0))
    valid = loss[~np.isnan(loss)]
    # At L-band (1.5 GHz) ionospheric loss is non-negligible → spatial variation
    assert valid.std() > 0.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_geometry_zenith():
    """Pixel at sub-satellite point should have elevation ~90°."""
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=0.0, origin_lon=0.0)
    slant, el = layer._calculate_geometry(0.0, 0.0, 0.0, 0.0, 550.0)
    assert slant > 0
    assert el > 85.0


def test_geometry_off_nadir():
    """Pixel 100 km away should have lower elevation than zenith."""
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=0.0, origin_lon=0.0)
    _, el_nadir = layer._calculate_geometry(0.0, 0.0, 0.0, 0.0, 550.0)
    _, el_off = layer._calculate_geometry(0.9, 0.0, 0.0, 0.0, 550.0)  # ~100 km north
    assert el_off < el_nadir


def test_geometry_slant_range_increases_with_distance():
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=0.0, origin_lon=0.0)
    r0, _ = layer._calculate_geometry(0.0, 0.0, 0.0, 0.0, 550.0)
    r1, _ = layer._calculate_geometry(1.0, 0.0, 0.0, 0.0, 550.0)
    assert r1 > r0


# ---------------------------------------------------------------------------
# Vectorized vs scalar consistency
# ---------------------------------------------------------------------------

def test_vectorized_matches_scalar_at_corners():
    """Vectorized compute should match scalar _calculate_geometry at corner pixels."""
    layer = L1MacroLayer(BASE_CONFIG, origin_lat=39.9, origin_lon=116.4)
    loss_map = layer.compute(datetime(2025, 1, 1, 0, 0, 0))

    from src.core.physics import free_space_path_loss, atmospheric_loss, ionospheric_loss
    from src.core.grid import Grid

    grid = layer.grid
    for i, j in [(0, 0), (0, 255), (255, 0), (255, 255)]:
        plat, plon = grid.pixel_to_latlon(i, j)
        slant, el = layer._calculate_geometry(
            plat, plon, layer.origin_lat, layer.origin_lon,
            layer.satellite_altitude_km
        )
        fspl = float(free_space_path_loss(slant, layer.frequency_ghz))
        atm = float(atmospheric_loss(el, layer.frequency_ghz))
        iono = float(ionospheric_loss(layer.frequency_ghz))
        expected = fspl + atm + iono
        assert abs(float(loss_map[i, j]) - expected) < 1e-4
