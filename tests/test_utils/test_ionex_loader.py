"""Tests for src/utils/ionex_loader.py"""

import numpy as np
import pytest

IONEX_PATH = "data/l1_space/data/cddis_data_2025/UPC0OPSRAP_20250020000_01D_15M_GIM.INX.gz"


@pytest.fixture(scope="module")
def loader():
    from src.utils.ionex_loader import IonexLoader
    return IonexLoader(IONEX_PATH)


def test_load_real_file_shape(loader):
    """97 maps, 71 lat bands (87.5→-87.5 step 2.5), 73 lon bands (-180→180 step 5)."""
    assert loader.tec_maps.shape == (97, 71, 73)


def test_epochs_range(loader):
    """Epochs should span 0 to ~86400 seconds (full day at 15-min intervals)."""
    assert loader.epochs[0] == 0
    assert loader.epochs[-1] < 86401
    assert len(loader.epochs) == 97


def test_lat_lon_axes(loader):
    """Latitude axis descending 87.5→-87.5; longitude -180→180."""
    assert loader.lats[0] == pytest.approx(87.5)
    assert loader.lats[-1] == pytest.approx(-87.5)
    assert loader.lons[0] == pytest.approx(-180.0)
    assert loader.lons[-1] == pytest.approx(180.0)


def test_tec_scalar_beijing(loader):
    """TEC over Beijing at midnight should be a physically plausible value."""
    tec = loader.get_tec(0.0, 39.9042, 116.4074)
    assert float(tec) > 0.0
    assert float(tec) < 300.0


def test_tec_array_input_shape(loader):
    """Array input should return same shape."""
    lats = np.linspace(31.0, 39.0, 256).reshape(16, 16)
    lons = np.linspace(104.0, 112.0, 256).reshape(16, 16)
    tec = loader.get_tec(0.0, lats, lons)
    assert tec.shape == (16, 16)


def test_tec_256x256(loader):
    """Full 256×256 grid query should work and return finite values."""
    lats = np.linspace(31.0, 39.0, 256 * 256).reshape(256, 256)
    lons = np.linspace(104.0, 112.0, 256 * 256).reshape(256, 256)
    tec = loader.get_tec(3600.0, lats, lons)
    assert tec.shape == (256, 256)
    assert np.all(np.isfinite(tec))


def test_temporal_interpolation(loader):
    """TEC at midpoint between two epochs should be between the two endpoint values."""
    lat, lon = 39.9, 116.4
    t0 = float(loader.epochs[0])
    t1 = float(loader.epochs[1])
    t_mid = (t0 + t1) / 2.0

    tec0 = float(loader.get_tec(t0, lat, lon))
    tec1 = float(loader.get_tec(t1, lat, lon))
    tec_mid = float(loader.get_tec(t_mid, lat, lon))

    lo, hi = min(tec0, tec1), max(tec0, tec1)
    # Allow small floating-point margin
    assert lo - 0.1 <= tec_mid <= hi + 0.1


def test_tec_clamped_non_negative(loader):
    """Output should never be negative."""
    lats = np.random.uniform(-87.0, 87.0, (10, 10))
    lons = np.random.uniform(-180.0, 180.0, (10, 10))
    tec = loader.get_tec(43200.0, lats, lons)
    assert np.all(tec >= 0.0)
