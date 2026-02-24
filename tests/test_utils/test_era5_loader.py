"""Tests for src/utils/era5_loader.py"""

import os
import numpy as np
import pytest


def test_none_when_path_none():
    from src.utils.era5_loader import load_era5
    assert load_era5(None) is None


def test_none_when_file_missing():
    from src.utils.era5_loader import load_era5
    assert load_era5("/nonexistent/path/era5.nc") is None


def test_none_on_corrupt_file(tmp_path):
    from src.utils.era5_loader import load_era5
    bad = tmp_path / "bad.nc"
    bad.write_bytes(b"not a netcdf file")
    assert load_era5(str(bad)) is None


def test_loads_mock_netcdf(tmp_path):
    """Create a minimal ERA5-like NetCDF4 file matching real file structure."""
    pytest.importorskip("netCDF4")
    import netCDF4 as nc

    path = str(tmp_path / "era5_mock.nc")
    ds = nc.Dataset(path, "w")

    # Dimensions — match real file structure
    ds.createDimension("valid_time", 24)
    ds.createDimension("pressure_level", 37)
    ds.createDimension("latitude", 9)    # 39→31°N descending
    ds.createDimension("longitude", 9)   # 104→112°E ascending

    # Variables
    lats_v = ds.createVariable("latitude", "f4", ("latitude",))
    lons_v = ds.createVariable("longitude", "f4", ("longitude",))
    time_v = ds.createVariable("valid_time", "i8", ("valid_time",))
    plev_v = ds.createVariable("pressure_level", "f4", ("pressure_level",))
    q_v = ds.createVariable("q", "f4", ("valid_time", "pressure_level", "latitude", "longitude"))

    # Descending lats (39→31), ascending lons, Unix timestamps for 2025-01-01
    lats_v[:] = np.linspace(39.0, 31.0, 9)
    lons_v[:] = np.linspace(104.0, 112.0, 9)
    base_ts = 1735689600  # 2025-01-01 00:00:00 UTC
    time_v[:] = np.array([base_ts + h * 3600 for h in range(24)], dtype=np.int64)
    # Pressure levels descending (1000→1 hPa) as in real file
    plev_v[:] = np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
                           750, 700, 650, 600, 550, 500, 450, 400, 350, 300,
                           250, 225, 200, 175, 150, 125, 100, 70, 50, 30,
                           20, 10, 7, 5, 3, 2, 1], dtype=np.float32)
    q_data = np.zeros((24, 37, 9, 9), dtype=np.float32)
    for lev in range(37):
        q_data[:, lev, :, :] = max(0.0, 5e-3 * (1.0 - lev / 37.0))
    q_v[:] = q_data
    ds.close()

    from src.utils.era5_loader import load_era5
    era5 = load_era5(path)
    assert era5 is not None

    lats_arr = np.linspace(32.0, 38.0, 6).reshape(2, 3)
    lons_arr = np.linspace(105.0, 111.0, 6).reshape(2, 3)
    iwv = era5.get_iwv(12.0, lats_arr, lons_arr)
    assert iwv.shape == (2, 3)
    assert np.all(iwv >= 0.0)
    assert np.all(np.isfinite(iwv))
