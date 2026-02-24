"""
ERA5 reanalysis data loader (optional).

Loads ERA5 pressure-level NetCDF data and computes Integrated Water Vapor (IWV)
for use in improved atmospheric attenuation estimates.

Returns None gracefully when the file is absent or netCDF4/xarray is not installed.
"""

import os
import numpy as np


def load_era5(path):
    """
    Load ERA5 NetCDF file and return an Era5Loader, or None on any failure.

    Args:
        path: Path to ERA5 NetCDF file, or None

    Returns:
        Era5Loader instance, or None
    """
    if not path or not os.path.exists(path):
        return None
    try:
        return Era5Loader(path)
    except Exception:
        return None


class Era5Loader:
    """
    Provides Integrated Water Vapor (IWV) interpolated from ERA5 data.

    IWV is computed by vertically integrating specific humidity over all
    pressure levels: IWV = (1/g) * |integral(q dp)|  [kg/m²]
    """

    def __init__(self, path: str):
        from scipy.interpolate import RegularGridInterpolator

        # Try netCDF4 first, fall back to xarray
        try:
            import netCDF4 as nc
            ds = nc.Dataset(path, 'r')
            q = np.array(ds.variables['q'][:])           # (T, lev, lat, lon)
            lats = np.array(ds.variables['latitude'][:])
            lons = np.array(ds.variables['longitude'][:])
            p_hpa = np.array(ds.variables['pressure_level'][:])  # read from file
            valid_time = np.array(ds.variables['valid_time'][:])  # Unix seconds
            ds.close()
        except ImportError:
            import xarray as xr
            ds = xr.open_dataset(path)
            q = ds['q'].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            p_hpa = ds['pressure_level'].values
            valid_time = ds['valid_time'].values.astype('int64') // 10**9
            ds.close()

        # Convert Unix timestamps → hours-of-day (0.0–23.0)
        times_h = (valid_time % 86400) / 3600.0

        # Compute IWV: (1/g) * |trapz(q, p_Pa, axis=lev)|
        # pressure_level may be descending (1000→1 hPa); trapz handles sign correctly
        g = 9.80665
        p_pa = p_hpa * 100.0  # hPa → Pa
        iwv = np.abs(np.trapezoid(q, p_pa, axis=1)) / g  # (T, lat, lon) kg/m²

        # Ensure lats ascending for RegularGridInterpolator
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            iwv = iwv[:, ::-1, :]

        self._interp = RegularGridInterpolator(
            (times_h, lats, lons), iwv,
            method='linear', bounds_error=False,
            fill_value=float(np.nanmedian(iwv))
        )
        self._median_iwv = float(np.nanmedian(iwv))

    def get_iwv(self, hour_utc: float,
                lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Interpolate IWV at given time and locations.

        Args:
            hour_utc : UTC hour (0.0–23.0)
            lat      : latitude array (any shape), degrees
            lon      : longitude array (same shape), degrees

        Returns:
            IWV in kg/m², same shape as lat/lon
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        orig_shape = lat.shape

        t = np.full(lat.size, float(hour_utc))
        pts = np.column_stack([t, lat.ravel(), lon.ravel()])
        result = self._interp(pts)
        return result.reshape(orig_shape)
