"""
Ionosphere geometry and Faraday-rotation helpers.

These utilities provide:
- Thin-shell IPP projection (ground -> ionospheric pierce point)
- Slant-factor mapping from VTEC to STEC
- Faraday rotation approximation for linear polarization mismatch
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
FARADAY_CONST = 2.36e4  # SI form with f in Hz, TEC in el/m^2, B in Tesla


def ipp_from_ground(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    az_to_sat_deg: np.ndarray,
    el_to_sat_deg: np.ndarray,
    shell_height_km: float = 350.0,
    max_mapping_factor: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project ground points onto a thin ionospheric shell.

    Args:
        lat_deg: Ground latitude array (degrees)
        lon_deg: Ground longitude array (degrees)
        az_to_sat_deg: Azimuth from ground toward satellite (degrees, north=0, clockwise)
        el_to_sat_deg: Elevation from ground toward satellite (degrees)
        shell_height_km: Ionospheric shell height
        max_mapping_factor: Clamp for obliquity/mapping factor

    Returns:
        ipp_lat_deg, ipp_lon_deg, mapping_factor
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    az = np.asarray(az_to_sat_deg, dtype=float)
    el = np.asarray(el_to_sat_deg, dtype=float)

    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    az_r = np.deg2rad(az)
    el_r = np.deg2rad(np.clip(el, 0.1, 89.9))

    ratio = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + float(shell_height_km))
    chi = np.arcsin(np.clip(ratio * np.cos(el_r), -1.0, 1.0))
    psi = (np.pi / 2.0) - el_r - chi

    sin_lat_ipp = np.sin(lat_r) * np.cos(psi) + np.cos(lat_r) * np.sin(psi) * np.cos(az_r)
    lat_ipp_r = np.arcsin(np.clip(sin_lat_ipp, -1.0, 1.0))
    lon_ipp_r = lon_r + np.arctan2(
        np.sin(psi) * np.sin(az_r),
        np.cos(lat_r) * np.cos(psi) - np.sin(lat_r) * np.sin(psi) * np.cos(az_r),
    )

    ipp_lat = np.rad2deg(lat_ipp_r)
    ipp_lon = (np.rad2deg(lon_ipp_r) + 540.0) % 360.0 - 180.0

    mapping = 1.0 / np.sqrt(np.maximum(1.0 - (ratio * np.cos(el_r)) ** 2, 1e-8))
    mapping = np.clip(mapping, 1.0, float(max_mapping_factor))
    return ipp_lat, ipp_lon, mapping


def faraday_rotation_deg(stec_tecu: np.ndarray, b_parallel_t: float, freq_hz: float) -> np.ndarray:
    """
    Approximate Faraday rotation angle for a path.

    Args:
        stec_tecu: Slant TEC in TECU
        b_parallel_t: Path-parallel geomagnetic field component in Tesla
        freq_hz: Carrier frequency in Hz

    Returns:
        Rotation angle in degrees (signed).
    """
    stec = np.asarray(stec_tecu, dtype=float) * 1e16  # TECU -> electrons/m^2
    f2 = max(float(freq_hz), 1.0) ** 2
    theta_rad = FARADAY_CONST * stec * float(b_parallel_t) / f2
    return np.rad2deg(theta_rad)


def polarization_mismatch_loss_db(mismatch_deg: np.ndarray) -> np.ndarray:
    """
    Vectorized linear-polarization mismatch loss.
    """
    mismatch = np.asarray(mismatch_deg, dtype=float)
    cos_val = np.abs(np.cos(np.deg2rad(mismatch)))
    return -20.0 * np.log10(np.maximum(cos_val, 1e-10))
