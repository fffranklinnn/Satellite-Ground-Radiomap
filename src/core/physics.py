"""
Physics utilities for radio propagation calculations.

This module provides fundamental RF propagation formulas including:
- Free Space Path Loss (FSPL)
- Atmospheric attenuation (ITU-R P.618)
- Ionospheric effects (ITU-R P.531)
- Polarization loss
"""

import numpy as np


def free_space_path_loss(distance_km, frequency_ghz):
    """
    Calculate Free Space Path Loss (FSPL).

    Formula: FSPL(dB) = 20*log10(d) + 20*log10(f) + 92.45
    where d is in km and f is in GHz

    Args:
        distance_km: Distance in kilometers (scalar or ndarray)
        frequency_ghz: Frequency in GHz (scalar or ndarray)

    Returns:
        Path loss in dB (same shape as inputs)
    """
    d = np.asarray(distance_km, dtype=float)
    f = np.asarray(frequency_ghz, dtype=float)
    safe_d = np.where(d > 0, d, np.nan)
    safe_f = np.where(f > 0, f, np.nan)
    fspl_db = 20 * np.log10(safe_d) + 20 * np.log10(safe_f) + 92.45
    return fspl_db


def atmospheric_loss(elevation_angle_deg, frequency_ghz, rain_rate_mm_h=0.0):
    """
    Calculate atmospheric attenuation based on ITU-R P.618 (simplified).

    Accepts scalars or numpy arrays.

    Args:
        elevation_angle_deg: Elevation angle in degrees (scalar or ndarray)
        frequency_ghz: Frequency in GHz (scalar or ndarray)
        rain_rate_mm_h: Rain rate in mm/hour (scalar or ndarray)

    Returns:
        Atmospheric loss in dB (same shape as inputs)
    """
    el = np.asarray(elevation_angle_deg, dtype=float)
    freq = np.asarray(frequency_ghz, dtype=float)
    rain = np.asarray(rain_rate_mm_h, dtype=float)

    # Clamp elevation to avoid division by zero; below 0 → large loss
    el_safe = np.clip(el, 0.1, 90.0)
    path_factor = 1.0 / np.sin(np.radians(el_safe))

    zenith_attenuation = 0.1 * (freq / 10.0)
    atm_loss = zenith_attenuation * path_factor

    # Simple rain attenuation
    rain_coeff = 0.01 * (freq / 10.0) ** 2
    atm_loss = atm_loss + rain_coeff * rain * path_factor

    # Mark below-horizon pixels
    atm_loss = np.where(el <= 0, 999.0, atm_loss)
    return atm_loss


def atmospheric_loss_era5(elevation_angle_deg, frequency_ghz, iwv_kg_m2):
    """
    Improved atmospheric attenuation using ERA5 Integrated Water Vapor.

    Uses ITU-R P.836 approximation for wet zenith delay at 10 GHz:
      wet zenith  ≈ 0.0173 * IWV  dB
      dry zenith  ≈ 0.046 dB  (standard atmosphere)
    Path loss = zenith_total / sin(elevation)

    Accepts scalars or numpy arrays.

    Args:
        elevation_angle_deg: Elevation angle in degrees (scalar or ndarray)
        frequency_ghz: Frequency in GHz (scalar or ndarray)
        iwv_kg_m2: Integrated Water Vapor in kg/m² (scalar or ndarray)

    Returns:
        Atmospheric loss in dB (same shape as inputs)
    """
    el = np.asarray(elevation_angle_deg, dtype=float)
    freq = np.asarray(frequency_ghz, dtype=float)
    iwv = np.asarray(iwv_kg_m2, dtype=float)

    el_safe = np.clip(el, 5.0, 90.0)
    sin_el = np.sin(np.radians(el_safe))

    # Scale dry/wet zenith with frequency (relative to 10 GHz reference)
    freq_scale = (freq / 10.0)
    dry_zenith = 0.046 * freq_scale
    wet_zenith = 0.0173 * iwv * freq_scale

    atm_loss = (dry_zenith + wet_zenith) / sin_el
    atm_loss = np.where(el <= 0, 999.0, atm_loss)
    return atm_loss


def ionospheric_loss(frequency_ghz, tec=10.0):
    """
    Calculate ionospheric effects based on ITU-R P.531.

    Effects decrease with frequency squared; negligible above ~3 GHz.
    Accepts scalars or numpy arrays.

    Args:
        frequency_ghz: Frequency in GHz (scalar or ndarray)
        tec: Total Electron Content in TECU (scalar or ndarray, default 10.0)

    Returns:
        Ionospheric loss in dB (same shape as inputs)
    """
    freq = np.asarray(frequency_ghz, dtype=float)
    tec_arr = np.asarray(tec, dtype=float)

    iono_loss = np.where(
        freq > 3.0,
        0.0,
        0.1 * tec_arr / np.where(freq > 0, freq ** 2, 1.0)
    )
    return iono_loss


def polarization_loss(tx_polarization: str, rx_polarization: str,
                      cross_pol_discrimination_db: float = 20.0) -> float:
    """
    Calculate polarization mismatch loss.

    Args:
        tx_polarization: Transmitter polarization ('H', 'V', 'RHCP', 'LHCP')
        rx_polarization: Receiver polarization ('H', 'V', 'RHCP', 'LHCP')
        cross_pol_discrimination_db: Cross-polarization discrimination in dB

    Returns:
        Polarization loss in dB
    """
    # Perfect match
    if tx_polarization == rx_polarization:
        return 0.0

    # Orthogonal linear polarizations (H/V)
    if {tx_polarization, rx_polarization} == {'H', 'V'}:
        return cross_pol_discrimination_db

    # Orthogonal circular polarizations (RHCP/LHCP)
    if {tx_polarization, rx_polarization} == {'RHCP', 'LHCP'}:
        return cross_pol_discrimination_db

    # Linear to circular: 3 dB loss
    if tx_polarization in ['H', 'V'] and rx_polarization in ['RHCP', 'LHCP']:
        return 3.0
    if tx_polarization in ['RHCP', 'LHCP'] and rx_polarization in ['H', 'V']:
        return 3.0

    # Unknown combination
    return 0.0


def db_to_linear(db: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (db / 10.0)


def linear_to_db(linear: float) -> float:
    """Convert linear scale to dB."""
    if linear <= 0:
        return -999.0  # Very large loss
    return 10 * np.log10(linear)


def combine_losses_db(*losses_db: float) -> float:
    """
    Combine multiple losses in dB domain.

    Args:
        *losses_db: Variable number of loss values in dB

    Returns:
        Total loss in dB (sum of all losses)
    """
    return sum(losses_db)
