"""
Physics utilities for radio propagation calculations.

This module provides fundamental RF propagation formulas including:
- Free Space Path Loss (FSPL)
- Atmospheric attenuation (ITU-R P.618)
- Ionospheric effects (ITU-R P.531)
- Polarization loss
"""

import numpy as np


def free_space_path_loss(distance_km: float, frequency_ghz: float) -> float:
    """
    Calculate Free Space Path Loss (FSPL).

    Formula: FSPL(dB) = 20*log10(d) + 20*log10(f) + 92.45
    where d is in km and f is in GHz

    Args:
        distance_km: Distance in kilometers
        frequency_ghz: Frequency in GHz

    Returns:
        Path loss in dB
    """
    if distance_km <= 0 or frequency_ghz <= 0:
        return 0.0

    fspl_db = 20 * np.log10(distance_km) + 20 * np.log10(frequency_ghz) + 92.45
    return fspl_db


def atmospheric_loss(elevation_angle_deg: float, frequency_ghz: float,
                     rain_rate_mm_h: float = 0.0) -> float:
    """
    Calculate atmospheric attenuation based on ITU-R P.618.

    This is a simplified model. For V1.0, we use basic atmospheric absorption.
    For V2.0, this should be replaced with full ITU-R P.618 implementation.

    Args:
        elevation_angle_deg: Elevation angle in degrees (0-90)
        frequency_ghz: Frequency in GHz
        rain_rate_mm_h: Rain rate in mm/hour (default 0 for clear sky)

    Returns:
        Atmospheric loss in dB

    TODO: Implement full ITU-R P.618 model for V2.0
    TODO: Add oxygen and water vapor absorption
    TODO: Add rain attenuation model
    """
    if elevation_angle_deg <= 0:
        return 999.0  # Below horizon

    # Simplified clear-sky atmospheric absorption
    # Typical values: 0.1-0.5 dB at L-band, higher at higher frequencies
    zenith_attenuation = 0.1 * (frequency_ghz / 10.0)  # Rough approximation

    # Path length factor (1/sin(elevation))
    path_factor = 1.0 / np.sin(np.radians(elevation_angle_deg))

    # Basic atmospheric loss
    atm_loss = zenith_attenuation * path_factor

    # Simple rain attenuation (placeholder)
    if rain_rate_mm_h > 0:
        # Simplified rain attenuation: ~0.1 dB/km at 10 GHz for 1 mm/h
        rain_coeff = 0.01 * (frequency_ghz / 10.0) ** 2
        rain_loss = rain_coeff * rain_rate_mm_h * path_factor
        atm_loss += rain_loss

    return atm_loss


def ionospheric_loss(frequency_ghz: float, tec: float = 10.0) -> float:
    """
    Calculate ionospheric effects based on ITU-R P.531.

    The ionosphere primarily causes phase delay and scintillation.
    Effects decrease with frequency squared.

    Args:
        frequency_ghz: Frequency in GHz
        tec: Total Electron Content in TECU (1 TECU = 10^16 electrons/m^2)
             Typical values: 5-50 TECU

    Returns:
        Ionospheric loss in dB

    TODO: Implement full ITU-R P.531 model for V2.0
    TODO: Add scintillation effects
    TODO: Add Faraday rotation
    """
    # Ionospheric effects are negligible above ~3 GHz
    if frequency_ghz > 3.0:
        return 0.0

    # Simplified model: loss proportional to TEC and inversely to f^2
    # This is a placeholder - actual ionospheric effects are complex
    iono_loss = 0.1 * tec / (frequency_ghz ** 2)

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
