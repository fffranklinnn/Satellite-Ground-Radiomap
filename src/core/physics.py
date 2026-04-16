"""
Physics utilities for radio propagation calculations.

This module provides fundamental RF propagation formulas including:
- Free Space Path Loss (FSPL)
- Atmospheric attenuation (ITU-R P.618)
- Ionospheric effects (ITU-R P.531)
- Polarization loss
- Phased array antenna gain (merged from branch_L1)
"""

from typing import Optional

import numpy as np

# ── Physical constants (merged from branch_L1) ────────────────────────────────
SPEED_OF_LIGHT = 2.998e8       # m/s
BOLTZMANN_K    = 1.380649e-23  # J/K

# Approximate ITU-R P.838 coefficient tables (frequency in GHz).
# Used for a frequency/polarization-aware rain attenuation model.
_RAIN_COEFF_FREQ_GHZ = np.array(
    [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0],
    dtype=float
)
_RAIN_KH = np.array(
    [3.87e-5, 1.54e-4, 6.50e-4, 1.75e-3, 3.50e-3, 6.20e-3, 1.01e-2,
     1.88e-2, 7.51e-2, 1.87e-1, 3.50e-1, 5.36e-1],
    dtype=float
)
_RAIN_AH = np.array(
    [0.912, 0.963, 1.121, 1.241, 1.320, 1.276, 1.217, 1.154,
     1.099, 1.021, 0.960, 0.900],
    dtype=float
)
_RAIN_KV = np.array(
    [3.52e-5, 1.38e-4, 5.90e-4, 1.55e-3, 3.15e-3, 5.65e-3, 9.00e-3,
     1.68e-2, 6.30e-2, 1.67e-1, 3.10e-1, 4.80e-1],
    dtype=float
)
_RAIN_AV = np.array(
    [0.880, 0.923, 1.075, 1.190, 1.265, 1.230, 1.190, 1.120,
     1.065, 1.000, 0.940, 0.880],
    dtype=float
)


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


def _rain_coefficients(frequency_ghz):
    """Interpolate rain-model k/alpha coefficients at the requested frequency."""
    freq = np.asarray(frequency_ghz, dtype=float)
    freq_clip = np.clip(freq, _RAIN_COEFF_FREQ_GHZ[0], _RAIN_COEFF_FREQ_GHZ[-1])

    k_h = np.interp(freq_clip, _RAIN_COEFF_FREQ_GHZ, _RAIN_KH)
    a_h = np.interp(freq_clip, _RAIN_COEFF_FREQ_GHZ, _RAIN_AH)
    k_v = np.interp(freq_clip, _RAIN_COEFF_FREQ_GHZ, _RAIN_KV)
    a_v = np.interp(freq_clip, _RAIN_COEFF_FREQ_GHZ, _RAIN_AV)
    return k_h, a_h, k_v, a_v


def rain_specific_attenuation_db_per_km(frequency_ghz,
                                        rain_rate_mm_h,
                                        elevation_angle_deg=45.0,
                                        polarization_tilt_deg=45.0):
    """
    Approximate specific attenuation gamma_R (dB/km) from rain intensity.

    The model is frequency- and polarization-aware (table-interpolated k/alpha).
    """
    freq = np.asarray(frequency_ghz, dtype=float)
    rain = np.asarray(rain_rate_mm_h, dtype=float)
    el = np.asarray(elevation_angle_deg, dtype=float)

    # Interpolate ITU-style horizontal/vertical coefficients to target freq.
    k_h, a_h, k_v, a_v = _rain_coefficients(freq)

    cos_el_sq = np.cos(np.radians(np.clip(el, 0.0, 90.0))) ** 2
    cos_2tau = np.cos(2.0 * np.radians(float(polarization_tilt_deg)))

    # Blend H/V polarization components to equivalent link polarization.
    k = 0.5 * (k_h + k_v + (k_h - k_v) * cos_el_sq * cos_2tau)
    k = np.maximum(k, 1e-10)

    num = (
        k_h * a_h + k_v * a_v +
        (k_h * a_h - k_v * a_v) * cos_el_sq * cos_2tau
    )
    alpha = num / (2.0 * k)

    rain_safe = np.maximum(rain, 0.0)
    return k * np.power(rain_safe, alpha)


def rain_attenuation_slant_path_db(elevation_angle_deg,
                                   frequency_ghz,
                                   rain_rate_mm_h,
                                   station_alt_km=0.4,
                                   rain_height_km=5.0,
                                   polarization_tilt_deg=45.0):
    """
    Approximate slant-path rain attenuation (dB).

    Uses:
    - frequency/polarization-aware specific attenuation gamma_R (dB/km)
    - effective slant path with a reduction factor inspired by ITU-R P.618
    """
    el = np.asarray(elevation_angle_deg, dtype=float)
    freq = np.asarray(frequency_ghz, dtype=float)
    rain = np.asarray(rain_rate_mm_h, dtype=float)

    el_safe = np.clip(el, 0.1, 90.0)
    sin_el = np.sin(np.radians(el_safe))

    # Effective rain layer depth above station.
    rain_layer_km = max(float(rain_height_km) - float(station_alt_km), 0.1)
    slant_len_km = rain_layer_km / sin_el

    gamma_r = rain_specific_attenuation_db_per_km(
        freq,
        rain,
        elevation_angle_deg=el_safe,
        polarization_tilt_deg=polarization_tilt_deg,
    )

    freq_safe = np.maximum(freq, 1e-3)
    term = np.maximum(slant_len_km * gamma_r / freq_safe, 0.0)
    # Empirical reduction factor approximates inhomogeneous rain cell effects.
    reduction = 1.0 / (
        1.0 +
        0.78 * np.sqrt(term) -
        0.38 * (1.0 - np.exp(-2.0 * slant_len_km))
    )
    reduction = np.clip(reduction, 0.05, 1.0)

    attenuation = gamma_r * slant_len_km * reduction
    attenuation = np.where(rain <= 0.0, 0.0, attenuation)
    attenuation = np.where(el <= 0.0, 999.0, attenuation)
    return attenuation


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

    # Clamp elevation to avoid division by zero; below 0 -> large loss
    el_safe = np.clip(el, 0.1, 90.0)
    path_factor = 1.0 / np.sin(np.radians(el_safe))

    # Baseline gaseous attenuation (clear-sky), frequency dependent.
    # This is an engineering approximation, not a full line-by-line model.
    dry_zenith = 0.035 * (freq / 10.0)
    wet_zenith = 0.012 * (freq / 10.0)
    atm_loss = (dry_zenith + wet_zenith) * path_factor

    # Refined rain component (frequency/polarization/slant-path aware).
    rain_loss = rain_attenuation_slant_path_db(
        elevation_angle_deg=el,
        frequency_ghz=freq,
        rain_rate_mm_h=rain,
    )
    atm_loss = atm_loss + np.clip(rain_loss, 0.0, 300.0)

    # Mark below-horizon pixels
    atm_loss = np.where(el <= 0, 999.0, atm_loss)
    return atm_loss


def atmospheric_loss_era5(elevation_angle_deg, frequency_ghz, iwv_kg_m2, rain_rate_mm_h=0.0):
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
    rain = np.asarray(rain_rate_mm_h, dtype=float)

    el_safe = np.clip(el, 5.0, 90.0)
    sin_el = np.sin(np.radians(el_safe))

    # Convert ERA5 IWV to wet-zenith attenuation and scale to operating freq.
    freq_scale = (freq / 10.0)
    dry_zenith = 0.046 * freq_scale
    wet_zenith = 0.0173 * iwv * freq_scale

    atm_loss = (dry_zenith + wet_zenith) / sin_el

    rain_loss = rain_attenuation_slant_path_db(
        elevation_angle_deg=el,
        frequency_ghz=freq,
        rain_rate_mm_h=rain,
    )
    atm_loss = atm_loss + np.clip(rain_loss, 0.0, 300.0)
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


# ── Branch-L1 antenna / link-budget functions ─────────────────────────────────
# Reference: ITU-R P.618, Remote Sensing paper (You Fu et al., 2026)
# Ku-band 14.5 GHz, 31×31 rectangular phased array

def fspl_db(slant_range_m: np.ndarray, freq_hz: float) -> np.ndarray:
    """
    Free Space Path Loss (meter / Hz variant).

    FSPL(dB) = 20·log10(4π·d·f / c)

    Args:
        slant_range_m: Slant range in meters (scalar or ndarray)
        freq_hz:       Carrier frequency in Hz

    Returns:
        Path loss in dB (same shape as slant_range_m)
    """
    d = np.maximum(np.asarray(slant_range_m, dtype=float), 1.0)
    return 20.0 * np.log10(4.0 * np.pi * d * freq_hz / SPEED_OF_LIGHT)


def polarization_loss_db(pol_mismatch_angle_deg: float = 0.0) -> float:
    """
    Polarization Loss Factor (PLF).

    PLF(dB) = -20·log10|cos(Δψ)|
    Same polarization (Δψ=0) → 0 dB loss.

    Args:
        pol_mismatch_angle_deg: Polarization mismatch angle in degrees

    Returns:
        Polarization loss in dB (positive value)
    """
    angle_rad = np.deg2rad(pol_mismatch_angle_deg)
    cos_val = max(abs(float(np.cos(angle_rad))), 1e-10)
    return float(-20.0 * np.log10(cos_val))


def gaussian_beam_gain_db(theta_az_deg: np.ndarray,
                          theta_el_deg: np.ndarray,
                          peak_gain_db: float,
                          hpbw_az_deg: float,
                          hpbw_el_deg: float) -> np.ndarray:
    """
    2-D Gaussian beam antenna gain model.

    G(θ_az, θ_el) = G_peak - 12·[(θ_az/HPBW_az)² + (θ_el/HPBW_el)²]  (dB)

    Args:
        theta_az_deg: Azimuth offset from boresight in degrees (ndarray)
        theta_el_deg: Elevation offset from boresight in degrees (ndarray)
        peak_gain_db: Peak gain in dBi
        hpbw_az_deg:  Azimuth half-power beam width in degrees
        hpbw_el_deg:  Elevation half-power beam width in degrees

    Returns:
        Antenna gain array in dBi (same shape as theta_az_deg)
    """
    return peak_gain_db - 12.0 * (
        (theta_az_deg / hpbw_az_deg) ** 2 +
        (theta_el_deg / hpbw_el_deg) ** 2
    )


def parabolic_rolloff_gain_db(theta_deg: np.ndarray,
                              peak_gain_db: float,
                              theta_3db_deg: float,
                              min_gain_db: Optional[float] = None) -> np.ndarray:
    """
    Axisymmetric antenna roll-off model.

    G(θ) = G_peak - 3·(θ / θ_3dB)^2  (dB)

    This form is convenient for satellite beam-footprint rendering because it
    produces smooth off-axis contours and preserves the 3 dB point explicitly.

    Args:
        theta_deg:     Off-axis angle from boresight in degrees (ndarray)
        peak_gain_db:  Peak boresight gain in dBi
        theta_3db_deg: Off-axis 3 dB angle in degrees
        min_gain_db:   Optional gain floor (dBi)

    Returns:
        Antenna gain array in dBi (same shape as theta_deg)
    """
    theta = np.maximum(np.asarray(theta_deg, dtype=float), 0.0)
    theta_3db = max(float(theta_3db_deg), 1e-6)
    gain = float(peak_gain_db) - 3.0 * (theta / theta_3db) ** 2
    if min_gain_db is not None:
        gain = np.maximum(gain, float(min_gain_db))
    return gain


def phased_array_peak_gain_db(n_elements: int,
                               element_gain_db: float = 5.0) -> float:
    """
    Estimate peak gain of a phased array antenna.

    G_peak(dB) = 10·log10(N) + G_element(dBi)

    Args:
        n_elements:     Total number of array elements (e.g. 31×31 = 961)
        element_gain_db: Single element gain in dBi (default 5 dBi)

    Returns:
        Peak array gain in dBi
    """
    return float(10.0 * np.log10(n_elements) + element_gain_db)


def phased_array_hpbw_deg(n_row: int, n_col: int,
                           wavelength_m: float,
                           element_spacing_m: float) -> tuple:
    """
    Estimate HPBW of a rectangular phased array (uniform excitation).

    HPBW ≈ 0.886 · λ / (N · d)  [radians] → converted to degrees

    Args:
        n_row:              Number of rows
        n_col:              Number of columns
        wavelength_m:       Operating wavelength in meters
        element_spacing_m:  Element spacing in meters (typically λ/2)

    Returns:
        Tuple (hpbw_az_deg, hpbw_el_deg)
    """
    hpbw_az = np.rad2deg(0.886 * wavelength_m / (n_col * element_spacing_m))
    hpbw_el = np.rad2deg(0.886 * wavelength_m / (n_row * element_spacing_m))
    return float(hpbw_az), float(hpbw_el)


def thermal_noise_power_dbw(bandwidth_hz: float,
                             temp_k: float = 290.0) -> float:
    """
    Thermal noise power  Pn = k·T·B.

    Args:
        bandwidth_hz: System bandwidth in Hz
        temp_k:       System noise temperature in K (default 290 K per ITU)

    Returns:
        Thermal noise power in dBW
    """
    pn_w = BOLTZMANN_K * temp_k * bandwidth_hz
    return float(10.0 * np.log10(pn_w))
