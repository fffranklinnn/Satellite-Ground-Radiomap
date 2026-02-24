"""
L1 Macro/Space Layer for SG-MRM project.

This layer handles:
- Satellite positioning (V1.0: fixed at zenith; V2.0: TLE-based)
- Free Space Path Loss
- Atmospheric attenuation (ITU-R P.618; ERA5-enhanced when data available)
- Ionospheric effects (ITU-R P.531; real TEC from IONEX when data available)

Coverage: 256 km, Resolution: 1000 m/pixel
"""

from datetime import datetime
from typing import Dict, Any
import numpy as np

from .base import BaseLayer
from ..core.physics import (
    free_space_path_loss,
    atmospheric_loss,
    atmospheric_loss_era5,
    ionospheric_loss,
)
from ..utils.ionex_loader import IonexLoader
from ..utils.era5_loader import load_era5
from ..utils.tle_loader import TleLoader


class L1MacroLayer(BaseLayer):
    """
    L1 Macro/Space Layer: Satellite-to-ground propagation.

    Computes wide-area electromagnetic loss from satellite to ground,
    including free space path loss, atmospheric effects, and ionospheric
    effects. All calculations are fully vectorized over the 256×256 grid.

    Coverage: 256 km × 256 km
    Resolution: 1000 m/pixel
    Grid: 256 × 256 pixels
    """

    def __init__(self, config: Dict[str, Any], origin_lat: float, origin_lon: float):
        """
        Initialize L1 Macro Layer.

        Args:
            config: Layer configuration. Recognized keys:
                grid_size, coverage_km, resolution_m  (required by BaseLayer)
                frequency_ghz          : operating frequency (default 10.0)
                satellite_altitude_km  : LEO altitude (default 550.0)
                tec                    : fallback TEC in TECU (default 10.0)
                rain_rate_mm_h         : fallback rain rate (default 0.0)
                ionex_file             : path to IONEX .gz file (optional)
                era5_file              : path to ERA5 NetCDF file (optional)
                tle_file               : path to TLE file (optional; enables multi-sat mode)
            origin_lat: Origin latitude in degrees
            origin_lon: Origin longitude in degrees
        """
        super().__init__(config, origin_lat, origin_lon)

        self.frequency_ghz = config.get('frequency_ghz', 10.0)
        self.satellite_altitude_km = config.get('satellite_altitude_km', 550.0)

        # Load real data sources (all optional — None on failure)
        ionex_path = config.get('ionex_file')
        self.ionex = IonexLoader(ionex_path) if ionex_path else None

        self.era5 = load_era5(config.get('era5_file'))

        tle_path = config.get('tle_file')
        self.tle_loader = TleLoader(tle_path) if tle_path else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, timestamp: datetime) -> np.ndarray:
        """
        Compute L1 macro layer loss map (fully vectorized).

        When tle_loader is present: iterates over all Starlink satellites,
        computes per-pixel loss for each visible satellite, and returns the
        minimum loss (best link) per pixel.

        When tle_loader is absent: falls back to single satellite fixed at zenith.

        Args:
            timestamp: Simulation timestamp

        Returns:
            256×256 array of total loss values in dB (NaN for no coverage)
        """
        coords = self.grid.get_pixel_centers()
        lats = coords[:, :, 0]   # (256, 256)
        lons = coords[:, :, 1]

        epoch_sec = (timestamp.hour * 3600
                     + timestamp.minute * 60
                     + timestamp.second)

        # TEC map (shared across all satellites)
        if self.ionex is not None:
            tec = self.ionex.get_tec(epoch_sec, lats, lons)
        else:
            tec = np.full_like(lats, self.config.get('tec', 10.0))

        # IWV map (shared across all satellites)
        if self.era5 is not None:
            iwv = self.era5.get_iwv(timestamp.hour, lats, lons)
        else:
            iwv = None

        if self.tle_loader is not None:
            return self._compute_constellation(
                timestamp, lats, lons, tec, iwv)
        else:
            return self._compute_single_sat(
                lats, lons, tec, iwv,
                self.origin_lat, self.origin_lon,
                self.satellite_altitude_km)

    def _compute_single_sat(self, lats, lons, tec, iwv,
                            sat_lat, sat_lon, sat_alt_km):
        """Compute loss map for a single satellite position."""
        slant_range_km, elevation_deg = self._calc_geometry_vec(
            lats, lons, sat_lat, sat_lon, sat_alt_km)

        fspl = free_space_path_loss(slant_range_km, self.frequency_ghz)
        iono = ionospheric_loss(self.frequency_ghz, tec)

        if iwv is not None:
            atm = atmospheric_loss_era5(elevation_deg, self.frequency_ghz, iwv)
        else:
            atm = atmospheric_loss(
                elevation_deg, self.frequency_ghz,
                self.config.get('rain_rate_mm_h', 0.0))

        loss_map = fspl + atm + iono
        loss_map = np.where(elevation_deg <= 5.0, np.nan, loss_map)
        return loss_map

    def _compute_constellation(self, timestamp, lats, lons, tec, iwv):
        """
        Compute best-satellite loss map over the full Starlink constellation.

        Processes satellites in batches of 50 to limit memory usage.
        Returns per-pixel minimum loss (NaN where no satellite is visible).
        """
        sat_lats, sat_lons, sat_alts = self.tle_loader.get_geodetic(timestamp)

        # Filter to plausible LEO altitudes
        mask = (sat_alts >= 300.0) & (sat_alts <= 700.0)
        sat_lats = sat_lats[mask]
        sat_lons = sat_lons[mask]
        sat_alts = sat_alts[mask]

        best = np.full_like(lats, np.inf)
        BATCH = 50

        for start in range(0, len(sat_lats), BATCH):
            sl = slice(start, start + BATCH)
            for slat, slon, salt in zip(sat_lats[sl], sat_lons[sl], sat_alts[sl]):
                slant, el = self._calc_geometry_vec(lats, lons, slat, slon, salt)
                # Skip satellites below 5° elevation for this pixel
                visible = el > 5.0
                if not np.any(visible):
                    continue

                fspl = free_space_path_loss(slant, self.frequency_ghz)
                iono = ionospheric_loss(self.frequency_ghz, tec)
                if iwv is not None:
                    atm = atmospheric_loss_era5(el, self.frequency_ghz, iwv)
                else:
                    atm = atmospheric_loss(
                        el, self.frequency_ghz,
                        self.config.get('rain_rate_mm_h', 0.0))

                loss = np.where(visible, fspl + atm + iono, np.inf)
                best = np.minimum(best, loss)

        # Pixels with no visible satellite → NaN
        result = np.where(np.isinf(best), np.nan, best)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calc_geometry_vec(self, pixel_lat: np.ndarray, pixel_lon: np.ndarray,
                           sat_lat: float, sat_lon: float,
                           sat_alt_km: float):
        """
        Vectorized slant range and elevation angle calculation.

        Uses spherical law of cosines for the central angle ψ between the
        sub-satellite point and each ground pixel, then derives slant range
        and elevation from the resulting triangle.

        Args:
            pixel_lat : (H, W) latitude array in degrees
            pixel_lon : (H, W) longitude array in degrees
            sat_lat   : sub-satellite latitude in degrees
            sat_lon   : sub-satellite longitude in degrees
            sat_alt_km: satellite altitude above Earth surface in km

        Returns:
            slant_range_km : (H, W) array in km
            elevation_deg  : (H, W) array in degrees
        """
        R_E = 6371.0  # Earth radius km
        r_sat = R_E + sat_alt_km

        # Central angle ψ via spherical law of cosines
        lat1 = np.radians(pixel_lat)
        lon1 = np.radians(pixel_lon)
        lat2 = np.radians(sat_lat)
        lon2 = np.radians(sat_lon)

        cos_psi = (np.sin(lat1) * np.sin(lat2)
                   + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2))
        cos_psi = np.clip(cos_psi, -1.0, 1.0)

        # Slant range from law of cosines in the Earth-center triangle
        slant_range_km = np.sqrt(R_E**2 + r_sat**2 - 2 * R_E * r_sat * cos_psi)

        # Elevation angle: angle at the ground station
        # sin(elevation) = (r_sat * cos_psi - R_E) / slant_range
        sin_el = (r_sat * cos_psi - R_E) / np.where(slant_range_km > 0,
                                                      slant_range_km, 1.0)
        sin_el = np.clip(sin_el, -1.0, 1.0)
        elevation_deg = np.degrees(np.arcsin(sin_el))

        return slant_range_km, elevation_deg

    def _calculate_geometry(self, pixel_lat: float, pixel_lon: float,
                            sat_lat: float, sat_lon: float,
                            sat_alt_km: float) -> tuple:
        """
        Scalar geometry calculation (kept for backward compatibility / tests).

        Returns:
            Tuple of (slant_range_km, elevation_deg)
        """
        slant, el = self._calc_geometry_vec(
            np.array([[pixel_lat]]), np.array([[pixel_lon]]),
            sat_lat, sat_lon, sat_alt_km
        )
        return float(slant[0, 0]), float(el[0, 0])
