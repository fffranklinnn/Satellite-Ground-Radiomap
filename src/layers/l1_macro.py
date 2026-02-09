"""
L1 Macro/Space Layer for SG-MRM project.

This layer handles:
- Satellite positioning from TLE data
- Antenna gain pattern projection
- Atmospheric attenuation (ITU-R P.618)
- Ionospheric effects (ITU-R P.531)

Coverage: 256 km, Resolution: 1000 m/pixel
"""

from datetime import datetime
from typing import Dict, Any
import numpy as np

from .base import BaseLayer
from ..core.physics import (
    free_space_path_loss,
    atmospheric_loss,
    ionospheric_loss
)


class L1MacroLayer(BaseLayer):
    """
    L1 Macro/Space Layer: Satellite-to-ground propagation.

    This layer computes the wide-area electromagnetic loss from satellite
    to ground, including free space path loss, atmospheric effects, and
    ionospheric effects.

    Coverage: 256 km × 256 km
    Resolution: 1000 m/pixel
    Grid: 256 × 256 pixels
    """

    def __init__(self, config: Dict[str, Any], origin_lat: float, origin_lon: float):
        """
        Initialize L1 Macro Layer.

        Args:
            config: Layer configuration containing:
                - grid_size: 256
                - coverage_km: 256.0
                - resolution_m: 1000.0
                - frequency_ghz: Operating frequency
                - satellite_altitude_km: Satellite altitude
                - tle_file: Path to TLE file (optional)
            origin_lat: Origin latitude
            origin_lon: Origin longitude
        """
        super().__init__(config, origin_lat, origin_lon)

        # Extract L1-specific parameters
        self.frequency_ghz = config.get('frequency_ghz', 10.0)
        self.satellite_altitude_km = config.get('satellite_altitude_km', 550.0)
        self.tle_file = config.get('tle_file', None)

    def compute(self, timestamp: datetime) -> np.ndarray:
        """
        Compute L1 macro layer loss map.

        For V1.0: Simplified model with fixed satellite position
        For V2.0: Full TLE-based satellite tracking

        Args:
            timestamp: Simulation timestamp

        Returns:
            256×256 array of loss values in dB
        """
        loss_map = np.zeros((self.grid_size, self.grid_size))

        # TODO V1.0: Implement basic satellite positioning
        # TODO V2.0: Implement full TLE-based orbit propagation

        # For now, use a simplified model with satellite at zenith
        satellite_lat = self.origin_lat
        satellite_lon = self.origin_lon
        satellite_alt_km = self.satellite_altitude_km

        # Calculate loss for each pixel
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get pixel coordinates
                pixel_lat, pixel_lon = self.grid.pixel_to_latlon(i, j)

                # Calculate slant range and elevation angle
                slant_range_km, elevation_deg = self._calculate_geometry(
                    pixel_lat, pixel_lon, satellite_lat, satellite_lon, satellite_alt_km
                )

                # Calculate free space path loss
                fspl = free_space_path_loss(slant_range_km, self.frequency_ghz)

                # Calculate atmospheric loss
                atm_loss = atmospheric_loss(elevation_deg, self.frequency_ghz)

                # Calculate ionospheric loss
                iono_loss = ionospheric_loss(self.frequency_ghz)

                # TODO: Add antenna gain pattern
                # antenna_gain = self._get_antenna_gain(elevation_deg, azimuth_deg)

                # Total loss (in dB domain, losses add)
                total_loss = fspl + atm_loss + iono_loss

                loss_map[i, j] = total_loss

        return loss_map

    def _calculate_geometry(self, pixel_lat: float, pixel_lon: float,
                           sat_lat: float, sat_lon: float,
                           sat_alt_km: float) -> tuple:
        """
        Calculate slant range and elevation angle.

        Args:
            pixel_lat: Pixel latitude
            pixel_lon: Pixel longitude
            sat_lat: Satellite latitude
            sat_lon: Satellite longitude
            sat_alt_km: Satellite altitude in km

        Returns:
            Tuple of (slant_range_km, elevation_deg)
        """
        # Simplified geometry calculation
        # TODO: Use proper geodetic calculations for V2.0

        # Calculate ground distance (approximate)
        dlat = pixel_lat - sat_lat
        dlon = pixel_lon - sat_lon
        ground_distance_km = np.sqrt(
            (dlat * 111.0) ** 2 +
            (dlon * 111.0 * np.cos(np.radians(pixel_lat))) ** 2
        )

        # Calculate slant range
        earth_radius_km = 6371.0
        slant_range_km = np.sqrt(
            ground_distance_km ** 2 +
            sat_alt_km ** 2 +
            2 * earth_radius_km * sat_alt_km
        )

        # Calculate elevation angle
        if ground_distance_km < 0.001:
            elevation_deg = 90.0
        else:
            elevation_rad = np.arctan(
                (sat_alt_km + earth_radius_km) / ground_distance_km -
                earth_radius_km / ground_distance_km
            )
            elevation_deg = np.degrees(elevation_rad)

        return slant_range_km, max(0.0, elevation_deg)

    def _get_antenna_gain(self, elevation_deg: float, azimuth_deg: float) -> float:
        """
        Get antenna gain at given angles.

        TODO V1.0: Implement basic antenna pattern
        TODO V2.0: Load antenna pattern from file

        Args:
            elevation_deg: Elevation angle in degrees
            azimuth_deg: Azimuth angle in degrees

        Returns:
            Antenna gain in dBi
        """
        # Placeholder: Simple cosine pattern
        gain_dbi = 20.0 * np.cos(np.radians(elevation_deg))
        return max(0.0, gain_dbi)
