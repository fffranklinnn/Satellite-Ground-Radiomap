"""
Grid coordinate system for SG-MRM project.

This module provides:
- Grid class: 256×256 pixel coordinate system (used by L1 src implementation)
- Module-level functions: get_grid_latlon(), get_azimuth_elevation()
  (merged from branch_L1, used by L1MacroLayer directly)
"""

import numpy as np
from typing import Tuple

# ── Constants (merged from branch_L1/src/core/grid.py) ───────────────────────
GRID_SIZE      = 256
L1_COVERAGE    = 256_000.0          # L1 physical coverage (m)
L1_PIXEL_M     = L1_COVERAGE / GRID_SIZE   # 1000 m/px
EARTH_RADIUS_M = 6_371_000.0        # mean Earth radius (m)


class Grid:
    """
    Manages the 256×256 grid coordinate system for radio map generation.

    The grid provides coordinate transformations between pixel indices (i, j)
    and geographic coordinates (latitude, longitude). It supports different
    coverage areas and resolutions for the three physical layers (L1/L2/L3).

    Attributes:
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        grid_size (int): Grid dimension (default 256)
        coverage_km (float): Physical coverage in kilometers
        resolution_m (float): Physical resolution in meters per pixel
    """

    def __init__(self, origin_lat: float, origin_lon: float,
                 grid_size: int = 256, coverage_km: float = 256.0):
        """
        Initialize the grid coordinate system.

        Args:
            origin_lat: Origin latitude in degrees (center of grid)
            origin_lon: Origin longitude in degrees (center of grid)
            grid_size: Number of pixels per dimension (default 256)
            coverage_km: Physical coverage area in kilometers
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.grid_size = grid_size
        self.coverage_km = coverage_km
        self.resolution_m = (coverage_km * 1000) / grid_size

        # Pre-compute conversion factors
        # Approximate: 1 degree latitude ≈ 111 km
        self.lat_per_km = 1.0 / 111.0
        # Longitude varies with latitude
        self.lon_per_km = 1.0 / (111.0 * np.cos(np.radians(origin_lat)))

    def pixel_to_latlon(self, i: int, j: int) -> Tuple[float, float]:
        """
        Convert pixel indices to geographic coordinates.

        Args:
            i: Row index (0 to grid_size-1)
            j: Column index (0 to grid_size-1)

        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        # Calculate offset from center in pixels
        center_pixel = self.grid_size / 2.0
        di = i - center_pixel
        dj = j - center_pixel

        # Convert to kilometers
        offset_km_lat = -di * self.resolution_m / 1000.0  # Negative: i increases downward
        offset_km_lon = dj * self.resolution_m / 1000.0

        # Convert to degrees
        lat = self.origin_lat + offset_km_lat * self.lat_per_km
        lon = self.origin_lon + offset_km_lon * self.lon_per_km

        return lat, lon

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel indices.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Tuple of (i, j) pixel indices
        """
        # Calculate offset in degrees
        dlat = lat - self.origin_lat
        dlon = lon - self.origin_lon

        # Convert to kilometers
        offset_km_lat = dlat / self.lat_per_km
        offset_km_lon = dlon / self.lon_per_km

        # Convert to pixels
        center_pixel = self.grid_size / 2.0
        i = int(center_pixel - offset_km_lat * 1000.0 / self.resolution_m)
        j = int(center_pixel + offset_km_lon * 1000.0 / self.resolution_m)

        return i, j

    def get_pixel_centers(self) -> np.ndarray:
        """
        Get geographic coordinates of all pixel centers.

        Returns:
            Array of shape (grid_size, grid_size, 2) where [:, :, 0] is latitude
            and [:, :, 1] is longitude
        """
        coords = np.zeros((self.grid_size, self.grid_size, 2))

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                coords[i, j] = self.pixel_to_latlon(i, j)

        return coords

    def get_distance_matrix(self) -> np.ndarray:
        """
        Calculate distance from origin for each pixel.

        Returns:
            Array of shape (grid_size, grid_size) with distances in kilometers
        """
        distances = np.zeros((self.grid_size, self.grid_size))
        center = self.grid_size / 2.0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                di = i - center
                dj = j - center
                distance_pixels = np.sqrt(di**2 + dj**2)
                distances[i, j] = distance_pixels * self.resolution_m / 1000.0

        return distances

    def is_within_bounds(self, i: int, j: int) -> bool:
        """
        Check if pixel indices are within grid bounds.

        Args:
            i: Row index
            j: Column index

        Returns:
            True if indices are valid, False otherwise
        """
        return 0 <= i < self.grid_size and 0 <= j < self.grid_size


# ── Module-level functions (merged from branch_L1/src/core/grid.py) ───────────
# Used directly by L1MacroLayer for vectorized grid generation.

def get_grid_latlon(origin_lat: float, origin_lon: float,
                    coverage_m: float = L1_COVERAGE,
                    grid_size: int = GRID_SIZE):
    """
    Generate a grid_size×grid_size geographic grid centred on origin.

    Args:
        origin_lat: Centre latitude in degrees
        origin_lon: Centre longitude in degrees
        coverage_m: Physical coverage in metres (default 256 km)
        grid_size:  Number of pixels per side (default 256)

    Returns:
        lat_grid : (grid_size, grid_size) latitude array in degrees
        lon_grid : (grid_size, grid_size) longitude array in degrees
        x_m      : (grid_size, grid_size) eastward offset array in metres
        y_m      : (grid_size, grid_size) northward offset array in metres
    """
    half    = coverage_m / 2.0
    pixel_m = coverage_m / grid_size

    # Pixel-centre offsets relative to origin (metres)
    offsets = np.arange(grid_size) * pixel_m - half + pixel_m / 2.0
    x_m, y_m = np.meshgrid(offsets, offsets[::-1])   # x→east, y↑north

    # Flat-Earth approximation (valid for < 500 km)
    lat_rad = np.deg2rad(origin_lat)
    dlat = np.rad2deg(y_m / EARTH_RADIUS_M)
    dlon = np.rad2deg(x_m / (EARTH_RADIUS_M * np.cos(lat_rad)))

    lat_grid = origin_lat + dlat
    lon_grid = origin_lon + dlon

    return lat_grid, lon_grid, x_m, y_m


def get_azimuth_elevation(sat_x_m: float, sat_y_m: float, sat_alt_m: float,
                          grid_x_m: np.ndarray, grid_y_m: np.ndarray):
    """
    Compute azimuth, elevation, and slant range from each grid pixel to satellite.

    Args:
        sat_x_m:   Satellite eastward offset from origin in metres
        sat_y_m:   Satellite northward offset from origin in metres
        sat_alt_m: Satellite altitude in metres
        grid_x_m:  (grid_size, grid_size) eastward offset array in metres
        grid_y_m:  (grid_size, grid_size) northward offset array in metres

    Returns:
        azimuth_deg   : (grid_size, grid_size) azimuth in degrees (0=N, clockwise)
        elevation_deg : (grid_size, grid_size) elevation in degrees
        slant_range_m : (grid_size, grid_size) slant range in metres
    """
    dx = grid_x_m - sat_x_m
    dy = grid_y_m - sat_y_m
    horiz_dist = np.sqrt(dx ** 2 + dy ** 2)

    elevation_deg = np.rad2deg(np.arctan2(sat_alt_m, horiz_dist))
    azimuth_deg   = np.rad2deg(np.arctan2(dx, dy)) % 360.0
    slant_range_m = np.sqrt(horiz_dist ** 2 + sat_alt_m ** 2)

    return azimuth_deg, elevation_deg, slant_range_m
