"""
Grid coordinate system for SG-MRM project.

This module provides the Grid class for managing the 256×256 pixel grid
and coordinate transformations between pixel indices and geographic coordinates.
"""

import numpy as np
from typing import Tuple


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
