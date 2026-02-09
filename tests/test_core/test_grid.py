"""Unit tests for core.grid module."""

import pytest
import numpy as np
from src.core.grid import Grid


def test_grid_initialization():
    """Test grid initialization."""
    grid = Grid(origin_lat=39.9, origin_lon=116.4, grid_size=256, coverage_km=256.0)

    assert grid.origin_lat == 39.9
    assert grid.origin_lon == 116.4
    assert grid.grid_size == 256
    assert grid.coverage_km == 256.0
    assert grid.resolution_m == 1000.0


def test_pixel_to_latlon():
    """Test pixel to lat/lon conversion."""
    grid = Grid(origin_lat=0.0, origin_lon=0.0, grid_size=256, coverage_km=256.0)

    # Center pixel should be at origin
    lat, lon = grid.pixel_to_latlon(128, 128)
    assert abs(lat - 0.0) < 0.01
    assert abs(lon - 0.0) < 0.01


def test_latlon_to_pixel():
    """Test lat/lon to pixel conversion."""
    grid = Grid(origin_lat=0.0, origin_lon=0.0, grid_size=256, coverage_km=256.0)

    # Origin should map to center pixel
    i, j = grid.latlon_to_pixel(0.0, 0.0)
    assert abs(i - 128) < 1
    assert abs(j - 128) < 1


def test_is_within_bounds():
    """Test bounds checking."""
    grid = Grid(origin_lat=0.0, origin_lon=0.0, grid_size=256, coverage_km=256.0)

    assert grid.is_within_bounds(0, 0) == True
    assert grid.is_within_bounds(255, 255) == True
    assert grid.is_within_bounds(128, 128) == True
    assert grid.is_within_bounds(-1, 0) == False
    assert grid.is_within_bounds(0, 256) == False


def test_get_distance_matrix():
    """Test distance matrix calculation."""
    grid = Grid(origin_lat=0.0, origin_lon=0.0, grid_size=256, coverage_km=256.0)

    distances = grid.get_distance_matrix()

    assert distances.shape == (256, 256)
    # Center should have zero distance
    assert distances[128, 128] < 1.0
    # Corners should have maximum distance
    corner_dist = distances[0, 0]
    assert corner_dist > 100.0  # Should be > 100 km
