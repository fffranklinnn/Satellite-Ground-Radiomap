"""Unit tests for layers.l1_macro module."""

import pytest
import numpy as np
from datetime import datetime
from src.layers.l1_macro import L1MacroLayer


def test_l1_initialization():
    """Test L1 layer initialization."""
    config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0
    }

    layer = L1MacroLayer(config, origin_lat=39.9, origin_lon=116.4)

    assert layer.grid_size == 256
    assert layer.coverage_km == 256.0
    assert layer.frequency_ghz == 10.0
    assert layer.satellite_altitude_km == 550.0


def test_l1_compute():
    """Test L1 layer compute method."""
    config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0
    }

    layer = L1MacroLayer(config, origin_lat=39.9, origin_lon=116.4)
    timestamp = datetime(2024, 1, 1, 12, 0, 0)

    loss_map = layer.compute(timestamp)

    # Check output shape
    assert loss_map.shape == (256, 256)

    # Check that losses are positive
    assert np.all(loss_map >= 0)

    # Check that losses are reasonable (not infinite)
    assert np.all(np.isfinite(loss_map))


def test_l1_geometry_calculation():
    """Test geometry calculation."""
    config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0
    }

    layer = L1MacroLayer(config, origin_lat=0.0, origin_lon=0.0)

    # Test at origin (should be zenith)
    slant_range, elevation = layer._calculate_geometry(0.0, 0.0, 0.0, 0.0, 550.0)

    assert slant_range > 0
    assert elevation > 80.0  # Should be near zenith
