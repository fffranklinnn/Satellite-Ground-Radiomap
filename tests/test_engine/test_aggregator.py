"""Unit tests for engine.aggregator module."""

import pytest
import numpy as np
from datetime import datetime
from src.layers.l1_macro import L1MacroLayer
from src.layers.l3_urban import L3UrbanLayer
from src.engine.aggregator import RadioMapAggregator


def test_aggregator_initialization():
    """Test aggregator initialization."""
    l1_config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0
    }

    l1_layer = L1MacroLayer(l1_config, origin_lat=39.9, origin_lon=116.4)
    aggregator = RadioMapAggregator(l1_layer=l1_layer)

    assert aggregator.l1_layer is not None
    assert aggregator.target_grid_size == 256


def test_aggregator_compute():
    """Test aggregator compute method."""
    l1_config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0
    }

    l1_layer = L1MacroLayer(l1_config, origin_lat=39.9, origin_lon=116.4)
    aggregator = RadioMapAggregator(l1_layer=l1_layer)

    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    composite_map = aggregator.aggregate(timestamp)

    # Check output shape
    assert composite_map.shape == (256, 256)

    # Check that values are finite
    assert np.all(np.isfinite(composite_map))


def test_aggregator_layer_contributions():
    """Test getting individual layer contributions."""
    l1_config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0
    }

    l1_layer = L1MacroLayer(l1_config, origin_lat=39.9, origin_lon=116.4)
    aggregator = RadioMapAggregator(l1_layer=l1_layer)

    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    contributions = aggregator.get_layer_contributions(timestamp)

    assert 'l1' in contributions
    assert 'composite' in contributions
    assert contributions['l1'].shape == (256, 256)
