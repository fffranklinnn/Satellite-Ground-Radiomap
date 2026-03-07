"""Unit tests for engine.aggregator module."""

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from src.layers.l1_macro import L1MacroLayer
from src.engine.aggregator import RadioMapAggregator

ROOT = Path(__file__).resolve().parents[2]
TLE_PATH = ROOT / "data" / "2025_0101.tle"


def _l1_config() -> dict:
    return {
        "grid_size": 256,
        "coverage_km": 256.0,
        "resolution_m": 1000.0,
        "frequency_ghz": 10.0,
        "tle_file": str(TLE_PATH),
    }


def test_aggregator_initialization():
    """Test aggregator initialization."""
    l1_layer = L1MacroLayer(_l1_config(), origin_lat=39.9, origin_lon=116.4)
    aggregator = RadioMapAggregator(l1_layer=l1_layer)

    assert aggregator.l1_layer is not None
    assert aggregator.target_grid_size == 256


def test_aggregator_compute():
    """Test aggregator compute method."""
    l1_layer = L1MacroLayer(_l1_config(), origin_lat=39.9, origin_lon=116.4)
    aggregator = RadioMapAggregator(l1_layer=l1_layer)

    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    composite_map = aggregator.aggregate(39.9, 116.4, timestamp=timestamp)

    # Check output shape
    assert composite_map.shape == (256, 256)

    # Check that values are finite
    assert np.all(np.isfinite(composite_map))


def test_aggregator_layer_contributions():
    """Test getting individual layer contributions."""
    l1_layer = L1MacroLayer(_l1_config(), origin_lat=39.9, origin_lon=116.4)
    aggregator = RadioMapAggregator(l1_layer=l1_layer)

    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    contributions = aggregator.get_layer_contributions(39.9, 116.4, timestamp=timestamp)

    assert "l1" in contributions
    assert "composite" in contributions
    assert contributions["l1"].shape == (256, 256)


class _DummyLayer:
    def __init__(self, value: float, coverage_km: float = 0.256):
        self.value = float(value)
        self.coverage_km = float(coverage_km)
        self.calls = 0

    def compute(self, origin_lat, origin_lon, timestamp=None, context=None):
        self.calls += 1
        return np.full((256, 256), self.value, dtype=np.float32)


def test_get_layer_contributions_calls_each_layer_once():
    l1 = _DummyLayer(1.0, coverage_km=0.256)
    l2 = _DummyLayer(2.0, coverage_km=0.256)
    l3 = _DummyLayer(3.0, coverage_km=0.256)
    agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2, l3_layer=l3)

    out = agg.get_layer_contributions(0.0, 0.0, timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc))

    assert l1.calls == 1
    assert l2.calls == 1
    assert l3.calls == 1
    assert out["composite"].shape == (256, 256)
    assert np.allclose(out["composite"], 6.0, atol=1e-6)
