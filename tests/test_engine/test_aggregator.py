"""Unit tests for engine.aggregator module (task6 / AC-2)."""

import warnings
import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path

from src.engine.aggregator import RadioMapAggregator, ConfigurationError
from src.context.coverage_spec import CoverageSpec, BlendPolicy


ROOT = Path(__file__).resolve().parents[2]
TLE_PATH = ROOT / "data" / "starlink-2025-tle" / "2025-01-03.tle"


# ---------------------------------------------------------------------------
# Dummy layer helper
# ---------------------------------------------------------------------------

class _DummyLayer:
    def __init__(self, value: float, coverage_km: float = 0.256):
        self.value = float(value)
        self.coverage_km = float(coverage_km)
        self.calls = 0

    def compute(self, origin_lat, origin_lon, timestamp=None, context=None):
        self.calls += 1
        return np.full((256, 256), self.value, dtype=np.float32)


def _make_coverage_spec(product_km: float = 0.256) -> CoverageSpec:
    return CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
        product_coverage_km=product_km, product_nx=256, product_ny=256,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_no_layers_raises():
    with pytest.raises(ValueError, match="At least one layer"):
        RadioMapAggregator()


def test_strict_without_coverage_spec_raises():
    l1 = _DummyLayer(1.0)
    with pytest.raises(ConfigurationError, match="CoverageSpec"):
        RadioMapAggregator(l1_layer=l1, strict=True)


def test_no_coverage_spec_warns():
    l1 = _DummyLayer(1.0)
    with pytest.warns(DeprecationWarning, match="CoverageSpec"):
        agg = RadioMapAggregator(l1_layer=l1)
    assert agg.coverage_spec is None


def test_with_coverage_spec_no_warning():
    l1 = _DummyLayer(1.0)
    cs = _make_coverage_spec()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agg = RadioMapAggregator(l1_layer=l1, coverage_spec=cs)
    assert agg.coverage_spec is cs


# ---------------------------------------------------------------------------
# _target_coverage_km
# ---------------------------------------------------------------------------

def test_target_coverage_km_from_spec():
    l1 = _DummyLayer(1.0)
    cs = _make_coverage_spec(product_km=1.0)
    agg = RadioMapAggregator(l1_layer=l1, coverage_spec=cs)
    assert agg._target_coverage_km() == pytest.approx(1.0)


def test_target_coverage_km_legacy_default():
    l1 = _DummyLayer(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agg = RadioMapAggregator(l1_layer=l1)
    assert agg._target_coverage_km() == pytest.approx(0.256)


# ---------------------------------------------------------------------------
# get_layer_contributions
# ---------------------------------------------------------------------------

def test_get_layer_contributions_calls_each_layer_once():
    l1 = _DummyLayer(1.0, coverage_km=0.256)
    l2 = _DummyLayer(2.0, coverage_km=0.256)
    l3 = _DummyLayer(3.0, coverage_km=0.256)
    cs = _make_coverage_spec()
    agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2, l3_layer=l3, coverage_spec=cs)

    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = agg.get_layer_contributions(0.0, 0.0, timestamp=ts)

    assert l1.calls == 1
    assert l2.calls == 1
    assert l3.calls == 1
    assert out["composite"].shape == (256, 256)
    assert np.allclose(out["composite"], 6.0, atol=1e-5)


def test_get_layer_contributions_keys():
    l1 = _DummyLayer(1.0)
    l2 = _DummyLayer(2.0)
    cs = _make_coverage_spec()
    agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2, coverage_spec=cs)

    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = agg.get_layer_contributions(0.0, 0.0, timestamp=ts)

    assert "l1" in out
    assert "l2" in out
    assert "l3" not in out
    assert "composite" in out


def test_aggregate_returns_composite():
    l1 = _DummyLayer(5.0)
    cs = _make_coverage_spec()
    agg = RadioMapAggregator(l1_layer=l1, coverage_spec=cs)

    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    result = agg.aggregate(0.0, 0.0, timestamp=ts)

    assert result.shape == (256, 256)
    assert np.allclose(result, 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Interpolation: larger coverage layers are cropped to target
# ---------------------------------------------------------------------------

def test_interpolation_larger_coverage():
    """L1 at 256 km coverage should be interpolated down to 0.256 km target."""
    l1 = _DummyLayer(10.0, coverage_km=256.0)
    cs = _make_coverage_spec(product_km=0.256)
    agg = RadioMapAggregator(l1_layer=l1, coverage_spec=cs)

    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = agg.get_layer_contributions(0.0, 0.0, timestamp=ts)

    assert out["l1"].shape == (256, 256)
    # Uniform input -> uniform output after interpolation
    assert np.allclose(out["l1"], 10.0, atol=1e-4)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_with_coverage_spec():
    l1 = _DummyLayer(1.0)
    cs = _make_coverage_spec(product_km=1.0)
    agg = RadioMapAggregator(l1_layer=l1, coverage_spec=cs)
    r = repr(agg)
    assert "L1" in r
    assert "1.000km" in r


def test_repr_legacy():
    l1 = _DummyLayer(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agg = RadioMapAggregator(l1_layer=l1)
    r = repr(agg)
    assert "0.256km" in r
