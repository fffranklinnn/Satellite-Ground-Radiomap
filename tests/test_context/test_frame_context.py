"""
Unit tests for FrameContext and FrameBuilder (task8 / AC-3).
"""

import pytest
from datetime import datetime, timezone, timedelta
import numpy as np

from src.context.frame_context import FrameContext, FrameMismatchError
from src.context.frame_builder import FrameBuilder
from src.context.grid_spec import GridSpec
from src.context.coverage_spec import CoverageSpec


def _grid(coverage_km=256.0):
    return GridSpec.from_legacy_args(34.0, 108.0, coverage_km, 256, 256)


def _ts():
    return datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# FrameContext construction
# ---------------------------------------------------------------------------

def test_frame_context_basic():
    fc = FrameContext(
        frame_id="test_frame",
        timestamp=_ts(),
        grid=_grid(),
    )
    assert fc.frame_id == "test_frame"
    assert fc.timestamp.tzinfo == timezone.utc
    assert fc.grid.center_lat == pytest.approx(34.0)


def test_frame_context_naive_timestamp_raises():
    with pytest.raises(ValueError, match="timezone-aware"):
        FrameContext(
            frame_id="f",
            timestamp=datetime(2025, 1, 3, 12, 0, 0),  # naive
            grid=_grid(),
        )


def test_frame_context_normalizes_to_utc():
    tz_plus8 = timezone(timedelta(hours=8))
    ts = datetime(2025, 1, 3, 20, 0, 0, tzinfo=tz_plus8)
    fc = FrameContext(frame_id="f", timestamp=ts, grid=_grid())
    assert fc.timestamp.tzinfo == timezone.utc
    assert fc.timestamp.hour == 12


def test_frame_context_wrong_grid_type():
    with pytest.raises(TypeError, match="GridSpec"):
        FrameContext(frame_id="f", timestamp=_ts(), grid="not_a_grid")  # type: ignore


def test_frame_context_frozen():
    fc = FrameContext(frame_id="f", timestamp=_ts(), grid=_grid())
    with pytest.raises(Exception):
        fc.frame_id = "other"  # type: ignore[misc]


def test_frame_context_strict_requires_coverage():
    with pytest.raises(ValueError, match="coverage"):
        FrameContext(frame_id="f", timestamp=_ts(), grid=_grid(), strict=True)


def test_frame_context_strict_with_coverage_ok():
    cs = CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
        product_coverage_km=0.256, product_nx=256, product_ny=256,
    )
    fc = FrameContext(
        frame_id="f", timestamp=_ts(), grid=_grid(),
        coverage=cs, norad_id="12345", sat_elevation_deg=45.0,
        strict=True,
    )
    assert fc.coverage is cs


# ---------------------------------------------------------------------------
# FrameMismatchError
# ---------------------------------------------------------------------------

def test_check_frame_id_match():
    fc = FrameContext(frame_id="abc", timestamp=_ts(), grid=_grid())
    fc.check_frame_id("abc")  # should not raise


def test_check_frame_id_mismatch():
    fc = FrameContext(frame_id="abc", timestamp=_ts(), grid=_grid())
    with pytest.raises(FrameMismatchError, match="abc"):
        fc.check_frame_id("xyz")


def test_frame_mismatch_error_is_value_error():
    assert issubclass(FrameMismatchError, ValueError)


# ---------------------------------------------------------------------------
# FrameBuilder
# ---------------------------------------------------------------------------

def test_frame_builder_basic():
    builder = FrameBuilder(grid=_grid())
    frame = builder.build(_ts())
    assert frame.timestamp.tzinfo == timezone.utc
    assert frame.grid.center_lat == pytest.approx(34.0)
    assert "20250103T120000Z" in frame.frame_id


def test_frame_builder_with_sat_info():
    builder = FrameBuilder(grid=_grid())
    sat_info = {
        "norad_id": "12345",
        "lat_deg": 35.0,
        "lon_deg": 110.0,
        "alt_m": 550_000.0,
        "elevation_deg": 45.0,
        "azimuth_deg": 180.0,
    }
    frame = builder.build(_ts(), sat_info=sat_info)
    assert frame.norad_id == "12345"
    assert frame.sat_lat_deg == pytest.approx(35.0)
    assert frame.sat_elevation_deg == pytest.approx(45.0)
    assert "12345" in frame.frame_id


def test_frame_builder_custom_frame_id():
    builder = FrameBuilder(grid=_grid())
    frame = builder.build(_ts(), frame_id="custom_id")
    assert frame.frame_id == "custom_id"


def test_frame_builder_wrong_grid_type():
    with pytest.raises(TypeError, match="GridSpec"):
        FrameBuilder(grid="not_a_grid")  # type: ignore


def test_frame_builder_with_coverage():
    cs = CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
        product_coverage_km=0.256, product_nx=256, product_ny=256,
    )
    builder = FrameBuilder(grid=_grid(), coverage=cs)
    frame = builder.build(_ts())
    assert frame.coverage is cs


def test_frame_builder_repr():
    builder = FrameBuilder(grid=_grid())
    r = repr(builder)
    assert "256x256" in r
