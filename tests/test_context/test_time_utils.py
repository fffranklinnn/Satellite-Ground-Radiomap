"""Unit tests for UTC timestamp utilities (task6 / AC-2)."""

import pytest
from datetime import datetime, timezone, timedelta
from src.context.time_utils import parse_iso_utc, require_utc, StrictModeError


# ---------------------------------------------------------------------------
# parse_iso_utc — strict mode (default)
# ---------------------------------------------------------------------------

def test_parse_utc_z_suffix():
    dt = parse_iso_utc("2024-01-15T12:00:00Z")
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 15
    assert dt.hour == 12


def test_parse_utc_plus_offset():
    dt = parse_iso_utc("2024-01-15T20:00:00+08:00")
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 12  # 20:00+08:00 == 12:00 UTC


def test_parse_utc_zero_offset():
    dt = parse_iso_utc("2024-01-15T12:00:00+00:00")
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 12


def test_parse_naive_strict_raises():
    with pytest.raises(StrictModeError, match="Naive datetime"):
        parse_iso_utc("2024-01-15T12:00:00", strict=True)


def test_parse_naive_strict_is_value_error():
    """StrictModeError must be a subclass of ValueError."""
    with pytest.raises(ValueError):
        parse_iso_utc("2024-01-15T12:00:00", strict=True)


# ---------------------------------------------------------------------------
# parse_iso_utc — non-strict mode
# ---------------------------------------------------------------------------

def test_parse_naive_non_strict_warns():
    with pytest.warns(DeprecationWarning, match="Naive datetime"):
        dt = parse_iso_utc("2024-01-15T12:00:00", strict=False)
    assert dt.tzinfo == timezone.utc
    assert dt.hour == 12


def test_parse_naive_non_strict_returns_utc():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dt = parse_iso_utc("2025-03-01T00:00:00", strict=False)
    assert dt.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# require_utc — strict mode
# ---------------------------------------------------------------------------

def test_require_utc_aware_passthrough():
    dt_in = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    dt_out = require_utc(dt_in)
    assert dt_out == dt_in
    assert dt_out.tzinfo == timezone.utc


def test_require_utc_converts_offset_to_utc():
    tz_plus8 = timezone(timedelta(hours=8))
    dt_in = datetime(2024, 1, 15, 20, 0, 0, tzinfo=tz_plus8)
    dt_out = require_utc(dt_in)
    assert dt_out.tzinfo == timezone.utc
    assert dt_out.hour == 12


def test_require_utc_naive_strict_raises():
    dt_naive = datetime(2024, 1, 15, 12, 0, 0)
    with pytest.raises(StrictModeError, match="Naive datetime"):
        require_utc(dt_naive, strict=True)


def test_require_utc_naive_non_strict_warns():
    dt_naive = datetime(2024, 1, 15, 12, 0, 0)
    with pytest.warns(DeprecationWarning):
        dt_out = require_utc(dt_naive, strict=False)
    assert dt_out.tzinfo == timezone.utc
    assert dt_out.hour == 12


# ---------------------------------------------------------------------------
# StrictModeError is ValueError
# ---------------------------------------------------------------------------

def test_strict_mode_error_is_value_error():
    assert issubclass(StrictModeError, ValueError)
