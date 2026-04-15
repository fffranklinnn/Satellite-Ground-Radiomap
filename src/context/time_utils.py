"""
Shared UTC timestamp parsing utility.

All timestamps entering the pipeline must be timezone-aware UTC.
In strict mode, naive datetime strings raise ValueError immediately.
In non-strict mode, naive strings are normalized with a DeprecationWarning.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone


class StrictModeError(ValueError):
    """Raised when a strict-mode constraint is violated."""


def parse_iso_utc(s: str, strict: bool = True) -> datetime:
    """
    Parse an ISO 8601 string and return a timezone-aware UTC datetime.

    Args:
        s:      ISO 8601 string, e.g. "2024-01-15T12:00:00Z" or
                "2024-01-15T12:00:00+08:00".
        strict: If True (default), raise ValueError for naive strings.
                If False, normalize naive strings to UTC with a DeprecationWarning.

    Returns:
        datetime with tzinfo=timezone.utc

    Raises:
        ValueError: if the string is naive and strict=True.
        StrictModeError: subclass of ValueError, same condition.
    """
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        if strict:
            raise StrictModeError(
                f"Naive datetime string '{s}' is not allowed in strict mode. "
                "Append 'Z' or a UTC offset (e.g. '+00:00') to make it timezone-aware."
            )
        warnings.warn(
            f"Naive datetime string '{s}' treated as UTC. "
            "This behavior is deprecated; use an explicit UTC offset.",
            DeprecationWarning,
            stacklevel=2,
        )
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def require_utc(dt: datetime, strict: bool = True) -> datetime:
    """
    Ensure a datetime object is timezone-aware UTC.

    Args:
        dt:     datetime object to validate.
        strict: If True, raise StrictModeError for naive datetimes.

    Returns:
        datetime with tzinfo=timezone.utc
    """
    if dt.tzinfo is None:
        if strict:
            raise StrictModeError(
                f"Naive datetime {dt!r} is not allowed in strict mode. "
                "Use datetime.now(timezone.utc) or attach tzinfo=timezone.utc."
            )
        warnings.warn(
            f"Naive datetime {dt!r} treated as UTC. "
            "This behavior is deprecated; attach tzinfo=timezone.utc explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
