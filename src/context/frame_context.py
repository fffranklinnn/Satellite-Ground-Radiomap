"""
FrameContext — immutable per-frame simulation contract.

Every simulation frame is driven by a single FrameContext instance built
via frame_builder.build(). No layer may independently construct its own
grid or timestamp; all geometry is derived from the FrameContext.

Usage:
    from src.context.frame_context import FrameContext, FrameMismatchError
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .grid_spec import GridSpec
from .coverage_spec import CoverageSpec
from .time_utils import require_utc


class FrameMismatchError(ValueError):
    """Raised when a state object's frame_id does not match the current frame."""


@dataclass(frozen=True)
class FrameContext:
    """
    Immutable description of a single simulation frame.

    Attributes:
        frame_id:       Unique identifier for this frame (e.g. ISO timestamp + norad_id).
        timestamp:      UTC datetime for satellite positioning and atmospheric queries.
        grid:           Spatial grid for this frame (deprecated; use coverage.l1_grid etc.).
        coverage:       Multi-scale coverage specification (optional; required in strict mode).
        norad_id:       NORAD catalog ID of the selected satellite (optional).
        sat_lat_deg:    Satellite sub-satellite latitude (optional).
        sat_lon_deg:    Satellite sub-satellite longitude (optional).
        sat_alt_m:      Satellite altitude in metres (optional).
        sat_elevation_deg: Elevation angle from grid center to satellite (optional).
        sat_azimuth_deg:   Azimuth angle from grid center to satellite (optional).
        sat_slant_range_m: Slant range from grid center to satellite (optional).
        strict:         If True, missing optional fields raise ValueError.
    """

    frame_id: str
    timestamp: datetime
    grid: GridSpec
    coverage: Optional[CoverageSpec] = None
    norad_id: Optional[str] = None
    sat_lat_deg: Optional[float] = None
    sat_lon_deg: Optional[float] = None
    sat_alt_m: Optional[float] = None
    sat_elevation_deg: Optional[float] = None
    sat_azimuth_deg: Optional[float] = None
    sat_slant_range_m: Optional[float] = None
    strict: bool = False

    def __post_init__(self) -> None:
        grid = object.__getattribute__(self, "grid")
        if not isinstance(grid, GridSpec):
            raise TypeError(f"FrameContext.grid must be a GridSpec, got {type(grid)!r}")
        ts = object.__getattribute__(self, "timestamp")
        if ts.tzinfo is None:
            raise ValueError(
                f"FrameContext.timestamp must be timezone-aware UTC, got naive {ts!r}. "
                "Use datetime.now(timezone.utc) or attach tzinfo=timezone.utc."
            )
        object.__setattr__(self, "timestamp", ts.astimezone(timezone.utc))

        strict = object.__getattribute__(self, "strict")
        if strict:
            if object.__getattribute__(self, "coverage") is None:
                raise ValueError("FrameContext requires coverage in strict mode.")
            if object.__getattribute__(self, "norad_id") is None:
                raise ValueError("FrameContext requires norad_id in strict mode.")
            if object.__getattribute__(self, "sat_elevation_deg") is None:
                raise ValueError("FrameContext requires sat_elevation_deg in strict mode.")

    def __getattribute__(self, name: str):
        if name == "grid":
            # Emit deprecation warning when accessing frame.grid on strict path
            strict = object.__getattribute__(self, "strict")
            if strict:
                warnings.warn(
                    "FrameContext.grid is deprecated on the canonical strict path. "
                    "Use frame.coverage.l1_grid, frame.coverage.l2_grid, etc. instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return object.__getattribute__(self, name)

    def check_frame_id(self, state_frame_id: str) -> None:
        """Raise FrameMismatchError if state_frame_id does not match this frame."""
        if state_frame_id != self.frame_id:
            raise FrameMismatchError(
                f"Frame ID mismatch: expected '{self.frame_id}', got '{state_frame_id}'. "
                "State objects must be produced from the same FrameContext."
            )

    def __repr__(self) -> str:
        ts = object.__getattribute__(self, "timestamp").isoformat()
        grid = object.__getattribute__(self, "grid")
        norad = object.__getattribute__(self, "norad_id")
        fid = object.__getattribute__(self, "frame_id")
        return (
            f"FrameContext(id={fid!r} ts={ts} "
            f"grid={grid.nx}x{grid.ny} "
            f"norad={norad})"
        )
