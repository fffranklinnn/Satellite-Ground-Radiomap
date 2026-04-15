"""
FrameContext — immutable per-frame simulation contract.

Every simulation frame is driven by a single FrameContext instance built
via frame_builder.build(). No layer may independently construct its own
grid or timestamp; all geometry is derived from the FrameContext.

Usage:
    from src.context.frame_context import FrameContext, FrameMismatchError
"""

from __future__ import annotations

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
        grid:           Spatial grid for this frame (center-anchored GridSpec).
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
        if not isinstance(self.grid, GridSpec):
            raise TypeError(f"FrameContext.grid must be a GridSpec, got {type(self.grid)!r}")
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"FrameContext.timestamp must be timezone-aware UTC, got naive {self.timestamp!r}. "
                "Use datetime.now(timezone.utc) or attach tzinfo=timezone.utc."
            )
        # Normalize to UTC
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))

        if self.strict:
            if self.coverage is None:
                raise ValueError("FrameContext requires coverage in strict mode.")
            if self.norad_id is None:
                raise ValueError("FrameContext requires norad_id in strict mode.")
            if self.sat_elevation_deg is None:
                raise ValueError("FrameContext requires sat_elevation_deg in strict mode.")

    def check_frame_id(self, state_frame_id: str) -> None:
        """Raise FrameMismatchError if state_frame_id does not match this frame."""
        if state_frame_id != self.frame_id:
            raise FrameMismatchError(
                f"Frame ID mismatch: expected '{self.frame_id}', got '{state_frame_id}'. "
                "State objects must be produced from the same FrameContext."
            )

    def __repr__(self) -> str:
        ts = self.timestamp.isoformat()
        return (
            f"FrameContext(id={self.frame_id!r} ts={ts} "
            f"grid={self.grid.nx}x{self.grid.ny} "
            f"norad={self.norad_id})"
        )
