"""
FrameBuilder — constructs FrameContext instances from config + satellite data.

Usage:
    builder = FrameBuilder(grid_spec, coverage_spec=cs)
    frame = builder.build(timestamp, sat_info=sat_info)

The builder is the single entry point for creating FrameContext objects.
No layer or script should construct FrameContext directly.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .frame_context import FrameContext
from .grid_spec import GridSpec
from .coverage_spec import CoverageSpec
from .time_utils import require_utc


class FrameBuilder:
    """
    Constructs FrameContext instances for simulation frames.

    Args:
        grid:       GridSpec defining the spatial domain for all frames.
        coverage:   CoverageSpec for multi-scale aggregation (optional).
        strict:     If True, missing satellite geometry raises ValueError.
    """

    def __init__(
        self,
        grid: GridSpec,
        coverage: Optional[CoverageSpec] = None,
        strict: bool = False,
    ) -> None:
        if not isinstance(grid, GridSpec):
            raise TypeError(f"FrameBuilder.grid must be a GridSpec, got {type(grid)!r}")
        self.grid = grid
        self.coverage = coverage
        self.strict = strict

    def build(
        self,
        timestamp: datetime,
        sat_info: Optional[Dict[str, Any]] = None,
        frame_id: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> FrameContext:
        """
        Build a FrameContext for a single simulation frame.

        Args:
            timestamp:  UTC datetime for this frame.
            sat_info:   Satellite geometry dict (from SatelliteSelector.select).
                        Expected keys: norad_id, lat_deg, lon_deg, alt_m,
                        elevation_deg, azimuth_deg, slant_range_m.
                        Required on canonical strict path.
            frame_id:   Override the auto-generated frame ID.
            strict:     Override the builder's strict setting for this frame.

        Returns:
            Frozen FrameContext.
        """
        use_strict = self.strict if strict is None else strict
        ts = require_utc(timestamp, strict=use_strict)

        if use_strict and sat_info is None:
            raise ValueError(
                "FrameBuilder.build() requires sat_info on the canonical strict path. "
                "Use SatelliteSelector.select() to obtain satellite geometry before "
                "building the frame."
            )
        if not use_strict and sat_info is None:
            warnings.warn(
                "Building a frame without sat_info is deprecated. "
                "Use SatelliteSelector.select() to pre-bind satellite geometry.",
                DeprecationWarning,
                stacklevel=2,
            )

        norad_id = None
        sat_lat = sat_lon = sat_alt = sat_el = sat_az = sat_slant = None

        if sat_info is not None:
            norad_id = str(sat_info.get("norad_id", sat_info.get("catalog_number", ""))) or None
            sat_lat = sat_info.get("lat_deg")
            sat_lon = sat_info.get("lon_deg")
            sat_alt = sat_info.get("alt_m")
            sat_el = sat_info.get("elevation_deg")
            sat_az = sat_info.get("azimuth_deg")
            sat_slant = sat_info.get("slant_range_m")

        if frame_id is None:
            ts_str = ts.strftime("%Y%m%dT%H%M%SZ")
            norad_str = norad_id or "nosat"
            frame_id = f"{ts_str}_{norad_str}"

        return FrameContext(
            frame_id=frame_id,
            timestamp=ts,
            grid=self.grid,
            coverage=self.coverage,
            norad_id=norad_id,
            sat_lat_deg=sat_lat,
            sat_lon_deg=sat_lon,
            sat_alt_m=sat_alt,
            sat_elevation_deg=sat_el,
            sat_azimuth_deg=sat_az,
            sat_slant_range_m=sat_slant,
            strict=use_strict,
        )

    def __repr__(self) -> str:
        return (
            f"FrameBuilder(grid={self.grid.nx}x{self.grid.ny} "
            f"coverage={'yes' if self.coverage else 'no'} "
            f"strict={self.strict})"
        )
