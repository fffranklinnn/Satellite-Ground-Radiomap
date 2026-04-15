"""
Legacy adapters for backward compatibility during the GridSpec migration.

These adapters allow existing callers that pass bare origin_lat/origin_lon
(SW-corner convention for L2) to continue working while emitting
DeprecationWarning. New code must use GridSpec directly.
"""

from __future__ import annotations

import warnings
from typing import Optional

from ..context.grid_spec import GridSpec


def sw_corner_to_grid_spec(
    sw_lat: float,
    sw_lon: float,
    coverage_km: float,
    nx: int,
    ny: int,
    crs: str = "WGS84",
) -> GridSpec:
    """
    Convert an L2-style SW-corner origin to a center-anchored GridSpec.

    This is the canonical translation for the L2 SW-corner bug fix.
    Emits DeprecationWarning to encourage callers to switch to GridSpec directly.

    Args:
        sw_lat:      South-west corner latitude (L2 legacy convention).
        sw_lon:      South-west corner longitude (L2 legacy convention).
        coverage_km: Coverage in km (e.g. 25.6 for L2).
        nx, ny:      Grid dimensions.

    Returns:
        GridSpec with anchor="center", center derived from SW corner + half coverage.
    """
    warnings.warn(
        "sw_corner_to_grid_spec() is a legacy adapter for the L2 SW-corner origin "
        "convention. Pass a GridSpec directly instead of bare origin_lat/origin_lon.",
        DeprecationWarning,
        stacklevel=2,
    )
    return GridSpec.from_sw_corner(sw_lat, sw_lon, coverage_km, nx, ny, crs)


def center_to_grid_spec(
    origin_lat: float,
    origin_lon: float,
    coverage_km: float,
    nx: int,
    ny: int,
    crs: str = "WGS84",
) -> GridSpec:
    """
    Convert a center-origin (L1/L3 legacy convention) to a GridSpec.

    Emits DeprecationWarning to encourage callers to switch to GridSpec directly.
    """
    warnings.warn(
        "center_to_grid_spec() is a legacy adapter. "
        "Use GridSpec.from_legacy_args() directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return GridSpec.from_legacy_args(origin_lat, origin_lon, coverage_km, nx, ny, crs)
