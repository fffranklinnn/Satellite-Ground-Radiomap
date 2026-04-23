"""
Unified spatial grid specification.

All layers must derive their grid geometry from a GridSpec instance.
No layer may independently interpret a bare origin_lat/origin_lon pair.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

VALID_ROLES = {"l1_macro", "l2_terrain", "l3_urban", "product", "legacy"}


@dataclass(frozen=True)
class GridSpec:
    """
    Immutable description of a geographic simulation grid.

    The only supported anchor is "center": center_lat/center_lon is the
    geographic center of the grid. All pixel coordinates are derived from
    this anchor using center-pixel registration.

    Row 0 is the northernmost row (north_to_south order).
    Column 0 is the westernmost column (west_to_east order).
    """

    crs: str
    anchor: str
    center_lat: float
    center_lon: float
    width_m: float
    height_m: float
    nx: int
    ny: int
    dx_m: float
    dy_m: float
    pixel_registration: str
    row_order: str
    col_order: str
    role: str = "legacy"

    def __post_init__(self) -> None:
        if self.anchor != "center":
            raise ValueError(
                f"GridSpec only supports anchor='center', got '{self.anchor}'. "
                "Use GridSpec.from_legacy_args() to convert SW-corner inputs."
            )
        if self.pixel_registration != "center":
            raise ValueError(
                f"GridSpec only supports pixel_registration='center', got '{self.pixel_registration}'."
            )
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"GridSpec.role must be one of {VALID_ROLES}, got '{self.role}'."
            )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_legacy_args(
        cls,
        origin_lat: float,
        origin_lon: float,
        coverage_km: float,
        nx: int,
        ny: int,
        crs: str = "WGS84",
        role: str = "legacy",
    ) -> "GridSpec":
        """
        Build a GridSpec from the legacy center-origin calling convention.

        origin_lat/origin_lon are treated as the grid center (L1/L3 convention).
        """
        width_m = coverage_km * 1000.0
        height_m = coverage_km * 1000.0
        dx_m = width_m / nx
        dy_m = height_m / ny
        return cls(
            crs=crs,
            anchor="center",
            center_lat=float(origin_lat),
            center_lon=float(origin_lon),
            width_m=width_m,
            height_m=height_m,
            nx=nx,
            ny=ny,
            dx_m=dx_m,
            dy_m=dy_m,
            pixel_registration="center",
            row_order="north_to_south",
            col_order="west_to_east",
            role=role,
        )

    @classmethod
    def from_sw_corner(
        cls,
        sw_lat: float,
        sw_lon: float,
        coverage_km: float,
        nx: int,
        ny: int,
        crs: str = "WGS84",
        role: str = "legacy",
    ) -> "GridSpec":
        """
        Build a GridSpec from a south-west corner origin (L2 legacy convention).

        Converts SW-corner to center by adding half the coverage in each direction.
        """
        half_km = coverage_km / 2.0
        lat_per_km = 1.0 / 111.0
        lon_per_km = 1.0 / (111.0 * math.cos(math.radians(sw_lat + half_km * lat_per_km)))
        center_lat = sw_lat + half_km * lat_per_km
        center_lon = sw_lon + half_km * lon_per_km
        return cls.from_legacy_args(center_lat, center_lon, coverage_km, nx, ny, crs, role)

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    def bbox(self) -> Tuple[float, float, float, float]:
        """Return (south, west, north, east) bounding box in degrees."""
        lat_per_m = 1.0 / 111_000.0
        lon_per_m = 1.0 / (111_000.0 * math.cos(math.radians(self.center_lat)))
        half_h = self.height_m / 2.0
        half_w = self.width_m / 2.0
        south = self.center_lat - half_h * lat_per_m
        north = self.center_lat + half_h * lat_per_m
        west = self.center_lon - half_w * lon_per_m
        east = self.center_lon + half_w * lon_per_m
        return (south, west, north, east)

    def sw_corner(self) -> Tuple[float, float]:
        """Return (lat, lon) of the south-west corner."""
        south, west, _, _ = self.bbox()
        return (south, west)

    def world_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert geographic coordinates to fractional pixel (row, col).

        Row 0 is north; col 0 is west. Returns floats (center-pixel registration).
        """
        lat_per_m = 1.0 / 111_000.0
        lon_per_m = 1.0 / (111_000.0 * math.cos(math.radians(self.center_lat)))

        dlat = lat - self.center_lat
        dlon = lon - self.center_lon

        dy_m = dlat / lat_per_m   # positive = north
        dx_m = dlon / lon_per_m   # positive = east

        # Row increases southward; col increases eastward
        row = (self.ny / 2.0) - (dy_m / self.dy_m)
        col = (self.nx / 2.0) + (dx_m / self.dx_m)
        return (row, col)

    def pixel_to_world(self, row: float, col: float) -> Tuple[float, float]:
        """
        Convert fractional pixel (row, col) to geographic coordinates (lat, lon).
        """
        lat_per_m = 1.0 / 111_000.0
        lon_per_m = 1.0 / (111_000.0 * math.cos(math.radians(self.center_lat)))

        dy_m = (self.ny / 2.0 - row) * self.dy_m
        dx_m = (col - self.nx / 2.0) * self.dx_m

        lat = self.center_lat + dy_m * lat_per_m
        lon = self.center_lon + dx_m * lon_per_m
        return (lat, lon)

    def contains_bbox(self, other: "GridSpec", tol_m: float = 1.0) -> bool:
        """Return True if this grid's bbox fully contains other's bbox (within tolerance)."""
        s1, w1, n1, e1 = self.bbox()
        s2, w2, n2, e2 = other.bbox()
        lat_tol = tol_m / 111_000.0
        lon_tol = tol_m / (111_000.0 * math.cos(math.radians(self.center_lat)))
        return (s2 >= s1 - lat_tol and n2 <= n1 + lat_tol and
                w2 >= w1 - lon_tol and e2 <= e1 + lon_tol)

    def same_center(self, other: "GridSpec", tol_m: float = 1.0) -> bool:
        """Return True if this grid shares the same center as other (within tolerance)."""
        lat_tol = tol_m / 111_000.0
        lon_tol = tol_m / (111_000.0 * math.cos(math.radians(self.center_lat)))
        return (abs(self.center_lat - other.center_lat) <= lat_tol and
                abs(self.center_lon - other.center_lon) <= lon_tol)

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "GridSpec":
        """Deserialize from a dict (inverse of to_dict)."""
        return cls(**d)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "GridSpec":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    def with_role(self, role: str) -> "GridSpec":
        """Return a copy of this GridSpec with a different role."""
        d = asdict(self)
        d["role"] = role
        return GridSpec(**d)

    def __repr__(self) -> str:
        return (
            f"GridSpec(center=({self.center_lat:.4f}N,{self.center_lon:.4f}E) "
            f"{self.nx}x{self.ny} dx={self.dx_m:.1f}m dy={self.dy_m:.1f}m "
            f"coverage={self.width_m/1000:.3f}km)"
        )
