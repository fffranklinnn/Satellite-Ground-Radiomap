"""
Projection utilities for multi-scale composition.

Projects layer state arrays from native grids to a common product grid
using field-type-specific interpolation contracts.

Projection contracts:
    - Loss fields (dB): bilinear interpolation
    - Boolean masks (occlusion, nlos): nearest-neighbor
    - Support masks: nearest-neighbor with conservative AND at boundaries
    - Angle fields (azimuth): bilinear with 0/360 wrap handling
    - Angle fields (elevation): bilinear (no wrap)
    - Visibility masks: nearest-neighbor
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates, zoom

from src.context.grid_spec import GridSpec


class FieldType(Enum):
    LOSS_DB = "loss_db"
    BOOLEAN_MASK = "boolean_mask"
    SUPPORT_MASK = "support_mask"
    AZIMUTH_DEG = "azimuth_deg"
    ELEVATION_DEG = "elevation_deg"
    VISIBILITY_MASK = "visibility_mask"
    SLANT_RANGE_M = "slant_range_m"


# Map field types to interpolation orders
_INTERP_ORDER = {
    FieldType.LOSS_DB: 1,           # bilinear
    FieldType.BOOLEAN_MASK: 0,      # nearest-neighbor
    FieldType.SUPPORT_MASK: 0,      # nearest-neighbor (+ conservative AND)
    FieldType.AZIMUTH_DEG: 1,       # bilinear with wrap
    FieldType.ELEVATION_DEG: 1,     # bilinear
    FieldType.VISIBILITY_MASK: 0,   # nearest-neighbor
    FieldType.SLANT_RANGE_M: 1,     # bilinear
}


class ProjectionContractError(ValueError):
    """Raised when a projection violates the field-type contract."""


# Placeholder for remaining content


@dataclass(frozen=True)
class ProjectedView:
    """A layer field projected onto the product grid."""
    source_grid: GridSpec
    product_grid: GridSpec
    field_type: FieldType
    values: np.ndarray
    frame_id: str

    def __post_init__(self) -> None:
        expected = (self.product_grid.ny, self.product_grid.nx)
        if self.values.shape != expected:
            raise ValueError(
                f"ProjectedView values shape {self.values.shape} != "
                f"product_grid shape {expected}"
            )


def _compute_coordinate_map(
    source_grid: GridSpec,
    target_grid: GridSpec,
) -> tuple:
    """Compute pixel coordinate mapping from target to source grid."""
    target_rows = np.arange(target_grid.ny, dtype=np.float64)
    target_cols = np.arange(target_grid.nx, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(target_cols, target_rows)

    # Target pixel -> world coordinates
    lat_per_m = 1.0 / 111_000.0
    lon_per_m = 1.0 / (111_000.0 * np.cos(np.radians(target_grid.center_lat)))

    dy_m = (target_grid.ny / 2.0 - row_grid) * target_grid.dy_m
    dx_m = (col_grid - target_grid.nx / 2.0) * target_grid.dx_m
    world_lat = target_grid.center_lat + dy_m * lat_per_m
    world_lon = target_grid.center_lon + dx_m * lon_per_m

    # World -> source pixel coordinates
    src_lat_per_m = 1.0 / 111_000.0
    src_lon_per_m = 1.0 / (111_000.0 * np.cos(np.radians(source_grid.center_lat)))
    dlat = world_lat - source_grid.center_lat
    dlon = world_lon - source_grid.center_lon
    src_dy = dlat / src_lat_per_m
    src_dx = dlon / src_lon_per_m
    src_row = (source_grid.ny / 2.0) - (src_dy / source_grid.dy_m)
    src_col = (source_grid.nx / 2.0) + (src_dx / source_grid.dx_m)

    return src_row, src_col


def project_field(
    source_array: np.ndarray,
    source_grid: GridSpec,
    target_grid: GridSpec,
    field_type: FieldType,
    frame_id: str = "",
) -> ProjectedView:
    """Project a single field from source grid to target grid.

    Enforces per-field-type interpolation contracts.
    """
    if source_grid == target_grid:
        # Identity projection — ensure correct dtype for boolean fields
        vals = source_array.copy()
        if field_type in (FieldType.BOOLEAN_MASK, FieldType.VISIBILITY_MASK, FieldType.SUPPORT_MASK):
            vals = vals > 0.5 if vals.dtype != bool else vals
        return ProjectedView(
            source_grid=source_grid,
            product_grid=target_grid,
            field_type=field_type,
            values=vals,
            frame_id=frame_id,
        )

    src_row, src_col = _compute_coordinate_map(source_grid, target_grid)
    coords = np.array([src_row, src_col])

    if field_type == FieldType.AZIMUTH_DEG:
        # Wrap-aware bilinear: decompose into sin/cos, interpolate, recombine
        az_rad = np.radians(source_array.astype(np.float64))
        sin_az = np.sin(az_rad)
        cos_az = np.cos(az_rad)
        sin_proj = map_coordinates(sin_az, coords, order=1, mode='nearest')
        cos_proj = map_coordinates(cos_az, coords, order=1, mode='nearest')
        result = np.degrees(np.arctan2(sin_proj, cos_proj)) % 360.0
        result = result.astype(np.float32)

    elif field_type == FieldType.SUPPORT_MASK:
        # Nearest-neighbor with conservative AND at boundaries
        float_arr = source_array.astype(np.float64)
        projected = map_coordinates(float_arr, coords, order=0, mode='constant', cval=0.0)
        # Conservative: also check bilinear < 1.0 means boundary
        bilinear = map_coordinates(float_arr, coords, order=1, mode='constant', cval=0.0)
        result = (projected > 0.5) & (bilinear > 0.5)

    elif field_type in (FieldType.BOOLEAN_MASK, FieldType.VISIBILITY_MASK):
        float_arr = source_array.astype(np.float64)
        projected = map_coordinates(float_arr, coords, order=0, mode='nearest')
        result = projected > 0.5

    else:
        order = _INTERP_ORDER[field_type]
        float_arr = source_array.astype(np.float64)
        result = map_coordinates(float_arr, coords, order=order, mode='nearest')
        result = result.astype(np.float32)

    return ProjectedView(
        source_grid=source_grid,
        product_grid=target_grid,
        field_type=field_type,
        values=result,
        frame_id=frame_id,
    )


def validate_projection_contract(field_type: FieldType, order: int) -> None:
    """Raise ProjectionContractError if the interpolation order violates the contract."""
    expected = _INTERP_ORDER[field_type]
    if field_type in (FieldType.BOOLEAN_MASK, FieldType.VISIBILITY_MASK, FieldType.SUPPORT_MASK):
        if order != 0:
            raise ProjectionContractError(
                f"Boolean/support masks require nearest-neighbor (order=0), "
                f"got order={order} for {field_type.value}"
            )
    elif field_type == FieldType.AZIMUTH_DEG:
        if order == 0:
            raise ProjectionContractError(
                f"Azimuth fields require wrap-aware bilinear, got order=0"
            )


def project_to_product_grid(
    product_grid: GridSpec,
    entry=None,
    terrain=None,
    urban=None,
    frame_id: str = "",
):
    """
    Project layer state fields onto product_grid for composition.

    Returns a dict with ProjectedView objects (not bare arrays):
        l1_view, l2_view, l3_residual_view, l3_support_view
    """
    result = {
        "l1_view": None, "l2_view": None,
        "l3_residual_view": None, "l3_support_view": None,
    }

    if entry is not None:
        result["l1_view"] = project_field(
            entry.total_loss_db, entry.native_grid, product_grid,
            FieldType.LOSS_DB, frame_id,
        )

    if terrain is not None:
        result["l2_view"] = project_field(
            terrain.loss_db, terrain.native_grid, product_grid,
            FieldType.LOSS_DB, frame_id,
        )

    if urban is not None:
        result["l3_residual_view"] = project_field(
            urban.urban_residual_db, urban.native_grid, product_grid,
            FieldType.LOSS_DB, frame_id,
        )
        result["l3_support_view"] = project_field(
            urban.support_mask.astype(np.float64), urban.native_grid, product_grid,
            FieldType.SUPPORT_MASK, frame_id,
        )

    return result
