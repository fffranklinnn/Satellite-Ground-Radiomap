"""
Coverage specification and blend policy for multi-scale aggregation.

CoverageSpec defines the geometry of coarse, urban, and product grids.
BlendPolicy defines how layers are aligned, cropped, and blended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .grid_spec import GridSpec


@dataclass(frozen=True)
class BlendPolicy:
    """
    Alignment, crop, and blend rules for multi-scale composition.

    alignment_rule: how grids are aligned ("same_center" or "same_bbox")
    crop_policy:    how larger grids are cropped to the product grid ("bbox_project")
    blend_policy:   how coarse and urban layers are combined
                    ("coarse_plus_masked_residual" = L3 only within support_mask)
    """

    alignment_rule: str
    crop_policy: str
    blend_policy: str

    def __post_init__(self) -> None:
        valid_alignment = {"same_center", "same_bbox"}
        valid_crop = {"bbox_project"}
        valid_blend = {"coarse_plus_masked_residual", "additive"}
        if self.alignment_rule not in valid_alignment:
            raise ValueError(f"alignment_rule must be one of {valid_alignment}")
        if self.crop_policy not in valid_crop:
            raise ValueError(f"crop_policy must be one of {valid_crop}")
        if self.blend_policy not in valid_blend:
            raise ValueError(f"blend_policy must be one of {valid_blend}")

    @classmethod
    def default(cls) -> "BlendPolicy":
        return cls(
            alignment_rule="same_center",
            crop_policy="bbox_project",
            blend_policy="coarse_plus_masked_residual",
        )


@dataclass(frozen=True)
class CoverageSpec:
    """
    Geometry specification for a multi-scale simulation product.

    coarse_grid:         grid for L1/L2 (e.g. 256 km or 25.6 km)
    urban_grid:          grid for L3 urban refinement (e.g. 256 m), or None
    target_product_grid: final output grid (replaces hardcoded 0.256 km constant)
    blend:               blend policy for composition
    """

    coarse_grid: GridSpec
    urban_grid: Optional[GridSpec]
    target_product_grid: GridSpec
    blend: BlendPolicy

    @classmethod
    def from_config(
        cls,
        origin_lat: float,
        origin_lon: float,
        coarse_coverage_km: float,
        coarse_nx: int,
        coarse_ny: int,
        product_coverage_km: float,
        product_nx: int,
        product_ny: int,
        urban_coverage_km: Optional[float] = None,
        urban_nx: Optional[int] = None,
        urban_ny: Optional[int] = None,
        blend: Optional[BlendPolicy] = None,
    ) -> "CoverageSpec":
        """Build a CoverageSpec from scalar config values."""
        coarse_grid = GridSpec.from_legacy_args(
            origin_lat, origin_lon, coarse_coverage_km, coarse_nx, coarse_ny
        )
        target_grid = GridSpec.from_legacy_args(
            origin_lat, origin_lon, product_coverage_km, product_nx, product_ny
        )
        urban_grid = None
        if urban_coverage_km is not None and urban_nx is not None and urban_ny is not None:
            urban_grid = GridSpec.from_legacy_args(
                origin_lat, origin_lon, urban_coverage_km, urban_nx, urban_ny
            )
        return cls(
            coarse_grid=coarse_grid,
            urban_grid=urban_grid,
            target_product_grid=target_grid,
            blend=blend or BlendPolicy.default(),
        )
