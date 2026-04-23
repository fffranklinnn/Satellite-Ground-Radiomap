"""
Coverage specification and blend policy for multi-scale aggregation.

CoverageSpec defines the geometry of L1, L2, L3, and product grids.
BlendPolicy defines how layers are aligned, cropped, and blended.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

from .grid_spec import GridSpec


VALID_CROP_RULES = {"encompassing", "centered_crop"}


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

    Canonical fields (4-grid model):
        l1_grid:      grid for L1 macro layer (e.g. 256 km, ~1000 m/px)
        l2_grid:      grid for L2 terrain layer (e.g. 25.6 km, ~100 m/px)
        l3_grid:      grid for L3 urban refinement (e.g. 256 m, ~1 m/px), or None
        product_grid: final composition/export target grid
        crop_rule:    how product_grid relates to native grids ("encompassing" or "centered_crop")
        blend:        blend policy for composition

    Geometry contract:
        - All grids must be center-anchored (WGS84)
        - All grids must be concentric (same center point)
        - l2_grid bbox must be contained within l1_grid bbox
        - l3_grid bbox (if present) must be contained within l2_grid bbox
    """

    l1_grid: GridSpec
    l2_grid: GridSpec
    l3_grid: Optional[GridSpec]
    product_grid: GridSpec
    crop_rule: str
    blend: BlendPolicy

    def __post_init__(self) -> None:
        if self.crop_rule not in VALID_CROP_RULES:
            raise ValueError(
                f"crop_rule must be one of {VALID_CROP_RULES}, got '{self.crop_rule}'."
            )
        # Concentric center validation
        if not self.l1_grid.same_center(self.l2_grid):
            raise ValueError(
                "CoverageSpec geometry contract violation: l1_grid and l2_grid must share "
                f"the same center. l1=({self.l1_grid.center_lat}, {self.l1_grid.center_lon}), "
                f"l2=({self.l2_grid.center_lat}, {self.l2_grid.center_lon})"
            )
        if not self.l1_grid.same_center(self.product_grid):
            raise ValueError(
                "CoverageSpec geometry contract violation: l1_grid and product_grid must share "
                "the same center."
            )
        if self.l3_grid is not None:
            if not self.l1_grid.same_center(self.l3_grid):
                raise ValueError(
                    "CoverageSpec geometry contract violation: l1_grid and l3_grid must share "
                    "the same center."
                )
        # Nesting validation: l2 within l1, l3 within l2
        if not self.l1_grid.contains_bbox(self.l2_grid):
            raise ValueError(
                "CoverageSpec geometry contract violation: l2_grid bbox must be contained "
                "within l1_grid bbox."
            )
        if self.l3_grid is not None:
            if not self.l2_grid.contains_bbox(self.l3_grid):
                raise ValueError(
                    "CoverageSpec geometry contract violation: l3_grid bbox must be contained "
                    "within l2_grid bbox."
                )
        # Role validation
        if self.l1_grid.role != "l1_macro":
            raise ValueError(f"l1_grid.role must be 'l1_macro', got '{self.l1_grid.role}'")
        if self.l2_grid.role != "l2_terrain":
            raise ValueError(f"l2_grid.role must be 'l2_terrain', got '{self.l2_grid.role}'")
        if self.l3_grid is not None and self.l3_grid.role != "l3_urban":
            raise ValueError(f"l3_grid.role must be 'l3_urban', got '{self.l3_grid.role}'")
        if self.product_grid.role != "product":
            raise ValueError(f"product_grid.role must be 'product', got '{self.product_grid.role}'")

    # ------------------------------------------------------------------
    # Deprecated compatibility properties
    # ------------------------------------------------------------------

    @property
    def coarse_grid(self) -> GridSpec:
        warnings.warn(
            "CoverageSpec.coarse_grid is deprecated; use l1_grid or l2_grid instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.l1_grid

    @property
    def urban_grid(self) -> Optional[GridSpec]:
        warnings.warn(
            "CoverageSpec.urban_grid is deprecated; use l3_grid instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.l3_grid

    @property
    def target_product_grid(self) -> GridSpec:
        warnings.warn(
            "CoverageSpec.target_product_grid is deprecated; use product_grid instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.product_grid

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

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
        l2_coverage_km: Optional[float] = None,
        l2_nx: Optional[int] = None,
        l2_ny: Optional[int] = None,
        crop_rule: str = "centered_crop",
    ) -> "CoverageSpec":
        """Build a CoverageSpec from scalar config values.

        If l2_coverage_km is not provided, l2_grid defaults to the same
        geometry as l1_grid (backward compatible with single coarse_grid).
        """
        l1_grid = GridSpec.from_legacy_args(
            origin_lat, origin_lon, coarse_coverage_km, coarse_nx, coarse_ny,
            role="l1_macro",
        )
        if l2_coverage_km is not None and l2_nx is not None and l2_ny is not None:
            l2_grid = GridSpec.from_legacy_args(
                origin_lat, origin_lon, l2_coverage_km, l2_nx, l2_ny,
                role="l2_terrain",
            )
        else:
            l2_grid = l1_grid.with_role("l2_terrain")

        product = GridSpec.from_legacy_args(
            origin_lat, origin_lon, product_coverage_km, product_nx, product_ny,
            role="product",
        )
        l3_grid = None
        if urban_coverage_km is not None and urban_nx is not None and urban_ny is not None:
            l3_grid = GridSpec.from_legacy_args(
                origin_lat, origin_lon, urban_coverage_km, urban_nx, urban_ny,
                role="l3_urban",
            )
        return cls(
            l1_grid=l1_grid,
            l2_grid=l2_grid,
            l3_grid=l3_grid,
            product_grid=product,
            crop_rule=crop_rule,
            blend=blend or BlendPolicy.default(),
        )
