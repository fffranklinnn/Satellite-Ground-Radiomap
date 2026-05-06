"""
MultiScaleMap: composite radio map assembled from typed layer states.

Replaces the legacy additive pattern (composite = l1 + l2 + l3) with a
masked residual compositor that applies L3 only within its support_mask.

Composition rule:
    composite = l1_loss + l2_loss + (l3_residual * support_mask)

Where:
    l1_loss      — EntryWaveState.total_loss_db (full grid)
    l2_loss      — TerrainState.loss_db (full grid, zero where no terrain)
    l3_residual  — UrbanRefinementState.urban_residual_db (masked)
    support_mask — UrbanRefinementState.support_mask (True where tile data exists)

The old additive pattern (l1 + l2 + l3 without masking) emits a
DeprecationWarning and is supported via MultiScaleMap.from_additive().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .frame_context import FrameMismatchError
from .grid_spec import GridSpec
from .layer_states import EntryWaveState, TerrainState, UrbanRefinementState


class ShapeError(ValueError):
    """Raised when layer arrays have incompatible shapes."""


class GridMismatchError(ValueError):
    """Raised when projected views have mismatched grid metadata."""


@dataclass(frozen=True)
class MultiScaleMap:
    """
    Composite radio map assembled from typed layer states.

    Attributes:
        frame_id:       Identifies the simulation frame.
        grid:           GridSpec of the composite map (L1/L2 resolution).
        composite_db:   Final composite path loss (dB), float32, shape (ny, nx).
        l1_db:          L1 contribution (dB), or None if L1 was absent.
        l2_db:          L2 contribution (dB), or None if L2 was absent.
        l3_db:          L3 masked residual (dB), or None if L3 was absent.
        l3_support_mask: Boolean mask where L3 tile data was available.
    """

    frame_id: str
    grid: GridSpec
    composite_db: np.ndarray
    l1_db: Optional[np.ndarray] = None
    l2_db: Optional[np.ndarray] = None
    l3_db: Optional[np.ndarray] = None
    l3_support_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridSpec):
            raise TypeError(f"MultiScaleMap.grid must be GridSpec, got {type(self.grid)!r}")
        expected = (self.grid.ny, self.grid.nx)
        if self.composite_db.shape != expected:
            raise ShapeError(
                f"MultiScaleMap.composite_db shape {self.composite_db.shape} != grid {expected}"
            )
        for name, arr in (
            ("l1_db", self.l1_db),
            ("l2_db", self.l2_db),
            ("l3_db", self.l3_db),
            ("l3_support_mask", self.l3_support_mask),
        ):
            if arr is not None and arr.shape != expected:
                raise ShapeError(
                    f"MultiScaleMap.{name} shape {arr.shape} != grid {expected}"
                )

    # ------------------------------------------------------------------
    # Factory: canonical projected compositor
    # ------------------------------------------------------------------

    @classmethod
    def compose(
        cls,
        frame_id: str,
        product_grid: GridSpec,
        l1_view=None,
        l2_view=None,
        l3_residual_view=None,
        l3_support_view=None,
    ) -> "MultiScaleMap":
        """
        Assemble a MultiScaleMap from ProjectedView objects on product_grid.

        This is the canonical composition API. All inputs must be ProjectedView
        instances whose product_grid matches the declared product_grid exactly
        (including resolution, not just shape and center).

        Raises:
            GridMismatchError: If any view's product_grid differs from product_grid.
            TypeError: If any input is not a ProjectedView.
        """
        from src.compose import ProjectedView

        expected = (product_grid.ny, product_grid.nx)
        composite = np.zeros(expected, dtype=np.float32)
        l1_db = l2_db = l3_db = l3_mask = None

        for label, view in [("l1", l1_view), ("l2", l2_view),
                            ("l3_residual", l3_residual_view), ("l3_support", l3_support_view)]:
            if view is None:
                continue
            if not isinstance(view, ProjectedView):
                raise TypeError(
                    f"MultiScaleMap.compose() requires ProjectedView inputs, "
                    f"got {type(view).__name__} for {label}. "
                    f"Use project_to_product_grid() to project states first."
                )
            if view.product_grid != product_grid:
                raise GridMismatchError(
                    f"{label}: ProjectedView.product_grid does not match the declared "
                    f"product_grid. View grid: {view.product_grid!r}, "
                    f"expected: {product_grid!r}. "
                    f"This can happen when arrays have matching shape but different "
                    f"resolution or extent."
                )

        if l1_view is not None:
            l1_db = l1_view.values.astype(np.float32, copy=False)
            composite = composite + l1_db

        if l2_view is not None:
            l2_db = l2_view.values.astype(np.float32, copy=False)
            composite = composite + l2_db

        if l3_residual_view is not None:
            if l3_support_view is not None:
                l3_mask = l3_support_view.values.astype(bool, copy=False)
                l3_db = np.where(l3_mask, l3_residual_view.values, 0.0).astype(np.float32)
            else:
                l3_db = l3_residual_view.values.astype(np.float32, copy=False)
            composite = composite + l3_db

        return cls(
            frame_id=frame_id,
            grid=product_grid,
            composite_db=composite,
            l1_db=l1_db,
            l2_db=l2_db,
            l3_db=l3_db,
            l3_support_mask=l3_mask,
        )

    # ------------------------------------------------------------------
    # Factory: legacy native-state compositor (deprecated)
    # ------------------------------------------------------------------

    @classmethod
    def compose_legacy(
        cls,
        frame_id: str,
        grid: GridSpec,
        entry: Optional[EntryWaveState] = None,
        terrain: Optional[TerrainState] = None,
        urban: Optional[UrbanRefinementState] = None,
    ) -> "MultiScaleMap":
        """
        Assemble a MultiScaleMap using the masked residual compositor.

        L3 residual is applied only within urban.support_mask.
        Absent layers contribute zero to the composite.

        Args:
            frame_id: Frame identifier (must match all provided states).
            grid:     GridSpec for the composite (typically L1/L2 grid).
            entry:    EntryWaveState from L1 (optional).
            terrain:  TerrainState from L2 (optional).
            urban:    UrbanRefinementState from L3 (optional).

        Returns:
            MultiScaleMap with composite_db and per-layer contributions.

        Raises:
            FrameMismatchError: If any state's frame_id does not match frame_id.
            ShapeError: If any layer array shape is incompatible with grid.
        """
        # Validate frame_id consistency across all provided states
        for state, label in ((entry, "entry"), (terrain, "terrain"), (urban, "urban")):
            if state is not None and state.frame_id != frame_id:
                raise FrameMismatchError(
                    f"MultiScaleMap.compose: {label}.frame_id {state.frame_id!r} "
                    f"!= frame_id {frame_id!r}"
                )

        expected = (grid.ny, grid.nx)
        composite = np.zeros(expected, dtype=np.float32)

        l1_db = l2_db = l3_db = l3_support = None

        if entry is not None:
            if entry.total_loss_db.shape != expected:
                raise ShapeError(
                    f"EntryWaveState.total_loss_db shape {entry.total_loss_db.shape} "
                    f"!= grid {expected}"
                )
            l1_db = entry.total_loss_db.astype(np.float32, copy=False)
            composite = composite + l1_db

        if terrain is not None:
            if terrain.loss_db.shape != expected:
                raise ShapeError(
                    f"TerrainState.loss_db shape {terrain.loss_db.shape} != grid {expected}"
                )
            l2_db = terrain.loss_db.astype(np.float32, copy=False)
            composite = composite + l2_db

        if urban is not None:
            if urban.urban_residual_db.shape != expected:
                raise ShapeError(
                    f"UrbanRefinementState.urban_residual_db shape "
                    f"{urban.urban_residual_db.shape} != grid {expected}"
                )
            l3_support = urban.support_mask.astype(bool, copy=False)
            l3_db = np.where(l3_support, urban.urban_residual_db, 0.0).astype(np.float32)
            composite = composite + l3_db

        return cls(
            frame_id=frame_id,
            grid=grid,
            composite_db=composite,
            l1_db=l1_db,
            l2_db=l2_db,
            l3_db=l3_db,
            l3_support_mask=l3_support,
        )

    # ------------------------------------------------------------------
    # Factory: deprecated bare-array projected composition
    # ------------------------------------------------------------------

    @classmethod
    def compose_projected(
        cls,
        frame_id: str,
        product_grid: GridSpec,
        l1_loss: Optional[np.ndarray] = None,
        l2_loss: Optional[np.ndarray] = None,
        l3_residual: Optional[np.ndarray] = None,
        l3_support: Optional[np.ndarray] = None,
        l1_grid: Optional[GridSpec] = None,
        l2_grid: Optional[GridSpec] = None,
        l3_grid: Optional[GridSpec] = None,
    ) -> "MultiScaleMap":
        """
        Assemble a MultiScaleMap from pre-projected arrays on product_grid.

        All input arrays must already be projected onto product_grid.
        Passing raw native-grid arrays with mismatched grid metadata raises
        GridMismatchError.

        Args:
            frame_id:     Frame identifier.
            product_grid: The target product grid all arrays are projected onto.
            l1_loss:      Projected L1 total loss (dB) on product_grid, or None.
            l2_loss:      Projected L2 terrain loss (dB) on product_grid, or None.
            l3_residual:  Projected L3 urban residual (dB) on product_grid, or None.
            l3_support:   Projected L3 support mask on product_grid, or None.
            l1_grid:      Source grid metadata for l1_loss (for validation).
            l2_grid:      Source grid metadata for l2_loss (for validation).
            l3_grid:      Source grid metadata for l3_residual (for validation).

        Returns:
            MultiScaleMap with composite_db and per-layer contributions.
        """
        # Validate grid metadata consistency
        for label, arr, src_grid in [
            ("l1_loss", l1_loss, l1_grid),
            ("l2_loss", l2_loss, l2_grid),
            ("l3_residual", l3_residual, l3_grid),
        ]:
            if arr is not None and src_grid is not None:
                # If source grid has different shape than product_grid but array
                # matches product_grid shape, projection happened — OK.
                # If source grid has same shape as product_grid but different
                # center/resolution, that's a metadata mismatch.
                if (src_grid.nx == product_grid.nx and src_grid.ny == product_grid.ny
                        and not src_grid.same_center(product_grid)):
                    raise GridMismatchError(
                        f"{label}: source grid and product_grid have matching shape "
                        f"({src_grid.nx}x{src_grid.ny}) but different centers. "
                        f"This suggests raw native-grid arrays were passed without projection."
                    )

        expected = (product_grid.ny, product_grid.nx)
        composite = np.zeros(expected, dtype=np.float32)

        l1_db = l2_db = l3_db = l3_mask = None

        if l1_loss is not None:
            if l1_loss.shape != expected:
                raise ShapeError(
                    f"l1_loss shape {l1_loss.shape} != product_grid {expected}. "
                    "Arrays must be projected to product_grid before composition."
                )
            l1_db = l1_loss.astype(np.float32, copy=False)
            composite = composite + l1_db

        if l2_loss is not None:
            if l2_loss.shape != expected:
                raise ShapeError(
                    f"l2_loss shape {l2_loss.shape} != product_grid {expected}. "
                    "Arrays must be projected to product_grid before composition."
                )
            l2_db = l2_loss.astype(np.float32, copy=False)
            composite = composite + l2_db

        if l3_residual is not None:
            if l3_residual.shape != expected:
                raise ShapeError(
                    f"l3_residual shape {l3_residual.shape} != product_grid {expected}. "
                    "Arrays must be projected to product_grid before composition."
                )
            if l3_support is not None:
                l3_mask = l3_support.astype(bool, copy=False)
                l3_db = np.where(l3_mask, l3_residual, 0.0).astype(np.float32)
            else:
                l3_db = l3_residual.astype(np.float32, copy=False)
            composite = composite + l3_db

        return cls(
            frame_id=frame_id,
            grid=product_grid,
            composite_db=composite,
            l1_db=l1_db,
            l2_db=l2_db,
            l3_db=l3_db,
            l3_support_mask=l3_mask,
        )

    # ------------------------------------------------------------------
    # Factory: legacy additive pattern (deprecated)
    # ------------------------------------------------------------------

    @classmethod
    def from_additive(
        cls,
        frame_id: str,
        grid: GridSpec,
        l1_map: Optional[np.ndarray] = None,
        l2_map: Optional[np.ndarray] = None,
        l3_map: Optional[np.ndarray] = None,
    ) -> "MultiScaleMap":
        """
        Build a MultiScaleMap from raw arrays using the old additive pattern.

        .. deprecated::
            Use MultiScaleMap.compose() with typed layer states instead.
            The additive pattern applies L3 unconditionally (no support_mask),
            which can over-apply urban residuals outside tile coverage.
        """
        warnings.warn(
            "MultiScaleMap.from_additive() is deprecated. "
            "Use MultiScaleMap.compose() with EntryWaveState/TerrainState/"
            "UrbanRefinementState to apply L3 only within support_mask.",
            DeprecationWarning,
            stacklevel=2,
        )
        expected = (grid.ny, grid.nx)
        composite = np.zeros(expected, dtype=np.float32)
        l1_db = l2_db = l3_db = None

        if l1_map is not None:
            l1_db = np.asarray(l1_map, dtype=np.float32)
            if l1_db.shape != expected:
                raise ShapeError(f"l1_map shape {l1_db.shape} != grid {expected}")
            composite = composite + l1_db

        if l2_map is not None:
            l2_db = np.asarray(l2_map, dtype=np.float32)
            if l2_db.shape != expected:
                raise ShapeError(f"l2_map shape {l2_db.shape} != grid {expected}")
            composite = composite + l2_db

        if l3_map is not None:
            l3_db = np.asarray(l3_map, dtype=np.float32)
            if l3_db.shape != expected:
                raise ShapeError(f"l3_map shape {l3_db.shape} != grid {expected}")
            composite = composite + l3_db

        return cls(
            frame_id=frame_id,
            grid=grid,
            composite_db=composite,
            l1_db=l1_db,
            l2_db=l2_db,
            l3_db=l3_db,
            l3_support_mask=None,
        )
