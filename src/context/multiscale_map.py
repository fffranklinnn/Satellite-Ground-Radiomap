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

from .grid_spec import GridSpec
from .layer_states import EntryWaveState, TerrainState, UrbanRefinementState


class ShapeError(ValueError):
    """Raised when layer arrays have incompatible shapes."""


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
    # Factory: masked residual compositor (canonical)
    # ------------------------------------------------------------------

    @classmethod
    def compose(
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
            ShapeError: If any layer array shape is incompatible with grid.
        """
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
