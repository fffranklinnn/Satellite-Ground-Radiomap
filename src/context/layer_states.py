"""
Typed per-layer state objects for the SG-MRM pipeline.

Each propagation step returns a typed, immutable state object that carries
the frame_id, native_grid, and all computed arrays. Downstream steps validate
frame_id consistency before consuming state.

State hierarchy:
    EntryWaveState   — L1 output (free-space + atmospheric + ionospheric)
    TerrainState     — L2 output (terrain occlusion + diffraction)
    UrbanRefinementState — L3 output (urban NLoS residual)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid_spec import GridSpec


@dataclass(frozen=True)
class EntryWaveState:
    """L1 propagation output: entry-wave loss components."""

    frame_id: str
    native_grid: GridSpec

    total_loss_db: np.ndarray
    fspl_db: np.ndarray
    atm_db: np.ndarray
    iono_db: np.ndarray
    pol_db: np.ndarray
    gain_db: np.ndarray

    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    slant_range_m: np.ndarray
    occlusion_mask: np.ndarray

    norad_id: Optional[str] = None
    sat_lat_deg: Optional[float] = None
    sat_lon_deg: Optional[float] = None
    sat_alt_m: Optional[float] = None

    @property
    def grid(self) -> GridSpec:
        """Deprecated compatibility alias for native_grid."""
        warnings.warn(
            "EntryWaveState.grid is deprecated; use native_grid instead.",
            DeprecationWarning, stacklevel=2,
        )
        return object.__getattribute__(self, "native_grid")

    def __post_init__(self) -> None:
        ng = object.__getattribute__(self, "native_grid")
        if not isinstance(ng, GridSpec):
            raise TypeError(f"EntryWaveState.native_grid must be GridSpec, got {type(ng)!r}")
        expected = (ng.ny, ng.nx)
        for name in ("total_loss_db", "fspl_db", "atm_db", "iono_db",
                     "pol_db", "gain_db", "elevation_deg", "azimuth_deg",
                     "slant_range_m", "occlusion_mask"):
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(
                    f"EntryWaveState.{name} shape {arr.shape} != grid shape {expected}"
                )


@dataclass(frozen=True)
class TerrainState:
    """L2 propagation output: terrain occlusion and diffraction loss."""

    frame_id: str
    native_grid: GridSpec

    loss_db: np.ndarray
    occlusion_mask: np.ndarray
    dem_grid: Optional[np.ndarray] = None

    @property
    def grid(self) -> GridSpec:
        """Deprecated compatibility alias for native_grid."""
        warnings.warn(
            "TerrainState.grid is deprecated; use native_grid instead.",
            DeprecationWarning, stacklevel=2,
        )
        return object.__getattribute__(self, "native_grid")

    def __post_init__(self) -> None:
        ng = object.__getattribute__(self, "native_grid")
        if not isinstance(ng, GridSpec):
            raise TypeError(f"TerrainState.native_grid must be GridSpec, got {type(ng)!r}")
        expected = (ng.ny, ng.nx)
        for name in ("loss_db", "occlusion_mask"):
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(
                    f"TerrainState.{name} shape {arr.shape} != grid shape {expected}"
                )


@dataclass(frozen=True)
class UrbanRefinementState:
    """L3 propagation output: urban NLoS residual loss."""

    frame_id: str
    native_grid: GridSpec
    urban_grid: GridSpec

    urban_residual_db: np.ndarray
    support_mask: np.ndarray
    nlos_mask: np.ndarray

    @property
    def grid(self) -> GridSpec:
        """Deprecated compatibility alias for native_grid."""
        warnings.warn(
            "UrbanRefinementState.grid is deprecated; use native_grid instead.",
            DeprecationWarning, stacklevel=2,
        )
        return object.__getattribute__(self, "native_grid")

    def __post_init__(self) -> None:
        ng = object.__getattribute__(self, "native_grid")
        if not isinstance(ng, GridSpec):
            raise TypeError(f"UrbanRefinementState.native_grid must be GridSpec, got {type(ng)!r}")
        expected = (ng.ny, ng.nx)
        for name in ("urban_residual_db", "support_mask", "nlos_mask"):
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(
                    f"UrbanRefinementState.{name} shape {arr.shape} != grid shape {expected}"
                )
