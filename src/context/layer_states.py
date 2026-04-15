"""
Typed per-layer state objects for the SG-MRM pipeline.

Each propagation step returns a typed, immutable state object that carries
the frame_id, grid, and all computed arrays. Downstream steps validate
frame_id consistency before consuming state.

State hierarchy:
    EntryWaveState   — L1 output (free-space + atmospheric + ionospheric)
    TerrainState     — L2 output (terrain occlusion + diffraction)
    UrbanRefinementState — L3 output (urban NLoS residual)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid_spec import GridSpec


@dataclass(frozen=True)
class EntryWaveState:
    """
    L1 propagation output: entry-wave loss components.

    All loss arrays are float32, shape (ny, nx), in dB.
    Positive values = loss (signal attenuation).
    gain_db is positive = gain (subtracted from total).

    total_loss_db = fspl_db + atm_db + iono_db + pol_db - gain_db
    """

    frame_id: str
    grid: GridSpec

    # Component maps (dB)
    total_loss_db: np.ndarray       # net loss = fspl + atm + iono + pol - gain
    fspl_db: np.ndarray             # free-space path loss
    atm_db: np.ndarray              # atmospheric loss
    iono_db: np.ndarray             # ionospheric loss
    pol_db: np.ndarray              # polarization loss
    gain_db: np.ndarray             # antenna gain (positive = gain)

    # Geometry maps
    elevation_deg: np.ndarray       # per-pixel elevation angle to satellite
    azimuth_deg: np.ndarray         # per-pixel azimuth angle to satellite
    slant_range_m: np.ndarray       # per-pixel slant range in metres
    occlusion_mask: np.ndarray      # True where satellite is below horizon

    # Satellite metadata
    norad_id: Optional[str] = None
    sat_lat_deg: Optional[float] = None
    sat_lon_deg: Optional[float] = None
    sat_alt_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridSpec):
            raise TypeError(f"EntryWaveState.grid must be GridSpec, got {type(self.grid)!r}")
        expected = (self.grid.ny, self.grid.nx)
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
    """
    L2 propagation output: terrain occlusion and diffraction loss.

    loss_db is float32, shape (ny, nx), in dB.
    occlusion_mask is bool, shape (ny, nx).
    """

    frame_id: str
    grid: GridSpec

    loss_db: np.ndarray             # terrain diffraction/occlusion loss (dB)
    occlusion_mask: np.ndarray      # True where terrain blocks LOS
    dem_grid: Optional[np.ndarray] = None  # raw DEM elevation (m), may be None

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridSpec):
            raise TypeError(f"TerrainState.grid must be GridSpec, got {type(self.grid)!r}")
        expected = (self.grid.ny, self.grid.nx)
        for name in ("loss_db", "occlusion_mask"):
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(
                    f"TerrainState.{name} shape {arr.shape} != grid shape {expected}"
                )


@dataclass(frozen=True)
class UrbanRefinementState:
    """
    L3 propagation output: urban NLoS residual loss.

    urban_residual_db is float32, shape (ny, nx), in dB.
    This is a *residual* — it is added only within support_mask.
    Outside support_mask (no urban tile), the value is zero.
    """

    frame_id: str
    grid: GridSpec                  # L3 tile grid (256 m coverage)
    urban_grid: GridSpec            # same as grid; kept for explicit typing

    urban_residual_db: np.ndarray   # NLoS residual loss (dB), zero outside support
    support_mask: np.ndarray        # True where urban tile data is available
    nlos_mask: np.ndarray           # True where NLoS condition holds

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridSpec):
            raise TypeError(f"UrbanRefinementState.grid must be GridSpec, got {type(self.grid)!r}")
        expected = (self.grid.ny, self.grid.nx)
        for name in ("urban_residual_db", "support_mask", "nlos_mask"):
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(
                    f"UrbanRefinementState.{name} shape {arr.shape} != grid shape {expected}"
                )
