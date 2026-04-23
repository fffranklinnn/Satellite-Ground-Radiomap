"""
Unit tests for typed layer state objects (task8 / AC-4).
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from src.context.grid_spec import GridSpec
from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState


def _grid(coverage_km=256.0, nx=256, ny=256):
    return GridSpec.from_legacy_args(34.0, 108.0, coverage_km, nx, ny)


def _zeros(grid):
    return np.zeros((grid.ny, grid.nx), dtype=np.float32)


def _bool_zeros(grid):
    return np.zeros((grid.ny, grid.nx), dtype=bool)


# ---------------------------------------------------------------------------
# EntryWaveState
# ---------------------------------------------------------------------------

def test_entry_wave_state_basic():
    g = _grid()
    state = EntryWaveState(
        frame_id="f1",
        native_grid=g,
        total_loss_db=_zeros(g),
        fspl_db=_zeros(g),
        atm_db=_zeros(g),
        iono_db=_zeros(g),
        pol_db=_zeros(g),
        gain_db=_zeros(g),
        elevation_deg=_zeros(g),
        azimuth_deg=_zeros(g),
        slant_range_m=_zeros(g),
        occlusion_mask=_bool_zeros(g),
    )
    assert state.frame_id == "f1"
    assert state.total_loss_db.shape == (256, 256)


def test_entry_wave_state_wrong_shape():
    g = _grid()
    bad = np.zeros((128, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        EntryWaveState(
            frame_id="f1", native_grid=g,
            total_loss_db=bad,
            fspl_db=_zeros(g), atm_db=_zeros(g), iono_db=_zeros(g),
            pol_db=_zeros(g), gain_db=_zeros(g),
            elevation_deg=_zeros(g), azimuth_deg=_zeros(g),
            slant_range_m=_zeros(g), occlusion_mask=_bool_zeros(g),
        )


def test_entry_wave_state_wrong_grid_type():
    with pytest.raises(TypeError, match="GridSpec"):
        EntryWaveState(
            frame_id="f1", native_grid="not_a_grid",  # type: ignore
            total_loss_db=np.zeros((256, 256)),
            fspl_db=np.zeros((256, 256)), atm_db=np.zeros((256, 256)),
            iono_db=np.zeros((256, 256)), pol_db=np.zeros((256, 256)),
            gain_db=np.zeros((256, 256)), elevation_deg=np.zeros((256, 256)),
            azimuth_deg=np.zeros((256, 256)), slant_range_m=np.zeros((256, 256)),
            occlusion_mask=np.zeros((256, 256), dtype=bool),
        )


def test_entry_wave_state_frozen():
    g = _grid()
    state = EntryWaveState(
        frame_id="f1", native_grid=g,
        total_loss_db=_zeros(g), fspl_db=_zeros(g), atm_db=_zeros(g),
        iono_db=_zeros(g), pol_db=_zeros(g), gain_db=_zeros(g),
        elevation_deg=_zeros(g), azimuth_deg=_zeros(g),
        slant_range_m=_zeros(g), occlusion_mask=_bool_zeros(g),
    )
    with pytest.raises(Exception):
        state.frame_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TerrainState
# ---------------------------------------------------------------------------

def test_terrain_state_basic():
    g = _grid(coverage_km=25.6)
    state = TerrainState(
        frame_id="f1",
        native_grid=g,
        loss_db=_zeros(g),
        occlusion_mask=_bool_zeros(g),
    )
    assert state.frame_id == "f1"
    assert state.dem_grid is None


def test_terrain_state_wrong_shape():
    g = _grid(coverage_km=25.6)
    bad = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        TerrainState(frame_id="f1", native_grid=g, loss_db=bad, occlusion_mask=_bool_zeros(g))


def test_terrain_state_frozen():
    g = _grid(coverage_km=25.6)
    state = TerrainState(frame_id="f1", native_grid=g, loss_db=_zeros(g), occlusion_mask=_bool_zeros(g))
    with pytest.raises(Exception):
        state.frame_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# UrbanRefinementState
# ---------------------------------------------------------------------------

def test_urban_refinement_state_basic():
    g = _grid(coverage_km=0.256)
    state = UrbanRefinementState(
        frame_id="f1",
        native_grid=g,
        urban_grid=g,
        urban_residual_db=_zeros(g),
        support_mask=np.ones((g.ny, g.nx), dtype=bool),
        nlos_mask=_bool_zeros(g),
    )
    assert state.frame_id == "f1"
    assert state.urban_residual_db.shape == (256, 256)


def test_urban_refinement_state_wrong_shape():
    g = _grid(coverage_km=0.256)
    bad = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        UrbanRefinementState(
            frame_id="f1", native_grid=g, urban_grid=g,
            urban_residual_db=bad,
            support_mask=np.ones((g.ny, g.nx), dtype=bool),
            nlos_mask=_bool_zeros(g),
        )


def test_urban_refinement_state_frozen():
    g = _grid(coverage_km=0.256)
    state = UrbanRefinementState(
        frame_id="f1", native_grid=g, urban_grid=g,
        urban_residual_db=_zeros(g),
        support_mask=np.ones((g.ny, g.nx), dtype=bool),
        nlos_mask=_bool_zeros(g),
    )
    with pytest.raises(Exception):
        state.frame_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Non-square grids
# ---------------------------------------------------------------------------

def test_states_with_non_square_grid():
    g = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 128, 256)
    state = TerrainState(
        frame_id="f1",
        native_grid=g,
        loss_db=np.zeros((256, 128), dtype=np.float32),
        occlusion_mask=np.zeros((256, 128), dtype=bool),
    )
    assert state.loss_db.shape == (256, 128)
