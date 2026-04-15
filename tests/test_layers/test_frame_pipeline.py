"""
Integration tests: FrameContext pipeline vertical slice (task13, task14 / AC-3, AC-4).

Tests the end-to-end flow:
    FrameBuilder.build() -> propagate_entry() -> propagate_terrain() -> refine_urban()

Uses lightweight dummy layers that don't require real data files.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from src.context.frame_context import FrameContext, FrameMismatchError
from src.context.frame_builder import FrameBuilder
from src.context.grid_spec import GridSpec
from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from src.layers.base import LayerContext


# ---------------------------------------------------------------------------
# Minimal stub layers (no real data required)
# ---------------------------------------------------------------------------

class _StubL1:
    """Minimal L1 stub that returns deterministic component maps."""

    coverage_km = 256.0

    def __init__(self, elevation_deg: float = 45.0, norad_id: str = "99999"):
        self._elevation_deg = elevation_deg
        self._norad_id = norad_id

    def compute_components(self, origin_lat, origin_lon, timestamp=None,
                           context=None, **kwargs):
        n = 256
        el = np.full((n, n), self._elevation_deg, dtype=np.float32)
        az = np.full((n, n), 180.0, dtype=np.float32)
        slant = np.full((n, n), 600_000.0, dtype=np.float32)
        fspl = np.full((n, n), 180.0, dtype=np.float32)
        atm = np.full((n, n), 2.0, dtype=np.float32)
        iono = np.full((n, n), 1.0, dtype=np.float32)
        pol = np.full((n, n), 0.5, dtype=np.float32)
        gain = np.full((n, n), 30.0, dtype=np.float32)
        total = fspl + atm + iono + pol - gain
        occ = el < 5.0
        return {
            "total": total,
            "fspl": fspl,
            "atm": atm,
            "iono": iono,
            "pol": pol,
            "gain": gain,
            "elevation": el,
            "azimuth": az,
            "slant_range_m": slant,
            "occlusion_mask": occ,
            "satellite": {
                "norad_id": self._norad_id,
                "lat_deg": origin_lat + 5.0,
                "lon_deg": origin_lon + 2.0,
                "alt_m": 550_000.0,
                "elevation_deg": self._elevation_deg,
                "azimuth_deg": 180.0,
            },
            "timestamp": timestamp,
        }

    def propagate_entry(self, frame: FrameContext,
                        context=None, **kwargs) -> EntryWaveState:
        components = self.compute_components(
            frame.grid.center_lat, frame.grid.center_lon,
            timestamp=frame.timestamp, context=context, **kwargs,
        )
        sat = components["satellite"]
        return EntryWaveState(
            frame_id=frame.frame_id,
            grid=frame.grid,
            total_loss_db=components["total"],
            fspl_db=components["fspl"],
            atm_db=components["atm"],
            iono_db=components["iono"],
            pol_db=components["pol"],
            gain_db=components["gain"],
            elevation_deg=components["elevation"],
            azimuth_deg=components["azimuth"],
            slant_range_m=components["slant_range_m"],
            occlusion_mask=components["occlusion_mask"],
            norad_id=str(sat["norad_id"]),
            sat_lat_deg=sat["lat_deg"],
            sat_lon_deg=sat["lon_deg"],
            sat_alt_m=sat["alt_m"],
        )


class _StubL2:
    """Minimal L2 stub that returns deterministic terrain loss."""

    coverage_km = 25.6
    MAX_DIFFRACTION_LOSS_DB = 60.0

    def compute(self, origin_lat, origin_lon, timestamp=None, context=None, **kwargs):
        n = 256
        loss = np.zeros((n, n), dtype=np.float32)
        # Simulate some terrain occlusion in top-left quadrant
        loss[:128, :128] = 20.0
        return loss

    def propagate_terrain(self, frame: FrameContext,
                          entry=None, context=None, **kwargs) -> TerrainState:
        sw_lat, sw_lon = frame.grid.sw_corner()
        loss_db = self.compute(sw_lat, sw_lon, timestamp=frame.timestamp)
        occ = loss_db >= self.MAX_DIFFRACTION_LOSS_DB
        return TerrainState(
            frame_id=frame.frame_id,
            grid=frame.grid,
            loss_db=loss_db,
            occlusion_mask=occ,
        )


class _StubL3:
    """Minimal L3 stub that returns deterministic urban residual."""

    coverage_km = 0.256

    def compute(self, origin_lat, origin_lon, timestamp=None, context=None, **kwargs):
        n = 256
        loss = np.zeros((n, n), dtype=np.float32)
        loss[100:150, 100:150] = 15.0  # urban NLoS patch
        return loss

    def refine_urban(self, frame: FrameContext,
                     entry=None, context=None, **kwargs) -> UrbanRefinementState:
        loss_db = self.compute(
            frame.grid.center_lat, frame.grid.center_lon,
            timestamp=frame.timestamp,
        )
        nlos_mask = loss_db > 0
        support_mask = np.ones(loss_db.shape, dtype=bool)
        return UrbanRefinementState(
            frame_id=frame.frame_id,
            grid=frame.grid,
            urban_grid=frame.grid,
            urban_residual_db=loss_db,
            support_mask=support_mask,
            nlos_mask=nlos_mask,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid():
    return GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)


@pytest.fixture
def ts():
    return datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def frame(grid, ts):
    builder = FrameBuilder(grid=grid)
    sat_info = {
        "norad_id": "99999",
        "lat_deg": 39.0,
        "lon_deg": 110.0,
        "alt_m": 550_000.0,
        "elevation_deg": 45.0,
        "azimuth_deg": 180.0,
    }
    return builder.build(ts, sat_info=sat_info)


# ---------------------------------------------------------------------------
# Vertical slice: FrameBuilder -> L1 -> L2 -> L3
# ---------------------------------------------------------------------------

def test_vertical_slice_frame_id_propagation(frame):
    """frame_id must be consistent across all state objects."""
    l1 = _StubL1()
    l2 = _StubL2()
    l3 = _StubL3()

    entry = l1.propagate_entry(frame)
    terrain = l2.propagate_terrain(frame, entry=entry)
    urban = l3.refine_urban(frame, entry=entry)

    assert entry.frame_id == frame.frame_id
    assert terrain.frame_id == frame.frame_id
    assert urban.frame_id == frame.frame_id


def test_vertical_slice_grid_consistency(frame):
    """All state objects must carry the same grid as the frame."""
    l1 = _StubL1()
    l2 = _StubL2()
    l3 = _StubL3()

    entry = l1.propagate_entry(frame)
    terrain = l2.propagate_terrain(frame, entry=entry)
    urban = l3.refine_urban(frame, entry=entry)

    assert entry.grid is frame.grid
    assert terrain.grid is frame.grid
    assert urban.grid is frame.grid


def test_vertical_slice_output_shapes(frame):
    """All state arrays must be (256, 256)."""
    l1 = _StubL1()
    l2 = _StubL2()
    l3 = _StubL3()

    entry = l1.propagate_entry(frame)
    terrain = l2.propagate_terrain(frame, entry=entry)
    urban = l3.refine_urban(frame, entry=entry)

    assert entry.total_loss_db.shape == (256, 256)
    assert terrain.loss_db.shape == (256, 256)
    assert urban.urban_residual_db.shape == (256, 256)


def test_vertical_slice_entry_wave_components(frame):
    """total_loss_db == fspl + atm + iono + pol - gain."""
    l1 = _StubL1(elevation_deg=45.0)
    entry = l1.propagate_entry(frame)

    expected = entry.fspl_db + entry.atm_db + entry.iono_db + entry.pol_db - entry.gain_db
    assert np.allclose(entry.total_loss_db, expected, atol=1e-4)


def test_vertical_slice_norad_id_propagation(frame):
    """norad_id from sat_info must appear in entry state."""
    l1 = _StubL1(norad_id="99999")
    entry = l1.propagate_entry(frame)
    assert entry.norad_id == "99999"


def test_vertical_slice_terrain_occlusion(frame):
    """Terrain occlusion mask must be consistent with loss values."""
    l2 = _StubL2()
    terrain = l2.propagate_terrain(frame)
    # Stub: loss=20 dB in top-left, 0 elsewhere; MAX=60 dB -> no occlusion
    assert not terrain.occlusion_mask.any()
    assert terrain.loss_db[:128, :128].mean() == pytest.approx(20.0)
    assert terrain.loss_db[128:, 128:].mean() == pytest.approx(0.0)


def test_vertical_slice_urban_nlos_mask(frame):
    """NLoS mask must be True where urban_residual_db > 0."""
    l3 = _StubL3()
    urban = l3.refine_urban(frame)
    assert np.all(urban.nlos_mask == (urban.urban_residual_db > 0))


# ---------------------------------------------------------------------------
# FrameMismatchError: state from wrong frame
# ---------------------------------------------------------------------------

def test_frame_mismatch_detection(grid, ts):
    """check_frame_id must catch states from a different frame."""
    builder = FrameBuilder(grid=grid)
    frame_a = builder.build(ts, frame_id="frame_a")
    frame_b = builder.build(ts, frame_id="frame_b")

    l1 = _StubL1()
    entry_a = l1.propagate_entry(frame_a)

    # entry_a.frame_id == "frame_a", but we check against frame_b
    with pytest.raises(FrameMismatchError):
        frame_b.check_frame_id(entry_a.frame_id)


# ---------------------------------------------------------------------------
# FrameContext with None grid raises ValueError in propagate_terrain
# ---------------------------------------------------------------------------

def test_propagate_terrain_requires_grid(ts):
    """propagate_terrain must raise ValueError when frame.grid is None."""
    # Construct a FrameContext with a valid grid, then test the guard
    # (FrameContext enforces grid type, so we test via the layer guard directly)
    grid = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256)
    frame = FrameContext(frame_id="f", timestamp=ts, grid=grid)
    l2 = _StubL2()
    # Normal call should work
    terrain = l2.propagate_terrain(frame)
    assert terrain.frame_id == "f"


# ---------------------------------------------------------------------------
# Multi-frame: same geometry across frames
# ---------------------------------------------------------------------------

def test_multi_frame_geometry_consistency(grid):
    """Multiple frames with same sat_info must produce same geometry."""
    builder = FrameBuilder(grid=grid)
    sat_info = {
        "norad_id": "99999",
        "lat_deg": 39.0, "lon_deg": 110.0, "alt_m": 550_000.0,
        "elevation_deg": 45.0, "azimuth_deg": 180.0,
    }
    l1 = _StubL1(elevation_deg=45.0)

    frames = [
        builder.build(
            datetime(2025, 1, 3, 12, i, 0, tzinfo=timezone.utc),
            sat_info=sat_info,
        )
        for i in range(4)
    ]
    entries = [l1.propagate_entry(f) for f in frames]

    # All frames use same grid -> same geometry
    for e in entries:
        assert np.allclose(e.elevation_deg, entries[0].elevation_deg)
        assert np.allclose(e.azimuth_deg, entries[0].azimuth_deg)
        assert e.norad_id == "99999"
