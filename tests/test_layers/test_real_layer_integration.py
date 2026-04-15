"""
Real-layer integration tests for FrameContext pipeline (task11 / AC-3, AC-4).

Uses the actual L1MacroLayer, L2TopoLayer, and L3UrbanLayer classes with
monkeypatched data loaders so no real data files are required.

Covers:
- StrictModeError raised by L1 when no timestamp in strict mode
- FrameMismatchError raised by propagate_terrain / refine_urban
- DeprecationWarning emitted for extras injection alongside FrameContext
- DeprecationWarning emitted for incident_dir injection alongside FrameContext+entry
"""

import warnings
import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.context.frame_context import FrameContext, FrameMismatchError
from src.context.frame_builder import FrameBuilder
from src.context.grid_spec import GridSpec
from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from src.context.time_utils import StrictModeError
from src.layers.base import LayerContext
from src.layers.l1_macro import L1MacroLayer
from src.layers.l2_topo import L2TopoLayer
from src.layers.l3_urban import L3UrbanLayer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ORIGIN_LAT = 34.0
ORIGIN_LON = 108.0
TS_UTC = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
GRID_256 = GridSpec.from_legacy_args(ORIGIN_LAT, ORIGIN_LON, 256.0, 256, 256)


@pytest.fixture
def l1_config():
    return {
        "enabled": True,
        "tle_file": "data/l1_space/data/tle_data.txt",
        "frequency_ghz": 2.0,
        "coverage_km": 256.0,
        "grid_size": 256,
        "peak_gain_db": 30.0,
        "hpbw_az_deg": 60.0,
        "hpbw_el_deg": 60.0,
        "strict_data": False,
    }


@pytest.fixture
def l2_config():
    return {
        "enabled": True,
        "dem_file": "data/l2_topo/dem.tif",
        "frequency_ghz": 2.0,
        "satellite_elevation_deg": 45.0,
        "satellite_azimuth_deg": 180.0,
        "satellite_altitude_km": 550.0,
        "coverage_km": 25.6,
        "grid_size": 256,
        "resolution_m": 100.0,
    }


@pytest.fixture
def l3_config():
    return {
        "enabled": True,
        "tile_cache_root": "data/l3_urban/tiles",
        "nlos_loss_db": 20.0,
        "incident_dir": {"az_deg": 180.0, "el_deg": 45.0},
        "grid_size": 256,
        "coverage_km": 0.256,
        "resolution_m": 1.0,
    }


@pytest.fixture
def frame():
    builder = FrameBuilder(grid=GRID_256)
    sat_info = {
        "norad_id": "25544",
        "lat_deg": 39.0,
        "lon_deg": 110.0,
        "alt_m": 550_000.0,
        "elevation_deg": 45.0,
        "azimuth_deg": 180.0,
    }
    return builder.build(TS_UTC, sat_info=sat_info)


@pytest.fixture
def entry_state(frame):
    """A valid EntryWaveState for the given frame."""
    n = 256
    ones = np.ones((n, n), dtype=np.float32)
    return EntryWaveState(
        frame_id=frame.frame_id,
        grid=frame.grid,
        total_loss_db=ones * 153.5,
        fspl_db=ones * 180.0,
        atm_db=ones * 2.0,
        iono_db=ones * 1.0,
        pol_db=ones * 0.5,
        gain_db=ones * 30.0,
        elevation_deg=ones * 45.0,
        azimuth_deg=ones * 180.0,
        slant_range_m=ones * 600_000.0,
        occlusion_mask=np.zeros((n, n), dtype=bool),
        norad_id="25544",
        sat_lat_deg=39.0,
        sat_lon_deg=110.0,
        sat_alt_m=550_000.0,
    )


# ---------------------------------------------------------------------------
# L1: StrictModeError when no timestamp in strict mode
# ---------------------------------------------------------------------------

class TestL1StrictMode:
    def test_strict_mode_raises_on_missing_timestamp(self, l1_config, monkeypatch):
        """L1 must raise StrictModeError when strict_data=True and no timestamp."""
        cfg = dict(l1_config)
        cfg["strict_data"] = True

        layer = L1MacroLayer.__new__(L1MacroLayer)
        # Minimal init to avoid TLE file loading
        layer.strict_data = True
        layer.origin_lat = ORIGIN_LAT
        layer.origin_lon = ORIGIN_LON
        layer._sim_time = None
        layer.ts = MagicMock()

        from src.context.time_utils import StrictModeError as _SE
        with pytest.raises(_SE):
            layer._resolve_sim_datetime(None)

    def test_non_strict_mode_warns_on_missing_timestamp(self, monkeypatch):
        """L1 must emit DeprecationWarning (not raise) when strict_data=False and no timestamp."""
        layer = L1MacroLayer.__new__(L1MacroLayer)
        layer.strict_data = False
        layer.origin_lat = ORIGIN_LAT
        layer.origin_lon = ORIGIN_LON
        layer._sim_time = None

        mock_ts = MagicMock()
        mock_ts.from_datetime.return_value = MagicMock()
        layer.ts = mock_ts

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = layer._resolve_sim_datetime(None)
            assert result is not None
            assert result.tzinfo is not None
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

    def test_utc_timestamp_accepted(self, monkeypatch):
        """L1 must accept a UTC-aware datetime without raising."""
        layer = L1MacroLayer.__new__(L1MacroLayer)
        layer.strict_data = True
        layer.origin_lat = ORIGIN_LAT
        layer.origin_lon = ORIGIN_LON
        layer._sim_time = None

        mock_ts = MagicMock()
        mock_ts.from_datetime.return_value = MagicMock()
        layer.ts = mock_ts

        result = layer._resolve_sim_datetime(TS_UTC)
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# L2: FrameMismatchError in propagate_terrain
# ---------------------------------------------------------------------------

class TestL2FrameMismatch:
    def _make_l2_with_mock_dem(self, l2_config):
        """Create L2TopoLayer with DEM loading monkeypatched."""
        layer = L2TopoLayer(l2_config, ORIGIN_LAT, ORIGIN_LON)
        # Patch _load_dem_patch to return a synthetic DEM
        dem = np.zeros((256, 256), dtype=np.float32)
        dem[50:100, 50:100] = 500.0  # some terrain
        layer._load_dem_patch = MagicMock(return_value=dem)
        layer._validate_bounds = MagicMock()  # skip bounds check
        return layer

    def test_frame_mismatch_raises(self, l2_config, frame, entry_state):
        """propagate_terrain must raise FrameMismatchError when entry.frame_id != frame.frame_id."""
        layer = self._make_l2_with_mock_dem(l2_config)

        # Build a second frame with a different frame_id
        builder = FrameBuilder(grid=GRID_256)
        other_frame = builder.build(TS_UTC, frame_id="other_frame")

        # entry_state belongs to `frame`, not `other_frame`
        with pytest.raises(FrameMismatchError):
            layer.propagate_terrain(other_frame, entry=entry_state)

    def test_matching_frame_id_succeeds(self, l2_config, frame, entry_state):
        """propagate_terrain must succeed when entry.frame_id == frame.frame_id."""
        layer = self._make_l2_with_mock_dem(l2_config)
        terrain = layer.propagate_terrain(frame, entry=entry_state)
        assert terrain.frame_id == frame.frame_id
        assert terrain.loss_db.shape == (256, 256)

    def test_extras_injection_emits_deprecation_warning(self, l2_config, frame, entry_state):
        """propagate_terrain must emit DeprecationWarning when geometry extras are injected."""
        layer = self._make_l2_with_mock_dem(l2_config)
        ctx = LayerContext(extras={"satellite_elevation_deg": 30.0})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer.propagate_terrain(frame, entry=entry_state, context=ctx)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_no_warning_without_geometry_extras(self, l2_config, frame, entry_state):
        """propagate_terrain must NOT warn when context has no geometry extras."""
        layer = self._make_l2_with_mock_dem(l2_config)
        ctx = LayerContext(extras={"some_other_key": "value"})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer.propagate_terrain(frame, entry=entry_state, context=ctx)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


# ---------------------------------------------------------------------------
# L3: FrameMismatchError in refine_urban
# ---------------------------------------------------------------------------

class TestL3FrameMismatch:
    def _make_l3_with_mock_tile(self, l3_config):
        """Create L3UrbanLayer with tile loading monkeypatched."""
        layer = L3UrbanLayer(l3_config, ORIGIN_LAT, ORIGIN_LON)
        height_m = np.zeros((256, 256), dtype=np.float32)
        height_m[80:120, 80:120] = 15.0  # building block
        layer._load_tile_cache = MagicMock(return_value=(height_m, None))
        return layer

    def test_frame_mismatch_raises(self, l3_config, frame, entry_state):
        """refine_urban must raise FrameMismatchError when entry.frame_id != frame.frame_id."""
        layer = self._make_l3_with_mock_tile(l3_config)

        builder = FrameBuilder(grid=GRID_256)
        other_frame = builder.build(TS_UTC, frame_id="other_frame")

        with pytest.raises(FrameMismatchError):
            layer.refine_urban(other_frame, entry=entry_state)

    def test_matching_frame_id_succeeds(self, l3_config, frame, entry_state):
        """refine_urban must succeed when entry.frame_id == frame.frame_id."""
        layer = self._make_l3_with_mock_tile(l3_config)
        urban = layer.refine_urban(frame, entry=entry_state)
        assert urban.frame_id == frame.frame_id
        assert urban.urban_residual_db.shape == (256, 256)

    def test_incident_dir_injection_emits_deprecation_warning(self, l3_config, frame, entry_state):
        """refine_urban must emit DeprecationWarning when incident_dir is in context alongside entry."""
        layer = self._make_l3_with_mock_tile(l3_config)
        ctx = LayerContext(incident_dir={"az_deg": 90.0, "el_deg": 30.0})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer.refine_urban(frame, entry=entry_state, context=ctx)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_no_warning_without_incident_dir_in_context(self, l3_config, frame, entry_state):
        """refine_urban must NOT warn when context has no incident_dir."""
        layer = self._make_l3_with_mock_tile(l3_config)
        ctx = LayerContext(extras={"tile_id": "test_tile"})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer.refine_urban(frame, entry=entry_state, context=ctx)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0

    def test_nlos_mask_consistent_with_loss(self, l3_config, frame, entry_state):
        """nlos_mask must be True exactly where urban_residual_db > 0."""
        layer = self._make_l3_with_mock_tile(l3_config)
        urban = layer.refine_urban(frame, entry=entry_state)
        assert np.all(urban.nlos_mask == (urban.urban_residual_db > 0))
