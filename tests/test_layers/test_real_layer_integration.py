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
        native_grid=frame.grid,
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


# ---------------------------------------------------------------------------
# Geometry consistency: single frame preserves az/el/slant_range/norad_id/timestamp
# across L1 → L2 → L3 propagation
# ---------------------------------------------------------------------------

class TestGeometryConsistency:
    """
    Verify that a single FrameContext carries consistent geometry metadata
    from L1 (EntryWaveState) into L2 (TerrainState) and L3 (UrbanRefinementState).

    The key invariants:
    - All states share the same frame_id
    - L2 uses center-pixel az/el from entry state for its satellite geometry
    - L3 uses center-pixel az/el from entry state for its incident direction
    - norad_id from entry state is preserved
    - timestamp from frame is preserved in all states
    """

    def _make_l2_with_mock_dem(self, l2_config):
        layer = L2TopoLayer(l2_config, ORIGIN_LAT, ORIGIN_LON)
        dem = np.zeros((256, 256), dtype=np.float32)
        layer._load_dem_patch = MagicMock(return_value=dem)
        layer._validate_bounds = MagicMock()
        return layer

    def _make_l3_with_mock_tile(self, l3_config):
        layer = L3UrbanLayer(l3_config, ORIGIN_LAT, ORIGIN_LON)
        height_m = np.zeros((256, 256), dtype=np.float32)
        layer._load_tile_cache = MagicMock(return_value=(height_m, None))
        return layer

    def test_frame_id_consistent_across_all_states(
        self, l2_config, l3_config, frame, entry_state
    ):
        """All three states must carry the same frame_id as the frame."""
        l2 = self._make_l2_with_mock_dem(l2_config)
        l3 = self._make_l3_with_mock_tile(l3_config)

        terrain = l2.propagate_terrain(frame, entry=entry_state)
        urban = l3.refine_urban(frame, entry=entry_state)

        assert entry_state.frame_id == frame.frame_id
        assert terrain.frame_id == frame.frame_id
        assert urban.frame_id == frame.frame_id

    def test_norad_id_preserved_in_entry_state(self, frame, entry_state):
        """norad_id from entry_state must match the frame's satellite."""
        assert entry_state.norad_id == "25544"

    def test_timestamp_preserved_in_frame(self, frame):
        """Frame timestamp must be UTC-aware and match the construction timestamp."""
        assert frame.timestamp.tzinfo is not None
        assert frame.timestamp == TS_UTC

    def test_center_pixel_elevation_consistent_l1_to_l2(
        self, l2_config, frame, entry_state
    ):
        """L2 must use center-pixel elevation from entry state for satellite geometry."""
        l2 = self._make_l2_with_mock_dem(l2_config)

        # Capture what elevation L2 uses by inspecting the compute call
        captured_kwargs = {}
        original_compute = l2.compute

        def capturing_compute(origin_lat, origin_lon, timestamp=None, context=None, **kwargs):
            captured_kwargs.update(context.extras if context else {})
            return original_compute(origin_lat, origin_lon, timestamp=timestamp, context=context, **kwargs)

        l2.compute = capturing_compute
        l2.propagate_terrain(frame, entry=entry_state)

        # L2 should have received the center-pixel elevation from entry_state
        cy, cx = entry_state.grid.ny // 2, entry_state.grid.nx // 2
        expected_el = float(entry_state.elevation_deg[cy, cx])
        assert "satellite_elevation_deg" in captured_kwargs
        assert abs(captured_kwargs["satellite_elevation_deg"] - expected_el) < 1e-4

    def test_center_pixel_azimuth_consistent_l1_to_l2(
        self, l2_config, frame, entry_state
    ):
        """L2 must use center-pixel azimuth from entry state for satellite geometry."""
        l2 = self._make_l2_with_mock_dem(l2_config)

        captured_kwargs = {}
        original_compute = l2.compute

        def capturing_compute(origin_lat, origin_lon, timestamp=None, context=None, **kwargs):
            captured_kwargs.update(context.extras if context else {})
            return original_compute(origin_lat, origin_lon, timestamp=timestamp, context=context, **kwargs)

        l2.compute = capturing_compute
        l2.propagate_terrain(frame, entry=entry_state)

        cy, cx = entry_state.grid.ny // 2, entry_state.grid.nx // 2
        expected_az = float(entry_state.azimuth_deg[cy, cx])
        assert "satellite_azimuth_deg" in captured_kwargs
        assert abs(captured_kwargs["satellite_azimuth_deg"] - expected_az) < 1e-4

    def test_center_pixel_elevation_consistent_l1_to_l3(
        self, l3_config, frame, entry_state
    ):
        """L3 must derive incident_dir from center-pixel az/el of entry state."""
        l3 = self._make_l3_with_mock_tile(l3_config)

        # Capture the incident_dir used by L3
        captured_incident = {}
        original_compute = l3.compute

        def capturing_compute(origin_lat=None, origin_lon=None, timestamp=None, context=None, **kwargs):
            if context is not None:
                from src.layers.base import LayerContext
                ctx = LayerContext.from_any(context)
                if ctx.incident_dir is not None:
                    captured_incident["incident_dir"] = ctx.incident_dir
            return original_compute(
                origin_lat=origin_lat, origin_lon=origin_lon,
                timestamp=timestamp, context=context, **kwargs
            )

        l3.compute = capturing_compute
        l3.refine_urban(frame, entry=entry_state)

        cy, cx = entry_state.grid.ny // 2, entry_state.grid.nx // 2
        expected_az = float(entry_state.azimuth_deg[cy, cx])
        expected_el = float(entry_state.elevation_deg[cy, cx])

        assert "incident_dir" in captured_incident
        inc = captured_incident["incident_dir"]
        assert abs(inc["az_deg"] - expected_az) < 1e-4
        assert abs(inc["el_deg"] - expected_el) < 1e-4

    def test_slant_range_propagated_to_l2(self, l2_config, frame, entry_state):
        """L2 uses satellite_altitude_km from frame for slant range estimation."""
        l2 = self._make_l2_with_mock_dem(l2_config)

        captured_kwargs = {}
        original_compute = l2.compute

        def capturing_compute(origin_lat, origin_lon, timestamp=None, context=None, **kwargs):
            captured_kwargs.update(context.extras if context else {})
            return original_compute(origin_lat, origin_lon, timestamp=timestamp, context=context, **kwargs)

        l2.compute = capturing_compute
        l2.propagate_terrain(frame, entry=entry_state)

        # L2 receives satellite_altitude_km from frame.sat_alt_m (set by FrameBuilder)
        # slant_range_km is estimated internally when not explicitly provided
        assert "satellite_elevation_deg" in captured_kwargs
        assert "satellite_azimuth_deg" in captured_kwargs
        # altitude is passed when frame.sat_alt_m is set
        if frame.sat_alt_m is not None:
            assert "satellite_altitude_km" in captured_kwargs
            assert abs(captured_kwargs["satellite_altitude_km"] - frame.sat_alt_m / 1000.0) < 0.1

    def test_grid_consistent_across_all_states(
        self, l2_config, l3_config, frame, entry_state
    ):
        """All states must carry the same GridSpec as the frame."""
        l2 = self._make_l2_with_mock_dem(l2_config)
        l3 = self._make_l3_with_mock_tile(l3_config)

        terrain = l2.propagate_terrain(frame, entry=entry_state)
        urban = l3.refine_urban(frame, entry=entry_state)

        assert entry_state.grid is frame.grid
        assert terrain.native_grid is object.__getattribute__(frame, "grid")
        assert urban.native_grid is object.__getattribute__(frame, "grid")
