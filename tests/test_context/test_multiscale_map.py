"""
Tests for MultiScaleMap: masked residual compositor and legacy additive path.
"""

import warnings
import pytest
import numpy as np
from datetime import datetime, timezone

from src.context.frame_context import FrameMismatchError
from src.context.grid_spec import GridSpec
from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from src.context.multiscale_map import MultiScaleMap, ShapeError


GRID = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
FRAME_ID = "test_frame"
N = 256


def _entry(frame_id=FRAME_ID, total=153.5, el=45.0, az=180.0):
    ones = np.ones((N, N), dtype=np.float32)
    return EntryWaveState(
        frame_id=frame_id, native_grid=GRID,
        total_loss_db=ones * total,
        fspl_db=ones * 180.0, atm_db=ones * 2.0,
        iono_db=ones * 1.0, pol_db=ones * 0.5, gain_db=ones * 30.0,
        elevation_deg=ones * el, azimuth_deg=ones * az,
        slant_range_m=ones * 600_000.0,
        occlusion_mask=np.zeros((N, N), dtype=bool),
    )


def _terrain(loss=20.0):
    loss_arr = np.full((N, N), loss, dtype=np.float32)
    return TerrainState(
        frame_id=FRAME_ID, native_grid=GRID,
        loss_db=loss_arr,
        occlusion_mask=np.zeros((N, N), dtype=bool),
    )


def _urban(residual=15.0, support_fraction=0.5):
    """Urban state with residual in top half, support in top half."""
    res = np.zeros((N, N), dtype=np.float32)
    support = np.zeros((N, N), dtype=bool)
    half = N // 2
    res[:half, :] = residual
    support[:half, :] = True
    nlos = res > 0
    return UrbanRefinementState(
        frame_id=FRAME_ID, native_grid=GRID,
        urban_grid=GRID,
        urban_residual_db=res,
        support_mask=support,
        nlos_mask=nlos,
    )


# ---------------------------------------------------------------------------
# compose(): masked residual compositor
# ---------------------------------------------------------------------------

class TestCompose:
    def test_all_layers_present(self):
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID, _entry(), _terrain(), _urban())
        assert msm.composite_db.shape == (N, N)
        assert msm.l1_db is not None
        assert msm.l2_db is not None
        assert msm.l3_db is not None
        assert msm.l3_support_mask is not None

    def test_composite_values_with_support_mask(self):
        """L3 residual applied only in top half (support_mask)."""
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID, _entry(total=153.5), _terrain(20.0), _urban(15.0))
        half = N // 2
        # Top half: l1 + l2 + l3
        np.testing.assert_allclose(msm.composite_db[:half, :], 153.5 + 20.0 + 15.0, atol=1e-4)
        # Bottom half: l1 + l2 only (no support)
        np.testing.assert_allclose(msm.composite_db[half:, :], 153.5 + 20.0, atol=1e-4)

    def test_l3_zero_outside_support(self):
        """l3_db must be zero outside support_mask."""
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID, _entry(), _terrain(), _urban())
        half = N // 2
        assert np.all(msm.l3_db[half:, :] == 0.0)

    def test_no_l1(self):
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID, entry=None, terrain=_terrain(20.0))
        assert msm.l1_db is None
        np.testing.assert_allclose(msm.composite_db, 20.0, atol=1e-4)

    def test_no_l2(self):
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID, entry=_entry(total=100.0), terrain=None)
        assert msm.l2_db is None
        np.testing.assert_allclose(msm.composite_db, 100.0, atol=1e-4)

    def test_no_l3(self):
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID, entry=_entry(total=100.0), terrain=_terrain(10.0))
        assert msm.l3_db is None
        assert msm.l3_support_mask is None
        np.testing.assert_allclose(msm.composite_db, 110.0, atol=1e-4)

    def test_all_absent_gives_zero(self):
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID)
        np.testing.assert_allclose(msm.composite_db, 0.0, atol=1e-4)

    def test_shape_error_on_mismatched_entry(self):
        bad_grid = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 128, 128)
        bad_entry = EntryWaveState(
            frame_id=FRAME_ID, native_grid=bad_grid,
            total_loss_db=np.ones((128, 128), dtype=np.float32),
            fspl_db=np.ones((128, 128), dtype=np.float32),
            atm_db=np.ones((128, 128), dtype=np.float32),
            iono_db=np.ones((128, 128), dtype=np.float32),
            pol_db=np.ones((128, 128), dtype=np.float32),
            gain_db=np.ones((128, 128), dtype=np.float32),
            elevation_deg=np.ones((128, 128), dtype=np.float32),
            azimuth_deg=np.ones((128, 128), dtype=np.float32),
            slant_range_m=np.ones((128, 128), dtype=np.float32),
            occlusion_mask=np.zeros((128, 128), dtype=bool),
        )
        with pytest.raises(ShapeError):
            MultiScaleMap.compose_legacy(FRAME_ID, GRID, entry=bad_entry)

    def test_frame_id_preserved(self):
        msm = MultiScaleMap.compose_legacy("my_frame", GRID, _entry("my_frame"))
        assert msm.frame_id == "my_frame"

    def test_grid_preserved(self):
        msm = MultiScaleMap.compose_legacy(FRAME_ID, GRID, _entry())
        assert msm.grid is GRID

    def test_frame_id_mismatch_raises(self):
        """compose() must raise FrameMismatchError when state.frame_id != frame_id."""
        wrong_entry = _entry("wrong_frame")
        with pytest.raises(FrameMismatchError):
            MultiScaleMap.compose_legacy(FRAME_ID, GRID, entry=wrong_entry)

    def test_terrain_frame_id_mismatch_raises(self):
        loss = np.zeros((N, N), dtype=np.float32)
        wrong_terrain = TerrainState(
            frame_id="wrong_frame", native_grid=GRID,
            loss_db=loss, occlusion_mask=np.zeros((N, N), dtype=bool),
        )
        with pytest.raises(FrameMismatchError):
            MultiScaleMap.compose_legacy(FRAME_ID, GRID, terrain=wrong_terrain)


# ---------------------------------------------------------------------------
# from_additive(): legacy path emits DeprecationWarning
# ---------------------------------------------------------------------------

class TestFromAdditive:
    def test_emits_deprecation_warning(self):
        l1 = np.full((N, N), 100.0, dtype=np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MultiScaleMap.from_additive(FRAME_ID, GRID, l1_map=l1)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1
            assert "from_additive" in str(dep[0].message)

    def test_additive_composite_values(self):
        l1 = np.full((N, N), 100.0, dtype=np.float32)
        l2 = np.full((N, N), 20.0, dtype=np.float32)
        l3 = np.full((N, N), 15.0, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            msm = MultiScaleMap.from_additive(FRAME_ID, GRID, l1_map=l1, l2_map=l2, l3_map=l3)
        np.testing.assert_allclose(msm.composite_db, 135.0, atol=1e-4)

    def test_no_support_mask_in_additive(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            msm = MultiScaleMap.from_additive(FRAME_ID, GRID, l1_map=np.ones((N, N), dtype=np.float32))
        assert msm.l3_support_mask is None

    def test_shape_error_on_bad_l2(self):
        with pytest.raises(ShapeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                MultiScaleMap.from_additive(
                    FRAME_ID, GRID,
                    l2_map=np.ones((128, 128), dtype=np.float32),
                )


# ---------------------------------------------------------------------------
# MultiScaleMap frozen dataclass validation
# ---------------------------------------------------------------------------

class TestMultiScaleMapValidation:
    def test_wrong_composite_shape_raises(self):
        with pytest.raises(ShapeError):
            MultiScaleMap(
                frame_id=FRAME_ID,
                grid=GRID,
                composite_db=np.zeros((128, 128), dtype=np.float32),
            )

    def test_wrong_l1_shape_raises(self):
        with pytest.raises(ShapeError):
            MultiScaleMap(
                frame_id=FRAME_ID,
                grid=GRID,
                composite_db=np.zeros((N, N), dtype=np.float32),
                l1_db=np.zeros((128, 128), dtype=np.float32),
            )

    def test_non_gridspec_raises(self):
        with pytest.raises(TypeError):
            MultiScaleMap(
                frame_id=FRAME_ID,
                grid="not_a_gridspec",
                composite_db=np.zeros((N, N), dtype=np.float32),
            )
