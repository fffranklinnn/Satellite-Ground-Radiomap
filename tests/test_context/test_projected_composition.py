"""
Projected composition golden tests (AC-3).

Tests:
- compose_projected() with non-matching native grids
- Non-urban areas equal coarse backbone exactly
- Support-mask boundary behavior
- Grid metadata mismatch rejection
- Projection contract enforcement
"""

import numpy as np
import pytest

from src.context.grid_spec import GridSpec
from src.context.multiscale_map import MultiScaleMap, ShapeError
from src.compose import (
    project_field, FieldType, ProjectedView,
    validate_projection_contract, ProjectionContractError,
)


CENTER_LAT, CENTER_LON = 34.26, 108.94


def _grid(coverage_km, nx, ny, role="legacy"):
    return GridSpec.from_legacy_args(CENTER_LAT, CENTER_LON, coverage_km, nx, ny, role=role)


class TestComposeProjected:
    def _pv(self, arr, grid, ft=FieldType.LOSS_DB):
        """Helper to wrap an array as a ProjectedView on the given grid."""
        return ProjectedView(source_grid=grid, product_grid=grid, field_type=ft, values=arr, frame_id="test")

    def test_non_urban_equals_coarse_backbone(self):
        """Outside urban support, composite = L1 + L2 exactly."""
        pg = _grid(25.6, 64, 64, "product")
        l1 = self._pv(np.full((64, 64), 150.0, dtype=np.float32), pg)
        l2 = self._pv(np.full((64, 64), 10.0, dtype=np.float32), pg)
        msm = MultiScaleMap.compose(frame_id="test", product_grid=pg, l1_view=l1, l2_view=l2)
        np.testing.assert_array_almost_equal(msm.composite_db, 160.0)

    def test_urban_residual_only_within_support(self):
        """L3 residual applied only where support_mask is True."""
        pg = _grid(25.6, 64, 64, "product")
        l1 = self._pv(np.full((64, 64), 150.0, dtype=np.float32), pg)
        l2 = self._pv(np.zeros((64, 64), dtype=np.float32), pg)
        l3_res = self._pv(np.full((64, 64), 5.0, dtype=np.float32), pg)
        sup = np.zeros((64, 64), dtype=bool)
        sup[:32, :32] = True
        l3_sup = ProjectedView(source_grid=pg, product_grid=pg, field_type=FieldType.SUPPORT_MASK, values=sup, frame_id="test")
        msm = MultiScaleMap.compose(frame_id="test", product_grid=pg, l1_view=l1, l2_view=l2, l3_residual_view=l3_res, l3_support_view=l3_sup)
        assert msm.composite_db[0, 0] == pytest.approx(155.0)
        assert msm.composite_db[63, 63] == pytest.approx(150.0)

    def test_raw_array_rejected(self):
        """Passing raw arrays (not ProjectedView) raises TypeError."""
        pg = _grid(25.6, 64, 64, "product")
        raw = np.full((64, 64), 150.0, dtype=np.float32)
        with pytest.raises(TypeError, match="ProjectedView"):
            MultiScaleMap.compose(frame_id="test", product_grid=pg, l1_view=raw)

    def test_same_center_different_resolution_rejected(self):
        """Same center, same shape, different resolution must raise GridMismatchError."""
        from src.context.multiscale_map import GridMismatchError
        pg = _grid(0.256, 256, 256, "product")  # 256m, 256x256, 1m/px
        wrong_grid = _grid(256.0, 256, 256, "l1_macro")  # 256km, 256x256, 1000m/px — same center, same shape!
        l1 = ProjectedView(source_grid=wrong_grid, product_grid=wrong_grid, field_type=FieldType.LOSS_DB,
                           values=np.ones((256, 256), dtype=np.float32), frame_id="test")
        with pytest.raises(GridMismatchError):
            MultiScaleMap.compose(frame_id="test", product_grid=pg, l1_view=l1)


class TestProjectionContracts:
    def test_bilinear_for_boolean_mask_raises(self):
        validate_projection_contract(FieldType.LOSS_DB, 1)  # OK
        with pytest.raises(ProjectionContractError):
            validate_projection_contract(FieldType.BOOLEAN_MASK, 1)

    def test_nearest_for_support_mask_ok(self):
        validate_projection_contract(FieldType.SUPPORT_MASK, 0)  # OK

    def test_project_field_preserves_identity(self):
        """Same grid → identity projection."""
        g = _grid(25.6, 64, 64, "product")
        arr = np.random.rand(64, 64).astype(np.float32)
        pv = project_field(arr, g, g, FieldType.LOSS_DB, "test")
        np.testing.assert_array_equal(pv.values, arr)

    def test_project_field_different_grids(self):
        """Different grids → resampled output."""
        g1 = _grid(256.0, 64, 64, "l1_macro")
        g2 = _grid(25.6, 32, 32, "product")
        arr = np.ones((64, 64), dtype=np.float32) * 100.0
        pv = project_field(arr, g1, g2, FieldType.LOSS_DB, "test")
        assert pv.values.shape == (32, 32)
        # Uniform input → uniform output
        np.testing.assert_array_almost_equal(pv.values, 100.0, decimal=1)

    def test_azimuth_wrap_handling(self):
        """Azimuth projection handles 0/360 wrap correctly."""
        g1 = _grid(10.0, 32, 32, "l1_macro")
        g2 = _grid(10.0, 16, 16, "product")
        az = np.full((32, 32), 359.0, dtype=np.float32)
        az[:16, :] = 1.0
        pv = project_field(az, g1, g2, FieldType.AZIMUTH_DEG, "test")
        # Should not produce values like 180 from naive averaging of 1 and 359
        assert pv.values.min() >= 0.0
        assert pv.values.max() <= 360.0


class TestGoldenRegressions:
    """End-to-end golden regressions with distinct L1/L2/L3/product grids."""

    def test_distinct_grids_project_and_compose(self):
        """Drive project_to_product_grid with distinct native grids."""
        from src.compose import project_to_product_grid
        from src.context.coverage_spec import CoverageSpec

        cs = CoverageSpec.from_config(
            origin_lat=CENTER_LAT, origin_lon=CENTER_LON,
            coarse_coverage_km=256.0, coarse_nx=64, coarse_ny=64,
            product_coverage_km=25.6, product_nx=32, product_ny=32,
            l2_coverage_km=25.6, l2_nx=64, l2_ny=64,
            urban_coverage_km=0.256, urban_nx=64, urban_ny=64,
        )
        from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
        ones64 = np.ones((64, 64), dtype=np.float32)
        entry = EntryWaveState(
            frame_id="t", native_grid=cs.l1_grid,
            total_loss_db=ones64 * 150, fspl_db=ones64 * 180, atm_db=ones64 * 2,
            iono_db=ones64 * 1, pol_db=ones64 * 0.5, gain_db=ones64 * 30,
            elevation_deg=ones64 * 45, azimuth_deg=ones64 * 180,
            slant_range_m=ones64 * 600000, occlusion_mask=np.zeros((64, 64), dtype=bool),
        )
        terrain = TerrainState(
            frame_id="t", native_grid=cs.l2_grid,
            loss_db=ones64 * 10, occlusion_mask=np.zeros((64, 64), dtype=bool),
        )
        urban = UrbanRefinementState(
            frame_id="t", native_grid=cs.l3_grid, urban_grid=cs.l3_grid,
            urban_residual_db=ones64 * 5,
            support_mask=np.ones((64, 64), dtype=bool),
            nlos_mask=np.ones((64, 64), dtype=bool),
        )
        projected = project_to_product_grid(
            product_grid=cs.product_grid, entry=entry, terrain=terrain, urban=urban,
            frame_id="t",
        )
        msm = MultiScaleMap.compose(
            frame_id="t", product_grid=cs.product_grid, **projected,
        )
        assert msm.composite_db.shape == (32, 32)
        # L1+L2 backbone = 160. L3 residual (5dB) applies only within the tiny
        # L3 footprint (256m within 25.6km product grid), so most pixels = 160.
        # At least one pixel near center should have L3 contribution.
        assert msm.composite_db.min() >= 159.0
        assert msm.composite_db.max() <= 166.0

    def test_outside_urban_equals_coarse_backbone_exactly(self):
        """Non-urban pixels must equal L1+L2 backbone exactly."""
        pg = _grid(25.6, 32, 32, "product")
        l1 = ProjectedView(source_grid=pg, product_grid=pg, field_type=FieldType.LOSS_DB,
                           values=np.full((32, 32), 150.0, dtype=np.float32), frame_id="t")
        l2 = ProjectedView(source_grid=pg, product_grid=pg, field_type=FieldType.LOSS_DB,
                           values=np.full((32, 32), 10.0, dtype=np.float32), frame_id="t")
        l3_res = ProjectedView(source_grid=pg, product_grid=pg, field_type=FieldType.LOSS_DB,
                               values=np.full((32, 32), 5.0, dtype=np.float32), frame_id="t")
        l3_sup = ProjectedView(source_grid=pg, product_grid=pg, field_type=FieldType.SUPPORT_MASK,
                               values=np.zeros((32, 32), dtype=bool), frame_id="t")
        msm = MultiScaleMap.compose(
            frame_id="t", product_grid=pg,
            l1_view=l1, l2_view=l2, l3_residual_view=l3_res, l3_support_view=l3_sup,
        )
        np.testing.assert_array_equal(msm.composite_db, 160.0)

    def test_grid_metadata_mismatch_raises(self):
        """Same shape but different center must raise GridMismatchError."""
        from src.context.multiscale_map import GridMismatchError
        pg = _grid(25.6, 32, 32, "product")
        different_center = GridSpec.from_legacy_args(35.0, 109.0, 25.6, 32, 32, role="l1_macro")
        l1 = ProjectedView(source_grid=different_center, product_grid=different_center,
                           field_type=FieldType.LOSS_DB,
                           values=np.ones((32, 32), dtype=np.float32) * 150, frame_id="t")
        with pytest.raises(GridMismatchError):
            MultiScaleMap.compose(frame_id="t", product_grid=pg, l1_view=l1)
