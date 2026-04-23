"""
Golden-scene test fixtures for grid geometry contracts (AC-1).

Tests:
- CoverageSpec 4-grid construction with role metadata
- Concentric center validation (positive and negative)
- Bbox nesting validation (positive and negative)
- GridSpec JSON round-trip serialization
- Deprecated field access warnings
- Grid role validation
"""

import json
import warnings

import pytest

from src.context.grid_spec import GridSpec, VALID_ROLES
from src.context.coverage_spec import CoverageSpec, BlendPolicy


# ── Fixtures ─────────────────────────────────────────────────────────────────

CENTER_LAT, CENTER_LON = 34.26, 108.94


def _make_coverage(
    l1_km=256.0, l2_km=25.6, l3_km=0.256, product_km=25.6,
    nx=256, ny=256,
) -> CoverageSpec:
    return CoverageSpec.from_config(
        origin_lat=CENTER_LAT, origin_lon=CENTER_LON,
        coarse_coverage_km=l1_km, coarse_nx=nx, coarse_ny=ny,
        product_coverage_km=product_km, product_nx=nx, product_ny=ny,
        l2_coverage_km=l2_km, l2_nx=nx, l2_ny=ny,
        urban_coverage_km=l3_km, urban_nx=nx, urban_ny=ny,
    )


# ── AC-1 Positive: 4-grid construction with role metadata ────────────────────

class TestCoverageSpec4Grid:
    def test_from_config_exposes_4_grids(self):
        cs = _make_coverage()
        assert cs.l1_grid.role == "l1_macro"
        assert cs.l2_grid.role == "l2_terrain"
        assert cs.l3_grid.role == "l3_urban"
        assert cs.product_grid.role == "product"

    def test_grid_widths_match_config(self):
        cs = _make_coverage()
        assert cs.l1_grid.width_m == pytest.approx(256_000.0)
        assert cs.l2_grid.width_m == pytest.approx(25_600.0)
        assert cs.l3_grid.width_m == pytest.approx(256.0)
        assert cs.product_grid.width_m == pytest.approx(25_600.0)

    def test_all_grids_concentric(self):
        cs = _make_coverage()
        assert cs.l1_grid.same_center(cs.l2_grid)
        assert cs.l1_grid.same_center(cs.l3_grid)
        assert cs.l1_grid.same_center(cs.product_grid)

    def test_bbox_nesting_l2_within_l1(self):
        cs = _make_coverage()
        assert cs.l1_grid.contains_bbox(cs.l2_grid)

    def test_bbox_nesting_l3_within_l2(self):
        cs = _make_coverage()
        assert cs.l2_grid.contains_bbox(cs.l3_grid)

    def test_crop_rule_stored(self):
        cs = _make_coverage()
        assert cs.crop_rule == "centered_crop"

    def test_no_l3_grid(self):
        cs = CoverageSpec.from_config(
            origin_lat=CENTER_LAT, origin_lon=CENTER_LON,
            coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
            product_coverage_km=25.6, product_nx=256, product_ny=256,
            l2_coverage_km=25.6, l2_nx=256, l2_ny=256,
        )
        assert cs.l3_grid is None

    def test_default_l2_equals_l1_geometry(self):
        cs = CoverageSpec.from_config(
            origin_lat=CENTER_LAT, origin_lon=CENTER_LON,
            coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
            product_coverage_km=25.6, product_nx=256, product_ny=256,
        )
        assert cs.l2_grid.width_m == cs.l1_grid.width_m
        assert cs.l2_grid.role == "l2_terrain"


# ── AC-1 Negative: geometry contract violations ──────────────────────────────

class TestCoverageSpecContractViolations:
    def test_non_concentric_l2_raises(self):
        l1 = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256, role="l1_macro")
        l2 = GridSpec.from_legacy_args(35.0, 108.0, 25.6, 256, 256, role="l2_terrain")
        prod = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256, role="product")
        with pytest.raises(ValueError, match="same center"):
            CoverageSpec(l1_grid=l1, l2_grid=l2, l3_grid=None,
                         product_grid=prod, crop_rule="centered_crop",
                         blend=BlendPolicy.default())

    def test_l3_exceeds_l2_raises(self):
        l1 = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256, role="l1_macro")
        l2 = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256, role="l2_terrain")
        big_l3 = GridSpec.from_legacy_args(34.0, 108.0, 300.0, 256, 256, role="l3_urban")
        prod = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256, role="product")
        with pytest.raises(ValueError, match="l3_grid bbox must be contained"):
            CoverageSpec(l1_grid=l1, l2_grid=l2, l3_grid=big_l3,
                         product_grid=prod, crop_rule="centered_crop",
                         blend=BlendPolicy.default())

    def test_wrong_role_raises(self):
        l1 = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256, role="legacy")
        l2 = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256, role="l2_terrain")
        prod = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256, role="product")
        with pytest.raises(ValueError, match="l1_grid.role must be"):
            CoverageSpec(l1_grid=l1, l2_grid=l2, l3_grid=None,
                         product_grid=prod, crop_rule="centered_crop",
                         blend=BlendPolicy.default())

    def test_invalid_crop_rule_raises(self):
        l1 = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256, role="l1_macro")
        l2 = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256, role="l2_terrain")
        prod = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 256, 256, role="product")
        with pytest.raises(ValueError, match="crop_rule"):
            CoverageSpec(l1_grid=l1, l2_grid=l2, l3_grid=None,
                         product_grid=prod, crop_rule="invalid",
                         blend=BlendPolicy.default())


# ── GridSpec JSON round-trip ─────────────────────────────────────────────────

class TestGridSpecSerialization:
    def test_to_dict_round_trip(self):
        g = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256, role="l1_macro")
        d = g.to_dict()
        g2 = GridSpec.from_dict(d)
        assert g == g2

    def test_to_json_round_trip(self):
        g = GridSpec.from_legacy_args(34.0, 108.0, 25.6, 128, 128, role="l2_terrain")
        j = g.to_json()
        g2 = GridSpec.from_json(j)
        assert g == g2

    def test_json_contains_role(self):
        g = GridSpec.from_legacy_args(34.0, 108.0, 0.256, 64, 64, role="l3_urban")
        d = json.loads(g.to_json())
        assert d["role"] == "l3_urban"

    def test_bbox_center_resolution_in_dict(self):
        g = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256, role="product")
        d = g.to_dict()
        assert d["center_lat"] == pytest.approx(34.0)
        assert d["center_lon"] == pytest.approx(108.0)
        assert d["dx_m"] == pytest.approx(1000.0)
        assert d["dy_m"] == pytest.approx(1000.0)
        assert d["width_m"] == pytest.approx(256_000.0)


# ── GridSpec role validation ─────────────────────────────────────────────────

class TestGridSpecRole:
    def test_valid_roles(self):
        for role in VALID_ROLES:
            g = GridSpec.from_legacy_args(34.0, 108.0, 10.0, 64, 64, role=role)
            assert g.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="role must be one of"):
            GridSpec.from_legacy_args(34.0, 108.0, 10.0, 64, 64, role="invalid")

    def test_with_role(self):
        g = GridSpec.from_legacy_args(34.0, 108.0, 10.0, 64, 64, role="l1_macro")
        g2 = g.with_role("l2_terrain")
        assert g2.role == "l2_terrain"
        assert g2.center_lat == g.center_lat


# ── Deprecated field access warnings ─────────────────────────────────────────

class TestDeprecatedAccess:
    def test_coarse_grid_warns(self):
        cs = _make_coverage()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = cs.coarse_grid
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_urban_grid_warns(self):
        cs = _make_coverage()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = cs.urban_grid
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_target_product_grid_warns(self):
        cs = _make_coverage()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = cs.target_product_grid
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
