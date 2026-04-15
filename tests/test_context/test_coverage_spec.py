"""Unit tests for CoverageSpec and BlendPolicy (task6 / AC-2)."""

import pytest
from src.context.grid_spec import GridSpec
from src.context.coverage_spec import BlendPolicy, CoverageSpec


# ---------------------------------------------------------------------------
# BlendPolicy
# ---------------------------------------------------------------------------

def test_blend_policy_default():
    bp = BlendPolicy.default()
    assert bp.alignment_rule == "same_center"
    assert bp.crop_policy == "bbox_project"
    assert bp.blend_policy == "coarse_plus_masked_residual"


def test_blend_policy_invalid_alignment():
    with pytest.raises(ValueError, match="alignment_rule"):
        BlendPolicy(alignment_rule="invalid", crop_policy="bbox_project",
                    blend_policy="additive")


def test_blend_policy_invalid_crop():
    with pytest.raises(ValueError, match="crop_policy"):
        BlendPolicy(alignment_rule="same_center", crop_policy="invalid",
                    blend_policy="additive")


def test_blend_policy_invalid_blend():
    with pytest.raises(ValueError, match="blend_policy"):
        BlendPolicy(alignment_rule="same_center", crop_policy="bbox_project",
                    blend_policy="invalid")


def test_blend_policy_additive_valid():
    bp = BlendPolicy(alignment_rule="same_center", crop_policy="bbox_project",
                     blend_policy="additive")
    assert bp.blend_policy == "additive"


def test_blend_policy_frozen():
    bp = BlendPolicy.default()
    with pytest.raises(Exception):
        bp.alignment_rule = "same_bbox"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CoverageSpec.from_config
# ---------------------------------------------------------------------------

def test_coverage_spec_from_config_basic():
    cs = CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
        product_coverage_km=0.256, product_nx=256, product_ny=256,
    )
    assert isinstance(cs.coarse_grid, GridSpec)
    assert isinstance(cs.target_product_grid, GridSpec)
    assert cs.urban_grid is None
    assert cs.coarse_grid.width_m == pytest.approx(256_000.0)
    assert cs.target_product_grid.width_m == pytest.approx(256.0)


def test_coverage_spec_from_config_with_urban():
    cs = CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=25.6, coarse_nx=256, coarse_ny=256,
        product_coverage_km=0.256, product_nx=256, product_ny=256,
        urban_coverage_km=1.0, urban_nx=256, urban_ny=256,
    )
    assert cs.urban_grid is not None
    assert cs.urban_grid.width_m == pytest.approx(1000.0)


def test_coverage_spec_from_config_default_blend():
    cs = CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
        product_coverage_km=0.256, product_nx=256, product_ny=256,
    )
    assert cs.blend == BlendPolicy.default()


def test_coverage_spec_from_config_custom_blend():
    bp = BlendPolicy(alignment_rule="same_bbox", crop_policy="bbox_project",
                     blend_policy="additive")
    cs = CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
        product_coverage_km=0.256, product_nx=256, product_ny=256,
        blend=bp,
    )
    assert cs.blend.alignment_rule == "same_bbox"


def test_coverage_spec_frozen():
    cs = CoverageSpec.from_config(
        origin_lat=34.0, origin_lon=108.0,
        coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
        product_coverage_km=0.256, product_nx=256, product_ny=256,
    )
    with pytest.raises(Exception):
        cs.coarse_grid = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# target_product_grid.width_m replaces hardcoded 0.256 km
# ---------------------------------------------------------------------------

def test_target_product_grid_width_matches_config():
    for product_km in [0.256, 1.0, 5.0, 25.6]:
        cs = CoverageSpec.from_config(
            origin_lat=34.0, origin_lon=108.0,
            coarse_coverage_km=256.0, coarse_nx=256, coarse_ny=256,
            product_coverage_km=product_km, product_nx=256, product_ny=256,
        )
        assert cs.target_product_grid.width_m == pytest.approx(product_km * 1000.0)
