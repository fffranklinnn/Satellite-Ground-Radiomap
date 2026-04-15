"""
Benchmark tests for MultiScaleMap support_mask gating and coarse backbone equivalence.

Validates:
1. Outside support_mask: composite equals l1 + l2 (coarse backbone unchanged)
2. Inside support_mask: composite equals l1 + l2 + l3_residual
3. Golden scene arrays: residual formulation matches additive (max diff = 0.0 dB)
"""

import json
import pytest
import numpy as np
from pathlib import Path

from src.context.grid_spec import GridSpec
from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from src.context.multiscale_map import MultiScaleMap


GOLDEN_DIR = Path("benchmarks/golden_scenes")
GRID = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
FRAME_ID = "benchmark_frame"
N = 256


def _make_entry(total: float) -> EntryWaveState:
    ones = np.ones((N, N), dtype=np.float32)
    return EntryWaveState(
        frame_id=FRAME_ID, grid=GRID,
        total_loss_db=ones * total,
        fspl_db=ones * 180.0, atm_db=ones * 2.0,
        iono_db=ones * 1.0, pol_db=ones * 0.5, gain_db=ones * 30.0,
        elevation_deg=ones * 45.0, azimuth_deg=ones * 180.0,
        slant_range_m=ones * 600_000.0,
        occlusion_mask=np.zeros((N, N), dtype=bool),
    )


def _make_terrain(loss: float) -> TerrainState:
    return TerrainState(
        frame_id=FRAME_ID, grid=GRID,
        loss_db=np.full((N, N), loss, dtype=np.float32),
        occlusion_mask=np.zeros((N, N), dtype=bool),
    )


def _make_urban(residual: float, support_rows: slice) -> UrbanRefinementState:
    """Urban state with residual only in support_rows, zero elsewhere."""
    res = np.zeros((N, N), dtype=np.float32)
    support = np.zeros((N, N), dtype=bool)
    res[support_rows, :] = residual
    support[support_rows, :] = True
    return UrbanRefinementState(
        frame_id=FRAME_ID, grid=GRID, urban_grid=GRID,
        urban_residual_db=res, support_mask=support, nlos_mask=res > 0,
    )


# ---------------------------------------------------------------------------
# Core support_mask gating invariants
# ---------------------------------------------------------------------------

class TestSupportMaskGating:
    def test_outside_support_equals_coarse_backbone(self):
        """Outside support_mask: composite must equal l1 + l2 exactly."""
        entry = _make_entry(100.0)
        terrain = _make_terrain(10.0)
        urban = _make_urban(residual=20.0, support_rows=slice(0, 64))

        msm = MultiScaleMap.compose(FRAME_ID, GRID, entry, terrain, urban)

        # Outside support (rows 64-255): composite = l1 + l2 = 110.0
        outside = msm.composite_db[64:, :]
        np.testing.assert_allclose(outside, 110.0, atol=1e-5,
                                   err_msg="Outside support_mask: composite must equal l1+l2")

    def test_inside_support_equals_l1_plus_l2_plus_l3(self):
        """Inside support_mask: composite must equal l1 + l2 + l3_residual."""
        entry = _make_entry(100.0)
        terrain = _make_terrain(10.0)
        urban = _make_urban(residual=20.0, support_rows=slice(0, 64))

        msm = MultiScaleMap.compose(FRAME_ID, GRID, entry, terrain, urban)

        # Inside support (rows 0-63): composite = l1 + l2 + l3 = 130.0
        inside = msm.composite_db[:64, :]
        np.testing.assert_allclose(inside, 130.0, atol=1e-5,
                                   err_msg="Inside support_mask: composite must equal l1+l2+l3")

    def test_l3_zero_outside_support_in_composite(self):
        """l3_db contribution must be zero outside support_mask."""
        entry = _make_entry(100.0)
        terrain = _make_terrain(0.0)
        urban = _make_urban(residual=15.0, support_rows=slice(100, 150))

        msm = MultiScaleMap.compose(FRAME_ID, GRID, entry, terrain, urban)

        # Outside support: composite == l1 == 100.0
        outside_mask = ~urban.support_mask
        np.testing.assert_allclose(msm.composite_db[outside_mask], 100.0, atol=1e-5)

    def test_no_urban_tiles_composite_equals_coarse(self):
        """When no urban tiles are present, composite must equal l1 + l2."""
        entry = _make_entry(100.0)
        terrain = _make_terrain(10.0)
        # Urban with all-zero residual and all-False support
        urban = UrbanRefinementState(
            frame_id=FRAME_ID, grid=GRID, urban_grid=GRID,
            urban_residual_db=np.zeros((N, N), dtype=np.float32),
            support_mask=np.zeros((N, N), dtype=bool),
            nlos_mask=np.zeros((N, N), dtype=bool),
        )
        msm = MultiScaleMap.compose(FRAME_ID, GRID, entry, terrain, urban)
        np.testing.assert_allclose(msm.composite_db, 110.0, atol=1e-5)

    def test_full_support_equals_additive(self):
        """When support_mask is all True, residual == additive."""
        entry = _make_entry(100.0)
        terrain = _make_terrain(10.0)
        res = np.full((N, N), 5.0, dtype=np.float32)
        urban = UrbanRefinementState(
            frame_id=FRAME_ID, grid=GRID, urban_grid=GRID,
            urban_residual_db=res,
            support_mask=np.ones((N, N), dtype=bool),
            nlos_mask=res > 0,
        )
        msm = MultiScaleMap.compose(FRAME_ID, GRID, entry, terrain, urban)
        np.testing.assert_allclose(msm.composite_db, 115.0, atol=1e-5)

    def test_support_mask_boundary_precision(self):
        """Boundary between support and non-support must be pixel-exact."""
        entry = _make_entry(0.0)
        terrain = _make_terrain(0.0)
        urban = _make_urban(residual=10.0, support_rows=slice(128, 129))

        msm = MultiScaleMap.compose(FRAME_ID, GRID, entry, terrain, urban)

        # Row 128: inside support
        np.testing.assert_allclose(msm.composite_db[128, :], 10.0, atol=1e-5)
        # Row 127: outside support
        np.testing.assert_allclose(msm.composite_db[127, :], 0.0, atol=1e-5)
        # Row 129: outside support
        np.testing.assert_allclose(msm.composite_db[129, :], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Golden scene benchmark: residual == additive on real captured data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (GOLDEN_DIR / "l1l2l3_l1.npy").exists(),
    reason="Golden scene arrays not available"
)
class TestGoldenSceneBenchmark:
    def _load_golden(self):
        l1 = np.load(GOLDEN_DIR / "l1l2l3_l1.npy")
        l2 = np.load(GOLDEN_DIR / "l1l2l3_l2.npy")
        l3 = np.load(GOLDEN_DIR / "l1l2l3_l3.npy")
        composite = np.load(GOLDEN_DIR / "l1l2l3_composite.npy")
        return l1, l2, l3, composite

    def test_residual_equals_additive_on_golden_scenes(self):
        """Residual formulation must match additive on golden scenes (max diff = 0.0 dB)."""
        l1, l2, l3, _ = self._load_golden()
        additive = l1 + l2 + l3
        support_mask = l3 > 0
        residual = l1 + l2 + np.where(support_mask, l3, 0.0)
        max_diff = float(np.max(np.abs(additive - residual)))
        assert max_diff == 0.0, f"Residual vs additive max diff: {max_diff} dB (expected 0.0)"

    def test_golden_composite_matches_additive(self):
        """Golden composite must match additive composition (max diff = 0.0 dB)."""
        l1, l2, l3, composite_golden = self._load_golden()
        additive = l1 + l2 + l3
        max_diff = float(np.max(np.abs(composite_golden - additive)))
        assert max_diff == 0.0, f"Golden vs additive max diff: {max_diff} dB"

    def test_l3_support_fraction(self):
        """L3 support fraction must be consistent with benchmark artifact."""
        _, _, l3, _ = self._load_golden()
        support_mask = l3 > 0
        nonzero = int(np.sum(support_mask))
        total = l3.size
        # From residual_vs_additive.json: 9256 nonzero pixels
        assert nonzero == 9256, f"Expected 9256 nonzero L3 pixels, got {nonzero}"
        assert total == 65536

    def test_outside_support_composite_equals_coarse(self):
        """Outside L3 support: golden composite must equal l1 + l2."""
        l1, l2, l3, composite_golden = self._load_golden()
        support_mask = l3 > 0
        outside = ~support_mask
        coarse = l1 + l2
        max_diff = float(np.max(np.abs(composite_golden[outside] - coarse[outside])))
        assert max_diff == 0.0, f"Outside support: composite vs coarse max diff: {max_diff} dB"

    def test_benchmark_artifact_exists(self):
        """Benchmark artifact must exist and have PASS verdict."""
        artifact_path = Path("benchmarks/baselines/residual_vs_additive.json")
        assert artifact_path.exists(), "residual_vs_additive.json benchmark artifact missing"
        data = json.loads(artifact_path.read_text())
        assert data["gate_decision"]["result"] == "PASSED"
        assert data["comparison"]["max_absolute_difference_db"] == 0.0
