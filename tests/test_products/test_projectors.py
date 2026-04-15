"""
Tests for product projectors and dataset export (task24 / AC-7).
"""

import json
import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from src.context.frame_context import FrameContext
from src.context.frame_builder import FrameBuilder
from src.context.grid_spec import GridSpec
from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from src.context.multiscale_map import MultiScaleMap
from src.products.manifest import ProductManifest
from src.products.projectors import (
    UnknownProductTypeError,
    project,
    export_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRID = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
TS_UTC = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
FRAME_ID = "20250601T120000Z_25544"
N = 256


@pytest.fixture
def frame():
    builder = FrameBuilder(grid=GRID)
    return builder.build(TS_UTC, frame_id=FRAME_ID)


@pytest.fixture
def entry(frame):
    ones = np.ones((N, N), dtype=np.float32)
    return EntryWaveState(
        frame_id=frame.frame_id, grid=GRID,
        total_loss_db=ones * 153.5,
        fspl_db=ones * 180.0, atm_db=ones * 2.0,
        iono_db=ones * 1.0, pol_db=ones * 0.5, gain_db=ones * 30.0,
        elevation_deg=ones * 45.0, azimuth_deg=ones * 180.0,
        slant_range_m=ones * 600_000.0,
        occlusion_mask=np.zeros((N, N), dtype=bool),
    )


@pytest.fixture
def terrain(frame):
    loss = np.zeros((N, N), dtype=np.float32)
    loss[:64, :64] = 20.0
    occ = loss >= 60.0
    return TerrainState(frame_id=frame.frame_id, grid=GRID, loss_db=loss, occlusion_mask=occ)


@pytest.fixture
def urban(frame):
    res = np.zeros((N, N), dtype=np.float32)
    res[100:150, 100:150] = 15.0
    support = np.zeros((N, N), dtype=bool)
    support[100:150, 100:150] = True
    nlos = res > 0
    return UrbanRefinementState(
        frame_id=frame.frame_id, grid=GRID, urban_grid=GRID,
        urban_residual_db=res, support_mask=support, nlos_mask=nlos,
    )


@pytest.fixture
def multiscale(frame, entry, terrain, urban):
    return MultiScaleMap.compose(frame.frame_id, GRID, entry, terrain, urban)


@pytest.fixture
def manifest():
    return ProductManifest.build(
        frame_id=FRAME_ID, timestamp_utc=TS_UTC.isoformat(),
        config={"frequency_ghz": 2.0}, data_snapshot_id="snap",
    )


# ---------------------------------------------------------------------------
# project(): individual projectors
# ---------------------------------------------------------------------------

class TestProject:
    def test_path_loss_map_from_multiscale(self, frame, entry, terrain, urban, multiscale):
        arr = project("path_loss_map", frame, entry=entry, terrain=terrain,
                      urban=urban, multiscale=multiscale)
        assert arr.shape == (N, N)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, multiscale.composite_db, atol=1e-5)

    def test_path_loss_map_fallback_compose(self, frame, entry, terrain, urban):
        """path_loss_map without multiscale falls back to MultiScaleMap.compose."""
        arr = project("path_loss_map", frame, entry=entry, terrain=terrain, urban=urban)
        assert arr.shape == (N, N)

    def test_visibility_mask(self, frame, entry):
        arr = project("visibility_mask", frame, entry=entry)
        assert arr.dtype == np.bool_
        assert arr.shape == (N, N)
        # occlusion_mask is all False -> visibility is all True
        assert arr.all()

    def test_visibility_mask_requires_entry(self, frame):
        with pytest.raises(ValueError, match="entry"):
            project("visibility_mask", frame)

    def test_elevation_field(self, frame, entry):
        arr = project("elevation_field", frame, entry=entry)
        assert arr.shape == (N, N)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, 45.0, atol=1e-4)

    def test_elevation_field_requires_entry(self, frame):
        with pytest.raises(ValueError, match="entry"):
            project("elevation_field", frame)

    def test_azimuth_field(self, frame, entry):
        arr = project("azimuth_field", frame, entry=entry)
        assert arr.shape == (N, N)
        np.testing.assert_allclose(arr, 180.0, atol=1e-4)

    def test_azimuth_field_requires_entry(self, frame):
        with pytest.raises(ValueError, match="entry"):
            project("azimuth_field", frame)

    def test_terrain_blockage(self, frame, terrain):
        arr = project("terrain_blockage", frame, terrain=terrain)
        assert arr.dtype == np.bool_
        assert arr.shape == (N, N)
        # loss[:64,:64]=20 < 60 -> no occlusion
        assert not arr.any()

    def test_terrain_blockage_requires_terrain(self, frame):
        with pytest.raises(ValueError, match="terrain"):
            project("terrain_blockage", frame)

    def test_urban_residual(self, frame, urban):
        arr = project("urban_residual", frame, urban=urban)
        assert arr.shape == (N, N)
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr[100:150, 100:150], 15.0, atol=1e-4)
        np.testing.assert_allclose(arr[0, 0], 0.0, atol=1e-4)

    def test_urban_residual_requires_urban(self, frame):
        with pytest.raises(ValueError, match="urban"):
            project("urban_residual", frame)

    def test_unknown_product_type_raises(self, frame):
        with pytest.raises(UnknownProductTypeError, match="unknown_type"):
            project("unknown_type", frame)

    def test_unknown_product_type_message_lists_valid_types(self, frame):
        with pytest.raises(UnknownProductTypeError) as exc_info:
            project("bad_product", frame)
        assert "path_loss_map" in str(exc_info.value)


# ---------------------------------------------------------------------------
# export_dataset(): file output
# ---------------------------------------------------------------------------

class TestExportDataset:
    def test_writes_npy_files(self, tmp_path, frame, entry, terrain, urban):
        written = export_dataset(
            tmp_path, frame,
            product_types=["path_loss_map", "visibility_mask"],
            entry=entry, terrain=terrain, urban=urban,
        )
        assert "path_loss_map" in written
        assert "visibility_mask" in written
        assert Path(written["path_loss_map"]).exists()
        assert Path(written["visibility_mask"]).exists()

    def test_npy_files_loadable(self, tmp_path, frame, entry):
        export_dataset(tmp_path, frame, ["elevation_field"], entry=entry)
        arr = np.load(tmp_path / "elevation_field.npy")
        assert arr.shape == (N, N)

    def test_writes_json_sidecar(self, tmp_path, frame, entry):
        export_dataset(tmp_path, frame, ["elevation_field"], entry=entry)
        sidecar_path = tmp_path / "dataset.json"
        assert sidecar_path.exists()
        data = json.loads(sidecar_path.read_text())
        assert data["frame_id"] == frame.frame_id
        assert "elevation_field" in data["products"]

    def test_sidecar_contains_shape_and_dtype(self, tmp_path, frame, entry):
        export_dataset(tmp_path, frame, ["elevation_field"], entry=entry)
        data = json.loads((tmp_path / "dataset.json").read_text())
        prod = data["products"]["elevation_field"]
        assert prod["shape"] == [N, N]
        assert "float32" in prod["dtype"]

    def test_manifest_embedded_in_sidecar(self, tmp_path, frame, entry, manifest):
        export_dataset(tmp_path, frame, ["elevation_field"], entry=entry, manifest=manifest)
        data = json.loads((tmp_path / "dataset.json").read_text())
        assert "manifest" in data
        assert data["manifest"]["frame_id"] == manifest.frame_id

    def test_prefix_applied_to_filenames(self, tmp_path, frame, entry):
        export_dataset(tmp_path, frame, ["elevation_field"], entry=entry, prefix="run01_")
        assert (tmp_path / "run01_elevation_field.npy").exists()
        assert (tmp_path / "run01_dataset.json").exists()

    def test_creates_output_dir(self, tmp_path, frame, entry):
        nested = tmp_path / "deep" / "nested"
        export_dataset(nested, frame, ["elevation_field"], entry=entry)
        assert nested.exists()

    def test_unknown_product_type_raises(self, tmp_path, frame):
        with pytest.raises(UnknownProductTypeError):
            export_dataset(tmp_path, frame, ["bad_product"])

    def test_all_product_types(self, tmp_path, frame, entry, terrain, urban, multiscale):
        all_types = [
            "path_loss_map", "visibility_mask", "elevation_field",
            "azimuth_field", "terrain_blockage", "urban_residual",
        ]
        written = export_dataset(
            tmp_path, frame, all_types,
            entry=entry, terrain=terrain, urban=urban, multiscale=multiscale,
        )
        assert len(written) == len(all_types)
        for pt in all_types:
            assert Path(written[pt]).exists()
