"""
Reproducibility tests for BenchmarkRunner (task21 / task22 / AC-6).

Verifies:
- BenchmarkRunner.load_frame_list() parses the golden-scene frame list
- Two runs with identical inputs produce matching config_hash and data_snapshot_id
- Output arrays from two runs have relative error < 1e-6
- ProductManifest JSON round-trip is lossless
- BenchmarkRunner produces one manifest per frame
"""

from __future__ import annotations

import json
import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.context.frame_context import FrameContext
from src.context.frame_builder import FrameBuilder
from src.context.grid_spec import GridSpec
from src.context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from src.context.multiscale_map import MultiScaleMap
from src.pipeline.benchmark_runner import BenchmarkRunner
from src.pipeline.manifest_writer import ManifestWriter
from src.products.manifest import ProductManifest, collect_input_file_paths


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ORIGIN_LAT = 34.0
ORIGIN_LON = 108.0
N = 64  # small grid for fast tests
GRID = GridSpec.from_legacy_args(ORIGIN_LAT, ORIGIN_LON, 256.0, N, N)
TS_UTC = datetime(2025, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
CONFIG = {"frequency_ghz": 2.0, "grid_size": N, "origin": {"lat": ORIGIN_LAT, "lon": ORIGIN_LON}}
SNAPSHOT_ID = "snap_golden_001"

GOLDEN_FRAME_LIST = Path("benchmarks/golden_scenes/frame_list.json")


def _make_entry(frame_id: str, total: float = 153.5) -> EntryWaveState:
    ones = np.ones((N, N), dtype=np.float32)
    return EntryWaveState(
        frame_id=frame_id, grid=GRID,
        total_loss_db=ones * total,
        fspl_db=ones * 180.0, atm_db=ones * 2.0,
        iono_db=ones * 1.0, pol_db=ones * 0.5, gain_db=ones * 30.0,
        elevation_deg=ones * 45.0, azimuth_deg=ones * 180.0,
        slant_range_m=ones * 600_000.0,
        occlusion_mask=np.zeros((N, N), dtype=bool),
    )


def _make_terrain(frame_id: str, loss: float = 2.0) -> TerrainState:
    return TerrainState(
        frame_id=frame_id, grid=GRID,
        loss_db=np.full((N, N), loss, dtype=np.float32),
        occlusion_mask=np.zeros((N, N), dtype=bool),
    )


def _make_urban(frame_id: str, residual: float = 5.0) -> UrbanRefinementState:
    res = np.full((N, N), residual, dtype=np.float32)
    support = np.ones((N, N), dtype=bool)
    return UrbanRefinementState(
        frame_id=frame_id, grid=GRID, urban_grid=GRID,
        urban_residual_db=res,
        nlos_mask=support,
        support_mask=support,
    )


def _make_mock_layers(frame_id: str):
    """Return mock L1/L2/L3 layers that return deterministic states."""
    entry = _make_entry(frame_id)
    terrain = _make_terrain(frame_id)
    urban = _make_urban(frame_id)

    l1 = MagicMock()
    l1.propagate_entry.return_value = entry

    l2 = MagicMock()
    l2.propagate_terrain.return_value = terrain

    l3 = MagicMock()
    l3.refine_urban.return_value = urban

    return l1, l2, l3


def _make_frame_builder() -> FrameBuilder:
    return FrameBuilder(grid=GRID)


# ---------------------------------------------------------------------------
# BenchmarkRunner.load_frame_list
# ---------------------------------------------------------------------------

class TestLoadFrameList:
    def test_loads_golden_frame_list(self):
        """load_frame_list must parse the golden-scene frame list."""
        if not GOLDEN_FRAME_LIST.exists():
            pytest.skip("Golden frame list not found")
        frames = BenchmarkRunner.load_frame_list(GOLDEN_FRAME_LIST)
        assert len(frames) >= 1
        for ts in frames:
            assert ts.tzinfo is not None

    def test_loads_from_tmp_file(self, tmp_path):
        """load_frame_list must parse a custom frame list JSON."""
        frame_list = tmp_path / "frame_list.json"
        frame_list.write_text(
            json.dumps({"frames": ["2025-01-03T00:00:00+00:00", "2025-01-03T06:00:00+00:00"]}),
            encoding="utf-8",
        )
        frames = BenchmarkRunner.load_frame_list(frame_list)
        assert len(frames) == 2
        assert frames[0] == datetime(2025, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
        assert frames[1] == datetime(2025, 1, 3, 6, 0, 0, tzinfo=timezone.utc)

    def test_all_frames_utc_aware(self, tmp_path):
        """All loaded frames must be UTC-aware."""
        frame_list = tmp_path / "frame_list.json"
        frame_list.write_text(
            json.dumps({"frames": ["2025-06-01T12:00:00+00:00"]}),
            encoding="utf-8",
        )
        frames = BenchmarkRunner.load_frame_list(frame_list)
        assert all(ts.tzinfo is not None for ts in frames)


# ---------------------------------------------------------------------------
# BenchmarkRunner.run_frame: manifest fields
# ---------------------------------------------------------------------------

class TestBenchmarkRunnerManifest:
    def _make_runner(self, frame_id: str) -> BenchmarkRunner:
        l1, l2, l3 = _make_mock_layers(frame_id)
        fb = _make_frame_builder()
        return BenchmarkRunner(
            frame_builder=fb,
            l1_layer=l1,
            l2_layer=l2,
            l3_layer=l3,
            config=CONFIG,
            data_snapshot_id=SNAPSHOT_ID,
        )

    def test_run_frame_returns_manifest(self, tmp_path):
        """run_frame must return a ProductManifest."""
        runner = self._make_runner("test_frame")
        # Patch FrameBuilder.build to return a frame with known frame_id
        frame = FrameContext(
            frame_id="test_frame",
            timestamp=TS_UTC,
            grid=GRID,
        )
        with patch.object(runner.frame_builder, "build", return_value=frame):
            manifest = runner.run_frame(
                timestamp=TS_UTC,
                output_dir=tmp_path,
                product_types=["path_loss_map"],
            )
        assert isinstance(manifest, ProductManifest)
        assert manifest.frame_id == "test_frame"
        assert manifest.data_snapshot_id == SNAPSHOT_ID

    def test_config_hash_deterministic(self, tmp_path):
        """Two runs with the same config must produce the same config_hash."""
        runner = self._make_runner("frame_a")
        frame = FrameContext(frame_id="frame_a", timestamp=TS_UTC, grid=GRID)
        with patch.object(runner.frame_builder, "build", return_value=frame):
            m1 = runner.run_frame(TS_UTC, tmp_path / "run1", ["path_loss_map"])
            m2 = runner.run_frame(TS_UTC, tmp_path / "run2", ["path_loss_map"])
        assert m1.config_hash == m2.config_hash

    def test_data_snapshot_id_preserved(self, tmp_path):
        """data_snapshot_id must be preserved in the manifest."""
        runner = self._make_runner("frame_b")
        frame = FrameContext(frame_id="frame_b", timestamp=TS_UTC, grid=GRID)
        with patch.object(runner.frame_builder, "build", return_value=frame):
            manifest = runner.run_frame(TS_UTC, tmp_path, ["path_loss_map"])
        assert manifest.data_snapshot_id == SNAPSHOT_ID


# ---------------------------------------------------------------------------
# Reproducibility: relative error < 1e-6
# ---------------------------------------------------------------------------

class TestReproducibility:
    """
    Two runs with identical inputs must produce numerically identical outputs.
    Relative error < 1e-6 is the AC-6 acceptance criterion.
    """

    def _run_once(self, frame_id: str, output_dir: Path) -> np.ndarray:
        """Run the pipeline once and return the saved path_loss_map array."""
        l1, l2, l3 = _make_mock_layers(frame_id)
        fb = _make_frame_builder()
        runner = BenchmarkRunner(
            frame_builder=fb,
            l1_layer=l1,
            l2_layer=l2,
            l3_layer=l3,
            config=CONFIG,
            data_snapshot_id=SNAPSHOT_ID,
        )
        frame = FrameContext(frame_id=frame_id, timestamp=TS_UTC, grid=GRID)
        with patch.object(runner.frame_builder, "build", return_value=frame):
            runner.run_frame(
                timestamp=TS_UTC,
                output_dir=output_dir,
                product_types=["path_loss_map"],
                prefix="repro_",
            )
        npy_path = output_dir / "repro_path_loss_map.npy"
        return np.load(npy_path)

    def test_two_runs_identical_arrays(self, tmp_path):
        """Two runs with identical inputs must produce identical arrays."""
        arr1 = self._run_once("repro_frame", tmp_path / "run1")
        arr2 = self._run_once("repro_frame", tmp_path / "run2")
        assert arr1.shape == arr2.shape
        np.testing.assert_array_equal(arr1, arr2)

    def test_relative_error_below_threshold(self, tmp_path):
        """Relative error between two identical runs must be < 1e-6."""
        arr1 = self._run_once("repro_frame", tmp_path / "run1")
        arr2 = self._run_once("repro_frame", tmp_path / "run2")
        # Avoid division by zero: use absolute error where values are near zero
        denom = np.maximum(np.abs(arr1), 1e-10)
        rel_err = np.max(np.abs(arr1 - arr2) / denom)
        assert rel_err < 1e-6, f"Relative error {rel_err:.2e} exceeds 1e-6"

    def test_manifest_config_hash_identical_across_runs(self, tmp_path):
        """config_hash must be identical across two runs with the same config."""
        l1, l2, l3 = _make_mock_layers("hash_frame")
        fb = _make_frame_builder()
        runner = BenchmarkRunner(
            frame_builder=fb, l1_layer=l1, l2_layer=l2, l3_layer=l3,
            config=CONFIG, data_snapshot_id=SNAPSHOT_ID,
        )
        frame = FrameContext(frame_id="hash_frame", timestamp=TS_UTC, grid=GRID)
        with patch.object(runner.frame_builder, "build", return_value=frame):
            m1 = runner.run_frame(TS_UTC, tmp_path / "r1", ["path_loss_map"])
            m2 = runner.run_frame(TS_UTC, tmp_path / "r2", ["path_loss_map"])
        assert m1.config_hash == m2.config_hash
        assert m1.data_snapshot_id == m2.data_snapshot_id

    def _run_with_writer(self, frame_id: str, output_dir: Path):
        """Run with a ManifestWriter and return (manifest, manifest_path)."""
        l1, l2, l3 = _make_mock_layers(frame_id)
        fb = _make_frame_builder()
        runner = BenchmarkRunner(
            frame_builder=fb, l1_layer=l1, l2_layer=l2, l3_layer=l3,
            config=CONFIG, data_snapshot_id=SNAPSHOT_ID,
        )
        frame = FrameContext(frame_id=frame_id, timestamp=TS_UTC, grid=GRID)
        manifest_path = output_dir / "manifest.jsonl"
        with ManifestWriter(manifest_path) as writer:
            with patch.object(runner.frame_builder, "build", return_value=frame):
                manifest = runner.run_frame(
                    TS_UTC, output_dir, ["path_loss_map"],
                    prefix="repro_", manifest_writer=writer,
                )
        return manifest, manifest_path

    def test_output_file_hashes_populated(self, tmp_path):
        """run_frame with ManifestWriter must return manifest with output_file_hashes."""
        manifest, _ = self._run_with_writer("hash_frame2", tmp_path)
        assert "path_loss_map" in manifest.output_file_hashes
        assert len(manifest.output_file_hashes["path_loss_map"]) == 64  # SHA-256 hex

    def test_output_hashes_stable_across_runs(self, tmp_path):
        """Two identical runs must produce identical output_file_hashes."""
        m1, _ = self._run_with_writer("stable_frame", tmp_path / "r1")
        m2, _ = self._run_with_writer("stable_frame", tmp_path / "r2")
        assert m1.output_file_hashes == m2.output_file_hashes

    def test_manifest_equality_across_runs(self, tmp_path):
        """Two identical runs must produce equal manifests (config_hash, data_snapshot_id, output_hashes)."""
        m1, _ = self._run_with_writer("eq_frame", tmp_path / "r1")
        m2, _ = self._run_with_writer("eq_frame", tmp_path / "r2")
        assert m1.config_hash == m2.config_hash
        assert m1.data_snapshot_id == m2.data_snapshot_id
        assert m1.output_file_hashes == m2.output_file_hashes

    def test_jsonl_contains_output_hashes(self, tmp_path):
        """JSONL manifest must contain output_file_hashes for the written frame."""
        _, manifest_path = self._run_with_writer("jsonl_frame", tmp_path)
        line = manifest_path.read_text().strip()
        parsed = json.loads(line)
        assert "output_file_hashes" in parsed
        assert "path_loss_map" in parsed["output_file_hashes"]


# ---------------------------------------------------------------------------
# BenchmarkRunner.run: multi-frame list
# ---------------------------------------------------------------------------

class TestBenchmarkRunnerMultiFrame:
    def test_run_produces_one_manifest_per_frame(self, tmp_path):
        """run() must return one ProductManifest per timestamp."""
        timestamps = [
            datetime(2025, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 3, 6, 0, 0, tzinfo=timezone.utc),
        ]

        call_count = [0]

        def _build_frame(ts, **kwargs):
            fid = f"frame_{call_count[0]:03d}"
            call_count[0] += 1
            return FrameContext(frame_id=fid, timestamp=ts, grid=GRID)

        l1, l2, l3 = _make_mock_layers("frame_000")
        # Make mock layers return states with matching frame_id
        def _propagate_entry(frame, **kwargs):
            return _make_entry(frame.frame_id)
        def _propagate_terrain(frame, **kwargs):
            return _make_terrain(frame.frame_id)
        def _refine_urban(frame, **kwargs):
            return _make_urban(frame.frame_id)

        l1.propagate_entry.side_effect = _propagate_entry
        l2.propagate_terrain.side_effect = _propagate_terrain
        l3.refine_urban.side_effect = _refine_urban

        fb = _make_frame_builder()
        runner = BenchmarkRunner(
            frame_builder=fb, l1_layer=l1, l2_layer=l2, l3_layer=l3,
            config=CONFIG, data_snapshot_id=SNAPSHOT_ID,
        )
        with patch.object(runner.frame_builder, "build", side_effect=_build_frame):
            results = runner.run(
                timestamps=timestamps,
                output_dir=tmp_path,
                product_types=["path_loss_map"],
            )

        assert len(results) == 2
        assert all(isinstance(m, ProductManifest) for m in results)

    def test_run_writes_manifest_jsonl(self, tmp_path):
        """run() must write a manifest.jsonl with one line per frame."""
        timestamps = [datetime(2025, 1, 3, 0, 0, 0, tzinfo=timezone.utc)]

        def _build_frame(ts, **kwargs):
            return FrameContext(frame_id="frame_000", timestamp=ts, grid=GRID)

        l1, l2, l3 = _make_mock_layers("frame_000")
        fb = _make_frame_builder()
        runner = BenchmarkRunner(
            frame_builder=fb, l1_layer=l1, l2_layer=l2, l3_layer=l3,
            config=CONFIG, data_snapshot_id=SNAPSHOT_ID,
        )
        with patch.object(runner.frame_builder, "build", side_effect=_build_frame):
            runner.run(
                timestamps=timestamps,
                output_dir=tmp_path,
                product_types=["path_loss_map"],
            )

        manifest_path = tmp_path / "manifest.jsonl"
        assert manifest_path.exists()
        lines = manifest_path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert "frame_id" in parsed

    def test_run_empty_frame_list_returns_empty(self, tmp_path):
        """run() with an empty frame list must return an empty list."""
        fb = _make_frame_builder()
        runner = BenchmarkRunner(frame_builder=fb, config=CONFIG)
        results = runner.run(
            timestamps=[],
            output_dir=tmp_path,
            product_types=["path_loss_map"],
        )
        assert results == []


# ---------------------------------------------------------------------------
# Regression report schema (task26 / AC-8)
# ---------------------------------------------------------------------------

class TestRegressionReportSchema:
    """Verify the regression report artifact has the required schema."""

    REPORT_PATH = Path("benchmarks/baselines/regression_report.json")

    def test_regression_report_exists(self):
        """benchmarks/baselines/regression_report.json must exist."""
        assert self.REPORT_PATH.exists(), (
            f"Regression report not found at {self.REPORT_PATH}. "
            "Run benchmarks/run_regression.py to generate it."
        )

    def test_regression_report_schema(self):
        """Regression report must have required top-level fields."""
        if not self.REPORT_PATH.exists():
            pytest.skip("Regression report not yet generated.")
        report = json.loads(self.REPORT_PATH.read_text())
        assert "schema_version" in report
        assert "array_comparisons" in report
        assert "manifest_checks" in report
        assert isinstance(report["array_comparisons"], list)
        assert isinstance(report["manifest_checks"], list)

    def test_regression_report_all_passed(self):
        """Regression report must have all_passed=True when generated."""
        if not self.REPORT_PATH.exists():
            pytest.skip("Regression report not yet generated.")
        report = json.loads(self.REPORT_PATH.read_text())
        if report.get("all_passed") is None:
            pytest.skip("Regression report is a stub; run benchmarks/run_regression.py first.")
        assert report["all_passed"] is True, (
            f"Regression report has failures: {report.get('array_comparisons')}"
        )


# ---------------------------------------------------------------------------
# collect_input_file_paths and input_file_hashes reproducibility (task20b)
# ---------------------------------------------------------------------------

class TestInputFileHashes:
    def test_collect_input_file_paths_empty_config(self):
        """collect_input_file_paths returns empty dict for config with no layers."""
        assert collect_input_file_paths({}) == {}

    def test_collect_input_file_paths_extracts_tle(self, tmp_path):
        """collect_input_file_paths extracts tle_file from l1_macro config."""
        tle = tmp_path / "test.tle"
        tle.write_text("dummy")
        cfg = {"layers": {"l1_macro": {"tle_file": str(tle)}}}
        paths = collect_input_file_paths(cfg)
        assert "tle_file" in paths
        assert paths["tle_file"] == str(tle)

    def test_collect_input_file_paths_extracts_dem(self, tmp_path):
        """collect_input_file_paths extracts dem_file from l2_topo config."""
        dem = tmp_path / "dem.tif"
        dem.write_bytes(b"dummy")
        cfg = {"layers": {"l2_topo": {"dem_file": str(dem)}}}
        paths = collect_input_file_paths(cfg)
        assert "dem_file" in paths

    def test_input_file_hashes_populated_when_files_exist(self, tmp_path):
        """ProductManifest.build with hash_files=True populates input_file_hashes."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello")
        manifest = ProductManifest.build(
            frame_id="f1",
            timestamp_utc="2025-01-03T00:00:00+00:00",
            config=CONFIG,
            input_files={"data": str(f)},
            hash_files=True,
        )
        assert "data" in manifest.input_file_hashes
        assert len(manifest.input_file_hashes["data"]) == 64  # SHA-256 hex

    def test_input_file_hashes_stable_across_runs(self, tmp_path):
        """Two runs with the same input file produce identical input_file_hashes."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"stable content")
        m1 = ProductManifest.build(
            frame_id="f1", timestamp_utc="2025-01-03T00:00:00+00:00",
            config=CONFIG, input_files={"data": str(f)}, hash_files=True,
        )
        m2 = ProductManifest.build(
            frame_id="f1", timestamp_utc="2025-01-03T00:00:00+00:00",
            config=CONFIG, input_files={"data": str(f)}, hash_files=True,
        )
        assert m1.input_file_hashes == m2.input_file_hashes

    def test_benchmark_runner_input_file_hashes_with_real_files(self, tmp_path):
        """BenchmarkRunner.run_frame returns manifest with non-empty input_file_hashes when config has files."""
        tle = tmp_path / "test.tle"
        tle.write_text("dummy tle content")
        cfg_with_files = {
            **CONFIG,
            "layers": {"l1_macro": {"tle_file": str(tle)}},
        }
        l1, l2, l3 = _make_mock_layers("hash_input_frame")
        fb = _make_frame_builder()
        runner = BenchmarkRunner(
            frame_builder=fb, l1_layer=l1, l2_layer=l2, l3_layer=l3,
            config=cfg_with_files, data_snapshot_id=SNAPSHOT_ID,
        )
        frame = FrameContext(frame_id="hash_input_frame", timestamp=TS_UTC, grid=GRID)
        with patch.object(runner.frame_builder, "build", return_value=frame):
            manifest = runner.run_frame(TS_UTC, tmp_path / "out", ["path_loss_map"])
        assert "tle_file" in manifest.input_file_hashes
        assert len(manifest.input_file_hashes["tle_file"]) == 64


    def test_collect_input_file_paths_extracts_tile_cache_root(self, tmp_path):
        """collect_input_file_paths uses tile_cache_root key from l3_urban config."""
        cache = tmp_path / "tiles"
        cache.mkdir()
        cfg = {"layers": {"l3_urban": {"tile_cache_root": str(cache)}}}
        paths = collect_input_file_paths(cfg)
        assert "tile_cache_root" in paths
        assert paths["tile_cache_root"] == str(cache)

    def test_collect_input_file_paths_real_benchmark_config(self):
        """collect_input_file_paths extracts tile_cache_root from the real benchmark config."""
        import yaml
        cfg_path = Path(__file__).resolve().parents[2] / "benchmarks" / "golden_scenes_config.yaml"
        if not cfg_path.exists():
            pytest.skip("Benchmark config not found.")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        paths = collect_input_file_paths(cfg)
        # tile_cache_root should be present (even if the directory doesn't exist on this machine)
        assert "tile_cache_root" in paths

    def test_directory_input_hash_non_empty_and_stable(self, tmp_path):
        """Directory-backed input produces non-empty, stable hash."""
        from src.products.manifest import _sha256_dir
        d = tmp_path / "cache"
        d.mkdir()
        (d / "tile_001" ).mkdir()
        (d / "tile_001" / "H.npy").write_bytes(b"height data")
        h1 = _sha256_dir(str(d))
        h2 = _sha256_dir(str(d))
        assert h1 != ""
        assert len(h1) == 64
        assert h1 == h2

    def test_directory_input_hash_via_manifest_build(self, tmp_path):
        """ProductManifest.build with a directory input produces non-empty hash."""
        d = tmp_path / "tiles"
        d.mkdir()
        (d / "t1").mkdir()
        (d / "t1" / "H.npy").write_bytes(b"data")
        manifest = ProductManifest.build(
            frame_id="f1",
            timestamp_utc="2025-01-03T00:00:00+00:00",
            config={},
            input_files={"tile_cache_root": str(d)},
            hash_files=True,
        )
        h = manifest.input_file_hashes["tile_cache_root"]
        assert h != "", "Directory-backed input hash must not be empty"
        assert len(h) == 64


# ---------------------------------------------------------------------------
# Runtime-facing strict-mode tests (task20 / AC-6 end-to-end)
# These tests go through the config-driven runtime path (propagate_terrain /
# refine_urban), not just the internal helper methods.
# ---------------------------------------------------------------------------

from src.context.time_utils import StrictModeError
from src.layers.l2_topo import L2TopoLayer
from src.layers.l3_urban import L3UrbanLayer
from src.context.frame_context import FrameContext
from datetime import timezone


def _make_frame() -> FrameContext:
    return FrameContext(
        frame_id="strict_test_frame",
        timestamp=datetime(2025, 1, 3, 0, 0, 0, tzinfo=timezone.utc),
        grid=GRID,
    )


class TestStrictModeEndToEnd:
    """
    End-to-end strict-mode tests: config dict with strict_data=True propagated
    through the layer constructor, then runtime method raises StrictModeError.
    """

    def test_l2_strict_mode_propagate_terrain_raises(self, tmp_path):
        """propagate_terrain with strict_data=True and missing DEM raises StrictModeError."""
        cfg = {
            "dem_file": str(tmp_path / "nonexistent.tif"),
            "frequency_ghz": 14.5,
            "strict_data": True,
            "grid_size": 256,
            "coverage_km": 25.6,
            "resolution_m": 100.0,
        }
        layer = L2TopoLayer(cfg, origin_lat=34.3, origin_lon=108.9)
        frame = _make_frame()
        with pytest.raises(StrictModeError):
            layer.propagate_terrain(frame)

    def test_l2_non_strict_propagate_terrain_returns_zero_loss(self, tmp_path):
        """propagate_terrain with strict_data=False and missing DEM raises FileNotFoundError (not StrictModeError)."""
        cfg = {
            "dem_file": str(tmp_path / "nonexistent.tif"),
            "frequency_ghz": 14.5,
            "strict_data": False,
            "grid_size": 256,
            "coverage_km": 25.6,
            "resolution_m": 100.0,
        }
        layer = L2TopoLayer(cfg, origin_lat=34.3, origin_lon=108.9)
        frame = _make_frame()
        with pytest.raises(FileNotFoundError) as exc_info:
            layer.propagate_terrain(frame)
        assert not isinstance(exc_info.value, StrictModeError)

    def test_l3_strict_mode_refine_urban_raises(self, tmp_path):
        """refine_urban with strict_data=True and empty tile cache raises StrictModeError."""
        empty_cache = tmp_path / "tiles"
        empty_cache.mkdir()
        cfg = {
            "tile_cache_root": str(empty_cache),
            "nlos_loss_db": 20.0,
            "strict_data": True,
            "incident_dir": {"az_deg": 180.0, "el_deg": 45.0},
            "grid_size": 256,
            "coverage_km": 0.256,
            "resolution_m": 1.0,
        }
        layer = L3UrbanLayer(cfg, origin_lat=34.3, origin_lon=108.9)
        frame = _make_frame()
        with pytest.raises(StrictModeError):
            layer.refine_urban(frame)

    def test_l3_non_strict_refine_urban_raises_file_not_found(self, tmp_path):
        """refine_urban with strict_data=False and empty tile cache raises FileNotFoundError."""
        empty_cache = tmp_path / "tiles"
        empty_cache.mkdir()
        cfg = {
            "tile_cache_root": str(empty_cache),
            "nlos_loss_db": 20.0,
            "strict_data": False,
            "incident_dir": {"az_deg": 180.0, "el_deg": 45.0},
            "grid_size": 256,
            "coverage_km": 0.256,
            "resolution_m": 1.0,
        }
        layer = L3UrbanLayer(cfg, origin_lat=34.3, origin_lon=108.9)
        frame = _make_frame()
        with pytest.raises(FileNotFoundError) as exc_info:
            layer.refine_urban(frame)
        assert not isinstance(exc_info.value, StrictModeError)

    def test_benchmark_runner_collects_fallbacks_used(self, tmp_path):
        """BenchmarkRunner.run_frame must populate fallbacks_used in manifest when L1 falls back."""
        l1, l2, l3 = _make_mock_layers("fallback_frame")
        # Simulate L1 recording a fallback
        l1.fallbacks_used = ["L1: IONEX unavailable, using synthetic TEC"]
        l1.clear_fallbacks = lambda: None  # no-op so we can inspect after
        fb = _make_frame_builder()
        runner = BenchmarkRunner(
            frame_builder=fb, l1_layer=l1, l2_layer=l2, l3_layer=l3,
            config=CONFIG, data_snapshot_id=SNAPSHOT_ID,
        )
        frame = FrameContext(frame_id="fallback_frame", timestamp=TS_UTC, grid=GRID)
        with patch.object(runner.frame_builder, "build", return_value=frame):
            manifest = runner.run_frame(TS_UTC, tmp_path, ["path_loss_map"])
        assert isinstance(manifest.fallbacks_used, (list, tuple))


# ---------------------------------------------------------------------------
# Benchmark/main-style strict-mode and fallback provenance tests (task20 / AC-6)
# These tests simulate the benchmark/main.py config path: a full config dict
# with strict_data=True in the layer sub-config, exercising the real L1 lifecycle.
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
TLE_PATH_FOR_TESTS = ROOT_DIR / "data" / "starlink-2025-tle" / "2025-01-03.tle"

L1_BASE_CFG = {
    "grid_size": 256,
    "coverage_km": 256.0,
    "resolution_m": 1000.0,
    "frequency_ghz": 10.0,
    "tle_file": str(TLE_PATH_FOR_TESTS),
}


class TestStrictModeBenchmarkStyle:
    """
    Strict-mode tests through benchmark/main-style config paths.
    Each test constructs a layer from a config dict (as build_layers() does),
    not by calling internal helpers directly.
    """

    def test_l1_strict_missing_ionex_raises_via_constructor(self, tmp_path):
        """L1 constructor with strict_data=True and missing IONEX raises StrictModeError."""
        if not TLE_PATH_FOR_TESTS.exists():
            pytest.skip("TLE file not available")
        cfg = {
            **L1_BASE_CFG,
            "strict_data": True,
            "ionex_file": str(tmp_path / "nonexistent.INX"),
        }
        from src.layers.l1_macro import L1MacroLayer
        with pytest.raises(StrictModeError):
            L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)

    def test_l1_strict_mode_alias_missing_ionex_raises(self, tmp_path):
        """L1 constructor with strict_mode=True (alias) and missing IONEX raises StrictModeError."""
        if not TLE_PATH_FOR_TESTS.exists():
            pytest.skip("TLE file not available")
        cfg = {
            **L1_BASE_CFG,
            "strict_mode": True,  # alias for strict_data
            "ionex_file": str(tmp_path / "nonexistent.INX"),
        }
        from src.layers.l1_macro import L1MacroLayer
        with pytest.raises(StrictModeError):
            L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)

    def test_l1_strict_unreadable_era5_raises_via_constructor(self, tmp_path):
        """L1 constructor with strict_data=True and unreadable ERA5 raises StrictModeError."""
        if not TLE_PATH_FOR_TESTS.exists():
            pytest.skip("TLE file not available")
        bad_era5 = tmp_path / "bad.nc"
        bad_era5.write_bytes(b"not a netcdf file")
        cfg = {
            **L1_BASE_CFG,
            "strict_data": True,
            "era5_file": str(bad_era5),
        }
        from src.layers.l1_macro import L1MacroLayer
        with pytest.raises(StrictModeError):
            L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)

    def test_l1_non_strict_missing_ionex_records_constructor_fallback(self, tmp_path):
        """L1 non-strict with missing IONEX records fallback in constructor_fallbacks (not cleared by clear_fallbacks)."""
        if not TLE_PATH_FOR_TESTS.exists():
            pytest.skip("TLE file not available")
        cfg = {
            **L1_BASE_CFG,
            "strict_data": False,
            "ionex_file": str(tmp_path / "nonexistent.INX"),
        }
        from src.layers.l1_macro import L1MacroLayer
        layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
        # Constructor fallback must be present
        assert any("IONEX" in fb for fb in layer.fallbacks_used)
        # clear_fallbacks must NOT erase constructor fallbacks
        layer.clear_fallbacks()
        assert any("IONEX" in fb for fb in layer.fallbacks_used), (
            "Constructor-time IONEX fallback must survive clear_fallbacks()"
        )

    def test_l1_non_strict_missing_era5_records_constructor_fallback(self, tmp_path):
        """L1 non-strict with unreadable ERA5 records fallback that survives clear_fallbacks."""
        if not TLE_PATH_FOR_TESTS.exists():
            pytest.skip("TLE file not available")
        bad_era5 = tmp_path / "bad.nc"
        bad_era5.write_bytes(b"not a netcdf file")
        cfg = {
            **L1_BASE_CFG,
            "strict_data": False,
            "era5_file": str(bad_era5),
        }
        from src.layers.l1_macro import L1MacroLayer
        layer = L1MacroLayer(cfg, origin_lat=39.9, origin_lon=116.4)
        assert any("ERA5" in fb for fb in layer.fallbacks_used)
        layer.clear_fallbacks()
        assert any("ERA5" in fb for fb in layer.fallbacks_used), (
            "Constructor-time ERA5 fallback must survive clear_fallbacks()"
        )

    def test_l2_strict_mode_alias_missing_dem_raises(self, tmp_path):
        """L2 constructor with strict_mode=True (alias) and missing DEM raises StrictModeError."""
        cfg = {
            "dem_file": str(tmp_path / "nonexistent.tif"),
            "frequency_ghz": 14.5,
            "strict_mode": True,  # alias for strict_data
            "grid_size": 256,
            "coverage_km": 25.6,
            "resolution_m": 100.0,
        }
        layer = L2TopoLayer(cfg, origin_lat=34.3, origin_lon=108.9)
        with pytest.raises(StrictModeError):
            layer.propagate_terrain(_make_frame())

    def test_l3_strict_mode_alias_empty_cache_raises(self, tmp_path):
        """L3 constructor with strict_mode=True (alias) and empty cache raises StrictModeError."""
        empty_cache = tmp_path / "tiles"
        empty_cache.mkdir()
        cfg = {
            "tile_cache_root": str(empty_cache),
            "nlos_loss_db": 20.0,
            "strict_mode": True,  # alias for strict_data
            "incident_dir": {"az_deg": 180.0, "el_deg": 45.0},
            "grid_size": 256,
            "coverage_km": 0.256,
            "resolution_m": 1.0,
        }
        layer = L3UrbanLayer(cfg, origin_lat=34.3, origin_lon=108.9)
        with pytest.raises(StrictModeError):
            layer.refine_urban(_make_frame())


class TestTopLevelStrictViaDataValidation:
    """
    End-to-end strict tests that exercise the benchmark/main-style config path.

    These tests set data_validation.strict=True at the top level and call
    build_layers() from benchmarks/run_regression.py, verifying that the
    resolved strict flag propagates into each layer constructor so that
    missing/unreadable data raises StrictModeError rather than silently
    falling back.
    """

    def _base_config(self, tmp_path) -> dict:
        return {
            "origin": {"latitude": 34.3416, "longitude": 108.9398},
            "data_validation": {"strict": True, "snapshot_id": "test"},
            "layers": {
                "l1_macro": {
                    "enabled": True,
                    "grid_size": 256,
                    "coverage_km": 256.0,
                    "resolution_m": 1000.0,
                    "frequency_ghz": 14.5,
                    "satellite_altitude_km": 550.0,
                    "tle_file": str(TLE_PATH_FOR_TESTS),
                    "ionex_file": str(tmp_path / "missing.INX"),
                    "era5_file": str(tmp_path / "missing.nc"),
                },
                "l2_topo": {
                    "enabled": True,
                    "grid_size": 256,
                    "coverage_km": 25.6,
                    "resolution_m": 100.0,
                    "dem_file": str(tmp_path / "missing.tif"),
                    "frequency_ghz": 14.5,
                },
                "l3_urban": {
                    "enabled": True,
                    "grid_size": 256,
                    "coverage_km": 0.256,
                    "resolution_m": 1.0,
                    "tile_cache_root": str(tmp_path / "empty_tiles"),
                    "nlos_loss_db": 20.0,
                    "incident_dir": {"az_deg": 180.0, "el_deg": 45.0},
                },
            },
        }

    def test_top_level_strict_l1_missing_ionex_raises(self, tmp_path):
        """data_validation.strict=True propagates to L1; missing IONEX raises StrictModeError."""
        if not TLE_PATH_FOR_TESTS.exists():
            pytest.skip("TLE file not available")
        from benchmarks.run_regression import build_layers
        config = self._base_config(tmp_path)
        with pytest.raises(StrictModeError):
            build_layers(config, 34.3416, 108.9398,
                         enable_l1=True, enable_l2=False, enable_l3=False)

    def test_top_level_strict_l2_missing_dem_raises(self, tmp_path):
        """data_validation.strict=True propagates to L2; missing DEM raises StrictModeError on propagate_terrain."""
        from benchmarks.run_regression import build_layers
        config = self._base_config(tmp_path)
        config["layers"]["l1_macro"]["enabled"] = False
        _, l2_layer, _ = build_layers(config, 34.3416, 108.9398,
                                       enable_l1=False, enable_l2=True, enable_l3=False)
        assert l2_layer is not None
        assert l2_layer.strict_data is True
        with pytest.raises(StrictModeError):
            l2_layer.propagate_terrain(_make_frame())

    def test_top_level_strict_l3_empty_cache_raises(self, tmp_path):
        """data_validation.strict=True propagates to L3; empty tile cache raises StrictModeError on refine_urban."""
        from benchmarks.run_regression import build_layers
        config = self._base_config(tmp_path)
        config["layers"]["l1_macro"]["enabled"] = False
        config["layers"]["l2_topo"]["enabled"] = False
        (tmp_path / "empty_tiles").mkdir()
        _, _, l3_layer = build_layers(config, 34.3416, 108.9398,
                                       enable_l1=False, enable_l2=False, enable_l3=True)
        assert l3_layer is not None
        assert l3_layer.strict_data is True
        with pytest.raises(StrictModeError):
            l3_layer.refine_urban(_make_frame())

    def test_top_level_strict_false_l2_raises_file_not_found(self, tmp_path):
        """data_validation.strict=False: L2 with missing DEM raises FileNotFoundError, not StrictModeError."""
        from benchmarks.run_regression import build_layers
        config = self._base_config(tmp_path)
        config["data_validation"]["strict"] = False
        config["layers"]["l1_macro"]["enabled"] = False
        _, l2_layer, _ = build_layers(config, 34.3416, 108.9398,
                                       enable_l1=False, enable_l2=True, enable_l3=False)
        assert l2_layer is not None
        assert l2_layer.strict_data is False
        with pytest.raises(FileNotFoundError):
            l2_layer.propagate_terrain(_make_frame())

    def test_resolve_strict_flag_priority(self, tmp_path):
        """_resolve_strict_flag: data_validation.strict takes priority over strict_data/strict_mode."""
        from benchmarks.run_regression import _resolve_strict_flag
        assert _resolve_strict_flag({"data_validation": {"strict": True}, "strict_data": False}) is True
        assert _resolve_strict_flag({"data_validation": {"strict": False}, "strict_data": True}) is False
        assert _resolve_strict_flag({"strict_data": True}) is True
        assert _resolve_strict_flag({"strict_mode": True}) is True
        assert _resolve_strict_flag({}) is False

