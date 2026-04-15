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
from src.products.manifest import ProductManifest


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

