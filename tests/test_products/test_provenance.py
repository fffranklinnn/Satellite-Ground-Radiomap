"""
Provenance reproducibility tests (AC-4, AC-4.1).

Tests:
- Same config + snapshot = identical manifest
- Benchmark mode failures for missing data, naive timestamps, empty snapshot
- Normal mode: unconfigured inputs silently skipped
- Normal mode: configured-but-missing inputs emit warnings
"""

import warnings

import pytest

from src.products.manifest import (
    ProductManifest, ProvenanceBlock, BenchmarkMode,
    collect_input_file_paths, _sha256_dict,
)
from src.context.time_utils import StrictModeError


class TestProvenanceReproducibility:
    def test_identical_config_produces_identical_manifest(self):
        config = {"layers": {"l1_macro": {"freq_ghz": 10.0}}}
        m1 = ProductManifest.build(
            frame_id="f1", timestamp_utc="2025-01-01T00:00:00+00:00",
            config=config, data_snapshot_id="snap1",
        )
        m2 = ProductManifest.build(
            frame_id="f1", timestamp_utc="2025-01-01T00:00:00+00:00",
            config=config, data_snapshot_id="snap1",
        )
        assert m1.config_hash == m2.config_hash
        assert m1.to_dict() == m2.to_dict()

    def test_provenance_block_round_trip(self):
        prov = ProvenanceBlock(
            benchmark_id="b1", input_snapshot_hash="h1",
            coverage_signature="cs1", frame_contract_hash="fc1",
            software_version="v1",
        )
        m = ProductManifest(
            frame_id="f1", timestamp_utc="2025-01-01T00:00:00+00:00",
            config_hash="ch", data_snapshot_id="snap",
            provenance=prov,
        )
        d = m.to_dict()
        m2 = ProductManifest.from_dict(d)
        assert m2.provenance is not None
        assert m2.provenance.benchmark_id == "b1"
        assert m2.provenance.schema_version == "2.0"

    def test_fallbacks_used_distinguishes_runs(self):
        m_normal = ProductManifest(
            frame_id="f1", timestamp_utc="2025-01-01T00:00:00+00:00",
            config_hash="ch", data_snapshot_id="snap",
            fallbacks_used=(),
        )
        m_fallback = ProductManifest(
            frame_id="f1", timestamp_utc="2025-01-01T00:00:00+00:00",
            config_hash="ch", data_snapshot_id="snap",
            fallbacks_used=("L1: synthetic TLE",),
        )
        assert m_normal.fallbacks_used == ()
        assert m_fallback.fallbacks_used == ("L1: synthetic TLE",)
        assert m_normal.to_dict() != m_fallback.to_dict()


class TestBenchmarkModeStrict:
    def test_empty_snapshot_id_raises(self):
        bm = BenchmarkMode()
        with pytest.raises(StrictModeError, match="data_snapshot_id"):
            bm.validate_manifest_inputs(
                "2025-01-01T00:00:00+00:00", "", [],
            )

    def test_fallback_raises(self):
        bm = BenchmarkMode()
        with pytest.raises(StrictModeError, match="fallback"):
            bm.validate_manifest_inputs(
                "2025-01-01T00:00:00+00:00", "snap", ["L1: synthetic"],
            )

    def test_naive_timestamp_raises(self):
        bm = BenchmarkMode()
        with pytest.raises(StrictModeError):
            bm.validate_manifest_inputs(
                "2025-01-01T00:00:00", "snap", [],
            )

    def test_valid_inputs_pass(self):
        bm = BenchmarkMode()
        bm.validate_manifest_inputs(
            "2025-01-01T00:00:00+00:00", "snap", [],
        )


class TestCollectInputFilePaths:
    def test_unconfigured_inputs_silently_skipped(self):
        paths = collect_input_file_paths({"layers": {}})
        assert paths == {}

    def test_configured_missing_warns_in_normal_mode(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paths = collect_input_file_paths(
                {"layers": {"l1_macro": {"tle_file": "/nonexistent.tle"}}},
                strict=False,
            )
            assert len(w) >= 1
            assert "not found" in str(w[0].message)
            assert "tle_file" in paths

    def test_configured_missing_raises_in_strict_mode(self):
        with pytest.raises(StrictModeError, match="tle_file"):
            collect_input_file_paths(
                {"layers": {"l1_macro": {"tle_file": "/nonexistent.tle"}}},
                strict=True,
            )

    def test_unconfigured_does_not_raise_in_normal_mode(self):
        paths = collect_input_file_paths(
            {"layers": {"l1_macro": {}}},
            strict=False,
        )
        assert "tle_file" not in paths
