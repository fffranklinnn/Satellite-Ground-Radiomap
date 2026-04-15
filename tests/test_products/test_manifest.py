"""
Tests for ProductManifest and ManifestWriter.
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.products.manifest import ProductManifest, _sha256_dict
from src.pipeline.manifest_writer import ManifestWriter


FRAME_ID = "20250601T120000Z_25544"
TS_UTC = "2025-06-01T12:00:00+00:00"
CONFIG = {"frequency_ghz": 2.0, "grid_size": 256, "origin": {"lat": 34.0, "lon": 108.0}}


# ---------------------------------------------------------------------------
# ProductManifest: construction and validation
# ---------------------------------------------------------------------------

class TestProductManifest:
    def test_basic_construction(self):
        m = ProductManifest(
            frame_id=FRAME_ID,
            timestamp_utc=TS_UTC,
            config_hash="abc123",
            data_snapshot_id="snap_001",
        )
        assert m.frame_id == FRAME_ID
        assert m.timestamp_utc == TS_UTC
        assert m.config_hash == "abc123"
        assert m.data_snapshot_id == "snap_001"
        assert m.input_file_hashes == {}
        assert m.output_file_hashes == {}
        assert m.fallbacks_used == ()
        assert m.metadata == {}

    def test_with_all_fields(self):
        m = ProductManifest(
            frame_id=FRAME_ID,
            timestamp_utc=TS_UTC,
            config_hash="abc",
            data_snapshot_id="snap",
            input_file_hashes={"tle": "deadbeef"},
            output_file_hashes={"composite": "cafebabe"},
            fallbacks_used=["L1: no TLE, used synthetic"],
            metadata={"version": "2.0"},
        )
        assert m.input_file_hashes["tle"] == "deadbeef"
        assert m.output_file_hashes["composite"] == "cafebabe"
        assert "L1: no TLE, used synthetic" in m.fallbacks_used
        assert m.metadata["version"] == "2.0"

    def test_frozen_prevents_reassignment(self):
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="x", data_snapshot_id="y",
        )
        with pytest.raises((AttributeError, TypeError)):
            m.frame_id = "other"

    def test_input_file_hashes_deeply_immutable(self):
        """input_file_hashes must be a read-only mapping (MappingProxyType)."""
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="x", data_snapshot_id="y",
            input_file_hashes={"tle": "abc"},
        )
        with pytest.raises(TypeError):
            m.input_file_hashes["new_key"] = "value"

    def test_fallbacks_used_is_tuple(self):
        """fallbacks_used must be a tuple (immutable sequence)."""
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="x", data_snapshot_id="y",
            fallbacks_used=["fallback_a"],
        )
        assert isinstance(m.fallbacks_used, tuple)
        assert m.fallbacks_used == ("fallback_a",)

    def test_metadata_deeply_immutable(self):
        """metadata must be a read-only mapping."""
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="x", data_snapshot_id="y",
            metadata={"k": "v"},
        )
        with pytest.raises(TypeError):
            m.metadata["new_key"] = "value"

    def test_nested_metadata_dict_is_immutable(self):
        """Nested dicts inside metadata must also be read-only (deep freeze)."""
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="x", data_snapshot_id="y",
            metadata={"nested": {"a": 1}},
        )
        with pytest.raises(TypeError):
            m.metadata["nested"]["a"] = 99

    def test_nested_metadata_list_is_immutable(self):
        """Nested lists inside metadata must be converted to tuples (deep freeze)."""
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="x", data_snapshot_id="y",
            metadata={"items": [1, 2, 3]},
        )
        assert isinstance(m.metadata["items"], tuple)
        with pytest.raises(AttributeError):
            m.metadata["items"].append(4)

    def test_to_dict_nested_metadata_is_plain_dict(self):
        """to_dict() must return plain dicts/lists, not MappingProxyType/tuple."""
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="x", data_snapshot_id="y",
            metadata={"nested": {"a": 1}, "items": [1, 2]},
        )
        d = m.to_dict()
        assert isinstance(d["metadata"], dict)
        assert isinstance(d["metadata"]["nested"], dict)
        assert isinstance(d["metadata"]["items"], list)
        # Must be JSON-serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# ProductManifest: JSON round-trip
# ---------------------------------------------------------------------------

class TestProductManifestJsonRoundTrip:
    def test_to_dict_from_dict(self):
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="abc", data_snapshot_id="snap",
            input_file_hashes={"tle": "hash1"},
            output_file_hashes={"npy": "hash2"},
            fallbacks_used=["fallback_a"],
            metadata={"k": "v"},
        )
        d = m.to_dict()
        m2 = ProductManifest.from_dict(d)
        assert m2.frame_id == m.frame_id
        assert m2.timestamp_utc == m.timestamp_utc
        assert m2.config_hash == m.config_hash
        assert m2.data_snapshot_id == m.data_snapshot_id
        assert m2.input_file_hashes == m.input_file_hashes
        assert m2.output_file_hashes == m.output_file_hashes
        assert m2.fallbacks_used == m.fallbacks_used
        assert m2.metadata == m.metadata

    def test_to_json_from_json(self):
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="abc", data_snapshot_id="snap",
        )
        s = m.to_json()
        m2 = ProductManifest.from_json(s)
        assert m2.frame_id == m.frame_id
        assert m2.config_hash == m.config_hash

    def test_json_is_valid_json(self):
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="abc", data_snapshot_id="snap",
        )
        parsed = json.loads(m.to_json())
        assert parsed["frame_id"] == FRAME_ID

    def test_round_trip_preserves_empty_lists_and_dicts(self):
        m = ProductManifest(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config_hash="abc", data_snapshot_id="snap",
        )
        m2 = ProductManifest.from_dict(m.to_dict())
        assert m2.input_file_hashes == {}
        assert m2.fallbacks_used == ()


# ---------------------------------------------------------------------------
# ProductManifest.build() factory
# ---------------------------------------------------------------------------

class TestProductManifestBuild:
    def test_build_computes_config_hash(self):
        m = ProductManifest.build(
            frame_id=FRAME_ID, timestamp_utc=TS_UTC,
            config=CONFIG, data_snapshot_id="snap",
        )
        expected_hash = _sha256_dict(CONFIG)
        assert m.config_hash == expected_hash

    def test_build_same_config_same_hash(self):
        m1 = ProductManifest.build(FRAME_ID, TS_UTC, CONFIG, "snap")
        m2 = ProductManifest.build(FRAME_ID, TS_UTC, CONFIG, "snap")
        assert m1.config_hash == m2.config_hash

    def test_build_different_config_different_hash(self):
        cfg2 = dict(CONFIG, frequency_ghz=5.0)
        m1 = ProductManifest.build(FRAME_ID, TS_UTC, CONFIG, "snap")
        m2 = ProductManifest.build(FRAME_ID, TS_UTC, cfg2, "snap")
        assert m1.config_hash != m2.config_hash

    def test_build_without_hash_files_gives_empty_hashes(self):
        m = ProductManifest.build(
            FRAME_ID, TS_UTC, CONFIG, "snap",
            input_files={"tle": "/nonexistent/tle.txt"},
            hash_files=False,
        )
        assert m.input_file_hashes["tle"] == ""

    def test_build_with_fallbacks(self):
        m = ProductManifest.build(
            FRAME_ID, TS_UTC, CONFIG, "snap",
            fallbacks_used=["L1: synthetic TLE"],
        )
        assert "L1: synthetic TLE" in m.fallbacks_used

    def test_build_with_metadata(self):
        m = ProductManifest.build(
            FRAME_ID, TS_UTC, CONFIG, "snap",
            metadata={"pipeline_version": "2.0"},
        )
        assert m.metadata["pipeline_version"] == "2.0"


# ---------------------------------------------------------------------------
# ManifestWriter: JSONL output
# ---------------------------------------------------------------------------

class TestManifestWriter:
    def _make_manifest(self, frame_id=FRAME_ID):
        return ProductManifest.build(
            frame_id=frame_id, timestamp_utc=TS_UTC,
            config=CONFIG, data_snapshot_id="snap",
        )

    def test_write_creates_file(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        with ManifestWriter(path) as w:
            w.write(self._make_manifest())
        assert path.exists()

    def test_write_appends_lines(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        with ManifestWriter(path) as w:
            w.write(self._make_manifest("frame_1"))
            w.write(self._make_manifest("frame_2"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_each_line_is_valid_json(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        with ManifestWriter(path) as w:
            w.write(self._make_manifest())
        for line in path.read_text().strip().splitlines():
            parsed = json.loads(line)
            assert "frame_id" in parsed

    def test_round_trip_via_jsonl(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        original = self._make_manifest()
        with ManifestWriter(path) as w:
            w.write(original)
        line = path.read_text().strip()
        recovered = ProductManifest.from_json(line)
        assert recovered.frame_id == original.frame_id
        assert recovered.config_hash == original.config_hash

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "manifest.jsonl"
        with ManifestWriter(path) as w:
            w.write(self._make_manifest())
        assert path.exists()

    def test_appends_across_instances(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        with ManifestWriter(path) as w:
            w.write(self._make_manifest("frame_1"))
        with ManifestWriter(path) as w:
            w.write(self._make_manifest("frame_2"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        ids = [json.loads(l)["frame_id"] for l in lines]
        assert "frame_1" in ids
        assert "frame_2" in ids
