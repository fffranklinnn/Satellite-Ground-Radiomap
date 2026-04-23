"""
ProductManifest: immutable record of a simulation product's provenance.

Captures:
- config_hash: SHA-256 of the serialized config dict
- data_snapshot_id: identifier for the input data snapshot (TLE file, DEM, etc.)
- input_file_hashes: {label: sha256_hex} for each input file
- output_file_hashes: {label: sha256_hex} for each output file
- fallbacks_used: list of fallback descriptions (e.g. "L1: no TLE, used synthetic")
- frame_id: the FrameContext frame_id this product belongs to
- timestamp_utc: ISO-8601 UTC string of the simulation timestamp
- metadata: arbitrary extra key-value pairs

JSON round-trip is guaranteed: ProductManifest.from_dict(manifest.to_dict()) == manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

from src.context.time_utils import StrictModeError


MANIFEST_SCHEMA_VERSION = "2.0"


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir(path: str) -> str:
    """
    Compute a deterministic SHA-256 snapshot hash for a directory.

    Hashes sorted relative paths + file contents so the result is stable
    across identical directory trees and non-empty for any non-empty directory.
    Returns "" for missing or empty directories.
    """
    root = Path(path)
    if not root.is_dir():
        return ""
    h = hashlib.sha256()
    found_any = False
    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        found_any = True
    return h.hexdigest() if found_any else ""


def _sha256_dict(d: dict) -> str:
    """Compute SHA-256 of a JSON-serialized dict (sorted keys)."""
    payload = json.dumps(d, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_input_file_paths(config: dict, strict: bool = False) -> Dict[str, str]:
    """
    Extract configured input data file paths from a mission config dict.

    Returns a {label: path_str} mapping for all configured data files.

    In strict (benchmark) mode:
    - Configured but missing files raise StrictModeError
    - Unconfigured optional inputs are silently skipped (no error)

    In normal mode:
    - Configured but missing files emit a warning
    - Unconfigured optional inputs are silently skipped
    """
    import warnings
    paths: Dict[str, str] = {}
    layers = config.get("layers", {})

    # Define expected input sources
    _INPUT_SOURCES = [
        ("l1_macro", "tle", lambda l1: (l1.get("tle", {}).get("file") if isinstance(l1.get("tle"), dict) else None) or l1.get("tle_file"), "tle_file"),
        ("l1_macro", "ionex", lambda l1: l1.get("ionex_file"), "ionex_file"),
        ("l1_macro", "era5", lambda l1: l1.get("era5_file"), "era5_file"),
        ("l2_topo", "dem", lambda l2: l2.get("dem_file"), "dem_file"),
        ("l3_urban", "tiles", lambda l3: l3.get("tile_cache_root") or l3.get("data_dir") or l3.get("tiles_dir"), "tile_cache_root"),
    ]

    for layer_key, source_name, extractor, label in _INPUT_SOURCES:
        layer_cfg = layers.get(layer_key, {})
        path_str = extractor(layer_cfg)
        if path_str is None:
            continue  # Unconfigured optional — silently skip
        path_str = str(path_str)
        p = Path(path_str)
        if not p.exists():
            if strict:
                raise StrictModeError(
                    f"Configured input '{label}' not found: {path_str}. "
                    f"In benchmark mode, all configured inputs must be present."
                )
            else:
                warnings.warn(
                    f"Configured input '{label}' not found: {path_str}. "
                    f"Hash will be empty.",
                    UserWarning,
                    stacklevel=2,
                )
        paths[label] = path_str

    return paths


def _deep_freeze(obj: Any) -> Any:
    """
    Recursively convert mutable containers to immutable equivalents.

    - dict  → MappingProxyType (recursively frozen values)
    - list  → tuple (recursively frozen elements)
    - tuple → tuple (recursively frozen elements)
    - other → unchanged

    This ensures that nested provenance payloads (e.g. metadata with nested
    dicts) cannot be mutated after ProductManifest construction.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    return obj


def _deep_thaw(obj: Any) -> Any:
    """
    Recursively convert frozen containers back to plain mutable equivalents.

    - MappingProxyType → dict (recursively thawed values)
    - tuple            → list (recursively thawed elements)
    - other            → unchanged

    Used by ProductManifest.to_dict() to produce JSON-serializable output.
    """
    if isinstance(obj, MappingProxyType):
        return {k: _deep_thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_deep_thaw(v) for v in obj]
    return obj


@dataclass(frozen=True)
class ProvenanceBlock:
    """Versioned provenance metadata nested within ProductManifest."""
    schema_version: str = MANIFEST_SCHEMA_VERSION
    benchmark_id: str = ""
    input_snapshot_hash: str = ""
    coverage_signature: str = ""
    frame_contract_hash: str = ""
    software_version: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "input_snapshot_hash": self.input_snapshot_hash,
            "coverage_signature": self.coverage_signature,
            "frame_contract_hash": self.frame_contract_hash,
            "software_version": self.software_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceBlock":
        return cls(**{k: str(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class BenchmarkMode:
    """Strict validation flags for benchmark-grade reproducibility."""
    strict_utc: bool = True
    strict_snapshot: bool = True
    allow_fallback: bool = False

    def validate_manifest_inputs(
        self,
        timestamp_utc: str,
        data_snapshot_id: str,
        fallbacks_used: List[str],
    ) -> None:
        """Validate manifest inputs against benchmark strictness."""
        if self.strict_utc:
            from src.context.time_utils import parse_iso_utc
            parse_iso_utc(timestamp_utc, strict=True)

        if self.strict_snapshot and not data_snapshot_id:
            raise StrictModeError(
                "BenchmarkMode requires a non-empty data_snapshot_id."
            )

        if not self.allow_fallback and fallbacks_used:
            raise StrictModeError(
                f"BenchmarkMode does not allow fallbacks: {fallbacks_used}"
            )


@dataclass(frozen=True)
class ProductManifest:
    """
    Immutable provenance record for a simulation product.

    Attributes:
        frame_id:           FrameContext frame identifier.
        timestamp_utc:      ISO-8601 UTC string of the simulation timestamp.
        config_hash:        SHA-256 of the serialized config dict.
        data_snapshot_id:   Identifier for the input data snapshot.
        input_file_hashes:  {label: sha256_hex} for each input file.
        output_file_hashes: {label: sha256_hex} for each output file.
        fallbacks_used:     Descriptions of any fallback behaviors triggered.
        metadata:           Arbitrary extra key-value pairs.
    """

    frame_id: str
    timestamp_utc: str
    config_hash: str
    data_snapshot_id: str
    input_file_hashes: Dict[str, str] = field(default_factory=dict)
    output_file_hashes: Dict[str, str] = field(default_factory=dict)
    fallbacks_used: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[ProvenanceBlock] = None

    def __post_init__(self) -> None:
        # Recursively freeze all container fields so nested provenance payloads
        # (e.g. metadata with nested dicts) cannot be mutated after construction.
        object.__setattr__(self, "input_file_hashes", _deep_freeze(dict(self.input_file_hashes)))
        object.__setattr__(self, "output_file_hashes", _deep_freeze(dict(self.output_file_hashes)))
        object.__setattr__(self, "fallbacks_used", tuple(self.fallbacks_used))
        object.__setattr__(self, "metadata", _deep_freeze(dict(self.metadata)))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = {
            "frame_id": self.frame_id,
            "timestamp_utc": self.timestamp_utc,
            "config_hash": self.config_hash,
            "data_snapshot_id": self.data_snapshot_id,
            "input_file_hashes": _deep_thaw(self.input_file_hashes),
            "output_file_hashes": _deep_thaw(self.output_file_hashes),
            "fallbacks_used": list(self.fallbacks_used),
            "metadata": _deep_thaw(self.metadata),
        }
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductManifest":
        """Deserialize from a dict (inverse of to_dict)."""
        prov = None
        if "provenance" in d and d["provenance"] is not None:
            prov = ProvenanceBlock.from_dict(d["provenance"])
        return cls(
            frame_id=str(d["frame_id"]),
            timestamp_utc=str(d["timestamp_utc"]),
            config_hash=str(d["config_hash"]),
            data_snapshot_id=str(d["data_snapshot_id"]),
            input_file_hashes=dict(d.get("input_file_hashes", {})),
            output_file_hashes=dict(d.get("output_file_hashes", {})),
            fallbacks_used=list(d.get("fallbacks_used", [])),
            metadata=dict(d.get("metadata", {})),
            provenance=prov,
        )

    @classmethod
    def from_json(cls, s: str) -> "ProductManifest":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        frame_id: str,
        timestamp_utc: str,
        config: dict,
        data_snapshot_id: str = "",
        input_files: Optional[Dict[str, str]] = None,
        output_files: Optional[Dict[str, str]] = None,
        fallbacks_used: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        hash_files: bool = False,
    ) -> "ProductManifest":
        """
        Build a ProductManifest from a config dict and optional file paths.

        Args:
            frame_id:           Frame identifier.
            timestamp_utc:      ISO-8601 UTC string.
            config:             Config dict (will be SHA-256 hashed).
            data_snapshot_id:   Input data snapshot identifier.
            input_files:        {label: file_path} — hashed if hash_files=True.
            output_files:       {label: file_path} — hashed if hash_files=True.
            fallbacks_used:     List of fallback descriptions.
            metadata:           Extra key-value pairs.
            hash_files:         If True, compute SHA-256 of each file path.
        """
        config_hash = _sha256_dict(config)

        def _hash_files(files: Optional[Dict[str, str]]) -> Dict[str, str]:
            if not files:
                return {}
            if not hash_files:
                return {k: "" for k in files}
            result = {}
            for label, path in files.items():
                try:
                    p = Path(path)
                    if p.is_dir():
                        result[label] = _sha256_dir(path)
                    else:
                        result[label] = _sha256_file(path)
                except (OSError, IOError):
                    result[label] = ""
            return result

        return cls(
            frame_id=frame_id,
            timestamp_utc=timestamp_utc,
            config_hash=config_hash,
            data_snapshot_id=data_snapshot_id,
            input_file_hashes=_hash_files(input_files),
            output_file_hashes=_hash_files(output_files),
            fallbacks_used=list(fallbacks_used or []),
            metadata=dict(metadata or {}),
        )
