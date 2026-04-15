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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple


def _sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dict(d: dict) -> str:
    """Compute SHA-256 of a JSON-serialized dict (sorted keys)."""
    payload = json.dumps(d, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        return {
            "frame_id": self.frame_id,
            "timestamp_utc": self.timestamp_utc,
            "config_hash": self.config_hash,
            "data_snapshot_id": self.data_snapshot_id,
            "input_file_hashes": _deep_thaw(self.input_file_hashes),
            "output_file_hashes": _deep_thaw(self.output_file_hashes),
            "fallbacks_used": list(self.fallbacks_used),
            "metadata": _deep_thaw(self.metadata),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductManifest":
        """Deserialize from a dict (inverse of to_dict)."""
        return cls(
            frame_id=str(d["frame_id"]),
            timestamp_utc=str(d["timestamp_utc"]),
            config_hash=str(d["config_hash"]),
            data_snapshot_id=str(d["data_snapshot_id"]),
            input_file_hashes=dict(d.get("input_file_hashes", {})),
            output_file_hashes=dict(d.get("output_file_hashes", {})),
            fallbacks_used=list(d.get("fallbacks_used", [])),
            metadata=dict(d.get("metadata", {})),
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
