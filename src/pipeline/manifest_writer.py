"""
ManifestWriter: writes ProductManifest records to a JSONL file.

Each call to write() appends one JSON line to the output file.
The writer is not thread-safe; use one instance per output file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from ..products.manifest import ProductManifest


class ManifestWriter:
    """
    Appends ProductManifest records to a JSONL file.

    Usage:
        writer = ManifestWriter("output/manifest.jsonl")
        writer.write(manifest)
        writer.close()

    Or as a context manager:
        with ManifestWriter("output/manifest.jsonl") as writer:
            writer.write(manifest)
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def write(self, manifest: ProductManifest) -> None:
        """Append one manifest record as a JSON line."""
        line = json.dumps(manifest.to_dict(), ensure_ascii=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
