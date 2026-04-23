"""
MapProduct: typed product abstraction for multi-product export (AC-5).

Each MapProduct carries product type, array, grid, field type, frame ID,
and a required manifest reference. Exporting without a manifest raises an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.context.grid_spec import GridSpec
from src.products.manifest import ProductManifest


class MissingManifestError(ValueError):
    """Raised when exporting a product without a valid manifest reference."""


@dataclass(frozen=True)
class MapProduct:
    """A typed simulation product with provenance traceability.

    Attributes:
        product_id:   Unique product identifier.
        product_type: Type of product (e.g. "path_loss", "visibility", "terrain_blockage").
        grid:         GridSpec the product array is defined on.
        values:       Product array (float32 or bool).
        units:        Physical units (e.g. "dB", "bool", "degrees").
        frame_id:     Frame this product belongs to.
        manifest:     Required ProductManifest for provenance.
    """

    product_id: str
    product_type: str
    grid: GridSpec
    values: np.ndarray
    units: str
    frame_id: str
    manifest: Optional[ProductManifest] = None

    def __post_init__(self) -> None:
        expected = (self.grid.ny, self.grid.nx)
        if self.values.shape != expected:
            raise ValueError(
                f"MapProduct values shape {self.values.shape} != grid {expected}"
            )

    def validate_manifest(self) -> None:
        """Raise MissingManifestError if no manifest is attached."""
        if self.manifest is None:
            raise MissingManifestError(
                f"MapProduct '{self.product_id}' has no manifest. "
                "Canonical export requires a valid ProductManifest reference."
            )

    def to_traceability_dict(self) -> Dict[str, Any]:
        """Return traceability metadata for dataset export."""
        self.validate_manifest()
        d = {
            "product_id": self.product_id,
            "product_type": self.product_type,
            "units": self.units,
            "frame_id": self.frame_id,
            "grid_role": self.grid.role,
            "grid_center": (self.grid.center_lat, self.grid.center_lon),
            "grid_resolution_m": (self.grid.dx_m, self.grid.dy_m),
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
        }
        if self.manifest is not None:
            d["manifest"] = {
                "config_hash": self.manifest.config_hash,
                "data_snapshot_id": self.manifest.data_snapshot_id,
                "timestamp_utc": self.manifest.timestamp_utc,
                "fallbacks_used": list(self.manifest.fallbacks_used),
            }
            if self.manifest.provenance is not None:
                d["manifest"]["provenance"] = self.manifest.provenance.to_dict()
        return d
