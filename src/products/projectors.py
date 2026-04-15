"""
Product projectors: extract named product arrays from typed layer states.

Each projector takes a FrameContext + layer states and returns a named
numpy array. The export function assembles a dataset dict and writes
NPY + JSON sidecar files.

Supported product types:
    path_loss_map       — composite path loss (dB), from MultiScaleMap
    visibility_mask     — boolean mask: True where satellite is visible
    elevation_field     — per-pixel elevation angle to satellite (deg)
    azimuth_field       — per-pixel azimuth angle to satellite (deg)
    terrain_blockage    — terrain occlusion mask (bool)
    urban_residual      — urban NLoS residual loss (dB)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from ..context.frame_context import FrameContext
from ..context.layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from ..context.multiscale_map import MultiScaleMap
from ..products.manifest import ProductManifest, _sha256_file

if TYPE_CHECKING:
    from ..pipeline.manifest_writer import ManifestWriter


class UnknownProductTypeError(ValueError):
    """Raised when an unrecognized product type is requested."""


# ---------------------------------------------------------------------------
# Projector registry
# ---------------------------------------------------------------------------

_PRODUCT_TYPES = {
    "path_loss_map",
    "visibility_mask",
    "elevation_field",
    "azimuth_field",
    "terrain_blockage",
    "urban_residual",
}


def project(
    product_type: str,
    frame: FrameContext,
    entry: Optional[EntryWaveState] = None,
    terrain: Optional[TerrainState] = None,
    urban: Optional[UrbanRefinementState] = None,
    multiscale: Optional[MultiScaleMap] = None,
) -> np.ndarray:
    """
    Extract a named product array from typed layer states.

    Args:
        product_type: One of the supported product type strings.
        frame:        FrameContext for this simulation frame.
        entry:        EntryWaveState from L1 (required for L1-derived products).
        terrain:      TerrainState from L2 (required for terrain products).
        urban:        UrbanRefinementState from L3 (required for urban products).
        multiscale:   MultiScaleMap (required for path_loss_map).

    Returns:
        numpy array for the requested product.

    Raises:
        UnknownProductTypeError: If product_type is not recognized.
        ValueError: If a required state is missing for the requested product.
    """
    if product_type not in _PRODUCT_TYPES:
        raise UnknownProductTypeError(
            f"Unknown product type: {product_type!r}. "
            f"Supported types: {sorted(_PRODUCT_TYPES)}"
        )

    if product_type == "path_loss_map":
        if multiscale is not None:
            return multiscale.composite_db.astype(np.float32, copy=False)
        # Fallback: assemble from states
        msm = MultiScaleMap.compose(
            frame_id=frame.frame_id,
            grid=frame.grid,
            entry=entry,
            terrain=terrain,
            urban=urban,
        )
        return msm.composite_db

    if product_type == "visibility_mask":
        if entry is None:
            raise ValueError("visibility_mask requires EntryWaveState (entry).")
        return (~entry.occlusion_mask).astype(np.bool_)

    if product_type == "elevation_field":
        if entry is None:
            raise ValueError("elevation_field requires EntryWaveState (entry).")
        return entry.elevation_deg.astype(np.float32, copy=False)

    if product_type == "azimuth_field":
        if entry is None:
            raise ValueError("azimuth_field requires EntryWaveState (entry).")
        return entry.azimuth_deg.astype(np.float32, copy=False)

    if product_type == "terrain_blockage":
        if terrain is None:
            raise ValueError("terrain_blockage requires TerrainState (terrain).")
        return terrain.occlusion_mask.astype(np.bool_)

    if product_type == "urban_residual":
        if urban is None:
            raise ValueError("urban_residual requires UrbanRefinementState (urban).")
        return urban.urban_residual_db.astype(np.float32, copy=False)

    # Should never reach here given the guard above
    raise UnknownProductTypeError(f"Unhandled product type: {product_type!r}")


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------

def export_dataset(
    output_dir: Union[str, Path],
    frame: FrameContext,
    product_types: List[str],
    entry: Optional[EntryWaveState] = None,
    terrain: Optional[TerrainState] = None,
    urban: Optional[UrbanRefinementState] = None,
    multiscale: Optional[MultiScaleMap] = None,
    manifest: Optional[ProductManifest] = None,
    prefix: str = "",
    manifest_writer: Optional["ManifestWriter"] = None,
) -> tuple:
    """
    Export a set of product arrays to NPY files with a JSON sidecar.

    Each product is saved as:
        <output_dir>/<prefix><product_type>.npy

    A JSON sidecar is written to:
        <output_dir>/<prefix>dataset.json

    If manifest_writer is provided, a ProductManifest with output file hashes
    is appended to the JSONL manifest file.

    Args:
        output_dir:       Directory to write output files.
        frame:            FrameContext for this simulation frame.
        product_types:    List of product type strings to export.
        entry:            EntryWaveState from L1.
        terrain:          TerrainState from L2.
        urban:            UrbanRefinementState from L3.
        multiscale:       MultiScaleMap (used for path_loss_map if provided).
        manifest:         Existing ProductManifest to embed in the JSON sidecar.
                          If None and manifest_writer is provided, a minimal one
                          is built from frame metadata.
        prefix:           Optional filename prefix.
        manifest_writer:  ManifestWriter to append the output manifest record.

    Returns:
        Tuple of (written, output_manifest) where:
          - written: Dict mapping product_type -> absolute file path (str).
          - output_manifest: The augmented ProductManifest with output_file_hashes
            (None if manifest_writer was not provided).

    Raises:
        UnknownProductTypeError: If any product_type is not recognized.
        ValueError: If a required state is missing for a requested product.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}
    sidecar: Dict[str, Any] = {
        "frame_id": frame.frame_id,
        "timestamp_utc": frame.timestamp.isoformat(),
        "products": {},
    }

    for pt in product_types:
        arr = project(
            pt, frame,
            entry=entry, terrain=terrain, urban=urban, multiscale=multiscale,
        )
        fname = f"{prefix}{pt}.npy"
        fpath = out / fname
        np.save(fpath, arr)
        written[pt] = str(fpath)
        sidecar["products"][pt] = {
            "file": fname,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

    output_manifest: Optional[ProductManifest] = None

    # Build output manifest with file hashes if a writer is provided
    if manifest_writer is not None:
        from ..pipeline.manifest_writer import ManifestWriter as _MW  # lazy import
        output_hashes = {pt: _sha256_file(written[pt]) for pt in written}
        if manifest is not None:
            output_manifest = ProductManifest(
                frame_id=manifest.frame_id,
                timestamp_utc=manifest.timestamp_utc,
                config_hash=manifest.config_hash,
                data_snapshot_id=manifest.data_snapshot_id,
                input_file_hashes=dict(manifest.input_file_hashes),
                output_file_hashes=output_hashes,
                fallbacks_used=list(manifest.fallbacks_used),
                metadata=dict(manifest.metadata),
            )
        else:
            output_manifest = ProductManifest(
                frame_id=frame.frame_id,
                timestamp_utc=frame.timestamp.isoformat(),
                config_hash="",
                data_snapshot_id="",
                output_file_hashes=output_hashes,
            )
        manifest_writer.write(output_manifest)
        sidecar["manifest"] = output_manifest.to_dict()
    elif manifest is not None:
        sidecar["manifest"] = manifest.to_dict()

    sidecar_path = out / f"{prefix}dataset.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    return written, output_manifest
