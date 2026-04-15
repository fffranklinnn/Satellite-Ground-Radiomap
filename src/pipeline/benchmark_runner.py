"""
BenchmarkRunner: deterministic fixed-frame-list pipeline execution.

Runs the full L1 → L2 → L3 → MultiScaleMap.compose() → export_dataset()
pipeline for a fixed list of timestamps, producing ProductManifest records
with output file hashes for reproducibility validation.

Usage:
    runner = BenchmarkRunner(
        frame_builder=fb,
        l1_layer=l1,
        l2_layer=l2,
        l3_layer=l3,
        config=config_dict,
        data_snapshot_id="snap_001",
    )
    results = runner.run(
        timestamps=[datetime(2025, 1, 3, tzinfo=timezone.utc)],
        output_dir=Path("output/benchmark"),
        product_types=["path_loss_map"],
    )
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..context.frame_builder import FrameBuilder
from ..context.multiscale_map import MultiScaleMap
from ..layers.l1_macro import L1MacroLayer
from ..layers.l2_topo import L2TopoLayer
from ..layers.l3_urban import L3UrbanLayer
from ..pipeline.manifest_writer import ManifestWriter
from ..products.manifest import ProductManifest
from ..products.projectors import export_dataset


class BenchmarkRunner:
    """
    Runs the simulation pipeline for a fixed frame list.

    Each frame is processed deterministically:
        FrameBuilder.build(ts) → propagate_entry → propagate_terrain
        → refine_urban → MultiScaleMap.compose() → export_dataset()

    A ProductManifest with output file hashes is produced for each frame.

    Args:
        frame_builder:    FrameBuilder for constructing FrameContext objects.
        l1_layer:         L1MacroLayer (optional; absent layers contribute zero).
        l2_layer:         L2TopoLayer (optional).
        l3_layer:         L3UrbanLayer (optional).
        config:           Config dict (hashed into ProductManifest.config_hash).
        data_snapshot_id: Identifier for the input data snapshot.
    """

    def __init__(
        self,
        frame_builder: FrameBuilder,
        l1_layer: Optional[L1MacroLayer] = None,
        l2_layer: Optional[L2TopoLayer] = None,
        l3_layer: Optional[L3UrbanLayer] = None,
        config: Optional[Dict[str, Any]] = None,
        data_snapshot_id: str = "",
    ) -> None:
        self.frame_builder = frame_builder
        self.l1_layer = l1_layer
        self.l2_layer = l2_layer
        self.l3_layer = l3_layer
        self.config = config or {}
        self.data_snapshot_id = data_snapshot_id

    def run_frame(
        self,
        timestamp: datetime,
        output_dir: Path,
        product_types: List[str],
        prefix: str = "",
        manifest_writer: Optional[ManifestWriter] = None,
    ) -> ProductManifest:
        """
        Run the pipeline for a single timestamp and export products.

        Args:
            timestamp:       UTC datetime for this frame.
            output_dir:      Directory to write output files.
            product_types:   Product types to export (e.g. ["path_loss_map"]).
            prefix:          Optional filename prefix for output files.
            manifest_writer: ManifestWriter to append the output manifest record.

        Returns:
            ProductManifest with output file hashes.
        """
        frame = self.frame_builder.build(timestamp)

        entry = self.l1_layer.propagate_entry(frame) if self.l1_layer else None
        terrain = (
            self.l2_layer.propagate_terrain(frame, entry=entry)
            if self.l2_layer else None
        )
        urban = (
            self.l3_layer.refine_urban(frame, entry=entry)
            if self.l3_layer else None
        )

        msm = MultiScaleMap.compose(
            frame_id=frame.frame_id,
            grid=frame.grid,
            entry=entry,
            terrain=terrain,
            urban=urban,
        )

        manifest = ProductManifest.build(
            frame_id=frame.frame_id,
            timestamp_utc=frame.timestamp.isoformat(),
            config=self.config,
            data_snapshot_id=self.data_snapshot_id,
        )

        export_dataset(
            output_dir=output_dir,
            frame=frame,
            product_types=product_types,
            entry=entry,
            terrain=terrain,
            urban=urban,
            multiscale=msm,
            manifest=manifest,
            prefix=prefix,
            manifest_writer=manifest_writer,
        )

        return manifest

    def run(
        self,
        timestamps: List[datetime],
        output_dir: Union[str, Path],
        product_types: List[str],
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> List[ProductManifest]:
        """
        Run the pipeline for all timestamps in the fixed frame list.

        Args:
            timestamps:    Ordered list of UTC datetimes (the fixed frame list).
            output_dir:    Root directory for output files.
            product_types: Product types to export per frame.
            manifest_path: Optional path for the JSONL manifest file.
                           Defaults to <output_dir>/manifest.jsonl.

        Returns:
            List of ProductManifest objects, one per frame.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if manifest_path is None:
            manifest_path = out / "manifest.jsonl"

        results: List[ProductManifest] = []
        with ManifestWriter(manifest_path) as writer:
            for idx, ts in enumerate(timestamps):
                prefix = f"frame_{idx:06d}_"
                manifest = self.run_frame(
                    timestamp=ts,
                    output_dir=out,
                    product_types=product_types,
                    prefix=prefix,
                    manifest_writer=writer,
                )
                results.append(manifest)

        return results

    @staticmethod
    def load_frame_list(path: Union[str, Path]) -> List[datetime]:
        """
        Load a fixed frame list from a JSON file.

        Expected format:
            {"frames": ["2025-01-03T00:00:00+00:00", ...]}

        Args:
            path: Path to the frame list JSON file.

        Returns:
            List of UTC-aware datetime objects.
        """
        from ..context.time_utils import parse_iso_utc

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [parse_iso_utc(ts) for ts in data["frames"]]
