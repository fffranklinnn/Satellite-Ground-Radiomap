#!/usr/bin/env python3
"""
SG-MRM Main Entry Point

This is the main entry point for the Satellite-Ground Multiscale Radio Map
simulation system. It loads configuration, initializes layers, and runs the
simulation over the specified time range.

Usage:
    python main.py [--config CONFIG_FILE] [--output OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import warnings
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext
from src.context import GridSpec, CoverageSpec, FrameBuilder
from src.context.multiscale_map import MultiScaleMap
from src.context.time_utils import require_utc
from src.products.manifest import ProductManifest, collect_input_file_paths
from src.products.projectors import export_dataset
from src.pipeline.manifest_writer import ManifestWriter
from src.utils import setup_logger, SimulationLogger, plot_radio_map, plot_layer_comparison
from src.utils import get_profiler
from src.utils.data_validation import (
    validate_data_integrity,
    format_data_validation_report,
)


def load_config(config_file: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def initialize_layers(config: dict):
    """
    Initialize physical layers based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (l1_layer, l2_layer, l3_layer)
    """
    origin_lat = config['origin']['latitude']
    origin_lon = config['origin']['longitude']

    # Initialize L1 Macro Layer
    l1_layer = None
    if config['layers']['l1_macro']['enabled']:
        l1_config = config['layers']['l1_macro']
        l1_layer = L1MacroLayer(l1_config, origin_lat, origin_lon)
        print(f"Initialized L1 Macro Layer: {l1_layer}")

    # Initialize L2 Terrain Layer
    l2_layer = None
    if config['layers']['l2_topo']['enabled']:
        l2_config = config['layers']['l2_topo']
        l2_layer = L2TopoLayer(l2_config, origin_lat, origin_lon)
        print(f"Initialized L2 Terrain Layer: {l2_layer}")

    # Initialize L3 Urban Layer
    l3_layer = None
    if config['layers']['l3_urban']['enabled']:
        l3_config = config['layers']['l3_urban']
        l3_layer = L3UrbanLayer(l3_config, origin_lat, origin_lon)
        print(f"Initialized L3 Urban Layer: {l3_layer}")

    return l1_layer, l2_layer, l3_layer


def build_frame_builder(config: dict) -> FrameBuilder:
    """
    Build a FrameBuilder from config.

    Constructs a GridSpec from the L1 coverage and origin, and a CoverageSpec
    from the product coverage. This replaces the legacy bare origin_lat/lon
    passing pattern.
    """
    origin_lat = config['origin']['latitude']
    origin_lon = config['origin']['longitude']

    l1_cfg = config['layers'].get('l1_macro', {})
    coarse_km = float(l1_cfg.get('coverage_km', 256.0))
    grid_size = int(l1_cfg.get('grid_size', 256))

    grid = GridSpec.from_legacy_args(origin_lat, origin_lon, coarse_km, grid_size, grid_size)

    product_km = float(config.get('product', {}).get('coverage_km', 0.256))
    product_nx = int(config.get('product', {}).get('grid_size', 256))
    coverage = CoverageSpec.from_config(
        origin_lat=origin_lat, origin_lon=origin_lon,
        coarse_coverage_km=coarse_km, coarse_nx=grid_size, coarse_ny=grid_size,
        product_coverage_km=product_km, product_nx=product_nx, product_ny=product_nx,
    )
    return FrameBuilder(grid=grid, coverage=coverage)


def run_simulation(config: dict, output_dir: Path):
    """
    Run the main simulation loop using the FrameBuilder pipeline.

    Each frame is driven by:
        FrameBuilder.build(ts) -> propagate_entry -> propagate_terrain -> refine_urban

    All timestamps are parsed through parse_iso_utc() for strict UTC enforcement.
    """
    import logging as _logging
    log_level = config['logging']['level']
    log_file = config['logging'].get('log_file')
    logger = setup_logger('sg_mrm', level=getattr(_logging, log_level), log_file=log_file)
    sim_logger = SimulationLogger()
    sim_logger.start_simulation(config)

    # Initialize layers
    l1_layer, l2_layer, l3_layer = initialize_layers(config)

    # Build FrameBuilder (replaces bare origin_lat/lon + legacy aggregator)
    frame_builder = build_frame_builder(config)

    # Parse time parameters through strict UTC helpers
    strict = bool(config.get('data_validation', {}).get('strict', False))
    from src.context.time_utils import parse_iso_utc
    start_time = parse_iso_utc(config['time']['start'], strict=strict)
    end_time   = parse_iso_utc(config['time']['end'],   strict=strict)
    step_hours = config['time']['step_hours']
    time_step  = timedelta(hours=step_hours)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Build a data_snapshot_id from config hash for provenance tracking
    from src.products.manifest import _sha256_dict as _cfg_hash
    data_snapshot_id = config.get('data_validation', {}).get('snapshot_id', '')
    input_file_paths = collect_input_file_paths(config)

    current_time = start_time
    frame_count  = 0

    manifest_path = output_dir / "manifest.jsonl"
    with ManifestWriter(manifest_path) as manifest_writer:
      while current_time <= end_time:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing timestamp: {current_time.isoformat()}")
        logger.info(f"{'='*60}")

        sim_logger.log_layer_start("FramePipeline", current_time)

        # Select satellite before building frame (canonical path)
        sat_info = None
        if l1_layer is not None:
            from src.planning.satellite_selector import SatelliteSelector
            tle_path = config['layers'].get('l1_macro', {}).get('tle', {})
            if isinstance(tle_path, dict):
                tle_path = tle_path.get('file', '')
            tle_path = tle_path or config['layers'].get('l1_macro', {}).get('tle_file', '')
            if tle_path:
                selector = SatelliteSelector(tle_path, strict=strict, min_elevation_deg=5.0)
                target_ids = config['layers'].get('l1_macro', {}).get('target_norad_ids')
                if target_ids:
                    target_ids = [str(x) for x in (target_ids if isinstance(target_ids, list) else [target_ids])]
                sat_info = selector.select(
                    timestamp=current_time,
                    center=(config['origin']['latitude'], config['origin']['longitude']),
                    target_ids=target_ids,
                    strict=strict,
                )

        # Build frame with pre-bound satellite geometry
        frame = frame_builder.build(current_time, sat_info=sat_info)

        # Clear per-frame fallback accumulator before propagation
        if l1_layer is not None:
            l1_layer.clear_fallbacks()

        # Typed state propagation
        entry   = l1_layer.propagate_entry(frame)   if l1_layer  else None
        terrain = l2_layer.propagate_terrain(frame, entry=entry) if l2_layer else None
        urban   = l3_layer.refine_urban(frame, entry=entry)      if l3_layer else None

        # Collect fallbacks recorded during this frame's propagation
        frame_fallbacks = l1_layer.fallbacks_used if l1_layer is not None else []

        # Assemble composite map via projected composition (canonical path)
        _grid = object.__getattribute__(frame, "grid")
        from src.compose import project_to_product_grid
        projected = project_to_product_grid(
            product_grid=_grid, entry=entry, terrain=terrain, urban=urban,
            frame_id=frame.frame_id,
        )
        msm = MultiScaleMap.compose_projected(
            frame_id=frame.frame_id,
            product_grid=_grid,
            **projected,
        )
        composite_map = msm.composite_db
        contributions = {}
        if msm.l1_db is not None:
            contributions['l1'] = msm.l1_db
        if msm.l2_db is not None:
            contributions['l2'] = msm.l2_db
        if msm.l3_db is not None:
            contributions['l3'] = msm.l3_db
        contributions['composite'] = composite_map

        sim_logger.log_layer_end("FramePipeline")

        # Save individual layers
        if config['output']['save_individual_layers']:
            if 'l1' in contributions:
                plot_radio_map(contributions['l1'],
                               title=f"L1 Macro - {current_time.isoformat()}",
                               output_file=str(output_dir / f"l1_macro_{frame_count:04d}.png"),
                               dpi=config['output']['dpi'])
            if 'l2' in contributions:
                plot_radio_map(contributions['l2'],
                               title=f"L2 Terrain - {current_time.isoformat()}",
                               output_file=str(output_dir / f"l2_topo_{frame_count:04d}.png"),
                               dpi=config['output']['dpi'])
            if 'l3' in contributions:
                plot_radio_map(contributions['l3'],
                               title=f"L3 Urban - {current_time.isoformat()}",
                               output_file=str(output_dir / f"l3_urban_{frame_count:04d}.png"),
                               dpi=config['output']['dpi'])
            plot_layer_comparison(
                l1_map=contributions.get('l1'),
                l2_map=contributions.get('l2'),
                l3_map=contributions.get('l3'),
                composite_map=composite_map,
                output_file=str(output_dir / f"comparison_{frame_count:04d}.png"),
                dpi=config['output']['dpi'],
            )

        if config['output']['save_composite']:
            plot_radio_map(composite_map,
                           title=f"Composite Radio Map - {current_time.isoformat()}",
                           output_file=str(output_dir / f"composite_{frame_count:04d}.png"),
                           dpi=config['output']['dpi'])
            from src.products.manifest import ProvenanceBlock, BenchmarkMode, MANIFEST_SCHEMA_VERSION, _sha256_dict
            import subprocess as _sp
            try:
                _git_rev = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=_sp.DEVNULL).strip()
            except Exception:
                _git_rev = "unknown"
            _prov = ProvenanceBlock(
                schema_version=MANIFEST_SCHEMA_VERSION,
                input_snapshot_hash=_sha256_dict({"snapshot": data_snapshot_id}),
                coverage_signature=_sha256_dict({"config_hash": _sha256_dict(config)}),
                frame_contract_hash=_sha256_dict({"frame_id": frame.frame_id, "ts": frame.timestamp.isoformat()}),
                software_version=_git_rev,
            )
            _bm = BenchmarkMode(strict_utc=True, strict_snapshot=True, allow_fallback=False) if strict else None
            frame_manifest = ProductManifest.build(
                frame_id=frame.frame_id,
                timestamp_utc=frame.timestamp.isoformat(),
                config=config,
                data_snapshot_id=data_snapshot_id,
                input_files=collect_input_file_paths(config, strict=strict),
                hash_files=True,
                fallbacks_used=frame_fallbacks,
                provenance=_prov,
                benchmark_mode=_bm,
            )
            written, _ = export_dataset(
                output_dir=output_dir,
                frame=frame,
                product_types=["path_loss_map"],
                multiscale=msm,
                manifest=frame_manifest,
                prefix=f"composite_{frame_count:04d}_",
                manifest_writer=manifest_writer,
            )
            logger.info(f"Saved composite map to {written['path_loss_map']}")

        logger.info(f"Composite map statistics:")
        logger.info(f"  Min loss: {np.min(composite_map):.2f} dB")
        logger.info(f"  Max loss: {np.max(composite_map):.2f} dB")
        logger.info(f"  Mean loss: {np.mean(composite_map):.2f} dB")
        logger.info(f"  Std loss: {np.std(composite_map):.2f} dB")

        current_time += time_step
        frame_count  += 1

    sim_logger.end_simulation()

    if config['performance']['enable_profiling']:
        profiler = get_profiler()
        profiler.print_summary()

    logger.info(f"\nSimulation complete. Generated {frame_count} frames.")
    logger.info(f"Output saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='SG-MRM: Satellite-Ground Multiscale Radio Map Simulation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mission_config.yaml',
        help='Path to configuration file (default: configs/mission_config.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: from config file)'
    )
    parser.add_argument(
        '--strict-data',
        action='store_true',
        help='Fail on missing/unreadable configured data and disable silent fallbacks in L1'
    )
    parser.add_argument(
        '--check-data-only',
        action='store_true',
        help='Run data integrity checks and exit without simulation'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Data integrity checks
    strict_from_cfg = bool(config.get('data_validation', {}).get('strict', False))
    strict_data = bool(args.strict_data or strict_from_cfg)
    if strict_data:
        config.setdefault('layers', {}).setdefault('l1_macro', {})['strict_data'] = True
        config.setdefault('layers', {}).setdefault('l2_topo', {})['strict_data'] = True
        config.setdefault('layers', {}).setdefault('l3_urban', {})['strict_data'] = True

    report = validate_data_integrity(config=config, project_root=Path(__file__).parent, strict=strict_data)
    print(format_data_validation_report(report))

    if report['errors']:
        print("\nData integrity check failed; aborting.")
        sys.exit(2)

    if args.check_data_only:
        print("\nData integrity check passed.")
        sys.exit(0)

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config['output']['directory'])

    # Run simulation
    try:
        run_simulation(config, output_dir)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
