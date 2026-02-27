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
from datetime import datetime, timedelta
import yaml
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext
from src.utils import setup_logger, SimulationLogger, plot_radio_map, plot_layer_comparison
from src.utils import get_profiler


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


def run_simulation(config: dict, output_dir: Path):
    """
    Run the main simulation loop.

    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    """
    # Setup logging
    log_level = config['logging']['level']
    log_file = config['logging'].get('log_file')
    logger = setup_logger('sg_mrm', level=getattr(__import__('logging'), log_level), log_file=log_file)
    sim_logger = SimulationLogger()

    # Start simulation
    sim_logger.start_simulation(config)

    # Initialize layers
    l1_layer, l2_layer, l3_layer = initialize_layers(config)

    # Initialize aggregator
    aggregator = RadioMapAggregator(l1_layer, l2_layer, l3_layer)
    logger.info("Initialized RadioMapAggregator")

    # Parse time parameters
    start_time = datetime.fromisoformat(config['time']['start'])
    end_time = datetime.fromisoformat(config['time']['end'])
    step_hours = config['time']['step_hours']
    time_step = timedelta(hours=step_hours)

    # Origin coordinates
    origin_lat = config['origin']['latitude']
    origin_lon = config['origin']['longitude']

    # Build LayerContext from config (incident_dir for L3)
    l3_cfg = config['layers'].get('l3_urban', {})
    incident_dir_cfg = l3_cfg.get('incident_dir')
    context = LayerContext.from_any({'incident_dir': incident_dir_cfg}) if incident_dir_cfg else None

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Simulation loop
    current_time = start_time
    frame_count = 0

    while current_time <= end_time:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing timestamp: {current_time}")
        logger.info(f"{'='*60}")

        # Compute composite radio map
        sim_logger.log_layer_start("Aggregator", current_time)
        composite_map = aggregator.aggregate(origin_lat, origin_lon, current_time, context)
        sim_logger.log_layer_end("Aggregator")

        # Get individual layer contributions
        if config['output']['save_individual_layers']:
            contributions = aggregator.get_layer_contributions(origin_lat, origin_lon, current_time, context)

            # Save individual layers
            if l1_layer and 'l1' in contributions:
                l1_file = output_dir / f"l1_macro_{frame_count:04d}.png"
                plot_radio_map(contributions['l1'], title=f"L1 Macro - {current_time}",
                             output_file=str(l1_file), dpi=config['output']['dpi'])

            if l2_layer and 'l2' in contributions:
                l2_file = output_dir / f"l2_topo_{frame_count:04d}.png"
                plot_radio_map(contributions['l2'], title=f"L2 Terrain - {current_time}",
                             output_file=str(l2_file), dpi=config['output']['dpi'])

            if l3_layer and 'l3' in contributions:
                l3_file = output_dir / f"l3_urban_{frame_count:04d}.png"
                plot_radio_map(contributions['l3'], title=f"L3 Urban - {current_time}",
                             output_file=str(l3_file), dpi=config['output']['dpi'])

            # Save layer comparison
            comparison_file = output_dir / f"comparison_{frame_count:04d}.png"
            plot_layer_comparison(
                l1_map=contributions.get('l1'),
                l2_map=contributions.get('l2'),
                l3_map=contributions.get('l3'),
                composite_map=contributions.get('composite'),
                output_file=str(comparison_file),
                dpi=config['output']['dpi']
            )

        # Save composite map
        if config['output']['save_composite']:
            composite_file = output_dir / f"composite_{frame_count:04d}.png"
            plot_radio_map(composite_map, title=f"Composite Radio Map - {current_time}",
                         output_file=str(composite_file), dpi=config['output']['dpi'])

            # Also save as numpy array
            npy_file = output_dir / f"composite_{frame_count:04d}.npy"
            np.save(npy_file, composite_map)
            logger.info(f"Saved composite map to {npy_file}")

        # Log statistics
        logger.info(f"Composite map statistics:")
        logger.info(f"  Min loss: {np.min(composite_map):.2f} dB")
        logger.info(f"  Max loss: {np.max(composite_map):.2f} dB")
        logger.info(f"  Mean loss: {np.mean(composite_map):.2f} dB")
        logger.info(f"  Std loss: {np.std(composite_map):.2f} dB")

        # Advance time
        current_time += time_step
        frame_count += 1

    # End simulation
    sim_logger.end_simulation()

    # Print performance summary
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

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

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
