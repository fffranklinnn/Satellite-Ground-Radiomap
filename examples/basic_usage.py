#!/usr/bin/env python3
"""
Basic usage example for SG-MRM.

This example demonstrates how to:
1. Create layers programmatically
2. Initialize the aggregator
3. Compute a single radio map
4. Visualize the results
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import L1MacroLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.utils import plot_radio_map, plot_layer_comparison


def main():
    """Run basic usage example."""
    print("=" * 60)
    print("SG-MRM Basic Usage Example")
    print("=" * 60)

    # Define origin (Beijing example)
    origin_lat = 39.9042
    origin_lon = 116.4074

    # Create L1 Macro Layer configuration
    l1_config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0
    }

    # Create L3 Urban Layer configuration
    l3_config = {
        'grid_size': 256,
        'coverage_km': 0.256,
        'resolution_m': 1.0,
        'satellite_azimuth_deg': 180.0,
        'satellite_elevation_deg': 45.0
    }

    # Initialize layers
    print("\nInitializing layers...")
    l1_layer = L1MacroLayer(l1_config, origin_lat, origin_lon)
    l3_layer = L3UrbanLayer(l3_config, origin_lat, origin_lon)
    print(f"L1 Layer: {l1_layer}")
    print(f"L3 Layer: {l3_layer}")

    # Initialize aggregator
    print("\nInitializing aggregator...")
    aggregator = RadioMapAggregator(l1_layer=l1_layer, l3_layer=l3_layer)

    # Compute radio map for a specific timestamp
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    print(f"\nComputing radio map for {timestamp}...")

    composite_map = aggregator.aggregate(timestamp)

    # Print statistics
    print("\nComposite Map Statistics:")
    print(f"  Shape: {composite_map.shape}")
    print(f"  Min loss: {np.min(composite_map):.2f} dB")
    print(f"  Max loss: {np.max(composite_map):.2f} dB")
    print(f"  Mean loss: {np.mean(composite_map):.2f} dB")
    print(f"  Std loss: {np.std(composite_map):.2f} dB")

    # Get individual layer contributions
    print("\nGetting individual layer contributions...")
    contributions = aggregator.get_layer_contributions(timestamp)

    # Visualize results
    print("\nVisualizing results...")
    output_dir = Path("output/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot composite map
    plot_radio_map(
        composite_map,
        title="Composite Radio Map",
        output_file=str(output_dir / "basic_composite.png")
    )

    # Plot layer comparison
    plot_layer_comparison(
        l1_map=contributions.get('l1'),
        l3_map=contributions.get('l3'),
        composite_map=contributions.get('composite'),
        output_file=str(output_dir / "basic_comparison.png")
    )

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
