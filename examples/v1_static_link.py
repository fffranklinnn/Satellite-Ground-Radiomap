#!/usr/bin/env python3
"""
V1.0 Static Link Example

This example demonstrates the V1.0 milestone: static link closure
with L1 (satellite positioning) and L3 (building shadows).
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import L1MacroLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.utils import plot_radio_map, export_radio_map_png


def main():
    """Run V1.0 static link example."""
    print("=" * 60)
    print("SG-MRM V1.0: Static Link Closure")
    print("=" * 60)

    # Define origin for a specific urban area
    origin_lat = 39.9042  # Beijing
    origin_lon = 116.4074

    print(f"\nOrigin: ({origin_lat}, {origin_lon})")
    print("Frequency: 10 GHz")
    print("Satellite altitude: 550 km")

    # L1 Configuration: Wide-area satellite coverage
    l1_config = {
        'grid_size': 256,
        'coverage_km': 256.0,
        'resolution_m': 1000.0,
        'frequency_ghz': 10.0,
        'satellite_altitude_km': 550.0,
        'tle_file': None  # V1.0: Fixed satellite position
    }

    # L3 Configuration: Urban building effects
    l3_config = {
        'grid_size': 256,
        'coverage_km': 0.256,
        'resolution_m': 1.0,
        'building_shapefile': None,  # V1.0: No building data yet
        'satellite_azimuth_deg': 180.0,
        'satellite_elevation_deg': 45.0
    }

    # Initialize layers
    print("\n[1/4] Initializing L1 Macro Layer...")
    l1_layer = L1MacroLayer(l1_config, origin_lat, origin_lon)
    print(f"      {l1_layer}")

    print("\n[2/4] Initializing L3 Urban Layer...")
    l3_layer = L3UrbanLayer(l3_config, origin_lat, origin_lon)
    print(f"      {l3_layer}")

    # Initialize aggregator
    print("\n[3/4] Initializing Radio Map Aggregator...")
    aggregator = RadioMapAggregator(l1_layer=l1_layer, l3_layer=l3_layer)

    # Compute static link for a single timestamp
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    print(f"\n[4/4] Computing static link for {timestamp}...")

    # Get individual contributions
    contributions = aggregator.get_layer_contributions(timestamp)

    l1_map = contributions['l1']
    l3_map = contributions['l3']
    composite_map = contributions['composite']

    # Print statistics
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    print("\nL1 Macro Layer (256 km coverage):")
    print(f"  Min loss: {np.min(l1_map):.2f} dB")
    print(f"  Max loss: {np.max(l1_map):.2f} dB")
    print(f"  Mean loss: {np.mean(l1_map):.2f} dB")

    print("\nL3 Urban Layer (256 m coverage):")
    print(f"  Min loss: {np.min(l3_map):.2f} dB")
    print(f"  Max loss: {np.max(l3_map):.2f} dB")
    print(f"  Mean loss: {np.mean(l3_map):.2f} dB")

    print("\nComposite Map:")
    print(f"  Min loss: {np.min(composite_map):.2f} dB")
    print(f"  Max loss: {np.max(composite_map):.2f} dB")
    print(f"  Mean loss: {np.mean(composite_map):.2f} dB")

    # Save outputs
    output_dir = Path("output/v1_static_link")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving outputs to {output_dir}...")

    # Save individual layers
    plot_radio_map(l1_map, title="L1 Macro Layer (256 km)",
                  output_file=str(output_dir / "l1_macro.png"))

    plot_radio_map(l3_map, title="L3 Urban Layer (256 m)",
                  output_file=str(output_dir / "l3_urban.png"))

    plot_radio_map(composite_map, title="Composite Radio Map",
                  output_file=str(output_dir / "composite.png"))

    # Export as simple PNG (256x256 pixels)
    export_radio_map_png(composite_map, str(output_dir / "composite_raw.png"))

    # Save as numpy array for further processing
    np.save(output_dir / "composite.npy", composite_map)

    print("\n" + "=" * 60)
    print("V1.0 Static Link Closure Complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNext steps for V2.0:")
    print("  - Add TLE-based satellite tracking")
    print("  - Integrate L2 terrain layer with DEM data")
    print("  - Implement time-series simulation")
    print("  - Add building shapefile support for L3")
    print("  - Implement GPU ray tracing for multipath")


if __name__ == '__main__':
    main()
