#!/usr/bin/env python3
"""
Debug script to investigate SINR calculation issues.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.l1_macro import L1MacroLayer


def main():
    config_path = project_root / "configs" / "mission_config.yaml"
    layer = L1MacroLayer(str(config_path))
    layer.enable_interference = True
    layer.max_interfering_sats = 3  # Use fewer satellites for debugging

    timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)

    print("Getting visible satellites...")
    visible_sats = layer.get_visible_satellites(
        34.3416, 108.9398, timestamp,
        min_elevation_deg=layer.MIN_ELEVATION_DEG
    )

    print(f"\nFound {len(visible_sats)} visible satellites")
    for i, sat in enumerate(visible_sats[:5]):
        print(f"  {i+1}. NORAD {sat['norad_id']}: el={sat['elevation_deg']:.1f}°, "
              f"az={sat['azimuth_deg']:.1f}°, range={sat['slant_range_km']:.1f} km")

    print("\nComputing received power for target satellite...")
    target_sat = visible_sats[0]
    p_target = layer._compute_received_power(target_sat, 34.3416, 108.9398, None)

    print(f"Target power stats:")
    print(f"  Min: {p_target.min():.2f} dBm")
    print(f"  Max: {p_target.max():.2f} dBm")
    print(f"  Mean: {p_target.mean():.2f} dBm")
    print(f"  Std: {p_target.std():.2f} dB")

    print("\nComputing received power for interfering satellites...")
    for i, sat in enumerate(visible_sats[1:4]):
        p_int = layer._compute_received_power(sat, 34.3416, 108.9398, None)
        print(f"Interferer {i+1} (NORAD {sat['norad_id']}):")
        print(f"  Min: {p_int.min():.2f} dBm")
        print(f"  Max: {p_int.max():.2f} dBm")
        print(f"  Mean: {p_int.mean():.2f} dBm")

    print("\nComputing full SINR...")
    sinr_db, metadata = layer.compute_multisat_sinr(34.3416, 108.9398, timestamp)

    print(f"\nSINR stats:")
    valid_mask = ~np.isinf(sinr_db)
    print(f"  Valid pixels: {valid_mask.sum()} / {sinr_db.size}")
    print(f"  Min: {sinr_db[valid_mask].min():.2f} dB")
    print(f"  Max: {sinr_db[valid_mask].max():.2f} dB")
    print(f"  Mean: {sinr_db[valid_mask].mean():.2f} dB")
    print(f"  Std: {sinr_db[valid_mask].std():.2f} dB")
    print(f"  Unique values: {len(np.unique(sinr_db[valid_mask]))}")

    # Check if all values are the same
    if len(np.unique(sinr_db[valid_mask])) == 1:
        print("\n⚠️  WARNING: All SINR values are identical!")
        print("This suggests a problem in the calculation.")

        # Debug: check intermediate values
        print("\nDebug info:")
        print(f"  tx_power_dbm: {layer.tx_power_dbm}")
        print(f"  noise_floor_dbm: {layer.noise_floor_dbm}")
        print(f"  Target power (center pixel): {p_target[128, 128]:.2f} dBm")


if __name__ == "__main__":
    main()
