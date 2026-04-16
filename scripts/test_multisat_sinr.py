#!/usr/bin/env python3
"""
Integration test for multi-satellite SINR calculation.

This script demonstrates the multi-satellite interference feature by:
1. Computing traditional single-satellite loss map
2. Computing multi-satellite SINR map
3. Comparing the results
4. Visualizing the difference
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.l1_macro import L1MacroLayer


def main():
    """Run integration test for multi-satellite SINR."""
    print("=" * 80)
    print("Multi-Satellite SINR Integration Test")
    print("=" * 80)

    # Configuration
    config_path = project_root / "configs" / "mission_config.yaml"
    origin_lat = 34.3416  # Xi'an
    origin_lon = 108.9398
    timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)

    print(f"\nTest Configuration:")
    print(f"  Location: ({origin_lat}°N, {origin_lon}°E)")
    print(f"  Timestamp: {timestamp.isoformat()}")
    print(f"  Config: {config_path.name}")

    # Initialize layer
    print("\n" + "-" * 80)
    print("Initializing L1 Macro Layer...")
    print("-" * 80)
    layer = L1MacroLayer(str(config_path))

    # Test 1: Single-satellite mode (traditional)
    print("\n" + "-" * 80)
    print("Test 1: Single-Satellite Mode (Traditional)")
    print("-" * 80)
    layer.enable_interference = False
    loss_single = layer.compute(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=timestamp
    )

    valid_mask_single = loss_single < layer.NO_COVERAGE_LOSS_DB
    print(f"\nResults:")
    print(f"  Coverage: {valid_mask_single.sum()} / {loss_single.size} pixels")
    print(f"  Loss range: {loss_single[valid_mask_single].min():.1f} ~ {loss_single[valid_mask_single].max():.1f} dB")
    print(f"  Loss mean: {loss_single[valid_mask_single].mean():.1f} dB")

    # Test 2: Multi-satellite mode with interference
    print("\n" + "-" * 80)
    print("Test 2: Multi-Satellite Mode (With Interference)")
    print("-" * 80)
    layer.enable_interference = True
    layer.max_interfering_sats = 10

    sinr_multi, metadata = layer.compute_multisat_sinr(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=timestamp
    )

    valid_mask_multi = ~np.isinf(sinr_multi)
    print(f"\nResults:")
    print(f"  Target satellite: {metadata['target_sat_norad_id']}")
    print(f"  Target elevation: {metadata['target_sat_elevation_deg']:.1f}°")
    print(f"  Interfering satellites: {metadata['num_interfering_sats']}")
    print(f"  Interferer IDs: {', '.join(metadata['interfering_sat_ids'][:5])}" +
          (f" ... (+{len(metadata['interfering_sat_ids'])-5} more)" if len(metadata['interfering_sat_ids']) > 5 else ""))
    print(f"  Coverage: {valid_mask_multi.sum()} / {sinr_multi.size} pixels")
    print(f"  SINR range: {sinr_multi[valid_mask_multi].min():.1f} ~ {sinr_multi[valid_mask_multi].max():.1f} dB")
    print(f"  SINR mean: {sinr_multi[valid_mask_multi].mean():.1f} dB")

    # Test 3: Compare single-sat vs multi-sat
    print("\n" + "-" * 80)
    print("Test 3: Comparison Analysis")
    print("-" * 80)

    # Convert loss to approximate received power
    p_rx_single = layer.tx_power_dbm - loss_single

    # Calculate degradation due to interference
    common_mask = valid_mask_single & valid_mask_multi
    if common_mask.any():
        degradation = p_rx_single[common_mask] - sinr_multi[common_mask]
        print(f"\nInterference Impact:")
        print(f"  Mean degradation: {degradation.mean():.1f} dB")
        print(f"  Max degradation: {degradation.max():.1f} dB")
        print(f"  Min degradation: {degradation.min():.1f} dB")
        print(f"  Std degradation: {degradation.std():.1f} dB")

    # Test 4: Physical consistency checks
    print("\n" + "-" * 80)
    print("Test 4: Physical Consistency Checks")
    print("-" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: SINR should be lower than single-sat scenario (when interference exists)
    checks_total += 1
    if metadata['num_interfering_sats'] > 0:
        if common_mask.any():
            if sinr_multi[common_mask].mean() < p_rx_single[common_mask].mean():
                print("  ✓ SINR < Single-sat power (interference effect verified)")
                checks_passed += 1
            else:
                print("  ✗ SINR >= Single-sat power (unexpected!)")
    else:
        print("  ⊘ No interferers present (check skipped)")

    # Check 2: SINR values should be in reasonable range
    checks_total += 1
    if valid_mask_multi.any():
        sinr_min = sinr_multi[valid_mask_multi].min()
        sinr_max = sinr_multi[valid_mask_multi].max()
        if -20 < sinr_min and sinr_max < 50:
            print(f"  ✓ SINR in reasonable range ({sinr_min:.1f} to {sinr_max:.1f} dB)")
            checks_passed += 1
        else:
            print(f"  ✗ SINR out of range ({sinr_min:.1f} to {sinr_max:.1f} dB)")

    # Check 3: Coverage should be similar
    checks_total += 1
    coverage_diff = abs(valid_mask_single.sum() - valid_mask_multi.sum())
    if coverage_diff < 0.1 * valid_mask_single.sum():
        print(f"  ✓ Coverage consistent ({coverage_diff} pixel difference)")
        checks_passed += 1
    else:
        print(f"  ✗ Coverage mismatch ({coverage_diff} pixel difference)")

    # Check 4: No NaN values
    checks_total += 1
    if not np.isnan(sinr_multi).any():
        print("  ✓ No NaN values in SINR map")
        checks_passed += 1
    else:
        print(f"  ✗ Found {np.isnan(sinr_multi).sum()} NaN values")

    print(f"\nPhysical Consistency: {checks_passed}/{checks_total} checks passed")

    # Test 5: Verify compute() method integration
    print("\n" + "-" * 80)
    print("Test 5: compute() Method Integration")
    print("-" * 80)

    # Test with interference flag
    layer.enable_interference = True
    result_interference = layer.compute(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=timestamp
    )

    # Verify it returns SINR (not loss)
    if np.allclose(result_interference[valid_mask_multi], sinr_multi[valid_mask_multi], rtol=1e-5):
        print("  ✓ compute() correctly returns SINR when interference enabled")
        checks_passed += 1
    else:
        print("  ✗ compute() output mismatch")
    checks_total += 1

    # Test without interference flag
    layer.enable_interference = False
    result_no_interference = layer.compute(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=timestamp
    )

    if np.allclose(result_no_interference[valid_mask_single], loss_single[valid_mask_single], rtol=1e-5):
        print("  ✓ compute() correctly returns loss when interference disabled")
        checks_passed += 1
    else:
        print("  ✗ compute() output mismatch")
    checks_total += 1

    # Final summary
    print("\n" + "=" * 80)
    print("Integration Test Summary")
    print("=" * 80)
    print(f"Total checks: {checks_total}")
    print(f"Passed: {checks_passed}")
    print(f"Failed: {checks_total - checks_passed}")
    print(f"Success rate: {100 * checks_passed / checks_total:.1f}%")

    if checks_passed == checks_total:
        print("\n✓ All tests PASSED! Multi-satellite SINR feature is working correctly.")
        return 0
    else:
        print(f"\n✗ {checks_total - checks_passed} test(s) FAILED. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
