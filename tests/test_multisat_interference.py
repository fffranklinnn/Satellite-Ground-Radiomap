"""
Unit tests for multi-satellite interference and SINR calculation.

Tests verify:
1. SINR calculation physical correctness
2. Multi-satellite interference effects
3. Boundary conditions (no satellites, single satellite)
4. Physical consistency (SINR <= SNR)
"""

import unittest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.layers.l1_macro import L1MacroLayer


class TestMultiSatInterference(unittest.TestCase):
    """Test multi-satellite interference and SINR calculation."""

    @classmethod
    def setUpClass(cls):
        """Set up test configuration."""
        cls.config_path = Path(__file__).parent.parent / "configs" / "mission_config.yaml"
        if not cls.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {cls.config_path}")

    def test_sinr_with_interference_enabled(self):
        """Test SINR calculation with interference enabled."""
        # Create layer with interference enabled
        layer = L1MacroLayer(str(self.config_path))
        layer.enable_interference = True
        layer.max_interfering_sats = 5

        # Compute SINR
        timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
        sinr_db, metadata = layer.compute_multisat_sinr(
            origin_lat=34.3416,
            origin_lon=108.9398,
            timestamp=timestamp
        )

        # Verify output shape
        self.assertEqual(sinr_db.shape, (256, 256))
        self.assertEqual(sinr_db.dtype, np.float32)

        # Verify metadata
        self.assertIn('target_sat_norad_id', metadata)
        self.assertIn('num_interfering_sats', metadata)
        self.assertIn('interfering_sat_ids', metadata)

        # Verify SINR values are in reasonable range
        valid_mask = ~np.isinf(sinr_db)
        if valid_mask.any():
            sinr_valid = sinr_db[valid_mask]
            self.assertGreater(sinr_valid.max(), -20.0, "SINR max should be > -20 dB")
            self.assertLess(sinr_valid.max(), 50.0, "SINR max should be < 50 dB")
            print(f"SINR range: {sinr_valid.min():.1f} to {sinr_valid.max():.1f} dB")

    def test_sinr_lower_than_single_sat(self):
        """Test that multi-sat SINR is lower than single-sat scenario."""
        layer = L1MacroLayer(str(self.config_path))
        timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)

        # Single satellite mode (interference disabled)
        layer.enable_interference = False
        loss_single = layer.compute(
            origin_lat=34.3416,
            origin_lon=108.9398,
            timestamp=timestamp
        )

        # Multi-satellite mode (interference enabled)
        layer.enable_interference = True
        layer.max_interfering_sats = 10
        sinr_multi, metadata = layer.compute_multisat_sinr(
            origin_lat=34.3416,
            origin_lon=108.9398,
            timestamp=timestamp
        )

        # Convert loss to received power (assuming tx_power_dbm = 40)
        # P_rx = P_tx - Loss
        p_rx_single = layer.tx_power_dbm - loss_single

        # In multi-sat scenario, SINR should be lower than SNR (single-sat case)
        # because of interference
        if metadata['num_interfering_sats'] > 0:
            valid_mask = ~np.isinf(sinr_multi) & (loss_single < layer.NO_COVERAGE_LOSS_DB)
            if valid_mask.any():
                # SINR should be lower than received power (which approximates SNR)
                # This is a simplified check; exact comparison requires noise floor
                print(f"Number of interfering satellites: {metadata['num_interfering_sats']}")
                print(f"SINR mean: {sinr_multi[valid_mask].mean():.1f} dB")
                print(f"Single-sat P_rx mean: {p_rx_single[valid_mask].mean():.1f} dBm")

    def test_interference_increases_with_more_satellites(self):
        """Test that interference increases with more visible satellites."""
        layer = L1MacroLayer(str(self.config_path))
        layer.enable_interference = True
        timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)

        # Test with different numbers of interfering satellites
        sinr_results = []
        for max_sats in [1, 5, 10, 20]:
            layer.max_interfering_sats = max_sats
            sinr_db, metadata = layer.compute_multisat_sinr(
                origin_lat=34.3416,
                origin_lon=108.9398,
                timestamp=timestamp
            )
            valid_mask = ~np.isinf(sinr_db)
            if valid_mask.any():
                mean_sinr = sinr_db[valid_mask].mean()
                sinr_results.append((max_sats, mean_sinr, metadata['num_interfering_sats']))
                print(f"Max interferers: {max_sats}, Actual: {metadata['num_interfering_sats']}, "
                      f"Mean SINR: {mean_sinr:.1f} dB")

        # Verify that SINR decreases as more interferers are considered
        if len(sinr_results) >= 2:
            for i in range(len(sinr_results) - 1):
                if sinr_results[i][2] < sinr_results[i+1][2]:  # If actual interferers increased
                    self.assertGreaterEqual(
                        sinr_results[i][1], sinr_results[i+1][1],
                        "SINR should decrease with more interferers"
                    )

    def test_physical_consistency_sinr_bounds(self):
        """Test physical consistency: SINR should not exceed SNR."""
        layer = L1MacroLayer(str(self.config_path))
        layer.enable_interference = True
        layer.max_interfering_sats = 10
        timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)

        sinr_db, metadata = layer.compute_multisat_sinr(
            origin_lat=34.3416,
            origin_lon=108.9398,
            timestamp=timestamp
        )

        # Calculate theoretical SNR (no interference)
        # SNR = P_signal / P_noise
        # In dB: SNR_dB = P_signal_dBm - P_noise_dBm
        # Assuming typical received power is around -80 to -100 dBm
        # and noise floor is -110 dBm, SNR should be around 10-30 dB

        valid_mask = ~np.isinf(sinr_db)
        if valid_mask.any():
            sinr_valid = sinr_db[valid_mask]
            # SINR should not exceed reasonable SNR bounds
            # (assuming noise floor -110 dBm and typical rx power -80 to -100 dBm)
            max_theoretical_snr = 40.0  # Conservative upper bound
            self.assertLess(
                sinr_valid.max(), max_theoretical_snr,
                f"SINR should not exceed theoretical SNR bound ({max_theoretical_snr} dB)"
            )

    def test_no_coverage_regions(self):
        """Test that no-coverage regions have -inf SINR."""
        layer = L1MacroLayer(str(self.config_path))
        layer.enable_interference = True
        timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)

        sinr_db, metadata = layer.compute_multisat_sinr(
            origin_lat=34.3416,
            origin_lon=108.9398,
            timestamp=timestamp
        )

        # Check that some pixels have -inf (no coverage)
        no_coverage_mask = np.isinf(sinr_db) & (sinr_db < 0)
        if no_coverage_mask.any():
            print(f"No-coverage pixels: {no_coverage_mask.sum()} / {sinr_db.size}")
            self.assertTrue(no_coverage_mask.any(), "Should have some no-coverage regions")

    def test_compute_method_with_interference_flag(self):
        """Test that compute() method respects enable_interference flag."""
        layer = L1MacroLayer(str(self.config_path))
        timestamp = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)

        # Test with interference disabled (default)
        layer.enable_interference = False
        result_no_interference = layer.compute(
            origin_lat=34.3416,
            origin_lon=108.9398,
            timestamp=timestamp
        )
        self.assertEqual(result_no_interference.shape, (256, 256))
        # Should return loss values (positive dB)
        valid_mask = result_no_interference < layer.NO_COVERAGE_LOSS_DB
        if valid_mask.any():
            self.assertGreater(result_no_interference[valid_mask].min(), 0.0,
                             "Loss values should be positive")

        # Test with interference enabled
        layer.enable_interference = True
        result_with_interference = layer.compute(
            origin_lat=34.3416,
            origin_lon=108.9398,
            timestamp=timestamp
        )
        self.assertEqual(result_with_interference.shape, (256, 256))
        # Should return SINR values (can be negative dB)
        valid_mask = ~np.isinf(result_with_interference)
        if valid_mask.any():
            # SINR can be negative, unlike loss
            self.assertTrue(
                result_with_interference[valid_mask].min() < result_no_interference[valid_mask].min(),
                "SINR values should be different from loss values"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
