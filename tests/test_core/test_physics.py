"""Unit tests for core.physics module."""

import pytest
import numpy as np
from src.core.physics import (
    free_space_path_loss,
    atmospheric_loss,
    ionospheric_loss,
    polarization_loss,
    db_to_linear,
    linear_to_db,
    combine_losses_db
)


def test_free_space_path_loss():
    """Test FSPL calculation."""
    # Test basic FSPL
    fspl = free_space_path_loss(100.0, 10.0)
    assert fspl > 0
    assert isinstance(fspl, float)

    # FSPL should increase with distance
    fspl1 = free_space_path_loss(100.0, 10.0)
    fspl2 = free_space_path_loss(200.0, 10.0)
    assert fspl2 > fspl1

    # FSPL should increase with frequency
    fspl1 = free_space_path_loss(100.0, 10.0)
    fspl2 = free_space_path_loss(100.0, 20.0)
    assert fspl2 > fspl1


def test_atmospheric_loss():
    """Test atmospheric loss calculation."""
    # Test at different elevation angles
    loss_low = atmospheric_loss(10.0, 10.0)
    loss_high = atmospheric_loss(90.0, 10.0)

    # Loss should be higher at low elevation (longer path)
    assert loss_low > loss_high

    # Test with rain
    loss_clear = atmospheric_loss(45.0, 10.0, rain_rate_mm_h=0.0)
    loss_rain = atmospheric_loss(45.0, 10.0, rain_rate_mm_h=10.0)
    assert loss_rain > loss_clear


def test_ionospheric_loss():
    """Test ionospheric loss calculation."""
    # Test at different frequencies
    loss_low_freq = ionospheric_loss(1.0, tec=10.0)
    loss_high_freq = ionospheric_loss(10.0, tec=10.0)

    # Ionospheric effects decrease with frequency
    assert loss_low_freq > loss_high_freq

    # Above 3 GHz should be negligible
    loss = ionospheric_loss(5.0, tec=10.0)
    assert loss == 0.0


def test_polarization_loss():
    """Test polarization loss calculation."""
    # Perfect match
    assert polarization_loss('H', 'H') == 0.0
    assert polarization_loss('RHCP', 'RHCP') == 0.0

    # Orthogonal polarizations
    loss_hv = polarization_loss('H', 'V')
    assert loss_hv > 10.0  # Should be significant

    # Linear to circular: 3 dB loss
    loss_lc = polarization_loss('H', 'RHCP')
    assert loss_lc == 3.0


def test_db_linear_conversion():
    """Test dB to linear conversion."""
    # Test conversion
    linear = db_to_linear(10.0)
    assert abs(linear - 10.0) < 0.01

    db = linear_to_db(10.0)
    assert abs(db - 10.0) < 0.01

    # Test round-trip
    original = 20.0
    converted = linear_to_db(db_to_linear(original))
    assert abs(converted - original) < 0.01


def test_combine_losses_db():
    """Test combining losses in dB domain."""
    loss1 = 10.0
    loss2 = 20.0
    loss3 = 5.0

    total = combine_losses_db(loss1, loss2, loss3)
    assert total == 35.0
