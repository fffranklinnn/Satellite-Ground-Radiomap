"""
Radio map aggregation engine for SG-MRM project.

This module combines the outputs from L1/L2/L3 layers to produce
the final composite 256×256 electromagnetic loss map.
"""

from datetime import datetime
from typing import Optional
import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..layers.base import BaseLayer
from ..layers.l1_macro import L1MacroLayer
from ..layers.l2_topo import L2TopoLayer
from ..layers.l3_urban import L3UrbanLayer


class RadioMapAggregator:
    """
    Aggregates multiple physical layers into a composite radio map.

    The aggregator combines L1 (macro), L2 (terrain), and L3 (urban) layers
    using dB domain addition. Layers with different resolutions are interpolated
    to match the target resolution.

    Formula: Result = L1 + Interpolate(L2) + L3

    Attributes:
        l1_layer: L1 Macro/Space layer (256 km coverage)
        l2_layer: L2 Terrain layer (25.6 km coverage)
        l3_layer: L3 Urban layer (256 m coverage)
        target_grid_size: Target output grid size (default 256)
    """

    def __init__(self,
                 l1_layer: Optional[L1MacroLayer] = None,
                 l2_layer: Optional[L2TopoLayer] = None,
                 l3_layer: Optional[L3UrbanLayer] = None,
                 target_grid_size: int = 256):
        """
        Initialize the radio map aggregator.

        Args:
            l1_layer: L1 Macro/Space layer (optional)
            l2_layer: L2 Terrain layer (optional)
            l3_layer: L3 Urban layer (optional)
            target_grid_size: Target output grid size (default 256)
        """
        self.l1_layer = l1_layer
        self.l2_layer = l2_layer
        self.l3_layer = l3_layer
        self.target_grid_size = target_grid_size

        # Validate that at least one layer is provided
        if not any([l1_layer, l2_layer, l3_layer]):
            raise ValueError("At least one layer must be provided")

    def aggregate(self, timestamp: datetime) -> np.ndarray:
        """
        Compute composite radio map by aggregating all layers.

        Args:
            timestamp: Simulation timestamp

        Returns:
            256×256 array of total loss values in dB
        """
        # Initialize composite map
        composite_map = np.zeros((self.target_grid_size, self.target_grid_size))

        # Add L1 contribution (wide area)
        if self.l1_layer is not None:
            l1_loss = self.l1_layer.compute(timestamp)
            l1_interpolated = self._interpolate_to_target(
                l1_loss,
                self.l1_layer.coverage_km,
                self.target_grid_size
            )
            composite_map += l1_interpolated

        # Add L2 contribution (terrain)
        if self.l2_layer is not None:
            l2_loss = self.l2_layer.compute(timestamp)
            l2_interpolated = self._interpolate_to_target(
                l2_loss,
                self.l2_layer.coverage_km,
                self.target_grid_size
            )
            composite_map += l2_interpolated

        # Add L3 contribution (urban)
        if self.l3_layer is not None:
            l3_loss = self.l3_layer.compute(timestamp)
            # L3 is already at target resolution (256m coverage, 1m/pixel)
            composite_map += l3_loss

        return composite_map

    def _interpolate_to_target(self, layer_output: np.ndarray,
                               layer_coverage_km: float,
                               target_size: int) -> np.ndarray:
        """
        Interpolate layer output to target grid size.

        For layers with different coverage areas (L1: 256km, L2: 25.6km),
        we need to extract the relevant region and interpolate to match
        the target resolution.

        Args:
            layer_output: Layer output array (256×256)
            layer_coverage_km: Physical coverage of the layer in km
            target_size: Target grid size (typically 256)

        Returns:
            Interpolated array of size (target_size, target_size)
        """
        input_size = layer_output.shape[0]

        # Create coordinate arrays for input
        x_in = np.linspace(0, layer_coverage_km, input_size)
        y_in = np.linspace(0, layer_coverage_km, input_size)

        # Create interpolation function
        interpolator = RectBivariateSpline(x_in, y_in, layer_output, kx=1, ky=1)

        # For L3 target (256m coverage), extract the center region from larger layers
        # Assume L3 is centered at the origin
        target_coverage_km = 0.256  # L3 coverage

        if layer_coverage_km > target_coverage_km:
            # Extract center region
            center = layer_coverage_km / 2.0
            half_target = target_coverage_km / 2.0
            x_out = np.linspace(center - half_target, center + half_target, target_size)
            y_out = np.linspace(center - half_target, center + half_target, target_size)
        else:
            # Layer is smaller than target, use full extent
            x_out = np.linspace(0, layer_coverage_km, target_size)
            y_out = np.linspace(0, layer_coverage_km, target_size)

        # Interpolate
        interpolated = interpolator(x_out, y_out)

        return interpolated

    def compute_composite_map(self, timestamp: datetime) -> np.ndarray:
        """
        Alias for aggregate() method.

        Args:
            timestamp: Simulation timestamp

        Returns:
            256×256 array of total loss values in dB
        """
        return self.aggregate(timestamp)

    def get_layer_contributions(self, timestamp: datetime) -> dict:
        """
        Get individual layer contributions separately.

        Useful for debugging and visualization.

        Args:
            timestamp: Simulation timestamp

        Returns:
            Dictionary with keys 'l1', 'l2', 'l3', 'composite'
        """
        contributions = {}

        if self.l1_layer is not None:
            l1_loss = self.l1_layer.compute(timestamp)
            contributions['l1'] = self._interpolate_to_target(
                l1_loss,
                self.l1_layer.coverage_km,
                self.target_grid_size
            )

        if self.l2_layer is not None:
            l2_loss = self.l2_layer.compute(timestamp)
            contributions['l2'] = self._interpolate_to_target(
                l2_loss,
                self.l2_layer.coverage_km,
                self.target_grid_size
            )

        if self.l3_layer is not None:
            contributions['l3'] = self.l3_layer.compute(timestamp)

        contributions['composite'] = self.aggregate(timestamp)

        return contributions

    def __repr__(self) -> str:
        """String representation of the aggregator."""
        layers = []
        if self.l1_layer: layers.append('L1')
        if self.l2_layer: layers.append('L2')
        if self.l3_layer: layers.append('L3')
        return f"RadioMapAggregator(layers={'+'.join(layers)})"
