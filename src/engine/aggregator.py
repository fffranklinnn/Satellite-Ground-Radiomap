"""
Radio map aggregation engine for SG-MRM project.

Combines L1/L2/L3 layer outputs into a composite 256x256 loss map using
dB-domain addition. Layers with different physical coverage are interpolated
to the L3 target resolution before summation.

Formula: composite = L1_interp + L2_interp + L3
"""

from datetime import datetime
from typing import Optional
import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..layers.base import BaseLayer, LayerContext
from ..layers.l1_macro import L1MacroLayer
from ..layers.l2_topo import L2TopoLayer
from ..layers.l3_urban import L3UrbanLayer


class RadioMapAggregator:
    """
    Aggregates L1/L2/L3 physical layers into a composite radio map.

    All layers are called with the unified interface:
        layer.compute(origin_lat, origin_lon, timestamp=ts, context=ctx)

    L1 (256 km) and L2 (25.6 km) outputs are interpolated down to the L3
    target coverage (256 m) before dB-domain summation.
    """

    def __init__(self,
                 l1_layer: Optional[L1MacroLayer] = None,
                 l2_layer: Optional[L2TopoLayer] = None,
                 l3_layer: Optional[L3UrbanLayer] = None,
                 target_grid_size: int = 256):
        self.l1_layer = l1_layer
        self.l2_layer = l2_layer
        self.l3_layer = l3_layer
        self.target_grid_size = target_grid_size

        if not any([l1_layer, l2_layer, l3_layer]):
            raise ValueError("At least one layer must be provided.")

    def aggregate(self,
                  origin_lat: float,
                  origin_lon: float,
                  timestamp: Optional[datetime] = None,
                  context: Optional[LayerContext] = None) -> np.ndarray:
        """
        Compute composite radio map by aggregating all enabled layers.

        Args:
            origin_lat: Origin latitude in degrees
            origin_lon: Origin longitude in degrees
            timestamp:  Simulation timestamp (forwarded to L1)
            context:    LayerContext with incident_dir etc. (forwarded to L3)

        Returns:
            256x256 float32 array of total loss values in dB
        """
        composite = np.zeros((self.target_grid_size, self.target_grid_size),
                             dtype=np.float64)

        if self.l1_layer is not None:
            l1_loss = self.l1_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            composite += self._interpolate_to_target(
                l1_loss, self.l1_layer.coverage_km)

        if self.l2_layer is not None:
            l2_loss = self.l2_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            composite += self._interpolate_to_target(
                l2_loss, self.l2_layer.coverage_km)

        if self.l3_layer is not None:
            l3_loss = self.l3_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            composite += l3_loss

        return composite.astype(np.float32)

    def _interpolate_to_target(self,
                               layer_output: np.ndarray,
                               layer_coverage_km: float) -> np.ndarray:
        """
        Interpolate a layer output to the target grid size.

        Extracts the central 256 m region from larger-coverage layers (L1/L2)
        and interpolates to target_grid_size pixels.
        """
        input_size = layer_output.shape[0]
        x_in = np.linspace(0, layer_coverage_km, input_size)
        y_in = np.linspace(0, layer_coverage_km, input_size)

        interpolator = RectBivariateSpline(x_in, y_in, layer_output, kx=1, ky=1)

        target_coverage_km = 0.256   # L3 tile coverage

        if layer_coverage_km > target_coverage_km:
            center = layer_coverage_km / 2.0
            half   = target_coverage_km / 2.0
            x_out  = np.linspace(center - half, center + half, self.target_grid_size)
            y_out  = np.linspace(center - half, center + half, self.target_grid_size)
        else:
            x_out = np.linspace(0, layer_coverage_km, self.target_grid_size)
            y_out = np.linspace(0, layer_coverage_km, self.target_grid_size)

        return interpolator(x_out, y_out)

    def compute_composite_map(self,
                              origin_lat: float,
                              origin_lon: float,
                              timestamp: Optional[datetime] = None,
                              context: Optional[LayerContext] = None) -> np.ndarray:
        """Alias for aggregate()."""
        return self.aggregate(origin_lat, origin_lon, timestamp, context)

    def get_layer_contributions(self,
                                origin_lat: float,
                                origin_lon: float,
                                timestamp: Optional[datetime] = None,
                                context: Optional[LayerContext] = None) -> dict:
        """
        Return individual layer contributions plus the composite map.

        Returns:
            Dict with keys 'l1', 'l2', 'l3', 'composite' (whichever are enabled).
        """
        contributions = {}

        if self.l1_layer is not None:
            l1_loss = self.l1_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            contributions['l1'] = self._interpolate_to_target(
                l1_loss, self.l1_layer.coverage_km)

        if self.l2_layer is not None:
            l2_loss = self.l2_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            contributions['l2'] = self._interpolate_to_target(
                l2_loss, self.l2_layer.coverage_km)

        if self.l3_layer is not None:
            contributions['l3'] = self.l3_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)

        contributions['composite'] = self.aggregate(
            origin_lat, origin_lon, timestamp, context)

        return contributions

    def __repr__(self) -> str:
        layers = []
        if self.l1_layer: layers.append('L1')
        if self.l2_layer: layers.append('L2')
        if self.l3_layer: layers.append('L3')
        return f"RadioMapAggregator(layers={'+'.join(layers)})"
