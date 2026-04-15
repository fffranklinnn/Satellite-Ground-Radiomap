"""
Radio map aggregation engine for SG-MRM project.

Combines L1/L2/L3 layer outputs into a composite 256x256 loss map using
dB-domain addition. Layers with different physical coverage are interpolated
to the target product grid before summation.

Formula: composite = L1_interp + L2_interp + L3

The target coverage is determined by CoverageSpec.target_product_grid.
Passing a CoverageSpec is required in strict mode; in non-strict mode the
legacy hardcoded 0.256 km default is used with a DeprecationWarning.
"""

import warnings
from datetime import datetime
from typing import Optional
import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..layers.base import BaseLayer, LayerContext
from ..layers.l1_macro import L1MacroLayer
from ..layers.l2_topo import L2TopoLayer
from ..layers.l3_urban import L3UrbanLayer


class ConfigurationError(ValueError):
    """Raised when RadioMapAggregator is misconfigured in strict mode."""


_LEGACY_TARGET_COVERAGE_KM = 0.256  # kept only for non-strict backward compatibility


class RadioMapAggregator:
    """
    Aggregates L1/L2/L3 physical layers into a composite radio map.

    All layers are called with the unified interface:
        layer.compute(origin_lat, origin_lon, timestamp=ts, context=ctx)

    L1 (256 km) and L2 (25.6 km) outputs are interpolated down to the
    target product coverage before dB-domain summation.

    Args:
        l1_layer, l2_layer, l3_layer: optional layer instances.
        coverage_spec: CoverageSpec that defines the target product grid.
                       Required in strict mode; optional otherwise (legacy default used).
        target_grid_size: output grid size (default 256).
        strict: if True, raise ConfigurationError when coverage_spec is absent.
    """

    def __init__(self,
                 l1_layer: Optional[L1MacroLayer] = None,
                 l2_layer: Optional[L2TopoLayer] = None,
                 l3_layer: Optional[L3UrbanLayer] = None,
                 target_grid_size: int = 256,
                 coverage_spec=None,
                 strict: bool = False):
        self.l1_layer = l1_layer
        self.l2_layer = l2_layer
        self.l3_layer = l3_layer
        self.target_grid_size = target_grid_size
        self.coverage_spec = coverage_spec
        self.strict = strict

        if not any([l1_layer, l2_layer, l3_layer]):
            raise ValueError("At least one layer must be provided.")

        if strict and coverage_spec is None:
            raise ConfigurationError(
                "RadioMapAggregator requires a CoverageSpec in strict mode. "
                "Pass coverage_spec=CoverageSpec(...) or set strict=False."
            )

        if coverage_spec is None:
            warnings.warn(
                "RadioMapAggregator initialized without a CoverageSpec. "
                f"Using legacy default target_coverage_km={_LEGACY_TARGET_COVERAGE_KM}. "
                "Pass a CoverageSpec to remove this warning.",
                DeprecationWarning,
                stacklevel=2,
            )

    def _target_coverage_km(self) -> float:
        """Return the target product coverage in km."""
        if self.coverage_spec is not None:
            g = self.coverage_spec.target_product_grid
            return g.width_m / 1000.0
        return _LEGACY_TARGET_COVERAGE_KM

    def aggregate(self,
                  origin_lat: float,
                  origin_lon: float,
                  timestamp: Optional[datetime] = None,
                  context: Optional[LayerContext] = None) -> np.ndarray:
        contributions = self.get_layer_contributions(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            timestamp=timestamp,
            context=context,
        )
        return contributions['composite']

    def _interpolate_to_target(self,
                               layer_output: np.ndarray,
                               layer_coverage_km: float) -> np.ndarray:
        """
        Interpolate a layer output to the target grid size.

        Extracts the central region matching target_coverage_km from
        larger-coverage layers (L1/L2) and interpolates to target_grid_size pixels.
        """
        input_size = layer_output.shape[0]
        x_in = np.linspace(0, layer_coverage_km, input_size)
        y_in = np.linspace(0, layer_coverage_km, input_size)

        interpolator = RectBivariateSpline(x_in, y_in, layer_output, kx=1, ky=1)

        target_coverage_km = self._target_coverage_km()

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
        contributions = {}
        composite = np.zeros((self.target_grid_size, self.target_grid_size), dtype=np.float64)

        if self.l1_layer is not None:
            l1_loss = self.l1_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            l1_interp = self._interpolate_to_target(
                l1_loss, self.l1_layer.coverage_km)
            contributions['l1'] = l1_interp.astype(np.float32)
            composite += l1_interp

        if self.l2_layer is not None:
            l2_loss = self.l2_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            l2_interp = self._interpolate_to_target(
                l2_loss, self.l2_layer.coverage_km)
            contributions['l2'] = l2_interp.astype(np.float32)
            composite += l2_interp

        if self.l3_layer is not None:
            l3_loss = self.l3_layer.compute(
                origin_lat, origin_lon, timestamp=timestamp, context=context)
            contributions['l3'] = l3_loss.astype(np.float32)
            composite += l3_loss

        contributions['composite'] = composite.astype(np.float32)

        return contributions

    def __repr__(self) -> str:
        layers = []
        if self.l1_layer: layers.append('L1')
        if self.l2_layer: layers.append('L2')
        if self.l3_layer: layers.append('L3')
        cov = f" coverage={self._target_coverage_km():.3f}km"
        return f"RadioMapAggregator(layers={'+'.join(layers)}{cov})"

