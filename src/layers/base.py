"""
Base interfaces for SG-MRM layers.

All physical layers (L1/L2/L3) must inherit from BaseLayer and implement
the compute() method to generate 256×256 loss maps.

LayerContext carries per-call context (e.g. incident_dir for L3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from ..core.grid import Grid


# ── LayerContext (merged from branch_L3) ──────────────────────────────────────

@dataclass
class LayerContext:
    """
    Context passed to layer compute calls.

    Attributes:
        incident_dir: Incoming signal direction for L3 NLoS calculation.
            Supports ENU vector [e, n, u], or dict with az_deg/el_deg keys.
        extras: Additional key-value pairs for future extensions.
    """

    incident_dir: Optional[Sequence[float]] = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, value: Any) -> "LayerContext":
        """Construct LayerContext from dict, existing instance, or None."""
        if isinstance(value, LayerContext):
            return value
        if value is None:
            return cls()
        if isinstance(value, Mapping):
            payload = dict(value)
            incident_dir = payload.pop("incident_dir", None)
            if incident_dir is None:
                incident_dir = payload.pop("incident_direction", None)
            return cls(incident_dir=incident_dir, extras=payload)
        raise TypeError(f"Unsupported context type: {type(value)!r}")

    def merged_with_kwargs(self, kwargs: Mapping[str, Any]) -> "LayerContext":
        """Return a new LayerContext with kwargs merged in."""
        incident_dir = kwargs.get("incident_dir", self.incident_dir)
        payload = dict(self.extras)
        payload.update(kwargs)
        payload.pop("incident_dir", None)
        return LayerContext(incident_dir=incident_dir, extras=payload)


# ── BaseLayer ─────────────────────────────────────────────────────────────────

class BaseLayer(ABC):
    """
    Abstract base class for all physical layers (L1/L2/L3).

    Each layer must implement compute() to return a 256×256 ndarray of
    electromagnetic loss values in dB.

    Attributes:
        config (Dict[str, Any]): Layer configuration dictionary
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        grid (Grid): Grid coordinate system for this layer
        grid_size (int): Grid dimension (always 256)
        coverage_km (float): Physical coverage in kilometers
        resolution_m (float): Physical resolution in meters per pixel
    """

    GRID_SIZE = 256  # fixed output resolution

    def __init__(self, config: Dict[str, Any], origin_lat: float, origin_lon: float):
        self.config = config
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon

        self.grid_size = config.get('grid_size', 256)
        self.coverage_km = config.get('coverage_km')
        self.resolution_m = config.get('resolution_m')

        self._validate_config()

        self.grid = Grid(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            grid_size=self.grid_size,
            coverage_km=self.coverage_km
        )

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_keys = ['grid_size', 'coverage_km', 'resolution_m']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        expected_resolution = (self.coverage_km * 1000) / self.grid_size
        if abs(self.resolution_m - expected_resolution) > 0.1:
            raise ValueError(
                f"Inconsistent grid parameters: "
                f"coverage={self.coverage_km}km, resolution={self.resolution_m}m, "
                f"grid_size={self.grid_size}"
            )

    @abstractmethod
    def compute(self,
                origin_lat: float,
                origin_lon: float,
                timestamp: Optional[datetime] = None,
                context: Optional[LayerContext] = None,
                **kwargs: Any) -> np.ndarray:
        """
        Compute electromagnetic loss map for this layer.

        Args:
            origin_lat: Origin latitude in degrees
            origin_lon: Origin longitude in degrees
            timestamp:  Simulation timestamp (used by L1 for satellite positioning)
            context:    LayerContext carrying incident_dir etc. (used by L3)
            **kwargs:   Additional layer-specific parameters

        Returns:
            256×256 numpy array of loss values in dB
        """
        pass

    def get_layer_info(self) -> Dict[str, Any]:
        """Return layer metadata summary."""
        return {
            'layer_type': self.__class__.__name__,
            'grid_size': self.grid_size,
            'coverage_km': self.coverage_km,
            'resolution_m': self.resolution_m,
            'origin_lat': self.origin_lat,
            'origin_lon': self.origin_lon,
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"coverage={self.coverage_km}km, "
                f"resolution={self.resolution_m}m)")
