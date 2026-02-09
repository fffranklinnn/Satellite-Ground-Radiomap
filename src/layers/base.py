"""
Base layer abstract class for SG-MRM project.

All physical layers (L1/L2/L3) must inherit from BaseLayer and implement
the compute() method to generate 256×256 loss maps.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
import numpy as np

from ..core.grid import Grid


class BaseLayer(ABC):
    """
    Abstract base class for all physical layers.

    Each layer must implement the compute() method to generate a 256×256
    array of electromagnetic loss values in dB.

    Attributes:
        config (Dict[str, Any]): Layer configuration dictionary
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        grid (Grid): Grid coordinate system for this layer
        grid_size (int): Grid dimension (always 256)
        coverage_km (float): Physical coverage in kilometers
        resolution_m (float): Physical resolution in meters per pixel
    """

    def __init__(self, config: Dict[str, Any], origin_lat: float, origin_lon: float):
        """
        Initialize the base layer.

        Args:
            config: Layer configuration dictionary containing:
                - grid_size: Grid dimension (default 256)
                - coverage_km: Physical coverage in kilometers
                - resolution_m: Physical resolution in meters per pixel
            origin_lat: Origin latitude in degrees
            origin_lon: Origin longitude in degrees
        """
        self.config = config
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon

        # Extract grid parameters
        self.grid_size = config.get('grid_size', 256)
        self.coverage_km = config.get('coverage_km')
        self.resolution_m = config.get('resolution_m')

        # Validate configuration
        self._validate_config()

        # Initialize grid coordinate system
        self.grid = Grid(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            grid_size=self.grid_size,
            coverage_km=self.coverage_km
        )

    def _validate_config(self):
        """Validate that required configuration parameters are present."""
        required_keys = ['grid_size', 'coverage_km', 'resolution_m']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Verify grid consistency
        expected_resolution = (self.coverage_km * 1000) / self.grid_size
        if abs(self.resolution_m - expected_resolution) > 0.1:
            raise ValueError(
                f"Inconsistent grid parameters: "
                f"coverage={self.coverage_km}km, resolution={self.resolution_m}m, "
                f"grid_size={self.grid_size}"
            )

    @abstractmethod
    def compute(self, timestamp: datetime) -> np.ndarray:
        """
        Compute electromagnetic loss for this layer at given timestamp.

        This is the core method that must be implemented by all layers.

        Args:
            timestamp: Simulation timestamp

        Returns:
            256×256 numpy array of loss values in dB

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    def get_layer_info(self) -> Dict[str, Any]:
        """
        Get layer information summary.

        Returns:
            Dictionary containing layer metadata
        """
        return {
            'layer_type': self.__class__.__name__,
            'grid_size': self.grid_size,
            'coverage_km': self.coverage_km,
            'resolution_m': self.resolution_m,
            'origin_lat': self.origin_lat,
            'origin_lon': self.origin_lon
        }

    def __repr__(self) -> str:
        """String representation of the layer."""
        return (f"{self.__class__.__name__}("
                f"coverage={self.coverage_km}km, "
                f"resolution={self.resolution_m}m)")
