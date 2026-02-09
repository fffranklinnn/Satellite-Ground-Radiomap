"""
L3 Urban/Micro Layer for SG-MRM project.

This layer handles:
- Building distribution from shapefiles
- Shadow and obstruction calculation
- Ray tracing for multipath effects (V2.0)

Coverage: 256 m, Resolution: 1 m/pixel
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np

from .base import BaseLayer


class L3UrbanLayer(BaseLayer):
    """
    L3 Urban/Micro Layer: Building-scale propagation effects.

    This layer computes electromagnetic loss due to urban structures,
    including buildings, shadows, reflections, and multipath.

    Coverage: 256 m × 256 m
    Resolution: 1 m/pixel
    Grid: 256 × 256 pixels
    """

    def __init__(self, config: Dict[str, Any], origin_lat: float, origin_lon: float):
        """
        Initialize L3 Urban Layer.

        Args:
            config: Layer configuration containing:
                - grid_size: 256
                - coverage_km: 0.256
                - resolution_m: 1.0
                - building_shapefile: Path to building shapefile
                - satellite_azimuth_deg: Satellite azimuth angle
                - satellite_elevation_deg: Satellite elevation angle
            origin_lat: Origin latitude
            origin_lon: Origin longitude
        """
        super().__init__(config, origin_lat, origin_lon)

        # Extract L3-specific parameters
        self.building_shapefile = config.get('building_shapefile', None)
        self.satellite_azimuth_deg = config.get('satellite_azimuth_deg', 180.0)
        self.satellite_elevation_deg = config.get('satellite_elevation_deg', 45.0)

        # Building data
        self.buildings = []  # List of building polygons with heights

    def compute(self, timestamp: datetime) -> np.ndarray:
        """
        Compute L3 urban layer loss map.

        For V1.0: Basic building shadow calculation
        For V2.0: Full ray tracing with reflections and multipath

        Args:
            timestamp: Simulation timestamp

        Returns:
            256×256 array of loss values in dB
        """
        loss_map = np.zeros((self.grid_size, self.grid_size))

        # TODO V1.0: Load building data from shapefile
        # TODO V2.0: Implement GPU-accelerated ray tracing

        if not self.buildings:
            # No building data: return zero loss
            return loss_map

        # Calculate shadow map based on satellite position
        shadow_map = self._calculate_shadows()

        # Apply shadow loss
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if shadow_map[i, j]:
                    # In shadow: high loss
                    loss_map[i, j] = 50.0  # dB (complete obstruction)
                else:
                    # Line of sight: minimal loss
                    loss_map[i, j] = 0.0

        # TODO V2.0: Add multipath effects
        # multipath_loss = self._calculate_multipath(i, j)
        # loss_map[i, j] += multipath_loss

        return loss_map

    def _load_buildings(self):
        """
        Load building data from shapefile.

        TODO V1.0: Implement shapefile loading
        Expected format: Polygon geometries with height attribute

        Each building should have:
        - Polygon coordinates (lat/lon or local coordinates)
        - Height in meters
        - Optional: Material properties for V2.0
        """
        if self.building_shapefile is None:
            return

        # Placeholder for building loading
        # import geopandas as gpd
        # buildings_gdf = gpd.read_file(self.building_shapefile)
        # self.buildings = self._convert_buildings(buildings_gdf)
        pass

    def _calculate_shadows(self) -> np.ndarray:
        """
        Calculate shadow map based on building positions and sun/satellite angle.

        Returns:
            256×256 boolean array (True = in shadow, False = line of sight)
        """
        shadow_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # TODO V1.0: Implement shadow casting algorithm
        # For each building:
        #   1. Project building footprint onto ground
        #   2. Calculate shadow direction from satellite azimuth
        #   3. Extend shadow based on building height and elevation angle
        #   4. Mark shadowed pixels

        for building in self.buildings:
            building_shadow = self._cast_building_shadow(building)
            shadow_map = np.logical_or(shadow_map, building_shadow)

        return shadow_map

    def _cast_building_shadow(self, building: Dict[str, Any]) -> np.ndarray:
        """
        Cast shadow for a single building.

        Args:
            building: Building dictionary with 'polygon' and 'height'

        Returns:
            256×256 boolean array of shadow for this building
        """
        shadow = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # TODO V1.0: Implement shadow casting
        # Calculate shadow length: L = H / tan(elevation)
        # Project shadow in azimuth direction

        return shadow

    def _calculate_multipath(self, i: int, j: int) -> float:
        """
        Calculate multipath effects at pixel (i, j).

        TODO V2.0: Implement ray tracing for multipath
        This requires:
        - Ray launching from satellite
        - Reflection from building surfaces
        - Path loss calculation for each ray
        - Coherent combination of multipath components

        Args:
            i: Row index
            j: Column index

        Returns:
            Multipath loss/gain in dB
        """
        # Placeholder: No multipath effects for V1.0
        return 0.0

    def _trace_ray(self, origin: Tuple[float, float, float],
                   direction: Tuple[float, float, float],
                   max_reflections: int = 3) -> List[Dict[str, Any]]:
        """
        Trace a ray through the urban environment.

        TODO V2.0: Implement GPU-accelerated ray tracing

        Args:
            origin: Ray origin (x, y, z) in meters
            direction: Ray direction (normalized vector)
            max_reflections: Maximum number of reflections to trace

        Returns:
            List of ray path segments with reflection points
        """
        # Placeholder for ray tracing
        return []

    def add_building(self, polygon: List[Tuple[float, float]], height: float,
                    material: str = 'concrete'):
        """
        Add a building to the urban environment.

        Args:
            polygon: List of (lat, lon) coordinates defining building footprint
            height: Building height in meters
            material: Building material (for reflection properties in V2.0)
        """
        building = {
            'polygon': polygon,
            'height': height,
            'material': material
        }
        self.buildings.append(building)
