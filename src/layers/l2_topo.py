"""
L2 Terrain/Topography Layer for SG-MRM project.
Merged from branch_L2/l2_topo.py.

Responsibilities:
- Window-based GeoTIFF DEM reading via rasterio (avoids loading ~2.6 GB file).
- Bilinear resampling from 30 m to 100 m resolution (256x256 output).
- Vectorised LOS occlusion detection using cumulative-maximum horizon algorithm.
- Knife-edge diffraction loss (V1.0: fixed 20 dB; V2.0: Fresnel-Kirchhoff).
- Output: 256x256 float32 loss matrix (dB).

Interface:
- __init__ accepts EITHER (config: dict, origin_lat, origin_lon) [main.py style]
                       OR (dem_path_str, freq_hz, sat_elevation_deg) [standalone]
- compute(origin_lat, origin_lon, timestamp=None, context=None)

Optimised for Shaanxi Province DEM (31N-39N, 104E-112E, WGS84, 30 m GeoTIFF).
Coverage: 25.6 km, Resolution: 100 m/pixel
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .base import BaseLayer, LayerContext
from ..context.frame_context import FrameContext
from ..context.layer_states import EntryWaveState, TerrainState
from ..context.time_utils import StrictModeError
from ..core.physics import SPEED_OF_LIGHT


class L2TopoLayer(BaseLayer):
    """
    L2 Terrain/Topography Layer: DEM-based terrain obstruction and diffraction.

    Pipeline:
      1. Window-clip: read only the 25.6 km tile from the large GeoTIFF.
      2. Resample: 30 m native -> 100 m/px, output 256x256.
      3. Vectorised LOS occlusion: cumulative-maximum horizon (pure numpy).
      4. Diffraction loss: V1.0 fixed 20 dB for occluded pixels.
    """

    GRID_SIZE           = 256
    RESOLUTION_M        = 100.0
    COVERAGE_M          = GRID_SIZE * RESOLUTION_M   # 25,600 m = 25.6 km
    DEM_LAT_MIN, DEM_LAT_MAX = 15.0, 57.0   # National DEM coverage (china_dem_30m.tif)
    DEM_LON_MIN, DEM_LON_MAX = 73.0, 139.0  # National DEM coverage
    DIFFRACTION_LOSS_DB = 20.0
    MAX_DIFFRACTION_LOSS_DB = 60.0

    def __init__(self,
                 config: Union[Dict[str, Any], str],
                 origin_lat: float = None,
                 origin_lon: float = None,
                 freq_hz: float = None,
                 sat_elevation_deg: float = None,
                 sat_azimuth_deg: float = 180.0):
        """
        Two calling conventions:

        1. main.py style:
               L2TopoLayer(config_dict, origin_lat, origin_lon)
           Config keys: dem_file, frequency_ghz, satellite_elevation_deg,
                        satellite_azimuth_deg

        2. Standalone style (matching branch_L2):
               L2TopoLayer(dem_path_str, freq_hz=2e9, sat_elevation_deg=45.0)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if isinstance(config, str):
            dem_path = config
            self.freq_hz           = float(freq_hz) if freq_hz is not None else 2.0e9
            self.sat_elevation_deg = float(sat_elevation_deg) if sat_elevation_deg is not None else 45.0
            self.sat_azimuth_deg   = float(sat_azimuth_deg)
            self.satellite_altitude_km = 550.0
            self.strict_data       = False
            base_config = {"grid_size": 256, "coverage_km": 25.6, "resolution_m": 100.0}
            origin_lat = origin_lat or 34.0
            origin_lon = origin_lon or 108.0
        else:
            dem_path               = config.get("dem_file")
            self.freq_hz           = float(config.get("frequency_ghz", 2.0)) * 1e9
            self.sat_elevation_deg = float(config.get("satellite_elevation_deg", 45.0))
            self.sat_azimuth_deg   = float(config.get("satellite_azimuth_deg", 180.0))
            self.satellite_altitude_km = float(config.get("satellite_altitude_km", 550.0))
            self.strict_data       = bool(config.get("strict_data", config.get("strict_mode", False)))
            base_config            = config

        super().__init__(base_config, origin_lat, origin_lon)

        self.dem_path = Path(dem_path) if dem_path else None
        self._src = None
        self._src_transform = None

    def _open_dem(self):
        """Lazy-open DEM file and cache the handle."""
        if self._src is not None:
            return
        if self.dem_path is None or not self.dem_path.exists():
            if getattr(self, "strict_data", False):
                raise StrictModeError(
                    f"[L2] strict_data: DEM file not found: {self.dem_path}. "
                    "Set dem_file in config or provide a valid path."
                )
            raise FileNotFoundError(
                f"[L2] DEM file not found: {self.dem_path}. "
                "Set dem_file in config or provide a valid path."
            )
        import rasterio
        self._src = rasterio.open(self.dem_path)
        self._src_transform = self._src.transform
        self.logger.info(
            f"[L2] DEM opened: {self.dem_path.name} | "
            f"size={self._src.width}x{self._src.height} | "
            f"CRS={self._src.crs} | res={self._src.res}"
        )

    def close(self):
        """Release DEM file handle (call after time-series simulation ends)."""
        if self._src is not None:
            self._src.close()
            self._src = None

    def compute(self,
                origin_lat: float = None,
                origin_lon: float = None,
                timestamp=None,
                context: Optional[LayerContext] = None,
                **kwargs) -> np.ndarray:
        """
        Compute L2 terrain loss map (256x256, float32, dB).

        Args:
            origin_lat: South-west corner latitude of the 25.6 km tile.
            origin_lon: South-west corner longitude of the 25.6 km tile.
            timestamp:  Unused in V1.0 (V2.0: receive satellite geometry from L1).
            context:    Unused by L2.

        Returns:
            256x256 float32 array of terrain loss in dB (>= 0).
            Returns zero-loss map when no DEM file is configured.
        """
        if origin_lat is None:
            origin_lat = self.origin_lat
        if origin_lon is None:
            origin_lon = self.origin_lon

        if self.dem_path is None:
            self.logger.warning("[L2] No DEM file configured — returning zero loss.")
            return np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)

        ctx = LayerContext.from_any(context).merged_with_kwargs(kwargs)
        sat_elevation_deg = float(ctx.extras.get("satellite_elevation_deg", self.sat_elevation_deg))
        sat_azimuth_deg = float(ctx.extras.get("satellite_azimuth_deg", self.sat_azimuth_deg))
        sat_slant_range_km = ctx.extras.get("satellite_slant_range_km")
        sat_altitude_km = float(ctx.extras.get("satellite_altitude_km", self.satellite_altitude_km))
        padding_m = max(float(ctx.extras.get("padding_m", ctx.extras.get("l2_padding_m", 0.0))), 0.0)
        if sat_slant_range_km is None:
            sat_slant_range_km = self._estimate_slant_range_km(sat_elevation_deg, sat_altitude_km)
        sat_slant_range_m = max(float(sat_slant_range_km) * 1000.0, 1.0)

        self.logger.info(f"[L2] compute @ ({origin_lat:.4f}N, {origin_lon:.4f}E) | padding={padding_m:.1f} m")
        self._validate_bounds(origin_lat, origin_lon, padding_m=padding_m)

        dem_grid, core_rows, core_cols = self._load_dem_patch_with_padding(origin_lat, origin_lon, padding_m=padding_m)
        occlusion_mask, excess_height_m, obstacle_distance_m = self._compute_occlusion_vectorized(
            dem_grid,
            sat_elevation_deg=sat_elevation_deg,
            sat_azimuth_deg=sat_azimuth_deg,
            return_profile=True,
        )
        loss_db_full = self._apply_diffraction_loss(
            dem=dem_grid,
            mask=occlusion_mask,
            excess_height_m=excess_height_m,
            obstacle_distance_m=obstacle_distance_m,
            sat_slant_range_m=sat_slant_range_m,
        )
        loss_db = loss_db_full[core_rows, core_cols]
        core_occlusion = occlusion_mask[core_rows, core_cols]

        self.logger.info(
            f"[L2] Done | occluded={core_occlusion.mean()*100:.1f}% | "
            f"loss={loss_db.min():.1f}~{loss_db.max():.1f} dB"
        )
        return loss_db

    def _estimate_slant_range_km(self, sat_elevation_deg: float, sat_altitude_km: float) -> float:
        """Estimate slant range from altitude and elevation for fallback use."""
        sin_el = np.sin(np.radians(max(float(sat_elevation_deg), 1.0)))
        return float(sat_altitude_km / max(sin_el, 1e-3))

    def _load_dem_patch(self, origin_lat: float, origin_lon: float) -> np.ndarray:
        """Read only the 25.6 km tile; resample 30 m -> 100 m/px on-the-fly."""
        data, _, _ = self._load_dem_patch_with_padding(origin_lat, origin_lon, padding_m=0.0)
        return data

    def _load_dem_patch_with_padding(self,
                                     origin_lat: float,
                                     origin_lon: float,
                                     padding_m: float = 0.0):
        """
        Read a DEM tile with an optional overlap buffer and return crop slices.

        The returned DEM is sampled at the native L2 resolution (100 m/px). When
        padding is requested, callers should compute occlusion on the full padded
        raster and only emit the central core indexed by the returned slices.
        """
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling

        self._open_dem()

        pad_px = int(np.ceil(max(float(padding_m), 0.0) / self.RESOLUTION_M))
        effective_padding_m = pad_px * self.RESOLUTION_M
        coverage_m = self.COVERAGE_M + 2.0 * effective_padding_m
        out_size = self.GRID_SIZE + 2 * pad_px

        center_lat = origin_lat + 0.5 * (self.COVERAGE_M / 111_000)
        delta_lat = coverage_m / 111_000
        delta_lon = coverage_m / (111_000 * np.cos(np.radians(center_lat)))
        pad_lat = effective_padding_m / 111_000
        pad_lon = effective_padding_m / (111_000 * np.cos(np.radians(center_lat)))

        west = origin_lon - pad_lon
        east = west + delta_lon
        south = origin_lat - pad_lat
        north = south + delta_lat

        window = from_bounds(
            left=west, bottom=south, right=east, top=north,
            transform=self._src_transform
        )
        data = self._src.read(
            1,
            window=window,
            out_shape=(out_size, out_size),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        data = np.flipud(data)   # rasterio: N->S; flip to S->N

        nodata = self._src.nodata
        if nodata is not None:
            data[data == nodata] = 0.0

        core_rows = slice(pad_px, pad_px + self.GRID_SIZE)
        core_cols = slice(pad_px, pad_px + self.GRID_SIZE)
        return data, core_rows, core_cols

    def _compute_occlusion_vectorized(self,
                                      dem: np.ndarray,
                                      sat_elevation_deg: Optional[float] = None,
                                      sat_azimuth_deg: Optional[float] = None,
                                      return_profile: bool = False):
        """
        Vectorised LOS occlusion with arbitrary azimuth directional scan.

        The DEM is indexed as row=S->N, col=W->E. We convert satellite azimuth
        (where the satellite appears in the sky) to incoming-wave scan azimuth:
            scan_az = (satellite_azimuth_deg + 180) % 360
        Then sweep each projected ray from the wavefront side using a per-ray
        cumulative horizon in the transformed coordinate system.
        """
        if dem.ndim != 2:
            raise ValueError("DEM input must be a 2-D array.")

        el_deg = self.sat_elevation_deg if sat_elevation_deg is None else float(sat_elevation_deg)
        az_deg = self.sat_azimuth_deg if sat_azimuth_deg is None else float(sat_azimuth_deg)

        tan_elev = float(np.tan(np.radians(el_deg)))
        if tan_elev <= 0.0:
            return np.zeros_like(dem, dtype=bool)

        # azimuth is where satellite is seen from ground; incoming scan is opposite.
        scan_az_deg = (az_deg + 180.0) % 360.0
        az_rad = np.deg2rad(scan_az_deg)
        ux = float(np.sin(az_rad))  # east component
        uy = float(np.cos(az_rad))  # north component

        rows, cols = dem.shape
        x = (np.arange(cols, dtype=np.float32) + 0.5) * self.RESOLUTION_M
        y = (np.arange(rows, dtype=np.float32) + 0.5) * self.RESOLUTION_M
        xx, yy = np.meshgrid(x, y)

        # Ray-aligned coordinates: u along scan direction, v groups parallel rays.
        u = xx * ux + yy * uy
        v = xx * (-uy) + yy * ux
        ray_id = np.floor(v / self.RESOLUTION_M).astype(np.int32)

        u_flat = u.ravel()
        ray_flat = ray_id.ravel()
        h_flat = dem.ravel().astype(np.float32, copy=False)
        order = np.lexsort((u_flat, ray_flat))

        blocked = np.zeros_like(h_flat, dtype=bool)
        excess_height = np.zeros_like(h_flat, dtype=np.float32)
        obstacle_distance = np.zeros_like(h_flat, dtype=np.float32)
        current_ray = np.int32(-2**31)
        max_virtual = -np.inf
        max_u = 0.0
        eps = 1e-6

        for idx in order:
            rid = ray_flat[idx]
            if rid != current_ray:
                current_ray = rid
                max_virtual = -np.inf
                max_u = 0.0

            ui = float(u_flat[idx])
            virtual_h = float(h_flat[idx]) + tan_elev * ui
            if max_virtual > virtual_h + eps:
                blocked[idx] = True
                excess_height[idx] = float(max_virtual - virtual_h)
                obstacle_distance[idx] = max(float(ui - max_u), self.RESOLUTION_M)

            if virtual_h > max_virtual:
                max_virtual = virtual_h
                max_u = ui

        blocked_grid = blocked.reshape(dem.shape)
        if not return_profile:
            return blocked_grid

        return (
            blocked_grid,
            excess_height.reshape(dem.shape),
            obstacle_distance.reshape(dem.shape),
        )

    def _apply_diffraction_loss(self,
                                dem: np.ndarray,
                                mask: np.ndarray,
                                excess_height_m: Optional[np.ndarray] = None,
                                obstacle_distance_m: Optional[np.ndarray] = None,
                                sat_slant_range_m: Optional[float] = None) -> np.ndarray:
        """
        Knife-edge diffraction loss model.

        When occlusion profile is unavailable, fall back to fixed loss.
        """
        if excess_height_m is None or obstacle_distance_m is None:
            loss = np.zeros(dem.shape, dtype=np.float32)
            loss[mask] = self.DIFFRACTION_LOSS_DB
            return loss

        d1 = np.maximum(obstacle_distance_m.astype(np.float64), self.RESOLUTION_M)
        if sat_slant_range_m is None or sat_slant_range_m <= 0.0:
            d2 = np.full_like(d1, 550_000.0, dtype=np.float64)
        else:
            d2 = np.full_like(d1, float(sat_slant_range_m), dtype=np.float64)

        lam = SPEED_OF_LIGHT / max(self.freq_hz, 1.0)
        h = np.maximum(excess_height_m.astype(np.float64), 0.0)

        # ITU-R knife-edge diffraction parameter:
        # v = h * sqrt( 2*(d1+d2) / (lambda*d1*d2) )
        v = h * np.sqrt((2.0 * (d1 + d2)) / np.maximum(lam * d1 * d2, 1e-12))
        term = np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1
        knife_edge_loss = np.where(
            v <= -0.78,
            0.0,
            6.9 + 20.0 * np.log10(np.maximum(term, 1e-8))
        )
        knife_edge_loss = np.clip(knife_edge_loss, 0.0, self.MAX_DIFFRACTION_LOSS_DB)

        loss = np.zeros(dem.shape, dtype=np.float32)
        loss[mask] = knife_edge_loss[mask].astype(np.float32)
        return loss

    def _validate_bounds(self, origin_lat: float, origin_lon: float, padding_m: float = 0.0):
        center_lat = origin_lat + 0.5 * (self.COVERAGE_M / 111_000)
        pad_lat = max(float(padding_m), 0.0) / 111_000
        pad_lon = max(float(padding_m), 0.0) / (111_000 * np.cos(np.radians(center_lat)))
        delta_lat = self.COVERAGE_M / 111_000
        delta_lon = self.COVERAGE_M / (111_000 * np.cos(np.radians(center_lat)))
        north = origin_lat + delta_lat + pad_lat
        east  = origin_lon + delta_lon + pad_lon
        south = origin_lat - pad_lat
        west = origin_lon - pad_lon
        if (south < self.DEM_LAT_MIN or north > self.DEM_LAT_MAX or
                west < self.DEM_LON_MIN or east  > self.DEM_LON_MAX):
            raise ValueError(
                f"[L2] Requested area ({south:.2f}~{north:.2f}N, "
                f"{west:.2f}~{east:.2f}E) is outside Shaanxi DEM coverage "
                f"({self.DEM_LAT_MIN}~{self.DEM_LAT_MAX}N, "
                f"{self.DEM_LON_MIN}~{self.DEM_LON_MAX}E)."
            )

    # ------------------------------------------------------------------
    # FrameContext-based interface (task10 / AC-3, AC-4)
    # ------------------------------------------------------------------

    def propagate_terrain(self,
                          frame: FrameContext,
                          entry: Optional[EntryWaveState] = None,
                          context: Optional[LayerContext] = None,
                          **kwargs) -> TerrainState:
        """
        Compute L2 terrain propagation for a FrameContext frame.

        The L2 layer uses the SW corner of frame.grid as its origin
        (L2 legacy convention). The GridSpec.sw_corner() method provides
        the correct SW corner from the center-anchored grid.

        Args:
            frame:   FrameContext for this simulation frame.
            entry:   EntryWaveState from L1 (used for sat geometry if available).
            context: Optional LayerContext for extra parameters.

        Returns:
            TerrainState with frame_id == frame.frame_id.
        """
        if frame.grid is None:
            raise ValueError("propagate_terrain requires frame.grid to be set.")

        # Validate entry frame_id consistency
        if entry is not None:
            frame.check_frame_id(entry.frame_id)

        # Warn if caller is injecting geometry via extras alongside a FrameContext
        if context is not None:
            ctx_check = LayerContext.from_any(context)
            _GEOMETRY_EXTRAS = {"satellite_elevation_deg", "satellite_azimuth_deg",
                                 "satellite_altitude_km", "satellite_slant_range_km"}
            if _GEOMETRY_EXTRAS & set(ctx_check.extras.keys()):
                warnings.warn(
                    "propagate_terrain: satellite geometry in LayerContext.extras is deprecated "
                    "when a FrameContext is provided. Geometry is derived from frame and entry state.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Resolve satellite geometry: prefer frame, fall back to entry, then layer defaults
        sat_elevation_deg = frame.sat_elevation_deg
        sat_azimuth_deg = frame.sat_azimuth_deg
        if sat_elevation_deg is None and entry is not None:
            # Use center-pixel elevation from entry state as representative value
            cy, cx = entry.grid.ny // 2, entry.grid.nx // 2
            sat_elevation_deg = float(entry.elevation_deg[cy, cx])
        if sat_azimuth_deg is None and entry is not None:
            cy, cx = entry.grid.ny // 2, entry.grid.nx // 2
            sat_azimuth_deg = float(entry.azimuth_deg[cy, cx])

        # Build context with satellite geometry
        ctx_extras = {}
        if sat_elevation_deg is not None:
            ctx_extras["satellite_elevation_deg"] = sat_elevation_deg
        if sat_azimuth_deg is not None:
            ctx_extras["satellite_azimuth_deg"] = sat_azimuth_deg
        if frame.sat_alt_m is not None:
            ctx_extras["satellite_altitude_km"] = frame.sat_alt_m / 1000.0
        if frame.sat_slant_range_m is not None:
            ctx_extras["satellite_slant_range_km"] = frame.sat_slant_range_m / 1000.0

        if context is not None:
            merged = LayerContext.from_any(context).merged_with_kwargs(ctx_extras)
        else:
            merged = LayerContext(extras=ctx_extras)

        # L2 uses SW corner as origin (legacy convention)
        sw_lat, sw_lon = frame.grid.sw_corner()

        loss_db = self.compute(
            origin_lat=sw_lat,
            origin_lon=sw_lon,
            timestamp=frame.timestamp,
            context=merged,
            **kwargs,
        )

        # Reconstruct occlusion mask: pixels at max diffraction loss are occluded
        occlusion_mask = loss_db >= self.MAX_DIFFRACTION_LOSS_DB

        return TerrainState(
            frame_id=frame.frame_id,
            grid=frame.grid,
            loss_db=loss_db,
            occlusion_mask=occlusion_mask,
        )
