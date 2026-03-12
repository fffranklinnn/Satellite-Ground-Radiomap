"""
L1 Macro/Space Layer for SG-MRM project.
Merged from branch_L1/SG-MRM-L1/src/layers/L1_macro.py (v1.1).

Responsibilities:
- Parse TLE file and select the best visible satellite (highest elevation,
  valid orbit altitude) using Skyfield.
- Build a 256x256 geographic grid centred on origin.
- Compute per-pixel azimuth / elevation / slant-range.
- Apply 2-D Gaussian beam antenna gain (31x31 phased array, Ku-band 14.5 GHz).
- Compute Free Space Path Loss and polarization loss.
- Output: 256x256 float32 net-loss matrix (dB).

Interface:
- __init__ accepts EITHER (config: dict, origin_lat, origin_lon) [main.py style]
                       OR (config_path: str)                      [standalone style]
- compute(origin_lat, origin_lon, timestamp=None, context=None)

Physics reference: ITU-R P.618; Remote Sensing paper (You Fu et al., 2026).
Coverage: 256 km, Resolution: 1000 m/pixel
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from skyfield.api import load, EarthSatellite, wgs84

from .base import BaseLayer, LayerContext
from ..core.grid import (
    get_grid_latlon,
    get_azimuth_elevation,
    L1_COVERAGE,
    GRID_SIZE,
    EARTH_RADIUS_M,
)
from ..core.physics import (
    fspl_db,
    polarization_loss_db,
    gaussian_beam_gain_db,
    parabolic_rolloff_gain_db,
    phased_array_peak_gain_db,
    phased_array_hpbw_deg,
    SPEED_OF_LIGHT,
    atmospheric_loss,
    atmospheric_loss_era5,
    ionospheric_loss,
)
from ..utils.ionex_loader import IonexLoader
from ..utils.era5_loader import load_era5
from ..utils.ionosphere import (
    ipp_from_ground,
    faraday_rotation_deg,
    polarization_mismatch_loss_db,
)


# ── TLE helpers ───────────────────────────────────────────────────────────────

def _parse_tle_file(tle_file_path: str) -> Tuple[List[EarthSatellite], List[Tuple[str, str]]]:
    """Parse a TLE file and return (satellites, tle_groups)."""
    tle_file = Path(tle_file_path)
    if not tle_file.exists():
        raise FileNotFoundError(f"TLE file not found: {tle_file_path}")

    with open(tle_file, "r", encoding="utf-8") as f:
        tle_lines = [line.strip() for line in f if line.strip()]

    satellites: List[EarthSatellite] = []
    valid_tle_groups: List[Tuple[str, str]] = []

    i = 0
    while i < len(tle_lines):
        if (tle_lines[i].startswith("1") and
                i + 1 < len(tle_lines) and
                tle_lines[i + 1].startswith("2")):
            line1, line2 = tle_lines[i], tle_lines[i + 1]
            valid_tle_groups.append((line1, line2))
            satellites.append(EarthSatellite(line1, line2))
            i += 2
        else:
            i += 1

    print(f"[L1] Parsed {len(satellites)} TLE entries from {tle_file.name}")
    return satellites, valid_tle_groups


def _get_norad_id(tle_group: Tuple[str, str]) -> str:
    return tle_group[1].split()[1]


def _get_sat_altitude_km(sat: EarthSatellite, t) -> float:
    """Compute orbital altitude (km) from geocentric position vector."""
    pos_km = sat.at(t).position.km
    return float(np.linalg.norm(pos_km)) - 6371.0


# ── L1MacroLayer ──────────────────────────────────────────────────────────────

class L1MacroLayer(BaseLayer):
    """
    L1 Macro/Space Layer: satellite-to-ground propagation loss.

    Output matrix semantics:
        net_loss_db[i,j] = FSPL[i,j] + Pol_Loss - Antenna_Gain[i,j]
    Higher value means greater electromagnetic loss at that pixel.
    """

    # Class-level constants (Ku-band, 31x31 array)
    FREQ_HZ_DEFAULT     = 14.5e9
    ARRAY_ROWS          = 31
    ARRAY_COLS          = 31
    ELEMENT_GAIN_DBI    = 5.0
    MIN_ELEVATION_DEG   = 5.0
    NO_COVERAGE_LOSS_DB = 200.0
    MIN_ORBIT_ALT_KM    = 200.0
    MAX_ORBIT_ALT_KM    = 40_000.0

    def __init__(self,
                 config: Union[Dict[str, Any], str],
                 origin_lat: float = None,
                 origin_lon: float = None):
        """
        Two calling conventions:
          1. main.py style:   L1MacroLayer(config_dict, origin_lat, origin_lon)
          2. Standalone style: L1MacroLayer("configs/mission_config.yaml")
        """
        if isinstance(config, str):
            cfg_path = Path(config)
            raw_cfg = self._load_yaml(cfg_path)
            if isinstance(raw_cfg.get("layers"), dict) and isinstance(raw_cfg["layers"].get("l1_macro"), dict):
                self.cfg = raw_cfg["layers"]["l1_macro"]
                origin_cfg = raw_cfg.get("origin", {})
            else:
                self.cfg = raw_cfg
                origin_cfg = self.cfg.get("origin", {})

            if origin_lat is None:
                origin_lat = float(origin_cfg.get("latitude", origin_cfg.get("lat", 34.3416)))
            if origin_lon is None:
                origin_lon = float(origin_cfg.get("longitude", origin_cfg.get("lon", 108.9398)))

            base_config = {
                "grid_size": int(self.cfg.get("grid_size", 256)),
                "coverage_km": float(self.cfg.get("coverage_km", 256.0)),
                "resolution_m": float(self.cfg.get("resolution_m", 1000.0)),
            }
        else:
            self.cfg = config
            base_config = {
                "grid_size": int(self.cfg.get("grid_size", 256)),
                "coverage_km": float(self.cfg.get("coverage_km", 256.0)),
                "resolution_m": float(self.cfg.get("resolution_m", 1000.0)),
            }

        super().__init__(base_config, origin_lat, origin_lon)

        # RF / antenna parameters
        freq_cfg = self.cfg.get("frequency", None)
        if isinstance(freq_cfg, dict) and freq_cfg.get("center_hz") is not None:
            self.freq_hz = float(freq_cfg["center_hz"])
        elif freq_cfg is not None and not isinstance(freq_cfg, dict):
            # Allow scalar config["frequency"] in GHz for backward compatibility.
            self.freq_hz = float(freq_cfg) * 1e9
        else:
            self.freq_hz = float(self.cfg.get("frequency_ghz", self.FREQ_HZ_DEFAULT / 1e9)) * 1e9

        self.frequency_ghz = self.freq_hz / 1e9
        self.wavelength_m = SPEED_OF_LIGHT / self.freq_hz
        self.element_spacing_m = 0.5 * self.wavelength_m

        n_elements = self.ARRAY_ROWS * self.ARRAY_COLS
        self.peak_gain_db = phased_array_peak_gain_db(n_elements, self.ELEMENT_GAIN_DBI)
        self.hpbw_az_deg, self.hpbw_el_deg = phased_array_hpbw_deg(
            self.ARRAY_ROWS, self.ARRAY_COLS,
            self.wavelength_m, self.element_spacing_m
        )

        antenna_cfg = self.cfg.get("antenna_pattern", {}) or {}
        self.antenna_pattern_model = str(
            antenna_cfg.get("model", "parabolic_rolloff")
        ).lower().strip()
        theta_3db_default = max(0.5 * (self.hpbw_az_deg + self.hpbw_el_deg) * 3.0, 0.1)
        self.theta_3db_deg = float(antenna_cfg.get("theta_3db_deg", theta_3db_default))
        max_rolloff_db = antenna_cfg.get("max_rolloff_db", 30.0)
        self.antenna_min_gain_db = None
        if max_rolloff_db is not None:
            self.antenna_min_gain_db = float(self.peak_gain_db - float(max_rolloff_db))

        pol_cfg = self.cfg.get("polarization", {})
        self.pol_mismatch_deg = float(pol_cfg.get("mismatch_angle_deg", 0.0))
        self.pol_loss_db = polarization_loss_db(self.pol_mismatch_deg)
        self.pol_mode = str(pol_cfg.get("mode", "linear")).lower().strip()

        iono_cfg = self.cfg.get("ionosphere", {})
        self.use_ipp = bool(iono_cfg.get("use_ipp", True))
        self.use_slant_tec = bool(iono_cfg.get("use_slant_tec", True))
        self.iono_shell_height_km = float(iono_cfg.get("shell_height_km", 350.0))
        self.max_mapping_factor = float(iono_cfg.get("max_mapping_factor", 8.0))
        self.enable_faraday = bool(iono_cfg.get("enable_faraday", False))
        self.faraday_linear_only = bool(iono_cfg.get("faraday_linear_only", True))
        self.fallback_b_t = float(iono_cfg.get("fallback_b_t", 45e-6))
        self.strict_data = bool(self.cfg.get("strict_data", self.cfg.get("strict_mode", False)))
        self._geomag_backend = "uninitialized"
        self._geomag_model = None

        target_ids = self.cfg.get("target_norad_ids", [])
        if isinstance(target_ids, (str, int, float)):
            target_ids = [target_ids]
        self.target_norad_ids: List[str] = [str(x) for x in target_ids]
        self.rain_rate_mm_h = float(self.cfg.get("rain_rate_mm_h", 0.0))
        self.default_tec = float(self.cfg.get("tec", 10.0))

        # Multi-satellite interference configuration
        interference_cfg = self.cfg.get("interference", {})
        self.enable_interference = bool(interference_cfg.get("enable_interference", False))
        self.tx_power_dbm = float(interference_cfg.get("tx_power_dbm", 40.0))
        self.noise_floor_dbm = float(interference_cfg.get("noise_floor_dbm", -110.0))
        self.max_interfering_sats = int(interference_cfg.get("max_interfering_sats", 20))

        # Resolve TLE file path
        tle_cfg = self.cfg.get("tle", {})
        tle_file = tle_cfg.get("file") if isinstance(tle_cfg, dict) else None
        if tle_file is None:
            tle_file = self.cfg.get("tle_file")
        if tle_file is None:
            raise ValueError("[L1] No TLE file specified (key: tle.file or tle_file).")

        tle_path = self._resolve_data_path(tle_file)
        self.satellites, self.tle_groups = _parse_tle_file(str(tle_path))

        # Optional real-data sources (graceful fallback when missing/unreadable)
        self.ionex = None
        ionex_file = self.cfg.get("ionex_file")
        if ionex_file:
            ionex_path = self._resolve_data_path(ionex_file)
            if ionex_path.exists():
                try:
                    self.ionex = IonexLoader(str(ionex_path))
                    print(f"[L1] IONEX loaded: {ionex_path.name}")
                except Exception as exc:
                    if self.strict_data:
                        raise RuntimeError(f"[L1] strict_data: failed to load IONEX ({ionex_path}): {exc}") from exc
                    print(f"[L1] IONEX unavailable ({exc}); using fallback TEC={self.default_tec:.1f}.")
            else:
                if self.strict_data:
                    raise FileNotFoundError(f"[L1] strict_data: IONEX file not found: {ionex_path}")
                print(f"[L1] IONEX file not found: {ionex_path}; using fallback TEC={self.default_tec:.1f}.")

        self.era5 = None
        era5_file = self.cfg.get("era5_file")
        if era5_file:
            era5_path = self._resolve_data_path(era5_file)
            self.era5 = load_era5(str(era5_path))
            if self.era5 is not None:
                print(f"[L1] ERA5 loaded: {era5_path.name}")
            else:
                if self.strict_data:
                    raise RuntimeError(f"[L1] strict_data: ERA5 unavailable or unreadable: {era5_path}")
                print("[L1] ERA5 unavailable; using simplified atmospheric model.")

        self.ts = load.timescale()
        self._sim_time = None

        self._print_antenna_params()

    @staticmethod
    def _load_yaml(cfg_path: Path) -> Dict[str, Any]:
        if not cfg_path.exists():
            print(f"[L1] Config not found at {cfg_path}, using defaults.")
            return {}
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _resolve_data_path(path_value: Union[str, Path]) -> Path:
        """Resolve a data path against project root when given a relative path."""
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate

        root = Path(__file__).resolve().parents[2]
        rooted = root / candidate
        if rooted.exists():
            return rooted
        return candidate

    def _resolve_sim_datetime(self, timestamp: Optional[datetime]) -> datetime:
        """Resolve simulation datetime in UTC and sync Skyfield time state."""
        if timestamp is not None:
            dt_utc = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
            self._sim_time = self.ts.from_datetime(dt_utc)
            return dt_utc

        if self._sim_time is not None:
            dt = self._sim_time.utc_datetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt

        dt_now = datetime.now(timezone.utc)
        self._sim_time = self.ts.from_datetime(dt_now)
        print("[L1] No simulation time set — using current UTC.")
        return dt_now

    def _print_antenna_params(self):
        print(f"[L1] {self.ARRAY_ROWS}x{self.ARRAY_COLS} array | "
              f"{self.freq_hz/1e9:.2f} GHz | "
              f"peak {self.peak_gain_db:.2f} dBi | "
              f"HPBW az={self.hpbw_az_deg:.2f}deg el={self.hpbw_el_deg:.2f}deg")

    def set_time(self, dt: datetime):
        """Set simulation time."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        self._sim_time = self.ts.from_datetime(dt)

    @staticmethod
    def _angular_difference_deg(a_deg: np.ndarray, b_deg: float) -> np.ndarray:
        return ((np.asarray(a_deg, dtype=np.float64) - float(b_deg) + 180.0) % 360.0) - 180.0

    def _compute_offaxis_angle_deg(self,
                                   sat_x_m: float,
                                   sat_y_m: float,
                                   sat_alt_m: float,
                                   x_m: np.ndarray,
                                   y_m: np.ndarray) -> np.ndarray:
        """Compute satellite-beam off-axis angle matrix relative to city-center boresight."""
        vx0 = -float(sat_x_m)
        vy0 = -float(sat_y_m)
        vz0 = -float(sat_alt_m)
        norm0 = max(float(np.sqrt(vx0 * vx0 + vy0 * vy0 + vz0 * vz0)), 1e-6)

        vx = x_m - float(sat_x_m)
        vy = y_m - float(sat_y_m)
        vz = np.full_like(vx, -float(sat_alt_m), dtype=np.float64)
        norm = np.sqrt(vx * vx + vy * vy + vz * vz)
        dot = vx * vx0 + vy * vy0 + vz * vz0
        cos_theta = np.clip(dot / np.maximum(norm * norm0, 1e-6), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta)).astype(np.float32)

    def _compute_antenna_gain_db(self,
                                 sat_info: Dict[str, float],
                                 azimuth_deg: np.ndarray,
                                 elevation_deg: np.ndarray,
                                 offaxis_theta_deg: np.ndarray) -> np.ndarray:
        model = self.antenna_pattern_model
        if model in {"legacy", "legacy_gaussian", "gaussian"}:
            az_to_sat_deg = (azimuth_deg + 180.0) % 360.0
            delta_az = self._angular_difference_deg(az_to_sat_deg, float(sat_info["azimuth_deg"]))
            delta_el = float(sat_info["elevation_deg"]) - elevation_deg
            return gaussian_beam_gain_db(
                theta_az_deg=delta_az,
                theta_el_deg=delta_el,
                peak_gain_db=self.peak_gain_db,
                hpbw_az_deg=self.hpbw_az_deg * 3.0,
                hpbw_el_deg=self.hpbw_el_deg * 3.0,
            )

        return parabolic_rolloff_gain_db(
            theta_deg=offaxis_theta_deg,
            peak_gain_db=self.peak_gain_db,
            theta_3db_deg=self.theta_3db_deg,
            min_gain_db=self.antenna_min_gain_db,
        )

    def compute(self,
                origin_lat: float = None,
                origin_lon: float = None,
                timestamp: Optional[datetime] = None,
                context: Optional[LayerContext] = None,
                **kwargs) -> np.ndarray:
        """
        Compute L1 net-loss matrix or SINR map (256x256, float32, dB).

        If enable_interference=True: returns SINR map (dB)
        If enable_interference=False: returns traditional loss map (dB)

        Pixels with elevation < MIN_ELEVATION_DEG are set to NO_COVERAGE_LOSS_DB or -inf (SINR mode).
        """
        if origin_lat is None:
            origin_lat = self.origin_lat
        if origin_lon is None:
            origin_lon = self.origin_lon

        # Multi-satellite interference mode
        if self.enable_interference:
            sinr_db, metadata = self.compute_multisat_sinr(
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                timestamp=timestamp,
                context=context
            )
            # Store metadata in context if available
            if context is not None and hasattr(context, 'extras'):
                context.extras['sinr_metadata'] = metadata
            return sinr_db

        # Traditional single-satellite loss mode (backward compatible)
        components = self.compute_components(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            timestamp=timestamp,
            context=context,
            **kwargs
        )
        return components["total"]

    def compute_components(self,
                           timestamp: Optional[datetime] = None,
                           origin_lat: float = None,
                           origin_lon: float = None,
                           context: Optional[LayerContext] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Compute L1 component maps (FSPL/ATM/IONO/GAIN/TOTAL).

        Returns:
            Dict with float32 component arrays and metadata.
        """
        if origin_lat is None:
            origin_lat = self.origin_lat
        if origin_lon is None:
            origin_lon = self.origin_lon

        ctx = LayerContext.from_any(context).merged_with_kwargs(kwargs)
        target_ids = ctx.extras.get("target_norad_ids", ctx.extras.get("norad_ids"))
        if isinstance(target_ids, (str, int, float)):
            target_ids = [str(target_ids)]
        elif isinstance(target_ids, (list, tuple, set)):
            target_ids = [str(x) for x in target_ids]
        else:
            target_ids = None

        rain_rate_mm_h = float(ctx.extras.get("rain_rate_mm_h", self.rain_rate_mm_h))

        sim_dt = self._resolve_sim_datetime(timestamp)

        print(f"[L1] Computing @ ({origin_lat:.4f}N, {origin_lon:.4f}E) | {sim_dt.isoformat()}")

        _, sat_info = self._select_best_satellite(origin_lat, origin_lon, target_norad_ids=target_ids)

        lat_grid, lon_grid, x_m, y_m = get_grid_latlon(origin_lat, origin_lon)

        lat_rad = np.deg2rad(origin_lat)
        sat_x_m = ((sat_info["lon_deg"] - origin_lon) *
                   np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(lat_rad))
        sat_y_m = ((sat_info["lat_deg"] - origin_lat) *
                   np.deg2rad(1.0) * EARTH_RADIUS_M)

        azimuth_deg, elevation_deg, slant_range_m = get_azimuth_elevation(
            sat_x_m, sat_y_m, sat_info["alt_m"], x_m, y_m
        )
        offaxis_theta_deg = self._compute_offaxis_angle_deg(
            sat_x_m=sat_x_m,
            sat_y_m=sat_y_m,
            sat_alt_m=float(sat_info["alt_m"]),
            x_m=x_m,
            y_m=y_m,
        )
        gain_db = self._compute_antenna_gain_db(
            sat_info=sat_info,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            offaxis_theta_deg=offaxis_theta_deg,
        )

        fspl = fspl_db(slant_range_m, self.freq_hz)

        epoch_sec = (
            sim_dt.hour * 3600 +
            sim_dt.minute * 60 +
            sim_dt.second +
            sim_dt.microsecond / 1e6
        )
        hour_utc = epoch_sec / 3600.0

        if self.ionex is not None:
            try:
                tec_query_lat = lat_grid
                tec_query_lon = lon_grid
                mapping_factor = np.ones_like(lat_grid, dtype=np.float32)
                if self.use_ipp:
                    # get_azimuth_elevation returns incoming-wave azimuth; convert to
                    # ground->satellite azimuth before IPP projection.
                    az_to_sat = (azimuth_deg + 180.0) % 360.0
                    tec_query_lat, tec_query_lon, mapping_factor = ipp_from_ground(
                        lat_grid,
                        lon_grid,
                        az_to_sat,
                        elevation_deg,
                        shell_height_km=self.iono_shell_height_km,
                        max_mapping_factor=self.max_mapping_factor,
                    )

                tec_vtec_map = self.ionex.get_tec(epoch_sec, tec_query_lat, tec_query_lon)
                if self.use_slant_tec:
                    tec_map = tec_vtec_map * mapping_factor
                else:
                    tec_map = tec_vtec_map
            except Exception as exc:
                if self.strict_data:
                    raise RuntimeError(f"[L1] strict_data: IONEX query failed at {sim_dt.isoformat()}: {exc}") from exc
                print(f"[L1] IONEX query failed ({exc}); fallback TEC={self.default_tec:.1f}.")
                tec_vtec_map = np.full_like(lat_grid, self.default_tec, dtype=np.float32)
                tec_map = np.full_like(lat_grid, self.default_tec, dtype=np.float32)
        else:
            tec_vtec_map = np.full_like(lat_grid, self.default_tec, dtype=np.float32)
            tec_map = np.full_like(lat_grid, self.default_tec, dtype=np.float32)

        iono_db = ionospheric_loss(self.frequency_ghz, tec_map)

        pol_db = np.full_like(lat_grid, self.pol_loss_db, dtype=np.float32)
        faraday_deg = np.zeros_like(lat_grid, dtype=np.float32)
        b_parallel_t = 0.0
        if self.enable_faraday and self.ionex is not None:
            if (not self.faraday_linear_only) or self.pol_mode == "linear":
                b_parallel_t = self._estimate_b_parallel_t(
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    sat_azimuth_deg=float(sat_info["azimuth_deg"]),
                    sat_elevation_deg=float(sat_info["elevation_deg"]),
                    timestamp=sim_dt,
                    sample_alt_km=self.iono_shell_height_km,
                )
                faraday_deg = faraday_rotation_deg(tec_map, b_parallel_t, self.freq_hz).astype(np.float32)
                pol_db = polarization_mismatch_loss_db(
                    self.pol_mismatch_deg + faraday_deg
                ).astype(np.float32)

        if self.era5 is not None:
            try:
                iwv_map = self.era5.get_iwv(hour_utc, lat_grid, lon_grid)
                atm_db = atmospheric_loss_era5(
                    elevation_deg,
                    self.frequency_ghz,
                    iwv_map,
                    rain_rate_mm_h=rain_rate_mm_h,
                )
            except Exception as exc:
                if self.strict_data:
                    raise RuntimeError(f"[L1] strict_data: ERA5 query failed at {sim_dt.isoformat()}: {exc}") from exc
                print(f"[L1] ERA5 query failed ({exc}); fallback simplified atmospheric model.")
                iwv_map = np.full_like(lat_grid, np.nan, dtype=np.float32)
                atm_db = atmospheric_loss(elevation_deg, self.frequency_ghz, rain_rate_mm_h)
        else:
            iwv_map = np.full_like(lat_grid, np.nan, dtype=np.float32)
            atm_db = atmospheric_loss(elevation_deg, self.frequency_ghz, rain_rate_mm_h)

        net_loss_db = fspl + atm_db + iono_db + pol_db - gain_db

        occlusion_mask = elevation_deg < self.MIN_ELEVATION_DEG
        net_loss_db[occlusion_mask] = self.NO_COVERAGE_LOSS_DB

        valid = net_loss_db < self.NO_COVERAGE_LOSS_DB
        if valid.any():
            print(f"[L1] Done | visible {int(np.sum(~occlusion_mask))}/{GRID_SIZE**2} px | "
                  f"loss {net_loss_db[valid].min():.1f}~{net_loss_db[valid].max():.1f} dB")

        return {
            "total": net_loss_db.astype(np.float32),
            "fspl": fspl.astype(np.float32),
            "atm": atm_db.astype(np.float32),
            "iono": iono_db.astype(np.float32),
            "gain": gain_db.astype(np.float32),
            "pol": pol_db.astype(np.float32),
            "faraday_deg": faraday_deg.astype(np.float32),
            "b_parallel_t": float(b_parallel_t),
            "elevation": elevation_deg.astype(np.float32),
            "azimuth": azimuth_deg.astype(np.float32),
            "slant_range_m": slant_range_m.astype(np.float32),
            "offaxis_theta_deg": offaxis_theta_deg.astype(np.float32),
            "tec": tec_map.astype(np.float32),
            "tec_vtec": tec_vtec_map.astype(np.float32),
            "iwv": iwv_map.astype(np.float32),
            "occlusion_mask": occlusion_mask,
            "satellite": sat_info,
            "timestamp": sim_dt,
        }

    def _select_best_satellite(self,
                               origin_lat: float,
                               origin_lon: float,
                               target_norad_ids: Optional[List[str]] = None) -> Tuple[EarthSatellite, Dict]:
        """Select satellite with highest elevation angle over origin."""
        observer = wgs84.latlon(origin_lat, origin_lon)
        best_sat  = None
        best_info: Dict = {}
        best_el   = -999.0
        skipped   = 0
        filter_ids = target_norad_ids if target_norad_ids is not None else self.target_norad_ids

        for sat, tg in zip(self.satellites, self.tle_groups):
            norad_id = _get_norad_id(tg)
            if filter_ids and norad_id not in filter_ids:
                continue
            try:
                diff = sat - observer
                topocentric = diff.at(self._sim_time)
                alt, az, dist = topocentric.altaz()
                el_deg  = float(alt.degrees)
                az_deg  = float(az.degrees)
                dist_km = float(dist.km)

                alt_km = _get_sat_altitude_km(sat, self._sim_time)
                if alt_km < self.MIN_ORBIT_ALT_KM or alt_km > self.MAX_ORBIT_ALT_KM:
                    skipped += 1
                    continue

                geocentric = sat.at(self._sim_time)
                subpoint   = wgs84.subpoint_of(geocentric)
                sat_lat    = float(subpoint.latitude.degrees)
                sat_lon    = float(subpoint.longitude.degrees)
            except Exception:
                continue

            if el_deg > best_el:
                best_el  = el_deg
                best_sat = sat
                best_info = {
                    "norad_id"      : norad_id,
                    "elevation_deg" : el_deg,
                    "azimuth_deg"   : az_deg,
                    "slant_range_km": dist_km,
                    "lat_deg"       : sat_lat,
                    "lon_deg"       : sat_lon,
                    "alt_m"         : alt_km * 1000.0,
                }

        if skipped:
            print(f"[L1] Skipped {skipped} satellites with invalid orbit altitude.")
        if best_sat is None:
            raise RuntimeError("[L1] No visible satellite found. Check TLE file or NORAD ID filter.")

        print(f"[L1] Selected NORAD {best_info['norad_id']} | "
              f"el={best_info['elevation_deg']:.2f}deg | "
              f"alt={best_info['alt_m']/1e3:.1f} km")
        return best_sat, best_info

    def get_visible_satellites(self,
                               origin_lat: float,
                               origin_lon: float,
                               timestamp: Optional[datetime] = None,
                               min_elevation_deg: Optional[float] = None,
                               max_elevation_deg: Optional[float] = None,
                               target_norad_ids: Optional[List[str]] = None,
                               max_count: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Enumerate visible satellites over an observer location.

        Returns satellites sorted by elevation (descending).
        """
        self._resolve_sim_datetime(timestamp)
        observer = wgs84.latlon(origin_lat, origin_lon)

        min_el = self.MIN_ELEVATION_DEG if min_elevation_deg is None else float(min_elevation_deg)
        max_el = None if max_elevation_deg is None else float(max_elevation_deg)
        filter_ids = target_norad_ids if target_norad_ids is not None else self.target_norad_ids
        if isinstance(filter_ids, (str, int, float)):
            filter_ids = [str(filter_ids)]
        elif isinstance(filter_ids, (list, tuple, set)):
            filter_ids = [str(x) for x in filter_ids]
        else:
            filter_ids = []

        best_per_norad: Dict[str, Dict[str, float]] = {}
        for sat, tg in zip(self.satellites, self.tle_groups):
            norad_id = _get_norad_id(tg)
            if filter_ids and norad_id not in filter_ids:
                continue
            try:
                diff = sat - observer
                topocentric = diff.at(self._sim_time)
                alt, az, dist = topocentric.altaz()
                el_deg = float(alt.degrees)
                if el_deg < min_el:
                    continue
                if max_el is not None and el_deg > max_el:
                    continue

                alt_km = _get_sat_altitude_km(sat, self._sim_time)
                if alt_km < self.MIN_ORBIT_ALT_KM or alt_km > self.MAX_ORBIT_ALT_KM:
                    continue

                geocentric = sat.at(self._sim_time)
                subpoint = wgs84.subpoint_of(geocentric)
                info = {
                    "norad_id": norad_id,
                    "elevation_deg": el_deg,
                    "azimuth_deg": float(az.degrees),
                    "slant_range_km": float(dist.km),
                    "lat_deg": float(subpoint.latitude.degrees),
                    "lon_deg": float(subpoint.longitude.degrees),
                    "alt_m": alt_km * 1000.0,
                }
                prev = best_per_norad.get(norad_id)
                if prev is None or info["elevation_deg"] > prev["elevation_deg"]:
                    best_per_norad[norad_id] = info
            except Exception:
                continue

        visible = list(best_per_norad.values())
        visible.sort(key=lambda x: x["elevation_deg"], reverse=True)
        if max_count is not None and max_count > 0:
            visible = visible[:int(max_count)]
        return visible

    @staticmethod
    def _decimal_year(dt: datetime) -> float:
        start = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
        end = datetime(dt.year + 1, 1, 1, tzinfo=dt.tzinfo)
        frac = (dt - start).total_seconds() / max((end - start).total_seconds(), 1.0)
        return dt.year + frac

    def _query_geomagnetic_ned_t(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
        timestamp: datetime,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Query geomagnetic field (N, E, D) in Tesla at one point.

        Backends are optional; returns None when unavailable.
        """
        if self._geomag_backend == "uninitialized":
            self._geomag_backend = "none"

            try:
                import geomag as geomag_pkg  # type: ignore
                if hasattr(geomag_pkg, "GeoMag"):
                    self._geomag_model = geomag_pkg.GeoMag()
                    self._geomag_backend = "geomag"
                elif hasattr(geomag_pkg, "geomag") and hasattr(geomag_pkg.geomag, "GeoMag"):
                    self._geomag_model = geomag_pkg.geomag.GeoMag()
                    self._geomag_backend = "geomag"
            except Exception:
                pass

            if self._geomag_backend == "none":
                try:
                    import igrf as igrf_pkg  # type: ignore
                    self._geomag_model = igrf_pkg
                    self._geomag_backend = "igrf"
                except Exception:
                    pass

        if self._geomag_backend == "geomag":
            try:
                # geomag commonly expects altitude in feet.
                result = self._geomag_model.GeoMag(  # type: ignore[attr-defined]
                    float(lat_deg),
                    float(lon_deg),
                    h=float(alt_km) * 3280.839895,
                )
                bn = float(getattr(result, "bx", np.nan)) * 1e-9
                be = float(getattr(result, "by", np.nan)) * 1e-9
                bd = float(getattr(result, "bz", np.nan)) * 1e-9
                if np.isfinite(bn) and np.isfinite(be) and np.isfinite(bd):
                    return bn, be, bd
            except Exception:
                return None

        if self._geomag_backend == "igrf":
            try:
                decimal_year = self._decimal_year(timestamp)
                model = self._geomag_model
                if hasattr(model, "igrf_value"):
                    vals = model.igrf_value(float(lat_deg), float(lon_deg), float(alt_km), float(decimal_year))
                    if isinstance(vals, (list, tuple)) and len(vals) >= 6:
                        # Common ordering: D, I, H, X, Y, Z, F
                        bn, be, bd = float(vals[3]), float(vals[4]), float(vals[5])
                        return bn * 1e-9, be * 1e-9, bd * 1e-9
            except Exception:
                return None

        return None

    def _estimate_b_parallel_t(
        self,
        origin_lat: float,
        origin_lon: float,
        sat_azimuth_deg: float,
        sat_elevation_deg: float,
        timestamp: datetime,
        sample_alt_km: float,
    ) -> float:
        """
        Estimate LOS-parallel magnetic component (Tesla).
        """
        ned = self._query_geomagnetic_ned_t(
            lat_deg=origin_lat,
            lon_deg=origin_lon,
            alt_km=sample_alt_km,
            timestamp=timestamp,
        )
        if ned is None:
            if self.strict_data and self.enable_faraday:
                raise RuntimeError(
                    "[L1] strict_data: Faraday rotation enabled but geomagnetic backend unavailable."
                )
            # Conservative fallback when no geomagnetic backend is available.
            return float(self.fallback_b_t)

        bn, be, bd = ned
        az_r = np.deg2rad(float(sat_azimuth_deg))
        el_r = np.deg2rad(float(sat_elevation_deg))

        los_n = np.cos(el_r) * np.cos(az_r)
        los_e = np.cos(el_r) * np.sin(az_r)
        los_d = -np.sin(el_r)
        return float(bn * los_n + be * los_e + bd * los_d)

    # V2.0 stubs
    def _compute_ionospheric_scintillation_db(self, *_) -> np.ndarray:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def _compute_tropospheric_rain_attenuation_db(self, *_) -> np.ndarray:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def _load_antenna_pattern_csv(self, pattern_file: str):
        return None

    # ── Multi-satellite interference methods ──────────────────────────────────

    def compute_multisat_sinr(self,
                              origin_lat: float,
                              origin_lon: float,
                              timestamp: Optional[datetime] = None,
                              context: Optional[LayerContext] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculate SINR map under multi-satellite co-channel interference.

        Physical model:
            SINR(dB) = 10*log10(P_signal / (P_interference + P_noise))

        Where:
            - P_signal: Received power from target satellite (linear domain, mW)
            - P_interference: Sum of interference power from all other visible satellites (linear domain, mW)
            - P_noise: Receiver noise power (linear domain, mW)

        Args:
            origin_lat: Grid center latitude (degrees)
            origin_lon: Grid center longitude (degrees)
            timestamp: Simulation time (UTC)
            context: Optional layer context

        Returns:
            sinr_db: SINR map (256, 256) in dB
            metadata: Dict containing target satellite info, interfering satellite list, etc.
        """
        self._resolve_sim_datetime(timestamp)

        # 1. Get all visible satellites (sorted by elevation, descending)
        visible_sats = self.get_visible_satellites(
            origin_lat, origin_lon, timestamp,
            min_elevation_deg=self.MIN_ELEVATION_DEG
        )

        if len(visible_sats) == 0:
            return np.full((GRID_SIZE, GRID_SIZE), -np.inf, dtype=np.float32), {}

        # 2. Select target satellite (highest elevation)
        target_sat = visible_sats[0]

        # 3. Calculate received power from target satellite (dBm → linear mW)
        p_desired_dbm = self._compute_received_power(
            target_sat, origin_lat, origin_lon, context
        )
        p_desired_linear = 10 ** (p_desired_dbm / 10.0)  # mW

        # 4. Calculate interference power (linear domain sum)
        p_interference_linear = np.zeros_like(p_desired_linear, dtype=np.float64)
        interfering_sats = visible_sats[1:self.max_interfering_sats + 1]

        for sat in interfering_sats:
            p_int_dbm = self._compute_received_power(
                sat, origin_lat, origin_lon, context
            )
            p_int_linear = 10 ** (p_int_dbm / 10.0)
            p_interference_linear += p_int_linear

        # 5. Calculate SINR (linear → dB domain)
        p_noise_linear = 10 ** (self.noise_floor_dbm / 10.0)
        sinr_linear = p_desired_linear / (p_interference_linear + p_noise_linear)
        sinr_db = 10 * np.log10(np.maximum(sinr_linear, 1e-30))

        # 6. Handle no-coverage regions
        lat_grid, lon_grid, x_m, y_m = get_grid_latlon(origin_lat, origin_lon, L1_COVERAGE, GRID_SIZE)
        slant_range, azimuth, elevation_deg = get_azimuth_elevation(
            lat_grid, lon_grid,
            target_sat["lat_deg"], target_sat["lon_deg"], target_sat["alt_m"]
        )
        occlusion_mask = elevation_deg < self.MIN_ELEVATION_DEG
        sinr_db[occlusion_mask] = -np.inf

        metadata = {
            'target_sat_norad_id': target_sat['norad_id'],
            'target_sat_elevation_deg': target_sat['elevation_deg'],
            'num_interfering_sats': len(interfering_sats),
            'interfering_sat_ids': [s['norad_id'] for s in interfering_sats],
            'timestamp': self._sim_time.utc_datetime() if self._sim_time is not None else None
        }

        print(f"[L1] SINR computed | target: {target_sat['norad_id']} | "
              f"interferers: {len(interfering_sats)} | "
              f"SINR range: {sinr_db[~occlusion_mask].min():.1f}~{sinr_db[~occlusion_mask].max():.1f} dB")

        return sinr_db.astype(np.float32), metadata

    def _compute_received_power(self,
                                sat_info: Dict[str, float],
                                origin_lat: float,
                                origin_lon: float,
                                context: Optional[LayerContext] = None) -> np.ndarray:
        """
        Calculate received power from a single satellite across the grid.

        Link budget:
            P_rx(dBm) = P_tx(dBm) + G_tx(dB) + G_rx(dB) - FSPL(dB) - L_atm(dB) - L_iono(dB) - L_pol(dB)

        Args:
            sat_info: Satellite information dict (from get_visible_satellites)
            origin_lat: Grid center latitude (degrees)
            origin_lon: Grid center longitude (degrees)
            context: Optional layer context

        Returns:
            rx_power_dbm: Received power map (256, 256) in dBm
        """
        # 1. Build geographic grid
        lat_grid, lon_grid, x_m, y_m = get_grid_latlon(origin_lat, origin_lon, L1_COVERAGE, GRID_SIZE)

        # 2. Calculate geometric parameters
        slant_range, azimuth, elevation_deg = get_azimuth_elevation(
            lat_grid, lon_grid,
            sat_info["lat_deg"], sat_info["lon_deg"], sat_info["alt_m"]
        )

        # 3. Calculate FSPL
        fspl = fspl_db(slant_range, self.freq_hz)

        # 4. Calculate antenna gain
        # Transmit side: satellite phased array pointing to ground
        # Simplified assumption: all satellites use main beam pointing to grid center
        # (Phase 2 will introduce beam scheduling and sidelobe modeling)
        if self.antenna_pattern_model == "gaussian":
            gain_db = gaussian_beam_gain_db(
                azimuth, elevation_deg,
                sat_info["azimuth_deg"], sat_info["elevation_deg"],
                self.hpbw_az_deg, self.hpbw_el_deg,
                self.peak_gain_db
            )
        else:  # parabolic_rolloff
            # Calculate off-boresight angle (simplified: assume boresight points to grid center)
            origin_slant, origin_az, origin_el = get_azimuth_elevation(
                np.array([[origin_lat]]), np.array([[origin_lon]]),
                sat_info["lat_deg"], sat_info["lon_deg"], sat_info["alt_m"]
            )
            # Off-boresight angle approximation (simplified for now)
            theta_off_boresight = np.abs(elevation_deg - origin_el[0, 0])
            gain_db = parabolic_rolloff_gain_db(
                theta_off_boresight,
                peak_gain_db=self.peak_gain_db,
                theta_3db_deg=self.theta_3db_deg,
                min_gain_db=self.antenna_min_gain_db
            )

        # Receive side: user terminal antenna (simplified as omnidirectional)
        rx_gain_db = 0.0

        # 5. Calculate atmospheric loss
        if self.era5 is not None:
            atm_db = atmospheric_loss_era5(
                elevation_deg, self.frequency_ghz, self.era5,
                lat_grid, lon_grid, self._sim_time.utc_datetime()
            )
        else:
            atm_db = atmospheric_loss(elevation_deg, self.frequency_ghz, self.rain_rate_mm_h)

        # 6. Calculate ionospheric loss
        if self.ionex is not None:
            # Use IPP-based TEC query
            tec_vtec_map = np.full_like(lat_grid, self.default_tec, dtype=np.float32)
            if self.use_ipp:
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        ipp_lat, ipp_lon = ipp_from_ground(
                            lat_grid[i, j], lon_grid[i, j],
                            sat_info["lat_deg"], sat_info["lon_deg"], sat_info["alt_m"],
                            self.iono_shell_height_km * 1000.0
                        )
                        tec_vtec_map[i, j] = self.ionex.get_tec(
                            ipp_lat, ipp_lon, self._sim_time.utc_datetime()
                        )
            else:
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        tec_vtec_map[i, j] = self.ionex.get_tec(
                            lat_grid[i, j], lon_grid[i, j], self._sim_time.utc_datetime()
                        )
            iono_db = ionospheric_loss(self.frequency_ghz, tec_vtec_map)
        else:
            iono_db = ionospheric_loss(self.frequency_ghz, self.default_tec)

        # 7. Calculate polarization loss
        pol_db = self.pol_loss_db

        # 8. Link budget
        rx_power_dbm = (
            self.tx_power_dbm +
            gain_db +
            rx_gain_db -
            fspl -
            atm_db -
            iono_db -
            pol_db
        )

        return rx_power_dbm
