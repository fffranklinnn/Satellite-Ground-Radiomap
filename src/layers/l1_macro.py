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
    phased_array_peak_gain_db,
    phased_array_hpbw_deg,
    SPEED_OF_LIGHT,
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
            self.cfg = self._load_yaml(cfg_path)
            origin_cfg = self.cfg.get("origin", {})
            origin_lat = float(origin_cfg.get("lat", 34.3416))
            origin_lon = float(origin_cfg.get("lon", 108.9398))
            base_config = {"grid_size": 256, "coverage_km": 256.0, "resolution_m": 1000.0}
        else:
            self.cfg = config
            base_config = config

        super().__init__(base_config, origin_lat, origin_lon)

        # RF / antenna parameters
        freq_cfg = self.cfg.get("frequency", {})
        if isinstance(freq_cfg, dict):
            self.freq_hz = float(freq_cfg.get("center_hz", self.FREQ_HZ_DEFAULT))
        else:
            self.freq_hz = float(self.cfg.get("frequency_ghz", self.FREQ_HZ_DEFAULT / 1e9)) * 1e9

        self.wavelength_m = SPEED_OF_LIGHT / self.freq_hz
        self.element_spacing_m = 0.5 * self.wavelength_m

        n_elements = self.ARRAY_ROWS * self.ARRAY_COLS
        self.peak_gain_db = phased_array_peak_gain_db(n_elements, self.ELEMENT_GAIN_DBI)
        self.hpbw_az_deg, self.hpbw_el_deg = phased_array_hpbw_deg(
            self.ARRAY_ROWS, self.ARRAY_COLS,
            self.wavelength_m, self.element_spacing_m
        )

        pol_cfg = self.cfg.get("polarization", {})
        self.pol_mismatch_deg = float(pol_cfg.get("mismatch_angle_deg", 0.0))
        self.pol_loss_db = polarization_loss_db(self.pol_mismatch_deg)

        self.target_norad_ids: List[str] = [
            str(x) for x in self.cfg.get("target_norad_ids", [])
        ]

        # Resolve TLE file path
        tle_cfg = self.cfg.get("tle", {})
        tle_file = tle_cfg.get("file") if isinstance(tle_cfg, dict) else None
        if tle_file is None:
            tle_file = self.cfg.get("tle_file")
        if tle_file is None:
            raise ValueError("[L1] No TLE file specified (key: tle.file or tle_file).")

        _root = Path(__file__).resolve().parents[2]
        tle_path = (_root / tle_file) if not Path(tle_file).is_absolute() else Path(tle_file)
        if not tle_path.exists():
            tle_path = Path(tle_file)
        self.satellites, self.tle_groups = _parse_tle_file(str(tle_path))

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

    def compute(self,
                origin_lat: float = None,
                origin_lon: float = None,
                timestamp: Optional[datetime] = None,
                context: Optional[LayerContext] = None,
                **kwargs) -> np.ndarray:
        """
        Compute L1 net-loss matrix (256x256, float32, dB).

        Pixels with elevation < MIN_ELEVATION_DEG are set to NO_COVERAGE_LOSS_DB.
        """
        if origin_lat is None:
            origin_lat = self.origin_lat
        if origin_lon is None:
            origin_lon = self.origin_lon

        if timestamp is not None:
            self.set_time(timestamp)
        if self._sim_time is None:
            self._sim_time = self.ts.now()
            print("[L1] No simulation time set — using current UTC.")

        print(f"[L1] Computing @ ({origin_lat:.4f}N, {origin_lon:.4f}E)")

        sat, sat_info = self._select_best_satellite(origin_lat, origin_lon)

        lat_grid, lon_grid, x_m, y_m = get_grid_latlon(origin_lat, origin_lon)

        lat_rad = np.deg2rad(origin_lat)
        sat_x_m = ((sat_info["lon_deg"] - origin_lon) *
                   np.deg2rad(1.0) * EARTH_RADIUS_M * np.cos(lat_rad))
        sat_y_m = ((sat_info["lat_deg"] - origin_lat) *
                   np.deg2rad(1.0) * EARTH_RADIUS_M)

        azimuth_deg, elevation_deg, slant_range_m = get_azimuth_elevation(
            sat_x_m, sat_y_m, sat_info["alt_m"], x_m, y_m
        )

        delta_el = sat_info["elevation_deg"] - elevation_deg
        delta_az = np.zeros_like(azimuth_deg)
        gain_db = gaussian_beam_gain_db(
            theta_az_deg=delta_az,
            theta_el_deg=delta_el,
            peak_gain_db=self.peak_gain_db,
            hpbw_az_deg=self.hpbw_az_deg * 3.0,
            hpbw_el_deg=self.hpbw_el_deg * 3.0,
        )

        fspl = fspl_db(slant_range_m, self.freq_hz)
        net_loss_db = fspl + self.pol_loss_db - gain_db

        occlusion_mask = elevation_deg < self.MIN_ELEVATION_DEG
        net_loss_db[occlusion_mask] = self.NO_COVERAGE_LOSS_DB

        valid = net_loss_db < self.NO_COVERAGE_LOSS_DB
        if valid.any():
            print(f"[L1] Done | visible {int(np.sum(~occlusion_mask))}/{GRID_SIZE**2} px | "
                  f"loss {net_loss_db[valid].min():.1f}~{net_loss_db[valid].max():.1f} dB")

        return net_loss_db.astype(np.float32)

    def _select_best_satellite(self, origin_lat: float,
                               origin_lon: float) -> Tuple[EarthSatellite, Dict]:
        """Select satellite with highest elevation angle over origin."""
        observer = wgs84.latlon(origin_lat, origin_lon)
        best_sat  = None
        best_info: Dict = {}
        best_el   = -999.0
        skipped   = 0

        for sat, tg in zip(self.satellites, self.tle_groups):
            norad_id = _get_norad_id(tg)
            if self.target_norad_ids and norad_id not in self.target_norad_ids:
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

    # V2.0 stubs
    def _compute_ionospheric_scintillation_db(self, *_) -> np.ndarray:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def _compute_tropospheric_rain_attenuation_db(self, *_) -> np.ndarray:
        return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    def _load_antenna_pattern_csv(self, pattern_file: str):
        return None

