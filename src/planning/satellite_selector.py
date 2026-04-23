"""
SatelliteSelector — deterministic satellite selection service.

Extracts satellite selection from L1MacroLayer into a shared service
so all layers in a frame share identical pre-bound link geometry.

Selection rules:
    1. Filter by target_norad_ids (if provided)
    2. Filter by orbital altitude (200-40000 km)
    3. Filter by minimum elevation angle (default 5 deg)
    4. Rank by highest elevation
    5. Tie-break by lowest numeric NORAD ID
    6. Return slant_range in meters (not km)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.context.time_utils import require_utc


class SatelliteSelectionError(ValueError):
    """Raised when satellite selection fails."""


class NoVisibleSatelliteError(SatelliteSelectionError):
    """Raised when no satellite meets visibility criteria."""


class InvalidNoradIdError(SatelliteSelectionError):
    """Raised when a requested NORAD ID is not found in the TLE catalog."""


# Placeholder for remaining content


MIN_ORBIT_ALT_KM = 200.0
MAX_ORBIT_ALT_KM = 40_000.0
DEFAULT_MIN_ELEVATION_DEG = 5.0


def _parse_tle_file(tle_file_path: str):
    """Parse a TLE file and return (satellites, tle_groups)."""
    from sgp4.api import Satrec
    from skyfield.api import EarthSatellite

    tle_file = Path(tle_file_path)
    if not tle_file.exists():
        raise FileNotFoundError(f"TLE file not found: {tle_file_path}")

    with open(tle_file, "r", encoding="utf-8") as f:
        tle_lines = [line.strip() for line in f if line.strip()]

    satellites = []
    tle_groups = []
    i = 0
    while i < len(tle_lines):
        if (tle_lines[i].startswith("1") and
                i + 1 < len(tle_lines) and
                tle_lines[i + 1].startswith("2")):
            line1, line2 = tle_lines[i], tle_lines[i + 1]
            tle_groups.append((line1, line2))
            satellites.append(EarthSatellite(line1, line2))
            i += 2
        else:
            i += 1
    return satellites, tle_groups


def _get_norad_id(tle_group: Tuple[str, str]) -> str:
    return tle_group[1].split()[1]


def _get_sat_altitude_km(sat, t) -> float:
    pos_km = sat.at(t).position.km
    return float(np.linalg.norm(pos_km)) - 6371.0


class SatelliteSelector:
    """Deterministic satellite selection service.

    Args:
        tle_path: Path to TLE file.
        strict: If True, raise errors instead of returning None.
        min_elevation_deg: Minimum elevation threshold for visibility.
    """

    def __init__(
        self,
        tle_path: str,
        strict: bool = True,
        min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
    ) -> None:
        self.tle_path = str(tle_path)
        self.strict = strict
        self.min_elevation_deg = min_elevation_deg
        self.satellites, self.tle_groups = _parse_tle_file(self.tle_path)
        self._catalog_norad_ids = {
            _get_norad_id(tg) for tg in self.tle_groups
        }

    def select(
        self,
        timestamp: datetime,
        center: Tuple[float, float],
        target_ids: Optional[List[str]] = None,
        min_elevation_deg: Optional[float] = None,
        strict: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select the best satellite for a frame.

        Returns a sat_info dict with keys:
            norad_id, elevation_deg, azimuth_deg, slant_range_m,
            lat_deg, lon_deg, alt_m

        Raises:
            InvalidNoradIdError: If a target NORAD ID is not in the catalog.
            NoVisibleSatelliteError: If no satellite meets criteria (strict mode).
        """
        from skyfield.api import wgs84, load

        use_strict = self.strict if strict is None else strict
        ts_utc = require_utc(timestamp, strict=use_strict)
        min_el = self.min_elevation_deg if min_elevation_deg is None else float(min_elevation_deg)
        center_lat, center_lon = center

        # Validate target NORAD IDs
        filter_ids = None
        if target_ids is not None:
            filter_ids = [str(x) for x in target_ids]
            unknown = set(filter_ids) - self._catalog_norad_ids
            if unknown:
                raise InvalidNoradIdError(
                    f"NORAD IDs not found in TLE catalog: {sorted(unknown)}"
                )

        candidates = self._collect_candidates(ts_utc, center_lat, center_lon, min_el, filter_ids)

        if not candidates:
            if use_strict:
                raise NoVisibleSatelliteError(
                    f"No satellite meets visibility criteria "
                    f"(min_elevation={min_el}deg) at {ts_utc.isoformat()} "
                    f"over ({center_lat}, {center_lon})."
                )
            return None

        # Deterministic sort: highest elevation first, then lowest NORAD ID
        candidates.sort(key=lambda c: (-c["elevation_deg"], int(c["norad_id"])))
        return candidates[0]

    def _collect_candidates(
        self,
        ts_utc,
        center_lat: float,
        center_lon: float,
        min_el: float,
        filter_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Collect visible satellite candidates. Extracted for testability."""
        from skyfield.api import wgs84, load

        ts = load.timescale()
        t = ts.from_datetime(ts_utc)
        observer = wgs84.latlon(center_lat, center_lon)

        candidates: List[Dict[str, Any]] = []
        for sat, tg in zip(self.satellites, self.tle_groups):
            norad_id = _get_norad_id(tg)
            if filter_ids and norad_id not in filter_ids:
                continue
            try:
                diff = sat - observer
                topocentric = diff.at(t)
                alt_obj, az_obj, dist_obj = topocentric.altaz()
                el_deg = float(alt_obj.degrees)
                az_deg = float(az_obj.degrees)
                dist_km = float(dist_obj.km)

                alt_km = _get_sat_altitude_km(sat, t)
                if alt_km < MIN_ORBIT_ALT_KM or alt_km > MAX_ORBIT_ALT_KM:
                    continue

                if el_deg < min_el:
                    continue

                geocentric = sat.at(t)
                subpoint = wgs84.subpoint_of(geocentric)
                sat_lat = float(subpoint.latitude.degrees)
                sat_lon = float(subpoint.longitude.degrees)

                candidates.append({
                    "norad_id": norad_id,
                    "elevation_deg": el_deg,
                    "azimuth_deg": az_deg,
                    "slant_range_m": dist_km * 1000.0,
                    "lat_deg": sat_lat,
                    "lon_deg": sat_lon,
                    "alt_m": alt_km * 1000.0,
                })
            except Exception:
                continue

        return candidates
