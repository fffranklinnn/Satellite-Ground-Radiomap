"""
TLE loader and SGP4 orbit propagator for SG-MRM project.

Parses a TLE file, filters satellites by inclination, and provides
ECI position propagation + geodetic coordinate conversion.
"""

import numpy as np
from datetime import datetime, timezone
from typing import Tuple

from sgp4.api import Satrec, jday


class TleLoader:
    """
    Loads TLE data and propagates satellite positions via SGP4.

    Args:
        tle_path  : Path to two-line element file
        inc_min   : Minimum inclination filter (degrees)
        inc_max   : Maximum inclination filter (degrees)
    """

    def __init__(self, tle_path: str,
                 inc_min: float = 52.9,
                 inc_max: float = 53.3):
        self.sats = self._load(tle_path, inc_min, inc_max)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.sats)

    def get_geodetic(self, dt: datetime
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagate all satellites to dt and return geodetic coordinates.

        Args:
            dt: UTC datetime

        Returns:
            lats : (N,) degrees
            lons : (N,) degrees  [-180, 180]
            alts : (N,) km above spherical Earth
        """
        jd, fr = self._dt_to_jd(dt)
        r_eci = self._propagate_all(jd, fr)          # (N, 3) km
        return self._eci_to_geodetic(r_eci, jd + fr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str, inc_min: float, inc_max: float):
        lines = []
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s:
                    lines.append(s)

        sats = []
        i = 0
        while i < len(lines) - 1:
            l1, l2 = lines[i], lines[i + 1]
            if l1.startswith('1 ') and l2.startswith('2 '):
                inc = float(l2.split()[2])
                if inc_min <= inc <= inc_max:
                    try:
                        sats.append(Satrec.twoline2rv(l1, l2))
                    except Exception:
                        pass
                i += 2
            else:
                i += 1
        return sats

    @staticmethod
    def _dt_to_jd(dt: datetime) -> Tuple[float, float]:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return jday(dt.year, dt.month, dt.day,
                    dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

    def _propagate_all(self, jd: float, fr: float) -> np.ndarray:
        """Return (N, 3) ECI positions in km, filtering propagation errors."""
        positions = []
        for sat in self.sats:
            e, r, _ = sat.sgp4(jd, fr)
            if e == 0:
                positions.append(r)
        return np.array(positions, dtype=float)  # (N, 3)

    @staticmethod
    def _eci_to_geodetic(r_eci: np.ndarray,
                         jd_full: float
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert ECI (km) to geodetic (lat, lon, alt).

        Uses spherical Earth approximation (R_E = 6371 km).
        GMST computed from Julian date.
        """
        x, y, z = r_eci[:, 0], r_eci[:, 1], r_eci[:, 2]
        r_mag = np.linalg.norm(r_eci, axis=1)

        # Greenwich Mean Sidereal Time (degrees)
        T = jd_full - 2451545.0
        gmst_deg = (280.46061837 + 360.98564724 * T) % 360.0
        gmst_rad = np.radians(gmst_deg)

        # ECI → ECEF rotation (z-axis rotation by GMST)
        x_ecef =  x * np.cos(gmst_rad) + y * np.sin(gmst_rad)
        y_ecef = -x * np.sin(gmst_rad) + y * np.cos(gmst_rad)

        lats = np.degrees(np.arcsin(np.clip(z / r_mag, -1.0, 1.0)))
        lons = np.degrees(np.arctan2(y_ecef, x_ecef))
        alts = r_mag - 6371.0

        return lats, lons, alts
