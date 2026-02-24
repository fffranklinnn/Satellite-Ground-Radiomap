"""
IONEX (Ionosphere Map Exchange) file loader.

Parses IONEX v1.0 files (plain or .gz) and provides bilinear spatial +
linear temporal interpolation of TEC (Total Electron Content) values.
"""

import gzip
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class IonexLoader:
    """
    Parse an IONEX file and interpolate TEC at arbitrary locations/times.

    Attributes:
        epochs    : np.ndarray (N,)       seconds-of-day for each map
        lats      : np.ndarray (n_lat,)   latitudes  87.5 → -87.5, step -2.5
        lons      : np.ndarray (n_lon,)   longitudes -180 → 180,   step  5.0
        tec_maps  : np.ndarray (N, n_lat, n_lon)  TEC in TECU; NaN where missing
    """

    def __init__(self, path: str):
        self.epochs, self.lats, self.lons, self.tec_maps = self._parse(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tec(self, epoch_sec: float,
                lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Bilinear spatial + linear temporal interpolation of TEC.

        Args:
            epoch_sec : seconds of day (0–86400)
            lat       : latitude array (any shape), degrees
            lon       : longitude array (same shape as lat), degrees

        Returns:
            TEC in TECU, same shape as lat/lon inputs
        """
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        orig_shape = lat.shape

        # Temporal bracketing
        i1 = np.searchsorted(self.epochs, epoch_sec)
        i1 = int(np.clip(i1, 1, len(self.epochs) - 1))
        i0 = i1 - 1
        t0, t1 = self.epochs[i0], self.epochs[i1]
        t_frac = (epoch_sec - t0) / (t1 - t0) if t1 != t0 else 0.0

        tec0 = self._interp_map(i0, lat.ravel(), lon.ravel())
        tec1 = self._interp_map(i1, lat.ravel(), lon.ravel())
        tec = tec0 + t_frac * (tec1 - tec0)

        # Clamp and reshape
        tec = np.clip(tec, 0.0, 300.0)
        return tec.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interp_map(self, map_idx: int,
                    lat_flat: np.ndarray, lon_flat: np.ndarray) -> np.ndarray:
        """Bilinear interpolation on a single TEC map."""
        tec_map = self.tec_maps[map_idx]

        # RegularGridInterpolator requires ascending axes
        # self.lats is descending (87.5 → -87.5), so flip
        lats_asc = self.lats[::-1]
        tec_asc = tec_map[::-1, :]

        # Fill NaN with map median before interpolation
        median_val = np.nanmedian(tec_asc)
        tec_filled = np.where(np.isnan(tec_asc), median_val, tec_asc)

        interp = RegularGridInterpolator(
            (lats_asc, self.lons), tec_filled,
            method='linear', bounds_error=False, fill_value=median_val
        )
        pts = np.column_stack([lat_flat, lon_flat])
        return interp(pts)

    @staticmethod
    def _parse(path: str):
        """Parse IONEX file; return (epochs, lats, lons, tec_maps)."""
        opener = gzip.open if path.endswith('.gz') else open

        epochs = []
        tec_maps = []
        lats_ref = None
        lons_ref = None

        with opener(path, 'rt', encoding='ascii', errors='replace') as fh:
            in_header = True
            in_map = False
            current_epoch = None
            current_lat_rows = []   # list of 1-D arrays, one per lat band
            current_lat_vals = []   # lat values in order
            current_row_buf = []    # integer buffer for current lat row
            current_lat = None
            n_lons = None

            for line in fh:
                label = line[60:].strip() if len(line) > 60 else ''

                # ---- header end ----
                if 'END OF HEADER' in label:
                    in_header = False
                    continue
                if in_header:
                    continue

                # ---- map boundaries ----
                if 'START OF TEC MAP' in label:
                    in_map = True
                    current_epoch = None
                    current_lat_rows = []
                    current_lat_vals = []
                    current_row_buf = []
                    current_lat = None
                    continue

                if 'END OF TEC MAP' in label:
                    # flush last lat row
                    if current_row_buf and current_lat is not None:
                        current_lat_rows.append(
                            (current_lat, np.array(current_row_buf, dtype=float))
                        )
                        current_row_buf = []

                    if current_epoch is not None and current_lat_rows:
                        # sort by lat descending (as in file)
                        current_lat_rows.sort(key=lambda x: -x[0])
                        lat_vals = np.array([r[0] for r in current_lat_rows])
                        rows = np.vstack([r[1] for r in current_lat_rows])
                        # replace missing (9999 → 999.9 after /10 later, flag now)
                        rows[rows == 9999.0] = np.nan
                        rows /= 10.0

                        if lats_ref is None:
                            lats_ref = lat_vals
                        if lons_ref is None and n_lons is not None:
                            lons_ref = np.linspace(-180.0, 180.0, n_lons)

                        epochs.append(current_epoch)
                        tec_maps.append(rows)

                    in_map = False
                    continue

                if not in_map:
                    continue

                # ---- epoch ----
                if 'EPOCH OF CURRENT MAP' in label:
                    parts = line[:60].split()
                    y, mo, d, h, mi, s = [int(x) for x in parts[:6]]
                    current_epoch = h * 3600 + mi * 60 + s
                    continue

                # ---- lat row header ----
                if 'LAT/LON1/LON2/DLON/H' in label:
                    # flush previous lat row buffer
                    if current_row_buf and current_lat is not None:
                        current_lat_rows.append(
                            (current_lat, np.array(current_row_buf, dtype=float))
                        )
                        current_row_buf = []

                    # Fixed-width format: 2X + 5×F6.1
                    # cols: lat[2:8], lon1[8:14], lon2[14:20], dlon[20:26]
                    current_lat = float(line[2:8])
                    lon1 = float(line[8:14])
                    lon2 = float(line[14:20])
                    dlon = float(line[20:26])
                    n_lons = int(round((lon2 - lon1) / dlon)) + 1
                    continue

                # ---- data line ----
                # Data lines have no label (cols 60+ are blank or spaces)
                if in_map and current_lat is not None:
                    vals = line.split()   # read full line, not just [:60]
                    if vals:
                        try:
                            current_row_buf.extend(int(v) for v in vals)
                        except ValueError:
                            pass

        if not tec_maps:
            raise ValueError(f"No TEC maps found in {path}")

        epochs_arr = np.array(epochs, dtype=float)
        tec_arr = np.array(tec_maps, dtype=float)   # (N, n_lat, n_lon)

        if lons_ref is None:
            lons_ref = np.linspace(-180.0, 180.0, tec_arr.shape[2])

        return epochs_arr, lats_ref, lons_ref, tec_arr
