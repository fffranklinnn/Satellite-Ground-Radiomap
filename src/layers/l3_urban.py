"""
L3 Urban/Micro Layer for SG-MRM project.
Merged from branch_L3/zhangyue/src/layers/l3_urban.py.

Responsibilities:
- Load per-tile building height raster cache (H.npy, Occ.npy).
- Compute NLoS mask via directional scan under parallel-wave assumption.
- Map NLoS / occupancy mask to loss_db for L3 aggregation.
- Output: 256x256 float32 loss matrix (dB).

Interface:
- __init__ accepts EITHER (config: dict, origin_lat, origin_lon) [main.py style]
                       OR (tile_cache_root, nlos_loss_db, ...)   [standalone]
- compute(origin_lat, origin_lon, timestamp=None, context=None)
  context.incident_dir is required for NLoS calculation.

Tile cache is produced by tools/build_l3_tile_cache.py.
Coverage: 256 m, Resolution: 1 m/pixel
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .base import BaseLayer, LayerContext
from ..context.frame_context import FrameContext
from ..context.layer_states import EntryWaveState, UrbanRefinementState
from ..context.time_utils import StrictModeError


# ── Incident direction helpers (from branch_L3) ───────────────────────────────

def _stable_tile_id(tile_origin: Any) -> str:
    """Derive a stable string tile ID from various tile_origin formats."""
    if isinstance(tile_origin, Mapping) and "tile_id" in tile_origin:
        return str(tile_origin["tile_id"])
    if isinstance(tile_origin, str):
        return tile_origin
    payload = json.dumps(tile_origin, ensure_ascii=True, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"tile_{digest}"


def _normalize_incident_direction(incident_dir: Any) -> Tuple[float, float, float]:
    """
    Normalise incident_dir to (hx, hy, tan_elevation).

    Supported formats:
    - [east, north, up]  ENU vector (unit or non-unit)
    - {"vector": [e, n, u]}
    - {"az_deg": float, "el_deg": float}
    """
    if incident_dir is None:
        raise ValueError("incident_dir is required for L3 urban occlusion.")

    if isinstance(incident_dir, Mapping):
        if "vector" in incident_dir:
            return _normalize_incident_direction(incident_dir["vector"])
        if "enu" in incident_dir:
            return _normalize_incident_direction(incident_dir["enu"])
        az = incident_dir.get("az_deg", incident_dir.get("azimuth_deg"))
        el = incident_dir.get("el_deg", incident_dir.get("elevation_deg"))
        if az is not None and el is not None:
            az_rad = np.deg2rad(float(az))
            el_rad = np.deg2rad(float(el))
            cos_el = float(np.cos(el_rad))
            hx = float(np.sin(az_rad) * cos_el)
            hy = float(np.cos(az_rad) * cos_el)
            tan_el = float(np.tan(el_rad))
            hnorm = float(np.hypot(hx, hy))
            if hnorm <= 1e-8:
                raise ValueError("incident_dir horizontal component is too small.")
            return hx / hnorm, hy / hnorm, tan_el
        if "incident_dir" in incident_dir:
            return _normalize_incident_direction(incident_dir["incident_dir"])
        raise ValueError("Unsupported incident_dir mapping schema.")

    vec = np.asarray(incident_dir, dtype=np.float64).reshape(-1)
    if vec.size != 3:
        raise ValueError("incident_dir vector must have exactly 3 values: [e, n, u].")
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError("incident_dir vector norm is zero.")
    vec = vec / norm
    hx, hy, uz = float(vec[0]), float(vec[1]), float(vec[2])
    hnorm = float(np.hypot(hx, hy))
    if hnorm <= 1e-8:
        raise ValueError("incident_dir horizontal component is too small.")
    return hx / hnorm, hy / hnorm, float(abs(uz) / hnorm)


def compute_nlos_mask(height_m: np.ndarray, incident_dir: Any) -> np.ndarray:
    """
    Compute NLoS mask via directional scan, O(N^2 log N), per-ray sweep.

    Args:
        height_m:    2-D building height raster (metres), shape [H, W].
        incident_dir: Incoming direction (ENU vector or az/el mapping).

    Returns:
        Boolean array of shape [H, W]; True = NLoS (blocked).
    """
    grid = np.asarray(height_m, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError("height_m must be a 2-D array.")

    hx, hy, tan_el = _normalize_incident_direction(incident_dir)
    if tan_el <= 1e-8:
        raise ValueError("Elevation angle must be > 0 to compute shadow mask.")

    rows, cols = grid.shape
    x = np.arange(cols, dtype=np.float32) + 0.5
    y = np.arange(rows, dtype=np.float32) + 0.5
    xx, yy = np.meshgrid(x, y)

    u = xx * hx + yy * hy
    v = xx * (-hy) + yy * hx
    # ray_id = np.rint(v).astype(np.int32)
    ray_id = np.floor(v).astype(np.int32)

    u_flat   = u.ravel()
    ray_flat = ray_id.ravel()
    h_flat   = grid.ravel()
    order    = np.lexsort((u_flat, ray_flat))

    blocked     = np.zeros_like(h_flat, dtype=bool)
    current_ray = np.int32(-2**31)
    max_term    = -np.inf
    eps         = 1e-6

    for idx in order:
        rid = ray_flat[idx]
        if rid != current_ray:
            current_ray = rid
            max_term    = -np.inf

        ui = float(u_flat[idx])
        if max_term >= tan_el * ui - eps:
            blocked[idx] = True

        term = float(h_flat[idx]) + tan_el * ui
        if term > max_term:
            max_term = term

    return blocked.reshape(grid.shape)


# ── L3UrbanLayer ──────────────────────────────────────────────────────────────

class L3UrbanLayer(BaseLayer):
    """
    L3 Urban/Micro Layer: building-scale NLoS propagation loss.

    Requires tile cache produced by tools/build_l3_tile_cache.py:
        <tile_cache_root>/<tile_id>/H.npy    — building height raster
        <tile_cache_root>/<tile_id>/Occ.npy  — occupancy mask (optional)
    """

    name = "l3_urban"

    def __init__(self,
                 config: Union[Dict[str, Any], str, Path],
                 origin_lat: float = None,
                 origin_lon: float = None,
                 nlos_loss_db: float = None,
                 occ_loss_db: float = None):
        """
        Two calling conventions:

        1. main.py style:
               L3UrbanLayer(config_dict, origin_lat, origin_lon)
           Config keys: tile_cache_root, nlos_loss_db, occ_loss_db,
                        incident_dir (optional default direction)

        2. Standalone style (matching branch_L3):
               L3UrbanLayer("data/l3_urban/tiles", nlos_loss_db=20.0)
        """
        if isinstance(config, (str, Path)):
            tile_cache_root = str(config)
            self.nlos_loss_db = float(nlos_loss_db) if nlos_loss_db is not None else 20.0
            self.occ_loss_db  = float(occ_loss_db)  if occ_loss_db  is not None else None
            self.strict_data  = False
            base_config = {"grid_size": 256, "coverage_km": 0.256, "resolution_m": 1.0}
            origin_lat = origin_lat or 0.0
            origin_lon = origin_lon or 0.0
        else:
            tile_cache_root   = config.get("tile_cache_root", "data/l3_urban/tiles")
            self.nlos_loss_db = float(config.get("nlos_loss_db", 20.0))
            _occ = config.get("occ_loss_db")
            self.occ_loss_db  = float(_occ) if _occ is not None else None
            self.strict_data  = bool(config.get("strict_data", config.get("strict_mode", False)))
            # Default incident_dir from config (can be overridden per compute call)
            self._default_incident_dir = config.get("incident_dir")
            base_config = config

        super().__init__(base_config, origin_lat, origin_lon)
        self.tile_cache_root = Path(tile_cache_root)
        self._tile_index: Optional[list] = None  # lazy-built list of (lon, lat, tile_id)

    def _build_tile_index(self):
        """Scan tile cache and build (lon, lat, tile_id) index from meta.json files."""
        index = []
        for tile_dir in self.tile_cache_root.iterdir():
            meta_path = tile_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ox = float(meta["origin"]["x"])
                oy = float(meta["origin"]["y"])
                index.append((ox, oy, tile_dir.name))
            except Exception:
                continue
        self._tile_index = index

    def _find_nearest_tile_id(self, lon: float, lat: float) -> str:
        """Return tile_id of the nearest cached tile to (lon, lat)."""
        if self._tile_index is None:
            self._build_tile_index()
        if not self._tile_index:
            if getattr(self, "strict_data", False):
                raise StrictModeError(f"[L3] strict_data: Tile cache is empty: {self.tile_cache_root}")
            raise FileNotFoundError(f"[L3] Tile cache is empty: {self.tile_cache_root}")
        best_id = min(self._tile_index, key=lambda t: (t[0] - lon) ** 2 + (t[1] - lat) ** 2)[2]
        return best_id

    def _load_tile_cache(self, tile_origin: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load H.npy and optionally Occ.npy for the given tile.

        Tile lookup order:
        1. Explicit tile_id string / mapping with 'tile_id' key.
        2. lat/lon dict: hash using build_l3_tile_cache format (lon|lat|EPSG:4326).
        3. Fallback: nearest tile in cache by Euclidean distance on lon/lat.
        """
        if isinstance(tile_origin, Mapping) and "lat" in tile_origin and "lon" in tile_origin:
            import hashlib as _hl
            lon = float(tile_origin["lon"])
            lat = float(tile_origin["lat"])
            token = f"{lon:.10f}|{lat:.10f}|EPSG:4326"
            digest = _hl.sha1(token.encode("utf-8")).hexdigest()[:16]
            tile_id = f"tile_{digest}"
        else:
            tile_id = _stable_tile_id(tile_origin)
            lon = lat = None

        tile_dir = self.tile_cache_root / tile_id
        h_path   = tile_dir / "H.npy"
        occ_path = tile_dir / "Occ.npy"

        if not h_path.exists():
            # Exact hash miss — fall back to nearest tile in cache
            if lon is not None and lat is not None:
                tile_id  = self._find_nearest_tile_id(lon, lat)
                tile_dir = self.tile_cache_root / tile_id
                h_path   = tile_dir / "H.npy"
                occ_path = tile_dir / "Occ.npy"
            if not h_path.exists():
                if getattr(self, "strict_data", False):
                    raise StrictModeError(f"[L3] strict_data: Missing tile cache: {h_path}")
                raise FileNotFoundError(f"[L3] Missing tile cache: {h_path}")

        height_m = np.load(h_path)
        occ      = np.load(occ_path) if occ_path.exists() else None
        return height_m, occ

    def compute(self,
                origin_lat: float = None,
                origin_lon: float = None,
                timestamp=None,
                context: Optional[LayerContext] = None,
                **kwargs) -> np.ndarray:
        """
        Compute L3 urban loss map (256x256, float32, dB).

        Args:
            origin_lat: Origin latitude (used to derive tile_id when no explicit
                        tile_id is provided in context.extras).
            origin_lon: Origin longitude.
            timestamp:  Unused by L3.
            context:    Must carry context.incident_dir for NLoS calculation.
                        Can also carry context.extras["tile_id"] to override
                        the default lat/lon-based tile lookup.

        Returns:
            256x256 float32 loss array in dB.
        """
        if origin_lat is None:
            origin_lat = self.origin_lat
        if origin_lon is None:
            origin_lon = self.origin_lon

        ctx = LayerContext.from_any(context).merged_with_kwargs(kwargs)

        # Resolve incident_dir: context > default config value
        incident_dir = ctx.incident_dir
        if incident_dir is None:
            incident_dir = getattr(self, "_default_incident_dir", None)
        if incident_dir is None:
            raise ValueError(
                "[L3] L3UrbanLayer.compute requires incident_dir via context "
                "or config key 'incident_dir'."
            )

        # Resolve tile_origin: explicit tile_id > lat/lon dict
        tile_id = ctx.extras.get("tile_id")
        if tile_id is not None:
            tile_origin = {"tile_id": tile_id}
        else:
            tile_origin = {"lat": origin_lat, "lon": origin_lon}

        height_m, occ = self._load_tile_cache(tile_origin)
        nlos_mask = compute_nlos_mask(height_m, incident_dir)

        loss_db = np.zeros(height_m.shape, dtype=np.float32)
        loss_db[nlos_mask] = self.nlos_loss_db

        if self.occ_loss_db is not None:
            occ_mask = occ.astype(bool) if occ is not None else (height_m > 0)
            loss_db[occ_mask] = np.maximum(loss_db[occ_mask], self.occ_loss_db)

        return loss_db

    # ------------------------------------------------------------------
    # FrameContext-based interface (task11 / AC-3, AC-4)
    # ------------------------------------------------------------------

    def refine_urban(self,
                     frame: FrameContext,
                     entry: Optional[EntryWaveState] = None,
                     context: Optional[LayerContext] = None,
                     **kwargs) -> UrbanRefinementState:
        """
        Compute L3 urban NLoS residual for a FrameContext frame.

        The incident direction is derived from the entry state's center-pixel
        azimuth/elevation if not provided in context.

        Args:
            frame:   FrameContext for this simulation frame.
            entry:   EntryWaveState from L1 (used for incident direction).
            context: Optional LayerContext; incident_dir overrides entry geometry.

        Returns:
            UrbanRefinementState with frame_id == frame.frame_id.
        """
        if frame.grid is None:
            raise ValueError("refine_urban requires frame.grid to be set.")

        # Validate entry frame_id consistency
        if entry is not None:
            frame.check_frame_id(entry.frame_id)

        # Warn if caller is injecting geometry via extras alongside a FrameContext
        if context is not None:
            ctx_check = LayerContext.from_any(context)
            if ctx_check.incident_dir is not None and entry is not None:
                warnings.warn(
                    "refine_urban: incident_dir in LayerContext is deprecated when a "
                    "FrameContext and EntryWaveState are provided. "
                    "Incident direction is derived from entry state.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Derive incident_dir from entry state if not in context
        ctx = LayerContext.from_any(context).merged_with_kwargs(kwargs)
        if ctx.incident_dir is None and entry is not None:
            cy, cx = entry.grid.ny // 2, entry.grid.nx // 2
            az_deg = float(entry.azimuth_deg[cy, cx])
            el_deg = float(entry.elevation_deg[cy, cx])
            ctx = LayerContext(
                incident_dir={"az_deg": az_deg, "el_deg": el_deg},
                extras=dict(ctx.extras),
            )

        loss_db = self.compute(
            origin_lat=frame.grid.center_lat,
            origin_lon=frame.grid.center_lon,
            timestamp=frame.timestamp,
            context=ctx,
        )

        nlos_mask = loss_db > 0
        # support_mask: True where tile data was available (non-zero loss or tile loaded)
        support_mask = np.ones(loss_db.shape, dtype=bool)

        return UrbanRefinementState(
            frame_id=frame.frame_id,
            grid=frame.grid,
            urban_grid=frame.grid,
            urban_residual_db=loss_db,
            support_mask=support_mask,
            nlos_mask=nlos_mask,
        )
