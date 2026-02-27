"""
Offline tile-cache builder for L3 urban layer.
L3 城市层的离线 tile 缓存构建脚本。

Purpose / 作用:
- Convert normalized building catalog into per-tile raster cache.
- 将规范化建筑库转为按 tile 存储的栅格缓存。

Main capabilities / 核心功能:
- Read catalog + tile list and crop buildings per tile.
- 读取 catalog 与 tile 列表，逐 tile 裁剪建筑。
- Rasterize to `H.npy` (height) and `Occ.npy` (occupancy).
- 栅格化输出 `H.npy`（高度）与 `Occ.npy`（占据）。
- Save tile metadata (`meta.json`) with CRS and affine information.
- 保存 `meta.json`（CRS、仿射参数等元信息）。

Interfaces / 接口:
- CLI entrypoint: `python tools/build_l3_tile_cache.py ...`
- 命令行入口：`python tools/build_l3_tile_cache.py ...`

Relationship / 与其他脚本关系:
- Input catalog comes from `tools/preprocess_buildings_catalog.py`.
- 输入 catalog 来自 `tools/preprocess_buildings_catalog.py`。
- Output tiles are consumed by `src/layers/l3_urban.py` and visualization scripts.
- 输出 tiles 被 `src/layers/l3_urban.py` 和可视化脚本消费。

Example / 调用示例:
```powershell
conda run -n radiodiff python tools/build_l3_tile_cache.py `
  --catalog data/l3_urban/xian/catalog/buildings_xian.parquet `
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv `
  --output-root data/l3_urban/xian/tiles_60 `
  --origin-crs EPSG:3857 `
  --tile-size-m 256 --resolution-m 1 --height-field height_m
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from shapely.ops import transform


LOGGER = logging.getLogger("build_l3_tile_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-tile L3 cache: H.npy / Occ.npy / meta.json"
    )
    parser.add_argument("--catalog", type=Path, required=True, help="Normalized catalog (.parquet/.gpkg).")
    parser.add_argument(
        "--tile-list",
        type=Path,
        required=True,
        help="CSV or JSON with tile specs: tile_id(optional), origin_x, origin_y",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/l3_urban/tiles"))
    parser.add_argument("--origin-crs", type=str, default="EPSG:4326", help="CRS of origin_x/origin_y in tile list.")
    parser.add_argument("--tile-size-m", type=float, default=256.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--height-field", type=str, default="height_m")
    parser.add_argument("--all-touched", action="store_true", help="Use rasterize(all_touched=True).")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _load_catalog(path: Path) -> gpd.GeoDataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Catalog {path} has no CRS.")
    if gdf.empty:
        raise ValueError(f"Catalog {path} is empty.")
    return gdf


def _load_tile_specs(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError("JSON tile list must be an array of objects.")
    else:
        raise ValueError("tile-list format must be .csv or .json")

    required = {"origin_x", "origin_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tile-list missing required columns: {missing}")
    return df


def _stable_tile_id(origin_x: float, origin_y: float, origin_crs: str) -> str:
    token = f"{origin_x:.10f}|{origin_y:.10f}|{origin_crs}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    return f"tile_{digest}"


def _local_aeqd_crs(lon: float, lat: float) -> CRS:
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")


def _prepare_shapes(
    gdf_local: gpd.GeoDataFrame,
    tile_bbox_local,
    height_field: str,
) -> tuple[list[tuple[Any, float]], list[tuple[Any, int]]]:
    clipped = gdf_local.copy()
    clipped["geometry"] = clipped.geometry.intersection(tile_bbox_local)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty]
    if clipped.empty:
        return [], []

    clipped = clipped[np.isfinite(clipped[height_field]) & (clipped[height_field] > 0)]
    if clipped.empty:
        return [], []

    height_shapes = [
        (geom, float(h))
        for geom, h in zip(clipped.geometry, clipped[height_field])
        if geom is not None and not geom.is_empty
    ]
    # Using replace with ascending height order yields max-height rasterization in overlaps.
    height_shapes.sort(key=lambda x: x[1])
    occ_shapes = [(geom, 1) for geom, _ in height_shapes]
    return height_shapes, occ_shapes


def _rasterize_tile(
    height_shapes: list[tuple[Any, float]],
    occ_shapes: list[tuple[Any, int]],
    tile_size_m: float,
    resolution_m: float,
    all_touched: bool,
) -> tuple[np.ndarray, np.ndarray]:
    tile_px = int(round(tile_size_m / resolution_m))
    if tile_px <= 0:
        raise ValueError("Invalid tile size / resolution.")

    out_shape = (tile_px, tile_px)
    affine = from_origin(0.0, tile_size_m, resolution_m, resolution_m)

    if height_shapes:
        h = rasterize(
            height_shapes,
            out_shape=out_shape,
            transform=affine,
            fill=0.0,
            all_touched=all_touched,
            dtype="float32",
            merge_alg=MergeAlg.replace,
        )
        occ = rasterize(
            occ_shapes,
            out_shape=out_shape,
            transform=affine,
            fill=0,
            all_touched=all_touched,
            dtype="uint8",
            merge_alg=MergeAlg.replace,
        )
    else:
        h = np.zeros(out_shape, dtype=np.float32)
        occ = np.zeros(out_shape, dtype=np.uint8)
    return h, occ


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    catalog = _load_catalog(args.catalog)
    if args.height_field not in catalog.columns:
        raise ValueError(f"Height field '{args.height_field}' not found in catalog columns.")
    specs = _load_tile_specs(args.tile_list)
    origin_crs = CRS.from_user_input(args.origin_crs)

    to_wgs84 = Transformer.from_crs(origin_crs, CRS.from_epsg(4326), always_xy=True)
    tile_bbox_local = box(0.0, 0.0, args.tile_size_m, args.tile_size_m)

    args.output_root.mkdir(parents=True, exist_ok=True)
    sindex = catalog.sindex
    built = 0

    for row in specs.to_dict(orient="records"):
        origin_x = float(row["origin_x"])
        origin_y = float(row["origin_y"])
        tile_id = str(row.get("tile_id") or _stable_tile_id(origin_x, origin_y, args.origin_crs))

        lon, lat = to_wgs84.transform(origin_x, origin_y)
        local_crs = _local_aeqd_crs(lon, lat)

        to_local = Transformer.from_crs(catalog.crs, local_crs, always_xy=True).transform
        to_catalog = Transformer.from_crs(local_crs, catalog.crs, always_xy=True).transform

        tile_bbox_catalog = transform(to_catalog, tile_bbox_local).envelope
        cand_idx = list(sindex.intersection(tile_bbox_catalog.bounds))
        if cand_idx:
            subset = catalog.iloc[cand_idx]
            subset = subset[subset.intersects(tile_bbox_catalog)]
            subset_local = subset.to_crs(local_crs)
        else:
            subset_local = gpd.GeoDataFrame(columns=catalog.columns, geometry="geometry", crs=local_crs)

        height_shapes, occ_shapes = _prepare_shapes(
            gdf_local=subset_local,
            tile_bbox_local=tile_bbox_local,
            height_field=args.height_field,
        )
        h, occ = _rasterize_tile(
            height_shapes=height_shapes,
            occ_shapes=occ_shapes,
            tile_size_m=args.tile_size_m,
            resolution_m=args.resolution_m,
            all_touched=args.all_touched,
        )

        out_dir = args.output_root / tile_id
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "H.npy", h)
        np.save(out_dir / "Occ.npy", occ)

        meta = {
            "tile_id": tile_id,
            "origin": {"x": origin_x, "y": origin_y, "crs": args.origin_crs},
            "tile_size_m": args.tile_size_m,
            "resolution_m": args.resolution_m,
            "shape": [int(h.shape[0]), int(h.shape[1])],
            "local_crs_wkt": local_crs.to_wkt(),
            "affine": [1.0, 0.0, 0.0, 0.0, -1.0, args.tile_size_m],
            "num_shapes": len(height_shapes),
            "bbox_catalog_crs": list(tile_bbox_catalog.bounds),
        }
        (out_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        built += 1

        if built % 50 == 0:
            LOGGER.info("Built %d tiles...", built)

    LOGGER.info("Done. Built %d tile caches under %s", built, args.output_root)


if __name__ == "__main__":
    main()
