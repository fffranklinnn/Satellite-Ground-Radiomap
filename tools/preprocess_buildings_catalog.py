"""
Offline preprocessing: raw shapefiles -> normalized building catalog.
离线预处理：原始 shapefile -> 规范化建筑 catalog。

Purpose / 作用:
- Scan regional shapefiles and normalize them into a unified building dataset.
- 扫描区域 shp 数据并规范化成统一建筑库。

Main capabilities / 核心功能:
- Validate sidecar files (`.shx/.dbf/.prj/.cpg`) per `.shp`.
- 校验每个 `.shp` 的配套侧车文件是否完整。
- Repair invalid geometries and keep Polygon/MultiPolygon only.
- 修复无效几何，仅保留 Polygon/MultiPolygon。
- Detect and map height field to standardized `height_m`.
- 自动识别并映射建筑高度字段到标准 `height_m`。
- Reproject to target CRS and export GeoParquet / GeoPackage.
- 重投影到目标 CRS，导出 GeoParquet / GeoPackage。

Interfaces / 接口:
- CLI entrypoint: `python tools/preprocess_buildings_catalog.py ...`
- 命令行入口：`python tools/preprocess_buildings_catalog.py ...`

Relationship / 与其他脚本关系:
- Output catalog is consumed by `tools/build_l3_tile_cache.py`.
- 输出 catalog 供 `tools/build_l3_tile_cache.py` 使用。
- Later visualization reads tile cache created from this catalog.
- 后续可视化基于该 catalog 构建出的 tile cache。

Example / 调用示例:
```powershell
conda run -n radiodiff python tools/preprocess_buildings_catalog.py `
  --input-root shanxisheng `
  --output data/l3_urban/catalog/buildings.parquet `
  --target-crs EPSG:4326 `
  --height-candidates height_m,pred_h_r,height,Height
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

try:
    from shapely.validation import make_valid as _make_valid
except ImportError:  # pragma: no cover
    _make_valid = None


LOGGER = logging.getLogger("preprocess_buildings_catalog")
POLYGON_TYPES = {"Polygon", "MultiPolygon"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize building shapefiles into a unified GeoParquet/GeoPackage catalog."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root folder containing .shp files.")
    parser.add_argument("--output", type=Path, required=True, help="Output catalog path (.parquet or .gpkg).")
    parser.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:4326",
        help="Target CRS for normalized catalog. Default: EPSG:4326.",
    )
    parser.add_argument(
        "--height-candidates",
        type=str,
        default="height_m,pred_h_r,height,height_m_,building_h,hgt,z,floor",
        help="Comma-separated candidate height field names.",
    )
    parser.add_argument(
        "--default-height",
        type=float,
        default=None,
        help="Fallback height (meters) for rows where height is missing.",
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=0.5,
        help="Drop geometries smaller than this area (computed in EPSG:3857).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def find_shapefiles(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.shp") if path.is_file())


def sidecar_complete(shp_path: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for suffix in (".shx", ".dbf", ".prj", ".cpg"):
        if not shp_path.with_suffix(suffix).exists():
            missing.append(suffix)
    return len(missing) == 0, missing


def _polygonal_only(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in POLYGON_TYPES:
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in POLYGON_TYPES and not g.is_empty]
        if not polys:
            return None
        return unary_union(polys)
    return None


def _repair_geometry(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None:
        return None
    fixed = _make_valid(geom) if _make_valid is not None else geom.buffer(0)
    return _polygonal_only(fixed)


def _detect_height_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _standardize_one(
    shp_path: Path,
    target_crs: str,
    height_candidates: list[str],
    default_height: Optional[float],
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry", "height_m", "source_name"], geometry="geometry", crs=target_crs)
    if gdf.crs is None:
        raise ValueError(f"{shp_path} has no CRS metadata.")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf["geometry"] = gdf.geometry.apply(_repair_geometry)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.geom_type.isin(POLYGON_TYPES)]
    gdf = gdf[~gdf.geometry.is_empty]

    if gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry", "height_m", "source_name"], geometry="geometry", crs=target_crs)

    height_col = _detect_height_column(gdf.columns, height_candidates)
    if height_col is None and default_height is None:
        raise ValueError(
            f"{shp_path} does not contain any height column in candidates {height_candidates} "
            "and --default-height was not provided."
        )

    if height_col is None:
        height_m = pd.Series(default_height, index=gdf.index, dtype=np.float32)
    else:
        height_m = pd.to_numeric(gdf[height_col], errors="coerce").astype(np.float32)
        if default_height is not None:
            height_m = height_m.fillna(default_height)

    valid_height = height_m.notna() & np.isfinite(height_m) & (height_m > 0)
    gdf = gdf.loc[valid_height].copy()
    height_m = height_m.loc[valid_height]
    gdf["height_m"] = height_m.astype(np.float32)
    gdf["source_name"] = shp_path.stem

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    if min_area_m2 > 0 and not gdf.empty:
        area_geo = gdf.to_crs("EPSG:3857")
        keep = area_geo.geometry.area >= float(min_area_m2)
        gdf = gdf.loc[keep].copy()

    return gdf[["geometry", "height_m", "source_name"]]


def save_catalog(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        gdf.to_parquet(output_path, index=False)
        return
    if suffix == ".gpkg":
        gdf.to_file(output_path, driver="GPKG", layer="buildings")
        return
    raise ValueError("Unsupported output format. Use .parquet or .gpkg.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    shp_files = find_shapefiles(args.input_root)
    if not shp_files:
        raise FileNotFoundError(f"No .shp files found under {args.input_root}")

    height_candidates = [x.strip() for x in args.height_candidates.split(",") if x.strip()]
    chunks: list[gpd.GeoDataFrame] = []
    skipped = 0

    for shp in shp_files:
        ok, missing = sidecar_complete(shp)
        if not ok:
            skipped += 1
            LOGGER.warning("Skip %s due to missing sidecars: %s", shp, ",".join(missing))
            continue

        try:
            part = _standardize_one(
                shp_path=shp,
                target_crs=args.target_crs,
                height_candidates=height_candidates,
                default_height=args.default_height,
                min_area_m2=args.min_area_m2,
            )
        except Exception as exc:
            skipped += 1
            LOGGER.exception("Failed on %s: %s", shp, exc)
            continue

        if not part.empty:
            chunks.append(part)

    if not chunks:
        raise RuntimeError("No valid building records were produced.")

    merged = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), geometry="geometry", crs=args.target_crs)
    merged["building_id"] = np.arange(1, len(merged) + 1, dtype=np.int64)
    merged = merged[["building_id", "geometry", "height_m", "source_name"]]

    save_catalog(merged, args.output)
    LOGGER.info(
        "Catalog saved: %s (records=%d, skipped_shapefiles=%d)",
        args.output,
        len(merged),
        skipped,
    )


if __name__ == "__main__":
    main()
