# tools 目录说明

本目录提供 L3 数据预处理、cache 构建和可视化工具脚本。

## 1. 脚本清单

- `preprocess_buildings_catalog.py`
  - 将原始建筑 `shp` 规范化为统一 catalog（`.parquet` / `.gpkg`）
- `build_l3_tile_cache.py`
  - 基于 catalog + tile list 生成每个 tile 的 `H.npy` / `Occ.npy` / `meta.json`
- `visualize_l3_tile.py`
  - 单 tile 或小范围 mosaic 的 2D/3D 可视化
- `batch_visualize_l3_tiles.py`
  - 按 tile list 批量导出可视化结果
- `visualize_l3_city_overview.py`
  - 从 tile cache 生成城市级总览图（高度/占用/覆盖）

## 2. 常用命令

建筑数据预处理：

```bash
python tools/preprocess_buildings_catalog.py \
  --input-root data/l3_urban/shanxisheng/陕西省 \
  --output data/l3_urban/xian/catalog/buildings_xian.parquet
```

构建 L3 tile cache：

```bash
python tools/build_l3_tile_cache.py \
  --catalog data/l3_urban/xian/catalog/buildings_xian.parquet \
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv \
  --output-root data/l3_urban/xian/tiles_60
```

查看单 tile：

```bash
python tools/visualize_l3_tile.py \
  --tile-id xian_000001 \
  --tiles-root data/l3_urban/xian/tiles_60 \
  --output-dir output/l3_tile_preview
```

批量可视化：

```bash
python tools/batch_visualize_l3_tiles.py \
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv \
  --tiles-root data/l3_urban/xian/tiles_60 \
  --output-dir output/l3_tiles_batch_preview
```

城市级总览：

```bash
python tools/visualize_l3_city_overview.py \
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv \
  --tiles-root data/l3_urban/xian/tiles_60 \
  --output-dir output/l3_city_overview
```

## 3. 与主流程关系

- 主流程（`main.py` / `scripts/*`）运行时仅读取 tile cache，不直接读取原始 `shp`。
- 推荐先用本目录脚本完成离线构建，再将 `layers.l3_urban.tile_cache_root` 指向对应 cache。
