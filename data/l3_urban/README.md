# L3 Urban Layer Data

L3 城市层数据目录，存放建筑高度栅格 tile cache 用于 NLoS 遮挡计算。

## 当前数据（陕西省原始建筑 + 西安缓存）

### `shanxisheng/陕西省/` — 陕西省原始建筑 shp 数据

- **数据形态**: 原始建筑矢量数据（`.shp/.dbf/.shx/.prj/.cpg`）
- **规模**: 70 套 shapefile（覆盖陕西省内多数城市区域）
- **坐标系**: 以 `EPSG:3857` 为主
- **用途**: 作为 L3 的原始输入，先预处理为 catalog，再构建 tile cache

### `xian/tiles_60/` — 西安市区 tile cache

- **tile 数量**: 1320 tiles（40×33 网格）
- **覆盖范围**: 34.22°~34.31°N, 108.90°~108.99°E（西安市区核心区）
- **tile 尺寸**: 256 m × 256 m，分辨率 1 m/pixel
- **每个 tile 包含**:
  - `H.npy` — 建筑高度栅格 (256×256 float32, 米)
  - `Occ.npy` — 建筑占用掩码 (256×256 bool)
  - `meta.json` — tile 元数据（origin 坐标、CRS 等）

### `xian/catalog/buildings_xian.parquet` — 建筑目录

- **来源**: 陕西省 70 个 shapefile（EPSG:3857）
- **记录数**: 359,691 栋建筑
- **标准字段**: `building_id`, `geometry`, `height_m`, `source_name`
- **高度范围**: 约 3~105 m

## Tile cache 构建流程

```bash
# 1. 预处理 shapefile → parquet 目录
python tools/preprocess_buildings_catalog.py \
  --input-root data/l3_urban/shanxisheng/陕西省 \
  --output data/l3_urban/xian/catalog/buildings_xian.parquet

# 2. 构建 tile cache
python tools/build_l3_tile_cache.py \
  --catalog data/l3_urban/xian/catalog/buildings_xian.parquet \
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv \
  --output-root data/l3_urban/xian/tiles_60/
```

## 配置

```yaml
layers:
  l3_urban:
    enabled: true
    tile_cache_root: "data/l3_urban/xian/tiles_60"
    nlos_loss_db: 20.0
    occ_loss_db: 30.0
    incident_dir:
      az_deg: 180.0
      el_deg: 45.0
```

## 扩展到其他城市

1. 准备建筑 shapefile（需包含 Height 字段）
2. 生成 tile list CSV（`origin_x,origin_y` 格式，EPSG:4326）
3. 运行上述构建流程，修改 `--output-root` 路径
4. 更新 `mission_config.yaml` 中的 `tile_cache_root`

## 说明

- 当前仓库中可直接用于 L3 计算的 cache 主要是西安（`xian/tiles_60/`）。
- 若要扩展到陕西省其他城市，需要基于 `shanxisheng/陕西省/` 原始 shp 重新生成对应城市的 `tile_list` 和 `tile cache`。
