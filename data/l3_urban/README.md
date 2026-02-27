# L3 Urban Layer Data

L3 城市层数据目录，存放建筑高度栅格 tile cache 用于 NLoS 遮挡计算。

## 当前数据（西安市区）

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
- **高度范围**: 3~122 m

## Tile cache 构建流程

```bash
# 1. 预处理 shapefile → parquet 目录
python tools/preprocess_buildings_catalog.py \
  --input-root data/l3_urban/shanxisheng/ \
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
