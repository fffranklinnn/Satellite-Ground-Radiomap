# data/l3_urban 说明

L3 城市层使用“原始建筑矢量 + 预构建 tile cache”两段式数据流程。

## 1. 当前目录角色

### `shanxisheng/陕西省/`

- 内容：陕西省范围建筑 shapefile（多城市）
- 作用：原始来源数据，不直接参与在线 L3 计算

### `xian/tiles_60/`

- 内容：西安可运行 tile cache（`H.npy`, `Occ.npy`, `meta.json`）
- 作用：L3 运行时直接读取

### `xian/catalog/buildings_xian.parquet`

- 作用：构建 cache 过程中的中间目录数据

## 2. 为什么要有 cache

L3 每个 tile 计算需要 1m 分辨率的 256x256 栅格输入。直接在运行时解析 shp 开销高且不稳定，因此采用离线构建：

1. shp -> parquet catalog
2. catalog + tile list -> tile cache
3. 在线仿真只读 cache

## 3. 构建流程

### 步骤 1：预处理目录

```bash
python tools/preprocess_buildings_catalog.py \
  --input-root data/l3_urban/shanxisheng/陕西省 \
  --output data/l3_urban/xian/catalog/buildings_xian.parquet
```

### 步骤 2：生成 tile cache

```bash
python tools/build_l3_tile_cache.py \
  --catalog data/l3_urban/xian/catalog/buildings_xian.parquet \
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv \
  --output-root data/l3_urban/xian/tiles_60/
```

## 4. 运行配置

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

## 5. 扩展到陕西省其他城市

1. 从 `shanxisheng/陕西省/` 选出目标城市范围
2. 生成该城市 tile list
3. 构建该城市 cache
4. 将 `tile_cache_root` 指向新 cache

## 6. 已知边界

- 仓库内现成可直接跑的大规模 cache 主要是西安。
- 省域级 L3 全量拼接依赖各城市 cache 是否完备。
