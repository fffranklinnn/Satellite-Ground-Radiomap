# data 目录说明

本目录管理 SG-MRM 的原始数据、示例数据与下载脚本入口。

## 1. 目录结构

```text
data/
├── 2025_0101.tle
├── l1_space/
│   └── data/
├── l2_topo/
└── l3_urban/
```

## 2. 按层的数据依赖

### L1（星地宏观层）

必需：

- `data/2025_0101.tle`（或你自己的 TLE）

可选增强：

- IONEX：`data/l1_space/data/*.INX.gz`
- ERA5 pressure-level：`data/l1_space/data/*.nc`
- ERA5 single-level（雨衰/云雾扩展准备）：同目录下载结果

当前仓库内下载脚本：

- `data/l1_space/data/NASAcddis.py`：IONEX 批量下载
- `data/l1_space/data/cds.py`：ERA5 pressure-level 下载请求
- `data/l1_space/data/cds_pressure_batch.py`：ERA5 pressure-level 批量下载（按月/按日）
- `data/l1_space/data/cds_single_level.py`：ERA5 single-level 下载请求

### L2（地形层）

- `data/l2_topo/全国DEM数据.tif`

说明：L2 通过窗口读取，不会一次性把全国 DEM 全部加载到内存。

### L3（城市层）

- 原始建筑矢量：`data/l3_urban/shanxisheng/陕西省/*.shp`
- 可运行缓存（西安）：`data/l3_urban/xian/tiles_60/`

说明：L3 主流程需要 tile cache，直接读取原始 shp 不参与在线计算。

## 3. 当前数据完整性（基于仓库现状）

| 项 | 现状 | 说明 |
|---|---|---|
| TLE | 有 | 2025-01-01 数据可用 |
| IONEX | 有 | 可用于 TEC 查询 |
| ERA5 pressure-level | 有 | 可提取 IWV |
| ERA5 single-level | 脚本有 | 变量下载脚本已提供，是否下载由本地数据决定 |
| DEM | 有 | 全国覆盖 |
| L3 原始 shp | 有 | 陕西省范围 |
| L3 多城市 cache | 部分 | 现成 cache 主要是西安 |

## 4. 数据准备建议

1. 以 `configs/mission_config.yaml` 为准，保证路径可解析。
2. 所有大文件保持在 `data/` 下并使用 `.gitignore` 规则。
3. 若做省域/全年批量，优先按时间片和区域片分批下载、分批计算。

## 5. 获取示例

首次使用 ERA5 脚本前，先安装依赖：

```bash
pip install cdsapi
```

TLE（示例）：

```bash
wget https://celestrak.com/NORAD/elements/starlink.txt -O data/2025_0101.tle
```

ERA5 pressure-level 下载请求（需要 CDS 账号和 API key）：

```bash
python data/l1_space/data/cds.py
```

ERA5 pressure-level 批量下载（推荐）：

```bash
python data/l1_space/data/cds_pressure_batch.py \
  --year 2025 --months 1-12 --chunk-mode month \
  --area 40,106,32,112 \
  --output-dir data/l1_space/data/era5_pressure_levels_2025
```

ERA5 single-level 下载请求：

```bash
python data/l1_space/data/cds_single_level.py
```

IONEX 批量下载：

```bash
python data/l1_space/data/NASAcddis.py
```

## 6. 相关文档

- L1 数据细节：[l1_space/README.md](l1_space/README.md)
- L2 DEM 细节：[l2_topo/README.md](l2_topo/README.md)
- L3 建筑数据与 cache：[l3_urban/README.md](l3_urban/README.md)
