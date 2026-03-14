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
  --data-format netcdf \
  --download-format unarchived \
  --output-dir data/l1_space/data/era5_pressure_levels_2025_nc
```

说明：

- 运行时 `layers.l1_macro.era5_file` 读取的是 NetCDF 文件路径（`.nc`）。
- `cds_pressure_batch.py` 默认 `--download-format zip`，会产生压缩包；压缩包不能直接作为 `era5_file`，需要先解压并指向解压后的 `.nc`。
- 如果你更关注“可直接跑”，建议如上显式使用 `--download-format unarchived`。

ERA5 single-level 下载请求：

```bash
python data/l1_space/data/cds_single_level.py
```

IONEX 批量下载：

```bash
python data/l1_space/data/NASAcddis.py
```

## 6. `output/datasets/sgmrm_v1/` 目录约定（draft）

除原始输入数据外，当前项目已经开始在 `output/datasets/sgmrm_v1/` 下整理 tile-level prototype / pilot 样本。

推荐结构：

```text
output/datasets/sgmrm_v1/
  pilot/
    manifest.jsonl
    logs/
    previews/
    samples/
  train/
  val/
  test/
```

说明：
- `pilot/`：用于 exporter、schema 和条件轴验证
- `train/val/test/`：预留给后续正式数据集切分
- 当前 arrays 以 `.npz` 为主，预览图以 `.png` 为辅，索引使用 `manifest.jsonl`

当前 pilot 已验证的小规模条件轴包括：
- rain sweep
- satellite sweep
- timestamp sweep

正式数据集生成前，建议先保证：
1. `sample_id` 命名规则稳定
2. `manifest.jsonl` 字段一致
3. 数组键集合不随脚本运行随机漂移
4. matched sweep 成员在 split 时保持同组，不被拆散

当前推荐的组语义是：
- `scenario_id`：稳定单样本场景标识
- `condition_axes`：该样本参与的条件轴列表
- `condition_groups`：该样本所属 matched sweep 组列表（注意是列表，不是单值）

## 7. 相关文档

- L1 数据细节：[l1_space/README.md](l1_space/README.md)
- L2 DEM 细节：[l2_topo/README.md](l2_topo/README.md)
- L3 建筑数据与 cache：[l3_urban/README.md](l3_urban/README.md)
- 脚本与 dataset prototype 说明：[../scripts/README.md](../scripts/README.md)
