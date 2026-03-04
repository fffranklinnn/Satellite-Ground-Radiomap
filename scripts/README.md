# scripts — 运行脚本

独立可执行脚本，用于生成无线电地图和批量仿真。

## `generate_l1_map.py`

生成 L1 宏观层无线电地图，包含四类输出用于物理建模验证。

**功能：**
- 全天24小时逐小时损耗图（4×6 面板）
- 降雨率参数扫描（0 / 10 / 50 / 100 mm/h 对比 + 差值图）
- 频率参数扫描（1 / 3 / 10 / 30 GHz 对比）
- 损耗分量分解图（FSPL / 大气 / 电离层 / 总损耗）

**运行方式：**

```bash
conda run -n sgmrm_test python scripts/generate_l1_map.py
```

**输出文件：**

```
output/l1_maps/
├── hourly/          — 24张逐小时图 + 4×6面板
├── rain_scan/       — 降雨率扫描面板 + 差值面板
├── freq_scan/       — 频率扫描面板
└── components/      — 分量分解图 + 对比面板
```

**依赖数据（可选，缺失时使用默认值）：**
- `data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz` — IONEX
- `data/l1_space/data/data_stream-oper_stepType-instant.nc` — ERA5
- `data/2025_0101.tle` — TLE

---

## `generate_global_comparison.py`

在全球6个典型地区生成 256 km × 256 km 的 L1 损耗图，展示 IONEX TEC 和 ERA5 IWV 的地理差异。

**地区：** 北京、新加坡（赤道）、莫斯科（高纬）、北极圈、圣保罗（南大西洋异常区）、迪拜（干燥）

**运行方式：**

```bash
conda run -n sgmrm_test python scripts/generate_global_comparison.py
```

**输出文件：**

```
output/global_comparison/
├── {city}_total.png        — 6张总损耗图
├── {city}_components.png   — 6张分量分解图
└── global_comparison_panel.png  — 2×3 对比面板
```

---

## `generate_global_map.py`

生成真正的全球无线电地图（等经纬度投影，0.5° 分辨率，720×360 像素）。

直接调用 `physics.py` 函数，不经过 `L1MacroLayer`，使各物理效应的全球空间分布清晰可见。

**运行方式：**

```bash
conda run -n sgmrm_test python scripts/generate_global_map.py
```

**输出文件：**

```
output/global_map/
├── global_fspl.png    — 全球自由空间路径损耗
├── global_atm.png     — 全球大气损耗
├── global_iono.png    — 全球电离层损耗（IONEX TEC）
├── global_total.png   — 全球总损耗
└── global_panel.png   — 2×2 对比面板
```

---

## `generate_full_radiomap.py`

在单一时刻、单一地点生成“全物理效应叠加”的高分辨率 radiomap（论文插图风格）。

叠加内容：
- L1：FSPL + 大气损耗（ERA5）+ 电离层损耗（IONEX）+ 天线增益/极化
- L2：地形遮挡/衍射损耗（支持任意方位角）
- L3：建筑 NLoS 与占用损耗

**运行方式：**

```bash
python scripts/generate_full_radiomap.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --show-decomposition
```

指定卫星集合（可重复 `--norad-id`）：

```bash
python scripts/generate_full_radiomap.py \
  --norad-id 25544 \
  --norad-id 44713
```

**输出文件：**

```
output/full_radiomap/
├── full_radiomap_*.png                 — 主图（300 DPI）
├── full_radiomap_*_decomposition.png   — 分量分解（可选）
├── full_radiomap_*_composite.npy
├── full_radiomap_*_l1.npy
├── full_radiomap_*_l2.npy
├── full_radiomap_*_l3.npy
├── full_radiomap_*_fspl.npy
├── full_radiomap_*_atm.npy
├── full_radiomap_*_iono.npy
└── ...mask.npy
```

---

## `generate_xian_city_radiomap.py`

将西安 `tile_list_xian_60.csv` 的全部 tile（默认 1320 个）拼接为一张城市级全物理效应 radiomap。

说明：
- 每个 tile 使用 L3 建筑损耗 + L2 地形损耗（按卫星方位角）；
- L1 宏观层在城市中心计算一次，并作为 tile 级基线模板叠加；
- 输出一张全市大图（约 10240×8448 像素）和 `composite.npy`。
- 当前该脚本直接消费西安 cache（`data/l3_urban/xian/tiles_60/`）；陕西省原始 shp 位于 `data/l3_urban/shanxisheng/陕西省/`，需先构建对应城市 cache 才能用于 L3 计算。

**运行方式（建议在 `sgmrm_test` 环境）：**

```bash
conda run -n sgmrm_test python scripts/generate_xian_city_radiomap.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --output-dir output/xian_city_radiomap \
  --dpi 300
```

调试小规模（先跑前 20 tiles）：

```bash
conda run -n sgmrm_test python scripts/generate_xian_city_radiomap.py \
  --max-tiles 20 --dpi 120
```

---

## `generate_shaanxi_radiomap.py`

生成陕西省尺度的地形损耗 radiomap（L2，基于 DEM 的遮挡/衍射损耗拼接）。

说明：
- 省域范围默认从 `data/l3_urban/shanxisheng/陕西省/*.shp` 自动推断；
- 按 `25.6 km` tile 步长拼接全省，输出省级大图（默认约 `9472×5120`）；
- 终端实时显示进度/速度/ETA，适合长任务观察。

**运行方式：**

默认参数（`el=45°`，遮挡较轻）：

```bash
conda run -n sgmrm_test python scripts/generate_shaanxi_radiomap.py \
  --timestamp 2025-01-01T06:00:00 \
  --output-dir output/shaanxi_radiomap
```

增强地形对比（推荐，低仰角）：

```bash
conda run -n sgmrm_test python scripts/generate_shaanxi_radiomap.py \
  --timestamp 2025-01-01T06:00:00 \
  --sat-elevation-deg 12 \
  --sat-azimuth-deg 180 \
  --output-prefix shaanxi_l2_terrain_el12 \
  --output-dir output/shaanxi_radiomap
```

调试小规模：

```bash
conda run -n sgmrm_test python scripts/generate_shaanxi_radiomap.py \
  --max-tiles 20 --dpi 120
```

---

## `generate_feature_showcase.py`

一键生成“功能展示图”脚本，输出分为三类目录：

- `terrain/`：地形 DEM 加载、遮挡掩码、L2 地形损耗
- `atmosphere/`：ERA5 IWV 加载、不同雨率下的大气损耗与差值
- `radiomap/`：一个或多个可见卫星在指定频段/区域的 radiomap

脚本会在输出根目录生成汇总文件：`*_showcase_summary.json`。

**运行方式：**

```bash
conda run -n sgmrm_test python scripts/generate_feature_showcase.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --output-root output/feature_showcase_demo \
  --output-tag xian_demo \
  --top-k-sats 3 \
  --rain-rates 0,10,25
```

如需增强地形遮挡效果（推荐低仰角）：

```bash
conda run -n sgmrm_test python scripts/generate_feature_showcase.py \
  --terrain-elevation-deg 12 \
  --terrain-azimuth-deg 180
```

---

## `batch_city_experiments.py`

面向“西安城市拼接图”的批量实验脚本。自动做时间序列和参数扫描，并记录实验索引（CSV/JSON）。

**支持的扫描维度：**
- 时间：`--start --end --step-hours`
- 频率：`--freqs-ghz`（逗号分隔）
- 雨率：`--rain-rates`（逗号分隔）

**核心特性：**
- 每个实验自动生成独立配置快照（保存在 `output/.../configs/`）
- 自动调用 `generate_xian_city_radiomap.py`
- 输出实验索引：`experiment_index.csv` / `experiment_index.json`
- 支持断点续跑：`--skip-existing`
- 支持调试：`--max-tiles`、`--dry-run`
- 支持并发运行：`--workers`
- 支持 GPU 卡位分配：`--gpu-ids 0,1,2,3`（按实验轮转设置 `CUDA_VISIBLE_DEVICES`）

说明：
- 当前 SG-MRM 主体计算仍以 CPU (numpy/scipy) 为主；`--gpu-ids` 主要用于并发任务的 GPU 绑定调度（便于后续接入 GPU 内核/依赖）。

**运行示例：**

1) 全天小时级（固定 14.5GHz、无雨）：

```bash
conda run -n sgmrm_test python scripts/batch_city_experiments.py \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T23:00:00 \
  --step-hours 1 \
  --freqs-ghz 14.5 \
  --rain-rates 0 \
  --output-root output/city_batch_day1
```

2) 参数扫描（2个频率 × 3个雨率 × 4个时刻）：

```bash
conda run -n sgmrm_test python scripts/batch_city_experiments.py \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T18:00:00 \
  --step-hours 6 \
  --freqs-ghz 10,14.5 \
  --rain-rates 0,10,25 \
  --workers 4 \
  --gpu-ids 0,1,2,3 \
  --max-tiles 50 \
  --output-root output/city_batch_sweep
```

---

## `report_satellite_visibility.py`

统计某一地点在时间区间内的可见卫星数量和 Top-K 卫星（按仰角降序）。

**运行方式：**

```bash
conda run -n sgmrm_test python scripts/report_satellite_visibility.py \
  --config configs/mission_config.yaml \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T23:00:00 \
  --step-hours 1 \
  --min-elevation-deg 5 \
  --top-k 5 \
  --output-csv output/visibility/xian_20250101_hourly.csv
```

---

## `visualize_batch.py`

将批量实验目录中的 `.npy` radiomap 可视化为 PNG，便于快速浏览结果。

**运行方式：**

```bash
conda run -n sgmrm_test python scripts/visualize_batch.py \
  --batch-dir output/city_batch_day1
```

---

## `batch_generate_all.py`

按预定义任务集批量生成示例结果（支持选择任务子集和并行 worker）。

**运行方式：**

```bash
conda run -n sgmrm_test python scripts/batch_generate_all.py \
  --tasks G1 G2 G3 \
  --workers 2
```

---

## 使用配置文件运行完整仿真

```bash
python main.py --config configs/mission_config.yaml --output output/
```

`main.py` 支持的参数：

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（默认 `configs/mission_config.yaml`） |
| `--output` | 输出目录（默认 `output/`） |
