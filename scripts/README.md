# scripts 说明

本目录包含可直接执行的实验脚本与图像生成脚本。

## 1. 主入口类型

### A. 单次结果生成

- `generate_full_radiomap.py`
  - 单时间、单位置，生成全物理叠加图
  - 可附带分量图与 `.npy` 导出
- `generate_multisat_timeseries_radiomap.py`
  - 多星、长时序逐帧生成（每帧多星融合）
  - 支持 `best-server` 与 `soft-combine` 融合
  - 固定输出：`png/` + `npy/` + `frame_json/` + `manifest.jsonl`
- `generate_global_map.py`
  - 直接基于物理函数生成全球分布图
- `generate_l1_map.py`
  - L1 时序图与参数扫描图

### B. 区域拼接

- `generate_xian_city_radiomap.py`
  - 按 tile list 拼接西安城市级图
- `generate_xian_full_from_shapefile.py`
  - 从西安 shapefile 自动生成全域 tile list + L3 cache + radiomap（一键）
- `generate_multisat_timeseries_radiomap.py`
  - 多卫星长时序数据集生成（PNG + NPY + frame JSON + manifest JSONL）
- `generate_shaanxi_radiomap.py`
  - 按省域网格拼接陕西 L2 地形图（带进度/ETA）

### C. 批量实验与可见星分析

- `batch_city_experiments.py`
  - 批量参数扫描（时间/频率/雨率）
  - 输出实验索引 CSV/JSON
  - 支持并发 worker 与 GPU ID 轮转绑定（调度层）
- `report_satellite_visibility.py`
  - 统计某地某时段可见星数量与 Top-K
- `check_data_integrity.py`
  - 按配置检查数据完整性（可 strict 模式）

### D. 展示与后处理

- `generate_feature_showcase.py`
  - 一次生成三类展示图：`terrain/`, `atmosphere/`, `radiomap/`
- `visualize_batch.py`
  - 将 batch 目录中的 `.npy` 转成 PNG 预览
- `postprocess_xian_timeseries.py`
  - 对西安小时级全域结果做后处理：分量帧图、时序大拼图、GIF
- `batch_generate_all.py`
  - 预设任务集批量跑图
- `generate_global_comparison.py`
  - 六城市对比图

## 2. 常用命令

### 全物理单图

```bash
python scripts/generate_full_radiomap.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --show-decomposition
```

### 多星长时序（逐帧融合）

```bash
python scripts/generate_multisat_timeseries_radiomap.py \
  --config configs/mission_config.yaml \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T03:00:00 \
  --step-minutes 30 \
  --fusion-mode best-server \
  --max-satellites 6 \
  --output-dir output/multisat_timeseries_demo
```

### 西安批量实验（含并发与 GPU 轮转）

```bash
python scripts/batch_city_experiments.py \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T23:00:00 \
  --step-hours 1 \
  --freqs-ghz 14.5 \
  --rain-rates 0,10 \
  --top-k-sats 10 \
  --sat-workers 8 \
  --workers 4 \
  --gpu-ids 0,1,2,3 \
  --output-root output/city_batch_demo
```

### 陕西省地形拼接（进度可见）

```bash
python scripts/generate_shaanxi_radiomap.py \
  --timestamp 2025-01-01T06:00:00 \
  --sat-elevation-deg 12 \
  --sat-azimuth-deg 180 \
  --output-dir output/shaanxi_radiomap
```

### 西安 shapefile 全域一键生成

```bash
python scripts/generate_xian_full_from_shapefile.py \
  --xian-shp data/l3_urban/shanxisheng/陕西省/processed_xian20221010_all.shp \
  --catalog data/l3_urban/xian/catalog/buildings_xian.parquet \
  --tile-size-m 1024 \
  --timestamp 2025-01-01T12:00:00 \
  --output-root output/xian_full_from_shp
```

### 多卫星长时序数据集（推荐）

```bash
python scripts/generate_multisat_timeseries_radiomap.py \
  --config configs/mission_config.yaml \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T23:00:00 \
  --step-minutes 60 \
  --fusion-mode soft-combine \
  --max-satellites 3 \
  --output-dir output/xian_multisat_timeseries
```

### 功能展示图

```bash
python scripts/generate_feature_showcase.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --output-root output/feature_showcase_demo \
  --output-tag xian_demo
```

### 数据完整性检查

```bash
python scripts/check_data_integrity.py \
  --config configs/mission_config.yaml \
  --strict \
  --output-json output/data_check/report.json
```

### 西安时序后处理（分量图 + 拼图 + GIF）

```bash
python scripts/postprocess_xian_timeseries.py \
  --batch-root output/xian_hourly_full_components_20250101 \
  --panel-cols 6 \
  --fps 2 \
  --output-dir output/xian_hourly_full_components_20250101/timeseries_visuals
```

## 3. 运行环境建议

- 推荐在 `sgmrm_test` 或等价环境运行。
- 大规模拼接/批量实验建议：
  1. 先用 `--max-tiles` 小规模验证。
  2. 再放开全量任务。
  3. 使用 `--skip-existing` 支持续跑。

## 4. 关于 GPU 参数

- `batch_city_experiments.py` 的 `--gpu-ids` 目前用于任务级 `CUDA_VISIBLE_DEVICES` 绑定。
- 当前核心计算仍以 `numpy/scipy` 为主，GPU 性能收益取决于你是否引入 GPU 内核。

## 5. 输出约定

不同脚本会在对应输出目录保存：

- 主要结果图（`.png`）
- 数组结果（`.npy`）
- 元信息索引（`.csv`/`.json`，仅批处理脚本）

建议将输出统一放在 `output/` 下，方便清理与归档。

`generate_multisat_timeseries_radiomap.py` 的输出结构固定为：

```text
<output-dir>/
  manifest.jsonl
  png/
    <prefix>_<frame_idx>_<timestamp>.png
  npy/
    <prefix>_<frame_idx>_<timestamp>.npy
  frame_json/
    <prefix>_<frame_idx>_<timestamp>.json
```

## 6. `generate_xian_city_radiomap.py` 数据集 prototype / pilot 输出

当使用：

```bash
python scripts/generate_xian_city_radiomap.py   --config configs/mission_config.yaml   --timestamp 2025-01-01T06:00:00   --max-tiles 1   --target-sat-id 51863   --dataset-prototype-out output/datasets/sgmrm_v1/pilot
```

脚本会在常规 PNG/NPY 城市输出之外，额外写出一个 tile-level prototype 样本：

```text
output/datasets/sgmrm_v1/pilot/
  manifest.jsonl
  logs/
  previews/
    <sample_id>.png
  samples/
    <sample_id>.npz
```

当前 `.npz` 固定包含：

- `composite`
- `l1`
- `l2`
- `l3`
- `height`
- `occ`

当前 `manifest.jsonl` 的推荐最小字段包括：

- `sample_id`
- `dataset_version`
- `split`（当前为 `pilot`）
- `sample_type`（当前为 `tile-level`）
- `source_script`
- `array_keys`
- `grid_shape`
- `tile_id`
- `timestamp_utc`
- `origin_lat` / `origin_lon`
- `frequency_ghz`
- `rain_rate_mm_h`
- `satellite_norad_id`
- `satellite_elevation_deg` / `satellite_azimuth_deg`
- `ionex_used` / `era5_used`
- `l2_padding_m`
- `npz_path` / `preview_png_path`
- `composite_mean` / `composite_std`

### 当前 pilot 的用途

这一路径适合做：

1. 单样本 exporter 验证
2. 小规模条件 sweep（rain / satellite / timestamp）
3. 后续正式数据集生成前的 schema 打磨

### 当前 pilot 约定

- 先使用**单 tile、单时间、单卫星**收口导出格式
- sweep 通过重复运行脚本累积到同一个 `manifest.jsonl`
- 建议保持 `sample_id` 可直接读出主要条件轴（tile / ts / sat / freq / rain）

## 7. SGMRM Dataset Spec v1（draft）

当前 draft 采用 **tile-level sample** 作为最小样本单位，目标是先把导出格式、命名规则、manifest 索引和 pilot sweep 约定收口，再扩展到正式数据集生成。

### 7.1 样本目录建议

```text
output/datasets/sgmrm_v1/
  pilot/
    manifest.jsonl
    logs/
    previews/
    samples/
  train/        # 预留
  val/          # 预留
  test/         # 预留
```

当前已落地的是 `pilot/`，后续正式 split 可以沿用同样的内部结构。

### 7.2 `sample_id` 命名约定

推荐保持：

```text
sgmrm_v1__tile-<tile_id>__ts-<timestamp>__sat-<norad>__f-<freq>__rain-<rain>
```

要求：
- 能直接读出主要条件轴
- 不依赖额外数据库即可做粗筛选
- 同一条件重复运行时，优先保持 deterministic 命名

### 7.3 `.npz` 约定

当前固定数组键：

- `composite`
- `l1`
- `l2`
- `l3`
- `height`
- `occ`

其中：
- `composite` 是主监督/主分析对象
- `l1/l2/l3` 保留可解释分量
- `height/occ` 保留 L3 几何上下文

### 7.4 `manifest.jsonl` 字段分层

#### 必填（当前应稳定存在）
- `sample_id`
- `dataset_version`
- `split`
- `sample_type`
- `source_script`
- `array_keys`
- `grid_shape`
- `tile_id`
- `timestamp_utc`
- `origin_lat`
- `origin_lon`
- `frequency_ghz`
- `rain_rate_mm_h`
- `satellite_norad_id`
- `satellite_elevation_deg`
- `satellite_azimuth_deg`
- `ionex_used`
- `era5_used`
- `l2_padding_m`
- `npz_path`
- `preview_png_path`

#### 统计型推荐字段
- `composite_mean`
- `composite_std`

#### 当前建议升格字段（建议在正式数据集前固定）
- `scenario_id`
- `split_reason`
- `condition_axes`
- `condition_groups`

说明：
- `scenario_id`：去掉数据集版本前缀后的稳定场景标识，适合跨 split / 跨批次追踪同一物理样本。
- `condition_axes`：当前样本参与了哪些 sweep 轴，例如 `rain_rate_mm_h`、`satellite_norad_id`、`timestamp_utc`。
- `condition_groups`：**列表字段**，因为同一个 baseline 样本可以同时属于多个 matched sweep 组，不能假定只有一个 group id。
- `split_reason`：记录样本为什么被放进某个 split，例如 `manual-pilot`、`tile-holdout`、`time-holdout`、`satellite-holdout`。

#### 其余后续可扩展字段（暂不强制）
- `tags`
- `notes`

### 7.5 Split / Group 规则（draft）

#### 当前规则
- `pilot/` 中所有样本的 `split = pilot`
- 当前 `split_reason` 默认可记为 `manual-pilot`
- pilot 阶段允许同一样本同时参与多个 sweep group，用于验证 schema 和条件轴设计

#### 正式数据集建议
1. **不要在 matched sweep 内部拆分 train/val/test**
   - 同一个 `condition_groups` 中的成员应整体进入同一个 split，避免信息泄漏。
2. **split 应优先按 group / scene 层做，而不是按单条 sample 随机切分**
   - 推荐至少在 `scenario_id` 或 `condition_groups` 层级做切分。
3. **优先采用较粗粒度的 holdout 规则**
   - 例如：按 tile、按时间窗、按 satellite、按城市做 holdout。
4. **pilot -> formal split 的过渡顺序**
   - 先在 `pilot/` 验证 exporter、manifest 和 group 语义
   - 再批量生成 `train/val/test/`
   - 最后补 `split_reason` 记录切分策略来源

### 7.6 Pilot sweep 约定

当前 pilot 已经验证三类最小条件轴：

1. `rain_rate_mm_h`
2. `satellite_norad_id`
3. `timestamp_utc`

推荐将 pilot 用作：
- exporter 正确性验证
- schema 打磨
- 条件轴设计验证
- 正式 train/val/test 生成前的 smoke test

### 7.6 当前边界

- 目前仍以 **单 tile 小样本** 为主，不代表正式大规模数据集已完成。
- `pilot/manifest.jsonl` 当前适合做 schema 验证，不宜直接当作最终训练索引。
- 后续如果扩展到多 tile / 多城市 / 多卫星 / 多时间，需要优先保证 `sample_id` 和 manifest 规则继续一致。
