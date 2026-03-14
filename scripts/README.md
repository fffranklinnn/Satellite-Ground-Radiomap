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

### 使用这些 sample 时，应该如何理解

#### 第一优先级：物理一致性

在把 pilot 或正式数据集用于训练/分析之前，优先检查：

1. `composite == l1 + l2 + l3`
2. 条件轴变化是否符合物理直觉：
   - 只改 `rain_rate_mm_h` 时，主要应表现为 L1 侧变化
   - 只改 `satellite_norad_id` 时，主要应表现为几何/方向相关变化
   - 只改 `timestamp_utc` 时，主要应表现为过境几何与环境变化
3. 各层语义是否清晰：
   - `l1` = 大尺度宏观贡献
   - `l2` = 地形贡献
   - `l3` = 城市建筑贡献

#### 第二优先级：图像大小与物理分辨率

当前 prototype / pilot 的 tile-level sample 应统一理解为：

- image size: `256 × 256`
- physical footprint: `256 m × 256 m`
- effective resolution: **`1 m/px`**

虽然 `l1` 和 `l2` 来自更粗尺度模型，但它们在导出时已经对齐到了 L3 tile grid，因此当前 `.npz` 中所有数组都应按 tile-level sample 来解释。

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


## 8. 物理一致性检查流程（可执行清单）

在启动大规模数据生成前，建议先对当前 `output/datasets/sgmrm_v1/pilot/` 做一轮自动检查 + 一轮人工检查。

### 8.1 自动检查（结构与导出正确性）

目标：确认当前样本不是“格式对了但内容写乱了”。

#### 自动检查项

1. `composite == l1 + l2 + l3`
2. 所有数组 shape 一致，且与 `grid_shape` 一致
3. `occ` 仍是二值 `{0,1}`
4. manifest 中的 `npz_path` / `preview_png_path` 可解析到真实文件
5. `.npz` 固定包含：
   - `composite`
   - `l1`
   - `l2`
   - `l3`
   - `height`
   - `occ`

#### 推荐执行方式（当前 pilot）

```bash
python - << 'PY'
import json
from pathlib import Path
import numpy as np

root = Path('output/datasets/sgmrm_v1/pilot')
manifest = root / 'manifest.jsonl'
records = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
required = {'composite','l1','l2','l3','height','occ'}
errors = []
for rec in records:
    npz_path = root / rec['npz_path']
    preview_path = root / rec['preview_png_path']
    if not npz_path.exists():
        errors.append((rec['sample_id'], 'missing_npz'))
        continue
    if not preview_path.exists():
        errors.append((rec['sample_id'], 'missing_preview'))
    data = np.load(npz_path)
    if set(data.files) != required:
        errors.append((rec['sample_id'], 'bad_keys', sorted(data.files)))
        continue
    comp = data['composite']
    l1 = data['l1']
    l2 = data['l2']
    l3 = data['l3']
    occ = data['occ']
    if comp.shape != (256, 256):
        errors.append((rec['sample_id'], 'bad_shape', comp.shape))
    if list(comp.shape) != rec['grid_shape']:
        errors.append((rec['sample_id'], 'grid_shape_mismatch'))
    if np.max(np.abs(comp - (l1 + l2 + l3))) > 1e-4:
        errors.append((rec['sample_id'], 'composite_not_sum'))
    if not set(np.unique(occ).tolist()).issubset({0, 1}):
        errors.append((rec['sample_id'], 'occ_not_binary'))
print('records', len(records))
print('errors', len(errors))
for e in errors[:20]:
    print('ERROR', e)
PY
```

#### 放行条件

- `errors = 0`
- 没有缺文件 / 缺数组键 / shape mismatch / `composite_not_sum`

---

### 8.2 条件轴单变量一致性检查（半自动）

目标：确认“只改一个条件轴时，变化符合物理直觉”。

#### A. Rain sweep

固定：
- `tile_id`
- `timestamp_utc`
- `satellite_norad_id`
- `frequency_ghz`

只改：
- `rain_rate_mm_h`

##### 预期
- 主变化应来自 `l1`
- `l2` 应保持不变
- `l3` 应保持不变
- `height/occ` 应保持不变

##### 推荐检查

```bash
python - << 'PY'
import numpy as np
from pathlib import Path
root = Path('output/datasets/sgmrm_v1/pilot/samples')
base = np.load(root/'sgmrm_v1__tile-xian_idx000001__ts-20250101T060000Z__sat-51863__f-14p5__rain-0.npz')
r10 = np.load(root/'sgmrm_v1__tile-xian_idx000001__ts-20250101T060000Z__sat-51863__f-14p5__rain-10.npz')
r25 = np.load(root/'sgmrm_v1__tile-xian_idx000001__ts-20250101T060000Z__sat-51863__f-14p5__rain-25.npz')
print('l2_diff_r10', float(np.max(np.abs(base['l2'] - r10['l2']))))
print('l3_diff_r10', float(np.max(np.abs(base['l3'] - r10['l3']))))
print('height_diff_r10', float(np.max(np.abs(base['height'] - r10['height']))))
print('occ_diff_r10', int(np.max(np.abs(base['occ'] - r10['occ']))))
print('l2_diff_r25', float(np.max(np.abs(base['l2'] - r25['l2']))))
print('l3_diff_r25', float(np.max(np.abs(base['l3'] - r25['l3']))))
PY
```

##### 放行条件
- `l2/l3/height/occ` 对 rain sweep 基本不变
- `composite` 变化可由 `l1` 变化解释

---

#### B. Satellite sweep

固定：
- `tile_id`
- `timestamp_utc`
- `rain_rate_mm_h`
- `frequency_ghz`

只改：
- `satellite_norad_id`

##### 预期
- `l1` 应变化明显
- `l2/l3` 可以因方向变化而变化
- `height/occ` 必须保持不变

##### 推荐检查
- 看 manifest 里的 `satellite_elevation_deg` / `satellite_azimuth_deg` 是否明显变化
- 检查 `height/occ` 是否完全一致

---

#### C. Timestamp sweep

固定：
- `tile_id`
- `satellite_norad_id`
- `rain_rate_mm_h`
- `frequency_ghz`

只改：
- `timestamp_utc`

##### 预期
- 变化应主要由过境几何变化解释
- `height/occ` 必须保持不变

##### 推荐检查
- 看 manifest 中同一星的 elevation / azimuth 是否随时间变化
- 检查 `height/occ` 是否完全一致

---

### 8.3 人工检查（看图）

自动检查只能保证结构正确，不能代替物理判断。

建议至少做三组人工 spot-check：

1. **Rain sweep 看图**
   - 只改 rain 时，整体强弱应变化，但城市结构纹理不应无故跳变
2. **Satellite sweep 看图**
   - 换星后，宏观方向性和局部遮挡关系可以变化，但建筑底图语义不应漂移
3. **Timestamp sweep 看图**
   - 同一星过境前后，变化应与几何变化匹配，而不是随机噪声

### 8.4 当前 pilot 的推荐验收门槛

在继续进入大规模生成前，建议满足：

- 自动检查 `errors = 0`
- rain / satellite / timestamp 三类 sweep 都做过至少一次 spot-check
- 已明确 sample 语义：
  - image size = `256 × 256`
  - physical footprint = `256 m × 256 m`
  - effective resolution = `1 m/px`
- 已明确这是一种 **tile-aligned multi-component physical sample**，而不是三张原始尺度图直接打包
