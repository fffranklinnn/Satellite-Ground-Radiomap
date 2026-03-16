# SG-MRM 项目实现快照

更新时间：2026-03-04

## 1. 代码能力范围

| 模块 | 文件 | 当前状态 | 说明 |
|---|---|---|---|
| L1 宏观层 | `src/layers/l1_macro.py` | 可用 | TLE 选星、FSPL、波束、IONEX/ERA5 增强、可选 Faraday |
| L2 地形层 | `src/layers/l2_topo.py` | 可用 | DEM 窗口读取、方向遮挡、衍射损耗 |
| L3 城市层 | `src/layers/l3_urban.py` | 可用 | tile cache 加载、NLoS 与占用损耗 |
| 聚合引擎 | `src/engine/aggregator.py` | 可用 | L1/L2 插值到 L3 尺度后 dB 叠加 |
| 数据加载器 | `src/utils/*_loader.py` | 可用 | IONEX/ERA5/TLE 解析与插值 |

## 2. 数据现状

| 数据类型 | 路径 | 可用性 |
|---|---|---|
| TLE | `data/2025_0101.tle` | 可用 |
| IONEX | `data/l1_space/data/*.INX.gz` | 可用 |
| ERA5 pressure-level | `data/l1_space/data/*.nc` | 可用 |
| ERA5 single-level 脚本 | `data/l1_space/data/cds_single_level.py` | 可用（脚本） |
| DEM | `data/l2_topo/china_dem_30m.tif` | 可用 |
| L3 原始 shp | `data/l3_urban/shanxisheng/陕西省/*.shp` | 可用 |
| L3 可运行 cache | `data/l3_urban/xian/tiles_60/` | 可用（西安） |

## 3. 结构检查结果（冗余与优化）

### 已识别的结构噪声

1. 根目录存在 `branch_L1/branch_L2/branch_L3` 历史快照，容易与 `src/` 混淆。
2. `output/` 结果体量大，若不按任务分目录，后期难以追溯。
3. `data/` 同时存放下载脚本、样例文件和超大原始数据，需要更明确的数据治理约定。

### 当前仓库已具备的治理基础

- `.gitignore` 已排除 `output/`、`branch_*`、大体量 data 文件。
- 核心运行代码集中在 `src/`。

### 建议执行策略（非破坏性）

1. 将 `src/` 作为唯一运行时真源。
2. 将 `branch_*` 视为仅本地备份，后续可归档到单独目录或仓库。
3. 批量任务统一落在 `output/<task_name>/`，并保留 `experiment_index.*`。
4. 数据下载流程统一通过 `data/l1_space/data/*.py` 脚本记录来源。

## 4. 当前边界

- 现有模型工程上可用，但未完全覆盖 ITU 严格统计链。
- L3 全省级精细建模取决于各城市 cache 是否构建齐全。
- GPU 并行目前主要是任务调度层绑定，核心内核仍以 CPU 为主。

## 5. 建议优先级

1. 文档与代码接口持续同步（已完成一轮 README 统一）。
2. 补全 L2/L3 边界测试与关键脚本 smoke test。
3. 若要提升 Ka/更高频段可信度，优先打通 ERA5 single-level 到主流程的统计映射链。
