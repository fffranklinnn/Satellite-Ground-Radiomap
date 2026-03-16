# scripts 目录说明

本目录提供项目级运行入口，负责把 `src/` 能力组织成可直接执行的任务（单次仿真、拼接、批量实验、后处理与诊断）。

## 1. 目录角色

- 统一命令行入口，减少手工调用底层模块的复杂度
- 沉淀可复现实验流程（参数、输出结构、日志）
- 面向论文图、批处理数据集与工程诊断的不同场景

## 2. 当前文件清单

### 核心仿真与拼接

- `generate_full_radiomap.py`
  - 单时刻、单位置全物理叠加图（L1+L2+L3）
- `generate_l1l2_radiomap.py`
  - L1+L2 组合仿真与可视化
- `generate_xian_city_radiomap.py`
  - 西安 tile 级拼接；支持 Top-K 卫星与数据集 prototype 导出
- `generate_xian_full_from_shapefile.py`
  - 从 shp 到 tile list/cache/城市图的一键流水线
- `generate_shaanxi_radiomap.py`
  - 省域 L2 拼接（带进度与 ETA）
- `generate_multisat_timeseries_radiomap.py`
  - 多星长时序帧生成（best-server/soft-combine）

### 批量实验与参数扫描

- `batch_city_experiments.py`
  - 时间/频率/雨率批量扫描，生成实验索引
- `batch_generate_all.py`
  - 预设任务全集批量运行脚本

### 专题图与展示

- `generate_feature_showcase.py`
  - 地形/大气/radiomap 三类展示图
- `generate_global_map.py`
  - 全球分布图
- `generate_global_comparison.py`
  - 多城市对比图
- `generate_l1_map.py`
  - L1 时序与参数变化展示
- `generate_beam_dwell.py`
  - 波束凝视/驻留场景分析

### 数据校验、后处理与诊断

- `check_data_integrity.py`
  - 运行前数据依赖检查
- `report_satellite_visibility.py`
  - 可见星统计报表
- `postprocess_xian_timeseries.py`
  - 时序结果后处理（拼图、GIF）
- `visualize_batch.py`
  - 批量将 `.npy` 渲染为 PNG
- `debug_sinr.py`
  - 多星 SINR 调试脚本
- `test_multisat_sinr.py`
  - 多星 SINR 快速验证脚本

### 辅助文件

- `__init__.py`
  - 包导入占位
- `generate_xian_city_radiomap.py.pre_p1_backup`
  - 历史备份文件（非主运行入口）

## 3. 常用命令

### 单次全物理仿真

```bash
python scripts/generate_full_radiomap.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --show-decomposition
```

### 西安城市级拼接

```bash
python scripts/generate_xian_city_radiomap.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --output-dir output/xian_city_radiomap
```

### 多星长时序

```bash
python scripts/generate_multisat_timeseries_radiomap.py \
  --config configs/mission_config.yaml \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T03:00:00 \
  --step-minutes 30 \
  --fusion-mode best-server \
  --max-satellites 6
```

### 批量参数扫描

```bash
python scripts/batch_city_experiments.py \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T23:00:00 \
  --step-hours 1 \
  --freqs-ghz 14.5 \
  --rain-rates 0,10 \
  --workers 4
```

### 数据检查

```bash
python scripts/check_data_integrity.py \
  --config configs/mission_config.yaml \
  --strict
```

## 4. 输出约定

- 大多数脚本默认将结果写入 `output/` 下的任务子目录。
- 时序脚本通常输出 `png/`、`npy/`、`frame_json/` 与 `manifest.jsonl`。
- 批处理脚本通常输出 `experiment_index.csv/json` 便于回溯。

## 5. 维护建议

1. 入口脚本只负责编排，核心算法尽量下沉到 `src/`。
2. 新脚本加入后，必须同步更新本 README 的“文件清单”。
3. 历史备份脚本建议显式标注 `backup`/`legacy`，避免误用。
