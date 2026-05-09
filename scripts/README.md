# scripts 目录说明

本目录提供项目级运行入口，负责把 `src/` 能力组织成可直接执行的任务（单次仿真、拼接、批量实验、后处理与诊断）。

## 1. 目录角色

- 统一命令行入口，减少手工调用底层模块的复杂度
- 沉淀可复现实验流程（参数、输出结构、日志）
- 面向论文图、批处理数据集与工程诊断的不同场景

## 2. 当前文件清单

### 推荐主入口

- `main.py`
  - 主仿真入口，支持 `--config`、`--output`、`--strict-data`、`--check-data-only`
- `scripts/check_data_integrity.py`
  - 独立的数据依赖检查
- `scripts/report_satellite_visibility.py`
  - 可见星统计报表
- `scripts/generate_multisat_timeseries_radiomap.py`
  - 多星长时序帧生成（best-server / soft-combine）

### 兼容包装器

- `scripts/generate_full_radiomap.py`
- `scripts/generate_feature_showcase.py`
- `scripts/generate_l1l2_radiomap.py`
- `scripts/generate_xian_city_radiomap.py`
- `scripts/generate_beam_dwell.py`
- `scripts/batch_generate_all.py`

以上文件目前都是 thin wrapper，实际实现已迁移到 `scripts/legacy/`。

### 其他脚本

- `batch_city_experiments.py`
- `generate_shaanxi_radiomap.py`
- `generate_xian_full_from_shapefile.py`
- `run_scene_smoke.py`
- `run_xian_urban_smoke.py`
- `run_qinling_smoke.py`
- `run_huashan_smoke.py`
- `run_loess_plateau_smoke.py`
- `generate_global_map.py`
- `generate_global_comparison.py`
- `generate_l1_map.py`
- `postprocess_xian_timeseries.py`
- `visualize_batch.py`
- `debug_sinr.py`
- `test_multisat_sinr.py`

## 3. 常用命令

### 主流程

```bash
python main.py \
  --config configs/mission_config.yaml \
  --output output/
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
python main.py --config configs/mission_config.yaml --check-data-only
python scripts/check_data_integrity.py \
  --config configs/mission_config.yaml \
  --strict
```

### 场景 smoke test

```bash
python scripts/run_xian_urban_smoke.py --output-root output/verify --gpu-id 0
python scripts/run_qinling_smoke.py --output-root output/verify --gpu-id 1 --max-frames 10
python scripts/run_huashan_smoke.py --start 2025-05-01T12:00:00 --end 2025-05-01T12:04:30
python scripts/run_loess_plateau_smoke.py --step-minutes 1.0
```

每次 smoke test 会在场景输出目录下额外保存：

- `run_config.yaml`
- `run_manifest.json`
- `preset_config.yaml`

## 4. 输出约定

- 大多数脚本默认将结果写入 `output/` 下的任务子目录。
- 时序脚本通常输出 `png/`、`npy/`、`frame_json/` 与 `manifest.jsonl`。
- 批处理脚本通常输出 `experiment_index.csv/json` 便于回溯。

## 5. 维护建议

1. 入口脚本只负责编排，核心算法尽量下沉到 `src/`。
2. 新脚本加入后，必须同步更新本 README 的“文件清单”。
3. 历史备份脚本建议显式标注 `backup`/`legacy`，避免误用。
