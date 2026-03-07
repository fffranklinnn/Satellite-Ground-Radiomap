# 快速入门

## 1. 环境准备

```bash
pip install -r requirements.txt
python main.py --help
```

推荐先确认以下关键文件存在：

- `data/2025_0101.tle`
- `data/l2_topo/全国DEM数据.tif`（L2）
- `data/l3_urban/xian/tiles_60/`（L3）
- `data/l1_space/data/*.INX.gz`（可选，IONEX）
- `data/l1_space/data/*.nc`（可选，ERA5）

## 2. 运行一次完整仿真

```bash
python main.py --config configs/mission_config.yaml --output output/
```

默认配置：

- 区域中心：西安（34.3416, 108.9398）
- 时间范围：2025-01-01 全天，步长 6 小时
- 频率：14.5 GHz
- 启用 L1/L2/L3

## 3. 输出结果说明

主程序典型输出：

```text
output/
├── composite_0000.png
├── composite_0000.npy
├── l1_macro_0000.png
├── l2_topo_0000.png
├── l3_urban_0000.png
└── comparison_0000.png
```

其中 `.npy` 为可复用数组，`.png` 为可视化。

## 4. 常用脚本入口

单时刻全效应图：

```bash
python scripts/generate_full_radiomap.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00
```

西安可见星统计：

```bash
python scripts/report_satellite_visibility.py \
  --config configs/mission_config.yaml \
  --start 2025-01-01T00:00:00 \
  --end 2025-01-01T23:00:00 \
  --step-hours 1 \
  --output-csv output/visibility/xian_20250101_hourly.csv
```

功能展示图（地形/大气/radiomap 三类）：

```bash
python scripts/generate_feature_showcase.py \
  --config configs/mission_config.yaml \
  --timestamp 2025-01-01T06:00:00 \
  --output-root output/feature_showcase_demo
```

数据完整性检查（推荐先执行）：

```bash
python main.py --config configs/mission_config.yaml --check-data-only
python scripts/check_data_integrity.py --config configs/mission_config.yaml --strict
```

## 5. 最常见回退行为

- IONEX 缺失：L1 使用配置中的默认 TEC（`tec`）
- ERA5 缺失：L1 使用简化大气模型
- DEM 缺失：L2 返回零损耗
- L3 tile 缺失：L3 计算会失败（需要可用 cache 或关闭 L3）

## 6. 典型问题

`PROJ` 版本警告可通过指定 `PROJ_DATA` 处理：

```bash
export PROJ_DATA=$(python -c "import pyproj; print(pyproj.datadir.get_data_dir())")
```

L3 cache 不存在时，先参考 [../data/l3_urban/README.md](../data/l3_urban/README.md) 构建。
