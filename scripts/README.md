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
conda run -n SatelliteRM python scripts/generate_l1_map.py
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
conda run -n SatelliteRM python scripts/generate_global_comparison.py
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
conda run -n SatelliteRM python scripts/generate_global_map.py
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

## 使用配置文件运行完整仿真

```bash
python main.py --config configs/mission_config.yaml --output output/
```

`main.py` 支持的参数：

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（默认 `configs/mission_config.yaml`） |
| `--output` | 输出目录（默认 `output/`） |
| `--profile` | 启用性能分析 |
