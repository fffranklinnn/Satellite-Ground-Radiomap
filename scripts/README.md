# scripts — 运行脚本

独立可执行脚本，用于生成无线电地图和批量仿真。

## `generate_l1_map.py`

生成 L1 宏观层无线电地图，支持多时刻批量输出。

**功能：**
- 以北京（39.9042°N, 116.4074°E）为原点，256 km 覆盖范围，10 GHz，550 km LEO
- 计算 2025-01-01 四个时刻（00:00、06:00、12:00、18:00 UTC）的损耗图
- 输出每个时刻的独立 PNG 图像
- 输出 2×2 对比面板图

**运行方式：**

```bash
python scripts/generate_l1_map.py
```

**输出文件（`output/` 目录）：**

```
output/
├── l1_map_20250101_0000.png
├── l1_map_20250101_0600.png
├── l1_map_20250101_1200.png
├── l1_map_20250101_1800.png
└── l1_map_comparison.png
```

**依赖数据（可选，缺失时使用默认值）：**
- `data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz` — IONEX
- `data/l1_space/data/data_stream-oper_stepType-instant.nc` — ERA5
- `data/2025_0101.tle` — TLE

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
