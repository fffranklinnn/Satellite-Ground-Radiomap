# SG-MRM Project Summary

本文件是当前仓库（`Satellite-Ground-Radiomap`）的实现快照，重点同步“代码能力”和“数据现状”。

## 当前实现范围

| 层 | 文件 | 当前状态 | 核心能力 |
|----|------|----------|----------|
| L1 宏观层 | `src/layers/l1_macro.py` | ✅ 已实现 | TLE 选星、FSPL、相控阵增益、极化损耗、IONEX/ERA5 加载 |
| L2 地形层 | `src/layers/l2_topo.py` | ✅ 已实现 | DEM 窗口读取、LOS 遮挡判定、地形损耗栅格 |
| L3 城市层 | `src/layers/l3_urban.py` | ✅ 已实现 | Tile cache 加载、方向性 NLoS 扫描、占用损耗 |
| 聚合引擎 | `src/engine/aggregator.py` | ✅ 已实现 | L1/L2/L3 多尺度插值与 dB 合成 |

## 数据现状（仓库内）

| 数据类型 | 路径 | 说明 |
|---------|------|------|
| TLE | `data/2025_0101.tle` | Starlink 星座轨道数据（2025-01-01） |
| IONEX | `data/l1_space/data/*.INX.gz` | 电离层 TEC |
| ERA5 | `data/l1_space/data/*.nc` | pressure-level 数据（q/z/r/t） |
| DEM | `data/l2_topo/全国DEM数据.tif` | L2 地形遮挡输入 |
| 建筑原始 shp | `data/l3_urban/shanxisheng/陕西省/*.shp` | 陕西省多城市原始建筑矢量 |
| 建筑缓存 | `data/l3_urban/xian/tiles_60/` | 当前可直接运行的西安 L3 tile cache |

说明：L3 的陕西省原始建筑数据覆盖范围大于当前可直接运行的 cache 范围。若扩展到其他城市，需先构建对应 tile cache。

## 目录概览

```text
Satellite-Ground-Radiomap/
├── configs/               # 配置文件
├── data/                  # 原始数据与预处理缓存
├── docs/                  # 说明文档
├── examples/              # 示例脚本
├── scripts/               # 批处理/可视化脚本
├── src/                   # 核心实现
│   ├── core/
│   ├── layers/
│   ├── engine/
│   └── utils/
├── tests/
├── main.py
└── README.md
```

## 常用入口

- 主文档：`README.md`
- 快速开始：`docs/QUICKSTART.md`
- 数据说明：`data/README.md`
- L3 数据与缓存构建：`data/l3_urban/README.md`
- 脚本说明：`scripts/README.md`

## 已知边界

- 仓库中“现成可跑”的 L3 cache 主要是西安；陕西省其他城市需先从原始 shp 构建 cache。
- 年尺度/省尺度批量实验可运行，但计算量大，建议按时间片和区域分批执行并使用进度监控脚本。

## 更新日期

- Last updated: 2026-03-03
