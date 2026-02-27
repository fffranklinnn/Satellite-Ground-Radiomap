# SG-MRM：星地多尺度无线电地图

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English README](README.md)

星地多尺度电磁损耗地图仿真系统，用于卫星到地面的无线电传播分析。

## 概述

SG-MRM 通过三层物理模型生成高分辨率电磁损耗图：

- **L1 宏观层** (256 km, 1000 m/px)：TLE 轨道传播选星、自由空间路径损耗、31×31 相控阵天线增益、大气衰减（ERA5 IWV）、电离层效应（IONEX TEC）
- **L2 地形层** (25.6 km, 100 m/px)：GeoTIFF DEM 加载与重采样、向量化累积最大值 LOS 遮挡分析、衍射损耗
- **L3 城市层** (256 m, 1 m/px)：建筑高度栅格 tile cache、方向性 NLoS 扫描、遮挡/占用损耗

各层输出 256×256 float32 dB 损耗矩阵，通过 dB 域叠加合成：

```
复合损耗 (dB) = Interp(L1) + Interp(L2) + L3
```

## 安装

```bash
pip install -r requirements.txt
```

主要依赖：numpy, scipy, matplotlib, pyyaml, skyfield, rasterio, geopandas, shapely, pyproj, pyarrow, pandas

## 快速开始

### 运行完整仿真

```bash
python main.py --config configs/mission_config.yaml --output output/
```

### 编程接口

```python
from datetime import datetime
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext
import yaml

config = yaml.safe_load(open('configs/mission_config.yaml'))
lat, lon = config['origin']['latitude'], config['origin']['longitude']

l1 = L1MacroLayer(config['layers']['l1_macro'], lat, lon)
l2 = L2TopoLayer(config['layers']['l2_topo'], lat, lon)
l3 = L3UrbanLayer(config['layers']['l3_urban'], lat, lon)

agg = RadioMapAggregator(l1, l2, l3)
ctx = LayerContext.from_any({'incident_dir': config['layers']['l3_urban']['incident_dir']})
composite = agg.aggregate(lat, lon, timestamp=datetime(2025, 1, 1, 6, 0, 0), context=ctx)
# composite: (256, 256) float32, dB
```

### 可视化脚本

```bash
python scripts/generate_l1_map.py              # L1 全天时序 + 参数扫描
python scripts/generate_global_comparison.py    # 全球六城市对比
python scripts/generate_global_map.py           # 全球损耗图 (720×360)
```

## 项目结构

```
Satellite-Ground-Radiomap/
├── configs/              # 仿真配置文件 (YAML)
├── data/
│   ├── 2025_0101.tle     # Starlink TLE 轨道数据
│   ├── l1_space/data/    # IONEX 电离层 + ERA5 气象数据
│   ├── l2_topo/          # 全国 DEM GeoTIFF
│   └── l3_urban/         # 建筑 tile cache (H.npy/Occ.npy)
├── src/
│   ├── core/             # 网格坐标系统 + RF 物理公式
│   ├── layers/           # L1/L2/L3 层实现
│   ├── engine/           # 多层聚合引擎
│   └── utils/            # 数据加载器、可视化、性能分析
├── tools/                # L3 tile cache 构建工具
├── scripts/              # 可视化与批量生成脚本
├── tests/                # 单元测试
├── examples/             # 使用示例
├── docs/                 # 架构文档
└── main.py               # 主入口
```

## 当前数据配置（西安 · 陕西省）

| 数据 | 文件 | 说明 |
|------|------|------|
| TLE | `data/2025_0101.tle` | Starlink 星座 14918 颗卫星 (2025-01-01) |
| IONEX | `data/l1_space/data/*.INX.gz` | UPC GIM 全球 TEC (15 分钟间隔) |
| ERA5 | `data/l1_space/data/*.nc` | ECMWF 气压层数据 (z/r/q/t) |
| DEM | `data/l2_topo/全国DEM数据.tif` | 全国 DEM (~30 m 分辨率) |
| 建筑 | `data/l3_urban/xian/tiles_60/` | 西安市区核心区 1320 tiles (256 m, 1 m/px) |

## 性能基准（西安，4 帧全天仿真）

| 阶段 | 耗时 | 峰值内存 |
|------|------|---------|
| L1 初始化（解析 14918 TLE） | 0.46 s | 31.0 MB |
| L1 compute（选星 + FSPL + 增益） | ~4.6 s | 6.7 MB |
| L2 compute（DEM 读取 + LOS） | ~0.003 s | 1.8 MB |
| L3 compute（tile 加载 + NLoS） | ~0.14 s | 2.1 MB |
| 单帧总计 | ~9.5 s | — |

## 开发路线

### V1.0（当前）✓

- [x] L1：TLE 选星 + FSPL + 相控阵增益 + 大气/电离层损耗
- [x] L2：GeoTIFF DEM 加载 + 向量化 LOS 遮挡 + 衍射损耗
- [x] L3：Tile cache 建筑栅格 + 方向性 NLoS 扫描 + 遮挡/占用损耗
- [x] 多层聚合引擎（dB 域叠加 + 双线性插值）
- [x] 数据加载器：IONEX、ERA5、TLE
- [x] 可视化工具 + 单元测试

### V2.0（规划）

- [ ] ITU-R P.526 Fresnel-Kirchhoff 刃形衍射模型 (L2)
- [ ] GPU 光线追踪多径效应 (L3)
- [ ] 完整 ITU-R P.618 雨衰模型
- [ ] 多日仿真：按日期自动切换 TLE/IONEX/ERA5 数据源
- [ ] 时序动画生成
- [ ] 并行计算支持

## 测试

```bash
pytest tests/
pytest --cov=src tests/
```

## 文档

- [架构设计](docs/architecture.md)
- [快速入门](docs/QUICKSTART.md)
- [English README](README.md)

## 许可证

MIT License - 详见 [LICENSE](LICENSE)
