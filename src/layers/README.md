# src/layers — 物理层实现

三层多尺度无线电地图架构，每层独立计算对应尺度的电磁损耗，通过聚合引擎合并。

## 层级概览

| 层 | 类 | 覆盖范围 | 分辨率 | 网格 | 状态 |
|----|----|---------|--------|------|------|
| L1 宏观层 | `L1MacroLayer` | 256 km × 256 km | 1000 m/pixel | 256×256 | ✅ 完整实现 |
| L2 地形层 | `L2TopoLayer` | 25.6 km × 25.6 km | 100 m/pixel | 256×256 | ✅ 完整实现 |
| L3 城市层 | `L3UrbanLayer` | 256 m × 256 m | 1 m/pixel | 256×256 | ✅ 完整实现 |

---

## 统一接口

所有层继承 `BaseLayer`，实现统一接口：

```python
def compute(self, origin_lat=None, origin_lon=None,
            timestamp=None, context=None, **kwargs) -> np.ndarray:
    """返回 256×256 float32 损耗数组，单位 dB。"""
```

构造函数：`Layer(config: dict, origin_lat: float, origin_lon: float)`

---

## L1 宏观层 (`l1_macro.py`)

卫星到地面的宽域电磁损耗，基于 Skyfield TLE 轨道传播。

**计算内容：**
- TLE 选星：遍历星座选择最高仰角卫星
- 自由空间路径损耗（FSPL）
- 31×31 相控阵天线增益（高斯波束模型）
- 极化失配损耗
- 仰角 < 5° 的像素标记为无覆盖 (200 dB)

**数据依赖：**
- `data/2025_0101.tle` — Starlink TLE（必需）
- `data/l1_space/data/*.INX.gz` — IONEX TEC（可选，回退到默认 10 TECU）
- `data/l1_space/data/*.nc` — ERA5 pressure-level 数据（q/z/r/t，可选，回退到简化大气模型）

---

## L2 地形层 (`l2_topo.py`)

DEM 地形遮挡与衍射损耗，基于 rasterio 窗口读取。

**计算内容：**
- GeoTIFF DEM 窗口读取 + 双线性重采样（30m → 100m/px）
- 向量化累积最大值 LOS 遮挡分析（纯 numpy，无 Python 循环）
- 遮挡像素固定 20 dB 衍射损耗（V2.0: ITU-R P.526）

**数据依赖：**
- `data/l2_topo/全国DEM数据.tif` — 全国 DEM（15~57°N, 73~139°E）

**配置：**
```yaml
l2_topo:
  enabled: true
  dem_file: "data/l2_topo/全国DEM数据.tif"
  satellite_elevation_deg: 45.0
  satellite_azimuth_deg: 180.0
```

---

## L3 城市层 (`l3_urban.py`)

建筑尺度 NLoS 遮挡损耗，基于预构建的 tile cache。

**计算内容：**
- 从 tile cache 加载建筑高度栅格 (H.npy) 和占用掩码 (Occ.npy)
- 方向性 NLoS 扫描（平行波假设，按入射方向逐射线扫描）
- NLoS 像素 20 dB + 建筑占用像素 30 dB 损耗

**数据依赖：**
- `data/l3_urban/shanxisheng/陕西省/*.shp` — 陕西省原始建筑矢量
- `data/l3_urban/xian/tiles_60/` — 可直接运行的西安 tile cache（由 `tools/build_l3_tile_cache.py` 构建）

**Tile cache 构建流程：**
```bash
# 1. 预处理建筑 shapefile → parquet
python tools/preprocess_buildings_catalog.py \
  --input-root data/l3_urban/shanxisheng/陕西省 \
  --output data/l3_urban/xian/catalog/buildings_xian.parquet

# 2. 构建 tile cache
python tools/build_l3_tile_cache.py \
  --catalog data/l3_urban/xian/catalog/buildings_xian.parquet \
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv \
  --output-root data/l3_urban/xian/tiles_60/
```

说明：仓库当前现成 cache 主要是西安。若需要陕西省其他城市，需要从 `shanxisheng/陕西省/` 原始 shp 生成对应城市的 tile list 与 tile cache。

---

## 导出接口

```python
from src.layers import BaseLayer, L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.layers.base import LayerContext
```
