# src/layers — 物理层实现

三层多尺度无线电地图架构，每层独立计算对应尺度的电磁损耗，通过聚合引擎合并。

## 层级概览

| 层 | 类 | 覆盖范围 | 分辨率 | 网格 | V1.0 状态 |
|----|----|---------|--------|------|-----------|
| L1 宏观层 | `L1MacroLayer` | 256 km × 256 km | 1000 m/pixel | 256×256 | ✅ 完整实现 |
| L2 地形层 | `L2TopoLayer` | 25.6 km × 25.6 km | 100 m/pixel | 256×256 | ⏳ 占位（零损耗） |
| L3 城市层 | `L3UrbanLayer` | 256 m × 256 m | 1 m/pixel | 256×256 | ⏳ 占位（零损耗） |

---

## 接口规范

所有层继承 `BaseLayer`，实现统一接口：

```python
def compute(self, timestamp: datetime) -> np.ndarray:
    """
    返回 256×256 的损耗数组，单位 dB。
    值越大表示损耗越高（信号越弱）。
    """
```

构造函数签名：

```python
Layer(config: dict, origin_lat: float, origin_lon: float)
```

---

## L1 宏观层（`l1_macro.py`）

计算卫星到地面的宽域电磁损耗，完全向量化。

**计算内容：**
- 自由空间路径损耗（FSPL）
- 大气衰减：有 ERA5 数据时用 `atmospheric_loss_era5()`，否则用简化模型
- 电离层损耗：有 IONEX 数据时用实测 TEC，否则用默认值 10 TECU
- 卫星几何：V1.0 固定天顶方向；TLE 加载器已集成，V2.0 启用动态定位

**配置参数：**

```yaml
frequency_ghz: 10.0
satellite_altitude_km: 550.0
ionex_file: data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz  # 可选
era5_file: data/l1_space/data/data_stream-oper_stepType-instant.nc          # 可选
tle_file: data/2025_0101.tle                                                 # 可选
```

---

## L2 地形层（`l2_topo.py`）— V2.0 待实现

**V1.0**：无 DEM 数据时始终返回零损耗数组。

**V2.0 规划：**
- 加载 GeoTIFF / SRTM HGT 格式 DEM 数据
- 视线（LOS）分析：逐像素检查卫星方向是否被地形遮挡
- 刃形衍射损耗（ITU-R P.526）

**配置参数：**

```yaml
dem_file: data/l2_topo/N39E116.hgt   # V2.0 需要
satellite_elevation_deg: 45.0
```

---

## L3 城市层（`l3_urban.py`）— V2.0 待实现

**V1.0**：无建筑数据时始终返回零损耗数组。

**V2.0 规划：**
- 从 Shapefile 加载建筑轮廓和高度
- 建筑阴影计算（基于卫星方位角/仰角）
- GPU 光线追踪多径效应

**配置参数：**

```yaml
building_shapefile: data/l3_urban/buildings.shp   # V2.0 需要
satellite_azimuth_deg: 180.0
satellite_elevation_deg: 45.0
```

---

## 导出接口

```python
from src.layers import BaseLayer, L1MacroLayer, L2TopoLayer, L3UrbanLayer
```
