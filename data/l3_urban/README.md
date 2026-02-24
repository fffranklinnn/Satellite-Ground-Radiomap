# L3 Urban Layer Data

L3 城市层数据目录，用于建筑阴影和多径效应计算。

**当前状态**：V1.0 中 L3 层为占位实现，无建筑数据时始终返回零损耗。V2.0 将启用完整城市传播计算。

---

## 预期数据格式（V2.0）

### 建筑 Shapefile (`.shp`)

- 来源：[OpenStreetMap](https://www.openstreetmap.org/)、本地 GIS 数据库、城市规划部门
- 必需属性：
  - `geometry`：建筑轮廓多边形（WGS84 坐标）
  - `height`：建筑高度（米）
- 可选属性：
  - `material`：建筑材料（`concrete`、`glass`、`metal`），用于 V2.0 反射系数计算
  - `floors`：楼层数

### 3D 城市模型（V2.0 GPU 光线追踪）

- 格式：OBJ (`.obj`)、glTF (`.gltf`)、CityGML
- 用于高精度多径效应仿真

---

## 数据获取

### 从 OpenStreetMap 提取建筑数据

```python
import osmnx as ox

# 下载指定区域的建筑数据（示例：北京朝阳区）
buildings = ox.geometries_from_place(
    "Chaoyang, Beijing, China",
    tags={'building': True}
)

# 保留必要字段并导出
buildings = buildings[['geometry', 'height']].dropna(subset=['geometry'])
buildings.to_file("data/l3_urban/buildings.shp")
```

---

## 配置方式（V2.0）

在 `configs/mission_config.yaml` 中指定建筑数据路径：

```yaml
layers:
  l3:
    enabled: true
    building_shapefile: data/l3_urban/buildings.shp
    satellite_azimuth_deg: 180.0
    satellite_elevation_deg: 45.0
```

---

## V2.0 计划功能

- Shapefile 加载与坐标投影到本地网格（256 m × 256 m，1 m/pixel）
- 建筑阴影计算（基于卫星方位角和仰角的投影遮挡）
- GPU 加速光线追踪（多径反射效应）
- 建筑材料反射系数建模
