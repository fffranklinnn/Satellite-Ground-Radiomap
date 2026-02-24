# L2 Terrain Layer Data

L2 地形层数据目录，用于 DEM（数字高程模型）地形遮挡计算。

**当前状态**：V1.0 中 L2 层为占位实现，此目录暂无数据文件。V2.0 将启用完整地形计算。

---

## 预期数据格式（V2.0）

### GeoTIFF (`.tif`)

- 来源：[ASTER GDEM](https://asterweb.jpl.nasa.gov/gdem.asp)、[Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)
- 分辨率：30 m 或 90 m
- 坐标系：WGS84

### SRTM HGT (`.hgt`)

- 来源：[USGS Earth Explorer](https://earthexplorer.usgs.gov/)、[NASA SRTM](https://www2.jpl.nasa.gov/srtm/)
- 分辨率：30 m（SRTM1）或 90 m（SRTM3）
- 文件命名：`N{lat}E{lon}.hgt`，例如 `N39E116.hgt`（覆盖北京区域）

---

## 数据获取

```bash
# 下载北京区域 SRTM 瓦片（需要 USGS 账号）
# 覆盖范围：N39°-N40°, E116°-E117°
wget "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N39E116.SRTMGL1.hgt.zip"
unzip N39E116.SRTMGL1.hgt.zip -d data/l2_topo/
```

---

## 配置方式（V2.0）

在 `configs/mission_config.yaml` 中指定 DEM 文件路径：

```yaml
layers:
  l2:
    enabled: true
    dem_file: data/l2_topo/N39E116.hgt
    satellite_elevation_deg: 45.0
```

---

## V2.0 计划功能

- DEM 数据加载与重采样到 100 m/pixel 网格（25.6 km 覆盖）
- 逐像素视线（LOS）分析：检查卫星方向是否被地形遮挡
- 刃形衍射损耗计算（ITU-R P.526）
- 地形剖面提取（Bresenham 直线算法）
