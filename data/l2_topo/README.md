# L2 Terrain Layer Data

L2 地形层数据目录，存放 DEM（数字高程模型）文件用于地形遮挡计算。

## 当前数据

### `全国DEM数据.tif` — 全国数字高程模型

- **格式**: GeoTIFF
- **覆盖范围**: 15°~57°N, 73°~139°E（全国）
- **分辨率**: ~0.000278°（约 30 m）
- **像素尺寸**: 237600 × 151200
- **坐标系**: WGS84

## L2 层处理流程

1. 根据 origin 坐标计算 25.6 km × 25.6 km 窗口范围
2. 通过 rasterio 窗口读取，双线性重采样到 256×256（100 m/px）
3. 向量化累积最大值 LOS 遮挡分析
4. 遮挡像素施加 20 dB 衍射损耗

## 配置

```yaml
layers:
  l2_topo:
    enabled: true
    dem_file: "data/l2_topo/全国DEM数据.tif"
    satellite_elevation_deg: 45.0
    satellite_azimuth_deg: 180.0
```

## 其他 DEM 数据源

如需替换或补充 DEM 数据：

- [SRTM](https://www2.jpl.nasa.gov/srtm/)：30 m / 90 m，`.hgt` 格式
- [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)：30 m GeoTIFF
- [ASTER GDEM](https://asterweb.jpl.nasa.gov/gdem.asp)：30 m GeoTIFF
