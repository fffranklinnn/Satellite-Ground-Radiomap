# data/l2_topo 说明

L2 地形层的数据目录，当前主要使用全国 DEM GeoTIFF。

## 1. 当前数据

- 文件：`china_dem_30m.tif`
- 格式：GeoTIFF
- 覆盖：约 15°~57°N, 73°~139°E
- 分辨率：约 30 m（原始）

## 2. 在 L2 中的使用方式

L2 不会整图加载，而是按每个 tile 的地理窗口读取：

1. 根据 tile 原点计算 25.6 km x 25.6 km 范围
2. 窗口读取 DEM
3. 重采样到 256x256（100 m/px）
4. 进行遮挡扫描并估计衍射损耗

这使得省域拼接任务仍可在可控内存下运行。

## 3. 配置示例

```yaml
layers:
  l2_topo:
    enabled: true
    dem_file: "data/l2_topo/china_dem_30m.tif"
    frequency_ghz: 14.5
    satellite_elevation_deg: 45.0
    satellite_azimuth_deg: 180.0
```

## 4. 常见问题

- 文件不存在：L2 会返回零损耗并给出日志警告。
- 区域越界：L2 会抛出边界错误（请求区域超出 DEM 覆盖）。
- 仰角过高：遮挡效应不明显，可用低仰角（如 10°~20°）做地形对比展示。

## 5. 可替换数据源

- SRTM
- Copernicus DEM
- ASTER GDEM

替换时需保证坐标系和地理覆盖可被当前配置区域访问。
