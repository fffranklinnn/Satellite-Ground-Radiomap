# src/layers 说明

`src/layers` 是 SG-MRM 的三层物理建模核心。

## 1. 层级与尺度

| 层 | 类 | 覆盖范围 | 分辨率 | 输出 |
|---|---|---:|---:|---|
| L1 | `L1MacroLayer` | 256 km x 256 km | 1000 m/px | 256x256 dB |
| L2 | `L2TopoLayer` | 25.6 km x 25.6 km | 100 m/px | 256x256 dB |
| L3 | `L3UrbanLayer` | 256 m x 256 m | 1 m/px | 256x256 dB |

## 2. 统一接口

所有层均继承 `BaseLayer`，核心方法：

```python
compute(origin_lat=None, origin_lon=None, timestamp=None, context=None, **kwargs) -> np.ndarray
```

返回统一为 `float32` 的 256x256 损耗矩阵（dB）。

## 3. L1 宏观层（`l1_macro.py`）

### 主要流程

1. 解析 TLE 并按时刻计算可见卫星
2. 选择最高仰角（或指定 NORAD）卫星
3. 计算网格级仰角/方位/斜距
4. 叠加 FSPL + 大气损耗 + 电离层损耗 + 极化损耗 - 波束增益
5. 仰角过低像素标注无覆盖（高损耗）

### 关键输入

- 必需：`tle_file`
- 可选：`ionex_file`, `era5_file`
- 可选增强：`ionosphere.use_ipp/use_slant_tec/enable_faraday`

### 关键输出

`compute_components()` 除 `total` 外还返回：

- `fspl`, `atm`, `iono`, `gain`, `pol`
- `tec`, `tec_vtec`, `iwv`
- `elevation`, `azimuth`, `slant_range_m`
- `satellite`（包含 norad、仰角、方位、斜距等）

### 回退行为

- IONEX 不可用 -> 使用配置 `tec` 默认值
- ERA5 不可用 -> 使用简化大气模型
- 地磁后端不可用 -> Faraday 使用回退 `B_parallel`

## 4. L2 地形层（`l2_topo.py`）

### 主要流程

1. 读取 DEM 窗口并重采样到 256x256
2. 按卫星入射方向做遮挡扫描
3. 基于遮挡剖面计算衍射损耗

### 关键输入

- `dem_file`
- `frequency_ghz`
- `satellite_elevation_deg`, `satellite_azimuth_deg`

### 上下文覆盖

`context.extras` 可覆盖：

- `satellite_elevation_deg`
- `satellite_azimuth_deg`
- `satellite_slant_range_km`
- `satellite_altitude_km`

### 回退行为

- 无 DEM -> 返回全零损耗图
- 请求区域超出 DEM 覆盖 -> 抛出异常

## 5. L3 城市层（`l3_urban.py`）

### 主要流程

1. 根据 `tile_id` 或 `origin(lat/lon)` 选择 cache tile
2. 加载 `H.npy` 与 `Occ.npy`
3. 依据 `incident_dir` 计算 NLoS mask
4. 映射 NLoS 与占用损耗

### 关键输入

- `tile_cache_root`
- `nlos_loss_db`
- `occ_loss_db`（可选）
- `incident_dir`（必须由配置或 context 提供）

### tile 选择逻辑

优先级：

1. `context.extras['tile_id']`
2. `origin_lat/origin_lon` 哈希匹配
3. 最近邻 tile 回退

## 6. 辅助文件

- `base.py`
  - `BaseLayer`
  - `LayerContext`

`LayerContext` 是跨层几何参数同步的关键对象。
