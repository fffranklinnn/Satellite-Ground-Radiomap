# src/core 说明

`src/core` 提供网格坐标与传播物理公式，是 L1/L2/L3 的公共基础。

## 文件总览

- `grid.py`
  - `Grid` 类：像素与地理坐标转换
  - 向量化网格函数：`get_grid_latlon()`、`get_azimuth_elevation()`
- `physics.py`
  - FSPL、大气损耗、电离层损耗、极化损耗
  - 雨衰比衰减与斜路径近似
  - 相控阵增益与波束宽度辅助函数

## 1. 网格模块 (`grid.py`)

### `Grid`

用于管理不同层的统一网格语义：

- 输入：`origin_lat/origin_lon/grid_size/coverage_km`
- 输出：
  - 像素 -> 经纬度
  - 经纬度 -> 像素
  - 距离矩阵

### 向量化函数

- `get_grid_latlon(origin_lat, origin_lon, coverage_m=256000, grid_size=256)`
  - 生成整张网格的 `lat/lon/x_m/y_m`
- `get_azimuth_elevation(sat_x_m, sat_y_m, sat_alt_m, grid_x_m, grid_y_m)`
  - 计算每像素到卫星的方位/仰角/斜距

## 2. 物理模块 (`physics.py`)

### 路损相关

- `free_space_path_loss(distance_km, frequency_ghz)`
- `fspl_db(slant_range_m, freq_hz)`

### 大气相关

- `atmospheric_loss(elevation_angle_deg, frequency_ghz, rain_rate_mm_h=0.0)`
  - 清空+雨衰近似
- `atmospheric_loss_era5(elevation_angle_deg, frequency_ghz, iwv_kg_m2, rain_rate_mm_h=0.0)`
  - 使用 IWV 的改进近似
- `rain_specific_attenuation_db_per_km(...)`
- `rain_attenuation_slant_path_db(...)`

### 电离层与极化

- `ionospheric_loss(frequency_ghz, tec=10.0)`
- `polarization_loss(...)`
- `polarization_loss_db(pol_mismatch_angle_deg)`

### 天线与链路辅助

- `gaussian_beam_gain_db(...)`
- `phased_array_peak_gain_db(...)`
- `phased_array_hpbw_deg(...)`
- `thermal_noise_power_dbw(...)`

## 3. 在项目中的调用关系

- L1：调用 `get_grid_latlon/get_azimuth_elevation` + 大多数物理函数
- L2：调用 `SPEED_OF_LIGHT` 计算衍射参数
- L3：不直接依赖 `core` 公式，主要是几何遮挡

## 4. 当前精度边界

- 雨衰和气体吸收为工程近似，不是完整 ITU 全流程统计模型。
- `ionospheric_loss` 是频率衰减近似，闪烁未并入主流程。
