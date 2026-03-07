# src/utils 说明

`src/utils` 提供数据加载、可视化、日志和性能分析等通用组件。

## 1. 数据加载器

### `IonexLoader` (`ionex_loader.py`)

- 输入：IONEX 文件（支持 `.INX` / `.INX.gz`）
- 主要方法：
  - `get_tec(epoch_sec, lat, lon)`
- 特点：
  - 空间双线性 + 时间线性插值
  - 缺失值使用统计值填充后插值

### `Era5Loader` / `load_era5` (`era5_loader.py`)

- 输入：ERA5 pressure-level NetCDF
- 主要方法：
  - `get_iwv(hour_utc, lat, lon)`
- 特点：
  - 从 `q` 与 `pressure_level` 积分得到 IWV
  - 文件缺失或依赖不可用时 `load_era5()` 返回 `None`

### `TleLoader` (`tle_loader.py`)

- 输入：TLE 文件
- 主要方法：
  - `get_geodetic(datetime_utc)`
- 特点：
  - 使用 `sgp4` 传播
  - 内含倾角范围过滤

## 2. 电离层几何与极化辅助 (`ionosphere.py`)

- `ipp_from_ground(...)`
  - 地面点投影到薄壳 IPP
- `faraday_rotation_deg(stec_tecu, b_parallel_t, freq_hz)`
  - 估计 Faraday 旋转角
- `polarization_mismatch_loss_db(mismatch_deg)`
  - 向量化线极化失配损耗

## 3. 数据完整性校验 (`data_validation.py`)

- `validate_data_integrity(config, project_root, strict=False)`
  - 对 L1/L2/L3 配置数据源做存在性与基础可读性检查
- `format_data_validation_report(report)`
  - 统一格式化检查结果

`main.py --check-data-only` 与 `scripts/check_data_integrity.py` 都基于该模块。

## 4. 可视化 (`plotter.py`)

常用接口：

- `plot_radio_map(...)`
- `plot_layer_comparison(...)`
- `plot_full_radiomap_paper(...)`
- `export_radio_map_png(...)`
- `plot_time_series(...)`
- `create_animation_frames(...)`

## 5. 日志 (`logger.py`)

- `setup_logger(...)`
- `get_logger(...)`
- `SimulationLogger`

用于统一输出层启动/结束、统计信息和错误。

## 6. 性能分析 (`performance.py`)

- `PerformanceTimer`
- `PerformanceProfiler`
- `timeit`
- `profile_layer_computation(...)`

可用于单层性能采样和总体耗时统计。

## 7. 导出接口

统一在 `src/utils/__init__.py` 暴露，建议从包级导入：

```python
from src.utils import IonexLoader, Era5Loader, TleLoader
```
