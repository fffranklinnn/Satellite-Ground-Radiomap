# src/utils — 工具模块

提供数据加载、可视化和性能分析工具。

## 数据加载器

### `IonexLoader` (`ionex_loader.py`)

加载 IONEX 格式全球电离层图（GIM），提供实测 TEC 值。

**支持格式**：IONEX（`.INX`，支持 gzip 压缩的 `.INX.gz`）

```python
from src.utils import IonexLoader

loader = IonexLoader('data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz')

# 获取指定时刻、位置的 TEC（TECU）
tec = loader.get_tec(timestamp, lat=39.9, lon=116.4)

# 获取插值后的 TEC 网格（256×256）
tec_grid = loader.get_tec_grid(timestamp, lat_grid, lon_grid)
```

IONEX 文件包含全球 2.5°×5° 分辨率的 TEC 网格，时间分辨率 15 分钟。

---

### `Era5Loader` / `load_era5()` (`era5_loader.py`)

加载 ERA5 再分析气象数据（NetCDF 格式），提供积分水汽（IWV）。

```python
from src.utils import Era5Loader

era5 = Era5Loader('data/l1_space/data/data_stream-oper_stepType-instant.nc')

# 获取指定位置的 IWV（kg/m²）
iwv = era5.get_iwv(lat=39.9, lon=116.4)

# 获取插值后的 IWV 网格（256×256）
iwv_grid = era5.get_iwv_grid(lat_grid, lon_grid)
```

IWV 用于 `atmospheric_loss_era5()` 计算湿延迟，比简化模型更准确。

下载脚本见 `data/l1_space/data/cds.py`（需要 ECMWF CDS API 账号）。

---

### `TleLoader` (`tle_loader.py`)

加载 TLE 卫星轨道数据，支持多卫星星座。

```python
from src.utils import TleLoader

loader = TleLoader('data/2025_0101.tle')

# 获取指定时刻在指定位置可见的卫星列表
satellites = loader.get_visible_satellites(
    timestamp, origin_lat=39.9, origin_lon=116.4,
    min_elevation_deg=10.0
)

# 获取指定卫星的位置（lat, lon, alt_km）
pos = loader.get_satellite_position(sat_name, timestamp)
```

当前 V1.0 中 TLE 加载器已集成到 `L1MacroLayer`，但卫星定位仍使用固定天顶方向；V2.0 将启用基于 TLE 的动态轨道传播。

---

## 可视化工具 (`plotter.py`)

```python
from src.utils import (
    plot_radio_map,
    plot_layer_comparison,
    export_radio_map_png,
    plot_time_series,
    create_animation_frames,
)
```

| 函数 | 说明 |
|------|------|
| `plot_radio_map(loss_map, title, output_file)` | 绘制单张损耗图（热力图） |
| `plot_layer_comparison(maps_dict, output_file)` | 多层对比图（子图网格） |
| `export_radio_map_png(loss_map, output_file)` | 导出为 PNG 文件 |
| `plot_time_series(maps_list, timestamps, output_file)` | 时序对比图 |
| `create_animation_frames(maps_list, output_dir)` | 生成动画帧序列 |

---

## 日志工具 (`logger.py`)

```python
from src.utils import setup_logger, get_logger, SimulationLogger

logger = setup_logger('my_sim', level='INFO', log_file='logs/sim.log')
sim_logger = SimulationLogger('simulation')
sim_logger.log_layer_result('L1', loss_map)
```

---

## 性能分析工具 (`performance.py`)

```python
from src.utils import PerformanceTimer, PerformanceProfiler, timeit, profile_layer_computation

# 上下文管理器计时
with PerformanceTimer('L1 compute'):
    result = l1_layer.compute(timestamp)

# 装饰器
@timeit
def my_function(): ...

# 层计算性能分析
profile_layer_computation(l1_layer, timestamp, n_runs=10)
```

---

## 导出接口

```python
from src.utils import (
    IonexLoader, Era5Loader, TleLoader,
    plot_radio_map, plot_layer_comparison, export_radio_map_png,
    plot_time_series, create_animation_frames,
    setup_logger, get_logger, SimulationLogger,
    PerformanceTimer, PerformanceProfiler, timeit, profile_layer_computation,
)
```
