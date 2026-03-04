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
import numpy as np
from src.utils import Era5Loader, load_era5

era5 = load_era5('data/l1_space/data/data_stream-oper_stepType-instant.nc')
if era5 is None:
    raise RuntimeError("ERA5 文件不存在或无法读取")

# 获取指定时刻、位置的 IWV（kg/m²）
hour_utc = 6.0
lat = np.array([[39.9]], dtype=float)
lon = np.array([[116.4]], dtype=float)
iwv = era5.get_iwv(hour_utc, lat, lon)

# 获取插值后的 IWV 网格（lat_grid / lon_grid 同形状）
# iwv_grid.shape == lat_grid.shape
iwv_grid = era5.get_iwv(hour_utc, lat_grid, lon_grid)
```

IWV 用于 `atmospheric_loss_era5()` 计算湿延迟，比简化模型更准确。

下载脚本见 `data/l1_space/data/cds.py`（需要 ECMWF CDS API 账号）。

---

### `TleLoader` (`tle_loader.py`)

加载 TLE 卫星轨道数据，并通过 SGP4 推进得到卫星地理位置。

```python
from datetime import datetime, timezone
from src.utils import TleLoader

loader = TleLoader('data/2025_0101.tle')

# 获取指定时刻所有卫星的地理坐标（lat/lon/alt_km）
dt = datetime(2025, 1, 1, 6, 0, 0, tzinfo=timezone.utc)
lats, lons, alts = loader.get_geodetic(dt)
```

可见星筛选与选星逻辑由 `L1MacroLayer` 负责（`get_visible_satellites()` / `compute_components()`）。

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
| `plot_layer_comparison(l1_map, l2_map, l3_map, composite_map, output_file)` | 多层对比图（子图网格） |
| `export_radio_map_png(loss_map, output_file)` | 导出为 PNG 文件 |
| `plot_time_series(timestamps, loss_values, output_file)` | 时序曲线图 |
| `create_animation_frames(loss_maps, timestamps, output_dir)` | 生成动画帧序列 |

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
    IonexLoader, Era5Loader, load_era5, TleLoader,
    plot_radio_map, plot_layer_comparison, export_radio_map_png,
    plot_time_series, create_animation_frames,
    setup_logger, get_logger, SimulationLogger,
    PerformanceTimer, PerformanceProfiler, timeit, profile_layer_computation,
)
```
