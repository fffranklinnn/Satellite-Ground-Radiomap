# 快速入门

## 安装

```bash
pip install -r requirements.txt
```

验证安装：

```bash
python main.py --help
```

## 运行仿真

### 方式一：使用配置文件

```bash
python main.py --config configs/mission_config.yaml --output output/
```

默认配置以西安为中心 (34.3416°N, 108.9398°E)，Ku-band 14.5 GHz，时间范围 2025-01-01 全天（6 小时步长）。

### 方式二：运行可视化脚本

```bash
# L1 全天时序 + 参数扫描
python scripts/generate_l1_map.py

# 全球六城市对比
python scripts/generate_global_comparison.py
```

### 方式三：编程接口

```python
from datetime import datetime
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext
import yaml

config = yaml.safe_load(open('configs/mission_config.yaml'))
lat, lon = config['origin']['latitude'], config['origin']['longitude']

# 初始化
l1 = L1MacroLayer(config['layers']['l1_macro'], lat, lon)
l2 = L2TopoLayer(config['layers']['l2_topo'], lat, lon)
l3 = L3UrbanLayer(config['layers']['l3_urban'], lat, lon)

# 计算
agg = RadioMapAggregator(l1, l2, l3)
ctx = LayerContext.from_any({'incident_dir': config['layers']['l3_urban']['incident_dir']})
composite = agg.aggregate(lat, lon, timestamp=datetime(2025, 1, 1, 6, 0, 0), context=ctx)

print(f"Shape: {composite.shape}, Range: {composite.min():.1f}~{composite.max():.1f} dB")
```

## 输出文件

```
output/
├── composite_0000.png   # 复合无线电地图
├── composite_0000.npy   # numpy 数组（可编程读取）
├── l1_macro_0000.png    # L1 宏观层
├── l2_topo_0000.png     # L2 地形层
├── l3_urban_0000.png    # L3 城市层
└── comparison_0000.png  # 四层对比图
```

## 自定义配置

编辑 `configs/mission_config.yaml`：

```yaml
# 修改仿真中心
origin:
  latitude: 34.3416
  longitude: 108.9398

# 修改时间范围
time:
  start: "2025-01-01T00:00:00"
  end: "2025-01-01T23:00:00"
  step_hours: 6

# 启用/禁用层
layers:
  l1_macro:
    enabled: true
  l2_topo:
    enabled: true
  l3_urban:
    enabled: true
```

## 数据依赖

| 层 | 数据 | 必需 |
|----|------|------|
| L1 | TLE 文件 (`data/2025_0101.tle`) | 是 |
| L1 | IONEX (`data/l1_space/data/*.INX.gz`) | 否（回退到默认 TEC） |
| L1 | ERA5 (`data/l1_space/data/*.nc`) | 否（回退到简化模型） |
| L2 | DEM GeoTIFF (`data/l2_topo/全国DEM数据.tif`) | 否（无文件返回零损耗） |
| L3 | Tile cache (`data/l3_urban/xian/tiles_60/`) | 是（需预构建） |

## 常见问题

**PROJ 库冲突警告**：如果出现 `proj.db contains DATABASE.LAYOUT.VERSION.MINOR` 警告，设置环境变量：

```bash
export PROJ_DATA=$(python -c "import pyproj; print(pyproj.datadir.get_data_dir())")
```

**L3 tile cache 不存在**：需要先构建，参见 `data/l3_urban/README.md`。
