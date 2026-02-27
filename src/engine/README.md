# src/engine — 多层聚合引擎

将 L1/L2/L3 各层的损耗图合并为最终的复合无线电地图。

## 聚合公式

```
L_total(dB) = Interpolate(L1 → L3覆盖) + Interpolate(L2 → L3覆盖) + L3
```

L1 (256 km) 和 L2 (25.6 km) 的输出通过双线性插值提取中心 256 m 区域，下采样到 L3 分辨率后在 dB 域直接相加。

## 类：`RadioMapAggregator`

### 构造函数

```python
RadioMapAggregator(
    l1_layer=None,   # L1MacroLayer 实例
    l2_layer=None,   # L2TopoLayer 实例
    l3_layer=None,   # L3UrbanLayer 实例
    target_grid_size=256
)
```

至少需要提供一个层，否则抛出 `ValueError`。

### 方法

#### `aggregate(origin_lat, origin_lon, timestamp=None, context=None)`

计算复合损耗图。返回 256×256 float32 dB 数组。

#### `get_layer_contributions(origin_lat, origin_lon, timestamp=None, context=None)`

返回各层独立贡献 + 复合结果：

```python
{
    'l1': np.ndarray,        # L1 损耗（已插值到 L3 覆盖）
    'l2': np.ndarray,        # L2 损耗（已插值到 L3 覆盖）
    'l3': np.ndarray,        # L3 损耗
    'composite': np.ndarray  # 合并结果
}
```

## 使用示例

```python
from datetime import datetime
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext

l1 = L1MacroLayer(config['layers']['l1_macro'], lat, lon)
l2 = L2TopoLayer(config['layers']['l2_topo'], lat, lon)
l3 = L3UrbanLayer(config['layers']['l3_urban'], lat, lon)

agg = RadioMapAggregator(l1, l2, l3)
ctx = LayerContext.from_any({'incident_dir': {'az_deg': 180, 'el_deg': 45}})

composite = agg.aggregate(lat, lon, timestamp=datetime(2025, 1, 1, 6, 0, 0), context=ctx)
contributions = agg.get_layer_contributions(lat, lon, timestamp=datetime(2025, 1, 1, 6, 0, 0), context=ctx)
```
