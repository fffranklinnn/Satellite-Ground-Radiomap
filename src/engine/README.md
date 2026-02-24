# src/engine — 多层聚合引擎

将 L1/L2/L3 各层的损耗图合并为最终的复合无线电地图。

## 聚合公式

```
L_total(dB) = L1 + Interpolate(L2 → L1分辨率) + L3
```

各层损耗在 dB 域直接相加（对数域叠加等价于线性域相乘）。

由于三层分辨率不同，L2（25.6 km）在合并前通过双三次样条插值（`scipy.interpolate.RectBivariateSpline`）上采样到 L1 分辨率（256 km）。L3（256 m）在最细分辨率下独立运行。

---

## 类：`RadioMapAggregator`

### 构造函数

```python
RadioMapAggregator(
    l1_layer=None,   # L1MacroLayer 实例
    l2_layer=None,   # L2TopoLayer 实例（可选）
    l3_layer=None,   # L3UrbanLayer 实例（可选）
)
```

至少需要提供一个层，否则抛出 `ValueError`。

### 方法

#### `aggregate(timestamp: datetime) -> np.ndarray`

计算指定时刻的复合损耗图。

- 调用各层的 `compute(timestamp)`
- 将 L2 插值到 L1 网格尺寸
- 返回 256×256 的 dB 损耗数组

#### `get_layer_contributions(timestamp: datetime) -> dict`

返回各层独立贡献，便于分析和调试：

```python
{
    'l1': np.ndarray,   # L1 损耗图（256×256）
    'l2': np.ndarray,   # L2 损耗图（256×256，已插值）
    'l3': np.ndarray,   # L3 损耗图（256×256）
    'total': np.ndarray # 合并结果
}
```

---

## 使用示例

```python
from datetime import datetime
from src.layers import L1MacroLayer
from src.engine import RadioMapAggregator

l1 = L1MacroLayer(config, origin_lat=39.9042, origin_lon=116.4074)
aggregator = RadioMapAggregator(l1_layer=l1)

# 计算单时刻
radio_map = aggregator.aggregate(datetime(2025, 1, 1, 12, 0, 0))

# 查看各层贡献
contributions = aggregator.get_layer_contributions(datetime(2025, 1, 1, 12, 0, 0))
print(contributions['l1'].mean(), "dB (L1 平均损耗)")
```

---

## 导出接口

```python
from src.engine import RadioMapAggregator
```
