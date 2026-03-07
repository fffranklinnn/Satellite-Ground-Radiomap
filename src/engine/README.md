# src/engine 说明

`src/engine/aggregator.py` 负责把 L1/L2/L3 损耗图聚合成单张复合 radiomap。

## 1. 核心类

- `RadioMapAggregator`

构造函数：

```python
RadioMapAggregator(l1_layer=None, l2_layer=None, l3_layer=None, target_grid_size=256)
```

要求至少提供一层，否则抛出 `ValueError`。

## 2. 聚合逻辑

复合公式：

```text
composite = Interp(L1) + Interp(L2) + L3
```

- L1 覆盖 256 km，L2 覆盖 25.6 km，都会先插值到 L3 覆盖范围（256 m）与 `target_grid_size`
- 插值器使用 `RectBivariateSpline`，`kx=1, ky=1`（双线性）
- 加法在 dB 域直接叠加

## 3. 公开方法

- `aggregate(origin_lat, origin_lon, timestamp=None, context=None)`
  - 返回 `np.ndarray(shape=(256,256), dtype=float32)`
- `get_layer_contributions(...)`
  - 返回 `{'l1','l2','l3','composite'}` 中实际启用层的键
- `compute_composite_map(...)`
  - `aggregate` 别名

## 4. 上下文透传

`context`（`LayerContext`）会透传到各层，用于携带：

- L1：目标 NORAD、雨率覆盖等
- L2：卫星仰角/方位/斜距
- L3：`incident_dir`、`tile_id`

## 5. 使用示例

```python
from src.engine import RadioMapAggregator
agg = RadioMapAggregator(l1_layer=l1, l2_layer=l2, l3_layer=l3)
loss = agg.aggregate(origin_lat, origin_lon, timestamp=ts, context=ctx)
```
