# src/core — 核心物理计算模块

提供网格坐标系统和 RF 传播物理公式，是所有层的计算基础。

## 模块

### `grid.py` — 网格坐标系统

**类：`Grid`**

管理 256×256 像素网格与地理坐标之间的映射关系。

| 方法 | 说明 |
|------|------|
| `pixel_to_latlon(i, j)` | 像素坐标 → (lat, lon) |
| `latlon_to_pixel(lat, lon)` | (lat, lon) → 像素坐标 |
| `get_distance_matrix()` | 返回每个像素到原点的距离矩阵（km），256×256 |
| `get_latlon_grids()` | 返回完整的纬度/经度网格数组 |

网格以 `origin_lat/lon` 为中心，覆盖范围和分辨率由 `coverage_km` 和 `resolution_m` 决定。

---

### `physics.py` — RF 传播物理公式

所有函数均支持标量和 NumPy 数组输入（完全向量化）。

#### `free_space_path_loss(distance_km, frequency_ghz)`

自由空间路径损耗（FSPL）。

```
FSPL(dB) = 20·log₁₀(d) + 20·log₁₀(f) + 92.45
```

- `d`：距离（km）
- `f`：频率（GHz）

---

#### `atmospheric_loss(elevation_angle_deg, frequency_ghz, rain_rate_mm_h=0.0)`

大气衰减，ITU-R P.618 简化模型。

- 仰角 ≤ 0° 时返回 999 dB（地平线以下）
- 含简单雨衰项：`0.01 · (f/10)² · R · (1/sin(el))`

---

#### `atmospheric_loss_era5(elevation_angle_deg, frequency_ghz, iwv_kg_m2)`

基于 ERA5 积分水汽（IWV）的改进大气衰减，ITU-R P.836 近似。

```
wet_zenith = 0.0173 · IWV · (f/10)  dB
dry_zenith = 0.046 · (f/10)          dB
loss = (dry + wet) / sin(elevation)
```

有 ERA5 数据时优先使用此函数。

---

#### `ionospheric_loss(frequency_ghz, tec=10.0)`

电离层效应，ITU-R P.531。

- 频率 > 3 GHz 时返回 0（可忽略）
- `tec`：总电子含量（TECU），默认 10.0，有 IONEX 数据时使用实测值

---

#### `polarization_loss(tx_polarization, rx_polarization, cross_pol_discrimination_db=20.0)`

极化失配损耗。

| 组合 | 损耗 |
|------|------|
| 相同极化 | 0 dB |
| H ↔ V 或 RHCP ↔ LHCP | `cross_pol_discrimination_db`（默认 20 dB） |
| 线极化 ↔ 圆极化 | 3 dB |

---

#### `combine_losses_db(*losses_db)`

dB 域损耗叠加（直接求和）。

---

## 导出接口

```python
from src.core import (
    Grid,
    free_space_path_loss,
    atmospheric_loss,
    atmospheric_loss_era5,
    ionospheric_loss,
    polarization_loss,
    combine_losses_db,
    db_to_linear,
    linear_to_db,
)
```
