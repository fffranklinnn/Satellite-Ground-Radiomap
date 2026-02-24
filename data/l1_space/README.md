# L1 Space Layer Data

This directory contains data for the L1 Macro/Space layer.

## Data Files (Current)

### `../../2025_0101.tle` — TLE 卫星轨道数据

- **内容**: Starlink 星座 TLE 数据（约 2.1 MB）
- **日期**: 2025 年 1 月 1 日
- **格式**: 标准 TLE 三行格式（卫星名 + Line 1 + Line 2）
- **用途**: 由 `TleLoader` 加载，用于获取可见卫星列表及仰角计算（V2.0 动态定位）

### `data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz` — IONEX 电离层数据

- **内容**: IGS UPC 分析中心发布的全球电离层图（GIM）
- **日期**: 2025 年第 1 天（2025-01-01），全天 97 个时间点，15 分钟间隔
- **格式**: IONEX 格式（gzip 压缩），包含全球 TEC 网格（2.5° × 5° 分辨率）
- **用途**: 由 `IonexLoader` 加载，提供实测 TEC 值用于电离层损耗计算（ITU-R P.531）

### `data/data_stream-oper_stepType-instant.nc` — ERA5 气象数据

- **内容**: ECMWF ERA5 再分析气象数据
- **格式**: NetCDF（`.nc`）
- **关键变量**: 积分水汽（IWV，单位 kg/m²）
- **用途**: 由 `Era5Loader` 加载，提供 IWV 用于大气湿延迟计算（ITU-R P.836 近似）

---

## 数据获取

### TLE 文件

```bash
# 从 CelesTrak 下载 Starlink TLE
wget https://celestrak.com/NORAD/elements/starlink.txt -O data/l1_space/starlink.tle
```

### IONEX 文件

从 IGS 数据中心下载（如 CDDIS）：

```
ftp://cddis.nasa.gov/gnss/products/ionex/YYYY/DDD/
```

文件命名格式：`{center}OPSRAP_{YYYY}{DDD}0000_01D_15M_GIM.INX.gz`

### ERA5 数据

通过 ECMWF CDS API 下载：

```python
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-single-levels', {
    'variable': 'total_column_water_vapour',
    'product_type': 'reanalysis',
    'date': '2025-01-01',
    'time': '00:00',
    'format': 'netcdf',
}, 'data/l1_space/data/era5_iwv.nc')
```

---

## 加载方式

```python
from src.utils import IonexLoader, Era5Loader, TleLoader

ionex = IonexLoader('data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz')
era5  = Era5Loader('data/l1_space/data/data_stream-oper_stepType-instant.nc')
tle   = TleLoader('data/2025_0101.tle')
```
