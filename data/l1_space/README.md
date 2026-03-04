# L1 Space Layer Data

This directory contains data for the L1 Macro/Space layer.

## Data Files (Current)

### `data/2025_0101.tle` — TLE 卫星轨道数据

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
- **关键变量**: `q`, `z`, `r`, `t`, `pressure_level`, `valid_time`, `latitude`, `longitude`
- **用途**: 由 `Era5Loader` 加载，基于 `q` 与 `pressure_level` 积分计算 IWV，用于大气损耗计算

---

## 数据获取

### TLE 文件

```bash
# 从 CelesTrak 下载 Starlink TLE（示例）
wget https://celestrak.com/NORAD/elements/starlink.txt -O data/2025_0101.tle
```

### IONEX 文件

从 IGS 数据中心下载（如 CDDIS）：

```
ftp://cddis.nasa.gov/gnss/products/ionex/YYYY/DDD/
```

文件命名格式：`{center}OPSRAP_{YYYY}{DDD}0000_01D_15M_GIM.INX.gz`

项目内提供批量下载脚本（按 DOY 1~365）：

```bash
python data/l1_space/data/NASAcddis.py
```

### ERA5 数据

通过 ECMWF CDS API 下载 `reanalysis-era5-pressure-levels`（与当前加载器一致）：

```python
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-pressure-levels', {
    'variable': ['geopotential', 'relative_humidity', 'specific_humidity', 'temperature'],
    'product_type': 'reanalysis',
    'year': '2025',
    'month': '01',
    'day': '01',
    'time': [f'{h:02d}:00' for h in range(24)],
    'pressure_level': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125',
                       '150', '175', '200', '225', '250', '300', '350', '400', '450', '500',
                       '550', '600', '650', '700', '750', '775', '800', '825', '850', '875',
                       '900', '925', '950', '975', '1000'],
    'data_format': 'netcdf',
    'download_format': 'zip',
})
```

项目内也提供下载请求脚本：

```bash
python data/l1_space/data/cds.py
```

下载后将 zip 内 `data_stream-oper_stepType-instant.nc` 解压到 `data/l1_space/data/`。

---

## 加载方式

```python
from src.utils import IonexLoader, Era5Loader, TleLoader

ionex = IonexLoader('data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz')
era5  = Era5Loader('data/l1_space/data/data_stream-oper_stepType-instant.nc')
tle   = TleLoader('data/2025_0101.tle')
```
