# data/l1_space 说明

本目录存放 L1 宏观层所需的数据与下载脚本。

## 1. 当前可见文件类型

| 文件类型 | 示例 | 用途 |
|---|---|---|
| TLE | `data/2025_0101.tle` | 卫星轨道传播与选星 |
| IONEX | `UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz` | TEC 查询 |
| ERA5 pressure-level | `data_stream-oper_stepType-instant.nc` | `q` 积分得到 IWV |
| ERA5 请求脚本 | `cds.py` | 下载 pressure-level 数据 |
| ERA5 批量下载脚本 | `cds_pressure_batch.py` | 按月/按日拆分下载全年 pressure-level |
| ERA5 single-level 请求脚本 | `cds_single_level.py` | 下载 `t2m/sp/tcwv/tp/tclw` |
| IONEX 请求脚本 | `NASAcddis.py` | 批量下载 IONEX |

> 注：下载脚本是请求入口，实际下载结果取决于本地执行和账号权限。

## 2. IONEX 数据要点

- 典型空间分辨率：2.5°（纬度）x 5.0°（经度）
- 典型时间分辨率：15 分钟
- 主流程使用 TEC；RMS 可用于后续误差建模扩展
- L1 中支持：
  - 直接地面点查询
  - IPP 穿刺点查询（`ionosphere.use_ipp=true`）
  - VTEC->STEC 映射（`ionosphere.use_slant_tec=true`）

## 3. ERA5 数据要点

### Pressure-level（当前主流程已用）

当前 `Era5Loader` 使用变量：

- `q`（specific humidity）
- `pressure_level`
- `latitude/longitude`
- `valid_time`

通过 `trapz(q, p)` 得到 IWV（kg/m^2），供 L1 大气损耗函数使用。

### Single-level（用于雨衰/云衰增强的准备）

推荐下载变量：

- `2m_temperature` (`t2m`)
- `surface_pressure` (`sp`)
- `total_column_water_vapour` (`tcwv`)
- `total_precipitation` (`tp`)
- `total_column_cloud_liquid_water` (`tclw`)

脚本：`data/l1_space/data/cds_single_level.py`

## 4. 典型下载命令

首次使用 CDS API 前：

```bash
pip install cdsapi
```

IONEX：

```bash
python data/l1_space/data/NASAcddis.py
```

ERA5 pressure-level：

```bash
python data/l1_space/data/cds.py
```

ERA5 pressure-level（批量，推荐）：

```bash
# 2025 全年，按月拆分（默认）
python data/l1_space/data/cds_pressure_batch.py \
  --year 2025 \
  --months 1-12 \
  --chunk-mode month \
  --area 40,106,32,112 \
  --output-dir data/l1_space/data/era5_pressure_levels_2025
```

```bash
# 如果按月仍超限，改为按日拆分
python data/l1_space/data/cds_pressure_batch.py \
  --year 2025 \
  --months 1-12 \
  --chunk-mode day \
  --area 40,106,32,112 \
  --output-dir data/l1_space/data/era5_pressure_levels_2025_day
```

脚本特性：

- 默认跳过已存在文件（断点续传友好）
- 支持 `--retries` 和 `--retry-wait-sec` 自动重试
- 支持 `--dry-run` 先检查请求计划
- 输出 `manifest_*.jsonl` 记录每个分片状态

ERA5 single-level：

```bash
python data/l1_space/data/cds_single_level.py
```

## 5. 在代码中的加载路径

L1 层配置（示例）：

```yaml
layers:
  l1_macro:
    tle_file: "data/2025_0101.tle"
    ionex_file: "data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz"
    era5_file: "data/l1_space/data/data_stream-oper_stepType-instant.nc"
```

## 6. 已知边界

1. 主流程当前没有直接消费 ERA5 single-level 的 `tp/tclw` 进行完整统计可用度建模。
2. Faraday 旋转是可选增强，且依赖可选地磁后端（`igrf` 或 `geomag`）。
3. 若缺少 IONEX 或 ERA5，L1 会回退到简化模型，不会中断整个流程。
