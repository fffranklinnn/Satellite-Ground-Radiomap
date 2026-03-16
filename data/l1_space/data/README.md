# data/l1_space/data 说明

本目录是 L1 宏观层的本地数据工作区，包含下载脚本与已落地的数据文件。

## 1. 目录角色

- 存放 L1 运行时直接读取的数据（IONEX、ERA5）
- 存放数据获取脚本（CDS / NASA CDDIS）
- 作为年度批量下载结果的归档根目录

## 2. 当前内容分类

```text
data/l1_space/data/
├── NASAcddis.py                     # IONEX 批量下载脚本
├── cds.py                           # ERA5 pressure-level 单请求脚本
├── cds_pressure_batch.py            # ERA5 pressure-level 批量下载脚本
├── cds_single_level.py              # ERA5 single-level 下载脚本
├── data_stream-oper_stepType-instant.nc
│                                     # 运行示例使用的 ERA5 pressure-level NetCDF
├── cddis_data_2025/                 # 2025 年 IONEX 数据归档
└── era5_pressure_levels_2025_day/   # ERA5 按日切片下载结果（示例）
```

## 3. 与运行配置的关系

典型配置项（`configs/mission_config.yaml`）：

- `layers.l1_macro.ionex_file`
- `layers.l1_macro.era5_file`

这些路径通常指向本目录或其子目录中的具体文件。

## 4. 使用与维护建议

- 年度归档建议按 `cddis_data_<year>/`、`era5_*_<year>_*` 命名。
- 批量下载优先使用 `cds_pressure_batch.py`，并保存 `manifest_*.jsonl` 便于追踪失败分片。
- 大文件不建议直接提交到版本库；保留下载脚本和目录约定即可复现。
