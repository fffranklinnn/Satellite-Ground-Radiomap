# configs 目录说明

本目录存放 SG-MRM 的 YAML 配置文件。

当前主配置：`mission_config.yaml`。

## 1. 设计原则

- 顶层只定义运行参数，不放业务代码。
- `layers.*` 与 `src/layers/*` 一一对应。
- 脚本可在命令行覆盖部分配置（例如时间、频率、雨率、区域）。

## 2. `mission_config.yaml` 结构

### `origin`

| 字段 | 类型 | 说明 |
|---|---|---|
| `latitude` | float | 区域中心纬度 |
| `longitude` | float | 区域中心经度 |
| `altitude_m` | float | 站点海拔（当前主流程未强依赖） |

### `time`

| 字段 | 类型 | 说明 |
|---|---|---|
| `start` | ISO string | 开始时间 |
| `end` | ISO string | 结束时间 |
| `step_hours` | float/int | 小时步长 |
| `timezone` | string | 文档标注用途，代码默认按 UTC 处理 |

### `layers.l1_macro`

核心字段：

| 字段 | 说明 |
|---|---|
| `enabled` | 是否启用 L1 |
| `grid_size/coverage_km/resolution_m` | L1 网格参数 |
| `frequency_ghz` | 载频（GHz） |
| `tle_file` | TLE 文件路径（必需） |
| `ionex_file` | IONEX 文件路径（可选） |
| `era5_file` | ERA5 pressure-level NetCDF（可选） |
| `rain_rate_mm_h` | 雨率输入（mm/h） |
| `tec` | IONEX 缺失时默认 TEC |

`ionosphere` 子配置：

| 字段 | 说明 |
|---|---|
| `use_ipp` | 使用 IPP 穿刺点查 TEC |
| `use_slant_tec` | 使用映射因子 VTEC->STEC |
| `shell_height_km` | 薄壳高度（默认 350km） |
| `max_mapping_factor` | 映射因子上限 |
| `enable_faraday` | 是否叠加 Faraday 旋转失配 |
| `faraday_linear_only` | 仅在线极化模式下启用 Faraday |
| `fallback_b_t` | 无地磁后端时的 `B_parallel` 回退值 |

`polarization` 子配置：

| 字段 | 说明 |
|---|---|
| `mismatch_angle_deg` | 基础极化失配角 |
| `mode` | `linear` / `circular` |

### `layers.l2_topo`

| 字段 | 说明 |
|---|---|
| `enabled` | 是否启用 L2 |
| `dem_file` | DEM GeoTIFF 路径 |
| `frequency_ghz` | 频率（影响衍射参数） |
| `satellite_elevation_deg` | 默认仰角 |
| `satellite_azimuth_deg` | 默认方位角 |
| `satellite_altitude_km` | slant-range 估算回退高度 |

说明：脚本或上下文可覆盖仰角/方位/斜距。

### `layers.l3_urban`

| 字段 | 说明 |
|---|---|
| `enabled` | 是否启用 L3 |
| `tile_cache_root` | tile cache 根目录 |
| `nlos_loss_db` | NLoS 像素损耗 |
| `occ_loss_db` | 建筑占用像素损耗（可选） |
| `incident_dir` | 默认入射方向（可被上下文覆盖） |

`incident_dir` 推荐格式：

```yaml
incident_dir:
  az_deg: 180.0
  el_deg: 45.0
```

### `output`

| 字段 | 说明 |
|---|---|
| `directory` | 默认输出目录 |
| `format` | 预留字段（当前主流程未直接读取） |
| `save_individual_layers` | 是否保存 L1/L2/L3 分层图 |
| `save_composite` | 是否保存复合图 |
| `dpi` | PNG DPI |
| `colormap` | 可视化配色 |

### 顶层兼容字段（当前主流程未直接消费）

`mission_config.yaml` 中还包含以下字段，主要用于元信息或历史兼容：

| 字段 | 当前状态 |
|---|---|
| `mission` | 元信息字段；当前 `main.py` 不读取 |
| `rf` | 预留 RF 描述；当前频率等以 `layers.*` 为准 |
| `satellite` | 早期结构兼容字段；当前卫星几何以 L1 配置 + TLE 计算为准 |
| `output.format` | 预留字段；当前输出行为由 `save_individual_layers/save_composite` 决定 |

### `data_validation`

| 字段 | 说明 |
|---|---|
| `strict` | 严格数据模式。也可由 `main.py --strict-data` 覆盖开启 |

### `performance` / `logging`

- `performance.enable_profiling`：开启后打印性能摘要。
- `logging.level/log_file`：日志等级与文件路径。

## 3. 常见配置修改场景

修改时间范围：

```yaml
time:
  start: "2025-01-01T00:00:00"
  end: "2025-01-31T23:00:00"
  step_hours: 1
```

关闭某一层：

```yaml
layers:
  l3_urban:
    enabled: false
```

固定卫星候选：

```yaml
layers:
  l1_macro:
    target_norad_ids: ["25544", "44713"]
```

## 4. 运行方式

```bash
python main.py --config configs/mission_config.yaml --output output/
```

或复制为自定义配置再运行。

数据完整性检查（不跑仿真）：

```bash
python main.py --config configs/mission_config.yaml --check-data-only
```
