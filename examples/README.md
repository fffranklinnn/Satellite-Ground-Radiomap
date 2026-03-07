# examples 说明

本目录提供示例脚本模板，用于说明如何通过代码调用 SG-MRM 组件。

## 1. 当前文件

- `basic_usage.py`
- `v1_static_link.py`

## 2. 使用建议

这些示例最初用于早期版本演示。当前主流程以 `configs/mission_config.yaml` + `main.py` / `scripts/*` 为主，若直接运行示例，建议先按当前接口检查并同步以下项：

1. `L1MacroLayer` 需要有效 `tle_file`。
2. `L3UrbanLayer` 需要有效 `tile_cache_root` 与 `incident_dir`。
3. `RadioMapAggregator.aggregate()` 需要显式传入 `origin_lat, origin_lon`。

## 3. 推荐替代入口

- 主流程：`python main.py --config configs/mission_config.yaml`
- 单次全物理图：`python scripts/generate_full_radiomap.py`
- 批量任务：`python scripts/batch_city_experiments.py`

## 4. 如果要维护示例

建议把示例改造成“读取主配置后只覆盖少量参数”的形式，这样能避免与主接口长期漂移。
