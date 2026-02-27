# Configuration Directory

SG-MRM 仿真配置文件目录。

## `mission_config.yaml`

主配置文件，定义所有仿真参数：

| 配置段 | 说明 |
|--------|------|
| `origin` | 仿真中心坐标（当前：西安 34.3416°N, 108.9398°E） |
| `rf` | 射频参数：频率 14.5 GHz (Ku-band)、发射功率、极化 |
| `time` | 仿真时间范围与步长 |
| `layers.l1_macro` | L1 宏观层：TLE 文件、IONEX/ERA5 数据路径 |
| `layers.l2_topo` | L2 地形层：DEM 文件路径、卫星仰角/方位角 |
| `layers.l3_urban` | L3 城市层：tile cache 路径、NLoS/占用损耗、入射方向 |
| `output` | 输出目录、格式、DPI |
| `performance` | 性能分析开关 |
| `logging` | 日志级别与文件 |

## 使用

```bash
# 使用默认配置
python main.py

# 指定配置文件
python main.py --config configs/mission_config.yaml --output output/

# 自定义配置
cp configs/mission_config.yaml configs/my_config.yaml
# 编辑 my_config.yaml 后运行
python main.py --config configs/my_config.yaml
```

## 各层启用/禁用

```yaml
layers:
  l1_macro:
    enabled: true    # TLE 文件为必需依赖
  l2_topo:
    enabled: true    # 无 DEM 文件时自动返回零损耗
  l3_urban:
    enabled: true    # 需要预构建 tile cache
```
