# tests — 单元测试

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 带覆盖率报告
pytest --cov=src tests/

# 运行特定模块
pytest tests/test_core/
pytest tests/test_layers/test_l1_macro.py
```

## 测试结构

```
tests/
├── test_core/
│   ├── test_grid.py       # Grid 坐标转换、距离矩阵
│   └── test_physics.py    # 物理公式数值正确性
├── test_layers/
│   └── test_l1_macro.py   # L1 层计算、数据加载器集成
├── test_engine/
│   └── test_aggregator.py # 多层聚合、插值
└── test_utils/
    ├── test_ionex_loader.py  # IONEX 解析、TEC 插值
    └── test_era5_loader.py   # ERA5 NetCDF 加载、IWV 提取
```

## 各测试文件覆盖范围

| 文件 | 主要测试内容 |
|------|------------|
| `test_grid.py` | 像素↔坐标转换精度、距离矩阵形状和值域 |
| `test_physics.py` | FSPL 公式验证、大气损耗仰角依赖、电离层频率依赖、极化损耗组合 |
| `test_l1_macro.py` | `compute()` 返回形状（256×256）、无数据时的默认行为、ERA5/IONEX 数据路径 |
| `test_aggregator.py` | 单层/多层聚合、L2 插值尺寸、`get_layer_contributions()` 键值 |
| `test_ionex_loader.py` | `.gz` 文件解析、TEC 网格插值、时间插值 |
| `test_era5_loader.py` | NetCDF 变量读取、IWV 空间插值 |

## 测试数量

当前共 49+ 个单元测试，覆盖核心物理计算和数据加载路径。
