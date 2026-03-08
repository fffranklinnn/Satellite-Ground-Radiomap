# tests 说明

本目录是 SG-MRM 的单元测试集合，覆盖核心公式、加载器、聚合器和 L1 主路径。

## 1. 运行方式

```bash
pytest tests/
pytest --cov=src tests/
```

按子模块运行：

```bash
pytest tests/test_core/
pytest tests/test_layers/test_l1_macro.py
pytest tests/test_engine/test_aggregator.py
```

## 2. 目录结构

```text
tests/
├── test_core/
│   ├── test_grid.py
│   └── test_physics.py
├── test_layers/
│   └── test_l1_macro.py
├── test_engine/
│   └── test_aggregator.py
└── test_utils/
    ├── test_data_validation.py
    ├── test_ionex_loader.py
    └── test_era5_loader.py
```

## 3. 当前覆盖范围

- `test_grid.py`
  - 网格坐标转换、距离矩阵等
- `test_physics.py`
  - FSPL / 大气 / 电离层 / 极化等公式基本正确性
- `test_l1_macro.py`
  - L1 初始化、组件输出结构、可见星与 NORAD 过滤
- `test_aggregator.py`
  - 聚合器初始化与输出形状
- `test_ionex_loader.py`
  - IONEX 解析、空间插值、时间插值
- `test_era5_loader.py`
  - ERA5 IWV 读取与插值
- `test_data_validation.py`
  - 配置数据完整性检查（缺失关键数据、全层关闭场景）

## 4. 当前测试规模

- `pytest --collect-only -q`：当前分支（2026-03-07）收集到 37 个测试用例。

## 5. 建议补充

1. L2/L3 更多几何边界测试（极端仰角、越界 tile、空 cache）。
2. 关键脚本的集成测试（至少 smoke test）。
3. 文档示例与真实 API 的一致性回归测试。
