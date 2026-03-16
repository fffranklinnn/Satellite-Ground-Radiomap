# tests/test_layers 说明

本目录覆盖 `src/layers` 中的层级实现测试（当前重点是 L1）。

## 当前文件

- `test_l1_macro.py`
  - L1 初始化与参数读取
  - 组件输出结构契约（`compute_components`）
  - 可见卫星筛选与 NORAD 过滤行为
  - IONEX 可用性与离轴角输出检查
