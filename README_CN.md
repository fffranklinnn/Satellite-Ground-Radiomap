# SG-MRM：星地多尺度无线电地图

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English README](README.md)

SG-MRM 是一个用于星地链路传播损耗建模的多尺度仿真框架，输出可用于 radiomap 生成、对比分析和批量实验。

## 1. 计算目标

对每个时间点和区域中心，系统输出各层 256x256 损耗图（dB）与复合图：

```text
L_total(dB) = Interp(L1 -> L3覆盖) + Interp(L2 -> L3覆盖) + L3
```

### 三层建模范围

| 层 | 空间尺度 | 分辨率 | 核心效应 |
|---|---:|---:|---|
| L1 宏观层 | 256 km x 256 km | 1000 m/px | TLE 选星、FSPL、相控阵增益、大气损耗、电离层 TEC 损耗 |
| L2 地形层 | 25.6 km x 25.6 km | 100 m/px | DEM 加载、LOS 遮挡、衍射损耗 |
| L3 城市层 | 256 m x 256 m | 1 m/px | 建筑 NLoS 阴影、占用损耗 |

## 2. 当前实现能力（按代码现状）

### 已实现

- L1：
  - 基于 TLE 的可见卫星枚举与最高仰角选星（`Skyfield`）
  - FSPL + 高斯波束近似相控阵增益 + 极化失配
  - IONEX TEC 读取；支持 IPP 穿刺点和 VTEC->STEC 映射
  - 可选 Faraday 旋转附加失配（需可选地磁后端）
  - ERA5 pressure-level (`q`) 积分得到 IWV，并用于改进大气损耗
- L2：
  - DEM 窗口读取与重采样
  - 向量化方向性遮挡扫描
  - 基于遮挡剖面的衍射损耗估计（带上限）
- L3：
  - tile cache (`H.npy`/`Occ.npy`) 加载
  - 方向性 NLoS 扫描
  - NLoS / 占用像素损耗映射
- 引擎与脚本：
  - 多层插值聚合
  - 城市/省域拼接、批量实验、可见星统计、功能展示图生成

### 尚未完全 ITU 严格化

- 尚未完成 P.618 可用度统计链路（如 `R0.01` 统计映射）
- 尚未做 P.676 全频谱逐层积分气体吸收
- 主流程未集成闪烁（`S4`）模型
- L3 尚未引入完整多径光线追踪核心

## 3. 安装

```bash
pip install -r requirements.txt
```

可选数据与依赖见 [data/README.md](data/README.md)。

### 可选：在 `sgmrm_test` 安装 PyTorch（用于后续热点迁移）

```bash
conda activate sgmrm_test
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python - << 'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 4. 快速开始

### 主程序

```bash
python main.py --config configs/mission_config.yaml --output output/
```

### 常用脚本

```bash
python scripts/generate_full_radiomap.py --timestamp 2025-01-01T06:00:00
python scripts/report_satellite_visibility.py --start 2025-01-01T00:00:00 --end 2025-01-01T23:00:00 --step-hours 1
python scripts/generate_feature_showcase.py --output-root output/feature_showcase_demo
python scripts/check_data_integrity.py --config configs/mission_config.yaml --strict
```

更多脚本说明见 [scripts/README.md](scripts/README.md)。

## 5. 仓库内数据快照

| 类型 | 路径 | 当前用途 |
|---|---|---|
| TLE | `data/2025_0101.tle` | 2025-01-01 主仿真轨道数据 |
| IONEX | `data/l1_space/data/*.INX.gz` | 电离层 TEC |
| ERA5 pressure-level | `data/l1_space/data/*.nc` | IWV 提取（`q` 积分） |
| DEM | `data/l2_topo/全国DEM数据.tif` | L2 地形遮挡 |
| L3 原始数据 | `data/l3_urban/shanxisheng/陕西省/*.shp` | 陕西省建筑矢量源 |
| L3 可运行缓存 | `data/l3_urban/xian/tiles_60/` | 西安现成 cache |

## 6. 项目结构与可优化点

```text
Satellite-Ground-Radiomap/
├── src/             # 运行时核心代码（core/layers/engine/utils）
├── scripts/         # 批处理与出图入口
├── configs/         # YAML 配置
├── data/            # 数据目录（含重数据入口）
├── docs/            # 文档
├── tests/           # 单元测试
├── examples/        # 旧示例
├── output/          # 生成结果（已 gitignore）
└── branch_L1/L2/L3/ # 本地历史分支快照（已 gitignore）
```

建议：

1. `src/` 作为唯一运行时真源，`branch_*` 仅做历史参考。
2. 大体量结果统一放 `output/`，原始下载放 `data/` 或 `cddis_data_2025/`。
3. 长期维护时可把 `branch_*` 归档或移出主目录，减少结构噪声。
4. 优先通过 `scripts/` 复现实验，避免不可追踪的临时流程。

## 7. 文档索引

- 文档入口：[docs/README.md](docs/README.md)
- 快速上手：[docs/QUICKSTART.md](docs/QUICKSTART.md)
- 配置说明：[configs/README.md](configs/README.md)
- 数据说明：[data/README.md](data/README.md)
- 三层实现：[src/layers/README.md](src/layers/README.md)
- 工具模块：[src/utils/README.md](src/utils/README.md)
- 测试说明：[tests/README.md](tests/README.md)

## 8. 测试

```bash
pytest tests/
pytest --cov=src tests/
```

## 许可证

MIT License，见 [LICENSE](LICENSE)。
