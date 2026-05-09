# 1. 项目名称

Satellite-Ground-Radiomap 场景自适应分层传播改造方案

# 2. 项目简介

Satellite-Ground-Radiomap 是一个面向卫星到地面链路传播建模的多层无线电地图生成项目。项目当前已经建立了三层传播框架：

- `L1`：宏观层，负责卫星几何、自由空间损耗、大气、电离层等大尺度传播效应；
- `L2`：地形层，负责地形遮挡、视距变化与地形附加损耗；
- `L3`：城市层，负责建筑阻挡、局部 NLoS 与城市精细修正。

项目经过前一轮语义统一与多尺度重构，已经基本建立了统一的 `FrameContext`、`CoverageSpec`、`native_grid`、`product_grid`、projection/composition 与 manifest/provenance 体系，主线测试目前可以通过。这说明项目已经从“早期研究脚本集合”进入到了“有统一运行语义的传播管线”阶段。

但与此同时，一个新的问题已经变得非常突出：当前系统默认任何场景都需要完整运行 `L1 + L2 + L3`，而这与真实的物理场景并不一致。对于城市平原区，建筑细节往往比地形起伏更重要；对于山区非城市区，地形传播效应显著，而建筑层甚至没有物理意义；对于空旷平原区，很多时候 `L1` 就已经足够。因此，项目下一阶段的重点，不应再是默认三层全开，而应是建立“面向场景的分层传播启用策略”。

# 3. 项目目标

这个项目从最开始到最终完善，真正要做的事情不是简单地把三个传播模块堆在一起，而是构建一套**面向不同地理场景、具有统一时空语义、可解释、可复现的卫星地面传播地图生成系统**。

从整体上看，项目应当完成以下四个目标：

## 3.1 建立统一的传播场景定义

同一张 radiomap 必须对应同一个物理场景定义，包括：

- 同一个时间语义；
- 同一个卫星绑定结果；
- 同一个空间覆盖范围；
- 同一个目标产品网格；
- 同一套启用层组合。

换句话说，系统不能再允许“图虽然能画出来，但不同层实际并不属于同一个物理场景”。

## 3.2 建立统一的多尺度传播表达

项目不能再依赖“数组 shape 相同就代表可以相加”的早期策略，而必须明确：

- L1、L2、L3 分别运行在什么原生网格上；
- 最终产品输出在哪个 `product_grid` 上；
- 不同层之间如何投影与组合；
- 哪些层是骨架层，哪些层是局部修正层。

这一步的目标不是让系统“更复杂”，而是让输出结果在空间语义上真实可信。

## 3.3 建立面向场景的层启用策略

项目最终不应把“三层传播”理解为“任何场景都必须计算三层”，而应该理解为“根据场景类型，选择合理的层组合”。

也就是说：

- 城市平原区可以是 `L1 + L3`；
- 山区非城市区可以是 `L1 + L2`；
- 混合区可以是 `L1 + L2 + L3`；
- 空旷平原区甚至可以是 `L1 only`。

这一步的目标，是让模型回到真实的物理主导因素，而不是被固定的实现框架绑架。

## 3.4 建立可复现、可追溯的实验与产品输出体系

项目不应只停留在“生成了一张图”，而应能回答：

- 这张图属于什么场景；
- 用了哪些层；
- 没用哪些层；
- 不用某层是因为不需要，还是因为数据缺失；
- 同样配置、同样数据快照、同样 frame list 是否可复现。

这要求项目最终形成完整的 benchmark/provenance/manifest 体系。

# 4. 功能需求

基于上述总目标，项目最终应具备以下功能。

## 4.1 场景定义功能

系统应允许用户明确声明当前仿真对应的场景类型，而不是让每个脚本自己隐式决定。

建议最小场景集合包括：

- `urban_flat`
- `mountain_rural`
- `suburban_mixed`
- `plain_sparse`

## 4.2 分层启用功能

系统应根据场景类型自动解析本次运行需要启用哪些传播层，并控制：

- 是否运行 `L2TopoLayer`
- 是否运行 `L3UrbanLayer`
- 最终组合采用哪些层

## 4.3 层缺失原因解释功能

系统应能正式区分以下情况：

- 某层由于场景策略被关闭；
- 某层本应启用，但输入数据缺失；
- 某层被用户手工关闭。

## 4.4 部分层组合输出功能

系统不仅要支持 `L1 + L2 + L3`，还应支持：

- `L1`
- `L1 + L2`
- `L1 + L3`

并保证这些输出仍然符合统一的 frame/grid/projection 语义。

## 4.5 场景化 benchmark 功能

Benchmark 不应继续默认“三层都开才是标准”，而应允许：

- 针对城市区 benchmark `L1 + L3`
- 针对山区 benchmark `L1 + L2`
- 针对混合区 benchmark `L1 + L2 + L3`

# 5. 技术方案

基于当前代码现实情况，这一阶段的技术方案不应推翻已有统一语义架构，而应在其上增加“场景自适应层策略”。

## 5.1 保留现有统一基础

以下基础已经存在，应继续复用：

- `FrameContext`
- `CoverageSpec`
- `GridSpec`
- `native_grid`
- `product_grid`
- `ProjectedView`
- `MultiScaleMap.compose(...)`
- `ProductManifest` / provenance

这说明项目不需要回到早期重新设计传播框架，而是只需在运行时调度层面增强。

更具体地说，当前主线已经基本具备一条完整的 canonical runtime：

1. `src/planning/satellite_selector.py` 先做卫星选择；
2. `src/context/frame_builder.py` 构建 `FrameContext`；
3. `src/layers/l1_macro.py`、`l2_topo.py`、`l3_urban.py` 分别产生 typed states；
4. `src/compose.project_to_product_grid(...)` 将不同 native grid 投影到统一的 `product_grid`；
5. `src/context.multiscale_map.MultiScaleMap.compose(...)` 负责 canonical 组合；
6. `src/products.projectors` 与 `src/products.manifest` 负责产品导出和 provenance 记录。

因此，scene-aware layer policy 的正确切入点不是重写已有传播框架，而是在这条主链路前面增加一层“场景 -> 层执行集合”的统一决策。

## 5.2 增加场景 profile

建议在配置中增加：

```yaml
scene:
  profile: urban_flat
```

它作为运行时层策略的上游输入。

## 5.3 增加统一层策略解析器

建议新增统一模块，例如：

- `LayerPolicyResolver`

输入：

- `scene.profile`
- 当前配置
- strict / benchmark mode
- 数据可用性检查结果

输出：

- `enabled_layers`
- `disabled_layers`
- `reason_type`
- `reason`

建议输出结构不要只是一组布尔量，而是更接近：

```python
LayerPolicy(
    scene_profile="mountain_rural",
    enabled_layers=("l1_macro", "l2_topo"),
    disabled_layers={
        "l3_urban": {
            "reason_type": "scene_policy",
            "reason": "mountain_rural profile disables urban refinement",
        }
    },
)
```

这样该对象既能驱动 runtime，也能直接进入 manifest。

## 5.4 在主运行路径接入策略

以下入口应统一走同一套层策略：

- `main.py`
- `BenchmarkRunner`
- 多星脚本

目标是避免“同一个项目，不同脚本各自决定该不该跑某层”。

结合当前项目入口，建议分两步接：

- 第一步：先接入 `main.py` 和 `src/pipeline/benchmark_runner.py`
  - 它们已经是当前 canonical runtime 的主要入口
- 第二步：再接入 `scripts/generate_multisat_timeseries_radiomap.py`
  - 这是高价值脚本，但相对更偏专项流程

这样可以优先保证主路径与 benchmark path 一致，而不是一开始就试图同时收拢所有脚本。

## 5.5 在 manifest 中记录层策略结果

建议为 `ProductManifest` 扩展：

- `scene_profile`
- `enabled_layers`
- `disabled_layers`
- `reason_type`
- `layer_policy_version`

这样每次结果都可解释。

从当前实现看，`ProductManifest` 已经能记录：

- `config_hash`
- `data_snapshot_id`
- `input_file_hashes`
- `output_file_hashes`
- `fallbacks_used`
- `ProvenanceBlock`

所以这一阶段不是“重新设计 manifest”，而是在现有 contract 上继续增加：

- `scene_profile`
- `enabled_layers`
- `disabled_layers`
- `reason_type`

真正关键的是把“层没有运行”的语义升级成正式输出，而不是只在日志里打印一句说明。

# 6. 项目结构

从当前代码状态出发，本项目的主要结构可概括为：

- `src/context/`
  - 统一 frame/grid/coverage/multiscale 语义
- `src/layers/`
  - L1/L2/L3 传播计算实现
- `src/compose/`
  - 投影与组合
- `src/products/`
  - 产品生成与 manifest
- `src/pipeline/`
  - benchmark 运行入口
- `main.py`
  - 主运行入口
- `scripts/`
  - 特定任务脚本，如多星时序脚本
- `docs/`
  - 架构、计划、草案与总结文档

下一阶段建议新增的结构位置是：

- `src/planning/` 或相近目录
  - 用于承载 scene-aware `LayerPolicyResolver`

如果按当前仓库实际职责再展开，可以更准确地理解为：

## 6.1 `src/context/`

这个目录现在已经是项目的统一时空语义核心，包括：

- `frame_context.py`
  - 定义 `FrameContext`
  - 统一 frame_id、timestamp、satellite geometry、coverage
- `frame_builder.py`
  - 负责实际构建 frame
- `coverage_spec.py`
  - 定义 `CoverageSpec`
  - 当前已支持 `l1_grid`、`l2_grid`、`l3_grid`、`product_grid`
- `grid_spec.py`
  - 定义基础网格几何
- `layer_states.py`
  - 定义 `EntryWaveState`、`TerrainState`、`UrbanRefinementState`
- `multiscale_map.py`
  - 负责 legacy/native-state 组合与 canonical projected 组合
- `time_utils.py`
  - 负责 UTC 约束

scene-aware policy 最终必须服从这一层已经建立好的 runtime contract，而不是绕开它另造一套调度模型。

## 6.2 `src/layers/`

这个目录承载三层传播资产：

- `l1_macro.py`
  - 输出宏观传播骨架
  - 当前已能读取 TLE、IONEX、ERA5 等输入
- `l2_topo.py`
  - 输出地形损耗和地形遮挡
- `l3_urban.py`
  - 输出城市局部 residual 与 support mask

这三层的“物理职责”当前已经比早期清楚得多。下一阶段不应再急着改它们的内部公式，而应优先明确“何时该调用它们”。

## 6.3 `src/compose/`

这个目录是当前 canonical projection contract 所在位置。

当前核心内容包括：

- `FieldType`
- `ProjectedView`
- `project_field(...)`
- `project_to_product_grid(...)`

由于 `project_to_product_grid(...)` 已经返回 `ProjectedView`，而 `MultiScaleMap.compose(...)` 已经基于 `ProjectedView` 组合，所以 scene-aware policy 可以直接利用这个现有接口：只要把本次实际启用的层投影过去即可。

## 6.4 `src/products/`

这个目录负责产品与输出层语义：

- `manifest.py`
  - `ProductManifest`
  - `ProvenanceBlock`
  - `BenchmarkMode`
  - `collect_input_file_paths(...)`
- `projectors.py`
  - 提供 `path_loss_map`、`visibility_mask`、`terrain_blockage`、`urban_residual` 等产品导出
- `map_product.py`
  - 负责产品对象封装

因此，scene-aware policy 不只影响 runtime，还会直接影响产品导出和 manifest 解释。

## 6.5 `src/pipeline/`

这个目录承载标准化执行路径，尤其是 benchmark path。

当前关键模块：

- `benchmark_runner.py`
  - 已经能够走：select satellite -> build frame -> run L1/L2/L3 -> project -> compose -> export manifest
- `manifest_writer.py`
  - 负责 JSONL 记录

这一层是 scene-aware policy 最需要优先接入的地方，因为 benchmark path 不能继续默认所有场景都该三层全开。

## 6.6 `src/planning/`

当前已存在：

- `satellite_selector.py`

这说明 `src/planning/` 已经承担“运行前决策逻辑”的角色，因此新增 `LayerPolicyResolver` 放在这里是很自然的。

## 6.7 脚本与入口

当前主要入口包括：

- `main.py`
- `scripts/generate_multisat_timeseries_radiomap.py`
- `benchmarks/capture_golden_scenes.py`

它们现在都已经较多依赖统一 frame/projector/composition 语义，所以后续不是推翻这些入口，而是把 scene-aware policy 嵌进去。

# 7. 核心模块

结合当前与后续目标，项目核心模块应包括：

## 7.1 FrameContext

负责统一一帧仿真的时间、卫星、几何与 coverage 契约。

## 7.2 CoverageSpec / GridSpec

负责统一原生传播网格和产品网格的空间语义。

## 7.3 L1 / L2 / L3 Layers

负责各自物理层的传播计算资产，但不再被视作“任何时候都必须同时启用”的固定三件套。

当前代码中这三层已经具备了比较清晰的输出语义：

- `L1MacroLayer` 产生 `EntryWaveState`
- `L2TopoLayer` 产生 `TerrainState`
- `L3UrbanLayer` 产生 `UrbanRefinementState`

也正因为如此，下一阶段可以把重点放在“层的启用策略”而不是“层的内部实现重写”上。

## 7.4 MultiScale Composition

负责将已投影的层结果组合到产品网格上。

当前这一层已经从早期的 shape-based 组合，演进到了：

- `project_to_product_grid(...)`
- `MultiScaleMap.compose(...)`

这意味着组合层已经具备足够好的基础去支持部分层集合：

- `L1`
- `L1+L2`
- `L1+L3`
- `L1+L2+L3`

所以 scene-aware policy 最终不应该改写 compose 逻辑，而是控制 compose 的输入集合。

## 7.5 LayerPolicyResolver

这是下一阶段新增的核心模块，用于把场景 profile 转成层启用结果。

建议它至少承担以下职责：

- 维护不同 scene profile 的默认层策略
- 判断某层是 required 还是 optional
- 根据 strict / benchmark mode 和数据可用性做最终决策
- 输出 `enabled_layers`、`disabled_layers`
- 给出禁用原因分类

## 7.6 ProductManifest

负责记录输出产品与其运行时合同。

当前 manifest 已经有较好的基础 contract，因此这一步更像“扩展输出语义”而不是“重做输出系统”。

# 8. 输入输出

## 8.1 输入

系统输入包括：

- 配置文件
- 时间与区域设置
- TLE
- IONEX / ERA5
- DEM
- 城市建筑或 tile 数据
- scene profile

从当前代码角度看，这些输入分别对应：

- TLE -> `satellite_selector.py` / `l1_macro.py`
- IONEX / ERA5 -> `l1_macro.py` / `src/utils`
- DEM -> `l2_topo.py`
- urban tiles / building cache -> `l3_urban.py`

因此，scene-aware policy 不只是多一个配置项，而是要把“哪一类输入在什么场景下是必需品”说清楚。

## 8.2 输出

系统输出不仅包括最终 radiomap，还应包括运行合同信息：

- 产品数组或产品文件
- `ProductManifest`
- `scene_profile`
- `enabled_layers`
- `disabled_layers`
- `disabled_reason`

结合当前产品结构，输出可以分成两层：

1. 产品层  
   如 `path_loss_map`、`visibility_mask`、`terrain_blockage`、`urban_residual`

2. 合同层  
   即 `ProductManifest` 与 provenance / layer policy 记录

未来 scene-aware 改造的关键，是让“产品层结果”始终能被“合同层解释”。

# 9. 实现计划

下面按阶段给出项目下一步的实现计划。

## 阶段一：统一基础语义

这一阶段的目标，是把时间、空间、frame、grid、projection、manifest 这些基础语义先统一起来，使项目从“能跑”进入“主路径一致”的状态。

这一阶段主要完成：

- `FrameContext` 的建立与主路径接入
- `CoverageSpec` / `GridSpec` 语义统一
- `native_grid` / `product_grid` 区分
- projection/composition canonical path 收口
- manifest/provenance 基础能力建立

结合当前项目实际情况，这一阶段已经基本完成，当前主线测试可通过就是重要证据。

更具体地说，当前已经完成的包括：

- `FrameContext` / `FrameBuilder`
- `CoverageSpec` 四网格模型
- `native_grid` state contract
- `ProjectedView` + canonical `compose(...)`
- manifest/provenance 基础框架

## 阶段二：引入场景语义

这一阶段的目标，是明确项目不再默认“三层永远同时启用”，而是为系统增加正式的场景分类能力。

这一阶段主要完成：

- 增加 `scene.profile`
- 固定不同场景的物理含义
- 明确哪些场景对应哪些推荐层组合

这一阶段是后续所有 scene-aware runtime 的前提。

## 阶段三：引入层策略解析器

这一阶段的目标，是把 scene profile 转换为项目级统一的层启用策略。

这一阶段主要完成：

- 增加 `LayerPolicyResolver`
- 输出启用层 / 关闭层 / 关闭原因
- 将该策略接入 `main.py`、`BenchmarkRunner` 和多星脚本

完成后，项目就不再依赖脚本作者自己决定该不该跑 L2 或 L3。

这里建议接入顺序是：

1. `main.py`
2. `BenchmarkRunner`
3. 多星脚本

## 阶段四：区分 scene_policy 与 missing_input

这一阶段的目标，是让系统可以严格区分：

- 某层因为场景本来不需要而不启用；
- 某层本应启用，但因为输入缺失而无法运行。

这一阶段主要完成：

- manifest 字段扩展
- strict / benchmark mode 行为调整
- disabled reason contract 固化

这是项目从“工程上能控制层开关”走向“科学上可解释”的关键一步。

## 阶段五：场景化 benchmark

这一阶段的目标，是让 benchmark 协议本身也接受“不同场景允许不同层组合”。

这一阶段主要完成：

- 按场景类型组织 benchmark
- 在 benchmark artifact 中记录实际层组合
- 区分 `L1`、`L1+L2`、`L1+L3`、`L1+L2+L3`

完成后，benchmark 就不会再隐含地强迫所有区域都跑完整三层。

## 阶段六：后续增强

在 scene-aware runtime 稳定后，才适合继续考虑：

- 自动场景识别
- 更复杂的启用 heuristics
- 更丰富的产品导出
- AI / dataset 接口

# 10. 测试方案

本项目的测试策略不应只验证“模块能不能运行”，还应验证“运行的场景合同是否正确”。

## 10.1 基础语义测试

验证：

- frame/grid/coverage 是否一致
- projection/composition 是否符合 contract

当前仓库这部分已经有现成基础测试：

- `tests/test_context/test_multiscale_map.py`
- `tests/test_context/test_projected_composition.py`
- `tests/test_products/test_projectors.py`

## 10.2 场景策略测试

验证：

- 不同 `scene.profile` 是否得到正确层组合
- 未知 profile 是否报错

## 10.3 运行路径测试

验证：

- `urban_flat` 可运行 `L1 + L3`
- `mountain_rural` 可运行 `L1 + L2`
- `plain_sparse` 可运行 `L1`

## 10.4 manifest 测试

验证：

- 是否正确记录 `scene_profile`
- 是否正确记录 `enabled_layers`
- 是否正确区分 `scene_policy` 和 `missing_input`

## 10.5 benchmark / strict 测试

验证：

- strict mode 下 required layer 缺失是否失败
- scene policy 禁用层是否不被视作错误

这一部分后续应建立在当前 `BenchmarkRunner`、`BenchmarkMode` 与 `collect_input_file_paths(...)` 的基础上扩展，而不是另起一套测试框架。

# 11. 当前进度

结合当前代码状态，可以对项目进度做如下判断。

## 11.1 已经完成的部分

- 三层传播主框架已经存在
- 统一 `FrameContext` / `CoverageSpec` / `GridSpec` 语义已经基本建立
- projection/composition canonical path 已经基本收口
- manifest/provenance 框架已经存在
- 当前全量测试可以通过

更精确地说，当前项目已经完成的是“统一三层框架”的大部分底层工作。

## 11.2 尚未完成的部分

- 还没有正式的 `scene.profile`
- 还没有统一的 `LayerPolicyResolver`
- 还没有把“层不启用的原因”升级成正式运行契约
- benchmark 仍未系统化表达“场景不同、层组合不同”的事实

也就是说，项目现在缺的已经不是基本的 L1/L2/L3 计算能力，而是“根据场景合理调度这些能力”的上层语义控制。

## 11.3 当前阶段判断

项目现在最主要的矛盾，已经不再是“底层时空语义不一致”，而是：

**系统已经具备统一三层框架，但还没有具备按场景合理裁剪层级的能力。**

因此，下一阶段最值得做的事，不是继续把三层都做得更复杂，而是先把：

- 场景类型
- 层启用策略
- 缺层原因解释
- 场景化 benchmark

这四件事正式做成项目级合同。
