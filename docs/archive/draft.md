• # 1. 当前项目诊断

  你的项目现在最大的问题，确实不是"模块能不能各自算出图"，而是"这些图是否来自同一个物理场景定义"。从现有代码看，这个判断是有代码证据支撑的，不只是架构层面的担忧：

  - src/core/grid.py:42 把 origin 定义为网格中心。
  - src/layers/l2_topo.py:129 却把 origin 定义成 L2 tile 的 south-west corner。
  - src/engine/aggregator.py:86 把目标覆盖范围硬编码为 0.256 km。
  - src/layers/l1_macro.py:274 和 src/layers/l1_macro.py:287 在缺省时间时会回退到 current UTC。
  - main.py:111 仍直接用 datetime.fromisoformat()，没有统一强制转 UTC。
  - scripts/generate_multisat_timeseries_radiomap.py:223 说明"统一几何上下文"现在还是靠脚本手工拼 LayerContext，还不是核心一等公民。

  ## 1.1 必须立即修复

  - 层间空间语义冲突：origin=center vs origin=SW corner
      - 性质：科学语义风险 + 工程契约风险
      - 这是目前最危险的问题。因为 L1/L3/聚合器默认都更像"中心锚点语义"，而 L2 明确按 southwest corner 读 DEM，这意味着同样的 (lat, lon) 输入，在不同层上不是同一个地面 footprint。
      - 如果不先修，后面任何"误差分析""层间对比""多星时序"都可能是在比较不同空间区域。
  - 层间几何未绑定到同一帧
      - 性质：科学语义风险
      - 目前 L1 在每帧选星并得到 az/el/slant range，但 L2/L3 并没有天然绑定同一份 frame geometry；多星脚本里是手工把卫星几何塞进 LayerContext，见 scripts/generate_multisat_timeseries_radiomap.py:223。
      - 这意味着系统级一致性依赖"脚本作者记得怎么拼 context"，而不是依赖统一 pipeline。
  - 聚合 coverage 写死
      - 性质：科学语义风险 + 可扩展性风险
      - src/engine/aggregator.py:86 的 target_coverage_km = 0.256 直接把"最终产品 footprint"偷换成"当前 L3 tile 大小"。
      - 这会阻断你后面要做的 coarse-to-fine 统一系统，也让"无 L3 场景""多产品导出"都语义不稳。
  - 缺省时间和静默 fallback 破坏严格复现
      - 性质：科学可信度风险
      - src/layers/l1_macro.py:287 会在没有时间时直接取当前 UTC；IONEX/ERA5 也存在 fallback 路径，见 src/layers/l1_macro.py:218、src/layers/l1_macro.py:414、src/layers/l1_macro.py:453。
      - 对科研原型来说，"跑出来了"不等于"可复验"。

  ## 1.2 应尽快修复

  - UTC 体系没有项目级统一
      - 性质：科学语义风险 + 工程风险
      - main.py:111 用 datetime.fromisoformat()，而另一个脚本已经有更严格的 parse_iso_utc()，见 scripts/generate_multisat_timeseries_radiomap.py:100。
      - 现在属于"部分脚本严格、部分脚本宽松"，这对时序传播仿真是隐患。
  - 旧脚本与核心接口漂移
      - 性质：工程风险 + 可维护性风险
      - 现在已经出现脚本直接调用私有方法 _interpolate_to_target() 的情况，见 scripts/generate_multisat_timeseries_radiomap.py:237。
      - 这说明"核心接口不够表达业务意图"，脚本开始绕过正式抽象。
  - L3 仍被当作平行全图层，而不是 urban refinement
      - 性质：科学建模风险 + 可扩展性风险
      - 当前聚合逻辑仍是 L1 + L2 + L3 三层并排求和，这适合早期 demo，不适合"30m 粗骨架 + 1m 城市精修"的多尺度统一模型。
  - 评估还停留在出图/统计，不是 benchmark/protocol
      - 性质：工程风险 + 科学评估风险
      - 现在更多是生成 PNG/NPY/对比图，而不是"固定帧集、固定输入快照、固定误差协议"的 benchmark。

  ## 1.3 后续增强

  - 把 L1 拆得更物理清晰
      - 性质：可扩展性问题
      - 后续应区分：自由空间基准、电离层、对流层/天气层、极化/Faraday 等，而不是只在 L1 内部堆组件。
  - 支持多种产品，不把 radio map 当唯一终点
      - 性质：可扩展性问题
      - 未来更合理的是：radio map 只是对传播状态场的一种投影；中间状态本身也应该可导出。
  - AI 接口和数据集导出
      - 性质：后续增强
      - 应建立在 FrameContext + State + Manifest 之上，而不是直接从 PNG/NPY 脚本生长出来。

  ———

  # 2. 分阶段修改路线图

  ## 阶段1

  - 目标
      - 统一时间、几何、坐标、coverage 语义。
      - 先修"接口契约"，不急着动传播公式。
  - 为什么先做这一阶段
      - 你当前最大的风险是"同一个参数名在不同模块里不是同一个物理含义"。
      - 如果这一层不先锁死，后面任何 frame-centric 改造都只是在更快地传播不一致。
  - 修改内容
      - 明确项目唯一合法空间语义：
          - origin 不再裸传，改为显式 anchor=center 或显式 bbox。
          - 统一像素语义：pixel center registration、row/col 方向、north/south 顺序。
          - coverage 不再隐含在聚合器或层默认值里。
      - 统一时间语义：
          - 项目级所有 timestamp 都必须是 timezone-aware UTC。
          - 禁止 batch 仿真默认"没给时间就取 now"。
  - 新增抽象
      - GridSpec
      - CoverageSpec
  - 对现有代码的最小侵入式改法
      - 保留旧接口：compute(origin_lat, origin_lon, timestamp=None, context=None)
        但内部第一步统一转成：frame_grid = GridSpec.from_legacy_args(...)
      - src/engine/aggregator.py:86 的硬编码 0.256 改为从 CoverageSpec.target_grid 读取。
      - src/layers/l2_topo.py:129 的 SW-corner 语义改成"仅限 legacy wrapper"，正式内部接口只接受 GridSpec 或 bbox。
  - 验收标准
      - 同一个 frame 下，L1/L2/L3 都能打印/导出同一份中心点、bbox、resolution、coverage。
      - 不再有模块单独解释 origin。
      - 单元测试新增：GridSpec(center)->bbox、bbox->pixel transform、aggregator 不再依赖硬编码 L3 footprint。
  - 风险
      - 这一步会暴露很多"之前默认能跑"的脚本。
      - 一些旧输出可能发生小范围空间偏移，这是好事，说明以前确实不一致。

  ## 阶段2

  - 目标
      - 让 L1/L2/L3 都绑定同一个 FrameContext，彻底解决层间不同步。
  - 为什么先做这一阶段
      - 阶段1只是把"坐标和契约"说清楚；阶段2才是把"每一帧的卫星几何"真正变成一等公民。
      - 这是从"多层脚本系统"升级到"统一时空上下文驱动模拟器"的分水岭。
  - 修改内容
      - 引入 FrameContext，由一个统一的 frame builder 生成。
      - L1 不再只是返回一张图，而是先返回 EntryWaveState。
      - L2/L3 不再自己猜卫星几何，而是消费同一个 FrameContext + EntryWaveState。
  - 新增抽象
      - FrameContext、EntryWaveState、TerrainState、UrbanRefinementState
  - 对现有代码的最小侵入式改法
      - 保留 L1MacroLayer.compute()，但新增：def propagate_entry(self, frame: FrameContext) -> EntryWaveState
      - 保留 L2TopoLayer.compute()，但新增：def propagate_terrain(self, frame: FrameContext, entry: EntryWaveState) -> TerrainState
      - 保留 L3UrbanLayer.compute()，但新增：def refine_urban(self, frame: FrameContext, entry: EntryWaveState, terrain: TerrainState | None = None) -> UrbanRefinementState
      - main.py 和现有 timeseries 脚本不再手工构造 LayerContext，而是：
        frame = frame_builder.build(config, timestamp_utc)
        entry = l1.propagate_entry(frame)
        terrain = l2.propagate_terrain(frame, entry)
        urban = l3.refine_urban(frame, entry, terrain)
        product = projector.to_radiomap(frame, entry, terrain, urban)
  - 验收标准
      - 对任意一帧，L2/L3 使用的 az/el/slant_range/norad_id/timestamp 与 L1 完全一致。
      - 不再允许脚本层拼接几何字段后塞 extras。
      - scripts/generate_multisat_timeseries_radiomap.py:223 这种手工注入逻辑被替换成统一 FrameContext。
  - 风险
      - 接口会有一次明显调整。
      - 某些脚本若依赖"L2 默认 45° elevation""L3 默认 incident_dir"，会被强制改正。

  ## 阶段3

  - 目标
      - 重构多尺度表示，明确 30m coarse map 与 1m urban refinement 的关系。
  - 为什么先做这一阶段
      - 只有在"同一帧、同一几何、同一 coverage"成立后，coarse-to-fine 才有物理意义。
      - 否则你会把"不同 footprint 的三张图"误当成多尺度。
  - 修改内容
      - 重新定义 L2/L3 的职责：L2 提供 coarse terrain backbone；L3 提供 urban local residual / refinement，而不是平行全图层。
      - 聚合逻辑从 total = l1 + l2 + l3 改为：
        base = project(entry, coarse_grid) + terrain.loss_db
        urban = refine(base, urban_residual, urban_mask)
        total = project_to_product(base, urban, target_grid)
  - 新增抽象
      - MultiScaleMap
  - 对现有代码的最小侵入式改法
      - 不推翻 src/layers/l2_topo.py 和 src/layers/l3_urban.py 的计算资产。
      - 只改它们的"输出语义"：TerrainState.loss_db = coarse 骨架损耗；UrbanRefinementState.residual_db = 仅对城市细节的局部增量；UrbanRefinementState.support_mask = 哪些区域允许 refinement 覆盖。
      - 聚合器逐步降级成 projector/compositor，不再是"硬编码中心裁切 + 求和器"。
  - 验收标准
      - 非城市区域：关闭 L3 后结果应基本等于 coarse backbone 投影。
      - 城市区域：L3 只在 urban support 范围内改变结果，不再生成"整张图同尺度平行层"。
      - 能同时输出：coarse 30m map、urban 1m local refinement、最终统一产品。
  - 风险
      - 这一步最容易被误做成"再加一个重采样器"。
      - 必须坚持：L3 是 refinement，不是另一个同级全图。

  ## 阶段4

  - 目标
      - 引入 manifest / provenance / strict UTC / strict snapshot / benchmark mode。
  - 为什么先做这一阶段
      - 到这一步，系统才真正开始具备"科研软件原型"的可信性。
      - 没有 provenance，后面做论文对比或城市批量实验都很难说清输入快照是否一致。
  - 修改内容
      - 建立项目级 strict policy：strict UTC、strict data snapshot、strict no-fallback、benchmark mode。
      - 输出 manifest：输入数据路径和哈希/版本、代码版本、frame 参数、选星结果、fallback/warning 记录。
      - benchmark mode 固定：frame 列表、区域集、产品类型、统计协议。
  - 新增抽象
      - ProductManifest
  - 对现有代码的最小侵入式改法
      - 把 scripts/generate_multisat_timeseries_radiomap.py:100 的 parse_iso_utc() 上移为共享工具，main.py:111 全部改用它。
      - L1 的 datetime.now() fallback 只允许在交互 demo 模式；严格模式下一律报错。
      - src/utils/data_validation.py 从"检查数据是否存在"升级为"生成 snapshot report"。
  - 验收标准
      - 同一 config + 同一 snapshot + 同一 frame list，重复运行得到相同 manifest 和可接受的数值一致性。
      - strict mode 下：缺 IONEX/ERA5/DEM/L3 tile cache 直接失败；出现 naive datetime 直接失败；出现 fallback 直接失败。
      - benchmark run 可直接产出 machine-readable manifest。
  - 风险
      - 早期会让"以前勉强能跑"的实验失败变多。
      - 但这正是科学可信度提升的代价。

  ## 阶段5

  - 目标
      - 再考虑电离层、极化、多种地图产品、数据集导出和后续 AI 接口。
  - 为什么最后做这一阶段
      - 在 frame/context/state 没有稳定前，继续加物理细节只会把不一致放大。
      - 现在先保证"语义对"，再追求"物理更全"。
  - 修改内容
      - 把 L1 内部进一步拆成更清晰的物理子层：free-space baseline、ionosphere、troposphere/weather、polarization/Faraday。
      - 增加产品投影器：path loss map、visibility mask、entry elevation/azimuth field、terrain blockage map、urban residual map、dataset sample bundle。
  - 新增抽象
      - 可在此阶段引入 MapProduct，作为 State -> Exportable Product 的统一封装。
  - 对现有代码的最小侵入式改法
      - 不要直接改层内所有公式。
      - 先把现有计算结果挂到标准 state 字段，再逐步增强物理分量。
  - 验收标准
      - 能在同一帧导出多个产品，而不仅是单一 radio map。
      - 数据集导出直接依赖 FrameContext + States + ProductManifest。
      - 后续 AI 管线不需要再从 PNG 倒推物理上下文。
  - 风险
      - 最容易在这一阶段"功能爆炸"。
      - 一定要先有 benchmark，再扩物理项。

  ———

  # 3. 核心数据结构建议

  优先级：FrameContext > GridSpec/CoverageSpec > EntryWaveState/TerrainState/UrbanRefinementState > MultiScaleMap > ProductManifest。

  ## 3.1 FrameContext

  @dataclass(frozen=True)
  class FrameContext:
      frame_id: str
      timestamp_utc: datetime
      region_id: str
      grid: GridSpec
      coverage: CoverageSpec
      sat_norad_id: str
      sat_name: str | None
      sat_azimuth_deg: float
      sat_elevation_deg: float
      sat_slant_range_km: float
      sat_lat_deg: float
      sat_lon_deg: float
      sat_alt_km: float
      frequency_hz: float
      polarization_mode: str
      strict_mode: bool
      benchmark_mode: bool
      data_snapshot_id: str | None
      config_hash: str | None
      extras: dict[str, Any]

  ## 3.2 GridSpec

  @dataclass(frozen=True)
  class GridSpec:
      crs: str                 # "WGS84"
      anchor: str              # "center"
      center_lat: float
      center_lon: float
      width_m: float
      height_m: float
      nx: int
      ny: int
      dx_m: float
      dy_m: float
      pixel_registration: str  # "center"
      row_order: str           # "north_to_south"
      col_order: str           # "west_to_east"

  ## 3.3 CoverageSpec

  @dataclass(frozen=True)
  class CoverageSpec:
      coarse_grid: GridSpec
      urban_grid: GridSpec | None
      target_product_grid: GridSpec
      alignment_rule: str      # "same_center" / "same_bbox"
      crop_policy: str         # "bbox_project"
      blend_policy: str        # "coarse_plus_masked_residual"

  ## 3.4 EntryWaveState

  @dataclass
  class EntryWaveState:
      frame_id: str
      grid: GridSpec
      fspl_db: np.ndarray
      atm_db: np.ndarray
      iono_db: np.ndarray
      pol_db: np.ndarray
      gain_db: np.ndarray
      total_entry_loss_db: np.ndarray
      azimuth_deg: np.ndarray
      elevation_deg: np.ndarray
      slant_range_m: np.ndarray
      tec: np.ndarray | None
      iwv: np.ndarray | None
      valid_mask: np.ndarray
      sat_meta: dict[str, Any]

  ## 3.5 TerrainState

  @dataclass
  class TerrainState:
      frame_id: str
      grid: GridSpec
      dem_elevation_m: np.ndarray
      occlusion_mask: np.ndarray
      horizon_excess_m: np.ndarray | None
      obstacle_distance_m: np.ndarray | None
      diffraction_loss_db: np.ndarray
      metadata: dict[str, Any]

  ## 3.6 UrbanRefinementState

  @dataclass
  class UrbanRefinementState:
      frame_id: str
      coarse_parent_grid: GridSpec
      urban_grid: GridSpec
      building_height_m: np.ndarray
      occupancy_mask: np.ndarray | None
      nlos_mask: np.ndarray
      residual_loss_db: np.ndarray
      support_mask: np.ndarray
      tile_id: str | None
      metadata: dict[str, Any]

  ## 3.7 MultiScaleMap

  @dataclass
  class MultiScaleMap:
      frame_id: str
      coarse_grid: GridSpec
      urban_grid: GridSpec | None
      product_grid: GridSpec
      base_loss_db: np.ndarray
      urban_residual_db: np.ndarray | None
      urban_support_mask: np.ndarray | None
      blend_policy: str

  ## 3.8 ProductManifest

  @dataclass
  class ProductManifest:
      product_id: str
      frame_id: str
      product_type: str
      created_at_utc: datetime
      timestamp_utc: datetime
      sat_norad_id: str
      region_id: str
      config_hash: str
      code_version: str
      data_snapshot_id: str
      strict_mode: bool
      benchmark_mode: bool
      input_files: list[dict[str, Any]]
      output_files: list[dict[str, Any]]
      warnings: list[str]
      fallbacks_used: list[str]

  ———

  # 4. 推荐目录结构

  src/
    context/
      frame.py
      grid_spec.py
      coverage_spec.py
      time_utils.py
    states/
      entry_wave.py
      terrain_state.py
      urban_refinement_state.py
      multiscale_map.py
    pipeline/
      frame_builder.py
      frame_runner.py
      projector.py
      manifest_writer.py
    layers/
      base.py
      l1_macro.py
      l2_topo.py
      l3_urban.py
      legacy_adapters.py
    engine/
      aggregator.py
      compositor.py
    core/
      grid.py
      physics.py
    products/
      radiomap.py
      masks.py
      dataset_export.py
      manifest.py
    dataio/
      ionex.py
      era5.py
      dem.py
      urban_tiles.py
      snapshot.py
    utils/
      data_validation.py
      logger.py
      plotter.py
      performance.py
      tle_loader.py

  scripts/
    apps/
      generate_full_radiomap.py
      generate_multisat_timeseries_radiomap.py
      batch_city_experiments.py
    legacy/
      generate_l1_map.py
      generate_global_map.py
      visualize_batch.py

  configs/
    benchmarks/
    experiments/
    datasets/

  benchmarks/
    frame_sets/
    protocols/
    baselines/

  tests/
    test_context/
    test_states/
    test_pipeline/
    test_layers/
    test_products/

  ———

  # 5. 最小迁移方案

  ## 动作1：引入 GridSpec/CoverageSpec，移除聚合器硬编码 footprint

  - 改哪里：src/engine/aggregator.py:71、src/core/grid.py:36、层初始化配置读取处
  - 为什么：这是当前空间语义混乱的总根源之一。不改它，L1/L2/L3 永远只是"碰巧能拼起来"。
  - 预期收益：统一"地图覆盖范围、分辨率、锚点语义"。为后续 coarse-to-fine 打地基。
  - 可能副作用：旧测试和脚本里对 0.256 km 的隐式假设会失效。

  ## 动作2：新增 FrameContext，由单一入口构建每一帧

  - 改哪里：新增 src/context/frame.py、修改 main.py、修改 scripts/generate_multisat_timeseries_radiomap.py:215
  - 为什么：现在 L2/L3 同步几何靠脚本手工拼 context，不可靠。
  - 预期收益：同一帧内选星、时间、coverage、区域完全统一。为 benchmark/provenance 提供天然主键 frame_id。
  - 可能副作用：接口会有一次"看起来更重"的变化，但长期会简化脚本。

  ## 动作3：把 L2 正式接口改成基于 GridSpec/bbox，废除裸 origin

  - 改哪里：src/layers/l2_topo.py:129、相关 DEM window 裁切逻辑
  - 为什么：当前 L2 的 SW-corner 语义与全局中心语义直接冲突。这是科学一致性里最硬的一根刺。
  - 预期收益：L2 与 L1/L3 可以真正对应同一地面 footprint。后续 coarse 30m / 100m / custom resolution 都更容易切换。
  - 可能副作用：某些基于"西南角输入"的单独脚本需要适配 wrapper。

  ## 动作4：建立 strict UTC + strict no-fallback + manifest 最小闭环

  - 改哪里：main.py:111、src/layers/l1_macro.py:274、src/utils/data_validation.py、输出脚本
  - 为什么：这是最小成本提升科研可信度的动作。时间和数据快照不严格，后面的误差对比都站不住。
  - 预期收益：结果更可复现、可复验、可追责。能开始形成"论文级软件原型"的实验协议。
  - 可能副作用：更多实验会早失败，但这是正确的失败。

  ———

  # 6. 不要做的事情

  - 不要一开始重写所有层的物理公式。
  - 不要先做大规模代码风格统一，而忽略物理语义一致性。
  - 不要继续让 origin 保持模糊含义。
  - 不要让 L2 再以 SW corner 的私有语义偷偷运行。
  - 不要让 L3 继续作为与 L2 平行的全图层。
  - 不要继续在 aggregator 里硬编码目标 coverage。
  - 不要保留"没给时间就取当前 UTC"的 batch 行为。
  - 不要保留静默 fallback，尤其是 IONEX/ERA5/DEM 缺失时。
  - 不要让脚本直接拼 LayerContext.extras 充当正式架构。
  - 不要让旧脚本继续调用私有方法 _interpolate_to_target()。
  - 不要在 benchmark 之前继续引入太多新物理细节。
  - 不要把 radio map 当成唯一产物；中间传播状态同样应该可导出。
