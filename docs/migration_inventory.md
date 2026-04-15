# Migration Inventory: GridSpec / CoverageSpec Adoption

Generated: 2026-04-15
Status: task1b — enumerate all grid construction sites requiring migration

---

## Summary

| Category | Count | Status |
|---|---|---|
| Layer `__init__` (bare origin_lat/lon) | 3 | Needs migration (tasks 7–14) |
| Layer `compute()` (bare origin_lat/lon) | 3 | Needs migration (tasks 7–14) |
| `src/core/grid.py` standalone functions | 2 | Needs migration |
| `src/engine/aggregator.py` | 1 | **Migrated** (Round 1) |
| `scripts/` callers | 5 files | Deferred (post-AC-5) |
| `benchmarks/` callers | 2 files | Deferred (post-AC-5) |
| `src/context/` (new) | 4 files | **Done** (Round 1) |
| `src/layers/legacy_adapters.py` (new) | 1 file | **Done** (Round 1) |

---

## Migrated (Round 1)

### `src/engine/aggregator.py`
- **Change**: Added `coverage_spec: Optional[CoverageSpec]` and `strict: bool` parameters.
- **Hardcode removed**: `0.256` km replaced by `CoverageSpec.target_product_grid.width_m / 1000.0`.
- **Backward compat**: `_LEGACY_TARGET_COVERAGE_KM = 0.256` with `DeprecationWarning` when no spec provided.
- **Strict mode**: `ConfigurationError` raised when `strict=True` and no spec.

---

## Needs Migration — Layer `__init__` (bare origin_lat/lon)

### `src/layers/base.py:83`
```python
def __init__(self, config: Dict[str, Any], origin_lat: float, origin_lon: float):
    self.origin_lat = origin_lat
    self.origin_lon = origin_lon
    self.coverage_km = config.get('coverage_km')
```
- **Convention**: center-anchored (L1/L3 convention).
- **Migration target**: Accept `GridSpec` directly; keep bare-args path with `DeprecationWarning`.
- **Blocking tasks**: task7 (FrameContext), task8 (typed states).

### `src/layers/l1_macro.py:125–160`
```python
def __init__(self, config, origin_lat=None, origin_lon=None):
    ...
    super().__init__(base_config, origin_lat, origin_lon)
```
- **Convention**: center-anchored (L1 convention).
- **Migration target**: Accept `GridSpec`; derive `origin_lat/lon` from `grid_spec.center_lat/lon`.
- **Blocking tasks**: task8 (EntryWaveState).

### `src/layers/l2_topo.py:54–89`
```python
def __init__(self, config, origin_lat=None, origin_lon=None):
    ...
    super().__init__(base_config, origin_lat, origin_lon)
```
- **Convention**: **SW-corner** (L2 legacy bug — documented in `baselines/spatial_delta_report.json`).
- **Migration target**: Accept `GridSpec.from_sw_corner()`; internal DEM load uses SW corner.
- **Blocking tasks**: task9 (TerrainState), task10 (L2 SW-corner fix).

### `src/layers/l3_urban.py:165–196`
```python
def __init__(self, config, origin_lat=None, origin_lon=None):
    ...
    super().__init__(base_config, origin_lat, origin_lon)
```
- **Convention**: center-anchored (L3 convention).
- **Migration target**: Accept `GridSpec`; tile_id derived from `grid_spec.center_lat/lon`.
- **Blocking tasks**: task11 (UrbanRefinementState).

---

## Needs Migration — Layer `compute()` (bare origin_lat/lon)

### `src/layers/l1_macro.py:326–354`
```python
def compute(self, origin_lat=None, origin_lon=None, timestamp=None, context=None):
    if origin_lat is None:
        origin_lat = self.origin_lat
    ...
    lat_grid, lon_grid, x_m, y_m = get_grid_latlon(origin_lat, origin_lon)
```
- **Migration target**: Accept `GridSpec` via `context`; fall back to `self.origin_lat/lon` with `DeprecationWarning`.

### `src/layers/l2_topo.py:120–159`
```python
def compute(self, origin_lat=None, origin_lon=None, timestamp=None, context=None):
    ...
    dem_grid = self._load_dem_patch(origin_lat, origin_lon)
```
- **SW-corner bug site**: `_load_dem_patch` at line 185 uses `origin_lat` as SW corner.
- **Migration target**: Extract SW corner from `GridSpec.sw_corner()`.

### `src/layers/l3_urban.py:263–305`
```python
def compute(self, origin_lat=None, origin_lon=None, timestamp=None, context=None):
    ...
    tile_origin = {"lat": origin_lat, "lon": origin_lon}
```
- **Migration target**: Derive `tile_origin` from `GridSpec.center_lat/lon`.

---

## Needs Migration — `src/core/grid.py`

### `src/core/grid.py:163` — `get_grid_latlon()`
```python
def get_grid_latlon(origin_lat, origin_lon, coverage_m=..., grid_size=256):
```
- Used by `l1_macro.py:358` and `l3_urban.py` indirectly.
- **Migration target**: Add `GridSpec`-accepting overload; keep bare-args path with `DeprecationWarning`.

### `src/core/grid.py:29` — `RadioGrid.__init__()`
```python
def __init__(self, origin_lat, origin_lon, grid_size=256):
    self.origin_lat = origin_lat
    self.origin_lon = origin_lon
```
- **Migration target**: Accept `GridSpec`; derive `origin_lat/lon` from center.

---

## Deferred — Scripts (post-AC-5)

| File | Lines | Notes |
|---|---|---|
| `scripts/batch_generate_all.py` | 86, 105, 117, 137–161, 366, 420, 468, 542, 655 | Multiple hardcoded `coverage_km` values; L2 called with bare origin |
| `scripts/generate_global_map.py` | TBD | Calls L1/L2/L3 with bare origin |
| `scripts/generate_global_comparison.py` | TBD | Same pattern |
| `scripts/generate_l1_map.py` | TBD | L1 only |
| `scripts/generate_feature_showcase.py` | 100–265 | Inline `coverage_km` arithmetic |

Scripts are deferred because they are visualization/export tools that depend on the migrated layer interfaces. They will be updated in task22 (dataset export) and task23 (product projectors).

---

## Deferred — Benchmarks (post-AC-5)

| File | Notes |
|---|---|
| `benchmarks/capture_golden_scenes.py` | Uses `RadioMapAggregator` directly; will adopt `CoverageSpec` in task15 |
| `benchmarks/measure_performance_baseline.py` | Timing harness; no grid construction |

---

## L2 SW-Corner Bug — Evidence

Documented in `benchmarks/baselines/spatial_delta_report.json`:

- L2 `origin_lat/lon` is treated as SW corner internally (`l2_topo.py:192–197`).
- L1/L3 treat the same coordinates as center.
- Mismatch: **+0.1153° lat, +0.1397° lon** (~12.8 km each direction).
- Aggregator crops to 0.256 km center of 25.6 km tile → L2 contribution is all-zeros in current pipeline.

Fix path: `GridSpec.from_sw_corner()` → `sw_corner()` → pass to `_load_dem_patch()`.

---

## Migration Priority Order

1. `src/context/` — **Done** (GridSpec, CoverageSpec, time_utils, legacy_adapters)
2. `src/engine/aggregator.py` — **Done** (CoverageSpec integration)
3. `src/layers/base.py` + L1/L2/L3 `__init__` — task8–11
4. `src/layers/l2_topo.py` SW-corner fix — task10
5. `src/core/grid.py` — task12 (vertical slice)
6. Scripts — task22–23
