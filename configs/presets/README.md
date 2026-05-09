# Presets

Scene-oriented runnable presets for small-batch experiments and smoke tests.

## Current presets

- `xian_urban.yaml`
  - Xi'an urban core
  - scene profile: `urban_flat`
  - intended layers: `L1 + L3`
  - default product coverage: `0.256 km`
- `qinling_mountain.yaml`
  - Qinling mountain area
  - scene profile: `mountain_rural`
  - intended layers: `L1 + L2`
  - default product coverage: `25.6 km`
- `huashan_mountain.yaml`
  - Huashan mountain area
  - scene profile: `mountain_rural`
  - intended layers: `L1 + L2`
  - default product coverage: `25.6 km`
- `loess_plateau.yaml`
  - Loess Plateau area
  - scene profile: `mountain_rural`
  - intended layers: `L1 + L2`
  - default product coverage: `25.6 km`

## Important usage notes

1. These presets are date-bound examples, not fully automatic annual configs.
2. If target date changes, update at least:
   - `time.start`
   - `time.end`
   - `layers.l1_macro.tle_file`
   - ideally the referenced `ionex_file` / `era5_file` too
3. `product.coverage_km` is now explicit and matters:
   - city scene keeps product at `256 m`
   - mountain scenes export product at `25.6 km`
4. For `scripts/generate_multisat_timeseries_radiomap.py`, use timezone-aware CLI overrides:

```bash
python scripts/generate_multisat_timeseries_radiomap.py \
  --config configs/presets/huashan_mountain.yaml \
  --start 2025-05-01T00:00:00Z \
  --end 2025-05-01T00:10:00Z \
  --step-minutes 0.5 \
  --output-dir output/experiments/huashan_mountain/2025-05-01 \
  --region-id huashan_mountain \
  --save-per-satellite
```

## Relationship with smoke scripts

These presets are the source configs used by:

- `scripts/run_xian_urban_smoke.py`
- `scripts/run_qinling_smoke.py`
- `scripts/run_huashan_smoke.py`
- `scripts/run_loess_plateau_smoke.py`

Those wrappers snapshot runtime parameters into the output directory for reproducibility.
