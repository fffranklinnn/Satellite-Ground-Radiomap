# SG-MRM: Satellite-Ground Multiscale Radio Map

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文版 README](README_CN.md)

SG-MRM is a multi-scale radio propagation mapping framework for satellite-to-ground links.

## 1. What It Computes

For each timestamp and region center, SG-MRM outputs a 256x256 loss map (dB) per layer and an aggregated map:

```text
L_total(dB) = Interp(L1 -> L3 footprint) + Interp(L2 -> L3 footprint) + L3
```

### Layer scope

| Layer | Spatial scale | Resolution | Main effects |
|---|---:|---:|---|
| L1 Macro | 256 km x 256 km | 1000 m/px | TLE-based satellite selection, FSPL, phased-array gain, atmospheric loss, ionospheric TEC loss |
| L2 Terrain | 25.6 km x 25.6 km | 100 m/px | DEM loading, LOS blockage, knife-edge-style diffraction loss |
| L3 Urban | 256 m x 256 m | 1 m/px | Building NLoS mask, occupancy-based loss |

## 2. Current Capability Status (Code Reality)

### Implemented

- L1:
  - Visible-satellite enumeration and best-satellite selection from TLE (`Skyfield`)
  - FSPL + phased-array Gaussian beam + polarization mismatch
  - IONEX TEC ingestion with optional IPP projection and VTEC->STEC mapping
  - Optional Faraday-rotation-induced extra polarization loss (requires optional geomagnetic backend)
  - ERA5 pressure-level loading (`q` integration -> IWV) and IWV-based atmospheric attenuation
- L2:
  - Window-based DEM read + bilinear resampling
  - Vectorized directional occlusion scan
  - Diffraction loss derived from occlusion profile (capped)
- L3:
  - Tile-cache loading (`H.npy` / `Occ.npy`)
  - Directional NLoS scan
  - NLoS / occupancy loss mapping
- Engine and tooling:
  - Multi-layer aggregation with interpolation
  - Batch scripts, province/city stitching scripts, satellite-visibility reporting

### Not fully ITU-complete yet

- No full ITU-R P.618 statistical availability chain (`R0.01`, long-term exceedance conversion)
- No pressure-level line-by-line gaseous attenuation integration (P.676 full form)
- No scintillation (`S4`) model in main pipeline
- No full multipath/ray-tracing kernel for L3

## 3. Installation

```bash
pip install -r requirements.txt
```

Optional datasets and backends are documented in [data/README.md](data/README.md).

### Optional: PyTorch in `sgmrm_test` (for future hotspot migration)

```bash
conda activate sgmrm_test
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Validate:

```bash
python - << 'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## 4. Quick Start

### Main simulation

```bash
python main.py --config configs/mission_config.yaml --output output/
```

### Common script entry points

```bash
python main.py --config configs/mission_config.yaml --check-data-only
python main.py --config configs/mission_config.yaml --strict-data
python scripts/report_satellite_visibility.py --config configs/mission_config.yaml --step-hours 1
python scripts/generate_multisat_timeseries_radiomap.py --config configs/mission_config.yaml --start 2025-01-01T00:00:00 --end 2025-01-01T03:00:00 --step-minutes 30
python scripts/check_data_integrity.py --config configs/mission_config.yaml --strict
```

`generate_full_radiomap.py` and `generate_feature_showcase.py` still run, but they are thin wrappers over `scripts/legacy/*`.

More script details: [scripts/README.md](scripts/README.md)

### Recommended scene smoke runs

For the current repo state, the most practical verification entry points are the scene presets + smoke scripts:

```bash
python scripts/run_xian_urban_smoke.py --output-root output/verify --gpu-id 0
python scripts/run_qinling_smoke.py --output-root output/verify --gpu-id 0
python scripts/run_huashan_smoke.py --output-root output/verify --gpu-id 0
python scripts/run_loess_plateau_smoke.py --output-root output/verify --gpu-id 0
```

These scripts automatically:

- copy the selected preset into the run directory
- persist runtime parameters (`run_config.yaml`, `run_manifest.json`, `preset_config.yaml`)
- call `scripts/generate_multisat_timeseries_radiomap.py`

### Time-series dataset generation

Example: generate Xi'an frames on 2025-05-01 every 30 seconds.

```bash
python scripts/generate_multisat_timeseries_radiomap.py \
  --config configs/presets/xian_urban.yaml \
  --start 2025-05-01T00:00:00Z \
  --end 2025-05-01T00:59:30Z \
  --step-minutes 0.5 \
  --fusion-mode best-server \
  --output-dir output/experiments/xian_urban/2025-05-01 \
  --region-id xian_urban \
  --save-per-satellite
```

Important:

- use timezone-aware timestamps (`Z` or `+00:00`)
- if target date changes, update `layers.l1_macro.tle_file`
- `best-server` is pixel-wise winner-take-all over satellites
- `soft-combine` is power-domain fusion across satellites

### Re-label existing PNG outputs without recomputing physics

```bash
python scripts/relabel_radiomap_pngs.py output/experiments/huashan_mountain/2025-05-01
```

This only re-renders PNGs from saved `.npy + .json`; it does not recompute L1/L2/L3.

## 5. Data Snapshot in This Repository

| Type | Path | Current role |
|---|---|---|
| TLE | `data/starlink-2025-tle/2025-01-01.tle` | Main satellite ephemeris source |
| IONEX | `data/l1_space/data/*.INX.gz` | TEC maps for ionosphere |
| ERA5 pressure-level | `data/l1_space/data/*.nc` | IWV extraction (`q` integration) |
| DEM | `data/l2_topo/china_dem_30.tif` | L2 terrain blockage |
| L3 raw source | `data/l3_urban/shanxisheng/陕西省/*.shp` | Province-wide raw building vector source |
| L3 runnable cache | `data/l3_urban/xian/tiles_60/` | Ready-to-run Xi'an building cache |

Notes:

- full-year TLE and IONEX may exist locally, but actual runtime depends on the path referenced by each config
- DEM is static and reusable across dates
- ERA5 availability is the most likely remaining blocker for new dates/regions

## 6. Repository Structure and Optimization Notes

```text
Satellite-Ground-Radiomap/
├── src/             # Runtime code (core/layers/engine/utils)
├── scripts/         # Batch and figure-generation entry points
├── configs/         # YAML configs
├── data/            # Data stubs + local heavy data roots
├── docs/            # Guides and summaries
├── tests/           # Unit tests
├── examples/        # Legacy demos
├── output/          # Generated artifacts (gitignored)
└── change/          # Local historical snapshot area (non-runtime)
```

Practical optimization recommendations:

1. Keep `src/` as the single runtime source of truth; avoid editing historical snapshot trees.
2. Keep all heavy artifacts under `output/` and data downloads under `data/` (use `data/l1_space/data/cddis_data_2025/` for yearly IONEX archives; already gitignored).
3. For long-term cleanliness, archive or remove local historical snapshots once no longer needed.
4. Prefer script-based reproducibility (`scripts/`) over ad-hoc notebook outputs.

## 7. Output Semantics

Typical time-series runs produce:

- `png/`: rendered radiomap images
- `npy/`: raw path-loss arrays
- `frame_json/`: per-frame metadata, satellite list, fusion stats
- `per_satellite/<timestamp_region>/`: per-satellite PNG/NPY/JSON artifacts
- `manifest.jsonl`: append-only run manifest

Current default PNG semantics:

- colorbar is `Loss (dB, lower is better)`
- lower dB means less propagation loss and therefore stronger effective reception
- brighter colors in `viridis` do not mean “better”; always follow the colorbar label first
- if you only want to refresh labels/styles, use `scripts/relabel_radiomap_pngs.py`

## 8. Documentation Map

- Project docs index: [docs/README.md](docs/README.md)
- Quick start: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- Config schema: [configs/README.md](configs/README.md)
- Data and acquisition: [data/README.md](data/README.md)
- Source tree index: [src/README.md](src/README.md)
- Layer implementation notes: [src/layers/README.md](src/layers/README.md)
- L3 cache tooling: [tools/README.md](tools/README.md)
- Utility modules: [src/utils/README.md](src/utils/README.md)
- Tests: [tests/README.md](tests/README.md)

## 9. Testing

```bash
pytest tests/
pytest --cov=src tests/
```

## License

MIT License, see [LICENSE](LICENSE).
