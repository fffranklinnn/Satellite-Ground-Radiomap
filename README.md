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
python scripts/generate_full_radiomap.py --timestamp 2025-01-01T06:00:00
python scripts/report_satellite_visibility.py --start 2025-01-01T00:00:00 --end 2025-01-01T23:00:00 --step-hours 1
python scripts/generate_feature_showcase.py --output-root output/feature_showcase_demo
python scripts/check_data_integrity.py --config configs/mission_config.yaml --strict
```

More script details: [scripts/README.md](scripts/README.md)

## 5. Data Snapshot in This Repository

| Type | Path | Current role |
|---|---|---|
| TLE | `data/2025_0101.tle` | Main satellite ephemeris source for Jan 1, 2025 runs |
| IONEX | `data/l1_space/data/*.INX.gz` | TEC maps for ionosphere |
| ERA5 pressure-level | `data/l1_space/data/*.nc` | IWV extraction (`q` integration) |
| DEM | `data/l2_topo/全国DEM数据.tif` | L2 terrain blockage |
| L3 raw source | `data/l3_urban/shanxisheng/陕西省/*.shp` | Province-wide raw building vector source |
| L3 runnable cache | `data/l3_urban/xian/tiles_60/` | Ready-to-run Xi'an building cache |

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
└── branch_L1/L2/L3/ # Local historical branch snapshots (gitignored)
```

Practical optimization recommendations:

1. Keep `src/` as the single runtime source of truth; avoid editing `branch_*` trees.
2. Keep all heavy artifacts under `output/` and data downloads under `data/` (use `data/l1_space/data/cddis_data_2025/` for yearly IONEX archives; already gitignored).
3. For long-term cleanliness, archive or remove local `branch_*` snapshots once no longer needed.
4. Prefer script-based reproducibility (`scripts/`) over ad-hoc notebook outputs.

## 7. Documentation Map

- Project docs index: [docs/README.md](docs/README.md)
- Quick start: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- Config schema: [configs/README.md](configs/README.md)
- Data and acquisition: [data/README.md](data/README.md)
- Layer implementation notes: [src/layers/README.md](src/layers/README.md)
- L3 cache tooling: [tools/README.md](tools/README.md)
- Utility modules: [src/utils/README.md](src/utils/README.md)
- Tests: [tests/README.md](tests/README.md)

## 8. Testing

```bash
pytest tests/
pytest --cov=src tests/
```

## License

MIT License, see [LICENSE](LICENSE).
