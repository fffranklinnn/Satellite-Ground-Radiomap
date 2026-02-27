# SG-MRM: Satellite-Ground Multiscale Radio Map

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文版 README](README_CN.md)

A multi-scale electromagnetic loss mapping system for satellite-to-ground radio propagation analysis.

## Overview

SG-MRM generates high-resolution electromagnetic loss maps by combining three physical layers:

- **L1 Macro Layer** (256 km, 1000 m/px): TLE orbit propagation, FSPL, 31×31 phased-array antenna gain, atmospheric attenuation (ERA5 IWV), ionospheric effects (IONEX TEC)
- **L2 Terrain Layer** (25.6 km, 100 m/px): GeoTIFF DEM loading & resampling, vectorized cumulative-max LOS occlusion, diffraction loss
- **L3 Urban Layer** (256 m, 1 m/px): Building height raster tile cache, directional NLoS scan, occlusion/occupancy loss

Each layer outputs a 256×256 float32 loss matrix in dB, combined via dB-domain summation:

```
Composite Loss (dB) = Interp(L1) + Interp(L2) + L3
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: numpy, scipy, matplotlib, pyyaml, skyfield, rasterio, geopandas, shapely, pyproj, pyarrow, pandas

## Quick Start

### Run full simulation

```bash
python main.py --config configs/mission_config.yaml --output output/
```

### Programmatic API

```python
from datetime import datetime
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext
import yaml

config = yaml.safe_load(open('configs/mission_config.yaml'))
lat, lon = config['origin']['latitude'], config['origin']['longitude']

l1 = L1MacroLayer(config['layers']['l1_macro'], lat, lon)
l2 = L2TopoLayer(config['layers']['l2_topo'], lat, lon)
l3 = L3UrbanLayer(config['layers']['l3_urban'], lat, lon)

agg = RadioMapAggregator(l1, l2, l3)
ctx = LayerContext.from_any({'incident_dir': config['layers']['l3_urban']['incident_dir']})
composite = agg.aggregate(lat, lon, timestamp=datetime(2025, 1, 1, 6, 0, 0), context=ctx)
# composite: (256, 256) float32, dB
```

### Visualization scripts

```bash
python scripts/generate_l1_map.py              # L1 hourly + parameter sweeps
python scripts/generate_global_comparison.py    # 6-city global comparison
python scripts/generate_global_map.py           # Global loss map (720×360)
```

## Project Structure

```
Satellite-Ground-Radiomap/
├── configs/              # Simulation config (YAML)
├── data/
│   ├── 2025_0101.tle     # Starlink TLE orbit data
│   ├── l1_space/data/    # IONEX TEC + ERA5 atmospheric data
│   ├── l2_topo/          # National DEM GeoTIFF
│   └── l3_urban/         # Building tile cache (H.npy/Occ.npy)
├── src/
│   ├── core/             # Grid coordinate system + RF physics
│   ├── layers/           # L1/L2/L3 layer implementations
│   ├── engine/           # Multi-layer aggregation engine
│   └── utils/            # Data loaders, plotting, profiling
├── tools/                # L3 tile cache build tools
├── scripts/              # Visualization & batch generation
├── tests/                # Unit tests
├── examples/             # Usage examples
├── docs/                 # Architecture docs
└── main.py               # Entry point
```

## Current Data (Xi'an, Shaanxi)

| Data | File | Description |
|------|------|-------------|
| TLE | `data/2025_0101.tle` | Starlink constellation, 14918 sats (2025-01-01) |
| IONEX | `data/l1_space/data/*.INX.gz` | UPC GIM global TEC (15 min interval) |
| ERA5 | `data/l1_space/data/*.nc` | ECMWF pressure-level data (z/r/q/t) |
| DEM | `data/l2_topo/全国DEM数据.tif` | National DEM (~30 m resolution) |
| Buildings | `data/l3_urban/xian/tiles_60/` | Xi'an urban core, 1320 tiles (256 m, 1 m/px) |

## Roadmap

### V1.0 (Current) ✓

- [x] L1: TLE satellite selection + FSPL + phased-array gain + atmospheric/ionospheric loss
- [x] L2: GeoTIFF DEM loading + vectorized LOS occlusion + diffraction loss
- [x] L3: Tile cache building raster + directional NLoS scan + occlusion/occupancy loss
- [x] Multi-layer aggregation engine (dB-domain summation + bilinear interpolation)
- [x] Data loaders: IONEX, ERA5, TLE
- [x] Visualization tools + unit tests

### V2.0 (Planned)

- [ ] ITU-R P.526 Fresnel-Kirchhoff knife-edge diffraction (L2)
- [ ] GPU ray tracing for multipath (L3)
- [ ] Full ITU-R P.618 rain attenuation model
- [ ] Time-series animation generation
- [ ] Parallel computation support

## Testing

```bash
pytest tests/
pytest --cov=src tests/
```

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Quick Start](docs/QUICKSTART.md)
- [中文版 README](README_CN.md)

## License

MIT License - see [LICENSE](LICENSE)
