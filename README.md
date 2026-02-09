# SG-MRM: Satellite-Ground Multiscale Radio Map

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-scale electromagnetic mapping simulation system for satellite-to-ground radio propagation analysis.

## Overview

SG-MRM (Satellite-Ground Multiscale Radio Map) is a Python-based simulation framework that generates high-resolution electromagnetic loss maps by combining three physical layers:

- **L1 Macro Layer** (256 km coverage, 1000 m/pixel): Satellite positioning, atmospheric effects, ionospheric effects
- **L2 Terrain Layer** (25.6 km coverage, 100 m/pixel): DEM-based terrain obstruction
- **L3 Urban Layer** (256 m coverage, 1 m/pixel): Building distribution, shadows, ray tracing

Each layer outputs a standardized 256×256 pixel image representing electromagnetic loss in dB, which are combined using a tile-stacking approach.

## Features

- **Multi-scale Architecture**: Three-layer system covering macro, terrain, and micro scales
- **Standardized Interface**: All layers implement a common `.compute()` interface
- **Modular Design**: High cohesion, low coupling between layers
- **Time-Series Support**: Generate dynamic radio maps over time
- **Visualization Tools**: Built-in plotting and animation capabilities
- **Extensible**: Easy to add new physical models and data sources

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Development Installation

For development with editable install:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from datetime import datetime
from src.layers import L1MacroLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.utils import plot_radio_map

# Define origin
origin_lat, origin_lon = 39.9042, 116.4074

# Configure L1 layer
l1_config = {
    'grid_size': 256,
    'coverage_km': 256.0,
    'resolution_m': 1000.0,
    'frequency_ghz': 10.0,
    'satellite_altitude_km': 550.0
}

# Initialize layers
l1_layer = L1MacroLayer(l1_config, origin_lat, origin_lon)

# Create aggregator
aggregator = RadioMapAggregator(l1_layer=l1_layer)

# Compute radio map
timestamp = datetime(2024, 1, 1, 12, 0, 0)
radio_map = aggregator.aggregate(timestamp)

# Visualize
plot_radio_map(radio_map, title="Radio Map", output_file="output.png")
```

### Using Configuration File

```bash
python main.py --config configs/mission_config.yaml --output output/
```

### Run Examples

```bash
# Basic usage example
python examples/basic_usage.py

# V1.0 static link example
python examples/v1_static_link.py
```

## Project Structure

```
SG-RM/
├── configs/              # Configuration files
├── data/                 # Data sources (TLE, DEM, shapefiles)
├── src/
│   ├── core/            # Grid system and physics utilities
│   ├── layers/          # L1/L2/L3 layer implementations
│   ├── engine/          # Aggregation engine
│   └── utils/           # Logging, plotting, performance tools
├── tests/               # Unit tests
├── examples/            # Usage examples
├── docs/                # Documentation
└── main.py              # Main entry point
```

## Development Roadmap

### V1.0: Static Link Closure ✓

- [x] L1 basic satellite positioning
- [x] L3 building shadow calculation
- [x] Layer aggregation
- [x] Basic visualization

### V2.0: Full Dynamic Simulation (Planned)

- [ ] TLE-based orbit propagation
- [ ] Full ITU-R P.618 atmospheric model
- [ ] DEM-based terrain obstruction (L2)
- [ ] GPU ray tracing for multipath
- [ ] Time-series animation
- [ ] Real-time weather integration

## Architecture

### Three-Layer Design

The system follows a "tile stacking" architecture where each layer operates independently:

1. **L1 Macro Layer**: Computes wide-area satellite coverage including free space path loss, atmospheric attenuation, and ionospheric effects

2. **L2 Terrain Layer**: Processes DEM data to calculate terrain-induced obstructions and diffraction losses

3. **L3 Urban Layer**: Analyzes building distributions for shadow casting and multipath effects

### Aggregation Formula

```
Composite Loss (dB) = L1 + Interpolate(L2) + L3
```

Losses are combined in the dB domain (logarithmic addition).

## Configuration

Edit `configs/mission_config.yaml` to customize:

- Geographic origin (latitude, longitude)
- RF parameters (frequency, power, polarization)
- Satellite parameters (altitude, TLE file)
- Time range and resolution
- Layer enable/disable flags
- Output format and visualization settings

## Testing

Run unit tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Documentation

- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Development Guide](docs/development_guide.md)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{sg_mrm_2024,
  title = {SG-MRM: Satellite-Ground Multiscale Radio Map},
  author = {SG-MRM Development Team},
  year = {2024},
  url = {https://github.com/yourusername/SG-RM}
}
```

## Contact

For questions and support, please open an issue on GitHub.

## Acknowledgments

- ITU-R recommendations for propagation models
- Open-source geospatial data providers
- Python scientific computing community
