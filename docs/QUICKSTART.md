# Quick Start Guide

This guide will help you get started with SG-MRM in 5 minutes.

## Installation

1. **Clone the repository** (or download the code):
```bash
cd SG-RM
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python main.py --help
```

## Run Your First Simulation

### Option 1: Run Example Scripts

The easiest way to get started:

```bash
# Basic usage example
python examples/basic_usage.py

# V1.0 static link example
python examples/v1_static_link.py
```

Output will be saved to `output/examples/` or `output/v1_static_link/`.

### Option 2: Run with Configuration File

```bash
python main.py --config configs/mission_config.yaml --output output/
```

This will:
1. Load configuration from `configs/mission_config.yaml`
2. Initialize L1 and L3 layers (L2 disabled by default)
3. Compute radio maps for the time range specified
4. Save results to `output/` directory

### Option 3: Programmatic Usage

Create a Python script:

```python
from datetime import datetime
from src.layers import L1MacroLayer
from src.engine import RadioMapAggregator
from src.utils import plot_radio_map

# Configure L1 layer
l1_config = {
    'grid_size': 256,
    'coverage_km': 256.0,
    'resolution_m': 1000.0,
    'frequency_ghz': 10.0,
    'satellite_altitude_km': 550.0
}

# Initialize
l1_layer = L1MacroLayer(l1_config, origin_lat=39.9, origin_lon=116.4)
aggregator = RadioMapAggregator(l1_layer=l1_layer)

# Compute
radio_map = aggregator.aggregate(datetime(2024, 1, 1, 12, 0, 0))

# Visualize
plot_radio_map(radio_map, title="My Radio Map", output_file="my_map.png")
```

## Understanding the Output

After running a simulation, you'll find:

- **PNG images**: Visual representations of radio maps
- **NPY files**: Raw numpy arrays for further processing
- **Comparison plots**: Side-by-side layer visualizations

### Reading Output Files

```python
import numpy as np
import matplotlib.pyplot as plt

# Load composite map
composite = np.load('output/composite_0000.npy')

# Print statistics
print(f"Min loss: {np.min(composite):.2f} dB")
print(f"Max loss: {np.max(composite):.2f} dB")
print(f"Mean loss: {np.mean(composite):.2f} dB")

# Plot
plt.imshow(composite, cmap='viridis')
plt.colorbar(label='Loss (dB)')
plt.title('Radio Map')
plt.show()
```

## Customizing Your Simulation

### Change Location

Edit `configs/mission_config.yaml`:

```yaml
origin:
  latitude: 40.7128   # New York
  longitude: -74.0060
```

### Change Frequency

```yaml
rf:
  frequency_ghz: 12.0  # Ku-band
```

### Change Time Range

```yaml
time:
  start: "2024-06-01T00:00:00"
  end: "2024-06-01T12:00:00"
  step_hours: 1
```

### Enable/Disable Layers

```yaml
layers:
  l1_macro:
    enabled: true
  l2_topo:
    enabled: false  # Disable terrain layer
  l3_urban:
    enabled: true
```

## Next Steps

1. **Explore examples**: Check `examples/` directory for more usage patterns
2. **Read documentation**: See `docs/` for detailed architecture and API reference
3. **Add your data**: Place TLE, DEM, or building data in `data/` directories
4. **Run tests**: `pytest tests/` to verify everything works
5. **Customize layers**: Modify layer implementations in `src/layers/`

## Common Issues

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd /path/to/SG-RM
python main.py
```

### Missing Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### No Output Generated

Check that the output directory exists and is writable:

```bash
mkdir -p output
python main.py
```

## Getting Help

- Check the [README](README.md) for overview
- Read [Architecture Guide](docs/architecture.md) for design details
- Review [examples](examples/) for usage patterns
- Open an issue on GitHub for bugs or questions

## What's Next?

Now that you have a working simulation:

1. Try different locations and frequencies
2. Add your own data sources (TLE, DEM, buildings)
3. Modify physical models in `src/core/physics.py`
4. Extend layers with new features
5. Create time-series animations

Happy simulating! 🛰️📡
