# Configuration Directory

This directory contains configuration files for the SG-MRM simulation system.

## Files

### mission_config.yaml

The main configuration file that defines all simulation parameters:

- **Mission metadata**: Name, description, version
- **Geographic origin**: Latitude, longitude, altitude
- **RF parameters**: Frequency, transmit power, polarization
- **Satellite parameters**: Altitude, TLE file path
- **Time parameters**: Start/end time, time step
- **Layer configurations**: Settings for L1/L2/L3 layers
- **Output configuration**: Output directory, format, visualization settings
- **Performance settings**: Profiling, parallel processing
- **Logging configuration**: Log level, log file

## Usage

Load the configuration in your Python code:

```python
import yaml

with open('configs/mission_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access configuration values
origin_lat = config['origin']['latitude']
frequency = config['rf']['frequency_ghz']
```

## Customization

To create a custom configuration:

1. Copy `mission_config.yaml` to a new file (e.g., `my_mission.yaml`)
2. Modify the parameters as needed
3. Run the simulation with your custom config:

```bash
python main.py --config configs/my_mission.yaml
```

## Layer Configuration

Each layer (L1/L2/L3) has specific parameters:

### L1 Macro Layer
- Coverage: 256 km × 256 km
- Resolution: 1000 m/pixel
- Handles: Satellite positioning, atmospheric effects, ionospheric effects

### L2 Terrain Layer
- Coverage: 25.6 km × 25.6 km
- Resolution: 100 m/pixel
- Handles: DEM data, terrain obstruction

### L3 Urban Layer
- Coverage: 256 m × 256 m
- Resolution: 1 m/pixel
- Handles: Building distribution, shadows, ray tracing

## Version Notes

- **V1.0**: L1 and L3 enabled, L2 disabled (flat terrain assumption)
- **V2.0**: All layers enabled with full physics models
