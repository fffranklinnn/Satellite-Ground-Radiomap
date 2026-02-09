# SG-MRM Architecture

This document describes the architecture and design principles of the SG-MRM system.

## System Overview

SG-MRM uses a three-layer tile-stacking architecture to model electromagnetic propagation from satellite to ground at multiple scales.

## Design Principles

### 1. Physical Phenomenon Determines Layer

Each layer corresponds to a specific physical scale:

- **L1**: Satellite-scale phenomena (hundreds of km)
- **L2**: Terrain-scale phenomena (tens of km)
- **L3**: Building-scale phenomena (hundreds of m)

### 2. Standardized Output

All layers output 256×256 pixel images where:
- Each pixel represents electromagnetic loss in dB
- Grid size is always 256×256
- Physical coverage varies by layer

### 3. Modular Independence

Layers are highly cohesive and loosely coupled:
- Each layer focuses on its specific domain
- Layers can be enabled/disabled independently
- No direct dependencies between layers

### 4. dB Domain Aggregation

Losses are combined using logarithmic addition:

```
Total Loss (dB) = L1 + L2 + L3
```

This is physically correct for independent loss mechanisms.

## Layer Details

### L1 Macro Layer

**Coverage**: 256 km × 256 km
**Resolution**: 1000 m/pixel
**Grid**: 256 × 256

**Physical Models**:
- Free Space Path Loss (FSPL)
- Atmospheric attenuation (ITU-R P.618)
- Ionospheric effects (ITU-R P.531)
- Satellite antenna gain pattern

**Inputs**:
- Satellite TLE (orbit data)
- Antenna pattern file
- Atmospheric parameters

**Output**: 256×256 array of loss values (dB)

### L2 Terrain Layer

**Coverage**: 25.6 km × 25.6 km
**Resolution**: 100 m/pixel
**Grid**: 256 × 256

**Physical Models**:
- DEM-based line-of-sight analysis
- Terrain obstruction calculation
- Knife-edge diffraction (ITU-R P.526)

**Inputs**:
- DEM data (.tif or .hgt)
- Satellite elevation angle

**Output**: 256×256 array of loss values (dB)

### L3 Urban Layer

**Coverage**: 256 m × 256 m
**Resolution**: 1 m/pixel
**Grid**: 256 × 256

**Physical Models**:
- Building shadow calculation
- Ray tracing for multipath (V2.0)
- Material-based reflection/transmission

**Inputs**:
- Building shapefiles with heights
- Satellite azimuth and elevation

**Output**: 256×256 array of loss values (dB)

## Class Hierarchy

```
BaseLayer (Abstract)
├── L1MacroLayer
├── L2TopoLayer
└── L3UrbanLayer

RadioMapAggregator
├── Combines layer outputs
└── Handles interpolation

Grid
└── Coordinate transformations
```

## Data Flow

```
Configuration (YAML)
    ↓
Layer Initialization
    ↓
For each timestamp:
    ├── L1.compute(timestamp) → 256×256 array
    ├── L2.compute(timestamp) → 256×256 array
    ├── L3.compute(timestamp) → 256×256 array
    ↓
Aggregator.aggregate()
    ├── Interpolate L1 to L3 scale
    ├── Interpolate L2 to L3 scale
    ├── Sum in dB domain
    ↓
Composite 256×256 radio map
    ↓
Visualization / Export
```

## Coordinate Systems

### Geographic Coordinates

- Origin: (latitude, longitude) in degrees
- All layers share the same origin
- WGS84 datum

### Pixel Coordinates

- (i, j) where i is row, j is column
- (0, 0) is top-left corner
- (128, 128) is center (at origin)

### Coordinate Transformations

The `Grid` class handles conversions:

```python
# Pixel to geographic
lat, lon = grid.pixel_to_latlon(i, j)

# Geographic to pixel
i, j = grid.latlon_to_pixel(lat, lon)
```

## Interpolation Strategy

When combining layers with different coverages:

1. **L1 → L3**: Extract center 256m region from 256km coverage
2. **L2 → L3**: Extract center 256m region from 25.6km coverage
3. Use bilinear interpolation for smooth transitions

## Performance Considerations

### V1.0 (Current)

- Single-threaded computation
- Python-based calculations
- Suitable for small-scale simulations

### V2.0 (Planned)

- Parallel layer computation
- GPU-accelerated ray tracing
- Optimized for large-scale time-series

## Extension Points

### Adding New Layers

1. Inherit from `BaseLayer`
2. Implement `compute(timestamp)` method
3. Return 256×256 numpy array
4. Add to aggregator

### Adding New Physical Models

1. Add functions to `src/core/physics.py`
2. Use in layer `compute()` methods
3. Document assumptions and references

### Custom Data Sources

1. Add loading functions to layer classes
2. Support standard formats (GeoTIFF, Shapefile, etc.)
3. Handle coordinate transformations

## References

- ITU-R P.618: Propagation data and prediction methods
- ITU-R P.531: Ionospheric propagation data
- ITU-R P.526: Propagation by diffraction
