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

## Dataset sample spatial semantics (current implementation)

This point is important for downstream dataset use.

### Raw layer scale

Each layer has its own native physical coverage:

- L1: `256 km × 256 km` at `1000 m/px`
- L2: `25.6 km × 25.6 km` at `100 m/px`
- L3: `256 m × 256 m` at `1 m/px`

### Exported tile-level sample scale

When a tile-level dataset sample is exported from `generate_xian_city_radiomap.py`, the arrays written into `.npz` are all aligned onto the **L3 tile grid**.

So for the current prototype / pilot dataset:

- `grid_shape = [256, 256]`
- physical footprint = **`256 m × 256 m`**
- effective spatial resolution = **`1 m/px`**

This means:
- `l1` inside the sample is **not** an L1-native 1000 m/px raster anymore; it is the L1 contribution after being interpolated/aligned to the tile grid.
- `l2` inside the sample is **not** an L2-native 100 m/px raster anymore; it is the L2 contribution after being interpolated/aligned to the tile grid.
- `l3`, `height`, and `occ` remain naturally interpretable on the tile grid.

In other words, the sample is best understood as:

> a **tile-aligned multi-component physical sample**, not three raw-scale maps simply stored together.

## Physical consistency checklist (current dataset view)

For dataset construction, physical consistency should be checked before split strategy or large-scale expansion.

### Level 0: export integrity

These are structural invariants and should always hold:

1. `composite == l1 + l2 + l3`
2. all exported arrays share the same `grid_shape`
3. `occ` remains binary
4. manifest paths resolve to real sample files

### Level 1: conditional consistency

When only one condition axis changes, the resulting change should match physical intuition.

#### Rain-rate sweep
If only `rain_rate_mm_h` changes:
- primary change should come from the L1-side attenuation chain
- L2 terrain contribution should remain unchanged for the same tile/time/geometry
- L3 building contribution should remain unchanged for the same tile

#### Satellite sweep
If only `satellite_norad_id` changes:
- geometry-dependent L1 contribution should change
- incident direction reaching L3 may change
- L2/L3 effects may change indirectly because the viewing geometry changes
- but static geographic metadata (`tile_id`, `origin_lat/lon`, `grid_shape`) should remain unchanged

#### Timestamp sweep
If only `timestamp_utc` changes:
- the change should be explainable by time-varying geometry / atmosphere / ionosphere
- not by random variation in array layout or sample structure

### Level 2: semantic consistency

The dataset should preserve clear meaning for each array:
- `composite`: final tile-level loss map
- `l1`: tile-aligned macro contribution
- `l2`: tile-aligned terrain contribution
- `l3`: tile-aligned urban contribution
- `height`: tile-aligned urban geometry context
- `occ`: tile-aligned occupancy / obstruction mask

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
