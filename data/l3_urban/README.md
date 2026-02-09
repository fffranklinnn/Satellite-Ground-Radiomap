# L3 Urban Layer Data

This directory contains building and urban structure data.

## Required Data

- **Building shapefiles**: .shp format with building footprints and heights
- **3D city models**: .obj or .gltf format (optional for V2.0)

## Data Sources

- OpenStreetMap: https://www.openstreetmap.org/
- Local GIS databases
- City planning departments

## Shapefile Requirements

Required attributes:
- `geometry`: Polygon geometry (building footprint)
- `height`: Building height in meters

## Example

Extract buildings from OpenStreetMap:

```python
import osmnx as ox
buildings = ox.geometries_from_place("Your City", tags={'building': True})
buildings.to_file("data/l3_urban/buildings.shp")
```
