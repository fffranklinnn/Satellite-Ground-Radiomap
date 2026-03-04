# Data Directory

This directory contains raw data sources for the SG-MRM simulation system.

## Directory Structure

```
data/
├── l1_space/     # L1 Macro Layer data
├── l2_topo/      # L2 Terrain Layer data
└── l3_urban/     # L3 Urban Layer data
```

## Data Sources by Layer

### L1 Space Data (`l1_space/`)

Data for satellite positioning and atmospheric effects:

- **TLE Files** (.txt): Two-Line Element sets for satellite orbit propagation
  - Format: Standard TLE format
  - Source: [CelesTrak](https://celestrak.com/), [Space-Track](https://www.space-track.org/)
  - Example: `satellite.tle`

- **Antenna Patterns** (.csv, .mat): Satellite antenna gain patterns
  - Format: Elevation/Azimuth vs Gain (dBi)
  - Example: `antenna_pattern.csv`

- **Weather Data** (.nc, .grib): Atmospheric parameters
  - Format: NetCDF or GRIB
  - Parameters: Temperature, pressure, humidity, rain rate
  - Source: [ECMWF](https://www.ecmwf.int/), [NOAA](https://www.noaa.gov/)

Current repository includes practical download scripts:
- `data/l1_space/data/cds.py` (ERA5 pressure-level via CDS API)
- `data/l1_space/data/NASAcddis.py` (IONEX daily batch download)

### L2 Topography Data (`l2_topo/`)

Digital Elevation Model (DEM) data:

- **DEM Files** (.tif, .hgt): Terrain elevation data
  - Format: GeoTIFF or SRTM HGT
  - Resolution: 30m, 90m, or higher
  - Source: [SRTM](https://www2.jpl.nasa.gov/srtm/), [ASTER GDEM](https://asterweb.jpl.nasa.gov/gdem.asp)
  - Example: `N39E116.hgt` (SRTM tile)

### L3 Urban Data (`l3_urban/`)

Building and urban structure data:

- **Building Shapefiles** (.shp): Building footprints with heights
  - Format: ESRI Shapefile
  - Attributes: Building height, material (optional)
  - Source: [OpenStreetMap](https://www.openstreetmap.org/), local GIS databases
  - Example: `buildings.shp`

Current repository includes Shaanxi-wide raw building shapefiles at:
- `data/l3_urban/shanxisheng/陕西省/*.shp`
and Xi'an ready-to-use cache at:
- `data/l3_urban/xian/tiles_60/`

- **3D City Models** (.obj, .gltf): Detailed 3D urban models (V2.0)
  - Format: OBJ, glTF, or CityGML
  - For advanced ray tracing

## Data Acquisition

### Free Data Sources

1. **Satellite TLE**: [CelesTrak](https://celestrak.com/NORAD/elements/)
2. **DEM Data**: [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
3. **Building Data**: [OpenStreetMap](https://www.openstreetmap.org/)

### Data Preparation

#### TLE Files

Download TLE data for your satellite:

```bash
# Example: Download Starlink TLE
wget https://celestrak.com/NORAD/elements/starlink.txt -O data/l1_space/starlink.tle
```

#### DEM Data

1. Download SRTM tiles covering your area of interest
2. Place .hgt files in `data/l2_topo/`
3. Update `mission_config.yaml` with file path

#### Building Shapefiles

Extract from OpenStreetMap:

```python
import osmnx as ox

# Download buildings for a location
buildings = ox.geometries_from_place("Beijing, China", tags={'building': True})
buildings.to_file("data/l3_urban/buildings.shp")
```

## Data Format Requirements

### TLE Format

```
STARLINK-1007
1 44713U 19074A   24001.50000000  .00001234  00000-0  12345-3 0  9999
2 44713  53.0000 123.4567 0001234  12.3456 347.6543 15.12345678123456
```

### Antenna Pattern CSV

```csv
elevation,azimuth,gain_dbi
0,0,0.0
10,0,5.2
20,0,12.5
...
```

### Building Shapefile Attributes

Required attributes:
- `geometry`: Polygon geometry
- `height`: Building height in meters

Optional attributes:
- `material`: Building material (concrete, glass, metal)
- `floors`: Number of floors

## .gitignore

Large data files are excluded from git by default. Add your data files to `.gitignore`:

```
data/**/*.tif
data/**/*.hgt
data/**/*.nc
data/**/*.shp
```

## Data Citation

When using external data sources, please cite appropriately:

- **SRTM**: NASA Shuttle Radar Topography Mission
- **OpenStreetMap**: © OpenStreetMap contributors
- **CelesTrak**: Dr. T.S. Kelso, CelesTrak

## Notes

- Keep data files organized by layer
- Use consistent coordinate reference systems (WGS84 recommended)
- Document data sources and acquisition dates
- Validate data quality before simulation
