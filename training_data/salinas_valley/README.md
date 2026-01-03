# Salinas Valley Geological Framework Data

## Source

**Title**: Digital data for the Salinas Valley Geological Framework, California
**Publisher**: U.S. Geological Survey
**Authors**: Marcelli, M.F., et al.
**Year**: 2023
**DOI**: https://doi.org/10.5066/P9IL8VBD
**ScienceBase URL**: https://www.sciencebase.gov/catalog/item/63221f38d34e71c6d67ab5be

## Download Date

2025-01-03

## License

USGS Public Domain - This data is in the public domain and may be freely used.

## Dataset Description

This dataset contains borehole stratigraphic and lithologic data from the Salinas Valley, California. It was created as part of a USGS study in cooperation with the Monterey County Water Resource Agency for numerical simulation of the hydrologic system.

## File Structure

```
salinas_valley/
├── README.md                              # This file
├── WellDataTables_CSV.zip                 # Original downloaded archive
├── WellData_Location.csv                  # Borehole locations (1,385 wells)
├── WellData_WellStratigraphy.csv          # Stratigraphic unit intervals (3,562 records)
├── WellData_WellLithology.csv             # Lithology at 10ft intervals (85,578 records)
├── WellData_ModelUnits_AtWells.csv        # Model unit assignments
├── WellData_ZoneCodes.csv                 # Zone code definitions
└── WellData_Texture_KrigingParameters.csv # Kriging parameters
```

## Column Definitions

### WellData_Location.csv (1,385 records)
| Column | Type | Description |
|--------|------|-------------|
| BoreholeID | int | Unique well identifier (primary key) |
| Easting | float | X coordinate in State Plane feet |
| Northing | float | Y coordinate in State Plane feet |
| Elevation | float | Ground surface elevation (ft above MSL) |
| TotalDepth | float | Total drilled depth (ft below surface) |
| AlternateID | string | Alternative well identifier |

### WellData_WellStratigraphy.csv (3,562 records)
| Column | Type | Description |
|--------|------|-------------|
| BoreholeID | int | Foreign key to Location table |
| IntervalTop_Depth | float | Top of interval (ft below surface) |
| IntervalBase_Depth | float | Base of interval (ft below surface), "---" = unknown |
| HydrostratigraphicUnit | string | Formation/aquifer name |

### WellData_WellLithology.csv (85,578 records)
| Column | Type | Description |
|--------|------|-------------|
| BoreholeID | int | Foreign key to Location table |
| IntervalTop_Depth | float | Top of interval (ft below surface) |
| IntervalBase_Depth | float | Base of interval (ft below surface) |
| Lithologic descriptor | string | Sediment/rock type (36 unique values) |
| Easting | float | X coordinate (duplicated from Location) |
| Northing | float | Y coordinate (duplicated from Location) |
| IntervalTop_Elevation | float | Top elevation (ft MSL) |
| IntervalBase_Elevation | float | Base elevation (ft MSL) |

## Stratigraphic Units (8 total, youngest to oldest)

1. **Shallow aquifer** - Shallowest unconfined aquifer
2. **Salinas aquitard** - Confining layer
3. **180ft aquifer** - Named for typical depth
4. **Middle aquitard** - Confining layer
5. **400ft aquifer** - Named for typical depth
6. **Deep aquitard** - Deepest confining layer
7. **Paso Robles** - Plio-Pleistocene formation
8. **Purisima** - Pliocene formation (oldest)

## Coordinate System

- **Type**: California State Plane Zone 4 (feet)
- **Datum**: NAD83
- **Units**: US Survey Feet
- **Easting range**: 5,710,033 to 6,012,482
- **Northing range**: 1,848,005 to 2,204,362
- **Elevation range**: -298 to 1,910 ft MSL

## Lithology Types (36 unique descriptors)

Common types include:
- boulders, cobbles, gravel (coarse)
- coarse sand, sand, fine sand (medium)
- silt, clay, hardite, hardite/clay (fine)
- sandstone, shale (consolidated)

## Usage with GeoSAE

To convert this data to GeoSAE format, join stratigraphy with location:

```python
import pandas as pd

loc = pd.read_csv('WellData_Location.csv')
strat = pd.read_csv('WellData_WellStratigraphy.csv')

# Join and compute z elevation
merged = strat.merge(loc[['BoreholeID', 'Easting', 'Northing', 'Elevation']], on='BoreholeID')
merged['z'] = merged['Elevation'] - merged['IntervalTop_Depth']

# Export in GeoSAE format
geosae_data = merged[['Easting', 'Northing', 'z', 'HydrostratigraphicUnit']]
geosae_data.columns = ['x', 'y', 'z', 'surface_code']
geosae_data.to_csv('geosae_input.csv', index=False)
```

## Citation

```
Marcelli, M.F., Shepherd, M.M., and Faunt, C.C., 2023, Digital data for the
Salinas Valley Geological Framework, California: U.S. Geological Survey
data release, https://doi.org/10.5066/P9IL8VBD.
```
