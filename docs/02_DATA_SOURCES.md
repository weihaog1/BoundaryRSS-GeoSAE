# Data Sources for GeoSAE Training

**Last Updated**: 2025-01-03

## GeoSAE Data Requirements

GeoSAE requires **3D borehole stratigraphic data** with:
- `x`: Easting/longitude coordinate
- `y`: Northing/latitude coordinate
- `z`: Elevation/depth (vertical position)
- `surface_code`: Stratigraphic unit name (formation identifier)

### Critical: 2D Surface Geology is NOT Compatible

Surface geology maps (like what Jun provided) only show rock units at the surface. GeoSAE needs subsurface data showing how formations vary with depth.

```
SURFACE GEOLOGY (2D - NOT usable):
    Surface view only
    ┌─────────────────────┐
    │  Qa  │  Kt  │  Jm  │  ← Only shows surface units
    └─────────────────────┘

BOREHOLE DATA (3D - Required):
    ┌─────────────────────┐
    │  Qa  │  Kt  │  Jm  │  ← Surface
    ├─────────────────────┤
    │      Kt      │ Kt  │  ← Depth 1
    ├─────────────────────┤
    │      Kt      │ Jm  │  ← Depth 2
    ├─────────────────────┤
    │      Jm      │ Jm  │  ← Depth 3
    └─────────────────────┘
           ↑
        Boreholes provide vertical sections
```

## Primary Dataset: Salinas Valley (USGS)

### Source Information
- **Title**: Digital data for the Salinas Valley Geological Framework, California
- **Publisher**: U.S. Geological Survey
- **DOI**: https://doi.org/10.5066/P9IL8VBD
- **ScienceBase**: https://www.sciencebase.gov/catalog/item/63221f38d34e71c6d67ab5be
- **Local Path**: `training_data/salinas_valley/`

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Number of boreholes | 1,385 |
| Stratigraphic intervals | 3,562 |
| Lithology intervals | 85,578 |
| Unique stratigraphic units | 8 |
| Coordinate system | State Plane (feet) |
| Easting range | 5,710,033 - 6,012,482 |
| Northing range | 1,848,005 - 2,204,362 |
| Elevation range | -298 to 1,910 ft |

### Stratigraphic Units (youngest to oldest)
1. Shallow aquifer
2. Salinas aquitard
3. 180ft aquifer
4. Middle aquitard
5. 400ft aquifer
6. Deep aquitard
7. Paso Robles
8. Purisima

### Files Downloaded
```
training_data/salinas_valley/
├── WellData_Location.csv              # 84 KB - Borehole coordinates
├── WellData_WellStratigraphy.csv      # 124 KB - Formation intervals
├── WellData_WellLithology.csv         # 6.0 MB - Lithology at 10ft intervals
├── WellData_ModelUnits_AtWells.csv    # 229 KB - Model unit assignments
├── WellData_ZoneCodes.csv             # 0.5 KB - Zone code definitions
└── WellData_Texture_KrigingParameters.csv  # 1.5 KB - Kriging params
```

### Column Definitions

**WellData_Location.csv**:
| Column | Description |
|--------|-------------|
| BoreholeID | Unique well identifier |
| Easting | X coordinate (State Plane feet) |
| Northing | Y coordinate (State Plane feet) |
| Elevation | Ground surface elevation (ft MSL) |
| TotalDepth | Total drilled depth (ft) |
| AlternateID | Alternative well ID |

**WellData_WellStratigraphy.csv**:
| Column | Description |
|--------|-------------|
| BoreholeID | Links to Location table |
| IntervalTop_Depth | Top of interval (ft below surface) |
| IntervalBase_Depth | Base of interval (ft below surface) |
| HydrostratigraphicUnit | Formation name |

**WellData_WellLithology.csv**:
| Column | Description |
|--------|-------------|
| BoreholeID | Links to Location table |
| IntervalTop_Depth | Top depth (ft) |
| IntervalBase_Depth | Base depth (ft) |
| Lithologic descriptor | Rock/sediment type |
| Easting, Northing | Coordinates (duplicated) |
| IntervalTop_Elevation | Top elevation (ft MSL) |
| IntervalBase_Elevation | Base elevation (ft MSL) |

## Alternative US Datasets

If more data is needed, these are also compatible:

### CVHM2 Central Valley Well Log Database
- **DOI**: https://doi.org/10.5066/P9IZRO3V
- **Boreholes**: 14,683
- **Coverage**: California Central Valley
- **Format**: Well logs with lithology classifications

### Texas Historical Well Logs (USGS)
- **DOI**: https://doi.org/10.5066/P973SMX5
- **Wells**: 15,475 logs
- **Coverage**: 229 Texas counties
- **Format**: CSV headers + scanned PDF logs

### GeoLog Locator (USGS)
- **URL**: https://webapps.usgs.gov/GeoLogLocator/
- **Wells**: 7,000+ digital logs nationwide
- **Format**: LAS files (need conversion to CSV)

## Data Conversion for GeoSAE

The Salinas Valley stratigraphy data needs to be converted to GeoSAE format:

```python
# Required output format for GeoSAE
# CSV with columns: x, y, z, surface_code

# Example conversion (pseudocode):
# 1. Join stratigraphy with location on BoreholeID
# 2. Compute z = Elevation - IntervalTop_Depth
# 3. Map HydrostratigraphicUnit → surface_code
# 4. Export as: x, y, z, surface_code
```

See `training_data/salinas_valley/prepare_data.py` for the actual conversion script.
