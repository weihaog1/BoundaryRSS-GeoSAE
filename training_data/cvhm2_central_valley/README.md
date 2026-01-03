# CVHM2 Central Valley Well Log Database

## Source

**Title**: Central Valley Hydrologic Model version 2 (CVHM2): Well Log Database
**Publisher**: U.S. Geological Survey
**Authors**: Marcelli, M.F., Shepherd, M.M., and Faunt, C.C.
**Year**: 2022
**DOI**: https://doi.org/10.5066/P9IZRO3V
**ScienceBase URL**: https://www.sciencebase.gov/catalog/item/61fc85bbd34e622189cc0941

## Download Date

2025-01-03

## License

USGS Public Domain - This data is in the public domain and may be freely used.

## Dataset Description

This dataset contains borehole lithology data from the California Central Valley, compiled for the CVHM2 groundwater model. It includes well logs from the California Department of Water Resources (DWR) Online System of Well Completion Reports (OSWCR) and the California Central Valley Groundwater-Surface Water Simulation Model (C2VSim).

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total boreholes | 14,683 |
| Total lithology intervals | 280,151 |
| Unique texture types | 28 |
| Geographic coverage | California Central Valley |
| Latitude range | 34.99° to 40.69° N |
| Longitude range | -122.61° to -118.75° W |
| Maximum depth | 4,027 ft |

## File Structure

```
cvhm2_central_valley/
├── README.md                    # This file
├── Well_Log_Database.xlsx       # Original USGS Excel file (12.5 MB)
├── prepare_data.py              # Conversion script for GeoSAE
└── geosae_input.csv             # Generated GeoSAE-compatible data
```

## Excel Sheet Structure

### Borehole_Information (14,683 records)
| Column | Type | Description |
|--------|------|-------------|
| BoreID | int | Unique borehole identifier |
| WCR_Number | string | Well Completion Report number |
| Legacy_Log_Number | string | Legacy log identifier |
| Well_Name | string | Well name |
| Latitude | float | WGS84 latitude (degrees) |
| Longitude | float | WGS84 longitude (degrees) |
| Total_Depth | float | Total drilled depth (ft) |
| Planned_Use | string | Intended use of well |
| Published_Source | string | Data source reference |

### Borehole_Lithology (280,151 records)
| Column | Type | Description |
|--------|------|-------------|
| BoreID | int | Foreign key to Borehole_Information |
| Top_Depth | float | Top of interval (ft below surface) |
| Bottom_Depth | float | Bottom of interval (ft below surface) |
| Texture_Qualifier | string | Qualifier (e.g., "very", "slightly") |
| Secondary_Texture_Modifier | string | Secondary modifier |
| Primary_Texture_Modifier | string | Primary modifier |
| Texture | string | Main texture classification |
| Color_Qualifier | string | Color qualifier |
| Primary_Color | string | Primary color |
| Secondary_Color | string | Secondary color |

## Texture Types (28 categories)

Main categories by frequency:
1. Clay (127,822 intervals) - 45.6%
2. Sand (101,009 intervals) - 36.1%
3. Gravel (17,064 intervals) - 6.1%
4. Top Soil (7,648 intervals)
5. Shale (7,343 intervals)
6. Silt (6,186 intervals)
7. Sandstone (3,839 intervals)
8. Rock (2,818 intervals)
9. Hard Pan (2,811 intervals)
10. Cobbles (1,267 intervals)
11. Others (Unknown, Lava, Loam, Asphalt, Siltstone, etc.)

## Comparison with Other Datasets

| Dataset | Boreholes | Intervals | Categories | Type |
|---------|-----------|-----------|------------|------|
| CVHM2 Central Valley | 14,683 | 280,151 | 28 | Lithology textures |
| Salinas Valley | 1,385 | 85,578 | 8 | Stratigraphic units |

**Note**: CVHM2 uses lithology texture classification (Clay, Sand, Gravel, etc.) while Salinas Valley uses named hydrostratigraphic units (aquifers and aquitards). Both are valid for GeoSAE training.

## Coordinate System

- **Type**: WGS84 Geographic
- **Units**: Decimal degrees (Lat/Lon)
- **Depth units**: Feet below ground surface

## Citation

```
Marcelli, M.F., Shepherd, M.M., and Faunt, C.C., 2022, Central Valley
Hydrologic Model version 2 (CVHM2): Well Log Lithology Database and
Texture Model: U.S. Geological Survey data release,
https://doi.org/10.5066/P9IZRO3V.
```
