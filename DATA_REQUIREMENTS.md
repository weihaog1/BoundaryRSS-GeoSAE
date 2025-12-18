# GeoSAE Training Data Requirements

## Overview
Data needed to train the GeoSAE 3D stratigraphic modeling system.

---

## Required Data

### 1. Borehole / Well Log Data
Points where stratigraphic layer boundaries were intersected.

| Field | Description | Example |
|-------|-------------|---------|
| Borehole ID | Unique well identifier | BH-001 |
| X | Easting coordinate | 110.352 |
| Y | Northing coordinate | 19.956 |
| Z (Top) | Depth/elevation of layer top | -25.5 m |
| Layer Code | Stratigraphic unit identifier | Qh3y, N1d |

**Minimum recommended:** 50+ boreholes across the study area

### 2. Stratigraphic Column / Sequence
- List of stratigraphic units from youngest to oldest
- Contact type between units (conformable / unconformable)

**Example:**
```
Qh3y (youngest) - unconformable
Qh2q - conformable
Qp2d - unconformable (volcanic)
Qp2b - unconformable
Qp1x - unconformable
N2h - unconformable
N1d (oldest)
```

### 3. Study Area Bounds
- X range (min, max)
- Y range (min, max)
- Z range (depth extent)
- Coordinate reference system (CRS/EPSG code)

---

## Optional Data (Improves Accuracy)

### 4. Structural Measurements
Dip and strike measurements at outcrop or well locations.

| Field | Description |
|-------|-------------|
| X, Y, Z | Measurement location |
| Dip | Angle of inclination (degrees) |
| Dip Direction | Azimuth of dip (degrees) |
| Layer Code | Associated stratigraphic unit |

### 5. Geological Cross-Sections
Digitized interpreted sections with georeferenced layer boundaries.

### 6. Surface Geology Map
Outcrop boundaries of stratigraphic units (as polygons or points).

---

## Preferred Formats
- CSV, Excel, or GeoJSON
- Shapefiles for spatial data
- LAS files for well logs (if available)

---

## Questions to Clarify
1. What is the target study area?
2. How many boreholes/wells are available?
3. What depth range needs to be modeled?
4. Are there known faults or structural complexities?
