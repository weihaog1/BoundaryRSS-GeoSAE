#!/usr/bin/env python3
"""
Prepare Salinas Valley data for GeoSAE training.

This script converts the USGS Salinas Valley borehole data into the format
expected by GeoSAE: a CSV with columns (x, y, z, surface_code).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_data(data_dir: Path) -> pd.DataFrame:
    """
    Load Salinas Valley data and convert to GeoSAE format.

    Args:
        data_dir: Path to salinas_valley data directory

    Returns:
        DataFrame with columns: x, y, z, surface_code
    """
    # Load location and stratigraphy data
    location = pd.read_csv(data_dir / 'WellData_Location.csv')
    stratigraphy = pd.read_csv(data_dir / 'WellData_WellStratigraphy.csv')

    print(f"Loaded {len(location)} well locations")
    print(f"Loaded {len(stratigraphy)} stratigraphic intervals")

    # Filter out rows with unknown depths
    stratigraphy = stratigraphy[stratigraphy['IntervalTop_Depth'] != ' ---']
    stratigraphy['IntervalTop_Depth'] = pd.to_numeric(stratigraphy['IntervalTop_Depth'])

    # Merge stratigraphy with location
    merged = stratigraphy.merge(
        location[['BoreholeID', 'Easting', 'Northing', 'Elevation']],
        on='BoreholeID'
    )

    # Compute z elevation (surface elevation minus depth)
    # z = Elevation - IntervalTop_Depth
    merged['z'] = merged['Elevation'] - merged['IntervalTop_Depth']

    # Create output DataFrame in GeoSAE format
    geosae_data = merged[['Easting', 'Northing', 'z', 'HydrostratigraphicUnit']].copy()
    geosae_data.columns = ['x', 'y', 'z', 'surface_code']

    # Remove any rows with NaN values
    geosae_data = geosae_data.dropna()

    print(f"\nPrepared {len(geosae_data)} data points for GeoSAE")
    print(f"Unique stratigraphic units: {geosae_data['surface_code'].nunique()}")
    print(f"\nSurface code distribution:")
    print(geosae_data['surface_code'].value_counts())

    return geosae_data


def get_stratigraphic_order() -> list:
    """
    Return stratigraphic units in order from youngest (top) to oldest (bottom).

    This order is critical for GeoSAE's sequence constraints.
    """
    return [
        'Shallow aquifer',    # Youngest/shallowest
        'Salinas aquitard',
        '180ft aquifer',
        'Middle aquitard',
        '400ft aquifer',
        'Deep aquitard',
        'Paso Robles',
        'Purisima',           # Oldest/deepest
    ]


def print_data_statistics(df: pd.DataFrame):
    """Print statistics about the prepared data."""
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)

    print(f"\nTotal points: {len(df)}")
    print(f"\nCoordinate ranges:")
    print(f"  x (Easting):  {df['x'].min():.1f} to {df['x'].max():.1f}")
    print(f"  y (Northing): {df['y'].min():.1f} to {df['y'].max():.1f}")
    print(f"  z (Elevation): {df['z'].min():.1f} to {df['z'].max():.1f}")

    print(f"\nStratigraphic units ({df['surface_code'].nunique()} total):")
    order = get_stratigraphic_order()
    for i, unit in enumerate(order):
        count = len(df[df['surface_code'] == unit])
        print(f"  {i}: {unit} ({count} points)")


def main():
    # Get the data directory
    script_dir = Path(__file__).parent

    # Prepare the data
    geosae_data = load_and_prepare_data(script_dir)

    # Print statistics
    print_data_statistics(geosae_data)

    # Save to CSV
    output_path = script_dir / 'geosae_input.csv'
    geosae_data.to_csv(output_path, index=False)
    print(f"\nSaved GeoSAE input data to: {output_path}")

    # Also save the stratigraphic order for reference
    order_path = script_dir / 'stratigraphic_order.txt'
    with open(order_path, 'w') as f:
        f.write("# Stratigraphic units from youngest (top) to oldest (bottom)\n")
        for i, unit in enumerate(get_stratigraphic_order()):
            f.write(f"{i}: {unit}\n")
    print(f"Saved stratigraphic order to: {order_path}")


if __name__ == '__main__':
    main()
