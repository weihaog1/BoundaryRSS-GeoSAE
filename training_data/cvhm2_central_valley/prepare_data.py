#!/usr/bin/env python3
"""
Prepare CVHM2 Central Valley data for GeoSAE training.

This script converts the USGS CVHM2 Well Log Database into the format
expected by GeoSAE: a CSV with columns (x, y, z, surface_code).

Note: This dataset uses lithology textures (Clay, Sand, Gravel, etc.)
as the surface codes, rather than named stratigraphic units.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_data(data_dir: Path, top_n_textures: int = 10) -> pd.DataFrame:
    """
    Load CVHM2 data and convert to GeoSAE format.

    Args:
        data_dir: Path to cvhm2_central_valley data directory
        top_n_textures: Only include the N most common texture types
                       (reduces class imbalance)

    Returns:
        DataFrame with columns: x, y, z, surface_code
    """
    xlsx_path = data_dir / 'Well_Log_Database.xlsx'

    print(f"Loading {xlsx_path}...")
    info = pd.read_excel(xlsx_path, sheet_name='Borehole_Information')
    lith = pd.read_excel(xlsx_path, sheet_name='Borehole_Lithology')

    print(f"Loaded {len(info)} boreholes")
    print(f"Loaded {len(lith)} lithology intervals")

    # Get top N most common textures
    top_textures = lith['Texture'].value_counts().head(top_n_textures).index.tolist()
    print(f"\nUsing top {top_n_textures} texture types: {top_textures}")

    # Filter to only include top textures
    lith_filtered = lith[lith['Texture'].isin(top_textures)].copy()
    print(f"Filtered to {len(lith_filtered)} intervals with top textures")

    # Merge with location data
    merged = lith_filtered.merge(
        info[['BoreID', 'Latitude', 'Longitude']],
        on='BoreID'
    )

    # Convert lat/lon to approximate meters (for better scaling)
    # Using simple equirectangular projection centered on Central Valley
    # 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(lat) meters
    center_lat = merged['Latitude'].mean()
    merged['x'] = merged['Longitude'] * 111000 * np.cos(np.radians(center_lat))
    merged['y'] = merged['Latitude'] * 111000

    # Compute z as negative depth (below surface = negative)
    # Using midpoint of interval
    merged['z'] = -(merged['Top_Depth'] + merged['Bottom_Depth']) / 2

    # Create output DataFrame
    geosae_data = merged[['x', 'y', 'z', 'Texture']].copy()
    geosae_data.columns = ['x', 'y', 'z', 'surface_code']

    # Remove any NaN values
    geosae_data = geosae_data.dropna()

    print(f"\nPrepared {len(geosae_data)} data points for GeoSAE")

    return geosae_data


def get_texture_order() -> list:
    """
    Return texture types ordered by typical depth occurrence.

    This is approximate - in reality, textures can occur at any depth.
    Using frequency-based ordering as a proxy.
    """
    return [
        'Top Soil',     # Surface
        'Sand',         # Common at various depths
        'Gravel',       # Often shallow aquifer material
        'Clay',         # Common aquitard material
        'Silt',         # Fine-grained
        'Hard Pan',     # Cemented layer
        'Shale',        # Consolidated fine-grained
        'Sandstone',    # Consolidated sand
        'Cobbles',      # Coarse material
        'Rock',         # Bedrock/consolidated
    ]


def print_data_statistics(df: pd.DataFrame):
    """Print statistics about the prepared data."""
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)

    print(f"\nTotal points: {len(df):,}")
    print(f"\nCoordinate ranges (meters):")
    print(f"  x: {df['x'].min():.0f} to {df['x'].max():.0f}")
    print(f"  y: {df['y'].min():.0f} to {df['y'].max():.0f}")
    print(f"  z: {df['z'].min():.0f} to {df['z'].max():.0f}")

    print(f"\nTexture distribution:")
    for texture in df['surface_code'].unique():
        count = len(df[df['surface_code'] == texture])
        pct = 100 * count / len(df)
        print(f"  {texture}: {count:,} ({pct:.1f}%)")


def main():
    script_dir = Path(__file__).parent

    # Prepare data with top 10 textures
    geosae_data = load_and_prepare_data(script_dir, top_n_textures=10)

    # Print statistics
    print_data_statistics(geosae_data)

    # Save to CSV
    output_path = script_dir / 'geosae_input.csv'
    geosae_data.to_csv(output_path, index=False)
    print(f"\nSaved GeoSAE input data to: {output_path}")

    # Save texture order
    order_path = script_dir / 'texture_order.txt'
    with open(order_path, 'w') as f:
        f.write("# Texture types (approximate depth ordering)\n")
        for i, texture in enumerate(get_texture_order()):
            f.write(f"{i}: {texture}\n")
    print(f"Saved texture order to: {order_path}")


if __name__ == '__main__':
    main()
