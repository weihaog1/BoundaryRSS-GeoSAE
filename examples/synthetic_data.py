"""
Synthetic data generation for GeoSAE examples.

This module provides functions to generate synthetic borehole and
stratigraphic data for testing and demonstration purposes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geosae.data import BoreholeData


def generate_synthetic_surfaces(
    x_range: Tuple[float, float] = (0, 10000),
    y_range: Tuple[float, float] = (0, 10000),
    z_range: Tuple[float, float] = (-100, 0),
    num_surfaces: int = 7,
    num_boreholes: int = 100,
    noise_std: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[List[BoreholeData], Dict[str, int], np.ndarray]:
    """
    Generate synthetic borehole data with stratigraphic layers.

    Creates a set of parallel, gently undulating stratigraphic surfaces
    that simulate realistic geological layering.

    Args:
        x_range: Range for x coordinates
        y_range: Range for y coordinates
        z_range: Range for z coordinates (depth)
        num_surfaces: Number of stratigraphic surfaces
        num_boreholes: Number of boreholes to generate
        noise_std: Standard deviation of noise added to surface depths
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
            - List of BoreholeData objects
            - Dictionary mapping surface code to index
            - Array of surface points (N, 4) with columns [x, y, z, surface_idx]
    """
    if seed is not None:
        np.random.seed(seed)

    # Surface codes (from youngest to oldest)
    surface_codes = [f"S{i}" for i in range(num_surfaces)]
    code_to_idx = {code: i for i, code in enumerate(surface_codes)}

    # Generate borehole locations
    borehole_x = np.random.uniform(x_range[0], x_range[1], num_boreholes)
    borehole_y = np.random.uniform(y_range[0], y_range[1], num_boreholes)

    # Generate base surface depths (evenly spaced)
    base_depths = np.linspace(z_range[1], z_range[0], num_surfaces + 1)[:-1]

    # Create undulating surfaces using sinusoidal functions
    # This simulates gentle folding
    wavelength_x = (x_range[1] - x_range[0]) / 3
    wavelength_y = (y_range[1] - y_range[0]) / 3
    amplitude = (z_range[1] - z_range[0]) / (num_surfaces * 4)

    boreholes = []
    all_surface_points = []

    for i in range(num_boreholes):
        x, y = borehole_x[i], borehole_y[i]

        # Calculate surface depths at this location
        undulation = amplitude * (
            np.sin(2 * np.pi * x / wavelength_x)
            + np.sin(2 * np.pi * y / wavelength_y)
        )

        layers = []
        prev_depth = z_range[1]  # Start at surface

        for j, (code, base_depth) in enumerate(zip(surface_codes, base_depths)):
            # Add undulation and noise
            depth = base_depth + undulation + np.random.normal(0, noise_std)

            # Ensure depth is below previous surface
            depth = min(depth, prev_depth - 1)

            layers.append({
                "surface_code": code,
                "top_z": depth,
                "bottom_z": depth - 10,  # Arbitrary thickness
            })

            # Store surface point
            all_surface_points.append([x, y, depth, j])
            prev_depth = depth

        borehole = BoreholeData(
            borehole_id=f"BH_{i:04d}",
            x=x,
            y=y,
            layers=layers,
        )
        boreholes.append(borehole)

    surface_points = np.array(all_surface_points)

    return boreholes, code_to_idx, surface_points


def generate_unconformity_data(
    x_range: Tuple[float, float] = (0, 10000),
    y_range: Tuple[float, float] = (0, 10000),
    z_range: Tuple[float, float] = (-100, 0),
    num_boreholes: int = 100,
    noise_std: float = 2.0,
    seed: Optional[int] = 42,
) -> Tuple[List[BoreholeData], Dict[str, int], np.ndarray]:
    """
    Generate synthetic data with an unconformity.

    Creates stratigraphic data that includes an erosional unconformity,
    similar to the Jiangdong new district example in the paper.

    Args:
        x_range: Range for x coordinates
        y_range: Range for y coordinates
        z_range: Range for z coordinates
        num_boreholes: Number of boreholes
        noise_std: Noise standard deviation
        seed: Random seed

    Returns:
        Same as generate_synthetic_surfaces
    """
    if seed is not None:
        np.random.seed(seed)

    # Surface codes mimicking the paper's example
    surface_codes = ["Qh3y", "Qh2q", "Qp2d", "Qp2b", "Qp1x", "N2h", "N1d"]
    code_to_idx = {code: i for i, code in enumerate(surface_codes)}

    num_surfaces = len(surface_codes)

    # Base depths for each surface
    z_total = z_range[1] - z_range[0]
    base_depths = [
        z_range[1] - z_total * 0.05,   # Qh3y
        z_range[1] - z_total * 0.15,   # Qh2q
        z_range[1] - z_total * 0.30,   # Qp2d (unconformity - volcanic)
        z_range[1] - z_total * 0.45,   # Qp2b
        z_range[1] - z_total * 0.60,   # Qp1x
        z_range[1] - z_total * 0.80,   # N2h
        z_range[1] - z_total * 0.95,   # N1d
    ]

    # Generate borehole locations
    borehole_x = np.random.uniform(x_range[0], x_range[1], num_boreholes)
    borehole_y = np.random.uniform(y_range[0], y_range[1], num_boreholes)

    # Wavelengths for undulation
    wavelength_x = (x_range[1] - x_range[0]) / 2
    wavelength_y = (y_range[1] - y_range[0]) / 2
    amplitude = z_total / 15

    boreholes = []
    all_surface_points = []

    for i in range(num_boreholes):
        x, y = borehole_x[i], borehole_y[i]

        # Base undulation
        undulation = amplitude * np.sin(2 * np.pi * x / wavelength_x) * np.cos(2 * np.pi * y / wavelength_y)

        # Simulate unconformity erosion pattern for Qp2d (volcanic)
        # In some areas, this surface cuts through underlying strata
        volcanic_extent = (
            np.sin(2 * np.pi * x / (wavelength_x * 0.5))
            * np.cos(2 * np.pi * y / (wavelength_y * 0.5))
        )

        layers = []
        prev_depth = z_range[1]

        for j, (code, base_depth) in enumerate(zip(surface_codes, base_depths)):
            # Add undulation and noise
            depth = base_depth + undulation + np.random.normal(0, noise_std)

            # Special handling for volcanic unconformity
            if code == "Qp2d":
                # Volcanic surface has more irregular morphology
                depth += volcanic_extent * amplitude * 2

            # Ensure depth is below previous surface
            depth = min(depth, prev_depth - 0.5)

            layers.append({
                "surface_code": code,
                "top_z": depth,
                "bottom_z": depth - 5,
            })

            all_surface_points.append([x, y, depth, j])
            prev_depth = depth

        borehole = BoreholeData(
            borehole_id=f"BH_{i:04d}",
            x=x,
            y=y,
            layers=layers,
        )
        boreholes.append(borehole)

    surface_points = np.array(all_surface_points)

    return boreholes, code_to_idx, surface_points


def save_synthetic_data_csv(
    surface_points: np.ndarray,
    surface_codes: List[str],
    output_path: str,
) -> None:
    """
    Save synthetic surface points to CSV.

    Args:
        surface_points: Array of shape (N, 4) with columns [x, y, z, surface_idx]
        surface_codes: List of surface code names
        output_path: Path to save CSV file
    """
    import pandas as pd

    df = pd.DataFrame({
        "x": surface_points[:, 0],
        "y": surface_points[:, 1],
        "z": surface_points[:, 2],
        "surface_code": [surface_codes[int(idx)] for idx in surface_points[:, 3]],
    })

    df.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")


if __name__ == "__main__":
    # Generate and save synthetic data for testing
    boreholes, code_to_idx, surface_points = generate_unconformity_data(
        num_boreholes=200, seed=42
    )

    surface_codes = list(code_to_idx.keys())

    print(f"Generated {len(boreholes)} boreholes")
    print(f"Total surface points: {len(surface_points)}")
    print(f"Surface codes: {surface_codes}")

    # Save to CSV
    save_synthetic_data_csv(surface_points, surface_codes, "synthetic_data.csv")
