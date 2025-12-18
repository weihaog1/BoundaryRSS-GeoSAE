"""
PyVista visualization utilities for GeoSAE.

This module provides functions for:
- Creating 3D grids from potential field predictions
- Extracting isosurfaces (stratigraphic interfaces)
- Visualizing 3D geological models
- Exporting to VTK format

Reference:
    Sullivan, C.; Kaszynski, A. PyVista: 3D Plotting and Mesh Analysis
    through a Streamlined Interface for the Visualization Toolkit (VTK).
    J. Open Source Softw. 2019, 4, 1450.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("PyVista not available. Install with: pip install pyvista")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_structured_grid(
    model: "GeoSAE",
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: Tuple[int, int, int] = (50, 50, 50),
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create a structured grid of potential field values.

    Args:
        model: Trained GeoSAE model
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        resolution: (nx, ny, nz) grid resolution
        device: Device for model inference

    Returns:
        Tuple of:
            - grid_coords: Coordinates of shape (nx, ny, nz, 3)
            - grid_values: Potential field values of shape (nx, ny, nz, num_fields)
            - grid_info: Dictionary with grid metadata
    """
    model.eval()
    model.to(device)

    nx, ny, nz = resolution

    # Create coordinate arrays
    x = np.linspace(bounds[0][0], bounds[0][1], nx)
    y = np.linspace(bounds[1][0], bounds[1][1], ny)
    z = np.linspace(bounds[2][0], bounds[2][1], nz)

    # Create meshgrid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([xx, yy, zz], axis=-1)

    # Flatten for model input
    flat_coords = grid_coords.reshape(-1, 3)
    flat_coords_tensor = torch.from_numpy(flat_coords).float().to(device)

    # Predict in batches
    batch_size = 10000
    all_values = []

    with torch.no_grad():
        for i in range(0, len(flat_coords_tensor), batch_size):
            batch = flat_coords_tensor[i:i + batch_size]
            values = model(batch, normalize=True)
            all_values.append(values.cpu().numpy())

    flat_values = np.concatenate(all_values, axis=0)

    # Reshape to grid
    num_fields = flat_values.shape[1]
    grid_values = flat_values.reshape(nx, ny, nz, num_fields)

    grid_info = {
        "bounds": bounds,
        "resolution": resolution,
        "spacing": (
            (bounds[0][1] - bounds[0][0]) / (nx - 1),
            (bounds[1][1] - bounds[1][0]) / (ny - 1),
            (bounds[2][1] - bounds[2][0]) / (nz - 1),
        ),
        "num_fields": num_fields,
    }

    return grid_coords, grid_values, grid_info


def extract_isosurfaces(
    model: "GeoSAE",
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    isovalues: Optional[Union[List[float], np.ndarray]] = None,
    resolution: Tuple[int, int, int] = (50, 50, 50),
    device: str = "cpu",
    field_idx: int = 0,
) -> List:
    """
    Extract isosurfaces from the potential field.

    Uses PyVista's contour functionality to extract surfaces at specified
    isovalues (corresponding to stratigraphic interfaces).

    Args:
        model: Trained GeoSAE model
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        isovalues: List of isovalues for each surface (if None, uses model's stored values)
        resolution: Grid resolution for sampling
        device: Device for model inference
        field_idx: Index of the potential field to extract from

    Returns:
        List of PyVista PolyData objects (one per isovalue)
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for isosurface extraction")

    # Create structured grid
    grid_coords, grid_values, grid_info = create_structured_grid(
        model, bounds, resolution, device
    )

    # Get isovalues
    if isovalues is None:
        if hasattr(model, 'surface_isovalues'):
            isovalues = model.surface_isovalues.cpu().numpy()
        else:
            raise ValueError("isovalues must be provided or model must have surface_isovalues")

    # Create PyVista grid
    nx, ny, nz = resolution
    grid = pv.StructuredGrid()

    # Set points
    grid.points = grid_coords.reshape(-1, 3)
    grid.dimensions = [nx, ny, nz]

    # Add scalar field data
    scalar_field = grid_values[:, :, :, field_idx].flatten(order='F')
    grid["potential_field"] = scalar_field

    # Extract isosurfaces
    surfaces = []
    for isovalue in isovalues:
        try:
            contour = grid.contour(isosurfaces=[isovalue], scalars="potential_field")
            if contour.n_points > 0:
                surfaces.append(contour)
            else:
                surfaces.append(None)
        except Exception as e:
            print(f"Warning: Could not extract isosurface at {isovalue}: {e}")
            surfaces.append(None)

    return surfaces


def extract_all_surfaces(
    model: "GeoSAE",
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: Tuple[int, int, int] = (50, 50, 50),
    device: str = "cpu",
) -> Dict[int, List]:
    """
    Extract isosurfaces from all potential fields.

    Args:
        model: Trained GeoSAE model
        bounds: Spatial bounds
        resolution: Grid resolution
        device: Device for inference

    Returns:
        Dictionary mapping field_idx to list of surfaces
    """
    all_surfaces = {}

    for field_idx in range(model.num_potential_fields):
        surfaces = extract_isosurfaces(
            model=model,
            bounds=bounds,
            resolution=resolution,
            device=device,
            field_idx=field_idx,
        )
        all_surfaces[field_idx] = surfaces

    return all_surfaces


def visualize_model(
    model: "GeoSAE",
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: Tuple[int, int, int] = (50, 50, 50),
    device: str = "cpu",
    surface_colors: Optional[List[str]] = None,
    opacity: float = 0.8,
    show_edges: bool = False,
    window_size: Tuple[int, int] = (1024, 768),
    background: str = "white",
    show: bool = True,
    screenshot_path: Optional[str] = None,
) -> Optional["pv.Plotter"]:
    """
    Visualize the 3D stratigraphic model.

    Args:
        model: Trained GeoSAE model
        bounds: Spatial bounds
        resolution: Grid resolution
        device: Device for inference
        surface_colors: List of colors for each surface
        opacity: Surface opacity
        show_edges: Whether to show mesh edges
        window_size: Window size for visualization
        background: Background color
        show: Whether to show the plot
        screenshot_path: Path to save screenshot (optional)

    Returns:
        PyVista Plotter object (if show=False)
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for visualization")

    # Extract surfaces
    surfaces = extract_isosurfaces(
        model=model,
        bounds=bounds,
        resolution=resolution,
        device=device,
    )

    # Default colors
    if surface_colors is None:
        surface_colors = [
            "#FFFF00",  # Yellow
            "#FFA500",  # Orange
            "#FF0000",  # Red
            "#800080",  # Purple
            "#0000FF",  # Blue
            "#008000",  # Green
            "#A52A2A",  # Brown
        ]

    # Create plotter
    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)

    # Add surfaces
    for i, surface in enumerate(surfaces):
        if surface is not None and surface.n_points > 0:
            color = surface_colors[i % len(surface_colors)]
            plotter.add_mesh(
                surface,
                color=color,
                opacity=opacity,
                show_edges=show_edges,
                label=f"Surface {i}",
            )

    # Add legend
    plotter.add_legend()

    # Add axes
    plotter.add_axes()

    # Save screenshot if requested
    if screenshot_path is not None:
        plotter.screenshot(screenshot_path)

    if show:
        plotter.show()
        return None
    else:
        return plotter


def visualize_potential_field(
    model: "GeoSAE",
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: Tuple[int, int, int] = (50, 50, 50),
    device: str = "cpu",
    field_idx: int = 0,
    cmap: str = "viridis",
    opacity: str = "linear",
    show: bool = True,
) -> Optional["pv.Plotter"]:
    """
    Visualize the 3D potential field as a volume rendering.

    Args:
        model: Trained GeoSAE model
        bounds: Spatial bounds
        resolution: Grid resolution
        device: Device for inference
        field_idx: Index of potential field to visualize
        cmap: Colormap name
        opacity: Opacity transfer function
        show: Whether to show the plot

    Returns:
        PyVista Plotter object (if show=False)
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for visualization")

    # Create structured grid
    grid_coords, grid_values, grid_info = create_structured_grid(
        model, bounds, resolution, device
    )

    nx, ny, nz = resolution

    # Create PyVista ImageData (UniformGrid)
    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=grid_info["spacing"],
        origin=(bounds[0][0], bounds[1][0], bounds[2][0]),
    )

    # Add scalar field
    scalar_field = grid_values[:, :, :, field_idx].flatten(order='F')
    grid["potential_field"] = scalar_field

    # Create plotter
    plotter = pv.Plotter()
    plotter.add_volume(grid, cmap=cmap, opacity=opacity, scalars="potential_field")
    plotter.add_axes()

    if show:
        plotter.show()
        return None
    else:
        return plotter


def export_vtk(
    model: "GeoSAE",
    output_path: str,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    resolution: Tuple[int, int, int] = (50, 50, 50),
    device: str = "cpu",
    export_surfaces: bool = True,
    export_grid: bool = True,
) -> None:
    """
    Export model results to VTK format.

    Args:
        model: Trained GeoSAE model
        output_path: Base path for output files (without extension)
        bounds: Spatial bounds
        resolution: Grid resolution
        device: Device for inference
        export_surfaces: Whether to export isosurfaces
        export_grid: Whether to export the full grid
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for VTK export")

    # Export grid
    if export_grid:
        grid_coords, grid_values, grid_info = create_structured_grid(
            model, bounds, resolution, device
        )

        nx, ny, nz = resolution
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=grid_info["spacing"],
            origin=(bounds[0][0], bounds[1][0], bounds[2][0]),
        )

        # Add all potential fields
        for i in range(grid_values.shape[-1]):
            scalar_field = grid_values[:, :, :, i].flatten(order='F')
            grid[f"potential_field_{i}"] = scalar_field

        grid.save(f"{output_path}_grid.vtk")
        print(f"Saved grid to {output_path}_grid.vtk")

    # Export surfaces
    if export_surfaces:
        surfaces = extract_isosurfaces(
            model=model,
            bounds=bounds,
            resolution=resolution,
            device=device,
        )

        for i, surface in enumerate(surfaces):
            if surface is not None and surface.n_points > 0:
                surface.save(f"{output_path}_surface_{i}.vtk")
                print(f"Saved surface {i} to {output_path}_surface_{i}.vtk")


def plot_training_history(
    history: Dict[str, List],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training history.

    Args:
        history: Dictionary with training metrics
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required for plotting")

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Total loss
    if "total_loss" in history:
        axes[0, 0].plot(history["total_loss"])
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_yscale("log")

    # Variance loss
    if "variance_loss" in history:
        axes[0, 1].plot(history["variance_loss"])
        axes[0, 1].set_title("Variance Loss")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Loss")

    # Sequence loss
    if "sequence_loss" in history:
        axes[1, 0].plot(history["sequence_loss"])
        axes[1, 0].set_title("Sequence Loss")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Loss")

    # Smoothness loss
    if "smoothness_loss" in history:
        axes[1, 1].plot(history["smoothness_loss"])
        axes[1, 1].set_title("Smoothness Loss")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Loss")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history plot to {save_path}")

    plt.show()
