"""
Data preprocessing utilities for GeoSAE.

This module provides functions for:
- Coordinate normalization
- Stratigraphic sequence relationship construction
- Data augmentation
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union


def normalize_coordinates(
    coords: Union[np.ndarray, torch.Tensor],
    coord_min: Optional[Union[np.ndarray, torch.Tensor]] = None,
    coord_max: Optional[Union[np.ndarray, torch.Tensor]] = None,
    target_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[Union[np.ndarray, torch.Tensor], Dict]:
    """
    Normalize coordinates to a target range.

    Args:
        coords: Coordinates of shape (N, 3)
        coord_min: Minimum coordinates (if None, computed from data)
        coord_max: Maximum coordinates (if None, computed from data)
        target_range: Target range for normalization (default: [-1, 1])

    Returns:
        Tuple of (normalized_coords, normalization_params)
    """
    is_tensor = isinstance(coords, torch.Tensor)

    if is_tensor:
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords

    if coord_min is None:
        coord_min = coords_np.min(axis=0)
    elif isinstance(coord_min, torch.Tensor):
        coord_min = coord_min.detach().cpu().numpy()

    if coord_max is None:
        coord_max = coords_np.max(axis=0)
    elif isinstance(coord_max, torch.Tensor):
        coord_max = coord_max.detach().cpu().numpy()

    # Avoid division by zero
    coord_range = coord_max - coord_min
    coord_range = np.where(coord_range == 0, 1.0, coord_range)

    # Normalize to [0, 1]
    normalized = (coords_np - coord_min) / coord_range

    # Scale to target range
    target_min, target_max = target_range
    normalized = normalized * (target_max - target_min) + target_min

    params = {
        "coord_min": coord_min,
        "coord_max": coord_max,
        "target_range": target_range,
    }

    if is_tensor:
        normalized = torch.from_numpy(normalized).to(coords.device).to(coords.dtype)
        params["coord_min"] = torch.from_numpy(coord_min).to(coords.device)
        params["coord_max"] = torch.from_numpy(coord_max).to(coords.device)

    return normalized, params


def denormalize_coordinates(
    coords: Union[np.ndarray, torch.Tensor],
    coord_min: Union[np.ndarray, torch.Tensor],
    coord_max: Union[np.ndarray, torch.Tensor],
    source_range: Tuple[float, float] = (-1.0, 1.0),
) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize coordinates back to original range.

    Args:
        coords: Normalized coordinates of shape (N, 3)
        coord_min: Original minimum coordinates
        coord_max: Original maximum coordinates
        source_range: Source range of normalized coordinates

    Returns:
        Denormalized coordinates
    """
    is_tensor = isinstance(coords, torch.Tensor)

    if is_tensor:
        coords_np = coords.detach().cpu().numpy()
        if isinstance(coord_min, torch.Tensor):
            coord_min = coord_min.detach().cpu().numpy()
        if isinstance(coord_max, torch.Tensor):
            coord_max = coord_max.detach().cpu().numpy()
    else:
        coords_np = coords

    source_min, source_max = source_range

    # Scale from source range to [0, 1]
    normalized = (coords_np - source_min) / (source_max - source_min)

    # Scale to original range
    coord_range = coord_max - coord_min
    denormalized = normalized * coord_range + coord_min

    if is_tensor:
        denormalized = torch.from_numpy(denormalized).to(coords.device).to(coords.dtype)

    return denormalized


def create_sequence_relations(
    stratigraphic_order: List[str],
    contact_types: Dict[str, str],
) -> Dict[int, Dict[str, List[int]]]:
    """
    Create stratigraphic sequence relations from geological information.

    Args:
        stratigraphic_order: List of stratigraphic codes from youngest to oldest
        contact_types: Dictionary mapping surface code to contact type
                      ("conformable", "unconformable", "overlap")

    Returns:
        Dictionary mapping surface index to relationship dictionary:
        {
            surface_idx: {
                "above": [list of surface indices this should be above],
                "below": [list of surface indices this should be below],
                "overlap": [list of surface indices this overlaps with]
            }
        }

    Example:
        stratigraphic_order = ["Qh3y", "Qh2q", "Qp2d", "Qp2b", "Qp1x", "N2h", "N1d"]
        contact_types = {
            "Qh3y": "unconformable",
            "Qh2q": "conformable",
            ...
        }
    """
    num_surfaces = len(stratigraphic_order)
    relations = {}

    for i, surface_code in enumerate(stratigraphic_order):
        relations[i] = {
            "above": [],  # This surface should be above these surfaces
            "below": [],  # This surface should be below these surfaces
            "overlap": [],  # This surface overlaps with these surfaces
        }

        # All surfaces below this one in the sequence
        # (surfaces with larger index are older/deeper)
        for j in range(i + 1, num_surfaces):
            relations[i]["above"].append(j)

        # All surfaces above this one in the sequence
        # (surfaces with smaller index are younger/shallower)
        for j in range(0, i):
            relations[i]["below"].append(j)

        # Check for overlaps based on contact types
        contact = contact_types.get(surface_code, "unconformable")
        if contact == "overlap" and i > 0:
            # Add overlap relation with the immediately overlying surface
            relations[i]["overlap"].append(i - 1)
            relations[i - 1]["overlap"].append(i)

    return relations


def create_default_sequence_relations(num_surfaces: int) -> Dict[int, Dict[str, List[int]]]:
    """
    Create default sequence relations assuming all surfaces are unconformable.

    Args:
        num_surfaces: Number of stratigraphic surfaces

    Returns:
        Sequence relations dictionary
    """
    relations = {}

    for i in range(num_surfaces):
        relations[i] = {
            "above": list(range(i + 1, num_surfaces)),  # Above all older surfaces
            "below": list(range(0, i)),  # Below all younger surfaces
            "overlap": [],  # No overlaps by default
        }

    return relations


def generate_planar_pretrain_data(
    plane_values: List[float],
    num_points_per_plane: int = 1000,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    z_range: Tuple[float, float] = (-1.0, 1.0),
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pre-training data with planar geometry.

    Creates points on parallel horizontal planes with specified z-values.

    Args:
        plane_values: List of normalized z-values for each plane (potential field targets)
        num_points_per_plane: Number of points to sample per plane
        x_range: Range for x coordinates
        y_range: Range for y coordinates
        z_range: Range for z coordinates (planes are distributed within this)
        noise_std: Standard deviation of noise to add to coordinates

    Returns:
        Tuple of (coordinates, target_values)
        - coordinates: Shape (num_planes * num_points_per_plane, 3)
        - target_values: Shape (num_planes * num_points_per_plane, 1)
    """
    num_planes = len(plane_values)
    all_coords = []
    all_targets = []

    # Distribute planes evenly in z_range
    z_positions = np.linspace(z_range[0], z_range[1], num_planes)

    for i, (z_pos, target_val) in enumerate(zip(z_positions, plane_values)):
        # Generate random x, y coordinates
        x = np.random.uniform(x_range[0], x_range[1], num_points_per_plane)
        y = np.random.uniform(y_range[0], y_range[1], num_points_per_plane)
        z = np.full(num_points_per_plane, z_pos)

        # Add noise
        if noise_std > 0:
            x += np.random.normal(0, noise_std, num_points_per_plane)
            y += np.random.normal(0, noise_std, num_points_per_plane)
            z += np.random.normal(0, noise_std, num_points_per_plane)

        coords = np.stack([x, y, z], axis=-1)
        targets = np.full((num_points_per_plane, 1), target_val)

        all_coords.append(coords)
        all_targets.append(targets)

    coords = np.concatenate(all_coords, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return coords, targets


def sample_modeling_region(
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    num_samples: int,
) -> np.ndarray:
    """
    Sample random points from the modeling region for smoothness constraint.

    Args:
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        num_samples: Number of points to sample

    Returns:
        Sampled coordinates of shape (num_samples, 3)
    """
    x = np.random.uniform(bounds[0][0], bounds[0][1], num_samples)
    y = np.random.uniform(bounds[1][0], bounds[1][1], num_samples)
    z = np.random.uniform(bounds[2][0], bounds[2][1], num_samples)

    return np.stack([x, y, z], axis=-1)
