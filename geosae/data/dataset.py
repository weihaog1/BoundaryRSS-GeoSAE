"""
Dataset classes for GeoSAE.

Provides PyTorch Dataset implementations for:
- Borehole data
- Stratigraphic surface sampling points
- Pre-training planar geometry data
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class BoreholeData:
    """
    Container for borehole data.

    Attributes:
        borehole_id: Unique identifier for the borehole
        x: X coordinate of the borehole
        y: Y coordinate of the borehole
        layers: List of dictionaries with layer information:
                [{"surface_code": str, "top_z": float, "bottom_z": float}, ...]
    """

    borehole_id: str
    x: float
    y: float
    layers: List[Dict] = field(default_factory=list)

    def to_surface_points(self) -> List[Dict]:
        """
        Convert borehole layers to surface sampling points.

        Returns:
            List of dictionaries with surface point information:
            [{"x": float, "y": float, "z": float, "surface_code": str}, ...]
        """
        points = []
        for layer in self.layers:
            points.append({
                "x": self.x,
                "y": self.y,
                "z": layer["top_z"],
                "surface_code": layer["surface_code"],
            })
        return points


class StratigraphicDataset(Dataset):
    """
    PyTorch Dataset for stratigraphic surface sampling points.

    This dataset provides batches of:
    - Coordinates (x, y, z)
    - Surface labels (indices)
    - Potential field targets (if available)
    """

    def __init__(
        self,
        coords: Union[np.ndarray, torch.Tensor],
        surface_labels: Union[np.ndarray, torch.Tensor],
        surface_codes: Optional[List[str]] = None,
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        normalize: bool = True,
        normalize_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Initialize the dataset.

        Args:
            coords: Coordinates of shape (N, 3)
            surface_labels: Surface index for each point of shape (N,)
            surface_codes: List of surface codes (for reference)
            targets: Target potential field values (optional, shape (N, F) or (N,))
            normalize: Whether to normalize coordinates
            normalize_range: Target range for normalization
        """
        # Convert to tensors if needed
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        if isinstance(surface_labels, np.ndarray):
            surface_labels = torch.from_numpy(surface_labels).long()
        if targets is not None and isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        self.original_coords = coords.clone()
        self.surface_labels = surface_labels
        self.surface_codes = surface_codes
        self.targets = targets
        self.normalize_range = normalize_range

        # Compute normalization parameters
        self.coord_min = coords.min(dim=0)[0]
        self.coord_max = coords.max(dim=0)[0]

        # Normalize coordinates
        if normalize:
            self.coords = self._normalize(coords)
        else:
            self.coords = coords

        # Compute number of unique surfaces
        self.num_surfaces = len(torch.unique(surface_labels))

    def _normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to target range."""
        coord_range = self.coord_max - self.coord_min
        coord_range = torch.where(coord_range == 0, torch.ones_like(coord_range), coord_range)

        normalized = (coords - self.coord_min) / coord_range
        target_min, target_max = self.normalize_range
        return normalized * (target_max - target_min) + target_min

    def _denormalize(self, coords: torch.Tensor) -> torch.Tensor:
        """Denormalize coordinates back to original range."""
        source_min, source_max = self.normalize_range
        normalized = (coords - source_min) / (source_max - source_min)
        coord_range = self.coord_max - self.coord_min
        return normalized * coord_range + self.coord_min

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "coords": self.coords[idx],
            "surface_label": self.surface_labels[idx],
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item

    def get_surface_points(self, surface_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all points belonging to a specific surface.

        Args:
            surface_idx: Index of the surface

        Returns:
            Tuple of (coordinates, indices)
        """
        mask = self.surface_labels == surface_idx
        indices = torch.where(mask)[0]
        return self.coords[mask], indices

    def get_normalization_params(self) -> Dict[str, torch.Tensor]:
        """Get normalization parameters."""
        return {
            "coord_min": self.coord_min,
            "coord_max": self.coord_max,
            "normalize_range": self.normalize_range,
        }

    @classmethod
    def from_boreholes(
        cls,
        boreholes: List[BoreholeData],
        surface_code_to_idx: Dict[str, int],
        **kwargs,
    ) -> "StratigraphicDataset":
        """
        Create dataset from borehole data.

        Args:
            boreholes: List of BoreholeData objects
            surface_code_to_idx: Mapping from surface code to index
            **kwargs: Additional arguments passed to __init__

        Returns:
            StratigraphicDataset instance
        """
        all_coords = []
        all_labels = []

        for borehole in boreholes:
            for point in borehole.to_surface_points():
                if point["surface_code"] in surface_code_to_idx:
                    all_coords.append([point["x"], point["y"], point["z"]])
                    all_labels.append(surface_code_to_idx[point["surface_code"]])

        coords = np.array(all_coords, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)

        surface_codes = list(surface_code_to_idx.keys())

        return cls(coords, labels, surface_codes=surface_codes, **kwargs)

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        surface_col: str = "surface_code",
        **kwargs,
    ) -> "StratigraphicDataset":
        """
        Create dataset from CSV file.

        Args:
            filepath: Path to CSV file
            x_col: Column name for x coordinates
            y_col: Column name for y coordinates
            z_col: Column name for z coordinates
            surface_col: Column name for surface codes
            **kwargs: Additional arguments passed to __init__

        Returns:
            StratigraphicDataset instance
        """
        df = pd.read_csv(filepath)

        coords = df[[x_col, y_col, z_col]].values.astype(np.float32)

        # Create surface code to index mapping
        unique_codes = df[surface_col].unique()
        code_to_idx = {code: i for i, code in enumerate(sorted(unique_codes))}
        labels = df[surface_col].map(code_to_idx).values.astype(np.int64)

        surface_codes = sorted(unique_codes)

        return cls(coords, labels, surface_codes=surface_codes, **kwargs)


class PreTrainingDataset(Dataset):
    """
    Dataset for pre-training with planar geometry.
    """

    def __init__(
        self,
        coords: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
    ):
        """
        Initialize pre-training dataset.

        Args:
            coords: Coordinates of shape (N, 3)
            targets: Target values of shape (N, 1) or (N,)
        """
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        self.coords = coords
        self.targets = targets.view(-1, 1) if targets.dim() == 1 else targets

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "coords": self.coords[idx],
            "target": self.targets[idx],
        }

    @classmethod
    def create_planar(
        cls,
        plane_values: List[float] = [0, 25, 50, 75],
        num_points_per_plane: int = 1000,
        normalize_targets: bool = True,
    ) -> "PreTrainingDataset":
        """
        Create pre-training dataset with planar geometry.

        Args:
            plane_values: Target values for each plane
            num_points_per_plane: Number of points per plane
            normalize_targets: Whether to normalize target values to [-1, 1]

        Returns:
            PreTrainingDataset instance
        """
        from .preprocessing import generate_planar_pretrain_data

        # Normalize target values if requested
        if normalize_targets:
            min_val = min(plane_values)
            max_val = max(plane_values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            normalized_values = [
                2.0 * (v - min_val) / range_val - 1.0 for v in plane_values
            ]
        else:
            normalized_values = plane_values

        coords, targets = generate_planar_pretrain_data(
            plane_values=normalized_values,
            num_points_per_plane=num_points_per_plane,
        )

        return cls(coords, targets)
