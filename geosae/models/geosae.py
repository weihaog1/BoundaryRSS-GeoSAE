"""
GeoSAE: Main model combining Stacked Autoencoder with geological constraints.

This is the main interface for the GeoSAE method, providing:
1. Model initialization with geological configuration
2. Prediction of potential fields
3. Extraction of stratigraphic surfaces

Reference:
    Yang, Y.; Zhou, J.; Ruan, M.; Xiao, H.; Hua, W.; Wei, W.
    GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint.
    Appl. Sci. 2025, 15, 1185.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple, Union
from .stacked_autoencoder import StackedAutoEncoder
from .autoencoder import AutoEncoder


class GeoSAE(nn.Module):
    """
    GeoSAE: Geological constraint-driven 3D stratigraphic modeling.

    This class wraps the StackedAutoEncoder and provides methods for:
    - Predicting potential field values for arbitrary coordinates
    - Computing stratigraphic surface isovalues
    - Generating 3D grids for visualization
    """

    def __init__(
        self,
        num_stratigraphic_surfaces: int,
        stratigraphic_codes: Optional[List[str]] = None,
        potential_field_mapping: Optional[Dict[str, int]] = None,
        input_dim: int = 3,
        encoder_channels: Optional[List[int]] = None,
        encoder_fc_dim: int = 256,
        decoder_channels: Optional[List[int]] = None,
        beta: float = 1.0,
    ):
        """
        Initialize GeoSAE model.

        Args:
            num_stratigraphic_surfaces: Number of stratigraphic surfaces to model
            stratigraphic_codes: List of codes/names for each surface (e.g., ["N1d", "N2h", ...])
            potential_field_mapping: Mapping from surface code to potential field index
            input_dim: Input dimension (3 for x, y, z)
            encoder_channels: Encoder channel dimensions
            encoder_fc_dim: Encoder fully connected layer dimension
            decoder_channels: Decoder channel dimensions
            beta: Softplus activation parameter
        """
        super().__init__()

        self.num_stratigraphic_surfaces = num_stratigraphic_surfaces

        # Setup stratigraphic codes
        if stratigraphic_codes is None:
            self.stratigraphic_codes = [f"S{i}" for i in range(num_stratigraphic_surfaces)]
        else:
            self.stratigraphic_codes = stratigraphic_codes

        # Setup potential field mapping
        if potential_field_mapping is None:
            # Each surface gets its own potential field by default
            self.potential_field_mapping = {
                code: i for i, code in enumerate(self.stratigraphic_codes)
            }
        else:
            self.potential_field_mapping = potential_field_mapping

        # Determine number of potential fields needed
        self.num_potential_fields = len(set(self.potential_field_mapping.values()))

        # Create the stacked autoencoder
        self.sae = StackedAutoEncoder(
            num_potential_fields=self.num_potential_fields,
            input_dim=input_dim,
            encoder_channels=encoder_channels,
            encoder_fc_dim=encoder_fc_dim,
            decoder_channels=decoder_channels,
            beta=beta,
        )

        # Store isovalues for each stratigraphic surface (learned during training)
        # These are the mean potential field values at each surface
        self.register_buffer(
            "surface_isovalues",
            torch.zeros(num_stratigraphic_surfaces),
        )

        # Store normalization parameters
        self.register_buffer("coord_min", torch.zeros(3))
        self.register_buffer("coord_max", torch.ones(3))
        self.register_buffer("coord_scale", torch.ones(3))

    def set_normalization_params(
        self,
        coord_min: Union[np.ndarray, torch.Tensor],
        coord_max: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """
        Set coordinate normalization parameters.

        Args:
            coord_min: Minimum coordinates (x_min, y_min, z_min)
            coord_max: Maximum coordinates (x_max, y_max, z_max)
        """
        if isinstance(coord_min, np.ndarray):
            coord_min = torch.from_numpy(coord_min).float()
        if isinstance(coord_max, np.ndarray):
            coord_max = torch.from_numpy(coord_max).float()

        self.coord_min = coord_min
        self.coord_max = coord_max
        self.coord_scale = coord_max - coord_min

    def normalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates to [-1, 1] range.

        Args:
            coords: Coordinates of shape (N, 3)

        Returns:
            Normalized coordinates of shape (N, 3)
        """
        # Normalize to [0, 1] then to [-1, 1]
        normalized = (coords - self.coord_min) / (self.coord_scale + 1e-8)
        return 2.0 * normalized - 1.0

    def denormalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Denormalize coordinates from [-1, 1] to original range.

        Args:
            coords: Normalized coordinates of shape (N, 3)

        Returns:
            Original coordinates of shape (N, 3)
        """
        # From [-1, 1] to [0, 1] then to original range
        normalized = (coords + 1.0) / 2.0
        return normalized * self.coord_scale + self.coord_min

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Predict potential field values for input coordinates.

        Args:
            x: Input coordinates of shape (batch_size, 3)
            normalize: Whether to normalize input coordinates

        Returns:
            Potential field values of shape (batch_size, num_potential_fields)
        """
        if normalize:
            x = self.normalize_coordinates(x)
        return self.sae(x)

    def predict_surface(
        self,
        x: torch.Tensor,
        surface_idx: int,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Predict potential field value for a specific stratigraphic surface.

        Args:
            x: Input coordinates of shape (batch_size, 3)
            surface_idx: Index of the stratigraphic surface
            normalize: Whether to normalize input coordinates

        Returns:
            Potential field value of shape (batch_size, 1)
        """
        if normalize:
            x = self.normalize_coordinates(x)

        # Get the potential field index for this surface
        surface_code = self.stratigraphic_codes[surface_idx]
        field_idx = self.potential_field_mapping[surface_code]

        return self.sae.forward_single(x, field_idx)

    def update_surface_isovalues(self, isovalues: torch.Tensor) -> None:
        """
        Update the stored isovalues for stratigraphic surfaces.

        These values are computed during training as the mean potential field
        values at each stratigraphic surface sampling point.

        Args:
            isovalues: Tensor of isovalues of shape (num_stratigraphic_surfaces,)
        """
        self.surface_isovalues.copy_(isovalues)

    def get_gradient(
        self,
        x: torch.Tensor,
        surface_idx: int,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute the gradient of the potential field for a specific surface.

        Args:
            x: Input coordinates of shape (batch_size, 3)
            surface_idx: Index of the stratigraphic surface
            normalize: Whether to normalize input coordinates

        Returns:
            Gradient tensor of shape (batch_size, 3)
        """
        if normalize:
            x = self.normalize_coordinates(x)

        surface_code = self.stratigraphic_codes[surface_idx]
        field_idx = self.potential_field_mapping[surface_code]

        return self.sae.get_gradient(x, field_idx)

    def predict_grid(
        self,
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        resolution: Tuple[int, int, int],
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict potential field values on a regular 3D grid.

        Args:
            bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
            resolution: (nx, ny, nz) grid resolution
            normalize: Whether to normalize coordinates

        Returns:
            Tuple of:
                - grid_coords: Coordinates of shape (nx*ny*nz, 3)
                - grid_values: Potential field values of shape (nx*ny*nz, num_potential_fields)
        """
        # Create grid coordinates
        x = torch.linspace(bounds[0][0], bounds[0][1], resolution[0])
        y = torch.linspace(bounds[1][0], bounds[1][1], resolution[1])
        z = torch.linspace(bounds[2][0], bounds[2][1], resolution[2])

        # Create meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        grid_coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

        # Move to same device as model
        grid_coords = grid_coords.to(self.coord_min.device)

        # Predict values (in batches to avoid memory issues)
        batch_size = 10000
        all_values = []

        with torch.no_grad():
            for i in range(0, len(grid_coords), batch_size):
                batch = grid_coords[i : i + batch_size]
                values = self.forward(batch, normalize=normalize)
                all_values.append(values)

        grid_values = torch.cat(all_values, dim=0)

        return grid_coords, grid_values

    def save(self, path: str) -> None:
        """Save model state to file."""
        state = {
            "model_state_dict": self.state_dict(),
            "num_stratigraphic_surfaces": self.num_stratigraphic_surfaces,
            "stratigraphic_codes": self.stratigraphic_codes,
            "potential_field_mapping": self.potential_field_mapping,
            "num_potential_fields": self.num_potential_fields,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GeoSAE":
        """Load model from file."""
        state = torch.load(path, map_location=device)

        model = cls(
            num_stratigraphic_surfaces=state["num_stratigraphic_surfaces"],
            stratigraphic_codes=state["stratigraphic_codes"],
            potential_field_mapping=state["potential_field_mapping"],
        )
        model.load_state_dict(state["model_state_dict"])
        model.to(device)

        return model
