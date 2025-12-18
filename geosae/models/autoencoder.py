"""
Autoencoder (AE) model for GeoSAE.

The autoencoder network consists of an encoder-decoder structure for predicting
potential field values from 3D coordinates.

Reference:
    Section 2.2.1 of the GeoSAE paper - Stacked Autoencoder Network Construction
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ParametricSoftplus(nn.Module):
    """
    Parametric Softplus activation function.

    σ(x) = (1/β) * log(1 + e^(βx))

    The smoothness of the modeled interface is controlled by the parameter β:
    - Smaller β values result in flatter interfaces
    - Larger β values emphasize local details
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 / self.beta) * torch.log(1 + torch.exp(self.beta * x))


class AutoEncoder(nn.Module):
    """
    Autoencoder network for predicting potential field values.

    Architecture (from paper):
        Encoder:
            - Conv1D(3 -> 128, kernel=1, stride=1)
            - Conv1D(128 -> 256, kernel=1, stride=1)
            - FC(256 -> 256)
        Decoder:
            - Conv1D(256 -> 128, kernel=1, stride=1)
            - Conv1D(128 -> 1, kernel=1, stride=1)

    The convolutional layers use kernel size of 1 and stride of 1, ensuring
    that the input tensor's dimensions are preserved while producing output
    tensors with the same height and width but varying channel depth.
    """

    def __init__(
        self,
        input_dim: int = 3,
        encoder_channels: Optional[List[int]] = None,
        encoder_fc_dim: int = 256,
        decoder_channels: Optional[List[int]] = None,
        beta: float = 1.0,
    ):
        """
        Initialize the AutoEncoder.

        Args:
            input_dim: Input dimension (typically 3 for x, y, z coordinates)
            encoder_channels: List of channel dimensions for encoder conv layers
                              Default: [3, 128, 256]
            encoder_fc_dim: Dimension of the fully connected layer in encoder
            decoder_channels: List of channel dimensions for decoder conv layers
                              Default: [256, 128, 1]
            beta: Parameter for the Softplus activation function
        """
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [input_dim, 128, 256]
        if decoder_channels is None:
            decoder_channels = [256, 128, 1]

        self.beta = beta

        # Build encoder
        encoder_layers = []
        for i in range(len(encoder_channels) - 1):
            encoder_layers.append(
                nn.Conv1d(encoder_channels[i], encoder_channels[i + 1], kernel_size=1, stride=1)
            )
            encoder_layers.append(ParametricSoftplus(beta))

        # Add fully connected layer
        encoder_layers.append(nn.Flatten())
        encoder_layers.append(nn.Linear(encoder_channels[-1], encoder_fc_dim))
        encoder_layers.append(ParametricSoftplus(beta))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        # First we need to reshape the FC output back to conv input
        self.decoder_fc = nn.Sequential(
            nn.Linear(encoder_fc_dim, decoder_channels[0]),
            ParametricSoftplus(beta),
        )

        for i in range(len(decoder_channels) - 1):
            decoder_layers.append(
                nn.Conv1d(decoder_channels[i], decoder_channels[i + 1], kernel_size=1, stride=1)
            )
            # No activation after the last layer
            if i < len(decoder_channels) - 2:
                decoder_layers.append(ParametricSoftplus(beta))

        self.decoder = nn.Sequential(*decoder_layers)

        # Store dimensions for reference
        self.input_dim = input_dim
        self.encoder_channels = encoder_channels
        self.encoder_fc_dim = encoder_fc_dim
        self.decoder_channels = decoder_channels

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input coordinates to latent representation.

        Args:
            x: Input tensor of shape (batch_size, 3) for coordinates (x, y, z)

        Returns:
            Encoded representation of shape (batch_size, encoder_fc_dim)
        """
        # Reshape for Conv1d: (batch, features) -> (batch, features, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to potential field value.

        Args:
            z: Latent representation of shape (batch_size, encoder_fc_dim)

        Returns:
            Potential field value of shape (batch_size, 1)
        """
        # Apply FC layer
        x = self.decoder_fc(z)
        # Reshape for Conv1d: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(-1)
        # Apply decoder conv layers
        x = self.decoder(x)
        # Reshape output: (batch, 1, 1) -> (batch, 1)
        return x.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: coordinates -> potential field value.

        Args:
            x: Input tensor of shape (batch_size, 3) for coordinates (x, y, z)

        Returns:
            Potential field value of shape (batch_size, 1)
        """
        z = self.encode(x)
        return self.decode(z)

    def get_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the potential field with respect to input coordinates.

        This is used for:
        1. Computing the signed distance for stratigraphic sequence constraints
        2. Enforcing the Eikonal constraint for smoothness

        Args:
            x: Input tensor of shape (batch_size, 3) with requires_grad=True

        Returns:
            Gradient tensor of shape (batch_size, 3)
        """
        x = x.clone().requires_grad_(True)
        phi = self.forward(x)

        # Compute gradients
        grad_outputs = torch.ones_like(phi)
        gradients = torch.autograd.grad(
            outputs=phi,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return gradients

    def get_gradient_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm of the potential field gradient.

        Used for the Eikonal constraint: ||∇S(x)|| = 1

        Args:
            x: Input tensor of shape (batch_size, 3)

        Returns:
            Gradient norm of shape (batch_size, 1)
        """
        gradients = self.get_gradient(x)
        return torch.norm(gradients, dim=-1, keepdim=True)
