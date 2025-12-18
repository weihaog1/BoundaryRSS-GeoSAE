"""
Pre-training module for GeoSAE.

This module implements the geometric initialization (pre-training) process
using planar geometry as described in Section 2.2.2 of the paper.

The pre-training:
1. Uses planar geometry (4 parallel planes)
2. Uses L1 and L2 loss functions
3. Initializes network parameters for faster convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from tqdm import tqdm

from ..models import AutoEncoder
from ..data import PreTrainingDataset
from ..losses.geological_constraints import PreTrainingLoss


class PreTrainer:
    """
    Pre-trainer for geometric initialization.

    Pre-trains a single autoencoder using planar geometry, then the
    pre-trained weights can be loaded into the stacked autoencoder.
    """

    def __init__(
        self,
        model: AutoEncoder,
        device: str = "cuda",
        learning_rate: float = 0.001,
        lr_decay_step: int = 500,
        lr_decay_factor: float = 0.5,
        l1_weight: float = 1.0,
        l2_weight: float = 0.1,
    ):
        """
        Initialize the pre-trainer.

        Args:
            model: AutoEncoder model to pre-train
            device: Device to use for training
            learning_rate: Initial learning rate
            lr_decay_step: Steps between learning rate decay
            lr_decay_factor: Factor to decay learning rate
            l1_weight: Weight for L1 loss
            l2_weight: Weight for L2 (gradient) loss
        """
        self.device = device
        self.model = model.to(device)

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_step, gamma=lr_decay_factor
        )

        self.loss_fn = PreTrainingLoss(l1_weight=l1_weight, l2_weight=l2_weight)

        self.history = {
            "loss": [],
            "l1_loss": [],
            "l2_loss": [],
            "learning_rate": [],
        }

    def train(
        self,
        num_iterations: int = 5000,
        batch_size: int = 128,
        plane_values: list = [0, 25, 50, 75],
        num_points_per_plane: int = 1000,
        log_interval: int = 100,
        callback: Optional[Callable] = None,
    ) -> Dict[str, list]:
        """
        Run pre-training.

        Args:
            num_iterations: Number of training iterations
            batch_size: Batch size
            plane_values: Target values for planar geometry
            num_points_per_plane: Number of points per plane
            log_interval: Interval for logging progress
            callback: Optional callback function called each iteration

        Returns:
            Training history dictionary
        """
        # Create pre-training dataset
        dataset = PreTrainingDataset.create_planar(
            plane_values=plane_values,
            num_points_per_plane=num_points_per_plane,
            normalize_targets=True,
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model.train()

        # Training loop
        pbar = tqdm(range(num_iterations), desc="Pre-training")
        data_iter = iter(dataloader)

        for iteration in pbar:
            # Get batch (cycle through dataset)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            coords = batch["coords"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(coords)

            # Compute loss
            losses = self.loss_fn(predictions, targets, self.model, coords)

            # Backward pass
            losses["total"].backward()
            self.optimizer.step()
            self.scheduler.step()

            # Record history
            self.history["loss"].append(losses["total"].item())
            self.history["l1_loss"].append(losses["l1"].item())
            self.history["l2_loss"].append(losses["l2"].item())
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])

            # Update progress bar
            if iteration % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.6f}",
                    "l1": f"{losses['l1'].item():.6f}",
                    "l2": f"{losses['l2'].item():.6f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                })

            # Callback
            if callback is not None:
                callback(iteration, losses)

        return self.history

    def get_pretrained_model(self) -> AutoEncoder:
        """Get the pre-trained model."""
        return self.model

    def save_checkpoint(self, path: str) -> None:
        """Save pre-training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load pre-training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]


def pretrain_autoencoder(
    input_dim: int = 3,
    encoder_channels: list = None,
    encoder_fc_dim: int = 256,
    decoder_channels: list = None,
    beta: float = 1.0,
    device: str = "cuda",
    num_iterations: int = 5000,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    plane_values: list = [0, 25, 50, 75],
) -> AutoEncoder:
    """
    Convenience function to pre-train an autoencoder.

    Args:
        input_dim: Input dimension
        encoder_channels: Encoder channel dimensions
        encoder_fc_dim: Encoder FC dimension
        decoder_channels: Decoder channel dimensions
        beta: Softplus parameter
        device: Device for training
        num_iterations: Number of iterations
        batch_size: Batch size
        learning_rate: Learning rate
        plane_values: Target values for planes

    Returns:
        Pre-trained AutoEncoder
    """
    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    # Create model
    model = AutoEncoder(
        input_dim=input_dim,
        encoder_channels=encoder_channels,
        encoder_fc_dim=encoder_fc_dim,
        decoder_channels=decoder_channels,
        beta=beta,
    )

    # Create pre-trainer and train
    pretrainer = PreTrainer(model, device=device, learning_rate=learning_rate)
    pretrainer.train(
        num_iterations=num_iterations,
        batch_size=batch_size,
        plane_values=plane_values,
    )

    return pretrainer.get_pretrained_model()
