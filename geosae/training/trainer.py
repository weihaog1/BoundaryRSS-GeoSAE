"""
Main training module for GeoSAE.

This module implements the complete training pipeline with geological
constraints as described in Section 2.3 of the paper.

Training stages:
1. Load pre-trained autoencoder weights into SAE
2. Train with geological constraint loss function
3. Continue until convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Tuple, Union
import numpy as np
from tqdm import tqdm

from ..models import GeoSAE, AutoEncoder
from ..data import StratigraphicDataset
from ..data.preprocessing import (
    create_default_sequence_relations,
    sample_modeling_region,
)
from ..losses import GeoConstraintLoss
from .pretrain import pretrain_autoencoder


class Trainer:
    """
    Main trainer for GeoSAE with geological constraints.

    This trainer:
    1. Optionally runs pre-training with planar geometry
    2. Loads pre-trained weights into the stacked autoencoder
    3. Trains with geological constraint loss functions
    4. Tracks training metrics and history
    """

    def __init__(
        self,
        model: GeoSAE,
        device: str = "cuda",
        learning_rate: float = 0.001,
        lr_decay_step: int = 500,
        lr_decay_factor: float = 0.5,
        lambda_variance: float = 1.0,
        lambda_smoothness: float = 0.1,
        lambda_sequence: float = 1.0,
        lambda_attitude: float = 1.0,
    ):
        """
        Initialize the trainer.

        Args:
            model: GeoSAE model to train
            device: Device to use for training
            learning_rate: Initial learning rate
            lr_decay_step: Steps between learning rate decay
            lr_decay_factor: Factor to decay learning rate
            lambda_variance: Weight for variance loss
            lambda_smoothness: Weight for smoothness loss
            lambda_sequence: Weight for sequence constraint losses
            lambda_attitude: Weight for attitude constraint loss
        """
        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, using CPU")

        self.device = device
        self.model = model.to(device)

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_decay_step, gamma=lr_decay_factor
        )

        self.loss_fn = GeoConstraintLoss(
            lambda_variance=lambda_variance,
            lambda_above=lambda_sequence,
            lambda_below=lambda_sequence,
            lambda_overlap=lambda_sequence,
            lambda_attitude=lambda_attitude,
            lambda_smoothness=lambda_smoothness,
        )

        self.history = {
            "total_loss": [],
            "variance_loss": [],
            "sequence_loss": [],
            "smoothness_loss": [],
            "attitude_loss": [],
            "learning_rate": [],
        }

        # Training state
        self._is_pretrained = False

    def pretrain(
        self,
        num_iterations: int = 5000,
        batch_size: int = 128,
        plane_values: list = [0, 25, 50, 75],
        learning_rate: float = 0.001,
    ) -> None:
        """
        Run pre-training with planar geometry.

        Pre-trains a single autoencoder and loads weights into all
        autoencoders in the stacked architecture.

        Args:
            num_iterations: Number of pre-training iterations
            batch_size: Batch size for pre-training
            plane_values: Target values for planar geometry
            learning_rate: Learning rate for pre-training
        """
        print("Starting pre-training with planar geometry...")

        # Get model configuration from the first autoencoder
        first_ae = self.model.sae.autoencoders[0]

        # Pre-train
        pretrained_ae = pretrain_autoencoder(
            input_dim=first_ae.input_dim,
            encoder_channels=first_ae.encoder_channels,
            encoder_fc_dim=first_ae.encoder_fc_dim,
            decoder_channels=first_ae.decoder_channels,
            beta=first_ae.beta,
            device=self.device,
            num_iterations=num_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            plane_values=plane_values,
        )

        # Load pre-trained weights into all autoencoders
        print("Loading pre-trained weights into stacked autoencoder...")
        self.model.sae.load_pretrained_weights(pretrained_ae)
        self.model.to(self.device)

        # Reinitialize optimizer with new model parameters
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.optimizer.defaults["lr"]
        )

        self._is_pretrained = True
        print("Pre-training complete.")

    def train(
        self,
        dataset: StratigraphicDataset,
        num_iterations: int = 5000,
        batch_size: int = 128,
        sequence_relations: Optional[Dict] = None,
        smoothness_samples_per_iter: int = 1000,
        log_interval: int = 100,
        callback: Optional[Callable] = None,
        attitude_data: Optional[Dict] = None,
    ) -> Dict[str, list]:
        """
        Run main training with geological constraints.

        Args:
            dataset: StratigraphicDataset with surface sampling points
            num_iterations: Number of training iterations
            batch_size: Batch size
            sequence_relations: Stratigraphic sequence relationships (auto-created if None)
            smoothness_samples_per_iter: Number of random samples for smoothness constraint
            log_interval: Interval for logging progress
            callback: Optional callback function
            attitude_data: Optional dictionary with attitude point data:
                {
                    "coords": tensor of shape (N, 3),
                    "vectors": tensor of shape (N, 3),
                    "field_indices": tensor of shape (N,)
                }

        Returns:
            Training history dictionary
        """
        # Create sequence relations if not provided
        if sequence_relations is None:
            sequence_relations = create_default_sequence_relations(dataset.num_surfaces)

        # Store normalization parameters in model
        norm_params = dataset.get_normalization_params()
        self.model.set_normalization_params(
            norm_params["coord_min"], norm_params["coord_max"]
        )

        # Create data loader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model.train()

        # Bounds for smoothness sampling (in normalized coordinates)
        bounds = ((-1, 1), (-1, 1), (-1, 1))

        # Training loop
        pbar = tqdm(range(num_iterations), desc="Training GeoSAE")
        data_iter = iter(dataloader)

        for iteration in pbar:
            # Get batch (cycle through dataset)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            coords = batch["coords"].to(self.device)
            surface_labels = batch["surface_label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(coords, normalize=False)  # Already normalized

            # Generate smoothness samples
            smoothness_coords = torch.from_numpy(
                sample_modeling_region(bounds, smoothness_samples_per_iter)
            ).float().to(self.device)

            # Prepare attitude data if available
            if attitude_data is not None:
                att_coords = attitude_data["coords"].to(self.device)
                att_vectors = attitude_data["vectors"].to(self.device)
                att_indices = attitude_data["field_indices"].to(self.device)
            else:
                att_coords = None
                att_vectors = None
                att_indices = None

            # Compute losses
            losses = self.loss_fn(
                model=self.model,
                coords=coords,
                surface_labels=surface_labels,
                predictions=predictions,
                sequence_relations=sequence_relations,
                num_surfaces=dataset.num_surfaces,
                num_potential_fields=self.model.num_potential_fields,
                attitude_coords=att_coords,
                attitude_vectors=att_vectors,
                attitude_field_indices=att_indices,
                smoothness_sample_coords=smoothness_coords,
            )

            # Backward pass
            losses["total"].backward()
            self.optimizer.step()
            self.scheduler.step()

            # Update surface isovalues (mean potential field values)
            with torch.no_grad():
                self.model.update_surface_isovalues(losses["mean_values"])

            # Record history
            self.history["total_loss"].append(losses["total"].item())
            self.history["variance_loss"].append(losses["variance"].item())
            self.history["sequence_loss"].append(losses["sequence_total"].item())
            self.history["smoothness_loss"].append(losses["smoothness"].item())
            self.history["attitude_loss"].append(losses["attitude"].item())
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])

            # Update progress bar
            if iteration % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{losses['total'].item():.6f}",
                    "var": f"{losses['variance'].item():.6f}",
                    "seq": f"{losses['sequence_total'].item():.6f}",
                    "smooth": f"{losses['smoothness'].item():.6f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                })

            # Callback
            if callback is not None:
                callback(iteration, losses)

        return self.history

    def train_full(
        self,
        dataset: StratigraphicDataset,
        pretrain_iterations: int = 5000,
        train_iterations: int = 5000,
        batch_size: int = 128,
        sequence_relations: Optional[Dict] = None,
        plane_values: list = [0, 25, 50, 75],
        **kwargs,
    ) -> Dict[str, list]:
        """
        Run full training pipeline: pre-training + main training.

        Args:
            dataset: StratigraphicDataset
            pretrain_iterations: Number of pre-training iterations
            train_iterations: Number of main training iterations
            batch_size: Batch size
            sequence_relations: Stratigraphic sequence relationships
            plane_values: Target values for pre-training planes
            **kwargs: Additional arguments passed to train()

        Returns:
            Training history dictionary
        """
        # Pre-training
        self.pretrain(
            num_iterations=pretrain_iterations,
            batch_size=batch_size,
            plane_values=plane_values,
        )

        # Main training
        history = self.train(
            dataset=dataset,
            num_iterations=train_iterations,
            batch_size=batch_size,
            sequence_relations=sequence_relations,
            **kwargs,
        )

        return history

    def evaluate(
        self,
        dataset: StratigraphicDataset,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        all_coords = dataset.coords.to(self.device)
        all_labels = dataset.surface_labels.to(self.device)

        with torch.no_grad():
            predictions = self.model(all_coords, normalize=False)

        # Compute metrics per surface
        metrics = {}
        for surface_idx in range(dataset.num_surfaces):
            mask = all_labels == surface_idx
            if mask.sum() > 0:
                surface_preds = predictions[mask, min(surface_idx, predictions.shape[1] - 1)]

                # Mean and std of predictions
                mean_pred = surface_preds.mean().item()
                std_pred = surface_preds.std().item()

                metrics[f"surface_{surface_idx}_mean"] = mean_pred
                metrics[f"surface_{surface_idx}_std"] = std_pred

        # Overall variance loss
        total_variance = 0
        for surface_idx in range(dataset.num_surfaces):
            mask = all_labels == surface_idx
            if mask.sum() > 1:
                surface_preds = predictions[mask]
                total_variance += surface_preds.var().item()

        metrics["total_variance"] = total_variance

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "is_pretrained": self._is_pretrained,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        self._is_pretrained = checkpoint.get("is_pretrained", False)

    def get_trained_model(self) -> GeoSAE:
        """Get the trained model."""
        return self.model
