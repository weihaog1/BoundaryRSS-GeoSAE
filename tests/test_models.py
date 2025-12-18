"""
Unit tests for GeoSAE models.

Tests cover:
- AutoEncoder forward pass and gradient computation
- StackedAutoEncoder multi-field prediction
- GeoSAE full model functionality
- Loss function computations
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geosae.models import AutoEncoder, StackedAutoEncoder, GeoSAE
from geosae.losses import GeoConstraintLoss, VarianceLoss
from geosae.data import StratigraphicDataset
from geosae.data.preprocessing import (
    normalize_coordinates,
    create_default_sequence_relations,
    generate_planar_pretrain_data,
)


class TestAutoEncoder:
    """Tests for AutoEncoder model."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = AutoEncoder(input_dim=3)
        x = torch.randn(32, 3)
        output = model(x)

        assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"

    def test_encode_decode(self):
        """Test encode and decode methods separately."""
        model = AutoEncoder(input_dim=3)
        x = torch.randn(32, 3)

        z = model.encode(x)
        assert z.shape == (32, 256), f"Expected (32, 256), got {z.shape}"

        output = model.decode(z)
        assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"

    def test_gradient_computation(self):
        """Test gradient computation for Eikonal constraint."""
        model = AutoEncoder(input_dim=3)
        x = torch.randn(10, 3, requires_grad=True)

        grad = model.get_gradient(x)

        assert grad.shape == (10, 3), f"Expected (10, 3), got {grad.shape}"
        assert not torch.isnan(grad).any(), "Gradient contains NaN values"

    def test_gradient_norm(self):
        """Test gradient norm computation."""
        model = AutoEncoder(input_dim=3)
        x = torch.randn(10, 3)

        grad_norm = model.get_gradient_norm(x)

        assert grad_norm.shape == (10, 1), f"Expected (10, 1), got {grad_norm.shape}"
        assert (grad_norm >= 0).all(), "Gradient norm should be non-negative"

    def test_custom_architecture(self):
        """Test custom encoder/decoder architecture."""
        model = AutoEncoder(
            input_dim=3,
            encoder_channels=[3, 64, 128],
            encoder_fc_dim=128,
            decoder_channels=[128, 64, 1],
            beta=2.0,
        )

        x = torch.randn(16, 3)
        output = model(x)

        assert output.shape == (16, 1)


class TestStackedAutoEncoder:
    """Tests for StackedAutoEncoder model."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = StackedAutoEncoder(num_potential_fields=5)
        x = torch.randn(32, 3)
        output = model(x)

        assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"

    def test_forward_single(self):
        """Test single potential field prediction."""
        model = StackedAutoEncoder(num_potential_fields=5)
        x = torch.randn(32, 3)

        output = model.forward_single(x, field_idx=2)

        assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"

    def test_get_gradient(self):
        """Test gradient computation for single field."""
        model = StackedAutoEncoder(num_potential_fields=3)
        x = torch.randn(10, 3)

        grad = model.get_gradient(x, field_idx=1)

        assert grad.shape == (10, 3), f"Expected (10, 3), got {grad.shape}"

    def test_get_all_gradients(self):
        """Test gradient computation for all fields."""
        model = StackedAutoEncoder(num_potential_fields=3)
        x = torch.randn(10, 3)

        grads = model.get_all_gradients(x)

        assert grads.shape == (10, 3, 3), f"Expected (10, 3, 3), got {grads.shape}"

    def test_load_pretrained_weights(self):
        """Test loading pre-trained weights."""
        pretrained = AutoEncoder(input_dim=3)
        model = StackedAutoEncoder(num_potential_fields=3)

        # Should not raise
        model.load_pretrained_weights(pretrained)

        # Check weights are loaded
        for ae in model.autoencoders:
            for (p1, p2) in zip(pretrained.parameters(), ae.parameters()):
                assert torch.allclose(p1, p2)


class TestGeoSAE:
    """Tests for GeoSAE model."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = GeoSAE(num_stratigraphic_surfaces=5)
        x = torch.randn(32, 3)

        output = model(x, normalize=False)

        assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"

    def test_normalization(self):
        """Test coordinate normalization."""
        model = GeoSAE(num_stratigraphic_surfaces=3)

        coord_min = torch.tensor([0.0, 0.0, -100.0])
        coord_max = torch.tensor([1000.0, 1000.0, 0.0])
        model.set_normalization_params(coord_min, coord_max)

        # Test point in the middle
        x = torch.tensor([[500.0, 500.0, -50.0]])
        normalized = model.normalize_coordinates(x)

        # Should be approximately [0, 0, 0]
        assert torch.allclose(normalized, torch.zeros(1, 3), atol=0.1)

    def test_denormalization(self):
        """Test coordinate denormalization."""
        model = GeoSAE(num_stratigraphic_surfaces=3)

        coord_min = torch.tensor([0.0, 0.0, -100.0])
        coord_max = torch.tensor([1000.0, 1000.0, 0.0])
        model.set_normalization_params(coord_min, coord_max)

        normalized = torch.tensor([[0.0, 0.0, 0.0]])
        denormalized = model.denormalize_coordinates(normalized)

        expected = torch.tensor([[500.0, 500.0, -50.0]])
        assert torch.allclose(denormalized, expected, atol=1.0)

    def test_save_load(self, tmp_path):
        """Test model save and load."""
        model = GeoSAE(
            num_stratigraphic_surfaces=5,
            stratigraphic_codes=["S0", "S1", "S2", "S3", "S4"],
        )

        # Forward pass before save
        x = torch.randn(10, 3)
        output_before = model(x, normalize=False)

        # Save
        save_path = tmp_path / "model.pt"
        model.save(str(save_path))

        # Load
        loaded_model = GeoSAE.load(str(save_path))

        # Forward pass after load
        output_after = loaded_model(x, normalize=False)

        assert torch.allclose(output_before, output_after)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_variance_loss(self):
        """Test variance loss computation."""
        loss_fn = VarianceLoss()

        predictions = torch.randn(100, 1)
        surface_labels = torch.randint(0, 5, (100,))

        loss = loss_fn(predictions, surface_labels, num_surfaces=5)

        assert loss.shape == (), "Loss should be scalar"
        assert loss >= 0, "Variance loss should be non-negative"

    def test_variance_loss_zero_for_constant(self):
        """Test that variance loss is zero when all predictions are the same."""
        loss_fn = VarianceLoss()

        # All predictions are 1.0
        predictions = torch.ones(100, 1)
        surface_labels = torch.zeros(100, dtype=torch.long)

        loss = loss_fn(predictions, surface_labels, num_surfaces=1)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    def test_normalize_coordinates(self):
        """Test coordinate normalization."""
        coords = np.array([
            [0, 0, 0],
            [100, 100, 100],
            [50, 50, 50],
        ], dtype=np.float32)

        normalized, params = normalize_coordinates(coords)

        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0
        assert "coord_min" in params
        assert "coord_max" in params

    def test_create_default_sequence_relations(self):
        """Test default sequence relations creation."""
        relations = create_default_sequence_relations(num_surfaces=5)

        assert len(relations) == 5

        # Surface 0 should be above all others
        assert relations[0]["above"] == [1, 2, 3, 4]
        assert relations[0]["below"] == []

        # Surface 4 should be below all others
        assert relations[4]["above"] == []
        assert relations[4]["below"] == [0, 1, 2, 3]

    def test_generate_planar_pretrain_data(self):
        """Test planar pre-training data generation."""
        coords, targets = generate_planar_pretrain_data(
            plane_values=[0, 25, 50, 75],
            num_points_per_plane=100,
        )

        assert coords.shape == (400, 3)
        assert targets.shape == (400, 1)


class TestStratigraphicDataset:
    """Tests for StratigraphicDataset."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        coords = np.random.rand(100, 3).astype(np.float32)
        labels = np.random.randint(0, 5, 100)

        dataset = StratigraphicDataset(coords, labels)

        assert len(dataset) == 100
        assert dataset.num_surfaces == 5

    def test_dataset_normalization(self):
        """Test that dataset normalizes coordinates."""
        coords = np.array([
            [0, 0, 0],
            [1000, 1000, 100],
        ], dtype=np.float32)
        labels = np.array([0, 1])

        dataset = StratigraphicDataset(coords, labels, normalize=True)

        # Check that coordinates are in [-1, 1] range
        assert dataset.coords.min() >= -1.0
        assert dataset.coords.max() <= 1.0

    def test_dataset_getitem(self):
        """Test dataset indexing."""
        coords = np.random.rand(100, 3).astype(np.float32)
        labels = np.random.randint(0, 5, 100)

        dataset = StratigraphicDataset(coords, labels)
        item = dataset[0]

        assert "coords" in item
        assert "surface_label" in item
        assert item["coords"].shape == (3,)

    def test_get_surface_points(self):
        """Test getting points for a specific surface."""
        coords = np.random.rand(100, 3).astype(np.float32)
        labels = np.array([0] * 50 + [1] * 50)

        dataset = StratigraphicDataset(coords, labels)
        surface_coords, indices = dataset.get_surface_points(0)

        assert len(surface_coords) == 50
        assert len(indices) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
