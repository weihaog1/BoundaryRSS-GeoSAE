"""
Geological Constraint Loss Functions for GeoSAE.

This module implements the loss functions that encode geological knowledge
as constraints during the training process:

1. Stratigraphic Consistency Constraint (Section 2.1.1)
2. Stratigraphic Sequence Constraints (Section 2.1.2)
3. Attitude Point Constraints (Section 2.1.3)
4. Stratigraphic Interface Smoothness Constraints (Section 2.1.4)

Reference:
    Yang, Y.; Zhou, J.; Ruan, M.; Xiao, H.; Hua, W.; Wei, W.
    GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint.
    Appl. Sci. 2025, 15, 1185.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class VarianceLoss(nn.Module):
    """
    Stratigraphic Surface Sampling Point Consistency Constraint.

    Equation (2) from the paper:
    Loss_I^Var = Σ Var(S_K^I_K)

    This ensures that all points sampled from the same stratigraphic surface
    have similar potential field values (variance near 0).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        surface_labels: torch.Tensor,
        num_surfaces: int,
    ) -> torch.Tensor:
        """
        Compute variance loss for stratigraphic consistency.

        Args:
            predictions: Predicted potential field values of shape (N, num_fields)
            surface_labels: Surface index for each point of shape (N,)
            num_surfaces: Total number of stratigraphic surfaces

        Returns:
            Variance loss scalar
        """
        total_loss = torch.tensor(0.0, device=predictions.device)

        for surface_idx in range(num_surfaces):
            mask = surface_labels == surface_idx
            if mask.sum() > 1:
                # Get predictions for this surface
                surface_preds = predictions[mask]

                # Compute variance
                variance = torch.var(surface_preds, dim=0)
                total_loss = total_loss + variance.sum()

        return total_loss


class SequenceConstraintLoss(nn.Module):
    """
    Stratigraphic Sequence Constraints.

    Implements the "Above", "Below", and "Overlap" relationships between
    stratigraphic surfaces as described in Section 2.1.2.

    Equations (7), (8), (9) from the paper:

    - Above: Points that should be above a surface should have larger potential field values
    - Below: Points that should be below a surface should have smaller potential field values
    - Overlap: Points at overlapping surfaces should have equal potential field values

    The loss uses the approximate signed distance δ:
    δ_{I_x,I_K} = (S_x^{I_X} - S̄_K^{I_K}) / ||∇S_x^{I_X}||
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        surface_labels: torch.Tensor,
        predictions: torch.Tensor,
        sequence_relations: Dict[int, Dict[str, List[int]]],
        mean_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sequence constraint losses.

        Args:
            model: The GeoSAE model (for gradient computation)
            coords: Coordinates of shape (N, 3)
            surface_labels: Surface index for each point of shape (N,)
            predictions: Predicted potential field values of shape (N, num_fields)
            sequence_relations: Dictionary mapping surface_idx to:
                {
                    "above": [list of surface indices this should be above],
                    "below": [list of surface indices this should be below],
                    "overlap": [list of surface indices this overlaps with]
                }
            mean_values: Mean potential field values for each surface of shape (num_surfaces,)

        Returns:
            Tuple of (above_loss, below_loss, overlap_loss)
        """
        device = predictions.device
        above_loss = torch.tensor(0.0, device=device)
        below_loss = torch.tensor(0.0, device=device)
        overlap_loss = torch.tensor(0.0, device=device)

        num_surfaces = len(sequence_relations)

        for surface_idx, relations in sequence_relations.items():
            mask = surface_labels == surface_idx
            if mask.sum() == 0:
                continue

            surface_coords = coords[mask]
            surface_preds = predictions[mask, 0]  # Assuming single potential field per surface

            # Compute gradient norm for signed distance calculation
            # We need to enable gradients for this computation
            surface_coords_grad = surface_coords.clone().requires_grad_(True)

            # Get predictions with gradient
            field_idx = surface_idx  # Simplified mapping
            if hasattr(model, 'sae'):
                pred_for_grad = model.sae.forward_single(surface_coords_grad, min(field_idx, model.sae.num_potential_fields - 1))
            else:
                pred_for_grad = model(surface_coords_grad)[:, 0:1]

            # Compute gradients
            grad_outputs = torch.ones_like(pred_for_grad)
            gradients = torch.autograd.grad(
                outputs=pred_for_grad,
                inputs=surface_coords_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient_norms = torch.norm(gradients, dim=-1, keepdim=False) + 1e-8

            # Above constraint: surface_preds should be > mean_values[below_surfaces]
            for below_idx in relations.get("above", []):
                if below_idx < len(mean_values):
                    diff = surface_preds - mean_values[below_idx]
                    # Penalize when diff < 0 (should be above but isn't)
                    signed_dist = torch.abs(diff) / gradient_norms
                    violation_mask = diff < 0
                    above_loss = above_loss + (signed_dist * violation_mask.float()).mean()

            # Below constraint: surface_preds should be < mean_values[above_surfaces]
            for above_idx in relations.get("below", []):
                if above_idx < len(mean_values):
                    diff = surface_preds - mean_values[above_idx]
                    # Penalize when diff > 0 (should be below but isn't)
                    signed_dist = torch.abs(diff) / gradient_norms
                    violation_mask = diff > 0
                    below_loss = below_loss + (signed_dist * violation_mask.float()).mean()

            # Overlap constraint: surface_preds should equal mean_values[overlap_surfaces]
            for overlap_idx in relations.get("overlap", []):
                if overlap_idx < len(mean_values):
                    diff = surface_preds - mean_values[overlap_idx]
                    signed_dist = torch.abs(diff) / gradient_norms
                    overlap_loss = overlap_loss + signed_dist.mean()

        return above_loss, below_loss, overlap_loss


class AttitudeConstraintLoss(nn.Module):
    """
    Attitude Point Constraints.

    Equation (13) from the paper:
    L_O = Σ (1/|O|) Σ |cos(θ_o) - cos(θ_o^K)|

    This ensures that the gradient direction of the potential field at
    attitude points aligns with the measured attitude direction (dip/strike).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        model: nn.Module,
        attitude_coords: torch.Tensor,
        attitude_vectors: torch.Tensor,
        field_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attitude constraint loss.

        Args:
            model: The GeoSAE model (for gradient computation)
            attitude_coords: Coordinates of attitude points of shape (N, 3)
            attitude_vectors: Measured attitude direction vectors of shape (N, 3)
            field_indices: Potential field index for each attitude point of shape (N,)

        Returns:
            Attitude constraint loss scalar
        """
        if len(attitude_coords) == 0:
            return torch.tensor(0.0, device=attitude_coords.device if len(attitude_coords) > 0 else "cpu")

        device = attitude_coords.device
        total_loss = torch.tensor(0.0, device=device)

        unique_fields = torch.unique(field_indices)

        for field_idx in unique_fields:
            mask = field_indices == field_idx
            coords = attitude_coords[mask]
            vectors = attitude_vectors[mask]

            if len(coords) == 0:
                continue

            # Normalize attitude vectors
            vectors = vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-8)

            # Compute gradients at attitude points
            coords_grad = coords.clone().requires_grad_(True)

            if hasattr(model, 'sae'):
                pred = model.sae.forward_single(coords_grad, int(field_idx))
            else:
                pred = model(coords_grad)[:, int(field_idx):int(field_idx)+1]

            grad_outputs = torch.ones_like(pred)
            gradients = torch.autograd.grad(
                outputs=pred,
                inputs=coords_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Normalize gradients
            gradient_norms = torch.norm(gradients, dim=-1, keepdim=True) + 1e-8
            gradients_normalized = gradients / gradient_norms

            # Compute cosine similarity (should be close to 1 or -1 for parallel)
            # cos(θ) = v · ∇S / (||v|| ||∇S||)
            cos_theta = torch.sum(vectors * gradients_normalized, dim=-1)

            # Loss: we want |cos(θ)| to be 1 (parallel or anti-parallel)
            # So loss = 1 - |cos(θ)|
            loss = (1 - torch.abs(cos_theta)).mean()
            total_loss = total_loss + loss

        return total_loss / max(len(unique_fields), 1)


class SmoothnessConstraintLoss(nn.Module):
    """
    Stratigraphic Interface Smoothness Constraint (Eikonal Constraint).

    Equation (15) from the paper:
    L_ξ = Σ (1/|Ω_X|) (||∇S^{Ω_X}(x)|| - 1)

    This constraint enforces that the gradient magnitude of the potential
    field is close to 1, promoting smooth and continuous stratigraphic surfaces.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        model: nn.Module,
        sample_coords: torch.Tensor,
        num_potential_fields: int,
    ) -> torch.Tensor:
        """
        Compute Eikonal smoothness constraint loss.

        Args:
            model: The GeoSAE model (for gradient computation)
            sample_coords: Sampled coordinates from the modeling region of shape (N, 3)
            num_potential_fields: Number of potential fields

        Returns:
            Smoothness constraint loss scalar
        """
        device = sample_coords.device
        total_loss = torch.tensor(0.0, device=device)

        sample_coords_grad = sample_coords.clone().requires_grad_(True)

        for field_idx in range(num_potential_fields):
            # Get predictions for this field
            if hasattr(model, 'sae'):
                pred = model.sae.forward_single(sample_coords_grad, field_idx)
            else:
                pred = model(sample_coords_grad)[:, field_idx:field_idx+1]

            # Compute gradients
            grad_outputs = torch.ones_like(pred)
            gradients = torch.autograd.grad(
                outputs=pred,
                inputs=sample_coords_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # Compute gradient norms
            gradient_norms = torch.norm(gradients, dim=-1)

            # Eikonal loss: ||∇S|| should be close to 1
            eikonal_loss = torch.abs(gradient_norms - 1.0).mean()
            total_loss = total_loss + eikonal_loss

        return total_loss / num_potential_fields


class GeoConstraintLoss(nn.Module):
    """
    Combined Geological Constraint Loss Function.

    Equation (16) from the paper:
    Loss = Loss_I^Var + Loss_I + Loss_O + λ·Loss_ξ

    Where:
    - Loss_I^Var: Stratigraphic consistency (variance) loss
    - Loss_I: Stratigraphic sequence relationship loss
    - Loss_O: Attitude constraint loss
    - Loss_ξ: Smoothness (Eikonal) constraint loss
    - λ: Weight for smoothness constraint
    """

    def __init__(
        self,
        lambda_variance: float = 1.0,
        lambda_above: float = 1.0,
        lambda_below: float = 1.0,
        lambda_overlap: float = 1.0,
        lambda_attitude: float = 1.0,
        lambda_smoothness: float = 0.1,
    ):
        """
        Initialize the combined loss function.

        Args:
            lambda_variance: Weight for variance loss
            lambda_above: Weight for above relationship loss
            lambda_below: Weight for below relationship loss
            lambda_overlap: Weight for overlap relationship loss
            lambda_attitude: Weight for attitude constraint loss
            lambda_smoothness: Weight for smoothness constraint loss
        """
        super().__init__()

        self.lambda_variance = lambda_variance
        self.lambda_above = lambda_above
        self.lambda_below = lambda_below
        self.lambda_overlap = lambda_overlap
        self.lambda_attitude = lambda_attitude
        self.lambda_smoothness = lambda_smoothness

        self.variance_loss = VarianceLoss()
        self.sequence_loss = SequenceConstraintLoss()
        self.attitude_loss = AttitudeConstraintLoss()
        self.smoothness_loss = SmoothnessConstraintLoss()

    def forward(
        self,
        model: nn.Module,
        coords: torch.Tensor,
        surface_labels: torch.Tensor,
        predictions: torch.Tensor,
        sequence_relations: Dict[int, Dict[str, List[int]]],
        num_surfaces: int,
        num_potential_fields: int,
        attitude_coords: Optional[torch.Tensor] = None,
        attitude_vectors: Optional[torch.Tensor] = None,
        attitude_field_indices: Optional[torch.Tensor] = None,
        smoothness_sample_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all geological constraint losses.

        Args:
            model: The GeoSAE model
            coords: Coordinates of shape (N, 3)
            surface_labels: Surface index for each point of shape (N,)
            predictions: Predicted potential field values of shape (N, num_fields)
            sequence_relations: Dictionary defining stratigraphic relationships
            num_surfaces: Number of stratigraphic surfaces
            num_potential_fields: Number of potential fields
            attitude_coords: Coordinates of attitude points (optional)
            attitude_vectors: Measured attitude vectors (optional)
            attitude_field_indices: Field indices for attitude points (optional)
            smoothness_sample_coords: Sampled coordinates for smoothness loss (optional)

        Returns:
            Dictionary with individual losses and total loss
        """
        device = predictions.device

        # Compute mean potential field values for each surface
        mean_values = torch.zeros(num_surfaces, device=device)
        for surface_idx in range(num_surfaces):
            mask = surface_labels == surface_idx
            if mask.sum() > 0:
                # Use first potential field column for simplicity
                # In practice, this should match the surface-to-field mapping
                field_idx = min(surface_idx, predictions.shape[1] - 1)
                mean_values[surface_idx] = predictions[mask, field_idx].mean()

        # 1. Variance loss
        var_loss = self.variance_loss(predictions, surface_labels, num_surfaces)

        # 2. Sequence constraint losses
        above_loss, below_loss, overlap_loss = self.sequence_loss(
            model, coords, surface_labels, predictions, sequence_relations, mean_values
        )

        # 3. Attitude constraint loss
        if attitude_coords is not None and attitude_vectors is not None and attitude_field_indices is not None:
            att_loss = self.attitude_loss(
                model, attitude_coords, attitude_vectors, attitude_field_indices
            )
        else:
            att_loss = torch.tensor(0.0, device=device)

        # 4. Smoothness constraint loss
        if smoothness_sample_coords is not None:
            smooth_loss = self.smoothness_loss(
                model, smoothness_sample_coords, num_potential_fields
            )
        else:
            smooth_loss = torch.tensor(0.0, device=device)

        # Combine losses
        sequence_loss_total = (
            self.lambda_above * above_loss
            + self.lambda_below * below_loss
            + self.lambda_overlap * overlap_loss
        )

        total_loss = (
            self.lambda_variance * var_loss
            + sequence_loss_total
            + self.lambda_attitude * att_loss
            + self.lambda_smoothness * smooth_loss
        )

        return {
            "total": total_loss,
            "variance": var_loss,
            "above": above_loss,
            "below": below_loss,
            "overlap": overlap_loss,
            "sequence_total": sequence_loss_total,
            "attitude": att_loss,
            "smoothness": smooth_loss,
            "mean_values": mean_values,
        }


class PreTrainingLoss(nn.Module):
    """
    Loss function for pre-training with planar geometry.

    Uses L1 and L2 losses as described in Section 2.2.2:
    - L1 loss measures the discrepancy between predicted and actual values
    - L2 loss addresses smoothing errors in the gradient
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pre-training loss.

        Args:
            predictions: Predicted values of shape (N, 1)
            targets: Target values of shape (N, 1)
            model: Model for gradient computation (optional, for L2 smoothness)
            coords: Coordinates for gradient computation (optional)

        Returns:
            Dictionary with losses
        """
        l1 = self.l1_loss(predictions, targets)

        # L2 loss for gradient smoothness (Eikonal-like)
        if model is not None and coords is not None:
            coords_grad = coords.clone().requires_grad_(True)
            pred = model(coords_grad)

            grad_outputs = torch.ones_like(pred)
            gradients = torch.autograd.grad(
                outputs=pred,
                inputs=coords_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient_norms = torch.norm(gradients, dim=-1)
            l2 = self.l2_loss(gradient_norms, torch.ones_like(gradient_norms))
        else:
            l2 = torch.tensor(0.0, device=predictions.device)

        total = self.l1_weight * l1 + self.l2_weight * l2

        return {
            "total": total,
            "l1": l1,
            "l2": l2,
        }
