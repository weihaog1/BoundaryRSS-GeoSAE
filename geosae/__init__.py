"""
GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint

This package implements the GeoSAE method for 3D geological modeling using
stacked autoencoders with geological constraints integrated into the loss function.

Reference:
    Yang, Y.; Zhou, J.; Ruan, M.; Xiao, H.; Hua, W.; Wei, W.
    GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint.
    Appl. Sci. 2025, 15, 1185. https://doi.org/10.3390/app15031185
"""

__version__ = "1.0.0"
__author__ = "BoundaryRSS"

from .models import AutoEncoder, StackedAutoEncoder, GeoSAE
from .losses import GeoConstraintLoss
from .training import Trainer, PreTrainer
from .data import StratigraphicDataset, normalize_coordinates
from .visualization import extract_isosurfaces, visualize_model

__all__ = [
    "AutoEncoder",
    "StackedAutoEncoder",
    "GeoSAE",
    "GeoConstraintLoss",
    "Trainer",
    "PreTrainer",
    "StratigraphicDataset",
    "normalize_coordinates",
    "extract_isosurfaces",
    "visualize_model",
]
