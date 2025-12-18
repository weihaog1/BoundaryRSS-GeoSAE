# GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of GeoSAE, a geological constraint-driven 3D stratigraphic modeling method using Stacked Autoencoders (SAE) with integrated geological constraints in the loss function.

## Reference

This implementation is based on:

> Yang, Y.; Zhou, J.; Ruan, M.; Xiao, H.; Hua, W.; Wei, W. **GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint.** *Appl. Sci.* 2025, 15, 1185. https://doi.org/10.3390/app15031185

## Features

- **Stacked Autoencoder Architecture**: Multi-potential field prediction using stacked autoencoders
- **Geological Constraint Loss Functions**:
  - Stratigraphic consistency constraint (variance loss)
  - Stratigraphic sequence constraints (above/below/overlap relationships)
  - Attitude point constraints (dip/strike alignment)
  - Smoothness constraint (Eikonal constraint)
- **Pre-training with Planar Geometry**: Geometric initialization for faster convergence
- **PyVista Integration**: 3D visualization and VTK export
- **Flexible Data Loading**: Support for borehole data, CSV files, and synthetic data

## Installation

```bash
# Clone the repository
git clone https://github.com/weihaog1/BoundaryRSS-GeoSAE.git
cd BoundaryRSS-GeoSAE

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Using Synthetic Data

```python
from geosae import GeoSAE, Trainer, StratigraphicDataset
from geosae.data.preprocessing import create_default_sequence_relations
from examples.synthetic_data import generate_unconformity_data

# Generate synthetic data
boreholes, code_to_idx, surface_points = generate_unconformity_data(
    num_boreholes=200, seed=42
)
surface_codes = list(code_to_idx.keys())

# Create dataset
dataset = StratigraphicDataset.from_boreholes(boreholes, code_to_idx)

# Create model
model = GeoSAE(
    num_stratigraphic_surfaces=len(surface_codes),
    stratigraphic_codes=surface_codes,
)

# Create trainer and train
trainer = Trainer(model, device="cuda", lambda_smoothness=0.1)
history = trainer.train_full(
    dataset=dataset,
    pretrain_iterations=2000,
    train_iterations=3000,
)

# Save model
model.save("geosae_model.pt")
```

### Loading from CSV

```python
# CSV format: x, y, z, surface_code
dataset = StratigraphicDataset.from_csv(
    "data.csv",
    x_col="x",
    y_col="y",
    z_col="z",
    surface_col="surface_code"
)
```

### Visualization

```python
from geosae.visualization import visualize_model, export_vtk

# Visualize the 3D model
visualize_model(
    model=trained_model,
    bounds=((0, 10000), (0, 10000), (-100, 0)),
    resolution=(50, 50, 50),
)

# Export to VTK for use in ParaView, etc.
export_vtk(
    model=trained_model,
    output_path="output/geosae",
    bounds=((0, 10000), (0, 10000), (-100, 0)),
)
```

## Architecture

### Model Architecture

The GeoSAE model uses a Stacked Autoencoder (SAE) architecture:

```
Input: (x, y, z) coordinates
    │
    ▼
┌─────────────────────────────────────┐
│  Stacked Autoencoder (SAE)          │
│  ┌─────────────────────────────┐    │
│  │  AutoEncoder 1 (Field S₀)   │────┼──► φ₀
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │  AutoEncoder 2 (Field S₁)   │────┼──► φ₁
│  └─────────────────────────────┘    │
│           ...                       │
│  ┌─────────────────────────────┐    │
│  │  AutoEncoder F (Field Sᶠ)   │────┼──► φᶠ
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
    ▼
Output: F potential field values
```

Each AutoEncoder:
- **Encoder**: Conv1D(3→128) → Conv1D(128→256) → FC(256→256)
- **Decoder**: FC(256→256) → Conv1D(256→128) → Conv1D(128→1)
- **Activation**: Parametric Softplus: σ(x) = (1/β) log(1 + e^(βx))

### Loss Function

The total loss combines multiple geological constraints:

```
Loss = Loss_var + Loss_seq + Loss_att + λ·Loss_smooth
```

Where:
- **Loss_var**: Stratigraphic consistency (points on same surface have similar values)
- **Loss_seq**: Sequence constraints (above/below/overlap relationships)
- **Loss_att**: Attitude constraints (gradient alignment with measured dip/strike)
- **Loss_smooth**: Eikonal constraint (||∇S|| ≈ 1 for smooth surfaces)

## Project Structure

```
GeoSAE/
├── geosae/
│   ├── __init__.py
│   ├── models/
│   │   ├── autoencoder.py       # Single autoencoder
│   │   ├── stacked_autoencoder.py  # Stacked autoencoders
│   │   └── geosae.py            # Main GeoSAE model
│   ├── losses/
│   │   └── geological_constraints.py  # All loss functions
│   ├── data/
│   │   ├── preprocessing.py     # Data normalization
│   │   └── dataset.py           # PyTorch datasets
│   ├── training/
│   │   ├── pretrain.py          # Pre-training module
│   │   └── trainer.py           # Main trainer
│   ├── visualization/
│   │   └── pyvista_utils.py     # PyVista visualization
│   └── utils/
│       └── config.py            # Configuration management
├── examples/
│   ├── synthetic_data.py        # Synthetic data generator
│   └── train_geosae.py          # Training example
├── tests/
│   └── test_models.py           # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Configuration

Create a configuration file (`config.yaml`):

```yaml
model:
  encoder_channels: [3, 128, 256]
  encoder_fc_dim: 256
  decoder_channels: [256, 128, 1]
  activation: softplus
  softplus_beta: 1.0
  num_potential_fields: 7

training:
  batch_size: 128
  learning_rate: 0.001
  lr_decay_step: 500
  lr_decay_factor: 0.5
  num_iterations: 5000
  pretrain_iterations: 5000
  lambda_smoothness: 0.1
  device: cuda

data:
  normalize_range: [-1.0, 1.0]
  grid_resolution: [50.0, 50.0, 10.0]
```

## Command Line Training

```bash
# Train with synthetic data
python examples/train_geosae.py --use-synthetic --output-dir ./output

# Train with CSV data
python examples/train_geosae.py --data-path data.csv --output-dir ./output

# With custom parameters
python examples/train_geosae.py \
    --use-synthetic \
    --device cuda \
    --pretrain-iterations 3000 \
    --train-iterations 5000 \
    --lambda-smoothness 0.1 \
    --batch-size 256
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- PyVista >= 0.44.0 (for visualization)
- Pandas >= 2.0.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0 (for plotting)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{yang2025geosae,
  title={GeoSAE: A 3D Stratigraphic Modeling Method Driven by Geological Constraint},
  author={Yang, Yongpeng and Zhou, Jinbo and Ruan, Ming and Xiao, Haiqing and Hua, Weihua and Wei, Wencheng},
  journal={Applied Sciences},
  volume={15},
  number={3},
  pages={1185},
  year={2025},
  publisher={MDPI}
}
```

## Acknowledgments

This implementation is part of the BoundaryRSS geospatial AI agent project for subsurface exploration and geological modeling.
