# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Requirements

### Documentation Workflow
1. **Always read `docs/` first** - Check `docs/01_PROJECT_STATUS.md` for current state before starting work
2. **Update docs progressively** - After completing significant work, update relevant docs
3. **Create new docs when needed** - If a markdown exceeds ~300 lines, create a new numbered file
4. **Commit frequently** - Make atomic commits with clear messages after each logical change

### Commit Workflow
1. After completing a feature/fix, run `git status` and `git diff`
2. Stage relevant files with `git add`
3. Commit with descriptive message following conventional commits
4. Keep commits atomic - one logical change per commit
5. **Do NOT include Claude signature** in commit messages

### Code Cleanliness
1. Remove unused files and imports
2. Keep the codebase organized
3. Don't leave debug code or commented-out blocks
4. Update `.gitignore` as needed

## Documentation Structure

```
docs/
├── 01_PROJECT_STATUS.md    # Current state, what's working, next steps
├── 02_DATA_SOURCES.md      # Training data locations and formats
├── 03_*.md                 # Additional docs as needed
```

**Key Rule**: Always update `docs/01_PROJECT_STATUS.md` when project state changes.

## Project Overview

GeoSAE is a PyTorch implementation of geological constraint-driven 3D stratigraphic modeling using Stacked Autoencoders. Based on Yang et al. 2025 (Applied Sciences).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/ -v
pytest tests/test_models.py::TestAutoEncoder -v  # Single test class

# Train with synthetic data
python examples/train_geosae.py --use-synthetic --output-dir ./output

# Train with CSV data
python examples/train_geosae.py --data-path data.csv --output-dir ./output

# Train with custom parameters
python examples/train_geosae.py --use-synthetic --device cuda --pretrain-iterations 3000 --train-iterations 5000 --lambda-smoothness 0.1 --batch-size 256
```

## Architecture

### Two-Stage Training Pipeline
1. **Pre-training** (`geosae/training/pretrain.py`): Trains single autoencoder on synthetic planar geometry for weight initialization
2. **Main training** (`geosae/training/trainer.py`): Trains StackedAutoEncoder with geological constraint losses

### Neural Network Flow
```
(x,y,z) → StackedAutoEncoder → F potential field values (φ₀...φᶠ)
```
- Each AutoEncoder: Conv1D(3→128→256) → FC(256) → Conv1D(256→128→1)
- Activation: Parametric Softplus σ(x) = (1/β)log(1 + e^(βx))
- Gradients computed via autograd for Eikonal and signed distance constraints

### Loss Functions (`geosae/losses/geological_constraints.py`)
- **VarianceLoss**: Points on same surface have similar potential values
- **SequenceConstraintLoss**: Above/below/overlap stratigraphic relationships using signed distance δ = (φ_x - φ̄_k) / ||∇φ_x||
- **AttitudeConstraintLoss**: Gradient alignment with measured dip/strike
- **SmoothnessConstraintLoss**: Eikonal constraint ||∇S|| ≈ 1

### Data Pipeline
- `StratigraphicDataset`: Handles coordinate normalization to [-1, 1], surface labels
- `BoreholeData`: Container for well/borehole layer information
- Input formats: CSV files, borehole data structures, synthetic data

### Model Hierarchy
- `GeoSAE` (`geosae/models/geosae.py`): Top-level wrapper with normalization and surface code mapping
- `StackedAutoEncoder` (`geosae/models/stacked_autoencoder.py`): F parallel autoencoders for multi-field prediction
- `AutoEncoder` (`geosae/models/autoencoder.py`): Single potential field predictor

## Key Implementation Details

- Coordinate normalization is critical: model expects [-1, 1] range, denormalization needed for visualization
- Pre-training weights transfer to all autoencoders in the stack
- Gradients use `torch.autograd.grad` with `create_graph=True` for higher-order derivatives
- PyVista used for 3D visualization and VTK export (ParaView compatible)

## Configuration

Dataclass-based config in `geosae/utils/config.py` with YAML serialization:
- `ModelConfig`: Network architecture (channels, activation, beta)
- `TrainingConfig`: Learning rate, iterations, loss weights (lambda_*)
- `DataConfig`: Normalization range, grid resolution

## Testing

Tests in `tests/test_models.py` cover:
- Model forward/backward passes and gradient computation
- Loss function edge cases
- Data preprocessing and normalization
- Dataset creation and indexing
