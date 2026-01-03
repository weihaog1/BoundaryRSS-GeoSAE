# Project Status

**Last Updated**: 2026-01-03

## Current State

GeoSAE is a PyTorch implementation of 3D stratigraphic modeling using Stacked Autoencoders with geological constraints. Based on Yang et al. 2025 paper.

### What's Working
- Complete model architecture (AutoEncoder, StackedAutoEncoder, GeoSAE wrapper)
- All loss functions implemented (Variance, Sequence, Attitude, Smoothness/Eikonal)
- Pre-training pipeline for weight initialization
- Main training loop with geological constraints
- Data preprocessing and normalization utilities
- PyVista visualization utilities
- Unit tests passing
- Training pipeline runs end-to-end with real data
- **NaN divergence fixed** - gradient clipping, numerical stability added

### Training Run History

#### Run 1 (2026-01-03) - NaN Divergence
- **Dataset**: Salinas Valley (2,845 points, 8 surfaces)
- **Pre-training**: Converged successfully (loss: 1.0 → 0.002)
- **Main training**: Started converging (loss: 8.4 → 1.4) but diverged to NaN
- **Root cause**: Smoothness loss explosion, gradient instability

#### Run 2 (2026-01-03) - SUCCESS with NaN Fixes
- **Dataset**: Synthetic data (1,400 points, 7 surfaces)
- **Pre-training**: Completed (loss: 0.74)
- **Main training**: Converged successfully (loss: 0.38 → 0.07)
- **Result**: Model weights 100% healthy (0 NaN/Inf out of 1.38M params)
- **Outputs**: `output/test_fixes/`

### NaN Fixes Applied
1. **Gradient clipping**: `clip_grad_norm_(params, max_norm=1.0)` in `trainer.py`
2. **Numerical stability**: Added `epsilon=1e-8` to gradient norms in `geological_constraints.py`
3. **Loss clamping**: Clamp eikonal loss to prevent extreme values
4. **Reduced lambda_smoothness**: Default changed from 0.1 to 0.01

### Analysis Scripts
- `scripts/analyze_results.py` - Check model health, analyze predictions, generate visualizations
- Output: `output/analysis_*/surface_distributions.png`, `surface_ordering.png`

## Data Status

### Available Training Data
| Dataset | Location | Points | Status |
|---------|----------|--------|--------|
| Salinas Valley (USGS) | `training_data/salinas_valley/` | 2,845 | Ready |
| CVHM2 Central Valley (USGS) | `training_data/cvhm2_central_valley/` | 277,807 | Ready |
| Jun's Surface Geology | `Jun-s-data/` | N/A | **Incompatible** (2D only) |

### Salinas Valley Dataset (Primary)
- **Source**: USGS Digital data for Salinas Valley Geological Framework, California
- **DOI**: https://doi.org/10.5066/P9IL8VBD
- **Boreholes**: 1,385 wells
- **Stratigraphic intervals**: 3,562 records
- **Lithology records**: 85,578 intervals
- **Stratigraphic units**: 8 (Shallow aquifer → Deep aquitard)
- **Format**: CSV with x, y, z coordinates and formation codes

### Jun's Data (Incompatible)
- Contains 2D surface geology polygons only
- No borehole/subsurface data
- No z/depth information
- See `DATA_REQUEST_FOR_JUN.md` for what's actually needed

## Key Files

```
geosae/
├── models/           # Neural network architectures
├── losses/           # Geological constraint loss functions
├── training/         # Pre-training and main training loops
├── data/             # Dataset classes and preprocessing
├── visualization/    # PyVista 3D visualization
└── utils/            # Config and utilities

training_data/
└── salinas_valley/   # USGS borehole data (compatible!)
    ├── WellData_Location.csv          # Borehole x,y,elevation
    ├── WellData_WellStratigraphy.csv  # Formation intervals
    └── WellData_WellLithology.csv     # Lithology at 10ft intervals

docs/                 # Project documentation (you are here)
```

## Training Requirements

**GPU Recommended**: Training on CPU is ~1 hour for 1000 iterations. With NVIDIA GPU, expect 10-20x speedup.

### Quick Start (GPU Instance)

```bash
# Clone/sync the repo to your GPU instance
git pull

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Option 1: Salinas Valley (small, quick test)
python examples/train_geosae.py \
    --data-path training_data/salinas_valley/geosae_input.csv \
    --output-dir output/salinas_valley \
    --device cuda \
    --pretrain-iterations 2000 \
    --train-iterations 5000
# Expected: ~5 minutes on RTX 5070

# Option 2: CVHM2 Central Valley (large, full training)
python examples/train_geosae.py \
    --data-path training_data/cvhm2_central_valley/geosae_input.csv \
    --output-dir output/cvhm2 \
    --device cuda \
    --pretrain-iterations 3000 \
    --train-iterations 10000 \
    --batch-size 512
# Expected: ~15-20 minutes on RTX 5070
```

### Training Output
- `output/salinas_valley/geosae_model.pt` - Trained model
- `output/salinas_valley/training_history.png` - Loss curves
- `output/salinas_valley/geosae_output.vtk` - 3D model for ParaView

## Next Steps

1. ~~Create data preprocessing script for Salinas Valley data~~ **DONE**
2. ~~Run GeoSAE training with real data~~ **DONE**
3. ~~Fix NaN divergence (gradient clipping, numerical stability)~~ **DONE**
4. Re-run training on Salinas Valley with GPU (NaN fixes applied)
5. Generate 3D VTK visualizations (requires PyVista)
6. ~~Clean up incompatible Jun-s-data folder~~ (in progress)
