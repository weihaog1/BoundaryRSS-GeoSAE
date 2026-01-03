# Project Status

**Last Updated**: 2025-01-03

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

### What's In Progress
- Testing with real borehole data (Salinas Valley dataset downloaded)
- Data preprocessing script for Salinas Valley → GeoSAE format conversion

### What's Not Yet Done
- End-to-end validation with real geological data
- Visualization outputs
- Cleanup of Jun's incompatible data

## Data Status

### Available Training Data
| Dataset | Location | Status | Compatible |
|---------|----------|--------|------------|
| Salinas Valley (USGS) | `training_data/salinas_valley/` | Downloaded | Yes |
| Jun's Surface Geology | `Jun-s-data/` | Present | **No** (2D only) |

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

# Run training with GPU
python examples/train_geosae.py \
    --data-path training_data/salinas_valley/geosae_input.csv \
    --output-dir output/salinas_valley \
    --device cuda \
    --pretrain-iterations 2000 \
    --train-iterations 5000

# Expected time: ~5-10 minutes on RTX 5070
```

### Training Output
- `output/salinas_valley/geosae_model.pt` - Trained model
- `output/salinas_valley/training_history.png` - Loss curves
- `output/salinas_valley/geosae_output.vtk` - 3D model for ParaView

## Next Steps

1. ~~Create data preprocessing script for Salinas Valley data~~ **DONE**
2. Run GeoSAE training with real data (needs GPU)
3. Generate 3D stratigraphic model visualizations
4. Clean up incompatible Jun-s-data folder
5. Remove synthetic data examples
