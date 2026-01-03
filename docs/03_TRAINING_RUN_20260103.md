# Training Run Analysis - 2026-01-03

## Summary

First end-to-end training run of GeoSAE on real geological data (Salinas Valley USGS dataset).

**Result**: Pre-training succeeded, main training partially converged then diverged to NaN.

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Salinas Valley (USGS) |
| Data points | 2,845 |
| Stratigraphic surfaces | 8 |
| Device | CUDA (GPU) |
| Pre-train iterations | 2,000 |
| Train iterations | 5,000 |
| Learning rate | 0.001 |
| Batch size | Default |

## Training Phases

### Phase 1: Pre-training (Planar Geometry)

**Status**: ✅ SUCCESS

Pre-training learns basic spatial relationships using synthetic planar geometry before fine-tuning on real data.

| Metric | Start | End | Reduction |
|--------|-------|-----|-----------|
| Total Loss | 1.005 | 0.002 | 99.8% |
| L1 Loss | 0.906 | 0.002 | 99.8% |
| L2 Loss | 0.998 | 0.0005 | 99.9% |

**Learning Rate Schedule**:
- Iterations 0-500: lr=0.001
- Iterations 500-1000: lr=0.0005
- Iterations 1000-1500: lr=0.00025
- Iterations 1500-2000: lr=0.000125

**Observations**:
- Convergence was smooth and stable
- Loss dropped rapidly in first 100 iterations (1.0 → 0.05)
- Final loss of 0.002 indicates good fit to planar geometry
- ~70 iterations/second throughput on GPU

### Phase 2: Main Training (Geological Constraints)

**Status**: ⚠️ PARTIAL - Diverged to NaN

Main training applies four geological constraint losses to the real borehole data.

#### Loss Components

| Loss Type | Initial | At iter 100 | At iter 200 | Purpose |
|-----------|---------|-------------|-------------|---------|
| Total | 8.41 | 1.47 | 1.40 | Combined |
| Variance | 0.87 | 0.61 | 0.39 | Same-surface points should have similar potential |
| Sequence | 7.54 | 0.79 | 0.93 | Enforce above/below relationships |
| Smoothness | 0.007 | 0.78 | 0.79 | Eikonal constraint ∥∇φ∥ ≈ 1 |

**Key Observations**:

1. **Initial convergence**: Loss dropped from 8.4 → 1.4 in first 200 iterations (83% reduction)
2. **Sequence loss**: Dropped dramatically from 7.5 → 0.8 (stratigraphic ordering learned quickly)
3. **Variance loss**: Steadily decreased 0.87 → 0.39 (surfaces becoming coherent)
4. **Smoothness loss**: Increased from 0.007 → 0.78 (concerning trend)
5. **Divergence**: Training diverged to NaN sometime after iteration 200

#### Root Cause Analysis

The NaN divergence is likely caused by:

1. **Smoothness loss explosion**: The Eikonal constraint (∥∇φ∥ = 1) increased 100x from 0.007 to 0.78, indicating gradient magnitudes were not stabilizing

2. **Gradient instability**: The sequence constraint uses signed distance δ = (φ - φ̄) / ∥∇φ∥. If ∥∇φ∥ → 0, this causes division by zero or very large gradients

3. **Loss weight imbalance**: Default weights may not be appropriate for this dataset size/complexity

## Output Files

| File | Size | Description |
|------|------|-------------|
| `geosae_model.pt` | 6.4 MB | Model checkpoint (contains NaN weights) |
| `geosae_output_grid.vtk` | 4.0 MB | 3D grid (50×50×50) - invalid due to NaN |
| `training_history.png` | 122 KB | Loss curve visualization |

## Recommended Fixes

### 1. Lower Learning Rate
```bash
python examples/train_geosae.py --lr 0.0005 ...
```

### 2. Reduce Smoothness Weight
The smoothness constraint is destabilizing training:
```bash
python examples/train_geosae.py --lambda-smoothness 0.01 ...
```

### 3. Add Gradient Clipping
Edit `geosae/training/trainer.py` to add:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Early Stopping
Monitor for NaN and stop training when loss starts increasing significantly.

### 5. Numerical Stability
Add small epsilon to gradient norm calculations:
```python
grad_norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Pre-training speed | ~70 it/s |
| Main training speed | ~6 it/s |
| Total training time | ~14 minutes |
| GPU memory | Not measured |

## Next Steps

1. Implement gradient clipping in trainer
2. Re-run with lower smoothness weight (0.01 instead of default)
3. Add NaN detection and early stopping
4. Consider normalizing loss components to similar scales
