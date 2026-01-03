"""
Analyze GeoSAE training results.

This script:
1. Checks model health (NaN values in weights)
2. Loads trained model and test data
3. Runs inference
4. Analyzes predictions by surface
5. Creates visualizations
6. Evaluates stratigraphic ordering

Usage:
    python scripts/analyze_results.py --model output/salinas_valley/geosae_model.pt \
        --data training_data/salinas_valley/geosae_input.csv
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geosae import GeoSAE, StratigraphicDataset


def check_model_health(model: GeoSAE) -> dict:
    """Check for NaN/Inf values in model weights."""
    health = {
        "total_params": 0,
        "nan_params": 0,
        "inf_params": 0,
        "layers_with_issues": [],
        "is_healthy": True,
    }

    for name, param in model.named_parameters():
        health["total_params"] += param.numel()
        nan_count = torch.isnan(param).sum().item()
        inf_count = torch.isinf(param).sum().item()

        if nan_count > 0 or inf_count > 0:
            health["nan_params"] += nan_count
            health["inf_params"] += inf_count
            health["layers_with_issues"].append({
                "name": name,
                "shape": list(param.shape),
                "nan_count": nan_count,
                "inf_count": inf_count,
            })
            health["is_healthy"] = False

    return health


def analyze_predictions(model: GeoSAE, dataset: StratigraphicDataset, device: str) -> dict:
    """Analyze model predictions on dataset."""
    model.eval()

    coords = dataset.coords.to(device)
    labels = dataset.surface_labels.to(device)

    with torch.no_grad():
        predictions = model(coords, normalize=False)

    # Check for NaN in predictions
    nan_predictions = torch.isnan(predictions).sum().item()

    # Analyze per-surface
    surface_stats = {}
    num_surfaces = dataset.num_surfaces
    num_fields = predictions.shape[1]

    for surface_idx in range(num_surfaces):
        mask = labels == surface_idx
        if mask.sum() == 0:
            continue

        # Use the corresponding field (or the last one if fewer fields than surfaces)
        field_idx = min(surface_idx, num_fields - 1)
        surface_preds = predictions[mask, field_idx]

        surface_stats[surface_idx] = {
            "count": mask.sum().item(),
            "mean": surface_preds.mean().item(),
            "std": surface_preds.std().item(),
            "min": surface_preds.min().item(),
            "max": surface_preds.max().item(),
            "has_nan": torch.isnan(surface_preds).any().item(),
        }

    return {
        "nan_predictions": nan_predictions,
        "total_predictions": predictions.numel(),
        "surface_stats": surface_stats,
        "predictions": predictions.cpu(),
        "labels": labels.cpu(),
    }


def evaluate_stratigraphic_ordering(surface_stats: dict) -> dict:
    """Check if surfaces follow proper stratigraphic ordering."""
    # Surfaces should have increasing mean potential field values (by index)
    ordered_means = []
    violations = []

    sorted_indices = sorted(surface_stats.keys())

    for i, idx in enumerate(sorted_indices):
        mean_val = surface_stats[idx]["mean"]
        ordered_means.append((idx, mean_val))

        if i > 0:
            prev_idx, prev_mean = ordered_means[i - 1]
            if mean_val < prev_mean:
                violations.append({
                    "surface_a": prev_idx,
                    "surface_b": idx,
                    "mean_a": prev_mean,
                    "mean_b": mean_val,
                })

    return {
        "ordered_means": ordered_means,
        "violations": violations,
        "is_properly_ordered": len(violations) == 0,
    }


def plot_surface_distributions(predictions: torch.Tensor, labels: torch.Tensor,
                                num_surfaces: int, save_path: str = None):
    """Plot distribution of predictions for each surface."""
    # Check for all-NaN predictions
    if torch.isnan(predictions).all():
        print("   Skipping distribution plot - all predictions are NaN")
        return

    fig, axes = plt.subplots(2, (num_surfaces + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    num_fields = predictions.shape[1]

    for surface_idx in range(num_surfaces):
        mask = labels == surface_idx
        if mask.sum() == 0:
            continue

        ax = axes[surface_idx]
        field_idx = min(surface_idx, num_fields - 1)
        surface_preds = predictions[mask, field_idx].numpy()

        # Skip if all NaN
        if np.isnan(surface_preds).all():
            ax.text(0.5, 0.5, 'All NaN', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Surface {surface_idx} (NaN)')
            continue

        ax.hist(surface_preds[~np.isnan(surface_preds)], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.nanmean(surface_preds), color='red', linestyle='--',
                   label=f'Mean: {np.nanmean(surface_preds):.2f}')
        ax.set_title(f'Surface {surface_idx}')
        ax.set_xlabel('Potential Field Value')
        ax.set_ylabel('Count')
        ax.legend()

    # Hide unused axes
    for i in range(num_surfaces, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved distribution plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_ordering_analysis(ordered_means: list, save_path: str = None):
    """Plot surface ordering by mean potential field value."""
    # Check for all-NaN values
    means = [x[1] for x in ordered_means]
    if all(np.isnan(m) for m in means):
        print("   Skipping ordering plot - all means are NaN")
        return

    indices = [x[0] for x in ordered_means]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Replace NaN with 0 for plotting
    plot_means = [0 if np.isnan(m) else m for m in means]
    bars = ax.bar(indices, plot_means, color='steelblue', edgecolor='black')

    # Highlight ordering violations
    for i in range(1, len(means)):
        if means[i] < means[i-1]:
            bars[i].set_color('red')
            bars[i-1].set_color('red')

    ax.set_xlabel('Surface Index')
    ax.set_ylabel('Mean Potential Field Value')
    ax.set_title('Stratigraphic Surface Ordering\n(Red = Ordering Violations)')
    ax.set_xticks(indices)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved ordering plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze GeoSAE training results")
    parser.add_argument(
        "--model",
        type=str,
        default="output/salinas_valley/geosae_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="training_data/salinas_valley/geosae_input.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/analysis",
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GeoSAE Training Results Analysis")
    print("=" * 60)

    # Load model
    print(f"\n1. Loading model from {args.model}...")
    try:
        model = GeoSAE.load(args.model, device=args.device)
        print(f"   Model loaded successfully")
        print(f"   - Stratigraphic surfaces: {model.num_stratigraphic_surfaces}")
        print(f"   - Potential fields: {model.num_potential_fields}")
        print(f"   - Surface codes: {model.stratigraphic_codes}")
    except Exception as e:
        print(f"   ERROR loading model: {e}")
        return

    # Check model health
    print("\n2. Checking model health...")
    health = check_model_health(model)
    print(f"   - Total parameters: {health['total_params']:,}")
    print(f"   - NaN parameters: {health['nan_params']:,}")
    print(f"   - Inf parameters: {health['inf_params']:,}")

    if health["is_healthy"]:
        print("   ✓ Model weights are healthy (no NaN/Inf)")
    else:
        print("   ✗ Model has corrupted weights!")
        print("   Affected layers:")
        for layer in health["layers_with_issues"]:
            print(f"     - {layer['name']}: {layer['nan_count']} NaN, {layer['inf_count']} Inf")

    # Load data
    print(f"\n3. Loading data from {args.data}...")
    try:
        dataset = StratigraphicDataset.from_csv(args.data)
        print(f"   - Data points: {len(dataset)}")
        print(f"   - Surfaces: {dataset.num_surfaces}")
    except Exception as e:
        print(f"   ERROR loading data: {e}")
        return

    # Analyze predictions
    print("\n4. Running inference and analyzing predictions...")
    analysis = analyze_predictions(model, dataset, args.device)

    print(f"   - NaN predictions: {analysis['nan_predictions']} / {analysis['total_predictions']}")

    print("\n   Per-surface statistics:")
    print("   " + "-" * 70)
    print(f"   {'Surface':<10} {'Count':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("   " + "-" * 70)

    for surface_idx, stats in sorted(analysis["surface_stats"].items()):
        status = "NaN!" if stats["has_nan"] else ""
        print(f"   {surface_idx:<10} {stats['count']:<10} {stats['mean']:<12.4f} "
              f"{stats['std']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f} {status}")

    # Evaluate stratigraphic ordering
    print("\n5. Evaluating stratigraphic ordering...")
    ordering = evaluate_stratigraphic_ordering(analysis["surface_stats"])

    if ordering["is_properly_ordered"]:
        print("   ✓ Surfaces are properly ordered by potential field value")
    else:
        print("   ✗ Ordering violations detected:")
        for v in ordering["violations"]:
            print(f"     - Surface {v['surface_a']} (mean={v['mean_a']:.4f}) > "
                  f"Surface {v['surface_b']} (mean={v['mean_b']:.4f})")

    # Generate plots
    print("\n6. Generating visualization plots...")

    dist_path = os.path.join(args.output_dir, "surface_distributions.png")
    plot_surface_distributions(
        analysis["predictions"],
        analysis["labels"],
        dataset.num_surfaces,
        save_path=dist_path
    )

    order_path = os.path.join(args.output_dir, "surface_ordering.png")
    plot_ordering_analysis(ordering["ordered_means"], save_path=order_path)

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Model health:          {'✓ Healthy' if health['is_healthy'] else '✗ Corrupted (NaN/Inf)'}")
    print(f"Predictions valid:     {'✓ Yes' if analysis['nan_predictions'] == 0 else '✗ Contains NaN'}")
    print(f"Stratigraphic order:   {'✓ Correct' if ordering['is_properly_ordered'] else '✗ Violations'}")
    print(f"\nOutput files:")
    print(f"  - {dist_path}")
    print(f"  - {order_path}")
    print("=" * 60)

    # Return summary for programmatic use
    return {
        "health": health,
        "analysis": analysis,
        "ordering": ordering,
    }


if __name__ == "__main__":
    main()
