"""
Example script for training GeoSAE.

This script demonstrates the complete training pipeline:
1. Generate or load stratigraphic data
2. Pre-train with planar geometry
3. Train with geological constraints
4. Visualize and export results

Usage:
    python train_geosae.py [--use-synthetic] [--device cuda] [--output-dir ./output]
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geosae import GeoSAE, Trainer, StratigraphicDataset
from geosae.data.preprocessing import create_sequence_relations
from geosae.visualization import visualize_model, export_vtk, plot_training_history
from synthetic_data import generate_unconformity_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoSAE model")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to CSV file with stratigraphic data",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--pretrain-iterations",
        type=int,
        default=2000,
        help="Number of pre-training iterations",
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=3000,
        help="Number of main training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lambda-smoothness",
        type=float,
        default=0.01,  # Reduced from 0.1 to prevent NaN divergence
        help="Weight for smoothness constraint",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Softplus activation parameter",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")

    # Load or generate data
    if args.data_path is not None:
        print(f"Loading data from {args.data_path}...")
        dataset = StratigraphicDataset.from_csv(args.data_path)
        surface_codes = dataset.surface_codes
        code_to_idx = {code: i for i, code in enumerate(surface_codes)}

        # Get bounds from data
        coords = dataset.original_coords.numpy()
        bounds = (
            (coords[:, 0].min(), coords[:, 0].max()),
            (coords[:, 1].min(), coords[:, 1].max()),
            (coords[:, 2].min(), coords[:, 2].max()),
        )

    elif args.use_synthetic:
        print("Generating synthetic data...")
        boreholes, code_to_idx, surface_points = generate_unconformity_data(
            num_boreholes=200,
            seed=42,
        )

        surface_codes = list(code_to_idx.keys())
        print(f"Surface codes: {surface_codes}")

        # Create dataset from boreholes
        dataset = StratigraphicDataset.from_boreholes(
            boreholes, code_to_idx, normalize=True
        )

        # Bounds from synthetic data
        bounds = (
            (0, 10000),
            (0, 10000),
            (-100, 0),
        )

    else:
        print("Error: Either --data-path or --use-synthetic must be specified")
        sys.exit(1)

    print(f"Dataset size: {len(dataset)} points")
    print(f"Number of surfaces: {dataset.num_surfaces}")

    # Create sequence relations
    contact_types = {code: "unconformable" for code in surface_codes}
    sequence_relations = create_sequence_relations(surface_codes, contact_types)

    # Create model
    print("\nCreating GeoSAE model...")
    model = GeoSAE(
        num_stratigraphic_surfaces=dataset.num_surfaces,
        stratigraphic_codes=surface_codes,
        beta=args.beta,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        lambda_smoothness=args.lambda_smoothness,
    )

    # Run training
    print("\nStarting training...")
    history = trainer.train_full(
        dataset=dataset,
        pretrain_iterations=args.pretrain_iterations,
        train_iterations=args.train_iterations,
        batch_size=args.batch_size,
        sequence_relations=sequence_relations,
    )

    # Save model
    model_path = os.path.join(args.output_dir, "geosae_model.pt")
    trainer.get_trained_model().save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save training history plot
    history_plot_path = os.path.join(args.output_dir, "training_history.png")
    try:
        plot_training_history(history, save_path=history_plot_path)
    except ImportError:
        print("Matplotlib not available, skipping training history plot")

    # Evaluate model
    print("\nEvaluating model...")
    metrics = trainer.evaluate(dataset)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # Export to VTK
    vtk_path = os.path.join(args.output_dir, "geosae_output")
    print(f"\nExporting to VTK: {vtk_path}")
    try:
        export_vtk(
            model=trainer.get_trained_model(),
            output_path=vtk_path,
            bounds=bounds,
            resolution=(50, 50, 50),
            device=args.device,
        )
    except ImportError:
        print("PyVista not available, skipping VTK export")

    # Visualize
    if not args.no_visualize:
        print("\nVisualizing model...")
        try:
            screenshot_path = os.path.join(args.output_dir, "model_visualization.png")
            visualize_model(
                model=trainer.get_trained_model(),
                bounds=bounds,
                resolution=(50, 50, 50),
                device=args.device,
                screenshot_path=screenshot_path,
            )
        except ImportError:
            print("PyVista not available, skipping visualization")
        except Exception as e:
            print(f"Visualization error: {e}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
