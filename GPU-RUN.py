#!/usr/bin/env python3
"""
GPU Training Script for GeoSAE.

Run this on your GPU instance to train the model with NaN-fixed parameters.

Usage:
    python GPU-RUN.py                    # Train on Salinas Valley (default)
    python GPU-RUN.py --dataset cvhm2    # Train on CVHM2 Central Valley
    python GPU-RUN.py --synthetic        # Quick test with synthetic data
"""

import subprocess
import sys
import os

# Training configurations
CONFIGS = {
    "salinas": {
        "data_path": "training_data/salinas_valley/geosae_input.csv",
        "output_dir": "output/salinas_valley_v2",
        "pretrain_iterations": 2000,
        "train_iterations": 5000,
        "batch_size": 128,
    },
    "cvhm2": {
        "data_path": "training_data/cvhm2_central_valley/geosae_input.csv",
        "output_dir": "output/cvhm2",
        "pretrain_iterations": 3000,
        "train_iterations": 10000,
        "batch_size": 512,
    },
    "synthetic": {
        "data_path": None,  # Uses --use-synthetic flag
        "output_dir": "output/synthetic_test",
        "pretrain_iterations": 500,
        "train_iterations": 1000,
        "batch_size": 128,
    },
}

# NaN-fix parameters (tested and working)
NAN_FIX_PARAMS = {
    "lambda_smoothness": 0.01,
    "lr": 0.0005,
}


def run_training(dataset="salinas", device="cuda"):
    """Run GeoSAE training with specified configuration."""

    config = CONFIGS.get(dataset)
    if not config:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(CONFIGS.keys())}")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable,
        "examples/train_geosae.py",
        "--output-dir", config["output_dir"],
        "--device", device,
        "--pretrain-iterations", str(config["pretrain_iterations"]),
        "--train-iterations", str(config["train_iterations"]),
        "--batch-size", str(config["batch_size"]),
        "--lambda-smoothness", str(NAN_FIX_PARAMS["lambda_smoothness"]),
        "--lr", str(NAN_FIX_PARAMS["lr"]),
    ]

    # Add data path or synthetic flag
    if config["data_path"]:
        cmd.extend(["--data-path", config["data_path"]])
    else:
        cmd.append("--use-synthetic")

    # Print command for reference
    print("=" * 60)
    print("GeoSAE GPU Training")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Output: {config['output_dir']}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print()

    # Run training
    subprocess.run(cmd)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GeoSAE training on GPU")
    parser.add_argument(
        "--dataset",
        type=str,
        default="salinas",
        choices=["salinas", "cvhm2", "synthetic"],
        help="Dataset to train on (default: salinas)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Shortcut for --dataset synthetic",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu, default: cuda)",
    )

    args = parser.parse_args()

    dataset = "synthetic" if args.synthetic else args.dataset
    run_training(dataset=dataset, device=args.device)


if __name__ == "__main__":
    main()
