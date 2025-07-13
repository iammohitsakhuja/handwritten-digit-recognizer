#!/usr/bin/env python3
"""
MNIST Handwritten Digit Recognition using FastAI

This script trains a convolutional neural network to recognize handwritten digits
from the MNIST dataset using the FastAI library.
"""

import torch
from pathlib import Path
import argparse
import sys

# Import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from mnist_recognizer.training import MNISTTrainer
from mnist_recognizer.utils import print_training_header, print_success_message


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train a CNN model for MNIST handwritten digit recognition using FastAI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--epochs", "-e", type=int, default=5, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size for training"
    )

    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=None,
        help="Learning rate (if not provided, will use lr_find)",
    )

    # Model and output parameters
    parser.add_argument(
        "--model-name",
        "-n",
        type=str,
        default="mnist_digit_recognizer",
        help="Name for the saved model",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models",
        help="Directory to save the trained model",
    )

    parser.add_argument(
        "--plot-dir", "-p", type=str, default=".", help="Directory to save plot images"
    )

    # Control flags
    parser.add_argument("--no-plots", action="store_true", help="Disable saving plots")

    parser.add_argument(
        "--no-evaluation", action="store_true", help="Skip model evaluation step"
    )

    parser.add_argument(
        "--no-predictions", action="store_true", help="Skip sample predictions step"
    )

    parser.add_argument(
        "--no-synthetic", action="store_true", help="Skip synthetic digit testing"
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_arguments()

    print(f"Arguments: {vars(args)}")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    try:
        # Initialize trainer
        trainer = MNISTTrainer(batch_size=args.batch_size)

        # Use the unified training pipeline with configurable components
        accuracy, model_path = trainer.full_training_pipeline(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_name=args.model_name,
            save_plots=not args.no_plots,
            output_dir=args.plot_dir,
            models_dir=args.output_dir,
            test_synthetic=not args.no_synthetic,
            run_evaluation=not args.no_evaluation,
            run_predictions=not args.no_predictions,
        )

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
