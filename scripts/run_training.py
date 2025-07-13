#!/usr/bin/env python3
"""
Quick Start Script for Handwritten Digit Recognizer

This script provides a simple way to train and save the MNIST digit recognition model.
"""

import sys
import argparse
import torch
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mnist_recognizer import MNISTTrainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Quick start script for MNIST digit recognition training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--epochs", "-e", type=int, default=3, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size for training"
    )

    parser.add_argument(
        "--model-name",
        "-n",
        type=str,
        default="mnist_digit_recognizer",
        help="Name for the saved model",
    )

    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="Fast training mode (minimal epochs, no plots)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode with minimal output"
    )

    return parser.parse_args()


def main():
    """Main function to run the training pipeline"""
    args = parse_arguments()

    if not args.quiet:
        print("🚀 Starting Handwritten Digit Recognizer Training")
        print("=" * 50)
        if args.fast:
            print("⚡ Fast training mode enabled")

    try:
        seed = 42

        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Initialize trainer
        trainer = MNISTTrainer(batch_size=args.batch_size, seed=seed)

        # Configure training parameters based on mode
        if args.fast:
            # Fast mode: minimal features for quick training
            accuracy, model_path = trainer.full_training_pipeline(
                epochs=args.epochs,
                model_name=args.model_name,
                save_plots=False,  # No plots in fast mode
                output_dir="out",
                models_dir="models",
                test_synthetic=False,  # Skip synthetic testing in fast mode
                run_evaluation=True,  # Keep evaluation for accuracy
                run_predictions=False,  # Skip predictions in fast mode
            )
        else:
            # Full mode: all features enabled
            accuracy, model_path = trainer.full_training_pipeline(
                epochs=args.epochs,
                model_name=args.model_name,
                save_plots=not args.quiet,  # No plots if quiet
                output_dir="out",
                models_dir="models",
                test_synthetic=True,
                run_evaluation=True,
                run_predictions=True,
            )

        if not args.quiet:
            print("\n" + "=" * 50)
            print("✅ Training completed successfully!")
            print(f"🎯 Final accuracy: {accuracy:.4f}")
            print(f"💾 Model saved to: {model_path}")
            print("📁 Check the 'models/' directory for saved models")
            print("🔍 Run 'python predict_digits.py --demo' to test predictions")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(
            "Make sure all dependencies are installed: pip install -r requirements.txt"
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
