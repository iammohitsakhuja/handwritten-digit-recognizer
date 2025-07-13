#!/usr/bin/env python3
"""
MNIST Model Inference Script

This script provides command-line interface for making predictions on handwritten digit images
using the trained MNIST model. It uses the MNISTPredictor class from the inference module.
"""

import sys
from pathlib import Path
import argparse

# Add the src directory to the path so we can import the inference module
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mnist_recognizer.inference import MNISTPredictor


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MNIST digit recognition model inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default="models/mnist_digit_recognizer.pkl",
        help="Path to the trained model file",
    )

    parser.add_argument(
        "--image", "-i", type=str, help="Path to an image file to predict"
    )

    parser.add_argument(
        "--images", "-I", nargs="+", help="Paths to multiple image files to predict"
    )

    parser.add_argument(
        "--demo", "-d", action="store_true", help="Run demo with generated test images"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Directory to save output plots",
    )

    parser.add_argument("--no-plots", action="store_true", help="Disable saving plots")

    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Show visualization of predictions",
    )

    return parser.parse_args()


def predict_single_image(
    model_path, image_path, visualize=False, save_plots=True, output_dir="."
):
    """Predict a single image"""
    predictor = MNISTPredictor(model_path)
    return predictor.predict_single_with_output(
        image_path, visualize, save_plots, output_dir
    )


def predict_multiple_images(model_path, image_paths, save_plots=True, output_dir="."):
    """Predict multiple images"""
    predictor = MNISTPredictor(model_path)
    return predictor.predict_multiple_with_summary(image_paths, save_plots, output_dir)


def main():
    """Main function to demonstrate model usage"""
    args = parse_arguments()

    print("MNIST Model Inference")
    print("=" * 40)

    success = False

    if args.image:
        # Predict single image
        print(f"Predicting single image: {args.image}")
        pred_class, confidence = predict_single_image(
            args.model_path,
            args.image,
            visualize=args.visualize,
            save_plots=not args.no_plots,
            output_dir=args.output_dir,
        )
        success = pred_class is not None

    elif args.images:
        # Predict multiple images
        print(f"Predicting {len(args.images)} images...")
        results = predict_multiple_images(
            args.model_path,
            args.images,
            save_plots=not args.no_plots,
            output_dir=args.output_dir,
        )
        success = len(results) > 0

    elif args.demo:
        # Run demo
        try:
            predictor = MNISTPredictor(args.model_path)
            predictor.demo_predictions(
                save_plots=not args.no_plots, output_dir=args.output_dir
            )
            success = True
        except Exception as e:
            print(f"Demo failed: {e}")
            success = False

    else:
        # Default: run demo
        print("No specific action specified. Running demo...")
        try:
            predictor = MNISTPredictor(args.model_path)
            predictor.demo_predictions(
                save_plots=not args.no_plots, output_dir=args.output_dir
            )
            success = True
        except Exception as e:
            print(f"Demo failed: {e}")
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
