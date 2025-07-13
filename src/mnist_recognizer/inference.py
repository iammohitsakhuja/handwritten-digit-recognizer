"""
Inference module for MNIST digit recognition
"""

import torch
import numpy as np
from fastai.vision.all import load_learner, PILImage
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from .utils import create_test_digit, ensure_directory


class MNISTPredictor:
    """Class for loading and using trained MNIST model"""

    def __init__(self, model_path="models/mnist_digit_recognizer.pkl"):
        """
        Initialize the MNIST predictor

        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = Path(model_path)
        self.learn = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        self.learn = load_learner(self.model_path)
        print("✅ Model loaded successfully!")

    def predict_digit(self, image_path_or_array):
        """
        Predict digit from image

        Args:
            image_path_or_array: Path to image file or numpy array/PIL Image

        Returns:
            tuple: (predicted_class, confidence, probabilities)
        """
        if self.learn is None:
            raise ValueError("Model not loaded")

        # Handle different input types
        if isinstance(image_path_or_array, (str, Path)):
            img = PILImage.create(image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            img = PILImage.create(Image.fromarray(image_path_or_array))
        elif isinstance(image_path_or_array, Image.Image):
            img = PILImage.create(image_path_or_array)
        else:
            raise ValueError("Unsupported image format")

        # Make prediction
        pred_class, pred_idx, outputs = self.learn.predict(img)
        confidence = outputs[pred_idx].item()
        probabilities = outputs.softmax(dim=0).numpy()

        return str(pred_class), confidence, probabilities

    def predict_batch(self, image_list):
        """
        Predict digits for a batch of images

        Args:
            image_list: List of image paths or arrays

        Returns:
            list: List of (predicted_class, confidence, probabilities) tuples
        """
        results = []
        for image in image_list:
            try:
                result = self.predict_digit(image)
                results.append(result)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
                results.append((None, 0.0, None))
        return results

    def visualize_prediction(self, image_path_or_array, save_path=None):
        """
        Visualize prediction with confidence scores

        Args:
            image_path_or_array: Input image
            save_path: Optional path to save the visualization
        """
        # Get prediction
        pred_class, confidence, probabilities = self.predict_digit(image_path_or_array)

        # Load and prepare image for display
        if isinstance(image_path_or_array, (str, Path)):
            img = Image.open(image_path_or_array).convert("L")
        elif isinstance(image_path_or_array, np.ndarray):
            img = Image.fromarray(image_path_or_array).convert("L")
        elif isinstance(image_path_or_array, Image.Image):
            img = image_path_or_array.convert("L")
        else:
            raise ValueError(f"Unsupported image type: {type(image_path_or_array)}")

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Show image
        ax1.imshow(img, cmap="gray")
        ax1.set_title(f"Predicted: {pred_class}\nConfidence: {confidence:.3f}")
        ax1.axis("off")

        # Show probability distribution
        classes = list(range(10))
        ax2.bar(classes, probabilities)
        ax2.set_xlabel("Digit Class")
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction Probabilities")
        ax2.set_xticks(classes)

        plt.tight_layout()

        if save_path:
            ensure_directory(Path(save_path).parent)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")

        plt.show()
        return fig

    def demo_predictions(self, save_plots=True, output_dir="out"):
        """Demonstrate model predictions with generated test images"""
        print("Running prediction demo with generated test images...")

        # Create test images for each digit
        test_images = []
        for digit in range(10):
            img = create_test_digit(digit)
            test_images.append((img, digit))

        # Make predictions
        results = []
        for img, true_digit in test_images:
            pred_class, confidence, probabilities = self.predict_digit(img)
            results.append((img, true_digit, pred_class, confidence))

        # Visualize results
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, (img, true_digit, pred_class, confidence) in enumerate(results):
            axes[i].imshow(img, cmap="gray")
            correct = "✅" if str(true_digit) == pred_class else "❌"
            axes[i].set_title(
                f"True: {true_digit}, Pred: {pred_class} {correct}\nConf: {confidence:.3f}"
            )
            axes[i].axis("off")

        plt.suptitle("MNIST Prediction Demo - Generated Test Images", fontsize=16)
        plt.tight_layout()

        if save_plots:
            demo_path = ensure_directory(output_dir) / "prediction_demo.png"
            plt.savefig(demo_path, dpi=150, bbox_inches="tight")
            print(f"Demo visualization saved to: {demo_path}")

        plt.show()

        # Print summary
        correct_predictions = sum(
            1
            for _, true_digit, pred_class, _ in results
            if str(true_digit) == pred_class
        )
        print(f"\nDemo Results: {correct_predictions}/10 correct predictions")

        return results

    def predict_single_with_output(
        self, image_path, visualize=False, save_plots=True, output_dir="out"
    ):
        """
        Predict a single image with detailed console output and optional visualization

        Args:
            image_path: Path to the image file
            visualize: Whether to show visualization
            save_plots: Whether to save visualization plots
            output_dir: Directory to save plots

        Returns:
            tuple: (predicted_class, confidence) or (None, None) if error
        """
        try:
            pred_class, confidence, probs = self.predict_digit(image_path)

            print(f"Image: {image_path}")
            print(f"Predicted digit: {pred_class}")
            print(f"Confidence: {confidence:.4f}")

            if visualize:
                save_path = None
                if save_plots:
                    save_path = (
                        ensure_directory(Path(output_dir))
                        / f"prediction_{Path(image_path).stem}.png"
                    )
                self.visualize_prediction(image_path, save_path=save_path)

            return pred_class, confidence

        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            return None, None

    def predict_multiple_with_summary(
        self, image_paths, save_plots=True, output_dir="out"
    ):
        """
        Predict multiple images with summary visualization

        Args:
            image_paths: List of image file paths
            save_plots: Whether to save summary plot
            output_dir: Directory to save plots

        Returns:
            list: List of (image_path, predicted_class, confidence) tuples
        """
        try:
            results = []

            for image_path in image_paths:
                pred_class, confidence, probs = self.predict_digit(image_path)
                results.append((image_path, pred_class, confidence))
                print(
                    f"{Path(image_path).name}: {pred_class} (confidence: {confidence:.4f})"
                )

            # Create a summary plot
            if len(results) > 1 and save_plots:
                fig, axes = plt.subplots(1, min(len(results), 8), figsize=(15, 3))
                if len(results) == 1:
                    axes = [axes]

                for i, (img_path, pred_class, confidence) in enumerate(results[:8]):
                    img = Image.open(img_path).convert("L")
                    axes[i].imshow(img, cmap="gray")
                    axes[i].set_title(f"{pred_class}\n({confidence:.3f})")
                    axes[i].axis("off")

                plt.suptitle("Batch Predictions")
                plt.tight_layout()

                batch_path = ensure_directory(output_dir) / "batch_predictions.png"
                plt.savefig(batch_path, dpi=150, bbox_inches="tight")
                print(f"Batch predictions saved to: {batch_path}")
                plt.show()

            return results

        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return []
