#!/usr/bin/env python3
"""
Tests for predict_digits.py and inference module

This module contains unit tests for the prediction functionality of the MNIST digit recognition project.
Tests cover both the CLI script (predict_digits.py) and the underlying inference module.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add the scripts and src directories to the path so we can import the modules
scripts_path = Path(__file__).parent.parent / "scripts"
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(scripts_path))
sys.path.insert(0, str(src_path))

# Import from scripts - handle import errors gracefully
try:
    import predict_digits

    PREDICT_DIGITS_AVAILABLE = True
except ImportError:
    PREDICT_DIGITS_AVAILABLE = False
    print("Warning: predict_digits module not available for testing")

# Import from inference module
try:
    from mnist_recognizer import create_test_digit, MNISTPredictor

    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    print("Warning: mnist_recognizer modules not available for testing")


class TestPredictDigits(unittest.TestCase):
    """Test cases for predict_digits.py"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close("all")  # Close all matplotlib figures

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_create_test_digit_default(self):
        """Test create_test_digit function with default parameters."""
        img = create_test_digit()

        # Check that image is created correctly
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (28, 28))
        self.assertEqual(img.mode, "L")  # Grayscale

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_create_test_digit_different_sizes(self):
        """Test create_test_digit with different sizes."""
        for size in [16, 32, 64]:
            img = create_test_digit(size=size)
            self.assertEqual(img.size, (size, size))

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_create_test_digit_different_digits(self):
        """Test create_test_digit with different digit values."""
        for digit in range(10):
            img = create_test_digit(digit=digit)
            self.assertIsInstance(img, Image.Image)

            # Convert to numpy array to check if image has content
            img_array = np.array(img)
            # Check that the image is not completely black (has some white pixels)
            self.assertGreater(np.max(img_array), 0)

    @unittest.skipUnless(
        PREDICT_DIGITS_AVAILABLE, "predict_digits module not available"
    )
    @unittest.skipUnless(
        PREDICT_DIGITS_AVAILABLE, "predict_digits module not available"
    )
    def test_parse_arguments_defaults(self):
        """Test that parse_arguments returns correct default values."""
        original_argv = sys.argv
        try:
            sys.argv = ["predict_digits.py"]
            args = predict_digits.parse_arguments()

            self.assertEqual(args.model_path, "models/mnist_digit_recognizer.pkl")
            self.assertIsNone(args.image)
            self.assertIsNone(args.images)
            self.assertFalse(args.demo)
            self.assertEqual(args.output_dir, ".")
            self.assertFalse(args.no_plots)
            self.assertFalse(args.visualize)
        finally:
            sys.argv = original_argv

    @unittest.skipUnless(
        PREDICT_DIGITS_AVAILABLE, "predict_digits module not available"
    )
    def test_parse_arguments_custom(self):
        """Test that parse_arguments correctly processes custom arguments."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "predict_digits.py",
                "--model-path",
                "custom/model.pkl",
                "--image",
                "test.png",
                "--demo",
                "--output-dir",
                "output",
                "--no-plots",
                "--visualize",
            ]
            args = predict_digits.parse_arguments()

            self.assertEqual(args.model_path, "custom/model.pkl")
            self.assertEqual(args.image, "test.png")
            self.assertTrue(args.demo)
            self.assertEqual(args.output_dir, "output")
            self.assertTrue(args.no_plots)
            self.assertTrue(args.visualize)
        finally:
            sys.argv = original_argv

    @unittest.skipUnless(
        PREDICT_DIGITS_AVAILABLE, "predict_digits module not available"
    )
    def test_parse_arguments_multiple_images(self):
        """Test parsing multiple image arguments."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "predict_digits.py",
                "--images",
                "img1.png",
                "img2.png",
                "img3.png",
            ]
            args = predict_digits.parse_arguments()

            self.assertEqual(args.images, ["img1.png", "img2.png", "img3.png"])
        finally:
            sys.argv = original_argv

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_create_test_images_for_prediction(self):
        """Test creating test images that can be used for prediction testing."""
        # Create test images and save them
        test_images = []
        for i in range(3):
            img = create_test_digit(digit=i)
            img_path = self.temp_path / f"test_digit_{i}.png"
            img.save(img_path)
            test_images.append(img_path)

            # Verify image was saved correctly
            self.assertTrue(img_path.exists())

            # Load and verify image
            loaded_img = Image.open(img_path)
            self.assertEqual(loaded_img.size, (28, 28))

    @unittest.skipUnless(
        PREDICT_DIGITS_AVAILABLE and INFERENCE_AVAILABLE,
        "Required modules not available",
    )
    def test_predict_single_image_no_model(self):
        """Test predict_single_image with non-existent model."""
        # Create a test image
        img = create_test_digit(digit=5)
        img_path = self.temp_path / "test.png"
        img.save(img_path)

        # Try to predict with non-existent model
        fake_model_path = "nonexistent/model.pkl"
        pred_class, confidence = predict_digits.predict_single_image(
            fake_model_path, str(img_path), visualize=False, save_plots=False
        )

        # Should return None for both values when model doesn't exist
        self.assertIsNone(pred_class)
        self.assertIsNone(confidence)

    @unittest.skipUnless(
        PREDICT_DIGITS_AVAILABLE and INFERENCE_AVAILABLE,
        "Required modules not available",
    )
    def test_predict_multiple_images_no_model(self):
        """Test predict_multiple_images with non-existent model."""
        # Create test images
        test_images = []
        for i in range(2):
            img = create_test_digit(digit=i)
            img_path = self.temp_path / f"test_{i}.png"
            img.save(img_path)
            test_images.append(str(img_path))

        # Try to predict with non-existent model
        fake_model_path = "nonexistent/model.pkl"
        results = predict_digits.predict_multiple_images(
            fake_model_path, test_images, save_plots=False
        )

        # Should return empty list when model doesn't exist
        self.assertEqual(results, [])


class TestImageCreation(unittest.TestCase):
    """Test image creation utilities."""

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_image_pixel_values(self):
        """Test that created images have correct pixel value ranges."""
        img = create_test_digit(digit=0)
        img_array = np.array(img)

        # Check pixel values are in valid range (0-255 for grayscale)
        self.assertGreaterEqual(np.min(img_array), 0)
        self.assertLessEqual(np.max(img_array), 255)

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_image_has_content(self):
        """Test that created images are not empty."""
        for digit in range(10):
            img = create_test_digit(digit=digit)
            img_array = np.array(img)

            # Check that image has some non-black pixels
            non_black_pixels = int(np.sum(img_array > 0))
            self.assertGreater(non_black_pixels, 0, f"Digit {digit} image is empty")

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_different_digits_create_different_images(self):
        """Test that different digits create visually different images."""
        img0 = create_test_digit(digit=0)
        img1 = create_test_digit(digit=1)

        arr0 = np.array(img0)
        arr1 = np.array(img1)

        # Images should be different (not exactly the same)
        self.assertFalse(np.array_equal(arr0, arr1))


class TestMNISTPredictor(unittest.TestCase):
    """Test cases for MNISTPredictor class from inference module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close("all")  # Close all matplotlib figures

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_predictor_initialization_invalid_model(self):
        """Test MNISTPredictor initialization with invalid model path."""
        fake_model_path = "nonexistent/model.pkl"

        with self.assertRaises(FileNotFoundError):
            MNISTPredictor(fake_model_path)

    def test_predict_batch_empty_list(self):
        """Test predict_batch with empty image list."""
        # Since we can't test with a real model, we'll just test the structure
        # This test would need a real model to run properly
        pass

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_predictor_demo_predictions_no_model(self):
        """Test demo_predictions method with non-existent model."""
        fake_model_path = "nonexistent/model.pkl"

        with self.assertRaises(FileNotFoundError):
            predictor = MNISTPredictor(fake_model_path)

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_predict_single_with_output_no_model(self):
        """Test predict_single_with_output with non-existent model."""
        # Create a test image
        img = create_test_digit(digit=5)
        img_path = self.temp_path / "test.png"
        img.save(img_path)

        fake_model_path = "nonexistent/model.pkl"

        with self.assertRaises(FileNotFoundError):
            predictor = MNISTPredictor(fake_model_path)

    @unittest.skipUnless(INFERENCE_AVAILABLE, "mnist_recognizer modules not available")
    def test_predict_multiple_with_summary_no_model(self):
        """Test predict_multiple_with_summary with non-existent model."""
        # Create test images
        test_images = []
        for i in range(2):
            img = create_test_digit(digit=i)
            img_path = self.temp_path / f"test_{i}.png"
            img.save(img_path)
            test_images.append(str(img_path))

        fake_model_path = "nonexistent/model.pkl"

        with self.assertRaises(FileNotFoundError):
            predictor = MNISTPredictor(fake_model_path)


if __name__ == "__main__":
    unittest.main()
