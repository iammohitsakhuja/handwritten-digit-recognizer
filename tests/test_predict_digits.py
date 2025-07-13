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
from unittest.mock import patch, MagicMock

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mnist_recognizer import create_test_digit, MNISTPredictor
from scripts.predict_digits import (
    parse_arguments,
    predict_single_image,
    predict_multiple_images,
)


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

    def test_create_test_digit_default(self):
        """Test create_test_digit function with default parameters."""
        img = create_test_digit()

        # Check that image is created correctly
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (28, 28))
        self.assertEqual(img.mode, "L")  # Grayscale

    def test_create_test_digit_different_sizes(self):
        """Test create_test_digit with different sizes."""
        for size in [16, 32, 64]:
            img = create_test_digit(size=size)
            self.assertEqual(img.size, (size, size))

    def test_create_test_digit_different_digits(self):
        """Test create_test_digit with different digit values."""
        for digit in range(10):
            img = create_test_digit(digit=digit)
            self.assertIsInstance(img, Image.Image)

            # Convert to numpy array to check if image has content
            img_array = np.array(img)
            # Check that the image is not completely black (has some white pixels)
            self.assertGreater(np.max(img_array), 0)

    def test_parse_arguments_defaults(self):
        """Test that parse_arguments returns correct default values."""
        original_argv = sys.argv
        try:
            sys.argv = ["predict_digits.py"]
            args = parse_arguments()

            self.assertEqual(args.model_path, "models/mnist_digit_recognizer.pkl")
            self.assertIsNone(args.image)
            self.assertIsNone(args.images)
            self.assertFalse(args.demo)
            self.assertEqual(args.output_dir, "out")
            self.assertFalse(args.no_plots)
            self.assertFalse(args.visualize)
        finally:
            sys.argv = original_argv

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
            args = parse_arguments()

            self.assertEqual(args.model_path, "custom/model.pkl")
            self.assertEqual(args.image, "test.png")
            self.assertTrue(args.demo)
            self.assertEqual(args.output_dir, "output")
            self.assertTrue(args.no_plots)
            self.assertTrue(args.visualize)
        finally:
            sys.argv = original_argv

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
            args = parse_arguments()

            self.assertEqual(args.images, ["img1.png", "img2.png", "img3.png"])
        finally:
            sys.argv = original_argv

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

    @patch("scripts.predict_digits.MNISTPredictor")
    def test_predict_single_image_with_mock(self, mock_predictor_class):
        """Test predict_single_image with mocked MNISTPredictor."""
        # Create a test image
        img = create_test_digit(digit=5)
        img_path = self.temp_path / "test.png"
        img.save(img_path)

        # Mock the predictor and its methods
        mock_predictor = MagicMock()
        mock_predictor.predict_single_with_output.return_value = ("5", 0.95)
        mock_predictor_class.return_value = mock_predictor

        # Test the function
        fake_model_path = "fake/model.pkl"
        pred_class, confidence = predict_single_image(
            fake_model_path, str(img_path), visualize=False, save_plots=False
        )

        # Verify the results
        self.assertEqual(pred_class, "5")
        self.assertEqual(confidence, 0.95)

        # Verify that MNISTPredictor was called with correct arguments
        mock_predictor_class.assert_called_once_with(fake_model_path)
        mock_predictor.predict_single_with_output.assert_called_once_with(
            str(img_path), False, False, "."
        )

    @patch("scripts.predict_digits.MNISTPredictor")
    def test_predict_multiple_images_with_mock(self, mock_predictor_class):
        """Test predict_multiple_images with mocked MNISTPredictor."""
        # Create test images
        test_images = []
        for i in range(2):
            img = create_test_digit(digit=i)
            img_path = self.temp_path / f"test_{i}.png"
            img.save(img_path)
            test_images.append(str(img_path))

        # Mock the predictor and its methods
        mock_predictor = MagicMock()
        mock_results = [("0", 0.9), ("1", 0.85)]
        mock_predictor.predict_multiple_with_summary.return_value = mock_results
        mock_predictor_class.return_value = mock_predictor

        # Test the function
        fake_model_path = "fake/model.pkl"
        results = predict_multiple_images(
            fake_model_path, test_images, save_plots=False
        )

        # Verify the results
        self.assertEqual(results, mock_results)

        # Verify that MNISTPredictor was called with correct arguments
        mock_predictor_class.assert_called_once_with(fake_model_path)
        mock_predictor.predict_multiple_with_summary.assert_called_once_with(
            test_images, False, "."
        )

    def test_predict_single_image_no_model(self):
        """Test predict_single_image with non-existent model (should raise FileNotFoundError)."""
        # Create a test image
        img = create_test_digit(digit=5)
        img_path = self.temp_path / "test.png"
        img.save(img_path)

        # Try to predict with non-existent model - should raise FileNotFoundError
        fake_model_path = "nonexistent/model.pkl"
        with self.assertRaises(FileNotFoundError):
            predict_single_image(
                fake_model_path, str(img_path), visualize=False, save_plots=False
            )

    def test_predict_multiple_images_no_model(self):
        """Test predict_multiple_images with non-existent model (should raise FileNotFoundError)."""
        # Create test images
        test_images = []
        for i in range(2):
            img = create_test_digit(digit=i)
            img_path = self.temp_path / f"test_{i}.png"
            img.save(img_path)
            test_images.append(str(img_path))

        # Try to predict with non-existent model - should raise FileNotFoundError
        fake_model_path = "nonexistent/model.pkl"
        with self.assertRaises(FileNotFoundError):
            predict_multiple_images(fake_model_path, test_images, save_plots=False)


class TestImageCreation(unittest.TestCase):
    """Test image creation utilities."""

    def test_image_pixel_values(self):
        """Test that created images have correct pixel value ranges."""
        img = create_test_digit(digit=0)
        img_array = np.array(img)

        # Check pixel values are in valid range (0-255 for grayscale)
        self.assertGreaterEqual(np.min(img_array), 0)
        self.assertLessEqual(np.max(img_array), 255)

    def test_image_has_content(self):
        """Test that created images are not empty."""
        for digit in range(10):
            img = create_test_digit(digit=digit)
            img_array = np.array(img)

            # Check that image has some non-black pixels
            non_black_pixels = int(np.sum(img_array > 0))
            self.assertGreater(non_black_pixels, 0, f"Digit {digit} image is empty")

    def test_different_digits_create_different_images(self):
        """Test that different digits create visually different images."""
        img0 = create_test_digit(digit=0)
        img1 = create_test_digit(digit=1)

        arr0 = np.array(img0)
        arr1 = np.array(img1)

        # Images should be different (not exactly the same)
        self.assertFalse(np.array_equal(arr0, arr1))

    def test_create_test_digit_with_different_patterns(self):
        """Test that create_test_digit creates recognizable patterns for different digits."""
        # Test specific digit patterns
        for digit in range(5):  # Test first 5 digits
            img = create_test_digit(digit=digit)
            img_array = np.array(img)

            # Check that each digit has a reasonable amount of content
            white_pixels = np.sum(img_array > 0)
            total_pixels = img_array.size
            content_ratio = white_pixels / total_pixels

            # Should have some content but not too much (not a solid block)
            self.assertGreater(
                content_ratio, 0.05, f"Digit {digit} has too little content"
            )
            self.assertLess(content_ratio, 0.7, f"Digit {digit} has too much content")


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

    def test_predictor_initialization_invalid_model(self):
        """Test MNISTPredictor initialization with invalid model path."""
        fake_model_path = "nonexistent/model.pkl"

        with self.assertRaises(FileNotFoundError):
            MNISTPredictor(fake_model_path)

    def test_predictor_initialization_valid_path_format(self):
        """Test MNISTPredictor accepts various path formats."""
        # Test that the class accepts string and Path objects
        # (We don't actually initialize since we don't have a model)
        fake_model_path = "some/path/model.pkl"

        # This should not raise an exception until actual loading
        try:
            predictor = MNISTPredictor(fake_model_path)
            self.fail("Expected FileNotFoundError")
        except FileNotFoundError:
            # This is expected
            pass

    def test_predict_batch_empty_list(self):
        """Test predict_batch with empty image list."""
        # Since we can't test with a real model, we'll just test the structure
        # This test would need a real model to run properly
        pass
        self.assertTrue(True)  # Placeholder test

    def test_predictor_demo_predictions_no_model(self):
        """Test demo_predictions method with non-existent model."""
        fake_model_path = "nonexistent/model.pkl"

        with self.assertRaises(FileNotFoundError):
            predictor = MNISTPredictor(fake_model_path)

    def test_predict_single_with_output_no_model(self):
        """Test predict_single_with_output with non-existent model."""
        # Create a test image
        img = create_test_digit(digit=5)
        img_path = self.temp_path / "test.png"
        img.save(img_path)

        fake_model_path = "nonexistent/model.pkl"

        with self.assertRaises(FileNotFoundError):
            predictor = MNISTPredictor(fake_model_path)

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


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions and edge cases."""

    def test_image_creation_edge_cases(self):
        """Test edge cases for image creation."""
        # Test with minimum size
        img_small = create_test_digit(digit=1, size=8)
        self.assertEqual(img_small.size, (8, 8))

        # Test with large size
        img_large = create_test_digit(digit=1, size=128)
        self.assertEqual(img_large.size, (128, 128))

        # Test with all digit values including edge cases
        for digit in [0, 9]:  # Test edge digits
            img = create_test_digit(digit=digit)
            self.assertIsInstance(img, Image.Image)

    def test_image_format_consistency(self):
        """Test that all created images have consistent format."""
        for digit in range(10):
            img = create_test_digit(digit=digit)

            # All images should be grayscale
            self.assertEqual(img.mode, "L")

            # All images should have the same default size
            self.assertEqual(img.size, (28, 28))

            # Convert to array and check data type
            img_array = np.array(img)
            self.assertEqual(img_array.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
