#!/usr/bin/env python3
"""
Tests for utils.py

This module contains unit tests for the utility functions of the MNIST digit recognition project, designed to test individual functions without triggering device setup.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import torch
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_recognizer import create_test_digit, save_plot


class TestSetupDevice(unittest.TestCase):
    """Test cases for setup_device function (mocked to avoid device setup)"""

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    @patch("torch.cuda.get_device_properties")
    @patch("builtins.print")
    def test_setup_device_cuda_available(
        self,
        mock_print,
        mock_get_props,
        mock_get_name,
        mock_cuda_available,
    ):
        """Test setup_device when CUDA is available"""
        from mnist_recognizer import setup_device

        # Mock CUDA availability
        mock_cuda_available.return_value = True
        mock_get_name.return_value = "NVIDIA GeForce RTX 3080"

        # Mock device properties
        mock_props = MagicMock()
        mock_props.total_memory = 10 * 1024**3  # 10 GB
        mock_get_props.return_value = mock_props

        device = setup_device()

        # Verify device is set to cuda
        self.assertEqual(device.type, "cuda")

    @patch("torch.cuda.is_available")
    @patch.object(torch.backends, "mps", create=True)
    @patch("builtins.print")
    def test_setup_device_mps_available(
        self, mock_print, mock_mps, mock_cuda_available
    ):
        """Test setup_device when MPS is available but CUDA is not"""
        from mnist_recognizer import setup_device

        # Mock availability
        mock_cuda_available.return_value = False
        mock_mps.is_available.return_value = True

        device = setup_device()

        # Verify device is set to mps
        self.assertEqual(device.type, "mps")

    @patch("torch.cuda.is_available")
    @patch.object(torch.backends, "mps", create=False)
    @patch("builtins.print")
    def test_setup_device_cpu_fallback(self, mock_print, mock_mps, mock_cuda_available):
        """Test setup_device falls back to CPU when no GPU is available"""
        from mnist_recognizer import setup_device

        # Mock no GPU availability
        mock_cuda_available.return_value = False
        mock_mps.is_available.return_value = False

        # Mock no MPS (by not patching it, it won't exist)
        device = setup_device()

        # Verify device is set to cpu
        self.assertEqual(device.type, "cpu")


class TestCreateTestDigit(unittest.TestCase):
    """Test cases for create_test_digit function"""

    def test_import_and_create_digit(self):
        """Test that we can import and create a test digit"""
        img = create_test_digit()

        # Verify image properties
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (28, 28))
        self.assertEqual(img.mode, "L")  # Grayscale

    def test_create_test_digit_different_sizes(self):
        """Test create_test_digit with different sizes"""
        for size in [16, 32, 64, 128]:
            with self.subTest(size=size):
                img = create_test_digit(size=size)
                self.assertEqual(img.size, (size, size))
                self.assertEqual(img.mode, "L")

    def test_create_test_digit_all_digits(self):
        """Test create_test_digit for all digit values 0-9"""
        for digit in range(10):
            with self.subTest(digit=digit):
                img = create_test_digit(digit=digit)

                # Verify image is created
                self.assertIsInstance(img, Image.Image)
                self.assertEqual(img.size, (28, 28))

                # Convert to numpy array and verify it has content
                img_array = np.array(img)

                # Check that image has some non-black pixels
                self.assertGreater(
                    np.max(img_array),
                    0,
                    f"Digit {digit} image should have white pixels",
                )
                self.assertLessEqual(
                    np.max(img_array),
                    255,
                    f"Digit {digit} pixel values should not exceed 255",
                )

    def test_create_test_digit_pixel_values(self):
        """Test that created images have correct pixel value ranges"""
        img = create_test_digit(digit=0)
        img_array = np.array(img)

        # Check pixel values are in valid range (0-255 for grayscale)
        self.assertGreaterEqual(np.min(img_array), 0)
        self.assertLessEqual(np.max(img_array), 255)

        # Check that we have both black (0) and white (255) pixels
        unique_values = np.unique(img_array)
        self.assertIn(0, unique_values, "Should have black background pixels")
        self.assertIn(255, unique_values, "Should have white foreground pixels")

    def test_create_test_digit_different_digits_are_different(self):
        """Test that different digits create visually different images"""
        img0 = create_test_digit(digit=0)
        img1 = create_test_digit(digit=1)

        arr0 = np.array(img0)
        arr1 = np.array(img1)

        # Images should be different
        self.assertFalse(
            np.array_equal(arr0, arr1),
            "Different digits should create different images",
        )

    def test_create_test_digit_large_digit_value(self):
        """Test create_test_digit with digit values > 9"""
        # Should fall back to default square with text
        img = create_test_digit(digit=15)

        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (28, 28))

        # Should have some content
        img_array = np.array(img)
        self.assertGreater(np.max(img_array), 0)

    def test_create_test_digit_edge_size_values(self):
        """Test create_test_digit with edge case sizes"""
        # Very small size
        img_small = create_test_digit(size=8)
        self.assertEqual(img_small.size, (8, 8))

        # Large size
        img_large = create_test_digit(size=256)
        self.assertEqual(img_large.size, (256, 256))


class TestEnsureDirectory(unittest.TestCase):
    """Test cases for ensure_directory function"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_directory_creates_new_directory(self):
        """Test that ensure_directory creates a new directory"""
        from mnist_recognizer import ensure_directory

        new_dir = self.temp_path / "new_directory"

        # Verify directory doesn't exist initially
        self.assertFalse(new_dir.exists())

        # Create directory
        result = ensure_directory(new_dir)

        # Verify directory was created
        self.assertTrue(new_dir.exists())
        self.assertTrue(new_dir.is_dir())

        # Verify return value is a Path object
        self.assertIsInstance(result, Path)
        self.assertEqual(result, new_dir)

    def test_ensure_directory_creates_nested_directories(self):
        """Test that ensure_directory creates nested directories"""
        from mnist_recognizer import ensure_directory

        nested_dir = self.temp_path / "level1" / "level2" / "level3"

        # Verify nested path doesn't exist
        self.assertFalse(nested_dir.exists())

        # Create nested directories
        result = ensure_directory(nested_dir)

        # Verify all levels were created
        self.assertTrue(nested_dir.exists())
        self.assertTrue(nested_dir.is_dir())
        self.assertTrue((self.temp_path / "level1").exists())
        self.assertTrue((self.temp_path / "level1" / "level2").exists())

    def test_ensure_directory_existing_directory(self):
        """Test that ensure_directory works with existing directories"""
        from mnist_recognizer import ensure_directory

        existing_dir = self.temp_path / "existing"
        existing_dir.mkdir()

        # Verify it exists
        self.assertTrue(existing_dir.exists())

        # Call ensure_directory on existing directory
        result = ensure_directory(existing_dir)

        # Should still exist and return the path
        self.assertTrue(existing_dir.exists())
        self.assertEqual(result, existing_dir)

    def test_ensure_directory_with_string_path(self):
        """Test that ensure_directory works with string paths"""
        from mnist_recognizer import ensure_directory

        new_dir_str = str(self.temp_path / "string_path")

        # Call with string path
        result = ensure_directory(new_dir_str)

        # Verify directory was created
        self.assertTrue(Path(new_dir_str).exists())
        self.assertIsInstance(result, Path)


class TestSavePlot(unittest.TestCase):
    """Test cases for save_plot function"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close("all")  # Close all matplotlib figures

    def test_save_plot_creates_file(self):
        """Test that save_plot creates a plot file"""
        from mnist_recognizer import save_plot

        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")

        # Save the plot
        with patch("builtins.print"):  # Suppress print output
            result_path = save_plot(fig, str(self.temp_path), "test_plot.png")

        # Verify file was created
        expected_path = self.temp_path / "test_plot.png"
        self.assertTrue(expected_path.exists())
        self.assertTrue(expected_path.is_file())

        # Verify return value
        self.assertEqual(result_path, expected_path)

    def test_save_plot_creates_directory(self):
        """Test that save_plot creates output directory if it doesn't exist"""
        from mnist_recognizer import save_plot

        new_dir = self.temp_path / "plots"

        # Verify directory doesn't exist
        self.assertFalse(new_dir.exists())

        # Create and save plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with patch("builtins.print"):
            save_plot(fig, str(new_dir), "test.png")

        # Verify directory and file were created
        self.assertTrue(new_dir.exists())
        self.assertTrue((new_dir / "test.png").exists())

    def test_save_plot_different_dpi(self):
        """Test that save_plot accepts different DPI values"""
        from mnist_recognizer import save_plot

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with patch("builtins.print"):
            result_path = save_plot(fig, str(self.temp_path), "dpi_test.png", dpi=300)

        # File should be created regardless of DPI
        self.assertTrue(result_path.exists())

    @patch("builtins.print")
    def test_save_plot_prints_message(self, mock_print):
        """Test that save_plot prints the expected message"""
        from mnist_recognizer import save_plot

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        result_path = save_plot(fig, str(self.temp_path), "message_test.png")

        # Verify print was called with the expected message
        mock_print.assert_called_once()
        args = mock_print.call_args[0]
        self.assertTrue(args[0].startswith("Plot saved to:"))
        self.assertIn("message_test.png", args[0])

    def test_save_plot_with_path_object(self):
        """Test save_plot with Path object instead of string"""
        from mnist_recognizer import save_plot

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with patch("builtins.print"):
            result_path = save_plot(fig, self.temp_path, "path_object.png")

        expected_path = self.temp_path / "path_object.png"
        self.assertTrue(expected_path.exists())
        self.assertEqual(result_path, expected_path)


class TestPrintFunctions(unittest.TestCase):
    """Test cases for print utility functions"""

    @patch("builtins.print")
    def test_print_training_header(self, mock_print):
        """Test print_training_header function"""
        from mnist_recognizer import print_training_header

        print_training_header()

        # Verify print was called twice (header + separator)
        self.assertEqual(mock_print.call_count, 2)

        # Check the content of the calls
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertEqual(
            calls[0], "Starting MNIST Handwritten Digit Recognition Training"
        )
        self.assertEqual(calls[1], "=" * 60)

    @patch("builtins.print")
    def test_print_success_message(self, mock_print):
        """Test print_success_message function"""
        from mnist_recognizer import print_success_message

        accuracy = 0.9876
        model_path = Path("/path/to/model.pkl")

        print_success_message(accuracy, model_path)

        # Verify print was called 4 times (newline + separator, success message, accuracy, model path)
        self.assertEqual(mock_print.call_count, 4)

        # Check the content of the calls
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertEqual(calls[0], "\n" + "=" * 60)
        self.assertEqual(calls[1], "Training completed successfully!")
        self.assertEqual(calls[2], f"Final Accuracy: {accuracy:.4f}")
        self.assertEqual(calls[3], f"Model saved to: {model_path}")

    @patch("builtins.print")
    def test_print_success_message_string_path(self, mock_print):
        """Test print_success_message with string path"""
        from mnist_recognizer import print_success_message

        accuracy = 0.5
        model_path = "models/test_model.pkl"

        print_success_message(accuracy, model_path)

        # Should work with string paths too
        calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertEqual(calls[3], f"Model saved to: {model_path}")

    @patch("builtins.print")
    def test_print_success_message_accuracy_formatting(self, mock_print):
        """Test that accuracy is formatted to 4 decimal places"""
        from mnist_recognizer import print_success_message

        test_cases = [
            (0.123456789, "0.1235"),
            (0.9, "0.9000"),
            (1.0, "1.0000"),
            (0.0, "0.0000"),
        ]

        for accuracy, expected_format in test_cases:
            with self.subTest(accuracy=accuracy):
                mock_print.reset_mock()
                print_success_message(accuracy, "test.pkl")

                # Check accuracy formatting in the third call
                calls = [call[0][0] for call in mock_print.call_args_list]
                accuracy_call = calls[2]  # Third call is the accuracy
                self.assertIn(f"Final Accuracy: {expected_format}", accuracy_call)


class TestUtilsIntegration(unittest.TestCase):
    """Integration tests for utils functions working together"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close("all")

    def test_create_digit_and_save_plot_integration(self):
        """Test creating a digit image and saving it as a plot"""
        # Create a test digit
        img = create_test_digit(digit=7)

        # Create a plot showing the digit
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.set_title("Test Digit 7")
        ax.axis("off")

        # Save the plot using save_plot function
        with patch("builtins.print"):
            saved_path = save_plot(fig, str(self.temp_path), "digit_7.png")

        # Verify everything worked
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.name, "digit_7.png")

    def test_workflow_simulation(self):
        """Test a complete workflow using multiple utils functions"""
        from mnist_recognizer import (
            print_training_header,
            ensure_directory,
            create_test_digit,
            save_plot,
            print_success_message,
        )

        # Start with training header
        with patch("builtins.print"):
            print_training_header()

        # Create output directory
        output_dir = ensure_directory(self.temp_path / "results")
        self.assertTrue(output_dir.exists())

        # Create some test images
        images = []
        for i in range(3):
            img = create_test_digit(digit=i)
            images.append(img)

        # Create a plot with multiple images
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Digit {i}")
            ax.axis("off")

        # Save the plot
        with patch("builtins.print"):
            saved_path = save_plot(fig, str(output_dir), "test_digits.png")

        # Verify everything worked
        self.assertTrue(saved_path.exists())

        # Simulate successful completion
        with patch("builtins.print") as mock_print:
            print_success_message(0.9876, saved_path)

        # Verify success message was printed
        self.assertEqual(mock_print.call_count, 4)


if __name__ == "__main__":
    unittest.main()
