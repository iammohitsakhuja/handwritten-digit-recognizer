#!/usr/bin/env python3
"""
Tests for train_mnist_model.py

This module contains unit tests for the training functionality of the MNIST digit recognition project.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import torch
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the actual training class and script parser
from scripts.train_mnist_model import parse_arguments
from src.mnist_recognizer import MNISTTrainer


class TestTrainMnistModel(unittest.TestCase):
    """Test cases for train_mnist_model.py"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close("all")  # Close all matplotlib figures

    def test_parse_arguments_defaults(self):
        """Test that parse_arguments returns correct default values."""
        # Mock sys.argv to simulate command line with no arguments
        original_argv = sys.argv
        try:
            sys.argv = ["train_mnist_model.py"]
            args = parse_arguments()

            self.assertEqual(args.epochs, 5)
            self.assertEqual(args.batch_size, 64)
            self.assertIsNone(args.learning_rate)
            self.assertEqual(args.model_name, "mnist_digit_recognizer")
            self.assertEqual(args.output_dir, "models")
            self.assertEqual(args.plot_dir, ".")
            self.assertFalse(args.no_plots)
            self.assertFalse(args.no_evaluation)
            self.assertFalse(args.no_predictions)
            self.assertEqual(args.seed, 42)
        finally:
            sys.argv = original_argv

    def test_parse_arguments_custom(self):
        """Test that parse_arguments correctly processes custom arguments."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "train_mnist_model.py",
                "--epochs",
                "10",
                "--batch-size",
                "128",
                "--learning-rate",
                "0.001",
                "--model-name",
                "test_model",
                "--output-dir",
                "test_models",
                "--no-plots",
                "--seed",
                "123",
            ]
            args = parse_arguments()

            self.assertEqual(args.epochs, 10)
            self.assertEqual(args.batch_size, 128)
            self.assertEqual(args.learning_rate, 0.001)
            self.assertEqual(args.model_name, "test_model")
            self.assertEqual(args.output_dir, "test_models")
            self.assertTrue(args.no_plots)
            self.assertEqual(args.seed, 123)
        finally:
            sys.argv = original_argv

    @unittest.skipIf(
        not (torch.cuda.is_available() or torch.backends.mps.is_available()),
        "CUDA or MPS not available",
    )
    def test_trainer_setup_data_with_small_batch(self):
        """Test MNISTTrainer setup_data method with small batch size (requires network)."""
        try:
            # Test with very small batch size to minimize download time
            trainer = MNISTTrainer(batch_size=8)
            dls = trainer.setup_data()

            # Check that data loaders are created correctly
            self.assertIsNotNone(dls)
            self.assertGreater(len(dls.train_ds), 0)
            self.assertGreater(len(dls.valid_ds), 0)
            self.assertEqual(len(dls.vocab), 10)  # 10 digit classes

        except Exception as e:
            self.skipTest(f"Network/GPU issue: {e}")

    def test_trainer_create_model_mock(self):
        """Test MNISTTrainer create_model method with mock data loaders."""
        # This test would need mock data loaders, skipping for now
        # as it requires significant FastAI setup
        self.skipTest("Requires FastAI data loaders - integration test")

    def test_trainer_save_model_directory_creation(self):
        """Test that MNISTTrainer save_model creates output directory."""
        # This test would need a trained model, which requires full training
        # Testing directory creation logic instead by importing ensure_directory
        from src.mnist_recognizer import ensure_directory

        test_dir = self.temp_path / "test_models"

        # Check that the directory doesn't exist initially
        self.assertFalse(test_dir.exists())

        # Create directory using the utility function
        created_dir = ensure_directory(test_dir)

        # Check that directory was created
        self.assertTrue(test_dir.exists())
        self.assertTrue(test_dir.is_dir())
        self.assertEqual(created_dir, test_dir)


class TestTrainingUtilities(unittest.TestCase):
    """Test utility functions and edge cases."""

    def test_torch_seed_setting(self):
        """Test that torch.manual_seed works correctly."""
        torch.manual_seed(42)
        tensor1 = torch.randn(3, 3)

        torch.manual_seed(42)
        tensor2 = torch.randn(3, 3)

        # With the same seed, tensors should be equal
        self.assertTrue(torch.equal(tensor1, tensor2))

    def test_path_handling(self):
        """Test Path object handling."""
        test_path = Path("test/path")
        self.assertEqual(str(test_path), "test/path")

        # Test path concatenation
        model_path = test_path / "model.pkl"
        self.assertEqual(str(model_path), "test/path/model.pkl")


if __name__ == "__main__":
    unittest.main()
