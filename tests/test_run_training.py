#!/usr/bin/env python3
"""
Tests for run_training.py

This module contains unit tests for the quick start training functionality.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_training import parse_arguments


class TestRunTraining(unittest.TestCase):
    """Test cases for run_training.py"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_arguments_defaults(self):
        """Test that parse_arguments returns correct default values."""
        original_argv = sys.argv
        try:
            sys.argv = ["run_training.py"]
            args = parse_arguments()

            self.assertEqual(args.epochs, 3)  # Default epochs for run_training is 3
            self.assertEqual(args.batch_size, 64)
            self.assertEqual(args.model_name, "mnist_digit_recognizer")
            self.assertFalse(args.fast)
            self.assertFalse(args.quiet)
        finally:
            sys.argv = original_argv

    def test_parse_arguments_custom(self):
        """Test that parse_arguments correctly processes custom arguments."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "run_training.py",
                "--epochs",
                "10",
                "--batch-size",
                "128",
                "--model-name",
                "custom_model",
                "--fast",
                "--quiet",
            ]
            args = parse_arguments()

            self.assertEqual(args.epochs, 10)
            self.assertEqual(args.batch_size, 128)
            self.assertEqual(args.model_name, "custom_model")
            self.assertTrue(args.fast)
            self.assertTrue(args.quiet)
        finally:
            sys.argv = original_argv

    def test_parse_arguments_short_flags(self):
        """Test that short flag arguments work correctly."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "run_training.py",
                "-e",
                "8",
                "-b",
                "256",
                "-n",
                "short_flags_model",
                "-f",
                "-q",
            ]
            args = parse_arguments()

            self.assertEqual(args.epochs, 8)
            self.assertEqual(args.batch_size, 256)
            self.assertEqual(args.model_name, "short_flags_model")
            self.assertTrue(args.fast)
            self.assertTrue(args.quiet)
        finally:
            sys.argv = original_argv

    def test_parse_arguments_mixed_flags(self):
        """Test mixing long and short arguments."""
        original_argv = sys.argv
        try:
            sys.argv = [
                "run_training.py",
                "--epochs",
                "3",
                "-b",
                "32",
                "--model-name",
                "mixed_model",
                "-f",
            ]
            args = parse_arguments()

            self.assertEqual(args.epochs, 3)
            self.assertEqual(args.batch_size, 32)
            self.assertEqual(args.model_name, "mixed_model")
            self.assertTrue(args.fast)
            self.assertFalse(args.quiet)
        finally:
            sys.argv = original_argv


class TestArgumentValidation(unittest.TestCase):
    """Test argument validation and edge cases."""

    def test_positive_epochs(self):
        """Test that positive epoch values work."""
        original_argv = sys.argv
        try:
            sys.argv = ["run_training.py", "--epochs", "1"]
            args = parse_arguments()
            self.assertEqual(args.epochs, 1)

            sys.argv = ["run_training.py", "--epochs", "100"]
            args = parse_arguments()
            self.assertEqual(args.epochs, 100)
        finally:
            sys.argv = original_argv

    def test_positive_batch_size(self):
        """Test that positive batch size values work."""
        original_argv = sys.argv
        try:
            sys.argv = ["run_training.py", "--batch-size", "1"]
            args = parse_arguments()
            self.assertEqual(args.batch_size, 1)

            sys.argv = ["run_training.py", "--batch-size", "512"]
            args = parse_arguments()
            self.assertEqual(args.batch_size, 512)
        finally:
            sys.argv = original_argv

    def test_model_name_strings(self):
        """Test various model name strings."""
        test_names = ["simple_model", "model-with-dashes", "model_123", "ModelWithCaps"]

        original_argv = sys.argv
        try:
            for name in test_names:
                sys.argv = ["run_training.py", "--model-name", name]
                args = parse_arguments()
                self.assertEqual(args.model_name, name)
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    unittest.main()
