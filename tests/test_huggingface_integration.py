#!/usr/bin/env python3
"""
Tests for HuggingFace Hub integration

This module tests the HuggingFace upload functionality.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock
import shutil

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mnist_recognizer.huggingface_hub import (
    HuggingFaceUploader,
    push_to_huggingface,
)


class TestHuggingFaceIntegration(unittest.TestCase):
    """Test cases for HuggingFace Hub integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create a mock model file
        self.mock_model_path = self.temp_path / "test_model.pkl"
        self.mock_model_path.write_text("mock model content")

    def tearDown(self):
        """Clean up after tests"""

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_huggingface_uploader_init_no_token(self):
        """Test HuggingFaceUploader initialization without token"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                HuggingFaceUploader()
            self.assertIn("HuggingFace token required", str(context.exception))

    def test_huggingface_uploader_init_with_token(self):
        """Test HuggingFaceUploader initialization with token"""
        with patch("src.mnist_recognizer.huggingface_hub.HfApi") as mock_api:
            uploader = HuggingFaceUploader(token="test_token")
            self.assertEqual(uploader.token, "test_token")
            mock_api.assert_called_once()

    def test_huggingface_uploader_init_with_env_token(self):
        """Test HuggingFaceUploader initialization with environment token"""
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "env_token"}):
            with patch("src.mnist_recognizer.huggingface_hub.HfApi") as mock_api:
                uploader = HuggingFaceUploader()
                self.assertEqual(uploader.token, "env_token")
                mock_api.assert_called_once()

    def test_create_model_card(self):
        """Test model card creation"""
        with patch("src.mnist_recognizer.huggingface_hub.HfApi"):
            uploader = HuggingFaceUploader(token="test_token")

            model_card = uploader.create_model_card(
                model_name="test-model",
                username="testuser",
                accuracy=0.9876,
                epochs=5,
                batch_size=64,
                learning_rate=0.001,
                additional_info={"test": "value"},
            )

            self.assertIn("test-model", model_card)
            self.assertIn("0.9876", model_card)
            # Updated to match new MarkdownBuilder format (Performance section)
            self.assertIn("**Training Epochs**: 5", model_card)
            self.assertIn("**Batch Size**: 64", model_card)
            self.assertIn("0.001", model_card)
            self.assertIn("test", model_card)
            # Check that username is in the citation URL
            self.assertIn("https://huggingface.co/testuser/test-model", model_card)

    def test_create_config_json(self):
        """Test configuration JSON creation"""
        with patch("src.mnist_recognizer.huggingface_hub.HfApi"):
            uploader = HuggingFaceUploader(token="test_token")

            config = uploader.create_config_json(
                model_name="test-model",
                accuracy=0.9876,
                epochs=5,
                batch_size=64,
                learning_rate=0.001,
                additional_metadata={"custom": "data"},
            )

            self.assertEqual(config["model_name"], "test-model")
            self.assertEqual(config["metrics"]["accuracy"], 0.9876)
            self.assertEqual(config["training"]["epochs"], 5)
            self.assertEqual(config["training"]["batch_size"], 64)
            self.assertEqual(config["training"]["learning_rate"], 0.001)
            self.assertEqual(config["custom"], "data")

    @patch("src.mnist_recognizer.huggingface_hub.upload_folder")
    @patch("src.mnist_recognizer.huggingface_hub.create_repo")
    def test_upload_model_file_not_found(self, mock_create_repo, mock_upload_folder):
        """Test upload_model with non-existent file"""
        with patch("src.mnist_recognizer.huggingface_hub.HfApi"):
            uploader = HuggingFaceUploader(token="test_token")

            with self.assertRaises(FileNotFoundError):
                uploader.upload_model(
                    model_path="nonexistent.pkl",
                    repo_name="test-repo",
                    accuracy=0.9,
                    epochs=5,
                    batch_size=64,
                )

    def test_push_to_huggingface_convenience_function(self):
        """Test the convenience function"""
        with patch(
            "src.mnist_recognizer.huggingface_hub.HuggingFaceUploader"
        ) as mock_uploader_class:
            mock_uploader = MagicMock()
            mock_uploader_class.return_value = mock_uploader
            mock_uploader.upload_model.return_value = "https://huggingface.co/test/repo"

            result = push_to_huggingface(
                model_path=str(self.mock_model_path),
                repo_name="test-repo",
                accuracy=0.9,
                epochs=5,
                batch_size=64,
                token="test_token",
            )

            mock_uploader_class.assert_called_once_with(token="test_token")
            mock_uploader.upload_model.assert_called_once()
            self.assertEqual(result, "https://huggingface.co/test/repo")


if __name__ == "__main__":
    unittest.main()
