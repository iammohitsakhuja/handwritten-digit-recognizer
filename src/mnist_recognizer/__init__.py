"""
MNIST Handwritten Digit Recognizer Package

A FastAI-based implementation for training and inference of handwritten digit recognition models.
"""

__version__ = "1.0.0"
__author__ = "Mohit Sakhuja"

from .training import MNISTTrainer
from .inference import MNISTPredictor
from .utils import (
    setup_device,
    create_test_digit,
    ensure_directory,
    save_plot,
    print_training_header,
    print_success_message,
    get_safe_learning_rate,
    get_inference_device,
    save_text_file,
)

__all__ = [
    "MNISTTrainer",
    "MNISTPredictor",
    "setup_device",
    "create_test_digit",
    "ensure_directory",
    "save_plot",
    "print_training_header",
    "print_success_message",
    "get_safe_learning_rate",
    "get_inference_device",
]
