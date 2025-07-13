"""
Utility functions for MNIST digit recognition
"""

import torch
from PIL import Image, ImageDraw
from pathlib import Path


def setup_device():
    """Setup and configure the device (GPU/CPU) for training"""
    print("Configuring device...")

    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ CUDA GPU detected: {device_name}")
        print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔧 Using device: {device}")

        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

        # Set default device for FastAI (CUDA works well with this)
        torch.set_default_device(device)

    # Check for MPS (Apple Silicon GPU)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Apple Metal Performance Shaders (MPS) detected")
        print("⚠️  Note: MPS has compatibility issues with FastAI training")
        print("🔧 Using CPU for training, MPS for inference")

        # Use CPU for training due to FastAI compatibility issues
        device = torch.device("cpu")
        print(f"🔧 Training device: {device}")

        # Set default device to CPU for training
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)
        print("💡 After training, models can be moved to MPS for faster inference")

    # Fallback to CPU
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected, using CPU")
        print(f"🔧 Using device: {device}")
        print(
            "💡 For faster training, consider using a machine with CUDA or MPS support"
        )

        # Set default device for CPU
        torch.set_default_device(device)

    print(f"🧠 PyTorch version: {torch.__version__}")
    print()

    return device


def create_test_digit(digit=5, size=28):
    """Create a simple test digit image"""
    img = Image.new("L", (size, size), color=0)  # Black background
    draw = ImageDraw.Draw(img)

    # Simple digit representations
    center_x, center_y = size // 2, size // 2
    margin = size // 6

    if digit == 0:
        # Draw oval for 0
        draw.ellipse(
            [margin, margin, size - margin, size - margin], outline=255, width=2
        )
    elif digit == 1:
        # Draw vertical line for 1
        draw.line([center_x, margin, center_x, size - margin], fill=255, width=3)
    elif digit == 2:
        # Draw crude 2
        draw.line([margin, margin + 2, size - margin, margin + 2], fill=255, width=2)
        draw.line([size - margin, margin, size - margin, center_y], fill=255, width=2)
        draw.line([margin, center_y, size - margin, center_y], fill=255, width=2)
        draw.line([margin, center_y, margin, size - margin], fill=255, width=2)
        draw.line(
            [margin, size - margin, size - margin, size - margin], fill=255, width=2
        )
    elif digit == 3:
        # Draw crude 3
        draw.line([margin, margin, size - margin, margin], fill=255, width=2)
        draw.line([margin, center_y, size - margin, center_y], fill=255, width=2)
        draw.line(
            [margin, size - margin, size - margin, size - margin], fill=255, width=2
        )
        draw.line(
            [size - margin, margin, size - margin, size - margin], fill=255, width=2
        )
    else:
        # Default: draw a square for other digits
        draw.rectangle(
            [margin, margin, size - margin, size - margin], outline=255, width=2
        )
        # Add digit number as text
        try:
            draw.text((center_x - 5, center_y - 8), str(digit), fill=255)
        except:
            pass  # If font is not available, just draw the square

    return img


def ensure_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_plot(fig, path, filename, dpi=150):
    """Save matplotlib figure to file"""
    ensure_directory(path)
    full_path = Path(path) / filename
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"Plot saved to: {full_path}")
    return full_path


def print_training_header():
    """Print a formatted header for training"""
    print("Starting MNIST Handwritten Digit Recognition Training")
    print("=" * 60)


def print_success_message(accuracy, model_path):
    """Print training completion message"""
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {model_path}")


def get_safe_learning_rate(device):
    """Get a safe default learning rate based on device type"""
    if device and str(device) == "mps":
        return 5e-4  # Conservative for Apple Silicon MPS
    elif device and device.type == "cuda":
        return 1e-3  # Standard for NVIDIA CUDA
    else:
        return 1e-3  # CPU default


def get_inference_device():
    """
    Get the best device for inference (can use MPS even if training was on CPU)

    Returns:
        torch.device: Best available device for inference
    """
    # Check for CUDA first (best performance)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Check for MPS (good for inference on Apple Silicon)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # Fallback to CPU
    else:
        return torch.device("cpu")
