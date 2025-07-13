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
        device = torch.device("mps")
        print("✅ Apple Metal Performance Shaders (MPS) detected")
        print(f"🔧 Using device: {device}")
        print("⚠️  Note: Ensuring float32 compatibility for MPS")

        # Set default dtype to float32 for MPS compatibility
        torch.set_default_dtype(torch.float32)
        print("🔧 Set default dtype to float32 for MPS compatibility")

        # Set default device for MPS
        torch.set_default_device(device)
        print("� MPS will provide accelerated training on Apple Silicon")

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


def ensure_mps_float32_compatibility(model):
    """
    Ensure all model components are float32 for MPS compatibility

    Args:
        model: PyTorch model to convert

    Returns:
        model: Model with all components converted to float32
    """

    def convert_to_float32(module):
        """Recursively convert all parameters and buffers to float32"""
        for child in module.children():
            convert_to_float32(child)
        for param in module.parameters(recurse=False):
            if param.dtype != torch.float32:
                param.data = param.data.float()
        for buffer in module.buffers(recurse=False):
            if buffer.dtype != torch.float32:
                buffer.data = buffer.data.float()

    # Set default dtype to float32
    torch.set_default_dtype(torch.float32)

    # Convert the model
    model = model.float()
    convert_to_float32(model)

    return model


def patch_fastai_for_mps():
    """
    Patch FastAI to work better with MPS by ensuring all tensors are float32
    """
    import fastai.learner
    import fastai.data.core

    # Patch tensor creation to always use float32 on MPS
    original_tensor = torch.tensor

    def mps_compatible_tensor(*args, **kwargs):
        """Ensure tensors are float32 when using MPS"""
        if torch.get_default_device().type == "mps":
            if "dtype" not in kwargs:
                # Check if the input suggests a float type
                if len(args) > 0 and hasattr(args[0], "__iter__"):
                    try:
                        # If it's numeric data, default to float32
                        float(
                            list(args[0])[0]
                            if hasattr(args[0], "__iter__")
                            else args[0]
                        )
                        kwargs["dtype"] = torch.float32
                    except (ValueError, TypeError, IndexError):
                        pass
        return original_tensor(*args, **kwargs)

    # Temporarily replace torch.tensor
    torch.tensor = mps_compatible_tensor

    return original_tensor


def unpatch_fastai_tensor(original_tensor):
    """Restore original torch.tensor function"""
    torch.tensor = original_tensor
