#!/usr/bin/env python3
"""
GPU Detection Demo for MNIST Training

This script demonstrates the GPU detection and configuration
functionality added to the MNIST training pipeline.
"""

import torch
import sys
from pathlib import Path
import psutil

# Add current directory to path
sys.path.append(str(Path(__file__).parent))


def demo_gpu_detection():
    """Demonstrate GPU detection functionality"""
    print("🚀 MNIST GPU Detection Demo")
    print("=" * 50)

    # Step 1: Show PyTorch capabilities
    print("1️⃣ PyTorch Capabilities:")
    print(f"   • PyTorch version: {torch.__version__}")
    print(f"   • CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   • CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"     - Device {i}: {torch.cuda.get_device_name(i)}")

    if hasattr(torch.backends, "mps"):
        print(f"   • MPS available: {torch.backends.mps.is_available()}")

    print()

    # Step 2: Demonstrate device selection
    print("2️⃣ Device Selection Logic:")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "NVIDIA CUDA GPU"
        performance = "🚀 Excellent (CUDA)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "Apple Metal Performance Shaders"
        performance = "⚡ Good for inference (CPU used for training)"
    else:
        device = torch.device("cpu")
        device_type = "CPU"
        performance = "🐌 Slower (CPU only)"

    print(f"   • Selected device: {device}")
    print(f"   • Device type: {device_type}")
    print(f"   • Expected performance: {performance}")
    print()

    # Step 3: Test tensor operations on selected device
    print("3️⃣ Device Performance Test:")

    try:
        # Create test tensors
        if device.type == "mps":
            # MPS requires float32
            x = torch.randn(1000, 1000, dtype=torch.float32, device=device)
            y = torch.randn(1000, 1000, dtype=torch.float32, device=device)
        else:
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)

        # Time matrix multiplication
        import time

        start = time.time()

        # Perform computation
        result = torch.matmul(x, y)

        # Synchronize for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        end = time.time()
        computation_time = (end - start) * 1000
        print(f"   • Matrix multiplication (1000x1000): {computation_time:.2f} ms")

        print(f"   • Result tensor shape: {result.shape}")
        print(f"   • Result tensor device: {result.device}")
        print("   ✅ Device test successful!")

    except Exception as e:
        print(f"   ❌ Device test failed: {e}")

    print()

    # Step 4: Training recommendations
    print("4️⃣ Training Recommendations:")

    # Get device-specific recommendations
    if device.type == "cuda":
        # Get GPU memory info
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Calculate optimal batch size based on GPU memory
        if gpu_memory_gb >= 8:
            recommended_batch_size = "128-256"
            training_time = "20-40 seconds"
        elif gpu_memory_gb >= 4:
            recommended_batch_size = "64-128"
            training_time = "30-60 seconds"
        else:
            recommended_batch_size = "32-64"
            training_time = "45-90 seconds"

        print("   • CUDA GPU detected - optimal for training")
        print(f"   • GPU Memory: {gpu_memory_gb:.1f} GB")
        print(f"   • Recommended batch size: {recommended_batch_size}")
        print(f"   • Expected training time: {training_time} per epoch")

    elif device.type == "mps":
        # MPS shares system memory, estimate based on available RAM
        system_memory_gb = psutil.virtual_memory().total / (1024**3)

        if system_memory_gb >= 16:
            recommended_batch_size = "64-128"
            training_time = "1-2 minutes"
        elif system_memory_gb >= 8:
            recommended_batch_size = "32-64"
            training_time = "1.5-3 minutes"
        else:
            recommended_batch_size = "16-32"
            training_time = "2-4 minutes"

        print("   • Apple MPS detected - available for inference")
        print(f"   • System Memory: {system_memory_gb:.1f} GB")
        print(f"   • Recommended batch size: {recommended_batch_size}")
        print(f"   • Expected training time: {training_time} per epoch")
        print("   • Note: Training will use CPU for stability, MPS for inference")
        print("   • This ensures compatibility with FastAI training pipeline")

    else:
        # CPU performance depends on cores and memory
        cpu_cores = psutil.cpu_count(logical=False)
        system_memory_gb = psutil.virtual_memory().total / (1024**3)

        if cpu_cores >= 8 and system_memory_gb >= 16:
            recommended_batch_size = "32-64"
            training_time = "3-6 minutes"
        elif cpu_cores >= 4 and system_memory_gb >= 8:
            recommended_batch_size = "16-32"
            training_time = "5-10 minutes"
        else:
            recommended_batch_size = "8-16"
            training_time = "8-15 minutes"

        print("   • CPU only - training will be slower")
        print(f"   • CPU Cores: {cpu_cores}, Memory: {system_memory_gb:.1f} GB")
        print(f"   • Recommended batch size: {recommended_batch_size}")
        print(f"   • Expected training time: {training_time} per epoch")
        print("   • Consider cloud GPU services for faster training")

    print()
    print("🎯 To start training with optimized device usage:")
    print("   python train_mnist_model.py --epochs 5 --batch-size 64")
    print()
    print("🔍 The script will automatically:")
    print("   • Use CUDA GPU for training (if available)")
    print("   • Use CPU for training on Apple Silicon (for stability)")
    print("   • Move models to MPS for faster inference (if available)")
    print("   • Fallback to CPU for both training and inference otherwise")


if __name__ == "__main__":
    demo_gpu_detection()
