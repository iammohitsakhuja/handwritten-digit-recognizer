#!/usr/bin/env python3
"""
GPU Detection Demo for MNIST Training

This script demonstrates the GPU detection and configuration
functionality added to the MNIST training pipeline.
"""

import torch
import sys
from pathlib import Path

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
        performance = "⚡ Very Good (MPS)"
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

    if device.type == "cuda":
        print("   • CUDA GPU detected - optimal for training")
        print("   • Recommended batch size: 64-128")
        print("   • Expected training time: ~30-60 seconds per epoch")
    elif device.type == "mps":
        print("   • Apple MPS detected - good for training")
        print("   • Recommended batch size: 32-64")
        print("   • Expected training time: ~1-2 minutes per epoch")
        print("   • Note: Some operations may fallback to CPU")
    else:
        print("   • CPU only - training will be slower")
        print("   • Recommended batch size: 16-32")
        print("   • Expected training time: ~5-10 minutes per epoch")
        print("   • Consider cloud GPU services for faster training")

    print()
    print("🎯 To start training with GPU support:")
    print("   python train_mnist_model.py --epochs 5 --batch-size 64")
    print()
    print("🔍 The script will automatically detect and use the best available device!")


if __name__ == "__main__":
    demo_gpu_detection()
