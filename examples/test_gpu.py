#!/usr/bin/env python3
"""
Test GPU detection functionality
"""

import sys
from pathlib import Path

# Add parent directory and src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))


def test_gpu_detection():
    """Test the GPU detection and setup"""
    print("\n" + "=" * 30)
    print("🔧 GPU Detection Test")
    print("=" * 30)

    try:
        from mnist_recognizer.utils import setup_device

        device = setup_device()
        print(f"✅ Device setup successful: {device}")
        return device
    except Exception as e:
        print(f"❌ Error during device setup: {e}")
        return None


def test_basic_pytorch():
    """Test basic PyTorch functionality"""
    print("\n" + "=" * 30)
    print("🐍 Basic PyTorch Test")
    print("=" * 30)

    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, "mps"):
            print(f"✅ MPS available: {torch.backends.mps.is_available()}")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False


if __name__ == "__main__":
    print("🔬 GPU Detection Test Suite")
    print("=" * 40)

    # Test basic PyTorch
    pytorch_ok = test_basic_pytorch()

    if pytorch_ok:
        # Test our GPU detection function
        device = test_gpu_detection()

        if device:
            print("\n" + "=" * 40)
            print("🎉 Test Results - SUCCESS!")
            print("=" * 40)
            print(f"🔧 Training will use device: {device}")
            print("=" * 40)
        else:
            print("\n" + "=" * 40)
            print("❌ Test Results - GPU SETUP FAILED")
            print("=" * 40)
    else:
        print("\n" + "=" * 40)
        print("❌ Test Results - PYTORCH TEST FAILED")
        print("=" * 40)
