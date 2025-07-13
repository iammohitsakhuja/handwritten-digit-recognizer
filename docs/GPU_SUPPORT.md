# GPU Support for MNIST Training

## Overview

The MNIST training pipeline uses an intelligent device strategy to balance performance and compatibility:

- **NVIDIA CUDA GPUs**: Used for both training and inference (optimal performance)
- **Apple Metal Performance Shaders (MPS)**: Used for both training and inference on Apple Silicon Macs
- **CPU**: Universal fallback for systems without GPU support

## Device Strategy

### Training Phase
- **CUDA GPU**: Used directly for training (best performance)
- **Apple Silicon**: Uses MPS for training and inference (good performance)
- **Other systems**: Falls back to CPU

### Inference Phase
- **CUDA GPU**: Continues using GPU for inference
- **Apple Silicon**: Uses MPS for accelerated inference
- **Other systems**: Uses CPU

This approach ensures optimal performance while maintaining broad compatibility.

## Automatic Device Detection

The system automatically detects and configures the optimal device strategy:

```python
def setup_device():
    """Setup device for training (returns training device)"""
    if torch.cuda.is_available():
        return torch.device('cuda')  # CUDA for both training and inference
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')   # MPS for both training and inference
    else:
        return torch.device('cpu')   # CPU fallback
```

## Device Information Display

When training starts, you'll see the device strategy:

```
Configuring device...
✅ Apple Metal Performance Shaders (MPS) detected
🔧 Using MPS for training and inference
🔧 Using device: mps
```

## Performance Characteristics

### NVIDIA CUDA GPU
- **Training**: 🚀 Excellent (~30-60 seconds per epoch)
- **Inference**: 🚀 Excellent
- **Batch size**: 64-128
- **Strategy**: Direct GPU usage for everything

### Apple Silicon (MPS Available)

- **Training**: ⚡ Very Good (~30 seconds - 1 minute per epoch with MPS)
- **Inference**: ⚡ Very Good (accelerated by MPS)
- **Batch size**: 32-64
- **Strategy**: Direct MPS usage for both training and inference

### CPU Only
- **Training**: 🐌 Slower (~5-10 minutes per epoch)
- **Inference**: 🐌 Slower
- **Batch size**: 16-32
- **Strategy**: CPU for everything

## Usage Examples

### Check Available Devices

```bash
# See what GPU support is available
python examples/demo_gpu.py
```

### Training (Automatic Device Selection)

```bash
# Device selection is automatic - no special flags needed
python scripts/train_mnist_model.py --epochs 5 --batch-size 64
```

## Technical Implementation

### Apple Silicon Specific Handling

1. **Training Phase**: Uses MPS for accelerated training
2. **Inference Phase**: Continues using MPS for fast inference
3. **Data Loading**: Configures appropriate worker processes for MPS

### Memory Management

- **CUDA**: Displays GPU memory usage and clears cache
- **MPS**: Shares system memory efficiently
- **CPU**: Standard system memory management

## Compatibility Notes

- **MPS Support**: Requires PyTorch 1.12+ and macOS 12.3+
- **CUDA Support**: Requires NVIDIA GPU with appropriate drivers
- **FastAI Compatibility**: Updated FastAI/PyTorch versions support MPS training

## Why This Strategy?

1. **Performance**: MPS training on Apple Silicon provides significant speed improvements
2. **Compatibility**: Modern FastAI versions work well with MPS
3. **Simplicity**: Automatic detection means no manual device management
4. **Universality**: Works on all platforms with appropriate fallbacks

The system prioritizes both training and inference performance where possible.
