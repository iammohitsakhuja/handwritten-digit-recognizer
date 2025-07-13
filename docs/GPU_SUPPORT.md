# GPU Support for MNIST Training

## Overview

The training pipeline automatically detects and uses the best available hardware acceleration:

- **NVIDIA CUDA GPUs**: Used for both training and inference (optimal performance)
- **Apple Metal Performance Shaders (MPS)**: Used on Apple Silicon Macs
- **CPU**: Universal fallback for systems without GPU support

## Automatic Device Detection

The system automatically configures the optimal device:

```python
def setup_device():
    """Setup device for training"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
```

When training starts, you'll see the device information:

```text
Configuring device...
✅ Apple Metal Performance Shaders (MPS) detected
🔧 Using MPS for training and inference
🔧 Using device: mps
```

## Performance Characteristics

### NVIDIA CUDA GPU

- **Training**: Fast training performance
- **Inference**: Good performance
- **Recommended batch size**: 64-128

### Apple Silicon (MPS)

- **Training**: Efficient training performance
- **Inference**: Good performance
- **Recommended batch size**: 32-64

### CPU Only

- **Training**: Standard performance
- **Inference**: Adequate
- **Recommended batch size**: 16-32

## Usage

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

## Compatibility Notes

- **MPS Support**: Requires PyTorch 1.12+ and macOS 12.3+
- **CUDA Support**: Requires NVIDIA GPU with appropriate drivers
- **FastAI Compatibility**: Modern FastAI/PyTorch versions support MPS training

The system prioritizes performance while maintaining broad compatibility across all platforms.
