# Handwritten Digit Recognizer

A FastAI-based handwritten digit recognition system for classifying handwritten digits. The project includes training scripts, inference tools, comprehensive testing, and GPU acceleration support.

## Project Structure

```text
handwritten-digit-recognizer/
├── mnist.sh                       # Main command interface
├── requirements.txt               # Dependencies
├── run_tests.py                   # Test runner
├── test.sh                        # Test automation
├── scripts/                       # Training and inference scripts
│   ├── train_mnist_model.py       # Advanced training with full options
│   ├── predict_digits.py          # Model inference and prediction
│   ├── run_training.py            # Quick start training
│   └── upload_to_huggingface.py   # Upload models to HuggingFace Hub
├── src/mnist_recognizer/          # Core library modules
├── notebooks/                     # Interactive Jupyter notebooks
├── examples/                      # GPU detection and demo scripts
├── tests/                         # Comprehensive test suite
├── models/                        # Saved models (generated)
└── docs/                          # Documentation
    ├── README.md                  # This file
    ├── USAGE.md                   # Command-line usage guide
    └── GPU_SUPPORT.md             # GPU acceleration details
```

## Quick Start

1. **Setup Environment**:

```bash
# Activate virtual environment (if using .envrc)
source .envrc

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Simple Interface

The `mnist.sh` script provides easy access to all functionality:

```bash
# Quick training with default settings
./mnist.sh train

# Fast training (fewer epochs, no plots)
./mnist.sh train-fast

# Full training with detailed outputs
./mnist.sh train-full

# Run prediction demo
./mnist.sh predict

# Clean output and model files
./mnist.sh clean

# Show help
./mnist.sh --help
```

### Python Scripts

#### Training

```bash
# Quick start training
python scripts/run_training.py

# Advanced training with custom parameters
python scripts/train_mnist_model.py --epochs 10 --batch-size 128

# Fast training without plots
python scripts/train_mnist_model.py --epochs 3 --no-plots
```

#### Prediction/Inference

```bash
# Run demo with test images
python scripts/predict_digits.py --demo

# Predict a single image
python scripts/predict_digits.py --image path/to/digit.png --visualize

# Predict multiple images
python scripts/predict_digits.py --images image1.png image2.png
```

For detailed command-line options, see [USAGE.md](USAGE.md) or run:

```bash
python scripts/train_mnist_model.py --help
python scripts/predict_digits.py --help
python scripts/run_training.py --help
```

### Interactive Notebook

Open the Jupyter notebook for an interactive experience:

```bash
jupyter notebook notebooks/mnist_digit_recognition.ipynb
```

The notebook includes:

- Data loading and exploration
- Model architecture setup
- Training with visualization
- Performance evaluation
- Model saving

## Model Details

- **Architecture**: ResNet18 (pre-trained, fine-tuned)
- **Framework**: FastAI + PyTorch
- **Dataset**: MNIST (70,000 images of handwritten digits)
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)

## Features

- **Data Augmentation**: Rotation, scaling, lighting changes
- **Transfer Learning**: Uses pre-trained ResNet18
- **Learning Rate Finding**: Automatic learning rate detection
- **Model Export**: Saves both FastAI and PyTorch formats
- **Comprehensive Evaluation**: Accuracy, confusion matrix, top losses
- **Interactive Visualization**: Sample predictions and error analysis
- **🤗 HuggingFace Hub Integration**: Upload and share your models easily

## HuggingFace Hub Integration

You can now upload your trained models to Hugging Face Hub for sharing and deployment:

```bash
# Upload via command line
python scripts/upload_to_huggingface.py \
    --model-path models/mnist_digit_recognizer.pkl \
    --repo-name my-mnist-model \
    --accuracy 0.9892 \
    --epochs 5 \
    --batch-size 64
```

```python
# Upload from Python
from mnist_recognizer import MNISTTrainer

trainer = MNISTTrainer()
# ... train your model ...

# Upload to HuggingFace
repo_url = trainer.push_to_huggingface(
    repo_name="my-mnist-model",
    private=True
)
```

## Expected Performance

The model achieves respectable accuracy on the MNIST test set with efficient training times on modern hardware.

## File Outputs

After training, you'll find:

- `models/mnist_digit_recognizer.pkl` - Complete FastAI learner
- `models/mnist_digit_recognizer_state_dict.pth` - PyTorch state dict
- Various visualization plots (confusion matrix, predictions, etc.)

## Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
./test.sh all

# Run only unit tests (fast)
./test.sh unit

# Check syntax
./test.sh syntax

# Run tests with coverage
./test.sh coverage
```

You can also use the Python test runner:

```bash
# Run specific test module
python run_tests.py --test test_predict_digits

# Quick unit tests only
python run_tests.py --quick

# List available tests
python run_tests.py --list
```

The test suite includes:

- **Unit tests**: Test individual functions and components
- **Integration tests**: Test script interactions and help commands
- **Syntax checking**: Verify Python syntax correctness
- **Code coverage**: Measure test coverage of the codebase

## GPU Support

The training pipeline automatically detects and uses the best available device:

- **NVIDIA CUDA GPUs** - Good performance for training and inference
- **Apple Metal Performance Shaders (MPS)** - Used on Apple Silicon Macs
- **CPU Fallback** - Works everywhere

### Device Strategy

- **Training**: Uses CUDA if available, otherwise MPS on Apple Silicon, otherwise CPU
- **Inference**: Uses the best available device for good performance

On Apple Silicon Macs:

- Training runs on MPS for good performance
- Inference also uses MPS for efficient predictions
- This approach provides good performance with modern FastAI/PyTorch versions

### Check GPU Status

```bash
# See what GPU support is available
python examples/demo_gpu.py
```

For detailed GPU information, see [GPU_SUPPORT.md](GPU_SUPPORT.md).

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FastAI 2.7+
- See `requirements.txt` for complete list

## License

This project is open source and available under the MIT License.
