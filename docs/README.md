# MNIST Handwritten Digit Recognition

This project implements a handwritten digit recognition system using FastAI and the MNIST dataset. The model can classify handwritten digits (0-9) with high accuracy.

## Project Structure

```text
handwritten-digit-recognizer/
├── .envrc                         # Environment setup script
├── .python-version                # Python version specification
├── requirements.txt               # Python dependencies
├── mnist.sh                       # Easy-to-use shell script for common tasks
├── train_mnist_model.py           # Main training script (executable)
├── predict_digits.py              # Model inference script (executable)
├── run_training.py                # Quick start training script (executable)
├── mnist_digit_recognition.ipynb  # Interactive Jupyter notebook
├── models/                        # Saved models directory
├── out/                           # Output artifacts directory (plots, visualizations)
├── tests/                         # Test suite directory
│   ├── test_train_mnist_model.py  # Training module tests
│   ├── test_predict_digits.py     # Prediction module tests
│   ├── test_run_training.py       # Quick start script tests
│   └── test_integration.py        # Integration tests
├── run_tests.py                   # Test runner script (executable)
├── test.sh                        # Test automation script (executable)
├── pytest.ini                     # Pytest configuration
├── USAGE.md                       # Detailed command-line usage guide
└── README.md                      # This file
```

## Setup

1. **Environment Setup**: The project uses a virtual environment managed by the `.envrc` script.

```bash
# If you have direnv installed, it will automatically activate
# Otherwise, manually activate the environment:
source .envrc
```

2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Recommended)

The project includes an easy-to-use shell script for common tasks:

```bash
# Quick training with default settings
./mnist.sh train

# Fast training (3 epochs, no plots)
./mnist.sh train-fast

# Full training (10 epochs with all outputs)
./mnist.sh train-full

# Run prediction demo
./mnist.sh predict

# Get help
./mnist.sh --help
```

### Option 1: Command Line Scripts

All Python scripts are executable and include comprehensive command-line options:

#### Training
```bash
# Quick start training
./run_training.py

# Advanced training with custom parameters
./train_mnist_model.py --epochs 10 --batch-size 128 --model-name my_model

# Fast training without plots
./train_mnist_model.py --epochs 3 --no-plots --no-evaluation
```

#### Prediction/Inference
```bash
# Run demo with generated test images
./predict_digits.py --demo

# Predict a single image
./predict_digits.py --image path/to/digit.png --visualize

# Predict multiple images
./predict_digits.py --images image1.png image2.png image3.png
```

For detailed command-line options, see the [USAGE.md](USAGE.md) file or run:
```bash
./train_mnist_model.py --help
./predict_digits.py --help
./run_training.py --help
```

### Option 2: Interactive Jupyter Notebook

Open and run the `mnist_digit_recognition.ipynb` notebook for an interactive experience:

```bash
jupyter notebook mnist_digit_recognition.ipynb
```

The notebook includes:
- Data loading and exploration
- Model architecture setup
- Training with visualization
- Performance evaluation
- Model saving

### Option 2: Standalone Training Script

Run the complete training pipeline:

```bash
python train_mnist_model.py
```

This will:
- Download the MNIST dataset
- Train a CNN model using ResNet18
- Evaluate performance with confusion matrix
- Save the trained model to `models/`

### Option 3: Model Inference

Use the trained model for predictions:

```bash
python predict_digits.py
```

## Model Details

- **Architecture**: ResNet18 (pre-trained, fine-tuned)
- **Framework**: FastAI + PyTorch
- **Dataset**: MNIST (70,000 images of handwritten digits)
- **Input**: 28x28 grayscale images
- **Output**: 10 classes (digits 0-9)

## Features

- **Data Augmentation**: Rotation, scaling, lighting changes
- **Transfer Learning**: Uses pre-trained ResNet18
- **Learning Rate Finding**: Automatic optimal learning rate detection
- **Model Export**: Saves both FastAI and PyTorch formats
- **Comprehensive Evaluation**: Accuracy, confusion matrix, top losses
- **Interactive Visualization**: Sample predictions and error analysis

## Expected Performance

The model typically achieves:
- **Accuracy**: >99% on the test set
- **Training Time**: ~2-5 minutes (depending on hardware)

## File Outputs

After training, you'll find:
- `models/mnist_digit_recognizer.pkl` - Complete FastAI learner
- `models/mnist_digit_recognizer_state_dict.pth` - PyTorch state dict
- Various visualization plots (confusion matrix, predictions, etc.)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- FastAI 2.7+
- See `requirements.txt` for complete list

## License

This project is open source and available under the MIT License.

### Option 3: Testing

The project includes a comprehensive test suite to ensure code quality and functionality:

```bash
# Run all tests
./test.sh all

# Run only unit tests (fast)
./test.sh unit

# Run integration tests
./test.sh integration

# Check syntax only
./test.sh syntax

# Run tests with coverage
./test.sh coverage

# Show test status
./test.sh status
```

You can also use the Python test runner directly:
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

## GPU Support 🚀

The training pipeline automatically detects and uses the best available device:

- **NVIDIA CUDA GPUs** - Optimal performance for training and inference (30-60 seconds per epoch)
- **Apple Metal Performance Shaders (MPS)** - Used for both training and inference on Apple Silicon Macs
- **CPU Fallback** - Works everywhere for training and inference (5-10 minutes per epoch)

### Device Strategy

- **Training**: Uses CUDA if available, otherwise MPS on Apple Silicon, otherwise CPU
- **Inference**: Uses the best available device (CUDA > MPS > CPU) for optimal performance

On Apple Silicon Macs:
- Training runs on MPS for excellent performance
- Inference also uses MPS for fast predictions
- This approach provides optimal performance with modern FastAI/PyTorch versions

### Check GPU Status

```bash
# See what GPU support is available
python demo_gpu.py
```

### GPU Training

No special commands needed - device detection is automatic:

```bash
# Will automatically use the best training device
./train_mnist_model.py --epochs 5 --batch-size 64

# Quick training
python run_training.py --epochs 3 --fast
```

The system will show device information when training starts:

```
Configuring device...
✅ Apple Metal Performance Shaders (MPS) detected
🔧 Using MPS for training and inference
🔧 Using device: mps
```

For detailed GPU information, see [GPU_SUPPORT.md](GPU_SUPPORT.md).
