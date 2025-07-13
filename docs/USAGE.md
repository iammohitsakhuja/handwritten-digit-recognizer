# Handwritten Digit Recognizer - Command Line Usage Guide

This guide describes how to use the command-line interface for the Handwritten Digit Recognizer project.

## Scripts Overview

### Main Interface

- `mnist.sh` - Main script providing easy access to all functionality (training, prediction, cleanup, etc.)

### Python Scripts

All Python scripts are located in the `scripts/` directory and include comprehensive command-line argument parsing:

- `scripts/train_mnist_model.py` - Main training script with full control over training parameters
- `scripts/predict_digits.py` - Model inference script for making predictions
- `scripts/run_training.py` - Quick start script for simplified training

## Usage Examples

### 1. Training Scripts

#### Basic Training (Quick Start)

```bash
# Simple training with default parameters
python scripts/run_training.py

# Fast training mode (fewer epochs, no plots)
python scripts/run_training.py --fast

# Custom epochs and batch size
python scripts/run_training.py --epochs 10 --batch-size 128
```

#### Advanced Training

```bash
# Full training script with all options
python scripts/train_mnist_model.py --help

# Custom training configuration
python scripts/train_mnist_model.py \
    --epochs 10 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --model-name my_mnist_model \
    --output-dir ./models \
    --plot-dir ./out

# Training without plots (faster)
python scripts/train_mnist_model.py --epochs 5 --no-plots

# Training without evaluation
python scripts/train_mnist_model.py --epochs 5 --no-evaluation

# Minimal training (no plots, evaluation, or predictions)
python scripts/train_mnist_model.py \
    --epochs 3 \
    --no-plots \
    --no-evaluation \
    --no-predictions
```

### 2. Prediction Script

#### Demo Mode

```bash
# Run demo with generated test images
python scripts/predict_digits.py --demo

# Demo with custom model and output directory
python scripts/predict_digits.py \
    --demo \
    --model-path ./models/my_model.pkl \
    --output-dir ./out
```

#### Single Image Prediction

```bash
# Predict a single image
python scripts/predict_digits.py --image path/to/digit.png

# Predict with visualization
python scripts/predict_digits.py --image path/to/digit.png --visualize

# Predict without saving plots
python scripts/predict_digits.py --image path/to/digit.png --no-plots
```

#### Batch Image Prediction

```bash
# Predict multiple images
python scripts/predict_digits.py --images image1.png image2.png image3.png

# Batch prediction with custom model
python scripts/predict_digits.py \
    --images *.png \
    --model-path ./models/custom_model.pkl \
    --output-dir ./out
```

## Command Line Arguments

### train_mnist_model.py

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--epochs` | `-e` | 5 | Number of training epochs |
| `--batch-size` | `-b` | 64 | Batch size for training |
| `--learning-rate` | `-lr` | auto | Learning rate (auto-detected if not provided) |
| `--model-name` | `-n` | mnist_digit_recognizer | Name for the saved model |
| `--output-dir` | `-o` | models | Directory to save the trained model |
| `--plot-dir` | `-p` | out | Directory to save plot images |
| `--no-plots` | | | Disable saving plots |
| `--no-evaluation` | | | Skip model evaluation step |
| `--no-predictions` | | | Skip sample predictions step |
| `--seed` | | 42 | Random seed for reproducibility |

### predict_digits.py

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model-path` | `-m` | models/mnist_digit_recognizer.pkl | Path to the trained model file |
| `--image` | `-i` | | Path to an image file to predict |
| `--images` | `-I` | | Paths to multiple image files to predict |
| `--demo` | `-d` | | Run demo with generated test images |
| `--output-dir` | `-o` | out | Directory to save output plots |
| `--no-plots` | | | Disable saving plots |
| `--visualize` | `-v` | | Show visualization of predictions |

### run_training.py

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--epochs` | `-e` | 3 | Number of training epochs |
| `--batch-size` | `-b` | 64 | Batch size for training |
| `--model-name` | `-n` | mnist_digit_recognizer | Name for the saved model |
| `--fast` | `-f` | | Fast training mode (minimal epochs, no plots) |
| `--quiet` | `-q` | | Quiet mode with minimal output |

## Getting Help

Each script provides detailed help information:

```bash
python scripts/train_mnist_model.py --help
python scripts/predict_digits.py --help
python scripts/run_training.py --help
```

## Example Workflows

### Complete Training and Testing Pipeline

```bash
# 1. Train a model with custom parameters
python scripts/train_mnist_model.py \
    --epochs 8 \
    --batch-size 128 \
    --model-name custom_model

# 2. Test the trained model
python scripts/predict_digits.py \
    --model-path ./models/custom_model.pkl \
    --demo

# 3. Predict your own images
python scripts/predict_digits.py \
    --images digit1.png digit2.png
```

### Quick Development Workflow

```bash
# Fast training for testing
python scripts/run_training.py --fast --quiet

# Quick demo
python scripts/predict_digits.py --demo --no-plots
```

## File Outputs

The scripts will create the following files and directories:

- `models/` - Trained model files (.pkl and .pth)
- `out/` - Various plot images (confusion matrix, predictions, etc.)
- Console output with training progress and results

All output locations can be customized using the command-line arguments.

## Cleanup

To clean up generated files and start fresh:

```bash
# Using the main script interface
./mnist.sh clean

# Manual cleanup
rm -rf models/* out/*
```

The clean command removes all contents from the `models/` and `out/` directories while preserving the directory structure.
