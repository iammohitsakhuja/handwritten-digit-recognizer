#!/usr/bin/env bash

# Handwritten Digit Recognizer - Easy Setup and Training Script
# This script makes it easy to set up the environment and train the model

set -e

echo "🔢 Handwritten Digit Recognizer"
echo "======================================"

# Check if we're in a virtual environment or if .venv exists
if [[ -z "$VIRTUAL_ENV" ]] && [[ ! -d ".venv" ]]; then
    echo "⚠️  No virtual environment detected."
    echo "💡 Consider running: source .envrc"
    echo ""
fi

# Show available commands
if [[ $# -eq 0 ]] || [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  train          Quick training with default settings"
    echo "  train-fast     Fast training (3 epochs, no plots)"
    echo "  train-full     Full training (10 epochs with all outputs)"
    echo "  predict        Run prediction demo"
    echo "  demo-gpu       Show GPU detection and capabilities"
    echo "  demo-markdown  Demo new markdown model card generation"
    echo "  test-gpu       Test GPU functionality"
    echo "  notebook       Launch Jupyter notebook tutorial"
    echo "  clean          Clean output and model directories"
    echo "  help           Show detailed help for individual scripts"
    echo ""
    echo "Examples:"
    echo "  $0 train                   # Quick training"
    echo "  $0 train-fast              # Fast training mode"
    echo "  $0 train-full              # Full training with plots"
    echo "  $0 predict                 # Run prediction demo"
    echo "  $0 demo-gpu                # Check GPU support"
    echo "  $0 demo-markdown           # Demo markdown model card generation"
    echo "  $0 test-gpu                # Test GPU functionality"
    echo "  $0 notebook                # Open interactive tutorial"
    echo "  $0 clean                   # Clean output and model files"
    echo "  $0 help train              # Show training script help"
    echo "  $0 help predict            # Show prediction script help"
    echo "  $0 help run                # Show run training script help"
    echo ""
    exit 0
fi

COMMAND="$1"
shift  # Remove the first argument

case "$COMMAND" in
    "train")
        echo "🚀 Starting quick training..."
        python scripts/run_training.py "$@"
        ;;

    "train-fast")
        echo "⚡ Starting fast training (minimal output)..."
        python scripts/run_training.py --fast --quiet "$@"
        ;;

    "train-full")
        echo "📊 Starting full training with all outputs..."
        python scripts/train_mnist_model.py --epochs 10 "$@"
        ;;

    "predict")
        echo "🔮 Running prediction demo..."
        python scripts/predict_digits.py --demo "$@"
        ;;

    "demo-gpu")
        echo "🔧 Running GPU detection demo..."
        python examples/demo_gpu.py "$@"
        ;;

    "demo-markdown")
        echo "📝 Running markdown model card generation demo..."
        python examples/demo_markdown_generation.py "$@"
        ;;

    "test-gpu")
        echo "🧪 Testing GPU functionality..."
        python examples/test_gpu.py "$@"
        ;;

    "notebook")
        echo "📓 Launching Jupyter notebook..."
        jupyter notebook notebooks/mnist_digit_recognition.ipynb
        ;;

    "clean")
        echo "🧹 Cleaning output and model directories..."

        # Clean the out directory
        if [[ -d "out" ]]; then
            echo "  Removing contents of 'out' directory..."
            rm -rf out/*
            echo "  ✅ Cleaned 'out' directory"
        else
            echo "  ℹ️  'out' directory does not exist"
        fi

        # Clean the models directory
        if [[ -d "models" ]]; then
            echo "  Removing contents of 'models' directory..."
            rm -rf models/*
            echo "  ✅ Cleaned 'models' directory"
        else
            echo "  ℹ️  'models' directory does not exist"
        fi

        echo "🎉 Cleanup complete!"
        ;;

    "help")
        if [[ $# -eq 0 ]]; then
            echo "Available help topics:"
            echo "  help train     - Show scripts/train_mnist_model.py options"
            echo "  help predict   - Show scripts/predict_digits.py options"
            echo "  help run       - Show scripts/run_training.py options"
        else
            case "$1" in
                "train")
                    python scripts/train_mnist_model.py --help
                    ;;
                "predict")
                    python scripts/predict_digits.py --help
                    ;;
                "run")
                    python scripts/run_training.py --help
                    ;;
                *)
                    echo "Unknown help topic: $1"
                    echo "Available topics: train, predict, run"
                    exit 1
                    ;;
            esac
        fi
        ;;

    *)
        echo "❌ Unknown command: $COMMAND"
        echo "Run '$0 --help' for usage information"
        exit 1
        ;;
esac
