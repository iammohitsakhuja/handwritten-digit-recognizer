#!/usr/bin/env python3
"""
Demo of the new MarkdownBuilder-based model card generation
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_recognizer import save_text_file


# Simple standalone test without imports
def create_demo_markdown():
    """Create a demo markdown using the new structured approach"""

    # Simulate the frontmatter
    frontmatter_yaml = """---
license: mit
tags:
- computer-vision
- image-classification
- fastai
- pytorch
- mnist
- digit-recognition
datasets:
- mnist
metrics:
- accuracy
model-index:
- name: demo-mnist-model
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      type: mnist
      name: MNIST
    metrics:
    - type: accuracy
      value: 0.9845
      name: Accuracy
---

# demo-mnist-model

## Model Description

This is a convolutional neural network trained to recognize handwritten digits (0-9) using the MNIST dataset. The model is built using FastAI and PyTorch.

## Model Details

- **Model Type**: Convolutional Neural Network (ResNet18 architecture)
- **Task**: Image Classification (Digit Recognition)
- **Dataset**: MNIST (70,000 images of handwritten digits)
- **Framework**: FastAI + PyTorch
- **License**: MIT

## Performance

- **Validation Accuracy**: 0.9845
- **Training Epochs**: 10
- **Batch Size**: 64
- **Learning Rate**: 0.001

## Usage

### Installation

```bash
pip install fastai torch pillow
```

### Loading the Model

```python
from fastai.vision.all import load_learner
import torch

# Load the model
learn = load_learner('path_to_model.pkl')

# Make predictions
pred_class, pred_idx, outputs = learn.predict(image)
```

### Inference Example

```python
from fastai.vision.all import load_learner, PILImage
from PIL import Image

# Load model
model = load_learner('demo-mnist-model.pkl')

# Load and predict an image
img = PILImage.create('path_to_digit_image.png')
pred_class, pred_idx, outputs = model.predict(img)
confidence = outputs[pred_idx].item()

print(f"Predicted digit: {pred_class}")
print(f"Confidence: {confidence:.4f}")
```

## Training Details

The model was trained using FastAI's transfer learning approach:

1. **Architecture**: ResNet18 pre-trained on ImageNet
2. **Fine-tuning**: One-cycle training policy
3. **Data Augmentation**: Random rotations, scaling, and lighting changes
4. **Optimization**: AdamW optimizer with learning rate scheduling

## Files

- `demo-mnist-model.pkl`: Complete FastAI learner (recommended for inference)
- `demo-mnist-model_state_dict.pth`: PyTorch state dictionary
- `config.json`: Model configuration and metadata

## Citation

If you use this model, please cite:

```bibtex
@misc{demo_mnist_model,
    title={MNIST Handwritten Digit Recognition using FastAI},
    author={Your Name},
    year={2025},
    publisher={Hugging Face},
    url={https://huggingface.co/your-username/demo-mnist-model}
}
```

## Additional Information

- **training_time**: 5 minutes
- **model_size**: 43.2 MB
- **inference_time**: < 100ms
- **hardware**: CPU training

"""

    return frontmatter_yaml


if __name__ == "__main__":
    print("🎯 Demo: New MarkdownBuilder-based Model Card Generation")
    print("=" * 60)

    demo_content = create_demo_markdown()

    print("✅ Generated structured model card successfully!")
    print(f"📄 Total length: {len(demo_content)} characters")
    print(f"📝 Number of lines: {len(demo_content.split(chr(10)))}")

    # Save demo output to the out directory using utils function
    output_file = save_text_file(demo_content, "out", "demo_model_card.md")
    print(f"💾 Saved demo model card to '{output_file}'")
