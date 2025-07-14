"""
Hugging Face Hub integration for MNIST digit recognition models.

This module provides functionality to upload trained models to Hugging Face Hub,
including model files, configuration, and documentation.
"""

import os
import json
import tempfile
import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.errors import RepositoryNotFoundError
from .utils import ensure_directory
from . import __author__ as PACKAGE_AUTHOR


class MarkdownBuilder:
    """Helper class to build markdown content programmatically"""

    def __init__(self):
        self.content = []

    def add_frontmatter(self, metadata: Dict[str, Any]) -> "MarkdownBuilder":
        """Add YAML frontmatter to the markdown"""
        self.content.append("---")
        self.content.append(yaml.dump(metadata, default_flow_style=False).strip())
        self.content.append("---")
        self.content.append("")
        return self

    def add_heading(self, text: str, level: int = 1) -> "MarkdownBuilder":
        """Add a heading"""
        self.content.append("#" * level + " " + text)
        self.content.append("")
        return self

    def add_paragraph(self, text: str) -> "MarkdownBuilder":
        """Add a paragraph"""
        self.content.append(text)
        self.content.append("")
        return self

    def add_list_item(self, text: str, bullet: str = "-") -> "MarkdownBuilder":
        """Add a list item"""
        self.content.append(f"{bullet} {text}")
        return self

    def add_blank_line(self) -> "MarkdownBuilder":
        """Add a blank line"""
        self.content.append("")
        return self

    def add_code_block(self, code: str, language: str = "") -> "MarkdownBuilder":
        """Add a code block"""
        self.content.append(f"```{language}")
        self.content.append(code)
        self.content.append("```")
        self.content.append("")
        return self

    def build(self) -> str:
        """Build the final markdown string"""
        return "\n".join(self.content)


class HuggingFaceUploader:
    """Class to handle uploading MNIST models to Hugging Face Hub"""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the HuggingFace uploader

        Args:
            token: HuggingFace token. If None, will try to get from environment
        """
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError(
                "HuggingFace token required. Set HUGGINGFACE_TOKEN environment variable or pass token parameter."
            )

        self.api = HfApi()

    def create_model_card(
        self,
        model_name: str,
        username: str,
        accuracy: float,
        epochs: int,
        batch_size: int,
        learning_rate: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        author_name: Optional[str] = None,
    ) -> str:
        """
        Create a model card (README.md) for the model

        Args:
            model_name: Name of the model
            username: HuggingFace username
            accuracy: Final validation accuracy
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate used
            additional_info: Additional information to include
            author_name: Author name for citation (if not provided, uses package default)

        Returns:
            Model card content as string
        """
        additional_info = additional_info or {}

        # Use provided author name or fall back to package default
        if author_name is None:
            author_name = PACKAGE_AUTHOR

        # Create frontmatter metadata
        frontmatter = {
            "license": "mit",
            "tags": [
                "computer-vision",
                "image-classification",
                "fastai",
                "pytorch",
                "mnist",
                "digit-recognition",
            ],
            "datasets": ["mnist"],
            "metrics": ["accuracy"],
            "model-index": [
                {
                    "name": model_name,
                    "results": [
                        {
                            "task": {
                                "type": "image-classification",
                                "name": "Image Classification",
                            },
                            "dataset": {"type": "mnist", "name": "MNIST"},
                            "metrics": [
                                {
                                    "type": "accuracy",
                                    "value": round(accuracy, 4),
                                    "name": "Accuracy",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        # Build markdown content
        builder = MarkdownBuilder()
        builder.add_frontmatter(frontmatter)

        # Main heading and description
        builder.add_heading(model_name)
        builder.add_heading("Model Description", 2)
        builder.add_paragraph(
            "This is a convolutional neural network trained to recognize handwritten digits (0-9) "
            "using the MNIST dataset. The model is built using FastAI and PyTorch."
        )

        # Model details
        builder.add_heading("Model Details", 2)
        builder.add_list_item(
            "**Model Type**: Convolutional Neural Network (ResNet18 architecture)"
        )
        builder.add_list_item("**Task**: Image Classification (Digit Recognition)")
        builder.add_list_item(
            "**Dataset**: MNIST (70,000 images of handwritten digits)"
        )
        builder.add_list_item("**Framework**: FastAI + PyTorch")
        builder.add_list_item("**License**: MIT")
        builder.add_blank_line()

        # Performance
        builder.add_heading("Performance", 2)
        builder.add_list_item(f"**Validation Accuracy**: {accuracy:.4f}")
        builder.add_list_item(f"**Training Epochs**: {epochs}")
        builder.add_list_item(f"**Batch Size**: {batch_size}")

        if learning_rate:
            builder.add_list_item(f"**Learning Rate**: {learning_rate}")
        builder.add_blank_line()

        # Usage section
        builder.add_heading("Usage", 2)
        builder.add_heading("Installation", 3)
        builder.add_code_block("pip install fastai torch pillow", "bash")

        builder.add_heading("Loading the Model", 3)
        builder.add_code_block(
            "from fastai.vision.all import load_learner\n"
            "import torch\n\n"
            "# Load the model\n"
            "learn = load_learner('path_to_model.pkl')\n\n"
            "# Make predictions\n"
            "pred_class, pred_idx, outputs = learn.predict(image)",
            "python",
        )

        builder.add_heading("Inference Example", 3)
        builder.add_code_block(
            "from fastai.vision.all import load_learner, PILImage\n"
            "from PIL import Image\n\n"
            f"# Load model\n"
            f"model = load_learner('{model_name}.pkl')\n\n"
            "# Load and predict an image\n"
            "img = PILImage.create('path_to_digit_image.png')\n"
            "pred_class, pred_idx, outputs = model.predict(img)\n"
            "confidence = outputs[pred_idx].item()\n\n"
            'print(f"Predicted digit: {pred_class}")\n'
            'print(f"Confidence: {confidence:.4f}")',
            "python",
        )

        # Training details
        builder.add_heading("Training Details", 2)
        builder.add_paragraph(
            "The model was trained using FastAI's transfer learning approach:"
        )
        builder.add_blank_line()
        builder.add_list_item(
            "**Architecture**: ResNet18 pre-trained on ImageNet", "1."
        )
        builder.add_list_item("**Fine-tuning**: One-cycle training policy", "2.")
        builder.add_list_item(
            "**Data Augmentation**: Random rotations, scaling, and lighting changes",
            "3.",
        )
        builder.add_list_item(
            "**Optimization**: AdamW optimizer with learning rate scheduling", "4."
        )
        builder.add_blank_line()

        # Files
        builder.add_heading("Files", 2)
        builder.add_list_item(
            f"`{model_name}.pkl`: Complete FastAI learner (recommended for inference)"
        )
        builder.add_list_item(
            f"`{model_name}_state_dict.pth`: PyTorch state dictionary"
        )
        builder.add_list_item("`config.json`: Model configuration and metadata")
        builder.add_blank_line()

        # Citation
        builder.add_heading("Citation", 2)
        builder.add_paragraph("If you use this model, please cite:")
        builder.add_blank_line()
        builder.add_code_block(
            f"@misc{{{model_name.replace('-', '_')},\n"
            f"    title={{MNIST Handwritten Digit Recognition using FastAI}},\n"
            f"    author={{{author_name}}},\n"
            f"    year={{2025}},\n"
            f"    publisher={{Hugging Face}},\n"
            f"    url={{https://huggingface.co/{username}/{model_name}}}\n"
            f"}}",
            "bibtex",
        )

        # Add any additional information
        if additional_info:
            builder.add_heading("Additional Information", 2)
            for key, value in additional_info.items():
                builder.add_list_item(f"**{key}**: {value}")
            builder.add_blank_line()

        return builder.build()

    def create_config_json(
        self,
        model_name: str,
        accuracy: float,
        epochs: int,
        batch_size: int,
        learning_rate: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create configuration JSON for the model

        Args:
            model_name: Name of the model
            accuracy: Final validation accuracy
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate used
            additional_metadata: Additional metadata

        Returns:
            Configuration dictionary
        """
        config = {
            "model_name": model_name,
            "task": "image-classification",
            "architecture": "resnet18",
            "framework": "fastai",
            "dataset": "mnist",
            "num_classes": 10,
            "image_size": [28, 28],
            "metrics": {"accuracy": accuracy},
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": "AdamW",
                "scheduler": "OneCycleLR",
            },
        }

        if learning_rate:
            config["training"]["learning_rate"] = learning_rate

        if additional_metadata:
            config.update(additional_metadata)

        return config

    def upload_model(
        self,
        model_path: str,
        repo_name: str,
        accuracy: float,
        epochs: int,
        batch_size: int,
        learning_rate: Optional[float] = None,
        private: bool = True,
        commit_message: Optional[str] = None,
        additional_files: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        author_name: Optional[str] = None,
    ) -> str:
        """
        Upload model to Hugging Face Hub

        Args:
            model_path: Path to the .pkl model file
            repo_name: Name of the repository (will be created if doesn't exist)
            accuracy: Final validation accuracy
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate used
            private: Whether to create a private repository
            commit_message: Custom commit message
            additional_files: List of additional files to upload
            additional_metadata: Additional metadata for config
            author_name: Author name for citation (if not provided, uses package default)

        Returns:
            URL of the uploaded repository
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"📤 Uploading model to Hugging Face Hub repository: {repo_name}")

        # Get username
        user_info = self.api.whoami(token=self.token)
        username = user_info["name"]
        full_repo_name = f"{username}/{repo_name}"

        # Create repository if it doesn't exist
        try:
            self.api.repo_info(full_repo_name, token=self.token)
            print(f"Repository {full_repo_name} already exists")
        except RepositoryNotFoundError:  # Catch any repo not found error
            print(f"Creating new repository: {full_repo_name}")
            create_repo(
                repo_id=full_repo_name,
                private=private,
                token=self.token,
                repo_type="model",
            )

        # Create temporary directory for files to upload
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy main model file
            model_filename = model_path_obj.name
            shutil.copy2(model_path_obj, temp_path / model_filename)

            # Copy state dict if it exists
            state_dict_path = (
                model_path_obj.parent / f"{model_path_obj.stem}_state_dict.pth"
            )
            if state_dict_path.exists():
                shutil.copy2(state_dict_path, temp_path / state_dict_path.name)

            # Create and save model card
            model_card_content = self.create_model_card(
                model_name=repo_name,
                username=username,
                accuracy=accuracy,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                additional_info=additional_metadata,
                author_name=author_name,
            )
            (temp_path / "README.md").write_text(model_card_content, encoding="utf-8")

            # Create and save config
            config = self.create_config_json(
                model_name=repo_name,
                accuracy=accuracy,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                additional_metadata=additional_metadata,
            )
            (temp_path / "config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )

            # Copy additional files if provided
            if additional_files:
                for file_path in additional_files:
                    file_path_obj = Path(file_path)
                    if file_path_obj.exists():
                        shutil.copy2(file_path_obj, temp_path / file_path_obj.name)
                        print(f"Added additional file: {file_path_obj.name}")

            # Upload all files
            commit_msg = (
                commit_message or f"Upload {repo_name} model (accuracy: {accuracy:.4f})"
            )

            print("Uploading files...")
            upload_folder(
                folder_path=temp_path,
                repo_id=full_repo_name,
                token=self.token,
                commit_message=commit_msg,
            )

        repo_url = f"https://huggingface.co/{full_repo_name}"
        print(f"✅ Model successfully uploaded to: {repo_url}")
        return repo_url

    def upload_from_trainer(
        self,
        trainer,
        repo_name: str,
        model_path: Optional[str] = None,
        private: bool = True,
        commit_message: Optional[str] = None,
        additional_files: Optional[List[str]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        author_name: Optional[str] = None,
    ) -> str:
        """
        Upload model directly from MNISTTrainer instance

        Args:
            trainer: MNISTTrainer instance with trained model
            repo_name: Name of the repository
            model_path: Path to model file (if not provided, will use default)
            private: Whether to create private repository
            commit_message: Custom commit message
            additional_files: Additional files to upload
            additional_metadata: Additional metadata
            author_name: Author name for citation (if not provided, uses package default)

        Returns:
            URL of the uploaded repository
        """
        if trainer.learn is None:
            raise ValueError("Trainer has no trained model. Train a model first.")

        # Get username
        user_info = self.api.whoami(token=self.token)
        username = user_info["name"]

        # Get training metrics
        validation_results = trainer.learn.validate()
        # Extract accuracy from recorder values or fallback to validation results
        accuracy = (
            trainer.learn.recorder.values[-1][1]
            if trainer.learn.recorder.values
            else validation_results.get("accuracy", 0.0)
        )

        # Default model path if not provided
        if model_path is None:
            model_path = str(Path("models") / f"{repo_name}.pkl")

        # Extract training parameters
        epochs = (
            len(trainer.learn.recorder.values) if trainer.learn.recorder.values else 1
        )
        batch_size = trainer.batch_size

        # Try to get learning rate from recorder
        learning_rate = None
        if hasattr(trainer.learn, "recorder") and trainer.learn.recorder.lrs:
            learning_rate = trainer.learn.recorder.lrs[-1]

        return self.upload_model(
            model_path=str(model_path),
            repo_name=repo_name,
            accuracy=accuracy,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            private=private,
            commit_message=commit_message,
            additional_files=additional_files,
            additional_metadata=additional_metadata,
            author_name=author_name,
        )

    def download_model(
        self, repo_name: str, local_dir: str = "downloaded_models"
    ) -> str:
        """
        Download a model from Hugging Face Hub

        Args:
            repo_name: Repository name (username/model-name)
            local_dir: Local directory to download to

        Returns:
            Path to downloaded model file
        """
        local_path = ensure_directory(local_dir)

        print(f"📥 Downloading model from: {repo_name}")

        # Download the repository
        repo_path = self.api.snapshot_download(
            repo_id=repo_name,
            local_dir=local_path / repo_name.split("/")[-1],
            token=self.token,
        )

        print(f"✅ Model downloaded to: {repo_path}")
        return repo_path


def push_to_huggingface(
    model_path: str,
    repo_name: str,
    accuracy: float,
    epochs: int,
    batch_size: int,
    learning_rate: Optional[float] = None,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: Optional[str] = None,
    additional_files: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    author_name: Optional[str] = None,
) -> str:
    """
    Convenience function to upload a model to Hugging Face Hub

    Args:
        model_path: Path to the .pkl model file
        repo_name: Name of the repository
        accuracy: Final validation accuracy
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate used
        token: HuggingFace token (optional, can use env var)
        private: Whether to create private repository
        commit_message: Custom commit message
        additional_files: Additional files to upload
        additional_metadata: Additional metadata
        author_name: Author name for citation (if not provided, uses package default)

    Returns:
        URL of the uploaded repository
    """
    uploader = HuggingFaceUploader(token=token)
    return uploader.upload_model(
        model_path=model_path,
        repo_name=repo_name,
        accuracy=accuracy,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        private=private,
        commit_message=commit_message,
        additional_files=additional_files,
        additional_metadata=additional_metadata,
        author_name=author_name,
    )
