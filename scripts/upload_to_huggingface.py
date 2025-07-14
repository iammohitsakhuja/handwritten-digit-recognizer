#!/usr/bin/env python3
"""
Upload MNIST Model to Hugging Face Hub

This script allows you to upload a trained MNIST model to Hugging Face Hub
with proper documentation and metadata.
"""

import sys
import argparse
from pathlib import Path
import json

# Add the src directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mnist_recognizer import HuggingFaceUploader


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Upload MNIST model to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to the trained model (.pkl file)",
    )

    parser.add_argument(
        "--repo-name",
        "-r",
        type=str,
        required=True,
        help="Name of the Hugging Face repository to create/update",
    )

    parser.add_argument(
        "--accuracy",
        "-a",
        type=float,
        required=True,
        help="Final validation accuracy of the model",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        required=True,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        required=True,
        help="Training batch size",
    )

    # Optional arguments
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        help="Learning rate used during training",
    )

    parser.add_argument(
        "--token",
        "-t",
        type=str,
        help="Hugging Face token (if not provided, will use HUGGINGFACE_TOKEN env var)",
    )

    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repository public (default is private)",
    )

    parser.add_argument(
        "--commit-message",
        "-c",
        type=str,
        help="Custom commit message",
    )

    parser.add_argument(
        "--additional-files",
        nargs="+",
        help="Additional files to upload with the model",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to JSON file with additional metadata",
    )

    parser.add_argument(
        "--description",
        type=str,
        help="Description of the model",
    )

    return parser.parse_args()


def load_metadata(metadata_path: str) -> dict:
    """Load additional metadata from JSON file"""
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        print(f"Warning: Metadata file not found at {metadata_path}: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON metadata from {metadata_path}: {e}")
        return {}


def main():
    """Main function to upload model"""
    args = parse_arguments()

    print("🚀 Uploading MNIST Model to Hugging Face Hub")
    print("=" * 50)

    try:
        # Load additional metadata if provided
        additional_metadata = {}
        if args.metadata:
            additional_metadata = load_metadata(args.metadata)

        # Add description to metadata if provided
        if args.description:
            additional_metadata["description"] = args.description

        # Initialize uploader
        uploader = HuggingFaceUploader(token=args.token)

        # Upload model
        repo_url = uploader.upload_model(
            model_path=args.model_path,
            repo_name=args.repo_name,
            accuracy=args.accuracy,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            private=not args.public,
            commit_message=args.commit_message,
            additional_files=args.additional_files,
            additional_metadata=additional_metadata,
        )

        print("\n" + "=" * 50)
        print("✅ Upload completed successfully!")
        print(f"🔗 Repository URL: {repo_url}")
        print("\n📝 Next steps:")
        print("1. Visit your repository to verify the upload")
        print("2. Test downloading and using the model")
        print("3. Share your model with the community!")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the model file exists and the path is correct.")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        print(
            "Make sure you have set HUGGINGFACE_TOKEN environment variable or pass --token"
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
