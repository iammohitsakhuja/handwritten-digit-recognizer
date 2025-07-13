"""
Training module for MNIST digit recognition
"""

import torch
from fastai.vision.all import *
from fastai.data.external import *
import matplotlib.pyplot as plt
import time

from .utils import (
    setup_device,
    ensure_directory,
    save_plot,
    print_training_header,
    print_success_message,
    create_test_digit,
    get_safe_learning_rate,
    ensure_mps_float32_compatibility,
    patch_fastai_for_mps,
    unpatch_fastai_tensor,
)


class MNISTTrainer:
    """Main training class for MNIST digit recognition"""

    def __init__(self, batch_size=64, device=None):
        """
        Initialize the MNIST trainer

        Args:
            batch_size (int): Batch size for training
            device (torch.device): Device to use for training
        """
        self.batch_size = batch_size
        self.device = device or setup_device()
        self.dls = None
        self.learn = None
        self._data_path = None  # Cache for dataset path

    def setup_data(self, path="mnist_data"):
        """Download and prepare MNIST dataset"""
        print("Setting up MNIST dataset...")

        # Download MNIST dataset (cache the path)
        if self._data_path is None:
            self._data_path = untar_data(URLs.MNIST)
        data_path = self._data_path
        print(f"Dataset downloaded to: {data_path}")

        # Create data loaders with GPU support
        self.dls = ImageDataLoaders.from_folder(
            data_path,
            train="training",
            valid="testing",
            item_tfms=Resize(28),
            batch_tfms=[
                *aug_transforms(size=28, min_scale=0.8),
                Normalize.from_stats(*imagenet_stats),
            ],
            bs=self.batch_size,
            device=self.device,
            num_workers=(0 if self.device and str(self.device) == "mps" else 2),
        )

        # For MPS, manually move data loaders to device after creation
        if self.device and str(self.device) == "mps":
            print("🔧 Moving data loaders to MPS device...")
            self.dls = self.dls.to(self.device)

            # Ensure all data is float32 for MPS compatibility
            print("🔧 Ensuring data loaders use float32 for MPS compatibility...")
            torch.set_default_dtype(torch.float32)

        print(f"Training samples: {len(self.dls.train_ds)}")
        print(f"Validation samples: {len(self.dls.valid_ds)}")
        print(f"Classes: {self.dls.vocab}")

        if self.device:
            print(f"🔧 Data loaders configured for device: {self.device}")

        return self.dls

    def create_model(self):
        """Create and configure the CNN model"""
        if self.dls is None:
            raise ValueError("Data loaders not set up. Call setup_data() first.")

        print("Creating CNN model...")

        # Create a CNN learner with ResNet18 architecture
        self.learn = vision_learner(
            self.dls,
            resnet18,
            metrics=[accuracy, error_rate],
            loss_func=CrossEntropyLossFlat(),
        )

        # Handle device-specific configurations
        if self.device:
            print(f"🔧 Model will use device: {self.device}")

            # For MPS, ensure all tensors are float32
            if str(self.device) == "mps":
                print("🔧 Ensuring model uses float32 for MPS compatibility...")
                # Use the utility function to ensure float32 compatibility
                self.learn.model = ensure_mps_float32_compatibility(self.learn.model)

            # Show GPU memory usage if CUDA
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                print(
                    f"💾 GPU Memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**2:.1f} MB"
                )
                print(
                    f"💾 GPU Memory cached: {torch.cuda.memory_reserved(self.device) / 1024**2:.1f} MB"
                )

        return self.learn

    def train_model(self, epochs=10, learning_rate=None):
        """Train the model with fine-tuning"""
        if self.learn is None:
            raise ValueError("Model not created. Call create_model() first.")

        print(f"Training model for {epochs} epochs...")

        # Find optimal learning rate if not provided
        if learning_rate is None:
            print("Finding optimal learning rate...")
            try:
                if str(self.device) == "mps":
                    print("⚠️  MPS device detected, using CPU for lr_find")
                    # Temporarily move model to CPU for lr_find compatibility with MPS
                    original_device = self.device
                    self.learn.model.cpu()

                    # Set default device to CPU to avoid MPS issues during lr_find
                    torch.set_default_device("cpu")

                    # Find learning rate on CPU (MPS has compatibility issues with lr_find)
                    (learning_rate,) = self.learn.lr_find()

                    # Restore original device settings
                    torch.set_default_device(original_device)
                    self.learn.model.to(
                        original_device
                    )  # Move model back to original device
                else:
                    learning_rate = self.learn.lr_find()
                print(f"\nSuggested learning rate: {learning_rate}")
            except Exception as e:
                # Handle MPS/GPU compatibility issues with lr_find
                if (
                    "mps" in str(e).lower()
                    or "numpy" in str(e).lower()
                    or "can't convert" in str(e).lower()
                ):
                    print(
                        f"⚠️  Learning rate finder not compatible with {self.device}: {str(e)}"
                    )
                    print(
                        "� This is a known issue with MPS devices and FastAI's lr_find visualization"
                    )
                    learning_rate = get_safe_learning_rate(self.device)
                    print(f"Using device-optimized learning rate: {learning_rate}")
                    print(
                        "💡 Tip: You can specify --learning-rate manually for custom control"
                    )
                else:
                    print(f"⚠️  Learning rate finder failed: {str(e)}")
                    learning_rate = get_safe_learning_rate(self.device)
                    print(f"Using fallback learning rate: {learning_rate}")
        else:
            print(f"Using provided learning rate: {learning_rate}")

        # Train the model
        start_time = time.time()
        try:
            # Handle MPS-specific issues during training
            if self.device and str(self.device) == "mps":
                # For MPS, ensure everything stays in float32
                print("🔧 Using MPS-compatible training approach...")

                # Apply FastAI patches for MPS compatibility
                original_tensor = patch_fastai_for_mps()

                try:
                    # Set global PyTorch defaults for MPS
                    torch.set_default_dtype(torch.float32)
                    torch.set_default_device(self.device)

                    # Ensure all tensors are float32 using utility function
                    self.learn.model = ensure_mps_float32_compatibility(
                        self.learn.model
                    )

                    # For MPS, use fit_one_cycle instead of fine_tune to avoid scheduler issues
                    print("🔧 Training with float32 tensors on MPS...")
                    self.learn.fit_one_cycle(epochs, lr_max=learning_rate)

                finally:
                    # Always restore original tensor function
                    unpatch_fastai_tensor(original_tensor)

            else:
                # Standard training for CUDA/CPU
                self.learn.fine_tune(epochs, base_lr=learning_rate)

        except Exception as e:
            # Check if this is still an MPS float64 conversion error
            if "Cannot convert a MPS Tensor to float64" in str(e):
                print(
                    "⚠️  MPS float64 conversion error detected. Trying CPU fallback..."
                )
                print(
                    "⚠️  This appears to be a deep FastAI compatibility issue with MPS"
                )

                # As a last resort, move to CPU for training but keep model on MPS for inference
                print(
                    "🔧 Moving to CPU for training, will move back to MPS afterward..."
                )

                # Save original device
                original_device = self.device

                # Move everything to CPU
                self.device = torch.device("cpu")
                torch.set_default_device("cpu")
                torch.set_default_dtype(torch.float32)

                # Move model and data to CPU
                self.learn.model = self.learn.model.cpu()
                self.learn.dls = self.learn.dls.cpu()

                # Train on CPU
                print("🔧 Training on CPU...")
                self.learn.fit_one_cycle(epochs, lr_max=learning_rate)

                # Move model back to MPS for inference
                print("🔧 Moving trained model back to MPS for inference...")
                self.device = original_device
                self.learn.model = self.learn.model.to(self.device)

                print(
                    "✅ Training completed on CPU, model available on MPS for inference"
                )

            else:
                # Print stack trace for debugging other errors
                import traceback

                print("⚠️  Exception occurred during training:")
                traceback.print_exc()
                raise e

        training_time = time.time() - start_time

        print(f"Training completed in {training_time:.2f} seconds")
        return self.learn

    def evaluate_model(self, save_plots=True, output_dir="."):
        """Evaluate the trained model"""
        if self.learn is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("Evaluating model performance...")

        # Get validation metrics
        interp = ClassificationInterpretation.from_learner(self.learn)

        # Show confusion matrix
        print("Confusion Matrix:")
        interp.plot_confusion_matrix(figsize=(8, 8))
        plt.title("MNIST Digit Recognition - Confusion Matrix")
        plt.tight_layout()

        if save_plots:
            save_plot(plt.gcf(), output_dir, "confusion_matrix.png")

        plt.show()

        # Show most confused classes
        print("Most confused pairs:")
        interp.most_confused(min_val=2)

        # Show top losses (worst predictions)
        print("Analyzing worst predictions...")
        interp.plot_top_losses(9, nrows=3, figsize=(12, 8))
        plt.suptitle("MNIST Digit Recognition - Worst Predictions")
        plt.tight_layout()

        if save_plots:
            save_plot(plt.gcf(), output_dir, "worst_predictions.png")

        plt.show()

        # Get final accuracy
        validation_results = self.learn.validate()
        valid_loss = validation_results[0] if validation_results else 0.0
        accuracy = self.learn.recorder.values[-1][
            1
        ]  # Get accuracy from last validation
        print(f"Final Validation Accuracy: {accuracy:.4f}")
        print(f"Final Validation Loss: {valid_loss:.4f}")

        return accuracy

    def save_model(self, model_name="mnist_digit_recognizer", output_dir="models"):
        """Save the trained model"""
        if self.learn is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print(f"Saving model as '{model_name}'...")

        # Create models directory if it doesn't exist
        models_dir = ensure_directory(output_dir)

        # Save the complete learner
        model_path = models_dir / f"{model_name}.pkl"
        self.learn.export(model_path)
        print(f"Model saved to: {model_path}")

        # Also save just the model state dict for PyTorch compatibility
        torch_model_path = models_dir / f"{model_name}_state_dict.pth"
        torch.save(self.learn.model.state_dict(), torch_model_path)
        print(f"PyTorch state dict saved to: {torch_model_path}")

        return model_path

    def show_sample_data(self, save_plots=True, output_dir="."):
        """Show sample training data"""
        if self.dls is None:
            raise ValueError("Data loaders not set up. Call setup_data() first.")

        print("Sample training data:")
        self.dls.show_batch(max_n=9, figsize=(8, 8))
        plt.title("MNIST Training Samples")
        plt.tight_layout()

        if save_plots:
            save_plot(plt.gcf(), output_dir, "training_samples.png")

        plt.show()

    def test_predictions(self, save_plots=True, output_dir="."):
        """Test the model with some sample predictions"""
        if self.learn is None or self.dls is None:
            raise ValueError("Model and data not ready. Complete training first.")

        print("Testing model predictions...")

        # Get a batch of validation data
        x, y = self.dls.valid.one_batch()

        # Make predictions
        with torch.no_grad():
            preds = self.learn.model(x)
            pred_classes = torch.argmax(preds, dim=1)

        # Show some predictions
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        # Precompute normalization tensors outside the loop
        mean_tensor = torch.as_tensor(imagenet_stats[0]).view(3, 1, 1)
        std_tensor = torch.as_tensor(imagenet_stats[1]).view(3, 1, 1)

        for i in range(8):
            img = x[i].cpu()
            # Denormalize the image for display
            img = img * std_tensor + mean_tensor
            img = torch.clamp(img, 0, 1)

            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_title(f"True: {y[i]}, Pred: {pred_classes[i]}")
            axes[i].axis("off")

        plt.suptitle("Sample Predictions")
        plt.tight_layout()

        if save_plots:
            save_plot(fig, output_dir, "sample_predictions.png")

        plt.show()

    def test_synthetic_digits(self, save_plots=True, output_dir="."):
        """Test the model with synthetic test digits created by utils.py"""
        if self.learn is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("Testing model with synthetic digits...")

        # Create synthetic test digits using the utility function
        test_digits = []
        for digit in range(10):
            img = create_test_digit(digit, size=28)
            test_digits.append((img, digit))

        # Show the synthetic digits and predictions
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, (img, true_digit) in enumerate(test_digits):
            # Convert PIL image to tensor format expected by the model
            # Convert to RGB first, then to tensor
            img_rgb = img.convert("RGB")
            import torchvision.transforms as transforms

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            img_tensor = transform(img_rgb).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                if self.device:
                    img_tensor = img_tensor.to(self.device)
                pred = self.learn.model(img_tensor)
                pred_digit = torch.argmax(pred, dim=1).item()
                confidence = torch.softmax(pred, dim=1).max().item()

            # Display the image and prediction
            axes[i].imshow(img, cmap="gray")
            color = "green" if pred_digit == true_digit else "red"
            axes[i].set_title(
                f"True: {true_digit}, Pred: {pred_digit}\nConf: {confidence:.2f}",
                color=color,
            )
            axes[i].axis("off")

        plt.suptitle("Synthetic Digit Predictions")
        plt.tight_layout()

        if save_plots:
            save_plot(fig, output_dir, "synthetic_digit_predictions.png")

        plt.show()

        # Calculate accuracy on synthetic digits
        correct = sum(
            1
            for img, true_digit in test_digits
            if self._predict_single_synthetic(img) == true_digit
        )
        accuracy = correct / len(test_digits)
        print(
            f"Synthetic digit accuracy: {accuracy:.2f} ({correct}/{len(test_digits)})"
        )

        return accuracy

    def _predict_single_synthetic(self, img):
        """Helper method to predict a single synthetic digit"""
        img_rgb = img.convert("RGB")
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_tensor = transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            if self.device:
                img_tensor = img_tensor.to(self.device)
            pred = self.learn.model(img_tensor)
            return torch.argmax(pred, dim=1).item()

    def full_training_pipeline(
        self,
        epochs=5,
        learning_rate=None,
        model_name="mnist_digit_recognizer",
        save_plots=True,
        output_dir=".",
        models_dir="models",
        test_synthetic=True,
        run_evaluation=True,
        run_predictions=True,
    ):
        """Run the complete training pipeline with configurable components"""
        # Print training header
        print_training_header()

        # Setup data
        self.setup_data()

        # Show sample data
        if save_plots:
            self.show_sample_data(save_plots=save_plots, output_dir=output_dir)

        # Create and train model
        self.create_model()
        self.train_model(epochs=epochs, learning_rate=learning_rate)

        # Save model
        model_path = self.save_model(model_name=model_name, output_dir=models_dir)

        # Evaluate model (conditional)
        accuracy = 0.0
        if run_evaluation:
            accuracy = self.evaluate_model(save_plots=save_plots, output_dir=output_dir)

        # Test predictions (conditional)
        if run_predictions:
            self.test_predictions(save_plots=save_plots, output_dir=output_dir)

        # Test with synthetic digits (conditional)
        if test_synthetic:
            synthetic_accuracy = self.test_synthetic_digits(
                save_plots=save_plots, output_dir=output_dir
            )
            print(f"Bonus: Synthetic digit test accuracy: {synthetic_accuracy:.2f}")

        # Print success message
        print_success_message(accuracy, model_path)

        return accuracy, model_path
