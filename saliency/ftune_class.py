import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from typing import Tuple, List, Union
from transformers import AutoModel, AutoImageProcessor
from torch import nn
from pathlib import Path
import cv2
from datetime import datetime


class LinearProbeDinoV2(nn.Module):
    def __init__(self, backbone, procssor, num_classes):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self, input: torch.Tensor, output_attentions: bool = False):
        # Get backbone features - output will contain hidden states since output_hidden_states=True
        outputs = self.backbone(
            input, output_attentions=output_attentions, return_dict=True
        )

        # Get CLS token embedding from last layer
        # DINOv2 uses CLS token as first token, like BERT
        cls_token = outputs.last_hidden_state[:, 0]

        # Pass through classifier head
        logits = self.classifier(cls_token)

        if output_attentions:
            # attentions is a tuple of attention matrices for each layer
            # Each attention matrix has shape (batch_size, num_heads, seq_length, seq_length)
            # Get last layer attention (equivalent to transformer_block_11_att)
            last_layer_attention = outputs.attentions[
                -1
            ]  # Shape: [B, num_heads, seq_len, seq_len]
            return logits, last_layer_attention

        return logits


class RealSyntheticDataset(Dataset):
    def __init__(
        self,
        real_path: Union[str, List[str]],
        synthetic_paths: Union[str, List[str]],
        transform=None,
        balance_classes: bool = True,
    ):
        """
        Args:
            real_path: Path or list of paths to directories containing real images
            synthetic_paths: Path or list of paths to directories containing synthetic images
            transform: Optional transform to be applied on images
            balance_classes: If True, undersample majority class to match minority class size
        """
        # Handle single path or list of paths for real images
        if isinstance(real_path, str):
            real_path = [real_path]
        self.real_paths = [Path(p) for p in real_path]

        # Handle single path or list of paths for synthetic images
        if isinstance(synthetic_paths, str):
            synthetic_paths = [synthetic_paths]
        self.synthetic_paths = [Path(p) for p in synthetic_paths]

        # Get real images from all paths
        self.real_images = []
        for real_p in self.real_paths:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]:
                self.real_images.extend(list(real_p.glob(ext)))
                self.real_images.extend(list(real_p.glob(ext.upper())))

        # Get synthetic images from all paths
        self.synthetic_images = []
        for syn_path in self.synthetic_paths:
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"]:
                self.synthetic_images.extend(list(syn_path.glob(ext)))
                self.synthetic_images.extend(list(syn_path.glob(ext.upper())))

        # Remove any potential duplicates
        self.real_images = list(set(self.real_images))
        self.synthetic_images = list(set(self.synthetic_images))

        # Print initial dataset statistics
        print(
            f"Initially found {len(self.real_images)} real images from {len(self.real_paths)} directories:"
        )
        for path in self.real_paths:
            real_count = len([x for x in self.real_images if str(path) in str(x)])
            print(f"  - {path}: {real_count} images")

        print(
            f"Initially found {len(self.synthetic_images)} synthetic images from {len(self.synthetic_paths)} directories:"
        )
        for path in self.synthetic_paths:
            syn_count = len([x for x in self.synthetic_images if str(path) in str(x)])
            print(f"  - {path}: {syn_count} images")

        if len(self.real_images) == 0:
            raise ValueError(f"No real images found in provided paths")
        if len(self.synthetic_images) == 0:
            raise ValueError(f"No synthetic images found in provided paths")

        # Balance classes if requested
        if balance_classes:
            min_size = min(len(self.real_images), len(self.synthetic_images))
            if len(self.real_images) > min_size:
                print(
                    f"Undersampling real images from {len(self.real_images)} to {min_size}"
                )
                self.real_images = random.sample(self.real_images, min_size)
            elif len(self.synthetic_images) > min_size:
                print(
                    f"Undersampling synthetic images from {len(self.synthetic_images)} to {min_size}"
                )
                self.synthetic_images = random.sample(self.synthetic_images, min_size)

            print("\nAfter balancing:")
            print(f"Real images: {len(self.real_images)}")
            print(f"Synthetic images: {len(self.synthetic_images)}")

        # Combine all paths and create labels
        self.image_paths = self.real_images + self.synthetic_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.synthetic_images)

        # Set default transform if none provided
        if transform is None:
            self.transform = T.Compose(
                [
                    T.Resize((224, 224)),  # DINOv2 default size
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a random different image
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {img_path}: {str(e)}")
                # Return a random different image
                new_idx = (idx + 1) % len(self)
                return self.__getitem__(new_idx)

        return image, label

    def get_image_path(self, idx: int) -> Path:
        """Get the file path for a given index"""
        return self.image_paths[idx]


def create_dataloaders(
    real_path: str,
    synthetic_path: str,
    output_dir: str = "split_info",
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
    transform=None,
    balance_classes: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation dataloaders and saves split information to JSON files

    Args:
        real_path: Path to real images
        synthetic_path: Path to synthetic images
        output_dir: Directory to save split information JSON files
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
        transform: Optional custom transform
        balance_classes: If True, undersample majority class to match minority class size
    """
    # Create dataset with class balancing
    dataset = RealSyntheticDataset(
        real_path, synthetic_path, transform, balance_classes
    )

    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    # Generate random indices for the split
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create the splits using the indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Prepare split information
    split_info = {
        "metadata": {
            "creation_date": datetime.now().isoformat(),
            "train_split_ratio": train_split,
            "total_images": len(dataset),
            "train_size": train_size,
            "val_size": val_size,
        },
        "splits": {
            "train": {"real": [], "synthetic": []},
            "val": {"real": [], "synthetic": []},
        },
    }

    # Record file paths for each split
    for idx in train_indices:
        path = str(dataset.get_image_path(idx))
        if dataset.labels[idx] == 0:
            split_info["splits"]["train"]["real"].append(path)
        else:
            split_info["splits"]["train"]["synthetic"].append(path)

    for idx in val_indices:
        path = str(dataset.get_image_path(idx))
        if dataset.labels[idx] == 0:
            split_info["splits"]["val"]["real"].append(path)
        else:
            split_info["splits"]["val"]["synthetic"].append(path)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save split information to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"split_info_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Split information saved to {json_path}")

    # Create and return dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import make_grid
import numpy as np


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor = None,
    num_images: int = 16,
):
    """
    Visualize a batch of images with their labels and predictions

    Args:
        images: Tensor of shape [B, C, H, W]
        labels: Tensor of shape [B]
        predictions: Optional tensor of shape [B] containing model predictions
        num_images: Number of images to display
    """
    # Select subset of images
    images = images[:num_images]
    labels = labels[:num_images]
    if predictions is not None:
        predictions = predictions[:num_images]

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean

    # Create grid of images
    grid = make_grid(images, nrow=4, padding=2, normalize=True)
    grid = grid.cpu().numpy().transpose((1, 2, 0))

    # Create figure
    plt.figure(figsize=(15, 15))
    plt.imshow(grid)

    # Add labels
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols
    for idx in range(num_images):
        row = idx // num_cols
        col = idx % num_cols

        # Calculate text position
        x = col * (images.shape[3] + 2) + images.shape[3] // 2
        y = row * (images.shape[2] + 2) + images.shape[2] + 1

        # Create label text
        label_text = f"Real" if labels[idx] == 0 else f"Synthetic"
        if predictions is not None:
            pred_text = "Real" if predictions[idx] == 0 else "Synthetic"
            label_text += f"\nPred: {pred_text}"

            # Add color based on correctness
            color = "green" if predictions[idx] == labels[idx] else "red"
        else:
            color = "white"

        plt.text(
            x,
            y,
            label_text,
            color=color,
            horizontalalignment="center",
            verticalalignment="top",
            bbox=dict(facecolor="black", alpha=0.8),
        )

    plt.axis("off")
    if wandb.run is not None:
        wandb.log({"batch_visualization": wandb.Image(plt)})
    plt.close()


import torch
import wandb
from tqdm import tqdm
from typing import Dict, Tuple
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_step(
    model: LinearProbeDinoV2,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str = "cuda",
) -> Dict[str, float]:
    """Single training step"""
    model.train()
    images, labels = [x.to(device) for x in batch]

    # Forward pass
    logits = model(images)
    loss = criterion(logits, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean()

    return {"train_loss": loss.item(), "train_acc": acc.item()}


@torch.no_grad()
def val_step(
    model: LinearProbeDinoV2,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str = "cuda",
) -> Dict[str, float]:
    """Validation step"""
    model.eval()
    val_loss = 0
    val_acc = 0

    for batch in val_loader:
        images, labels = [x.to(device) for x in batch]

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        val_loss += loss.item()
        val_acc += acc.item()

    # Average metrics
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    return {"val_loss": val_loss, "val_acc": val_acc}


def train(
    model: LinearProbeDinoV2,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    wandb_config: Dict = None,
    log_dir: str = ".",
    visualize_first_batch: bool = True,  # New parameter
) -> None:
    """Main training loop"""

    # Initialize wandb
    if wandb_config:
        wandb.init(
            project=wandb_config.get("project", "dino-real-synthetic"),
            name=wandb_config.get("name", "experiment"),
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": train_loader.batch_size,
                "model": "dinov2-base-linear-probe",
            },
        )

    # Setup training
    model = model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0
    first_batch_visualized = False

    for epoch in range(num_epochs):
        # Training loop
        train_metrics = {"train_loss": 0.0, "train_acc": 0.0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch_metrics = train_step(model, batch, optimizer, criterion, device)

            # Visualize first batch of first epoch
            if visualize_first_batch and not first_batch_visualized:
                images, labels = [x.to(device) for x in batch]
                with torch.no_grad():
                    logits = model(images)
                    predictions = torch.argmax(logits, dim=1)
                visualize_batch(images, labels, predictions)
                first_batch_visualized = True

            # Update metrics
            for k, v in batch_metrics.items():
                train_metrics[k] += v

            # Update progress bar
            pbar.set_postfix(
                {"loss": batch_metrics["train_loss"], "acc": batch_metrics["train_acc"]}
            )

        # Rest of the training loop remains the same...
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
        val_metrics = val_step(model, val_loader, criterion, device)
        scheduler.step()

        # Log metrics
        metrics = {**train_metrics, **val_metrics}
        if wandb_config:
            wandb.log(metrics)

        # Save best model
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            torch.save(model.state_dict(), Path(log_dir, "best_model.pt"))

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.4f}"
        )
        print(
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.4f}"
        )
        print(f"Best Val Acc: {best_val_acc:.4f}")
        print("-" * 50)


def attention_maps():

    image_path = (
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/Turing/synth/seed0059439276.jpeg"
    )
    device = "cuda"
    # Set up model
    backbone = AutoModel.from_pretrained(
        "facebook/dinov2-base", output_hidden_states=True
    )
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    num_classes = 2
    model = LinearProbeDinoV2(backbone, processor, num_classes)
    ckpt = torch.load("./data/linear_probing/KSA_MAMOG/best_model.pt")
    model.load_state_dict(ckpt)
    model.to(device)
    # Prepare image

    image = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get model predictions and attention
    model.eval()
    with torch.no_grad():
        logits, attention = model(image_tensor, output_attentions=True)
        pred = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)

    # Get CLS token attention to patches
    cls_attention = attention[0, :, 0, 1:]  # [num_heads, num_patches]

    # Reshape attention to square grid
    patch_size = 16  # DINOv2 base uses 16x16 patches
    num_patches = int(np.sqrt(cls_attention.shape[1]))
    attention_maps = cls_attention.reshape(-1, num_patches, num_patches)

    # Denormalize image for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img = image_tensor * std + mean
    img = img[0].permute(1, 2, 0).cpu().numpy()

    # Plot
    num_heads = attention_maps.shape[0]
    fig = plt.figure(figsize=(20, 10))

    # Plot original image
    ax = plt.subplot(2, num_heads // 2 + 1, 1)
    ax.imshow(img)
    ax.set_title(
        f'Original Image\nPrediction: {"Synthetic" if pred == 1 else "Real"}\nConfidence: {probs[0][pred]:.2f}'
    )
    ax.axis("off")

    # Plot attention maps
    for idx, attention_map in enumerate(attention_maps):
        ax = plt.subplot(2, num_heads // 2 + 1, idx + 2)
        ax.imshow(img)  # Show original image

        # Normalize and overlay attention map
        attention_map = attention_map.cpu().numpy()
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min()
        )

        # Resize attention map to match image size
        attention_map_resized = cv2.resize(attention_map, (224, 224))
        ax.imshow(attention_map_resized, cmap="hot", alpha=0.5)
        ax.set_title(f"Attention Head {idx}")
        ax.axis("off")

    plt.suptitle("DINO-v2 Attention Maps", fontsize=16)
    plt.tight_layout()

    plt.savefig(
        "./data/linear_probing/KSA_MAMOG/best_model.png", bbox_inches="tight", dpi=300
    )

    plt.show()
    plt.close()


import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModel, AutoImageProcessor
import random

from math import ceil, sqrt


def create_dense_attention_grid(
    split_info_path: str, model_path: str, output_dir: str, samples_per_class: int = 16
):
    """
    Create dense grids of attention maps for each class with minimal spacing

    Args:
        split_info_path: Path to the JSON file containing split information
        model_path: Path to the saved model checkpoint
        output_dir: Directory to save visualizations
        samples_per_class: Number of samples to show for each class
    """
    # Load split information
    with open(split_info_path, "r") as f:
        split_info = json.load(f)

    # Correctly extract validation paths from split info
    validation_data = split_info["splits"]["val"]
    val_real = validation_data["real"]
    val_synthetic = validation_data["synthetic"]

    # Separate synthetic into GAN and Diffusion
    val_diffusion = [p for p in val_synthetic if "DiT" in p]
    val_gan = [p for p in val_synthetic if "DiT" not in p]

    print(f"Found {len(val_real)} real validation images")
    print(f"Found {len(val_gan)} GAN validation images")
    print(f"Found {len(val_diffusion)} Diffusion validation images")

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = AutoModel.from_pretrained(
        "facebook/dinov2-base", output_hidden_states=True
    )
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    num_classes = 2
    model = LinearProbeDinoV2(backbone, processor, num_classes)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def calculate_accuracy(results):
        """Calculate accuracy metrics for a set of results"""
        correct = sum(1 for r in results if r["prediction"] == r["true_label"])
        total = len(results)
        return (correct / total * 100) if total > 0 else 0

    def process_image_batch(paths, true_label):
        """Process a batch of images and return attention maps and predictions"""
        results = []
        for path in paths:
            try:
                image = Image.open(path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, attention = model(image_tensor, output_attentions=True)
                    pred = torch.argmax(logits, dim=1).item()
                    probs = torch.softmax(logits, dim=1)

                # Get 6th head attention
                cls_attention = attention[0, 5, 0, 1:]
                num_patches = int(np.sqrt(cls_attention.shape[0]))
                attention_map = cls_attention.reshape(num_patches, num_patches)

                # Denormalize image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                img = image_tensor * std + mean
                img = img[0].permute(1, 2, 0).cpu().numpy()

                results.append(
                    {
                        "image": img,
                        "attention_map": attention_map.cpu().numpy(),
                        "prediction": pred,
                        "confidence": probs[0][pred].item(),
                        "true_label": true_label,
                    }
                )
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                continue
        return results

    def process_all_validation_samples(paths, true_label):
        """Process all validation samples for accuracy calculation"""
        results = []
        for path in paths:
            try:
                image = Image.open(path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, _ = model(image_tensor, output_attentions=True)
                    pred = torch.argmax(logits, dim=1).item()
                    probs = torch.softmax(logits, dim=1)

                results.append(
                    {
                        "prediction": pred,
                        "confidence": probs[0][pred].item(),
                        "true_label": true_label,
                    }
                )
            except Exception as e:
                print(f"Error processing image {path}: {str(e)}")
                continue
        return results

    # Calculate overall accuracies first
    print("Calculating overall accuracies...")
    real_all_results = process_all_validation_samples(val_real, 0)
    gan_all_results = process_all_validation_samples(val_gan, 1)
    diff_all_results = process_all_validation_samples(val_diffusion, 1)

    real_acc = calculate_accuracy(real_all_results)
    gan_acc = calculate_accuracy(gan_all_results)
    diff_acc = calculate_accuracy(diff_all_results)

    # Process sample images for visualization
    real_results = process_image_batch(
        random.sample(val_real, min(samples_per_class, len(val_real))), 0
    )
    gan_results = process_image_batch(
        random.sample(val_gan, min(samples_per_class, len(val_gan))), 1
    )
    diff_results = process_image_batch(
        random.sample(val_diffusion, min(samples_per_class, len(val_diffusion))), 1
    )

    def create_dense_grid(results, title, overall_acc):
        """Create a dense grid of images with attention maps and accuracy stats"""
        n = len(results)
        grid_size = ceil(sqrt(n))

        # Create figure with minimal spacing
        fig = plt.figure(figsize=(15, 15))
        plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.92)

        for idx, result in enumerate(results):
            ax = plt.subplot(grid_size, grid_size, idx + 1)

            # Show image
            ax.imshow(result["image"])

            # Overlay attention map
            attention_map = result["attention_map"]
            attention_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min()
            )
            attention_map_resized = cv2.resize(attention_map, (224, 224))
            ax.imshow(attention_map_resized, cmap="hot", alpha=0.5)

            # Add small colored dot for prediction correctness
            correct = result["prediction"] == result["true_label"]
            color = "green" if correct else "red"
            confidence = result["confidence"]

            # Add small colored rectangle in corner for prediction
            rect = plt.Rectangle(
                (0.9, 0.9), 0.1, 0.1, transform=ax.transAxes, facecolor=color, alpha=0.8
            )
            ax.add_patch(rect)

            # Add tiny confidence text
            ax.text(
                0.91,
                0.91,
                f"{confidence:.2f}",
                transform=ax.transAxes,
                color="white",
                fontsize=8,
                weight="bold",
            )

            ax.axis("off")

        plt.suptitle(
            f"{title}\nOverall Accuracy: {overall_acc:.1f}% ({len(results)} samples shown)",
            fontsize=16,
            y=0.98,
        )
        return fig

    # Create and save grid for each class
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Real images grid
    fig_real = create_dense_grid(
        real_results, "Real Images - 6th Head Attention", real_acc
    )
    fig_real.savefig(
        output_path / "real_attention_grid.png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
    )
    plt.close(fig_real)

    # GAN images grid
    fig_gan = create_dense_grid(
        gan_results, "GAN-Generated Images - 6th Head Attention", gan_acc
    )
    fig_gan.savefig(
        output_path / "gan_attention_grid.png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
    )
    plt.close(fig_gan)

    # Diffusion images grid
    fig_diff = create_dense_grid(
        diff_results, "Diffusion-Generated Images - 6th Head Attention", diff_acc
    )
    fig_diff.savefig(
        output_path / "diffusion_attention_grid.png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
    )
    plt.close(fig_diff)

    # Print overall statistics
    print(f"\nOverall Accuracy Statistics:")
    print(f"Real Images: {real_acc:.1f}% ({len(real_all_results)} total samples)")
    print(f"GAN Images: {gan_acc:.1f}% ({len(gan_all_results)} total samples)")
    print(f"Diffusion Images: {diff_acc:.1f}% ({len(diff_all_results)} total samples)")


import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import umap
from PIL import Image
import torchvision.transforms as T
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def visualize_feature_embeddings(
    split_info_path: str,
    model_path: str,
    output_dir: str,
    batch_size: int = 32,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    force_recompute: bool = False,
):
    """
    Visualize feature embeddings using UMAP, separating real, GAN, and diffusion images

    Args:
        split_info_path: Path to the JSON file containing split information
        model_path: Path to the saved model checkpoint
        output_dir: Directory to save visualizations
        batch_size: Batch size for processing images
        n_neighbors: UMAP parameter for local neighborhood size
        min_dist: UMAP parameter for minimum distance between points
        random_state: Random seed for reproducibility
        force_recompute: If True, recompute embeddings even if cached version exists
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    embeddings_cache_path = output_path / "cached_embeddings.npz"

    # Check if cached embeddings exist
    if not force_recompute and embeddings_cache_path.exists():
        print("Loading cached embeddings...")
        cached_data = np.load(embeddings_cache_path, allow_pickle=True)
        real_features = cached_data["real_features"]
        gan_features = cached_data["gan_features"]
        diff_features = cached_data["diff_features"]
        real_paths = cached_data["real_paths"]
        gan_paths = cached_data["gan_paths"]
        diff_paths = cached_data["diff_paths"]
        print("Cached embeddings loaded successfully!")
    else:
        # Load split information
        with open(split_info_path, "r") as f:
            split_info = json.load(f)

        # Extract validation paths
        validation_data = split_info["splits"]["val"]
        val_real = validation_data["real"]
        val_synthetic = validation_data["synthetic"]
        val_diffusion = [p for p in val_synthetic if "DiT" in p]
        val_gan = [p for p in val_synthetic if "DiT" not in p]

        print(f"Processing feature embeddings for:")
        print(f"- {len(val_real)} real images")
        print(f"- {len(val_gan)} GAN images")
        print(f"- {len(val_diffusion)} diffusion images")

        # Setup device and model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        backbone = AutoModel.from_pretrained(
            "facebook/dinov2-base", output_hidden_states=True
        )
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        num_classes = 2
        model = LinearProbeDinoV2(backbone, processor, num_classes)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        def extract_features_batch(paths):
            """Extract features for a batch of images"""
            images = []
            valid_paths = []

            for path in paths:
                try:
                    image = Image.open(path).convert("RGB")
                    image_tensor = transform(image)
                    images.append(image_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading image {path}: {str(e)}")
                    continue

            if not images:
                return [], []

            # Stack images into a batch
            batch = torch.stack(images).to(device)

            with torch.no_grad():
                features = model.backbone(batch).last_hidden_state[
                    :, 0
                ]  # Get CLS token features

            return features.cpu().numpy(), valid_paths

        def process_dataset(paths):
            """Process all images in a dataset"""
            all_features = []
            valid_paths = []
            length = len(paths)
            for i in range(0, length, batch_size):
                batch_paths = paths[i : i + batch_size]
                features, batch_paths = extract_features_batch(batch_paths)
                if len(features) > 0:
                    all_features.append(features)
                    valid_paths.extend(batch_paths)

                if (i + batch_size) % (batch_size * 10) == 0:
                    print(f"Processed {i + batch_size} images...")

            if all_features:
                return np.vstack(all_features), valid_paths
            return np.array([]), []

        # Extract features for all datasets
        print("Extracting features for real images...")
        real_features, real_paths = process_dataset(val_real)
        print(real_features.shape)
        print("Extracting features for GAN images...")
        gan_features, gan_paths = process_dataset(val_gan)
        print(gan_features.shape)
        print("Extracting features for diffusion images...")
        diff_features, diff_paths = process_dataset(val_diffusion)
        print(diff_features.shape)

        # Save embeddings to cache
        print("Saving embeddings to cache...")
        np.savez_compressed(
            embeddings_cache_path,
            real_features=real_features,
            gan_features=gan_features,
            diff_features=diff_features,
            real_paths=real_paths,
            gan_paths=gan_paths,
            diff_paths=diff_paths,
        )
        print("Embeddings cached successfully!")

    # Combine features and create labels
    all_features = np.vstack([real_features, gan_features, diff_features])

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features)

    # Apply UMAP
    print("Applying UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    embeddings = reducer.fit_transform(scaled_features)

    # Create labels for plotting
    labels = (
        ["Real"] * len(real_features)
        + ["GAN"] * len(gan_features)
        + ["Diffusion"] * len(diff_features)
    )

    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )
    from scipy.spatial import ConvexHull
    import seaborn as sns
    from matplotlib.patches import Polygon

    def evaluate_clustering(embeddings, labels, cluster_labels=None):
        """Evaluate clustering quality using various metrics"""
        if cluster_labels is None:
            # Use the true labels
            unique_labels = np.unique(labels)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_map[label] for label in labels])
        else:
            numeric_labels = cluster_labels

        metrics = {
            "silhouette": silhouette_score(embeddings, numeric_labels),
            "calinski_harabasz": calinski_harabasz_score(embeddings, numeric_labels),
            "davies_bouldin": davies_bouldin_score(embeddings, numeric_labels),
        }
        return metrics

    def plot_cluster_boundaries(embeddings, labels, ax, alpha=0.2):
        """Plot convex hull boundaries for each cluster"""
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            points = embeddings[mask]
            if len(points) >= 3:  # Need at least 3 points for convex hull
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                polygon = Polygon(hull_points, alpha=alpha, label=f"{label} region")
                ax.add_patch(polygon)

    # Create main visualization with enhanced plotting
    fig = plt.figure(figsize=(20, 10))

    # Create subplot for the main scatter plot with boundaries
    ax1 = plt.subplot(121)

    # Set style
    plt.style.use("seaborn-v0_8-deep")

    # Create scatterplot with different colors and markers for each class
    colors = sns.color_palette("deep")
    for idx, (label, marker) in enumerate(
        zip(["Real", "GAN", "Diffusion"], ["o", "s", "^"])
    ):
        mask = np.array(labels) == label
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=label,
            marker=marker,
            alpha=0.6,
            s=50,
            color=colors[idx],
        )

    # Add cluster boundaries
    plot_cluster_boundaries(embeddings, np.array(labels), plt.gca())

    plt.title(
        "UMAP Visualization with Cluster Boundaries\nDINOv2 Embeddings", fontsize=14
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(fontsize=10, loc="upper center")

    # Add density plots
    ax2 = plt.subplot(122)

    # Create separate density plots for each class
    for idx, label in enumerate(["Real", "GAN", "Diffusion"]):
        mask = np.array(labels) == label
        sns.kdeplot(
            x=embeddings[mask, 0],
            y=embeddings[mask, 1],
            color=colors[idx],
            alpha=0.5,
            fill=True,
            label=label,
        )

    plt.title("Density Distribution of Clusters", fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(fontsize=10)

    # Calculate clustering metrics
    metrics_true = evaluate_clustering(embeddings, labels)

    # Try different clustering algorithms
    n_clusters = len(np.unique(labels))

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(embeddings)
    metrics_kmeans = evaluate_clustering(embeddings, labels, kmeans_labels)

    # DBSCAN clustering with optimized parameters
    from sklearn.neighbors import NearestNeighbors

    # Estimate epsilon for DBSCAN
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(embeddings)
    distances, _ = neigh.kneighbors(embeddings)
    eps = np.percentile(distances[:, -1], 90)  # Use 90th percentile of distances

    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan_labels = dbscan.fit_predict(embeddings)
    if -1 not in dbscan_labels:  # Only evaluate if no noise points
        metrics_dbscan = evaluate_clustering(embeddings, labels, dbscan_labels)
    else:
        metrics_dbscan = None

    # Add text with clustering metrics and dataset sizes
    info_text = (
        f"Dataset sizes:\n"
        f"Real: {len(real_features)}\n"
        f"GAN: {len(gan_features)}\n"
        f"Diffusion: {len(diff_features)}\n\n"
        f"Clustering Metrics:\n"
        f"True Labels:\n"
        f'- Silhouette: {metrics_true["silhouette"]:.3f}\n'
        f'- Calinski-Harabasz: {metrics_true["calinski_harabasz"]:.3f}\n'
        f'- Davies-Bouldin: {metrics_true["davies_bouldin"]:.3f}\n\n'
        f"K-means:\n"
        f'- Silhouette: {metrics_kmeans["silhouette"]:.3f}\n'
        f'- Calinski-Harabasz: {metrics_kmeans["calinski_harabasz"]:.3f}\n'
        f'- Davies-Bouldin: {metrics_kmeans["davies_bouldin"]:.3f}'
    )

    if metrics_dbscan:
        info_text += (
            f"\n\nDBSCAN:\n"
            f'- Silhouette: {metrics_dbscan["silhouette"]:.3f}\n'
            f'- Calinski-Harabasz: {metrics_dbscan["calinski_harabasz"]:.3f}\n'
            f'- Davies-Bouldin: {metrics_dbscan["davies_bouldin"]:.3f}'
        )

    # Add text to figure
    plt.figtext(
        1.02,
        0.5,
        info_text,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="center",
    )

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save enhanced visualization
    plt.savefig(
        output_path / "umap_features_enhanced.png",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )
    plt.close()

    # Print interpretation of metrics
    print("\nClustering Analysis:")
    print("\nMetrics Interpretation:")
    print("1. Silhouette Score (-1 to 1):")
    print("   - Higher is better")
    print("   - Above 0.5 indicates good separation")
    print("   - Your score:", f"{metrics_true['silhouette']:.3f}")

    print("\n2. Calinski-Harabasz Index:")
    print("   - Higher is better")
    print("   - No fixed range, but higher values indicate better clustering")
    print("   - Your score:", f"{metrics_true['calinski_harabasz']:.3f}")

    print("\n3. Davies-Bouldin Index:")
    print("   - Lower is better")
    print("   - Values closer to 0 indicate better separation")
    print("   - Your score:", f"{metrics_true['davies_bouldin']:.3f}")

    # Calculate and print pairwise distances between cluster centers
    print("\nInter-cluster Distances:")
    cluster_centers = {}
    for label in ["Real", "GAN", "Diffusion"]:
        mask = np.array(labels) == label
        cluster_centers[label] = np.mean(embeddings[mask], axis=0)

    for label1 in cluster_centers:
        for label2 in cluster_centers:
            if label1 < label2:
                dist = np.linalg.norm(cluster_centers[label1] - cluster_centers[label2])
                print(f"{label1} vs {label2}: {dist:.3f}")

    # Calculate overlap percentages
    def calculate_overlap(embeddings, labels, label1, label2, threshold=1.0):
        """Calculate approximate overlap between two clusters"""
        mask1 = np.array(labels) == label1
        mask2 = np.array(labels) == label2
        points1 = embeddings[mask1]
        points2 = embeddings[mask2]

        # Calculate distances from each point in cluster 1 to nearest point in cluster 2
        tree = NearestNeighbors(n_neighbors=1)
        tree.fit(points2)
        distances, _ = tree.kneighbors(points1)

        # Calculate percentage of points that are closer than threshold
        overlap_percentage = np.mean(distances < threshold) * 100
        return overlap_percentage

    print(
        "\nCluster Overlap Analysis (% of points with close neighbors from other clusters):"
    )
    threshold = np.percentile(distances[:, -1], 10)  # Use 10th percentile as threshold
    for label1 in ["Real", "GAN", "Diffusion"]:
        for label2 in ["Real", "GAN", "Diffusion"]:
            if label1 < label2:
                overlap = calculate_overlap(
                    embeddings, labels, label1, label2, threshold
                )
                print(f"{label1} -> {label2}: {overlap:.1f}%")

    # Save metrics to file
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj

    metrics_dict = {
        "true_labels": convert_to_json_serializable(metrics_true),
        "kmeans": convert_to_json_serializable(metrics_kmeans),
        "dbscan": convert_to_json_serializable(metrics_dbscan)
        if metrics_dbscan
        else "No valid clusters",
        "cluster_centers": {
            k: convert_to_json_serializable(v) for k, v in cluster_centers.items()
        },
    }

    # Save with pretty printing
    with open(output_path / "clustering_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)


def main():
    save_dir = "./data/linear_probing"
    real_path = [
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/microcalc/train/microcalc",
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/uncond_normal/Normal",
    ]
    synthetic_path = [
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/logs/009-DiT-XL-2/DiT-XL-2-0010000-size-512-class1-vae-ema-cfg-1.5-seed-0--steps350-final",
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/logs/009-DiT-XL-2/DiT-XL-2-0010000-size-512-class0-vae-ema-cfg-1.5-seed-0--steps350-final",
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/synthetic-data-generation/sample_run/microcalc_2048/00006-train-mirror-auto2-batch4-resumecustom-freezed4/gen_2099",
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/synthetic-data-generation/sample_run/normal_2048/00011-train-mirror-auto2-batch8-resumecustom-freezed0/gen_988",
    ]
    log_dir = Path(os.path.join(save_dir, "KSA_MAMOG"))
    num_classes = 2

    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        real_path=real_path,
        synthetic_path=synthetic_path,
        output_dir=log_dir,
        batch_size=64,
    )

    backbone = AutoModel.from_pretrained(
        "facebook/dinov2-base", output_hidden_states=True
    )
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

    wandb_config = {"project": "dino-real-synthetic", "name": "experiment-1"}

    model = LinearProbeDinoV2(backbone, processor, num_classes)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        learning_rate=1e-4,
        wandb_config=wandb_config,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    # main()

    visualize_feature_embeddings(
        split_info_path="data/linear_probing/KSA_MAMOG/split_info_20241023_135651.json",
        model_path="./data/linear_probing/KSA_MAMOG/best_model.pt",
        output_dir="./data/linear_probing/KSA_MAMOG/feature_analysis",
        batch_size=32,
        n_neighbors=15,
        min_dist=0.1,
    )
    # create_dense_attention_grid(
    #    split_info_path="./data/linear_probing/KSA_MAMOG/split_info_20241023_135651.json",
    #    model_path="./data/linear_probing/KSA_MAMOG/best_model.pt",
    #    output_dir="./data/linear_probing/KSA_MAMOG/attention_maps",
    #    samples_per_class=36
    # )

    # attention_maps()
