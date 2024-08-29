import torch
import argparse
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from torch.optim import lr_scheduler
import os
import random
from torchvision import datasets
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
from torchvision import models
from utils import load_model_from_hub

class SiameseNetwork(nn.Module):
    """
    A Siamese Network architecture based on a ResNet-50 backbone for feature extraction.

    Parameters:
    -----------
    network : str, default='ResNet-50'
        The backbone network to be used for feature extraction. Currently, only 'ResNet-50' is supported.
    in_channels : int, default=1
        Number of input channels for the images. Must be either 1 (grayscale) or 3 (RGB).
    n_features : int, default=128
        The dimensionality of the feature vector output by the network.

    Raises:
    -------
    ValueError:
        If `in_channels` is not 1 or 3.
        If `network` is not 'ResNet-50'.

    Methods:
    --------
    forward_once(x):
        Forward function for one branch to get the n_features-dim feature vector before merging.

    forward(input1, input2):
        Forward function that returns the feature vectors from each branch of the Siamese network.
    """

    def __init__(self, network="ResNet-50", in_channels=1, n_features=128):
        super(SiameseNetwork, self).__init__()
        self.network = network
        self.in_channels = in_channels
        self.n_features = n_features

        if self.network == "ResNet-50":
            self.model = models.resnet50(pretrained=False)
            self.model.load_state_dict(load_model_from_hub('molinamarc/syntheva', 'resnet50-19c8e357.pth'))

            # Adjust the input layer for different channels
            if self.in_channels == 1:
                self.model.conv1 = nn.Conv2d(
                    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
            elif self.in_channels != 3:
                raise ValueError(
                    "Invalid in_channels: {}. Choose either 1 or 3.".format(
                        self.in_channels
                    )
                )

            # Replace the classification layer with a custom layer for feature extraction
            self.model.fc = nn.Linear(
                in_features=2048, out_features=self.n_features, bias=True
            )
        else:
            raise ValueError(
                "Invalid network: {}. Currently, only ResNet-50 is supported.".format(
                    self.network
                )
            )

    def forward_once(self, x):
        """
        Forward function for one branch to get the n_features-dim feature vector before merging.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        --------
        torch.Tensor
            Feature vector of shape (batch_size, n_features).
        """
        x = self.model(x)
        return x

    def forward(self, input1, input2):
        """
        Forward function that returns the feature vectors from each branch of the Siamese network.

        Parameters:
        -----------
        input1 : torch.Tensor
            Input tensor for the first branch of the Siamese network.
        input2 : torch.Tensor
            Input tensor for the second branch of the Siamese network.

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the feature vectors from each branch.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class StandardContrastiveLoss(nn.Module):
    """
    A standard contrastive loss function with a margin.

    Parameters:
    -----------
    margin : float, default=1.0
        The margin by which negatives should be farther from the anchor than positives.

    Methods:
    --------
    forward(anchor, positive, negatives):
        Computes the contrastive loss with the specified margin.
    """

    def __init__(self, margin=1.0):
        """
        Initializes the contrastive loss with a margin.

        Parameters:
        -----------
        margin : float, default=1.0
            The margin by which negatives should be farther from the anchor than positives.
        """
        super(StandardContrastiveLoss, self).__init__()
        self.margin = margin  # Set the margin

    def forward(self, anchor, positive, negatives):
        """
        Computes the contrastive loss with margin.

        Parameters:
        -----------
        anchor : torch.Tensor
            The embeddings for the anchor samples.
        positive : torch.Tensor
            The embeddings for the positive samples.
        negatives : list of torch.Tensor or torch.Tensor
            The embeddings for the negative samples.

        Returns:
        --------
        torch.Tensor
            The computed loss.
        """
        pos_dist = torch.sum(
            (anchor - positive) ** 2, dim=1
        )  # Squared Euclidean distance for positive pairs

        # Handle list of negative samples
        if isinstance(negatives, list):
            neg_dist = torch.stack(
                [torch.sum((anchor - neg) ** 2, dim=1) for neg in negatives]
            )
            neg_dist = torch.min(neg_dist, dim=0)[
                0
            ]  # Take the minimum distance across all negatives
        else:
            neg_dist = torch.sum(
                (anchor - negatives) ** 2, dim=1
            )  # Squared Euclidean distance for negative pairs

        # Compute loss with margin
        loss = torch.mean(
            torch.relu(pos_dist - neg_dist + self.margin)
        )  # Include the margin in the ReLU function

        return loss


class NTxentLoss(nn.Module):
    """
    The NT-Xent (Normalized Temperature-scaled Cross Entropy) loss function.

    This loss function is widely used in contrastive learning frameworks to learn embeddings
    by maximizing the agreement between positive pairs and minimizing it for negative pairs.

    Parameters:
    -----------
    temperature : float, default=0.5
        The temperature parameter that scales the similarity scores.
    device : str, default='cpu'
        The device to run the computations on, either 'cpu' or 'cuda'.
    eps : float, default=1e-8
        A small epsilon value to prevent division by zero in logarithm calculations.

    Methods:
    --------
    forward(anchor, positive, negatives):
        Computes the NT-Xent loss for a given set of anchor, positive, and negative embeddings.
    """

    def __init__(self, temperature=0.5, device="cpu", eps=1e-8):
        """
        Initializes the NT-Xent loss with the given temperature, device, and epsilon value.

        Parameters:
        -----------
        temperature : float, default=0.5
            The temperature parameter that scales the similarity scores.
        device : str, default='cpu'
            The device to run the computations on, either 'cpu' or 'cuda'.
        eps : float, default=1e-8
            A small epsilon value to prevent division by zero in logarithm calculations.
        """
        super(NTxentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.eps = eps

    def forward(self, anchor, positive, negatives):
        """
        Computes the NT-Xent loss for a given set of anchor, positive, and negative embeddings.

        Parameters:
        -----------
        anchor : torch.Tensor
            The embeddings for the anchor samples.
        positive : torch.Tensor
            The embeddings for the positive samples.
        negatives : list of torch.Tensor or torch.Tensor
            The embeddings for the negative samples.

        Returns:
        --------
        torch.Tensor
            The computed NT-Xent loss.
        """
        embeddings = torch.cat([anchor, positive] + negatives, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Masking setup to consider only relevant negatives
        batch_size = embeddings.shape[0]
        sim_mask = torch.eye(batch_size, device=self.device).bool()
        sim_matrix = sim_matrix.masked_fill(sim_mask, float("-inf"))

        # Positives are always in the first two indices
        positives = sim_matrix[:1, 1:2]  # Anchor vs Positive
        negatives = sim_matrix[:1, 2:]  # Anchor vs Negatives

        exp_pos = torch.exp(positives)
        exp_neg = torch.sum(torch.exp(negatives), dim=1, keepdim=True) + self.eps

        log_prob = torch.log(exp_pos / (exp_pos + exp_neg))
        loss = -torch.mean(log_prob)
        return loss


class ContrastiveDataset(Dataset):
    """
    A PyTorch Dataset class for loading images for contrastive learning.

    This dataset prepares anchor, positive, and negative samples for contrastive learning tasks.

    Parameters:
    -----------
    image_paths : list of str
        A list of file paths to the images.
    transform : tuple
        A tuple containing two transformations. The first is applied to all images, and the second is an augmentation applied only to the positive sample.
    num_negatives : int, default=5
        The number of negative samples to generate for each anchor.

    Methods:
    --------
    __len__():
        Returns the number of samples in the dataset.
    __getitem__(idx):
        Returns a set of transformed images consisting of an anchor, a positive (augmented anchor), and the specified number of negatives.
    """

    def __init__(self, image_paths, transform=None, num_negatives=5):
        """
        Initializes the ContrastiveDataset with image paths, transformations, and the number of negatives.

        Parameters:
        -----------
        image_paths : list of str
            A list of file paths to the images.
        transform : tuple
            A tuple containing two transformations. The first is applied to all images, and the second is an augmentation applied only to the positive sample.
        num_negatives : int, default=5
            The number of negative samples to generate for each anchor.
        """
        self.image_paths = image_paths
        self.transform, self.augmentation = transform[0], transform[1]
        self.num_negatives = num_negatives

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
        --------
        int
            The number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns a set of transformed images consisting of an anchor, a positive (augmented anchor), and the specified number of negatives.

        Parameters:
        -----------
        idx : int
            The index of the anchor image.

        Returns:
        --------
        torch.Tensor
            A tensor containing the transformed images (anchor, positive, and negatives).
        """
        image_path = self.image_paths[idx]
        anchor_image = Image.open(image_path).convert("RGB")

        negatives = []
        while len(negatives) < self.num_negatives:
            neg_idx = random.randint(0, len(self.image_paths) - 1)
            if neg_idx != idx:
                neg_image = Image.open(self.image_paths[neg_idx]).convert("RGB")
                negatives.append(neg_image)

        images = [anchor_image, anchor_image] + negatives
        if self.transform:
            images = [self.transform(img) for img in images]
        if self.augmentation:
            images[1] = self.augmentation(anchor_image)

        return torch.stack(images)


def training_loop(
    dataloader, model, criterion, optimizer, scheduler, device, epochs=5, save_dir="./"
):
    """
    Training loop for a Siamese network with contrastive loss.

    Parameters:
    -----------
    dataloader : DataLoader
        The dataloader providing batches of anchor, positive, and negative images.
    model : nn.Module
        The Siamese network model to be trained.
    criterion : nn.Module
        The loss function, such as contrastive loss.
    optimizer : Optimizer
        The optimizer for training the model.
    scheduler : _LRScheduler
        The learning rate scheduler.
    device : str or torch.device
        The device to run the training on ('cpu' or 'cuda').
    epochs : int, default=5
        The number of epochs to train the model.
    save_dir : str, default='./'
        The directory to save the best model checkpoints.

    Returns:
    --------
    best_model : nn.Module
        The trained model with the lowest loss.
    loss_history : list of float
        The history of losses per epoch.
    """
    loss_history = []
    best_loss = float("inf")  # Initialize best_loss to a very high value

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        for i, batch in pbar:
            batch = batch.to(device)
            anchor, positive = batch[:, 0, :, :], batch[:, 1, :, :]
            negatives = [batch[:, j, :, :] for j in range(2, batch.shape[1])]

            # Forward pass through the model
            anchor_emb, positive_emb = model(anchor, positive)
            negative_embs = [model.forward_once(n) for n in negatives]

            # Check for NaNs in the embeddings
            if torch.isnan(anchor_emb).any() or torch.isnan(positive_emb).any():
                print(
                    f"NaN detected in embeddings at batch {i + 1}, skipping backpropagation."
                )
                continue

            # Compute the loss
            loss = criterion(anchor_emb, positive_emb, negative_embs)
            if torch.isnan(loss):
                print(
                    f"NaN detected in loss at batch {i + 1}, skipping backpropagation."
                )
                continue

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update tqdm description with current loss and learning rate
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_description(
                f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Batch Loss: {loss.item():.4f}"
            )

        scheduler.step()  # Update the learning rate
        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)

        # Log the average loss at the end of each epoch
        print(
            f"Epoch [{epoch+1}/{epochs}], LR: {current_lr:.6f}, Average Loss: {epoch_loss:.4f}"
        )

        # Check if the current model is the best one; if so, save it
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss,
                "lr": current_lr,
            }
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model checkpoint to {best_model_path}")

    return model, loss_history


def plot_loss(loss_history, save_path):
    """
    Plots the training loss over epochs and saves the plot to a specified path.

    Parameters:
    -----------
    loss_history : list of float
        The history of losses per epoch.
    save_path : str
        The path to save the plot image (including filename and extension).

    Returns:
    --------
    None
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()

    print(f"Loss plot saved to {save_path}")


def run_training(
    image_paths,
    save_dir,
    epochs,
    num_negatives,
    temp,
    lr,
    loss,
    device="cpu",
    batch_size=32,
    image_size=64,
    pretrained=None,
):
    """
    Runs the training process for the contrastive learning model.

    Parameters:
    -----------
    image_paths : list of str
        List of paths to the images for training.
    save_dir : str
        Directory to save the trained model and loss plot.
    epochs : int
        Number of training epochs.
    num_negatives : int
        Number of negative samples to use per anchor.
    temp : float
        Temperature parameter for the NT-Xent loss.
    lr : float
        Learning rate for the optimizer.
    loss : str
        Type of loss function to use ('xent' for NT-Xent or 'contrastive' for standard contrastive loss).
    device : str
        Device to run the training on ('cpu' or 'cuda').
    batch_size : int
        Batch size for training.
    image_size : int
        Size to resize the images to.
    pretrained : str, optional
        Path to a pretrained model to load before training.

    Returns:
    --------
    model : nn.Module
        The trained model.
    """
    # Transformations and dataset loading
    transform, augmentations = get_transforms(image_size)
    dataset = ContrastiveDataset(
        image_paths, transform=[transform, augmentations], num_negatives=num_negatives
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(
        f"Initialized dataset with {len(dataset)} items. Batch size set to {batch_size}."
    )

    # Model setup
    if device.startswith("cuda"):
        device = "cuda"
    model = SiameseNetwork(in_channels=1).to(device)
    print(f"Model initialized and moved to {device}.")

    # Load pretrained model if specified
    current_epoch = 0
    if pretrained is not None:
        pretrained_model_dict = load_model_from_hub('molinamarc/syntheva','SSCN/best_model.pth')
        model.load_state_dict(pretrained_model_dict["model_state_dict"])
        print(pretrained_model_dict.keys())
        current_epoch = pretrained_model_dict['epoch']
        print(f"Loaded pretrained model from {pretrained}.")

    # Loss function setup
    if loss == "xent":
        criterion = NTxentLoss(temperature=temp, device=device)
        print(f"Using NT-Xent loss with temperature {temp}.")
    else:
        criterion = StandardContrastiveLoss(margin=50)
        print(f"Using standard contrastive loss with margin.")

    # Optimizer and scheduler setup
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    print(f"Optimizer and scheduler initialized. Learning rate starts at {lr}.")

    # Training loop
    if pretrained is not None and current_epoch < epochs:
        model, loss_history = training_loop(
            dataloader, model, criterion, optimizer, scheduler, device, epochs, save_dir
        )
        plot_loss(
            loss_history, os.path.join(save_dir, "history_sscn.png")
        )  # Plot the loss after training
        print("Training completed. Loss history plotted.")

    return model


def get_transforms(size):
    """
    Generates transformation pipelines for image preprocessing and augmentation.

    This function creates two sets of transformations:
    1. A base transformation that standardizes the input images.
    2. An augmentation transformation to apply to positive images, enhancing data diversity.

    Parameters:
    -----------
    size : int
        The target size (height and width) to resize the images to.

    Returns:
    --------
    tuple
        A tuple containing two `torchvision.transforms.Compose` objects:
        - The first for base transformations applied to both anchor and positive images.
        - The second for augmentations applied to positive images.
    """
    # Base transformations for both anchor and positive
    base_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.22]),
        ]
    )

    # Augmentations for the positive images; apply RGB transformations before converting to grayscale
    augmentations = transforms.Compose(
        [
            transforms.Resize((size, size)),  # Ensure all images are the same size
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2
            ),  # Adjust brightness and contrast
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.ToTensor(),  # Convert to tensor before applying grayscale
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.22]),
        ]
    )
    return base_transform, augmentations


def ss_net_train(
    train_database_dir,
    save_dir,
    epochs,
    num_negatives,
    temp,
    lr,
    loss,
    bs,
    size,
    device,
    pretrained,
):
    """
    Trains a Siamese Network for image similarity using contrastive learning.

    This function initializes the training process for a Siamese Network, which is used for image similarity
    learning via contrastive loss. It prepares the dataset, sets up the model, loss function, optimizer, and
    scheduler, and executes the training loop.

    Parameters:
    -----------
    train_database_dir : str
        The directory containing the training images.
    save_dir : str
        The directory where training checkpoints and logs will be saved.
    epochs : int
        The number of training epochs.
    num_negatives : int
        The number of negative samples to be generated for each anchor-positive pair.
    temp : float
        The temperature parameter for the NT-Xent loss function.
    lr : float
        The initial learning rate for the optimizer.
    loss : str
        The type of loss function to use ('xent' for NT-Xent loss, any other value for standard contrastive loss).
    bs : int
        The batch size for training.
    size : int
        The size (height and width) to resize the input images to.
    device : str
        The device to perform training on ('cpu' or 'cuda:0' for GPU).
    pretrained : str or None
        Path to a pretrained model checkpoint to initialize the model weights.

    Returns:
    --------
    torch.nn.Module
        The trained Siamese Network model.

    """
    image_paths = []

    # Walk through the directory tree
    for root, _, files in os.walk(train_database_dir):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                # Construct the full path to the image
                image_path = os.path.join(root, file)
                # Add the image path to the list
                image_paths.append(image_path)

    # Run the training process
    model = run_training(
        image_paths,
        save_dir,
        epochs,
        num_negatives,
        temp,
        lr,
        loss,
        batch_size=bs,
        device=device,
        image_size=size,
        pretrained=pretrained,
    )

    return model


def compute_embeddings(dataloader, model, device):
    """
    Computes embeddings for a dataset using a given model.

    This function calculates embeddings for a dataset using a provided model. It iterates through the
    dataloader, moves images to the specified device, and extracts embeddings using the model's
    forward_once method.

    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        The dataloader containing the dataset for which embeddings are to be computed.
    model : torch.nn.Module
        The model used to compute embeddings.
    device : str
        The device to perform computations on ('cpu' or 'cuda:0' for GPU).

    Returns:
    --------
    torch.Tensor
        Tensor containing the computed embeddings for the entire dataset.
    """
    embeddings = []
    for data in dataloader:
        images = data["image"].to(device)
        embedding = model.forward_once(images)  # Get embeddings
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def compute_pairwise_distance(embeddings1, embeddings2, metric="euclidean"):
    """
    Computes pairwise distances between two sets of embeddings.

    This function calculates pairwise distances between two sets of embeddings using either Euclidean
    distance or cosine similarity as the metric.

    Parameters:
    -----------
    embeddings1 : torch.Tensor
        Tensor containing embeddings for the first set of samples.
    embeddings2 : torch.Tensor
        Tensor containing embeddings for the second set of samples.
    metric : {'euclidean', 'cosine'}, optional
        The distance metric to be used. Default is 'euclidean'.

    Returns:
    --------
    torch.Tensor
        Tensor containing the pairwise distances between the samples.
        If the metric is 'euclidean', the distances are Euclidean distances.
        If the metric is 'cosine', the distances are cosine distances.

    Raises:
    -------
    ValueError
        If an unsupported metric is provided. Supported metrics are 'euclidean' and 'cosine'.
    """
    if metric == "euclidean":
        # Compute Euclidean pairwise distance
        diff = embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0)
        pairwise_dist = torch.sqrt(torch.sum(diff**2, dim=2))
    elif metric == "cosine":
        # Compute cosine similarity and convert it to a distance metric
        cosine_sim = F.cosine_similarity(
            embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=2
        )
        pairwise_dist = 1 - cosine_sim  # Convert similarity to distance
    else:
        raise ValueError("Unsupported metric: choose 'euclidean' or 'cosine'")
    return pairwise_dist


def compute_pairwise_distance_batch(
    model, dataloader1, dataloader2, device, metric="euclidean"
):
    """
    Computes pairwise distances between batches of embeddings from two different dataloaders.

    Parameters:
    -----------
    model : torch.nn.Module
        The model used for generating embeddings.
    dataloader1 : torch.utils.data.DataLoader
        Dataloader containing the first set of samples.
    dataloader2 : torch.utils.data.DataLoader
        Dataloader containing the second set of samples.
    device : str
        Device where the computations will be performed ('cpu' or 'cuda').
    metric : {'euclidean', 'cosine'}, optional
        The distance metric to be used. Default is 'euclidean'.

    Returns:
    --------
    torch.Tensor
        Tensor containing the pairwise distances between all pairs of samples from the two dataloaders.
        If the metric is 'euclidean', the distances are Euclidean distances.
        If the metric is 'cosine', the distances are cosine distances.

    Raises:
    -------
    ValueError
        If an unsupported metric is provided. Supported metrics are 'euclidean' and 'cosine'.
    """
    model.eval()
    distances = []  # This will store all the distances

    with torch.no_grad():
        # Precompute embeddings for dataloader2
        all_embeddings2 = []
        for batch2 in tqdm(dataloader2):
            images2 = batch2["image"].to(device)
            embeddings2 = model.forward_once(images2)
            all_embeddings2.append(embeddings2)
        all_embeddings2 = torch.cat(all_embeddings2, dim=0)

        for batch1 in tqdm(dataloader1):
            images1 = batch1["image"].to(device)
            embeddings1 = model.forward_once(images1)

            # Compute distances
            if metric == "euclidean":
                diff = embeddings1.unsqueeze(1) - all_embeddings2.unsqueeze(0)
                dist = torch.sqrt(torch.sum(diff**2, dim=2))
            elif metric == "cosine":
                cosine_sim = F.cosine_similarity(
                    embeddings1.unsqueeze(1), all_embeddings2.unsqueeze(0), dim=2
                )
                dist = 1 - cosine_sim
            else:
                raise ValueError(
                    "Unsupported metric provided. Supported metrics are 'euclidean' and 'cosine'."
                )

            distances.append(dist)

    return torch.cat(distances, dim=0)


def find_closest_samples_and_distances(pairwise_dist):
    """
    Finds the closest samples and their corresponding distances in pairwise distance matrix.

    This function finds the indices and distances of the closest samples for each sample in the pairwise distance matrix.

    Parameters:
    -----------
    pairwise_dist : torch.Tensor
        Tensor containing the pairwise distances between samples.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple containing two tensors:
            - min_indices: Tensor containing the indices of the closest samples for each row in the distance matrix.
            - min_distances: Tensor containing the corresponding distances of the closest samples for each row.

    """
    min_distances, min_indices = pairwise_dist.min(
        dim=1
    )  # Find the minimum distance and indices for each row
    return min_indices, min_distances


def plot_similarity_distribution(clos_val, clos_syn, save_dir, title):
    """
    Plots the distribution of similarity scores between validation and synthetic samples.

    This function plots the distribution of similarity scores (pairwise distances) between validation and synthetic samples,
    and calculates the mean values and overlap between the distributions.

    Parameters:
    -----------
    clos_val : torch.Tensor
        Tensor containing the similarity scores (distances) for validation samples.
    clos_syn : torch.Tensor
        Tensor containing the similarity scores (distances) for synthetic samples.
    save_dir : str
        Directory where the plot will be saved.
    title : str
        Title of the plot.
    """
    # Convert tensors to numpy arrays
    val_array = clos_val.detach().cpu().numpy()
    syn_array = clos_syn.detach().cpu().numpy()

    # Calculate the 95th percentile for the validation distances
    val_95th_percentile = np.percentile(val_array, 99)
    syn_95th_percentile = np.percentile(syn_array, 99)

    threshold = max(val_95th_percentile, syn_95th_percentile)

    # Filter arrays to include only values up to the 95th percentile
    val_array = val_array[val_array <= threshold]
    syn_array = syn_array[syn_array <= threshold]

    # Calculate means
    mean_val = np.mean(val_array)
    mean_syn = np.mean(syn_array)

    # Calculate overlap (using a simple histogram intersection method as an example)
    hist_val, bins_val = np.histogram(val_array, bins=30, density=True)
    hist_syn, bins_syn = np.histogram(syn_array, bins=30, density=True)
    overlap = np.sum(np.minimum(hist_val, hist_syn))

    # Plot distributions
    sns.histplot(val_array, bins=30, kde=True, color="blue", label="Validation")
    sns.histplot(syn_array, bins=30, kde=True, color="red", label="Synthetic")

    # Plot mean lines
    plt.axvline(
        mean_val, color="blue", linestyle="--", label=f"Mean Validation: {mean_val:.2f}"
    )
    plt.axvline(
        mean_syn, color="red", linestyle="--", label=f"Mean Synthetic: {mean_syn:.2f}"
    )

    # Set title and labels
    plt.title(title)
    plt.xlabel("Distance Metric")
    plt.ylabel("Frequency")

    # Add overlap to the legend
    plt.legend()  # title=f'Dataset (Overlap: {overlap:.2f})')

    # Save the plot
    plt.savefig(os.path.join(save_dir, title + ".png"), transparent=True)
    plt.close()


def get_dataloaders(
    train_database_dir, val_database_dir, syn_database_dir, batch_size, size
):
    """
    Returns dataloaders for training, validation, and synthetic datasets.

    This function creates dataloaders for the training, validation, and synthetic datasets using the specified directories
    and batch size. It applies the same transformations to all datasets.

    Parameters:
    -----------
    train_database_dir : str
        Directory containing the training dataset.
    val_database_dir : str
        Directory containing the validation dataset.
    syn_database_dir : str
        Directory containing the synthetic dataset.
    batch_size : int
        Batch size for the dataloaders.
    size : int
        Size to which the images will be resized.

    Returns:
    --------
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for the training dataset.
    val_dataloader : torch.utils.data.DataLoader
        Dataloader for the validation dataset.
    syn_dataloader : torch.utils.data.DataLoader
        Dataloader for the synthetic dataset.
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),  # Resize images if necessary
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.22]),
        ]
    )

    # Create datasets
    from evaluation.dataset import CustomImageFolderDataset

    train_dataset = CustomImageFolderDataset(
        root_dir=train_database_dir, size=size, transform=transform
    )
    val_dataset = CustomImageFolderDataset(
        root_dir=val_database_dir, size=size, transform=transform
    )
    syn_dataset = CustomImageFolderDataset(
        root_dir=syn_database_dir, size=size, transform=transform
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    syn_dataloader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, syn_dataloader


def plot_candidate_pairs(
    dataloaders, closest_indices, distances, save_dir, plot_samples=10
):
    """
    Plots candidate pairs of synthetic and real images based on their distances.

    This function visualizes pairs of synthetic and real images that are potential matches based on their distances.
    It selects a specified number of sample pairs with the smallest distances and plots them in a grid format.

    Parameters:
    -----------
    dataloaders : dict
        Dictionary containing dataloaders for the synthetic and real datasets.
    closest_indices : dict
        Dictionary containing the indices of the closest real images for each synthetic image.
    distances : dict
        Dictionary containing the distances between synthetic and real images.
    save_dir : str
        Directory path where the plots and JSON file will be saved.
    plot_samples : int, optional
        Number of sample pairs to plot (default is 10).

    Returns:
    --------
    within_2nd_percentile_count : int
        Count of synthetic images within the 2nd percentile of validation distances.
    """
    # Calculate the number of rows needed for 4 columns
    num_columns = 4
    fifth_percentile_val_dist = np.percentile(distances["real"], 2)
    within_2nd_percentile_indices = np.where(
        distances["synthetic"] <= fifth_percentile_val_dist
    )[0]

    # Ensure we plot at most `plot_samples` images
    num_samples_to_plot = min(len(within_2nd_percentile_indices), plot_samples)
    sorted_indices = np.argsort(distances["synthetic"][within_2nd_percentile_indices])[
        :num_samples_to_plot
    ]
    selected_indices = within_2nd_percentile_indices[sorted_indices]

    num_rows = int(np.ceil(num_samples_to_plot / num_columns))
    fig, axs = plt.subplots(num_rows, num_columns * 2, figsize=(20, 5 * num_rows))

    # Ensure axs is always a 2D array
    if num_rows == 1:
        axs = np.array([axs])
    else:
        axs = np.array(axs)

    data_for_json = []

    print(f"Total synthetic images: {len(dataloaders['synthetic'].dataset)}")
    print(f"Total real images: {len(dataloaders['real'].dataset)}")
    print(f"2nd Percentile of Validation Distances: {fifth_percentile_val_dist}")
    print(
        f"Synthetic images within the 2nd percentile: {len(within_2nd_percentile_indices)}"
    )
    print(f"Number of samples to plot: {num_samples_to_plot}")

    for i, index in enumerate(selected_indices):
        print(f"Index: {index}, Distance: {distances['synthetic'][index]}")

        try:
            syn_idx = index
            real_idx = closest_indices["synthetic"][index]

            synth_image = dataloaders["synthetic"].dataset[syn_idx]["image"]
            real_image = dataloaders["real"].dataset[real_idx]["image"]

            row, col = divmod(i, num_columns)
            axs[row, col * 2].imshow(synth_image.squeeze().cpu(), cmap="gray")
            axs[row, col * 2 + 1].imshow(real_image.squeeze().cpu(), cmap="gray")
            axs[row, col * 2].set_title(
                f"Synthetic Image {syn_idx} (Dist: {distances['synthetic'][syn_idx]:.2f})"
            )
            axs[row, col * 2 + 1].set_title(f"Real Image {real_idx}")
            axs[row, col * 2].axis("off")
            axs[row, col * 2 + 1].axis("off")

            data_for_json.append(
                {
                    "synthetic_image_index": int(syn_idx),
                    "real_image_index": int(real_idx),
                    "distance": round(float(distances["synthetic"][syn_idx]), 2),
                }
            )
        except Exception as e:
            print(f"Error: {e}. Synthetic index: {syn_idx}, Real index: {real_idx}")

    # Remove unused subplots
    for j in range(i + 1, num_rows * num_columns):
        row, col = divmod(j, num_columns)
        fig.delaxes(axs[row, col * 2])
        fig.delaxes(axs[row, col * 2 + 1])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "selected_candidates.png"))
    json_path = os.path.join(save_dir, "eligible_image_pairs.json")
    with open(json_path, "w") as f:
        json.dump(data_for_json, f, indent=4)
    print(f"JSON saved to {json_path}")

    return len(within_2nd_percentile_indices)


def distances_train_val_synth(
    model,
    train_database_dir,
    val_database_dir,
    syn_database_dir,
    save_dir,
    metric,
    batch_size,
    size,
    device,
):
    """
    Computes pairwise distances between validation and synthetic datasets based on a given metric,
    and plots candidate pairs of synthetic and real images.

    This function calculates the pairwise distances between validation and synthetic datasets
    using the provided model and specified metric. It then visualizes potential matches by
    plotting pairs of synthetic and real images with the smallest distances.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model used for computing embeddings.
    train_database_dir : str
        Directory containing the training dataset.
    val_database_dir : str
        Directory containing the validation dataset.
    syn_database_dir : str
        Directory containing the synthetic dataset.
    save_dir : str
        Directory path where the plots and evaluation results will be saved.
    metric : str
        Distance metric to compute pairwise distances ('euclidean' or 'cosine').
    batch_size : int
        Batch size for loading data during inference.
    size : int
        Size to resize images to before processing.
    device : str
        Device to run inference on ('cpu' or 'cuda').

    Returns:
    --------
    None
    """
    train_dataloader, val_dataloader, syn_dataloader = get_dataloaders(
        train_database_dir, val_database_dir, syn_database_dir, batch_size, size
    )
    print(
        f"Loaded datasets with sizes - Train: {len(train_dataloader.dataset)}, Validation: {len(val_dataloader.dataset)}, Synthetic: {len(syn_dataloader.dataset)}"
    )

    pairwise_dist_val = compute_pairwise_distance_batch(
        model, train_dataloader, val_dataloader, device, metric
    )
    pairwise_dist_syn = compute_pairwise_distance_batch(
        model, train_dataloader, syn_dataloader, device, metric
    )
    pairwise_dist_syn_syn = compute_pairwise_distance_batch(
        model, syn_dataloader, syn_dataloader, device, metric
    )

    print(
        f"Pairwise distances computed - Validation: {pairwise_dist_val.shape}, Synthetic: {pairwise_dist_syn.shape}"
    )

    closest_validation_indices, closest_validation_distances = (
        find_closest_samples_and_distances(pairwise_dist_val)
    )
    closest_synthetic_indices, closest_synthetic_distances = (
        find_closest_samples_and_distances(pairwise_dist_syn)
    )
    print(
        f"Closest indices - Validation: {closest_validation_indices[:10]}, Synthetic: {closest_synthetic_indices[:10]}"
    )
    print(
        f"Closest Distances - Validation: {closest_validation_distances[:10]}, Synthetic: {closest_synthetic_distances[:10]}"
    )

    plot_similarity_distribution(
        closest_validation_distances,
        closest_synthetic_distances,
        save_dir,
        f"Distribution of Closest Distances ({metric.capitalize()} Metric)",
    )

    within_2nd_percentile_count = plot_candidate_pairs(
        dataloaders={"synthetic": syn_dataloader, "real": train_dataloader},
        closest_indices={
            "synthetic": closest_synthetic_indices,
            "real": closest_validation_indices,
        },
        distances={
            "synthetic": closest_synthetic_distances.cpu().detach().numpy(),
            "real": closest_validation_distances.cpu().detach().numpy(),
        },
        save_dir=save_dir,
        plot_samples=10,
    )

    results_file_path = max(
        [
            os.path.join(save_dir, file)
            for file in os.listdir(save_dir)
            if file.startswith("eval_results")
        ],
        key=os.path.getctime,
    )

    with open(results_file_path, "r") as f:
        results = json.load(f)

    results.update(
        {
            "SSCN_Diversity": str(round(torch.mean(pairwise_dist_syn_syn).item(), 3)),
            "SSCN_Diversity std": str(
                round(torch.std(pairwise_dist_syn_syn).item(), 3)
            ),
            "E_Distance": str(
                round(np.mean(closest_synthetic_distances.cpu().detach().numpy()), 3)
            ),
            "E_Distance std": str(
                round(np.std(closest_synthetic_distances.cpu().detach().numpy()), 3)
            ),
            "Validation E_Distance": str(
                round(np.mean(closest_validation_distances.cpu().detach().numpy()), 3)
            ),
            "Validatlion E_Distance std": str(
                round(np.std(closest_validation_distances.cpu().detach().numpy()), 3)
            ),
            "Copy candidates": str(within_2nd_percentile_count),
        }
    )

    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)


def main(
    conf,
    train_db_dir,
    val_db_dir,
    synth_dir,
    save_dir,
    epochs,
    num_negatives,
    temp,
    lr,
    loss,
    bs,
    size,
    device,
    pretrained,
):
    """
    Main function to train a Siamese network and evaluate synthetic data.

    This function serves as the entry point for training a Siamese network on the provided training dataset,
    and then evaluates the synthetic data using the trained model.

    Parameters:
    -----------
    conf : dict
        Configuration parameters.
    train_db_dir : str
        Directory containing the training dataset.
    val_db_dir : str
        Directory containing the validation dataset.
    synth_dir : str
        Directory containing the synthetic dataset.
    save_dir : str
        Directory path where the plots and evaluation results will be saved.
    epochs : int
        Number of epochs for training the Siamese network.
    num_negatives : int
        Number of negative samples to use during training.
    temp : float
        Temperature parameter for NT-Xent loss.
    lr : float
        Learning rate for training the Siamese network.
    loss : str
        Loss function to use ('xent' for NT-Xent loss, otherwise 'standard').
    bs : int
        Batch size for loading data during training and evaluation.
    size : int
        Size to resize images to before processing.
    device : str
        Device to run training and evaluation on ('cpu' or 'cuda').
    pretrained : str
        Path to a pretrained model checkpoint (optional).

    Returns:
    --------
    None
    """
    model = ss_net_train(
        train_db_dir,
        save_dir,
        epochs,
        num_negatives,
        temp,
        lr,
        loss,
        bs,
        size,
        device,
        pretrained,
    )

    _ = distances_train_val_synth(
        model,
        train_db_dir,
        val_db_dir,
        synth_dir,
        save_dir,
        metric="euclidean",
        batch_size=bs,
        size=size,
        device=device,
    )
