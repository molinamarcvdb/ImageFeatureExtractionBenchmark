"""
prdc from https://github.com/clovaai/generative-evaluation-prdc
Copyright (c) 2020-present NAVER Corp.
MIT license
Modified to also report realism score from https://arxiv.org/abs/1904.06991
"""
import numpy as np
import sklearn.metrics
import sys
import torch

import torch
import sys

__all__ = ["compute_prdc"]


def ensure_torch_tensor(data, device="cuda"):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    else:
        raise TypeError("Input must be either a PyTorch tensor or a numpy array")


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: torch.Tensor([N, feature_dim], dtype=torch.float32)
        data_y: torch.Tensor([N, feature_dim], dtype=torch.float32)
    Returns:
        torch.Tensor([N, N], dtype=torch.float32) of pairwise distances.
    """
    data_x = ensure_torch_tensor(data_x)
    if data_y is None:
        data_y = data_x
    else:
        data_y = ensure_torch_tensor(data_y)
    return torch.cdist(data_x, data_y)


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: torch.Tensor of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    return torch.kthvalue(unsorted, k, dim=axis)[0]


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k, realism=False):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    # real_features = ensure_torch_tensor(real_features)
    # fake_features = ensure_torch_tensor(fake_features)

    print(
        f"Num real: {real_features.shape[0]} Num fake: {fake_features.shape[0]}",
        file=sys.stderr,
    )

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k
    )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k
    )
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
        (distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1))
        .any(dim=0)
        .float()
        .mean()
    )
    recall = (
        (distance_real_fake < fake_nearest_neighbour_distances.unsqueeze(0))
        .any(dim=1)
        .float()
        .mean()
    )
    print(precision)

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
    ).sum(dim=0).float().mean()

    coverage = (
        (distance_real_fake.min(dim=1)[0] < real_nearest_neighbour_distances)
        .float()
        .mean()
    )

    d = dict(
        precision=precision,
        recall=recall,
        density=density,
        coverage=coverage,
    )

    if realism:
        mask = real_nearest_neighbour_distances < torch.median(
            real_nearest_neighbour_distances
        )
        d["realism"] = torch.mean(
            (
                real_nearest_neighbour_distances[mask].unsqueeze(1)
                / distance_real_fake[mask]
            ).max(dim=0)[0]
        )
    print(d)

    return d


import torch
import numpy as np
import sys


class PRDCLoss(torch.nn.Module):
    def __init__(self, nearest_k=5, metric="precision"):
        """
        Initialize PRDC Loss with specific metric

        Args:
            nearest_k: int, number of nearest neighbors
            metric: str, one of ['precision', 'recall', 'density', 'coverage', 'realism']
        """
        super().__init__()
        self.nearest_k = nearest_k
        if metric not in ["precision", "recall", "density", "coverage", "realism"]:
            raise ValueError(
                f"Metric must be one of ['precision', 'recall', 'density', 'coverage', 'realism'], got {metric}"
            )
        self.metric = metric

    def compute_pairwise_distance(self, data_x, data_y=None):
        """Compute pairwise distances while maintaining gradient information"""
        if data_y is None:
            data_y = data_x

        # Compute squared Euclidean distance using explicit operations to maintain gradients
        x_norm = (data_x**2).sum(1).view(-1, 1)
        y_norm = (data_y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(data_x, data_y.t())
        # Ensure no negative distances due to numerical errors
        dist = torch.clamp(dist, min=0.0)
        dist = torch.sqrt(
            dist + 1e-8
        )  # Add small epsilon to avoid numerical instability
        return dist

    def compute_nearest_neighbour_distances(self, input_features):
        """Compute k-nearest neighbor distances with gradient tracking"""
        distances = self.compute_pairwise_distance(input_features)
        # Sort distances and get kth value
        sorted_distances, _ = torch.sort(distances, dim=-1)
        # Add 1 to k because the closest point is always itself
        radii = sorted_distances[:, self.nearest_k]
        return radii

    def forward(self, real_features, fake_features):
        """
        Compute single PRDC metric as a differentiable loss

        Args:
            real_features: torch.Tensor([N, feature_dim])
            fake_features: torch.Tensor([N, feature_dim])
        Returns:
            Single metric loss value
        """
        # Ensure inputs have gradients enabled
        real_features = (
            real_features.detach()
        )  # We don't need gradients for real features
        fake_features = (
            fake_features.requires_grad_()
            if not fake_features.requires_grad
            else fake_features
        )

        real_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            real_features
        )
        fake_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            fake_features
        )
        distance_real_fake = self.compute_pairwise_distance(
            real_features, fake_features
        )

        if self.metric == "precision":
            # Use smooth approximation of indicator function
            temp = 0.1  # Temperature parameter for smoothing
            indicators = torch.sigmoid(
                (real_nearest_neighbour_distances.unsqueeze(1) - distance_real_fake)
                / temp
            )
            loss = torch.mean(torch.max(indicators, dim=0)[0])

        elif self.metric == "recall":
            temp = 0.1
            indicators = torch.sigmoid(
                (fake_nearest_neighbour_distances.unsqueeze(0) - distance_real_fake)
                / temp
            )
            loss = torch.mean(torch.max(indicators, dim=1)[0])

        elif self.metric == "density":
            temp = 0.1
            indicators = torch.sigmoid(
                (real_nearest_neighbour_distances.unsqueeze(1) - distance_real_fake)
                / temp
            )
            loss = (1.0 / float(self.nearest_k)) * torch.mean(
                torch.sum(indicators, dim=0)
            )

        elif self.metric == "coverage":
            temp = 0.1
            min_distances = torch.min(distance_real_fake, dim=1)[0]
            indicators = torch.sigmoid(
                (real_nearest_neighbour_distances - min_distances) / temp
            )
            loss = torch.mean(indicators)

        elif self.metric == "realism":
            mask = real_nearest_neighbour_distances < torch.median(
                real_nearest_neighbour_distances
            )
            ratios = real_nearest_neighbour_distances[mask].unsqueeze(1) / (
                distance_real_fake[mask] + 1e-8
            )
            loss = torch.mean(torch.max(ratios, dim=0)[0])

        return -loss  # Return negative for minimization

    def get_metric_value(self, real_features, fake_features):
        """Get the raw metric value without gradients"""
        with torch.no_grad():
            return -self.forward(real_features, fake_features)
