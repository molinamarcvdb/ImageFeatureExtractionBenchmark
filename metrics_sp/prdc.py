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
    real_features = ensure_torch_tensor(real_features)
    fake_features = ensure_torch_tensor(fake_features)

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

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
    ).sum(dim=0).float().mean()

    coverage = (
        (distance_real_fake.min(dim=1)[0] < real_nearest_neighbour_distances)
        .float()
        .mean()
    )

    d = dict(
        precision=precision.item(),
        recall=recall.item(),
        density=density.item(),
        coverage=coverage.item(),
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
        ).item()

    return d
