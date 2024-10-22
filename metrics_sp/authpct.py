import torch
import torch.nn as nn
from typing import Dict, Any
import torch.linalg as LA
from tqdm import tqdm


class BaseMetric(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def compute_statistics(self, features):
        raise NotImplementedError

    def update_statistics(self, old_stats, new_stats, num_images):
        raise NotImplementedError

    def compute_loss(self, real_stats, gen_stats):
        raise NotImplementedError


def compute_authpct(train_feat, gen_feat, device="cuda"):
    # Ensure inputs are PyTorch tensors with gradients enabled
    train_feat = torch.tensor(
        train_feat, dtype=torch.float32, device=device, requires_grad=True
    )
    gen_feat = torch.tensor(
        gen_feat, dtype=torch.float32, device=device, requires_grad=True
    )

    real_dists = torch.cdist(train_feat, train_feat)

    # Use a large value instead of infinity to avoid potential numerical issues
    real_dists.fill_diagonal_(1e10)
    gen_dists = torch.cdist(train_feat, gen_feat)

    real_min_dists, real_min_indices = real_dists.min(axis=0)
    gen_min_dists, gen_min_indices = gen_dists.min(dim=0)

    # For every synthetic point, find its closest real point, d1
    # Then, for that real point, find its closest real point(not itself), d2
    # if d2<d1, then it's authentic
    authen = real_min_dists[gen_min_indices] < gen_min_dists
    authpct = 100 * torch.sum(authen.float()) / len(authen)

    return authpct


import torch
import torch.nn as nn
from typing import Dict, Union, Optional


class AuthPctMetric(BaseMetric):
    def __init__(self, device):
        super().__init__(device)

    def compute_statistics(self, features: torch.Tensor) -> torch.Tensor:
        """Simply return features as statistics."""
        return (
            features.detach()
        )  # Detach real features stats as they don't need gradients

    def update_statistics(
        self, old_stats: torch.Tensor, new_features: torch.Tensor, num_images: int
    ) -> torch.Tensor:
        """Concatenate new features with old ones."""
        return torch.cat([old_stats, new_features], dim=0)

    def compute_loss(
        self, real_stats: torch.Tensor, gen_stats: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute AuthPct score as a differentiable loss.

        Args:
            real_stats: Real features [N, D]
            gen_stats: Generated features [M, D]
        Returns:
            AuthPct score as a differentiable tensor
        """
        # Ensure real features don't require gradients (they're fixed)
        real_features = real_stats.detach().to(self.device)

        # Ensure generated features require gradients
        if not gen_stats.requires_grad:
            gen_features = gen_stats.clone().requires_grad_(True)
        else:
            gen_features = gen_stats
        gen_features = gen_features.to(self.device)

        # Compute pairwise distances while maintaining gradients
        real_dists = torch.cdist(real_features, real_features)
        real_dists.fill_diagonal_(1e10)
        gen_dists = torch.cdist(real_features, gen_features)

        # Find minimum distances
        real_min_dists, _ = real_dists.min(dim=0)
        gen_min_dists, gen_min_indices = gen_dists.min(dim=0)

        # Compare distances using differentiable operations
        d2 = real_min_dists[gen_min_indices]  # distances between real pairs
        d1 = gen_min_dists  # distances between real and generated pairs

        # Compute authenticity using smooth approximation
        temperature = 0.1  # Controls smoothness of the approximation
        authen = torch.sigmoid((d2 - d1) / temperature)

        # Compute percentage while maintaining gradients
        authpct = 100 * torch.mean(authen)

        return -authpct  # Return negative for minimization

    def get_metric_value(
        self, real_stats: torch.Tensor, gen_stats: torch.Tensor
    ) -> float:
        """
        Compute the actual metric value without gradients.
        Returns positive score (not negated).
        """
        with torch.no_grad():
            return -self.compute_loss(real_stats, gen_stats).item()


# Factory function
def create_authpct_metric(device: str) -> AuthPctMetric:
    return AuthPctMetric(device)
