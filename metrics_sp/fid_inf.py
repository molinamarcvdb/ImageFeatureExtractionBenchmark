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


import torch
import torch.nn as nn
from typing import Dict, Any, Union
import torch.linalg as LA
from tqdm import tqdm

import torch
import torch.nn as nn
from typing import Dict, Any, Union
import torch.linalg as LA
from tqdm import tqdm


class FIDInfinityMetric(BaseMetric):
    def __init__(self, device, num_points=15, min_batch=5000):
        super().__init__(device)
        self.num_points = num_points
        self.min_batch = min_batch
        self.batch_sizes = None

    def ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 2D."""
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def compute_statistics(
        self, features: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute mean and covariance statistics."""
        if isinstance(features, dict):
            features = features.get(
                "features", features.get("mean", features.get("output", None))
            )
            if features is None:
                raise ValueError(
                    f"Could not find features in dictionary. Keys: {list(features.keys())}"
                )

        features = self.ensure_2d(features)

        # Ensure features are 2D: [N, D]
        if features.dim() == 1:
            features = features.unsqueeze(0)

        mean = torch.mean(features, dim=0)
        cov = self.compute_cov(features)

        # Ensure proper dimensions
        mean = mean.view(1, -1) if mean.dim() == 1 else mean
        return {"mean": mean, "cov": cov}

    def compute_cov(self, x: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix in a numerically stable way."""
        # Ensure input is 2D
        x = self.ensure_2d(x)

        # Center the features
        mean = torch.mean(x, dim=0, keepdim=True)
        x_centered = x - mean

        # Compute covariance
        n = x.size(0)
        cov = (x_centered.T @ x_centered) / max(1, n - 1)

        # Ensure the covariance matrix is 2D
        if cov.dim() == 0:
            cov = cov.view(1, 1)
        elif cov.dim() == 1:
            cov = cov.view(-1, 1)

        return cov

    def update_statistics(
        self,
        old_stats: Dict[str, torch.Tensor],
        new_features: torch.Tensor,
        num_images: int,
    ) -> Dict[str, torch.Tensor]:
        """Update running statistics with new batch."""
        if isinstance(new_features, dict):
            new_features = new_features.get(
                "features", new_features.get("mean", new_features.get("output"))
            )

        new_features = self.ensure_2d(new_features)
        old_mean = old_stats["mean"]
        if old_mean.dim() == 1:
            old_mean = old_mean.unsqueeze(0)

        delta = (new_features - old_mean) / num_images
        new_mean = old_stats["mean"] + delta.mean(dim=0)

        # Ensure new_mean is 2D
        new_mean = new_mean.view(1, -1) if new_mean.dim() == 1 else new_mean

        x_centered = new_features - new_mean
        new_cov = (
            old_stats["cov"] * (num_images - 1) + x_centered.T @ x_centered
        ) / num_images

        return {"mean": new_mean, "cov": new_cov}

    def compute_frechet_distance(
        self, stats1: Dict[str, torch.Tensor], stats2: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute Frechet distance between two sets of statistics."""
        m1 = stats1["mean"]
        m2 = stats2["mean"]
        s1 = stats1["cov"]
        s2 = stats2["cov"]

        # Ensure proper dimensions
        m1 = m1.view(-1) if m1.dim() > 1 else m1
        m2 = m2.view(-1) if m2.dim() > 1 else m2

        # Compute mean term
        mean_diff = m1 - m2
        mean_term = torch.sum(mean_diff * mean_diff)

        # Ensure covariance matrices are 2D
        s1 = s1.view(s1.size(-1), s1.size(-1)) if s1.dim() != 2 else s1
        s2 = s2.view(s2.size(-1), s2.size(-1)) if s2.dim() != 2 else s2

        # Compute matrix product and eigenvalues
        try:
            product = torch.matmul(s2, s1)
            eigenvalues = LA.eigvals(product).real

            # Compute trace terms
            trace_term = torch.trace(s1) + torch.trace(s2)
            sqrt_term = 2 * torch.sum(torch.sqrt(torch.abs(eigenvalues) + 1e-16))

            return mean_term + trace_term - sqrt_term

        except RuntimeError as e:
            print(
                f"Error in FID computation. Shapes: m1={m1.shape}, m2={m2.shape}, s1={s1.shape}, s2={s2.shape}"
            )
            raise e

    def compute_loss(
        self,
        real_stats: Dict[str, torch.Tensor],
        gen_stats: Union[Dict[str, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute FID-infinity loss."""
        # Extract features from gen_stats
        if isinstance(gen_stats, dict):
            gen_features = gen_stats.get(
                "features", gen_stats.get("mean", gen_stats.get("output"))
            )
        else:
            gen_features = gen_stats

        gen_features = self.ensure_2d(gen_features)

        # Initialize batch sizes if not already done
        if self.batch_sizes is None:
            self.batch_sizes = self.get_batch_sizes(len(gen_features))

        # Compute FID for different batch sizes
        fids = []
        for batch_size in self.batch_sizes:
            # Sample batch of fake features
            indices = torch.randperm(len(gen_features))[:batch_size]
            gen_batch = gen_features[indices]

            # Compute statistics for generated batch
            gen_batch_stats = self.compute_statistics(gen_batch)

            # Compute FID
            fid = self.compute_frechet_distance(real_stats, gen_batch_stats)
            fids.append(fid)

        # Stack FIDs and prepare for regression
        fids = torch.stack(fids).unsqueeze(1)
        x = (1 / self.batch_sizes.float()).unsqueeze(1)

        # Compute FID-infinity through linear regression
        fid_infinity = self.linear_regression(x, fids)

        return fid_infinity

    def get_batch_sizes(self, total_size: int) -> torch.Tensor:
        """Compute batch sizes for FID evaluation."""
        min_size = min(self.min_batch, max(total_size // 10, 2))
        return torch.linspace(
            min_size, total_size, self.num_points, device=self.device
        ).long()

    def linear_regression(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform linear regression to compute FID-infinity."""
        X = torch.cat([x, torch.ones_like(x)], dim=1)
        weights = LA.solve(X.t() @ X, X.t() @ y.float())
        return weights[1]
