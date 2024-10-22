import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Union, Optional

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


class VendiMetric(BaseMetric):
    def __init__(self, device, q=1, normalize=True, kernel="linear"):
        super().__init__(device)
        self.q = q
        self.normalize = normalize
        self.kernel = kernel

    def compute_statistics(self, features: torch.Tensor) -> torch.Tensor:
        """Simply return features as statistics."""
        return features

    def update_statistics(
        self, old_stats: torch.Tensor, new_features: torch.Tensor, num_images: int
    ) -> torch.Tensor:
        """Concatenate new features with old ones."""
        return torch.cat([old_stats, new_features], dim=0)

    def entropy_q(self, p: torch.Tensor, q: Union[float, int] = 1) -> torch.Tensor:
        """
        Compute q-entropy while maintaining gradients.

        Args:
            p: Eigenvalues tensor
            q: Order of entropy
        Returns:
            q-entropy tensor
        """
        # Keep only positive eigenvalues
        p = p[p > 0]

        if q == 1:
            return -(p * torch.log(p)).sum()
        if q == float("inf"):
            return -torch.log(torch.max(p))
        return torch.log((p**q).sum()) / (1 - q)

    def compute_kernel_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix while maintaining gradients.

        Args:
            X: Input features [N, D]
        Returns:
            Kernel matrix [N, N]
        """
        if self.normalize:
            X = F.normalize(X, p=2, dim=1)

        if self.kernel == "linear":
            S = torch.mm(X, X.t())
        elif self.kernel == "polynomial":
            S = (torch.mm(X, X.t()) + 1) ** 3
        else:
            raise NotImplementedError(f"Kernel {self.kernel} not implemented")

        return S

    def compute_vendi_score(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute Vendi score while maintaining gradients.

        Args:
            X: Input features [N, D]
        Returns:
            Vendi score as a tensor with gradients
        """
        X = X.to(self.device)
        n = X.shape[0]

        # Compute kernel matrix
        S = self.compute_kernel_matrix(X)

        # Compute eigenvalues
        w = torch.linalg.eigvalsh(S / n)

        # Compute entropy and exponential
        return torch.exp(self.entropy_q(w, q=self.q))

    def compute_per_class_vendi_scores(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-class Vendi scores while maintaining gradients.

        Args:
            features: Input features [N, D]
            labels: Class labels [N]
        Returns:
            Tensor of per-class Vendi scores
        """
        num_classes = len(torch.unique(labels))
        vendi_per_class = []

        for i in range(num_classes):
            features_class = features[labels == i]
            if len(features_class) > 0:  # Check if class has samples
                score = self.compute_vendi_score(features_class)
                vendi_per_class.append(score)
            else:
                vendi_per_class.append(torch.tensor(0.0, device=self.device))

        return torch.stack(vendi_per_class)

    def compute_loss(
        self,
        real_stats: torch.Tensor,
        gen_stats: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Vendi score(s) as a loss.

        Args:
            real_stats: Real features (unused in Vendi score)
            gen_stats: Generated features to compute Vendi score for
            labels: Optional class labels for per-class computation
        Returns:
            Vendi score(s) as a differentiable tensor
        """
        if labels is not None:
            return self.compute_per_class_vendi_scores(gen_stats, labels)
        return self.compute_vendi_score(gen_stats)


# Factory function
def create_vendi_metric(
    device: str, q: float = 1, normalize: bool = True, kernel: str = "linear"
) -> VendiMetric:
    return VendiMetric(device, q, normalize, kernel)
