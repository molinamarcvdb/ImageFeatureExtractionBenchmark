import numpy as np
import torch
from abc import ABC, abstractmethod
from scipy import linalg
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
from enum import Enum
from metrics_sp.fls import compute_fls_overfit, compute_fls

# from metrics_sp.vendi import compute_vendi_score
from metrics_sp.authpct import compute_authpct
from metrics_sp.sw import sw_approx
from metrics_sp.prdc import compute_prdc, PRDCLoss
from metrics import KID


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


class FIDMetric(BaseMetric):
    def compute_statistics(self, features):
        mean = torch.mean(features, dim=0)
        cov = self.compute_cov(features)
        return {"mean": mean, "cov": cov}

    def compute_loss(self, real_stats, gen_stats):
        m1, s1 = real_stats["mean"], real_stats["cov"]
        m2, s2 = gen_stats["mean"], gen_stats["cov"]

        mean_term = torch.sum(torch.square(m1 - m2.squeeze(0)))

        # Compute eigenvalues directly without full eigendecomposition
        eigenvalues = torch.linalg.eigvals(torch.matmul(s2, s1)).real

        # Compute trace terms and sum of square roots of eigenvalues
        trace_term = torch.trace(s1) + torch.trace(s2)
        sqrt_term = 2 * torch.sum(torch.sqrt(torch.abs(eigenvalues) + 1e-16))

        fid = mean_term + trace_term - sqrt_term

        return fid

    def update_statistics(self, old_stats, new_features, num_images):
        delta = (new_features - old_stats["mean"].unsqueeze(0)) / num_images
        new_mean = old_stats["mean"] + delta.squeeze(0)
        new_cov = (
            old_stats["cov"] * (num_images - 1)
            + torch.matmul(
                (new_features - new_mean).t(),
                new_features - old_stats["mean"].unsqueeze(0),
            )
        ) / (num_images - 1)
        return {"mean": new_mean, "cov": new_cov}

    def compute_cov(self, x):
        x_centered = x - x.mean(dim=0, keepdim=True)
        return (x_centered.T @ x_centered) / (x.size(0) - 1)


class PRDCMetricType(Enum):
    PRECISION = "precision"
    RECALL = "recall"
    DENSITY = "density"
    COVERAGE = "coverage"
    REALISM = "realism"


class PRDCBaseMetric(BaseMetric):
    def __init__(self, device, metric_type: PRDCMetricType, nearest_k=5):
        super().__init__(device)
        self.nearest_k = nearest_k
        self.metric_type = metric_type
        self.criterion = PRDCLoss(nearest_k=nearest_k, metric=metric_type.value)

    def compute_statistics(self, features):
        return features

    def update_statistics(self, old_stats, new_features, num_images):
        updated_features = torch.cat([old_stats, new_features], dim=0)
        return updated_features

    def compute_loss(self, real_stats, gen_stats):
        return self.criterion(real_stats, gen_stats)


class PrecisionMetric(PRDCBaseMetric):
    def __init__(self, device, nearest_k=5):
        super().__init__(device, PRDCMetricType.PRECISION, nearest_k)


class RecallMetric(PRDCBaseMetric):
    def __init__(self, device, nearest_k=5):
        super().__init__(device, PRDCMetricType.RECALL, nearest_k)


class DensityMetric(PRDCBaseMetric):
    def __init__(self, device, nearest_k=5):
        super().__init__(device, PRDCMetricType.DENSITY, nearest_k)


class CoverageMetric(PRDCBaseMetric):
    def __init__(self, device, nearest_k=5):
        super().__init__(device, PRDCMetricType.COVERAGE, nearest_k)


class RealismMetric(PRDCBaseMetric):
    def __init__(self, device, nearest_k=5):
        super().__init__(device, PRDCMetricType.REALISM, nearest_k)


class MMDMetric(BaseMetric):
    def __init__(self, device, kernel="rbf"):
        super().__init__(device)
        self.kernel = kernel

    def compute_statistics(self, features):
        return {"features": features}

    def compute_loss(self, real_stats, gen_stats):
        real_features = real_stats["features"]
        gen_features = gen_stats["features"]

        real_real = self.kernel_function(real_features, real_features)
        real_gen = self.kernel_function(real_features, gen_features)
        gen_gen = self.kernel_function(gen_features, gen_features)

        mmd = real_real.mean() + gen_gen.mean() - 2 * real_gen.mean()
        return mmd

    def update_statistics(self, old_stats, new_features, num_images):
        updated_features = torch.cat([old_stats["features"], new_features], dim=0)
        return {"features": updated_features}

    def kernel_function(self, x, y):
        if self.kernel == "rbf":
            return torch.exp(-torch.cdist(x, y, p=2) / (2 * x.shape[1]))
        elif self.kernel == "multiscale":
            scales = torch.tensor([0.2, 0.5, 0.9, 1.3], device=self.device)
            return torch.sum(
                torch.stack(
                    [torch.exp(-torch.cdist(x, y, p=2) / (2 * s)) for s in scales]
                ),
                dim=0,
            )


class ISMetric(BaseMetric):
    def compute_statistics(self, features):
        return {"features": features}

    def compute_loss(self, real_stats, gen_stats):
        gen_features = gen_stats["features"]
        scores = torch.nn.functional.softmax(gen_features, dim=1)
        p_y = torch.mean(scores, dim=0)
        E_p_y = torch.sum(-p_y * torch.log(p_y + 1e-16))
        E_p_yx = torch.mean(torch.sum(-scores * torch.log(scores + 1e-16), dim=1))
        return torch.exp(E_p_y - E_p_yx)

    def update_statistics(self, old_stats, new_features, num_images):
        updated_features = torch.cat([old_stats["features"], new_features], dim=0)
        return {"features": updated_features}


class KIDMetric(BaseMetric):
    def compute_statistics(self, features):
        return features

    def compute_loss(self, real_stats, gen_stats):
        real_features = real_stats
        gen_features = gen_stats
        kid, _ = KID(real_features, gen_features, gen_features.shape[0] // 10)

        return kid

    def update_statistics(self, old_stats, new_features, num_images):
        updated_features = torch.cat([old_stats, new_features], dim=0)
        return updated_features


class VendiMetric(BaseMetric):
    def compute_statistics(self, features):
        return features

    def compute_loss(self, real_stats, gen_stats):
        vendi_score = compute_vendi_score(gen_stats)
        return vendi_score

    def update_statistics(self, old_stats, new_features, num_images):
        updated_features = torch.cat([old_stats, new_features], dim=0)
        return updated_features


class AuthPctMetric(BaseMetric):
    def compute_statistics(self, features):
        return features

    def compute_loss(self, real_stats, gen_stats):
        auth_pct = compute_authpct(real_stats, gen_stats)
        return torch.tensor(auth_pct, device=self.device)

    def update_statistics(self, old_stats, new_features, num_images):
        updated_features = torch.cat([old_stats, new_features], dim=0)
        return updated_features


class SWMetric(BaseMetric):
    def compute_statistics(self, features):
        return features

    def compute_loss(self, real_stats, gen_stats):
        sw = sw_approx(real_stats, gen_stats)
        return sw

    def update_statistics(self, old_stats, new_features, num_images):
        updated_features = torch.cat([old_stats, new_features], dim=0)
        return updated_features


# Usage examples:
# fid_metric = FIDMetric(device)
# precision_metric = PrecisionMetric(device, nearest_k=5)
# recall_metric = RecallMetric(device, nearest_k=5)
# density_metric = DensityMetric(device, nearest_k=5)
# coverage_metric = CoverageMetric(device, nearest_k=5)
# mmd_metric = MMDMetric(device, kernel='rbf')
# is_metric = ISMetric(device)
# kid_metric = KIDMetric(device)
# vendi_metric = VendiMetric(device)
# auth_pct_metric = AuthPctMetric(device)
# sw_metric = SWMetric(device)
