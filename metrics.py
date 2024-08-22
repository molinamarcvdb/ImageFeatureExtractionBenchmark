import numpy as np
from scipy import linalg
import warnings
from typing import Tuple
from prdc import compute_prdc

class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert sigma1.shape == sigma2.shape, f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "FID calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))
        return fid

def compute_statistics(features: np.ndarray) -> FIDStatistics:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return FIDStatistics(mu, sigma)

def calculate_fid(features1: np.ndarray, features2: np.ndarray) -> float:
    stats1 = compute_statistics(features1)
    stats2 = compute_statistics(features2)
    return stats1.frechet_distance(stats2)

def calculate_metrics(ref_features: np.ndarray, sample_features: np.ndarray) -> dict:
    metrics = {}
    metrics['fid'] = calculate_fid(ref_features, sample_features)
    prdc_results = compute_prdc(ref_features, sample_features, nearest_k=5)
    metrics.update(prdc_results)
    return metrics
