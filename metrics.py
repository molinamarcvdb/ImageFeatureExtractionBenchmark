import numpy as np
from scipy import linalg
import warnings
from typing import Tuple
from prdc import compute_prdc
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["KernelInceptionDistance.plot"]

__doctest_requires__ = {("KernelInceptionDistance", "KernelInceptionDistance.plot"): ["torch_fidelity"]}

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



def maximum_mean_discrepancy(k_xx: Tensor, k_xy: Tensor, k_yy: Tensor) -> Tensor:
    """Adapted from `KID Score`_."""
    m = k_xx.shape[0]

    diag_x = torch.diag(k_xx)
    diag_y = torch.diag(k_yy)

    kt_xx_sums = k_xx.sum(dim=-1) - diag_x
    kt_yy_sums = k_yy.sum(dim=-1) - diag_y
    k_xy_sums = k_xy.sum(dim=0)

    kt_xx_sum = kt_xx_sums.sum()
    kt_yy_sum = kt_yy_sums.sum()
    k_xy_sum = k_xy_sums.sum()

    value = (kt_xx_sum + kt_yy_sum) / (m * (m - 1))
    value -= 2 * k_xy_sum / (m**2)
    return value


def poly_kernel(f1: Tensor, f2: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0) -> Tensor:
    """Adapted from `KID Score`_."""
    if gamma is None:
        gamma = 1.0 / f1.shape[1]
    return (f1 @ f2.T * gamma + coef) ** degree


def poly_mmd(
    f_real: Tensor, f_fake: Tensor, degree: int = 3, gamma: Optional[float] = None, coef: float = 1.0
) -> Tensor:
    """Adapted from `KID Score`_."""
    k_11 = poly_kernel(f_real, f_real, degree, gamma, coef)
    k_22 = poly_kernel(f_fake, f_fake, degree, gamma, coef)
    k_12 = poly_kernel(f_real, f_fake, degree, gamma, coef)
    return maximum_mean_discrepancy(k_11, k_12, k_22)


class KernelInceptionDistance(Metric):
    r"""Calculate Kernel Inception Distance (KID) which is used to access the quality of generated images.
    Torchmetrics implementation
    .. math::
        KID = MMD(f_{real}, f_{fake})^2

    where :math:`MMD` is the maximum mean discrepancy and :math:`I_{real}, I_{fake}` are extracted features
    from real and fake images, see `kid ref1`_ for more details. In particular, calculating the MMD requires the
    evaluation of a polynomial kernel function :math:`k`

    .. math::
        k(x,y) = (\gamma * x^T y + coef)^{degree}

    which controls the distance between two features. In practise the MMD is calculated over a number of
    subsets to be able to both get the mean and standard deviation of KID.

    Using the default feature extraction (Inception v3 using the original weights from `kid ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3xHxW)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.

    Using custom feature extractor is also possible. One can give a torch.nn.Module as `feature` argument. This
    custom feature extractor is expected to have output shape of ``(1, num_features)`` This would change the
    used feature extractor from default (Inception v3) to the given network. ``normalize`` argument won't have any
    effect and update method expects to have the tensor given to `imgs` argument to be in the correct shape and
    type that is compatible to the custom feature extractor.

    .. note:: using this metric with the default feature extractor requires that ``torch-fidelity``
        is installed. Either install as ``pip install torchmetrics[image]`` or
        ``pip install torch-fidelity``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``imgs`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor of shape ``(N,C,H,W)``
    - ``real`` (`bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``kid_mean`` (:class:`~torch.Tensor`): float scalar tensor with mean value over subsets
    - ``kid_std`` (:class:`~torch.Tensor`): float scalar tensor with standard deviation value over subsets

    Args:
        feature: Either an str, integer or ``nn.Module``:

            - an str or integer will indicate the inceptionv3 feature layer to choose. Can be one of the following:
              'logits_unbiased', 64, 192, 768, 2048
            - an ``nn.Module`` for using a custom feature extractor. Expects that its forward method returns
              an ``(N,d)`` matrix where ``N`` is the batch size and ``d`` is the feature size.

        subsets: Number of subsets to calculate the mean and standard deviation scores over
        subset_size: Number of randomly picked samples in each subset
        degree: Degree of the polynomial kernel function
        gamma: Scale-length of polynomial kernel. If set to ``None`` will be automatically set to the feature size
        coef: Bias term in the polynomial kernel.
        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``feature`` is set to an ``int`` (default settings) and ``torch-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in ``(64, 192, 768, 2048)``
        ValueError:
            If ``subsets`` is not an integer larger than 0
        ValueError:
            If ``subset_size`` is not an integer larger than 0
        ValueError:
            If ``degree`` is not an integer larger than 0
        ValueError:
            If ``gamma`` is neither ``None`` or a float larger than 0
        ValueError:
            If ``coef`` is not an float larger than 0
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.image.kid import KernelInceptionDistance
        >>> kid = KernelInceptionDistance(subset_size=50)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
        >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        >>> kid.update(imgs_dist1, real=True)
        >>> kid.update(imgs_dist2, real=False)
        >>> kid.compute()
        (tensor(0.0337), tensor(0.0023))

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    real_features: List[Tensor]
    fake_features: List[Tensor]
    inception: Module
    feature_network: str = "inception"

    def __init__(
        self,
        feature: Union[str, int, Module] = 2048,
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: Optional[float] = None,
        coef: float = 1.0,
        reset_real_features: bool = True,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `Kernel Inception Distance` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        self.used_custom_model = False

        if isinstance(feature, (str, int)):
            if not _TORCH_FIDELITY_AVAILABLE:
                raise ModuleNotFoundError(
                    "Kernel Inception Distance metric requires that `Torch-fidelity` is installed."
                    " Either install as `pip install torchmetrics[image]` or `pip install torch-fidelity`."
                )
            valid_int_input = ("logits_unbiased", 64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of {valid_int_input}, but got {feature}."
                )

            self.inception: Module = NoTrainInceptionV3(name="inception-v3-compat", features_list=[str(feature)])
        elif isinstance(feature, Module):
            self.inception = feature
            self.used_custom_model = True
        else:
            raise TypeError("Got unknown input to argument `feature`")

        if not (isinstance(subsets, int) and subsets > 0):
            raise ValueError("Argument `subsets` expected to be integer larger than 0")
        self.subsets = subsets

        if not (isinstance(subset_size, int) and subset_size > 0):
            raise ValueError("Argument `subset_size` expected to be integer larger than 0")
        self.subset_size = subset_size

        if not (isinstance(degree, int) and degree > 0):
            raise ValueError("Argument `degree` expected to be integer larger than 0")
        self.degree = degree

        if gamma is not None and not (isinstance(gamma, float) and gamma > 0):
            raise ValueError("Argument `gamma` expected to be `None` or float larger than 0")
        self.gamma = gamma

        if not (isinstance(coef, float) and coef > 0):
            raise ValueError("Argument `coef` expected to be float larger than 0")
        self.coef = coef

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        # states for extracted features
        self.add_state("real_features", [], dist_reduce_fx=None)
        self.add_state("fake_features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor, real: bool, feature: bool) -> None:
        """Update the state with extracted features.

        Args:
            imgs: Input img tensors to evaluate. If used custom feature extractor please
                make sure dtype and size is correct for the model.
            real: Whether given image is real or fake.

        """
        if feature:
            features = imgs
        else:
            imgs = (imgs * 255).byte() if self.normalize and (not self.used_custom_model) else imgs
            features = self.inception(imgs)

        if real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Calculate KID score based on accumulated extracted features from the two distributions.

        Implementation inspired by `Fid Score`_

        Returns:
            kid_mean (:class:`~torch.Tensor`): float scalar tensor with mean value over subsets
            kid_std (:class:`~torch.Tensor`): float scalar tensor with standard deviation value over subsets

        """
        real_features = dim_zero_cat(self.real_features)
        fake_features = dim_zero_cat(self.fake_features)

        n_samples_real = real_features.shape[0]
        if n_samples_real < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")
        n_samples_fake = fake_features.shape[0]
        if n_samples_fake < self.subset_size:
            raise ValueError("Argument `subset_size` should be smaller than the number of samples")

        kid_scores_ = []
        for _ in range(self.subsets):
            perm = torch.randperm(n_samples_real)
            f_real = real_features[perm[: self.subset_size]]
            perm = torch.randperm(n_samples_fake)
            f_fake = fake_features[perm[: self.subset_size]]

            o = poly_mmd(f_real, f_fake, self.degree, self.gamma, self.coef)
            kid_scores_.append(o)
        kid_scores = torch.stack(kid_scores_)
        return kid_scores.mean(), kid_scores.std(unbiased=False)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            # remove temporarily to avoid resetting
            value = self._defaults.pop("real_features")
            super().reset()
            self._defaults["real_features"] = value
        else:
            super().reset()


def MMD(x, y, kernel):
    """
    Empirical maximum mean discrepancy (MMD). The lower the result,
    the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P (numpy.ndarray or torch.Tensor)
        y: second sample, distribution Q (numpy.ndarray or torch.Tensor)
        kernel: kernel type such as "multiscale" or "rbf"
    """
    # Convert NumPy arrays to torch tensors if needed
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32).to(device)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.float32).to(device)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    elif kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    
    return torch.mean(XX + YY - 2. * XY).item()  # Convert the result to a Python float

def KID(ref_features, sample_features, set_size):
    # Initialize KID metric
    kid_metric = KernelInceptionDistance(subset_size=set_size)
    
    # Convert features to torch tensors if they are numpy arrays
    if isinstance(ref_features, np.ndarray):
        ref_features = torch.tensor(ref_features).to(device)#.to(torch.uint8)
    if isinstance(sample_features, np.ndarray):
        sample_features = torch.tensor(sample_features).to(device)#.to(torch.uint8)
    
    # Update KID metric with real and fake features
    kid_metric.update(ref_features, real=True, feature=True)
    kid_metric.update(sample_features, real=False, feature=True)
    
    # Compute KID mean and std
    kid_mean, kid_std = kid_metric.compute()

    return kid_mean.cpu().numpy(), kid_std.cpu().numpy()

def compute_statistics(features: np.ndarray) -> FIDStatistics:
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return FIDStatistics(mu, sigma)

def calculate_fid(features1: np.ndarray, features2: np.ndarray) -> float:
    stats1 = compute_statistics(features1)
    stats2 = compute_statistics(features2)
    return stats1.frechet_distance(stats2)

def compute_inception_score(activations: np.ndarray, split_size: int = 5000) -> float:
    """
    Compute the Inception Score (IS) for the given activations.

    Args:
        activations: The activations of the inception model, typically after the softmax layer (numpy.ndarray or torch.Tensor).
        split_size: The size of each split for computing the inception score.

    Returns:
        Inception Score (IS) as a float.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()

    # Small constant to avoid log(0)
    epsilon = 1e-10

    softmax_out = []
    for i in range(0, len(activations), split_size):
        part = activations[i: i + split_size]
        # Ensure the activations are in the [0,1] range and add epsilon to avoid log(0)
        part = np.clip(part, epsilon, 1 - epsilon)

        # Compute KL divergence
        mean_part = np.mean(part, axis=0)
        kl = part * (np.log(part + epsilon) - np.log(mean_part + epsilon))
        kl = np.mean(np.sum(kl, axis=1))
        
        softmax_out.append(np.exp(kl))
    
    is_score = np.mean(softmax_out)
    
    return float(is_score)

def calculate_metrics(ref_features: np.ndarray, sample_features: np.ndarray, set_size: int) -> dict:
    metrics = {}

    # Compute FID
    metrics['fid'] = calculate_fid(ref_features, sample_features)

    # Compute PRDC metrics
    prdc_results = compute_prdc(ref_features, sample_features, nearest_k=5)
    metrics.update(prdc_results)

    # Choose the kernel for MMD
    mmd_kernel = 'rbf'  # or 'multiscale'
    metrics['mmd'] = MMD(ref_features, sample_features, kernel=mmd_kernel)
    
    # IS
    metrics['is'] = compute_inception_score(sample_features, set_size)

    # KID
    metrics['kid'], _ = KID(ref_features, sample_features, set_size)
    print(metrics)
    return metrics

# Dummy function to simulate feature extraction
def generate_dummy_features(num_samples, feature_dim):
    """
    Generate random features to simulate extracted features from images.

    Args:
        num_samples (int): Number of samples (images).
        feature_dim (int): Dimension of the feature vector for each sample.

    Returns:
        np.ndarray: A (num_samples, feature_dim) array of random features.
    """
    return np.random.rand(num_samples, feature_dim)

# Test the calculate_metrics function
def test_calculate_metrics():
    # Simulate features for real and generated datasets
    num_samples = 1000  # Number of images
    feature_dim = 2048   # Dimension of the feature vector (e.g., output of a CNN layer)

    real_features = generate_dummy_features(num_samples, feature_dim)
    generated_features = generate_dummy_features(num_samples, feature_dim)

    # Calculate metrics
    metrics = calculate_metrics(real_features, generated_features)

    # Print the results
    print("Calculated Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

# Run the test
# test_calculate_metrics()
