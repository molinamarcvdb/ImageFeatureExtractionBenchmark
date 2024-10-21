"""

Obtained from https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/heatmaps/
"""


from typing import Any, List
import cv2
import math
import PIL
import numpy as np
import torch
import time
import torch.nn.functional as F

from saliency.metrics_grad import (
    FIDMetric,
    PrecisionMetric,
    RecallMetric,
    DensityMetric,
    CoverageMetric,
    MMDMetric,
    ISMetric,
    KIDMetric,
    VendiMetric,
    AuthPctMetric,
    SWMetric,
)


class GradCAM:
    METRIC_MAP = {
        "fid": FIDMetric,
        "precision": PrecisionMetric,
        "recall": RecallMetric,
        "density": DensityMetric,
        "coverage": CoverageMetric,
        "mmd": MMDMetric,
        "is": ISMetric,
        "kid": KIDMetric,
        "vendi": VendiMetric,
        "authpct": AuthPctMetric,
        "sw": SWMetric,
    }

    def __init__(
        self, model, reps_real, reps_gen, device, model_name, metric_name, **kwargs
    ):
        self.acts_and_gradients = ActivationsAndGradients(
            network=model, model_name=model_name
        )

        self.model = model
        self.device = device
        self.model_name = model_name

        # Select and initialize the appropriate metric
        if metric_name not in self.METRIC_MAP:
            raise ValueError(f"Unsupported metric: {metric_name}")

        self.metric = self.METRIC_MAP[metric_name](device, **kwargs)

        # Compute statistics for real and generated features
        self.real_stats = self.metric.compute_statistics(
            torch.from_numpy(reps_real).to(device)
        )
        self.gen_stats = self.metric.compute_statistics(
            torch.from_numpy(reps_gen).to(device)
        )

        self.reps_gen = torch.from_numpy(reps_gen).to(device)
        self.num_images = len(self.reps_gen)

    def get_map(self, image, idx):
        start_time = time.time()
        self.acts_and_gradients.eval()
        features = get_features(self.acts_and_gradients, image)

        # Ensure features is 2D: (1, feature_dim)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Update statistics with the new features
        new_gen_stats = self.metric.update_statistics(
            self.gen_stats, features, self.num_images
        )

        # Compute loss using the selected metric
        loss = self.metric.compute_loss(self.real_stats, new_gen_stats)
        loss.backward()

        heatmap = self._get_heatmap_from_grads()
        overlay = show_heatmap_on_image(heatmap, image)

        label = (
            self.model.get_label(features) if hasattr(self.model, "get_label") else None
        )

        return overlay, label

    def _get_heatmap_from_grads(self):
        # Get activations and gradients from the target layer by accessing hooks.
        activations = self.acts_and_gradients.activations[-1]
        gradients = self.acts_and_gradients.gradients[-1]

        if len(activations.shape) == 3:
            dim = int(activations.shape[-1] ** 0.5)
            activations = activations[:, :, 1:].reshape(
                *activations.shape[:-1], dim, dim
            )
            gradients = gradients[:, :, 1:].reshape(*gradients.shape[:-1], dim, dim)

        # Turn gradients and activation into heatmap.
        weights = np.mean(gradients**2, axis=(2, 3), keepdims=True)
        heatmap = (weights * activations).sum(axis=1)
        return heatmap[0]


MODEL_TO_LAYER_NAME_MAP = {
    "inception": "blocks.3.2",
    "clip": "visual.transformer.resblocks.11.ln_1",
    "mae": "blocks.23.norm1",
    "swav": "layer4.2",
    "dinov2": "blocks.23.norm1",
    "convnext": "stages.3.blocks.2",
    "data2vec": "model.encoder.layer.23.layernorm_before",
    "simclr": "net.4.blocks.2.net.3",
}

MODEL_TO_LAYER_NAME_MAP = {
    "rad_inception": "Mixed_7c",
    "inception": "Mixed_7c",
    "resnet50": "layer4.2.conv3",
    "rad_resnet50": "layer4.2.conv3",
    "densenet121": "features.denseblock4.denselayer16.conv2",
    "rad_densenet": "features.denseblock4.denselayer16.conv2",
    "clip": "vision_model.encoder.layers.11.layer_norm1",
    "rad_clip": "vision_model.encoder.layers.11.layer_norm1",
    "dino": "encoder.layer.11.norm1",
    "rad_dino": "encoder.layer.11.norm1",
}

MODEL_TO_TRANSFORM_MAP = {
    "inception": lambda x: x,
    "rad_inception": lambda x: x,
    "resnet50": lambda x: x,
    "rad_resnet50": lambda x: x,
    "densenet121": lambda x: x,
    "rad_densenet": lambda x: x,
    "clip": lambda x: -x.transpose(0, 2, 1),
    "rad_clip": lambda x: -x.transpose(0, 2, 1),
    "mae": lambda x: x.transpose(0, 2, 1),
    "swav": lambda x: x,
    "dino": lambda x: -x.transpose(0, 2, 1),
    "rad_dino": lambda x: -x.transpose(0, 2, 1),
    "convnext": lambda x: -x,
    "data2vec": lambda x: x.transpose(0, 2, 1),
    "simclr": lambda x: -x,
}


class ActivationsAndGradients:
    """Class to obtain intermediate activations and gradients.
    Adapted from: https://github.com/jacobgil/pytorch-grad-cam"""

    def __init__(
        self, network: Any, model_name: str, network_kwargs: dict = None
    ) -> None:
        self.network = network
        self.network_kwargs = network_kwargs if network_kwargs is not None else {}
        self.gradients: List[np.ndarray] = []
        self.activations: List[np.ndarray] = []
        self.transform = MODEL_TO_TRANSFORM_MAP.get(model_name)

        target_layer_name = MODEL_TO_LAYER_NAME_MAP.get(model_name)
        try:
            target_layer = dict(network.model.named_modules()).get(target_layer_name)
        except:
            target_layer = dict(network.named_modules()).get(target_layer_name)
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module: Any, input: Any, output: Any) -> None:
        """Saves forward pass activations."""
        activation = output

        self.activations.append(self.transform(activation.detach().cpu().numpy()))

    def save_gradient(self, module: Any, grad_input: Any, grad_output: Any) -> None:
        """Saves backward pass gradients."""
        # Gradients are computed in reverse order.
        grad = grad_output[0]
        self.gradients = [
            self.transform(grad.detach().cpu().numpy())
        ] + self.gradients  # Prepend current gradients.

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Resets hooked activations and gradients and calls model forward pass."""
        self.gradients = []
        self.activations = []

        return self.network(x, **self.network_kwargs)

    def eval(self):
        self.network.eval()


# def wasserstein2_loss(mean_reals: torch.Tensor,
#                      mean_gen: torch.Tensor,
#                      cov_reals: torch.Tensor,
#                      cov_gen: torch.Tensor,
#                      eps: float = 1e-12) -> torch.Tensor:
#    """Computes 2-Wasserstein distance."""
#    mean_term = torch.sum(torch.square(mean_reals - mean_gen.squeeze(0)))
#    eigenvalues = torch.real(torch.linalg.eig(torch.matmul(cov_gen, cov_reals))[0])
#    cov_term = torch.trace(cov_reals) + torch.trace(cov_gen) - 2 * torch.sum(torch.sqrt(abs(eigenvalues) + eps))
#    return mean_term + cov_term


def wasserstein2_loss(
    mean_reals: torch.Tensor,
    mean_gen: torch.Tensor,
    cov_reals: torch.Tensor,
    cov_gen: torch.Tensor,
    eps: float = 1e-12,
    device: str = "cuda",
) -> torch.Tensor:

    """Computes 2-Wasserstein distance."""
    mean_reals = mean_reals.to(device)
    mean_gen = mean_gen.to(device)
    cov_reals = cov_reals.to(device)
    cov_gen = cov_gen.to(device)

    mean_term = torch.sum(torch.square(mean_reals - mean_gen.squeeze(0)))

    # Compute eigenvalues directly without full eigendecomposition
    eigenvalues = torch.linalg.eigvals(torch.matmul(cov_gen, cov_reals)).real

    # Compute trace terms and sum of square roots of eigenvalues
    trace_term = torch.trace(cov_reals) + torch.trace(cov_gen)
    sqrt_term = 2 * torch.sum(torch.sqrt(torch.abs(eigenvalues) + eps))

    return mean_term + trace_term - sqrt_term


def get_features(model, image):
    features = model(image)[0]
    if not torch.is_tensor(features):  # Some encoders output tuples or lists
        features = features[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if features.dim() > 2:
        pred_sh = features.shape
        # Remove the class token (usually the first token)
        patch_embeddings = features[:, 1:, :]  # Shape: [32, 256, 768]
        rshp_num = int(math.sqrt(patch_embeddings.shape[1]))
        # Reshape to [32, 768, 16, 16] assuming 16x16 patches
        patch_embeddings = patch_embeddings.transpose(1, 2).view(
            pred_sh[0], pred_sh[2], rshp_num, rshp_num
        )

        # Apply adaptive average pooling
        pooled_output = F.adaptive_avg_pool2d(
            patch_embeddings, (1, 1)
        )  # Shape: [32, 768, 1, 1]

        # Flatten the pooled output
        features = pooled_output.view(pred_sh[0], pred_sh[2]).squeeze()

        # if features.size(2) != 1 or features.size(3) != 1:

        #    features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1))

        # features = features.squeeze(3).squeeze(2)
    if features.dim() == 1:
        features = features.unsqueeze(0)

    return features


def zero_one_scaling(image: np.ndarray) -> np.ndarray:
    """Scales an image to range [0, 1]."""
    if np.all(image == 0):
        return image
    image = image.astype(np.float32)
    if (image.max() - image.min()) == 0:
        return image
    return (image - image.min()) / (image.max() - image.min())


def show_heatmap_on_image(
    heatmap, image, colormap: int = cv2.COLORMAP_PARULA, heatmap_weight: float = 1.0
):

    image_np = image.detach().cpu().numpy()[0]
    _, h, w = image_np.shape

    if heatmap.size == 0:
        print("Error: Heatmap is empty")
        return None

    # Scale heatmap values between 0 and 255.
    heatmap = zero_one_scaling(heatmap)
    heatmap = np.clip((heatmap * 255.0).astype(np.uint8), 0.0, 255.0)

    # Save scaled heatmap

    # Scale to original image size.
    heatmap = np.array(
        PIL.Image.fromarray(heatmap)
        .resize((w, h), resample=PIL.Image.LANCZOS)
        .convert("L")
    )

    # Save resized heatmap

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    # Save color mapped heatmap

    # Overlay original RGB image and heatmap with specified weights.
    scaled_image = zero_one_scaling(image_np)
    overlay = heatmap_weight * heatmap.transpose(2, 0, 1) + scaled_image
    overlay = zero_one_scaling(overlay)
    overlay = np.clip(overlay * 255, 0.0, 255.0).astype(np.uint8)

    # Save final overlay
    # cv2.imwrite('debug_overlay_final.png', cv2.cvtColor(overlay.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

    if overlay.size == 0:
        print("Error: Overlay is empty")
        return None

    return overlay
