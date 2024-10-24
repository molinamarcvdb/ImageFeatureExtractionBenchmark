"""
Slightly modified from:
https://github.com/layer6ai-labs/dgm-eval/blob/master/dgm_eval/heatmaps/heatmaps.py#L28

"""
import os
import json
import sys
from abc import ABC, abstractmethod

import torch.nn as nn


from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import numpy as np
import torch
from tqdm import tqdm

from typing import List, Optional, Tuple

import PIL
import cv2
import numpy as np
import torch

from saliency.gradcam import GradCAM

from privacy_benchmark import initialize_model
from saliency.dataloading import get_dataloader
from saliency.representations import (
    save_outputs,
    get_representations,
    load_reps_from_path,
)


class Encoder(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.setup(*args, **kwargs)
        self.name = "encoder"

    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, x):
        """Converts a PIL Image to an input for the model"""
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def visualize_heatmaps(
    reps_real: np.array,
    reps_gen: np.array,
    model: Encoder,
    model_name: str,
    dataset: torch.utils.data.Dataset,
    results_dir: str,
    results_suffix: str = "default",
    dataset_name: str = None,
    num_rows: int = 4,
    num_cols: int = 4,
    device: torch.device = torch.device("cpu"),
    perturbation: bool = False,
    human_exp_indices: str = None,
    random_seed: int = 0,
    metric_name: str = "fid",
) -> None:

    """Visualizes to which regions in the images FID is the most sensitive to."""

    visualizer = GradCAM(
        model, reps_real, reps_gen, device, model_name, metric_name=metric_name
    )

    # ----------------------------------------------------------------------------
    # Visualize FID sensitivity heatmaps.
    heatmaps, labels, images = [], [], []

    # Sampling image indices
    rnd = np.random.RandomState(random_seed)
    if human_exp_indices is not None:
        with open(human_exp_indices, "r") as f_in:
            index_to_score = json.load(f_in)
        indices = [
            int(idx) for idx in list(index_to_score.keys()) if int(idx) < len(dataset)
        ]
        if len(indices) < len(index_to_score):
            raise RuntimeWarning(
                "The datasets were subsampled so the human experiment indices will not be accurate. "
                "Please use '--nmax_images' with a higher value"
            )
        vis_images_indices = [
            idx for idx in rnd.choice(indices, size=num_rows * num_cols, replace=False)
        ]
        vis_images_scores = [index_to_score[str(idx)] for idx in vis_images_indices]
        vis_images_indices = [
            idx for _, idx in sorted(zip(vis_images_scores, vis_images_indices))
        ]  # sorting indices in ascending human score
    else:
        vis_images_indices = rnd.choice(
            np.arange(len(dataset)), size=num_rows * num_cols, replace=False
        )

    print("Visualizing heatmaps...")
    for idx in tqdm(vis_images_indices):

        # ----------------------------------------------------------------------------
        # Get selected image and do required transforms
        image = get_image(dataset, idx, device, perturbation=perturbation)
        # ----------------------------------------------------------------------------
        # Compute and visualize a sensitivity map.
        heatmap, label = visualizer.get_map(image, idx)

        heatmaps.append(heatmap)
        labels.append(label)
        images.append(
            np.clip(
                zero_one_scaling(image=image.detach().cpu().numpy().squeeze(0)) * 255,
                0.0,
                255.0,
            ).astype(np.uint8)
        )

    human_scores = labels
    if human_exp_indices is not None:
        human_scores = [
            f"{index_to_score[str(idx)]:0.2f}" for idx in vis_images_indices
        ]

    # ----------------------------------------------------------------------------
    # Create a grid of overlay heatmaps.
    heatmap_grid = create_grid(
        images=heatmaps, labels=labels, num_rows=num_rows, num_cols=num_cols
    )
    image_grid = create_grid(
        images=images, labels=human_scores, num_rows=num_rows, num_cols=num_cols
    )
    heatmap_grid.save(
        os.path.join(
            results_dir, f"sensitivity_{metric_name}_grid_{results_suffix}.png"
        )
    )
    image_grid.save(os.path.join(results_dir, f"images_grid_{results_suffix}.png"))


def get_image(dataset, idx, device, perturbation=False):
    image = dataset[idx]

    if isinstance(image, tuple):
        # image is likely tuple[images, label]
        image = image[0]
    if isinstance(image, torch.Tensor):
        # add batch dimension
        image.unsqueeze_(0)
    else:  # Special case of data2vec
        image = image.data["pixel_values"]
    # Convert grayscale to RGB
    if image.ndim == 3:
        image.unsqueeze_(1)
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if perturbation:
        image = perturb_image(image)
    image = image.to(device)
    image.requires_grad = True
    return image


def get_features(model, image):
    features = model(image)[0]

    if not torch.is_tensor(features):  # Some encoders output tuples or lists
        features = features[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if features.dim() > 2:
        if features.size(2) != 1 or features.size(3) != 1:
            features = torch.nn.functional.adaptive_avg_pool2d(
                features, output_size=(1, 1)
            )

        features = features.squeeze(3).squeeze(2)

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

    # Scale heatmap values between 0 and 255.
    heatmap = zero_one_scaling(image=heatmap)
    heatmap = np.clip((heatmap * 255.0).astype(np.uint8), 0.0, 255.0)

    # Scale to original image size.
    heatmap = np.array(
        PIL.Image.fromarray(heatmap)
        .resize((w, h), resample=PIL.Image.LANCZOS)
        .convert("L")
    )

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    # Overlay original RGB image and heatmap with specified weights.
    scaled_image = zero_one_scaling(image=image_np)
    overlay = heatmap_weight * heatmap.transpose(2, 0, 1) + scaled_image
    overlay = zero_one_scaling(image=overlay)
    overlay = np.clip(overlay * 255, 0.0, 255.0).astype(np.uint8)

    return overlay


import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np
from typing import List, Optional, Tuple

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np
from typing import List, Optional, Tuple


def create_grid(
    images: List[np.ndarray],
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    labels: Optional[List[str]] = None,
    label_loc: Tuple[int, int] = (0, 0),
    fontsize: int = 32,
    font_path: str = "./data/times-new-roman.ttf",
) -> PIL.Image:
    """Creates an image grid."""
    if not images:
        raise ValueError("The images list is empty.")

    h, w = 256, 256
    num_images = len(images)

    # Determine grid size if not specified
    if num_rows is None and num_cols is None:
        num_cols = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_cols))
    elif num_rows is None:
        num_rows = int(np.ceil(num_images / num_cols))
    elif num_cols is None:
        num_cols = int(np.ceil(num_images / num_rows))

    if labels is None:
        labels = [None] * num_images
    assert len(images) == len(labels), "Number of images and labels must match"

    try:
        font = PIL.ImageFont.truetype(font_path, fontsize)
    except IOError:
        print(f"Warning: Font file not found at {font_path}. Using default font.")
        font = PIL.ImageFont.load_default()

    grid = PIL.Image.new("RGB", size=(num_cols * w, num_rows * h))

    for idx, (image, label) in enumerate(zip(images, labels)):
        i, j = divmod(idx, num_cols)

        # Ensure image is in the correct format (H, W, C)
        if image.ndim == 3 and image.shape[0] == 3:  # If in (C, H, W) format
            image = image.transpose((1, 2, 0))
        elif image.ndim == 2:  # If grayscale
            image = np.stack([image] * 3, axis=-1)

        im = cv2.resize(image, dsize=(h, w), interpolation=cv2.INTER_CUBIC)
        im = PIL.Image.fromarray(im.astype("uint8"))

        if label is not None:
            draw = PIL.ImageDraw.Draw(im)
            draw.text(
                label_loc, f"{label}".capitalize(), font=font, fill=(255, 255, 255)
            )  # White text

        grid.paste(im, box=(j * w, i * h))

    return grid


def perturb_image(image):
    # image is (B, N, H, W)
    _, _, h, w = image.shape
    image[
        :, :, int(2 * h / 10) : int(3 * h / 10), int(2 * w / 10) : int(3 * w / 10)
    ] = 0
    return image


def pil_resize(x, output_size):
    s1, s2 = output_size

    def resize_single_channel(x):
        img = Image.fromarray(x, mode="F")
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    x = np.array(x.convert("RGB")).astype(np.float32)
    x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)
    return to_tensor(x) / 255


def compute_repr(netw_name, model, DL, device, save=None, processor=None):
    if save:
        print(f"Loading saved representations from: {save}\n", file=sys.stderr)
        repsi = load_reps_from_path(save, netw_name, None, DL)
        if repsi is not None:
            return repsi

        print(f"No saved representations found: {save}\n", file=sys.stderr)

    repsi = get_representations(
        model, DL, device, normalized=False, processor=processor
    )
    if save:
        print(f"Saving representations to {save}\n", file=sys.stderr)
        save_outputs(save, repsi, netw_name, None, DL)
    return repsi


def remove_fc_by_identity(model, model_type, target_layer_name):
    import torchvision.models as models

    if model_type == "torch":
        if isinstance(model, models.ResNet):
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif isinstance(model, models.DenseNet):
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()
        elif isinstance(model, models.Inception3):
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            if hasattr(model, "AuxLogits"):
                model.AuxLogits = None
        else:
            raise ValueError("Unsupported Torch model type")
    if model_type == "hugging_face":
        model = replace_layers_with_identity(model, target_layer_name)

    return model


def replace_layers_with_identity(model, target_layer_name):
    found_target = False
    for name, module in model.named_children():
        if found_target:
            # Replace the module with an identity operation
            setattr(model, name, nn.Identity())
        elif name == target_layer_name:
            found_target = True
        else:
            # Recursively process nested modules
            replace_layers_with_identity(module, target_layer_name)
    return model


def saliency_representations(config: dict):
    real_path = config["real_dataset_path"]
    synth_path = config["synthetic_dataset_path"]
    network_names = config["networks"]
    timestamp_dir = f"./data/features/{config['timestamp']}"
    metric_names = config["metrics"]

    list_real = [file for file in os.listdir(real_path) if file.endswith(".jpeg")]
    list_synth = [file for file in os.listdir(synth_path) if file.endswith(".png")]
    nsample = max(len(list_real), len(list_synth))

    for network_name in network_names:
        print()
        print(f"Processing saliency fetures with GradCam for {network_name}")
        print()
        out_dir = os.path.join(timestamp_dir, network_name, "saliency")
        os.makedirs(out_dir, exist_ok=True)
        num_workers = 4
        bs = 32
        size = (224, 224)

        model, model_type, processor = initialize_model(network_name)

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

        target_layer_name = MODEL_TO_LAYER_NAME_MAP.get(network_name)

        featextractor = remove_fc_by_identity(model, model_type, target_layer_name)
        print(featextractor)

        # if not processor:

        real_dataloader = get_dataloader(
            real_path,
            nsample,
            bs,
            num_workers,
            seed=42,
            sample_w_replacement=False,
            transform=lambda x: pil_resize(x, size),
        )
        synth_dataloader = get_dataloader(
            synth_path,
            nsample,
            bs,
            num_workers,
            seed=42,
            sample_w_replacement=False,
            transform=lambda x: pil_resize(x, size),
        )
        # else:

        #     real_dataloader = get_dataloader(real_path, nsample, bs, num_workers, seed=42,
        #                            sample_w_replacement=False,
        #                      transform=lambda x: processor(images=x))
        #     synth_dataloader = get_dataloader(synth_path, nsample, bs, num_workers, seed=42,
        #                            sample_w_replacement=False,
        #                            transform=lambda x: processor(images=x))

        results_suffix = f"{network_name}_{real_dataloader.dataset_name}_{synth_dataloader.dataset_name}"

        reps_gen = compute_repr(
            network_name,
            featextractor,
            synth_dataloader,
            device="cuda",
            save=out_dir,
            processor=processor,
        )
        reps_real = compute_repr(
            network_name,
            featextractor,
            real_dataloader,
            device="cuda",
            save=out_dir,
            processor=processor,
        )

        for metric_name in metric_names:

            visualize_heatmaps(
                reps_real,
                reps_gen,
                model,
                network_name,
                dataset=synth_dataloader.data_set,
                results_dir=out_dir,
                results_suffix=results_suffix,
                dataset_name=synth_dataloader.dataset_name,
                device="cuda",
                random_seed=42,
                metric_name=metric_name,
            )


def _test():

    real_path = "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/output_images_512_all"
    synth_path = "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/logs/009-DiT-XL-2/DiT-XL-2-0010000-SIZE-512-CLASS1-VAE-EMA-CFG-1.5-SEED-0--STEPS-350-FINAL"
    list_real = [file for file in os.listdir(real_path) if file.endswith(".jpeg")]
    list_synth = [file for file in os.listdir(synth_path) if file.endswith(".png")]
    nsample = max(len(list_real), len(list_synth))

    network_name = "clip"
    out_dir = f"/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/features/20240930_150033/{network_name}/saliency"
    os.makedirs(out_dir, exist_ok=True)
    num_workers = 4
    bs = 32
    size = (224, 224)

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

    model, model_type, processor = initialize_model(network_name)

    # print(dir(model))
    print(dict(model.named_modules()))
    target_layer_name = MODEL_TO_LAYER_NAME_MAP.get(network_name)

    print(dict(model.named_modules()).get(target_layer_name))

    featextractor = remove_fc_by_identity(model, model_type, target_layer_name)

    real_dataloader = get_dataloader(
        real_path,
        nsample,
        bs,
        num_workers,
        seed=42,
        sample_w_replacement=False,
        transform=lambda x: pil_resize(x, size),
    )

    synth_dataloader = get_dataloader(
        synth_path,
        nsample,
        bs,
        num_workers,
        seed=42,
        sample_w_replacement=False,
        transform=lambda x: pil_resize(x, size),
    )

    results_suffix = (
        f"{network_name}_{real_dataloader.dataset_name}_{synth_dataloader.dataset_name}"
    )

    reps_gen = compute_repr(
        network_name,
        featextractor,
        synth_dataloader,
        device="cuda",
        save=out_dir,
        processor=processor,
    )
    reps_real = compute_repr(
        network_name,
        featextractor,
        real_dataloader,
        device="cuda",
        save=out_dir,
        processor=processor,
    )

    visualize_heatmaps(
        reps_real,
        reps_gen,
        model,
        network_name,
        dataset=synth_dataloader.data_set,
        results_dir=out_dir,
        results_suffix=results_suffix,
        dataset_name=synth_dataloader.dataset_name,
        device="cuda",
        random_seed=42,
    )
