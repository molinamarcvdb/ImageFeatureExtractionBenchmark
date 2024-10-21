import os
import math
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import torch
import pathlib

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def get_representations(model, DataLoader, device, normalized=False, processor=None):
    """Extracts features from all images in DataLoader given model.

    Params:
    -- model       : Instance of Encoder such as inception or CLIP or dinov2
    -- DataLoader  : DataLoader containing image files, or torchvision.dataset

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    start_idx = 0

    for ibatch, batch in enumerate(tqdm(DataLoader.data_loader)):
        if isinstance(batch, list):
            # batch is likely list[array(images), array(labels)]
            batch = batch[0]

        if not torch.is_tensor(batch):
            # assume batch is then e.g. AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
            batch = batch["pixel_values"][0]
            # batch = batch[:,0]
        # Convert grayscale to RGB
        if batch.ndim == 3:
            batch.unsqueeze_(1)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = batch.to(device)

        with torch.no_grad():
            try:
                pred = model(batch)
            except:
                pred = model.vision_model(batch)
            if not torch.is_tensor(pred):  # Some encoders output tuples or lists
                pred = pred[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        print(pred.shape)
        if pred.dim() > 2:
            pred_sh = pred.shape
            # Remove the class token (usually the first token)
            patch_embeddings = pred[:, 1:, :]  # Shape: [32, 256, 768]
            rshp_num = int(math.sqrt(patch_embeddings.shape[1]))
            # Reshape to [32, 768, 16, 16] assuming 16x16 patches
            patch_embeddings = patch_embeddings.transpose(1, 2).view(
                pred_sh[0], pred_sh[2], rshp_num, rshp_num
            )

            # Apply adaptive average pooling
            pooled_output = adaptive_avg_pool2d(
                patch_embeddings, (1, 1)
            )  # Shape: [32, 768, 1, 1]

            # Flatten the pooled output
            pred = pooled_output.view(pred_sh[0], pred_sh[2])

        if normalized:
            pred = torch.nn.functional.normalize(pred, dim=-1)
        pred = pred.cpu().numpy()

        if ibatch == 0:
            # initialize output array with full dataset size
            dims = pred.shape[-1]
            pred_arr = np.empty((DataLoader.nimages, dims))
        print(pred.shape)
        print()
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def save_outputs(output_dir, reps, model, checkpoint, DataLoader):
    """Save representations and other info to disk at file_path"""
    out_path = get_path(output_dir, model, checkpoint, DataLoader)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    hyperparams = vars(DataLoader).copy()  # Remove keys that can't be pickled
    hyperparams.pop("transform")
    hyperparams.pop("data_loader")
    hyperparams.pop("data_set")

    np.savez(out_path, model=model, reps=reps, hparams=hyperparams)


def load_reps_from_path(saved_dir, model, checkpoint, DataLoader):
    """Save representations and other info to disk at file_path"""
    save_path = get_path(saved_dir, model, checkpoint, DataLoader)
    reps = None
    print("Loading from:", save_path)
    if os.path.exists(f"{save_path}.npz"):
        saved_file = np.load(f"{save_path}.npz")
        reps = saved_file["reps"]
    return reps


def get_path(output_dir, model, checkpoint, DataLoader):
    train_str = "train" if DataLoader.train_set else "test"

    ckpt_str = (
        ""
        if checkpoint is None
        else f"_ckpt-{os.path.splitext(os.path.basename(checkpoint))[0]}"
    )

    hparams_str = f"reps_{DataLoader.dataset_name}_{model}{ckpt_str}_nimage-{len(DataLoader.data_set)}_{train_str}"
    return os.path.join(output_dir, hparams_str)
