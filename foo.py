import torch
import json
import numpy as np
import os
from time import time
import torchio as tio
from torch.utils.data import DataLoader, RandomSampler

MANDATORY_TRANSFORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
])

json_path = '/home/ksamamov/GitLab/Notebooks/feat_ext_bench/checkpoints/20240923_103236_rad_inception/20240923_103236_rad_inception_image_paths.json'
target_res = (224, 224, 1)

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, target_resolution, transforms, num_channels=3):
        self.paths = file_paths
        self.target_resolution = target_resolution
        self.transforms = transforms
        self.resolution_transform = tio.Resize(target_resolution)
        self.num_channels = num_channels

        self.dummy_image  = torch.rand(1, *target_resolution)

    def __len__(self):
        return len(self.paths)

    def process_image(self, img_path, aug=False):
        img = tio.ScalarImage(tensor = self.dummy_image)

        # Apply resolution transform
        img = self.resolution_transform(img)

        # Apply other transforms
        if aug:
            img = self.transforms(img)
        else:
            mandatory_aug = tio.Compose([
                tio.RescaleIntensity(out_min_max=(0, 1)),
            ])
            img = mandatory_aug(img)

        # Get the data tensor and remove the extra dimension
        img_data = img.data.squeeze()

        # Ensure the tensor is in the format (3, H, W)
        if img_data.ndim == 2:  # If the image is 2D (H, W)
            img_data = img_data.unsqueeze(0).repeat(3, 1, 1)  # Add channel dimension and repeat to 3 channels
        elif img_data.ndim == 3:
            if img_data.shape[0] == 1:
                img_data = img_data.repeat(3, 1, 1)  # Repeat single channel to 3 channels
            elif img_data.shape[0] == 3:
                pass  # Already in the correct format
            elif img_data.shape[2] == 3:
                img_data = img_data.permute(2, 0, 1)  # Permute from (H, W, C) to (C, H, W)
            else:
                raise ValueError(f"Unexpected channel configuration: {img_data.shape}")
        else:
            raise ValueError(f"Unexpected number of dimensions: {img_data.ndim}")

        # Ensure the image has 3 channels
        assert img_data.shape[0] == 3, f"Image should have 3 channels, but has shape {img_data.shape}"

        return img_data 

    def __getitem__(self, index):
        img = self.process_image(self.paths[index])
        img_pos = self.process_image(self.paths[index], aug = True)  # Same image, different augmentation
        # Get a negative sample
        index_neg = np.random.choice(np.delete(np.arange(len(self.paths)), index))
        img_neg = self.process_image(self.paths[index_neg])

        label = np.nan  # or implement your label logic here
        img_id = os.path.basename(self.paths[index])

        return {
            'data': img,
            'data_pos': img_pos,
            'data_neg': img_neg,
            'cond': label,
            'path': self.paths[index],
            'img_id': img_id
        }
def load_image_paths(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['train_paths'], data['val_paths']

train_paths, val_paths = load_image_paths(json_path)

train_dataset = ContrastiveDataset(
    file_paths=train_paths,
    target_resolution=target_res,
    transforms=MANDATORY_TRANSFORMS,
    num_channels=3
)

# Create a RandomSampler
train_sampler = RandomSampler(train_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler,  # Use the RandomSampler instead of shuffle
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

# Warm-up loop
print("Warming up...")
warm_up_loader = DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False,
)
for _ in warm_up_loader:
    break

print("Starting main loop...")
start_time = time()
for i, batch in enumerate(train_loader):
    # Process your batch here
    end_time = time()
    print(f"Batch {i+1} loading time: {end_time - start_time:.4f} seconds")
    start_time = time()  # Reset start_time for the next iteration
