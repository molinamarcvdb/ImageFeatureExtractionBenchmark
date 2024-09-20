
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchio as tio
from pytorch_metric_learning.distances import  CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms import transforms

def make_contrastive_data(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    target_res= 224
    ):
    
    train_paths, val_paths = train_val_paths(root_path)

    train_dataset = ContrastiveDataset(
        file_paths=train_paths,
        target_resolution=target_res,
        trans=transform,
        num_channels=3
    )

    val_dataset = ContrastiveDataset(
        file_paths=val_paths,
        target_resolution=target_res,
        trans=transform,
        num_channels=3    )

    # Create samplers 
    train_dist_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset=train_dataset,
                        num_replicas=world_size,
                        rank=rank)

    val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset=val_dataset,
                        num_replicas=world_size,
                        rank=rank)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collator,
        sampler=train_dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=collator,
        sampler=val_dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    return train_dataset, val_dataset, train_data_loader, val_data_loader,train_dist_sampler, val_dist_sampler
                         
                         
def train_val_paths(data_path):

    all_paths = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith((".png", ".jpeg", ".jpg")):
                all_paths.append(os.path.join(root, file))
    all_paths.sort()

    # Split paths into train and validation
    train_paths, val_paths = train_test_split(all_paths, test_size=0.8, random_state=42)

    return train_paths, val_paths


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, target_resolution, trans, num_channels=3):
        self.paths = file_paths
        self.target_resolution = target_resolution
        self.trans = trans
        self.resolution_transform = tio.Resize(target_resolution)
        self.num_channels = num_channels
        self.mandatory_trans = transforms.Compose([
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    def __len__(self):
        return len(self.paths)

    def process_image(self, img_path, aug=False):
        img = Image.open(img_path).convert('RGB')

        # Apply resolution transform
        #img = self.resolution_transform(img)

        # Apply other transforms
        if aug:
            img = self.trans(img)
        else:
            img = self.mandatory_trans(img)        # Get the data tensor and remove the extra dimension

        return img 

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
