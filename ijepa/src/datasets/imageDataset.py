import json
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

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

def convert_to_npy(input_paths, output_dir, shape):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    expected_npy_files = [os.path.splitext(os.path.basename(path))[0] + '.npy' for path in input_paths]
    existing_npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    
    if set(expected_npy_files) == set(existing_npy_files):
        print("All files already converted. Skipping conversion.")
        return [os.path.join(output_dir, f) for f in existing_npy_files]
    
    # If mismatch, clear the output directory and perform conversion
    print("Mismatch in files. Clearing output directory and performing conversion.")
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    new_paths = []
    file_sizes = []

    for input_path in tqdm(input_paths):
        if input_path.endswith((".png", ".jpeg", ".jpg", ".tiff")):
            # Open the image file
            img = Image.open(input_path)
# Resize the image
            img_resized = img.resize((shape[1], shape[0]), Image.LANCZOS)
            # Convert to numpy array
            img_array = np.array(img_resized)
            # Create output filename
            base_name = os.path.basename(input_path)
            npy_filename = os.path.splitext(base_name)[0] + '.npy'
            output_path = os.path.join(output_dir, npy_filename)
            
            # Save as NPY file
            np.save(output_path, img_array)
            
            # Verify file size
            file_size = os.path.getsize(output_path)
            file_sizes.append(file_size)
            
            new_paths.append(output_path)

    # Check if all file sizes are the same
    if len(set(file_sizes)) == 1:
        print(f"Warning: All converted files have the same size of {file_sizes[0]} bytes.")
    else:
        print(f"File sizes vary. Min: {min(file_sizes)}, Max: {max(file_sizes)}, Average: {sum(file_sizes)/len(file_sizes)}")

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
    target_res= 224,
    folder=None,
    dataset_info=None,
    unique_individual_id=None,
    unique_image_id=None,
    split_ratio=None,
    image_extension=None,
    seed=None
    ):
    
    target_res = (*(target_res, target_res), 1)

    train_paths, val_paths = create_train_val_split(
            patient_data_dir = root_path,
            patient_info_path = dataset_info,
            patient_id = unique_individual_id,
            unique_identifier_col = unique_image_id,
            train_ratio = split_ratio,
            extension = image_extension,
            seed = seed
            )


    # Save paths to JSON file
    paths_dict = {
        "train_paths": train_paths,
        "val_paths": val_paths
    }
    json_file_path = os.path.join(folder, f"image_paths.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(paths_dict, json_file, indent=2)



    train_paths = convert_to_npy(train_paths, os.path.join('./preprocessed', 'train'), target_res)
    val_paths = convert_to_npy(val_paths, os.path.join('./preprocessed', 'val'), target_res)
    
    train_dataset = ContrastiveDataset(
        file_paths=train_paths,
        target_resolution=target_res,
        transforms=transform,
        num_channels=3
    )

    val_dataset = ContrastiveDataset(
        file_paths=val_paths,
        target_resolution=target_res,
        transforms=transform,
        num_channels=3    )
    print(train_dataset, type(train_dataset))
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
                         
                         
class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, target_resolution, transforms, num_channels=3):
        self.paths = file_paths
        self.target_resolution = target_resolution
        self.transforms = transforms
        self.resolution_transform = tio.Resize(target_resolution)
        self.num_channels = num_channels

    def __len__(self):
        return len(self.paths)

    def process_image(self, img_path, aug=False):

        img = np.load(img_path, mmap_mode='r+')

        # Apply other transforms
        if aug:
            img = self.transforms(np.asarray(img))
        else:
            img= torch.from_numpy(img).float()
            return img.permute(2, 1, 0)
        
        # Get the data tensor and remove the extra 'z' dimension
        img_data = img.data.squeeze(-1).squeeze(0)
        if img_data.shape[0] != 3:
            img_data = img_data.repeat(3, 1, 1)
        # Ensure the image has 3 channels

        return img_data

    def __getitem__(self, index):
        img = self.process_image(self.paths[index], aug = True)
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

import os
import sys
import logging
import pandas as pd
import polars as pl
import random
from collections import defaultdict

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def create_train_val_split(
        patient_data_dir, 
        patient_info_path,
        patient_id: str,
        unique_identifier_col: str,
        train_ratio=0.8, 
        extension: str = '.jpeg',
        seed: int = None
        ):

    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Get all image paths
    image_paths = []
    for root, dirs, filenames in sorted(os.walk(patient_data_dir)):
        for file in sorted(filenames):
            if file.endswith((extension)):
                image_paths.append(os.path.join(patient_data_dir, file))

    total_number_imgs = len(image_paths)

    # Read patient info
    df = pd.read_csv(patient_info_path)
    dfl = pl.from_pandas(df)

    # Group images by patient
    patient_images = dfl.group_by(patient_id).agg(pl.col(unique_identifier_col).alias('image_list')).sort(by=patient_id)
    
    # Create dictionary of patient to image paths
    patient_to_images = defaultdict(list)
    for patient, images in sorted(zip(patient_images[patient_id], patient_images['image_list'])):
        for image in sorted(images):
            full_path = os.path.join(patient_data_dir, os.path.splitext(image)[0] + extension)
            if full_path in image_paths:
                patient_to_images[patient].append(full_path)

    # Get the list of patients and shuffle it
    patients = sorted(list(patient_to_images.keys()))
    random.Random(seed).shuffle(patients)

    train_paths: list[str] = []
    val_paths: list[str] = []

    img_count = 0
    train_full = False

    img_train_idx = int(total_number_imgs * train_ratio)
    patient_count = 0 
    for patient in patients:
        images = patient_to_images[patient]
        img_count += len(images) 
        
        if not train_full:
            patient_count += 1
            train_paths.extend(images)
        else:
            val_paths.extend(images)

        if img_count >= img_train_idx:
            train_full = True
    
    logger.info(f'Train val splitting resulting in {len(train_paths)} training images and {len(val_paths)} validation images')
    logger.info(f'Out of all {len(patients)} individuals, {patient_count} resulted in training set')
    
    return train_paths, val_paths

