import os
import sys
import logging
import pandas as pd
import polars as pl
import random
from collections import defaultdict
from typing import List

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def create_train_val_split(
    patient_data_dir,
    patient_info_path,
    patient_id: str,
    secondary_ids: List,
    unique_identifier_col: str,
    train_ratio=0.8,
    extension: str = ".jpeg",
    seed: int = None,
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

    # Read patient info
    df = pd.read_csv(patient_info_path)
    dfl = pl.from_pandas(df)

    # Group images by patient_id
    extended_ids = [patient_id]
    if secondary_ids:
        extended_ids.extend(secondary_ids)
    unique_images = (
        dfl.sort(
            unique_identifier_col
        )  # Sort to ensure consistent selection of the "first" image
        .group_by(extended_ids)
        .agg(pl.col(unique_identifier_col).first().alias("unique_image"))
    )
    # Group the unique images by patient
    patient_images = (
        unique_images.group_by(patient_id)
        .agg(pl.col("unique_image").alias("image_list"))
        .sort(patient_id)
    )
    print(patient_images)
    # Create dictionary of patient to image paths
    patient_to_images = defaultdict(list)
    for patient, images in sorted(
        zip(patient_images[patient_id], patient_images["image_list"])
    ):
        for image in sorted(images):
            full_path = os.path.join(
                patient_data_dir, os.path.splitext(image)[0] + extension
            )
            if full_path in image_paths:
                print(full_path)
                patient_to_images[patient].append(full_path)

    # Get the list of patients and shuffle it
    patients = sorted(list(patient_to_images.keys()))
    random.Random(seed).shuffle(patients)

    imgs = []

    for patient in patients:
        imgs.extend(patient_to_images[patient])

    total_number_imgs = len(imgs)

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

    logger.info(
        f"Train val splitting resulting in {len(train_paths)} training images and {len(val_paths)} validation images"
    )
    logger.info(
        f"Out of all {len(patients)} individuals, {patient_count} resulted in training set"
    )

    return train_paths, val_paths


## Usage
# patient_data_dir = '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/output_images_512_all'
# patient_info_path = '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/Notebooks/dicom_metadata.csv'

# # Use a fixed seed for reproducibility
# seed = 42
# train_paths, val_paths = create_train_val_split(patient_data_dir, patient_info_path, patient_id='Patient ID', unique_identifier_col='Filename', secondary_ids= ['Laterality', 'Projection'], train_ratio=0.8, seed=seed)

# print(f"Number of training images: {len(train_paths)}")
# print(f"Number of validation images: {len(val_paths)}")
# print(f"Final split ratio {len(train_paths)/(len(train_paths)+len(val_paths))}")
