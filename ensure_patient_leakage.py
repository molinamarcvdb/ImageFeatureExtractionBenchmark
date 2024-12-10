import os
import sys
import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def create_train_val_split(
    patient_data_dir: str,
    patient_info_path: Optional[str] = None,
    patient_id: Optional[str] = None,
    secondary_ids: Optional[List[str]] = None,
    unique_identifier_col: Optional[str] = None,
    train_ratio: float = 0.8,
    extension: str = ".jpeg",
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """
    Create train-validation split of image paths. If patient_info_path is provided,
    splits will maintain patient separation. Otherwise, performs simple random split.
    """
    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Get all image paths
    image_paths = []
    patient_data_path = Path(patient_data_dir)
    for file_path in patient_data_path.rglob(f"*{extension}"):
        image_paths.append(str(file_path))

    if not image_paths:
        logger.warning(
            f"No images found with extension {extension} in {patient_data_dir}"
        )
        return [], []

    # If no CSV file is provided or it doesn't exist, do simple random split
    if patient_info_path is None or not os.path.isfile(patient_info_path):
        logger.info(
            "No valid patient info file provided. Performing simple random split."
        )

        # Shuffle all image paths
        random.shuffle(image_paths)

        # Calculate split index
        split_idx = int(len(image_paths) * train_ratio)

        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        logger.info(
            f"Simple split resulted in {len(train_paths)} training images "
            f"and {len(val_paths)} validation images"
        )

        return train_paths, val_paths

    # If CSV exists, try to use it for patient-based splitting
    try:
        import pandas as pd
        import polars as pl

        df = pd.read_csv(patient_info_path)
        dfl = pl.from_pandas(df)

        # Check if required columns exist
        required_cols = [
            col for col in [patient_id, unique_identifier_col] if col is not None
        ]
        if not all(col in df.columns for col in required_cols):
            logger.warning(
                "Required columns not found in CSV. Falling back to simple random split."
            )
            random.shuffle(image_paths)
            split_idx = int(len(image_paths) * train_ratio)
            return image_paths[:split_idx], image_paths[split_idx:]

        # Original patient-based splitting logic
        extended_ids = [patient_id]
        if secondary_ids:
            extended_ids.extend(secondary_ids)

        unique_images = (
            dfl.sort(unique_identifier_col)
            .group_by(extended_ids)
            .agg(pl.col(unique_identifier_col).first().alias("unique_image"))
        )

        patient_images = (
            unique_images.group_by(patient_id)
            .agg(pl.col("unique_image").alias("image_list"))
            .sort(patient_id)
        )

        # Create dictionary of patient to image paths
        patient_to_images = {}
        for patient, images in zip(
            patient_images[patient_id], patient_images["image_list"]
        ):
            if patient is not None:
                patient_paths = []
                for image in images:
                    if image is not None:
                        full_path = os.path.join(
                            patient_data_dir, os.path.splitext(image)[0] + extension
                        )
                        if os.path.exists(full_path):
                            patient_paths.append(full_path)
                if patient_paths:
                    patient_to_images[patient] = patient_paths

        if not patient_to_images:
            logger.warning(
                "No valid patient-image mappings found. Falling back to simple random split."
            )
            random.shuffle(image_paths)
            split_idx = int(len(image_paths) * train_ratio)
            return image_paths[:split_idx], image_paths[split_idx:]

        # Split patients
        patients = sorted(list(patient_to_images.keys()))
        random.shuffle(patients)

        train_paths = []
        val_paths = []
        current_train_images = 0
        total_images = sum(len(images) for images in patient_to_images.values())
        target_train_images = int(total_images * train_ratio)

        for patient in patients:
            images = patient_to_images[patient]
            if current_train_images < target_train_images:
                train_paths.extend(images)
                current_train_images += len(images)
            else:
                val_paths.extend(images)

        logger.info(
            f"Patient-based split resulted in {len(train_paths)} training images "
            f"and {len(val_paths)} validation images"
        )

        return train_paths, val_paths

    except Exception as e:
        logger.warning(
            f"Error processing CSV file: {e}. Falling back to simple random split."
        )
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * train_ratio)
        return image_paths[:split_idx], image_paths[split_idx:]
