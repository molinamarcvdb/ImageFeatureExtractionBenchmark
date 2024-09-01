#import torch
#import os
#import torch.nn as nn
#from torchvision import models
#from transformers import CLIPModel, CLIPConfig, AutoModel, CLIPProcessor, AutoImageProcessor
#from typing import Tuple, Optional
#
#from privacy_benchmark import SiameseNetwork, initialize_model
#
#def test_model(model_name: str):
#    print(f"Testing {model_name}")
#    #try:
#    backbone, backbone_type, processor = initialize_model(model_name)
#    model = SiameseNetwork(backbone, backbone_type, processor)
#    model.eval()
#
#    # Create dummy input
#    if backbone_type == 'torch':
#        input_shape = (1, 3, 224, 224)
#        if isinstance(backbone, models.Inception3):
#            input_shape = (1, 3, 299, 299)
#    elif backbone_type == 'huggingface':
#        if processor:
#            # Use a dummy image for huggingface models with processors
#            dummy_image = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
#            processed = processor(images=dummy_image, return_tensors="pt")
#            dummy_input1 = processed['pixel_values'].to('cuda')
#            dummy_input2 = processed['pixel_values'].to('cuda')
#        else:
#            input_shape = (1, 3, 224, 224)
#            dummy_input1 = torch.randn(input_shape).to('cuda')
#            dummy_input2 = torch.randn(input_shape).to('cuda')
#    else:
#        raise ValueError(f"Unsupported backbone type: {backbone_type}")
#
#    if 'dummy_input1' not in locals():
#        dummy_input1 = torch.randn(input_shape).to('cuda')
#        dummy_input2 = torch.randn(input_shape).to('cuda')
#
#    with torch.no_grad():
#        print(dummy_input1.shape)
#        output = model(dummy_input1, resnet_only=True)
#        print(f"Success! Output shape: {output[0].shape}")
#        print(output)  
#        Labels = torch.arange(output.shape[0], device='cuda')
##except Exception as e:
#    #    print(f"Error: {str(e)}")
#    print()
#
## Test different model types
#test_model('resnet50')
#test_model('resnet18')
#test_model('inception')
#test_model('densenet121')
#test_model('clip')
#test_model('rad_clip')
#test_model('rad_dino')
#test_model('dino')
#test_model('rad_inception')
#test_model('rad_resnet50')
#test_model('rad_densenet')
import shutil
import pandas as pd
from tqdm import tqdm
import os
def organize_dataset_by_view_with_info(csv_path, image_root_dir, output_root_dir):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # List of all conditions
    conditions = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                  'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    
    # Create directories for each condition and view type
    for condition in conditions:
        os.makedirs(os.path.join(output_root_dir, f"{condition}_Frontal"), exist_ok=True)
        os.makedirs(os.path.join(output_root_dir, f"{condition}_Lateral"), exist_ok=True)
    
    # Process each row in the dataframe
    for _, row in tqdm(df.iterrows()):
        image_path = os.path.join(image_root_dir, row['Path'])
        view_type = row['Frontal/Lateral']
        
        # Extract patient and study info from the path
        path_parts = row['Path'].split('/')
        patient_id = path_parts[-3]  # Assuming patient ID is always the third-to-last part
        study_id = path_parts[-2]    # Assuming study ID is always the second-to-last part
        original_filename = os.path.basename(row['Path'])
        
        # Create new filename
        new_filename = f"{patient_id}_{study_id}_{original_filename}"
        
        # Check each condition
        for condition in conditions:
            if row[condition] == 1.0:  # Assuming 1.0 indicates presence of condition
                dest_folder = f"{condition}_{view_type}"
                dest_path = os.path.join(output_root_dir, dest_folder, new_filename)
                
                # Use shutil  to save space, or copy if symlink is not suitable
                if os.path.exists(image_path):
                    if not os.path.exists(dest_path):
                        shutil.copyfile(image_path, dest_path)
                        # Alternatively, use shutil.copy2(image_path, dest_path) to copy the file
                else:
                    print(f"Warning: Image not found - {image_path}")

    # Print summary of the dataset organization
    print_dataset_summary(output_root_dir)

def print_dataset_summary(output_root_dir):
    print("\nDataset Summary:")
    counts = 0
    for folder in sorted(os.listdir(output_root_dir)):
        count = len(os.listdir(os.path.join(output_root_dir, folder)))
        print(f"{folder}: {count} images")
        counts += count
    print('Total ammount of images sent:', counts)
# Usage
#organize_dataset_by_view_with_info('/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/CheXpert-v1.0-small/train.csv', '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/', '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/CheXpert-v1.0-small/ImageSubfolder')
print_dataset_summary('/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/CheXpert-v1.0-small/ImageSubfolder')
