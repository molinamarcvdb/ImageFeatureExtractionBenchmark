import torch
import torch.nn as nn
import torchvision.models as models
from tensorflow.keras.applications import InceptionV3, ResNet50, InceptionResNetV2, DenseNet121
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoImageProcessor
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from utils import load_model_from_hub
import re
from collections import OrderedDict
from datetime import datetime
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import torchio as tio
import yaml
from torch.nn import CosineSimilarity
from tqdm import tqdm
import wandb
import functools
import inspect
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
import gc
from transformers import CLIPConfig
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import random
import glob
from tqdm import tqdm
import h5py
import traceback
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kendalltau
from pytorch_metric_learning import miners, losses
import json
from sklearn.model_selection import train_test_split
from networks import IJEPAEncoder
from ijepa.infer import AttentivePooler
from scipy.stats import wasserstein_distance
import torch.distributed as dist
import sys
import traceback
import logging
from utils import load_ijepa_not_contrastive

def setup_logging(dire):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(dire,  f'error_{timestamp}.err.log')
    logging.basicConfig(filename=log_filename, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')

class SiameseNetwork(nn.Module):

    def __init__(self, backbone_model, backbone_type='torch', processor=None, in_channels=1, n_features=128):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone_model
        self.backbone_type = backbone_type
        self.n_features = n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor
        self.in_channels = in_channels

        if self.backbone_type == 'torch':
            if isinstance(self.backbone, models.ResNet):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif isinstance(self.backbone, models.DenseNet):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Identity()
            elif isinstance(self.backbone, models.Inception3):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
                if hasattr(self.backbone, 'AuxLogits'):
                    self.backbone.AuxLogits = None
            else:
                raise ValueError("Unsupported Torch model type")

            if self.in_channels == 1:
                self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            elif self.in_channels == 3:
                pass
            else:
                raise Exception(
                    'Invalid argument: ' + str(self.in_channels) + '\nChoose either in_channels=1 or in_channels=3')

            self.fc = nn.Linear(in_features, self.n_features, bias=True)

        elif self.backbone_type == 'huggingface':
            if isinstance(self.backbone.config, CLIPConfig):
                in_features = self.backbone.config.vision_config.hidden_size
            elif hasattr(self.backbone.config, 'hidden_size'):
                in_features = self.backbone.config.hidden_size
            else:
                raise ValueError(f"Unsupported model configuration: {type(self.backbone.config)}")

            self.fc = nn.Linear(in_features, self.n_features, bias=True)

            for param in self.backbone.parameters():
                param.requires_grad = True

            for param in self.fc.parameters():
                param.requires_grad = True

        elif self.backbone_type == 'keras': ##this is deprecated and trianign with keras models is not possible all teh trianing logic is in torc issue: network graphh
            # Keras model handling remains the same
            base_model = Model(inputs=self.backbone.input, outputs=self.backbone.layers[-2].output)
            in_features = base_model.output.shape[-1]
            new_output = Dense(n_features, name='new_fc')(base_model.output)
            self.backbone = Model(inputs=base_model.input, outputs=new_output)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

        self.fc_end = nn.Linear(self.n_features, 1)
        self.to(self.device)

    def _get_features(self, x):
        """Helper method to flexibly extract features from various model outputs"""
        if hasattr(x, 'logits'):
            return x.logits
        elif isinstance(x, dict) and 'logits' in x:
            return x['logits']
        elif isinstance(x, (tuple, list)):
            return x[0]  # Assume the first element contains the main output
        else:
            return x  # Assume x is already the feature tensor we want
    
    def process_input(self, x):
        if self.backbone_type == 'huggingface' and self.processor is not None:
            # Process the input using the CLIP processor
            processed = self.processor(images=x, return_tensors="pt", do_rescale=False)
            return processed['pixel_values'].to(self.device)
        return x

    def forward_once(self, x):
        if self.backbone_type == 'torch':
            output  = self.backbone(x)

        elif self.backbone_type == 'huggingface':
            x = x.permute(0, 3, 2, 1)
            if isinstance(self.backbone.config, CLIPConfig):
                features = self.backbone.vision_model(x).last_hidden_state
            else:
                features = self.backbone(x).last_hidden_state
            output = self.fc(features)

        #elif self.backbone_type == 'keras':
        #    output = self.backbone.predict(x.cpu().numpy())
        #    output = torch.tensor(output).to(x.device)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

        return torch.sigmoid(output)

    def forward(self, input1=None, input2=None, resnet_only=False):
        if resnet_only:
            if self.backbone_type == 'torch':
                output = self.backbone(input1)
                features = self.fc(self._get_features(output))

                return features


            elif self.backbone_type == 'huggingface':
                #input1 = self.process_input(input1)
                if isinstance(self.backbone.config, CLIPConfig):
                    return self.fc(self.backbone.vision_model(input1).pooler_output)
                else:
                    return self.fc(self.backbone(input1).last_hidden_state.mean(dim=1))
            else:
                raise ValueError(f"resnet_only not supported for backbone type: {self.backbone_type}")
        else:
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)

            difference = torch.abs(output1 - output2)
            output = self.fc_end(difference)

        return output, output1, output2

class ContrastiveNetwork(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        feature_dim: int = 768,
        attentive_probing: bool = False,
        device: str = 'cuda'
    ):
        super(ContrastiveNetwork, self).__init__()
        
        self.attentive_probing = attentive_probing
        self.feature_dim = feature_dim
        self.device = device

        self.init_backbone(backbone_name)
        self.append_head()

        #print(self.backbone)

    def init_backbone(self, backbone_name):
        self.backbone, self.backbone_type, self.processor = initialize_model(backbone_name)

    def append_head(self):
        
        self.get_backbone_out_dim()

        print(f"Backbone output dimension: {self.backbone_output_dim}")   
        
        if self.attentive_probing:
            self.fc = AttentivePooler(
                embed_dim=self.backbone_output_dim,
                num_queries=1,
                num_heads=8,
                mlp_ratio=4.0,
                depth=1
            ) 
            self.final_projection = nn.Linear(self.backbone_output_dim, self.feature_dim)
        else:
            
            self.fc = nn.Linear(self.backbone_output_dim, self.feature_dim)

    def get_backbone_out_dim(self):
        if hasattr(self.backbone, 'fc') and hasattr(self.backbone.fc, 'in_features'):
            self.backbone_output_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            #self.backbone.avgpool = nn.Identity()
            if hasattr(self.backbone, 'AuxLogits'):
                self.backbone.AuxLogits = None

        elif hasattr(self.backbone, 'classifier') and hasattr(self.backbone.classifier, 'in_features'):
            self.backbone_output_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'vision_config') and hasattr(self.backbone.config.vision_config, 'hidden_size'):
            self.backbone_output_dim = self.backbone.config.vision_config.hidden_size

        elif hasattr(self.backbone, 'encoder_output_dim'):
            self.backbone_output_dim = self.backbone.encoder_output_dim
        
        elif hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'hidden_size'):
            self.backbone_output_dim = self.backbone.config.hidden_size
        
        else:
            raise 'No ouput dim feture found'

    def _get_features(self, x):
        """Helper method to flexibly extract features from various model outputs"""
        if hasattr(x, 'logits'):
            return x.logits
        elif isinstance(x, dict) and 'logits' in x:
            return x['logits']
        elif isinstance(x, (tuple, list)):
            return x[0] # Assume the first element contains the main output
        elif hasattr(x, 'last_hidden_state'):
            return x.last_hidden_state
        else:
            return x  # Assume x is already the feature tensor we want
              
    def forward(self, x):
        
        if hasattr(self.backbone, 'config') and isinstance(self.backbone.config, CLIPConfig):

            features = self._get_features(self.backbone.vision_model(x.to(self.device))).to(self.device)
        
        else:
            features = self._get_features(self.backbone(x.to(self.device))).to(self.device)

        if self.attentive_probing:
            if len(features.shape) == 2:
                features = features.unsqueeze(1)

            features = self.fc(features)  # This should output [batch_size, num_queries, backbone_output_dim]
            features = features.squeeze(1)  # Remove the num_queries dimension
            features = self.final_projection(features)
        else:
            if len(features.shape) == 3: 
                features = features.mean(dim=1)  # Global average pooling
            features = self.fc(features)
        return features

def remap_keys(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = re.sub(r'norm\.(\d+)', r'norm\1', k)
        new_key = re.sub(r'conv\.(\d+)', r'conv\1', new_key)
        new_state_dict[new_key] = v
    return new_state_dict

def load_and_remap_state_dict(model, filename, repo_id='molinamarc/syntheva'):
    try:
        state_dict = load_model_from_hub(repo_id, filename)
        remapped_state_dict = remap_keys(state_dict)
        
        # Try to load with strict=True first
        try:
            model.load_state_dict(remapped_state_dict, strict=True)
            print(f"Successfully loaded {filename} with strict=True")
        except RuntimeError as e:
            print(f"Couldn't load {filename} with strict=True. Trying with strict=False. Error: {str(e)}")
            model.load_state_dict(remapped_state_dict, strict=False)
            print(f"Successfully loaded {filename} with strict=False")
        
        return model
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return model  # Return the original model if loading failsi

def loadIJEPA_state_dict(model, repo_id='molinamarc/syntheva'):
    load_path = 'IN22K-vit.h.14-900e.pth.tar' 
    ckpt = torch.load(load_path, map_location=torch.device('cpu'))
    pretrained_dict = ckpt['encoder']

    # Loading encoder
    model_dict = model.state_dict()
    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            k = k[len("module."):]
        if k in model_dict:
            model_dict[k].copy_(v)

    
    model.load_state_dict(model_dict)

    return model 

def initialize_model(network_name):
    
    processor = None

    if network_name.lower() == "resnet50":
        backbone = models.resnet50(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'resnet50.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "resnet18":
        backbone = models.resnet18(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'resnet18.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "inception":
        backbone = models.inception_v3(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'inception_v3.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "densenet121":
        backbone = models.densenet121(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'densenet121.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "clip":
        backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_hidden_states=True)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        backbone_type = 'huggingface'
    elif network_name.lower() == "rad_clip":
        backbone = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32", output_hidden_states=True)
        processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        backbone_type = 'huggingface'
    elif network_name.lower() == "rad_dino":
        backbone = AutoModel.from_pretrained("microsoft/rad-dino", output_hidden_states=True)
        processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
        backbone_type = 'huggingface'
    elif network_name.lower() == "dino":
        backbone = AutoModel.from_pretrained("facebook/dinov2-base", output_hidden_states=True)
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        backbone_type = 'huggingface'
    elif network_name.lower() == "rad_inception":
        backbone = models.inception_v3(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'RadImageNet-InceptionV3_notop.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "resnet50_keras":
        backbone = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        backbone_type = 'keras'
    elif network_name.lower() == "rad_resnet50":
        backbone = models.resnet50(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'RadImageNet-ResNet50_notop.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "inceptionresnet":
        backbone = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
        backbone_type = 'keras'
    elif network_name.lower() == "rad_inceptionresnet":
        backbone = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
        backbone_type = 'keras'
    elif network_name.lower() == "rad_densenet":
        backbone = models.densenet121(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'RadImageNet-DenseNet121_notop.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "ijepa":
        backbone = IJEPAEncoder()
        backbone = loadIJEPA_state_dict(backbone)
        backbone_type = 'torch'
    else:
        raise ValueError(f"Unsupported network name: {network_name}")

    if backbone_type in ['torch', 'huggingface']:
        backbone = backbone.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return backbone, backbone_type, processor

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

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, target_resolution, transforms, num_channels=3, inference=False):
        self.paths = file_paths
        self.target_resolution = target_resolution
        self.transforms = transforms
        self.resolution_transform = tio.Resize(target_resolution)
        self.num_channels = num_channels
        self.inference = inference
    def __len__(self):
        return len(self.paths)

    def process_image(self, img_path, aug=False):

        img = np.load(img_path, mmap_mode='r+')
# Convert to PyTorch tensor
        img = torch.from_numpy(img).float().permute(2, 1, 0).unsqueeze(0)

        # Ensure the tensor is in the format (C, H, W)
        if img.ndim == 2:  # If the image is 2D (H, W)
            img = img.unsqueeze(0).repeat(3, 1, 1)  # Add channel dimension and repeat to 3 channels
        elif img.ndim == 3:
            if img.shape[2] == 3:
                img = img.permute(2, 0, 1)  # Permute from (H, W, C) to (C, H, W)
            elif img.shape[0] != 3:
                img = img.repeat(3, 1, 1)  # Repeat single channel to 3 channels
        
        # Add a dummy 'z' dimension to make it 4D (C, H, W, Z)
        
        # Convert to ScalarImage for TorchIO transforms
        img = tio.ScalarImage(tensor=img)

        # Apply resolution transform
        #img = self.resolution_transform(img)

        # Apply other transforms
        if aug:
            img = self.transforms(img)
        else:
            mandatory_aug = tio.Compose([
                tio.RescaleIntensity(out_min_max=(0, 1)),
            ])
            img = mandatory_aug(img)

        # Get the data tensor and remove the extra 'z' dimension
        img_data = img.data.squeeze(-1).squeeze(0)
        if img_data.shape[0] != 3:
            img_data = img_data.repeat(3, 1, 1)
        # Ensure the image has 3 channels

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
def clear_device_cache(garbage_collection=False):
    """
    Clears the device cache by calling `torch.{backend}.empty_cache`. Can also run `gc.collect()`, but do note that
    this is a *considerable* slowdown and should be used sparingly.
    """
    #if garbage_collection:
    #    gc.collect()

    #if is_xpu_available():
    #    torch.xpu.empty_cache()
    #elif is_mlu_available():
    #    torch.mlu.empty_cache()
    #elif is_musa_available():
    #    torch.musa.empty_cache()
    #elif is_npu_available():
    #    torch.npu.empty_cache()
    #elif is_mps_available(min_version="2.0"):
    #    torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

def find_executable_batch_size(function: callable = None, starting_batch_size: int = 32):
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size
    has_run = False  # Track if the function has been run during the first epoch

    def decorator(*args, **kwargs):
        nonlocal batch_size, has_run

        # If it's not the first epoch, skip batch size adjustment
        if has_run:
            return function(batch_size, *args, **kwargs)

        # First epoch: run the batch size adjustment logic
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")

            try:
                # Call the function with the current batch size
                result = function(batch_size, *args, **kwargs)
                has_run = True  # Mark as run after first successful call
                return result

            except Exception as e:
                # Handle exception and reduce batch size if necessary
                if should_reduce_batch_size(e):
                    clear_device_cache(garbage_collection=True)
                    batch_size //= 2
                else:
                    raise

    return decorator


def train_val_paths(data_path, config):
    
    from ensure_patient_leakage import create_train_val_split

    train_paths, val_paths = create_train_val_split(
            patient_data_dir = config['real_dataset_path'],
            patient_info_path = config['dataset_info'],
            patient_id = config['unique_individual_id'],
            secondary_ids = config['secondary_ids'],
            unique_identifier_col = config['unique_image_id'],
            train_ratio = config['split_ratio'],
            extension = config['image_extension'],
            seed = config['seed']
            )

    return train_paths, val_paths

def setup_training(root_dir, network_name, **kwargs):
    """
    Sets up the dataset, dataloaders, and device for training, and initializes the training process.
    
    Args:
        root_dir (str): Path to the root directory containing the image data.
        network_name (str): Name of the network architecture to use.
        **kwargs: Additional keyword arguments to customize the training setup.
    
    Returns:
        DataLoader, DataLoader, torch.device: Training DataLoader, Validation DataLoader, and device.
    """
    
    
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    # Set up DistributedDataParallel
    dist.init_process_group("nccl", rank=rank, world_size=local_world_size)
    torch.cuda.set_device(rank) 
    #dist.barrier()
    
    
    if rank == 0:

        # Update defaults with provided kwargs
        config = {**kwargs}
        print(f'Starting trainign wiht the following config: \n {config}')

        # Select target reolution for eahc specific model 
        target_res = (224, 224)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create a unique directory name combining timestamp and network name
        run_name = f"{timestamp}_{network_name}"
        wandb.init(project="privacy_benchmark", name=run_name, config=config)

        # Set up checkpoint directory
        ckpt_dir = os.path.join('./checkpoints', run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save config in a json
        json_file_path = os.path.join(ckpt_dir, f"config.json")

        with open(json_file_path, 'w') as json_file:
            json.dump(config, json_file, indent=2)

        # Create the dataset
        train_paths, val_paths = train_val_paths(root_dir, config)

        # Ensure target_res is 3D
        if len(target_res) == 2:
            target_res = (*target_res, 1)
        
        num_channels = 3    # Create separate datasets for train and validation
        
        from utils import convert_to_npy
        
        # Save paths to JSON file
        paths_dict = {
            "train_paths": train_paths,
            "val_paths": val_paths
        }
        json_file_path = os.path.join(ckpt_dir, f"{run_name}_image_paths.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(paths_dict, json_file, indent=2)

        print(f"Image paths saved to {json_file_path}")    
        
        train_paths = convert_to_npy(train_paths, os.path.join(config['output_preprocessing'], 'train'), target_res)
        val_paths = convert_to_npy(val_paths, os.path.join(config['output_preprocessing'], 'val'), target_res)
        # Prepare data for broadcasting
        data_to_broadcast = {
            'config': config,
            'run_name': run_name,
            'ckpt_dir': ckpt_dir,
            'timestamp': timestamp,
            'train_paths': train_paths,
            'val_paths': val_paths,
            'target_res': target_res,
            'num_channels': num_channels
        }
    else:
        data_to_broadcast = None


    # Broadcast data from rank 0 to all other ranks
    data_to_broadcast = [data_to_broadcast]
    dist.broadcast_object_list(data_to_broadcast, src=0, device=torch.cuda.current_device())
    data_to_broadcast = data_to_broadcast[0]
    # Ensure all processes have received the broadcasted data

    # Unpack the broadcasted data
    if rank != 0:
        config = data_to_broadcast['config']
        run_name = data_to_broadcast['run_name']
        ckpt_dir = data_to_broadcast['ckpt_dir']
        train_paths = data_to_broadcast['train_paths']
        val_paths = data_to_broadcast['val_paths']
        timestamp = data_to_broadcast['timestamp']
        target_res = data_to_broadcast['target_res']
        num_channels = data_to_broadcast['num_channels']

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # Create modle and move it to gpu wiht rank id X
    device = rank % torch.cuda.device_count()


    if 'clip' in network_name:
        config['batch_size'] = 64 # Define model
    try:
        
        model = ContrastiveNetwork(network_name, feature_dim = config['n_features'], attentive_probing = config['attentive_probing'], device = device)
    except ValueError as e:
        print(network_name, e)
        raise ValueError(f'Error {e} with {network_name}')


    #if isinstance(model.backbone, models.Inception3):
    #    target_res = (299, 299)

    model.to(device)
    #if 'clip' in network_name or 'dino' in network_name:
    model = DDP(model, device_ids = [device], output_device=rank, broadcast_buffers=False, find_unused_parameters=True)
    #else:

    #    model = DDP(model, device_ids = [device], output_device=rank, broadcast_buffers=False)

    


   # Define preprocessing and training transforms
    MANDATORY_TRANSFORMS = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])

# Define augmentation transforms for training
    AUGMENTATION_TRANSFORMS = tio.Compose([
        Custom2DRotation(degrees=20, p=0.7),
        tio.RandomAffine(
            degrees=(-0, 0),  # Reduced rotation range
            scales=(0.8, 1.2),  # Reduced scaling range
            default_pad_value='minimum',
            p=0.5
        ),
        tio.RandomFlip(axes=(2), flip_probability=0.5),
        tio.RandomFlip(axes=(1), flip_probability=0.5),
        tio.RandomBiasField(
            coefficients=0.3,  # Reduced coefficient for less intense bias field
            order=3,
            p=0.4
        ),
        tio.RandomGamma(
            log_gamma=(-0.1, 0.1),  # Reduced range for less intense gamma correction
            p=0.4
        ),
        tio.RandomNoise(std=(0, 0.05), p=0.3),  # Added some noise for texture
        tio.RandomBlur(std=(0, 1), p=0.2),  # Added slight blur for realism
        tio.RandomMotion(degrees=5, translation=5, p=0.2),
    ])



    #AUGMENTATION_TRANSFORMS = tio.Compose([
    #    Custom2DRotation(degrees=20, p=0.7),
    #    tio.RandomAffine(
    #        degrees=(-0, 0),  # Reduced rotation range
    #        scales=(0.8, 1.2),  # Tighter scaling range
    #        image_interpolation='bspline',
    #        p=0.7,
    #    ),
    #    tio.RandomFlip(axes=(2,)),
    #    tio.RandomNoise(std=(0, 0.02)),
    #    tio.RandomBlur(std=(0, 0.5)),
    #    tio.RandomMotion(degrees=5, translation=5),
    #   
    #])
# Combine mandatory and augmentation transforms for training
    TRAIN_TRANSFORMS = tio.Compose([
        MANDATORY_TRANSFORMS,
        AUGMENTATION_TRANSFORMS
    ])

# Validation only uses mandatory transforms
    VAL_TRANSFORMS = MANDATORY_TRANSFORMS

    
    train_dataset = ContrastiveDataset(
        file_paths=train_paths,
        target_resolution=target_res,
        transforms=TRAIN_TRANSFORMS,
        num_channels=num_channels
    )

    val_dataset = ContrastiveDataset(
        file_paths=val_paths,
        target_resolution=target_res,
        transforms=VAL_TRANSFORMS,
        num_channels=num_channels
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=local_world_size, rank=rank, shuffle = False, drop_last = False)
    val_sampler = DistributedSampler(val_dataset, num_replicas=local_world_size, rank=rank, shuffle = False, drop_last = False) 

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True,
        sampler=train_sampler
    )
    from utils import MultiEpochsDataLoader

    loader_args = dict(
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            persistent_workers=True,
            sampler=train_sampler
            )

    train_loader = MultiEpochsDataLoader(
            train_dataset,
            **loader_args 
            )
    #print('Saving a set of iamges and its augmented versions')
    #visualize_augmentations_contrastive(train_loader, num_images=4, num_augmentations=3)
    loader_args = dict(
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            persistent_workers=True,
            sampler=val_sampler
            )

    val_loader = MultiEpochsDataLoader(
            val_dataset,
            **loader_args 
            )

  
      # Define optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['base_lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['gamma'])

       # Choose loss function based on config
    if config['loss_type'].lower() == 'ntxent':
        LossTr = NTXentLoss(temperature=config['temperature'])
    elif config['loss_type'].lower() == 'triplet':
        LossTr = losses.TripletMarginLoss(
            margin=config['margin'],
            swap=config['swap'],
            smooth_loss=config['smooth_loss'],
            triplets_per_anchor=config['triplets_per_anchor']
        )
    else:
        raise ValueError(f"Unsupported loss type: {config['loss_type']}. Choose 'ntxent' or 'triplet'.")
 
    miner = miners.BatchHardMiner()
    cosine_similarity = CosineSimilarity()
    
    # Training loop
    val_losses = []

    try:
        # Freeze the backbone initially
        freeze_backbone(model)
        
        print('Starting training')    
        for epoch in range(config['n_epochs']):

            if epoch == config['unfreeze_epoch']:
                print(f"Unfreezing the entire network at epoch {epoch}")
                unfreeze_backbone(model)
                #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['n_epochs'], eta_min=config['base_lr']/config['downscaling_after_freezing'])

            epoch_loss = train_epoch(config['batch_size'], model, train_loader, optimizer, scheduler, device, LossTr, epoch, config, miner)

            val_loss, pos_sim, euc_pos_dist, euc_neg_dist = validate(model, val_loader, device, LossTr, config, miner)

            
            val_losses.append(val_loss)

            
            # Save model checkpoints
            if rank ==0:
                if (epoch + 1) % config['save_model_interval'] == 0 or epoch == 0:
                    save_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth")
                    torch.save(model.module.state_dict() if config['multi_gpu'] else model.state_dict(), save_path)
                    #wandb.save(save_path)

                # Save best model
                if val_loss <= min(val_losses):
                    best_path = os.path.join(ckpt_dir, "model_best.pth")
                    torch.save(model.module.state_dict() if config['multi_gpu'] else model.state_dict(), best_path)
                    #wandb.save(best_path)

            
            # Logging with wandbi
            if rank ==0:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "batch_size": train_loader.batch_sampler.batch_size,
                    #"positive_samples": wandb.Histogram(euc_pos_dist),
                    #"negative_samples": wandb.Histogram(euc_neg_dist),
                    "learning rate": optimizer.param_groups[0]['lr']
                })

            dist.barrier()            

        wandb.finish()
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        setup_logging(ckpt_dir) 
        error_msg = f"An error occurred:\n{traceback.format_exc()}"
        logging.error(error_msg)
        
        raise

    return train_loader, val_loader, device

def freeze_backbone(model):
    print('Freezing backbone')
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True

#@find_executable_batch_size(starting_batch_size=config['batch_size'])
def train_epoch(batch_size, model, train_loader, optimizer, scheduler, device, LossTr, epoch, config, miner=None):
    torch.autograd.set_detect_anomaly(True)

    model.train()
    epoch_loss = 0
    train_loader.batch_sampler.batch_size = batch_size  # Update batch size
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")

    for train_step, batch in progress_bar:
        optimizer.zero_grad()
        Pos11 = batch['data'].to(device)
        Pos12 = batch['data_pos'].to(device)
        
        # Ensure input is 4D: [batch_size, channels, height, width]
        if Pos11.dim() != 4:
            raise ValueError(f"Expected 4D input, got {Pos11.dim()}D with shape {Pos11.shape}")
       
        PosEmb11 = model(Pos11)
        PosEmb12 = model(Pos12)

        # Create labels and positives mask
        batch_size = PosEmb11.shape[0]
        labels = torch.arange(batch_size, device=device)
        combined_labels = torch.cat([labels, labels])
        
        positives_mask = torch.eye(batch_size, device=device).bool()
        combined_positives_mask = torch.cat([positives_mask, positives_mask], dim=1)

        # Combine embeddings
        embeddings = torch.cat((PosEmb11, PosEmb12), dim=0)

        # Use miner if provided
        if miner is not None:
            miner_output = miner(embeddings, combined_labels)
        else:
            miner_output = None

        # Compute loss based on the specified loss type
        LossPos1 = LossTr(embeddings, combined_labels, miner_output)
       
     
        LossPos1.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += LossPos1.item()
        progress_bar.set_postfix({'loss': f'{LossPos1.item():.4f}'})
        
    return epoch_loss / len(train_loader)

def validate(model, val_loader, device, LossTr, config, miner=None):
    model.eval()
    val_loss = 0
    pos_sim, neg_sim, neg_sim_aug = [], [], []
    euc_pos_dist, euc_neg_dist = [], []
    
    for batch in val_loader:
        Pos11 = batch['data'].to(device)
        Pos12 = batch['data_pos'].to(device)
        with torch.no_grad():
            PosEmb11 = model(Pos11)
            PosEmb12 = model(Pos12)

        batch_size = PosEmb11.shape[0]
        labels = torch.arange(batch_size, device=device)
        combined_labels = torch.cat([labels, labels])
        
        positives_mask = torch.eye(batch_size, device=device).bool()
        combined_positives_mask = torch.cat([positives_mask, positives_mask], dim=1)

        embeddings = torch.cat((PosEmb11, PosEmb12), dim=0)

        if miner is not None:
            miner_output = miner(embeddings, combined_labels)
        else:
            miner_output = None

        val_loss += LossTr(embeddings, combined_labels, miner_output).item()
     
        cosine_similarity = CosineSimilarity()
        similarity_pos = cosine_similarity(PosEmb11, PosEmb12).cpu().numpy()
        similarity_neg = cosine_similarity(PosEmb11, PosEmb11).cpu().numpy()

        pos_sim.append(np.diag(similarity_pos))
        neg_sim.append(similarity_neg[np.triu_indices_from(similarity_neg, k=1)])
        neg_sim_aug.append(similarity_pos[np.triu_indices_from(similarity_pos, k=1)])

        # Euclidean distance calculations
        from torch.nn.functional import pairwise_distance
        euclidean_distance_pos = pairwise_distance(PosEmb11, PosEmb12).cpu().numpy()
        euclidean_distance_neg = pairwise_distance(PosEmb11, PosEmb11).cpu().numpy()
        
        # Ensure euclidean_distance_neg is 2D
        if euclidean_distance_neg.ndim == 1:
            euclidean_distance_neg = euclidean_distance_neg.reshape(1, -1)
        
        euc_pos_dist.append(np.diag(euclidean_distance_pos))
        euc_neg_dist.append(euclidean_distance_neg[np.triu_indices_from(euclidean_distance_neg, k=1)])

    torch.cuda.empty_cache()

    val_loss /= len(val_loader)
     # Flatten the lists of arrays into 1D arrays
    #pos_sim = np.concatenate(pos_sim)
    #neg_sim = np.concatenate(neg_sim)
    #neg_sim_aug = np.concatenate(neg_sim_aug)
    #euc_pos_dist = np.concatenate(euc_pos_dist)
    #euc_neg_dist = np.concatenate(euc_neg_dist)

    return val_loss, pos_sim, euc_pos_dist, euc_neg_dist



def train_by_models(real_data_dir: str, network_names: list, **kwargs):
    """
    Train models using specified network architectures and configurations.
    
    Args:
        real_data_dir (str): Directory containing the real data.
        network_names (list): List of network architectures to train.
        config_path (str, optional): Path to the YAML configuration file.
        **kwargs: Additional keyword arguments to override config file settings.
    """
    
    for network_name in network_names:
        print(f"Training {network_name}...")

        try:
            train_loader, val_loader, device = setup_training(
                root_dir=real_data_dir,
                network_name=network_name,
            )

        except Exception as e:

            error_msg = f"An error ocurredL \n {traceback.format_exc()}"
            logging.error(error_msg)

            raise
        

def load_best_model_for_inference(network_name, config, checkpoints_dir='./checkpoints', batch_size=32, num_workers=4):
    print(f"Loading best model for inference for network: {network_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the model

    if config['contrastive']:
        model = ContrastiveNetwork(network_name, feature_dim = config['n_features'], attentive_probing = config['attentive_probing'], device=device)

        # Find the most recent checkpoint directory for this network
        network_checkpoints = glob.glob(os.path.join(checkpoints_dir, f"*_{network_name}"))
        print(f"Found checkpoint directories: {network_checkpoints}")
        
        if not network_checkpoints:
            raise ValueError(f"No checkpoints found for {network_name}")

        latest_checkpoint_dir = max(network_checkpoints, key=os.path.getctime)
        print(f"Latest checkpoint directory: {latest_checkpoint_dir}")

        # Extract date from the latest checkpoint directory
        checkpoint_date = os.path.basename(latest_checkpoint_dir).split('_')[0]
        print(f"Extracted checkpoint date: {checkpoint_date}")

        # Look for the best model in this directory
        best_model_path = os.path.join(latest_checkpoint_dir, "model_best.pth")

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found at {best_model_path}")

        # Load the state dict
        state_dict = torch.load(best_model_path, map_location='cpu')

        # Load the state dict into the model
        try:
            model.load_state_dict(state_dict, strict = True)
        except:
            model.load_state_dict(state_dict, strict = False)

        # Find the JSON file with a matching pattern
        json_pattern = os.path.join(latest_checkpoint_dir, f"*{network_name}_image_paths.json")

    elif not config['contrastive'] and network_name == 'ijepa':

        model = load_ijepa_not_contrastive(config)
   
        json_pattern = os.path.join(config['ijepa_model_dir'], 'image_paths.json')


    model.eval()

    # Move the model to GPU if available
    if config['multi_gpu']:
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load dataloaders

    json_files = glob.glob(json_pattern)
    print(f"Found JSON files: {json_files}")

    if not json_files:
        raise FileNotFoundError(f"No JSON file found matching pattern {json_pattern}")

    # Sort JSON files by modification time and select the most recent one
    json_path = max(json_files, key=os.path.getmtime)
    print(f"Selected JSON file: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    train_paths, val_paths = load_image_paths(json_path)
    synthetic_paths = [os.path.join(config['synthetic_dataset_path'], file) for file in os.listdir(config['synthetic_dataset_path']) if file.endswith(('.jpg', '.jpeg', '.png'))]
    train_standard_loader, val_loader, synth_loader = create_dataloaders(
        train_paths, val_paths, synthetic_paths, config)

    return model, device, train_standard_loader, val_loader, synth_loader

class ImageDataset(Dataset):
    def __init__(self, image_paths, target_resolution: int = 224, transform=None):
        self.image_paths = image_paths
        self.target_resolution = target_resolution
        self.transform = transform
        self.resolution_transform = tio.Resize(target_resolution)
        self.mandatory_aug = tio.Compose([
                tio.RescaleIntensity(out_min_max=(0, 1)),
            ])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = tio.ScalarImage(img_path)
        
        # Print original shape for debugging
        #print(f"Original shape: {img.shape}")
        
        # Get the data as a PyTorch tensor
        img_data = img.data
        
        # Ensure the tensor is 4D (batch, channels, height, width)
        if len(img_data.shape) == 2:  # (height, width)
            img_data = img_data.unsqueeze(0).unsqueeze(0)
        elif len(img_data.shape) == 3:  # (channels, height, width)
            img_data = img_data.unsqueeze(0)
        elif len(img_data.shape) == 4:  # (batch, channels, height, width)
            pass  # Already in the correct format
        else:
            raise ValueError(f"Unexpected tensor shape: {img_data.shape}") 

        # Print shape after dimension adjustment
        #print(f"Shape after adjustment: {img_data.shape}")
        
        # Create a new ScalarImage with the adjusted data
        img = tio.ScalarImage(tensor=img_data)
        
        # Apply resolution transform
        img_data = self.resolution_transform(img)

        # Print shape after resizing
        #print(f"Shape after resizing: {img_data.shape}")
        
        if self.transform:
            print('Applying transform')
            img = self.transform(img_data)
            print(f'SHpare after transform {img}')
        else:
            img = self.mandatory_aug(img)
            print(img.shape)

        return img, img_path

class AdversarialDataset(ImageDataset):
    def __init__(self, image_paths, target_resolution):
        super().__init__(image_paths)
        self.transform = self.create_strong_augmentation()
        self.target_resolution = target_resolution

    def create_strong_augmentation(self):
        return tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            Custom2DRotation(degrees=20, p=0.7),
            tio.RandomAffine(
                degrees=0,
                scales=(0.8, 1.2),
                default_pad_value='minimum',
                p=0.5
            ),
            tio.RandomFlip(axes=(1, 2), flip_probability=0.5),  # Changed from (0, 1) to (1, 2)
            tio.RandomBiasField(
                coefficients=0.3,
                order=3,
                p=0.4
            ),
            tio.RandomGamma(
                log_gamma=(-0.1, 0.1),
                p=0.4
            ),
            tio.RandomNoise(std=(0, 0.05), p=0.3),
            tio.RandomBlur(std=(0, 1), p=0.2),
            tio.RandomMotion(degrees=5, translation=5, p=0.2),
        ])
        
def load_image_paths(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['train_paths'], data['val_paths']

def create_dataloaders(train_paths, val_paths, synthetic_paths, config):
    MANDATORY_TRANSFORMS = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])
    if config['augmentation_strength'] == 'weak':
        AUGMENTATION_TRANSFORMS = tio.Compose([
            Custom2DRotation(degrees=20, p=0.7),
            tio.RandomAffine(
                degrees=(-0, 0),  # Reduced rotation range
                scales=(0.8, 1.2),  # Reduced scaling range
                default_pad_value='minimum',
                p=0.5
            ),
            tio.RandomFlip(axes=(2), flip_probability=0.5),
            tio.RandomFlip(axes=(1), flip_probability=0.5),
            tio.RandomBiasField(
                coefficients=0.3,  # Reduced coefficient for less intense bias field
                order=3,
                p=0.4
            ),
            tio.RandomGamma(
                log_gamma=(-0.1, 0.1),  # Reduced range for less intense gamma correction
                p=0.4
            ),
            tio.RandomNoise(std=(0, 0.05), p=0.3),  # Added some noise for texture
            tio.RandomBlur(std=(0, 1), p=0.2),  # Added slight blur for realism
            tio.RandomMotion(degrees=5, translation=5, p=0.2),
        ])
    elif config['augmentation_strength'] == 'strong':
        
        AUGMENTATION_TRANSFORMS = tio.Compose([
            Custom2DRotation(degrees=20, p=1),
            tio.RandomAffine(
                degrees=(-0, 0),  # Reduced rotation range
                scales=(0.6, 1.4),  # Reduced scaling range
                default_pad_value='minimum',
                p=1
            ),
            tio.RandomFlip(axes=(2), flip_probability=0.6),
            tio.RandomFlip(axes=(1), flip_probability=0.6),
            tio.RandomBiasField(
                coefficients=0.3,  # Reduced coefficient for less intense bias field
                order=3,
                p=0.8
            ),
            tio.RandomGamma(
                log_gamma=(-0.1, 0.1),  # Reduced range for less intense gamma correction
                p=0.8
            ),
            tio.RandomNoise(std=(0, 0.05), p=0.8),  # Added some noise for texture
            tio.RandomBlur(std=(0, 1), p=0.8),  # Added slight blur for realism
            tio.RandomMotion(degrees=5, translation=5, p=0.7),
        ])


    # Combine mandatory and augmentation transforms for training
    TRAIN_TRANSFORMS = tio.Compose([
        MANDATORY_TRANSFORMS,
        AUGMENTATION_TRANSFORMS
    ])

    VAL_TRANSFORMS = TRAIN_TRANSFORMS

    # Ensure target_res is 3D
    target_res = config['target_resolution']

    if len(target_res) == 2:
        target_res = (*target_res, 1)
    
    num_channels = 3    # Create separate datasets for train and validation

    from utils import convert_to_npy
    
    train_paths = convert_to_npy(train_paths, os.path.join(config['output_preprocessing'], 'train'), target_res)
    val_paths = convert_to_npy(val_paths, os.path.join(config['output_preprocessing'], 'val'), target_res)
    synthetic_paths = convert_to_npy(synthetic_paths, os.path.join(config['output_preprocessing'], 'synth'), target_res)

    train_dataset = ContrastiveDataset(
        file_paths=train_paths,
        target_resolution=target_res,
        transforms=TRAIN_TRANSFORMS,
        num_channels=num_channels,
        inference=True
    )

    val_dataset = ContrastiveDataset(
        file_paths=val_paths,
        target_resolution=target_res,
        transforms=VAL_TRANSFORMS,
        num_channels=num_channels,
        inference=True
    )

    synth_dataset = ContrastiveDataset(
            file_paths = synthetic_paths,
            target_resolution=target_res,
            transforms=MANDATORY_TRANSFORMS,
            num_channels=num_channels,
            inference=True
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['inference_bs'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['inference_bs'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True
    )

    synth_loader = DataLoader(
        synth_dataset,
        batch_size=config['inference_bs'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True

        )

    return train_loader, val_loader, synth_loader

def inference_and_save_embeddings(model, device, train_standard_loader, val_standard_loader, synth_loader, network_name, output_dir, config):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    visualize_augmentations_contrastive(train_standard_loader)
    def process_dataloader(dataloader, name, model, device, adversarial=False):
        embeddings = []
        embeddings_adv = []
        image_ids = []
        image_ids_adv = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Processing {name} data"):
                try:
                    images = batch['data']
                    if adversarial:
                        images_adv = batch['data_pos']
                    
                    images = images.to(device)

                    if adversarial:
                        images_adv = images_adv.to(device)
                    
                    current_image_ids = batch['img_id']

                    output = model(images)
                    if len(output.shape)==3:
                        output = output.mean(dim=1)
                    if adversarial:
                        output_adv = model(images_adv)
                        if len(output_adv.shape)==3:
                            output_adv = output_adv.mean(dim=1)

                    if not isinstance(output, torch.Tensor):
                        output = output.logits if hasattr(output, 'logits') else output[0]

                    embeddings.append(output.cpu().numpy())
                    image_ids.extend(current_image_ids)

                    if adversarial:
                        if not isinstance(output_adv, torch.Tensor):
                            output_adv = output_adv.logits if hasattr(output_adv, 'logits') else output[0]

                        embeddings_adv.append(output_adv.cpu().numpy())
                        image_ids_adv.extend(current_image_ids)

                except Exception as e:
                    print(f"Error processing batch in {name} dataloader: {str(e)}")
                    print(f"Batch type: {type(batch)}")
                    if isinstance(images, torch.Tensor):
                        print(f"Images shape: {images.shape}")
                    raise
        if adversarial:
            return np.vstack(embeddings), image_ids, np.vstack(embeddings_adv), image_ids_adv
        else:
            return np.vstack(embeddings), image_ids, None, None
    try:
        print('Proccessing train standard')
        train_standard_embeddings, train_standard_ids, train_adversarial_embeddings, train_adversarial_ids= process_dataloader(train_standard_loader, "train_standard", model, device, adversarial=True)
        print()
        print('Processing val')
        val_standard_embeddings, val_standard_ids, val_adversarial_embeddings, val_adversarial_ids= process_dataloader(val_standard_loader, "val_standard", model, device, adversarial=True)
        print()
        print('Processing synthetic')
        synth_standard_embeddings, synth_standard_ids, _, _ = process_dataloader(synth_loader, "synth_standard", model, device, adversarial = False)

        output_file = os.path.join(output_dir, f"{network_name}_embeddings_aug_{config['augmentation_strength']}.h5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('train_standard/embeddings', data=train_standard_embeddings)
            f.create_dataset('train_standard/image_ids', data=np.array(train_standard_ids, dtype=h5py.special_dtype(vlen=str)))

            f.create_dataset('train_adversarial/embeddings', data=train_adversarial_embeddings)
            f.create_dataset('train_adversarial/image_ids', data=np.array(train_adversarial_ids, dtype=h5py.special_dtype(vlen=str)))

            f.create_dataset('val_standard/embeddings', data=val_standard_embeddings)
            f.create_dataset('val_standard/image_ids', data=np.array(val_standard_ids, dtype=h5py.special_dtype(vlen=str)))
            
            f.create_dataset('val_adversarial/embeddings', data=val_adversarial_embeddings)
            f.create_dataset('val_adversarial/image_ids', data=np.array(val_adversarial_ids, dtype=h5py.special_dtype(vlen=str)))

            f.create_dataset('synth_standard/embeddings', data=synth_standard_embeddings)
            f.create_dataset('synth_standard/image_ids', data=np.array(synth_standard_ids, dtype=h5py.special_dtype(vlen=str)))

        print(f"Embeddings saved to {output_file}")
    except Exception as e:
        print(f"Error processing {network_name}: {str(e)}")
        print(traceback.format_exc())
        raise Exception

    return output_file


def compute_pearson_correlation(X, Y):
    X_centered = X - X.mean(axis=1, keepdims=True)
    Y_centered = Y - Y.mean(axis=1, keepdims=True)
    X_normalized = X_centered / np.linalg.norm(X_centered, axis=1, keepdims=True)
    Y_normalized = Y_centered / np.linalg.norm(Y_centered, axis=1, keepdims=True)
    correlations = np.dot(X_normalized, Y_normalized.T)
    return np.mean(correlations, axis=1)

def compute_spearman_correlation(X, Y):
    def rank_data(data):
        return np.argsort(np.argsort(data, axis=1), axis=1) + 1

    X_ranked = rank_data(X)
    Y_ranked = rank_data(Y)
    return compute_pearson_correlation(X_ranked, Y_ranked)

def compute_distances_and_plot(embeddings_file, output_dir, config, methods='euclidean'):
    print(f"Loading embeddings from {embeddings_file}")
    with h5py.File(embeddings_file, 'r') as f:
        train_standard_embeddings = f['train_standard/embeddings'][:]
        train_adversarial_embeddings = f['train_adversarial/embeddings'][:]
        val_standard_embeddings = f['val_standard/embeddings'][:]
        val_adversarial_embeddings = f['val_adversarial/embeddings'][:]
        synth_standard_embeddings = f['synth_standard/embeddings'][:]

    print(f"Shapes: train_standard={train_standard_embeddings.shape}, "
          f"train_adversarial={train_adversarial_embeddings.shape}, "
          f"val_standard={val_standard_embeddings.shape}, "
          f"val_adversarial={val_adversarial_embeddings.shape}, "
          f"synth_standard={synth_standard_embeddings.shape}")

    if isinstance(methods, str):
        methods = [methods]

    print(f"Computing distances/correlations for methods: {methods}")

    results = {}
    for method in methods:
        print(f"\nProcessing method: {method}")
        if method in ['euclidean', 'sqeuclidean']:
            print(f"Computing {method} distances")
            train_val_distances = cdist(train_standard_embeddings, val_standard_embeddings, metric=method)
            adversarial_train_distances = cdist(train_adversarial_embeddings, train_standard_embeddings, metric=method)
            adversarial_val_distances = cdist(val_adversarial_embeddings, val_standard_embeddings, metric=method)
            synth_train_distances = cdist(synth_standard_embeddings, train_standard_embeddings, metric=method)

            min_train_val_distances = np.min(train_val_distances, axis=1)
            min_adversarial_train_distances = np.min(adversarial_train_distances, axis=1)
            min_adversarial_val_distances = np.min(adversarial_val_distances, axis=1)
            min_synth_train_distances = np.min(synth_train_distances, axis=1)

            print(f"Distance shapes: train_val={min_train_val_distances.shape}, "
                  f"adversarial_train={min_adversarial_train_distances.shape}, "
                  f"adversarial_val={min_adversarial_val_distances.shape}, "
                  f"synth_train={min_synth_train_distances.shape}")

            results[method] = {
                'train_val': min_train_val_distances,
                'adversarial_train': min_adversarial_train_distances,
                'adversarial_val': min_adversarial_val_distances,
                'synth_train': min_synth_train_distances
            }

        elif method in ['pearson', 'spearman']:
            print(f"Computing {method} correlation")

            if method == 'pearson':
                train_val_corr = compute_pearson_correlation(train_standard_embeddings, val_standard_embeddings)
                adversarial_train_corr = compute_pearson_correlation(train_adversarial_embeddings, train_standard_embeddings)
                adversarial_val_corr = compute_pearson_correlation(val_adversarial_embeddings, val_standard_embeddings)
                synth_train_corr = compute_pearson_correlation(synth_standard_embeddings, train_standard_embeddings)
            elif method == 'spearman':
                train_val_corr = compute_spearman_correlation(train_standard_embeddings, val_standard_embeddings)
                adversarial_train_corr = compute_spearman_correlation(train_adversarial_embeddings, train_standard_embeddings)
                adversarial_val_corr = compute_spearman_correlation(val_adversarial_embeddings, val_standard_embeddings)
                synth_train_corr = compute_spearman_correlation(synth_standard_embeddings, train_standard_embeddings)

            print(f"Correlation shapes: train_val={train_val_corr.shape}, "
                  f"adversarial_train={adversarial_train_corr.shape}, "
                  f"adversarial_val={adversarial_val_corr.shape}, "
                  f"synth_train={synth_train_corr.shape}")

            results[method] = {
                'train_val': train_val_corr,
                'adversarial_train': adversarial_train_corr,
                'adversarial_val': adversarial_val_corr,
                'synth_train': synth_train_corr
            }
    # Selected how many copy candidates are found
    method = methods[0]
    ndpctl = np.percentile(results[method]['train_val'], 2)
    ndpctl_high = np.percentile(results[method]['train_val'], 98) 

    adv_train_detection_count = 0
    adv_val_detection_count = 0 

    total_adv_train = len(results[method]['adversarial_train'])
    total_adv_val = len(results[method]['adversarial_val'])

    for dist in results[method]['adversarial_train']:
        if method in ['euclidean', 'sqeuclidean']:
            if dist < ndpctl:
                adv_train_detection_count += 1
        else:

            if dist > ndpctl_high:
                adv_train_detection_count += 1

    for dist in results[method]['adversarial_val']:
        if method in ['euclidean', 'sqeuclidean']:
            if dist < ndpctl:
                adv_val_detection_count += 1
        else:
            if dist > ndpctl_high:
                adv_val_detection_count += 1

    dist_trval_synth = wasserstein_distance(results[method]['synth_train'], results[method]['train_val'])
    dist_adv_train_trval = wasserstein_distance(results[method]['adversarial_train'], results[method]['train_val'])
    dist_adv_val_trval = wasserstein_distance(results[method]['adversarial_val'], results[method]['train_val'])
    
    detected_adv_train_pct = adv_train_detection_count / total_adv_train
    detected_adv_val_pct = adv_val_detection_count / total_adv_val
    
    adv_results = {
            'wasserstein_1d_train_val_train_synth': dist_trval_synth,
            'wasserstein_1d_train_val_train_train_adv': dist_adv_train_trval,
            'wasserstein_1d_train_val_train_val_adv': dist_adv_val_trval,
            'detection_ratio_train': detected_adv_train_pct,
            'detection_ratio_val': detected_adv_val_pct
            }

    json_file_path = os.path.join(output_dir, os.path.basename(embeddings_file).replace('.h5', f"aug_{config['augmentation_strength']}_results.json"))

    with open(json_file_path, 'w') as json_file:
        json.dump(adv_results, json_file)

    print("\nPlotting results")
    n_methods = len(methods)

    fig, axs = plt.subplots(n_methods, 1, figsize=(12, 10*n_methods), dpi=300)
    if n_methods == 1:
        axs = [axs]

    for ax, (method, data) in zip(axs, results.items()):
        print(f"Plotting for method: {method}")
        for key in ['train_val', 'adversarial_train', 'adversarial_val', 'synth_train']:
            valid_data = data[key][~np.isnan(data[key]) & ~np.isinf(data[key])]
            print(f"{key}: {len(valid_data)} valid data points")
            if len(valid_data) > 0:
                sns.histplot(valid_data, bins=80, kde=True, stat="density", alpha=0.3, label=key.replace('_', '-').title(), ax=ax)
                sns.kdeplot(valid_data, ax=ax, linewidth=2)
            else:
                print(f"Warning: No valid data for {key} in {method}")

        ax.set_xlabel(f'{method.capitalize()} {"Distance" if method in ["euclidean", "sqeuclidean"] else "Correlation"}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of {method.capitalize()} {"Distances" if method in ["euclidean", "sqeuclidean"] else "Correlations"}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Add a text box with statistics
    plt.tight_layout()
    plot_file = os.path.join(output_dir, os.path.basename(embeddings_file).replace('.h5', f"_aug_{config['augmentation_strength']}_{'-'.join(methods)}_distribution.png"))
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to {plot_file}")

    print("\nComputing statistics")
    stats = {}
    for method, data in results.items():
        for key in ['train_val', 'adversarial_train', 'synth_train']:
            valid_data = data[key][~np.isnan(data[key]) & ~np.isinf(data[key])]
            if len(valid_data) > 0:
                stats[f'{method}_{key}_mean'] = np.mean(valid_data)
                stats[f'{method}_{key}_median'] = np.median(valid_data)
                print(f"{method}_{key}: mean={stats[f'{method}_{key}_mean']}, median={stats[f'{method}_{key}_median']}")
            else:
                stats[f'{method}_{key}_mean'] = np.nan
                stats[f'{method}_{key}_median'] = np.nan
                print(f"Warning: No valid data for {key} in {method}")

    return stats



import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import seaborn as sns 

def decode_byte_strings(data):
    if isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, np.ndarray):
        return np.array([item.decode('utf-8') if isinstance(item, bytes) else item for item in data])
    else:
        return data
import re

def extract_model_name(file_path):
    # Extract the filename from the path
    base_name = os.path.basename(file_path)
    
    # Remove the file extension
    name_without_extension = os.path.splitext(base_name)[0]
    
    # Define patterns to match
    patterns = [
        r'^(\w+)_embeddings$',  # matches "densenet121_embeddings"
        r'^rad_(\w+)_embeddings$',  # matches "rad_densenet_embeddings"
        r'^(\w+)$'  # fallback to match any word characters if above patterns fail
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.match(pattern, name_without_extension)
        if match:
            return match.group(1)
    
    # If no pattern matches, return the whole name without extension
    return name_without_extension

def prepare_image_for_display(img):
    """Prepare the image for display, handling different formats without rotation."""
    if img.shape[0] == 3 or img.shape[0] == 1:  # (C, H, W) format
        img = img.T
    #
    # Normalize the image if it's not in 0-255 range
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    return img

def find_and_plot_similar_images(embeddings_file, train_dataloader, val_dataloader, synth_dataloader, output_dir, plot_percentage=1.0, validation_percentage=1):
    if not 0 <= plot_percentage <= 1 or not 0 < validation_percentage <= 1:
        raise ValueError("plot_percentage and validation_percentage must be between 0 and 1")

    print(f"Loading embeddings from {embeddings_file}")
    with h5py.File(embeddings_file, 'r') as f:
        train_standard_embeddings = f['train_standard/embeddings'][:]
        val_standard_embeddings = f['val_standard/embeddings'][:]
        synth_standard_embeddings = f['synth_standard/embeddings'][:]
        train_image_ids = decode_byte_strings(f['train_standard/image_ids'][:])
        val_image_ids = decode_byte_strings(f['val_standard/image_ids'][:])
        synth_image_ids = decode_byte_strings(f['synth_standard/image_ids'][:])

    print(f"Original shapes: train={train_standard_embeddings.shape}, val={val_standard_embeddings.shape}, synth={synth_standard_embeddings.shape}")

    # Subsample embeddings and image_ids for validation
    num_train = max(int(len(train_image_ids) * validation_percentage), 1)
    num_val = max(int(len(val_image_ids) * validation_percentage), 1)
    num_synth = max(int(len(synth_image_ids) * validation_percentage), 1)

    train_indices = random.sample(range(len(train_image_ids)), num_train)
    val_indices = random.sample(range(len(val_image_ids)), num_val)
    synth_indices = random.sample(range(len(synth_image_ids)), num_synth)

    train_standard_embeddings = train_standard_embeddings[train_indices]
    val_standard_embeddings = val_standard_embeddings[val_indices]
    synth_standard_embeddings = synth_standard_embeddings[synth_indices]
    train_image_ids = train_image_ids[train_indices]
    val_image_ids = val_image_ids[val_indices]
    synth_image_ids = synth_image_ids[synth_indices]

    print(f"Subsampled shapes: train={train_standard_embeddings.shape}, val={val_standard_embeddings.shape}, synth={synth_standard_embeddings.shape}")

    print("Computing distances between synthetic and training/validation images")
    train_distances = cdist(synth_standard_embeddings, train_standard_embeddings, metric='sqeuclidean')
    val_distances = cdist(synth_standard_embeddings, val_standard_embeddings, metric='sqeuclidean')

    print(f"Distance shapes: train={train_distances.shape}, val={val_distances.shape}")

    print("Finding most similar images for each synthetic image")
    min_train_distances = np.min(train_distances, axis=1)
    min_val_distances = np.min(val_distances, axis=1)
    
    is_train_more_similar = min_train_distances <= min_val_distances
    most_similar_train_indices = np.argmin(train_distances, axis=1)
    most_similar_val_indices = np.argmin(val_distances, axis=1)
    min_distances = np.minimum(min_train_distances, min_val_distances)

    # Sort synthetic images by their similarity to the closest image
    sorted_indices = np.argsort(min_distances)
    num_images_to_plot = max(int(len(sorted_indices) * plot_percentage), 1)
    indices_to_plot = sorted_indices[:num_images_to_plot]

    print(f"Number of images to plot: {num_images_to_plot}")

    # Create output directory
    output_dir = os.path.join(output_dir, f'synthetic_similar_pairs_{plot_percentage:.2f}_validation_{validation_percentage:.2f}_{extract_model_name(embeddings_file)}')
    os.makedirs(output_dir, exist_ok=True)

    # Create sets of image IDs we need to load
    train_ids_to_load = set(train_image_ids[most_similar_train_indices[indices_to_plot[is_train_more_similar[indices_to_plot]]]])
    val_ids_to_load = set(val_image_ids[most_similar_val_indices[indices_to_plot[~is_train_more_similar[indices_to_plot]]]])
    synth_ids_to_load = set(synth_image_ids[indices_to_plot])

    print(f"IDs to load: train={len(train_ids_to_load)}, val={len(val_ids_to_load)}, synth={len(synth_ids_to_load)}")

    # Create dictionaries to store only the needed images
    train_images = {}
    val_images = {}
    synth_images = {}

    def load_images(dataloader, ids_to_load, images_dict, set_name):
        print(f"Loading necessary {set_name} images")
        for batch in tqdm(dataloader, total=int(len(dataloader)*validation_percentage)):
            for img, img_id in zip(batch['data'], batch['img_id']):
                if img_id in ids_to_load:
                    images_dict[img_id] = img.cpu().numpy()
                    ids_to_load.remove(img_id)
            if not ids_to_load or random.random() > validation_percentage:
                break
        if ids_to_load:
            print(f"Warning: Could not find the following {set_name} images: {ids_to_load}")
        print(f"Loaded {len(images_dict)} {set_name} images")

    load_images(train_dataloader, train_ids_to_load, train_images, "training")
    load_images(val_dataloader, val_ids_to_load, val_images, "validation")
    load_images(synth_dataloader, synth_ids_to_load, synth_images, "synthetic")

    print(f"Plotting and saving the top {num_images_to_plot} most similar image pairs")
    plots_created = 0
    for i, synth_idx in enumerate(tqdm(indices_to_plot)):
        synth_id = synth_image_ids[synth_idx]
        
        if is_train_more_similar[synth_idx]:
            real_idx = most_similar_train_indices[synth_idx]
            real_id = train_image_ids[real_idx]
            real_img = train_images.get(real_id)
            distance = min_train_distances[synth_idx]
            set_name = "Training"
        else:
            real_idx = most_similar_val_indices[synth_idx]
            real_id = val_image_ids[real_idx]
            real_img = val_images.get(real_id)
            distance = min_val_distances[synth_idx]
            set_name = "Validation"

        synth_img = synth_images.get(synth_id)

        if synth_img is None or real_img is None:
            print(f"Warning: Missing image for synthetic ID {synth_id} or {set_name} ID {real_id}")
            continue
        
        # Prepare images for display
        synth_img = prepare_image_for_display(synth_img)
        real_img = prepare_image_for_display(real_img)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(synth_img)
        ax1.set_title(f"Synthetic Image\nID: {synth_id}")
        ax1.axis('off')

        ax2.imshow(real_img)
        ax2.set_title(f"Most Similar {set_name} Image\nID: {real_id}")
        ax2.axis('off')

        plt.suptitle(f"MSE Distance: {distance:.4f}\nRank: {i+1}/{num_images_to_plot}")
        plt.tight_layout()

        output_file = os.path.join(output_dir, f"pair_{i:04d}_synth_{synth_id}_{set_name.lower()}_{real_id}.png")
        plt.savefig(output_file)
        plt.close()
        plots_created += 1

    print(f"Total plots created: {plots_created}")
    print(f"All image pairs saved in {output_dir}")



def visualize_augmentations(dataloader, num_images=6, num_augmentations=5):
    # Get a batch of images
    images, paths = next(iter(dataloader))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_images, num_augmentations + 1, figsize=(20, 4 * num_images))
    fig.suptitle("Original vs Augmented Images", fontsize=16)

    # Inverse normalization function
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for i in range(num_images):
        # Display original image
        original_img = Image.open(paths[i]).convert('RGB')
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # Display augmented images
        for j in range(1, num_augmentations + 1):
            aug_img = dataloader.dataset.transform(original_img)
            aug_img = inv_normalize(aug_img)
            aug_img = torch.clamp(aug_img, 0, 1)
            axes[i, j].imshow(aug_img.permute(1, 2, 0))
            axes[i, j].set_title(f"Augmented {j}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('data/image_aug.png')

def visualize_augmentations_contrastive(dataloader, num_images=20, num_augmentations=4):
    batch = next(iter(dataloader))
    
    fig, axes = plt.subplots(num_images, num_augmentations + 1, figsize=(20, 5 * num_images))
    fig.suptitle("Anchor vs Augmented Images", fontsize=16)

    refined_transforms = tio.Compose([
        Custom2DRotation(degrees=20, p=0.7),
        tio.RandomAffine(
            degrees=(-0, 0),  # Reduced rotation range
            scales=(0.8, 1.2),  # Tighter scaling range
            image_interpolation='bspline',
            p=0.7,
        ),
        tio.RandomFlip(axes=(2,)),
        tio.RandomNoise(std=(0, 0.02)),
        tio.RandomBlur(std=(0, 0.5)),
        tio.RandomMotion(degrees=5, translation=5),
       
    ])
    
    
    refined_transforms = tio.Compose([
        Custom2DRotation(degrees=20, p=0.7),
        tio.RandomAffine(
            degrees=(-0, 0),  # Reduced rotation range
            scales=(0.8, 1.2),  # Reduced scaling range
            default_pad_value='minimum',
            p=0.5
        ),
        tio.RandomFlip(axes=(2), flip_probability=0.5),
        tio.RandomFlip(axes=(1), flip_probability=0.5),
        tio.RandomBiasField(
            coefficients=0.3,  # Reduced coefficient for less intense bias field
            order=3,
            p=0.4
        ),
        tio.RandomGamma(
            log_gamma=(-0.1, 0.1),  # Reduced range for less intense gamma correction
            p=0.4
        ),
        tio.RandomNoise(std=(0, 0.05), p=0.3),  # Added some noise for texture
        tio.RandomBlur(std=(0, 1), p=0.2),  # Added slight blur for realism
        tio.RandomMotion(degrees=5, translation=5, p=0.2),
    ])

    for i in range(num_images):

        # Convert RGB to grayscale and display anchor image
        anchor_img = batch['data'][i].cpu()
        anchor_img_gray = anchor_img.mean(dim=0)  # Average across color channels
        anchor_img_gray = torch.clamp(anchor_img_gray, 0, 1)
        axes[i, 0].imshow(anchor_img_gray, cmap='gray')
        axes[i, 0].set_title("Anchor")
        axes[i, 0].axis('off')

        # Create a TorchIO subject from the grayscale anchor image
        subject = tio.Subject(image=tio.ScalarImage(tensor=anchor_img_gray.unsqueeze(0).unsqueeze(0)))

        # Display different augmentations
        for j in range(1, num_augmentations + 1):
            augmented_subject = refined_transforms(subject)
            aug_img = augmented_subject.image.data.squeeze()
            aug_img = torch.clamp(aug_img, 0, 1)
            axes[i, j].imshow(aug_img, cmap='gray')
            axes[i, j].set_title(f"Aug {j}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('data/image_aug.png')
    plt.close()


def rotate_2d(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Apply 2D rotation to a 3D tensor (C, H, W)."""
    rad = torch.deg2rad(torch.tensor(angle))
    cos, sin = torch.cos(rad), torch.sin(rad)
    rotation_matrix = torch.tensor([[cos, -sin, 0],
                                    [sin, cos, 0]])
    grid = torch.nn.functional.affine_grid(rotation_matrix.unsqueeze(0), x.unsqueeze(0).size(), align_corners=False)
    return torch.nn.functional.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)

class Custom2DRotation(tio.transforms.Transform):
    def __init__(self, degrees=10, p=0.5):
        super().__init__(p=p)
        self.degrees = degrees

    def apply_transform(self, subject):
        for image in subject.get_images(intensity_only=True):
            angle = torch.randint(-self.degrees, self.degrees + 1, (1,)).item()
            image_data = image.data[0]  # (C, H, W) format
            rotated_data = rotate_2d(image_data, angle)
            image.set_data(rotated_data.unsqueeze(0))  # Back to (1, C, H, W)
        return subject
