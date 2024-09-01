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
from torch.utils.data import Dataset
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

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_model, backbone_type='torch', processor=None, n_features=128):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone_model
        self.backbone_type = backbone_type
        self.n_features = n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

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

            self.fc = nn.Linear(in_features, self.n_features)

        elif self.backbone_type == 'huggingface':
            if isinstance(self.backbone.config, CLIPConfig):
                in_features = self.backbone.config.vision_config.hidden_size
            elif hasattr(self.backbone.config, 'hidden_size'):
                in_features = self.backbone.config.hidden_size
            else:
                raise ValueError(f"Unsupported model configuration: {type(self.backbone.config)}")

            self.fc = nn.Linear(in_features, self.n_features)
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif self.backbone_type == 'keras':
            # Keras model handling remains the same
            base_model = Model(inputs=self.backbone.input, outputs=self.backbone.layers[-2].output)
            in_features = base_model.output.shape[-1]
            new_output = Dense(n_features, name='new_fc')(base_model.output)
            self.backbone = Model(inputs=base_model.input, outputs=new_output)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

        self.fc_end = nn.Linear(self.n_features, 1)
        self.to(self.device)
    
    def process_input(self, x):
        if self.backbone_type == 'huggingface' and self.processor is not None:
            # Process the input using the CLIP processor
            processed = self.processor(images=x, return_tensors="pt", do_rescale=False)
            return processed['pixel_values'].to(self.device)
        return x

    def forward_once(self, x):
        if self.backbone_type == 'torch':
            features = self.backbone(x)
            output = self.fc(features)
        elif self.backbone_type == 'huggingface':
            x = x.permute(0, 3, 2, 1)
            if isinstance(self.backbone.config, CLIPConfig):
                features = self.backbone.vision_model(x).pooler_output
            else:
                features = self.backbone(x).last_hidden_state[:, 0, :]
            output = self.fc(features)
        elif self.backbone_type == 'keras':
            output = self.backbone.predict(x.cpu().numpy())
            output = torch.tensor(output).to(x.device)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

        return torch.sigmoid(output)

    def forward(self, input1=None, input2=None, resnet_only=False):
        if resnet_only:
            if self.backbone_type == 'torch':
                a = self.backbone(input1)
                return a
            elif self.backbone_type == 'huggingface':
                #input1 = self.process_input(input1)
                if isinstance(self.backbone.config, CLIPConfig):
                    return self.backbone.get_image_features(input1)
                else:
                    return self.backbone(input1).pooler_output
            else:
                raise ValueError(f"resnet_only not supported for backbone type: {self.backbone_type}")

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        difference = torch.abs(output1 - output2)
        output = self.fc_end(difference)

        return output, output1, output2

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
        return model  # Return the original model if loading fails

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
        backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", output_hidden_states=True)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
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

# Define preprocessing and training transforms
PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomAffine(degrees=(-5, 5), scales=(0.9, 1.1), default_pad_value='minimum', p=0.5),
    tio.RandomFlip(axes=(2), flip_probability=0.5)
])

VAL_TRANSFORMS = None

class ContrastiveDataset(Dataset):
    def __init__(self, root_dir, split='train', training_samples=1000, validation_samples=200, augmentation=False, target_resolution=(1024, 1024), single_labeled=False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.preprocessing = PREPROCESSING_TRANSFORMS
        self.transforms = TRAIN_TRANSFORMS if augmentation else VAL_TRANSFORMS
        self.target_resolution = target_resolution
        self.single_labeled = single_labeled

        # Get all image paths and labels
        self.paths = self._get_file_paths()
        self.labels = self._generate_dummy_labels(len(self.paths))  # Replace with actual labels if available

        self.resolution_transform = tio.Resize((self.target_resolution[0], self.target_resolution[1], 1))

    def _get_file_paths(self):
        file_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith((".png", ".jpeg", ".jpg")):
                    file_paths.append(os.path.join(root, file))
        file_paths.sort()

        if self.single_labeled:
            file_paths = [file_paths[ii] for ii in self._filter_single_labeled(file_paths)]

        if self.split == 'train':
            return file_paths[:self.training_samples]
        else:
            return file_paths[-self.validation_samples:]

    def _filter_single_labeled(self, file_paths):
        # Implement logic to filter out images with only one label if labels are available
        return range(len(file_paths))  # Dummy implementation, replace with actual logic

    def _generate_dummy_labels(self, num_samples):
        # Generate dummy labels (replace with actual label loading logic)
        return torch.randint(0, 2, (num_samples, 5))  # Assuming 5 possible labels for example purposes

    def _get_image_resolution(self, img_path):
        with Image.open(img_path) as img:
            return img.size  # Returns (width, height)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = read_image(self.paths[index])  # Image shape: (C, H, W)
        
        # Add a singleton dimension for the z-axis (depth) to make it 4D: (C, H, W) -> (C, H, W, 1)
        img = img.unsqueeze(dim=-1)  # Now img shape is (C, H, W, 1)
        
        # Apply preprocessing
        img = self.preprocessing(img)
        
        # Apply resolution transform (resize)
        img = self.resolution_transform(img)

        # Continue with the rest of your processing...
        img_pos = self.transforms(img) if self.transforms else img
        
        # Get a negative sample
        index_neg_all = np.setdiff1d(range(self.__len__()), index)
        index_neg = np.random.choice(index_neg_all)
        img_neg = read_image(self.paths[index_neg])
        
        # Ensure the negative image has the correct dimensions
        img_neg = img_neg.unsqueeze(dim=-1)  # Add the z-axis dimension
        img_neg = self.preprocessing(img_neg)
        img_neg = self.resolution_transform(img_neg)

        # Get label
        #label = torch.nonzero(self.labels[index], as_tuple=False)
        #if label.shape[0] > 1:
        #    label = label[np.random.choice(label.shape[0])]
        #else:
        #    label = label[0]
        label = np.nan
        img_id = os.path.basename(self.paths[index])  # Image ID as the file name

        # Remove the singleton dimension for returning to the original shape
        img = img.squeeze(-1)
        img_pos = img_pos.squeeze(-1)
        img_neg = img_neg.squeeze(-1)

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
    def decorator(*args, **kwargs):
        nonlocal batch_size
        clear_device_cache(garbage_collection=True)
        params = list(inspect.signature(function).parameters.keys())
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    clear_device_cache(garbage_collection=True)
                    batch_size //= 2
                else:
                    raise
    return decorator


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
    
    # Default values
    defaults = {
        'n_features': 128,
        'batch_size': 32,
        'target_resolution': (512, 512),
        'split_ratio': 0.8,
        'num_workers': 4,
        'pin_memory': True,
        'base_lr': 1e-3,
        'n_epochs': 10,
        'temperature': 0.5,
        'save_model_interval': 5,
        'multi_gpu': False
    }
    
    # Update defaults with provided kwargs
    config = {**defaults, **kwargs}
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model
    try:
        backbone_model, backbone_type, processor = initialize_model(network_name)
        
        model = SiameseNetwork(backbone_model, backbone_type, processor, config['n_features'])
    except ValueError as e:
        print(network_name, e)
        raise ValueError(f'Error {e} with {network_name}')

    if config['multi_gpu']:
        model = torch.nn.DataParallel(model)
   
    model.to(device)
    
    # Select target reolution for eahc specific model 
    if backbone_type == 'torch':
        target_res = (224, 224)
        if isinstance(backbone_model, model.Inception3):
            target_res = (299, 299)
    elif backbone_type == 'huggingface':
        target_res = (224, 224)
    # Define preprocessing and transforms
    preprocessing_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize(target_res)
    ])
    
    train_transforms = tio.Compose([
        tio.RandomAffine(degrees=(-5, 5, 0, 0, 0, 0), scales=0, default_pad_value='minimum', p=0.5),
        tio.RandomFlip(axes=(2), flip_probability=0.5)
    ])
    
    val_transforms = None  # No augmentation for validationgg


    # Create the dataset
    dataset = ContrastiveDataset(
        root_dir=root_dir,
        split='train',
        target_resolution=target_res,
        augmentation=True
    )
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(config['split_ratio'] * dataset_size)
    val_size = dataset_size - train_size
    
    # Split dataset into training and validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    # Create DataLoaders with a BatchSampler
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size=config['batch_size'], drop_last=False)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True
    )

    val_sampler = BatchSampler(SequentialSampler(val_dataset), batch_size=config['batch_size'], drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=True
    )

    
    # Define optimizer, scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['n_epochs'], eta_min=config['base_lr'] / 2.0)
    LossTr = NTXentLoss(temperature=config['temperature'])
    cosine_similarity = CosineSimilarity()
    
    # Set up TensorBoard writer
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a unique directory name combining timestamp and network name
    run_name = f"{timestamp}_{network_name}"
    wandb.init(project="privacy_benchmark", name=run_name, config=config)

    # Set up checkpoint directory
    ckpt_dir = os.path.join('./checkpoints', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
 
    
    @find_executable_batch_size(starting_batch_size=config['batch_size'])
    def train_epoch(batch_size, model, train_loader, optimizer, scheduler, device, LossTr):
        model.train()
        epoch_loss = 0
        train_loader.batch_sampler.batch_size = batch_size  # Update batch size
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")

        for train_step, batch in progress_bar:
            optimizer.zero_grad()
            Pos11 = batch['data'].to(device)
            Pos12 = batch['data_pos'].to(device)

            # Forward pass
            PosEmb11 = model(Pos11, resnet_only=True)
            PosEmb12 = model(Pos12, resnet_only=True)

            try:
                # First, try to handle the output as a tensor
                PosEmb11 = PosEmb11.requires_grad_()
                PosEmb12 = PosEmb12.requires_grad_()
            except AttributeError:
                # If it's not a tensor, it might be InceptionOutputs
                try:
                    PosEmb11 = PosEmb11.logits
                    PosEmb12 = PosEmb12.logits
                except AttributeError:
                    # If it's neither a tensor nor has logits, print the type for debugging
                    print(f"Unexpected output type: {type(PosEmb11)}")
                    raise 
           
            Labels  = torch.arange(PosEmb11.shape[0], device=device)
            LossPos1 = LossTr(torch.cat((PosEmb11, PosEmb12), dim=0), torch.cat((Labels, Labels), dim=0))

            LossPos1.backward()
            optimizer.step()

            epoch_loss += LossPos1.item()
            progress_bar.set_postfix({'loss': f'{LossPos1.item():.4f}'})

        return epoch_loss / len(train_loader)

    def validate(model, val_loader, device, LossTr):
        model.eval()
        val_loss = 0
        pos_sim, neg_sim, neg_sim_aug = [], [], []
        
        for batch in val_loader:
            Pos11 = batch['data'].to(device)
            Pos12 = batch['data_pos'].to(device)
            with torch.no_grad():
                PosEmb11 = model(Pos11, resnet_only=True)
                PosEmb12 = model(Pos12, resnet_only=True)

            Labels = torch.arange(PosEmb11.shape[0], device=device)
            val_loss += LossTr(torch.cat((PosEmb11, PosEmb12), dim=0), torch.cat((Labels, Labels), dim=0)).item()

        similarity_pos = cosine_similarity(PosEmb11, PosEmb12).cpu().numpy()
        similarity_neg = cosine_similarity(PosEmb11, PosEmb11).cpu().numpy()

        pos_sim.append( np.diag(similarity_pos))
        neg_sim.append (similarity_neg[np.triu_indices_from(similarity_neg, k=1)])
        neg_sim_aug.append (similarity_pos[np.triu_indices_from(similarity_pos, k=1)])
        torch.cuda.empty_cache()

        val_loss /= len(val_loader)

        return val_loss, pos_sim, neg_sim, neg_sim_aug

    # Training loop
    val_losses = []
    
    for epoch in range(config['n_epochs']):
        epoch_loss = train_epoch(model, train_loader, optimizer, scheduler, device, LossTr)
        scheduler.step()

        val_loss, pos_sim, neg_sim, neg_sim_aug = validate(model, val_loader, device, LossTr)
        val_losses.append(val_loss)
        # Logging with wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "batch_size": train_loader.batch_sampler.batch_size,
            "positive_samples": wandb.Histogram(pos_sim),
            "negative_samples": wandb.Histogram(neg_sim),
            "negative_samples_augmented": wandb.Histogram(neg_sim_aug)
        })

        # Save model checkpoints
        if (epoch + 1) % config['save_model_interval'] == 0 or epoch == 0:
            save_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.module.state_dict() if config['multi_gpu'] else model.state_dict(), save_path)
            wandb.save(save_path)

        # Save best model
        if (epoch > 1) and (val_loss < min(val_losses[:-1])):
            best_path = os.path.join(ckpt_dir, "model_best.pth")
            torch.save(model.module.state_dict() if config['multi_gpu'] else model.state_dict(), best_path)
            wandb.save(best_path)

    wandb.finish()
    return train_loader, val_loader, device

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
        train_loader, val_loader, device = setup_training(
            root_dir=real_data_dir,
            network_name=network_name,
        )
        























































































