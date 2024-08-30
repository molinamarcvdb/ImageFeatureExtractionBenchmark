import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPModel, AutoModel, AutoImageProcessor
from tensorflow.keras.applications import InceptionV3, ResNet50, InceptionResNetV2, DenseNet121
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

class SiameseNetwork(nn.Module):
    def __init__(self, backbone_model, backbone_type='torch', n_features=128):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone_model
        self.backbone_type = backbone_type
        self.n_features = n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.backbone_type == 'torch':
            if isinstance(self.backbone, models.DenseNet):
                in_features = self.backbone.features[-1].num_features
            elif isinstance(self.backbone, models.ResNet):
                in_features = self.backbone.fc.in_features
            else:
                raise ValueError("Unsupported Torch model type")
            
            self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
            self.fc = nn.Linear(in_features, self.n_features)

        elif self.backbone_type == 'huggingface':
            # Huggingface model handling remains the same
            in_features = self.backbone.config.hidden_size
            self.new_head = nn.Linear(in_features, self.n_features)
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.new_head.parameters():
                param.requires_grad = True

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

    def forward_once(self, x):
        if self.backbone_type == 'torch':
            features = self.feature_extractor(x)
            features = features.view(features.size(0), -1)
            output = self.fc(features)
        elif self.backbone_type == 'huggingface':
            x = x.permute(0,3,2,1)
            outputs = self.backbone(x)
            last_hidden_state = outputs.last_hidden_state
            output = self.new_head(last_hidden_state)
        elif self.backbone_type == 'keras':
            output = self.backbone.predict(x.cpu().numpy())
            output = torch.tensor(output).to(x.device)
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")

        output = torch.sigmoid(output)
        return output

    def forward(self, input1=None, input2=None, resnet_only=False):
        if resnet_only:
            features = self.feature_extractor(input1)
            return features.view(features.size(0), -1)  # Return flattened features

        # Forward pass for both inputs through the network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Compute the absolute difference between the feature vectors
        difference = torch.abs(output1 - output2)

        # Pass the difference through the final fully connected layer
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
    if network_name.lower() == "resnet50":
        backbone = models.resnet50(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'resnet50-19c8e357.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "resnet18":
        backbone = models.resnet18(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'resnet18-f37072fd.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "inception":
        backbone = models.inception_v3(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'inception_v3.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "densenet121":
        backbone = models.densenet121(pretrained=False)
        backbone = load_and_remap_state_dict(backbone, 'densenet121-a639ec97.pth')
        backbone_type = 'torch'
    elif network_name.lower() == "clip":
        backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", output_hidden_states=True)
        backbone_type = 'huggingface'
    elif network_name.lower() == "rad_clip":
        backbone = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32", output_hidden_states=True)
        backbone_type = 'huggingface'
    elif network_name.lower() == "rad_dino":
        backbone = AutoModel.from_pretrained("microsoft/rad-dino", output_hidden_states=True)
        backbone_type = 'huggingface'
    elif network_name.lower() == "dino":
        backbone = AutoModel.from_pretrained("facebook/dinov2-base", output_hidden_states=True)
        backbone_type = 'huggingface'
    elif network_name.lower() == "rad_inception":
        backbone = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        backbone_type = 'keras'
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

    return backbone, backbone_type

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
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torchio as tio

def setup_training(root_dir, network_name, n_features=128, batch_size=32, target_resolution=(512, 512), split_ratio=0.8, num_workers=4, pin_memory=True, base_lr=1e-3, n_epochs=10, temperature=0.5, save_model_interval=5, multi_gpu=False):
    """

    Sets up the dataset, dataloaders, and device for training, and initializes the training process.
    
    Args:
        root_dir (str): Path to the root directory containing the image data.
        batch_size (int): Batch size for training and validation dataloaders.
        target_resolution (tuple): Target resolution for resizing images.
        split_ratio (float): Ratio to split dataset into training and validation sets.
        num_workers (int): Number of workers for the DataLoader.
        pin_memory (bool): Whether to pin memory for DataLoader.
        base_lr (float): Base learning rate for the optimizer.
        n_epochs (int): Number of epochs for training.
        temperature (float): Temperature parameter for NT-Xent loss.
        save_model_interval (int): Interval for saving the model.
        multi_gpu (bool): Whether to use multiple GPUs.

    Returns:
        DataLoader, DataLoader, torch.device: Training DataLoader, Validation DataLoader, and device.
    """
    
    # Define preprocessing and transforms
    preprocessing_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize(target_resolution)
    ])
    
    train_transforms = tio.Compose([
        tio.RandomAffine(degrees=(-5, 5, 0, 0, 0, 0), scales=0, default_pad_value='minimum', p=0.5),
        tio.RandomFlip(axes=(2), flip_probability=0.5)
    ])
    
    val_transforms = None  # No augmentation for validation

    # Create the dataset
    dataset = ContrastiveDataset(
        root_dir=root_dir,
        split='train',
        target_resolution=target_resolution,
        augmentation=True
    )
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    # Split dataset into training and validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True  # Set to True for persistent worker processes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True  # Set to True for persistent worker processes
    )
    
    # Set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Define model
    backbone_model, backbone_type = initialize_model(network_name)
    
    model = SiameseNetwork(backbone_model, backbone_type, n_features)
    
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Define optimizer, scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=base_lr / 2.0)
    LossTr = NTXentLoss(temperature=temperature)
    cosine_similarity = CosineSimilarity()
    
    # Set up TensorBoard writer
    ckpt_dir = './checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=ckpt_dir)
    
    # Training loop
    epoch_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")    

        epoch_loss = 0
        model.train()
        for train_step, batch in progress_bar:
            optimizer.zero_grad()
            Pos11 = batch['data'].to(device)
            Pos12 = batch['data_pos'].to(device)

            # Obtain positive embeddings
            PosEmb11 = model(Pos11, resnet_only=True)
            PosEmb12 = model(Pos12, resnet_only=True)

            Labels = torch.arange(PosEmb11.shape[0], device=device)
            LossPos1 = LossTr(torch.cat((PosEmb11, PosEmb12), dim=0), torch.cat((Labels, Labels), dim=0))
   
            LTotal = LossPos1  
            
            LTotal.backward()
            epoch_loss += LTotal.item()

            optimizer.step()
        scheduler.step()
        epoch_losses.append(epoch_loss / (train_step + 1))

        # Validation
        val_loss = 0
        pos_sim = []
        neg_sim = []
        neg_sim_aug = []
        model.eval()
        for val_step, batch in enumerate(val_loader):
            Pos11 = batch['data'].to(device)
            Pos12 = batch['data_pos'].to(device)
            with torch.no_grad():
                PosEmb11 = model(Pos11, resnet_only=True)
                PosEmb12 = model(Pos12, resnet_only=True)

            Labels = torch.arange(PosEmb11.shape[0], device=device)
            val_loss += LossTr(torch.cat((PosEmb11, PosEmb12), dim=0), torch.cat((Labels, Labels), dim=0)).item()       
            similarity_pos = cosine_similarity(PosEmb11, PosEmb12).cpu().numpy()
            similarity_neg = cosine_similarity(PosEmb11, PosEmb11).cpu().numpy()

            pos_sim.append(np.diag(similarity_pos))
            neg_sim.append(similarity_neg[np.triu_indices_from(similarity_neg, k=1)])
            neg_sim_aug.append(similarity_pos[np.triu_indices_from(similarity_pos, k=1)])
            torch.cuda.empty_cache()
        val_loss /= (val_step + 1)
        val_losses.append(val_loss)

        # Logging
        writer.add_histogram('Positive samples', np.hstack(pos_sim), epoch)
        writer.add_histogram('Negative samples', np.hstack(neg_sim), epoch)
        writer.add_histogram('Negative samples Augmented', np.hstack(neg_sim_aug), epoch)
        writer.add_scalar('Train/Loss', epoch_losses[-1], epoch)
        writer.add_scalar('Val/Loss', val_losses[-1], epoch)

        # Save model checkpoints
        if (epoch + 1) % save_model_interval == 0 or epoch == 0:  
            if multi_gpu:
                torch.save(model.module.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))
        
        # Save best model
        if (epoch > 1) and (val_loss < min(val_losses[:-1])):
            if multi_gpu:
                torch.save(model.module.state_dict(), os.path.join(ckpt_dir, "model_best.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_best.pth"))

    # Close TensorBoard writer
    writer.close()

    return train_loader, val_loader, device

train_loader, val_loader, device = setup_training(
    root_dir='./data/real/',
    network_name='densenet121',
    n_features=128,
    batch_size=32, 
    target_resolution=(512, 512), 
    split_ratio=0.8, 
    num_workers=4, 
    pin_memory=True, 
    base_lr=1e-3, 
    n_epochs=10, 
    temperature=0.5, 
    save_model_interval=5, 
    multi_gpu=True
)



#backbones = ["dino", "resnet50_keras", "rad_inception", "inception", "rad_densenet"]  # Add more backbones as needed
#for backbone_name in backbones:
#
#    n_features = 128
#    backbone_model, backbone_type = initialize_model(backbone_name)
#    siamese_network = SiameseNetwork(backbone_model=backbone_model, backbone_type=backbone_type, n_features=n_features)
#    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    # Example usage with some input tensors
#    input1 = torch.randn(1, 224, 224, 3).to(device)  # Example input, adjust according to your needs
#    input2 = torch.randn(1, 224, 224, 3).to(device)
#    
#    # Forward pass
#    output, output1, output2 = siamese_network(input1=input1, input2=input2)
#    print(f"Output using {backbone_name}: {output.shape, output1.shape, output2.shape }")
#
