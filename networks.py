import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoImageProcessor
from tensorflow.keras.applications import InceptionV3, ResNet50, InceptionResNetV2, DenseNet121
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.preprocessing import image as keras_image

from ijepa.src.helper import init_model
from ijepa.src.models.vision_transformer import vit_huge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rad_imagenet_dict = {
    'rad_densenet': '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/pretrained/RadImageNet_models/RadImageNet-DenseNet121_notop.h5',
    'rad_inception': '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/pretrained/RadImageNet_models/RadImageNet-InceptionV3_notop.h5',
    'rad_inceptionresnet': '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/pretrained/RadImageNet_models/RadImageNet-IRV2_notop.h5',
    'rad_resnet50': '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/syntheva/pretrained/RadImageNet_models/RadImageNet-ResNet50_notop.h5'
}

def load_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  # Ensure image is in RGB mode
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_images(images, processor, target_size=None, preprocess_fn=None):
    if processor:
        inputs = processor(images=images, return_tensors="pt")
        return inputs['pixel_values'].to(device)
    else:
        processed_images = []
        for img in images:
            if target_size:
                img = img.resize(target_size)
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_fn(img_array)
            processed_images.append(img_array)
        return np.vstack(processed_images)

def extract_features(network_name, model, images, processor=None, target_size=None, preprocess_fn=None):
    inputs = preprocess_images(images, processor, target_size, preprocess_fn)
    with torch.no_grad():
        if network_name.lower() in ["clip", "rad_clip"]:
            features = model.get_image_features(inputs).cpu().numpy()
        elif network_name.lower() in ["rad_dino", 'dino']:
            outputs = model(inputs)
            features = outputs.pooler_output.cpu().numpy()
        else:
            features = model.predict(inputs)
    return features

def initialize_model(network_name):
    target_size = None
    preprocess_fn = None
    processor = None
    if network_name.lower() == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif network_name.lower() == "rad_clip":
        model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
    elif network_name.lower() == "rad_dino":
        model = AutoModel.from_pretrained("microsoft/rad-dino").to(device)
        processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    elif network_name.lower() == "dino":
        model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    elif network_name.lower() == "inception":
        model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        target_size = (299, 299)
        preprocess_fn = inception_preprocess
    elif network_name.lower() == "rad_inception":
        model = InceptionV3(weights=rad_imagenet_dict[network_name], include_top=False, pooling='avg')
        target_size = (299, 299)
        preprocess_fn = inception_preprocess
    elif network_name.lower() == "resnet50":
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        target_size = (224, 224)
        preprocess_fn = resnet_preprocess
    elif network_name.lower() == "rad_resnet50":
        model = ResNet50(weights=rad_imagenet_dict[network_name], include_top=False, pooling='avg')
        target_size = (224, 224)
        preprocess_fn = resnet_preprocess
    elif network_name.lower() == "inceptionresnet":
        model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
        target_size = (299, 299)
        preprocess_fn = inception_resnet_preprocess
    elif network_name.lower() == "rad_inceptionresnet":
        model = InceptionResNetV2(weights=rad_imagenet_dict[network_name], include_top=False, pooling='avg')
        target_size = (299, 299)
        preprocess_fn = inception_resnet_preprocess
    elif network_name.lower() == "densenet121":
        model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
        target_size = (224, 224)
        preprocess_fn = densenet_preprocess
    elif network_name.lower() == "rad_densenet":
        model = DenseNet121(weights=rad_imagenet_dict[network_name], include_top=False, pooling='avg')
        target_size = (224, 224)
        preprocess_fn = densenet_preprocess
    else:
        raise ValueError(f"Unsupported network name: {network_name}")
    return model, processor, target_size, preprocess_fn

class IJEPAEncoder(nn.Module):
    def __init__(self, device='cuda', patch_size=14, crop_size=224, pred_depth=12, 
                 pred_emb_dim=384, model_name='vit_huge', output_dim=None, 
                 use_attentive_pooling=False, num_queries=1, num_heads=16, 
                 mlp_ratio=4.0, pooler_depth=1, init_std=0.02, qkv_bias=True, 
                 complete_block=True):
        super().__init__()
        self.device = device

        # Initialize the model
        self.encoder, _ = init_model(
            device=self.device,
            patch_size=patch_size,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_emb_dim=pred_emb_dim,
            model_name=model_name
        )
        

        self.encoder_output_dim = self.encoder.norm.normalized_shape[0]

    def forward(self, x):

        return self.encoder(x.to(self.device))




def extract_features_from_directory(list_of_img, network_name, batch_size=16):
    model, processor, target_size, preprocess_fn = initialize_model(network_name)
    features = []
    filenames = []
    image_batch = []
    batch_filenames = []

    # if not os.path.isdir(directory):
    #     raise ValueError(f"Directory not found: {directory}")

    for filename in list_of_img:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image = load_image(filename)
            if image:
                image_batch.append(image)
                batch_filenames.append(filename)

                if len(image_batch) == batch_size:
                    batch_features = extract_features(network_name, model, image_batch, processor, target_size, preprocess_fn)
                    features.append(batch_features)
                    filenames.extend(batch_filenames)
                    image_batch = []
                    batch_filenames = []

    if image_batch:
        batch_features = extract_features(network_name, model, image_batch, processor, target_size, preprocess_fn)
        features.append(batch_features)
        filenames.extend(batch_filenames)

    features = np.vstack(features)
    return filenames, features
