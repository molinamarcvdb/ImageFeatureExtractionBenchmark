from main import load_config
from typing import List, Dict, Optional, Tuple
from utils import load_jsonl
import random
from datetime import timedelta, datetime
import os
import json
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

config = load_config('config.yml')

jsonl_dict_ex =list(load_jsonl(config['jsonl_path'][0]))

# Initialize Azure Blob Service Client
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

if not connection_string:
    raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING environment variable not set.")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_name = os.getenv('CONTAINER_NAME')
container_client = blob_service_client.get_container_client(container_name)

# Function to list blobs and generate image URLs from the final_imgs folder
def get_image_blobs() -> Dict[str, List[str]]:
    image_folders = {
        "real_calc": "real_calc",
        "real_normal": "real_normal",
        "synth_gan_calc": "synth_gan_calc",
        "synth_gan_normal": "synth_gan_normal",
        "synth_diff_normal": "synth_diff_normal",
        "synth_diff_calc": "synth_diff_calc",
    }

    all_images = []
    for category, folder in image_folders.items():
        # print(f"Fetching images from folder: {folder}")
        blobs = container_client.list_blobs(name_starts_with=f"final_imgs/{folder}/")
        for blob in blobs:
            if blob.name.endswith(('.png', '.jpg', '.jpeg')):
                # Generate SAS URL for blob access
                sas_token = generate_blob_sas(
                    account_name=blob_service_client.account_name,
                    container_name=container_name,
                    blob_name=blob.name,
                    account_key=blob_service_client.credential.account_key,
                    permission=BlobSasPermissions(read=True),
                    expiry=datetime.utcnow() + timedelta(hours=1)  # 1 hour validity
                )
                image_url = f"{blob_service_client.get_blob_client(container_name, blob.name).url}?{sas_token}"
                all_images.append((image_url, category))
                # print(f"Found image: {image_url} in category: {category}")
    
    return all_images

SHUFFLE_SEED = 42

# Function to shuffle images deterministically
def deterministic_shuffle(images: List[Tuple[str, str]], seed: int) -> List[Tuple[str, str]]:
    # Initialize the random number generator with the seed
    rng = random.Random(seed)
    # Create a copy of the image list to shuffle
    images_copy = images[:]
    rng.shuffle(images_copy)
    return images_copy

# TOTAL IMAGES TO DISPLAY
TOTAL_IMAGES = int(os.getenv('TOTAL_IMAGES'))

# Gather and balance images from the blobs
all_images = get_image_blobs()
     
images_urls = [url[0] for url in all_images]

jsonl_dict_keys = jsonl_dict_ex[0].keys()

def generate_sample_data(list_of_files):
    data = []       
    
    for i, (url, cat) in enumerate(list_of_files):
        sample = {
            "user_id": "4e26daae-530d-430b-81be-107704de6a9e",
            "image_path": url,
            "category": cat,
            "is_real": random.choice([True, False]),
            "realism_score": round(random.uniform(0, 100), 2),
            "calcification_seen": random.choice([True, False]),
            "image_duration": random.randint(3, 60),
            "index": i,
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
        }
        data.append(sample)
    return data

data = generate_sample_data(all_images)

def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


write_jsonl(data, './data/turing_tests/evaluations_dummy.jsonl')











