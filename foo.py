import json
import numpy as np
import h5py
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance_nd, wasserstein_distance

file_path = '/home/ksamamov/GitLab/Notebooks/feat_ext_bench/checkpoints/20240919_072022_rad_inception/20240919_072022_rad_inception_image_paths.json'


with open(file_path, 'r') as fh:
    trval_js = json.load(fh)


train_paths = trval_js['train_paths']
val_paths = trval_js['val_paths']



from privacy_benchmark import compute_distances_and_plot

def decode_byte_strings(data):
    if isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, np.ndarray):
        return np.array([item.decode('utf-8') if isinstance(item, bytes) else item for item in data])
    else:
        return data


embeddings_file = '/home/ksamamov/GitLab/Notebooks/feat_ext_bench/embeddings/rad_dino_embeddings.h5'
method = 'sqeuclidean'

print(f"Loading embeddings from {embeddings_file}")

with h5py.File(embeddings_file, 'r') as f:

    train_standard_embeddings = f['train_standard/embeddings'][:]
    train_adversarial_embeddings = f['train_adversarial/embeddings'][:]
    val_standard_embeddings = f['val_standard/embeddings'][:]
    synth_standard_embeddings = f['synth_standard/embeddings'][:]
    train_image_ids = decode_byte_strings(f['train_standard/image_ids'][:])
    val_image_ids = decode_byte_strings(f['val_standard/image_ids'][:])
    synth_image_ids = decode_byte_strings(f['synth_standard/image_ids'][:])

train_val_distances = cdist(train_standard_embeddings, val_standard_embeddings, metric=method)

synth_val_distances = cdist(synth_standard_embeddings, train_standard_embeddings, metric=method)

#print('Wasserstein ND:', wasserstein_distance_nd(train_val_distances, synth_val_distances))
print(train_val_distances.shape)
print()
min_val_distances = np.min(train_val_distances, axis=1)
min_synth_distances = np.min(synth_val_distances, axis=1)
print('Wasserstein 1D:', wasserstein_distance(min_val_distances, min_synth_distances))

ndpctl = np.percentile(sorted(min_val_distances), 2)
print('2ndpercentile', ndpctl)
print(min_val_distances)
print(min_val_distances.shape)
print()

most_similar_val_indices = np.argmin(train_val_distances, axis=1)
print(most_similar_val_indices)
print(most_similar_val_indices.shape)
print()

sorted_indices = np.argsort(min_val_distances)
print(sorted_indices)
val_ids_to_load = val_image_ids[most_similar_val_indices]

import matplotlib.pyplot as plt
from PIL import Image
import os

def compare_images(path1, path2, distance):
    # Open the images
    img1 = Image.open(path1)
    img2 = Image.open(path2)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image
    ax1.imshow(img1)
    ax1.set_title('Image 1')
    ax1.axis('off')  # Hide the axis

    # Display the second image
    ax2.imshow(img2)
    ax2.set_title('Image 2')
    ax2.axis('off')  # Hide the axis

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig(os.path.join('/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/images_possibly_same_patients', os.path.basename(path1) + '_' + os.path.basename(path2) + f'{distance}.png'))

case_dir = '/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/512/output_images_512_all'

for i, err_idx in enumerate(sorted_indices):
    train_id = train_image_ids[err_idx]    
    val_idx = most_similar_val_indices[err_idx]
    val_img = val_image_ids[val_idx]
    distance = min_val_distances[err_idx]

    if distance <10:
        compare_images(os.path.join(case_dir, train_id), os.path.join(case_dir, val_img), distance)
    
