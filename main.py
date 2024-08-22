import os
import yaml
import csv
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from networks import extract_features_from_directory, initialize_model
from metrics import calculate_metrics
from utils import link_azure_local, get_sets_content, get_realism_set_dict, realism_corr_net
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_features(features_dir, network_name, set_name, filenames, features):
    # Ensure directories exist
    os.makedirs(features_dir, exist_ok=True)
    
    # Save features
    np.save(os.path.join(features_dir, f'{set_name}_filenames.npy'), filenames)
    np.save(os.path.join(features_dir, f'{set_name}_features.npy'), features)
    print(f"Features saved for {network_name}, set {set_name}")

def save_metrics(metrics_dir, network_name, all_set_metrics):
    # Ensure directories exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save all metrics to a single YAML file
    yaml_path = os.path.join(metrics_dir, f'{network_name}_metrics.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(all_set_metrics, file, default_flow_style=False)
    
    # Save all metrics to a single CSV file
    csv_path = os.path.join(metrics_dir, f'{network_name}_metrics.csv')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=all_set_metrics[0].keys())
        writer.writeheader()
        for metrics in all_set_metrics:
            writer.writerow(metrics)
    
    print(f"All metrics saved for {network_name} in single files")


def split_dataset(dataset_path, num_sets):
    all_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sets = np.array_split(all_files, num_sets)
    return sets

def process_dataset(dataset_path, network_name, batch_size, num_sets, features_dir):
    print(f"Processing dataset: {dataset_path} with network: {network_name}")
    sets = split_dataset(dataset_path, num_sets)
    for i, file_set in tqdm(enumerate(sets)):
        filenames, features = extract_features_from_directory(file_set, network_name, batch_size)
        set_name = f'set_{i+1}'
        save_features(features_dir, network_name, set_name, filenames, features)

def evaluate_metrics(real_features_dir, synthetic_features_dir, network_name, num_sets, data_dir):
    all_set_metrics = []
    
    for i in range(num_sets):
        set_name = f'set_{i+1}'
        real_features_path = os.path.join(real_features_dir, f'{set_name}_features.npy')
        synthetic_features_path = os.path.join(synthetic_features_dir, f'{set_name}_features.npy')

        # Load features
        real_features = np.load(real_features_path)
        synthetic_features = np.load(synthetic_features_path)
        
        # Calculate metrics
        metrics = calculate_metrics(real_features, synthetic_features)
        
        # Include set name in the metrics for clarity
        metrics['set_name'] = set_name
        
        # Add metrics to the list
        all_set_metrics.append(metrics)
    
    # Save all metrics at once
    save_metrics(data_dir, network_name, all_set_metrics)

def main():
    config = load_config('config.yml')
    
    real_dataset_path = config['real_dataset_path']
    synthetic_dataset_path = config['synthetic_dataset_path']
    networks = config['networks']
    batch_size = config['batch_size']
    num_sets = config['num_sets']
    metrics = config['metrics']
    # Create a timestamp for the run
    global timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if config['feature_extraction']:
        for network_name in networks:
            print(f"Processing network: {network_name}")
            
            # Prepare directories
            features_dir = os.path.join('data', 'features', timestamp, network_name)
            real_features_dir = os.path.join(features_dir, 'real')
            synthetic_features_dir = os.path.join(features_dir, 'synthetic')
            data_dir = os.path.join('data', 'features', timestamp, network_name, 'metrics')  # Updated directory for metrics
            
            # Process real dataset
            process_dataset(real_dataset_path, network_name, batch_size, num_sets, real_features_dir)
            
            # Process synthetic dataset
            process_dataset(synthetic_dataset_path, network_name, batch_size, num_sets, synthetic_features_dir)
            
            # Evaluate metrics
            evaluate_metrics(real_features_dir, synthetic_features_dir, network_name, num_sets, data_dir)

    if config['realism_correlation']:

        if config['timestamp'] is not None: timestamp = config['timestamp']

        # Azure to local image path linking
        grouped_data = link_azure_local(config['jsonl_path'])

        # Record of each image linking it with the local_path
        # All images sets should contain the same images only iterating for first network
        net_sets_dict = get_sets_content(timestamp)

        # Get realism dictioanry as qualitative metric in dict format
        dict_sets_realism = get_realism_set_dict(grouped_data, net_sets_dict)
        print(dict_sets_realism)

        realism_corr_net(dict_sets_realism, metrics, timestamp)
if __name__ == "__main__":
    main()
