import os
import yaml
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from networks import extract_features_from_directory, initialize_model
from metrics import calculate_metrics
from utils import link_azure_local, get_sets_content, get_realism_set_dict, realism_corr_net
from single_image_metrics import main_single_metric_eval, compute_ground_truth_correlations
from realism import realism_handling
from privacy_benchmark import setup_training, create_dataloaders, visualize_augmentations, load_best_model_for_inference, inference_and_save_embeddings, compute_distances_and_plot   
from tqdm import tqdm
import traceback

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

def numpy_to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(i) for i in obj]
    else:
        return obj

def save_metrics(metrics_dir, network_name, all_set_metrics, aggregated_metrics):
    # Ensure directories exist
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Convert numpy values to Python types
    all_set_metrics = numpy_to_python(all_set_metrics)
    
    # Save detailed metrics to YAML
    yaml_path = os.path.join(metrics_dir, f'{network_name}_detailed_metrics.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(all_set_metrics, file, default_flow_style=False)
    
    # Save aggregated metrics to CSV (no change needed here)
    csv_path = os.path.join(metrics_dir, f'{network_name}_aggregated_metrics.csv')
    df = pd.DataFrame(aggregated_metrics)
    df.to_csv(csv_path, index=False)
    
    print(f"Detailed metrics saved to YAML and aggregated metrics saved to CSV for {network_name}")

def split_dataset(dataset_path, num_sets):
    all_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Ensure the number of files is divisible by num_sets
    num_files = len(all_files)
    files_per_set = num_files // num_sets
    
    # Trim the list to ensure equal set sizes
    all_files = all_files[:files_per_set * num_sets]
    
    sets = np.array_split(all_files, num_sets)
    return sets

def process_dataset(dataset_path, network_name, batch_size, num_sets, features_dir):
    print(f"Processing dataset: {dataset_path} with network: {network_name}")
    sets = split_dataset(dataset_path, num_sets)
    for i, file_set in tqdm(enumerate(sets)):
        print(f"Processing set {i+1} with {len(file_set)} images")
        filenames, features = extract_features_from_directory(file_set, network_name, batch_size)
        set_name = f'set_{i+1}'
        save_features(features_dir, network_name, set_name, filenames, features)

def evaluate_metrics(real_features_dir, synthetic_features_dir, network_name, num_sets, data_dir, metrics):
    all_set_metrics = []
    aggregated_metrics = []
    
    # Load all real features
    real_features_sets = []
    for i in range(num_sets):
        real_features_path = os.path.join(real_features_dir, f'set_{i+1}_features.npy')
        real_features_sets.append(np.load(real_features_path))
    
    # For each synthetic set
    for i in range(num_sets):
        synthetic_features_path = os.path.join(synthetic_features_dir, f'set_{i+1}_features.npy')
        synthetic_features = np.load(synthetic_features_path)
        
        set_size = len(synthetic_features)
        
        set_comparisons = []
        # Compare with all real sets
        for j, real_features in enumerate(real_features_sets):
            # Calculate metrics
            set_metrics = calculate_metrics(real_features, synthetic_features, set_size)
            
            # Include set names in the metrics for clarity
            set_metrics['synthetic_set'] = f'set_{i+1}'
            set_metrics['real_set'] = f'set_{j+1}'
            
            set_comparisons.append(set_metrics)
        
        # Aggregate metrics for this synthetic set
        agg_metrics = {
            'synthetic_set': f'set_{i+1}',
            'num_comparisons': len(set_comparisons)
        }
        for metric in metrics:
            values = [comp[metric] for comp in set_comparisons]
            agg_metrics[f'{metric}_mean'] = np.mean(values)
            agg_metrics[f'{metric}_std'] = np.std(values)
        
        aggregated_metrics.append(agg_metrics)
        all_set_metrics.append({
            'synthetic_set': f'set_{i+1}',
            'comparisons': set_comparisons
        })
    
    # Save detailed metrics to YAML and aggregated metrics to CSV
    save_metrics(data_dir, network_name, all_set_metrics, aggregated_metrics)

def main():
    config = load_config('config.yml')
    
    real_dataset_path = config['real_dataset_path']
    synthetic_dataset_path = config['synthetic_dataset_path']
    networks = config['networks']
    batch_size = config['batch_size']
    num_sets = config['num_sets']
    metrics = config['metrics']
    do_z_score = config['do_z_score']

    network_list = ['inception', 'resnet50', 'resnet18', 'clip', 'densenet121', 'rad_clip', 'rad_dino', 'dino', 'rad_inception', 'rad_resnet50', 'rad_densenet']
# Create a timestamp for the run
    global timestamp
    
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    elif config['feature_extraction']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        print("No timestamp provided in config. Please specify a timestamp for evaluation or realism correlation.")
        return

    if config['feature_extraction']:
        real_sets = split_dataset(real_dataset_path, num_sets)
        synthetic_sets = split_dataset(synthetic_dataset_path, num_sets)

        if len(real_sets[0]) != len(synthetic_sets[0]):
            print(f"Warning: Real dataset has {len(real_sets[0])} images per set, but synthetic dataset has {len(synthetic_sets[0])} images per set.")
            print("Adjusting number of images to ensure equal set sizes.")
            min_set_size = min(len(real_sets[0]), len(synthetic_sets[0]))
            num_sets = min(len(real_sets), len(synthetic_sets))

        for network_name in networks:
            print(f"Processing network: {network_name}")
            
            # Prepare directories
            features_dir = os.path.join('data', 'features', timestamp, network_name)
            real_features_dir = os.path.join(features_dir, 'real')
            synthetic_features_dir = os.path.join(features_dir, 'synthetic')
            data_dir = os.path.join('data', 'features', timestamp, network_name, 'metrics')
            
            # Process real dataset
            process_dataset(real_dataset_path, network_name, batch_size, num_sets, real_features_dir)
            
            # Process synthetic dataset
            process_dataset(synthetic_dataset_path, network_name, batch_size, num_sets, synthetic_features_dir)
            
            # Evaluate metrics
            evaluate_metrics(real_features_dir, synthetic_features_dir, network_name, num_sets, data_dir, metrics)

    elif config['eval_only']:

        for network_name in networks:
            print(f"Evaluating metrics for network: {network_name}")
            
            # Prepare directories
            features_dir = os.path.join('data', 'features', timestamp, network_name)
            real_features_dir = os.path.join(features_dir, 'real')
            synthetic_features_dir = os.path.join(features_dir, 'synthetic')
            data_dir = os.path.join('data', 'features', timestamp, network_name, 'metrics')
            
            # Evaluate metrics
            evaluate_metrics(real_features_dir, synthetic_features_dir, network_name, num_sets, data_dir, metrics)
    if config['sing_image_eval']:

        output_dir = os.path.join('data', 'features', timestamp)
        main_single_metric_eval(synthetic_dataset_path, real_dataset_path, output_dir)
        
    if config['realism_correlation']:

        if config['timestamp'] is not None: timestamp = config['timestamp']

        # Azure to local image path linking
        grouped_data = link_azure_local(config['jsonl_path'])
        
        # Function to obtian only realism scores and file names and output z_score normalization
        mean_realism_z_scored = realism_handling(grouped_data)
   
        # Record of each image linking it with the local_path
        # All images sets should contain the same images only iterating for first network
        net_sets_dict = get_sets_content(timestamp)

        dict_sets_realism = get_realism_set_dict(grouped_data, net_sets_dict, mean_realism_z_scored, do_z_score)
        # Get realism dictioanry as qualitative metric in dict format
        #dict_sets_realism = get_realism_set_dict(grouped_data, net_sets_dict)
        #dict_sets_realism = {
        #    1: [72, 50, 50, 22, 50, 22, 50, 22, 50, 22, 50, 22, 50, 23], 
        #    2: [50, 79, 67, 30, 50, 50, 22, 50, 22, 50, 50, 50, 22,  23], 
        #    3: [4, 78, 23, 45, 67, 20, 50, 22, 50, 22, 1, 50, 1, 50 , 23], 
        #    4: [4, 78, 23, 45, 67, 50, 50, 22, 50, 22, 1, 50, 1, 50 , 23], 
        #    5: [4, 78, 23, 45, 67, 40, 50, 22, 50, 22, 1, 50, 1, 50 , 23], 
        #    6: [4, 78, 23, 45, 67, 50, 50, 22, 50, 22, 1, 50, 1, 50 , 23], 
        #    7: [4, 78, 23, 25, 67, 50, 50, 22, 50, 22, 1, 50, 1, 50 , 23], 
        #    8: [4, 78, 23, 45, 67, 50, 50, 22, 40, 22, 1, 50, 1, 50 , 23], 
        #    9: [4, 78, 23, 45, 67, 50, 50, 22, 50, 22, 90, 50, 1, 50 , 23], 
        #    10: [4, 78, 23, 45, 67, 10, 50, 22, 50, 22, 1, 50, 1, 100 , 23]
        #}
        #Set based correlationa analyses of distribution-based metrics
        realism_corr_net(dict_sets_realism, metrics, timestamp)

        #COmpute correlations of single image metric NR and FR with human judgement
        output_dir = os.path.join('data', 'features', timestamp)
        compute_ground_truth_correlations(output_dir, mean_realism_z_scored, do_z_score)
    
    if config.get('privacy_benchmark', False):
        print("Running privacy benchmark...")
        # Extract relevant parameters from config
        privacy_params = {
            'n_features': config.get('n_features', 128),
            'batch_size': config.get('batch_size', 32),
            'target_resolution': tuple(config.get('target_resolution', (224, 224))),
            'split_ratio': config.get('split_ratio', 0.8),
            'num_workers': config.get('num_workers', 4),
            'pin_memory': config.get('pin_memory', True),
            'base_lr': config.get('base_lr', 1e-3),
            'n_epochs': config.get('n_epochs', 10),
            'temperature': config.get('temperature', 0.5),
            'save_model_interval': config.get('save_model_interval', 5),
            'multi_gpu': config.get('multi_gpu', False),
            'loss_type': config.get('loss_type', 'triplet'),
            'margin': config.get('margin', 1.0),
            'swap': config.get('swap',  False),
            'smooth_loss': config.get('smooth_loss', False),
            'triplets_per_anchor': 'all',
            'unfreeze_epoch': config.get('unfreeze_epoch', 10)
        }

        for network_name in config['networks']:
            print(f"Processing network: {network_name}")
            if network_name in network_list:
                train_loader, val_loader, device = setup_training(
                    root_dir=config['real_dataset_path'],
                    network_name=network_name,
                    **privacy_params
                )
    if config.get('adversarial_privacy_assesment', False):

        output_dir = './embeddings'
        network_names = [i for i in config['networks'] if i in network_list]

        for network_name in network_names:
            try:
                model, processor, device, train_standard_loader, train_adversarial_loader, val_standard_loader = load_best_model_for_inference(network_name, config)

                # Perform inference and save embeddings
                embeddings_file = inference_and_save_embeddings(model, processor, device, train_standard_loader, train_adversarial_loader, val_standard_loader, network_name, output_dir)

                # Compute MSD between train_standard and val_standard (baseline)
                # and between train_adversarial and train_standard
                if os.path.exists(embeddings_file):
                    print(f"Processing {network_name}")
                    stats = compute_distances_and_plot(embeddings_file, output_dir, methods=['euclidean', 'pearson', 'spearman'])
                    print(f"Statistics for {network_name}:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")

            except Exception as e:
                print(f"Error processing {network_name}: {str(e)}")
                print(traceback.format_exc())

if __name__ == "__main__":
    main()
