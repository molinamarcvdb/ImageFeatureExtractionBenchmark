import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModel

# Link azure to local path and group answers
def link_azure_local(jsonl_path):
    import json
    import os
    import urllib.parse

    # Function to read JSONL file
    def load_jsonl(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield json.loads(line)

    # Load JSONL data
    data = list(load_jsonl(jsonl_path))

    # Define path mappings
    dict_map = {
        'synth_diff_calc':'./data/synthetic/diffusion',
        'synth_diff_normal':'./data/synthetic/diffusion',
        'synth_gan_calc':'./data/synthetic/gan',
        'synth_gan_normal':'./data/synthetic/gan',
        'real_normal':'./data/real',
        'real_calc':'./data/real'
    }

    # Group assessments by filename
    grouped_data = {}

    for i, record in enumerate(data):
        azure_path = record['image_path']
        category = record['category']

        # Extract filename from the Azure path
        filename = azure_path.split('/')[6].split('?')[0]

        # Generate the local file path based on the category
        if category in dict_map:
            local_path = None
            if category == 'synth_gan_calc':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_calc.jpeg')
            elif category == 'synth_diff_calc':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_calc.png')
            elif category == 'synth_gan_normal':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_normal.jpeg')
            elif category == 'synth_diff_normal':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_normal.png')
            else:
                local_path = os.path.join(dict_map[category], filename)

            if local_path:
                # Check if the local file exists
                if os.path.exists(local_path):
                    record['local_path'] = local_path
                    grouped_data[i] = record
                else: print(local_path)
    return grouped_data

import os
import numpy as np
# Link azure to local path and group answers
def link_azure_local(jsonl_path):
    import json
    import os

    # Function to read JSONL file
    def load_jsonl(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield json.loads(line)

    # Load JSONL data
    data = list(load_jsonl(jsonl_path))

    # Define path mappings
    dict_map = {
        'synth_diff_calc':'./data/synthetic/diffusion',
        'synth_diff_normal':'./data/synthetic/diffusion',
        'synth_gan_calc':'./data/synthetic/gan',
        'synth_gan_normal':'./data/synthetic/gan',
        'real_normal':'./data/real',
        'real_calc':'./data/real'
    }

    # Group assessments by filename
    grouped_data = {}

    for i, record in enumerate(data):
        azure_path = record['image_path']
        category = record['category']

        # Extract filename from the Azure path
        filename = azure_path.split('/')[6].split('?')[0]

        # Generate the local file path based on the category
        if category in dict_map:
            local_path = None
            if category == 'synth_gan_calc':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_calc.jpeg')
            elif category == 'synth_diff_calc':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_calc.png')
            elif category == 'synth_gan_normal':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_normal.jpeg')
            elif category == 'synth_diff_normal':
                local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0]+'_normal.png')
            else:
                local_path = os.path.join(dict_map[category], filename)

            if local_path:
                # Check if the local file exists
                if os.path.exists(local_path):
                    record['local_path'] = local_path
                    grouped_data[i] = record
                else: print(local_path)
    return grouped_data

def get_sets_content(timestamp):
    case_dir = os.path.join('./data/features', timestamp)

    net_sets_dict = {}
    #Itreate over networks
    for network in os.listdir(case_dir):
        # Iterate over sets
        for i, file in enumerate([file for file in os.listdir(os.path.join(case_dir, network, 'synthetic')) if file.endswith('filenames.npy')]):
            actual_set = i+1
            list_files_set = np.load(os.path.join(case_dir, network, 'synthetic', file))
            net_sets_dict[actual_set] = list(list_files_set)
        break
    
    return net_sets_dict


def get_realism_set_dict(grouped_data, net_sets_dict):

    dict_sets_realism = {i+1: [] for i in range(len(net_sets_dict.keys()))}

    for i in grouped_data:
        local_path = grouped_data[i]['local_path']
        for j in net_sets_dict:
            if local_path in net_sets_dict[j]:
                dict_sets_realism[j].append(grouped_data[i]['realism_score'])
                break
    return dict_sets_realism

def realism_corr_net(dict_sets_realism, metrics, timestamp):
    realism_doc_1 = [np.mean(dict_sets_realism[i]) for i in dict_sets_realism]

    timestamp_dir = os.path.join('data/features', timestamp)

    # Initialize an empty list to store modified DataFrames
    dfs = []
    networks = [net_dir for net_dir in os.listdir(timestamp_dir) if os.path.isdir(os.path.join(timestamp_dir, net_dir))]
    
    for network in networks:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(timestamp_dir, network, f'metrics/{network}_aggregated_metrics.csv'))
        
        # Filter for only the mean columns
        mean_cols = [col for col in df.columns if col.endswith('_mean')]
        df = df[mean_cols]
        
        # Rename columns to remove '_mean' suffix
        df.columns = [col.replace('_mean', '') for col in df.columns]
        
        # Prefix the columns with the network name
        df.columns = [f'{network}_{col}' for col in df.columns]
        
        # Append the modified DataFrame to the list
        dfs.append(df)

    # Concatenate all the DataFrames into a single DataFrame
    aggregated_df = pd.concat(dfs, axis=1)

    aggregated_df['realism_doc_1'] = realism_doc_1

    # Initialize a new DataFrame to store correlations
    correlation_results = pd.DataFrame(index=networks, columns=metrics)

    # Negate metrics where lower values are better (e.g., FID)
    negate_metrics = ['fid']

    # Loop over each network and compute correlations for each metric with realism_doc_1
    for network in networks:
        for metric in metrics:
            col_name = f'{network}_{metric}'
            if col_name in aggregated_df.columns:
                corr_value = aggregated_df[col_name].corr(aggregated_df['realism_doc_1'])
                if metric in negate_metrics:
                    corr_value = -corr_value  # Negate for better interpretation
                correlation_results.loc[network, metric] = corr_value

    # Convert the correlation results to numeric values, handling NaN values
    correlation_results = correlation_results.astype(float).fillna(0)

    # Create the figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 15), gridspec_kw={'height_ratios': [1, 0.4], 'width_ratios': [0.5, 0.2]})

    # 1. Overall Correlation Heatmap
    sns.heatmap(correlation_results, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[0, 0])
    axs[0, 0].set_title('Overall Correlation Heatmap of Networks and Metrics with Realism')
    
    # 3. Best Network: Compare normally pretrained and "rad" networks
    rad_networks = correlation_results.index.str.startswith('rad')
    non_rad_networks = ~rad_networks
    avg_corr_rad = correlation_results.loc[rad_networks].mean(axis=0)
    avg_corr_non_rad = correlation_results.loc[non_rad_networks].mean(axis=0)
    avg_corr_network = pd.DataFrame({
        'rad': avg_corr_rad,
        'non-rad': avg_corr_non_rad
    })

    avg_corr_network = avg_corr_network
    sns.heatmap(avg_corr_network, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[0, 1])
    axs[0, 1].set_title('Average Correlation of Metrics by Network Type')
    axs[0, 1].set_yticklabels(axs[0 , 1].get_yticklabels(), rotation=0)



    # 2. Best Metric: Aggregate correlations across all networks
    best_metric = correlation_results.mean(axis=0)
    best_metric_df = best_metric.to_frame(name='Correlation').T
    sns.heatmap(best_metric_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[1, 0])
    axs[1, 0].set_title('Average Correlation of Each Metric Across All Networks')
    axs[1, 0].set_yticklabels(axs[1, 0].get_yticklabels(), rotation=0)

   
    # 4. Aggregated Correlations by Base Network
    def get_base_network(name):
        if 'rad' in name:
            return name.replace('rad_', '')
        elif 'med' in name:
            return name.replace('med_', '')
        else:
            return name

    base_network_names = correlation_results.index.map(get_base_network)
    agg_corr_by_base_network_metric = pd.DataFrame(index=pd.unique(base_network_names), columns=correlation_results.columns)

    for base_network in pd.unique(base_network_names):
        for metric in correlation_results.columns:
            base_correlations = correlation_results.loc[base_network_names == base_network, metric]
            if not base_correlations.isna().all():
                avg_corr = base_correlations.mean()
                agg_corr_by_base_network_metric.loc[base_network, metric] = avg_corr
            else:
                agg_corr_by_base_network_metric.loc[base_network, metric] = float('nan')

    agg_corr_by_base_network_metric = agg_corr_by_base_network_metric.apply(pd.to_numeric, errors='coerce')
    best_network = agg_corr_by_base_network_metric.mean(axis=1).to_frame(name='Correlation')
    sns.heatmap(best_network, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[1, 1])
    axs[1, 1].set_title('Aggregated Correlation of Base Networks and Metrics')
    axs[1, 1].set_yticklabels(axs[1, 1].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(timestamp_dir, 'corr_turing_quant.png'))
    correlation_results.to_csv(os.path.join(timestamp_dir, 'corr_turing_quant.csv'))

    return correlation_results



import os
import pickle
import torch
import tensorflow as tf
import tensorflow_hub as hub
from huggingface_hub import hf_hub_download

def load_model_from_hub(model_repo_id, file_name):
    # Define the mapping of file extensions to loading functions
    def load_pytorch_model(file_path):
        print(f"Loading PyTorch model from {file_path}...")
        return torch.load(file_path, map_location='cpu')

    def load_tensorflow_model(file_path):
        print(f"Loading TensorFlow model from {file_path}...")
        return tf.keras.models.load_model(file_path)

    def load_graph_def(file_path):
        print(f"Loading TensorFlow GraphDef from {file_path}...")
        with open(file_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

    def load_pickled_model(file_path):
        print(f"Loading Pickle model from {file_path}...")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def load_musiq_model():
        print(f"Loading MUSIQ model './pretrained/MUSIQ/'...")
        
        return hub.KerasLayer('./pretrained/MUSIQ/', signature="serving_default", output_key="output_0")

    file_ext = os.path.splitext(file_name)[1]

    # Download the file from the Hugging Face Hub
    if not file_name.endswith('MUSIQ'):
        model_path = hf_hub_download(repo_id=model_repo_id, filename=file_name)
    else:

        save_dir = './pretrained/'
        os.makedirs(save_dir, exist_ok=True)

        hf_hub_download(repo_id=model_repo_id, filename='MUSIQ/musiq_spaq_ckpt.npz', local_dir=save_dir)
        hf_hub_download(repo_id=model_repo_id, filename='MUSIQ/saved_model.pb', local_dir=save_dir)
        hf_hub_download(repo_id=model_repo_id, filename='MUSIQ/archive.tar', local_dir=save_dir)
        hf_hub_download(repo_id=model_repo_id, filename='MUSIQ/variables/variables.data-00000-of-00001', local_dir=save_dir)
        hf_hub_download(repo_id=model_repo_id, filename='MUSIQ/variables/variables.index', local_dir=save_dir)
        

    if file_ext in ['.pth', '.pt']:
        # Load PyTorch model
        return load_pytorch_model(model_path)

    elif file_ext in ['.bin', '.pt']:
        # Load Hugging Face PyTorch model
        print(f"Loading Hugging Face PyTorch model from {model_path}...")
        return AutoModel.from_pretrained(model_repo_id)

    elif file_ext in ['.h5', '.pb']:
        # Load TensorFlow/Keras model or GraphDef
        if file_ext == '.pb':
            # Load TensorFlow GraphDef
            return load_graph_def(model_path)
        else:
            # Load TensorFlow/Keras model
            return load_tensorflow_model(model_path)

    elif file_ext == '.ckpt':
        # Load PyTorch checkpoint
        return load_pytorch_model(model_path)

    elif file_ext == '.pkl':
        # Load Pickle model
        return load_pickled_model(model_path)

    elif file_name.endswith('MUSIQ'):
        # Load MUSIQ model
        return load_musiq_model()

    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")