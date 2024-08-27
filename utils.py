import os
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModel
from scipy.stats import spearmanr, kendalltau
import pickle
import torch
import tensorflow as tf
import tensorflow_hub as hub
from huggingface_hub import hf_hub_download

# Link azure to local path and group answers
def link_azure_local(jsonl_paths):
    """
    Process one or more JSONL files and link Azure paths to local paths based on predefined categories.

    Parameters:
    jsonl_paths (str or list): A string representing a single JSONL file path or a list of JSONL file paths.

    Returns:
    list: A list containing dictionaries of grouped data from each processed JSONL file.
    """

    # Function to read JSONL file
    def load_jsonl(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield json.loads(line)

    # Define path mappings
    dict_map = {
        'synth_diff_calc': './data/synthetic/diffusion',
        'synth_diff_normal': './data/synthetic/diffusion',
        'synth_gan_calc': './data/synthetic/gan',
        'synth_gan_normal': './data/synthetic/gan',
        'real_normal': './data/real',
        'real_calc': './data/real'
    }

    # Process a single JSONL file
    def process_single_file(jsonl_path):
        data = list(load_jsonl(jsonl_path))
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
                    local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0] + '_calc.jpeg')
                elif category == 'synth_diff_calc':
                    local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0] + '_calc.png')
                elif category == 'synth_gan_normal':
                    local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0] + '_normal.jpeg')
                elif category == 'synth_diff_normal':
                    local_path = os.path.join(dict_map[category], os.path.splitext(filename)[0] + '_normal.png')
                else:
                    local_path = os.path.join(dict_map[category], filename)

                if local_path:
                    # Check if the local file exists
                    if os.path.exists(local_path):
                        record['local_path'] = local_path
                        grouped_data[i] = record
                    else:
                        print(local_path)

        return grouped_data

    # Check if input is a list of JSONL files
    if isinstance(jsonl_paths, list):
        results = []
        for jsonl_path in jsonl_paths:
            grouped_data = process_single_file(jsonl_path)
            results.append(grouped_data)
        return results
    else:
        # Process a single JSONL file
        return process_single_file(jsonl_paths)

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
def get_realism_set_dict(grouped_data, net_sets_dict, mean_realism_z_scored, do_z_score):
    """
    Generates a dictionary of realism scores (or Z-scores) for each set in net_sets_dict.

    Parameters:
    - grouped_data (dict or list of dict): The original data containing file paths and realism scores. 
      If it's a list, it means multiple JSONL files were provided.
    - net_sets_dict (dict): Dictionary where keys are set indices and values are lists of local paths.
    - mean_realism_z_scored (pd.DataFrame): DataFrame with filenames as index, containing Realism Score, Z-Score, and Averaged Z-Score.
    - do_z_score (bool): Determines whether to use the Z-Score or the Averaged Z-Score.

    Returns:
    - dict: A dictionary where keys are set indices and values are lists of the selected realism score.
    """

    # Initialize a dictionary to store realism scores or Z-scores for each set
    dict_sets_realism = {i + 1: [] for i in range(len(net_sets_dict.keys()))}

    # If grouped_data is a list, take the first element for local path mapping
    if isinstance(grouped_data, list):
        first_group = grouped_data[0]
  
		# Determine which score to use based on do_z_score
        score_column = 'Averaged Z-Score' if do_z_score else 'Averaged Realism Score'
    
    else:
        first_group = grouped_data

		# Determine which score to use based on do_z_score
        score_column = 'Z-Score' if do_z_score else 'Realism Score'
    
    # Iterate through the first_group to match local paths with net_sets_dict and populate dict_sets_realism
    for i in first_group:
        local_path = first_group[i]['local_path']
        
        # Ensure local_path is in the mean_realism_z_scored DataFrame
        if os.path.basename(local_path) in mean_realism_z_scored.index:
            selected_score = mean_realism_z_scored.loc[os.path.basename(local_path), score_column]

            for j in net_sets_dict:
                if local_path in net_sets_dict[j]:
                    dict_sets_realism[j].append(selected_score)
                    break

    return dict_sets_realism

#def get_realism_set_dict(grouped_data, net_sets_dict):
#
#    dict_sets_realism = {i+1: [] for i in range(len(net_sets_dict.keys()))}
#
#    for i in grouped_data:
#        local_path = grouped_data[i]['local_path']
#        for j in net_sets_dict:
#            if local_path in net_sets_dict[j]:
#                dict_sets_realism[j].append(grouped_data[i]['realism_score'])
#                break
#    return dict_sets_realism
def realism_corr_net(dict_sets_realism, metrics, timestamp):
    # Compute mean realism scores (already Z-score normalized)
    realism_doc_1 = [np.mean(dict_sets_realism[i]) for i in dict_sets_realism]

    timestamp_dir = os.path.join('data/features', timestamp)
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

    # Add realism scores to the aggregated DataFrame
    aggregated_df['realism_doc_1'] = realism_doc_1

    # Initialize a new DataFrame to store SRCC and KRCC correlations
    correlation_results_srcc = pd.DataFrame(index=networks, columns=metrics)
    correlation_results_krcc = pd.DataFrame(index=networks, columns=metrics)

    # List of metrics where lower values are better (e.g., FID)
    negate_metrics = ['fid']

    # Loop over each network and compute SRCC and KRCC correlations
    for network in networks:
        for metric in metrics:
            col_name = f'{network}_{metric}'
            if col_name in aggregated_df.columns:
                # Spearman Rank Correlation (SRCC)
                srcc_value, _ = spearmanr(aggregated_df[col_name], aggregated_df['realism_doc_1'])
                # Kendall Rank Correlation (KRCC)
                krcc_value, _ = kendalltau(aggregated_df[col_name], aggregated_df['realism_doc_1'])

                # Negate if necessary
                if metric in negate_metrics:
                    srcc_value = -srcc_value
                    krcc_value = -krcc_value

                # Store the absolute values of the correlations
                correlation_results_srcc.loc[network, metric] = abs(srcc_value)
                correlation_results_krcc.loc[network, metric] = abs(krcc_value)

    # Convert the correlation results to numeric values, handling NaN values
    correlation_results_srcc = correlation_results_srcc.astype(float).fillna(0)
    correlation_results_krcc = correlation_results_krcc.astype(float).fillna(0)
	
    plot_correlation_results(correlation_results_srcc, correlation_results_krcc, timestamp_dir)

    return correlation_results_srcc, correlation_results_krcc 

def plot_correlation_results(correlation_results_srcc, correlation_results_krcc, timestamp_dir):
    """
    Plots the correlation results for SRCC and KRCC in multiple heatmaps.

    Parameters:
    - correlation_results_srcc: DataFrame containing SRCC correlation results.
    - correlation_results_krcc: DataFrame containing KRCC correlation results.
    - timestamp_dir: Directory where plots and CSV files will be saved.
    """
    # Create the figure and axes
    fig, axs = plt.subplots(2, 4, figsize=(30, 15), gridspec_kw={'height_ratios': [1, 0.4], 'width_ratios': [0.5, 0.2, 0.5, 0.2]})

    # Plot for SRCC
    sns.heatmap(correlation_results_srcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[0, 0])
    axs[0, 0].set_title('SRCC: Overall Correlation Heatmap of Networks and Metrics with Realism')

    # Plot for KRCC
    sns.heatmap(correlation_results_krcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[0, 2])
    axs[0, 2].set_title('KRCC: Overall Correlation Heatmap of Networks and Metrics with Realism')

    # SRCC: Compare normally pretrained and "rad" networks
    rad_networks = correlation_results_srcc.index.str.startswith('rad')
    non_rad_networks = ~rad_networks
    avg_corr_rad_srcc = correlation_results_srcc.loc[rad_networks].mean(axis=0)
    avg_corr_non_rad_srcc = correlation_results_srcc.loc[non_rad_networks].mean(axis=0)
    avg_corr_network_srcc = pd.DataFrame({
        'rad': avg_corr_rad_srcc,
        'non-rad': avg_corr_non_rad_srcc
    })
    sns.heatmap(avg_corr_network_srcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[0, 1])
    axs[0, 1].set_title('SRCC: Average Correlation of Metrics by Network Type')
    axs[0, 1].set_yticklabels(axs[0 , 1].get_yticklabels(), rotation=0)

    # KRCC: Compare normally pretrained and "rad" networks
    avg_corr_rad_krcc = correlation_results_krcc.loc[rad_networks].mean(axis=0)
    avg_corr_non_rad_krcc = correlation_results_krcc.loc[non_rad_networks].mean(axis=0)
    avg_corr_network_krcc = pd.DataFrame({
        'rad': avg_corr_rad_krcc,
        'non-rad': avg_corr_non_rad_krcc
    })
    sns.heatmap(avg_corr_network_krcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[0, 3])
    axs[0, 3].set_title('KRCC: Average Correlation of Metrics by Network Type')
    axs[0, 3].set_yticklabels(axs[0 , 3].get_yticklabels(), rotation=0)

    # SRCC: Best Metric
    best_metric_srcc = correlation_results_srcc.mean(axis=0)
    best_metric_df_srcc = best_metric_srcc.to_frame(name='Correlation').T
    sns.heatmap(best_metric_df_srcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[1, 0])
    axs[1, 0].set_title('SRCC: Average Correlation of Each Metric Across All Networks')
    axs[1, 0].set_yticklabels(axs[1, 0].get_yticklabels(), rotation=0)

    # KRCC: Best Metric
    best_metric_krcc = correlation_results_krcc.mean(axis=0)
    best_metric_df_krcc = best_metric_krcc.to_frame(name='Correlation').T
    sns.heatmap(best_metric_df_krcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[1, 2])
    axs[1, 2].set_title('KRCC: Average Correlation of Each Metric Across All Networks')
    axs[1, 2].set_yticklabels(axs[1, 2].get_yticklabels(), rotation=0)

    # SRCC: Aggregated Correlations by Base Network
    def get_base_network(name):
        if 'rad' in name:
            return name.replace('rad_', '')
        elif 'med' in name:
            return name.replace('med_', '')
        else:
            return name

    base_network_names = correlation_results_srcc.index.map(get_base_network)
    agg_corr_by_base_network_metric_srcc = pd.DataFrame(index=pd.unique(base_network_names), columns=correlation_results_srcc.columns)
    for base_network in pd.unique(base_network_names):
        for metric in correlation_results_srcc.columns:
            base_correlations = correlation_results_srcc.loc[base_network_names == base_network, metric]
            if not base_correlations.isna().all():
                avg_corr = base_correlations.mean()
                agg_corr_by_base_network_metric_srcc.loc[base_network, metric] = avg_corr
            else:
                agg_corr_by_base_network_metric_srcc.loc[base_network, metric] = float('nan')

    agg_corr_by_base_network_metric_srcc = agg_corr_by_base_network_metric_srcc.apply(pd.to_numeric, errors='coerce')
    best_network_srcc = agg_corr_by_base_network_metric_srcc.mean(axis=1).to_frame(name='Correlation')
    sns.heatmap(best_network_srcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[1, 1])
    axs[1, 1].set_title('SRCC: Aggregated Correlation of Base Networks and Metrics')
    axs[1, 1].set_yticklabels(axs[1, 1].get_yticklabels(), rotation=0)

    # KRCC: Aggregated Correlations by Base Network
    agg_corr_by_base_network_metric_krcc = pd.DataFrame(index=pd.unique(base_network_names), columns=correlation_results_krcc.columns)
    for base_network in pd.unique(base_network_names):
        for metric in correlation_results_krcc.columns:
            base_correlations = correlation_results_krcc.loc[base_network_names == base_network, metric]
            if not base_correlations.isna().all():
                avg_corr = base_correlations.mean()
                agg_corr_by_base_network_metric_krcc.loc[base_network, metric] = avg_corr
            else:
                agg_corr_by_base_network_metric_krcc.loc[base_network, metric] = float('nan')

    agg_corr_by_base_network_metric_krcc = agg_corr_by_base_network_metric_krcc.apply(pd.to_numeric, errors='coerce')
    best_network_krcc = agg_corr_by_base_network_metric_krcc.mean(axis=1).to_frame(name='Correlation')
    sns.heatmap(best_network_krcc, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[1, 3])
    axs[1, 3].set_title('KRCC: Aggregated Correlation of Base Networks and Metrics')
    axs[1, 3].set_yticklabels(axs[1, 3].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(timestamp_dir, 'corr_turing_quant_srcc_krcc.png'))

    correlation_results_srcc.to_csv(os.path.join(timestamp_dir, 'corr_turing_quant_srcc.csv'))
    correlation_results_krcc.to_csv(os.path.join(timestamp_dir, 'corr_turing_quant_krcc.csv'))



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













































































































