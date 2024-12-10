import os
import yaml
import pandas as pd
from collections import Counter
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
import scipy.stats as stats


def load_jsonl(file_path):
    with open(file_path, "r") as file:
        for line in file:
            yield json.loads(line)


import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil


def convert_to_npy(input_paths, output_dir, shape):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    expected_npy_files = [
        os.path.splitext(os.path.basename(path))[0] + ".npy" for path in input_paths
    ]
    existing_npy_files = [f for f in os.listdir(output_dir) if f.endswith(".npy")]

    if set(expected_npy_files) == set(existing_npy_files):
        print("All files already converted. Skipping conversion.")
        return [os.path.join(output_dir, f) for f in existing_npy_files]

    # If mismatch, clear the output directory and perform conversion
    print("Mismatch in files. Clearing output directory and performing conversion.")
    try:
        shutil.rmtree(output_dir)
    except FileNotFoundError as e:
        print("FileNotFoundError", e)
        pass
    os.makedirs(output_dir, exist_ok=True)

    new_paths = []
    file_sizes = []

    for input_path in tqdm(input_paths):
        if input_path.endswith((".png", ".jpeg", ".jpg", ".tiff")):
            # Open the image file
            img = Image.open(input_path)
            # Resize the image
            img_resized = img.resize((shape[1], shape[0]), Image.LANCZOS)
            # Convert to numpy array
            img_array = np.array(img_resized)
            # Create output filename
            base_name = os.path.basename(input_path)
            npy_filename = os.path.splitext(base_name)[0] + ".npy"
            output_path = os.path.join(output_dir, npy_filename)

            # Save as NPY file
            np.save(output_path, img_array)

            # Verify file size
            file_size = os.path.getsize(output_path)
            file_sizes.append(file_size)

            new_paths.append(output_path)

    # Check if all file sizes are the same
    if len(set(file_sizes)) == 1:
        pass
        # print(f"Warning: All converted files have the same size of {file_sizes[0]} bytes.")
    else:
        print(
            f"File sizes vary. Min: {min(file_sizes)}, Max: {max(file_sizes)}, Average: {sum(file_sizes)/len(file_sizes)}"
        )

    # Verify that files can be loaded
    # for path in new_paths:
    #    try:
    #        loaded_array = np.load(path, mmap_mode='r')
    #        if loaded_array.shape != shape:
    #            print(f"Warning: Shape mismatch in {path}. Expected {shape}, got {loaded_array.shape}")
    #    except Exception as e:
    #        print(f"Error loading {path}: {str(e)}")

    return new_paths


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


from ijepa.src.helper import load_checkpoint, init_model
import torch

import os
import yaml
from collections import OrderedDict


def load_ddp_model(model, checkpoint_path, network_key):
    # Load the state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")[network_key]

    # Check if it's a DDP saved model
    if not hasattr(model, "module") and all(
        k.startswith("module.") for k in state_dict.keys()
    ):
        # Create a new OrderedDict without the 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v

        # Load the modified state dict
        model.load_state_dict(new_state_dict)
    else:
        # If it's not a DDP saved model, load normally
        model.load_state_dict(state_dict)

    return model


def load_ijepa_not_contrastive(config):

    model_dir = config["ijepa_model_dir"]

    yaml_file = os.path.join(
        model_dir, [file for file in os.listdir(model_dir) if file.endswith(".yaml")][0]
    )
    tar_file = os.path.join(
        model_dir,
        [file for file in os.listdir(model_dir) if file.endswith(".pth.tar")][0],
    )

    with open(yaml_file, "r") as fh:

        config = yaml.safe_load(fh)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    encoder, _ = init_model(
        device=device,
        patch_size=config["mask"]["patch_size"],
        crop_size=config["data"]["crop_size"],
        pred_depth=config["meta"]["pred_depth"],
        pred_emb_dim=config["meta"]["pred_emb_dim"],
        model_name=config["meta"]["model_name"],
    )

    encoder = load_ddp_model(encoder, tar_file, "encoder")

    return encoder


# Link azure to local path and group answers
def link_azure_local(config):
    """
    Process one or more JSONL files and link Azure paths to local paths based on predefined categories.

    Parameters:
    jsonl_paths (str or list): A string representing a single JSONL file path or a list of JSONL file paths.

    Returns:
    list: A list containing dictionaries of grouped data from each processed JSONL file.
    """

    # Function to read JSONL file
    jsonl_paths = config["jsonl_path"]
    # Define path mappings
    dict_map = {
        "synth_diff_calc": "./data/synthetic/diff",
        "synth_diff_normal": "./data/synthetic/diff",
        "synth_gan_calc": "./data/synthetic/gan",
        "synth_gan_normal": "./data/synthetic/gan",
        "real_normal": "./data/real",
        "real_calc": "./data/real",
    }

    # Process a single JSONL file
    def process_single_file(jsonl_path):
        data = list(load_jsonl(jsonl_path))
        grouped_data = {}

        for i, record in enumerate(data):
            azure_path = record["image_path"]
            category = record["category"]

            # Extract filename from the Azure path
            filename = azure_path.split("/")[6].split("?")[0]

            # Generate the local file path based on the category
            if category in dict_map:
                local_path = None
                if category == "synth_gan_calc":
                    local_path = os.path.join(
                        dict_map[category], os.path.splitext(filename)[0] + "_calc.jpeg"
                    )
                elif category == "synth_diff_calc":
                    local_path = os.path.join(
                        dict_map[category], os.path.splitext(filename)[0] + "_calc.png"
                    )
                elif category == "synth_gan_normal":
                    local_path = os.path.join(
                        dict_map[category],
                        os.path.splitext(filename)[0] + "_normal.jpeg",
                    )
                elif category == "synth_diff_normal":
                    local_path = os.path.join(
                        dict_map[category],
                        os.path.splitext(filename)[0] + "_normal.png",
                    )
                else:
                    local_path = os.path.join(dict_map[category], filename)

                if local_path:
                    # Check if the local file exists
                    if os.path.exists(local_path):
                        record["local_path"] = local_path
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


def get_sets_content(timestamp, model_to_seek, source="synthetic"):
    case_dir = os.path.join("./data/features", timestamp)

    if not isinstance(model_to_seek, list):
        model_to_seek = list(model_to_seek)

    set_dict_models = []
    for mname in model_to_seek:
        net_sets_dict = {}
        # Itreate over networks
        for network in os.listdir(case_dir):
            if os.path.isdir(os.path.join(case_dir, network)):
                # Iterate over sets
                for i, file in enumerate(
                    [
                        file
                        for file in os.listdir(os.path.join(case_dir, network, source))
                        if file.endswith("filenames.npy") and file.startswith(mname)
                    ]
                ):
                    actual_set = i + 1
                    list_files_set = np.load(
                        os.path.join(case_dir, network, source, file)
                    )
                    net_sets_dict[actual_set] = list(list_files_set)
                break  # We jsut need to iterate once as all networks have the same data split
        set_dict_models.append(net_sets_dict)

    return set_dict_models


def get_realism_set_dict(
    grouped_data, net_sets_all, mean_realism_z_scored, do_z_score, model_to_seek
):
    """
    Generates a dictionary of realism scores (or Z-scores) for each set in net_sets_dict.

    Parameters:
    - grouped_data (dict or list of dict): The original data containing file paths and realism scores.
      If it's a list, it means multiple JSONL files were provided.
    - net_sets_dict (dict): Dictionary where keys are set indices and values are lists of local paths.
    - mean_realism_z_scored (pd.DataFrame): DataFrame with filenames as index, containing Realism Score, Z-Score, and Averaged Z-Score.
    - do_z_score (bool): Determines whether to use the Averaged Realism Score  or the Averaged Z-Score.

    Returns:
    - dict: A dictionary where keys are set indices and values are lists of the selected realism score.
    """
    sets_realism, sets_her = [], []

    if not isinstance(model_to_seek, list):

        model_to_seek = [model_to_seek]
    for net_sets_dict, mname in zip(net_sets_all, model_to_seek):

        # Initialize a dictionary to store realism scores or Z-scores for each set
        dict_sets_realism = {i + 1: [] for i in range(len(net_sets_dict.keys()))}
        dict_sets_her = {i + 1: [] for i in range(len(net_sets_dict.keys()))}

        # If grouped_data is a list, take the first element for local path mapping
        if isinstance(grouped_data, list):
            first_group = grouped_data[0]

            # Determine which score to use based on do_z_score
            score_column = (
                "Averaged Z-Score" if do_z_score else "Averaged Realism Score"
            )
            scnd_score_column = "AggConfMatScore"
        else:
            first_group = grouped_data

            # Determine which score to use based on do_z_score
            score_column = "Z-Score" if do_z_score else "Realism Score"

        # Iterate through the first_group to match local paths with net_sets_dict and populate dict_sets_realism
        for i in first_group:
            local_path = first_group[i]["local_path"][2:]
            # Ensure local_path is in the mean_realism_z_scored DataFrame
            if (
                os.path.basename(local_path) in mean_realism_z_scored.index
                and mname in local_path
            ):

                selected_score = mean_realism_z_scored.loc[
                    os.path.basename(local_path), score_column
                ]
                second_selected_score = mean_realism_z_scored.loc[
                    os.path.basename(local_path), scnd_score_column
                ]

                for j in net_sets_dict:

                    if local_path in net_sets_dict[j]:

                        dict_sets_realism[j].append(selected_score)
                        dict_sets_her[j].append(second_selected_score)

                        break
        dict_sets_her = compute_agg_human_error(dict_sets_her, mname)

        sets_realism.append(dict_sets_realism)
        sets_her.append(dict_sets_her)

    return sets_realism, sets_her


def compute_agg_human_error(dict_sets_her, source):

    for sete, values in dict_sets_her.items():

        FN, FP, TN, TP = 0, 0, 0, 0
        her = []
        for idx in range(len(values[0])):
            list_conf_mat = [values[i][idx] for i in range(len(values))]
            dict_conf_mat = Counter(list_conf_mat)
            if source != "real":
                her.append(
                    dict_conf_mat["FN"] / (dict_conf_mat["FN"] + dict_conf_mat["TP"])
                )
            else:

                her.append(
                    dict_conf_mat["FP"] / (dict_conf_mat["FP"] + dict_conf_mat["TN"])
                )

        her = np.mean(her)

        dict_sets_her[sete] = her

    return dict_sets_her


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def realism_corr_net(dict_sets_all, metrics, timestamp, name, model_to_seek):
    # Compute mean realism scores (already Z-score normalized)

    aggregated_models_df = []
    for mname, dict_sets_realism in zip(model_to_seek, dict_sets_all):

        realism_doc_1 = [
            np.mean(dict_sets_realism[i])
            if isinstance(dict_sets_realism[i], list)
            else dict_sets_realism[i]
            for i in dict_sets_realism
        ]

        timestamp_dir = os.path.join("data/features", timestamp)
        dfs = []
        networks = [
            net_dir
            for net_dir in os.listdir(timestamp_dir)
            if os.path.isdir(os.path.join(timestamp_dir, net_dir))
        ]

        for network in networks:
            try:
                # Load the CSV file into a DataFrame
                df = pd.read_csv(
                    os.path.join(
                        timestamp_dir,
                        network,
                        f"metrics/{mname}_{network}_aggregated_metrics.csv",
                    )
                )

                # Filter for only the mean columns
                mean_cols = [col for col in df.columns if col.endswith("_mean")]
                df = df[mean_cols]

                # Rename columns to remove '_mean' suffix
                df.columns = [col.replace("_mean", "") for col in df.columns]

                # Prefix the columns with the network name
                df.columns = [f"{network}_{col}" for col in df.columns]

                # Append the modified DataFrame to the list
                dfs.append(df)
            except:
                pass
        # Concatenate all the DataFrames into a single DataFrame
        aggregated_df = pd.concat(dfs, axis=1)

        # Add realism scores to the aggregated DataFrame
        aggregated_df["realism_doc_1"] = realism_doc_1

        aggregated_models_df.append(aggregated_df)

    aggregated_df = combine_dataframes_with_renamed_index(aggregated_models_df)
    # Initialize a new DataFrame to store SRCC and KRCC correlations
    correlation_results_srcc = pd.DataFrame(index=networks, columns=metrics)
    correlation_results_krcc = pd.DataFrame(index=networks, columns=metrics)

    # List of metrics where lower values are better (e.g., FID)
    negate_metrics = ["fid", "kid", "mmd", "fid_inf", "sw"]

    # Initialize DataFrames for correlations and p-values
    correlation_results_srcc = pd.DataFrame(index=networks, columns=metrics)
    correlation_results_krcc = pd.DataFrame(index=networks, columns=metrics)
    pvalues_srcc = pd.DataFrame(index=networks, columns=metrics)
    pvalues_krcc = pd.DataFrame(index=networks, columns=metrics)

    # Loop over each network and compute SRCC and KRCC correlations
    for network in networks:
        for metric in metrics:
            col_name = f"{network}_{metric}"
            if col_name in aggregated_df.columns:
                # Spearman Rank Correlation (SRCC)
                srcc_value, srcc_pvalue = spearmanr(
                    aggregated_df[col_name], aggregated_df["realism_doc_1"]
                )
                # Kendall Rank Correlation (KRCC)
                krcc_value, krcc_pvalue = kendalltau(
                    aggregated_df[col_name], aggregated_df["realism_doc_1"]
                )

                # Negate if necessary
                if metric in negate_metrics:
                    srcc_value = -srcc_value
                    krcc_value = -krcc_value

                # Store correlations and p-values
                correlation_results_srcc.loc[network, metric] = srcc_value
                correlation_results_krcc.loc[network, metric] = krcc_value
                pvalues_srcc.loc[network, metric] = srcc_pvalue
                pvalues_krcc.loc[network, metric] = krcc_pvalue

    # Convert results to numeric values
    correlation_results_srcc = correlation_results_srcc.astype(float).fillna(0)
    correlation_results_krcc = correlation_results_krcc.astype(float).fillna(0)
    pvalues_srcc = pvalues_srcc.astype(float).fillna(1)  # Default to 1 for NaN p-values
    pvalues_krcc = pvalues_krcc.astype(float).fillna(1)

    # Pass both correlations and p-values to plotting function
    plot_correlation_results(
        correlation_results_srcc,
        pvalues_srcc,
        timestamp_dir,
        name,
        model_to_seek,
    )
    return correlation_results_srcc, correlation_results_krcc


def combine_dataframes_with_renamed_index(df_list):
    if not df_list:
        return pd.DataFrame()

    dfs = []
    for i, df in enumerate(df_list):
        df_copy = df.copy()
        if i > 0:
            df_copy.index = [f"{idx}_{i}" for idx in df.index]
        dfs.append(df_copy)

    return pd.concat(dfs)


def corr_analysis_single_img_by_set(
    dict_sets_all, dict_sets_her_all, net_sets_dict_all, timestamp, model_to_seek
):
    if not isinstance(model_to_seek, list):
        model_to_seek = [model_to_seek]

    sim_sets_model_df = []
    for mname, dict_sets_realism, dict_sets_her, net_sets_dict in zip(
        model_to_seek, dict_sets_all, dict_sets_her_all, net_sets_dict_all
    ):
        timestamp_dir = os.path.join("data/features", timestamp)

        realism_metric = [
            np.mean(dict_sets_realism[i])
            if isinstance(dict_sets_realism[i], list)
            else dict_sets_realism[i]
            for i in dict_sets_realism
        ]
        her_metric = [
            np.mean(dict_sets_her)
            if isinstance(dict_sets_her, list)
            else dict_sets_her[i]
            for i in dict_sets_her
        ]

        # Look in timestamp_dir the filenames for the file with the single image metrics
        if mname != "baseline":
            sim_df = pd.read_csv(
                os.path.join(timestamp_dir, f"{mname}_single_metric_eval.csv")
            )
        else:
            sim_df = pd.read_csv(
                os.path.join(timestamp_dir, "single_metric_eval_baseline.csv")
            )

        # Obtain aggregated emtric by set in a dict
        sim_results = []
        for key, val in net_sets_dict.items():
            list_of_files = [file.split("/")[-1] for file in net_sets_dict[key]]

            # Obtain smi dataframe partition with the set files
            partial_df = sim_df[sim_df["Image"].isin(list_of_files)].drop(
                columns="Image"
            )

            sim_results.append(partial_df.mean(axis=0))

        sim_set_df = pd.concat(sim_results, axis=1).T

        sim_set_df["realism_metric"] = realism_metric
        sim_set_df["her_metric"] = her_metric

        sim_sets_model_df.append(sim_set_df)

    sim_set_df = combine_dataframes_with_renamed_index(sim_sets_model_df).reset_index(
        drop=True
    )
    print(sim_set_df)
    sim_results = {}
    for col in sim_set_df.columns:
        if col not in ["realism_metric", "her_metric"]:
            d = {}
            d["Realism"], d["Realism_p_value"] = spearmanr(
                sim_set_df[col], sim_set_df["realism_metric"]
            )
            d["HER"], d["HER_p_value"] = spearmanr(
                sim_set_df[col], sim_set_df["her_metric"]
            )
            sim_results[col] = d

    sim_results = pd.DataFrame(sim_results)

    fig = plot_top_correlated_metrics(sim_set_df, sim_results)

    plt.savefig(
        os.path.join(timestamp_dir, f"{model_to_seek}_zscore_metrics_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    sim_results.to_csv(
        os.path.join(
            timestamp_dir, f"{model_to_seek}_single_metric_by_set_corr_analyses.csv"
        )
    )


def plot_top_correlated_metrics(sim_set_df, sim_results, top_n=3):
    """
    Plot top n z-scored metrics most highly correlated with realism and HER.

    Parameters:
    - sim_set_df: pandas DataFrame containing the metrics for each set
    - sim_results: pandas DataFrame containing correlation results
    - top_n: int, number of top correlated metrics to plot (default: 5)

    Returns:
    - fig: matplotlib figure object
    """
    # Define metrics where lower values are better
    lower_is_better = [
        "DISTS",
        "LPIPS",
        "GMSD",
        "MS-GMSD",
        "MDSI",
        "MAD",
        "NLPD",
        "BRISQUE",
        "ILNIQE",
        "NIQE",
    ]
    # Z-score the metrics
    z_scored_df = sim_set_df.apply(stats.zscore)

    # Invert z-scores for 'lower is better' metrics
    for col in z_scored_df.columns:
        if col in lower_is_better:
            z_scored_df[col] = -z_scored_df[col]

    # Get top n correlated metrics for realism and HER
    top_realism = sim_results.loc["Realism"].nlargest(top_n).index
    top_her = sim_results.loc["HER"].nlargest(top_n).index

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot top realism correlated metrics
    ax1.plot(
        z_scored_df.index,
        z_scored_df["realism_metric"],
        marker="*",
        color="k",
        label=f"Mean Realism",
    )
    for column in top_realism:
        ax1.plot(
            z_scored_df.index,
            z_scored_df[column],
            marker="o",
            label=f"{column} ({sim_results.loc['Realism', column]:.2f})",
        )

    ax1.set_title(f"Top {top_n} Metrics Correlated with Realism")
    ax1.set_ylabel("Z-Score")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot top HER correlated metrics
    ax2.plot(
        z_scored_df.index,
        z_scored_df["her_metric"],
        marker="*",
        color="k",
        label=f"Mean HER",
    )
    for column in top_her:
        ax2.plot(
            z_scored_df.index,
            z_scored_df[column],
            marker="o",
            label=f"{column} ({sim_results.loc['HER', column]:.2f})",
        )

    ax2.set_title(f"Top {top_n} Metrics Correlated with HER")
    ax2.set_xlabel("Set Number")
    ax2.set_ylabel("Z-Score")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent cutting off legends
    plt.tight_layout()

    return fig


def plot_correlation_results(
    correlation_results,
    pvalues,
    timestamp_dir,
    name,
    model_to_seek,
):
    """
    Plots the correlation results for Pearson in multiple heatmaps, including significance markers.
    """

    def format_annotation(val, p_val):
        stars = ""
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"
        return f"{val:.2f} \n {stars}"

    # Create the figure and axes
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(20, 15),
        gridspec_kw={"height_ratios": [1, 0.4], "width_ratios": [0.7, 0.3]},
    )

    # Create annotation matrices for main heatmap
    annot_corr = pd.DataFrame(
        [
            [format_annotation(val, p) for val, p in zip(row, p_row)]
            for row, p_row in zip(correlation_results.values, pvalues.values)
        ],
        index=correlation_results.index,
        columns=correlation_results.columns,
    )

    # Plot for Pearson Correlation
    sns.heatmap(
        correlation_results,
        annot=annot_corr,
        cmap="coolwarm",
        fmt="",
        linewidths=0.5,
        ax=axs[0, 0],
    )
    axs[0, 0].set_title(
        "Pearson: Overall Correlation Heatmap of Networks and Metrics with Realism"
    )

    # Compare normally pretrained and "rad" networks
    rad_networks = correlation_results.index.str.startswith("rad")
    non_rad_networks = ~rad_networks
    avg_corr_rad = correlation_results.loc[rad_networks].mean(axis=0)
    avg_corr_non_rad = correlation_results.loc[non_rad_networks].mean(axis=0)
    avg_corr_network = pd.DataFrame({"rad": avg_corr_rad, "non-rad": avg_corr_non_rad})
    sns.heatmap(
        avg_corr_network,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("Pearson: Average Correlation of Metrics by Network Type")
    axs[0, 1].set_yticklabels(axs[0, 1].get_yticklabels(), rotation=0)

    # Best Metric
    best_metric = correlation_results.mean(axis=0)
    best_metric_df = best_metric.to_frame(name="Correlation").T
    sns.heatmap(
        best_metric_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=axs[1, 0],
    )
    axs[1, 0].set_title(
        "Pearson: Average Correlation of Each Metric Across All Networks"
    )
    axs[1, 0].set_yticklabels(axs[1, 0].get_yticklabels(), rotation=0)

    # Aggregate correlations by base network
    def get_base_network(name):
        if "rad" in name:
            return name.replace("rad_", "")
        elif "med" in name:
            return name.replace("med_", "")
        else:
            return name

    base_network_names = correlation_results.index.map(get_base_network)

    # Aggregated Correlations
    agg_corr_by_base_network_metric = pd.DataFrame(
        index=pd.unique(base_network_names), columns=correlation_results.columns
    )

    for base_network in pd.unique(base_network_names):
        for metric in correlation_results.columns:
            base_correlations = correlation_results.loc[
                base_network_names == base_network, metric
            ]
            if not base_correlations.isna().all():
                avg_corr = base_correlations.mean()
                agg_corr_by_base_network_metric.loc[base_network, metric] = avg_corr
            else:
                agg_corr_by_base_network_metric.loc[base_network, metric] = float("nan")

    agg_corr_by_base_network_metric = agg_corr_by_base_network_metric.apply(
        pd.to_numeric, errors="coerce"
    )
    best_network = agg_corr_by_base_network_metric.mean(axis=1).to_frame(
        name="Correlation"
    )

    sns.heatmap(
        best_network,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=axs[1, 1],
    )
    axs[1, 1].set_title("Pearson: Aggregated Correlation of Base Networks and Metrics")
    axs[1, 1].set_yticklabels(axs[1, 1].get_yticklabels(), rotation=0)

    # Add legend for significance stars
    fig.text(
        0.98,
        0.02,
        "* p < 0.05\n** p < 0.01\n*** p < 0.001",
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            timestamp_dir, f"{model_to_seek}_{name}_corr_turing_quant_pearson.png"
        )
    )

    # Save correlation results
    correlation_results.to_csv(
        os.path.join(
            timestamp_dir, f"{model_to_seek}_{name}_corr_turing_quant_pearson.csv"
        )
    )


def load_model_from_hub(model_repo_id, file_name):
    # Define the mapping of file extensions to loading functions
    def load_pytorch_model(file_path):
        grint(f"Loading PyTorch model from {file_path}...")
        return torch.load(file_path, map_location="cpu")

    def load_tensorflow_model(file_path):
        print(f"Loading TensorFlow model from {file_path}...")
        return tf.keras.models.load_model(file_path)

    def load_graph_def(file_path):
        print(f"Loading TensorFlow GraphDef from {file_path}...")
        with open(file_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

    def load_pickled_model(file_path):
        print(f"Loading Pickle model from {file_path}...")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def load_musiq_model():
        print(f"Loading MUSIQ model './pretrained/MUSIQ/'...")

        return hub.KerasLayer(
            "./pretrained/MUSIQ/", signature="serving_default", output_key="output_0"
        )

    file_ext = os.path.splitext(file_name)[1]

    # Download the file from the Hugging Face Hub
    if not file_name.endswith("MUSIQ"):
        model_path = hf_hub_download(repo_id=model_repo_id, filename=file_name)
    else:

        save_dir = "./pretrained/"
        os.makedirs(save_dir, exist_ok=True)

        hf_hub_download(
            repo_id=model_repo_id,
            filename="MUSIQ/musiq_spaq_ckpt.npz",
            local_dir=save_dir,
        )
        hf_hub_download(
            repo_id=model_repo_id, filename="MUSIQ/saved_model.pb", local_dir=save_dir
        )
        hf_hub_download(
            repo_id=model_repo_id, filename="MUSIQ/archive.tar", local_dir=save_dir
        )
        hf_hub_download(
            repo_id=model_repo_id,
            filename="MUSIQ/variables/variables.data-00000-of-00001",
            local_dir=save_dir,
        )
        hf_hub_download(
            repo_id=model_repo_id,
            filename="MUSIQ/variables/variables.index",
            local_dir=save_dir,
        )

    if file_ext in [".pth", ".pt"]:
        # Load PyTorch model
        return load_pytorch_model(model_path)

    elif file_ext in [".bin", ".pt"]:
        # Load Hugging Face PyTorch model
        print(f"Loading Hugging Face PyTorch model from {model_path}...")
        return AutoModel.from_pretrained(model_repo_id)

    elif file_ext in [".h5", ".pb"]:
        # Load TensorFlow/Keras model or GraphDef
        if file_ext == ".pb":
            # Load TensorFlow GraphDef
            return load_graph_def(model_path)
        else:
            # Load TensorFlow/Keras model
            return load_tensorflow_model(model_path)

    elif file_ext == ".ckpt":
        # Load PyTorch checkpoint
        return load_pytorch_model(model_path)

    elif file_ext == ".pkl":
        # Load Pickle model
        return load_pickled_model(model_path)

    elif file_name.endswith("MUSIQ"):
        # Load MUSIQ model
        return load_musiq_model()

    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def save_features(
    features_dir, network_name, set_name, filenames, features, model_to_seek
):
    # Ensure directories exist
    os.makedirs(features_dir, exist_ok=True)

    # Save features
    np.save(
        os.path.join(features_dir, f"{model_to_seek}_{set_name}_filenames.npy"),
        filenames,
    )
    np.save(
        os.path.join(features_dir, f"{model_to_seek}_{set_name}_features.npy"), features
    )

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


def save_metrics(
    metrics_dir, network_name, all_set_metrics, aggregated_metrics, model_to_seek
):
    # Ensure directories exist
    os.makedirs(metrics_dir, exist_ok=True)

    # Convert numpy values to Python types
    all_set_metrics = numpy_to_python(all_set_metrics)

    # Save detailed metrics to YAML
    yaml_path = os.path.join(
        metrics_dir, f"{model_to_seek}_{network_name}_detailed_metrics.yaml"
    )
    with open(yaml_path, "w") as file:
        yaml.dump(all_set_metrics, file, default_flow_style=False)

    # Save aggregated metrics to CSV (no change needed here)
    csv_path = os.path.join(
        metrics_dir, f"{model_to_seek}_{network_name}_aggregated_metrics.csv"
    )
    df = pd.DataFrame(aggregated_metrics)
    df.to_csv(csv_path, index=False)

    print(
        f"Detailed metrics saved to YAML and aggregated metrics saved to CSV for {network_name}"
    )
