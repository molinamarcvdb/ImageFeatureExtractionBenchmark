import pandas as pd
import os
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np


def realism_handling(data):
    """
    Processes a dictionary of dictionaries or a list of such dictionaries to
    extract realism scores, compute Z-scores, and calculate averaged Z-scores.

    Parameters:
    data (dict or list): A dictionary where each value is another dictionary
                         containing various metadata, or a list of such dictionaries.

    Returns:
    pd.DataFrame: A DataFrame where the index is the filename, and the columns
                  contain the realism scores for each JSONL file, normalized Z-scores,
                  and averaged Z-scores.
    """

    def get_conf_mat_val(response_real, category):
        if response_real and "real" in category:
            return "TN"
        elif response_real and "synth" in category:
            return "FN"
        elif not response_real and "real" in category:
            return "FP"
        elif not response_real and "synth" in category:
            return "TP"
        else:
            raise f"Something went wrong obtaining the confusion matrix {response_real}, {category}"

    def extract_realism_scores(single_data):
        """Helper function to extract realism scores from a single dict of dicts."""
        realism_scores = {}
        conf_mat_scores = {}
        for key, value in single_data.items():
            # Extract the local_path (filename) and realism_score
            filename = value.get("local_path")
            realism_score = value.get("realism_score")
            conf_matrix_val = get_conf_mat_val(
                bool(value.get("is_real")), value.get("category")
            )
            # Use the filename as the key and realism score as the value
            if filename is not None:
                realism_scores[os.path.basename(filename)] = realism_score
                conf_mat_scores[os.path.basename(filename)] = conf_matrix_val

        return realism_scores, conf_mat_scores

    # Handle multiple JSONL files (list of dict of dicts)
    if isinstance(data, list):
        combined_realism_scores = {}

        for idx, single_data in enumerate(data):
            # Extract realism scores for each JSONL file and store them with a unique column name
            realism_scores, conf_mat_scores = extract_realism_scores(single_data)
            combined_realism_scores[f"Realism Score {idx+1}"] = pd.Series(
                realism_scores
            )
            combined_realism_scores[f"Conf mat score {idx+1}"] = pd.Series(
                conf_mat_scores
            )

        # Create a DataFrame with columns for each JSONL file's realism scores
        df = pd.DataFrame(combined_realism_scores)

        # Compute Z-scores for each JSONL file
        for idx in range(len(data)):
            df[f"Z-Score {idx+1}"] = zscore(
                df[f"Realism Score {idx+1}"], nan_policy="omit"
            )

        # Calculate the averaged Z-score across all JSONL files
        z_score_columns = [f"Z-Score {idx+1}" for idx in range(len(data))]
        df["Averaged Z-Score"] = df[z_score_columns].mean(axis=1)

        conf_mat_columns = [f"Conf mat score {idx+1}" for idx in range(len(data))]
        df["AggConfMatScore"] = df[conf_mat_columns].apply(
            lambda row: row.dropna().tolist(), axis=1
        )

        # Calculate the averaged Z-score across all JSONL files
        realism_score_columns = [f"Realism Score {idx+1}" for idx in range(len(data))]
        df["Averaged Realism Score"] = df[realism_score_columns].mean(axis=1)

    # Handle a single JSONL file (dict of dicts)
    elif isinstance(data, dict):
        realism_scores = extract_realism_scores(data)
        df = pd.DataFrame.from_dict(
            realism_scores, orient="index", columns=["Realism Score"]
        )

        # Compute the Z-score for the single JSONL file
        df["Z-Score"] = zscore(df["Realism Score"], nan_policy="omit")

    else:
        raise ValueError("Input data must be either a dict or a list of dicts.")

    return df


from matplotlib.colors import LinearSegmentedColormap


def calculate_ema(data, span=15):
    return pd.Series(data).ewm(span=span, adjust=False).mean()


def calculate_error_ci(data, span=15):
    return pd.Series(data).ewm(span=span, adjust=False).std()


def visualize_turing_tests(df, output_dir):
    # Separate synthetic and real images
    synthetic_df = df[
        df["AggConfMatScore"].apply(lambda x: any(score in ["TP", "FN"] for score in x))
    ]
    real_df = df[
        df["AggConfMatScore"].apply(lambda x: all(score in ["TN", "FP"] for score in x))
    ]

    # Further separate synthetic images into GAN and diffusion model
    gan_df = synthetic_df[synthetic_df.index.str.startswith("seed")]
    diffusion_df = synthetic_df[~synthetic_df.index.str.startswith("seed")]

    def calculate_cumulative_error(group_df, is_synthetic):
        respondent_errors = {}
        for col in [
            col for col in group_df.columns if col.startswith("Conf mat score")
        ]:
            correct, incorrect = 0, 0
            error_rates = []
            for score in group_df[col]:
                if is_synthetic and score == "TP":
                    correct += 1
                elif is_synthetic and score == "FN":
                    incorrect += 1
                elif not is_synthetic and score == "TN":
                    correct += 1
                elif not is_synthetic and score == "FP":
                    incorrect += 1
                total = correct + incorrect
                error_rate = incorrect / total if total > 0 else 0
                error_rates.append(error_rate)
            overall_error = (
                incorrect / (correct + incorrect) if (correct + incorrect) > 0 else 0
            )
            respondent_errors[col] = (error_rates, overall_error)
        return respondent_errors

    def process_data(df, is_synthetic):
        respondent_errors = calculate_cumulative_error(df, is_synthetic)
        realism_scores = {}
        realism_emas = {}
        stds = {}
        for col in [col for col in df.columns if col.startswith("Realism Score")]:
            realism_scores[col] = df[col]
            realism_emas[col] = calculate_ema(df[col])
            stds[col] = calculate_error_ci(df[col])

        return realism_scores, realism_emas, respondent_errors, stds

    synthetic_data = process_data(synthetic_df, True)
    real_data = process_data(real_df, False)
    gan_data = process_data(gan_df, True)
    diffusion_data = process_data(diffusion_df, True)

    # Create the plot with four subplots in a 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))

    # Create a custom colormap for Turing test responses with stronger colors
    colors = ["#0000FF", "#FF8000"]
    n_bins = 4
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    def plot_data(
        ax, realism_scores, realism_emas, respondent_errors, stds, title, is_synthetic
    ):
        num_respondents = len(realism_scores)
        colors = cmap(np.linspace(0, 1, num_respondents))

        ax_realism = ax
        ax_realism.set_xlabel("Cases")
        ax_realism.set_ylabel("Realism Score")

        ax_error = ax.twinx()
        ax_error.set_ylabel("Human Error Rate")

        overall_errors = []

        for (
            (realism_col, realism),
            (ema_col, ema),
            (error_col, (error, overall_error)),
            (std_col, std_roll),
            color,
        ) in zip(
            realism_scores.items(),
            realism_emas.items(),
            respondent_errors.items(),
            stds.items(),
            colors,
        ):
            x = range(len(realism))
            ax_realism.plot(
                x, ema, color=color, label=f"{realism_col} (Error: {overall_error:.2f})"
            )
            ax_realism.fill_between(
                x, ema - std_roll, ema + std_roll, alpha=0.2, color=color
            )
            ax_error.plot(x, error, linestyle="--", color=color)
            overall_errors.append(overall_error)

        ax_realism.set_ylim(0, 100)
        ax_error.set_ylim(0, 1)

        ax.set_title(title)

        # Add mini-legend with overall error rates
        overall_error_text = f"Mean Error: {np.mean(overall_errors):.2f}"
        # ax.text(0.05, 0.95, overall_error_text, transform=ax.transAxes,
        #        verticalalignment='top', fontsize=10,
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add legend for individual lines
        ax_realism.legend(loc="upper left", fontsize=13)

        return (
            ax_realism.get_lines()[0],
            ax_error.get_lines()[0],
        )  # Return one line from each axis for the main legend

    # Plot data for each subplot and collect legend handles
    handle1_realism, handle1_error = plot_data(
        ax1,
        synthetic_data[0],
        synthetic_data[1],
        synthetic_data[2],
        synthetic_data[3],
        "All Synthetic Images",
        True,
    )
    handle2_realism, handle2_error = plot_data(
        ax2,
        real_data[0],
        real_data[1],
        real_data[2],
        real_data[3],
        "Real Images",
        False,
    )
    handle3_realism, handle3_error = plot_data(
        ax3,
        gan_data[0],
        gan_data[1],
        gan_data[2],
        gan_data[3],
        "GAN-generated Images",
        True,
    )
    handle4_realism, handle4_error = plot_data(
        ax4,
        diffusion_data[0],
        diffusion_data[1],
        diffusion_data[2],
        diffusion_data[3],
        "Diffusion Model Images",
        True,
    )

    # Create a single legend outside the subplots for realism score and error rate
    fig.legend(
        [handle1_realism, handle1_error],
        ["Realism Score (EMA)", "Human Error Rate"],
        loc="center left",
        bbox_to_anchor=(0.9, 1.005),
        fontsize=14,
    )

    # Create color legend for Turing test responses
    response_labels = [
        "True Positive (TP)",
        "False Negative (FN)",
        "True Negative (TN)",
        "False Positive (FP)",
    ]
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=color, ec="none") for color in colors
    ]
    # fig.legend(legend_elements, response_labels, loc='center left', bbox_to_anchor=(1, 0.3),
    #               title="Turing Test Responses", fontsize='small')

    # Adjust layout and display the plot
    fig.suptitle("Realism and Human Error Rate Progression", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(right=0.98)  # Make room for the legends
    plt.savefig(f"{output_dir}/turing_test_results.png", dpi=300, bbox_inches="tight")
    plt.close()


# Examimport json
from itertools import combinations
from typing import List
import numpy as np
from sklearn.metrics import cohen_kappa_score
import json
import urllib.parse


def calculate_inter_rater_agreement(
    file_paths: List[str], debug: bool = False
) -> float:
    """
    Calculate the average inter-rater agreement for 'is_real' classification
    across multiple JSONL files containing Turing test results.

    Args:
    file_paths (List[str]): List of paths to JSONL files containing Turing test results.
    debug (bool): If True, print debugging information.

    Returns:
    float: Average Cohen's Kappa score across all pairs of raters.
    """

    def get_image_id(url):
        # Parse the URL and remove the query string (SAS token)
        parsed_url = urllib.parse.urlparse(url)
        return urllib.parse.urlunparse(parsed_url._replace(query=""))

    # Load data from all files
    all_ratings = {}
    image_ids = set()

    for file_path in file_paths:
        if debug:
            print(f"Processing file: {file_path}")

        with open(file_path, "r") as f:
            ratings = {}
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    image_id = get_image_id(data["image_path"])
                    is_real = int(
                        data["is_real"]
                    )  # Convert boolean to int for easier calculation
                    ratings[image_id] = is_real
                    image_ids.add(image_id)
                    if debug and i < 5:  # Print first 5 entries for debugging
                        print(f"  Entry {i}: Image ID: {image_id}, Is Real: {is_real}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {i+1} in file {file_path}")
                except KeyError as e:
                    print(f"Missing key {e} on line {i+1} in file {file_path}")

        all_ratings[file_path] = ratings
        if debug:
            print(f"  Processed {len(ratings)} entries from {file_path}")

    # Ensure all raters have rated all images
    for rater, ratings in all_ratings.items():
        missing_images = image_ids - set(ratings.keys())
        if missing_images:
            raise ValueError(
                f"Rater {rater} is missing ratings for {len(missing_images)} images"
            )

    # Calculate Cohen's Kappa for each pair of raters
    kappa_scores = []
    for (rater1, ratings1), (rater2, ratings2) in combinations(all_ratings.items(), 2):
        y1 = [ratings1[img] for img in image_ids]
        y2 = [ratings2[img] for img in image_ids]
        kappa = cohen_kappa_score(y1, y2)
        kappa_scores.append(kappa)
        if debug:
            print(f"Kappa score between {rater1} and {rater2}: {kappa:.4f}")

    # Calculate and return the average Kappa score
    return np.mean(kappa_scores)


if __name__ == "__main__":
    file_paths = [
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/Turing/evaluations_ed33ee11-c112-4a5c-8dd6-626b978e7e8d_Nadine_Benz.jsonl",
        "/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/turing_tests/evaluations_e15ce52f-ab76-45f2-9b94-8a7e140c3bbb_DANIELA_RAMIREZ.jsonl",
    ]

    average_kappa = calculate_inter_rater_agreement(file_paths)
    print(f"Average Inter-rater Agreement (Cohen's Kappa): {average_kappa:.4f}")
