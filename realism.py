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


def calculate_ema(data, span=15):
    return pd.Series(data).ewm(span=span, adjust=False).mean()


def calculate_error_ci(data, span=15):

    return pd.Series(data).ewm(span=span, adjust=False).std()

    return data.expanding().std()

    return data.rolling(window=span).std()


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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    def plot_data(ax, realism_scores, realism_emas, respondent_errors, stds, title):
        colors = plt.cm.rainbow(np.linspace(0, 1, len(realism_scores)))

        ax_realism = ax
        ax_realism.set_xlabel("Cases")
        ax_realism.set_ylabel("Realism Score")

        ax_error = ax.twinx()
        ax_error.set_ylabel("Cumulative Human Error Rate")

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
            # ax_realism.plot(x, realism, alpha=0.3, color=color, label=f'{realism_col} Raw')
            ax_realism.plot(x, ema, color=color, label=f"{realism_col} EMA")
            ax_realism.fill_between(
                x,
                ema - std_roll,
                ema + std_roll,
                alpha=0.2,
                color=color,
                label=f"{realism_col} std",
            )
            ax_error.plot(
                x,
                error,
                linestyle="--",
                color=color,
                label=f"{error_col} (Overall: {float(overall_error):.2f})",
            )

        ax_realism.set_ylim(0, 100)
        ax_error.set_ylim(0, 1)

        ax.set_title(title)

        # Combine legends
        lines1, labels1 = ax_realism.get_legend_handles_labels()
        lines2, labels2 = ax_error.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2, loc="upper left", fontsize="x-small"
        )

    # Plot data for each subplot
    plot_data(
        ax1,
        synthetic_data[0],
        synthetic_data[1],
        synthetic_data[2],
        synthetic_data[3],
        "All Synthetic Images",
    )
    plot_data(
        ax2, real_data[0], real_data[1], real_data[2], real_data[3], "Real Images"
    )
    plot_data(
        ax3, gan_data[0], gan_data[1], gan_data[2], gan_data[3], "GAN-generated Images"
    )
    plot_data(
        ax4,
        diffusion_data[0],
        diffusion_data[1],
        diffusion_data[2],
        diffusion_data[3],
        "Diffusion Model Images",
    )

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/turing_test_results.png", dpi=300, bbox_inches="tight")
    plt.close()


# Example usage:
if __name__ == "__main__":
    data = [
        {
            0: {
                "user_id": "4e26daae-530d-430b-81be-107704de6a9e",
                "image_path": "https://example.com/image1.png",
                "category": "real_calc",
                "is_real": False,
                "realism_score": 2.5,
                "image_duration": 4.76,
                "index": 39,
                "timestamp": "2024-08-19T09:26:32.421254",
                "local_path": "./data/real/image1.png",
            },
            31: {
                "user_id": "4e26daae-530d-430b-81be-107704de6a9e",
                "image_path": "https://example.com/image2.png",
                "category": "synth_diff_calc",
                "is_real": False,
                "realism_score": 1.0,
                "image_duration": 25.58,
                "index": 1,
                "timestamp": "2024-08-19T08:37:34.299025",
                "local_path": "./data/synthetic/image2.png",
            },
        },
        {
            0: {
                "user_id": "4e26daae-530d-430b-81be-107704de6a9e",
                "image_path": "https://example.com/image1.png",
                "category": "real_calc",
                "is_real": False,
                "realism_score": 3.0,
                "image_duration": 4.76,
                "index": 39,
                "timestamp": "2024-08-19T09:26:32.421254",
                "local_path": "./data/real/image1.png",
            },
            31: {
                "user_id": "4e26daae-530d-430b-81be-107704de6a9e",
                "image_path": "https://example.com/image2.png",
                "category": "synth_diff_calc",
                "is_real": False,
                "realism_score": 0.5,
                "image_duration": 25.58,
                "index": 1,
                "timestamp": "2024-08-19T08:37:34.299025",
                "local_path": "./data/synthetic/image2.png",
            },
        },
    ]

    df = realism_handling(data)
    print(df)
    print(df.columns)
