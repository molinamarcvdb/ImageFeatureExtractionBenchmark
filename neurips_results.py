import os
import itertools
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def load_and_process_data(case_dir):
    data = []
    for file in os.listdir(case_dir):
        if file.endswith(".json"):
            with open(os.path.join(case_dir, file), "r") as f:
                results = json.load(f)

            splits = file.split("_")
            pretrain = "RadImageNet" if splits[0].startswith("rad") else "ImageNet"
            model_name = splits[1] if pretrain == "RadImageNet" else splits[0]

            # Extract loss function from filename
            if "ntxent" in file:
                loss_function = "InfoNCE"
            elif "triplet" in file:
                loss_function = "Triplet"
            else:
                loss_function = "Unknown"

            for metric, metric_results in results.items():
                data.append(
                    {
                        "Model": model_name,
                        "Pretrain": pretrain,
                        "Loss": loss_function,
                        "Metric": metric,
                        "Val Detection": metric_results.get("detection_ratio_val", 0),
                    }
                )

    return pd.DataFrame(data)


def plot_aggregated_detection_ratios(df):
    plt.figure(figsize=(16, 10))

    models = df["Model"].unique()
    loss_functions = df["Loss"].unique()
    pretrains = ["ImageNet", "RadImageNet"]
    x = np.arange(len(models))
    width = 0.2

    color_scheme = {
        ("ImageNet", "InfoNCE"): "skyblue",
        ("ImageNet", "Triplet"): "lightblue",
        ("RadImageNet", "InfoNCE"): "salmon",
        ("RadImageNet", "Triplet"): "lightsalmon",
    }

    p_values_pretraining_triplet = []
    p_values_networks = []
    p_values_loss = []
    performance_results = []

    for i, model in enumerate(models):
        for j, pretrain in enumerate(pretrains):
            for k, loss in enumerate(loss_functions):
                data = df[
                    (df["Model"] == model)
                    & (df["Pretrain"] == pretrain)
                    & (df["Loss"] == loss)
                ]
                mean = data["Val Detection"].mean()
                sem = data["Val Detection"].sem()

                performance_results.append(
                    {
                        "Model": model,
                        "Pretrain": pretrain,
                        "Loss": loss,
                        "Mean": mean,
                        "SEM": sem,
                    }
                )

                color = color_scheme[(pretrain, loss)]
                pos = i + (j * 2 + k - 1.5) * width

                plt.bar(
                    pos,
                    mean,
                    width,
                    yerr=sem,
                    capsize=5,
                    label=f"{pretrain} {loss}" if i == 0 else "",
                    color=color,
                    ecolor="black",
                )

                if not np.isnan(mean):
                    plt.text(
                        pos - 0.1,
                        mean,
                        f"{mean:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        rotation=0,
                    )

    plt.xlabel("Model", fontsize=15)
    plt.ylabel("Average Validation Detection Ratio", fontsize=15)
    plt.title("Comparison of backbone, pretraining and contrastive loss", fontsize=20)
    plt.xticks(x, models, rotation=0, ha="right", fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15, loc="upper center")

    plt.tight_layout()
    plt.savefig(
        "validation_detection_ratio_comparison_by_loss.png",
        format="png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Calculate p-values for all network comparisons (using best configuration for each network)
    best_configs = {}
    for model in models:
        model_data = df[df["Model"] == model]
        best_config = model_data.loc[model_data["Val Detection"].idxmax()]
        best_configs[model] = best_config

    for (model1, config1), (model2, config2) in itertools.combinations(
        best_configs.items(), 2
    ):
        _, p_value = stats.ttest_ind(
            df[
                (df["Model"] == model1)
                & (df["Pretrain"] == config1["Pretrain"])
                & (df["Loss"] == config1["Loss"])
            ]["Val Detection"],
            df[
                (df["Model"] == model2)
                & (df["Pretrain"] == config2["Pretrain"])
                & (df["Loss"] == config2["Loss"])
            ]["Val Detection"],
        )
        p_values_networks.append(
            {
                "Model1": f"{model1} ({config1['Pretrain']}, {config1['Loss']})",
                "Model2": f"{model2} ({config2['Pretrain']}, {config2['Loss']})",
                "p-value": p_value,
                "Model1_Mean": config1["Val Detection"],
                "Model2_Mean": config2["Val Detection"],
            }
        )

    # Calculate p-values for Triplet vs InfoNCE
    for model in models:
        for pretrain in ["ImageNet", "RadImageNet"]:
            triplet_data = df[
                (df["Model"] == model)
                & (df["Pretrain"] == pretrain)
                & (df["Loss"] == "Triplet")
            ]["Val Detection"]
            infonce_data = df[
                (df["Model"] == model)
                & (df["Pretrain"] == pretrain)
                & (df["Loss"] == "InfoNCE")
            ]["Val Detection"]

            if not triplet_data.empty and not infonce_data.empty:
                _, p_value = stats.ttest_ind(triplet_data, infonce_data)
                p_values_loss.append(
                    {
                        "Model": model,
                        "Pretrain": pretrain,
                        "Comparison": "Triplet vs InfoNCE",
                        "p-value": p_value,
                        "Triplet_Mean": np.mean(triplet_data),
                        "InfoNCE_Mean": np.mean(infonce_data),
                    }
                )

            imagenet_data = df[
                (df["Model"] == model)
                & (df["Pretrain"] == "ImageNet")
                & (df["Loss"] == "Triplet")
            ]["Val Detection"]
            radimagenet_data = df[
                (df["Model"] == model)
                & (df["Pretrain"] == "RadImageNet")
                & (df["Loss"] == "Triplet")
            ]["Val Detection"]

            if not imagenet_data.empty and not radimagenet_data.empty:
                _, p_value = stats.ttest_ind(imagenet_data, radimagenet_data)
                p_values_pretraining_triplet.append(
                    {
                        "Model": model,
                        "Comparison": "ImageNet vs RadImageNet (Triplet)",
                        "p-value": p_value,
                        "ImageNet_Mean": np.mean(imagenet_data),
                        "RadImageNet_Mean": np.mean(radimagenet_data),
                    }
                )

    p_values_networks = []

    # Find the best configuration for each network
    best_configs = {}
    for model in models:
        model_data = df[df["Model"] == model]
        best_config = (
            model_data.groupby(["Pretrain", "Loss"])["Val Detection"].mean().idxmax()
        )
        best_pretrain, best_loss = best_config
        best_configs[model] = {
            "Pretrain": best_pretrain,
            "Loss": best_loss,
            "Mean": model_data[
                (model_data["Pretrain"] == best_pretrain)
                & (model_data["Loss"] == best_loss)
            ]["Val Detection"].mean(),
        }

    # Perform pairwise comparisons between networks
    for model1, model2 in itertools.combinations(models, 2):
        config1 = best_configs[model1]
        config2 = best_configs[model2]

        data1 = df[
            (df["Model"] == model1)
            & (df["Pretrain"] == config1["Pretrain"])
            & (df["Loss"] == config1["Loss"])
        ]["Val Detection"]
        data2 = df[
            (df["Model"] == model2)
            & (df["Pretrain"] == config2["Pretrain"])
            & (df["Loss"] == config2["Loss"])
        ]["Val Detection"]

        _, p_value = stats.ttest_ind(data1, data2)
        p_values_networks.append(
            {
                "Model1": f"{model1} ({config1['Pretrain']}, {config1['Loss']})",
                "Model2": f"{model2} ({config2['Pretrain']}, {config2['Loss']})",
                "p-value": p_value,
                "Model1_Mean": config1["Mean"],
                "Model2_Mean": config2["Mean"],
            }
        )

    # Save results to CSV
    performance_df = pd.DataFrame(performance_results)
    print("Debug - Performance Results:")
    print(performance_df)
    performance_df.to_csv("performance_results.csv", index=False)

    pd.DataFrame(p_values_pretraining_triplet).to_csv(
        "p_values_pretraining_comparison.csv", index=False
    )
    pd.DataFrame(p_values_networks).to_csv(
        "p_values_network_comparison.csv", index=False
    )
    pd.DataFrame(p_values_loss).to_csv("p_values_loss_comparison.csv", index=False)
    print("Results and p-values saved to CSV files")


def plot_metric_comparison_by_loss(df):
    loss_functions = df["Loss"].unique()

    all_p_values = []
    all_performance_data = []

    for loss in loss_functions:
        plt.figure(figsize=(20, 12))

        # Filter data for the current loss function
        loss_df = df[df["Loss"] == loss]

        # Aggregate data by metric
        metric_data = (
            loss_df.groupby("Metric")["Val Detection"]
            .agg(["mean", "sem"])
            .reset_index()
        )

        # Sort metrics by mean validation detection ratio
        metric_data = metric_data.sort_values("mean", ascending=False)

        # Calculate confidence intervals
        metric_data["ci"] = metric_data["sem"] * stats.t.ppf(
            (1 + 0.95) / 2, len(loss_df) - 1
        )

        # Add performance data to the list
        for _, row in metric_data.iterrows():
            all_performance_data.append(
                {
                    "Loss": loss,
                    "Metric": row["Metric"],
                    "Mean Val Detection": row["mean"],
                    "SEM": row["sem"],
                    "CI": row["ci"],
                }
            )

        # Generate a color map with progressive change
        color_map = plt.cm.get_cmap("Blues")
        colors = color_map(np.linspace(0.3, 1, len(metric_data)))

        # Plot sorted bar chart
        bars = plt.bar(
            metric_data["Metric"],
            metric_data["mean"],
            yerr=metric_data["ci"],
            capsize=5,
            color=colors,
            ecolor="black",
        )

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right", fontsize=15)
        plt.yticks(fontsize=15)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2 - 0.20,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=15,
            )

        plt.xlabel("Metric", fontsize=15)
        plt.ylabel("Average Validation Detection Ratio", fontsize=15)
        plt.title(f"Comparison of Metrics ({loss} Loss)", fontsize=20)

        plt.tight_layout()
        plt.savefig(
            f"metric_comparison_all_networks_{loss.lower()}.png",
            format="png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"Metric comparison plot for {loss} loss saved as metric_comparison_all_networks_{loss.lower()}.png"
        )

        # Calculate p-values for all metric comparisons
        metrics = metric_data["Metric"].tolist()
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                metric1_data = loss_df[loss_df["Metric"] == metrics[i]]["Val Detection"]
                metric2_data = loss_df[loss_df["Metric"] == metrics[j]]["Val Detection"]
                _, p_value = stats.ttest_ind(metric1_data, metric2_data)
                all_p_values.append(
                    {
                        "Loss": loss,
                        "Metric1": metrics[i],
                        "Metric2": metrics[j],
                        "p-value": p_value,
                    }
                )

    # Save p-values to CSV
    pd.DataFrame(all_p_values).to_csv("p_values_metric_comparison.csv", index=False)
    print("P-values for metric comparison saved to p_values_metric_comparison.csv")

    # Save performance data to CSV
    pd.DataFrame(all_performance_data).to_csv(
        "metric_performance_comparison.csv", index=False
    )
    print("Metric performance data saved to metric_performance_comparison.csv")


def output_top_5_combinations(df):
    # Group by Model, Pretrain, and Loss, then calculate mean Val Detection
    grouped = (
        df.groupby(["Model", "Pretrain", "Loss"])["Val Detection"].mean().reset_index()
    )

    # Sort by Val Detection in descending order and get top 5
    top_5 = grouped.sort_values("Val Detection", ascending=False).head(5)

    print("Top 5 Individual Combinations:")
    for i, row in top_5.iterrows():
        print(
            f"{i+1}. Model: {row['Model']}, Pretrain: {row['Pretrain']}, Loss: {row['Loss']}, "
            f"Val Detection: {row['Val Detection']:.4f}"
        )

    # Save to CSV
    top_5.to_csv("top_5_combinations.csv", index=False)
    print("Top 5 combinations saved to 'top_5_combinations.csv'")


def output_top_combinations(df, aggregate_over=None, top_n=5):
    """
    Output top combinations based on maximum Val Detection,
    while tracking which metric yielded the max value.

    Parameters:
    df (DataFrame): The input data
    aggregate_over (list): Columns to aggregate over. If None, default to ['Model', 'Pretrain', 'Loss'].
    top_n (int): Number of top combinations to output
    """
    if aggregate_over is None:
        aggregate_over = ["Model", "Pretrain", "Loss"]

    # Group by specified columns, but also track which metric provided the max value
    def agg_func(sub_df):
        max_row = sub_df.loc[
            sub_df["Val Detection"].idxmax()
        ]  # Get row with max 'Val Detection'
        return pd.Series(
            {
                "max_val_detection": max_row["Val Detection"],
                "metric_with_max": max_row[
                    "Metric"
                ],  # Keep track of the metric that led to the max value
                "mean_val_detection": sub_df["Val Detection"].mean(),
                "std_val_detection": sub_df["Val Detection"].std(),
                "count": sub_df["Val Detection"].count(),
            }
        )

    # Apply aggregation
    grouped = df.groupby(aggregate_over).apply(agg_func).reset_index()

    # Sort by max_val_detection in descending order and get top n
    top_n_combinations = grouped.sort_values("max_val_detection", ascending=False).head(
        top_n
    )

    # Display the top combinations
    print(f"Top {top_n} Combinations (aggregated over {', '.join(aggregate_over)}):")
    for i, row in top_n_combinations.iterrows():
        print(f"{i+1}. ", end="")
        for col in aggregate_over:
            print(f"{col}: {row[col]}, ", end="")
        print(
            f"Max Val Detection: {row['max_val_detection']:.4f}, Metric: {row['metric_with_max']}, "
            f"Mean Val Detection: {row['mean_val_detection']:.4f}, Std: {row['std_val_detection']:.4f}, "
            f"Count: {row['count']}"
        )

    # Save to CSV
    filename = f"top_{top_n}_combinations_{'_'.join(aggregate_over)}.csv"
    top_n_combinations.to_csv(filename, index=False)
    print(f"Top {top_n} combinations saved to '{filename}'")


# Main execution
case_dir = "/home/ksamamov/GitLab/Notebooks/feat_ext_bench/embeddings"
df = load_and_process_data(case_dir)
print("Loaded DataFrame:")
print(df.groupby(["Model", "Pretrain", "Loss"]).size().unstack(fill_value=0))
plot_aggregated_detection_ratios(df)
plot_metric_comparison_by_loss(df)
output_top_combinations(df, ["Model", "Pretrain", "Loss"], 20)
