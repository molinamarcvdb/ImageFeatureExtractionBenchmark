import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_comparison(
    diffusion_csv, gan_csv, baseline_csv, output_filename=None
):
    """
    Create a comparison plot of correlation coefficients for diffusion, GAN, and baseline metrics.

    Parameters:
    diffusion_csv: Path to CSV file containing diffusion model correlations
    gan_csv: Path to CSV file containing GAN model correlations
    baseline_csv: Path to CSV file containing baseline (real image) correlations
    output_filename: Optional filename to save the plot
    """
    # Read the CSV files
    diff_df = pd.read_csv(diffusion_csv)
    gan_df = pd.read_csv(gan_csv)
    baseline_df = pd.read_csv(baseline_csv)

    # Create sets of metrics for each type
    diff_metrics = set(diff_df["Metric"])
    gan_metrics = set(gan_df["Metric"])
    baseline_metrics = set(baseline_df["Metric"])

    # Get non-reference metrics (ones that appear in baseline)
    nonref_metrics = baseline_metrics

    # Get reference metrics (ones that only appear in synthetic datasets)
    ref_metrics = diff_metrics - baseline_metrics

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # Function to plot metrics on a given axis
    def plot_metric_set(ax, metrics, title):
        # Filter and sort dataframes for the given metrics
        diff_filtered = diff_df[diff_df["Metric"].isin(metrics)].copy()
        gan_filtered = gan_df[gan_df["Metric"].isin(metrics)].copy()
        baseline_filtered = baseline_df[baseline_df["Metric"].isin(metrics)].copy()

        # Sort by absolute correlation value of diffusion results
        diff_filtered["Abs_Correlation"] = diff_filtered["Kendall Correlation"]
        diff_filtered = diff_filtered.sort_values("Abs_Correlation", ascending=True)

        metrics_order = diff_filtered["Metric"].values
        y_pos = range(len(metrics_order))

        # Plot bars
        ax.barh(
            y_pos,
            diff_filtered["Kendall Correlation"],
            height=0.25,
            label="Diffusion Models",
            color="#2196F3",
            alpha=0.7,
        )

        # Add GAN correlations
        gan_filtered = (
            gan_filtered.set_index("Metric").reindex(metrics_order).reset_index()
        )
        ax.barh(
            [y + 0.25 for y in y_pos],
            gan_filtered["Kendall Correlation"],
            height=0.25,
            label="GANs",
            color="#FF9800",
            alpha=0.7,
        )

        # Add baseline correlations if available
        if not baseline_filtered.empty:
            baseline_filtered = (
                baseline_filtered.set_index("Metric")
                .reindex(metrics_order)
                .reset_index()
            )
            baseline_values = baseline_filtered["Kendall Correlation"].values
            ax.barh(
                [y + 0.5 for y in y_pos],
                baseline_values,
                height=0.25,
                label="Real Images (Baseline)",
                color="#4CAF50",
                alpha=0.7,
            )

        # Customize the plot
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        # Add labels
        ax.set_xlabel("Kendall Correlation Coefficient")
        ax.set_ylabel("Quality Metrics")
        ax.set_title(title, pad=10)

        # Customize y-axis
        ax.set_yticks([y + 0.25 for y in y_pos])
        ax.set_yticklabels(metrics_order)

        # Add legend
        ax.legend(loc="lower right")

        return ax

    # Plot non-reference metrics
    plot_metric_set(
        ax1,
        nonref_metrics,
        "Non-Reference Quality Metrics\nCorrelation with Human Assessment",
    )

    # Plot reference metrics
    plot_metric_set(
        ax2, ref_metrics, "Reference Quality Metrics\nCorrelation with Human Assessment"
    )

    # Adjust layout
    plt.tight_layout()

    # Save if filename provided
    if output_filename:
        plt.savefig(output_filename, dpi=600, bbox_inches="tight")

    return plt


# Example usage:
# Assuming you have your CSV files saved as 'diffusion_correlations.csv' and 'gan_correlations.csv'
plot_correlation_comparison(
    "/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/features/20240930_150033/diff_ground_truth_correlations.csv",
    "/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/features/20240930_150033/gan_ground_truth_correlations.csv",
    "/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/features/20240930_150033/baseline_ground_truth_correlations.csv",
    "/home/ksamamov/GitLab/Notebooks/feat_ext_bench/data/features/20240930_150033/kendall_sim_quality_metrics_correlation.png",
)
