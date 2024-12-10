import os
import json
import time
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import asdict


class ResultSaver:
    def __init__(self, base_dir: str = "benchmark_results"):
        """Initialize result saver with base directory"""
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / f"run_{self.timestamp}"
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories"""
        # Create main directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.run_dir / "plots", exist_ok=True)
        os.makedirs(self.run_dir / "data", exist_ok=True)
        os.makedirs(self.run_dir / "metrics", exist_ok=True)

    def save_config(self, config):
        """Save configuration"""
        config_dict = asdict(config)
        # Convert numpy arrays to lists for JSON serialization
        config_dict["severity_levels"] = config_dict["severity_levels"].tolist()

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)

    def save_stats(self, stats):
        """Save statistical results"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_stats = self._make_serializable(stats)

        with open(self.run_dir / "data" / "statistics.json", "w") as f:
            json.dump(serializable_stats, f, indent=4)

    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable types to Python native types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        return obj

    def save_plot(self, fig, name):
        """Save plot in multiple formats"""
        if isinstance(fig, plt.Figure):
            fig.savefig(
                self.run_dir / "plots" / f"{name}.png", dpi=300, bbox_inches="tight"
            )
            fig.savefig(self.run_dir / "plots" / f"{name}.pdf", bbox_inches="tight")
            plt.close(fig)
        else:
            # If fig is plt module (from plt.show()), save current figure
            plt.savefig(
                self.run_dir / "plots" / f"{name}.png", dpi=300, bbox_inches="tight"
            )
            plt.savefig(self.run_dir / "plots" / f"{name}.pdf", bbox_inches="tight")
            plt.close()

    def save_raw_results(self, results, name):
        """Save raw results using pickle"""
        with open(self.run_dir / "data" / f"{name}.pkl", "wb") as f:
            pickle.dump(results, f)

    def save_metrics(self, metrics_dict, name):
        """Save individual metrics results"""
        with open(self.run_dir / "metrics" / f"{name}.json", "w") as f:
            json.dump(self._make_serializable(metrics_dict), f, indent=4)


import numpy as np
import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


@dataclass
class BenchmarkConfig:
    batch_size: int = 50
    n_runs: int = 1
    severity_levels: np.ndarray = np.linspace(0, 0.9, 5)
    feature_dim: int = 64
    save_results: bool = True

    # Degradation configs
    n_clusters_base: int = 10
    max_noise_std: float = 0.1
    k_neighbors: int = 50
    semantic_distance_threshold: float = 0.5


class ImprovedDiversityBenchmark:
    def __init__(
        self,
        features: Union[torch.Tensor, np.ndarray],
        calculate_metrics_fn,
        config: BenchmarkConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.device = device
        self.features = self._prepare_features(features)
        self.calculate_metrics_fn = calculate_metrics_fn
        self.results_cache = {}

        self.saver = ResultSaver()

        # Dictionary to track whether metrics should be inverted (lower is better)
        self.invert_metrics = {
            "fid": True,
            "fid_inf": True,
            "kid_mean": True,
            "mmd": True,
            "sw": True,
            "precision": False,
            "recall": False,
            "density": False,
            "coverage": False,
            "is": False,
            "vendi": False,
            "authpct": False,
        }

    def _prepare_features(self, features):
        """Convert features to appropriate format"""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        return features.to(self.device).float()

    def calculate_feature_similarity(self, features):
        """Calculate semantic similarity between features using cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity

        return cosine_similarity(features)

    def get_valid_pairs(self, similarities, threshold):
        """Return pairs with similarity above threshold"""
        valid_pairs = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i, j] > threshold:
                    valid_pairs.append((i, j))
        return valid_pairs

    def adjust_interpolation_severity(self, severity, semantic_distance):
        """Scale severity based on semantic distance"""
        return severity * (1 - semantic_distance)

    def mode_collapse_degradation(self, severity_levels):
        """Simulate mode collapse by moving features towards cluster centers"""
        results = []
        features = self.features.cpu().numpy()

        for severity in severity_levels:
            # Number of clusters decreases as severity increases
            n_clusters = max(2, int((1 - severity) * self.config.n_clusters_base))

            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(features)
            centers = kmeans.cluster_centers_

            # Move features towards their cluster centers
            degraded = features.copy()
            for i in range(n_clusters):
                mask = clusters == i
                if not np.any(mask):
                    continue
                degraded[mask] = (1 - severity) * features[mask] + severity * centers[i]

            # Calculate metrics
            metrics = self.calculate_metrics_fn(
                ref_features=features, sample_features=degraded, device=self.device
            )

            results.append(
                {
                    "degradation_type": "mode_collapse",
                    "severity": severity,
                    "metrics": metrics,
                }
            )

        return results

    def save_all_results(self, stats, single_run_results):
        """Save all benchmark results"""
        # Save configuration
        self.saver.save_config(self.config)

        # Save statistics
        self.saver.save_stats(stats)

        # Save raw results
        self.saver.save_raw_results(single_run_results, "single_run_results")

        # Save plots with aggregated metrics
        fig = self.plot_statistical_results(stats)
        self.saver.save_plot(fig, "aggregated_metrics_by_degradation")

        # Correlation matrices for each degradation type
        fig = self.plot_correlation_matrix(single_run_results)
        self.saver.save_plot(fig, "correlation_matrices")

        # Save individual degradation results
        for result in single_run_results:
            deg_type = result["degradation_type"]
            severity = result["severity"]
            self.saver.save_metrics(
                result["metrics"], f"{deg_type}_severity_{severity:.2f}"
            )

        print(f"Results saved in: {self.saver.run_dir}")

    def interpolation_degradation(self, severity_levels):
        """Create interpolated features between pairs, handling odd-sized arrays"""
        results = []
        features = self.features.cpu().numpy()
        n_samples = len(features)

        for severity in severity_levels:
            # Create a copy for degradation
            degraded = features.copy()

            # Generate random permutation
            indices = np.random.permutation(n_samples)

            # Handle odd number of samples by excluding last element if necessary
            n_pairs = n_samples // 2
            pairs = indices[: n_pairs * 2].reshape(-1, 2)

            # Perform interpolation for pairs
            for i, j in pairs:
                degraded[j] = (1 - severity) * features[j] + severity * features[i]

            # Handle leftover sample if array length is odd
            if n_samples % 2 == 1:
                last_idx = indices[-1]
                # Interpolate with random sample from paired ones
                random_pair_idx = np.random.choice(indices[:-1])
                degraded[last_idx] = (1 - severity) * features[
                    last_idx
                ] + severity * features[random_pair_idx]

            metrics = self.calculate_metrics_fn(
                ref_features=features, sample_features=degraded, device=self.device
            )

            results.append(
                {
                    "degradation_type": "interpolation",
                    "severity": severity,
                    "metrics": metrics,
                    "n_pairs": n_pairs,
                    "odd_sample_handled": n_samples % 2 == 1,
                }
            )

        return results

    def semantic_interpolation_degradation(self, severity_levels, k_neighbors=5):
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        import torch
        from typing import List

        results = []
        features = self.features.cpu().numpy()
        n_samples = len(features)

        print(
            f"Starting semantic interpolation with {n_samples} samples and k={k_neighbors} neighbors"
        )

        t0 = time.time()
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine").fit(features)
        distances, indices = nbrs.kneighbors(features)
        print(f"Initial kNN fit & search took {time.time()-t0:.2f}s")
        print(
            f"Average neighbor distance: {distances.mean():.4f} ± {distances.std():.4f}"
        )

        for severity in severity_levels:
            print(f"\nProcessing severity {severity:.2f}")
            t0 = time.time()
            degraded = features.copy()
            pairs = []
            used_samples = set()

            # Create pairs based on semantic similarity
            t1 = time.time()
            for i in range(n_samples):
                if i in used_samples:
                    continue
                for neighbor_idx in range(k_neighbors):
                    potential_pair = indices[i, neighbor_idx]
                    if potential_pair not in used_samples and potential_pair != i:
                        pairs.append((i, potential_pair))
                        used_samples.add(i)
                        used_samples.add(potential_pair)
                        break
            print(f"Pair creation took {time.time()-t1:.2f}s")

            unpaired = set(range(n_samples)) - used_samples
            print(f"Created {len(pairs)} pairs, {len(unpaired)} unpaired samples")

            if unpaired:
                t1 = time.time()
                print(f"Processing {len(unpaired)} unpaired samples")
                unpaired_features = features[list(unpaired)].reshape(len(unpaired), -1)
                paired_features = features[list(used_samples)].reshape(
                    len(used_samples), -1
                )
                print(f"Feature reshaping took {time.time()-t1:.2f}s")

                t1 = time.time()
                nbrs_unpaired = NearestNeighbors(n_neighbors=1, metric="cosine").fit(
                    paired_features
                )
                distances_unpaired, indices_unpaired = nbrs_unpaired.kneighbors(
                    unpaired_features
                )
                print(f"Unpaired kNN search took {time.time()-t1:.2f}s")

                t1 = time.time()
                for idx, (dist, paired_idx) in enumerate(
                    zip(distances_unpaired, indices_unpaired)
                ):
                    original_idx = list(unpaired)[idx]
                    closest_paired_idx = list(used_samples)[paired_idx[0]]
                    adjusted_severity = severity * (1 - dist[0])
                    degraded[original_idx] = (1 - adjusted_severity) * features[
                        original_idx
                    ] + adjusted_severity * features[closest_paired_idx]
                print(f"Unpaired interpolation took {time.time()-t1:.2f}s")

            t1 = time.time()
            metrics = self.calculate_metrics_fn(
                ref_features=features, sample_features=degraded, device=self.device
            )
            print(f"Metrics calculation took {time.time()-t1:.2f}s")

            print(f"Total severity iteration took {time.time()-t0:.2f}s")

            results.append(
                {
                    "degradation_type": "semantic_interpolation",
                    "severity": severity,
                    "metrics": metrics,
                    "n_pairs": len(pairs),
                    "n_unpaired": len(unpaired),
                    "semantic_distances": distances.mean(axis=1).tolist(),
                }
            )

        return results

    def noise_degradation(self, severity_levels):
        """Add increasing levels of Gaussian noise"""
        results = []
        features = self.features.cpu().numpy()

        for severity in severity_levels:
            # Add noise proportional to feature standard deviation
            feature_std = np.std(features, axis=0, keepdims=True)
            noise = (
                np.random.randn(*features.shape)
                * feature_std
                * severity
                * self.config.max_noise_std
            )
            degraded = features + noise

            metrics = self.calculate_metrics_fn(
                ref_features=features, sample_features=degraded, device=self.device
            )

            results.append(
                {"degradation_type": "noise", "severity": severity, "metrics": metrics}
            )

        return results

    def sample_repetition_degradation(self, severity_levels):
        """
        Reduce diversity by repeating samples with increasing severity.
        Higher severity means more repetition of fewer unique samples.

        Args:
            severity_levels: List of severity values between 0 and 1
        """
        results = []
        features = self.features.cpu().numpy()
        n_samples = len(features)

        for severity in severity_levels:
            # Calculate number of unique samples to keep
            # At severity 0: keep all samples
            # At severity 1: keep only 10% of samples
            n_unique = max(int(n_samples * (1 - 0.9 * severity)), 1)

            # Randomly select unique samples
            unique_indices = np.random.choice(n_samples, size=n_unique, replace=False)
            unique_features = features[unique_indices]

            # Create degraded set by repeating unique samples
            repeats = n_samples // n_unique
            remainder = n_samples % n_unique

            # Create the degraded set
            degraded = np.concatenate(
                [
                    # Repeat samples evenly
                    np.repeat(unique_features, repeats, axis=0),
                    # Add remaining samples to reach original size
                    unique_features[:remainder],
                ]
            )

            # Shuffle the degraded set
            np.random.shuffle(degraded)

            # Calculate metrics
            metrics = self.calculate_metrics_fn(
                ref_features=features, sample_features=degraded, device=self.device
            )

            # Add diversity statistics
            metrics.update(
                {
                    "unique_samples": n_unique,
                    "unique_ratio": n_unique / n_samples,
                    "average_repetition": n_samples / n_unique,
                }
            )

            results.append(
                {
                    "degradation_type": "sample_repetition",
                    "severity": severity,
                    "metrics": metrics,
                }
            )

        return results

    def evaluate_all_degradations(self):
        """Run all degradation types"""
        all_results = []
        all_results.extend(self.mode_collapse_degradation(self.config.severity_levels))
        all_results.extend(self.interpolation_degradation(self.config.severity_levels))
        all_results.extend(self.noise_degradation(self.config.severity_levels))
        all_results.extend(
            self.sample_repetition_degradation(self.config.severity_levels)
        )
        all_results.extend(
            self.semantic_interpolation_degradation(self.config.severity_levels)
        )

        return all_results

    def run_statistical_evaluation(self):
        """Run multiple evaluations for statistical significance"""
        all_results = []
        for _ in range(self.config.n_runs):
            results = self.evaluate_all_degradations()
            all_results.append(results)

        return self._compute_statistics(all_results)

    def _compute_statistics(self, all_results):
        """Compute mean and std of metrics across runs"""
        stats = {}
        # Add sample_repetition to degradation types
        degradation_types = [
            "mode_collapse",
            "interpolation",
            "semantic_interpolation",
            "noise",
            "sample_repetition",
        ]
        degradation_titles = [
            "Mode Collapse",
            "Interpolation",
            "Semantic Interpolation",
            "Noise",
            "Sample Repetition",
        ]

        for deg_type in degradation_types:
            stats[deg_type] = {"mean": {}, "std": {}, "confidence": {}}

            # Compute statistics for each metric
            # Get metrics from first result that matches this degradation type
            first_result = next(
                r
                for run in all_results
                for r in run
                if r["degradation_type"] == deg_type
            )
            metrics = first_result["metrics"].keys()

            for metric in metrics:
                values = self._extract_metric_values(all_results, deg_type, metric)
                if len(values) > 0:  # Check if we have values for this metric
                    stats[deg_type]["mean"][metric] = np.mean(values, axis=0)
                    stats[deg_type]["std"][metric] = np.std(values, axis=0)

        return stats

    def _extract_metric_values(self, all_results, deg_type, metric):
        """Extract values for a specific metric and degradation type"""
        values = []
        for run_results in all_results:
            run_values = [
                float(r["metrics"][metric])  # Convert to float here
                for r in run_results
                if r["degradation_type"] == deg_type
            ]
            values.append(run_values)
        return np.array(values)

    def plot_statistical_results(self, stats, metrics_to_plot=None):
        """Plot all metrics in a single plot for each degradation type"""
        if metrics_to_plot is None:
            metrics_to_plot = list(stats["mode_collapse"]["mean"].keys())

        degradation_types = [
            "mode_collapse",
            "interpolation",
            "semantic_interpolation",
            "noise",
            "sample_repetition",
        ]
        degradation_titles = [
            "Mode Collapse",
            "Interpolation",
            "Semantic Interpolation",
            "Noise",
            "Sample Repetition",
        ]

        # Create subplots
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))

        colors = plt.cm.tab20(np.linspace(0, 1, len(metrics_to_plot)))
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

        for deg_idx, (deg_type, deg_title) in enumerate(
            zip(degradation_types, degradation_titles)
        ):
            if deg_type not in stats:
                continue

            ax = axes[deg_idx]

            for metric_idx, metric in enumerate(metrics_to_plot):
                if metric not in stats[deg_type]["mean"]:
                    continue

                mean = stats[deg_type]["mean"][metric]
                std = stats[deg_type]["std"][metric]

                if self.invert_metrics.get(metric, False):
                    mean = -mean

                # Normalize mean to [0,1] range
                mean_min, mean_max = np.min(mean), np.max(mean)
                if mean_min != mean_max:
                    mean_norm = (mean - mean_min) / (mean_max - mean_min)
                    std_norm = std / (mean_max - mean_min)
                else:
                    mean_norm = np.zeros_like(mean)
                    std_norm = std

                color = colors[metric_idx]
                marker = markers[metric_idx % len(markers)]

                ax.plot(
                    self.config.severity_levels,
                    mean_norm,
                    label=metric.upper(),
                    color=color,
                    marker=marker,
                    markersize=8,
                    linewidth=2,
                )

                ax.fill_between(
                    self.config.severity_levels,
                    mean_norm - std_norm,
                    mean_norm + std_norm,
                    color=color,
                    alpha=0.2,
                )

            # In plot_statistical_results:
            if deg_type == "semantic_interpolation":
                ax2 = ax.twinx()
                # Calculate average semantic distance for each severity level
                semantic_distances = []
                for severity in self.config.severity_levels:
                    severity_results = [
                        r
                        for r in self.results_cache.get(deg_type, [])
                        if r["severity"] == severity
                    ]
                    if severity_results:
                        avg_distance = np.mean(
                            [np.mean(r["semantic_distances"]) for r in severity_results]
                        )
                        semantic_distances.append(avg_distance)

                if semantic_distances:
                    ax2.plot(
                        self.config.severity_levels,
                        semantic_distances,
                        "--k",
                        label="Mean Semantic Distance",
                    )
                    ax2.set_ylabel("Mean Semantic Distance")
                    ax2.legend(loc="upper right")

            ax.set_title(
                f"{deg_title}\nMetric Responses (Normalized)", fontsize=12, pad=10
            )
            ax.set_xlabel("Degradation Severity", fontsize=10)
            ax.set_ylabel("Normalized Score", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_ylim(-0.1, 1.1)

            if deg_idx == len(degradation_types) - 1:
                ax.legend(
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                    fontsize=10,
                )

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, results: List[Dict]):
        """Plot correlation matrix for each degradation type including semantic metrics"""
        degradation_results = {}
        for result in results:
            deg_type = result["degradation_type"]
            if deg_type not in degradation_results:
                degradation_results[deg_type] = []
            degradation_results[deg_type].append(result)

        n_types = len(degradation_results)
        fig, axes = plt.subplots(1, n_types, figsize=(10 * n_types, 10))
        if n_types == 1:
            axes = [axes]

        for idx, (deg_type, deg_results) in enumerate(degradation_results.items()):
            metrics = list(deg_results[0]["metrics"].keys())
            values_dict = {metric: [] for metric in metrics}
            values_dict["severity"] = []

            if deg_type == "semantic_interpolation":
                values_dict["semantic_distance"] = []

            for result in deg_results:
                for metric in metrics:
                    value = result["metrics"][metric]
                    if self.invert_metrics.get(metric, False):
                        value = -value
                    values_dict[metric].append(value)
                values_dict["severity"].append(result["severity"])

                if deg_type == "semantic_interpolation":
                    values_dict["semantic_distance"].append(
                        np.mean(result["semantic_distances"])
                    )

            metrics_with_severity = ["severity"]
            if deg_type == "semantic_interpolation":
                metrics_with_severity.append("semantic_distance")
            metrics_with_severity.extend(metrics)

            values_array = np.array([values_dict[m] for m in metrics_with_severity])

            n = len(metrics_with_severity)
            corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    corr_matrix[i, j] = self._correlation_with_constants(
                        values_array[i], values_array[j]
                    )

            ax = axes[idx]
            sns.heatmap(
                corr_matrix,
                xticklabels=metrics_with_severity,
                yticklabels=metrics_with_severity,
                annot=True,
                fmt=".2f",
                cmap="RdBu",
                vmin=-1,
                vmax=1,
                center=0,
                ax=ax,
                square=True,
            )

            ax.set_title(f'{deg_type.replace("_", " ").title()}\nCorrelation Matrix')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        return fig

    def _correlation_with_constants(self, x, y):
        """Calculate correlation handling constant arrays"""
        x_std = np.std(x)
        y_std = np.std(y)

        if x_std == 0 or y_std == 0:
            return 0

        return np.corrcoef(x, y)[0, 1]


import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple


class MetricAnalyzer:
    def __init__(self, stats: Dict, severity_levels: np.ndarray):
        self.stats = stats
        self.severity_levels = severity_levels

    def analyze_metric_behavior(self, metric: str, degradation_type: str) -> Dict:
        """Analyze individual metric behavior for a specific degradation"""
        mean_values = self.stats[degradation_type]["mean"][metric]
        std_values = self.stats[degradation_type]["std"][metric]

        # Calculate key characteristics
        monotonicity = self._calculate_monotonicity(mean_values)
        sensitivity = self._calculate_sensitivity(mean_values)
        stability = 1 - np.mean(std_values) / (
            np.max(mean_values) - np.min(mean_values)
        )
        response_linearity = self._calculate_linearity(mean_values)

        return {
            "monotonicity": float(monotonicity),
            "sensitivity": float(sensitivity),
            "stability": float(stability),
            "response_linearity": float(response_linearity),
        }

    def _calculate_monotonicity(self, values: np.ndarray) -> float:
        """Calculate how consistently the metric changes with severity"""
        differences = np.diff(values)
        consistent_direction = np.sum(
            np.sign(differences[0]) == np.sign(differences)
        ) / len(differences)
        return consistent_direction

    def _calculate_sensitivity(self, values: np.ndarray) -> float:
        """Calculate the overall range of the metric's response"""
        return (np.max(values) - np.min(values)) / np.mean(values)

    def _calculate_linearity(self, values: np.ndarray) -> float:
        """Calculate how linear the metric's response is"""
        linear_fit = np.polyfit(self.severity_levels, values, 1)
        predicted = np.polyval(linear_fit, self.severity_levels)
        r2 = 1 - np.sum((values - predicted) ** 2) / np.sum(
            (values - np.mean(values)) ** 2
        )
        return r2

    def get_best_metrics(self, min_monotonicity=0.7, min_stability=0.7) -> Dict:
        """Identify best metrics for each degradation type"""
        recommendations = {}

        degradation_criteria = {
            "mode_collapse": {
                "description": "Strong monotonic decrease with clustering, high sensitivity to mode reduction",
                "priority": ["monotonicity", "sensitivity", "stability"],
            },
            "interpolation": {
                "description": "Moderate response to interpolation, high stability",
                "priority": ["stability", "sensitivity", "monotonicity"],
            },
            "noise": {
                "description": "High stability at low severity, sharp response at high severity",
                "priority": ["stability", "sensitivity", "response_linearity"],
            },
            "sample_repetition": {
                "description": "Linear response to unique sample reduction, high monotonicity",
                "priority": ["response_linearity", "monotonicity", "stability"],
            },
            "semantic_interpolation": {
                "description": "Semantic-aware interpolation response, high stability",
                "priority": ["stability", "monotonicity", "sensitivity"],
            },
        }

        for deg_type in self.stats.keys():
            metrics_analysis = {}
            for metric in self.stats[deg_type]["mean"].keys():
                behavior = self.analyze_metric_behavior(metric, deg_type)
                if (
                    behavior["monotonicity"] >= min_monotonicity
                    and behavior["stability"] >= min_stability
                ):
                    metrics_analysis[metric] = behavior

            # Sort metrics based on degradation-specific criteria
            criteria = degradation_criteria[deg_type]
            sorted_metrics = self._rank_metrics(metrics_analysis, criteria["priority"])

            recommendations[deg_type] = {
                "description": criteria["description"],
                "recommended_metrics": sorted_metrics,
                "analysis": metrics_analysis,
            }

        return recommendations

    def _rank_metrics(
        self, metrics_analysis: Dict, priority_order: List[str]
    ) -> List[Tuple[str, float]]:
        """Rank metrics based on specified criteria priority"""
        metric_scores = {}
        for metric, behavior in metrics_analysis.items():
            score = sum(
                behavior[criterion] * (len(priority_order) - i)
                for i, criterion in enumerate(priority_order)
            )
            metric_scores[metric] = score

        return sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)

    def save_recommendations(self, output_path: str = "metric_recommendations.json"):
        """Save metric recommendations to JSON"""
        recommendations = self.get_best_metrics()

        # Add metadata
        output_data = {
            "metadata": {
                "analysis_timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "severity_levels": self.severity_levels.tolist(),
            },
            "recommendations": recommendations,
        }

        # Save to JSON
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        return output_path
