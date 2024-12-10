import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
import time
from typing import Dict, List, Union

# Configuration
@dataclass
class BenchmarkConfig:
    batch_size: int = 50
    n_runs: int = 1
    severity_levels: np.ndarray = np.linspace(0, 0.9, 5)
    feature_dim: int = 768
    k_neighbors: int = 5
    semantic_distance_threshold: float = 0.5


# Simplified metrics calculator for testing
def calculate_metrics(ref_features, sample_features, device="cuda"):
    return {
        "fid": np.random.rand(),
        "recall": np.random.rand(),
        "precision": np.random.rand(),
    }


# Run semantic interpolation
def semantic_interpolation_test(features, severity_levels, k_neighbors=5):
    results = []
    features = features.cpu().numpy()
    n_samples = len(features)

    print(f"Starting semantic interpolation with {n_samples} samples")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine").fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Store semantic distances for each severity
    semantic_distances_by_severity = []
    metric_values_by_severity = []

    for severity in severity_levels:
        print(f"\nProcessing severity {severity:.2f}")
        degraded = features.copy()
        pairs = []
        used_samples = set()

        # Create pairs and interpolate
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

        # Store semantic distances for this severity
        pair_distances = [distances[i][neighbor_idx] for i, _ in pairs]
        semantic_distances_by_severity.append(np.mean(pair_distances))

        # Calculate metrics
        metrics = calculate_metrics(features, degraded)
        metric_values_by_severity.append(metrics)

    return semantic_distances_by_severity, metric_values_by_severity, severity_levels


# Generate dummy data
n_samples = 10000
feature_dim = 768
features = torch.randn(n_samples, feature_dim)
features = torch.nn.functional.normalize(features, p=2, dim=1)


# Run test and plot results
config = BenchmarkConfig()
semantic_distances, metrics, severities = semantic_interpolation_test(
    features, config.severity_levels, config.k_neighbors
)

# Plot results
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot metrics on left y-axis
for metric_name in ["recall", "precision", "fid"]:
    metric_values = [m[metric_name] for m in metrics]
    ax1.plot(severities, metric_values, "-o", label=metric_name.upper())

# Plot semantic distances on right y-axis
ax2.plot(severities, semantic_distances, "--k", label="Semantic Distance")

ax1.set_xlabel("Severity")
ax1.set_ylabel("Metric Value")
ax2.set_ylabel("Mean Semantic Distance")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

plt.title("Semantic Interpolation Results")
plt.tight_layout()
plt.savefig("./diversity/foo/note_semantic_interp.png")
