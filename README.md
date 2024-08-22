
# README for `main.py`

## Overview

The `main.py` script is designed for extracting features from image datasets using various networks, evaluating these features with different metrics, and performing realism correlation analysis. The script leverages configurations from a YAML file to control feature extraction, metric computation, and realism correlation tasks.

## Prerequisites

- Python 3.x
- Required libraries: `numpy`, `pandas`, `yaml`, `scikit-learn`, `tqdm`, `seaborn`, `matplotlib`, etc.
- Custom modules: `networks`, `metrics`, `utils`

## Configuration File

The script uses a configuration file (`config.yml`) to define various settings. Here is an example configuration file and explanations for each field:

```yaml
# Configuration file for feature extraction and metric computation
feature_extraction: false
realism_correlation: true

# Paths to the real and synthetic datasets
real_dataset_path: './data/real'
synthetic_dataset_path: './data/synthetic/diffusion'

# List of networks to use for feature extraction
networks:
  - 'rad_inception'
  - 'resnet50'
  - 'rad_resnet50'
  - 'inceptionresnet'
  - 'rad_inceptionresnet'
  - 'densenet'
  - 'rad_densenet'
  - 'clip'
  - 'rad_clip'
  - 'rad_dino'
  - 'dino'
  - 'inception'

metrics:
  - fid
  - precision 
  - recall
  - density
  - coverage

# Batch size for feature extraction
batch_size: 64

# Number of sets to divide each dataset into
num_sets: 10

# Directory to save the metrics results
metrics_output_dir: 'metrics'

# Directory to save features and metrics
features_output_dir: 'data/features'

# Path to jsonl where turign test responses are contained
jsonl_path: ''
timestamp: '' # for specific features previously computed, use with feature_extraction=false
```

### Fields

- `feature_extraction`: Boolean flag indicating whether to perform feature extraction.
- `realism_correlation`: Boolean flag indicating whether to perform realism correlation analysis.
- `real_dataset_path`: Directory path where real images are stored.
- `synthetic_dataset_path`: Directory path where synthetic images are stored.
- `networks`: List of network names to use for feature extraction.
- `metrics`: List of metrics to compute.
- `batch_size`: Batch size for processing images.
- `num_sets`: Number of sets to split each dataset into.
- `metrics_output_dir`: Directory to save metric results.
- `features_output_dir`: Directory to save features and metrics.
- `jsonl_path`: Path to the JSONL file containing Turing test responses.
- `timestamp`: Timestamp for referencing specific previously computed features (used only if `feature_extraction` is `false`).

## Usage

1. **Prepare Configuration File**: Ensure you have a `config.yml` file in the same directory as `main.py` with appropriate settings.

2. **Run the Script**: Execute the script using the following command:
    ```bash
    python main.py
    ```

3. **Feature Extraction**:
   - If `feature_extraction` is set to `true` in the configuration, the script will process the datasets, extract features using specified networks, and save them.

4. **Metric Evaluation**:
   - The script calculates metrics for the extracted features and saves them in both YAML and CSV formats if `feature_extraction` is enabled.

5. **Realism Correlation**:
   - If `realism_correlation` is set to `true`, the script will perform realism correlation analysis using the metrics and save the results.

6. **Using Previously Computed Features**:
   - If `feature_extraction` is set to `false`, ensure that the `timestamp` matches the directory of previously computed features for the script to use them.

## Custom Modules

- **`networks`**:
  - Functions for extracting features from images and initializing models.
- **`metrics`**:
  - Functions for calculating various metrics between real and synthetic features.
- **`utils`**:
  - Utility functions for linking Azure paths to local paths, managing datasets, and performing realism correlation.

## Example

To run the script with feature extraction enabled and realism correlation analysis:

1. Set `feature_extraction` and `realism_correlation` to `true` in `config.yml`.
2. Update paths and settings in `config.yml` according to your setup.
3. Execute:
    ```bash
    python main.py
    ```

The script will process the datasets, extract features, compute metrics, and perform realism correlation analysis.

