import torch
import numpy as np
import os
from pathlib import Path
import time
import json
from dataclasses import asdict
from diversity.div_bench import (
    ImprovedDiversityBenchmark,
    BenchmarkConfig,
    MetricAnalyzer,
)
from privacy_benchmark import initialize_model
from metrics import calculate_metrics
from PIL import Image
from tqdm import tqdm


class BenchmarkRunner:
    def __init__(
        self, base_dir="benchmark_results", use_dummy=False, n_dummy_samples=100
    ):
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir)
        self.use_dummy = use_dummy
        self.n_dummy_samples = n_dummy_samples
        self.setup_directories()

    def setup_directories(self):
        """Create organized directory structure for results"""
        self.run_dir = self.base_dir / f"run_{self.timestamp}"

        self.dirs = {
            "features": self.run_dir / "features",
            "diversity": self.run_dir / "diversity",
            "metrics": self.run_dir / "metrics",
            "plots": self.run_dir / "plots",
            "recommendations": self.run_dir / "recommendations",
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def generate_dummy_features(self, feature_dim=768):
        """Generate random dummy features for testing"""
        print(
            f"Generating dummy features: {self.n_dummy_samples} samples with dimension {feature_dim}"
        )
        features = torch.randn(self.n_dummy_samples, feature_dim)

        # Normalize features to unit length (common in embedding spaces)
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # Save dummy features
        torch.save(features, self.dirs["features"] / "extracted_features.pt")

        # Save feature metadata
        with open(self.dirs["features"] / "feature_metadata.json", "w") as f:
            json.dump(
                {
                    "n_samples": self.n_dummy_samples,
                    "feature_dim": feature_dim,
                    "data_type": "dummy",
                    "normalization": "L2",
                },
                f,
                indent=4,
            )

        return features

    def process_folder(self, backbone, processor, data_dir):
        """Process images in folder and extract features"""
        features_list = []
        image_paths = (
            list(Path(data_dir).rglob("*.jpg"))
            + list(Path(data_dir).rglob("*.png"))
            + list(Path(data_dir).rglob("*.jpeg"))
        )

        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                image = Image.open(img_path)
                inputs = processor(images=image, return_tensors="pt").to("cuda:1")
                with torch.no_grad():
                    features = backbone(**inputs).last_hidden_state.mean(dim=1)
                features_list.append(features.cpu())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        features = torch.cat(features_list, dim=0)
        torch.save(features, self.dirs["features"] / "extracted_features.pt")

        return features

    def run_complete_pipeline(self, data_dir=None):
        """Run complete benchmark pipeline with option for dummy data"""
        try:
            if self.use_dummy:
                features = self.generate_dummy_features()
                feature_dim = features.shape[1]
                model_type = "dummy"
            else:
                backbone, model_type, processor = initialize_model("dino")
                backbone.to("cuda:1")
                features = self.process_folder(backbone, processor, data_dir)
                feature_dim = features.shape[1]

            results = self.run_benchmark(features, feature_dim)

            with open(self.run_dir / "summary.json", "w") as f:
                json.dump(
                    {
                        "timestamp": self.timestamp,
                        "data_dir": str(data_dir) if data_dir else "dummy_data",
                        "model_type": model_type,
                        "feature_dim": feature_dim,
                        "n_samples": len(features),
                        "is_dummy": self.use_dummy,
                        "status": "completed",
                    },
                    f,
                    indent=4,
                )

            return results

        except Exception as e:
            print(f"Error in pipeline: {e}")
            with open(self.run_dir / "error_log.json", "w") as f:
                json.dump(
                    {"timestamp": self.timestamp, "error": str(e), "status": "failed"},
                    f,
                    indent=4,
                )
            raise

    def run_benchmark(self, features, feature_dim):
        """Run complete benchmark suite"""
        config = BenchmarkConfig(
            batch_size=32,
            n_runs=1,
            severity_levels=np.linspace(0, 0.9, 10),
            feature_dim=feature_dim,
        )

        # Save configuration
        # with open(self.run_dir / "config.json", "w") as f:
        #    json.dump(asdict(config), f, indent=4)

        benchmark = ImprovedDiversityBenchmark(
            features=features,
            calculate_metrics_fn=calculate_metrics,
            config=config,
            device="cuda",
        )

        stats = benchmark.run_statistical_evaluation()
        single_run_results = benchmark.evaluate_all_degradations()

        benchmark.save_all_results(
            stats,
            single_run_results,
        )

        analyzer = MetricAnalyzer(stats, config.severity_levels)
        recommendations = analyzer.save_recommendations()

        return {
            "stats": stats,
            "results": single_run_results,
            "recommendations": recommendations,
        }


if __name__ == "__main__":
    # Example usage with dummy data
    # runner = BenchmarkRunner(use_dummy=True, n_dummy_samples=10000)
    # results = runner.run_complete_pipeline()

    data_dir = (
        "/mnt/DV-MICROK/Syn.Dat/Marc/GitLab/datasets/CSAW/images/hg_dataset/train"
    )

    runner = BenchmarkRunner()
    results = runner.run_complete_pipeline(data_dir)
