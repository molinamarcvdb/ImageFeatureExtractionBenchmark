import os
import numpy as np
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, VisualInformationFidelity
from DISTS import DISTS
from haarPsi import haar_psi
import LPIPS_l
from typing import Callable
from PIL import Image
import random
import cv2
import pandas as pd
from transformers import AutoModelForCausalLM
import pyiqa
from tqdm import tqdm
from typing import Callable, Dict, List
import tensorflow as tf
import tensorflow_hub as hub
import io
from piq import information_weighted_ssim, fsim, srsim, gmsd, multi_scale_gmsd, vsi, dss, haarpsi, mdsi
from utils import load_model_from_hub
import matplotlib.pyplot as plt
import seaborn as sns

# Metric functions
def compute_ms_ssim(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    ref_img_tensor = np_to_tensor(ref_img)
    deg_img_tensor = np_to_tensor(deg_img)
    return float(ms_ssim(deg_img_tensor, ref_img_tensor))

def compute_psnr(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ref_img_tensor = np_to_tensor(ref_img)
    deg_img_tensor = np_to_tensor(deg_img)
    return float(psnr(deg_img_tensor, ref_img_tensor))

def compute_ssim(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ref_img_tensor = np_to_tensor(ref_img)
    deg_img_tensor = np_to_tensor(deg_img)
    return float(ssim(deg_img_tensor, ref_img_tensor))

def compute_vif(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    vif = VisualInformationFidelity()
    ref_img_tensor = np_to_tensor(ref_img)
    deg_img_tensor = np_to_tensor(deg_img)
    return float(vif(deg_img_tensor, ref_img_tensor))

def compute_haarpsi(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_img = ref_img * 255
    deg_img = deg_img * 255
    return haar_psi(ref_img, deg_img)[0]

def compute_dists(dists_network, ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_img_rgb = grey_to_rgb(ref_img)
    deg_img_rgb = grey_to_rgb(deg_img)
    return float(dists_network(ref_img_rgb, deg_img_rgb).detach().numpy())

def compute_lpips(lpips_network, ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    # Normalize images
    ref_img = ref_img * 2 - 1
    deg_img = deg_img * 2 - 1
    
    # Convert images to RGB if they are grayscale
    ref_img_rgb = grey_to_rgb(ref_img) if ref_img.ndim == 2 else torch.tensor(ref_img[None, :, :, :], dtype=torch.float32)
    deg_img_rgb = grey_to_rgb(deg_img) if deg_img.ndim == 2 else torch.tensor(deg_img[None, :, :, :], dtype=torch.float32)
    
    return float(lpips_network.forward(ref_img_rgb, deg_img_rgb).detach().numpy()[0,0,0,0])
    
def compute_piq_haarpsi(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return haarpsi(ref_tensor, deg_tensor).item()

def compute_information_weighted_ssim(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return information_weighted_ssim(ref_tensor, deg_tensor).item()

def compute_fsim(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return fsim(ref_tensor, deg_tensor).item()

def compute_srsim(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return srsim(ref_tensor, deg_tensor).item()

def compute_gmsd(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return gmsd(ref_tensor, deg_tensor).item()

def compute_multi_scale_gmsd(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return multi_scale_gmsd(ref_tensor, deg_tensor).item()

def compute_vsi(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return vsi(ref_tensor, deg_tensor).item()

def compute_dss(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return dss(ref_tensor, deg_tensor).item()

def compute_mdsi(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return mdsi(ref_tensor, deg_tensor).item()

# def compute_BRISQUE(ref_img: Image) -> float:
#     image_np = np.array(ref_img).astype(np.float32) / 255.0
#     image_np = np.transpose(image_np, (2, 0, 1))   
#     ref_tensor = np_to_tensor(image_np)
#     brisq = brisque(ref_tensor).item()

#     return {
#         'BRISQUE': brisq
#     }
def compute_topiq_fr(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return topiq_fr(ref_tensor, deg_tensor).item()

def compute_ahiq(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return ahiq(ref_tensor, deg_tensor).item()

def compute_pieapp(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return pieapp(ref_tensor, deg_tensor).item()

def compute_wadiqam(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return wadiqam_fr(ref_tensor, deg_tensor).item()

def compute_cw_ssim(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return cw_ssim(ref_tensor, deg_tensor).item()

def compute_nlpd(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return nlpd(ref_tensor, deg_tensor).item()

def compute_mad(ref_img: np.ndarray, deg_img: np.ndarray) -> float:
    ref_tensor = np_to_tensor(ref_img)
    deg_tensor = np_to_tensor(deg_img)
    return mad(ref_tensor, deg_tensor).item()

def create_pyiqa_metric(metric_name: str):
    print(metric_name)
    model = pyiqa.create_metric(metric_name)
    
    def compute_metric(image: Image.Image) -> dict:
        # Convert PIL Image to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))  # From (H, W, C) to (C, H, W)
        img = np_to_tensor(image_np)     

        with torch.no_grad():
            score = model(img).item()
        
        return {metric_name: score}
    
    return compute_metric

# Utility functions
def np_to_tensor(image: np.ndarray) -> torch.Tensor:
    """ Convert a numpy array to a PyTorch tensor with the shape [N, C, H, W]. """
    if len(image.shape) == 2:  # Grayscale image
        image_tensor = torch.tensor(image[None, None, :, :], dtype=torch.float32)  # Add channel and batch dimensions
    elif len(image.shape) == 3:  # RGB image
        image_tensor = torch.tensor(image[None, :, :, :], dtype=torch.float32)  # Add batch dimension only
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    
    return image_tensor
def grey_to_rgb(image_grey: np.ndarray) -> torch.Tensor:
    """ Convert a grey-scale image to RGB format. """
    if len(image_grey.shape) == 2:  # Check if it's a grayscale image
        h, w = image_grey.shape
        image_rgb = np.empty((1, 3, h, w), dtype=image_grey.dtype)
        image_rgb[0, 0, :, :] = image_grey
        image_rgb[0, 1, :, :] = image_grey
        image_rgb[0, 2, :, :] = image_grey
        return torch.tensor(image_rgb, device='cpu', dtype=torch.float32)
    elif len(image_grey.shape) == 3 and image_grey.shape[0] == 3:  # Already an RGB image
        return torch.tensor(image_grey[None, :, :, :], device='cpu', dtype=torch.float32)
    else:
        raise ValueError(f"Unexpected image shape: {image_grey.shape}")

# Utility function to load an image and convert it to a numpy array
def load_image_as_np_array(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')  # Convert image to RGB to ensure 3 channels
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1] range
    # Rearrange the dimensions to make channels the first dimension
    image_np = np.transpose(image_np, (2, 0, 1))  # From (H, W, C) to (C, H, W)
    
    return image_np


def process_images(synthetic_dir: str, real_dir: str, metric_func: Callable, metric_name: str) -> Dict[str, float]:
    """ Compute a specified metric for synthetic images against all real images and aggregate results. """
    metric_values = {}

    real_images = [f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]

    # Track the progress of processing synthetic images
    synthetic_files = [f for f in os.listdir(synthetic_dir) if os.path.isfile(os.path.join(synthetic_dir, f))]
    for synthetic_file in tqdm(synthetic_files, desc=f'Computing {metric_name}', unit='image'):
        synthetic_path = os.path.join(synthetic_dir, synthetic_file)
        if os.path.isfile(synthetic_path):
            # Load the synthetic image
            synthetic_img = load_image_as_np_array(synthetic_path)

            # Initialize a list to store metric values for this synthetic image
            metric_list = []

            # Track the progress of processing real images
            for real_image_file in tqdm(real_images, desc='Real Images', unit='file', leave=False):
                real_image_path = os.path.join(real_dir, real_image_file)
                real_img = load_image_as_np_array(real_image_path)

                # Compute the metric
                metric_value = metric_func(real_img, synthetic_img)
                metric_list.append(metric_value)
                break
            # Aggregate the metric values (e.g., mean)
            aggregated_metric_value = np.mean(metric_list)
            metric_values[os.path.basename(synthetic_file)] = aggregated_metric_value

    return metric_values

def compute_non_ref_metrics(synthetic_dir: str, pyiqa_metrics: List[str]) -> Dict[str, Dict[str, float]]:
    all_non_ref_metrics = {}
    
    synthetic_files = [f for f in os.listdir(synthetic_dir) if os.path.isfile(os.path.join(synthetic_dir, f))]
    
    for metric_name in pyiqa_metrics:
        metric_results = []
        
        # Initialize the metric model here
        metric_model = pyiqa.create_metric(metric_name)

        for synthetic_file in tqdm(synthetic_files, desc=f'Computing {metric_name}', unit='image'):
            synthetic_path = os.path.join(synthetic_dir, synthetic_file)
            if os.path.isfile(synthetic_path):
                # Load the synthetic image
                synthetic_img = Image.open(synthetic_path).convert('RGB')
                
                # Compute the metric
                output = metric_model(synthetic_img)
                metric_results.append({"Image": os.path.basename(synthetic_file), metric_name: output.item()})
        
        # Store the results for this metric
        all_non_ref_metrics[metric_name] = metric_results
        
        # Unload the metric model to free up memory
        del metric_model
        
        # Clear CUDA cache if using GPU to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_non_ref_metrics


# Initialize metric networks once
def initialize_metrics():
    dists_network = DISTS()
    lpips_network = LPIPS_l.LPIPS(net='alex')
    topiq_fr = pyiqa.create_metric('topiq_fr')
    ahiq = pyiqa.create_metric('ahiq')
    pieapp = pyiqa.create_metric('pieapp')
    wadiqam_fr = pyiqa.create_metric('wadiqam_fr')
    cw_ssim =  pyiqa.create_metric('cw_ssim')
    nlpd = pyiqa.create_metric('nlpd')
    mad = pyiqa.create_metric('mad')

    return dists_network, lpips_network, topiq_fr, ahiq, pieapp, wadiqam_fr, cw_ssim, nlpd, mad

def main_single_metric_eval(synthetic_images_dir: str, real_images_dir: str, output_dir: str) -> None:

    # Initialize networks
    global dists_network, lpips_network, topiq_fr, ahiq, pieapp, wadiqam_fr, cw_ssim, nlpd, mad
    dists_network, lpips_network, topiq_fr, ahiq, pieapp, wadiqam_fr, cw_ssim, nlpd, mad = initialize_metrics()

    # Reference-based metrics
    metrics = {
            "MS-SSIM": compute_ms_ssim,
            "PSNR": compute_psnr,
            "SSIM": compute_ssim,
            "VIF": compute_vif,
            "DISTS": lambda ref, deg: compute_dists(dists_network, ref, deg),
            "LPIPS": lambda ref, deg: compute_lpips(lpips_network, ref, deg),
            "HaarPSI": compute_haarpsi,
            "HaarPSI_PIQ": compute_piq_haarpsi,
            "IW-SSIM": compute_information_weighted_ssim,
            "FSIM": compute_fsim,
            "SR-SIM": compute_srsim,
            "GMSD": compute_gmsd,
            "MS-GMSD": compute_multi_scale_gmsd,
            "VSI": compute_vsi,
            "DSS": compute_dss,
            "MDSI": compute_mdsi,
            "TOPIQ_FR": lambda ref, deg: compute_topiq_fr(ref, deg),
            "AHIQ": lambda ref, deg: compute_ahiq(ref, deg),
            "PIEAPP": lambda ref, deg: compute_pieapp(ref, deg),
            "WADIQAM_FR": lambda ref, deg: compute_wadiqam(ref, deg),
            "CW-SSIM": lambda ref, deg: compute_cw_ssim(ref, deg),
            "NLPD": lambda ref, deg: compute_nlpd(ref, deg),
            "MAD": lambda ref, deg: compute_mad(ref, deg),
    }

    # Non-reference metrics
    pyiqa_metrics = [
        'brisque', 'ilniqe', 'qalign', 'pi', 'nrqm', 'cnniqa', 'wadiqam_nr',
        'nima', 'nima-vgg16-ava', 'hyperiqa', 'musiq', 'musiq-spaq',
        'musiq-paq2piq', 'musiq-ava', 'maniqa', 'maniqa-kadid', 'maniqa-pipal',
        'clipiqa', 'clipiqa+', 'clipiqa+_vitL14_512', 'clipiqa+_rn50_512', 
        'topiq_nr', 'topiq_nr-flive', 'topiq_nr-spaq', 'liqe', 'liqe_mix', 'niqe'
    ] 

    # Initialize an empty list to store each metric's results
    all_metrics = []
    all_metrics_baseline = []
    
    # Process metrics that require both reference and synthetic images
    for metric_name, metric_func in metrics.items():
        metric_values = process_images(synthetic_images_dir, real_images_dir, metric_func, metric_name)
        
        # Convert metric results to DataFrame format and append to the list
        metric_df = pd.DataFrame.from_dict(metric_values, orient='index', columns=[metric_name])
        metric_df.index.name = 'Image'
        all_metrics.append(metric_df)

    # Process non-reference metrics one by one
    non_ref_results = compute_non_ref_metrics(synthetic_images_dir, pyiqa_metrics)
    non_ref_results_baseline = compute_non_ref_metrics(real_images_dir, pyiqa_metrics)

    # Convert non-reference metrics results to DataFrame format and append to the list
        
    for metric_name, metric_data in non_ref_results.items(): # Prepare DataFrame from the metric data 
        metric_df = pd.DataFrame.from_records(metric_data).set_index('Image')
        all_metrics.append(metric_df)
    
    for metric_name, metric_data in non_ref_results_baseline.items(): # Prepare DataFrame from the metric data 
        
        metric_df = pd.DataFrame.from_records(metric_data).set_index('Image')
        all_metrics_baseline.append(metric_df)

    # Concatenate all metrics DataFrames into a single DataFrame
    result_df = pd.concat(all_metrics, axis=1)
           
    result_baseline_df = pd.concat(all_metrics_baseline, axis=1)

     # Save the DataFrame to a CSV file
    result_df.to_csv(os.path.join(output_dir, "single_metric_eval.csv"))

    result_baseline_df.to_csv(os.path.join(output_dir, "single_metric_eval_baseline.csv"))

def compute_ground_truth_correlations(output_dir: str, mean_realism_z_scored: pd.DataFrame, do_z_score: bool) -> None:
    # Define the paths for the results files
    single_metric_eval_path = os.path.join(output_dir, "single_metric_eval.csv")
    single_metric_eval_baseline_path = os.path.join(output_dir, "single_metric_eval_baseline.csv")
    

    if not os.path.exists(single_metric_eval_path) or not os.path.exists(single_metric_eval_baseline_path):
        print(f"Output files not found in {output_dir}. Ensure that 'main_single_metric_eval' has been run.")
        return
    
    # Load the metric evaluation results
    single_metric_df = pd.read_csv(single_metric_eval_path, index_col='Image')
    single_metric_baseline_df = pd.read_csv(single_metric_eval_baseline_path, index_col='Image')

    # Combine the metric evaluation results with the ground truth (realism scores)
    combined_df = single_metric_df.merge(mean_realism_z_scored, left_index=True, right_index=True, how='inner')

    # Initialize a list to store the correlation results
    correlation_results = []
    
    # Select column of realism raw or z-scored
    if do_z_score:
        realism_column = 'Averaged Realism Score'
    else:
        realism_column = 'Averaged Z-Score'
        
    # Compute correlations for each metric
    for metric in single_metric_df.columns:
        pearson_corr = combined_df[metric].corr(combined_df[realism_column], method='pearson')
        kendall_corr = combined_df[metric].corr(combined_df[realism_column], method='kendall')
        correlation_results.append({
            'Metric': metric,
            'Pearson Correlation': pearson_corr,
            'Kendall Correlation': kendall_corr
        })
    
    # Convert the results into a DataFrame
    correlation_df = pd.DataFrame(correlation_results)
    
    # Save the correlation results to a CSV file
    correlation_csv_path = os.path.join(output_dir, "ground_truth_correlations.csv")
    correlation_df.to_csv(correlation_csv_path, index=False)
    
    plot_metric_comparison(single_metric_df, single_metric_baseline_df, output_dir)

    print(f"Correlation results saved to {correlation_csv_path}")



def plot_metric_comparison(single_metric_df: pd.DataFrame, single_metric_baseline_df: pd.DataFrame, output_dir: str) -> None:
    # Ensure that the dataframes have the same metrics
    common_metrics = list(set(single_metric_df.columns) & set(single_metric_baseline_df.columns))
    single_metric_df = single_metric_df[common_metrics]
    single_metric_baseline_df = single_metric_baseline_df[common_metrics]
    
    # Create a grid of subplots
    num_metrics = len(common_metrics)
    num_cols = 4  # Number of columns in the grid
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Calculate number of rows needed
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration
    
    # Plot each metric
    for i, metric in enumerate(common_metrics):
        sns.kdeplot(single_metric_df[metric], ax=axs[i], label='Synthetic', color='red')
        sns.kdeplot(single_metric_baseline_df[metric], ax=axs[i], label='Real', color='blue')
        
        axs[i].set_title(f"Comparison of {metric}")
        axs[i].legend()
        axs[i].set_xlabel('Metric Value')
        axs[i].set_ylabel('Density')
    
    # Remove any empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_comparison_real_vs_synthetic.png'))



# TEST --------------------------------------

# def generate_random_image(w: int, h: int) -> np.ndarray:
#     """ Generate a random grayscale image. """
#     return np.random.rand(h, w).astype(np.float32)

# def create_dummy_image_files(directory: str, filename: str, img_size: tuple) -> None:
#     """ Create dummy grayscale image files. """
#     ref_img = generate_random_image(*img_size)
#     deg_img = generate_random_image(*img_size)
#     np.save(os.path.join(directory, filename.replace(".npz", "_ref.npy")), ref_img)
#     np.save(os.path.join(directory, filename.replace(".npz", "_deg.npy")), deg_img)



# import tempfile
# import shutil

# def setup_test_environment() -> tuple:
#     """ Set up directories and create dummy files. """
#     synthetic_images_dir = tempfile.mkdtemp(prefix='synthetic_images_')
#     real_images_dir = tempfile.mkdtemp(prefix='real_images_')
    
#     # Create dummy image files
#     for i in range(5):  # Create 5 dummy files
#         filename = f'synthetic_image_{i}.npz'
#         create_dummy_image_files(synthetic_images_dir, filename, (2048, 2048))
#         create_dummy_image_files(real_images_dir, filename, (2048, 2048))

#     return synthetic_images_dir, real_images_dir

# def cleanup_test_environment(dirs: tuple) -> None:
#     """ Remove the temporary directories and files. """
#     for dir in dirs:
#         shutil.rmtree(dir)

# if __name__ == "__main__":
#     # Set up the test environment
#     synthetic_images_dir, real_images_dir = setup_test_environment()
    
#     try:
#         # Run the main function from the metrics module
#         main(synthetic_images_dir, real_images_dir)
#     finally:
#         # Clean up the test environment
#         cleanup_test_environment((synthetic_images_dir, real_images_dir))
