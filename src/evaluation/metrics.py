"""
Evaluation script for image inpainting models.
Computes PSNR, SSIM, and LPIPS metrics as described in the paper.

Usage:
    python evaluate.py --model models/attention_unet_mse_50.h5
    python evaluate.py --test  # Run with dummy data to test metrics
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================================
# PSNR - Peak Signal-to-Noise Ratio
# ============================================================

def calculate_psnr(original, reconstructed, max_pixel=1.0):
    """
    Calculate PSNR between two images.
    
    Formula: PSNR = 10 * log10(MAX^2 / MSE)
    
    Args:
        original: Ground truth image (H, W, C) normalized to [0, 1]
        reconstructed: Reconstructed image (H, W, C) normalized to [0, 1]
        max_pixel: Maximum pixel value (1.0 for normalized, 255 for uint8)
    
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


# ============================================================
# SSIM - Structural Similarity Index
# ============================================================

def calculate_ssim(original, reconstructed, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Calculate SSIM between two images.
    
    Args:
        original: Ground truth image (H, W, C) normalized to [0, 1]
        reconstructed: Reconstructed image (H, W, C) normalized to [0, 1]
    
    Returns:
        SSIM value between -1 and 1 (higher is better)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(original, reconstructed, channel_axis=2, data_range=1.0)
    except ImportError:
        # Fallback to simple implementation
        mu_x = np.mean(original)
        mu_y = np.mean(reconstructed)
        
        sigma_x = np.var(original)
        sigma_y = np.var(reconstructed)
        sigma_xy = np.cov(original.flatten(), reconstructed.flatten())[0, 1]
        
        ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim_val


# ============================================================
# LPIPS - Learned Perceptual Image Patch Similarity
# ============================================================

def calculate_lpips(original, reconstructed, net='alex'):
    """
    Calculate LPIPS between two images using pretrained network.
    Lower LPIPS = better perceptual similarity.
    
    Args:
        original: Ground truth image (H, W, C) normalized to [0, 1]
        reconstructed: Reconstructed image (H, W, C) normalized to [0, 1]
        net: Network to use ('alex', 'vgg', 'squeeze')
    
    Returns:
        LPIPS distance (lower is better)
    """
    try:
        import torch
        import lpips
        
        # Initialize LPIPS model
        loss_fn = lpips.LPIPS(net=net)
        
        # Convert numpy to torch tensors
        # LPIPS expects (N, C, H, W) format with values in [-1, 1]
        original_t = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float()
        reconstructed_t = torch.from_numpy(reconstructed).permute(2, 0, 1).unsqueeze(0).float()
        
        # Scale from [0, 1] to [-1, 1]
        original_t = original_t * 2 - 1
        reconstructed_t = reconstructed_t * 2 - 1
        
        with torch.no_grad():
            distance = loss_fn(original_t, reconstructed_t)
        
        return distance.item()
        
    except ImportError:
        print("Warning: lpips package not installed. Using VGG-based approximation.")
        return calculate_lpips_vgg(original, reconstructed)


def calculate_lpips_vgg(original, reconstructed):
    """Approximate LPIPS using VGG19 features (TensorFlow implementation)."""
    # Load VGG19
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    
    # Get intermediate layer outputs
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
    outputs = [vgg.get_layer(name).output for name in layer_names]
    feature_model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
    
    # Preprocess images for VGG
    original_vgg = tf.keras.applications.vgg19.preprocess_input(original * 255)
    reconstructed_vgg = tf.keras.applications.vgg19.preprocess_input(reconstructed * 255)
    
    # Add batch dimension
    original_vgg = np.expand_dims(original_vgg, 0)
    reconstructed_vgg = np.expand_dims(reconstructed_vgg, 0)
    
    # Extract features
    original_features = feature_model.predict(original_vgg, verbose=0)
    reconstructed_features = feature_model.predict(reconstructed_vgg, verbose=0)
    
    # Calculate normalized L2 distance
    total_distance = 0
    for orig_f, recon_f in zip(original_features, reconstructed_features):
        # Normalize features
        orig_f = orig_f / (np.linalg.norm(orig_f, axis=-1, keepdims=True) + 1e-10)
        recon_f = recon_f / (np.linalg.norm(recon_f, axis=-1, keepdims=True) + 1e-10)
        total_distance += np.mean((orig_f - recon_f) ** 2)
    
    return total_distance / len(layer_names)


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_single_image(original, reconstructed):
    """Evaluate a single image pair with all metrics."""
    psnr = calculate_psnr(original, reconstructed)
    ssim = calculate_ssim(original, reconstructed)
    lpips_val = calculate_lpips(original, reconstructed)
    
    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips_val
    }


def evaluate_model(model, test_data, test_gt):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test input data (N, H, W, 4)
        test_gt: Ground truth images (N, H, W, 3)
    
    Returns:
        Dictionary with average metrics
    """
    print("Generating predictions...")
    predictions = model.predict(test_data, verbose=1)
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    print("Calculating metrics...")
    for i in range(len(predictions)):
        metrics = evaluate_single_image(test_gt[i], predictions[i])
        psnr_scores.append(metrics['PSNR'])
        ssim_scores.append(metrics['SSIM'])
        lpips_scores.append(metrics['LPIPS'])
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(predictions)} images")
    
    results = {
        'PSNR': {
            'mean': np.mean(psnr_scores),
            'std': np.std(psnr_scores),
            'min': np.min(psnr_scores),
            'max': np.max(psnr_scores)
        },
        'SSIM': {
            'mean': np.mean(ssim_scores),
            'std': np.std(ssim_scores),
            'min': np.min(ssim_scores),
            'max': np.max(ssim_scores)
        },
        'LPIPS': {
            'mean': np.mean(lpips_scores),
            'std': np.std(lpips_scores),
            'min': np.min(lpips_scores),
            'max': np.max(lpips_scores)
        }
    }
    
    return results


def print_results(results, model_name="Model"):
    """Print evaluation results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*60}")
    print(f"{'Metric':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*60}")
    
    for metric, values in results.items():
        print(f"{metric:<10} {values['mean']:>10.4f} {values['std']:>10.4f} "
              f"{values['min']:>10.4f} {values['max']:>10.4f}")
    
    print(f"{'='*60}\n")


# ============================================================
# MAIN
# ============================================================

def test_metrics():
    """Test metrics with dummy data."""
    print("Testing metrics with dummy data...\n")
    
    # Create dummy images
    np.random.seed(42)
    original = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Create similar reconstructed image
    noise = np.random.rand(256, 256, 3).astype(np.float32) * 0.1
    reconstructed = np.clip(original + noise, 0, 1)
    
    print("Testing identical images:")
    metrics = evaluate_single_image(original, original)
    print(f"  PSNR: {metrics['PSNR']:.2f} dB (expected: inf)")
    print(f"  SSIM: {metrics['SSIM']:.4f} (expected: 1.0)")
    print(f"  LPIPS: {metrics['LPIPS']:.4f} (expected: 0.0)")
    
    print("\nTesting slightly different images:")
    metrics = evaluate_single_image(original, reconstructed)
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  SSIM: {metrics['SSIM']:.4f}")
    print(f"  LPIPS: {metrics['LPIPS']:.4f}")
    
    print("\nMetrics test complete!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate inpainting model')
    parser.add_argument('--model', type=str, help='Path to trained model (.h5 file)')
    parser.add_argument('--test', action='store_true', help='Run metric tests with dummy data')
    args = parser.parse_args()
    
    if args.test:
        test_metrics()
        return
    
    if args.model:
        print(f"Loading model from {args.model}...")
        model = tf.keras.models.load_model(args.model)
        
        # Load test data
        from main import load_and_prepare_data
        (_, _), (_, _), (test_data, test_gt) = load_and_prepare_data()
        
        # Evaluate
        results = evaluate_model(model, test_data, test_gt)
        print_results(results, os.path.basename(args.model))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
