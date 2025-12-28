"""
Inference script for image inpainting models.
Use this to run object removal on your own images.

Usage (from project root):
    python test/script/infer.py --image test/imgs/image.jpg --mask test/imgs/mask.tif --model models/unet_mse_50.h5 --train_size 100
    
Output: inpainted_unet_mse_50_img<image_name>_train100.png
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get project root directory (parent of test/script/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TEST_IMGS_DIR = os.path.join(PROJECT_ROOT, "test", "imgs")


def load_and_preprocess(image_path, mask_path, size=(256, 256)):
    """Load and preprocess image and mask."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image = image.resize(size)
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Load mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(size)
    mask_array = np.array(mask).astype(np.float32) / 255.0
    mask_array = np.expand_dims(mask_array, axis=-1)
    
    return image_array, mask_array, original_size


def run_inpainting(model, image, mask):
    """Run the inpainting model on the input."""
    # Combine image and mask (4 channels: RGB + Mask)
    input_data = np.concatenate([image, mask], axis=-1)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    
    # Run prediction
    output = model.predict(input_data, verbose=0)
    output = output[0]  # Remove batch dimension
    output = np.clip(output, 0, 1)  # Ensure valid range
    
    return output


def visualize_results(original, mask, result, save_path=None):
    """Visualize the inpainting results."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Mask
    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Mask (white = remove)")
    axes[1].axis("off")
    
    # Masked input (show what model sees)
    masked_input = original * (1 - mask)
    axes[2].imshow(masked_input)
    axes[2].set_title("Masked Input")
    axes[2].axis("off")
    
    # Result
    axes[3].imshow(result)
    axes[3].set_title("Inpainted Result")
    axes[3].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def save_result(result, output_path, original_size=None):
    """Save the result image."""
    result_uint8 = (result * 255).astype(np.uint8)
    result_image = Image.fromarray(result_uint8)
    
    if original_size:
        result_image = result_image.resize(original_size)
    
    result_image.save(output_path)
    print(f"Saved result to: {output_path}")


def generate_output_filename(model_path, image_path, train_size=None):
    """Generate descriptive output filename."""
    # Extract model name (e.g., 'unet_mse_50' from 'models/unet_mse_50.h5')
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Extract image name (e.g., '000000000260' from '000000000260.jpg')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Build filename
    if train_size:
        filename = f"inpainted_{model_name}_img{image_name}_train{train_size}.png"
    else:
        filename = f"inpainted_{model_name}_img{image_name}.png"
    
    return filename


def main():
    # Default model path
    default_model = os.path.join(MODELS_DIR, "unet_mse_50.h5")
    
    parser = argparse.ArgumentParser(description='Run inpainting on an image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to mask image (white = region to remove)')
    parser.add_argument('--model', type=str, default=default_model, help='Path to trained model')
    parser.add_argument('--output', type=str, default=None, help='Output image path (auto-generated if not specified)')
    parser.add_argument('--train_size', type=int, default=None, help='Training data size (for filename)')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    if not os.path.exists(args.mask):
        print(f"Error: Mask not found: {args.mask}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print(f"Available models in '{MODELS_DIR}':")
        if os.path.exists(MODELS_DIR):
            for f in os.listdir(MODELS_DIR):
                print(f"  - {os.path.join(MODELS_DIR, f)}")
        return
    
    # Generate output filename if not specified
    if args.output is None:
        args.output = generate_output_filename(args.model, args.image, args.train_size)
    
    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)
    
    print(f"Processing image: {args.image}")
    image, mask, original_size = load_and_preprocess(args.image, args.mask)
    
    print("Running inpainting...")
    result = run_inpainting(model, image, mask)
    
    # Save result
    save_result(result, args.output, original_size)
    
    # Visualize if requested
    if args.visualize:
        comparison_path = args.output.replace('.png', '_comparison.png')
        visualize_results(image, mask, result, save_path=comparison_path)
    
    print("Done!")


if __name__ == "__main__":
    main()

