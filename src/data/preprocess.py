"""
Image preprocessing utilities for inpainting dataset.
Handles loading, resizing, and normalization of input images, masks, and ground truth.
"""

from PIL import Image
import numpy as np


def preprocess_images(image_paths, mask_paths, gt_paths):
    """
    Load and preprocess images for training.
    
    Args:
        image_paths: List of paths to input images
        mask_paths: List of paths to mask images
        gt_paths: List of paths to ground truth images
    
    Returns:
        Tuple of (images_array, masks_array, gts_array)
        All normalized to [0, 1] range and resized to (256, 256)
    """
    input_size = (256, 256)
    images_array = []
    masks_array = []
    gts_array = []
    
    for image_path, mask_path, gt_path in zip(image_paths, mask_paths, gt_paths):
        # Load and preprocess input image
        image = Image.open(image_path).convert("RGB")
        image = image.resize(input_size)
        image_array = np.array(image).astype(np.float32) / 255.0
        images_array.append(image_array)

        # Load and preprocess mask image
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = mask.resize(input_size)
        mask_array = np.array(mask).astype(np.float32) / 255.0
        mask_array = np.expand_dims(mask_array, axis=-1)  # Add channel dimension
        masks_array.append(mask_array)

        # Load and preprocess ground truth image
        gt = Image.open(gt_path).convert("RGB")
        gt = gt.resize(input_size)
        gt_array = np.array(gt).astype(np.float32) / 255.0
        gts_array.append(gt_array)

    images_array = np.array(images_array)
    masks_array = np.array(masks_array)
    gts_array = np.array(gts_array)

    return images_array, masks_array, gts_array
