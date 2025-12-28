#!/usr/bin/env python
"""
IMPBÆ: Image Inpainting for Object Removal
Main entry point for training inpainting models.

Settings are loaded from settings.json
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError

from src.models import AttentionUNet, UNetLikeModel
from src.training import InpaintingTrainer
from src.data import preprocess_images

# ============================================================
# CONFIGURATION
# ============================================================

def load_settings():
    """Load settings from settings.json"""
    settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
    with open(settings_path, 'r') as f:
        return json.load(f)

# Load settings
SETTINGS = load_settings()
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths from settings
DATASET_DIR = os.path.join(PROJECT_ROOT, SETTINGS['paths']['dataset'])
INPUT_DIR = os.path.join(PROJECT_ROOT, SETTINGS['paths']['original_images'])
MASK_DIR = os.path.join(PROJECT_ROOT, SETTINGS['paths']['masks'])
GT_DIR = os.path.join(PROJECT_ROOT, SETTINGS['paths']['ground_truth'])
MODELS_DIR = os.path.join(PROJECT_ROOT, SETTINGS['paths']['models'])
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, SETTINGS['paths']['predictions'])

# Training parameters from settings
TRAIN_SIZE = SETTINGS['training']['train_size']
TEST_SIZE = SETTINGS['training']['test_size']
VAL_SIZE = SETTINGS['training']['val_size']
BATCH_SIZE = SETTINGS['training']['batch_size']
LEARNING_RATE = SETTINGS['training']['learning_rate']
IMAGE_SIZE = tuple(SETTINGS['image']['size'])


# ============================================================
# DATA LOADING
# ============================================================

def get_matching_files(input_dir, mask_dir, gt_dir, max_count=None):
    """Find matching files across input, mask, and ground truth directories."""
    # Get input files (JPG format)
    input_files = {}
    for f in os.listdir(input_dir):
        if f.endswith(('.jpg', '.png', '.tif')):
            mscoco_id = os.path.splitext(f)[0]
            input_files[mscoco_id] = f
    
    # Get mask files (TIF format: 0_ID.tif)
    mask_files = {}
    for f in os.listdir(mask_dir):
        if f.endswith(('.tif', '.jpg', '.png')):
            parts = os.path.splitext(f)[0].split('_')
            if len(parts) >= 2:
                mscoco_id = parts[1]
                mask_files[mscoco_id] = f
    
    # Get GT files (TIF format: 0_ID.tif)
    gt_files = {}
    for f in os.listdir(gt_dir):
        if f.endswith(('.tif', '.jpg', '.png')):
            parts = os.path.splitext(f)[0].split('_')
            if len(parts) >= 2:
                mscoco_id = parts[1]
                gt_files[mscoco_id] = f
    
    # Find common IDs
    common_ids = set(input_files.keys()) & set(mask_files.keys()) & set(gt_files.keys())
    common_ids = sorted(list(common_ids))
    
    if max_count:
        common_ids = common_ids[:max_count]
    
    input_paths = [os.path.join(input_dir, input_files[id]) for id in common_ids]
    mask_paths = [os.path.join(mask_dir, mask_files[id]) for id in common_ids]
    gt_paths = [os.path.join(gt_dir, gt_files[id]) for id in common_ids]
    
    return input_paths, mask_paths, gt_paths


def load_and_prepare_data():
    """Load and prepare train, validation, and test datasets."""
    print("Loading dataset...")
    
    all_input, all_mask, all_gt = get_matching_files(INPUT_DIR, MASK_DIR, GT_DIR)
    total_files = len(all_input)
    print(f"Found {total_files} matching image sets.")
    
    if total_files < TRAIN_SIZE + TEST_SIZE + VAL_SIZE:
        print("Warning: Not enough data. Using proportional split.")
        train_end = int(total_files * 0.7)
        test_end = int(total_files * 0.85)
    else:
        train_end = TRAIN_SIZE
        test_end = TRAIN_SIZE + TEST_SIZE
    
    # Split data
    train_input_paths = all_input[:train_end]
    train_mask_paths = all_mask[:train_end]
    train_gt_paths = all_gt[:train_end]
    
    test_input_paths = all_input[train_end:test_end]
    test_mask_paths = all_mask[train_end:test_end]
    test_gt_paths = all_gt[train_end:test_end]
    
    val_input_paths = all_input[test_end:test_end + VAL_SIZE]
    val_mask_paths = all_mask[test_end:test_end + VAL_SIZE]
    val_gt_paths = all_gt[test_end:test_end + VAL_SIZE]
    
    print(f"Train: {len(train_input_paths)}, Test: {len(test_input_paths)}, Val: {len(val_input_paths)}")
    
    # Preprocess
    print("Preprocessing train images...")
    train_input, train_masks, train_gts = preprocess_images(train_input_paths, train_mask_paths, train_gt_paths)
    
    print("Preprocessing validation images...")
    val_input, val_masks, val_gts = preprocess_images(val_input_paths, val_mask_paths, val_gt_paths)
    
    print("Preprocessing test images...")
    test_input, test_masks, test_gts = preprocess_images(test_input_paths, test_mask_paths, test_gt_paths)
    
    # Combine input with masks (4-channel)
    train_data = np.concatenate((train_input, train_masks), axis=-1)
    val_data = np.concatenate((val_input, val_masks), axis=-1)
    test_data = np.concatenate((test_input, test_masks), axis=-1)
    
    return (train_data, train_gts), (val_data, val_gts), (test_data, test_gts)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_unet_mse(train_data, train_gts, val_data, val_gts, epochs=50):
    """Train U-Net Like model with MSE loss."""
    print(f"\n{'='*60}")
    print(f"Training: U-Net Like + MSE @ {epochs} epochs")
    print(f"{'='*60}")
    
    model = UNetLikeModel().model
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=MeanSquaredError())
    
    history = model.fit(
        train_data, train_gts,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(val_data, val_gts)
    )
    
    model.save(os.path.join(MODELS_DIR, "unet_mse_50.h5"))
    return model, history


def train_attention_unet_mse(train_data, train_gts, val_data, val_gts, epochs=50):
    """Train Attention U-Net model with MSE loss."""
    print(f"\n{'='*60}")
    print(f"Training: Attention U-Net + MSE @ {epochs} epochs")
    print(f"{'='*60}")
    
    model = AttentionUNet()
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=MeanSquaredError())
    
    history = model.fit(
        train_data, train_gts,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(val_data, val_gts)
    )
    
    model.save(os.path.join(MODELS_DIR, "attention_unet_mse_50.h5"))
    return model, history


def train_attention_unet_perceptual(train_data, train_gts, val_data, val_gts, test_data, epochs=50):
    """Train Attention U-Net model with Perceptual loss."""
    print(f"\n{'='*60}")
    print(f"Training: Attention U-Net + Perceptual Loss @ {epochs} epochs")
    print(f"{'='*60}")
    
    model = AttentionUNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    trainer = InpaintingTrainer(
        model=model,
        train_data=train_data,
        train_gts=train_gts,
        val_data=val_data,
        val_gts=val_gts,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        test_input_data=test_data,
        test_index_to_print=None
    )
    
    model = trainer.train()
    model.save(os.path.join(MODELS_DIR, f"attention_unet_perceptual_{epochs}.h5"))
    return model


# ============================================================
# MAIN
# ============================================================

def main():
    """Main training function."""
    
    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    # Load data
    (train_data, train_gts), (val_data, val_gts), (test_data, test_gts) = load_and_prepare_data()
    
    # Get model choice from settings
    MODEL_CHOICE = SETTINGS['training']['model']
    print(f"\nModel selected: {MODEL_CHOICE}")
    
    if MODEL_CHOICE == "UNetLikeMSE50":
        model, _ = train_unet_mse(train_data, train_gts, val_data, val_gts, epochs=50)
        
    elif MODEL_CHOICE == "AttentionUNetMSE50":
        model, _ = train_attention_unet_mse(train_data, train_gts, val_data, val_gts, epochs=50)
        
    elif MODEL_CHOICE == "AttentionUNetPerceptual20":
        model = train_attention_unet_perceptual(train_data, train_gts, val_data, val_gts, test_data, epochs=20)
        
    elif MODEL_CHOICE == "AttentionUNetPerceptual50":
        model = train_attention_unet_perceptual(train_data, train_gts, val_data, val_gts, test_data, epochs=50)
        
    elif MODEL_CHOICE == "ALL":
        print("Training all models...")
        train_unet_mse(train_data, train_gts, val_data, val_gts, epochs=50)
        train_attention_unet_mse(train_data, train_gts, val_data, val_gts, epochs=50)
        train_attention_unet_perceptual(train_data, train_gts, val_data, val_gts, test_data, epochs=20)
        model = train_attention_unet_perceptual(train_data, train_gts, val_data, val_gts, test_data, epochs=50)
    else:
        raise ValueError(f"Unknown model: {MODEL_CHOICE}. Check settings.json")
    
    # Generate predictions
    print("\nGenerating predictions on test set...")
    predictions = model.predict(test_data)
    
    # Save sample predictions
    print("Saving sample predictions...")
    for i in range(min(10, len(predictions))):
        pred_img = Image.fromarray((predictions[i] * 255).astype(np.uint8))
        pred_img.save(os.path.join(PREDICTIONS_DIR, f"pred_{i}.png"))
    
    print("\nTraining complete!")
    print(f"Model saved to: {MODELS_DIR}")
    print(f"Predictions saved to: {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()
