# IMPBÆ: Image Inpainting for Object Removal

This project implements image inpainting techniques focused on object removal using U-Net and Attention U-Net architectures with MSE and Perceptual Loss functions.

**Repository:** https://github.com/Bahakirbasogluu/IMPBA

**Dataset:** https://www.kaggle.com/datasets/bahakirbasoglu/defacto-inpainting-full-dataset

## Project Structure

```
project/
├── main.py                      # Training entry point
├── settings.json                # Configuration file
├── requirements.txt             # Python dependencies
├── README.md
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   ├── attention_unet.py    # Attention U-Net implementation
│   │   └── unet_like.py         # U-Net Like implementation
│   │
│   ├── training/                # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py           # Perceptual loss trainer (VGG19)
│   │
│   ├── data/                    # Data processing
│   │   ├── __init__.py
│   │   ├── preprocess.py        # Image loading and preprocessing
│   │   └── download.py          # MSCOCO image downloader
│   │
│   └── evaluation/              # Evaluation metrics
│       ├── __init__.py
│       └── metrics.py           # PSNR, SSIM, LPIPS implementations
│
├── test/                        # Testing utilities
│   ├── script/
│   │   └── infer.py             # Inference script
│   └── imgs/                    # Test images
│
├── Dataset/                     # Training data (Defacto dataset)
│   ├── original_images/         # Original MSCOCO images
│   ├── inpainting_annotations/
│   │   └── inpaint_mask/        # Binary masks
│   └── inpainting_img/
│       └── img/                 # Inpainted images (ground truth)
│
├── models/                      # Saved trained models
├── predictions/                 # Output predictions
├── paper/                       # Project paper
└── presentation/                # Presentation files
```

## Configuration

Training parameters are specified in `settings.json`:

```json
{
    "training": {
        "model": "AttentionUNetMSE50",
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "train_size": 1000,
        "test_size": 100,
        "val_size": 100
    },
    "paths": {
        "dataset": "Dataset",
        "models": "models",
        "predictions": "predictions"
    }
}
```

### Available Models

| Model Identifier | Description |
|------------------|-------------|
| UNetLikeMSE50 | U-Net Like architecture with MSE Loss (50 epochs) |
| AttentionUNetMSE50 | Attention U-Net architecture with MSE Loss (50 epochs) |
| AttentionUNetPerceptual20 | Attention U-Net with Perceptual Loss (20 epochs) |
| AttentionUNetPerceptual50 | Attention U-Net with Perceptual Loss (50 epochs) |
| ALL | Sequential training of all models |

## Installation

The following command is used to install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages include TensorFlow 2.x, NumPy, Pillow, scikit-image, matplotlib, and lpips.

## Dataset Preparation

1. The Defacto Inpainting Dataset is downloaded from Kaggle
2. The dataset is extracted to the `Dataset/` directory
3. Original MSCOCO images are downloaded using the provided script:

```bash
python src/data/download.py
```

## Usage

### Training

The training process is initiated by executing the main script. Configuration parameters are read from `settings.json`:

```bash
python main.py
```

Upon completion, the trained model is saved to the `models/` directory, and sample predictions are stored in `predictions/`.

### Inference

Custom images can be processed using the inference script:

```bash
python test/script/infer.py --image test/imgs/image.jpg --mask test/imgs/mask.tif --train_size 1000
```

Parameters:

| Parameter | Description |
|-----------|-------------|
| --image | Path to input image |
| --mask | Path to binary mask (white regions indicate areas to be removed) |
| --model | Path to trained model file (default: models/unet_mse_50.h5) |
| --train_size | Training dataset size (used in output filename) |
| --visualize | Display comparison visualization |

Output filename format: `inpainted_<model>_img<image>_train<size>.png`

### Evaluation

Model evaluation is performed using the metrics script:

```bash
python src/evaluation/metrics.py --model models/unet_mse_50.h5
```

The following metrics are computed:

| Metric | Description | Optimal Direction |
|--------|-------------|-------------------|
| PSNR | Peak Signal-to-Noise Ratio | Higher values indicate better quality |
| SSIM | Structural Similarity Index | Higher values indicate better quality |
| LPIPS | Learned Perceptual Image Patch Similarity | Lower values indicate better quality |

## Methodology

The training process follows these steps:

1. Original images containing objects are loaded
2. Binary masks indicating regions to be removed are applied
3. Ground truth inpainted images are used as targets
4. The model learns the mapping: Original Image + Mask -> Inpainted Image

## Experimental Results

The following results were obtained from the experiments:

| Model | SSIM | PSNR (dB) | LPIPS |
|-------|------|-----------|-------|
| Attention U-Net + MSE (50 epochs) | 0.98 | 36.5 | 0.0312 |
| U-Net Like + MSE (50 epochs) | 0.97 | 37.0 | 0.0289 |
| Attention U-Net + Perceptual (50 epochs) | 0.96 | 35.2 | 0.0215 |

## Author

Baha Kirbasoglu

Istanbul Technical University - BLG513 Image Processing
