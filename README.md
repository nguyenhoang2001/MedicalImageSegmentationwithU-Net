# Medical Image Segmentation with U-Net 🏥

A complete PyTorch implementation of U-Net architecture for binary image segmentation tasks, featuring comprehensive training, evaluation, and inference pipelines.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project implements a professional U-Net model for semantic segmentation, suitable for various imaging tasks:

- 🫁 **Medical Imaging**: Lung segmentation, tumor detection, organ segmentation
- 🔬 **Biological**: Cell segmentation, microscopy analysis
- 🏥 **Clinical**: Skin lesion segmentation (ISIC dataset)
- 🚗 **General**: Car segmentation (Carvana dataset), road scene segmentation

### Key Features

✅ **Complete Implementation**: Fully implemented U-Net with encoder-decoder and skip connections  
✅ **Multiple Loss Functions**: BCE, Dice Loss, and Combined Loss  
✅ **Comprehensive Metrics**: Dice, IoU, Precision, Recall, F1, Specificity  
✅ **Data Augmentation**: Extensive augmentation pipeline with Albumentations  
✅ **Training Pipeline**: Full training loop with validation and checkpointing  
✅ **Evaluation Tools**: Detailed evaluation with confusion matrix and visualizations  
✅ **Inference Support**: Single image and batch prediction capabilities  
✅ **Visualization**: Beautiful plots and overlays for predictions

## 🏗️ Project Structure

```
Medical Image Segmentation with U-Net/
├── data/                      # Dataset directory
│   ├── images/               # Input images
│   └── masks/                # Ground truth masks
├── models/                    # Model architecture
│   ├── __init__.py
│   └── unet.py               # U-Net implementation
├── checkpoints/              # Saved model checkpoints
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── inference.py              # Inference script
├── prepare_data.py           # Data preparation utility
├── utils.py                  # Utility functions
├── requirements.txt          # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## � Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/nguyenhoang2001/MedicalImageSegmentationwithU-Net.git
cd "Medical Image Segmentation with U-Net"

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- albumentations >= 1.3.0
- opencv-python >= 4.7.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.2.0
- tqdm >= 4.65.0
- numpy, Pillow

## 📊 Dataset Preparation

### Recommended Datasets

1. **Carvana Image Masking Challenge** (Beginner-friendly)
   - [Kaggle Competition](https://www.kaggle.com/c/carvana-image-masking-challenge)
   - 5,000+ car images with masks
   - Good for learning and testing

2. **ISIC Skin Lesion Dataset** (Medical)
   - [ISIC Archive](https://challenge.isic-archive.com/)
   - Skin lesion images with segmentation masks
   - Real medical imaging application

3. **Custom Dataset**
   - Any paired images and binary masks
   - Supported formats: JPG, PNG, TIF, TIFF

### Dataset Structure

Your dataset should follow this structure:

```
data/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── masks/
    ├── image001.png
    ├── image002.png
    └── ...
```

**Important Notes:**

- Image and mask filenames should match (e.g., `image001.jpg` → `image001.png`)
- Masks should be binary (0 for background, 255 for foreground)
- Masks can have suffixes like `_mask` (e.g., `image001_mask.png`)

### Preparing Your Data

If your data is in a different structure, use the preparation script:

```bash
python prepare_data.py \
    --source_images /path/to/your/images \
    --source_masks /path/to/your/masks \
    --output data/
```

This will organize your dataset into the required structure.

## 🎓 Usage Guide

### 1. Training

Train the U-Net model on your dataset:

```bash
# Basic training
python train.py --data_dir data/ --epochs 50 --batch_size 8

# Advanced training with all options
python train.py \
    --data_dir data/ \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --loss combined \
    --scheduler \
    --image_size 256 \
    --val_split 0.2
```

#### Training Arguments

| Argument           | Description                              | Default       |
| ------------------ | ---------------------------------------- | ------------- |
| `--data_dir`       | Path to dataset directory                | `data`        |
| `--epochs`         | Number of training epochs                | `50`          |
| `--batch_size`     | Batch size                               | `8`           |
| `--lr`             | Learning rate                            | `0.001`       |
| `--loss`           | Loss function: `bce`, `dice`, `combined` | `combined`    |
| `--scheduler`      | Use learning rate scheduler              | `False`       |
| `--image_size`     | Input image size                         | `256`         |
| `--val_split`      | Validation split ratio                   | `0.2`         |
| `--checkpoint_dir` | Directory for checkpoints                | `checkpoints` |
| `--save_every`     | Save checkpoint every N epochs           | `10`          |

#### Training Outputs

- `checkpoints/best_model.pth` - Best model based on validation Dice score
- `checkpoints/final_model.pth` - Final model after all epochs
- `checkpoints/checkpoint_epoch_N.pth` - Periodic checkpoints
- `checkpoints/training_history.png` - Training curves
- `checkpoints/sample_predictions.png` - Sample predictions

### 2. Evaluation

Evaluate a trained model on a test dataset:

```bash
# Basic evaluation
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/

# With custom settings
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/test/ \
    --batch_size 16 \
    --output_dir evaluation_results/
```

#### Evaluation Outputs

- `evaluation_report.txt` - Text summary of metrics
- `evaluation_report.json` - JSON format results
- `confusion_matrix.png` - Confusion matrix heatmap
- `metrics_distribution.png` - Distribution of metrics across samples
- `metrics_summary.png` - Bar chart of average metrics
- `sample_predictions.png` - Visual comparison of predictions

#### Metrics Computed

- **Dice Score**: Overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over union
- **Precision**: Positive predictive value
- **Recall (Sensitivity)**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **Pixel Accuracy**: Overall accuracy

### 3. Inference

Run predictions on new images:

#### Single Image

```bash
# Basic inference
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_path sample.jpg

# With ground truth comparison
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_path sample.jpg \
    --mask_path sample_mask.png \
    --save_overlay \
    --show_probability
```

#### Batch Processing

```bash
# Process entire directory
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_dir test_images/ \
    --output_dir predictions/ \
    --save_overlay \
    --threshold 0.5
```

#### Inference Arguments

| Argument             | Description                  | Default       |
| -------------------- | ---------------------------- | ------------- |
| `--model_path`       | Path to trained model        | _required_    |
| `--image_path`       | Single image path            | -             |
| `--image_dir`        | Directory of images          | -             |
| `--mask_path`        | Ground truth mask (optional) | -             |
| `--threshold`        | Prediction threshold         | `0.5`         |
| `--save_overlay`     | Save colored overlay         | `False`       |
| `--show_probability` | Show probability map         | `False`       |
| `--output_dir`       | Output directory             | `predictions` |

#### Inference Outputs

- `{filename}_mask.png` - Binary prediction mask
- `{filename}_overlay.png` - Colored overlay on original image
- `{filename}_visualization.png` - Multi-panel comparison

## 📈 Model Architecture

### U-Net Structure

The implemented U-Net follows the original architecture with modern improvements:

```
Input (3, 256, 256)
    ↓
[Encoder]
    DoubleConv(3 → 64)
    ↓ MaxPool
    DoubleConv(64 → 128)
    ↓ MaxPool
    DoubleConv(128 → 256)
    ↓ MaxPool
    DoubleConv(256 → 512)
    ↓ MaxPool
[Bottleneck]
    DoubleConv(512 → 1024)
    ↓
[Decoder]
    Up + Concat ← (512)
    DoubleConv(1024 → 512)
    ↓
    Up + Concat ← (256)
    DoubleConv(512 → 256)
    ↓
    Up + Concat ← (128)
    DoubleConv(256 → 128)
    ↓
    Up + Concat ← (64)
    DoubleConv(128 → 64)
    ↓
Output Conv(64 → 1)
    ↓
Output (1, 256, 256)
```

**Features:**

- **31M parameters** for standard U-Net
- **Skip connections** preserve spatial information
- **Batch normalization** for stable training
- **ReLU activation** for non-linearity
- **Optional bilinear upsampling** for memory efficiency

## 🔧 Technical Details

### Data Augmentation

Training uses extensive augmentation via Albumentations:

- Horizontal and vertical flips
- 90-degree rotations
- Affine transformations (scale, translate, rotate)
- Brightness and contrast adjustments
- Gaussian noise
- ImageNet normalization

### Loss Functions

1. **BCE Loss**: Binary Cross-Entropy with Logits
2. **Dice Loss**: 1 - Dice Coefficient (handles class imbalance)
3. **Combined Loss**: Weighted combination of BCE and Dice (recommended)

### Optimization

- **Optimizer**: Adam with weight decay
- **Learning Rate**: 0.001 (default)
- **Scheduler**: ReduceLROnPlateau (optional)
- **Early Stopping**: Based on validation Dice score

## 📊 Example Results

After training on your dataset, you should see:

- **Training/Validation curves** showing convergence
- **Dice scores** typically > 0.85 on good datasets
- **IoU scores** typically > 0.75 on good datasets
- **Visual predictions** closely matching ground truth

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**

```bash
# Reduce batch size
python train.py --batch_size 4

# Use smaller image size
python train.py --image_size 128

# Use bilinear upsampling
python train.py --bilinear
```

**2. Poor Performance**

- Ensure masks are binary (0 and 255)
- Check train/val split is reasonable (80/20)
- Try different loss functions (`--loss combined`)
- Increase training epochs
- Enable learning rate scheduling (`--scheduler`)

**3. Dataset Not Found**

```bash
# Verify directory structure
ls data/images/
ls data/masks/

# Use prepare_data.py to organize files
python prepare_data.py --source_images ... --source_masks ...
```

## 🔧 System Requirements

### Minimum

- CPU: Multi-core processor
- RAM: 8 GB
- Storage: 2 GB for code + dataset size

### Recommended

- GPU: NVIDIA GPU with 6+ GB VRAM (GTX 1060 or better)
- RAM: 16 GB
- Storage: SSD with 10+ GB

### Tested On

- Ubuntu 20.04 / macOS 12+ / Windows 10
- Python 3.8, 3.9, 3.10, 3.11, 3.12
- PyTorch 2.0, 2.1, 2.2
- CUDA 11.8, 12.1 (for GPU)

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={Medical Image Computing and Computer-Assisted Intervention},
  year={2015}
}
```

## � License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Original U-Net paper by Ronneberger et al.
- PyTorch team for the excellent framework
- Albumentations for data augmentation tools
- The open-source community

## 📧 Contact

- **GitHub**: [@nguyenhoang2001](https://github.com/nguyenhoang2001)
- **Repository**: [MedicalImageSegmentationwithU-Net](https://github.com/nguyenhoang2001/MedicalImageSegmentationwithU-Net)

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the medical imaging and computer vision community**

---

**Status**: 🚧 Work in Progress
