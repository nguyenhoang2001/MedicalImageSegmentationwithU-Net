# Medical Image Segmentation with U-Net

A PyTorch implementation of U-Net architecture for binary image segmentation tasks.

## 🎯 Project Overview

This project implements a U-Net model for semantic segmentation, suitable for medical imaging tasks such as:
- Lung segmentation
- Cell segmentation
- Skin lesion segmentation (ISIC dataset)
- Car segmentation (Carvana dataset)

## 🏗️ Project Structure

```
├── data/                   # Dataset directory
├── models/                 # Model architecture
│   └── unet.py            # U-Net implementation
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference and visualization
├── utils.py               # Utility functions
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🚀 Features

- **U-Net Architecture**: Fully implemented encoder-decoder with skip connections
- **Loss Functions**: BCE with Logits Loss and Dice Loss
- **Metrics**: IoU (Intersection over Union) and Dice Score
- **Visualization**: Compare predicted masks with ground truth
- **Training**: Configurable training with validation split

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "Medical Image Segmentation with U-Net"

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

This project supports any binary segmentation dataset. Recommended datasets:

- **Carvana Image Masking Challenge**: [Kaggle Link](https://www.kaggle.com/c/carvana-image-masking-challenge)
- **ISIC Skin Lesion Dataset**: [ISIC Archive](https://challenge.isic-archive.com/)

### Dataset Structure

Place your dataset in the `data/` directory with the following structure:

```
data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── masks/
    ├── img1.png
    ├── img2.png
    └── ...
```

## 🎓 Usage

### Training

```bash
python train.py --data_dir data/ --epochs 50 --batch_size 8 --lr 0.001
```

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir data/
```

### Inference

```bash
python inference.py --model_path checkpoints/best_model.pth --image_path data/images/test.jpg
```

## 📈 Model Architecture

The U-Net architecture consists of:
- **Encoder**: Downsampling path with convolutional layers
- **Bottleneck**: Bridge between encoder and decoder
- **Decoder**: Upsampling path with transpose convolutions
- **Skip Connections**: Concatenate encoder features with decoder features

## 📊 Evaluation Metrics

- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth masks
- **Dice Score**: Harmonic mean of precision and recall

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue.

---

**Status**: 🚧 Work in Progress
