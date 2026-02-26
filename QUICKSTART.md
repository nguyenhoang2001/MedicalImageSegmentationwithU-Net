# Quick Start Guide 🚀

This guide will help you get started with Medical Image Segmentation using U-Net in under 10 minutes.

## Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/nguyenhoang2001/MedicalImageSegmentationwithU-Net.git
cd "Medical Image Segmentation with U-Net"

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Get a Dataset (5 minutes)

### Option A: Carvana Dataset (Recommended for beginners)

1. Go to [Kaggle Carvana Competition](https://www.kaggle.com/c/carvana-image-masking-challenge/data)
2. Download `train.zip` and `train_masks.zip`
3. Extract both files
4. Prepare the data:

```bash
python prepare_data.py \
    --source_images /path/to/train \
    --source_masks /path/to/train_masks \
    --output data/
```

### Option B: ISIC Skin Lesion Dataset

1. Visit [ISIC Challenge](https://challenge.isic-archive.com/)
2. Download training images and segmentation masks
3. Prepare the data as shown above

### Option C: Custom Dataset

Organize your data:

```
data/
├── images/
│   └── (your images here)
└── masks/
    └── (your masks here)
```

## Step 3: Train Your First Model (30 seconds to start)

### Quick Training (Small dataset, fast testing)

```bash
python train.py --data_dir data/ --epochs 10 --batch_size 4
```

### Full Training (Better results)

```bash
python train.py \
    --data_dir data/ \
    --epochs 50 \
    --batch_size 8 \
    --loss combined \
    --scheduler
```

**Training will show:**

- Progress bars for each epoch
- Loss, Dice, and IoU metrics
- Automatic saving of best model

## Step 4: Evaluate Your Model

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --data_dir data/
```

**You'll get:**

- Comprehensive metrics report
- Confusion matrix
- Sample predictions
- Performance visualizations

## Step 5: Make Predictions

### Single Image

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_path sample.jpg \
    --save_overlay
```

### Batch Processing

```bash
python inference.py \
    --model_path checkpoints/best_model.pth \
    --image_dir test_images/ \
    --save_overlay
```

## 🎯 Expected Results

### After 10 epochs on Carvana:

- Dice Score: ~0.75-0.85
- IoU Score: ~0.65-0.75
- Training time: ~10-30 minutes (depending on GPU)

### After 50 epochs on Carvana:

- Dice Score: >0.90
- IoU Score: >0.82
- Training time: ~1-3 hours (depending on GPU)

## 🔧 Troubleshooting

### "CUDA out of memory"

```bash
python train.py --batch_size 2 --image_size 128
```

### "No images found"

Check your data directory structure matches the required format.

### Slow training

- Use smaller `--image_size` (e.g., 128 or 192)
- Reduce `--batch_size`
- Use `--bilinear` flag for lighter model

## 📚 Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, loss functions
2. **Data augmentation**: Modify `utils.py` for custom augmentations
3. **Fine-tuning**: Resume training with `--resume checkpoints/best_model.pth`
4. **Deploy**: Use `inference.py` for production predictions

## 💡 Tips for Better Results

1. **More data = better results**: Aim for 1000+ images if possible
2. **Balanced dataset**: Ensure variety in your training images
3. **Quality masks**: Clean, accurate masks are crucial
4. **Train longer**: 50-100 epochs usually gives best results
5. **Use scheduler**: Add `--scheduler` flag for adaptive learning rate
6. **Monitor training**: Watch the training curves in `checkpoints/training_history.png`

## 🆘 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Open an issue on [GitHub](https://github.com/nguyenhoang2001/MedicalImageSegmentationwithU-Net/issues)
- Review the troubleshooting section in README.md

---

Happy Segmenting! 🎉
