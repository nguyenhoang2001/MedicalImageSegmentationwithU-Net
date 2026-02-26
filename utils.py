"""
Utility Functions for Medical Image Segmentation

This module contains:
- Custom Dataset class for loading images and masks
- Evaluation metrics (IoU, Dice Score)
- Loss functions (Dice Loss, Combined Loss)
- Helper functions for data processing and visualization
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import cv2


class SegmentationDataset(Dataset):
    """
    Custom Dataset for Image Segmentation
    
    Loads images and their corresponding binary masks.
    Applies data augmentation and preprocessing.
    """
    
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir (str): Directory containing input images
            masks_dir (str): Directory containing ground truth masks
            transform (albumentations.Compose): Augmentation pipeline
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Get list of image files
        self.images = sorted([f for f in os.listdir(images_dir) 
                             if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Found {len(self.images)} images in {images_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (C, H, W)
            mask: Tensor of shape (1, H, W) with values in [0, 1]
        """
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Try different mask extensions
        mask_name = os.path.splitext(img_name)[0]
        mask_path = None
        for ext in ['.png', '.jpg', '.gif', '.tif', '.tiff']:
            potential_path = os.path.join(self.masks_dir, mask_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        # Also try with _mask suffix
        if mask_path is None:
            for ext in ['.png', '.jpg', '.gif', '.tif', '.tiff']:
                potential_path = os.path.join(self.masks_dir, mask_name + '_mask' + ext)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    break
        
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image {img_name}")
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Add channel dimension to mask
        mask = mask.unsqueeze(0)
        
        return image, mask


def get_train_transform(image_size=256):
    """
    Data augmentation pipeline for training
    
    Args:
        image_size (int): Size to resize images
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(-15, 15), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size=256):
    """
    Preprocessing pipeline for validation/testing (no augmentation)
    
    Args:
        image_size (int): Size to resize images
    
    Returns:
        albumentations.Compose: Preprocessing pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ==================== Metrics ====================

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice Coefficient (F1 Score for segmentation)
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Args:
        pred (torch.Tensor): Predicted mask (B, 1, H, W) - logits or probabilities
        target (torch.Tensor): Ground truth mask (B, 1, H, W) - binary [0, 1]
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Dice coefficient
    """
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred = (pred > 0.5).float()  # Binarize
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU / Jaccard Index)
    
    IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        pred (torch.Tensor): Predicted mask (B, 1, H, W) - logits or probabilities
        target (torch.Tensor): Ground truth mask (B, 1, H, W) - binary [0, 1]
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: IoU score
    """
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred = (pred > 0.5).float()  # Binarize
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred (torch.Tensor): Predicted mask (B, 1, H, W)
        target (torch.Tensor): Ground truth mask (B, 1, H, W)
    
    Returns:
        float: Pixel accuracy
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    correct = (pred == target).sum()
    total = target.numel()
    
    return (correct / total).item()


# ==================== Loss Functions ====================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    
    Dice Loss = 1 - Dice Coefficient
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted logits (B, 1, H, W)
            target (torch.Tensor): Ground truth mask (B, 1, H, W)
        
        Returns:
            torch.Tensor: Dice loss
        """
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE and Dice Loss
    
    This loss function combines Binary Cross Entropy with Dice Loss
    to leverage the benefits of both:
    - BCE: Good for pixel-wise optimization
    - Dice: Good for handling class imbalance
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted logits (B, 1, H, W)
            target (torch.Tensor): Ground truth mask (B, 1, H, W)
        
        Returns:
            torch.Tensor: Combined loss
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ==================== Helper Functions ====================

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Save model checkpoint
    
    Args:
        state (dict): Dictionary containing model state, optimizer state, etc.
        filename (str): Path to save checkpoint
    """
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (nn.Module): Model to load weights into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
    
    Returns:
        int: Epoch number from checkpoint
    """
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"=> Loaded checkpoint from epoch {epoch}")
    
    return epoch


def visualize_predictions(images, masks, predictions, num_samples=4):
    """
    Visualize images, ground truth masks, and predictions
    
    Args:
        images (torch.Tensor): Input images (B, C, H, W)
        masks (torch.Tensor): Ground truth masks (B, 1, H, W)
        predictions (torch.Tensor): Predicted masks (B, 1, H, W)
        num_samples (int): Number of samples to display
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get mask and prediction
        mask = masks[i, 0].cpu().numpy()
        pred = torch.sigmoid(predictions[i, 0]).cpu().detach().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_binary, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


def get_device():
    """
    Get the best available device (CUDA, MPS, or CPU)
    
    Returns:
        torch.device: Device to use for training/inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the dataset and transformations
    print("Testing utility functions...")
    
    # Test transforms
    train_transform = get_train_transform(256)
    val_transform = get_val_transform(256)
    print("✓ Transforms created successfully")
    
    # Test metrics with dummy data
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    acc = pixel_accuracy(pred, target)
    
    print(f"✓ Metrics computed - Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}")
    
    # Test loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    combined_loss = CombinedLoss()
    
    loss_bce = bce_loss(pred, target)
    loss_dice = dice_loss(pred, target)
    loss_combined = combined_loss(pred, target)
    
    print(f"✓ Loss functions work - BCE: {loss_bce:.4f}, Dice: {loss_dice:.4f}, Combined: {loss_combined:.4f}")
    
    # Test device detection
    device = get_device()
    
    print("\n✓ All utility functions tested successfully!")
