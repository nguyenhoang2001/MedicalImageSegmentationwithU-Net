"""
Training Script for Medical Image Segmentation with U-Net

This script handles:
- Loading and splitting the dataset
- Training loop with validation
- Learning rate scheduling
- Checkpoint saving
- Progress tracking and logging
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.unet import UNet
from utils import (
    SegmentationDataset,
    get_train_transform,
    get_val_transform,
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    DiceLoss,
    CombinedLoss,
    save_checkpoint,
    get_device,
    count_parameters,
    visualize_predictions
)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch
    
    Args:
        model: U-Net model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        dict: Training metrics (loss, dice, iou, accuracy)
    """
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)
        
        # Update running metrics
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        running_acc += acc
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': running_loss / num_batches,
        'dice': running_dice / num_batches,
        'iou': running_iou / num_batches,
        'accuracy': running_acc / num_batches
    }
    
    return metrics


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model
    
    Args:
        model: U-Net model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        dict: Validation metrics (loss, dice, iou, accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [VAL]  ')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            acc = pixel_accuracy(outputs, masks)
            
            # Update running metrics
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            running_acc += acc
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    metrics = {
        'loss': running_loss / num_batches,
        'dice': running_dice / num_batches,
        'iou': running_iou / num_batches,
        'accuracy': running_acc / num_batches
    }
    
    return metrics


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str): Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice Score
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Validation')
    axes[0, 1].set_title('Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU Score
    axes[1, 0].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_iou'], 'r-', label='Validation')
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Accuracy
    axes[1, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1, 1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")


def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Medical Image Segmentation with U-Net")
    print("="*60 + "\n")
    
    # ==================== Dataset ====================
    print("Loading dataset...")
    
    # Prepare data directories
    train_images_dir = os.path.join(args.data_dir, 'images')
    train_masks_dir = os.path.join(args.data_dir, 'masks')
    
    # Create dataset with training transforms
    full_dataset = SegmentationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=get_train_transform(args.image_size)
    )
    
    # Split into train and validation
    train_size = int((1 - args.val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = get_val_transform(args.image_size)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # ==================== Model ====================
    print(f"\nInitializing U-Net model...")
    model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # ==================== Loss & Optimizer ====================
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCE Loss")
    elif args.loss == 'dice':
        criterion = DiceLoss()
        print("Using Dice Loss")
    elif args.loss == 'combined':
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        print("Using Combined Loss (BCE + Dice)")
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    if args.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        print("Using ReduceLROnPlateau scheduler")
    
    # ==================== Training Loop ====================
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60 + "\n")
    
    best_dice = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Learning rate scheduling
        if args.scheduler:
            scheduler.step(val_metrics['dice'])
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': args
            }
            save_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(checkpoint, save_path)
            print(f"  ✓ New best model saved! (Dice: {best_dice:.4f})")
        
        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': args
            }
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(checkpoint, save_path)
        
        print()
    
    # ==================== Save Final Results ====================
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print("="*60 + "\n")
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_dice': best_dice,
        'args': args
    }
    save_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(final_checkpoint, save_path)
    
    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Visualize some predictions
    print("Generating sample predictions...")
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images)
        
        fig = visualize_predictions(images, masks, predictions, num_samples=4)
        viz_path = os.path.join(args.checkpoint_dir, 'sample_predictions.png')
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {viz_path}")
    
    print("\nAll done! 🎉")


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for Image Segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory containing images/ and masks/ folders')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images (default: 256)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    
    # Model parameters
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling instead of transpose convolution')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['bce', 'dice', 'combined'],
                        help='Loss function (default: combined)')
    parser.add_argument('--scheduler', action='store_true',
                        help='Use learning rate scheduler')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
