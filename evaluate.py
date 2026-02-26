"""
Evaluation Script for Medical Image Segmentation with U-Net

This script handles:
- Loading trained model
- Evaluating on test dataset
- Computing comprehensive metrics
- Generating visualizations and reports
- Saving evaluation results
"""

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from models.unet import UNet
from utils import (
    SegmentationDataset,
    get_val_transform,
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    load_checkpoint,
    get_device,
    visualize_predictions
)


def compute_metrics_batch(predictions, targets):
    """
    Compute metrics for a batch of predictions
    
    Args:
        predictions (torch.Tensor): Predicted masks (B, 1, H, W)
        targets (torch.Tensor): Ground truth masks (B, 1, H, W)
    
    Returns:
        dict: Dictionary of metrics
    """
    dice = dice_coefficient(predictions, targets)
    iou = iou_score(predictions, targets)
    acc = pixel_accuracy(predictions, targets)
    
    # Compute precision, recall, and F1 score
    preds_binary = (torch.sigmoid(predictions) > 0.5).float()
    targets_binary = targets
    
    tp = (preds_binary * targets_binary).sum()
    fp = (preds_binary * (1 - targets_binary)).sum()
    fn = ((1 - preds_binary) * targets_binary).sum()
    tn = ((1 - preds_binary) * (1 - targets_binary)).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    
    return {
        'dice': dice,
        'iou': iou,
        'accuracy': acc,
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'specificity': specificity.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'fn': fn.item(),
        'tn': tn.item()
    }


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: U-Net model
        dataloader: Data loader for evaluation
        device: Device to evaluate on
    
    Returns:
        dict: Dictionary containing evaluation metrics and predictions
    """
    model.eval()
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'specificity': []
    }
    
    all_images = []
    all_masks = []
    all_predictions = []
    
    confusion_stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute metrics
            metrics = compute_metrics_batch(outputs, masks)
            
            # Store metrics
            for key in all_metrics.keys():
                all_metrics[key].append(metrics[key])
            
            # Update confusion matrix stats
            confusion_stats['tp'] += metrics['tp']
            confusion_stats['fp'] += metrics['fp']
            confusion_stats['fn'] += metrics['fn']
            confusion_stats['tn'] += metrics['tn']
            
            # Store samples for visualization (first batch only)
            if len(all_images) == 0:
                all_images = images.cpu()
                all_masks = masks.cpu()
                all_predictions = outputs.cpu()
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {key: np.std(values) for key, values in all_metrics.items()}
    
    results = {
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'all_metrics': all_metrics,
        'confusion_stats': confusion_stats,
        'sample_images': all_images,
        'sample_masks': all_masks,
        'sample_predictions': all_predictions
    }
    
    return results


def plot_confusion_matrix(confusion_stats, save_path):
    """
    Plot confusion matrix
    
    Args:
        confusion_stats (dict): Dictionary with TP, FP, FN, TN
        save_path (str): Path to save the plot
    """
    cm = np.array([
        [confusion_stats['tn'], confusion_stats['fp']],
        [confusion_stats['fn'], confusion_stats['tp']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_metrics_distribution(all_metrics, save_path):
    """
    Plot distribution of metrics across all samples
    
    Args:
        all_metrics (dict): Dictionary of metric lists
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metrics Distribution Across Test Samples', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lavender']
    
    for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        ax = axes[idx // 3, idx % 3]
        data = all_metrics[metric]
        
        ax.hist(data, bins=30, color=color, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.4f}')
        ax.set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics distribution plot saved to {save_path}")


def plot_metrics_summary(avg_metrics, std_metrics, save_path):
    """
    Plot summary bar chart of metrics
    
    Args:
        avg_metrics (dict): Dictionary of average metrics
        std_metrics (dict): Dictionary of standard deviations
        save_path (str): Path to save the plot
    """
    metrics = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'f1', 'specificity']
    values = [avg_metrics[m] for m in metrics]
    errors = [std_metrics[m] for m in metrics]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(metrics)), values, yerr=errors, 
                    capsize=5, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Summary', fontsize=14, fontweight='bold')
    plt.xticks(range(len(metrics)), [m.upper() for m in metrics], rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics summary plot saved to {save_path}")


def save_evaluation_report(results, save_path):
    """
    Save evaluation report as JSON and text file
    
    Args:
        results (dict): Evaluation results
        save_path (str): Path to save the report
    """
    report = {
        'average_metrics': {k: float(v) for k, v in results['avg_metrics'].items()},
        'std_metrics': {k: float(v) for k, v in results['std_metrics'].items()},
        'confusion_matrix': results['confusion_stats']
    }
    
    # Save JSON
    json_path = save_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Evaluation report (JSON) saved to {json_path}")
    
    # Save text report
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Model Evaluation Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Average Metrics:\n")
        f.write("-"*60 + "\n")
        for metric, value in results['avg_metrics'].items():
            std = results['std_metrics'][metric]
            f.write(f"  {metric.upper():<15}: {value:.4f} ± {std:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Confusion Matrix Statistics:\n")
        f.write("-"*60 + "\n")
        cm = results['confusion_stats']
        f.write(f"  True Positives  (TP): {cm['tp']:,.0f}\n")
        f.write(f"  False Positives (FP): {cm['fp']:,.0f}\n")
        f.write(f"  False Negatives (FN): {cm['fn']:,.0f}\n")
        f.write(f"  True Negatives  (TN): {cm['tn']:,.0f}\n")
        
        total = cm['tp'] + cm['fp'] + cm['fn'] + cm['tn']
        f.write(f"\n  Total Pixels: {total:,.0f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Evaluation report (TXT) saved to {save_path}")


def evaluate(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    # Get device
    device = get_device()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60 + "\n")
    
    # ==================== Load Model ====================
    print(f"Loading model from: {args.model_path}")
    
    # Initialize model
    model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    if 'best_dice' in checkpoint:
        print(f"Model's best training Dice score: {checkpoint['best_dice']:.4f}")
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    
    # ==================== Load Dataset ====================
    print(f"\nLoading test dataset from: {args.data_dir}")
    
    test_images_dir = os.path.join(args.data_dir, 'images')
    test_masks_dir = os.path.join(args.data_dir, 'masks')
    
    test_dataset = SegmentationDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        transform=get_val_transform(args.image_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # ==================== Evaluate ====================
    results = evaluate_model(model, test_loader, device)
    
    # ==================== Print Results ====================
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nAverage Metrics:")
    print("-"*60)
    for metric, value in results['avg_metrics'].items():
        std = results['std_metrics'][metric]
        print(f"  {metric.upper():<15}: {value:.4f} ± {std:.4f}")
    
    print("\n" + "="*60)
    print("Confusion Matrix Statistics:")
    print("-"*60)
    cm = results['confusion_stats']
    print(f"  True Positives  (TP): {cm['tp']:,.0f}")
    print(f"  False Positives (FP): {cm['fp']:,.0f}")
    print(f"  False Negatives (FN): {cm['fn']:,.0f}")
    print(f"  True Negatives  (TN): {cm['tn']:,.0f}")
    print("="*60 + "\n")
    
    # ==================== Save Results ====================
    print("Saving evaluation results...")
    
    # Save report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    save_evaluation_report(results, report_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_stats'], cm_path)
    
    # Plot metrics distribution
    dist_path = os.path.join(args.output_dir, 'metrics_distribution.png')
    plot_metrics_distribution(results['all_metrics'], dist_path)
    
    # Plot metrics summary
    summary_path = os.path.join(args.output_dir, 'metrics_summary.png')
    plot_metrics_summary(results['avg_metrics'], results['std_metrics'], summary_path)
    
    # Visualize predictions
    print("\nGenerating sample predictions...")
    fig = visualize_predictions(
        results['sample_images'],
        results['sample_masks'],
        results['sample_predictions'],
        num_samples=min(8, len(results['sample_images']))
    )
    viz_path = os.path.join(args.output_dir, 'sample_predictions.png')
    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Sample predictions saved to {viz_path}")
    
    print("\n" + "="*60)
    print("Evaluation complete! 🎉")
    print(f"All results saved to: {args.output_dir}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate U-Net Model for Image Segmentation')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--bilinear', action='store_true',
                        help='Use bilinear upsampling (must match training)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test data directory containing images/ and masks/ folders')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size to resize images (default: 256)')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results (default: evaluation_results)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    evaluate(args)


if __name__ == "__main__":
    main()
