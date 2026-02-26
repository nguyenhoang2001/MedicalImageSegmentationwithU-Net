"""
Inference Script for Medical Image Segmentation with U-Net

This script handles:
- Single image prediction
- Batch image prediction
- Visualization of results
- Saving predictions
- Optional ground truth comparison
"""

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

from models.unet import UNet
from utils import get_device, dice_coefficient, iou_score


def get_inference_transform(image_size=256):
    """
    Get preprocessing transform for inference

    Args:
        image_size (int): Size to resize images

    Returns:
        albumentations.Compose: Preprocessing pipeline
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def load_model(model_path, device, bilinear=False):
    """
    Load trained model from checkpoint

    Args:
        model_path (str): Path to model checkpoint
        device: Device to load model on
        bilinear (bool): Whether model uses bilinear upsampling

    Returns:
        model: Loaded U-Net model
    """
    print(f"Loading model from: {model_path}")

    model = UNet(n_channels=3, n_classes=1, bilinear=bilinear)
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if "best_dice" in checkpoint:
        print(f"Model's best training Dice score: {checkpoint['best_dice']:.4f}")
    if "epoch" in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")

    return model


def load_image(image_path):
    """
    Load and preprocess a single image

    Args:
        image_path (str): Path to image file

    Returns:
        np.ndarray: Image as numpy array (H, W, C)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def predict_single_image(model, image, transform, device, threshold=0.5):
    """
    Predict mask for a single image

    Args:
        model: U-Net model
        image (np.ndarray): Input image (H, W, C)
        transform: Preprocessing transform
        device: Device to run inference on
        threshold (float): Threshold for binary prediction

    Returns:
        tuple: (predicted_mask, probability_map)
    """
    # Apply transform
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
        prediction = (probability > threshold).astype(np.uint8)

    return prediction, probability


def visualize_prediction(
    image,
    prediction,
    probability=None,
    mask=None,
    save_path=None,
    show_probability=True,
):
    """
    Visualize prediction results

    Args:
        image (np.ndarray): Original image
        prediction (np.ndarray): Binary prediction
        probability (np.ndarray, optional): Probability map
        mask (np.ndarray, optional): Ground truth mask
        save_path (str, optional): Path to save visualization
        show_probability (bool): Whether to show probability map
    """
    num_plots = 2 if mask is None else 3
    if show_probability and probability is not None:
        num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Original image
    axes[plot_idx].imshow(image)
    axes[plot_idx].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[plot_idx].axis("off")
    plot_idx += 1

    # Ground truth (if available)
    if mask is not None:
        axes[plot_idx].imshow(mask, cmap="gray")
        axes[plot_idx].set_title("Ground Truth", fontsize=14, fontweight="bold")
        axes[plot_idx].axis("off")
        plot_idx += 1

    # Prediction
    axes[plot_idx].imshow(prediction, cmap="gray")
    axes[plot_idx].set_title("Prediction", fontsize=14, fontweight="bold")
    axes[plot_idx].axis("off")
    plot_idx += 1

    # Probability map (if requested)
    if show_probability and probability is not None:
        im = axes[plot_idx].imshow(probability, cmap="jet", vmin=0, vmax=1)
        axes[plot_idx].set_title("Probability Map", fontsize=14, fontweight="bold")
        axes[plot_idx].axis("off")
        plt.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def overlay_prediction(image, prediction, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay prediction mask on original image

    Args:
        image (np.ndarray): Original image (H, W, C)
        prediction (np.ndarray): Binary prediction (H, W)
        alpha (float): Transparency of overlay
        color (tuple): RGB color for mask overlay

    Returns:
        np.ndarray: Image with overlay
    """
    # Resize prediction to match image size
    if prediction.shape[:2] != image.shape[:2]:
        prediction = cv2.resize(
            prediction,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Create colored mask
    mask_colored = np.zeros_like(image)
    mask_colored[prediction > 0] = color

    # Blend with original image
    overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)

    return overlay


def save_prediction(prediction, save_path):
    """
    Save prediction mask as image

    Args:
        prediction (np.ndarray): Binary prediction
        save_path (str): Path to save mask
    """
    # Convert to 0-255 range
    mask_image = (prediction * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_image)
    print(f"Prediction mask saved to: {save_path}")


def process_single_image(args, model, device, transform):
    """
    Process a single image

    Args:
        args: Command line arguments
        model: U-Net model
        device: Device to run inference on
        transform: Preprocessing transform
    """
    print("\n" + "=" * 60)
    print("Single Image Inference")
    print("=" * 60 + "\n")

    # Load image
    print(f"Loading image: {args.image_path}")
    image = load_image(args.image_path)
    original_size = image.shape[:2]
    print(f"Image size: {original_size[1]}x{original_size[0]}")

    # Load mask if available
    mask = None
    if args.mask_path and os.path.exists(args.mask_path):
        print(f"Loading ground truth mask: {args.mask_path}")
        mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

    # Predict
    print("Running inference...")
    prediction, probability = predict_single_image(
        model, image, transform, device, args.threshold
    )

    # Resize to original size
    prediction_resized = cv2.resize(
        prediction.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    probability_resized = cv2.resize(
        probability,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Calculate metrics if mask is available
    if mask is not None:
        # Prepare tensors for metric calculation
        pred_tensor = (
            torch.from_numpy(prediction_resized).unsqueeze(0).unsqueeze(0).float()
        )
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

        # Convert prediction to logits (inverse sigmoid)
        pred_logits = torch.log(pred_tensor / (1 - pred_tensor + 1e-7))

        dice = dice_coefficient(pred_logits, mask_tensor)
        iou = iou_score(pred_logits, mask_tensor)

        print(f"\nMetrics:")
        print(f"  Dice Score: {dice:.4f}")
        print(f"  IoU Score:  {iou:.4f}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get base filename
    base_name = Path(args.image_path).stem

    # Save prediction mask
    if args.save_mask:
        mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
        save_prediction(prediction_resized, mask_path)

    # Save visualization
    viz_path = os.path.join(args.output_dir, f"{base_name}_visualization.png")
    visualize_prediction(
        image,
        prediction_resized,
        probability_resized,
        mask,
        save_path=viz_path,
        show_probability=args.show_probability,
    )

    # Save overlay
    if args.save_overlay:
        overlay = overlay_prediction(
            image, prediction_resized, alpha=args.overlay_alpha
        )
        overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Overlay saved to: {overlay_path}")

    print("\n" + "=" * 60)
    print("Inference complete! 🎉")
    print("=" * 60 + "\n")


def process_directory(args, model, device, transform):
    """
    Process all images in a directory

    Args:
        args: Command line arguments
        model: U-Net model
        device: Device to run inference on
        transform: Preprocessing transform
    """
    print("\n" + "=" * 60)
    print("Batch Image Inference")
    print("=" * 60 + "\n")

    # Get all image files
    image_dir = Path(args.image_dir)
    image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    image_files = [
        f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images in {args.image_dir}")

    if len(image_files) == 0:
        print("No images found!")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")

        # Load and predict
        image = load_image(str(image_path))
        prediction, probability = predict_single_image(
            model, image, transform, device, args.threshold
        )

        # Resize to original size
        original_size = image.shape[:2]
        prediction_resized = cv2.resize(
            prediction.astype(np.uint8),
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Save prediction mask
        base_name = image_path.stem
        mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
        save_prediction(prediction_resized, mask_path)

        # Save overlay if requested
        if args.save_overlay:
            overlay = overlay_prediction(
                image, prediction_resized, alpha=args.overlay_alpha
            )
            overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("\n" + "=" * 60)
    print(f"Batch inference complete! Processed {len(image_files)} images")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="U-Net Inference for Image Segmentation"
    )

    # Model parameters
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear upsampling (must match training)",
    )

    # Input parameters (choose one)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image_path", type=str, help="Path to single input image"
    )
    input_group.add_argument(
        "--image_dir", type=str, help="Path to directory containing multiple images"
    )

    # Optional ground truth
    parser.add_argument(
        "--mask_path",
        type=str,
        help="Path to ground truth mask (only for single image)",
    )

    # Inference parameters
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size to resize images for model input (default: 256)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary prediction (default: 0.5)",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save predictions (default: predictions)",
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=True,
        help="Save prediction mask (default: True)",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Save overlay of prediction on original image",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.5,
        help="Transparency of overlay (default: 0.5)",
    )
    parser.add_argument(
        "--show_probability",
        action="store_true",
        help="Show probability map in visualization",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if args.image_path and not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    if args.image_dir and not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    # Get device
    device = get_device()

    # Load model
    model = load_model(args.model_path, device, args.bilinear)

    # Get transform
    transform = get_inference_transform(args.image_size)

    # Process images
    if args.image_path:
        process_single_image(args, model, device, transform)
    else:
        process_directory(args, model, device, transform)


if __name__ == "__main__":
    main()
