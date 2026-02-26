"""
Generate Synthetic Dataset for Testing

This script creates a small synthetic dataset with random shapes
to test the U-Net training, evaluation, and inference pipeline.

NOTE: This is for testing purposes only. For real applications,
use actual medical imaging datasets.
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse


def generate_random_shape_mask(size=256):
    """
    Generate a random shape (circle, ellipse, or rectangle) mask

    Args:
        size (int): Image size

    Returns:
        np.ndarray: Binary mask (0 or 255)
    """
    mask = np.zeros((size, size), dtype=np.uint8)

    # Random shape type
    shape_type = np.random.choice(["circle", "ellipse", "rectangle", "polygon"])

    # Random position
    center_x = np.random.randint(size // 4, 3 * size // 4)
    center_y = np.random.randint(size // 4, 3 * size // 4)

    if shape_type == "circle":
        radius = np.random.randint(30, size // 4)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    elif shape_type == "ellipse":
        axes_length = (
            np.random.randint(30, size // 4),
            np.random.randint(30, size // 4),
        )
        angle = np.random.randint(0, 180)
        cv2.ellipse(mask, (center_x, center_y), axes_length, angle, 0, 360, 255, -1)

    elif shape_type == "rectangle":
        width = np.random.randint(50, size // 3)
        height = np.random.randint(50, size // 3)
        x1 = max(0, center_x - width // 2)
        y1 = max(0, center_y - height // 2)
        x2 = min(size, center_x + width // 2)
        y2 = min(size, center_y + height // 2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    else:  # polygon
        num_points = np.random.randint(5, 8)
        points = []
        for _ in range(num_points):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.randint(50, size // 4)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    return mask


def generate_image_from_mask(mask, size=256):
    """
    Generate a synthetic image based on the mask

    Args:
        mask (np.ndarray): Binary mask
        size (int): Image size

    Returns:
        np.ndarray: RGB image
    """
    # Create base image with random background
    background_color = np.random.randint(50, 150, 3)
    image = np.ones((size, size, 3), dtype=np.uint8) * background_color

    # Object color (different from background)
    object_color = np.random.randint(100, 255, 3)

    # Apply object color where mask is white
    image[mask > 0] = object_color

    # Add some noise
    noise = np.random.normal(0, 20, (size, size, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Add some blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image


def generate_dataset(output_dir, num_train=100, num_val=20, num_test=20, size=256):
    """
    Generate synthetic dataset with train, validation, and test splits

    Args:
        output_dir (str): Output directory
        num_train (int): Number of training samples
        num_val (int): Number of validation samples
        num_test (int): Number of test samples
        size (int): Image size
    """
    print("Generating Synthetic Dataset for Testing")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Training samples: {num_train}")
    print(f"Validation samples: {num_val}")
    print(f"Test samples: {num_test}")
    print(f"Image size: {size}x{size}")
    print("=" * 60 + "\n")

    # Create directories
    splits = {"train": num_train, "val": num_val, "test": num_test}

    for split, num_samples in splits.items():
        images_dir = os.path.join(output_dir, split, "images")
        masks_dir = os.path.join(output_dir, split, "masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        print(f"Generating {split} set ({num_samples} samples)...")

        for i in tqdm(range(num_samples)):
            # Generate mask
            mask = generate_random_shape_mask(size)

            # Generate image from mask
            image = generate_image_from_mask(mask, size)

            # Save files
            img_filename = f"{split}_{i:04d}.jpg"
            mask_filename = f"{split}_{i:04d}.png"

            cv2.imwrite(os.path.join(images_dir, img_filename), image)
            cv2.imwrite(os.path.join(masks_dir, mask_filename), mask)

    print("\n" + "=" * 60)
    print("✓ Dataset generation complete!")
    print("=" * 60)
    print("\nDirectory structure:")
    print(f"{output_dir}/")
    print("├── train/")
    print("│   ├── images/ ({} images)".format(num_train))
    print("│   └── masks/ ({} masks)".format(num_train))
    print("├── val/")
    print("│   ├── images/ ({} images)".format(num_val))
    print("│   └── masks/ ({} masks)".format(num_val))
    print("└── test/")
    print("    ├── images/ ({} images)".format(num_test))
    print("    └── masks/ ({} masks)".format(num_test))

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print(
        f"1. Train: python train.py --data_dir {output_dir}/train --epochs 5 --batch_size 4"
    )
    print(
        f"2. Evaluate: python evaluate.py --model_path checkpoints/best_model.pth --data_dir {output_dir}/test"
    )
    print(
        f"3. Predict: python inference.py --model_path checkpoints/best_model.pth --image_path {output_dir}/test/images/test_0000.jpg"
    )
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Dataset for Testing"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=100,
        help="Number of training samples (default: 100)",
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=20,
        help="Number of validation samples (default: 20)",
    )
    parser.add_argument(
        "--num_test", type=int, default=20, help="Number of test samples (default: 20)"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="Image size (default: 256)"
    )

    args = parser.parse_args()

    generate_dataset(
        args.output_dir, args.num_train, args.num_val, args.num_test, args.size
    )


if __name__ == "__main__":
    main()
