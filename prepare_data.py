"""
Data Preparation Script

This script helps organize your dataset into the required structure:
data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── masks/
    ├── img1.png
    ├── img2.png
    └── ...

Usage:
    python prepare_data.py --source_images /path/to/images --source_masks /path/to/masks --output data/
"""

import os
import shutil
import argparse
from pathlib import Path


def prepare_dataset(source_images, source_masks, output_dir):
    """
    Organize images and masks into the required directory structure

    Args:
        source_images (str): Path to directory containing source images
        source_masks (str): Path to directory containing source masks
        output_dir (str): Output directory for organized dataset
    """
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"Creating dataset structure in: {output_dir}")
    print(f"  Images will be copied to: {images_dir}")
    print(f"  Masks will be copied to: {masks_dir}")

    # Copy images
    image_files = list(Path(source_images).glob("*"))
    image_files = [
        f
        for f in image_files
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    ]

    print(f"\nCopying {len(image_files)} images...")
    for img_file in image_files:
        dest = os.path.join(images_dir, img_file.name)
        shutil.copy2(img_file, dest)

    # Copy masks
    mask_files = list(Path(source_masks).glob("*"))
    mask_files = [
        f
        for f in mask_files
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff"]
    ]

    print(f"Copying {len(mask_files)} masks...")
    for mask_file in mask_files:
        dest = os.path.join(masks_dir, mask_file.name)
        shutil.copy2(mask_file, dest)

    print("\n✓ Dataset preparation complete!")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total masks: {len(mask_files)}")

    if len(image_files) != len(mask_files):
        print("\n⚠️  Warning: Number of images and masks don't match!")
        print("   Please ensure each image has a corresponding mask.")

    print(f"\nYou can now train the model using:")
    print(f"  python train.py --data_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")

    parser.add_argument(
        "--source_images",
        type=str,
        required=True,
        help="Path to directory containing source images",
    )
    parser.add_argument(
        "--source_masks",
        type=str,
        required=True,
        help="Path to directory containing source masks",
    )
    parser.add_argument(
        "--output", type=str, default="data", help="Output directory (default: data)"
    )

    args = parser.parse_args()

    # Validate input directories
    if not os.path.exists(args.source_images):
        raise ValueError(f"Source images directory not found: {args.source_images}")

    if not os.path.exists(args.source_masks):
        raise ValueError(f"Source masks directory not found: {args.source_masks}")

    prepare_dataset(args.source_images, args.source_masks, args.output)


if __name__ == "__main__":
    main()
