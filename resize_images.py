#!/usr/bin/env python3
"""
Resize all images in specified directories

Usage:
    python resize_images.py --input_dir ~/field_images/plowing --target_size 512 --backup
"""

import os
import argparse
import cv2
import numpy as np
from glob import glob
import shutil
from tqdm import tqdm


def resize_image(image_path, target_size, maintain_aspect=False):
    """
    Resize single image

    Args:
        image_path: Path to the image
        target_size: Target size (if maintain_aspect=False, this is both width and height)
                     (if maintain_aspect=True, this is the max dimension)
        maintain_aspect: If True, maintain aspect ratio

    Returns:
        Resized image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    h, w = img.shape[:2]

    if maintain_aspect:
        # Maintain aspect ratio - resize so largest dimension is target_size
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
    else:
        # Square resize
        new_h = target_size
        new_w = target_size

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def resize_directory(input_dir, target_size, maintain_aspect=False, backup=True, dry_run=False, rename_sequential=True):
    """
    Resize all images in a directory 

    Args:
        input_dir: Directory containing images
        target_size: Target size for resizing
        maintain_aspect: Whether to maintain aspect ratio
        backup: Whether to create backup before resizing
        dry_run: If True, just show what would be done without actually doing it
        rename_sequential: If True, rename images sequentially (1.jpg, 2.jpg, etc.)
    """
    # Find all images
    image_paths = sorted(glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) +
                        glob(os.path.join(input_dir, "*.[pP][nN][gG]")))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return 0

    print(f"\nProcessing directory: {input_dir}")
    print(f"Found {len(image_paths)} images")

    if dry_run:
        print("DRY RUN - no changes will be made")
        # Show first image info
        img = cv2.imread(image_paths[0])
        if img is not None:
            h, w = img.shape[:2]
            print(f"Current size example: {w}x{h}")
            if maintain_aspect:
                if h > w:
                    new_h = target_size
                    new_w = int(w * (target_size / h))
                else:
                    new_w = target_size
                    new_h = int(h * (target_size / w))
            else:
                new_h = new_w = target_size
            print(f"New size: {new_w}x{new_h}")
        if rename_sequential:
            print("Images will be renamed sequentially (1.jpg, 2.jpg, etc.)")
        return 0

    # Create backup if requested
    if backup:
        backup_dir = input_dir + "_backup"
        if not os.path.exists(backup_dir):
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(input_dir, backup_dir)
        else:
            print(f"Backup already exists: {backup_dir}")

    # Resize images
    success_count = 0
    temp_files = []  # Track temporary files for renaming

    iterator = tqdm(image_paths, desc="Resizing images")

    # First pass: resize and save to temporary names
    for i, image_path in enumerate(iterator):
        try:
            resized = resize_image(image_path, target_size, maintain_aspect)

            if rename_sequential:
                # Get file extension from original file
                _, ext = os.path.splitext(image_path)
                # Create temporary filename to avoid conflicts
                temp_path = os.path.join(input_dir, f"temp_{i}{ext}")
                cv2.imwrite(temp_path, resized)
                temp_files.append((temp_path, i + 65, ext))
            else:
                cv2.imwrite(image_path, resized)

            success_count += 1
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")

    # Second pass: rename temp files to sequential names if requested
    if rename_sequential and temp_files:
        print("\nRenaming files sequentially...")
        # Remove original files
        for image_path in image_paths:
            if os.path.exists(image_path):
                os.remove(image_path)

        # Rename temp files to sequential names
        for temp_path, num, ext in tqdm(temp_files, desc="Renaming"):
            final_path = os.path.join(input_dir, f"{num}{ext}")
            os.rename(temp_path, final_path)

    return success_count


def main():
    parser = argparse.ArgumentParser(description="Resize images in directories")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Root directory containing train/test/valid folders")
    parser.add_argument("--target_size", type=int, default=512,
                       help="Target size for images (default: 512)")
    parser.add_argument("--maintain_aspect", action="store_true",
                       help="Maintain aspect ratio (default: square resize)")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup before resizing")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be done without doing it")
    parser.add_argument("--rename_sequential", action="store_true", default=True,
                       help="Rename images sequentially starting from 1 (default: True)")
    parser.add_argument("--no_rename", action="store_true",
                       help="Keep original filenames (disable sequential renaming)")
    parser.add_argument("--folders", type=str, nargs="+", default=["train", "test", "valid"],
                       help="Folders to process (default: train test valid)")

    args = parser.parse_args()

    # Determine if renaming should happen
    rename_sequential = not args.no_rename

    print("=" * 70)
    print("Image Resizing Tool")
    print("=" * 70)
    print(f"Root directory: {args.input_dir}")
    print(f"Target size: {args.target_size}x{args.target_size}" if not args.maintain_aspect else f"Target max dimension: {args.target_size}")
    print(f"Maintain aspect ratio: {args.maintain_aspect}")
    print(f"Create backup: {args.backup}")
    print(f"Sequential renaming: {rename_sequential}")
    print(f"Folders to process: {', '.join(args.folders)}")
    print("=" * 70)

    if not args.dry_run:
        response = input("\nThis will REPLACE all images in the specified folders. Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return

    total_processed = 0

    # Check if we should process the input_dir directly (no subfolders)
    if args.folders == ["."] or not any(os.path.exists(os.path.join(args.input_dir, f)) for f in args.folders):
        # Process input_dir directly
        if os.path.exists(args.input_dir):
            count = resize_directory(
                args.input_dir,
                args.target_size,
                args.maintain_aspect,
                args.backup,
                args.dry_run,
                rename_sequential
            )
            total_processed += count
        else:
            print(f"\nError: Directory does not exist: {args.input_dir}")
    else:
        # Process subfolders
        for folder in args.folders:
            folder_path = os.path.join(args.input_dir, folder)

            if not os.path.exists(folder_path):
                print(f"\nWarning: Folder does not exist: {folder_path}")
                continue

            count = resize_directory(
                folder_path,
                args.target_size,
                args.maintain_aspect,
                args.backup,
                args.dry_run,
                rename_sequential
            )
            total_processed += count

    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE")
    else:
        print("Resizing Complete")
        print(f"Total images processed: {total_processed}")
        if args.backup:
            print("\nBackups created with '_backup' suffix")
    print("=" * 70)


if __name__ == "__main__":
    main()
