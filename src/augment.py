import os
import cv2
import numpy as np
from pathlib import Path

# Image parameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3

# Input paths
TRAIN_PATH = '/home/volcani/Unet/data/plowing/train/'
MASK_PATH = '/home/volcani/Unet/data/plowing/mask_train/'

# Output paths
TRAIN_AUGMENT_PATH = '/home/volcani/Unet/data/plowing/train_augment/'
MASK_AUGMENT_PATH = '/home/volcani/Unet/data/plowing/mask_train_augment/'

# Create output directories if they don't exist
os.makedirs(TRAIN_AUGMENT_PATH, exist_ok=True)
os.makedirs(MASK_AUGMENT_PATH, exist_ok=True)

def horizontal_flip(image):
    """Flip image horizontally"""
    return cv2.flip(image, 1)

def vertical_flip(image):
    """Flip image vertically"""
    return cv2.flip(image, 0)

def rotate_90(image):
    """Rotate image 90 degrees clockwise"""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def rotate_180(image):
    """Rotate image 180 degrees"""
    return cv2.rotate(image, cv2.ROTATE_180)

def rotate_270(image):
    """Rotate image 270 degrees clockwise"""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def adjust_brightness(image, factor=1.2):
    """Adjust brightness (only for training images, not masks)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, factor=1.3):
    """Adjust contrast (only for training images, not masks)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = l.astype(np.float32)
    l = ((l / 255.0 - 0.5) * factor + 0.5) * 255.0
    l = np.clip(l, 0, 255).astype(np.uint8)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Define augmentation functions
augmentations = [
    ('hflip', horizontal_flip, True),
    ('vflip', vertical_flip, True),
    ('rot90', rotate_90, True),
    ('rot180', rotate_180, True),
    ('rot270', rotate_270, True),
    ('bright', lambda img: adjust_brightness(img, 1.2), False),
    ('dark', lambda img: adjust_brightness(img, 0.8), False),
    ('contrast', lambda img: adjust_contrast(img, 1.3), False),
]

def get_image_files(directory):
    """Get all image files from directory"""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG', '.JPEG'}
    files = []
    for file in os.listdir(directory):
        if Path(file).suffix in valid_extensions:
            files.append(file)
    return files

def build_file_mapping(train_files, mask_files):
    """Build mapping between train and mask files"""
    # Create lookup by stem 
    mask_dict = {}
    for mask_file in mask_files:
        stem = Path(mask_file).stem
        mask_dict[stem] = mask_file

    pairs = []
    unmatched = []
    
    for train_file in train_files:
        stem = Path(train_file).stem
        if stem in mask_dict:
            pairs.append((train_file, mask_dict[stem]))
        else:
            unmatched.append(train_file)
    
    return pairs, unmatched

def augment_dataset():
    """Main function to augment dataset"""
    print("="*70)
    print("STARTING AUGMENTATION")
    print("="*70)
    
    print("\nScanning directories...")
    train_files = get_image_files(TRAIN_PATH)
    mask_files = get_image_files(MASK_PATH)
    
    print(f"Found {len(train_files)} training files")
    print(f"Found {len(mask_files)} mask files")
    
    # Show sample files
    print("\nSample training files (first 5):")
    for f in sorted(train_files)[:5]:
        print(f"  {f}")
    
    print("\nSample mask files (first 5):")
    for f in sorted(mask_files)[:5]:
        print(f"  {f}")

    print("\nMatching files...")
    pairs, unmatched = build_file_mapping(train_files, mask_files)
    
    print(f"Successfully matched: {len(pairs)} pairs")
    print(f"Unmatched train files: {len(unmatched)}")
    
    if unmatched and len(unmatched) <= 10:
        print("\nUnmatched files:")
        for f in unmatched:
            print(f"  {f}")
    elif unmatched:
        print(f"\nFirst 10 unmatched files:")
        for f in unmatched[:10]:
            print(f"  {f}")
    
    if len(pairs) == 0:
        print("\nERROR: No matching pairs found!")
        print("Please check that train and mask files have the same names.")
        return

    print(f"\nFirst 3 matched pairs:")
    for train_f, mask_f in sorted(pairs)[:3]:
        print(f"  {train_f} <-> {mask_f}")
    
    # Starting ID for augmented images
    augment_id = 423
    total_augmentations = 0
    
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(pairs)} IMAGE PAIRS")
    print(f"{'='*70}\n")
    
    # Process each pair
    for idx, (train_file, mask_file) in enumerate(sorted(pairs), 1):
        # Read images
        train_img_path = os.path.join(TRAIN_PATH, train_file)
        mask_img_path = os.path.join(MASK_PATH, mask_file)
        
        train_img = cv2.imread(train_img_path)
        mask_img = cv2.imread(mask_img_path)
        
        if train_img is None:
            print(f"[{idx}/{len(pairs)}] ERROR: Could not read {train_file}, skipping...")
            continue
            
        if mask_img is None:
            print(f"[{idx}/{len(pairs)}] ERROR: Could not read {mask_file}, skipping...")
            continue
        
        # Resize if necessary
        if train_img.shape[0] != IMG_HEIGHT or train_img.shape[1] != IMG_WIDTH:
            train_img = cv2.resize(train_img, (IMG_WIDTH, IMG_HEIGHT))
        if mask_img.shape[0] != IMG_HEIGHT or mask_img.shape[1] != IMG_WIDTH:
            mask_img = cv2.resize(mask_img, (IMG_WIDTH, IMG_HEIGHT))
        
        print(f"[{idx}/{len(pairs)}] Processing: {train_file} <-> {mask_file}")
        
        for aug_name, aug_func, apply_to_both in augmentations:
            # Augment training image
            augmented_train = aug_func(train_img.copy())
            
            # Augment mask 
            if apply_to_both:
                augmented_mask = aug_func(mask_img.copy())
            else:
                # For brightness/contrast, keep original mask
                augmented_mask = mask_img.copy()
            
            # Save augmented images
            train_output_name = f"{augment_id:03d}.png"
            mask_output_name = f"{augment_id:03d}.png"
            
            train_output_path = os.path.join(TRAIN_AUGMENT_PATH, train_output_name)
            mask_output_path = os.path.join(MASK_AUGMENT_PATH, mask_output_name)
            
            cv2.imwrite(train_output_path, augmented_train)
            cv2.imwrite(mask_output_path, augmented_mask)
            
            augment_id += 1
            total_augmentations += 1
        
        # Show progress every 50 images
        if idx % 50 == 0:
            print(f"  ... processed {idx} pairs, created {total_augmentations} augmentations so far")
    
    print(f"\n{'='*70}")
    print("AUGMENTATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Processed pairs: {len(pairs)}")
    print(f"Total augmented images created: {total_augmentations}")
    print(f"Starting ID: 423")
    print(f"Final ID: {augment_id - 1}")
    print(f"Output directories:")
    print(f"  Train: {TRAIN_AUGMENT_PATH}")
    print(f"  Masks: {MASK_AUGMENT_PATH}")
    print(f"{'='*70}")

if __name__ == "__main__":
    augment_dataset()

