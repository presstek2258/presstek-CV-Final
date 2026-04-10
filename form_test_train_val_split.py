# SCRIPT WRITTEN BY GEMINI 3 PRO
import os
import shutil
import random

# --- Configuration ---
# Your unzipped folders
SOURCE_IMAGES = "images"
SOURCE_LABELS = "labels"

# Where you want the final YOLO dataset built
OUTPUT_DIR = "custom_dataset"

# Standard split ratios (70% Train, 20% Validation, 10% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
# Test ratio automatically takes the remaining 10%

def create_yolo_directories():
    """Creates the YOLOv8/11 standard directory tree."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def process_split(file_list, split_name):
    """Copies a list of images and their matching .txt labels to the target split folder."""
    print(f"Processing {split_name} split ({len(file_list)} images)...")
    
    missing_labels = 0
    for img_filename in file_list:
        # 1. Copy the Image
        src_img_path = os.path.join(SOURCE_IMAGES, img_filename)
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split_name, img_filename)
        shutil.copy(src_img_path, dst_img_path)

        # 2. Find and copy the matching label
        # Replaces .jpg / .png with .txt
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        src_label_path = os.path.join(SOURCE_LABELS, label_filename)
        dst_label_path = os.path.join(OUTPUT_DIR, 'labels', split_name, label_filename)

        # Label Studio only generates a .txt file if you actually drew a box.
        # If an image has no objects in it, it won't have a label file, which is valid in YOLO.
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            missing_labels += 1
            
    if missing_labels > 0:
        print(f"  -> Note: {missing_labels} background images (no labels) included in {split_name}.")

def main():
    print("=== YOLO Dataset Splitter ===")
    
    if not os.path.exists(SOURCE_IMAGES) or not os.path.exists(SOURCE_LABELS):
        print("Error: Could not find 'images' or 'labels' folder. Are you running this in the right directory?")
        return

    create_yolo_directories()

    # Get all images and shuffle them randomly
    all_images = [f for f in os.listdir(SOURCE_IMAGES) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42) # Keeps the shuffle consistent if you run it multiple times
    random.shuffle(all_images)
    
    total_images = len(all_images)
    print(f"Found {total_images} total images.")

    # Calculate index breakpoints
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    # Slice the list
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]

    # Execute the copying
    process_split(train_images, 'train')
    process_split(val_images, 'val')
    process_split(test_images, 'test')

    print(f"\nSuccess! Your dataset is ready in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
