import os
import shutil
import random
from tqdm import tqdm

# Define paths
DATASET_PATH = "data/original"
OUTPUT_DIR = "data"

# Split percentages
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Ensure the output directories exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# Split dataset
for class_folder in tqdm(os.listdir(DATASET_PATH), desc="Splitting dataset"):
    class_path = os.path.join(DATASET_PATH, class_folder)

    if not os.path.isdir(class_path):
        continue

    # Get all image filenames
    images = os.listdir(class_path)
    random.shuffle(images)  # Shuffle before splitting

    # Compute split indices
    total_images = len(images)
    train_split = int(total_images * TRAIN_RATIO)
    val_split = int(total_images * (TRAIN_RATIO + VAL_RATIO))

    # Assign images to each set
    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    # Copy images to respective directories
    for split, files in splits.items():
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_folder)
        os.makedirs(split_class_dir, exist_ok=True)

        for file in files:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(split_class_dir, file)
            shutil.copy(src_path, dest_path)

print("Dataset successfully split into train, val, and test sets!")