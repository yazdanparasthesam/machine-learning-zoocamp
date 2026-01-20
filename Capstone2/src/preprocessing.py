"""
Shared preprocessing logic for training and inference.

Responsibilities:
- Image transformations (train / validation)
- Stratified train/val/test split
- Split statistics logging

Raw dataset structure:
data/raw/
├── mask/
└── no_mask/

Generated dataset structure:
data/processed/
├── train/
│   ├── mask/
│   └── no_mask/
├──val/
|   ├── mask/
|   └── no_mask/
├── test/
|   ├── mask/
|   └── no_mask/
"""

import random
import shutil
import logging
from pathlib import Path
from torchvision import transforms

# =========================
# Configuration
# =========================

RANDOM_SEED = 42
CLASSES = ["mask", "no_mask"]

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
LOG_FILE = "logs/preprocessing.log"

# =========================
# Logging Setup
# =========================

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Image Transformations
# =========================

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# =========================
# Dataset Splitting
# =========================

def create_dataset_splits(
    raw_data_dir: str = RAW_DATA_DIR,
    output_dir: str = PROCESSED_DATA_DIR,
) -> None:
    """
    Create stratified train/val/test splits.

    Structure:
    data/processed/
    ├── train/
    ├── val/
    └── test/
    """

    random.seed(RANDOM_SEED)

    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)

    # Create directories
    for split in SPLITS.keys():
        for cls in CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    stats = {}

    for cls in CLASSES:
        images = list((raw_data_dir / cls).glob("*"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLITS["train"])
        n_val = int(n_total * SPLITS["val"])

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        split_map = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }

        stats[cls] = {}

        for split, imgs in split_map.items():
            for img in imgs:
                shutil.copy(img, output_dir / split / cls / img.name)

            stats[cls][split] = len(imgs)

    log_split_stats(stats)
    print("✅ Stratified train/val/test split completed")


# =========================
# Split Statistics Logging
# =========================

def log_split_stats(stats: dict) -> None:
    """
    Log dataset split statistics
    """
    logging.info("Dataset split statistics:")

    for cls, splits in stats.items():
        logging.info(f"Class: {cls}")
        for split, count in splits.items():
            logging.info(f"  {split}: {count} images")

# =========================
# Script Entry Point
# =========================

if __name__ == "__main__":
    create_dataset_splits()