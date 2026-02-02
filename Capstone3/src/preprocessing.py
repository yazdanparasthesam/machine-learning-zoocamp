# src/preprocessing.py
# Cleaning + train/val/test split + saving processed CSVs + logging

import re
import string
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# Logging Setup
# =========================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

#logging.basicConfig(
#    filename=LOG_DIR / "preprocessing.log",
#    level=logging.INFO,
#    format="%(asctime)s - %(levelname)s - %(message)s",
#)

logger = logging.getLogger(__name__)


# =========================
# Text Cleaning
# =========================

def clean_text(text: str) -> str:
    """
    Basic text cleaning for NLP:
    - lowercase
    - remove URLs
    - remove digits
    - remove punctuation
    - strip whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text


# =========================
# Dataset Processing
# =========================

def load_and_split_dataset(
    raw_dir: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Load Fake.csv and True.csv, merge them, clean text,
    split into train/val/test, and save CSV files.

    Raw structure:
        data/raw/
        ├── Fake.csv
        └── True.csv

    Output structure:
        data/processed/
        ├── train.csv
        ├── val.csv
        └── test.csv
    """

    logger.info("Starting preprocessing pipeline")

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load raw datasets
    # --------------------------------------------------
    logger.info("Loading raw datasets")
    fake_df = pd.read_csv(raw_dir / "Fake.csv")
    real_df = pd.read_csv(raw_dir / "True.csv")

    fake_df["label"] = 0   # fake news
    real_df["label"] = 1   # real news

    # --------------------------------------------------
    # Combine datasets
    # --------------------------------------------------
    df = pd.concat([fake_df, real_df], ignore_index=True)
    logger.info(f"Combined dataset size: {len(df)}")

    if "text" not in df.columns:
        logger.error("Missing 'text' column in dataset")
        raise ValueError("Dataset must contain a 'text' column")
        
    # --------------------------------------------------
    # Enforce label type (CRITICAL)
    # --------------------------------------------------
    df["label"] = df["label"].astype("int64")

    # Safety check
    assert set(df["label"].unique()) <= {0, 1}, "Invalid labels detected"
 

    # --------------------------------------------------
    # Clean text
    # --------------------------------------------------
    logger.info("Cleaning text data")
    df["text_clean"] = df["text"].apply(clean_text)

    # --------------------------------------------------
    # Train / Validation / Test split
    # --------------------------------------------------
    logger.info("Splitting dataset into train/val/test")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    val_relative_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    # --------------------------------------------------
    # Ensure label dtype after split
    # --------------------------------------------------
    for df_ in (train_df, val_df, test_df):
        df_["label"] = df_["label"].astype("int64")


    # --------------------------------------------------
    # Save processed files
    # --------------------------------------------------
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # --------------------------------------------------
    # Logging statistics
    # --------------------------------------------------
    logger.info("Preprocessing completed successfully")
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Validation size: {len(val_df)}")
    logger.info(f"Test size: {len(test_df)}")
    logger.info("Train class distribution:")
    logger.info(train_df["label"].value_counts().to_dict())
    logger.info(f"Label dtype (train): {train_df['label'].dtype}")
    logger.info(f"Label sample: {train_df['label'].head().tolist()}")


    print("✅ Dataset preprocessing completed")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df
# =========================
# Script Entry Point
# =========================

if __name__ == "__main__":
    logger.info("Running preprocessing script directly")

    load_and_split_dataset(
        raw_dir="data/raw",
        output_dir="data/processed",
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )
    logger.info("Preprocessing script finished")    
