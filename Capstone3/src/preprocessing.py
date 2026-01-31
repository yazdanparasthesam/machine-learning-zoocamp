# src/preprocessing.py
#cleaning + train/val/test split + logging statistics

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

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

def load_and_split_dataset(csv_path: str, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load dataset from CSV and split into train, validation, test sets.
    Assumes CSV has 'text' and 'label' columns.
    
    Returns:
        train_df, val_df, test_df
    """
    df = pd.read_csv(csv_path)
    
    # Clean text
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Stratified split
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )
    
    val_relative_size = val_size / (1 - test_size)  # adjust val size for remaining data
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, stratify=train_val_df['label'], random_state=random_state
    )
    
    print("Dataset split statistics:")
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    print("Train class distribution:\n", train_df['label'].value_counts())
    
    return train_df, val_df, test_df