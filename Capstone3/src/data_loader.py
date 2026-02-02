# src/data_loader.py
# custom Dataset + DataLoaders + logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast

class NewsDataset(Dataset):
    """
    Custom PyTorch Dataset for Fake News detection.
    """
    def __init__(self, texts, labels, tokenizer: DistilBertTokenizerFast, max_len=512):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.astype("int64").values
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Safety check (optional but recommended)
        assert set(self.labels).issubset({0, 1}), "Invalid label values detected"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts.iloc[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_dataloaders(train_df, val_df, test_df, tokenizer, max_len=512, batch_size=16):
    """
    Create PyTorch DataLoaders for train, val, test sets
    """
    train_ds = NewsDataset(train_df['text_clean'], train_df['label'], tokenizer, max_len)
    val_ds = NewsDataset(val_df['text_clean'], val_df['label'], tokenizer, max_len)
    test_ds = NewsDataset(test_df['text_clean'], test_df['label'], tokenizer, max_len)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    
    print("DataLoaders created:")
    print(f"Train batches: {len(train_dl)}, Validation batches: {len(val_dl)}, Test batches: {len(test_dl)}")
    
    return train_dl, val_dl, test_dl