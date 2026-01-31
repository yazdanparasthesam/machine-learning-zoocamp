# Production-style training script (NLP / Transformer)
# ✔ Reproducibility
# ✔ Script-based training
# ✔ Notebook separation
# ✔ YAML-driven configuration

from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm
import yaml

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AdamW,
)

from src.preprocessing import load_and_split_dataset
from src.data_loader import create_dataloaders


# =========================
# Load Configuration
# =========================

with open("config/model.py", "r") as f:
    cfg = yaml.safe_load(f)

BATCH_SIZE = cfg["training"]["batch_size"]
EPOCHS = cfg["training"]["epochs"]
LR = float(cfg["training"]["learning_rate"])

MODEL_NAME = cfg["model"]["name"]
NUM_CLASSES = cfg["model"]["num_classes"]

TRAIN_FILE = cfg["data"]["train_file"]
VAL_FILE = cfg["data"]["val_file"]
TEST_FILE = cfg["data"]["test_file"]
MAX_LEN = cfg["data"]["max_length"]

DEVICE_CFG = cfg["runtime"]["device"]
NUM_WORKERS = cfg["runtime"]["num_workers"]
PIN_MEMORY = cfg["runtime"]["pin_memory"]

MODEL_PATH = "models/model.pt"


# =========================
# Training
# =========================

def train():
    # Device selection
    if DEVICE_CFG == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE_CFG

    torch.manual_seed(cfg["data"]["random_seed"])

    # --------------------------------------------------
    # Load & preprocess dataset
    # --------------------------------------------------
    train_df, val_df, test_df = load_and_split_dataset(
        csv_path=TRAIN_FILE,
        test_size=cfg["data"]["test_split"],
        val_size=cfg["data"]["val_split"],
        random_state=cfg["data"]["random_seed"],
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------
    train_dl, val_dl, test_dl = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
    ).to(device)

    # --------------------------------------------------
    # Optimizer & Loss
    # --------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_dl, criterion, device)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {total_loss / len(train_dl):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


# =========================
# Script Entry Point
# =========================

if __name__ == "__main__":
    train()
