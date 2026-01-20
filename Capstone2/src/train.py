# Production-style training script
# ✔ Reproducibility
# ✔ Script-based training
# ✔ Notebook separation
# ✔ YAML-driven configuration

from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import build_model
from preprocessing import (
    get_train_transforms,
    get_val_transforms,
    create_dataset_splits,
)
from config import load_config


# =========================
# Load Configuration
# =========================

cfg = load_config("config/model.yaml")

BATCH_SIZE = cfg["training"]["batch_size"]
EPOCHS = cfg["training"]["epochs"]
LR = cfg["training"]["learning_rate"]

TRAIN_DIR = cfg["data"]["train_dir"]
VAL_DIR = cfg["data"]["val_dir"]
MODEL_PATH = "models/model.pt"

DEVICE_CFG = cfg["runtime"]["device"]


# =========================
# Training
# =========================

def train():
    if DEVICE_CFG == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE_CFG

    # Create dataset split only if it does not exist
    if not Path(TRAIN_DIR).exists():
        print("🔄 Creating train/val/test split...")
        create_dataset_splits(
            raw_data_dir=cfg["data"]["raw_dir"],
            output_dir=cfg["data"]["processed_dir"],
        )

    train_ds = ImageFolder(
        TRAIN_DIR,
        transform=get_train_transforms()
    )
    val_ds = ImageFolder(
        VAL_DIR,
        transform=get_val_transforms()
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cfg["runtime"]["num_workers"],
        pin_memory=cfg["runtime"]["pin_memory"],
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=cfg["runtime"]["num_workers"],
        pin_memory=cfg["runtime"]["pin_memory"],
    )

    model = build_model(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_dl, criterion, device)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {total_loss / len(train_dl):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}"
        )

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy


# =========================
# Script Entry Point
# =========================

if __name__ == "__main__":
    train()
