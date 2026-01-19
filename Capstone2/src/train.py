#Production-style training script
#✔ Reproducibility
#✔ Script-based training
#✔ Notebook separation
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import build_model
from preprocessing import get_train_transforms, get_val_transforms


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ImageFolder("data/train", transform=get_train_transforms())
    val_ds = ImageFolder("data/val", transform=get_val_transforms())

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dl):.4f}")

    torch.save(model.state_dict(), "models/model.pt")
    print("Model saved to models/model.pt")


if __name__ == "__main__":
    train()
