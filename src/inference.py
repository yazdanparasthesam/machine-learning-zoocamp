from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms, models
from torch.nn import functional as F

app = FastAPI()

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    return {
        "mask": float(probs[1]),
        "no_mask": float(probs[0])
    }