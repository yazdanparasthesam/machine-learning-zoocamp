#FastAPI inference service
#✔ Deployment
#✔ Monitoring
#✔ API clarity
import torch
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from model import build_model
from preprocessing import get_val_transforms
from monitoring import log_prediction

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model()
model.load_state_dict(torch.load("models/model.pt", map_location=device))
model.eval()
model.to(device)

transform = get_val_transforms()
classes = ["mask", "no_mask"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    response = dict(zip(classes, probs.tolist()))
    log_prediction(response)

    return response
