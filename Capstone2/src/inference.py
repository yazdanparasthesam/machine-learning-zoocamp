# FastAPI inference service
# ✔ Deployment
# ✔ Monitoring
# ✔ API clarity

import torch
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from src.model import build_model
from src.preprocessing import get_val_transforms
from src.monitoring import log_prediction
from src.drift import confidence_drift


# =========================
# App & Runtime Setup
# =========================

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained model (no pretrained weights needed at inference time)
model = build_model(pretrained=False)
model.load_state_dict(
    torch.load("models/model.pt", map_location=device)
)
model.eval()
model.to(device)

transform = get_val_transforms()
classes = ["mask", "no_mask"]

# =========================
# Health & Metadata Endpoints
# =========================

@app.get("/health")
def health():
    """
    Health check endpoint for Kubernetes liveness/readiness probes
    """
    return {"status": "ok"}


@app.get("/info")
def info():
    """
    Model metadata endpoint
    """
    return {
        "model": "resnet18",
        "num_classes": len(classes),
        "classes": classes,
        "device": device,
    }

# =================================
# Inference Endpoint for Monitoring
# =================================
#Wrap logging in a try/except so inference never fails because of monitoring.
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    response = dict(zip(classes, probs.tolist()))

    # 🔍 Monitoring (non-blocking)
    try:
        log_prediction(response)
    except Exception as e:
        print(f"[Monitoring warning] {e}")

    return response



@app.get("/drift")
def drift():
    return confidence_drift()

