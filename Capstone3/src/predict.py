# src/predict.py
# FastAPI inference service (Capstone 3 – NLP)

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import DistilBertTokenizer
from src.model import build_model
from config.config import load_config

# =========================
# App & Runtime Setup
# =========================

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = load_config("config/model.yaml")

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(
    cfg["model"]["name"]
)

# Model
model = build_model(
    name=cfg["model"]["name"],
    num_classes=cfg["model"]["num_classes"],
    pretrained=False,
)

model.load_state_dict(
    torch.load("models/model.pt", map_location=device)
)

model.to(device)
model.eval()

classes = ["negative", "positive"]  # adjust if needed

# =========================
# Request Schema
# =========================

class TextRequest(BaseModel):
    text: str

# =========================
# Health & Metadata
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "model": cfg["model"]["name"],
        "num_classes": len(classes),
        "classes": classes,
        "device": device,
    }

# =========================
# Prediction Endpoint
# =========================

@app.post("/predict")
def predict(request: TextRequest):
    inputs = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    response = {
        cls: float(probs[i])
        for i, cls in enumerate(classes)
    }

    return response
