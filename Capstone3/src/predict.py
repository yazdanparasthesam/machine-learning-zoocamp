# src/predict.py
# FastAPI inference service (Capstone 3 – NLP Fake News Detection)

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast

from model import build_model
from config.config import load_config

# =========================
# App & Runtime Setup
# =========================

app = FastAPI(
    title="Fake News Detection API",
    description="Inference service for DistilBERT-based fake news classifier",
    version="1.0.0",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Configuration Loading
# =========================

cfg = load_config("config/model.yaml")

# =========================
# Tokenizer Initialization
# =========================

tokenizer = DistilBertTokenizerFast.from_pretrained(
    cfg["model"]["name"]
)

# =========================
# Model Loading
# =========================

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

# Class labels must match training labels
classes = ["fake", "real"]

# =========================
# Request Schema
# =========================

class TextRequest(BaseModel):
    """
    Request body schema for prediction endpoint.
    """
    text: str

# =========================
# Health & Metadata Endpoints
# =========================

@app.get("/health")
def health():
    """
    Health check endpoint.
    Useful for Kubernetes / load balancers.
    """
    return {"status": "ok"}

@app.get("/info")
def info():
    """
    Model metadata and runtime information.
    """
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
    """
    Perform fake news classification on input text.
    Returns class probabilities.
    """

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

    return {
        cls: float(probs[i])
        for i, cls in enumerate(classes)
    }
