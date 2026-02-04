# src/predict.py
# FastAPI inference service (Capstone 3 – NLP Fake News Detection)

import torch
import time
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.model import build_model
from src.config.config import load_config

# =========================
# App Setup
# =========================

app = FastAPI(
    title="Fake News Detection API",
    description="Inference service for fine-tuned fake news classifier",
    version="1.0.0",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load config
# =========================

cfg = load_config("config/model.yaml")

# =========================
# Tokenizer (LOCAL ONLY)
# =========================

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "/models/tokenizer",
    local_files_only=True
)

# =========================
# Build model architecture
# =========================

model = build_model(
    name=cfg["model"]["name"],
    num_classes=cfg["model"]["num_classes"],
    pretrained=False,
)

model.load_state_dict(
    torch.load("/models/model.pt", map_location=device)
)

model.to(device)
model.eval()

classes = ["fake", "real"]

# =========================
# Request schema
# =========================

class TextRequest(BaseModel):
    text: str

# =========================
# Prometheus Metrics Expose
# =========================

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"]
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total number of model predictions"
)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    HTTP_REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(latency)

    return response

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# =========================
# Health & info
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
# Prediction
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

    MODEL_PREDICTIONS_TOTAL.inc()

    return {
        cls: float(probs[i])
        for i, cls in enumerate(classes)
    }
