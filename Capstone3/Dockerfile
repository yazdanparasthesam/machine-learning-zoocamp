############################################
# 🏗️ Stage 1 — Builder
# Purpose:
# - Install Python dependencies
# - Build wheels for faster runtime install
# - Cache heavy packages like torch & transformers
############################################

FROM python:3.10-slim AS builder

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# System-level dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first (Docker cache optimization)
COPY requirements.txt .

# Upgrade pip and build wheels for all dependencies
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt


############################################
# 🚀 Stage 2 — Runtime
# Purpose:
# - Minimal image
# - Install only prebuilt wheels
# - Bake ML model artifacts into the image
# - Run FastAPI inference service
############################################

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy prebuilt Python wheels from builder stage
COPY --from=builder /wheels /wheels

# Install Python dependencies from wheels (offline-safe)
RUN pip install --no-cache-dir /wheels/*

############################################
# 🧠 Download HuggingFace model at BUILD time
# Purpose:
# - Avoid runtime internet dependency
# - Ensure Kubernetes / air-gapped compatibility
# - Faster startup and reproducibility
############################################

RUN python - <<EOF
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "/models/distilbert"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.save_pretrained(MODEL_DIR)
EOF

############################################
# 📦 Copy application source code
############################################

COPY src/ src/
COPY config/ config/

# (Optional) keep models dir if you have fine-tuned weights
COPY models/ models/

# Expose FastAPI port
EXPOSE 8000

############################################
# 🚀 Run FastAPI app
############################################

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
