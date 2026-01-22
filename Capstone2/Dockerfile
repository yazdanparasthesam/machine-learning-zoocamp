############################
# Stage 1 — Builder
############################
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# System deps needed to compile wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch FIRST
RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2 \
    torchvision==0.17.2 \
    torchaudio==2.2.2

# Install remaining deps
RUN pip install -r requirements.txt

# Build wheels (optional but recommended)
RUN pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

############################
# Stage 2 — Runtime
############################
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local /usr/local

# Copy built wheels (optional optimization)
COPY --from=builder /wheels /wheels

# Copy application code
COPY src/ src/
COPY models/ models/
COPY config/ config/

EXPOSE 8000

CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]
