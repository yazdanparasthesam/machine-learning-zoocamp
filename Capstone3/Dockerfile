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

# Copy only dependency definitions first (for Docker cache efficiency)
COPY requirements.txt .

# Upgrade pip and build wheels for all dependencies
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt


############################################
# 🚀 Stage 2 — Runtime
# Purpose:
# - Minimal image
# - Install only prebuilt wheels
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

# Install Python dependencies from wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application source code
COPY src/ src/
COPY config/ config/
COPY models/ models/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app using Uvicorn
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
