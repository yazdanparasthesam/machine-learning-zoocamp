FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (for PIL / torchvision)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY src/ src/
COPY models/ models/
COPY config/ config/
COPY logs/ logs/

EXPOSE 8000

CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]