# ============================================================
# Capstone 3 — NLP Training & Inference
# ✔ Ease of use
# ✔ Reproducible ML workflow
# ✔ Production-ready engineering
# ============================================================

PROJECT_NAME=capstone3
IMAGE_NAME=capstone3-inference
PORT=8000

# ----------------------------
# 🧹 Housekeeping
# ----------------------------

.PHONY: help clean

help:
	@echo "Available commands:"
	@echo "  make deps           Install Python dependencies (uv)"
	@echo "  make lock           Generate requirements.txt"
	@echo "  make preprocess     Run data preprocessing"
	@echo "  make train          Train the model"
	@echo "  make infer          Run inference API locally"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo "  make clean          Remove artifacts"

clean:
	rm -rf __pycache__ .pytest_cache logs/*.log

# ----------------------------
# 📦 Dependency Management
# ----------------------------

deps:
	uv sync

lock:
	uv pip compile pyproject.toml -o requirements.txt

# ----------------------------
# 🧠 ML Pipeline
# ----------------------------

preprocess:
	python src/preprocessing.py

train:
	python src/train.py

# ----------------------------
# 🚀 Inference
# ----------------------------

infer:
	uvicorn src.predict:app --host 0.0.0.0 --port $(PORT) --reload

# ----------------------------
# 🐳 Docker
# ----------------------------

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(PORT):8000 $(IMAGE_NAME)
