# Base image
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /code

# Install dependencies
RUN pip install --upgrade pip fastapi uvicorn scikit-learn

# Copy pipeline and FastAPI app into the container
COPY pipeline_v1.bin .
COPY app.py .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
