# Use a lightweight Python image
FROM python:3.10-slim

# Install uv by copying it from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy only the requirements first (to cache the layer)
COPY requirements.txt .

# Install dependencies into the system environment inside the container
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the model and the prediction script
COPY ["predict.py", "model.bin", "./"]

# Expose the port Flask is running on
EXPOSE 9696

# Run the service using gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]