# Step 1: Use the prepared base image
FROM agrigorev/zoomcamp-model:2025

# Step 2: Set working directory inside the container
WORKDIR /code

# Step 3: Copy pyproject.toml (for dependencies) and install them
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install -r <(uv export --without-hashes)

# Step 4: Copy your FastAPI script into the container
COPY app.py .

# Step 5: Expose port for FastAPI
EXPOSE 8000

# Step 6: Run FastAPI server using uvicorn
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
