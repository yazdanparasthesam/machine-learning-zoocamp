# Fake News Detection – Capstone3 Project (ML Zoomcamp)

## 🎯 Problem Description
The rapid spread of misinformation and fake news through online platforms has become a serious societal challenge. Manual verification of news articles by human fact-checkers is slow, expensive, and does not scale to the massive volume of content produced every day.

This project addresses the problem by building an automated Natural Language Processing (NLP) system that classifies news articles as either **fake** or **real** based on their textual content.

---

## 👤 Target Users
The model is designed to be used by:

- Media organizations and content moderation teams  
- Researchers studying misinformation and media credibility  
- Developers and ML engineers building NLP-based content analysis systems  
- Platforms that require automated pre-screening of textual content  

---

## 🧠 Decision Supported by the Model
Given a news article (title and body text), the model predicts whether the article is **fake** or **real**.

This prediction supports automated decisions such as:

- Flagging potentially fake news for human review  
- Reducing the workload of manual fact-checking teams  
- Enabling downstream workflows in content moderation pipelines  
- Supporting research and analysis of misinformation patterns  

---

## 💡 Why This Problem Matters
Manual fake news detection:

- Does not scale to large volumes of online content  
- Is costly and time-consuming  
- Is prone to subjectivity and human bias  

An automated fake news detection system enables:

- Scalable and consistent content screening  
- Faster identification of suspicious articles  
- Integration into existing NLP and moderation systems  

This makes the solution valuable for both research and real-world content management applications.

The system is built end-to-end using machine learning, API-based inference, containerization, and production-oriented ML engineering practices.

This project is developed as a capstone project for the **Machine Learning Zoomcamp**.

---

## 🎯 Business / Application Context
Automatically detecting fake news can help:

- Reduce the spread of misinformation  
- Assist content moderation and fact-checking workflows  
- Enable scalable analysis of large text corpora  
- Support research into misinformation dynamics  

The model can be integrated into news aggregation systems, moderation tools, or NLP analytics pipelines.

---

## 🗂️ Dataset Description
The dataset consists of labeled news articles collected for the task of fake news detection. Each sample includes textual information such as:

- Article title  
- Article body text  
- Ground-truth label (`fake` or `real`)  

The dataset is designed for binary text classification.

---

Dataset Labels
Each article belongs to one of two classes:

- `fake` – the article contains misinformation or fabricated content  
- `real` – the article is from a reliable or verified source  

The dataset includes variations in:

- Writing styles  
- Article lengths  
- Topics and domains  
- Linguistic patterns  

---

### Dataset Source

The dataset is based on publicly available fake news datasets
commonly used for natural language processing classification tasks.

Example sources include:
- Kaggle Fake and Real News datasets
- Public research datasets and repositories

The dataset is used strictly for educational and research purposes.

---

### 📊 Dataset Exploration
Exploratory Data Analysis (EDA) is performed in `notebooks/eda.ipynb`, including:

- Class distribution analysis  
- Article length statistics  
- Vocabulary inspection  
- Identification of potential data quality issues  

The notebook focuses on understanding the dataset rather than production training.

---


### 📦 Dataset Access

Due to the size of the dataset, it is **not included** in this repository.

To reproduce the results, download the dataset from one of the following
public sources and place it in the `data/raw` directory:

- Fake and Real News Dataset  
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

- The dataset is split into train, validation, and test sets using a
reproducible stratified split (70/15/15) implemented in `src/preprocessing.py`.
- Split statistics are logged for transparency and reproducibility.

After downloading, organize the dataset as follows:

---

### Dataset structure:

The raw dataset consists of two separate files (`fake.csv` and `true.csv`).
During preprocessing, these files are merged, labeled, cleaned, and split
into train, validation, and test sets, which are saved as CSV files under
`data/processed/`.


Each file contains a `text` column representing the news content.

### 🏷 Labeling
- `fake` → label `0`
- `real` → label `1`

### 🧹 Preprocessing Steps
The preprocessing pipeline performs:
- Lowercasing text
- Removing URLs, digits, and punctuation
- Merging fake and real datasets
- Stratified splitting into:
  - **Train**
  - **Validation**
  - **Test**

### 📁 Output Structure
After running:
```bash
python src/preprocessing.py
```
The following files are generated:
```css
data/processed/
├── train.csv
├── val.csv
└── test.csv
```

![alt text](1.png)

we can see the preprocessing logs in `logs/preprocessing.log`

![alt text](2.png)

The notebook and training scripts expect this directory structure.(We split train/validation/test 70/15/15.)

## 🧠 Solution Overview

This project implements an **end-to-end NLP system** for automated fake news detection, covering the full machine learning lifecycle from data exploration to production deployment.

The solution consists of the following components:

1. **Data Exploration & Validation**  
   - Exploratory data analysis is performed in `notebooks/eda.ipynb`  
   - Class balance, article lengths, and vocabulary diversity are inspected  
   - The notebook is used only for analysis and experimentation

2. **Model Training and Tuning**  
   - A text classification model is trained using **TF-IDF + Logistic Regression** as baseline, with optional transformer upgrade (BERT / DistilBERT)  
   - Article titles and body text are preprocessed (cleaning, tokenization)  
   - The model predicts probabilities for two classes: `fake` and `real`  
   - Training and validation logic is implemented in standalone scripts

3. **Inference Service**  
   - The trained model is served through a **FastAPI** application  
   - A REST endpoint (`POST /predict`) accepts a news article and returns class probabilities  
   - The service is stateless and suitable for horizontal scaling

4. **Containerization & Deployment**  
   - The inference service is packaged into a Docker container  
   - The container can be deployed locally or on a cloud cluster  
   - Deployment ensures high availability and reproducibility  
   - The API is exposed for external access

5. **Monitoring & Analysis**  
   - Prediction outputs and input statistics are logged for monitoring  
   - **Evidently** is used to analyze prediction distributions and detect potential data drift  
   - Input text length, vocabulary coverage, and class distribution are monitored

6. **Configuration & Reproducibility**  
   - Model and training parameters are managed via YAML configuration files  
   - Dependencies are pinned in `pyproject.toml` or `requirements.txt`  
   - The entire system can be reproduced using the provided scripts and manifests

This architecture demonstrates production-ready machine learning practices, including modular code, containerized inference, scalable deployment, monitoring, and reproducibility.

---

## 🚀 Development
The source code follows a `src/`-based layout. The `src` directory is treated
as a Python package to ensure clean imports and compatibility with testing,
Docker, and CI environments.

---

## 📓 Exploratory Data Analysis & Feature Analysis (Notebook)
The notebook file (`notebook.ipynb`) includes:
- Dataset size inspection
- Class distribution analysis
- Text length and vocabulary statistics
- The dataset is split into train, validation, and test sets using a
  reproducible stratified split (70/15/15) implemented in `src/preprocessing.py`.
- Text preprocessing (cleaning, tokenization)
- Baseline model training (TF-IDF + Logistic Regression)
- Evaluation metrics (F1-score, precision, recall, confusion matrix)

> The `notebook.ipynb` file is used **only for exploration and validation**.  
> Final training and inference are implemented in standalone scripts.

![alt text](1.png)

---

## 📝 Notebook Overview

This project includes **two separate notebooks** to explore and experiment with different approaches to fake news detection:

1. **TF-IDF + Classical ML Notebook(tfidf_experiments.ipynb)**  
   - Implements baseline models using **TF-IDF text features**.  
   - Experiments with algorithms such as Logistic Regression, Random Forest, and SVM.  
   - Provides **quick insights** on feature importance and baseline performance.  
   - Useful for **understanding dataset structure** and **establishing a benchmark**.

2. **Transformer-based Notebook(bert_exploration.ipynb) (BERT / DistilBERT)**  
   - Implements **state-of-the-art NLP models** using **pretrained Transformers**.  
   - Handles tokenization, sequence length, attention masks, and fine-tuning.  
   - Provides **higher accuracy and more robust predictions** compared to TF-IDF baselines.  
   - Used as the **final production model** for training and inference.

> Having both notebooks allows for **progressive model development**:  
> first exploring simple features and classical ML, then moving to more complex **deep learning approaches** for production-grade performance.

---

## 🧠 TF-IDF vs BERT: Preprocessing Differences

This project includes **two separate notebooks** for text representation and modeling:
- **TF-IDF + Classical ML**
- **BERT (Transformer-based Deep Learning)**

They intentionally use **different preprocessing strategies**, because these models learn language in fundamentally different ways.

---

### 📄 TF-IDF Preprocessing (tfidf_experiments.ipynb)

TF-IDF relies on **explicit word statistics**, so heavier text normalization is required to reduce noise and dimensionality.

**Applied steps:**
- Lowercasing all text
- Removing URLs
- Removing digits and punctuation
- Stripping extra whitespace
- Optional stopword removal and n-grams
- Vectorization using `TfidfVectorizer`

**Why this matters:**
- TF-IDF treats text as a **bag of words**
- Noise directly increases feature space
- Cleaning improves signal-to-noise ratio and model stability

**Best suited for:**
- Linear models (Logistic Regression, SVM)
- Fast experimentation and baselines
- Interpretability (feature importance)

---

### 🤖 BERT Preprocessing (bert_exploration.ipynb)

BERT is pretrained on **raw natural language** and uses **subword tokenization**, so minimal cleaning is applied.

**Applied steps:**
- Lowercasing (handled internally by tokenizer)
- Tokenization using `DistilBertTokenizer`
- Padding and truncation to fixed sequence length
- Attention mask generation

**Not applied:**
- ❌ Stopword removal  
- ❌ Aggressive punctuation removal  
- ❌ Manual token splitting  

**Why this matters:**
- BERT learns **context and syntax**
- Removing words or punctuation can harm meaning
- The tokenizer is optimized for pretrained weights

**Best suited for:**
- Context-aware classification
- Higher accuracy on complex language
- Production-grade NLP systems

---

### 🔍 Why Two Notebooks Are Necessary

| Aspect | TF-IDF | BERT |
|------|-------|------|
| Feature type | Sparse vectors | Dense contextual embeddings |
| Preprocessing | Heavy | Minimal |
| Training speed | Very fast | Slower (GPU preferred) |
| Interpretability | High | Lower |
| Accuracy ceiling | Medium | High |

Keeping these pipelines **separate and explicit**:
- Improves clarity
- Avoids incorrect preprocessing reuse
- Demonstrates understanding of NLP fundamentals
- Makes experimental comparison fair and reproducible

---

### ✅ Takeaway

> **Preprocessing is model-dependent.**  
> Applying TF-IDF cleaning rules to BERT would degrade performance, while using raw text with TF-IDF would introduce noise.

This project deliberately separates both approaches to ensure **correct, principled, and production-ready NLP workflows**.

---

## 🤖 Model Training & Tuning
- Architecture: **DistilBERT (Transformer-based model, transfer learning)**
- Loss function: Cross-Entropy Loss
- Optimizer: AdamW
- Output: Probability of `fake` vs `real`
- Hyperparameters passed to the model via `config/model.py` file

Training Done on google colab to use GPU and we see the GPU `Tesla4` is available:

![alt text](3.png)

after that we started the training and it was ok and `model.pt` saved in directory model and you can see the training logs in `logs/training.log`

![alt text](4.png)

---

Text inputs are tokenized using the **DistilBERT tokenizer**, padded and truncated
to a maximum sequence length (**512 tokens**) during preprocessing.


## 🚀 Inference Service

The trained **Fake News Detection model** is exposed via a **FastAPI REST API**
for real-time inference.

The service loads the trained **Transformer-based text classification model**
and returns **class probabilities** for the labels `fake` and `real`.

### 📁 src/config/__init__.py

#### `src/config/__init__.py`

Initializes the configuration package and exposes configuration objects for easy import across the project.  
This file allows modules to access configuration settings using a unified interface (e.g. `from src.config import settings`).

### 📁 src/config/config.py

#### `src/config/config.py`

Central configuration module for the project.  
Defines all runtime settings such as dataset paths, model hyperparameters, training options, inference parameters, and environment-specific values.  
Acts as a single source of truth to ensure consistency across training, evaluation, and inference.

### 📁 src/model.py

#### `src/data_loader.py`

Handles dataset loading and preprocessing.  
Includes utilities to read raw data, clean and tokenize inputs, create training/validation/test splits, and return data loaders compatible with the training and inference pipelines.


### 📁 src/data_loader.py

#### `src/data_loader.py`

Handles dataset loading and preprocessing.  
Includes utilities to read raw data, clean and tokenize inputs, create training/validation/test splits, and return data loaders compatible with the training and inference pipelines.

## Project Structure

- `src/config/` – Configuration management
- `src/model.py` – Model architecture definition
- `src/data_loader.py` – Data loading and preprocessing


---

## 🔌 Endpoint (POST `/predict`)

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Breaking news: Scientists discover water on Mars."
      }'
```

### Example Response
```json
{
  "fake": 0.91,
  "real": 0.09
}
```
---

## 💻 Deploy Locally & Test Inference

Run the inference service locally using Uvicorn:

```bash
uvicorn src.predict:app --host 0.0.0.0 --port 8000
```
![alt text](5.png)

### 🩺 Health & Metadata Endpoints

Verify that the service is running correctly:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
```

Example Responses

```json
{ "status": "ok" }
```
```json
{
  "model": "distilbert-base-uncased",
  "num_classes": 2,
  "classes": ["fake", "real"],
  "device": "cpu"
}
```
![alt text](6.png)



Logs of health and info:

![alt text](7.png)

### 🔮 Test Prediction Locally

Send a sample text to the model:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "text": "You will not believe what this politician said next!"
      }'
```

Example Output
```json
{
  "fake": 0.9998301267623901,
  "real": 0.00016987029812298715
}
```


![alt text](8.png)

### 📜 Inference Logs

Prediction requests are logged for monitoring, debugging, and drift analysis.

Logs of predict:

![alt text](9.png)

---

## 🐳 Containerization

The inference service is containerized using Docker.

Actually I have implemented Multi-Stage Dockerfile

### 🧱 Why Multi-Stage Docker?

| Benefit           | Why it matters              |
| ----------------- | --------------------------- |
| Smaller image     | No build tools in runtime   |
| Faster startup    | Lean final container        |
| Cleaner security  | No compilers / caches       |
| Industry standard | Used in real ML deployments |


### 🧠 Strategy for This Project

This project is designed as a **production-ready NLP inference service**, following best practices from MLOps and cloud deployment.

We split the workflow into **two clear stages**:

---

### 🚧 Stage 1 — Model Build & Training

Purpose: train and persist a reproducible NLP model.

**Responsibilities**
- Install Python dependencies
- Load and preprocess dataset
- Build and train the Transformer-based model
- Save trained artifacts (model + tokenizer)

**Key components**
- `src/data_loader.py` – dataset loading & preprocessing
- `src/model.py` – model architecture and initialization
- `src/train.py` – training pipeline and model persistence
- `src/config/` – centralized configuration management

Training is executed **once**, and the trained model is reused for inference.

---

### 🚀 Stage 2 — Runtime Inference Service

Purpose: serve predictions with minimal runtime overhead.

**Responsibilities**
- Load trained model and tokenizer
- Expose REST APIs using FastAPI
- Handle health checks, metadata, and predictions
- Run efficiently in containerized environments

**Key components**
- `src/predict.py` – FastAPI application and prediction logic
- `uvicorn` – ASGI server for production inference
- Minimal runtime dependencies (no training-only packages)

---

## 🐳 Docker-Based Deployment Flow

A **multi-stage Docker strategy** is used to ensure:
- Smaller image size
- Faster startup time
- Clear separation between build and runtime

---

### 1-Build the Docker Image
```bash
docker build -t capstone3-nlp .
```

![alt text](11.png)

![alt text](12.png)

#### 2-Verify the Docker image:
```bash
docker images | grep capstone3-nlp
```
![alt text](13.png)

#### 3-Run the container:
```bash
docker run --rm -p 8001:8000 capstone3-nlp:latest
```
![alt text](14.png)

#### 4-Test endpoints health and info:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/info
```

![alt text](15.png)

#### 5-Docker logs of health and info:

![alt text](16.png)

#### 6-Test endpoints prediction:

```bash
curl -X POST http://localhost:8001/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic"}'
```
![alt text](17.png)

#### 7-Docker logs of prediction:

![alt text](18.png)

#### 8-Swagger UI:
```bash
http://localhost:8001/docs
```
![alt text](19.png)

#### 9-Swagger UI docker logs:

![alt text](20.png)

#### 10-Swagger UI health test:

![alt text](21.png)

#### 11-Swagger UI info test:

![alt text](22.png)

#### 12-Swagger UI predict test with image upload:

![alt text](23.png)


---

# ☸️ Kubernetes Orchestration (kind) — ML Inference Platform

This guide explains how to deploy the **ML Inference API (Torch + Transformers)** on a **local Kubernetes cluster using kind**, including:

- GPU-ready container image
- Kubernetes Deployment & Service
- Health checks & metrics
- Horizontal scaling
- Local testing & debugging

---

## 1️⃣ Install kind

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.22.0/kind-linux-amd64
chmod +x kind
sudo mv kind /usr/local/bin/
```

## 2️⃣ Verify kind Installation
```bash
kind version
```

![alt text](27.png)

## 3️⃣ Create a Kubernetes Cluster
```bash
kind create cluster --name ml-inference
```

📌 This creates a single-node control plane suitable for development and testing.

## 4️⃣ Verify Cluster Status
```bash
kubectl cluster-info
```

You should see:

- Kubernetes master
- CoreDNS
- API server

## 5️⃣ Build Docker Image (LOCAL)

⚠️ Image must be built locally before loading into kind.
```bash
docker build -t ml-inference-api:latest .
```

This image includes:

- PyTorch
- Transformers
- FastAPI
- Prometheus metrics
- Healthcheck endpoint

## 6️⃣ Verify Docker Image
```bash
docker images | grep capstone3-nlp
```

## 7️⃣ Load Docker Image into kind

⚠️ Critical step
kind does NOT automatically see host Docker images.
```bash
kind load docker-image capstone3-nlp:latest --name ml-inference
```
## 8️⃣ Verify Image Inside kind Node
```bash
docker ps | grep kind
```

Then:
```bash
docker exec -it ml-inference-control-plane crictl images | grep ml-inference
```

✔ Confirms the image is available to Kubernetes.

![alt text](28.png)

![alt text](29.png)


## 9️⃣ Apply Kubernetes Manifests
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

📦 Resources deployed:

- Namespace
- Deployment (replicas > 1)
- NodePort Service

## 🔟 Verify Kubernetes Resources ✅

### 🔟-1 Pods
```bash
kubectl get pods -n ml-inference
```

Expected:
```bash
ml-inference-api-xxxx   Running
```
### 🔟-2 Services
```bash
kubectl get svc -n ml-inference
```

Expected:
```bash
ml-inference-service   NodePort
```
### 🔟-3 Logs
```bash
kubectl logs -n ml-inference deploy/ml-inference-api
```

Logs should show:

- Model loading
- API startup
- Metrics enabled

### 🔟-4 Horizontal Pod Autoscaler
```bash
kubectl get hpa -n ml-inference
```

✔ Confirms autoscaling is active.

1️⃣1️⃣ Access API from Host (Method 1 — NodePort)
1️⃣1️⃣-1 Get Node IP
```bash
docker inspect ml-inference-control-plane | grep IPAddress
```

Example:
```bash
"IPAddress": "172.18.0.3"
```

1️⃣1️⃣-2 Test Health & Info
```bash
curl http://172.18.0.3:30080/health
curl http://172.18.0.3:30080/info
```

Expected:
```bash
{"status":"ok"}
```

1️⃣1️⃣-3 Swagger UI (Kubernetes)

Open in browser:
```bash
http://172.18.0.3:30080/docs
```

📌 You can test:

- `/health`
- `/metrics`
- `/predict`

directly from Swagger UI.

 Steps:

- Select POST /predict
- Click Try it out
- Enter text input:
```bash
{
  "text": "Breaking news: Scientists discover water on Mars"
}
```

- Click Execute

Expected response:
```bash
{
  "fake": 0.9998301267623901,
  "real": 0.00016987029812298715
}
```

1️⃣1️⃣-4 Test `/predict` from CLI
```bash
curl -X POST http://172.18.0.3:30080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news: Scientists discover water on Mars"
  }'
```

Expected Response:
```bash
{
  "fake": 0.9998301267623901,
  "real": 0.00016987029812298715
}
```


1️⃣2️⃣ Access API (Method 2 — kubectl port-forward)
```bash
kubectl port-forward -n ml-inference svc/ml-inference-service 8000:80
```

Now access locally:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
```

Swagger UI:
```bash
http://localhost:8000/docs
```

📊 Prometheus Metrics

Metrics endpoint:
```bash
GET /metrics
```

Compatible with:

- Prometheus
- Grafana dashboards
- HPA custom metrics

🧠 Deployment Architecture

Kubernetes Resources Used:

- Namespace
- Deployment
 - Multi-replica
 - Health probes
 - Rolling updates
- NodePort Service
- Horizontal Pod Autoscaler
- Prometheus-ready metrics
---

### Configuration File
The main configuration file is located at:
```
config/model.yaml
```

Example configuration:
```yaml
model:
  name: distilbert-base-uncased
  num_classes: 2
  pretrained: true

training:
  batch_size: 16
  epochs: 3
  learning_rate: 2e-5
  optimizer: adamw
  loss: cross_entropy

data:
  raw_dir: data/raw
  processed_dir: data/processed
  train_file: data/processed/train.csv
  val_file: data/processed/val.csv
  test_file: data/processed/test.csv
  max_length: 512
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42

runtime:
  device: auto
  num_workers: 2
  pin_memory: true
```

## 📦 Dependency Management

This project uses **uv**, a fast and modern Python package manager,
for dependency resolution and locking.

Dependencies are declared in `pyproject.toml` and compiled into a
reproducible `requirements.txt` file for compatibility with Docker,
CI/CD, and standard Python environments.

---

### ➕ Adding Dependencies

To add a new dependency:

```bash
uv add fastapi uvicorn torch transformers pandas evidently dynaconf python-multipart
```
![alt text](24.png)

![alt text](25.png)

⚠️ We explicitly pin NumPy to <2 for PyTorch compatibility.

```bash
uv add "numpy<2"
```
### 📌 Generating requirements.txt

A fully pinned requirements.txt is generated using:
```bash
uv pip compile pyproject.toml -o requirements.txt
```

![alt text](26.png)

This file must be committed to the repository.

### 🐳 Why requirements.txt is still used

Although uv is used for development, `requirements.txt` ensures:

- Docker compatibility
- Faster CI builds
- Deterministic deployments
- Kubernetes & air-gapped support