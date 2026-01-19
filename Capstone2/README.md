# Face Mask Image Classification – Capstone 2 (ML Zoomcamp)


## 🎯 Problem Description

Monitoring compliance with face mask regulations in public and industrial
environments is typically done manually by human operators or security staff.
This approach is expensive, does not scale well, and is prone to human error,
especially in crowded or continuous-monitoring scenarios.

This project addresses the problem by building an **automated image-based
classification system** that determines whether a person in an image is
wearing a face mask or not.

## 👤 Target Users

The model is designed to be used by:

- Organizations responsible for **public safety enforcement**
- **Facility operators** (airports, factories, hospitals, campuses)
- **Developers and ML engineers** integrating computer vision models into
  monitoring or surveillance pipelines


## 🧠 Decision Supported by the Model

Given an input image containing a human face, the model predicts whether
the person is wearing a face mask or not.

This prediction supports automated decisions such as:
- Triggering alerts when mask compliance is violated
- Logging compliance statistics over time
- Enabling downstream workflows in monitoring systems


## 💡 Why This Problem Matters

Manual monitoring of mask usage:
- Does not scale to large or continuous environments
- Is costly and labor-intensive
- Can miss violations due to fatigue or human error

An automated face mask detection system enables:
- Scalable real-time monitoring
- Consistent and objective enforcement
- Integration into existing computer vision systems

This makes the solution valuable in both **public health** and
**industrial safety** contexts.




The system is built end-to-end using **deep learning**, **API-based inference**,
**containerization**, and **Kubernetes orchestration**.

This project is developed as **Capstone 2** for the **Machine Learning Zoomcamp**.

---

## 🎯 Business Problem
Automatically detecting mask usage can help:
- Enforce safety regulations in public or industrial environments
- Reduce manual monitoring costs
- Enable scalable, automated compliance systems

The model can be integrated into surveillance pipelines or image-based monitoring systems.

---

## 🗂️ Dataset Description

The dataset consists of labeled images of human faces collected for the task
of face mask classification.

Each image belongs to one of two classes:
- `mask` – the person is wearing a face mask
- `no_mask` – the person is not wearing a face mask

The dataset includes variations in:
- Lighting conditions
- Face orientations
- Mask types
- Background environments

---

### Dataset Source

The dataset is based on publicly available face mask image datasets
commonly used for computer vision classification tasks.

Example sources include:
- Kaggle Face Mask Detection datasets
- Public GitHub image repositories

The dataset is used strictly for educational and research purposes.


---

### Dataset Exploration

Exploratory data analysis (EDA) is performed in `notebook.ipynb`, including:
- Class distribution analysis
- Visual inspection of sample images
- Identification of potential data quality issues

The notebook focuses on understanding the dataset rather than production training.

---


## 📦 Dataset Access

Due to the size of the dataset, it is **not included** in this repository.

To reproduce the results, download the dataset from one of the following
public sources and place it in the `data/` directory:

- Face Mask Detection Dataset  
  https://data.mendeley.com/datasets/7bt2d592b9

After downloading, organize the dataset as follows:

---

### Dataset structure:
```css
The dataset is organized into training and validation splits:

data/
├── train/
│ ├── mask/
│ └── no_mask/
├── val/
│ ├── mask/
│ └── no_mask/


The notebook and training scripts expect this directory structure.(We split train/validation 80/20.)
```



## 🧠 Solution Overview
The solution consists of:
- A **CNN-based image classification model** (ResNet18)
- A **FastAPI inference service**
- A **Dockerized deployment**
- A **Kubernetes (kind) cluster**
- **Monitoring hooks** for prediction analysis

---

```bash
Images are resized to **224×224** and normalized during preprocessing.
```

---

## 📓 Exploratory Data Analysis (Notebook)
The notebook (`notebook.ipynb`) includes:
- Dataset size inspection
- Visual inspection of sample images
- Image transformations
- Baseline CNN training
- Evaluation metrics (classification report, confusion matrix)

> The notebook is used **only for exploration and validation**.  
> Final training and inference are implemented in standalone scripts.

---

## 🧠 Model
- Architecture: **ResNet18 (transfer learning)**
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Output: Probability of `mask` vs `no_mask`

---

## 🚀 Inference Service
The trained model is exposed via a **FastAPI REST API**.


##  Endpoint(POST /predict)

### Example Request
```bash
curl -X POST \
  -F "file=@image.jpg" \
  http://localhost:30008/predict
```

### Example Response
```json
{
  "mask": 0.87,
  "no_mask": 0.13
}
```


## 🐳 Containerization

The inference service is containerized using Docker.

To build the image:

```bash
docker build -t face-mask .
```
---

## ☸️ Kubernetes Deployment (kind)

The system is deployed on a local Kubernetes cluster using kind.

Kubernetes resources:

- Namespace
- Deployment (replicas > 1)
- NodePort Service

To deploy:

```bash
kind create cluster --name face-mask
docker build -t face-mask .
kind load docker-image face-mask --name face-mask

kubectl apply -f k8s/
```
---

## 📊 Monitoring

Basic monitoring is implemented by:

- Logging prediction probabilities
- Tracking class distribution over time
- Generating drift analysis reports (optional extension with Evidently)
- Prediction drift is monitored using Evidently.
- Model predictions are logged and periodically compared against a reference
distribution to detect data drift.


---

## 🧪 Reproducibility

- All dependencies are listed in requirements.txt
- Training, inference, and deployment are script-based
- The project can be fully reproduced using the instructions in this README

---

## 🛠️ Tech Stack

- Python
- PyTorch
- FastAPI
- Docker
- Kubernetes (kind)
- NumPy, PIL, torchvision

---


## 🧱 Project Structure

```css
capstone2-face-mask-k8s/
│
├── data/
│   └── .gitkeep
│
|
├── k8s/
│   ├── deployment.yaml
│   ├── hpa.yaml
|   ├── namespace.yaml
│   └── service.yaml
|
├── logs/
│
├── models/
│   └── model.pt
│
├──monitoring/
|  └── evidently_report.py
|
├── src/
│   ├── inference.py
│   ├── model.py
│   ├── monitoring.py
│   ├── preprocessing.py
│   └── train.py
|
├── Dockerfile
├── Makefile
├── notebook.ipynb
├── README.md
└── requirements.txt

```
---

## ⚠️ Limitations & Future Work

### Limitations

While the current system demonstrates the feasibility of automated face mask
classification, it has several limitations:

- **Limited dataset diversity**  
  The dataset may not fully represent all real-world conditions such as
  extreme lighting, heavy occlusions, low-resolution surveillance images,
  or uncommon face coverings.

- **Binary classification only**  
  The model distinguishes only between `mask` and `no_mask` and does not
  handle partial compliance cases (e.g., mask worn incorrectly).

- **Static image inference**  
  The system operates on individual images and does not leverage temporal
  information available in video streams.

- **No automated retraining pipeline**  
  Model retraining and updates are currently manual and not triggered by
  detected data drift.

- **Limited fairness analysis**  
  The model has not been evaluated for potential bias across demographic
  groups, which may affect real-world deployment.

---

### Future Work

Potential improvements and extensions include:

- **Dataset expansion and augmentation**  
  Incorporating more diverse images and applying advanced augmentation
  techniques to improve generalization.

- **Multi-class classification**  
  Extending the model to detect incorrect mask usage (e.g., nose uncovered).

- **Video stream support**  
  Integrating the model with real-time video pipelines and temporal smoothing
  for more stable predictions.

- **Automated monitoring and retraining**  
  Adding data drift detection and automated retraining workflows using tools
  such as Evidently and MLflow.

- **Model optimization for edge deployment**  
  Applying model quantization or pruning to enable deployment on edge devices
  with limited computational resources.

- **Fairness and robustness evaluation**  
  Performing systematic evaluations across demographics and environmental
  conditions to ensure responsible deployment.
