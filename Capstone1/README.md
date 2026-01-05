# Bank Marketing Term Deposit Prediction

## Project Description
This project predicts whether a client will subscribe to a term deposit based on a marketing campaign by a Portuguese banking institution. It uses a machine learning pipeline involving EDA, feature engineering with `DictVectorizer`, and an **XGBoost** classifier.

## Data
The dataset used is the [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 
- `data/bank-full.csv`: The full dataset used for training.

## Dependency Management
I used **uv** (a fast Python package manager written in Rust) to manage dependencies and virtual environments.

### Setting up the environment:
1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create venv: `uv venv`
3. Activate venv: `. .venv/bin/activate` 
4. Install dependencies: `uv pip install -r requirements.txt`

## Files
* `notebook.ipynb`: Exploratory Data Analysis, Feature Importance, and Model Tuning.
* `train.py`: Script to train the final model on the full dataset and save it to `model.bin`.
* `predict.py`: Flask web service to serve predictions.
* `ModelVerify.py`: Script to test the running web service.
* `model.bin`: The serialized XGBoost model and DictVectorizer.
* `Dockerfile`: Instructions to containerize the application using `uv`.
* `requirements.txt`: List of project dependencies.

## Model Evaluation
The model was evaluated using the **AUC (Area Under the ROC Curve)** metric to ensure performance on the imbalanced dataset.



* **Validation AUC**: ~0.90+ (High performance in distinguishing subscribers from non-subscribers).
* **Key Findings**: The most important feature for prediction was `duration` (length of the last contact), followed by `month` and `poutcome`.



## Training the Model
To retrain the model and regenerate `model.bin`, run:
```bash
python train.py
```


# Running the Service with Docker

To build and run the prediction service in a container:

## 1. Build the image:

```bash
docker build -t bank-prediction .
```

## 2. Run the container:

```bash
docker run -it -p 9696:9696 bank-prediction:latest
```

# Testing the Service

While the Docker container is running, you can test the API using the provided verification script:


## Run the verification script:

```bash
python ModelVerify.py
```


## Script Content (`ModelVerify.py`):

```bash
import requests

url = "http://localhost:9696/predict"
client = {
    "age": 35, 
    "job": "management", 
    "marital": "married", 
    "education": "tertiary",
    "housing": "yes",
    "loan": "no",
    "balance": 1500
}

response = requests.post(url, json=client).json()
print(f"Prediction result: {response}")
```