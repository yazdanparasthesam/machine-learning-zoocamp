import requests

url = "http://localhost:9696/predict"

# Example customer data
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