import requests

URL = "http://172.18.0.3:30080/predict"
IMAGE_PATH = "data/processed/test/mask/with_mask8.png"

with open(IMAGE_PATH, "rb") as f:
    files = {"file": f}
    response = requests.post(URL, files=files)

print("Status code:", response.status_code)
print("Prediction:", response.json())
