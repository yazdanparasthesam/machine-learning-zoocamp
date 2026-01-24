import requests
import time

url = "http://localhost:8000/predict"

files = {"file": open("/home/yazadanparast/Downloads/Capstone2/data/processed/test/mask/with_mask8.png", "rb")}

while True:
    try:
        requests.post(url, files=files, timeout=1)
    except:
        pass
    time.sleep(0.0005)