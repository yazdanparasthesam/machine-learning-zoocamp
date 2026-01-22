import requests
import time

url = "http://localhost:8080/predict"

files = {"file": open("with_mask8.png", "rb")}

while True:
    try:
        requests.post(url, files=files, timeout=1)
    except:
        pass
    time.sleep(0.05)