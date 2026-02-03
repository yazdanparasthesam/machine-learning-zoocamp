import requests

# =========================
# Configuration
# =========================

URL = "http://172.18.0.3:30080/predict"  # Node IP + NodePort

payload = {
    "text": (
        "Breaking news! Scientists confirm the discovery of water on Mars, "
        "raising hopes for future human missions."
    )
}

headers = {
    "Content-Type": "application/json"
}

# =========================
# Send request
# =========================

response = requests.post(URL, json=payload, headers=headers)

# =========================
# Output
# =========================

print("Status code:", response.status_code)

if response.ok:
    print("Prediction:", response.json())
else:
    print("Error:", response.text)
