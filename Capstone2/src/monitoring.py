import json
from datetime import datetime
from pathlib import Path
import pandas as pd

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "predictions.jsonl"


def log_prediction(probs: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "mask_prob": probs["mask"],
        "no_mask_prob": probs["no_mask"],
        "prediction": "mask" if probs["mask"] > probs["no_mask"] else "no_mask"
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_predictions() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame()

    return pd.read_json(LOG_FILE, lines=True)
