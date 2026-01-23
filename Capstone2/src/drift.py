from src.monitoring import load_predictions

def basic_drift_summary():
    df = load_predictions()

    if df.empty:
        return {
            "status": "no_data",
            "message": "No predictions logged yet"
        }

    return {
        "status": "ok",
        "num_predictions": len(df),
        "mask_rate": float((df["prediction"] == "mask").mean()),
        "no_mask_rate": float((df["prediction"] == "no_mask").mean()),
        "time_range": {
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max()
        }
    }
