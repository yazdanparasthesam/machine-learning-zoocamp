from src.monitoring import load_predictions

def confidence_drift(threshold=0.2):
    df = load_predictions()
    if df.empty:
        return None

    df["confidence"] = df[["mask_prob", "no_mask_prob"]].max(axis=1)

    recent = df.tail(100)
    mean_conf = recent["confidence"].mean()

    return {
        "mean_confidence": float(mean_conf),
        "drift_detected": mean_conf < threshold
    }