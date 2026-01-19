import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.monitoring import load_predictions


def generate_drift_report():
    df = load_predictions()

    if len(df) < 50:
        print("Not enough data for drift detection")
        return

    reference = df.iloc[: len(df)//2]
    current = df.iloc[len(df)//2 :]

    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(reference_data=reference, current_data=current)

    report.save_html("reports/data_drift_report.html")
    print("Drift report saved to reports/data_drift_report.html")


if __name__ == "__main__":
    generate_drift_report()
