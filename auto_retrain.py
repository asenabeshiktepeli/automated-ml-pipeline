import os
import json
import glob
from datetime import datetime

def check_latest_drift():
    reports = glob.glob("reports/drift_report_*.json")
    if not reports:
        print("No drift reports found.")
        return False
    latest = max(reports, key=os.path.getctime)
    with open(latest) as f:
        data = json.load(f)
    return data.get("drift_detected", False)

def check_latest_accuracy():
    reports = glob.glob("reports/report_*.txt")
    if not reports:
        return 1.0
    latest = max(reports, key=os.path.getctime)
    with open(latest, encoding="utf-8") as f:
        for line in f:
            if "Model Accuracy:" in line:
                try:
                    return float(line.split(":")[1].strip())
                except:
                    return 1.0
    return 1.0

def run_auto_retrain():
    print("\n" + "="*60)
    print("AUTO RETRAIN CHECK — STARTING")
    print("="*60)

    drift_detected  = check_latest_drift()
    latest_accuracy = check_latest_accuracy()

    print(f"Drift detected : {drift_detected}")
    print(f"Latest accuracy: {latest_accuracy:.4f}")

    should_retrain = drift_detected or latest_accuracy < 0.85

    if should_retrain:
        print("\n⚠️  Retraining triggered!")
        print("Running main pipeline...")
        os.system("python main_pipeline.py")
        print("✅ Retraining complete.")
    else:
        print("\n✅ No retraining needed.")

    print("="*60)

if __name__ == "__main__":
    run_auto_retrain()