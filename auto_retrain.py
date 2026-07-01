import os
import json
import glob
from datetime import datetime
from mlflow.tracking import MlflowClient

MLFLOW_URI = "sqlite:///mlflow.db"
MODEL_NAME = "retail_return_predictor"


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
    """
    data/last_accuracy.json'dan okur (main_pipeline.py tarafından her
    çalıştırmada güncelleniyor). Eski text-parsing yöntemi kırılgandı —
    LLM rapor formatı değişirse sessizce yanlış sonuç (1.0) döndürüyordu.
    """
    path = "data/last_accuracy.json"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found — assuming retrain needed.")
        return 0.0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data["accuracy"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"WARNING: Could not parse {path} ({e}) — assuming retrain needed.")
        return 0.0


def promote_if_better():
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print(f"WARNING: Model registry okunamadi: {e} - promotion atlandi.")
        return

    if not versions:
        print("WARNING: Registry'de model bulunamadi - promotion atlandi.")
        return

    challenger = max(versions, key=lambda v: int(v.version))
    challenger_run = client.get_run(challenger.run_id)
    challenger_acc = challenger_run.data.metrics.get("accuracy", 0.0)

    prod_versions = [v for v in versions if v.current_stage == "Production"]

    if not prod_versions:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=challenger.version, stage="Production"
        )
        print(f"OK: Ilk model production'a alindi (v{challenger.version}, "
              f"accuracy={challenger_acc:.4f}).")
        return

    champion = prod_versions[0]
    champion_run = client.get_run(champion.run_id)
    champion_acc = champion_run.data.metrics.get("accuracy", 0.0)

    print(f"Champion (v{champion.version}): accuracy={champion_acc:.4f}")
    print(f"Challenger (v{challenger.version}): accuracy={challenger_acc:.4f}")

    if challenger_acc > champion_acc:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=challenger.version, stage="Production"
        )
        client.transition_model_version_stage(
            name=MODEL_NAME, version=champion.version, stage="Archived"
        )
        print(f"OK: Yeni model daha iyi - production'a alindi (v{challenger.version}).")
    else:
        print(f"BLOCKED: Yeni model daha iyi degil - production'da v{champion.version} kaliyor "
              f"(otomatik rollback korundu).")


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
        print("Retraining run complete — evaluating new model...")

        promote_if_better()
    else:
        print("\n✅ No retraining needed.")

    print("="*60)


if __name__ == "__main__":
    run_auto_retrain()