import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import os
import json
from datetime import datetime

DATA_PATH   = "data/data.csv"
REPORTS_DIR = "reports"

def load_data():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "InvoiceNo"  : "invoice_no",
        "StockCode"  : "stock_code",
        "Description": "description",
        "Quantity"   : "quantity",
        "InvoiceDate": "date",
        "UnitPrice"  : "price",
        "CustomerID" : "customer_id",
        "Country"    : "country"
    })
    df["returned"]    = (df["quantity"] < 0).astype(int)
    df = df[df["price"] > 0]
    df.dropna(subset=["customer_id"], inplace=True)
    df["date"]        = pd.to_datetime(df["date"])
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    le = LabelEncoder()
    df["country_enc"]    = le.fit_transform(df["country"])
    df["stock_code_enc"] = le.fit_transform(df["stock_code"].astype(str))
    return df

def check_drift(reference, current, feature, threshold=0.05):
    stat, p_value = stats.ks_2samp(reference[feature], current[feature])
    drifted = p_value < threshold
    return {"feature": feature, "p_value": round(p_value, 4), "drifted": bool(drifted)}

def run_drift_monitor():
    print("\n" + "="*60)
    print("DRIFT MONITOR — STARTING")
    print("="*60)

    df       = load_data()
    features = ["stock_code_enc", "country_enc", "price", "month", "day_of_week"]

    split     = int(len(df) * 0.7)
    reference = df.iloc[:split]
    current   = df.iloc[split:]

    print(f"Reference: {len(reference)} rows | Current: {len(current)} rows\n")

    results      = []
    drift_count  = 0

    for feature in features:
        result = check_drift(reference, current, feature)
        results.append(result)
        status = "⚠️  DRIFT" if result["drifted"] else "✅ OK"
        print(f"{status} | {feature:20} | p-value: {result['p_value']}")
        if result["drifted"]:
            drift_count += 1

    drift_detected = drift_count > len(features) // 2

    print(f"\n{'='*60}")
    if drift_detected:
        print("⚠️  OVERALL: DATA DRIFT DETECTED — Retraining recommended!")
    else:
        print("✅ OVERALL: No significant drift detected.")

    # Raporu kaydet
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORTS_DIR}/drift_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp"     : timestamp,
            "drift_detected": bool(drift_detected),
            "features"      : results
        }, f, indent=2)

    print(f"Report saved: {report_path}")
    print("="*60)
    return drift_detected

if __name__ == "__main__":
    run_drift_monitor()