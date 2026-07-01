import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import os
import json
from datetime import datetime

DATA_PATH   = "data/sales_data.csv"
REPORTS_DIR = "reports"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["quantity"] * df["price"]
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df.dropna(inplace=True)

    le = LabelEncoder()
    df["product_enc"]  = le.fit_transform(df["product"].astype(str))
    df["category_enc"] = le.fit_transform(df["category"].astype(str))
    df["region_enc"]   = le.fit_transform(df["region"].astype(str))

    return df

def check_drift(reference, current, feature, threshold=0.05):
    stat, p_value = stats.ks_2samp(reference[feature], current[feature])
    drifted = p_value < threshold
    return {"feature": feature, "p_value": round(p_value, 4), "drifted": bool(drifted)}

def run_drift_monitor():
    print("\n" + "="*60)
    print("DRIFT MONITOR — STARTING")
    print("="*60)

    df = load_data()

    # Davranışsal / sürekli özellikler — takvim özellikleri (month, day_of_week)
    # kasıtlı olarak hariç tutuldu, çünkü farklı zaman dilimlerini karşılaştırırken
    # bunlar yapısal olarak her zaman farklı çıkar ve gerçek bir model performans
    # kaybını göstermez.
    features = ["product_enc", "category_enc", "region_enc", "quantity", "price", "customer_age"]

    # Not: Veri seti şu an çok küçük (prototip aşaması). İstatistiksel olarak
    # anlamlı sonuçlar için en az birkaç yüz satır gerekir. Üretim ortamında
    # gerçek satış verisi biriktikçe bu test daha güvenilir hale gelecektir.
    if len(df) < 30:
        print(f"⚠️  UYARI: Veri seti çok küçük ({len(df)} satır). Drift testi sonuçları "
              f"istatistiksel olarak güvenilir olmayabilir. Anlamlı sonuçlar için "
              f"en az 30+ satır önerilir.\n")

    # Rastgele bölme (konumsal/zamansal bölme YERİNE) — küçük/büyüyen veri
    # setlerinde tutarlı ve önyargısız bir karşılaştırma sağlar.
    reference, current = train_test_split(df, test_size=0.3, random_state=42)

    print(f"Reference: {len(reference)} rows | Current: {len(current)} rows\n")

    results = []
    drift_count = 0

    for feature in features:
        result = check_drift(reference, current, feature)
        results.append(result)
        status = "⚠️ DRIFT" if result["drifted"] else "✅ OK"
        print(f"{status} | {feature:20} | p-value: {result['p_value']}")
        if result["drifted"]:
            drift_count += 1

    drift_detected = drift_count > len(features) // 2

    print(f"\n{'='*60}")
    if drift_detected:
        print("⚠️ OVERALL: DATA DRIFT DETECTED — Retraining recommended!")
    else:
        print("✅ OVERALL: No significant drift detected.")

    # Raporu kaydet
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{REPORTS_DIR}/drift_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "sample_size": len(df),
            "drift_detected": bool(drift_detected),
            "features": results
        }, f, indent=2)

    print(f"Report saved: {report_path}")
    print("="*60)
    return drift_detected

if __name__ == "__main__":
    run_drift_monitor()