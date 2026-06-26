import ollama
import pandas as pd
import numpy as np
import json
import os
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ────────────────────────────────────────────────
DATA_PATH    = "data/sales_data.csv"
REPORTS_DIR  = "reports"
MODEL_NAME   = "llama3.1:8b"
TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
MLFLOW_URI   = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("sales_pipeline")

# ── STEP 1: LOAD & CLEAN ──────────────────────────────────
def load_and_clean(path):
    print("[1/5] Loading and cleaning data...")
    df = pd.read_csv(path)
    df["date"]        = pd.to_datetime(df["date"])
    df["revenue"]     = df["quantity"] * df["price"]
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df.dropna(inplace=True)
    print(f"      {len(df)} records loaded.")
    return df

# ── STEP 2: ANALYZE ───────────────────────────────────────
def analyze(df):
    print("[2/5] Analyzing data...")
    stats = {
        "total_revenue"    : round(df["revenue"].sum(), 2),
        "avg_order_value"  : round(df["revenue"].mean(), 2),
        "total_orders"     : len(df),
        "return_rate"      : round(df["returned"].mean() * 100, 2),
        "top_category"     : df.groupby("category")["revenue"].sum().idxmax(),
        "top_product"      : df.groupby("product")["revenue"].sum().idxmax(),
        "top_region"       : df.groupby("region")["revenue"].sum().idxmax(),
        "avg_customer_age" : round(df["customer_age"].mean(), 1),
    }
    return stats

# ── STEP 3: TRAIN MODEL + MLFLOW ──────────────────────────
def train_model(df):
    print("[3/5] Training model + logging to MLflow...")
    le = LabelEncoder()
    df["product_enc"]  = le.fit_transform(df["product"])
    df["category_enc"] = le.fit_transform(df["category"])
    df["region_enc"]   = le.fit_transform(df["region"])

    features = ["product_enc", "category_enc", "region_enc",
                "quantity", "price", "customer_age", "month", "day_of_week"]

    X = df[features]
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {"n_estimators": 100, "random_state": 42, "max_depth": 5}

    with mlflow.start_run():
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        accuracy    = accuracy_score(y_test, predictions)
        report      = classification_report(y_test, predictions)
        importance  = dict(zip(features, clf.feature_importances_.round(3)))

        # Log everything to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("return_rate", df["returned"].mean())
        mlflow.log_metric("total_revenue", df["revenue"].sum())
        mlflow.sklearn.log_model(clf, "random_forest_model")

        print(f"      Accuracy: {accuracy:.4f} — logged to MLflow ✓")

    return accuracy, report, importance

# ── STEP 4: LLM REPORT ────────────────────────────────────
def generate_report(stats, accuracy, report, feature_importance):
    print("[4/5] Generating LLM report...")
    prompt = f"""
You are a senior data scientist presenting to the CEO.
Analyze this sales pipeline report and provide executive insights.

BUSINESS METRICS:
- Total Revenue: ${stats['total_revenue']:,}
- Average Order Value: ${stats['avg_order_value']}
- Total Orders: {stats['total_orders']}
- Return Rate: {stats['return_rate']}%
- Top Category: {stats['top_category']}
- Top Product: {stats['top_product']}
- Top Region: {stats['top_region']}
- Average Customer Age: {stats['avg_customer_age']}

RETURN PREDICTION MODEL:
- Accuracy: {accuracy:.4f}
- Feature Importance: {json.dumps(feature_importance, indent=2)}

Classification Report:
{report}

Provide a structured executive report with:
1. Executive Summary (2-3 sentences)
2. Key Business Insights (3 bullet points)
3. Risk Analysis
4. Model Performance Assessment
5. Top 3 Actionable Recommendations
"""
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content

# ── STEP 5: SAVE REPORT ───────────────────────────────────
def save_report(stats, accuracy, llm_report):
    print("[5/5] Saving report...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = f"{REPORTS_DIR}/report_{TIMESTAMP}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AUTOMATED DATA PIPELINE REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key:25}: {value}\n")
        f.write(f"\nModel Accuracy: {accuracy:.4f}\n\n")
        f.write("LLM EXECUTIVE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(llm_report)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Report: {path}")
    print(f"Total Revenue : ${stats['total_revenue']:,}")
    print(f"Return Rate   : {stats['return_rate']}%")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"{'='*70}\n")

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AUTOMATED DATA PIPELINE — STARTING")
    print("="*70 + "\n")

    df                           = load_and_clean(DATA_PATH)
    stats                        = analyze(df)
    accuracy, report, importance = train_model(df)
    llm_report                   = generate_report(stats, accuracy, report, importance)
    save_report(stats, accuracy, llm_report)