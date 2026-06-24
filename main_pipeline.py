import ollama
import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)

import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ────────────────────────────────────────────────
DATA_PATH   = "data/data.csv"
REPORTS_DIR = "reports"
MODEL_NAME  = "llama3.1:8b"
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")

mlflow.set_experiment("ecommerce_pipeline")

# ── STEP 1: LOAD & CLEAN ──────────────────────────────────
def load_and_clean(path):
    print("[1/5] Loading and cleaning data...")
    df = pd.read_csv(path, encoding="ISO-8859-1")

    # Rename columns
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

    # Clean
    df["returned"] = (df["quantity"] < 0).astype(int)
    df = df[df["price"] > 0]
    df.dropna(subset=["customer_id"], inplace=True)

    df["date"]        = pd.to_datetime(df["date"])
    df["revenue"]     = df["quantity"] * df["price"]
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek



    print(f"      {len(df)} records loaded")
    return df

# ── STEP 2: ANALYZE ───────────────────────────────────────
def analyze(df):
    print("[2/5] Analyzing data...")
    return {
        "total_revenue"   : round(df["revenue"].sum(), 2),
        "avg_order_value" : round(df["revenue"].mean(), 2),
        "total_orders"    : len(df),
        "return_rate"     : round(df["returned"].mean() * 100, 2),
        "top_country"     : df.groupby("country")["revenue"].sum().idxmax(),
        "top_product"     : df.groupby("description")["revenue"].sum().idxmax(),
        "unique_customers": int(df["customer_id"].nunique()),
        "avg_quantity"    : round(df["quantity"].mean(), 1),
    }

# ── STEP 3: TRAIN MODEL ───────────────────────────────────
def train_model(df):
    print("[3/5] Training return prediction model...")
    le = LabelEncoder()
    df["country_enc"]     = le.fit_transform(df["country"])
    df["stock_code_enc"]  = le.fit_transform(df["stock_code"].astype(str))

    features = ["stock_code_enc", "country_enc",
            "price", "month", "day_of_week"]

    X = df[features]
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        accuracy    = accuracy_score(y_test, predictions)
        report      = classification_report(y_test, predictions)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("return_rate", df["returned"].mean())
        mlflow.sklearn.log_model(clf, "random_forest_model")

        print(f"      Model accuracy: {accuracy:.4f}")
        print(f"      MLflow run logged ✅")

    feature_importance = dict(zip(features, clf.feature_importances_.round(3)))
    return accuracy, report, feature_importance

# ── STEP 4: LLM REPORT ────────────────────────────────────
def generate_report(stats, accuracy, report, feature_importance):
    print("[4/5] Generating LLM report...")
    prompt = f"""
You are a senior data scientist presenting to the CEO.
Analyze this e-commerce pipeline report and provide executive insights.

BUSINESS METRICS:
- Total Revenue: ${stats['total_revenue']:,}
- Average Order Value: ${stats['avg_order_value']}
- Total Orders: {stats['total_orders']}
- Return Rate: {stats['return_rate']}%
- Top Country: {stats['top_country']}
- Top Product: {stats['top_product']}
- Unique Customers: {stats['unique_customers']}
- Average Quantity per Order: {stats['avg_quantity']}

RETURN PREDICTION MODEL:
- Accuracy: {accuracy:.4f}
- Feature Importance: {json.dumps(feature_importance, indent=2)}

Classification Report:
{report}

Provide a structured executive report with:
1. Executive Summary (2-3 sentences)
2. Key Business Insights (3 bullet points)
3. Risk Analysis (return rate concerns)
4. Model Performance Assessment
5. Top 3 Actionable Recommendations
"""
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.message.content

# ── STEP 5: SAVE REPORT ───────────────────────────────────
def save_report(stats, accuracy, llm_report):
    print("[5/5] Saving report...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = f"{REPORTS_DIR}/report_{TIMESTAMP}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AUTOMATED DATA PIPELINE REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write("BUSINESS METRICS\n")
        f.write("-" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key:25}: {value}\n")
        f.write(f"\nModel Accuracy: {accuracy:.4f}\n\n")
        f.write("LLM EXECUTIVE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(llm_report)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Report: {report_path}")
    print(f"Accuracy: {accuracy:.4f}")
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