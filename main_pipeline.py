import pandas as pd
import numpy as np
import json
import os
import sqlite3
import glob
import re
import subprocess
import duckdb
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
from groq import Groq
import rag_utils

# ── CONFIG ──────────────────────────────────────────────
load_dotenv()

DBT_PROJECT_DIR = "dbt_project"
DUCKDB_PATH      = "data/warehouse.duckdb"
REPORTS_DIR  = "reports"
MODEL_NAME   = "llama-3.3-70b-versatile"
TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
MLFLOW_URI   = "sqlite:///mlflow.db"

# Groq client (cloud-hosted LLM — far more reliable than the local
# Ollama model, which was prone to repetition-loop glitches).
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("sales_pipeline")

# ── STEP 1: dbt TRANSFORM + LOAD ──────────────────────────
def load_and_clean(path=None):
    print("[1/5] Running dbt transformations...")

    run_result = subprocess.run(
        ["dbt", "run", "--profiles-dir", r"C:\Users\User\.dbt"],
        cwd=DBT_PROJECT_DIR,
        capture_output=True, text=True
    )
    print(run_result.stdout[-1500:])
    if run_result.returncode != 0:
        print(run_result.stderr[-1500:])
        raise RuntimeError("dbt run failed — see output above.")

    print("      Running dbt data-quality tests...")
    test_result = subprocess.run(
        ["dbt", "test", "--profiles-dir", r"C:\Users\User\.dbt"],
        cwd=DBT_PROJECT_DIR,
        capture_output=True, text=True
    )
    print(test_result.stdout[-1000:])
    if test_result.returncode != 0:
        print("      WARNING: dbt data-quality tests failed — check output above. Continuing anyway.")

    con = duckdb.connect(DUCKDB_PATH)
    df = con.execute("SELECT * FROM retail_clean").df()
    con.close()

    df.dropna(inplace=True)
    print(f"      {len(df):,} records loaded from dbt warehouse.")
    return df

# ── STEP 2: ANALYZE ──────────────────────────────────────
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
    }
    return stats

# ── STEP 3: TRAIN MODEL + MLFLOW ──────────────────────────
def train_model(df):
    print("[3/5] Training model + logging to MLflow...")
    le_product  = LabelEncoder()
    le_category = LabelEncoder()
    le_region   = LabelEncoder()

    df["product_enc"]  = le_product.fit_transform(df["product"])
    df["category_enc"] = le_category.fit_transform(df["category"])
    df["region_enc"]   = le_region.fit_transform(df["region"])

    features = ["product_enc", "category_enc", "region_enc",
                "quantity_abs", "price", "month", "day_of_week",
                "customer_prior_orders", "customer_avg_order_value",
                "customer_return_rate", "customer_recency_days"]

    X = df[features]
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": 5,
        "class_weight": "balanced",
    }

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
        mlflow.sklearn.log_model(
            clf, "random_forest_model",
            registered_model_name="retail_return_predictor"
        )

        # Encoder'ları da MLflow'a logla (tahmin aşamasında gerekli)
        os.makedirs("encoders", exist_ok=True)
        joblib.dump(le_product,  "encoders/le_product.pkl")
        joblib.dump(le_category, "encoders/le_category.pkl")
        joblib.dump(le_region,   "encoders/le_region.pkl")
        mlflow.log_artifacts("encoders", artifact_path="encoders")

        print(f"      Accuracy: {accuracy:.4f} — logged to MLflow ✓")
        print(f"\n{report}")

    return accuracy, report, importance

# ── STEP 4: LLM REPORT ──────────────────────────────────
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
- Top Category: {str(stats['top_category']).replace("'", "")}
- Top Product: {str(stats['top_product']).replace("'", "")}
- Top Region: {stats['top_region']}

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
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content

# ── STEP 4.5: AGENT-BASED TREND INSIGHTS ─────────────────
def get_previous_report_summary():
    """En son kaydedilmiş raporu (şu anki hariç) okuyup özetini döndürür."""
    files = sorted(glob.glob(f"{REPORTS_DIR}/report_*.txt"))
    if len(files) < 2:
        return {"error": "Karşılaştırılacak önceki rapor yok"}
    prev_file = files[-2]
    with open(prev_file, "r", encoding="utf-8") as f:
        content = f.read()
    return {"previous_report_file": prev_file, "content_preview": content[:500]}

def compare_accuracy_trend():
    """MLflow veritabanından son 5 run'ın accuracy'sini döndürür."""
    conn = sqlite3.connect("mlflow.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT run_uuid, value FROM metrics
        WHERE key = 'accuracy'
        ORDER BY run_uuid DESC LIMIT 5
    """)
    rows = cur.fetchall()
    conn.close()
    return {"recent_accuracy_values": [r[1] for r in rows]}

def search_similar_reports(query="model performansı ve iade oranı"):
    """DuckDB'de saklanan geçmiş rapor embedding'leri arasında anlamsal arama yapar."""
    try:
        results = rag_utils.search_similar_reports(DUCKDB_PATH, query, top_k=3)
        if not results:
            return {"info": "Henüz aranabilir geçmiş rapor yok."}
        return {"similar_past_reports": results}
    except Exception as e:
        return {"error": f"Semantik arama başarısız: {e}"}

def generate_agent_insights():
    print("[Agent] Generating trend insights...")

    agent_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_previous_report_summary",
                "description": "Bir önceki pipeline raporunun özetini getirir, trend karşılaştırması için kullanılır",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_accuracy_trend",
                "description": "Son 5 model çalıştırmasının accuracy değerlerini getirir",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_similar_reports",
                "description": "Geçmiş raporlar arasında anlamsal (embedding tabanlı) arama yapar; konuya en yakın önceki raporları bulur",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Aranacak konu, örn. 'yüksek iade oranı' veya 'düşük model doğruluğu'"}
                    },
                    "required": []
                }
            }
        }
    ]

    available_functions = {
        "get_previous_report_summary": get_previous_report_summary,
        "compare_accuracy_trend": compare_accuracy_trend,
        "search_similar_reports": search_similar_reports,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful data analyst. Only describe a trend if the numbers "
                "you retrieved are actually different between periods. If the data is "
                "identical or there isn't enough historical data to compare, say so "
                "explicitly instead of inventing a trend. Never describe a single "
                "snapshot of data as if it shows change over time. You also have "
                "access to a semantic search tool over past reports — use it if it "
                "would help ground your answer in similar historical situations. "
                "Keep your answer to 3-4 sentences."
            )
        },
        {"role": "user", "content": "Model performansımız geçen çalıştırmalara göre nasıl gidiyor? Trend var mı? Önceki raporu ve accuracy geçmişini incele, kısa bir değerlendirme yap."}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=agent_tools,
            temperature=0.3,
        )
        response_message = response.choices[0].message

        # Build a plain dict for the assistant turn (Groq/OpenAI format)
        assistant_msg = {"role": "assistant", "content": response_message.content}
        if response_message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response_message.tool_calls
            ]
        messages.append(assistant_msg)

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                func = available_functions[tool_call.function.name]
                result = func()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            final_response = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.3,
            )
            return final_response.choices[0].message.content
        else:
            return response_message.content
    except Exception as e:
        print(f"[Agent] Hata: {e}")
        return "Trend analizi şu an oluşturulamadı."

# ── STEP 5: SAVE REPORT ──────────────────────────────────
def save_report(stats, accuracy, llm_report, agent_insights=None):
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

        # Tekrar döngüsü ve bozuk karakterleri temizle
        llm_clean = re.sub(r'@+', '', llm_report)          # @ karakterlerini sil
        llm_clean = re.sub(r'(.)\1{10,}', '', llm_clean)   # 10+ tekrar eden karakter sil
        llm_clean = re.sub(r'\n{4,}', '\n\n', llm_clean)   # Fazla boş satırları azalt
        llm_clean = llm_clean.strip()

        f.write(llm_clean)

        agent_clean = ""
        if agent_insights:
            f.write("\n\n")
            f.write("AGENT TREND ANALYSIS\n")
            f.write("-" * 40 + "\n")
            agent_clean = re.sub(r'@+', '', agent_insights)
            agent_clean = re.sub(r'(.)\1{10,}', '', agent_clean)
            agent_clean = agent_clean.strip()
            f.write(agent_clean)

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Report: {path}")
    print(f"Total Revenue : ${stats['total_revenue']:,}")
    print(f"Return Rate   : {stats['return_rate']}%")
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"{'='*70}\n")

    # RAG: bu raporun embedding'ini gelecekteki anlamsal aramalar için sakla
    try:
        print("[RAG] Storing report embedding for semantic search...")
        full_text = f"{llm_clean}\n\n{agent_clean if agent_insights else ''}"
        rag_utils.store_report_embedding(DUCKDB_PATH, report_id=path, report_text=full_text)
    except Exception as e:
        print(f"[RAG] Embedding storage failed (non-fatal): {e}")

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AUTOMATED DATA PIPELINE — STARTING")
    print("="*70 + "\n")

    df                           = load_and_clean()
    stats                        = analyze(df)
    accuracy, report, importance = train_model(df)
    llm_report                   = generate_report(stats, accuracy, report, importance)
    agent_insights                = generate_agent_insights()
    save_report(stats, accuracy, llm_report, agent_insights)

    # Accuracy'yi alert_monitor.py'nin okuyabilmesi için dosyaya kaydet
    os.makedirs("data", exist_ok=True)
    with open("data/last_accuracy.json", "w") as f:
        json.dump({"accuracy": accuracy, "timestamp": datetime.now().isoformat()}, f)