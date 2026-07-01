import ollama
import json
import sqlite3
import glob

MODEL_NAME = "llama3.1:8b"

# ── Veriyi doğrudan kendimiz topluyoruz (modelin karar vermesine gerek yok) ──

def read_alert_log_history():
    try:
        with open("/opt/airflow/dags/project/logs/alerts.log", "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        blocks = content.strip().split("=" * 60)
        recent = [b.strip() for b in blocks if b.strip()][-5:]
        degradation_count = sum(1 for b in recent if "MODEL_DEGRADATION" in b)
        return {
            "recent_alert_checks": recent,
            "model_degradation_alert_count_in_recent_checks": degradation_count,
            "total_checks_reviewed": len(recent)
        }
    except Exception as e:
        return {"error": str(e)}

def get_accuracy_history():
    conn = sqlite3.connect("/opt/airflow/dags/project/mlflow.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT run_uuid, value FROM metrics
        WHERE key = 'accuracy'
        ORDER BY run_uuid DESC LIMIT 5
    """)
    rows = cur.fetchall()
    conn.close()
    return {"recent_accuracy_values": [r[1] for r in rows]}

def read_latest_report_metrics():
    files = sorted(glob.glob("/opt/airflow/dags/project/reports/report_*.txt"))
    if not files:
        return {"error": "Hiç rapor bulunamadı"}
    with open(files[-1], "r", encoding="utf-8") as f:
        content = f.read()
    metrics_section = content.split("LLM EXECUTIVE ANALYSIS")[0]
    return {"metrics": metrics_section.strip()}

print("=== INVESTIGATOR AGENT BAŞLIYOR ===\n")

# Veriyi önceden topla
alert_data = read_alert_log_history()
accuracy_data = get_accuracy_history()
report_data = read_latest_report_metrics()

print("→ Alert log verisi toplandı")
print("→ Accuracy geçmişi toplandı")
print("→ Rapor metrikleri toplandı\n")

prompt = f"""Bir model accuracy uyarısı tetiklendi. Aşağıdaki verilere bakarak kısa bir teşhis yaz (4-5 cümle, Türkçe).

Son alert kontrollerinde kaç kez MODEL_DEGRADATION uyarısı görüldü: {alert_data.get('model_degradation_alert_count_in_recent_checks', 'bilinmiyor')} / {alert_data.get('total_checks_reviewed', 'bilinmiyor')}

Son accuracy değerleri: {accuracy_data.get('recent_accuracy_values', 'bilinmiyor')}

Sadece yukarıdaki verilere dayanarak yorum yap. Veride olmayan bir şeyi uydurma. Eğer bu tekrar eden bir sorunsa açıkça belirt."""

response = ollama.chat(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.2, "repeat_penalty": 1.4, "repeat_last_n": 64}
)

print("=== AGENT'IN TEŞHİSİ ===")
print(response.message.content)