import smtplib
import sqlite3
import pandas as pd
import json
import os
import glob
import ollama
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────
# Fill in your Gmail address and App Password
EMAIL_SENDER   = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "your_email@gmail.com"

# Alert thresholds
ACCURACY_THRESHOLD    = 0.85  # Alert if accuracy drops below 85%
RETURN_RATE_THRESHOLD = 25.0  # Alert if return rate exceeds 25%

DB_PATH = "data/pipeline.db"

MODEL_NAME = "llama3.1:8b"

# ── LOAD METRICS ────────────────────────────────────────
def load_metrics():
    print("Loading metrics from database...")
    conn   = sqlite3.connect(DB_PATH)
    df     = pd.read_sql("SELECT * FROM sales_data", conn)
    conn.close()

    metrics = {
        "total_revenue"   : round(df["revenue"].sum(), 2),
        "total_orders"    : len(df),
        "return_rate"     : round(df["returned"].mean() * 100, 2),
        "avg_order_value" : round(df["revenue"].mean(), 2),
        "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return metrics

# ── CHECK ALERTS ────────────────────────────────────────
def check_alerts(metrics, model_accuracy=1.0):
    alerts = []

    if model_accuracy < ACCURACY_THRESHOLD:
        alerts.append({
            "type"    : "MODEL_DEGRADATION",
            "severity": "HIGH",
            "message" : f"Model accuracy dropped to {model_accuracy:.4f} (threshold: {ACCURACY_THRESHOLD})",
        })

    if metrics["return_rate"] > RETURN_RATE_THRESHOLD:
        alerts.append({
            "type"    : "HIGH_RETURN_RATE",
            "severity": "MEDIUM",
            "message" : f"Return rate is {metrics['return_rate']}% (threshold: {RETURN_RATE_THRESHOLD}%)",
        })

    return alerts

# ── INVESTIGATOR AGENT ──────────────────────────────────
def read_alert_log_history():
    try:
        with open("logs/alerts.log", "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        blocks = content.strip().split("=" * 60)
        recent = [b.strip() for b in blocks if b.strip()][-5:]
        degradation_count = sum(1 for b in recent if "MODEL_DEGRADATION" in b)
        return {
            "model_degradation_alert_count_in_recent_checks": degradation_count,
            "total_checks_reviewed": len(recent)
        }
    except Exception as e:
        return {"error": str(e)}

def get_accuracy_history():
    try:
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
    except Exception as e:
        return {"error": str(e)}

def investigate_with_agent(alerts):
    """Sadece uyarı varsa çalışır. Geçmiş veriye bakıp kısa bir teşhis üretir."""
    print("\n[Investigator Agent] Uyarılar inceleniyor...")

    alert_data    = read_alert_log_history()
    accuracy_data = get_accuracy_history()

    alert_types = ", ".join(a["type"] for a in alerts)

    prompt = f"""Şu uyarılar tetiklendi: {alert_types}

Son alert kontrollerinde kaç kez MODEL_DEGRADATION uyarısı görüldü: {alert_data.get('model_degradation_alert_count_in_recent_checks', 'bilinmiyor')} / {alert_data.get('total_checks_reviewed', 'bilinmiyor')}

Son accuracy değerleri: {accuracy_data.get('recent_accuracy_values', 'bilinmiyor')}

Sadece yukarıdaki verilere dayanarak kısa bir teşhis yaz (4-5 cümle, Türkçe). Veride olmayan bir şeyi uydurma. Eğer bu tekrar eden bir sorunsa açıkça belirt."""

    def is_broken(text):
        """Modelin döngüye girip anlamsız çıktı (ör. '@@@@@@') ürettiğini tespit eder."""
        if not text:
            return True
        return text.count("@") > 10 or len(set(text)) < 5

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "repeat_penalty": 1.4, "repeat_last_n": 64}
        )
        diagnosis = response.message.content.strip()

        # Bozuk/döngü çıktısı tespit edilirse bir kez daha, daha güçlü ayarlarla dene
        if is_broken(diagnosis):
            print("[Investigator Agent] Bozuk çıktı tespit edildi, tekrar deneniyor...")
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "repeat_penalty": 1.6, "repeat_last_n": 128}
            )
            diagnosis = response.message.content.strip()

        # İkinci deneme de bozuksa, kullanıcıya temiz bir mesaj göster
        if is_broken(diagnosis):
            diagnosis = "Investigator agent bu çalıştırmada güvenilir bir teşhis üretemedi (model çıktı hatası). Sonraki çalıştırmada tekrar denenecek."

        print(f"[Investigator Agent] Teşhis:\n{diagnosis}\n")
        return diagnosis
    except Exception as e:
        print(f"[Investigator Agent] Hata: {e}")
        return None

# ── SEND EMAIL ──────────────────────────────────────────
def send_alert_email(alerts, metrics):
    if not alerts:
        print("No alerts to send.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[PIPELINE ALERT] {len(alerts)} issue(s) detected"
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER

    body = f"""
AUTOMATED PIPELINE ALERT
Generated: {metrics['timestamp']}
{'='*50}

ALERTS DETECTED: {len(alerts)}

"""
    for i, alert in enumerate(alerts, 1):
        body += f"""
Alert {i}: {alert['type']}
Severity : {alert['severity']}
Message  : {alert['message']}
{'-'*40}
"""

    body += f"""
CURRENT METRICS:
- Total Revenue   : ${metrics['total_revenue']:,}
- Total Orders    : {metrics['total_orders']}
- Return Rate     : {metrics['return_rate']}%
- Avg Order Value : ${metrics['avg_order_value']}

{'='*50}
Automated Data Pipeline Monitor
"""

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"Alert email sent to {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"Email failed: {e}")
        print("(Configure EMAIL_SENDER and EMAIL_PASSWORD to enable email alerts)")

# ── LOG ALERT ───────────────────────────────────────────
def log_alerts(alerts, metrics, agent_diagnosis=None):
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/alerts.log"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Alert Check: {metrics['timestamp']}\n")
        f.write(f"Return Rate: {metrics['return_rate']}%\n")

        if alerts:
            for alert in alerts:
                f.write(f"[{alert['severity']}] {alert['type']}: {alert['message']}\n")
        else:
            f.write("Status: OK - No alerts\n")

        if agent_diagnosis:
            f.write(f"\n[Investigator Agent Diagnosis]\n{agent_diagnosis}\n")

    print(f"Alert log saved: {log_path}")

# ── MAIN ────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ALERT MONITOR - STARTING")
    print("="*60 + "\n")

    metrics = load_metrics()

    # Gerçek model accuracy'sini dosyadan oku
    try:
        with open("data/last_accuracy.json", "r") as f:
            accuracy_data = json.load(f)
            model_accuracy = accuracy_data["accuracy"]
    except (FileNotFoundError, KeyError):
        print("Uyarı: last_accuracy.json bulunamadı, varsayılan 1.0 kullanılıyor")
        model_accuracy = 1.0

    alerts  = check_alerts(metrics, model_accuracy=model_accuracy)

    print(f"Return Rate   : {metrics['return_rate']}%")
    print(f"Total Revenue: ${metrics['total_revenue']:,}")
    print(f"Alerts found : {len(alerts)}")

    agent_diagnosis = None
    if alerts:
        print("\nALERTS:")
        for alert in alerts:
            print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")

        # Sadece gerçek bir uyarı varsa investigator agent çalışır
        agent_diagnosis = investigate_with_agent(alerts)

    log_alerts(alerts, metrics, agent_diagnosis)
    send_alert_email(alerts, metrics)

    print("\nMonitor complete.")