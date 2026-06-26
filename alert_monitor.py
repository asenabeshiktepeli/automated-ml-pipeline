import smtplib
import sqlite3
import pandas as pd
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────
# Fill in your Gmail address and App Password
EMAIL_SENDER   = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "your_email@gmail.com"

# Alert thresholds
ACCURACY_THRESHOLD    = 0.85   # Alert if accuracy drops below 85%
RETURN_RATE_THRESHOLD = 25.0   # Alert if return rate exceeds 25%

DB_PATH = "data/pipeline.db"

# ── LOAD METRICS ──────────────────────────────────────────
def load_metrics():
    print("Loading metrics from database...")
    conn   = sqlite3.connect(DB_PATH)
    df     = pd.read_sql("SELECT * FROM sales_data", conn)
    conn.close()

    metrics = {
        "total_revenue"  : round(df["revenue"].sum(), 2),
        "total_orders"   : len(df),
        "return_rate"    : round(df["returned"].mean() * 100, 2),
        "avg_order_value": round(df["revenue"].mean(), 2),
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return metrics

# ── CHECK ALERTS ──────────────────────────────────────────
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

# ── SEND EMAIL ────────────────────────────────────────────
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
- Total Revenue  : ${metrics['total_revenue']:,}
- Total Orders   : {metrics['total_orders']}
- Return Rate    : {metrics['return_rate']}%
- Avg Order Value: ${metrics['avg_order_value']}

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

# ── LOG ALERT ─────────────────────────────────────────────
def log_alerts(alerts, metrics):
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
            f.write("Status: OK — No alerts\n")

    print(f"Alert log saved: {log_path}")

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ALERT MONITOR — STARTING")
    print("="*60 + "\n")

    metrics  = load_metrics()
    alerts   = check_alerts(metrics)

    print(f"Return Rate  : {metrics['return_rate']}%")
    print(f"Total Revenue: ${metrics['total_revenue']:,}")
    print(f"Alerts found : {len(alerts)}")

    if alerts:
        print("\nALERTS:")
        for alert in alerts:
            print(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")

    log_alerts(alerts, metrics)
    send_alert_email(alerts, metrics)

    print("\nMonitor complete.")