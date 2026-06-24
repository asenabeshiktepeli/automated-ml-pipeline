import pandas as pd
import logging
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────
DATA_PATH         = "data/sales_data.csv"
LOG_DIR           = "logs"
RETURN_THRESHOLD  = 0.15   # Alert if return rate exceeds 15%
REVENUE_MIN       = 10000  # Alert if total revenue drops below $10,000

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/alerts.log"),
        logging.StreamHandler()
    ]
)

# ── LOAD DATA ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["revenue"] = df["quantity"] * df["price"]
    return df

# ── CHECK ALERTS ──────────────────────────────────────────
def check_alerts(df):
    alerts_triggered = []

    return_rate   = df["returned"].mean()
    total_revenue = df["revenue"].sum()

    # Alert 1 — High return rate
    if return_rate > RETURN_THRESHOLD:
        msg = (f"ALERT — High return rate detected: "
               f"{return_rate:.1%} (threshold: {RETURN_THRESHOLD:.1%})")
        logging.warning(msg)
        alerts_triggered.append(msg)
    else:
        logging.info(f"Return rate OK: {return_rate:.1%}")

    # Alert 2 — Low revenue
    if total_revenue < REVENUE_MIN:
        msg = (f"ALERT — Revenue below threshold: "
               f"${total_revenue:,.0f} (min: ${REVENUE_MIN:,})")
        logging.warning(msg)
        alerts_triggered.append(msg)
    else:
        logging.info(f"Revenue OK: ${total_revenue:,.0f}")

    # Alert 3 — Region performance drop
    region_revenue = df.groupby("region")["revenue"].sum()
    weak_regions   = region_revenue[region_revenue < 4000].index.tolist()
    if weak_regions:
        msg = f"ALERT — Underperforming regions: {weak_regions}"
        logging.warning(msg)
        alerts_triggered.append(msg)
    else:
        logging.info("All regions performing above threshold.")

    return alerts_triggered

# ── SUMMARY ───────────────────────────────────────────────
def print_summary(df, alerts):
    print("\n" + "="*60)
    print("ALERT MONITOR — SUMMARY")
    print("="*60)
    print(f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total orders: {len(df)}")
    print(f"Return rate : {df['returned'].mean():.1%}")
    print(f"Revenue     : ${df['revenue'].sum():,.0f}")
    print("-"*60)
    if alerts:
        print(f"ALERTS TRIGGERED: {len(alerts)}")
        for a in alerts:
            print(f"  ⚠  {a}")
    else:
        print("ALL SYSTEMS NORMAL — No alerts.")
    print("="*60 + "\n")

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    df     = load_data()
    alerts = check_alerts(df)
    print_summary(df, alerts)