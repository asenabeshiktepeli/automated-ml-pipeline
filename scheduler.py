import schedule
import time
import subprocess
import logging
import os
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────
LOG_DIR  = "logs"
LOG_FILE = f"{LOG_DIR}/scheduler.log"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ── PIPELINE RUNNER ───────────────────────────────────────
def run_pipeline():
    logging.info("Pipeline triggered — starting...")
    try:
        result = subprocess.run(
            ["python", "main_pipeline.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logging.info("Pipeline completed successfully.")
            logging.info(result.stdout[-500:])
        else:
            logging.error("Pipeline failed.")
            logging.error(result.stderr[-500:])
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

# ── SCHEDULE ──────────────────────────────────────────────
# Runs every day at 02:00 AM automatically
schedule.every().day.at("02:00").do(run_pipeline)

# Also runs immediately once on startup
logging.info("Scheduler started. Pipeline will run daily at 02:00 AM.")
logging.info("Running pipeline now for initial test...")
run_pipeline()

# ── MAIN LOOP ─────────────────────────────────────────────
logging.info("Entering schedule loop. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)