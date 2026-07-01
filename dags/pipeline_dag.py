from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import logging

default_args = {
    "owner"      : "data_pipeline",
    "retries"    : 1,
    "retry_delay": timedelta(minutes=5),
    "start_date" : datetime(2026, 1, 1),
}

PROJECT_DIR = "/opt/airflow/dags/project"

def run_script(script_name):
    logging.info(f"Running {script_name}...")
    result = subprocess.run(
        ["python", f"{PROJECT_DIR}/{script_name}"],
        capture_output=True, text=True,
        cwd=PROJECT_DIR
    )
    logging.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"{script_name} failed: {result.stderr}")

def run_pipeline():
    run_script("main_pipeline.py")

def run_drift_check():
    run_script("drift_monitor.py")

def run_auto_retrain():
    run_script("auto_retrain.py")

def run_alert_check():
    run_script("alert_monitor.py")

with DAG(
    dag_id="sales_data_pipeline",
    default_args=default_args,
    description="Automated sales pipeline with drift monitoring and auto-retrain",
    schedule="0 2 * * *",
    catchup=False,
    tags=["sales", "ml", "llm", "drift", "retrain"],
) as dag:

    start = BashOperator(
        task_id="start_notification",
        bash_command='echo "Pipeline starting at $(date)"'
    )

    run = PythonOperator(
        task_id="run_pipeline",
        python_callable=run_pipeline,
    )

    drift_check = PythonOperator(
        task_id="check_drift",
        python_callable=run_drift_check,
    )

    auto_retrain = PythonOperator(
        task_id="auto_retrain_if_needed",
        python_callable=run_auto_retrain,
    )

    alert_check = PythonOperator(
        task_id="check_alerts",
        python_callable=run_alert_check,
    )

    end = BashOperator(
        task_id="end_notification",
        bash_command='echo "Pipeline completed at $(date)"'
    )

    start >> run >> drift_check >> auto_retrain >> alert_check >> end