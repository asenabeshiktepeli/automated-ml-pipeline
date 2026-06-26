from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import logging

default_args = {
    "owner"           : "data_pipeline",
    "retries"         : 1,
    "retry_delay"     : timedelta(minutes=5),
    "start_date"      : datetime(2026, 1, 1),
}

def run_pipeline():
    logging.info("Running main pipeline...")
    result = subprocess.run(
        ["python", "/opt/airflow/dags/main_pipeline.py"],
        capture_output=True, text=True
    )
    logging.info(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Pipeline failed: {result.stderr}")

with DAG(
    dag_id="sales_data_pipeline",
    default_args=default_args,
    description="Automated sales data pipeline",
    schedule="0 2 * * *",  # Every day at 02:00 AM
    catchup=False,
    tags=["sales", "ml", "llm"],
) as dag:

    start = BashOperator(
        task_id="start_notification",
        bash_command='echo "Pipeline starting at $(date)"'
    )

    run = PythonOperator(
        task_id="run_pipeline",
        python_callable=run_pipeline,
    )

    end = BashOperator(
        task_id="end_notification",
        bash_command='echo "Pipeline completed at $(date)"'
    )

    start >> run >> end