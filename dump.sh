cd /opt/airflow/dags/project
for f in agent.py alert_monitor.py api.py auto_retrain.py check_mlflow.py dags/pipeline_dag.py dashboard.py data_fetcher.py db_setup.py drift_monitor.py hyperparameter_tuning.py main_pipeline.py pipeline.py scheduler.py requirements.txt docker-compose.yml Dockerfile Dockerfile.airflow Procfile README.md
do
  echo "=== $f ==="
  cat "$f" 2>/dev/null
  echo
done
