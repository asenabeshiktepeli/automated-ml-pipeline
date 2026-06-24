# Automated ML Pipeline

End-to-end automated data pipeline with ML modeling, LLM-powered reporting, and Docker deployment.

## What This Does

- Loads and cleans sales data automatically
- Trains a Random Forest model to predict product returns
- Tracks every model run with MLflow
- Generates executive reports using a local LLM (Ollama)
- Monitors anomalies and triggers alerts
- Visualizes everything on a live Plotly Dash dashboard
- Runs on a schedule — fully automated
- Fully containerized with Docker

## Tech Stack

- Python 3.13
- scikit-learn — ML modeling
- MLflow — experiment tracking
- Ollama (llama3.1:8b) — local LLM reporting
- Plotly Dash — interactive dashboard
- Docker + Docker Compose — containerization
- schedule — automated pipeline execution

## Project Structure

data_pipeline/

├── main_pipeline.py # Core pipeline: load → clean → train → report

├── dashboard.py # Live visual dashboard

├── scheduler.py # Automated nightly execution

├── alert_monitor.py # Anomaly detection and alerting

├── Dockerfile # Container definition

├── docker-compose.yml # Multi-service orchestration

├── requirements.txt # Dependencies

└── data/ # Input data

## Quick Start

```bash
# Run with Docker
docker compose up --build

# Or run locally
pip install -r requirements.txt
python main_pipeline.py
```

## Dashboard

Open after starting the dashboard service.
`http://localhost:8050`
