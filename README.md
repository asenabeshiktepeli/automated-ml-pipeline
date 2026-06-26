# Automated E-Commerce ML Pipeline

End-to-end automated ML system processing 400K+ real e-commerce transactions with AI agent, drift monitoring, auto-retraining, and REST API.

## Architecture

Data (400K+ records)

↓

ML Pipeline (Random Forest, 91.97% accuracy)

↓

MLflow Experiment Tracking

↓

Drift Monitor → Auto Retrain

↓

FastAPI Model Serving

↓

AI Agent (natural language queries)

↓

Plotly Dash Dashboard + LLM Reports

## Features

- **ML Pipeline** — loads, cleans, and trains on 400K+ real transactions
- **Return Prediction** — Random Forest classifier, 91.97% accuracy
- **MLflow Tracking** — every model run logged and versioned
- **LLM Reporting** — Ollama generates executive reports automatically
- **Drift Monitoring** — KS-test statistical drift detection
- **Auto Retraining** — triggers pipeline when drift or accuracy drop detected
- **FastAPI REST API** — real-time return prediction endpoint
- **AI Agent** — answer business questions in natural language
- **Live Dashboard** — Plotly Dash with revenue, returns, trends
- **Docker** — fully containerized, single command deployment
- **Scheduler** — automated nightly execution

## Tech Stack

- Python 3.13
- scikit-learn — ML modeling
- MLflow — experiment tracking
- Ollama (llama3.1:8b) — local LLM
- FastAPI + Uvicorn — REST API
- Plotly Dash — dashboard
- SciPy — statistical drift detection
- Docker + Docker Compose — containerization
- schedule — automation

## Results

| Metric         | Value                    |
| -------------- | ------------------------ |
| Dataset        | 400K+ real transactions  |
| Model Accuracy | 91.97%                   |
| Return Rate    | 2.19%                    |
| Total Revenue  | $8.3M                    |
| Top Market     | United Kingdom (84%)     |
| Top Product    | REGENCY CAKESTAND 3 TIER |

## Project Structure

data_pipeline/

├── main_pipeline.py # Core ML pipeline

├── dashboard.py # Live Plotly Dash dashboard

├── scheduler.py # Automated scheduling

├── alert_monitor.py # Anomaly detection

├── drift_monitor.py # Statistical drift monitoring

├── auto_retrain.py # Automatic retraining

├── api.py # FastAPI REST endpoint

├── agent.py # AI data analysis agent

├── Dockerfile

├── docker-compose.yml

└── requirements.txt

## Quick Start

```bash
# Docker
docker compose up --build

# Local
pip install -r requirements.txt
python main_pipeline.py

# Start API
python api.py
# → http://localhost:8000/docs

# Start Dashboard
python dashboard.py
# → http://localhost:8050

# Run AI Agent
python agent.py

# Check drift & auto retrain
python drift_monitor.py
python auto_retrain.py
```

## API Usage

```bash
POST http://localhost:8000/predict
{
  "stock_code": "85123A",
  "country": "United Kingdom",
  "price": 2.55,
  "month": 6,
  "day_of_week": 3
}

→ {"prediction": 0, "probability": 0.94, "label": "Not returned"}
```

## Recent Updates

### Live API Integration

- Added `data_fetcher.py` — pulls real-time data from Fake Store API
- Automatic data refresh: products + carts → cleaned sales dataset
- Data stored in both CSV and SQLite database

### Database Layer

- Added SQLite database (`data/pipeline.db`)
- Tables: `sales_data`, `pipeline_runs`
- Ready for PostgreSQL migration via Docker

### Airflow Orchestration

- Full Docker Compose setup with Airflow 2.9.0
- DAG: `sales_data_pipeline` — runs daily at 02:00 AM
- Services: PostgreSQL + Airflow Webserver + Scheduler

### Alert Monitor

- Threshold-based alerting (accuracy, return rate)
- Log file: `logs/alerts.log`
- Email alert support (Gmail App Password required)

## Running the Full Stack

```bash
# 1. Fetch live data
python data_fetcher.py

# 2. Run pipeline
python main_pipeline.py

# 3. Start Airflow
docker-compose up -d

# 4. Monitor alerts
python alert_monitor.py

# 5. Start dashboard
python dashboard.py
# → http://localhost:8050

# 6. Check MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```
