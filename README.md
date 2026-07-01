# Automated E-Commerce ML Pipeline

End-to-end automated ML system processing 400K+ real e-commerce transactions with AI agent, drift monitoring, auto-retraining, and REST API.

## 🚀 Live API

The prediction API is deployed and publicly accessible:

**Base URL:** https://automated-ml-pipeline-gf15.onrender.com

- `GET /` — health/status check
- `GET /health` — health check
- `GET /docs` — interactive Swagger UI (try it in the browser)
- `POST /predict` — return prediction endpoint

> ⚠️ **Note:** This runs on Render's free tier, which spins down after periods of inactivity. The first request after idle time may take up to ~50 seconds while the instance wakes up — subsequent requests are fast.

### Try it in the browser

Open [https://automated-ml-pipeline-gf15.onrender.com/docs](https://automated-ml-pipeline-gf15.onrender.com/docs) and use the Swagger UI to test `/predict` interactively.

### Try it with curl

```bash
curl -X POST https://automated-ml-pipeline-gf15.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product": " I LOVE LONDON MINI BACKPACK",
    "category": "CHRISTMAS",
    "region": "United Kingdom",
    "quantity_abs": 1,
    "price": 15.5,
    "month": 12,
    "day_of_week": 3,
    "customer_prior_orders": 5,
    "customer_avg_order_value": 120,
    "customer_return_rate": 0.15,
    "customer_recency_days": 20
  }'
```

### Try it with PowerShell

```powershell
$body = @{
    product = " I LOVE LONDON MINI BACKPACK"
    category = "CHRISTMAS"
    region = "United Kingdom"
    quantity_abs = 1
    price = 15.5
    month = 12
    day_of_week = 3
    customer_prior_orders = 5
    customer_avg_order_value = 120
    customer_return_rate = 0.15
    customer_recency_days = 20
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://automated-ml-pipeline-gf15.onrender.com/predict" -Method Post -Body $body -ContentType "application/json"
```

**Example response:**

```json
{
  "prediction": 1,
  "probability": 0.9526,
  "label": "Will be returned"
}
```

> `product`, `category`, and `region` must match values the model's encoders were trained on. Unrecognized values will return an error.

## Architecture

Data (400K+ records)

↓

ML Pipeline (Random Forest, 91.97% accuracy)

↓

MLflow Experiment Tracking

↓

Drift Monitor → Auto Retrain

↓

FastAPI Model Serving (deployed on Render)

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
- **FastAPI REST API** — real-time return prediction endpoint, deployed live on Render
- **AI Agent** — answer business questions in natural language
- **Live Dashboard** — Plotly Dash with revenue, returns, trends
- **Docker** — fully containerized, single command deployment
- **Scheduler** — automated nightly execution
- **CI/CD** — GitHub Actions running pytest on every push

## Tech Stack

- Python 3.13
- scikit-learn — ML modeling
- MLflow (mlflow-skinny for serving) — experiment tracking & model loading
- Ollama (llama3.1:8b) — local LLM
- FastAPI + Uvicorn — REST API
- Plotly Dash — dashboard
- SciPy — statistical drift detection
- Docker + Docker Compose — containerization
- schedule — automation
- Render — cloud deployment for the prediction API

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

├── api.py # FastAPI REST endpoint (deployed on Render)

├── agent.py # AI data analysis agent

├── model/ # Serialized model + encoders (loaded directly by api.py)

├── Dockerfile

├── docker-compose.yml

├── Procfile # Render start command

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

### Local

```bash
POST http://localhost:8000/predict
```

### Live (deployed)

```bash
POST https://automated-ml-pipeline-gf15.onrender.com/predict
```

**Request body:**

```json
{
  "product": " I LOVE LONDON MINI BACKPACK",
  "category": "CHRISTMAS",
  "region": "United Kingdom",
  "quantity_abs": 1,
  "price": 15.5,
  "month": 12,
  "day_of_week": 3,
  "customer_prior_orders": 5,
  "customer_avg_order_value": 120,
  "customer_return_rate": 0.15,
  "customer_recency_days": 20
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.9526,
  "label": "Will be returned"
}
```

## Recent Updates

### Cloud Deployment

- API deployed live to Render (`api.py` runs independently of any local MLflow tracking server)
- Model and encoders are serialized and bundled directly under `model/`, loaded from disk at startup
- `requirements.txt` trimmed to only what the API needs (`mlflow-skinny` instead of full `mlflow`, plus `pandas`, `scikit-learn`, `skops`, `fastapi`, `uvicorn`, `pydantic`, `joblib`, `numpy`)
- `Procfile` defines the Render start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

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
