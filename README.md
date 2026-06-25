# Automated E-Commerce ML Pipeline

End-to-end automated ML pipeline processing 400K+ real e-commerce transactions with drift monitoring and auto-retraining.

## What This Does

- Loads and cleans 400K+ real e-commerce transactions (Kaggle dataset)
- Trains a Random Forest model to predict product returns (91.97% accuracy)
- Tracks every model run with MLflow
- Generates executive reports using a local LLM (Ollama)
- Monitors data drift using KS-test statistical analysis
- Auto-retrains the model when drift is detected or accuracy drops
- Visualizes everything on a live Plotly Dash dashboard
- Sends anomaly alerts via alert monitoring system
- Runs on a schedule — fully automated
- Fully containerized with Docker

## Tech Stack

- Python 3.13
- scikit-learn — ML modeling
- MLflow — experiment tracking
- Ollama (llama3.1:8b) — local LLM reporting
- Plotly Dash — interactive dashboard
- SciPy — statistical drift detection
- Docker + Docker Compose — containerization
- schedule — automated pipeline execution

## Project Structure

data_pipeline/

├── main_pipeline.py # Core pipeline: load → clean → train → report

├── dashboard.py # Live visual dashboard

├── scheduler.py # Automated nightly execution

├── alert_monitor.py # Anomaly detection and alerting

├── drift_monitor.py # Statistical data drift monitoring

├── auto_retrain.py # Automatic model retraining

├── Dockerfile # Container definition

├── docker-compose.yml # Multi-service orchestration

├── requirements.txt # Dependencies

└── data/ # Input data (Kaggle e-commerce dataset)

## Quick Start

```bash
# Run with Docker
docker compose up --build

# Or run locally
pip install -r requirements.txt
python main_pipeline.py

# Run drift monitor
python drift_monitor.py

# Run auto retrain check
python auto_retrain.py
```

## Dashboard

Open `http://localhost:8050` after starting the dashboard service.

## Results

- **Dataset**: 400K+ real e-commerce transactions
- **Model Accuracy**: 91.97%
- **Return Rate**: 2.19%
- **Top Market**: United Kingdom (84.1% of revenue)
- **Top Product**: REGENCY CAKESTAND 3 TIER
