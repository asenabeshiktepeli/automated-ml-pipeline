import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────
DB_PATH     = "data/pipeline.db"
MLFLOW_URI  = "sqlite:///mlflow.db"
N_TRIALS    = 30  # number of optimization trials

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("hyperparameter_tuning")

# ── LOAD DATA ─────────────────────────────────────────────
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql("SELECT * FROM sales_data", conn)
    conn.close()

    le = LabelEncoder()
    df["product_enc"]  = le.fit_transform(df["product"].astype(str))
    df["category_enc"] = le.fit_transform(df["category"].astype(str))
    df["region_enc"]   = le.fit_transform(df["region"].astype(str))

    features = ["product_enc", "category_enc", "region_enc",
                "quantity", "price", "customer_age"]

    X = df[features]
    y = df["returned"]
    return X, y

# ── OPTUNA OBJECTIVE ──────────────────────────────────────
def objective(trial):
    model_name = trial.suggest_categorical(
        "model", ["random_forest", "xgboost", "lightgbm"]
    )

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators = trial.suggest_int("n_estimators", 50, 300),
            max_depth    = trial.suggest_int("max_depth", 3, 15),
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
            random_state = 42
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators  = trial.suggest_int("n_estimators", 50, 300),
            max_depth     = trial.suggest_int("max_depth", 3, 10),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3),
            random_state  = 42,
            eval_metric   = "logloss",
            verbosity     = 0
        )
    else:
        model = LGBMClassifier(
            n_estimators  = trial.suggest_int("n_estimators", 50, 300),
            max_depth     = trial.suggest_int("max_depth", 3, 10),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3),
            random_state  = 42,
            verbosity     = -1
        )

    score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    return score

# ── RUN OPTIMIZATION ──────────────────────────────────────
def run_optimization(X, y):
    print(f"\nRunning Optuna optimization — {N_TRIALS} trials...")
    print("This will test Random Forest, XGBoost, and LightGBM.\n")

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best Model     : {best.params.get('model')}")
    print(f"Best Accuracy  : {best.value:.4f}")
    print(f"Best Params    : {best.params}")
    return study

# ── TRAIN BEST MODEL ──────────────────────────────────────
def train_best_model(study, X, y):
    best_params = study.best_trial.params
    model_name  = best_params.get("model")

    print(f"\nTraining best model: {model_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators      = best_params.get("n_estimators", 100),
            max_depth         = best_params.get("max_depth", 5),
            min_samples_split = best_params.get("min_samples_split", 2),
            random_state      = 42
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators  = best_params.get("n_estimators", 100),
            max_depth     = best_params.get("max_depth", 5),
            learning_rate = best_params.get("learning_rate", 0.1),
            random_state  = 42,
            eval_metric   = "logloss",
            verbosity     = 0
        )
    else:
        model = LGBMClassifier(
            n_estimators  = best_params.get("n_estimators", 100),
            max_depth     = best_params.get("max_depth", 5),
            learning_rate = best_params.get("learning_rate", 0.1),
            random_state  = 42,
            verbosity     = -1
        )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy    = accuracy_score(y_test, predictions)
    report      = classification_report(y_test, predictions)

    # Log to MLflow
    with mlflow.start_run(run_name=f"best_{model_name}"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("optuna_best_score", study.best_value)
        mlflow.sklearn.log_model(model, f"best_{model_name}")

    print(f"\nFinal Accuracy  : {accuracy:.4f}")
    print(f"Optuna Best CV  : {study.best_value:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"\nLogged to MLflow ✓")

    return model, accuracy

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING — STARTING")
    print("="*60)

    X, y  = load_data()
    print(f"Dataset: {len(X)} records, {y.sum()} returns")

    study         = run_optimization(X, y)
    model, acc    = train_best_model(study, X, y)

    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE")
    print(f"Best accuracy: {acc:.4f}")
    print(f"{'='*60}")