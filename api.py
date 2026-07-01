from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import joblib
import os
import uvicorn

app = FastAPI(title="E-Commerce Return Prediction API")

# ── CONFIG ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model", "random_forest_model")
ENCODER_DIR = os.path.join(BASE_DIR, "model", "encoders")

# ── MODEL + ENCODER LOADING ON STARTUP ────────────────────
def load_artifacts():
    print("Loading model from local disk...")
    clf = mlflow.sklearn.load_model(MODEL_DIR)

    print("Loading encoders from local disk...")
    le_product  = joblib.load(os.path.join(ENCODER_DIR, "le_product.pkl"))
    le_category = joblib.load(os.path.join(ENCODER_DIR, "le_category.pkl"))
    le_region   = joblib.load(os.path.join(ENCODER_DIR, "le_region.pkl"))

    return clf, le_product, le_category, le_region

model, le_product, le_category, le_region = load_artifacts()
print("Model ready ✅")

# ── REQUEST SCHEMA ──────────────────────────────────────
class PredictRequest(BaseModel):
    product                  : str
    category                 : str
    region                   : str
    quantity_abs              : float
    price                     : float
    month                     : int
    day_of_week               : int
    customer_prior_orders     : float
    customer_avg_order_value  : float
    customer_return_rate      : float
    customer_recency_days     : float

# ── ENDPOINTS ────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "E-Commerce Return Prediction API"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        product_enc  = le_product.transform([req.product])[0]
        category_enc = le_category.transform([req.category])[0]
        region_enc   = le_region.transform([req.region])[0]
    except ValueError as e:
        return {"error": f"Unknown value: {str(e)}"}

    features = [[
        product_enc, category_enc, region_enc,
        req.quantity_abs, req.price, req.month, req.day_of_week,
        req.customer_prior_orders, req.customer_avg_order_value,
        req.customer_return_rate, req.customer_recency_days
    ]]

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][int(pred)]

    return {
        "prediction" : int(pred),
        "probability": round(float(prob), 4),
        "label"      : "Will be returned" if pred == 1 else "Not returned"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)