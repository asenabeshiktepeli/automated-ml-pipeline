from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import uvicorn

app = FastAPI(title="E-Commerce Return Prediction API")

# ── MODEL TRAINING ON STARTUP ─────────────────────────────
def train():
    df = pd.read_csv("data/data.csv", encoding="ISO-8859-1")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "InvoiceNo"  : "invoice_no",
        "StockCode"  : "stock_code",
        "Quantity"   : "quantity",
        "InvoiceDate": "date",
        "UnitPrice"  : "price",
        "CustomerID" : "customer_id",
        "Country"    : "country"
    })
    df["returned"] = (df["quantity"] < 0).astype(int)
    df = df[df["price"] > 0]
    df.dropna(subset=["customer_id"], inplace=True)
    df["date"]        = pd.to_datetime(df["date"])
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    le_country    = LabelEncoder()
    le_stock      = LabelEncoder()
    df["country_enc"]    = le_country.fit_transform(df["country"])
    df["stock_code_enc"] = le_stock.fit_transform(df["stock_code"].astype(str))

    features = ["stock_code_enc", "country_enc", "price", "month", "day_of_week"]
    X = df[features]
    y = df["returned"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)

    return clf, le_country, le_stock

print("Training model...")
model, le_country, le_stock = train()
print("Model ready ✅")

# ── REQUEST SCHEMA ────────────────────────────────────────
class PredictRequest(BaseModel):
    stock_code : str
    country    : str
    price      : float
    month      : int
    day_of_week: int

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "E-Commerce Return Prediction API"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        stock_enc   = le_stock.transform([req.stock_code])[0]
        country_enc = le_country.transform([req.country])[0]
    except ValueError as e:
        return {"error": f"Unknown value: {str(e)}"}

    features = [[stock_enc, country_enc, req.price, req.month, req.day_of_week]]
    pred     = model.predict(features)[0]
    prob     = model.predict_proba(features)[0][int(pred)]

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