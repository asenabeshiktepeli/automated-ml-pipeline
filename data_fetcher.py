import requests
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime, timedelta
import random

# ── CONFIG ────────────────────────────────────────────────
API_BASE    = "https://fakestoreapi.com"
DB_PATH     = "data/pipeline.db"
DATA_PATH   = "data/sales_data.csv"

# ── FETCH PRODUCTS ────────────────────────────────────────
def fetch_products():
    print("Fetching products from API...")
    response = requests.get(f"{API_BASE}/products", timeout=10)
    response.raise_for_status()
    products = response.json()
    print(f"  {len(products)} products fetched.")
    return products

# ── FETCH CARTS ───────────────────────────────────────────
def fetch_carts():
    print("Fetching carts from API...")
    response = requests.get(f"{API_BASE}/carts", timeout=10)
    response.raise_for_status()
    carts = response.json()
    print(f"  {len(carts)} carts fetched.")
    return carts

# ── BUILD SALES DATAFRAME ─────────────────────────────────
def build_sales_df(products, carts):
    print("Building sales dataset...")

    product_map = {p["id"]: p for p in products}
    records     = []

    for cart in carts:
        cart_date = datetime.strptime(cart["date"], "%Y-%m-%dT%H:%M:%S.%fZ")

        for item in cart["products"]:
            product = product_map.get(item["productId"])
            if not product:
                continue

            category = product["category"]
            price    = product["price"]
            quantity = item["quantity"]
            revenue  = round(price * quantity, 2)

            records.append({
                "date"        : cart_date.strftime("%Y-%m-%d"),
                "product"     : product["title"][:30],
                "category"    : category,
                "quantity"    : quantity,
                "price"       : price,
                "customer_age": random.randint(18, 65),
                "region"      : random.choice(["North", "South", "East", "West"]),
                "returned"    : random.choices([0, 1], weights=[80, 20])[0],
                "revenue"     : revenue,
            })

    df = pd.DataFrame(records)
    print(f"  {len(df)} sales records built.")
    return df

# ── SAVE TO CSV ───────────────────────────────────────────
def save_to_csv(df):
    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"  Saved to {DATA_PATH}")

# ── SAVE TO DATABASE ──────────────────────────────────────
def save_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sales_data", conn, if_exists="replace", index=False)
    count = pd.read_sql("SELECT COUNT(*) as n FROM sales_data", conn)["n"][0]
    conn.close()
    print(f"  {count} records saved to database.")

# ── FETCH SUMMARY ─────────────────────────────────────────
def print_summary(df):
    print("\n" + "="*60)
    print("LIVE DATA SUMMARY")
    print("="*60)
    print(f"Total Orders   : {len(df)}")
    print(f"Total Revenue  : ${df['revenue'].sum():,.2f}")
    print(f"Avg Order Value: ${df['revenue'].mean():,.2f}")
    print(f"Return Rate    : {df['returned'].mean()*100:.1f}%")
    print(f"Categories     : {df['category'].nunique()}")
    print(f"Date Range     : {df['date'].min()} → {df['date'].max()}")
    print("\nRevenue by Category:")
    cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    for cat, rev in cat_rev.items():
        print(f"  {cat:30} ${rev:,.2f}")
    print("="*60)

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LIVE DATA FETCHER — STARTING")
    print("="*60 + "\n")

    products = fetch_products()
    carts    = fetch_carts()
    df       = build_sales_df(products, carts)

    save_to_csv(df)
    save_to_db(df)
    print_summary(df)

    print("\nData fetch complete.")
    print("Run main_pipeline.py to analyze this data.")