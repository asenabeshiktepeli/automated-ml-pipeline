"""
data/data.csv (Online Retail dataset, cp1254 encoding) -> data/retail_data.parquet

Run once (or whenever data.csv changes):
    python convert_to_parquet.py
"""
import pandas as pd
import os

SRC_PATH = "data/data.csv"
DST_PATH = "data/retail_data.parquet"

def main():
    print(f"Reading {SRC_PATH} (encoding=cp1254)...")
    df = pd.read_csv(SRC_PATH, encoding="cp1254")
    print(f"  {len(df):,} rows, {len(df.columns)} columns loaded.")
    print(f"  Columns: {df.columns.tolist()}")

    os.makedirs("data", exist_ok=True)
    df.to_parquet(DST_PATH, index=False)

    size_mb = os.path.getsize(DST_PATH) / (1024 * 1024)
    print(f"Parquet written: {DST_PATH} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()