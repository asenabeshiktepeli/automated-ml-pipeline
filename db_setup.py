from sqlalchemy import create_engine, text
import pandas as pd
import os

# ── CONFIG ────────────────────────────────────────────────
DB_PATH = "data/pipeline.db"
DB_URL  = f"sqlite:///{DB_PATH}"

def setup_database():
    print("Setting up SQLite database...")
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sales_data (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                date         TEXT,
                product      TEXT,
                category     TEXT,
                quantity     INTEGER,
                price        REAL,
                customer_age INTEGER,
                region       TEXT,
                returned     INTEGER,
                revenue      REAL,
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date       TEXT DEFAULT (datetime('now')),
                total_revenue  REAL,
                total_orders   INTEGER,
                return_rate    REAL,
                model_accuracy REAL,
                llm_report     TEXT
            )
        """))
        conn.commit()
        print("Tables created successfully.")

    # Load CSV into database
    df = pd.read_csv("data/sales_data.csv")
    df["date"]    = pd.to_datetime(df["date"]).astype(str)
    df["revenue"] = df["quantity"] * df["price"]

    df.to_sql("sales_data", engine, if_exists="replace", index=False)
    print(f"{len(df)} records loaded into database.")

    # Verify
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM sales_data"))
        count  = result.scalar()
        print(f"Verification: {count} records confirmed.")

    print("\nDatabase setup complete.")
    print(f"Database file: {DB_PATH}")

if __name__ == "__main__":
    setup_database()