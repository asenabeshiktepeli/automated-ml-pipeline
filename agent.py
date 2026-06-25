import pandas as pd
import requests
import json

# ── DATA LOAD ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv("data/data.csv", encoding="ISO-8859-1")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "InvoiceNo"  : "invoice_no",
        "StockCode"  : "stock_code",
        "Description": "description",
        "Quantity"   : "quantity",
        "InvoiceDate": "date",
        "UnitPrice"  : "price",
        "CustomerID" : "customer_id",
        "Country"    : "country"
    })
    df["returned"] = (df["quantity"] < 0).astype(int)
    df = df[df["price"] > 0]
    df.dropna(subset=["customer_id"], inplace=True)
    df["revenue"] = df["quantity"] * df["price"]
    return df

df = load_data()

# ── TOOLS ─────────────────────────────────────────────────
def get_summary_stats():
    return f"""
Total Revenue: ${df['revenue'].sum():,.2f}
Total Orders: {len(df):,}
Unique Customers: {df['customer_id'].nunique():,}
Return Rate: {df['returned'].mean()*100:.2f}%
Top Country: {df.groupby('country')['revenue'].sum().idxmax()}
Top Product: {df.groupby('description')['revenue'].sum().idxmax()}
"""

def get_top_countries(n=5):
    result = df.groupby("country")["revenue"].sum().nlargest(n)
    return result.to_string()

def get_return_rate_by_country(n=5):
    result = df.groupby("country")["returned"].mean().nlargest(n) * 100
    return result.round(2).to_string()

def get_top_products(n=5):
    result = df.groupby("description")["revenue"].sum().nlargest(n)
    return result.to_string()

TOOLS = {
    "get_summary_stats"        : get_summary_stats,
    "get_top_countries"        : get_top_countries,
    "get_return_rate_by_country": get_return_rate_by_country,
    "get_top_products"         : get_top_products,
}

# ── AGENT ─────────────────────────────────────────────────
def ask_agent(question: str) -> str:
    # Veri özetini hazırla
    data_context = get_summary_stats()
    top_countries = get_top_countries(10)
    return_rates  = get_return_rate_by_country(10)
    top_products  = get_top_products(10)

    prompt = f"""You are a data analyst AI agent for an e-commerce company.
You have access to the following real data:

SUMMARY:
{data_context}

TOP COUNTRIES BY REVENUE:
{top_countries}

RETURN RATES BY COUNTRY:
{return_rates}

TOP PRODUCTS BY REVENUE:
{top_products}

Answer this question based on the data above:
{question}

Give a concise, data-driven answer."""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# ── RUN ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("E-COMMERCE AI AGENT")
    print("="*60)

    questions = [
        "Which country has the highest return rate and why might that be?",
        "What are the top 3 products by revenue?",
        "Give me a business summary with key insights."
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        print("-"*40)
        answer = ask_agent(q)
        print(f"Answer: {answer}")
        print("="*60)