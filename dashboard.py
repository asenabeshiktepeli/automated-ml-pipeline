import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import os
import glob
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────
DATA_PATH   = "data/data.csv"
REPORTS_DIR = "reports"

# ── LOAD DATA ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
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
    df["date"]    = pd.to_datetime(df["date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df

def load_latest_report():
    reports = glob.glob(f"{REPORTS_DIR}/report_*.txt")
    if not reports:
        return "No reports generated yet."
    latest = max(reports, key=os.path.getctime)
    with open(latest, "r", encoding="utf-8") as f:
        return f.read()

# ── APP ───────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Data Pipeline Dashboard"

df = load_data()

total_revenue   = df["revenue"].sum()
total_orders    = len(df)
return_rate     = df["returned"].mean() * 100
avg_order_value = df["revenue"].mean()

# Top 10 products by revenue
top_products = (
    df.groupby("description")["revenue"]
    .sum()
    .nlargest(10)
    .reset_index()
)

# Revenue by country (top 10)
top_countries = (
    df.groupby("country")["revenue"]
    .sum()
    .nlargest(10)
    .reset_index()
)

# ── LAYOUT ────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("Automated Data Pipeline Dashboard",
                style={"color": "#ffffff", "margin": "0", "fontSize": "24px"}),
        html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               style={"color": "#aaaaaa", "margin": "4px 0 0 0", "fontSize": "13px"})
    ], style={
        "backgroundColor": "#1e1e2e",
        "padding": "20px 30px",
        "borderBottom": "2px solid #444"
    }),

    # KPI Cards
    html.Div([
        html.Div([
            html.H3("Total Revenue", style={"color": "#aaaaaa", "fontSize": "13px", "margin": "0"}),
            html.H2(f"${total_revenue:,.0f}", style={"color": "#00d4aa", "margin": "8px 0 0 0"})
        ], style={"backgroundColor": "#2a2a3e", "padding": "20px", "borderRadius": "8px", "flex": "1"}),

        html.Div([
            html.H3("Total Orders", style={"color": "#aaaaaa", "fontSize": "13px", "margin": "0"}),
            html.H2(f"{total_orders:,}", style={"color": "#7c6af7", "margin": "8px 0 0 0"})
        ], style={"backgroundColor": "#2a2a3e", "padding": "20px", "borderRadius": "8px", "flex": "1"}),

        html.Div([
            html.H3("Return Rate", style={"color": "#aaaaaa", "fontSize": "13px", "margin": "0"}),
            html.H2(f"{return_rate:.1f}%", style={"color": "#ff6b6b", "margin": "8px 0 0 0"})
        ], style={"backgroundColor": "#2a2a3e", "padding": "20px", "borderRadius": "8px", "flex": "1"}),

        html.Div([
            html.H3("Avg Order Value", style={"color": "#aaaaaa", "fontSize": "13px", "margin": "0"}),
            html.H2(f"${avg_order_value:,.2f}", style={"color": "#ffa94d", "margin": "8px 0 0 0"})
        ], style={"backgroundColor": "#2a2a3e", "padding": "20px", "borderRadius": "8px", "flex": "1"}),

        html.Div([
            html.H3("Unique Customers", style={"color": "#aaaaaa", "fontSize": "13px", "margin": "0"}),
            html.H2(f"{df['customer_id'].nunique():,}", style={"color": "#74c0fc", "margin": "8px 0 0 0"})
        ], style={"backgroundColor": "#2a2a3e", "padding": "20px", "borderRadius": "8px", "flex": "1"}),

    ], style={"display": "flex", "gap": "16px", "padding": "24px 30px"}),

    # Charts Row 1
    html.Div([
        html.Div([
            dcc.Graph(figure=px.bar(
                top_products,
                x="revenue", y="description",
                orientation="h",
                title="Top 10 Products by Revenue",
                color="revenue",
                color_continuous_scale="teal",
                template="plotly_dark"
            ).update_layout(yaxis={"categoryorder": "total ascending"}))
        ], style={"flex": "2", "backgroundColor": "#2a2a3e", "borderRadius": "8px"}),

        html.Div([
            dcc.Graph(figure=px.pie(
                top_countries,
                values="revenue", names="country",
                title="Revenue by Country",
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Plasma
            ))
        ], style={"flex": "1", "backgroundColor": "#2a2a3e", "borderRadius": "8px"}),

    ], style={"display": "flex", "gap": "16px", "padding": "0 30px"}),

    # Charts Row 2
    html.Div([
        html.Div([
            dcc.Graph(figure=px.line(
                df.groupby(df["date"].dt.to_period("M").astype(str))["revenue"]
                .sum().reset_index(),
                x="date", y="revenue",
                title="Monthly Revenue Trend",
                template="plotly_dark",
                color_discrete_sequence=["#00d4aa"]
            ))
        ], style={"flex": "1", "backgroundColor": "#2a2a3e", "borderRadius": "8px"}),

        html.Div([
            dcc.Graph(figure=px.bar(
                df.groupby("country")["returned"].mean().nlargest(10).reset_index(),
                x="country", y="returned",
                title="Return Rate by Country (Top 10)",
                template="plotly_dark",
                color="returned",
                color_continuous_scale="reds"
            ))
        ], style={"flex": "1", "backgroundColor": "#2a2a3e", "borderRadius": "8px"}),

    ], style={"display": "flex", "gap": "16px", "padding": "16px 30px"}),

    # LLM Report
    html.Div([
        html.H3("LLM Executive Report",
                style={"color": "#ffffff", "marginBottom": "12px"}),
        html.Pre(load_latest_report(),
                 style={
                     "color": "#cccccc",
                     "backgroundColor": "#1a1a2e",
                     "padding": "20px",
                     "borderRadius": "8px",
                     "fontSize": "13px",
                     "whiteSpace": "pre-wrap",
                     "maxHeight": "400px",
                     "overflowY": "auto"
                 })
    ], style={"padding": "0 30px 30px"}),

], style={"backgroundColor": "#13131f", "minHeight": "100vh", "fontFamily": "Arial, sans-serif"})

# ── RUN ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Dashboard running at: http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)