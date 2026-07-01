import duckdb
con = duckdb.connect('../data/warehouse.duckdb')
result = con.execute("""
    select customer_id, invoice_date, customer_prior_orders, customer_avg_order_value
    from retail_clean
    where customer_prior_orders = 0
    limit 5
""").fetchall()
print(result)